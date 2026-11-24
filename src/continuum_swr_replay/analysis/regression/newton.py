"""Penalized Newton fit at fixed penalty strengths, in float64.

Minimizes ``F(beta) = -l(beta) + beta' S beta / 2`` with ``S`` the summed,
embedded penalty ``sum_j lam_j S_j`` (zero for an unpenalized fit). Each step
solves ``A step = g`` by Cholesky, with ``g`` the gradient of ``-F`` and
``A = H + S`` (``H`` the information), and the fit stops when the Newton
decrement ``g' A^-1 g / 2`` -- the gain in ``-F`` the step predicts -- is below
the tolerance. A step is shortened so no coefficient moves more than
``max_step``, then halved until ``F`` does not rise; a step whose ``F`` is not
finite is rejected the same way. A quadratic likelihood (the Gaussian) takes
its full step, uncapped: it is the exact optimum, whatever the scale of the
outcome. ``eta`` is never clipped.

Before the first step the inputs are checked (finite, of matching shapes, and
what the family takes) and so is the design: the coefficients the penalty
leaves free (the unpenalized columns and the penalties' null spaces) must be
identified by ``X``. A small decrement is not enough to stop: when the step it
belongs to still moves the linear predictor by more than
``NO_MAXIMUM_ETA_STEP`` on some row, the likelihood has no finite maximum
along that step (a 0/1 column whose rows all have the same outcome, a Poisson
cell with no events): the decrement shrinks to zero while a coefficient runs
to infinity. Near a finite maximum the step shrinks with the decrement. A
quadratic likelihood with ``A`` positive definite always has a finite maximum,
and its linear predictor is on the outcome's scale, so the rule is not applied
to it.

Once the decrement is below the tolerance the fit still takes that last full
step, when ``F`` does not rise by more than a few rounding units of ``F``
(``8 eps max(1, |F|)``) and the decrement where it lands is below the
tolerance too, and reports ``A`` and the decrement there; otherwise it stays
where it was, whose decrement is below the tolerance. Stopping before the step
would leave a coefficient off by up to about ``sqrt(2 tol / lambda_min)``
(``lambda_min`` the smallest eigenvalue of ``A``): 4e-5 on a content fit of 80
events. After it the error is of the order of the square of that.

A fit with no finite maximum records its last step and the columns running
away along it (:func:`running_away_columns`): those whose part of the step
moves the linear predictor by more than ``NO_MAXIMUM_ETA_STEP`` on some row,
so that a caller can leave them out of the session.

Every fit ends with one of the statuses below; only ``CONVERGED`` is a fit.

The numerical work runs in JAX, in float64 (``runtime``), on the rows padded
to a bucket (``Likelihood.pad``): one jitted kernel evaluates ``F`` at given
coefficients, and one makes a Newton iteration (``A``, its Cholesky factor,
the step, the decrement, ``max |X step|`` and the step-halving search). The
linear predictor is carried on the device from one iteration to the next,
``eta + scale * (X @ step)``, rather than recomputed as ``X @ beta``. The
rules above, the statuses and the design check are a host loop in Python and
NumPy around them. That loop is written for a batch of fits on one design (a
leading axis of fits, one per fit; :func:`penalized_newton` fits one), and
the kernels map over that axis. The rule constants ``NEWTON_DECREMENT_TOL``,
``MAX_STEP_HALVINGS``, ``NO_MAXIMUM_ETA_STEP``, ``NO_MAXIMUM_CAPPED_STEPS``,
``IDENTIFIABILITY_TOL`` and ``RUNNING_AWAY_TIE_DECIMALS`` are read from this
module at each call (the first two, with the step cap ``max_step``, are passed
to the iteration kernel as arguments), so a changed value takes effect in a
kernel already compiled. The defaults of :func:`penalized_newton`'s
``max_iter`` and ``max_step`` are ``NEWTON_MAX_STEPS`` and
``MAX_STEP_LENGTH`` as they were when the module was imported. A Cholesky factorization that fails gives NaN
in JAX rather than raising: ``A`` is not positive definite when ``A`` or its
factor is not finite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np

from .constants import (
    IDENTIFIABILITY_TOL,
    MAX_STEP_HALVINGS,
    MAX_STEP_LENGTH,
    NEWTON_DECREMENT_TOL,
    NEWTON_MAX_STEPS,
    NO_MAXIMUM_CAPPED_STEPS,
    NO_MAXIMUM_ETA_STEP,
    RUNNING_AWAY_TIE_DECIMALS,
)
from .likelihoods import Likelihood, Rows
from .runtime import float64_kernel, in_float64

__all__ = [
    "CONVERGED",
    "HALVING_FAILED",
    "NOT_IDENTIFIED",
    "NOT_POSITIVE_DEFINITE",
    "NO_FINITE_MAXIMUM",
    "STEP_LIMIT",
    "FitStatus",
    "NewtonFit",
    "penalized_newton",
    "running_away_columns",
]

FitStatus = Literal[
    "converged",
    "design not identified",
    "information not positive definite",
    "step halving failed",
    "step limit reached",
    "no finite maximum",
]
#: The decrement is below the tolerance at a finite maximum.
CONVERGED: FitStatus = "converged"
#: The design cannot identify a coefficient the penalty leaves free (a
#: duplicated column, a column zero on every row, a linear term beside a smooth
#: whose penalty's null space holds it); no step is taken.
NOT_IDENTIFIED: FitStatus = "design not identified"
#: ``A`` is not positive definite or not finite (a start whose predictions
#: overflow).
NOT_POSITIVE_DEFINITE: FitStatus = "information not positive definite"
#: No halving of a step lowered ``F``.
HALVING_FAILED: FitStatus = "step halving failed"
#: ``max_iter`` steps without the decrement falling below the tolerance.
STEP_LIMIT: FitStatus = "step limit reached"
#: The decrement is below the tolerance but the step still moves the linear
#: predictor by more than ``NO_MAXIMUM_ETA_STEP``, or the fit ran out of steps
#: while each of its last ``NO_MAXIMUM_CAPPED_STEPS`` was cut to the step cap,
#: lowered ``F`` and left a full step no shorter: a coefficient runs to
#: infinity.
NO_FINITE_MAXIMUM: FitStatus = "no finite maximum"


@dataclass(frozen=True)
class NewtonFit:
    """The end state of :func:`penalized_newton`.

    Attributes
    ----------
    beta : np.ndarray, shape (p,)
        The coefficients.
    A : np.ndarray, shape (p, p)
        ``H + S`` at ``beta``; its inverse is the Bayesian covariance. NaN when
        the design is not identified (no step is taken).
    status : FitStatus
        How the fit ended; ``CONVERGED`` is the only fit.
    newton_decrement : float
        ``g' A^-1 g / 2`` at ``beta``; NaN when ``A`` is not positive definite
        or the design is not identified.
    n_iter : int
        Newton steps taken.
    loglik : float
        The log-likelihood at ``beta`` (without the penalty).
    step : np.ndarray or None, shape (p,)
        For a fit with no finite maximum, the Newton step at ``beta``: the
        direction along which the likelihood rises without bound. None for
        every other status.
    running_away : np.ndarray of int, shape (k,)
        For a fit with no finite maximum, the columns running away along
        ``step`` (:func:`running_away_columns`), the one moving the linear
        predictor most first; empty for every other status.
    unidentified : np.ndarray or None, shape (p, k)
        For a design that is not identified, the combinations of coefficients
        it does not identify, with orthonormal columns; None for every other
        status. A combination weights each free coordinate after its column of
        the design is scaled to unit length, so the weights say which
        coordinates take part whatever their units, and ``X @ unidentified``
        is zero only when those columns have one length.
    """

    beta: np.ndarray
    A: np.ndarray
    status: FitStatus
    newton_decrement: float
    n_iter: int
    loglik: float
    step: np.ndarray | None = None
    running_away: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    unidentified: np.ndarray | None = None

    @property
    def converged(self) -> bool:
        """True when ``status`` is ``CONVERGED``."""
        return self.status == CONVERGED


def _free_columns(S: np.ndarray) -> np.ndarray:
    """The unit vectors of the columns ``S`` leaves unpenalized, shape ``(p, k)``."""
    unpenalized = ~(S != 0).any(axis=0)
    return np.eye(S.shape[0])[:, unpenalized]


def _unidentified(X: np.ndarray, free_basis: np.ndarray) -> np.ndarray:
    """The combinations ``free_basis @ v`` that ``X`` does not identify, shape
    ``(p, k)``; ``k = 0`` when it identifies every coordinate of ``free_basis``.

    The columns of ``X @ free_basis`` are scaled to unit length (so a column's
    units do not matter). A column zero on every row is returned alone; else
    each ``v`` is a unit eigenvector of their Gram matrix whose eigenvalue is at
    most ``IDENTIFIABILITY_TOL``.
    """
    free = X @ free_basis
    norms = np.linalg.norm(free, axis=0)
    if not np.all(norms > 0):
        return free_basis[:, ~(norms > 0)]
    unit = free / norms
    eigenvalues, vectors = np.linalg.eigh(unit.T @ unit)
    return free_basis @ vectors[:, eigenvalues <= IDENTIFIABILITY_TOL]


def _check_design_finite(X: np.ndarray) -> None:
    """Raise ``ValueError``, with their count, when entries of ``X`` are not
    finite."""
    if not np.all(np.isfinite(X)):
        raise ValueError(f"design not finite in {np.sum(~np.isfinite(X))} entries")


def _check_inputs(
    lik: Likelihood,
    X: np.ndarray,
    y: np.ndarray,
    aux: object,
    S: np.ndarray,
    beta0: np.ndarray,
) -> None:
    """Raise ``ValueError``, naming the problem, for inputs the fit cannot take."""
    if X.ndim != 2 or y.shape != (X.shape[0],):
        raise ValueError(f"design of shape {X.shape} for a response of shape {y.shape}")
    p = X.shape[1]
    if S.shape != (p, p) or np.shape(beta0) != (p,):
        raise ValueError(
            f"penalty of shape {S.shape} and start of shape {np.shape(beta0)} for "
            f"{p} columns"
        )
    _check_design_finite(X)
    if not np.all(np.isfinite(y)):
        raise ValueError(f"response not finite on {np.sum(~np.isfinite(y))} rows")
    if not (np.all(np.isfinite(S)) and np.all(np.isfinite(beta0))):
        raise ValueError("penalty or start not finite")
    lik.check(y, aux)


def running_away_columns(X: np.ndarray, step: np.ndarray) -> np.ndarray:
    """The columns whose part of ``step`` moves the linear predictor by more
    than ``NO_MAXIMUM_ETA_STEP`` on some row.

    Column ``j``'s part is ``max_i |X_ij step_j|``. At a stop with no finite
    maximum the columns along the direction of unbounded increase move the
    linear predictor by about one each step and the others by about the
    square root of the decrement (below ``1e-3`` on the content model's
    separated sessions); a direction that combines several columns (an arm
    column with a role column) lists each of them.

    Parameters
    ----------
    X : np.ndarray, shape (n_rows, p)
    step : np.ndarray, shape (p,)

    Returns
    -------
    np.ndarray of int, shape (k,)
        Column indices, the largest part first; parts equal but for rounding
        (``RUNNING_AWAY_TIE_DECIMALS``) in column order, so that the order
        does not depend on the machine.
    """
    part = np.max(np.abs(X * step), axis=0)
    columns = np.flatnonzero(part > NO_MAXIMUM_ETA_STEP)
    size = np.round(np.log10(part[columns]), RUNNING_AWAY_TIE_DECIMALS)
    return columns[np.argsort(-size, kind="stable")]


def _objective_one(lik, X, y, aux, weight, S, beta):
    """``F(beta) = -l(beta) + beta' S beta / 2``, ``l(beta)`` and
    ``eta = X @ beta``, of one fit."""
    eta = X @ beta
    loglik = lik._jax_loglik(eta, y, aux, weight)
    return -loglik + 0.5 * beta @ S @ beta, loglik, eta


@float64_kernel(static_argnames=("lik",))
def _evaluate(lik, X, Y, aux, weight, S, beta):
    """``F`` and ``l`` of each fit of the batch, stacked in one array of shape
    ``(2, B)``, and ``eta``, shape ``(B, n_bucket)``; ``Y`` and ``beta`` have
    a leading axis of fits."""
    F, loglik, eta = jax.vmap(
        lambda y, b: _objective_one(lik, X, y, aux, weight, S, b)
    )(Y, beta)
    return jnp.stack([F, loglik]), eta


#: The scalars of an iteration of one fit, in the order of the kernel's stack;
#: the flags among them are 0 or 1.
_SCALARS = (
    "ok",
    "decrement",
    "eta_step",
    "capped",
    "F_new",
    "loglik_new",
    "halving_ok",
)
_FLAGS = ("ok", "capped", "halving_ok")


def _iteration_one(
    lik, X, y, aux, weight, S, beta, eta, F, cap, decrement_tol, n_halvings
) -> dict[str, Any]:
    """One Newton iteration of one fit from ``beta``, whose linear predictor is
    ``eta`` and objective ``F``.

    The system: ``g`` the gradient of ``-F``, ``A = H + S``, its Cholesky
    factor (``ok`` when ``A`` and the factor are finite), the step
    ``A^-1 g``, the decrement ``g' step / 2`` and ``max |X step|``. Then, unless
    the factor failed or the decrement is below ``decrement_tol``, the step
    shortened to ``cap`` (``capped``) and halved, at most ``n_halvings``
    times, until ``F`` is finite and does not rise: the ``candidate``, its
    ``F_new`` and ``loglik_new``, and ``halving_ok`` when one was found.
    A trial's linear predictor is ``eta + scale * (X @ step)``, so the
    iteration multiplies by ``X`` twice whatever the halvings, and the
    candidate's is returned to start the next iteration. Returned as the
    scalars in one array, in the order of ``_SCALARS``, and ``A``, ``step``,
    ``candidate`` and its ``eta`` by name.
    """
    # score @ X, not X.T @ score: XLA would copy the transposed design.
    g = lik._jax_score(eta, y, aux, weight) @ X - S @ beta
    A = lik._jax_information(X, eta, y, aux, weight) + S
    factor = jax.scipy.linalg.cho_factor(A, lower=True)
    ok = jnp.all(jnp.isfinite(A)) & jnp.all(jnp.isfinite(factor[0]))
    step = jax.scipy.linalg.cho_solve(factor, g)
    decrement = 0.5 * (g @ step)
    # initial=0: a fit with no columns has no step to shorten.
    ratio = cap / jnp.max(jnp.abs(step), initial=0.0)
    eta_step = X @ step
    scale0 = jnp.where(ratio < 1.0, ratio, 1.0)  # Python's min(1.0, ratio)

    def untried(state):
        k, _, _, _, _, _, found = state
        return (k < n_halvings) & ~found

    def try_scale(state):
        k, scale, _, _, _, _, _ = state
        candidate = beta + scale * step
        eta_new = eta + scale * eta_step
        loglik_new = lik._jax_loglik(eta_new, y, aux, weight)
        F_new = -loglik_new + 0.5 * candidate @ S @ candidate
        found = jnp.isfinite(F_new) & (F_new <= F)
        return (
            k + 1,
            jnp.where(found, scale, 0.5 * scale),
            candidate,
            eta_new,
            F_new,
            loglik_new,
            found,
        )

    search = ok & ~(decrement < decrement_tol)
    start = jnp.where(search, 0, n_halvings)
    _, _, candidate, eta_new, F_new, loglik_new, found = jax.lax.while_loop(
        untried, try_scale, (start, scale0, beta, eta, F, jnp.nan * F, jnp.array(False))
    )
    scalars = {
        "ok": ok,
        "decrement": decrement,
        "eta_step": jnp.max(jnp.abs(eta_step)),
        "capped": scale0 < 1.0,
        "F_new": F_new,
        "loglik_new": loglik_new,
        "halving_ok": found,
    }
    stacked = jnp.stack(
        [jnp.asarray(scalars[name], dtype=F.dtype) for name in _SCALARS]
    )
    return stacked, {"A": A, "step": step, "candidate": candidate, "eta": eta_new}


@float64_kernel(static_argnames=("lik",))
def _iterate(lik, X, Y, aux, weight, S, beta, eta, F, cap, decrement_tol, n_halvings):
    """:func:`_iteration_one` of each fit of the batch; every output has a
    leading axis of fits (the scalars, shape ``(B, len(_SCALARS))``)."""
    return jax.vmap(
        lambda y, b, e, f: _iteration_one(
            lik, X, y, aux, weight, S, b, e, f, cap, decrement_tol, n_halvings
        )
    )(Y, beta, eta, F)


@dataclass(frozen=True)
class _Problem:
    """A batch of fits on one design, its rows padded and on the device.

    Attributes
    ----------
    lik : Likelihood
    X : np.ndarray, shape (n_rows, p)
        The unpadded design, for the design check and the columns running away.
    aux : object
        The unpadded aux, for the design check.
    S : np.ndarray, shape (p, p)
    rows : Rows
        The padded design, aux and weights as JAX arrays (no response).
    Y : jax.Array, shape (B, n_bucket)
        The padded response of each fit.
    S_device : jax.Array, shape (p, p)
    """

    lik: Likelihood
    X: np.ndarray
    aux: object
    S: np.ndarray
    rows: Rows
    Y: Any
    S_device: Any

    @classmethod
    def of(
        cls, lik: Likelihood, X: np.ndarray, y: np.ndarray, aux: object, S: np.ndarray
    ) -> _Problem:
        """One fit's problem (a batch of one)."""
        hash(lik)  # a family that cannot key a kernel fails here, as itself
        padded = lik.pad(X, y, aux)
        S = np.asarray(S, dtype=float)
        # One transfer of every array, the batch axis added in NumPy.
        X_device, aux_device, weight, Y, S_device = jax.device_put(
            (padded.X, padded.aux, padded.weight, padded.y[None], S)
        )
        rows = Rows(X_device, None, aux_device, weight, padded.n_real)
        return cls(lik, X, aux, S, rows, Y, S_device)

    def evaluate(self, beta: np.ndarray) -> tuple[np.ndarray, np.ndarray, Any]:
        """``F`` and ``l`` at ``beta`` of shape ``(B, p)``, each shape ``(B,)``,
        and ``eta``, shape ``(B, n_bucket)``, left on the device."""
        r = self.rows
        stacked, eta = _evaluate(
            self.lik, r.X, self.Y, r.aux, r.weight, self.S_device, beta
        )
        F, loglik = np.array(stacked)
        return F, loglik, eta

    def iterate(
        self, beta: np.ndarray, eta: Any, F: np.ndarray, cap: float
    ) -> _Iteration:
        """:func:`_iteration_one` at ``beta`` (B, p), whose linear predictors
        are ``eta`` (B, n_bucket; on the device) and objectives ``F`` (B,),
        the rule constants read now."""
        r = self.rows
        scalars, arrays = _iterate(
            self.lik,
            r.X,
            self.Y,
            r.aux,
            r.weight,
            self.S_device,
            beta,
            eta,
            F,
            np.float64(cap),
            np.float64(NEWTON_DECREMENT_TOL),
            np.int64(MAX_STEP_HALVINGS),
        )
        return _Iteration(np.asarray(scalars), arrays)


class _Iteration(dict):
    """One iteration of a batch (:func:`_iterate`) on the host, by name.

    The scalars of every fit (``_SCALARS``, shape ``(B,)`` each; the flags as
    bools) come over in one transfer. ``A``, ``step`` and ``candidate`` stay
    on the device until first read, since the loop reads ``A`` and ``step``
    only where a fit stops, then as NumPy arrays; the candidates' ``eta``
    stays there (:meth:`on_device`).
    """

    def __init__(self, scalars: np.ndarray, arrays: dict[str, Any]) -> None:
        super().__init__(
            (name, scalars[:, i] != 0 if name in _FLAGS else scalars[:, i])
            for i, name in enumerate(_SCALARS)
        )
        self._device = arrays

    def __missing__(self, name: str) -> np.ndarray:
        self[name] = value = np.asarray(self._device[name])
        return value

    def on_device(self, name: str) -> Any:
        """Output ``name`` as the kernel returned it, not brought over."""
        return self._device[name]


@in_float64
def penalized_newton(
    lik: Likelihood,
    X: np.ndarray,
    y: np.ndarray,
    aux: object,
    S: np.ndarray,
    beta0: np.ndarray,
    *,
    free_basis: np.ndarray | None = None,
    max_iter: int = NEWTON_MAX_STEPS,
    max_step: float = MAX_STEP_LENGTH,
) -> NewtonFit:
    """Fit ``beta`` by penalized Newton from ``beta0``.

    Parameters
    ----------
    lik : Likelihood
        The family (:mod:`.likelihoods`).
    X : np.ndarray, shape (n_rows, p)
        Design.
    y : np.ndarray, shape (n_rows,)
        Response.
    aux : object
        The family's extra data (offset, scale, group ids).
    S : np.ndarray, shape (p, p)
        The summed penalty at the chosen strengths; zeros for none.
    beta0 : np.ndarray, shape (p,)
        Start.
    free_basis : np.ndarray or None, shape (p, k)
        A basis of the coefficients the penalty leaves free, whose
        identification by ``X`` is checked before the fit: the unpenalized
        columns and the null spaces of the penalty blocks
        (:func:`.reml.reml_and_derivatives` passes it). None: the columns
        where ``S`` is zero, which misses a penalty's null space.
    max_iter : int
        Most Newton steps; the default is ``NEWTON_MAX_STEPS`` as imported.
    max_step : float
        Longest step, as the largest change of any coefficient; not applied
        to a quadratic likelihood, whose full step is the optimum The
        default is ``MAX_STEP_LENGTH`` as imported.

    Returns
    -------
    NewtonFit
        With ``status`` saying how it ended (the statuses above); a fit that
        is not ``CONVERGED`` is returned without raising.

    Raises
    ------
    ValueError
        If the inputs cannot be fitted: shapes that do not match, a design,
        response, penalty or start that is not finite, or a response or
        ``aux`` the family does not take (``Likelihood.check``).
    """
    return _fit_one(
        lik, X, y, aux, S, beta0, free_basis, max_iter=max_iter, max_step=max_step
    )[0]


def _fit_one(
    lik: Likelihood,
    X: np.ndarray,
    y: np.ndarray,
    aux: object,
    S: np.ndarray,
    beta0: np.ndarray,
    free_basis: np.ndarray | None,
    *,
    max_iter: int = NEWTON_MAX_STEPS,
    max_step: float = MAX_STEP_LENGTH,
) -> tuple[NewtonFit, _Problem]:
    """:func:`penalized_newton`, with the problem it fitted, whose padded rows
    are on the device (what REML's kernels read after the fit). Run inside
    ``runtime.in_float64``."""
    _check_inputs(lik, X, y, aux, S, beta0)
    problem = _Problem.of(lik, X, y, aux, S)
    beta = np.array(beta0, dtype=float)[None]
    return _fit(problem, beta, free_basis, max_iter, max_step)[0], problem


def _fit(
    problem: _Problem,
    beta0: np.ndarray,
    free_basis: np.ndarray | None,
    max_iter: int,
    max_step: float,
) -> list[NewtonFit]:
    """The fits of a batch from ``beta0``, shape ``(B, p)``: the host loop of
    :func:`penalized_newton`, each rule applied to every fit of the batch still
    stepping."""
    lik = problem.lik
    n_fits, p = beta0.shape
    cap = np.inf if lik.quadratic else max_step
    beta = beta0.copy()
    F, loglik, eta = problem.evaluate(beta)
    unidentified = _unidentified(
        lik.identifying_design(problem.X, problem.aux),
        _free_columns(problem.S) if free_basis is None else free_basis,
    )
    if unidentified.shape[1]:
        nan = np.full((p, p), np.nan)
        return [
            NewtonFit(
                beta[b].copy(),
                nan,
                NOT_IDENTIFIED,
                np.nan,
                0,
                float(loglik[b]),
                unidentified=unidentified,
            )
            for b in range(n_fits)
        ]
    fits: list[NewtonFit | None] = [None] * n_fits
    stepping = np.ones(n_fits, dtype=bool)
    eta_steps = np.full((n_fits, max_iter + 1), np.nan)
    capped_and_fell = np.zeros((n_fits, max_iter), dtype=bool)

    def stop(members, r, status, n_iter):
        """End each fit of ``members`` where it is, after iteration ``r``."""
        for b in np.flatnonzero(members):
            decrement = np.nan if status == NOT_POSITIVE_DEFINITE else r["decrement"][b]
            if status == NO_FINITE_MAXIMUM:
                fits[b] = _no_finite_maximum(
                    problem.X,
                    beta[b].copy(),
                    r["A"][b].copy(),
                    float(decrement),
                    n_iter,
                    float(loglik[b]),
                    r["step"][b].copy(),
                )
            else:
                fits[b] = NewtonFit(
                    beta[b].copy(),
                    r["A"][b].copy(),
                    status,
                    float(decrement),
                    n_iter,
                    float(loglik[b]),
                )

    for n_iter in range(max_iter + 1):
        r = problem.iterate(beta, eta, F, cap)
        stop(stepping & ~r["ok"], r, NOT_POSITIVE_DEFINITE, n_iter)
        small = stepping & r["ok"] & (r["decrement"] < NEWTON_DECREMENT_TOL)
        runs_away = small & (not lik.quadratic) & (r["eta_step"] > NO_MAXIMUM_ETA_STEP)
        stop(runs_away, r, NO_FINITE_MAXIMUM, n_iter)
        at_maximum = small & ~runs_away
        if at_maximum.any():
            last = _last_step(problem, beta, r, at_maximum, n_iter, cap)
            for b in np.flatnonzero(at_maximum):
                fits[b] = last(b)
        stepping &= r["ok"] & ~(r["decrement"] < NEWTON_DECREMENT_TOL)
        eta_steps[stepping, n_iter] = r["eta_step"][stepping]
        if n_iter == max_iter or not stepping.any():
            break
        stop(stepping & ~r["halving_ok"], r, HALVING_FAILED, n_iter)
        stepping &= r["halving_ok"]
        if not stepping.any():
            break
        capped_and_fell[stepping, n_iter] = (r["capped"] & (r["F_new"] < F))[stepping]
        beta[stepping] = r["candidate"][stepping]
        # The candidates' eta stays on the device (the next iteration's start).
        eta = (
            r.on_device("eta")
            if stepping.all()
            else jnp.where(stepping[:, None], r.on_device("eta"), eta)
        )
        F[stepping] = r["F_new"][stepping]
        loglik[stepping] = r["loglik_new"][stepping]
    if stepping.any():
        away = (not lik.quadratic) & _runs_away_at_the_cap(capped_and_fell, eta_steps)
        stop(stepping & away, r, NO_FINITE_MAXIMUM, n_iter)
        stop(stepping & ~away, r, STEP_LIMIT, n_iter)
    return fits


def _no_finite_maximum(
    X: np.ndarray,
    beta: np.ndarray,
    A: np.ndarray,
    decrement: float,
    n_iter: int,
    loglik: float,
    step: np.ndarray,
) -> NewtonFit:
    """A fit with no finite maximum, with its last step and the columns
    running away along it."""
    return NewtonFit(
        beta,
        A,
        NO_FINITE_MAXIMUM,
        decrement,
        n_iter,
        loglik,
        step,
        running_away_columns(X, step),
    )


def _runs_away_at_the_cap(
    capped_and_fell: np.ndarray, eta_steps: np.ndarray
) -> np.ndarray:
    """Per fit of a batch that ran out of steps, whether it was running away at
    the step cap, shape ``(B,)``.

    ``capped_and_fell`` has shape ``(B, max_iter)``, one entry per step taken;
    ``eta_steps`` has shape ``(B, max_iter + 1)``, the full step's
    ``max |X step|`` at each iteration, the last one untaken.

    True when each of the last ``NO_MAXIMUM_CAPPED_STEPS`` steps was shortened
    to ``max_step`` and lowered ``F``, and the full Newton step's movement of
    the linear predictor (``max |X step|``) did not shrink from one step to the
    next, through the step left untaken. Along a direction without a finite
    maximum the curvature vanishes and the full step grows while the cap keeps
    the progress linear; toward a finite maximum, however far, the full step
    shrinks as the fit approaches it.
    """
    k = NO_MAXIMUM_CAPPED_STEPS
    n_fits, n_steps = capped_and_fell.shape
    if n_steps < k:
        return np.zeros(n_fits, dtype=bool)
    growing = np.diff(eta_steps[:, -(k + 1) :], axis=1) >= 0.0
    return capped_and_fell[:, -k:].all(axis=1) & growing.all(axis=1)


def _last_step(
    problem: _Problem,
    beta: np.ndarray,
    r: dict,
    members: np.ndarray,
    n_iter: int,
    cap: float,
):
    """The converged fits of ``members`` (a mask over the batch), each moved
    by its full Newton step ``r["step"]`` when that is safe.

    ``beta`` are the batch's coefficients and ``r`` the iteration at
    ``beta``, whose decrement is below ``NEWTON_DECREMENT_TOL`` for every
    member. ``F`` is evaluated afresh at ``beta`` and at the moved point, by
    the evaluation kernel on ``X @ beta`` both times (not on the iteration's
    carried linear predictor), so that the comparison is of like with like,
    and the log-likelihood returned is that evaluation's. A member's step is
    taken when ``F`` does not rise by more than its rounding,
    ``8 eps max(1, |F|)``, and it lands where ``A`` is positive definite and
    the decrement is below ``NEWTON_DECREMENT_TOL``; otherwise the fit stays
    at ``beta``. A gain below the rounding of ``F`` is still taken: without it
    a coefficient would stay off by up to ``sqrt(2 eps |F| / lambda_min)``,
    about 1e-7 on a fit with ``F`` near 400, which the REML score inherits
    through ``log|A|``.

    Returns
    -------
    callable
        The member's index to its ``NewtonFit``.
    """
    F, loglik, _ = problem.evaluate(beta)
    moved = np.where(members[:, None], beta + r["step"], beta)
    F_new, loglik_new, eta_moved = problem.evaluate(moved)
    rounding = 8.0 * np.finfo(float).eps * np.maximum(1.0, np.abs(F))
    take = members & np.isfinite(F_new) & (F_new <= F + rounding)
    if take.any():
        landed = problem.iterate(moved, eta_moved, F_new, cap)
        take &= landed["ok"] & (landed["decrement"] < NEWTON_DECREMENT_TOL)

    def fit(b: int) -> NewtonFit:
        if take[b]:
            return NewtonFit(
                moved[b].copy(),
                landed["A"][b].copy(),
                CONVERGED,
                float(landed["decrement"][b]),
                n_iter + 1,
                float(loglik_new[b]),
            )
        return NewtonFit(
            beta[b].copy(),
            r["A"][b].copy(),
            CONVERGED,
            float(r["decrement"][b]),
            n_iter,
            float(loglik[b]),
        )

    return fit
