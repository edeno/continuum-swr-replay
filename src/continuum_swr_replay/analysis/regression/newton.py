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

Before the first step the design is checked: the coefficients the penalty
leaves free (the unpenalized columns and the penalties' null spaces) must be
identified by ``X``. A small decrement is not enough to stop: when the step it
belongs to still moves the linear predictor by more than
``NO_MAXIMUM_ETA_STEP`` on some row, the likelihood has no finite maximum
along that step (a 0/1 column whose rows all have the same outcome, a Poisson
cell with no events): the decrement shrinks to zero while a coefficient runs
to infinity. Near a finite maximum the step shrinks with the decrement.

Once the decrement is below the tolerance the fit still takes that last full
step, and reports ``A`` and the decrement where it lands. Stopping before it
would leave a coefficient off by up to about ``sqrt(2 tol / lambda_min)``
(``lambda_min`` the smallest eigenvalue of ``A``): 4e-5 on a content fit of 80
events. After it the error is of the order of the square of that.

Every fit ends with one of the statuses below; only ``CONVERGED`` is a fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.linalg

from .constants import (
    IDENTIFIABILITY_TOL,
    MAX_STEP_HALVINGS,
    MAX_STEP_LENGTH,
    NEWTON_DECREMENT_TOL,
    NEWTON_MAX_STEPS,
    NO_MAXIMUM_ETA_STEP,
)
from .likelihoods import Likelihood

__all__ = [
    "CONVERGED",
    "FIT_STATUSES",
    "HALVING_FAILED",
    "NOT_IDENTIFIED",
    "NOT_POSITIVE_DEFINITE",
    "NO_FINITE_MAXIMUM",
    "STEP_LIMIT",
    "FitStatus",
    "NewtonFit",
    "penalized_newton",
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
#: predictor by more than ``NO_MAXIMUM_ETA_STEP``: a coefficient runs to
#: infinity.
NO_FINITE_MAXIMUM: FitStatus = "no finite maximum"
FIT_STATUSES: tuple[FitStatus, ...] = (
    CONVERGED,
    NOT_IDENTIFIED,
    NOT_POSITIVE_DEFINITE,
    HALVING_FAILED,
    STEP_LIMIT,
    NO_FINITE_MAXIMUM,
)


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
    """

    beta: np.ndarray
    A: np.ndarray
    status: FitStatus
    newton_decrement: float
    n_iter: int
    loglik: float

    @property
    def converged(self) -> bool:
        """True when ``status`` is ``CONVERGED``."""
        return self.status == CONVERGED


def _free_columns(S: np.ndarray) -> np.ndarray:
    """The unit vectors of the columns ``S`` leaves unpenalized, shape ``(p, k)``."""
    unpenalized = ~(S != 0).any(axis=0)
    return np.eye(S.shape[0])[:, unpenalized]


def _identified(X: np.ndarray, free_basis: np.ndarray) -> bool:
    """Whether ``X`` identifies the coefficients spanned by ``free_basis``.

    The columns of ``X @ free_basis`` are scaled to unit length (so a column's
    units do not matter) and their Gram matrix's smallest eigenvalue must
    exceed ``IDENTIFIABILITY_TOL``; a column that is zero on every row fails.
    """
    if free_basis.shape[1] == 0:
        return True
    free = X @ free_basis
    norms = np.linalg.norm(free, axis=0)
    if not np.all(norms > 0):
        return False
    unit = free / norms
    return bool(np.linalg.eigvalsh(unit.T @ unit)[0] > IDENTIFIABILITY_TOL)


def _eta_step(X: np.ndarray, step: np.ndarray) -> float:
    """``max |X @ step|``: the most a step moves the linear predictor on a row."""
    return float(np.max(np.abs(X @ step)))


def penalized_newton(
    lik: Likelihood,
    X: np.ndarray,
    y: np.ndarray,
    aux: object,
    S: np.ndarray,
    beta0: np.ndarray,
    *,
    free_basis: np.ndarray | None = None,
    tol: float = NEWTON_DECREMENT_TOL,
    max_iter: int = NEWTON_MAX_STEPS,
    max_halvings: int = MAX_STEP_HALVINGS,
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
    tol : float
        Stop when the Newton decrement is below this.
    max_iter : int
        Most Newton steps.
    max_halvings : int
        Most halvings of one step.
    max_step : float
        Longest step, as the largest change of any coefficient; not applied
        to a quadratic likelihood, whose full step is the optimum.

    Returns
    -------
    NewtonFit
        With ``status`` saying how it ended (:data:`FIT_STATUSES`); a fit that
        is not ``CONVERGED`` is returned without raising.
    """
    beta = np.array(beta0, dtype=float)
    cap = np.inf if lik.quadratic else max_step

    def objective(b: np.ndarray) -> tuple[float, float]:
        loglik = lik.loglik(X @ b, y, aux)
        return -loglik + 0.5 * b @ S @ b, loglik

    F, loglik = objective(beta)
    if not _identified(X, _free_columns(S) if free_basis is None else free_basis):
        nan = np.full((beta.size, beta.size), np.nan)
        return NewtonFit(beta, nan, NOT_IDENTIFIED, np.nan, 0, loglik)
    for n_iter in range(max_iter + 1):
        eta = X @ beta
        g = X.T @ lik.score_eta(eta, y, aux) - S @ beta
        A = lik.information(X, eta, y, aux) + S
        try:
            factor = scipy.linalg.cho_factor(A, lower=True)
        except (np.linalg.LinAlgError, ValueError):  # not PD, or not finite
            return NewtonFit(beta, A, NOT_POSITIVE_DEFINITE, np.nan, n_iter, loglik)
        step = scipy.linalg.cho_solve(factor, g)
        decrement = 0.5 * float(g @ step)
        if decrement < tol:
            if _eta_step(X, step) > NO_MAXIMUM_ETA_STEP:
                return NewtonFit(beta, A, NO_FINITE_MAXIMUM, decrement, n_iter, loglik)
            return _last_step(
                lik,
                X,
                y,
                aux,
                S,
                NewtonFit(beta, A, CONVERGED, decrement, n_iter, loglik),
                step,
            )
        if n_iter == max_iter:
            break
        scale = min(1.0, cap / np.max(np.abs(step)))
        for _ in range(max_halvings):
            candidate = beta + scale * step
            F_new, loglik_new = objective(candidate)
            if np.isfinite(F_new) and F_new <= F:
                break
            scale *= 0.5
        else:
            return NewtonFit(beta, A, HALVING_FAILED, decrement, n_iter, loglik)
        beta, F, loglik = candidate, F_new, loglik_new
    return NewtonFit(beta, A, STEP_LIMIT, decrement, n_iter, loglik)


def _last_step(
    lik: Likelihood,
    X: np.ndarray,
    y: np.ndarray,
    aux: object,
    S: np.ndarray,
    fit: NewtonFit,
    step: np.ndarray,
) -> NewtonFit:
    """The converged ``fit`` moved by its full Newton ``step``.

    The step is taken when it lowers ``F`` or, when its gain is below the
    rounding of ``F`` (``F`` cannot tell the two points apart), when it lowers
    the decrement: otherwise a coefficient would stay off by up to
    ``sqrt(2 eps |F| / lambda_min)``, about 1e-7 on a fit with ``F`` near 400,
    which the REML score inherits through ``log|A|``. ``fit`` is returned
    unchanged when neither holds or the step lands where ``A`` is not
    positive definite.
    """
    beta = fit.beta + step
    eta = X @ beta
    loglik = lik.loglik(eta, y, aux)
    F_old = -fit.loglik + 0.5 * fit.beta @ S @ fit.beta
    F_new = -loglik + 0.5 * beta @ S @ beta
    if not np.isfinite(F_new):
        return fit
    g = X.T @ lik.score_eta(eta, y, aux) - S @ beta
    A = lik.information(X, eta, y, aux) + S
    try:
        factor = scipy.linalg.cho_factor(A, lower=True)
    except (np.linalg.LinAlgError, ValueError):  # not PD, or not finite
        return fit
    decrement = 0.5 * float(g @ scipy.linalg.cho_solve(factor, g))
    if not (F_new <= F_old or decrement < fit.newton_decrement):
        return fit
    return NewtonFit(beta, A, CONVERGED, decrement, fit.n_iter + 1, loglik)
