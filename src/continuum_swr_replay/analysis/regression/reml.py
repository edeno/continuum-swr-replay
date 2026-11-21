"""Penalty strengths by restricted maximum likelihood (REML), with exact derivatives.

A model has ``m`` quadratic penalties on disjoint sets of coefficients (one per
smooth term), with strengths ``lam_j = exp(rho_j)``. For one session the
Laplace-approximate REML score is

    V(rho) = -l(b) + b' S b / 2 - log|S|_+ / 2 + log|H + S| / 2 - M_p log(2 pi) / 2

with ``b`` the penalized fit at ``rho`` (:func:`.newton.penalized_newton`),
``S = sum_j lam_j S_j``, ``H`` the information at ``b`` and ``M_p`` the
dimension of the penalty's null space. Because the blocks are disjoint,
``log|S|_+ = sum_j (r_j rho_j + log|S_j|_+)`` with ``r_j`` the rank of
``S_j``. ``V`` is mgcv's REML score less the constants the likelihoods leave
out (for the Poisson, ``sum log y!``), given the same ``S_j``. The gradient and
Hessian in ``rho`` are exact (they differentiate ``b`` and ``H`` through the
optimum, with the likelihood's ``dinformation`` and ``d2information``).

The Gaussian scale is profiled: its penalty is on the scale of the residual sum
of squares, so the fit minimizes ``||y - X b||^2 + b' S b`` (``lam_j`` are
mgcv's ``sp``), the scale is ``phi = D_p / (n - M_p)`` with ``D_p`` that
minimum, and ``V`` is Wood (2011, eq. 4) at that scale; it equals the Laplace
score above at the penalty ``S / phi``.

Pooled REML shares the strengths across sessions, each with its own
coefficients: the pooled score, gradient and Hessian are the sums over
sessions, and one session whose inner fit fails makes the pooled score
infinite; the score records which session and the fit's status, and every
error names them. A session may leave out unpenalized columns it cannot
estimate (``SessionData.active``): its fit is that of its design without them.
:func:`select_penalties` starts from the best point of a grid (the score can
have two optima) and refines it by Newton on ``rho`` with the exact Hessian;
:func:`fit_at_penalties` fits at given strengths without a search. A search
that does not converge raises.
"""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import scipy.linalg

from .constants import (
    AT_BOUNDARY_MARGIN,
    LOG_PENALTY_BOUNDS,
    MAX_STEP_HALVINGS,
    NULL_SPACE_EDF_TOL,
    PENALTY_ROUNDING_TOL,
    REML_DECREMENT_TOL,
    REML_EIGEN_FLOOR,
    REML_GRID_CENTER,
    REML_GRID_PASSES,
    REML_GRID_POINTS,
    REML_MAX_STEPS,
)
from .likelihoods import Gaussian, Likelihood
from .newton import FitStatus, NewtonFit, penalized_newton
from .pspline import penalty_rank

__all__ = [
    "PenaltyBlock",
    "PenaltySearch",
    "PooledFit",
    "PooledScore",
    "RemlScore",
    "SessionData",
    "SessionFit",
    "assemble_penalty",
    "fit_at_penalties",
    "free_basis",
    "pooled_reml",
    "refine_penalties",
    "reml_and_derivatives",
    "select_penalties",
]

_LOG_2PI = float(np.log(2.0 * np.pi))


@dataclass(frozen=True)
class PenaltyBlock:
    """One quadratic penalty on a set of coefficients.

    Attributes
    ----------
    columns : np.ndarray, shape (k,)
        The coefficients (columns of the design) it penalizes.
    S : np.ndarray, shape (k, k)
        The penalty matrix, symmetric positive semi-definite.
    rank : int
        ``penalty_rank(S)``, set on construction.
    log_pdet : float
        ``log|S|_+``, the sum of the logs of its ``rank`` largest
        eigenvalues, set on construction.
    null_space : np.ndarray, shape (k, k - rank)
        An orthonormal basis of the null space of ``S`` (for a P-spline, the
        straight lines), set on construction.
    """

    columns: np.ndarray
    S: np.ndarray
    rank: int = field(init=False)
    log_pdet: float = field(init=False)
    null_space: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        columns = np.asarray(self.columns, dtype=int)
        S = np.asarray(self.S, dtype=float)
        if S.shape != (columns.size, columns.size):
            raise ValueError(f"penalty of shape {S.shape} for {columns.size} columns")
        scale = float(np.max(np.abs(S))) if S.size else 0.0
        if np.max(np.abs(S - S.T), initial=0.0) > PENALTY_ROUNDING_TOL * scale:
            raise ValueError("penalty matrix is not symmetric")
        rank = penalty_rank(S)
        eigenvalues, vectors = np.linalg.eigh(S)
        if eigenvalues.size and eigenvalues[0] < -PENALTY_ROUNDING_TOL * max(
            abs(eigenvalues[0]), abs(eigenvalues[-1])
        ):
            raise ValueError(
                f"penalty matrix is not positive semidefinite (eigenvalue "
                f"{eigenvalues[0]:.3g})"
            )
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "S", S)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(
            self,
            "log_pdet",
            float(np.log(eigenvalues[eigenvalues.size - rank :]).sum()),
        )
        object.__setattr__(self, "null_space", vectors[:, : eigenvalues.size - rank])


@dataclass(frozen=True)
class SessionData:
    """One session's data.

    Attributes
    ----------
    X : np.ndarray, shape (n_rows, p)
        Design.
    y : np.ndarray, shape (n_rows,)
        Response.
    aux : object
        The family's extra data (offset, group ids); None for the Bernoulli
        and the Gaussian (whose scale is profiled).
    beta0 : np.ndarray or None, shape (p,)
        Start of the first inner fit; zeros when None.
    name : str or None
        A label for messages (the session's key); its index when None.
    active : np.ndarray of bool or None, shape (p,)
        The columns this session estimates; all when None. An unpenalized
        column the session cannot estimate (a 0/1 term with no support in it)
        is inactive: the session's fit leaves it out, and its coefficient and
        covariance rows and columns are NaN. A column in a penalty block may
        not be inactive (the penalty identifies it).
    """

    X: np.ndarray
    y: np.ndarray
    aux: object = None
    beta0: np.ndarray | None = None
    name: str | None = None
    active: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.active is not None:
            active = np.asarray(self.active)
            if active.dtype != bool or active.shape != (self.X.shape[1],):
                raise ValueError(
                    f"active must be a boolean mask over the {self.X.shape[1]} "
                    f"columns, not {active.dtype} of shape {active.shape}"
                )
            object.__setattr__(self, "active", active)


@dataclass(frozen=True)
class RemlScore:
    """One session's REML score at ``rho`` (:func:`reml_and_derivatives`).

    Attributes
    ----------
    score : float
        ``V(rho)``; inf when the inner fit did not converge.
    gradient : np.ndarray or None, shape (m,)
        ``dV/drho``; None when not asked for or the score is infinite.
    hessian : np.ndarray or None, shape (m, m)
        ``d2V/drho2``; None likewise.
    fit : NewtonFit
        The inner fit. For the Gaussian, ``A`` is ``(X'X + S) / scale`` and
        ``loglik`` is at the profiled scale.
    scale : float
        The profiled Gaussian scale; 1 for the other families.
    """

    score: float
    gradient: np.ndarray | None
    hessian: np.ndarray | None
    fit: NewtonFit
    scale: float


@dataclass(frozen=True)
class PooledScore:
    """The pooled REML score at ``rho`` (:func:`pooled_reml`).

    Attributes
    ----------
    score : float
        Sum of the sessions' scores; inf when any inner fit failed.
    gradient, hessian : np.ndarray or None
        Sums of the sessions', shapes (m,) and (m, m); None when not asked for
        or the score is infinite.
    sessions : tuple of RemlScore
        One per session, in order; empty when the score is infinite (the
        sessions after the first failure are not evaluated). A session's
        ``fit.beta`` and ``fit.A`` are NaN at its inactive columns.
    failed_session : int or None
        The index of the first session whose inner fit failed; None when the
        score is finite.
    failed_status : FitStatus or None
        That fit's status.
    failure : str or None
        ``failed_session`` and ``failed_status`` in words, with the session's
        name when it has one.
    """

    score: float
    gradient: np.ndarray | None
    hessian: np.ndarray | None
    sessions: tuple[RemlScore, ...]
    failed_session: int | None = None
    failed_status: FitStatus | None = None
    failure: str | None = None


@dataclass(frozen=True)
class SessionFit:
    """One session's fit at the chosen strengths.

    Attributes
    ----------
    fit : NewtonFit
        The inner fit (``beta``, ``A``, convergence).
    covariance : np.ndarray, shape (p, p)
        ``A^-1``, the Bayesian covariance of ``beta`` (mgcv's ``Vp``).
    edf : np.ndarray, shape (m,)
        Effective degrees of freedom per penalty block: the trace of the
        block's rows and columns of ``A^-1 H`` (mgcv's ``edf``, summed over
        the term's coefficients).
    edf_total : float
        ``tr(A^-1 H)`` over all coefficients.
    scale : float
        The profiled Gaussian scale; 1 for the other families.
    score : float
        The session's REML score.
    in_null_space : np.ndarray of bool, shape (m,)
        True for a term shrunk to its penalty's null space (a P-spline drawn
        as a straight line): its edf is within ``NULL_SPACE_EDF_TOL`` of the
        null space's dimension within the term's columns (the number of
        columns less the penalty's rank). The search can stop on the score's
        plateau before the bound, so ``PenaltySearch.at_boundary`` misses
        such a term.
    """

    fit: NewtonFit
    covariance: np.ndarray
    edf: np.ndarray
    edf_total: float
    scale: float
    score: float
    in_null_space: np.ndarray


@dataclass(frozen=True)
class PenaltySearch:
    """How :func:`select_penalties` or :func:`refine_penalties` found ``rho``.

    There is one only for a converged search (the Newton decrement on the
    log penalties below ``REML_DECREMENT_TOL``; a term at a bound whose
    gradient points out of the bounds is held there): a search that does not
    converge raises.

    Attributes
    ----------
    start_rho : np.ndarray, shape (m,)
        Where Newton started: the best grid point, or the given start.
    start_score : float
        The pooled score there.
    n_grid_points : int
        Grid points evaluated (0 for a given start).
    n_grid_not_finite : int
        Grid points whose pooled score was not finite (an inner fit failed).
    grid_failures : dict of str to int
        Those points counted by the status of their first failing session's
        fit.
    n_outer_steps : int
        Newton steps on ``rho`` taken.
    at_boundary : np.ndarray of bool, shape (m,)
        True for a term whose log penalty is within ``AT_BOUNDARY_MARGIN`` of
        a bound.
    """

    start_rho: np.ndarray
    start_score: float
    n_grid_points: int
    n_grid_not_finite: int
    grid_failures: dict[str, int]
    n_outer_steps: int
    at_boundary: np.ndarray


@dataclass(frozen=True)
class PooledFit:
    """The sessions' fits at shared penalty strengths.

    Attributes
    ----------
    rho : np.ndarray, shape (m,)
        Log penalty strengths (for the Gaussian, on the residual-sum-of-squares
        scale).
    score : float
        Pooled REML score at ``rho``.
    gradient : np.ndarray or None, shape (m,)
        Its gradient there; None when the strengths were given.
    sessions : tuple of SessionFit
        One per session, in order.
    search : PenaltySearch or None
        None when the strengths were given (:func:`fit_at_penalties`).
    """

    rho: np.ndarray
    score: float
    gradient: np.ndarray | None
    sessions: tuple[SessionFit, ...]
    search: PenaltySearch | None


def _check_blocks(blocks: Sequence[PenaltyBlock], n_columns: int) -> None:
    """Raise when a block's columns are outside the design or blocks overlap."""
    seen: set[int] = set()
    for block in blocks:
        columns = block.columns.tolist()
        if any(c < 0 or c >= n_columns for c in columns):
            raise ValueError(f"penalty columns outside the {n_columns} design columns")
        if len(set(columns)) != len(columns) or seen.intersection(columns):
            raise ValueError("penalty blocks overlap; each needs its own columns")
        seen.update(columns)


def _embedded(block: PenaltyBlock, strength: float, n_columns: int) -> np.ndarray:
    """``strength * S_j`` embedded in a ``(p, p)`` matrix."""
    out = np.zeros((n_columns, n_columns))
    out[np.ix_(block.columns, block.columns)] = strength * block.S
    return out


def assemble_penalty(
    blocks: Sequence[PenaltyBlock], rho: Sequence[float], n_columns: int
) -> np.ndarray:
    """``sum_j exp(rho_j) S_j``, embedded, shape ``(n_columns, n_columns)``."""
    _check_blocks(blocks, n_columns)
    rho = np.asarray(rho, dtype=float)
    if rho.shape != (len(blocks),):
        raise ValueError(f"{rho.size} log penalties for {len(blocks)} blocks")
    out = np.zeros((n_columns, n_columns))
    for block, r in zip(blocks, rho, strict=True):
        out[np.ix_(block.columns, block.columns)] += np.exp(r) * block.S
    return out


def free_basis(blocks: Sequence[PenaltyBlock], n_columns: int) -> np.ndarray:
    """The coefficients no penalty touches: the unpenalized columns and each
    block's null space, embedded, shape ``(n_columns, k)``.

    The design must identify them (:func:`.newton.penalized_newton` checks);
    a penalty identifies the rest.
    """
    penalized = np.zeros(n_columns, dtype=bool)
    parts = []
    for block in blocks:
        penalized[block.columns] = True
        embedded = np.zeros((n_columns, block.null_space.shape[1]))
        embedded[block.columns] = block.null_space
        parts.append(embedded)
    return np.hstack([np.eye(n_columns)[:, ~penalized], *parts])


def _log_pdet(blocks: Sequence[PenaltyBlock], rho: np.ndarray) -> float:
    """``log|S|_+`` of the disjoint blocks at ``rho``."""
    return float(sum(b.rank * r + b.log_pdet for b, r in zip(blocks, rho, strict=True)))


def reml_and_derivatives(
    lik: Likelihood,
    X: np.ndarray,
    y: np.ndarray,
    aux: object,
    blocks: Sequence[PenaltyBlock],
    rho: Sequence[float],
    beta0: np.ndarray | None = None,
    *,
    derivatives: bool = True,
) -> RemlScore:
    """One session's REML score at ``rho``, with its gradient and Hessian.

    Parameters
    ----------
    lik : Likelihood
        The family. For :class:`.likelihoods.Gaussian` the scale is profiled
        and ``aux`` must be None.
    X : np.ndarray, shape (n_rows, p)
    y : np.ndarray, shape (n_rows,)
    aux : object
        The family's extra data.
    blocks : sequence of PenaltyBlock, length m
        Disjoint penalties.
    rho : sequence of float, length m
        Log penalty strengths.
    beta0 : np.ndarray or None, shape (p,)
        Start of the inner fit; zeros when None.
    derivatives : bool
        Compute the gradient and Hessian (the score alone needs only the fit).

    Returns
    -------
    RemlScore
        Score inf, no derivatives, when the inner fit did not converge.

    Raises
    ------
    ValueError
        If the blocks overlap or leave the design, or ``aux`` is given for the
        Gaussian.
    """
    rho = np.asarray(rho, dtype=float)
    n_columns = X.shape[1]
    S = assemble_penalty(blocks, rho, n_columns)
    start = np.zeros(n_columns) if beta0 is None else beta0
    if isinstance(lik, Gaussian):
        if aux is not None:
            raise ValueError("the Gaussian scale is profiled; pass aux=None")
        return _gaussian_profiled(X, y, blocks, rho, S, start, derivatives)

    fit = penalized_newton(
        lik, X, y, aux, S, start, free_basis=free_basis(blocks, n_columns)
    )
    if not fit.converged:
        return RemlScore(np.inf, None, None, fit, 1.0)
    b, A = fit.beta, fit.A
    factor = scipy.linalg.cho_factor(A, lower=True)
    null_dim = n_columns - sum(block.rank for block in blocks)
    score = (
        -fit.loglik
        + 0.5 * b @ S @ b
        - 0.5 * _log_pdet(blocks, rho)
        + float(np.log(np.diag(factor[0])).sum())
        - 0.5 * null_dim * _LOG_2PI
    )
    if not derivatives:
        return RemlScore(float(score), None, None, fit, 1.0)

    eta = X @ b
    m = len(blocks)
    S_j = [
        _embedded(block, np.exp(r), n_columns)
        for block, r in zip(blocks, rho, strict=True)
    ]
    S_j_b = [Sj @ b for Sj in S_j]
    b_j = [-scipy.linalg.cho_solve(factor, Sjb) for Sjb in S_j_b]
    A_j = [
        Sj + lik.dinformation(X, eta, y, aux, X @ bj)
        for Sj, bj in zip(S_j, b_j, strict=True)
    ]
    Ainv_A_j = [scipy.linalg.cho_solve(factor, Aj) for Aj in A_j]
    gradient = 0.5 * np.array(
        [b @ S_j_b[j] - blocks[j].rank + np.trace(Ainv_A_j[j]) for j in range(m)]
    )
    hessian = np.empty((m, m))
    for j in range(m):
        for k in range(j, m):
            b_jk = -scipy.linalg.cho_solve(factor, A_j[k] @ b_j[j] + S_j[j] @ b_j[k])
            if j == k:
                b_jk += b_j[j]
            A_jk = lik.d2information(
                X, eta, y, aux, X @ b_j[j], X @ b_j[k]
            ) + lik.dinformation(X, eta, y, aux, X @ b_jk)
            if j == k:
                A_jk += S_j[j]
            hessian[j, k] = hessian[k, j] = 0.5 * (
                (b @ S_j_b[j] if j == k else 0.0)
                + 2.0 * b_j[k] @ S_j_b[j]
                - np.sum(Ainv_A_j[k] * Ainv_A_j[j].T)
                + np.trace(scipy.linalg.cho_solve(factor, A_jk))
            )
    return RemlScore(float(score), gradient, hessian, fit, 1.0)


def _gaussian_profiled(
    X: np.ndarray,
    y: np.ndarray,
    blocks: Sequence[PenaltyBlock],
    rho: np.ndarray,
    S: np.ndarray,
    beta0: np.ndarray,
    derivatives: bool,
) -> RemlScore:
    """The Gaussian score with the scale profiled out (Wood 2011, eq. 4).

    ``b`` minimizes ``D_p = ||y - X b||^2 + b' S b``, ``phi = D_p / (n - M_p)``
    and

        V = (n - M_p) (1 + log(2 pi phi)) / 2 - log|S|_+ / 2 + log|X'X + S| / 2.

    The Newton fit that finds ``b`` runs at the working scale ``var(y)`` with
    the penalty ``S / var(y)`` (the same minimizer), so that its decrement, and
    the tolerance it is held to, do not depend on the outcome's units.

    ``D_p`` has derivatives ``b' S_j b`` and ``delta_jk b' S_j b + 2 b_k' S_j b``
    (``b_k = -(X'X + S)^-1 S_k b``), as in the Laplace case with ``H`` fixed.
    """
    n_rows, n_columns = X.shape
    working = float(np.var(y))
    pls = penalized_newton(
        Gaussian(),
        X,
        y,
        working,
        S / working,
        beta0,
        free_basis=free_basis(blocks, n_columns),
    )
    if not pls.converged:
        return RemlScore(np.inf, None, None, pls, np.nan)
    b = pls.beta
    gram = X.T @ X + S
    residual = y - X @ b
    S_b = S @ b
    D_p = float(residual @ residual + b @ S_b)
    dof = n_rows - (n_columns - sum(block.rank for block in blocks))
    scale = D_p / dof
    factor = scipy.linalg.cho_factor(gram, lower=True)
    score = (
        0.5 * dof * (1.0 + _LOG_2PI + np.log(scale))
        - 0.5 * _log_pdet(blocks, rho)
        + float(np.log(np.diag(factor[0])).sum())
    )
    fit = replace(
        pls,
        A=gram / scale,
        newton_decrement=pls.newton_decrement * working / scale,
        loglik=Gaussian().loglik(X @ b, y, scale),
    )
    if not derivatives:
        return RemlScore(float(score), None, None, fit, scale)

    m = len(blocks)
    S_j = [
        _embedded(block, np.exp(r), n_columns)
        for block, r in zip(blocks, rho, strict=True)
    ]
    S_j_b = [Sj @ b for Sj in S_j]
    b_j = [-scipy.linalg.cho_solve(factor, Sjb) for Sjb in S_j_b]
    Ainv_S_j = [scipy.linalg.cho_solve(factor, Sj) for Sj in S_j]
    D_j = np.array([b @ Sjb for Sjb in S_j_b])
    D_jk = np.diag(D_j) + 2.0 * np.array(
        [[b_j[k] @ S_j_b[j] for k in range(m)] for j in range(m)]
    )
    D_jk = 0.5 * (D_jk + D_jk.T)
    trace_j = np.array([np.trace(M) for M in Ainv_S_j])
    trace_jk = np.array(
        [[np.sum(Ainv_S_j[k] * Ainv_S_j[j].T) for k in range(m)] for j in range(m)]
    )
    ranks = np.array([block.rank for block in blocks], dtype=float)
    gradient = 0.5 * (dof * D_j / D_p - ranks + trace_j)
    hessian = 0.5 * (
        dof * (D_jk / D_p - np.outer(D_j, D_j) / D_p**2) + np.diag(trace_j) - trace_jk
    )
    return RemlScore(float(score), gradient, hessian, fit, scale)


def _label(data: SessionData, index: int) -> str:
    """``session <index>``, with the session's name when it has one."""
    return f"session {index}" + ("" if data.name is None else f" ({data.name})")


def _check_active(
    data: SessionData, index: int, blocks: Sequence[PenaltyBlock]
) -> None:
    """Raise when a column of a penalty block is inactive in the session."""
    if data.active is None:
        return
    penalized = np.concatenate([np.empty(0, int), *(b.columns for b in blocks)])
    inactive = penalized[~data.active[penalized]]
    if inactive.size:
        raise ValueError(
            f"{_label(data, index)}: column {inactive[0]} is in a penalty block and "
            "may not be inactive (the penalty identifies it)"
        )


def _session_score(
    lik: Likelihood,
    data: SessionData,
    blocks: Sequence[PenaltyBlock],
    rho: Sequence[float],
    start: np.ndarray | None,
    derivatives: bool,
) -> RemlScore:
    """One session's score, its fit left without the inactive columns.

    The returned fit's ``beta`` and ``A`` are NaN at the inactive columns.
    """
    if data.active is None or data.active.all():
        return reml_and_derivatives(
            lik, data.X, data.y, data.aux, blocks, rho, start, derivatives=derivatives
        )
    active = data.active
    position = np.cumsum(active) - 1  # a column's index among the active ones
    reduced = [replace(block, columns=position[block.columns]) for block in blocks]
    session = reml_and_derivatives(
        lik,
        data.X[:, active],
        data.y,
        data.aux,
        reduced,
        rho,
        None if start is None else start[active],
        derivatives=derivatives,
    )
    p = active.size
    beta = np.full(p, np.nan)
    beta[active] = session.fit.beta
    A = np.full((p, p), np.nan)
    A[np.ix_(active, active)] = session.fit.A
    return replace(session, fit=replace(session.fit, beta=beta, A=A))


def pooled_reml(
    lik: Likelihood,
    sessions: Sequence[SessionData],
    blocks: Sequence[PenaltyBlock],
    rho: Sequence[float],
    starts: Sequence[np.ndarray] | None = None,
    *,
    derivatives: bool = True,
) -> PooledScore:
    """The pooled REML score at ``rho``: the sum over sessions.

    Parameters
    ----------
    lik : Likelihood
    sessions : sequence of SessionData
        Each with its own coefficients; the blocks apply to every session.
        A session's inactive columns are left out of its fit.
    blocks : sequence of PenaltyBlock, length m
    rho : sequence of float, length m
    starts : sequence of np.ndarray or None
        One inner-fit start per session (warm starts), shape ``(p,)`` with
        anything at the inactive columns; each session's ``beta0`` (or zeros)
        when None.
    derivatives : bool
        Also sum the gradients and Hessians.

    Returns
    -------
    PooledScore
        Score inf, with no sessions, as soon as one session's inner fit fails;
        it records which session and why.

    Raises
    ------
    ValueError
        If a session marks a column of a penalty block inactive or its inputs
        cannot be fitted; the message names the session.
    """
    for index, data in enumerate(sessions):
        _check_active(data, index, blocks)
    if starts is None:
        starts = [data.beta0 for data in sessions]
    scores = []
    for index, (data, start) in enumerate(zip(sessions, starts, strict=True)):
        try:
            session = _session_score(lik, data, blocks, rho, start, derivatives)
        except ValueError as error:
            raise ValueError(f"{_label(data, index)}: {error}") from error
        if not np.isfinite(session.score):
            status = session.fit.status
            failure = f"{_label(data, index)}: {status}"
            return PooledScore(np.inf, None, None, (), index, status, failure)
        scores.append(session)
    score = float(sum(s.score for s in scores))
    if not derivatives:
        return PooledScore(score, None, None, tuple(scores))
    return PooledScore(
        score,
        np.sum([s.gradient for s in scores], axis=0),
        np.sum([s.hessian for s in scores], axis=0),
        tuple(scores),
    )


class _Evaluator:
    """Pooled scores with each session warm-started from its last accepted fit."""

    def __init__(
        self,
        lik: Likelihood,
        sessions: Sequence[SessionData],
        blocks: Sequence[PenaltyBlock],
    ) -> None:
        self.lik, self.sessions, self.blocks = lik, sessions, blocks
        self.starts = [data.beta0 for data in sessions]

    def __call__(self, rho: np.ndarray, derivatives: bool) -> PooledScore:
        pooled = pooled_reml(
            self.lik,
            self.sessions,
            self.blocks,
            rho,
            self.starts,
            derivatives=derivatives,
        )
        if np.isfinite(pooled.score):
            self.starts = [s.fit.beta for s in pooled.sessions]
        return pooled


def _grid(m: int) -> list[np.ndarray]:
    """The grid points for one to three terms."""
    lo, hi = LOG_PENALTY_BOUNDS
    if m == 1:
        return [np.array([r]) for r in np.arange(lo, hi + 0.5)]
    return [np.array(point) for point in itertools.product(REML_GRID_POINTS, repeat=m)]


def _grid_start(
    evaluate: _Evaluator, m: int
) -> tuple[np.ndarray, int, Counter[tuple[str, str]]]:
    """The best grid point, the number of points evaluated, and the points
    whose score was not finite counted by (first failing session, status).

    Raises
    ------
    ValueError
        If no grid point gives a finite score; the message names the sessions
        that failed and why.
    """
    best_rho, best_score, n_points = np.full(m, np.nan), np.inf, 0
    failures: Counter[tuple[str, str]] = Counter()

    def visit(rho: np.ndarray) -> None:
        nonlocal best_rho, best_score, n_points
        pooled = evaluate(rho, derivatives=False)
        n_points += 1
        if not np.isfinite(pooled.score):
            index = pooled.failed_session
            failures[_label(evaluate.sessions[index], index), pooled.failed_status] += 1
        elif pooled.score < best_score:
            best_rho, best_score = rho, pooled.score

    if m <= 3:
        for rho in _grid(m):
            visit(rho)
    else:
        lo, hi = LOG_PENALTY_BOUNDS
        current = np.full(m, REML_GRID_CENTER)
        for _ in range(REML_GRID_PASSES):
            for j in range(m):
                for value in np.arange(lo, hi + 0.5):
                    rho = current.copy()
                    rho[j] = value
                    visit(rho)
                if np.isfinite(best_score):
                    current = best_rho.copy()
    if not np.isfinite(best_score):
        detail = "; ".join(
            f"{label}: {status} at {count} of {n_points} points"
            for (label, status), count in failures.items()
        )
        raise ValueError(
            f"no finite REML score on the grid over {LOG_PENALTY_BOUNDS}: no penalty "
            f"gives a converged inner fit in every session ({detail})"
        )
    return best_rho, n_points, failures


def _free(rho: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    """The terms not held at a bound by a gradient pointing out of it."""
    lo, hi = LOG_PENALTY_BOUNDS
    return ~(((rho <= lo) & (gradient > 0)) | ((rho >= hi) & (gradient < 0)))


def _newton_step(gradient: np.ndarray, hessian: np.ndarray) -> np.ndarray:
    """``-H~^-1 g`` with ``H``'s eigenvalues made positive and floored."""
    eigenvalues, vectors = np.linalg.eigh(hessian)
    floor = REML_EIGEN_FLOOR * max(1.0, float(np.max(np.abs(eigenvalues))))
    eigenvalues = np.maximum(np.abs(eigenvalues), floor)
    return -vectors @ ((vectors.T @ gradient) / eigenvalues)


def _outer_step(rho: np.ndarray, current: PooledScore) -> np.ndarray:
    """The Newton step on the free terms, zero on the terms held at a bound."""
    free = _free(rho, current.gradient)
    step = np.zeros_like(rho)
    if free.any():
        step[free] = _newton_step(
            current.gradient[free], current.hessian[np.ix_(free, free)]
        )
    return step


def _try_step(
    evaluate: _Evaluator,
    rho: np.ndarray,
    current: PooledScore,
    step: np.ndarray,
    max_halvings: int,
    slack: float = 0.0,
) -> tuple[np.ndarray, PooledScore] | str:
    """``step``, halved until the score does not rise by more than ``slack``.

    Returns the new point and its score, or why there is none: the step is
    empty (the bounds hold it), or no halving lowers the score.
    """
    scale, trial = 1.0, None
    for _ in range(max_halvings):
        candidate = np.clip(rho + scale * step, *LOG_PENALTY_BOUNDS)
        if np.array_equal(candidate, rho):
            return "its step is empty at the bounds"
        trial = evaluate(candidate, derivatives=True)
        if np.isfinite(trial.score) and trial.score <= current.score + slack:
            return candidate, trial
        scale *= 0.5
    reason = "no halving of its step lowered the pooled score"
    if trial is not None and trial.failure is not None:
        reason += f" (the last halving failed in {trial.failure})"
    return reason


def _search(
    evaluate: _Evaluator,
    start: np.ndarray,
    n_grid_points: int,
    failures: Counter[tuple[str, str]],
) -> tuple[np.ndarray, PooledScore, PenaltySearch]:
    """Newton on the log penalties from ``start``.

    Stops when the Newton decrement in the log penalties, ``g' H~^-1 g / 2``
    on the free terms with the floored Hessian ``H~`` (the gain in the score
    the next step predicts), is below ``REML_DECREMENT_TOL``, then takes that
    last step when the score does not rise by more than its rounding,
    ``8 eps max(1, |V|)`` (as the inner fit does), so ``rho`` is the optimum
    to rounding.

    Raises
    ------
    ValueError
        If the score at ``start`` is not finite, or the search does not
        converge: ``REML_MAX_STEPS`` steps, an empty step or no halving that
        lowers the score. The message gives the last log penalties, the
        gradient and the steps taken.
    """
    rho = np.array(start, dtype=float)
    current = evaluate(rho, derivatives=True)
    if not np.isfinite(current.score):
        raise ValueError(f"no finite REML score at the start {rho}: {current.failure}")
    start_score = current.score
    n_steps = 0
    while True:
        step = _outer_step(rho, current)
        if -0.5 * float(current.gradient @ step) < REML_DECREMENT_TOL:
            rounding = 8.0 * np.finfo(float).eps * max(1.0, abs(current.score))
            moved = _try_step(evaluate, rho, current, step, 1, rounding)
            if not isinstance(moved, str):
                (rho, current), n_steps = moved, n_steps + 1
            break
        if n_steps == REML_MAX_STEPS:
            reason = f"no stationary point in {REML_MAX_STEPS} steps"
        else:
            moved = _try_step(evaluate, rho, current, step, MAX_STEP_HALVINGS)
            if not isinstance(moved, str):
                (rho, current), n_steps = moved, n_steps + 1
                continue
            reason = moved
        raise ValueError(
            f"the penalty search did not converge: {reason}, after {n_steps} "
            f"steps; log penalties {rho}, gradient {current.gradient}"
        )
    lo, hi = LOG_PENALTY_BOUNDS
    at_boundary = (rho - lo <= AT_BOUNDARY_MARGIN) | (hi - rho <= AT_BOUNDARY_MARGIN)
    search = PenaltySearch(
        np.array(start, dtype=float),
        start_score,
        n_grid_points,
        sum(failures.values()),
        dict(Counter({status: n for (_, status), n in failures.items()})),
        n_steps,
        at_boundary,
    )
    return rho, current, search


def _pooled_fit(
    data: Sequence[SessionData],
    rho: np.ndarray,
    pooled: PooledScore,
    blocks: Sequence[PenaltyBlock],
    search: PenaltySearch | None,
) -> PooledFit:
    """The per-session fits, covariances and edf of a pooled score.

    A session's covariance is NaN in the rows and columns of its inactive
    columns; its edf counts the active ones.
    """
    sessions = []
    for session_data, session in zip(data, pooled.sessions, strict=True):
        p = session.fit.beta.size
        active = (
            np.ones(p, dtype=bool)
            if session_data.active is None
            else session_data.active
        )
        A = session.fit.A[np.ix_(active, active)]
        factor = scipy.linalg.cho_factor(A, lower=True)
        S = assemble_penalty(blocks, rho, p)[np.ix_(active, active)] / session.scale
        covariance = np.full((p, p), np.nan)
        covariance[np.ix_(active, active)] = scipy.linalg.cho_solve(
            factor, np.eye(A.shape[0])
        )
        F = np.full((p, p), np.nan)
        F[np.ix_(active, active)] = covariance[np.ix_(active, active)] @ (A - S)
        edf = np.array([np.trace(F[np.ix_(b.columns, b.columns)]) for b in blocks])
        null_dim = np.array([b.columns.size - b.rank for b in blocks])
        sessions.append(
            SessionFit(
                session.fit,
                covariance,
                edf,
                float(np.trace(F[np.ix_(active, active)])),
                session.scale,
                session.score,
                edf - null_dim <= NULL_SPACE_EDF_TOL,
            )
        )
    return PooledFit(rho, pooled.score, pooled.gradient, tuple(sessions), search)


def refine_penalties(
    lik: Likelihood,
    sessions: Sequence[SessionData],
    blocks: Sequence[PenaltyBlock],
    rho0: Sequence[float],
) -> PooledFit:
    """Newton on the pooled REML score from ``rho0``, without the grid.

    A local search: from a poor start it can stop at the worse of two optima,
    which is why :func:`select_penalties` starts from a grid.

    Raises
    ------
    ValueError
        If the score at ``rho0`` is not finite or the search does not converge.
    """
    evaluate = _Evaluator(lik, sessions, blocks)
    rho0 = np.asarray(rho0, dtype=float)
    rho, pooled, search = _search(evaluate, rho0, 0, Counter())
    return _pooled_fit(sessions, rho, pooled, blocks, search)


def select_penalties(
    lik: Likelihood,
    sessions: Sequence[SessionData],
    blocks: Sequence[PenaltyBlock],
) -> PooledFit:
    """One penalty strength per block, shared by the sessions, by pooled REML.

    The grid start (one term: every integer of the bounds; two or three: the
    product of ``REML_GRID_POINTS``; more: one-term integer grids, two passes
    from all terms at ``REML_GRID_CENTER``), then Newton on the log penalties
    with the exact Hessian, eigenvalue floor, step halving and the bounds.

    Parameters
    ----------
    lik : Likelihood
    sessions : sequence of SessionData
    blocks : sequence of PenaltyBlock, length m >= 1

    Returns
    -------
    PooledFit

    Raises
    ------
    ValueError
        If there is no block, no grid point gives a finite pooled score, or
        the search does not converge (no fallback: the model is not fitted).
    """
    if not blocks:
        raise ValueError("no penalty to select; use fit_at_penalties")
    evaluate = _Evaluator(lik, sessions, blocks)
    start, n_points, failures = _grid_start(evaluate, len(blocks))
    rho, pooled, search = _search(evaluate, start, n_points, failures)
    return _pooled_fit(sessions, rho, pooled, blocks, search)


def fit_at_penalties(
    lik: Likelihood,
    sessions: Sequence[SessionData],
    blocks: Sequence[PenaltyBlock],
    rho: Sequence[float],
) -> PooledFit:
    """The sessions' fits at given log penalties, without a search.

    Raises
    ------
    ValueError
        If a session's inner fit does not converge; the message names it and
        says why.
    """
    rho = np.asarray(rho, dtype=float)
    pooled = pooled_reml(lik, sessions, blocks, rho)
    if not np.isfinite(pooled.score):
        raise ValueError(
            f"an inner fit did not converge at log penalties {rho}: {pooled.failure}"
        )
    return _pooled_fit(sessions, rho, pooled, blocks, None)
