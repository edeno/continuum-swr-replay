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

Each session's fit and score are computed with every block's coefficients in
the eigenbasis of its penalty, where the penalty is diagonal and exactly zero on
its null space (:func:`reml_and_derivatives`), so that the fit at a very strong
penalty is the fit in the null space rather than a casualty of rounding.

Pooled REML shares the strengths across sessions, each with its own
coefficients: the pooled score, gradient and Hessian are the sums over
sessions, and one session whose inner fit fails makes the pooled score
infinite; the score records which session and the fit's status, and every
error names them. A session may leave out unpenalized columns it cannot
estimate (``SessionData.active``): its fit is that of its design without them.
:func:`select_penalties` first sets aside the sessions whose fit has no
finite maximum at a reference point (the grid's center), naming the columns
running away (any other failure there raises); it then starts from the best
point of a grid over the others (the score can have two optima), refines it by
Newton on ``rho`` with the exact Hessian, and fits the set-aside sessions at
the chosen penalties, leaving out their unpenalized running-away columns; more
than ``MAX_FAILING_SHARE`` of the sessions without a fit raises.
:func:`fit_at_penalties` fits at given strengths without a search. A search
that does not converge raises.
"""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import NamedTuple

import numpy as np
import scipy.linalg

from .constants import (
    AT_BOUNDARY_MARGIN,
    IDENTIFIABILITY_TOL,
    LOG_PENALTY_BOUNDS,
    MAX_FAILING_SHARE,
    MAX_STEP_HALVINGS,
    NULL_SPACE_EDF_TOL,
    PENALTY_ROUNDING_TOL,
    REFIT_MAX_ROUNDS,
    REML_DECREMENT_TOL,
    REML_EIGEN_FLOOR,
    REML_GRID_CENTER,
    REML_GRID_PASSES,
    REML_GRID_POINTS,
    REML_MAX_STEPS,
)
from .likelihoods import Gaussian, Likelihood
from .newton import (
    NO_FINITE_MAXIMUM,
    NOT_IDENTIFIED,
    FitStatus,
    NewtonFit,
    penalized_newton,
    running_away_columns,
)
from .pspline import penalty_rank

__all__ = [
    "PenaltyBlock",
    "PenaltySearch",
    "PooledFit",
    "PooledScore",
    "RemlScore",
    "SessionData",
    "SessionFit",
    "SetAsideSession",
    "FreeDirection",
    "assemble_penalty",
    "fit_at_penalties",
    "free_basis",
    "pooled_reml",
    "refine_penalties",
    "reml_and_derivatives",
    "select_penalties",
]

_LOG_2PI = float(np.log(2.0 * np.pi))


class FreeDirection(NamedTuple):
    """One coordinate along which a fit with no finite maximum runs away, in
    the coordinates the penalty leaves free or penalizes.

    Attributes
    ----------
    block : int or None
        None for an unpenalized design column; else the penalty block.
    index : int
        The design column; or, in a block, the coordinate in the eigenbasis of
        its penalty (``PenaltyBlock.eigenvectors``), its null space first.
    unpenalized : bool
        True for a design column the penalty does not touch and for a block's
        null-space coordinate (for a P-spline, its straight line): a block's
        coordinate cannot be made inactive.
    """

    block: int | None
    index: int
    unpenalized: bool

    def __str__(self) -> str:
        if self.block is None:
            return f"column {self.index}"
        if self.unpenalized:
            return f"block {self.block}'s null space (direction {self.index})"
        return f"block {self.block}, penalized direction {self.index}"


def _directions(
    blocks: Sequence[PenaltyBlock], columns: np.ndarray
) -> tuple[FreeDirection, ...]:
    """The :class:`FreeDirection` of each of ``columns``, indices into the
    design in the blocks' eigenbasis."""
    owner = {}
    for b, block in enumerate(blocks):
        for q, column in enumerate(block.columns):
            owner[int(column)] = (b, q, q < block.columns.size - block.rank)
    out = []
    for column in columns.tolist():
        if column in owner:
            b, q, null = owner[column]
            out.append(FreeDirection(b, q, null))
        else:
            out.append(FreeDirection(None, column, True))
    return tuple(out)


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
    eigenvalues : np.ndarray, shape (k,)
        The eigenvalues of ``S``, ascending, with the ``k - rank`` smallest
        set to zero: ``S`` as its null space and ``rank`` define it (a matrix
        computed in floating point, such as ``Z' D' D Z``, has rounding there,
        about ``1e-16`` of its largest eigenvalue). Set on construction.
    eigenvectors : np.ndarray, shape (k, k)
        The orthonormal eigenvectors, in the order of ``eigenvalues``; the
        first ``k - rank`` are :attr:`null_space`. Set on construction.
    """

    columns: np.ndarray
    S: np.ndarray
    rank: int = field(init=False)
    log_pdet: float = field(init=False)
    eigenvalues: np.ndarray = field(init=False)
    eigenvectors: np.ndarray = field(init=False)

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
        n_null = eigenvalues.size - rank
        object.__setattr__(
            self,
            "eigenvalues",
            np.concatenate([np.zeros(n_null), eigenvalues[n_null:]]),
        )
        object.__setattr__(self, "eigenvectors", vectors)

    @property
    def null_space(self) -> np.ndarray:
        """An orthonormal basis of the null space of ``S`` (for a P-spline, the
        straight lines), shape ``(k, k - rank)``: the first ``k - rank``
        eigenvectors."""
        return self.eigenvectors[:, : self.columns.size - self.rank]

    @cached_property
    def in_eigenbasis(self) -> PenaltyBlock:
        """The block in the eigenbasis of its penalty: the same columns, the
        penalty ``diag(eigenvalues)``, exactly zero on its null space. Built
        once per block, when first read."""
        return PenaltyBlock(self.columns, np.diag(self.eigenvalues))


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
    covariance : np.ndarray or None, shape (p, p)
        ``A^-1``, computed in the penalties' eigenbasis (see
        :func:`reml_and_derivatives`); None when the score is infinite.
    edf : np.ndarray or None, shape (m,)
        Per block, the trace of the block's rows and columns of ``A^-1 H``;
        None when the score is infinite.
    edf_total : float
        ``tr(A^-1 H)``; NaN when the score is infinite.
    running_away_directions : tuple of FreeDirection
        For a fit with no finite maximum, the coordinates it runs away along
        (``fit.running_away`` names the design columns); empty otherwise.
    """

    score: float
    gradient: np.ndarray | None
    hessian: np.ndarray | None
    fit: NewtonFit
    scale: float
    covariance: np.ndarray | None = None
    edf: np.ndarray | None = None
    edf_total: float = np.nan
    running_away_directions: tuple[FreeDirection, ...] = ()


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
        The inner fit (``beta``, ``A``, status); ``beta`` and ``A`` are NaN at
        the session's inactive columns.
    covariance : np.ndarray, shape (p, p)
        ``A^-1``, the Bayesian covariance of ``beta`` (mgcv's ``Vp``); NaN in
        the rows and columns of the inactive columns.
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
    in_penalty_choice : bool
        True when the session's score was part of the pooled score that chose
        the penalties; False for a session set aside before the search and
        fitted afterward at the chosen penalties (``PenaltySearch.set_aside``).
    """

    fit: NewtonFit
    covariance: np.ndarray
    edf: np.ndarray
    edf_total: float
    scale: float
    score: float
    in_null_space: np.ndarray
    in_penalty_choice: bool = True


@dataclass(frozen=True)
class SetAsideSession:
    """A session left out of the penalty choice: its fit had no finite maximum
    at the search's reference point (:func:`select_penalties`).

    After the search it is fitted again at the chosen log penalties; while
    that fit has no finite maximum along unpenalized design columns only,
    those columns are made inactive and it is fitted again, at most
    ``REFIT_MAX_ROUNDS`` times.

    Attributes
    ----------
    index : int
        Its place among the sessions given.
    name : str or None
        ``SessionData.name``.
    reference : RemlScore
        Its score and fit at the reference point.
    refits : tuple of RemlScore
        Its fits at the chosen log penalties, in order: as given, then after
        each round of columns made inactive.
    left_out : np.ndarray of int, shape (k,)
        The columns made inactive in its last refit, in the order they were.
    reason : str
        How it ended, in words.
    """

    index: int
    name: str | None
    reference: RemlScore
    refits: tuple[RemlScore, ...]
    left_out: np.ndarray
    reason: str

    @property
    def reference_status(self) -> FitStatus:
        """How its fit at the reference point ended."""
        return self.reference.fit.status

    @property
    def final_status(self) -> FitStatus:
        """How its last fit at the chosen log penalties ended."""
        return self.refits[-1].fit.status

    @property
    def fitted(self) -> bool:
        """True when its last fit at the chosen log penalties converged."""
        return self.refits[-1].fit.converged

    @property
    def running_away(self) -> np.ndarray:
        """The columns running away at the reference point
        (:func:`.newton.running_away_columns`)."""
        return self.reference.fit.running_away


@dataclass(frozen=True)
class PenaltySearch:
    """How :func:`select_penalties` or :func:`refine_penalties` found ``rho``.

    There is one only for a converged search (the Newton decrement on the
    log penalties below ``REML_DECREMENT_TOL``; a term at a bound whose
    gradient points out of the bounds is held there): a search that does not
    converge raises.

    Attributes
    ----------
    reference_rho : np.ndarray, shape (m,)
        Where every session was fitted before the search: the sessions whose
        fit has no finite maximum there are set aside, and the search runs
        over the others.
    in_choice : np.ndarray of int, shape (n_in_choice,)
        The indices of the sessions whose scores chose the penalties (the
        pooled score's sessions); two models are compared on the same set.
    set_aside : tuple of SetAsideSession
        The sessions set aside, in order, each with its refits at the chosen
        penalties.
    start_score : float
        The pooled score where Newton started: the best grid point, or the
        given start.
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

    reference_rho: np.ndarray
    in_choice: np.ndarray
    set_aside: tuple[SetAsideSession, ...]
    start_score: float
    n_grid_points: int
    n_grid_not_finite: int
    grid_failures: dict[str, int]
    n_outer_steps: int
    at_boundary: np.ndarray

    @property
    def share_without_fit(self) -> float:
        """The share of the sessions given (those in the choice and those set
        aside) that end without a fit (set aside and not fitted by their
        refits); at most ``MAX_FAILING_SHARE``."""
        n_sessions = self.in_choice.size + len(self.set_aside)
        return sum(not r.fitted for r in self.set_aside) / n_sessions


@dataclass(frozen=True)
class PooledFit:
    """The sessions' fits at shared penalty strengths.

    Attributes
    ----------
    rho : np.ndarray, shape (m,)
        Log penalty strengths (for the Gaussian, on the residual-sum-of-squares
        scale).
    score : float
        Pooled REML score at ``rho`` over the sessions in the penalty choice
        only (``search.in_choice``; every session when the strengths were
        given), not those set aside: two models' scores are comparable only
        on the same set.
    gradient : np.ndarray or None, shape (m,)
        Its gradient there; None when the strengths were given.
    sessions : tuple of SessionFit
        One per session given, in order. A session set aside by the search
        (``search.set_aside``) and fitted afterward has its fit at ``rho``
        with ``in_penalty_choice`` False (NaN at the columns its refit left
        out); one not fitted has NaN coefficients, covariance, edf and score,
        and ``fit.status`` says how its last fit ended.
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
    return _summed(_strength_blocks(blocks, rho, n_columns), n_columns)


def _strength_blocks(
    blocks: Sequence[PenaltyBlock], rho: np.ndarray, n_columns: int
) -> list[np.ndarray]:
    """Each ``exp(rho_j) S_j``, embedded, shape ``(n_columns, n_columns)``."""
    return [
        _embedded(block, np.exp(r), n_columns)
        for block, r in zip(blocks, rho, strict=True)
    ]


def _summed(S_j: Sequence[np.ndarray], n_columns: int) -> np.ndarray:
    """The sum of the embedded blocks ``S_j`` (zeros for none)."""
    return sum(S_j, np.zeros((n_columns, n_columns)))


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


def _eigenbasis(
    blocks: Sequence[PenaltyBlock], n_columns: int
) -> tuple[list[np.ndarray], list[PenaltyBlock]]:
    """Each block's eigenvectors, and the blocks in that basis (diagonal).

    The coefficients ``gamma = V_j' beta_j`` of a block are an orthogonal
    change of its coefficients in which its penalty is
    ``diag(PenaltyBlock.eigenvalues)``, exactly zero on its null space.
    """
    _check_blocks(blocks, n_columns)
    vectors = [block.eigenvectors for block in blocks]
    rotated = [block.in_eigenbasis for block in blocks]
    return vectors, rotated


def _rotate(
    blocks: Sequence[PenaltyBlock],
    vectors: Sequence[np.ndarray],
    M: np.ndarray,
    *,
    inverse: bool = False,
) -> np.ndarray:
    """``Q M``, or ``Q' M`` when ``inverse``, with ``Q`` the block-diagonal
    rotation (identity off the blocks).

    ``M`` has shape ``(p,)`` or ``(p, k)``.
    """
    out = np.array(M, dtype=float)
    for block, V in zip(blocks, vectors, strict=True):
        out[block.columns] = (V.T if inverse else V) @ M[block.columns]
    return out


def _rotate_both(
    blocks: Sequence[PenaltyBlock], vectors: Sequence[np.ndarray], M: np.ndarray
) -> np.ndarray:
    """``Q M Q'`` for a symmetric ``M`` of shape ``(p, p)``."""
    return _rotate(blocks, vectors, _rotate(blocks, vectors, M).T)


def _null_dimension(blocks: Sequence[PenaltyBlock], n_columns: int) -> int:
    """``M_p``, the dimension of the summed penalty's null space."""
    return n_columns - sum(block.rank for block in blocks)


def _half_log_det(factor: tuple[np.ndarray, bool]) -> float:
    """``log|A| / 2`` from the Cholesky ``factor`` of ``A``."""
    return float(np.log(np.diag(factor[0])).sum())


def _coefficient_derivatives(
    S_j: Sequence[np.ndarray], b: np.ndarray, factor: tuple[np.ndarray, bool]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per block, ``S_j b`` and ``b_j = db/drho_j = -A^-1 S_j b``, each shape
    ``(p,)``, from the Cholesky ``factor`` of ``A``."""
    S_j_b = [Sj @ b for Sj in S_j]
    return S_j_b, [-scipy.linalg.cho_solve(factor, Sjb) for Sjb in S_j_b]


def _covariance_and_edf(
    factor: tuple[np.ndarray, bool],
    A: np.ndarray,
    S: np.ndarray,
    blocks: Sequence[PenaltyBlock],
) -> tuple[np.ndarray, np.ndarray, float]:
    """``A^-1``, the edf per block and in all, from the Cholesky ``factor`` of ``A``.

    ``A - S`` is ``H``; the edf of a block is the trace of its rows and
    columns of ``A^-1 H``.
    """
    covariance = scipy.linalg.cho_solve(factor, np.eye(A.shape[0]))
    F = covariance @ (A - S)
    edf = np.array([np.trace(F[np.ix_(b.columns, b.columns)]) for b in blocks])
    return covariance, edf, float(np.trace(F))


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

    The fit, score and derivatives are computed with each block's
    coefficients in the eigenbasis of its penalty (:func:`_eigenbasis`), an
    orthogonal change of coefficients that leaves the score, its derivatives
    and the edf unchanged, and the coefficients, ``A`` and the covariance are
    rotated back. In that basis the penalty is diagonal and exactly zero on
    its null space, so ``b' S b`` is a sum of positive terms and ``A`` is
    well conditioned after scaling by its diagonal. Computed with ``S`` as
    given, at a log penalty of 16 and above, ``b' S b`` (near ``1e-7``) was
    off by ``eps |S| |b|^2`` (``3e-7``), more than the gain of the fit's last
    steps (``3e-8``), and the Newton fit stopped as "step halving failed";
    the rounding of ``S`` on its null space, ``exp(rho)`` times ``1e-16`` of
    its largest eigenvalue, also penalized the null space.

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
    if isinstance(lik, Gaussian) and aux is not None:
        raise ValueError("the Gaussian scale is profiled; pass aux=None")
    vectors, rotated = _eigenbasis(blocks, n_columns)
    if rho.shape != (len(blocks),):
        raise ValueError(f"{rho.size} log penalties for {len(blocks)} blocks")
    X_rotated = X.copy()
    for block, V in zip(blocks, vectors, strict=True):
        X_rotated[:, block.columns] = X[:, block.columns] @ V
    start = np.zeros(n_columns) if beta0 is None else np.asarray(beta0, dtype=float)
    session = _reml_in_basis(
        lik,
        X_rotated,
        y,
        aux,
        rotated,
        rho,
        _rotate(blocks, vectors, start, inverse=True),
        derivatives,
    )
    fit = replace(
        session.fit,
        beta=_rotate(blocks, vectors, session.fit.beta),
        A=_rotate_both(blocks, vectors, session.fit.A),
    )
    covariance = (
        None
        if session.covariance is None
        else _rotate_both(blocks, vectors, session.covariance)
    )
    directions: tuple[FreeDirection, ...] = ()
    if fit.step is not None:
        directions = _directions(rotated, fit.running_away)
        step = _rotate(blocks, vectors, fit.step)
        fit = replace(fit, step=step, running_away=running_away_columns(X, step))
    return replace(
        session,
        fit=fit,
        covariance=covariance,
        running_away_directions=directions,
    )


def _reml_in_basis(
    lik: Likelihood,
    X: np.ndarray,
    y: np.ndarray,
    aux: object,
    blocks: Sequence[PenaltyBlock],
    rho: np.ndarray,
    start: np.ndarray,
    derivatives: bool,
) -> RemlScore:
    """:func:`reml_and_derivatives` in the basis of ``X`` and ``blocks`` as given."""
    n_columns = X.shape[1]
    S_j = _strength_blocks(blocks, rho, n_columns)
    S = _summed(S_j, n_columns)
    if isinstance(lik, Gaussian):
        return _gaussian_profiled(X, y, blocks, rho, S_j, S, start, derivatives)

    fit = penalized_newton(
        lik, X, y, aux, S, start, free_basis=free_basis(blocks, n_columns)
    )
    if not fit.converged:
        return RemlScore(np.inf, None, None, fit, 1.0)
    b, A = fit.beta, fit.A
    factor = scipy.linalg.cho_factor(A, lower=True)
    covariance, edf, edf_total = _covariance_and_edf(factor, A, S, blocks)
    score = (
        -fit.loglik
        + 0.5 * b @ S @ b
        - 0.5 * _log_pdet(blocks, rho)
        + _half_log_det(factor)
        - 0.5 * _null_dimension(blocks, n_columns) * _LOG_2PI
    )
    summary = {"covariance": covariance, "edf": edf, "edf_total": edf_total}
    if not derivatives:
        return RemlScore(float(score), None, None, fit, 1.0, **summary)

    eta = X @ b
    m = len(blocks)
    S_j_b, b_j = _coefficient_derivatives(S_j, b, factor)
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
    return RemlScore(float(score), gradient, hessian, fit, 1.0, **summary)


def _gaussian_profiled(
    X: np.ndarray,
    y: np.ndarray,
    blocks: Sequence[PenaltyBlock],
    rho: np.ndarray,
    S_j: Sequence[np.ndarray],
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
    if not working > 0:
        raise ValueError(
            "the Gaussian outcome is constant (variance zero): it has no scale to "
            "profile"
        )
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
    dof = n_rows - _null_dimension(blocks, n_columns)
    scale = D_p / dof
    factor = scipy.linalg.cho_factor(gram, lower=True)
    score = (
        0.5 * dof * (1.0 + _LOG_2PI + np.log(scale))
        - 0.5 * _log_pdet(blocks, rho)
        + _half_log_det(factor)
    )
    fit = replace(
        pls,
        A=gram / scale,
        newton_decrement=pls.newton_decrement * working / scale,
        loglik=Gaussian().loglik(X @ b, y, scale),
    )
    covariance, edf, edf_total = _covariance_and_edf(factor, gram, S, blocks)
    summary = {"covariance": scale * covariance, "edf": edf, "edf_total": edf_total}
    if not derivatives:
        return RemlScore(float(score), None, None, fit, scale, **summary)

    m = len(blocks)
    S_j_b, b_j = _coefficient_derivatives(S_j, b, factor)
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
    return RemlScore(float(score), gradient, hessian, fit, scale, **summary)


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
    covariance = None
    if session.covariance is not None:
        covariance = np.full((p, p), np.nan)
        covariance[np.ix_(active, active)] = session.covariance
    fit = replace(session.fit, beta=beta, A=A)
    directions = session.running_away_directions
    if fit.step is not None:
        columns = np.flatnonzero(active)
        step = np.full(p, np.nan)
        step[active] = fit.step
        fit = replace(fit, step=step, running_away=columns[fit.running_away])
        directions = tuple(
            d._replace(index=int(columns[d.index])) if d.block is None else d
            for d in directions
        )
    return replace(
        session, fit=fit, covariance=covariance, running_away_directions=directions
    )


def _failure(data: SessionData, index: int, session: RemlScore) -> str:
    """Why session ``index``'s fit failed, in words, with the columns running
    away and their coordinates for a fit with no finite maximum."""
    fit = session.fit
    text = f"{_label(data, index)}: {fit.status}"
    if fit.running_away.size:
        along = ", ".join(str(d) for d in session.running_away_directions)
        text += f" (columns running away: {fit.running_away.tolist()}; along {along})"
    return text


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
    if starts is None:
        starts = [data.beta0 for data in sessions]
    return _pooled_reml(
        lik, sessions, range(len(sessions)), blocks, rho, starts, derivatives
    )


def _pooled_reml(
    lik: Likelihood,
    sessions: Sequence[SessionData],
    indices: Sequence[int],
    blocks: Sequence[PenaltyBlock],
    rho: Sequence[float],
    starts: Sequence[np.ndarray | None],
    derivatives: bool,
) -> PooledScore:
    """:func:`pooled_reml`, with each session labeled by its index in ``indices``
    (its place among all of a search's sessions)."""
    for index, data in zip(indices, sessions, strict=True):
        _check_active(data, index, blocks)
    scores = []
    for index, data, start in zip(indices, sessions, starts, strict=True):
        try:
            session = _session_score(lik, data, blocks, rho, start, derivatives)
        except ValueError as error:
            raise ValueError(f"{_label(data, index)}: {error}") from error
        if not np.isfinite(session.score):
            failure = _failure(data, index, session)
            return PooledScore(
                np.inf, None, None, (), index, session.fit.status, failure
            )
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
        indices: Sequence[int],
        blocks: Sequence[PenaltyBlock],
    ) -> None:
        self.lik, self.blocks = lik, blocks
        self.sessions, self.indices = list(sessions), list(indices)
        self.labels = {
            i: _label(data, i) for i, data in zip(indices, sessions, strict=True)
        }
        self.starts = [data.beta0 for data in sessions]

    def __call__(self, rho: np.ndarray, derivatives: bool) -> PooledScore:
        pooled = _pooled_reml(
            self.lik,
            self.sessions,
            self.indices,
            self.blocks,
            rho,
            self.starts,
            derivatives,
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
            label = evaluate.labels[pooled.failed_session]
            failures[label, pooled.failed_status] += 1
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


def _term_scale(hessian: np.ndarray) -> np.ndarray:
    """Per term, ``sqrt(max(1, |H_jj|))``: the Hessian divided by it on both
    sides has a diagonal of at most one."""
    return np.sqrt(np.maximum(1.0, np.abs(np.diag(hessian))))


@dataclass(frozen=True)
class _Curvature:
    """The Hessian of the pooled score on the free terms, decomposed once per
    outer step.

    With ``D = diag(_term_scale(H))``, ``eigenvalues`` and ``vectors`` are
    those of ``D^-1 H D^-1``, and ``floor`` is
    ``REML_EIGEN_FLOOR * max(1, max |eigenvalue|)``: a term of curvature at
    most one keeps the absolute floor ``REML_EIGEN_FLOOR``, and a stiff term
    does not set the floor of a weak one (floored over the whole Hessian, a
    term of curvature 1000 floored a term of curvature 2e-7 at 1e-3, and the
    search stopped eight units short of that term's optimum).
    """

    free: np.ndarray
    scale: np.ndarray
    eigenvalues: np.ndarray
    vectors: np.ndarray
    floor: float

    @classmethod
    def at(cls, rho: np.ndarray, current: PooledScore) -> _Curvature:
        """The curvature at ``rho`` on the terms not held at a bound."""
        free = _free(rho, current.gradient)
        hessian = current.hessian[np.ix_(free, free)]
        scale = _term_scale(hessian)
        if not free.any():
            return cls(free, scale, np.empty(0), np.empty((0, 0)), np.inf)
        eigenvalues, vectors = np.linalg.eigh(hessian / np.outer(scale, scale))
        floor = REML_EIGEN_FLOOR * max(1.0, float(np.max(np.abs(eigenvalues))))
        return cls(free, scale, eigenvalues, vectors, floor)

    def newton_step(self, gradient: np.ndarray) -> np.ndarray:
        """``-H~^-1 g`` on the free terms, with the scaled Hessian's
        eigenvalues made positive and floored; zero on the terms held at a
        bound."""
        step = np.zeros_like(gradient)
        if self.free.any():
            eigenvalues = np.maximum(np.abs(self.eigenvalues), self.floor)
            g = gradient[self.free] / self.scale
            step[self.free] = (
                -(self.vectors @ ((self.vectors.T @ g) / eigenvalues)) / self.scale
            )
        return step

    def downhill_probes(self, gradient: np.ndarray) -> list[np.ndarray]:
        """Steps along the direction of most negative curvature on the free
        terms, when the Hessian there is not positive semidefinite to the
        floor; empty otherwise. The probes are one scaled unit along the
        eigenvector, the downhill sign first."""
        if not self.free.any() or self.eigenvalues[0] >= -self.floor:
            return []
        direction = np.zeros_like(gradient)
        direction[self.free] = self.vectors[:, 0] / self.scale
        if gradient @ direction > 0:
            direction = -direction
        return [direction, -direction]


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
    evaluate: _Evaluator, start: np.ndarray
) -> tuple[np.ndarray, PooledScore, float, int]:
    """Newton on the log penalties from ``start``.

    Returns the log penalties found, the pooled score there, the score at
    ``start`` and the number of steps taken.

    Stops when the Newton decrement in the log penalties, ``g' H~^-1 g / 2``
    on the free terms with the floored Hessian ``H~`` (the gain in the score
    the next step predicts; :class:`_Curvature`), is below
    ``REML_DECREMENT_TOL`` or below the rounding of the score,
    ``8 eps max(1, |V|)`` (a gain no halving can show), and the Hessian on the
    free terms is positive semidefinite to rounding, then takes that last
    step when the score does not rise by more than its rounding (as the inner
    fit does), so ``rho`` is the optimum to rounding. Where the decrement is
    small but the curvature negative (near a maximum or a saddle of the
    score) it steps along the direction of negative curvature instead
    (:meth:`_Curvature.downhill_probes`).

    On a flat score (the data do not support curvature; the score falls
    toward its limit in the null space as ``c exp(-rho)``) the gradient is
    ``-c exp(-rho)``, the gain left on the way to the bound, and the Hessian
    the same, below the eigenvalue floor ``REML_EIGEN_FLOOR``: the search
    stops where that gain is below ``sqrt(2 REML_DECREMENT_TOL
    REML_EIGEN_FLOOR)`` (1.4e-7), or at the bound, where a gradient pointing
    out of it holds the term.

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
    start_score, n_steps = current.score, 0
    while True:
        curvature = _Curvature.at(rho, current)
        step = curvature.newton_step(current.gradient)
        rounding = 8.0 * np.finfo(float).eps * max(1.0, abs(current.score))
        small = -0.5 * float(current.gradient @ step) < max(
            REML_DECREMENT_TOL, rounding
        )
        probes = curvature.downhill_probes(current.gradient) if small else []
        if probes and n_steps < REML_MAX_STEPS:
            moved = _probe(evaluate, rho, current, probes)
            if moved is None:
                raise _not_converged(
                    "it stopped where the score's curvature is negative and no step "
                    "along that direction lowered it",
                    n_steps,
                    rho,
                    current,
                )
        elif small:  # the last step, taken when the score does not rise
            moved = _try_step(evaluate, rho, current, step, 1, rounding)
            if not isinstance(moved, str):
                (rho, current), n_steps = moved, n_steps + 1
            return rho, current, start_score, n_steps
        elif n_steps == REML_MAX_STEPS:
            raise _not_converged(
                f"no stationary point in {REML_MAX_STEPS} steps", n_steps, rho, current
            )
        else:
            moved = _try_step(evaluate, rho, current, step, MAX_STEP_HALVINGS)
            if isinstance(moved, str):
                raise _not_converged(moved, n_steps, rho, current)
        (rho, current), n_steps = moved, n_steps + 1


def _probe(
    evaluate: _Evaluator,
    rho: np.ndarray,
    current: PooledScore,
    probes: list[np.ndarray],
) -> tuple[np.ndarray, PooledScore] | None:
    """The first of ``probes`` (each halved as needed) that lowers the score
    strictly, and its score; None when none does."""
    for probe in probes:
        moved = _try_step(evaluate, rho, current, probe, MAX_STEP_HALVINGS)
        if not isinstance(moved, str) and moved[1].score < current.score:
            return moved
    return None


def _not_converged(
    reason: str, n_steps: int, rho: np.ndarray, current: PooledScore
) -> ValueError:
    """The error of a search that did not converge, with its last point."""
    return ValueError(
        f"the penalty search did not converge: {reason}, after {n_steps} "
        f"steps; log penalties {rho}, gradient {current.gradient}"
    )


def _at_boundary(rho: np.ndarray) -> np.ndarray:
    """True for a log penalty within ``AT_BOUNDARY_MARGIN`` of a bound."""
    lo, hi = LOG_PENALTY_BOUNDS
    return (rho - lo <= AT_BOUNDARY_MARGIN) | (hi - rho <= AT_BOUNDARY_MARGIN)


def _screen(
    lik: Likelihood,
    sessions: Sequence[SessionData],
    blocks: Sequence[PenaltyBlock],
    reference: np.ndarray,
) -> tuple[list[int], list[tuple[int, RemlScore]]]:
    """Fit every session at ``reference``.

    Returns the indices of the sessions whose fit converged, and the index and
    score of each session whose fit has no finite maximum, in order.

    Raises
    ------
    ValueError
        If a session's design is not identified (naming its columns that are
        zero on every row; the remedy is to mark the columns it cannot
        estimate inactive), its fit fails otherwise (a numerical failure,
        named with its status), or no session converges.
    """
    kept, set_aside = [], []
    for index, data in enumerate(sessions):
        _check_active(data, index, blocks)
        try:
            session = _session_score(lik, data, blocks, reference, data.beta0, False)
        except ValueError as error:
            raise ValueError(f"{_label(data, index)}: {error}") from error
        status = session.fit.status
        if np.isfinite(session.score):
            kept.append(index)
        elif status == NO_FINITE_MAXIMUM:
            set_aside.append((index, session))
        elif status == NOT_IDENTIFIED:
            raise ValueError(
                f"{_label(data, index)}: design not identified at the reference log "
                f"penalties {reference} ({_zero_columns(lik, data, blocks)}); mark the "
                "columns the session cannot estimate inactive (SessionData.active)"
            )
        else:
            raise ValueError(
                f"{_label(data, index)}: {status} at the reference log penalties "
                f"{reference} (a numerical failure of the fit, not a lack of support "
                "in the data)"
            )
    if not kept:
        raise ValueError(
            f"no session has a finite maximum at the reference log penalties "
            f"{reference}: "
            + "; ".join(_failure(sessions[i], i, score) for i, score in set_aside)
        )
    return kept, set_aside


def _zero_columns(
    lik: Likelihood, data: SessionData, blocks: Sequence[PenaltyBlock]
) -> str:
    """The active unpenalized columns the likelihood cannot see (zero on every
    row; for the grouped softmax, constant within every group), in words: the
    common cause of a design that is not identified."""
    free = np.ones(data.X.shape[1], dtype=bool)
    for block in blocks:
        free[block.columns] = False
    if data.active is not None:
        free &= data.active
    seen = lik.identifying_design(data.X, data.aux)
    zero = np.flatnonzero(
        free & ~np.any(np.abs(seen) > 1e-12 * np.abs(data.X).max(), axis=0)
    )
    if zero.size:
        return (
            "columns the likelihood cannot see (zero on every row; for the grouped "
            f"softmax, constant within every group): {zero.tolist()}"
        )
    # The free coordinates (unpenalized columns, blocks' null spaces) the
    # likelihood sees, each scaled to unit length; the right singular vectors
    # of the smallest singular values name the combinations it cannot tell
    # apart.
    names, parts = [], []
    for column in np.flatnonzero(free):
        names.append(FreeDirection(None, int(column), True))
        parts.append(seen[:, column])
    for b, block in enumerate(blocks):
        null = seen[:, block.columns] @ block.null_space
        for q in range(null.shape[1]):
            names.append(FreeDirection(b, q, True))
            parts.append(null[:, q])
    coordinates = np.column_stack(parts)
    coordinates = coordinates / np.linalg.norm(coordinates, axis=0)
    _, singular, vectors = np.linalg.svd(coordinates, full_matrices=False)
    collinear = sorted(
        {
            j
            for k in np.flatnonzero(singular**2 <= IDENTIFIABILITY_TOL)
            for j in np.flatnonzero(np.abs(vectors[k]) > 0.1)
        }
    )
    return "collinear to the likelihood: " + ", ".join(str(names[j]) for j in collinear)


def _refit(
    lik: Likelihood,
    data: SessionData,
    index: int,
    blocks: Sequence[PenaltyBlock],
    rho: np.ndarray,
    reference: RemlScore,
) -> SetAsideSession:
    """A set-aside session fitted at the chosen ``rho``.

    Fitted as given first; then, while its fit has no finite maximum along
    unpenalized design columns only, with those columns made inactive, at most
    ``REFIT_MAX_ROUNDS`` times (a direction spread over near-collinear columns
    can name only part of them at a time). A direction through a penalty
    block (a smooth's own line, or its penalized coordinates) cannot be left
    out, and the session stays without a fit.
    """
    active = (
        np.ones(data.X.shape[1], dtype=bool)
        if data.active is None
        else data.active.copy()
    )
    left_out: list[int] = []
    refits = [_session_score(lik, data, blocks, rho, data.beta0, False)]
    reason = ""
    while not refits[-1].fit.converged:
        last = refits[-1]
        if last.fit.status != NO_FINITE_MAXIMUM:
            reason = f"its fit at the chosen log penalties ended as {last.fit.status}"
            break
        along = last.running_away_directions
        if any(d.block is not None for d in along):
            reason = (
                "no finite maximum along a penalty block ("
                + ", ".join(str(d) for d in along)
                + "), whose columns cannot be made inactive"
            )
            break
        if len(refits) > REFIT_MAX_ROUNDS:
            reason = (
                f"no finite maximum after {REFIT_MAX_ROUNDS} rounds of columns "
                f"made inactive (still running away: {last.fit.running_away.tolist()})"
            )
            break
        new = [int(c) for c in last.fit.running_away if active[c]]
        active[new] = False
        left_out.extend(new)
        refits.append(
            _session_score(
                lik, replace(data, active=active.copy()), blocks, rho, data.beta0, False
            )
        )
    if refits[-1].fit.converged:
        reason = "converged at the chosen log penalties" + (
            f" with columns {left_out} inactive" if left_out else ""
        )
    return SetAsideSession(
        index,
        data.name,
        reference,
        tuple(refits),
        np.array(left_out, dtype=int),
        reason,
    )


def _refit_set_aside(
    lik: Likelihood,
    sessions: Sequence[SessionData],
    blocks: Sequence[PenaltyBlock],
    rho: np.ndarray,
    set_aside: Sequence[tuple[int, RemlScore]],
) -> tuple[SetAsideSession, ...]:
    """Each set-aside session refitted at ``rho`` (:func:`_refit`).

    Raises
    ------
    ValueError
        If that share exceeds ``MAX_FAILING_SHARE``; the message names each
        session without a fit, its statuses and why.
    """
    records = tuple(
        _refit(lik, sessions[i], i, blocks, rho, score) for i, score in set_aside
    )
    unfitted = [r for r in records if not r.fitted]
    share = len(unfitted) / len(sessions)
    if share > MAX_FAILING_SHARE:
        detail = "; ".join(
            f"{_failure(sessions[r.index], r.index, r.reference)} at the reference "
            f"point, then {r.reason}"
            for r in unfitted
        )
        raise ValueError(
            f"{len(unfitted)} of {len(sessions)} sessions ({share:.0%}, more than "
            f"MAX_FAILING_SHARE = {MAX_FAILING_SHARE:.0%}) have no fit at the log "
            f"penalties {rho}: {detail}"
        )
    return records


def _session_fit(
    session: RemlScore, blocks: Sequence[PenaltyBlock], in_penalty_choice: bool
) -> SessionFit:
    """The :class:`SessionFit` of a converged session score."""
    null_dim = np.array([b.columns.size - b.rank for b in blocks])
    return SessionFit(
        session.fit,
        session.covariance,
        session.edf,
        session.edf_total,
        session.scale,
        session.score,
        session.edf - null_dim <= NULL_SPACE_EDF_TOL,
        in_penalty_choice,
    )


def _pooled_fit(
    rho: np.ndarray,
    pooled: PooledScore,
    blocks: Sequence[PenaltyBlock],
    search: PenaltySearch | None,
) -> PooledFit:
    """The per-session fits, covariances and edf of a pooled score.

    A session's covariance is NaN in the rows and columns of its inactive
    columns; its edf counts the active ones. A session the search set aside
    is put back in its place: its refit at ``rho`` when that converged
    (``in_penalty_choice`` False), else NaN values.
    """
    sessions = [_session_fit(session, blocks, True) for session in pooled.sessions]
    for left_out in () if search is None else search.set_aside:
        last = left_out.refits[-1]
        if left_out.fitted:
            entry = _session_fit(last, blocks, False)
        else:
            p = last.fit.beta.size
            nan = np.full((p, p), np.nan)
            entry = SessionFit(
                replace(last.fit, beta=np.full(p, np.nan), A=nan),
                nan,
                np.full(len(blocks), np.nan),
                np.nan,
                np.nan,
                np.nan,
                np.zeros(len(blocks), dtype=bool),
                False,
            )
        sessions.insert(left_out.index, entry)
    return PooledFit(rho, pooled.score, pooled.gradient, tuple(sessions), search)


def _screened_search(
    lik: Likelihood,
    sessions: Sequence[SessionData],
    blocks: Sequence[PenaltyBlock],
    reference: np.ndarray,
    start: np.ndarray | None,
) -> PooledFit:
    """Screen at ``reference``, search over the sessions kept (from the grid
    when ``start`` is None), refit the sessions set aside at the chosen
    penalties."""
    kept, set_aside = _screen(lik, sessions, blocks, reference)
    evaluate = _Evaluator(lik, [sessions[i] for i in kept], kept, blocks)
    if start is None:
        start, n_points, failures = _grid_start(evaluate, len(blocks))
    else:
        n_points, failures = 0, Counter()
    rho, pooled, start_score, n_steps = _search(evaluate, start)
    search = PenaltySearch(
        reference_rho=reference.copy(),
        in_choice=np.array(kept, dtype=int),
        set_aside=_refit_set_aside(lik, sessions, blocks, rho, set_aside),
        start_score=start_score,
        n_grid_points=n_points,
        n_grid_not_finite=sum(failures.values()),
        grid_failures=dict(Counter({status: n for (_, status), n in failures.items()})),
        n_outer_steps=n_steps,
        at_boundary=_at_boundary(rho),
    )
    return _pooled_fit(rho, pooled, blocks, search)


def refine_penalties(
    lik: Likelihood,
    sessions: Sequence[SessionData],
    blocks: Sequence[PenaltyBlock],
    rho0: Sequence[float],
) -> PooledFit:
    """Newton on the pooled REML score from ``rho0``, without the grid.

    A local search: from a poor start it can stop at the worse of two optima,
    which is why :func:`select_penalties` starts from a grid. The sessions
    are screened at ``rho0`` as :func:`select_penalties` screens them at the
    grid's center.

    Raises
    ------
    ValueError
        As :func:`select_penalties`, with ``rho0`` the reference point.
    """
    rho0 = np.asarray(rho0, dtype=float)
    return _screened_search(lik, sessions, blocks, rho0, rho0)


def select_penalties(
    lik: Likelihood,
    sessions: Sequence[SessionData],
    blocks: Sequence[PenaltyBlock],
) -> PooledFit:
    """One penalty strength per block, shared by the sessions, by pooled REML.

    First every session is fitted at the reference point, every log penalty
    at ``REML_GRID_CENTER``. A session whose fit has no finite maximum there
    (its unpenalized columns, or a smooth's unpenalized line, separate the
    outcome: no penalty gives it one) is set aside; a session whose design is
    not identified, or whose fit fails for a numerical reason, is an error.
    The search runs over the other sessions, the same ones throughout, so
    that its objective does not change as it goes (``PenaltySearch.in_choice``);
    a session kept that fails at another point makes that point's score
    infinite. After the search each set-aside session is fitted at the chosen
    log penalties (its penalty does not depend on its own data), with the
    columns running away made inactive while they are all unpenalized design
    columns, at most ``REFIT_MAX_ROUNDS`` times (``PenaltySearch.set_aside``
    records every step). More than ``MAX_FAILING_SHARE`` of the sessions
    without a fit at the end is an error.

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
        One entry per session given; a set-aside session has its refit (with
        ``in_penalty_choice`` False), or NaN values when it has no fit. The
        score is over ``search.in_choice`` only.

    Raises
    ------
    ValueError
        If there is no block; a session's design is not identified or its fit
        fails numerically at the reference point; no session converges there;
        no grid point gives a finite pooled score over the sessions kept; the
        search does not converge; or more than ``MAX_FAILING_SHARE`` of the
        sessions end without a fit (no fallback: the model is not fitted).
    """
    if not blocks:
        raise ValueError("no penalty to select; use fit_at_penalties")
    reference = np.full(len(blocks), REML_GRID_CENTER)
    return _screened_search(lik, sessions, blocks, reference, None)


def fit_at_penalties(
    lik: Likelihood,
    sessions: Sequence[SessionData],
    blocks: Sequence[PenaltyBlock],
    rho: Sequence[float],
) -> PooledFit:
    """The sessions' fits at given log penalties, without a search.

    The score's gradient and Hessian are not computed (``gradient`` is None).

    Raises
    ------
    ValueError
        If a session's inner fit does not converge; the message names it and
        says why.
    """
    rho = np.asarray(rho, dtype=float)
    pooled = pooled_reml(lik, sessions, blocks, rho, derivatives=False)
    if not np.isfinite(pooled.score):
        raise ValueError(
            f"an inner fit did not converge at log penalties {rho}: {pooled.failure}"
        )
    return _pooled_fit(rho, pooled, blocks, None)
