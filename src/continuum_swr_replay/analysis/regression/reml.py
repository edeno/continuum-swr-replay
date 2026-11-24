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
:func:`select_penalties` chooses the strengths: it sets aside the sessions
without a finite maximum, searches from a grid by Newton on ``rho`` with the
exact Hessian, and refits the sessions set aside (its docstring states the
rules) by :func:`refit_session`, which a per-session fit at given penalties
calls too. :func:`fit_at_penalties` fits at given strengths without a search.

The score at a converged fit, ``A^-1``, the edf and the gradient and Hessian
in ``rho`` are jitted kernels in float64 (``runtime``), on one Cholesky
factor of the fit's ``A`` and on the padded rows the Newton fit put on the
device: the rows are padded and moved once for a fit and its derivatives.
The kernel of the derivatives is one per family (the profiled Gaussian has its
own); the number of penalties is the leading axis of its arrays.
"""

from __future__ import annotations

import itertools
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from functools import cached_property
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg
import numpy as np

from .constants import (
    AT_BOUNDARY_MARGIN,
    LOG_PENALTY_BOUNDS,
    MAX_FAILING_SHARE,
    MAX_STEP_HALVINGS,
    NAMED_SHARE,
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
from .likelihoods import Gaussian, Likelihood, _gram, _mirrored, _upper_pairs
from .newton import (
    NO_FINITE_MAXIMUM,
    NOT_IDENTIFIED,
    FitStatus,
    NewtonFit,
    _check_design_finite,
    _factored,
    _fit_one,
    _rounding_slack,
    running_away_columns,
)
from .pspline import penalty_rank
from .runtime import float64_kernel, in_float64

__all__ = [
    "ALONG_THE_INTERCEPT",
    "PenaltyBlock",
    "PenaltySearch",
    "PooledFit",
    "PooledScore",
    "RefitEnd",
    "RemlScore",
    "SessionData",
    "SessionRefit",
    "SetAsideSession",
    "FreeDirection",
    "assemble_penalty",
    "fit_at_penalties",
    "free_basis",
    "in_null_space",
    "pooled_reml",
    "refine_penalties",
    "refit_session",
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

    def at_columns(self, columns: np.ndarray) -> PenaltyBlock:
        """The same penalty on other ``columns``, its decomposition (and its
        block in the eigenbasis, once read) copied, not computed again."""
        block = object.__new__(PenaltyBlock)
        for name in ("S", "rank", "log_pdet", "eigenvalues", "eigenvectors"):
            object.__setattr__(block, name, getattr(self, name))
        object.__setattr__(block, "columns", np.asarray(columns, dtype=int))
        if "in_eigenbasis" in self.__dict__:
            block.__dict__["in_eigenbasis"] = self.in_eigenbasis.at_columns(columns)
        return block


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
    unidentified_directions : tuple of FreeDirection
        For a design that is not identified, the coordinates in
        ``fit.unidentified``, unpenalized columns first; empty otherwise.
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
    unidentified_directions: tuple[FreeDirection, ...] = ()

    def without_fit(self, n_blocks: int) -> RemlScore:
        """The score of a session left without a fit: the score, scale,
        coefficients, ``A``, covariance and edf NaN; the inner fit's status,
        Newton decrement and steps kept.

        Parameters
        ----------
        n_blocks : int
            The penalty blocks, the length of ``edf``.
        """
        p = self.fit.beta.size
        nan = np.full((p, p), np.nan)
        return replace(
            self,
            score=np.nan,
            fit=replace(self.fit, beta=np.full(p, np.nan), A=nan),
            scale=np.nan,
            covariance=nan,
            edf=np.full(n_blocks, np.nan),
            edf_total=np.nan,
        )


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
    failed : RemlScore or None
        That session's score, with its fit.
    """

    score: float
    gradient: np.ndarray | None
    hessian: np.ndarray | None
    sessions: tuple[RemlScore, ...]
    failed_session: int | None = None
    failed: RemlScore | None = None

    @property
    def failed_status(self) -> FitStatus | None:
        """How the failed session's fit ended; None when none failed."""
        return None if self.failed is None else self.failed.fit.status


def in_null_space(session: RemlScore, blocks: Sequence[PenaltyBlock]) -> np.ndarray:
    """Per block, True for a term shrunk to its penalty's null space (a P-spline
    drawn as a straight line), shape ``(m,)``.

    Its edf is within ``NULL_SPACE_EDF_TOL`` of the null space's dimension
    within the term's columns (the number of columns less the penalty's rank).
    The search can stop on the score's plateau before the bound, so
    ``PenaltySearch.at_boundary`` misses such a term. False for a session
    without a fit (its edf is NaN).
    """
    null_dim = np.array([b.columns.size - b.rank for b in blocks])
    return session.edf - null_dim <= NULL_SPACE_EDF_TOL


@dataclass(frozen=True)
class SetAsideSession:
    """A session left out of the penalty choice and refitted after the search,
    by the rules of :func:`select_penalties`.

    Attributes
    ----------
    index : int
        Its place among the sessions given.
    name : str or None
        ``SessionData.name``.
    reference : RemlScore
        Its score and fit at the reference point.
    refit : SessionRefit
        Its refit at the chosen log penalties (:func:`refit_session`).
    """

    index: int
    name: str | None
    reference: RemlScore
    refit: SessionRefit

    @property
    def refits(self) -> tuple[RemlScore, ...]:
        """Its fits at the chosen log penalties, in order: as given, then
        after each round of columns made inactive."""
        return self.refit.refits

    @property
    def left_out(self) -> np.ndarray:
        """The columns made inactive in its last refit, in the order they
        were, shape ``(k,)``."""
        return self.refit.left_out

    @property
    def reason(self) -> str:
        """How it ended, in words."""
        return self.refit.reason

    @property
    def fitted(self) -> bool:
        """True when its last fit at the chosen log penalties converged."""
        return self.refits[-1].fit.converged


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
    grid_failures : dict of str to int
        Grid points whose pooled score was not finite (an inner fit failed),
        counted by the status of their first failing session's fit.
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
        Pooled REML score at ``rho`` over the sessions in the penalty choice
        only (``search.in_choice``; every session when the strengths were
        given), not those set aside: two models' scores are comparable only
        on the same set.
    gradient : np.ndarray or None, shape (m,)
        Its gradient there; None when the strengths were given.
    sessions : tuple of RemlScore
        One per session given, in order, each with its fit, covariance
        (``A^-1``, mgcv's ``Vp``), edf per block (mgcv's ``edf`` summed over
        the term's coefficients) and score; :func:`in_null_space` flags its
        terms shrunk to a line. A session set aside by the search
        (``search.set_aside``; not in ``search.in_choice``) and fitted
        afterward has its fit at ``rho`` (NaN at the columns its refit left
        out); one not fitted has NaN coefficients, covariance, edf, scale and
        score, and ``fit.status`` says how its last fit ended.
    search : PenaltySearch or None
        None when the strengths were given (:func:`fit_at_penalties`).
    """

    rho: np.ndarray
    score: float
    gradient: np.ndarray | None
    sessions: tuple[RemlScore, ...]
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


def assemble_penalty(
    blocks: Sequence[PenaltyBlock], rho: Sequence[float], n_columns: int
) -> np.ndarray:
    """``sum_j exp(rho_j) S_j``, embedded, shape ``(n_columns, n_columns)``."""
    _check_blocks(blocks, n_columns)
    rho = np.asarray(rho, dtype=float)
    if rho.shape != (len(blocks),):
        raise ValueError(f"{rho.size} log penalties for {len(blocks)} blocks")
    return _summed(_penalty_stack(blocks, rho, n_columns))


def _penalty_stack(
    blocks: Sequence[PenaltyBlock], rho: np.ndarray, n_columns: int
) -> np.ndarray:
    """Each ``exp(rho_j) S_j``, embedded, stacked, shape ``(m, p, p)``."""
    stack = np.zeros((len(blocks), n_columns, n_columns))
    for j, (block, r) in enumerate(zip(blocks, rho, strict=True)):
        stack[j][np.ix_(block.columns, block.columns)] = np.exp(r) * block.S
    return stack


def _summed(stack: np.ndarray) -> np.ndarray:
    """The sum of the stacked blocks, shape ``(p, p)`` (zeros for none): added
    in order from zeros, so that a -0 of a block's penalty is summed to 0."""
    return np.add.reduce(stack, axis=0, initial=0.0)


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


def _eigenbasis(blocks: Sequence[PenaltyBlock], n_columns: int) -> list[PenaltyBlock]:
    """The blocks in the eigenbasis of their penalties (diagonal).

    The coefficients ``gamma = V_j' beta_j`` of a block are an orthogonal
    change of its coefficients in which its penalty is
    ``diag(PenaltyBlock.eigenvalues)``, exactly zero on its null space.
    """
    _check_blocks(blocks, n_columns)
    return [block.in_eigenbasis for block in blocks]


def _rotate(
    blocks: Sequence[PenaltyBlock], M: np.ndarray, *, inverse: bool = False
) -> np.ndarray:
    """``Q M``, or ``Q' M`` when ``inverse``, with ``Q`` the block-diagonal
    rotation by the blocks' eigenvectors (identity off the blocks).

    ``M`` has shape ``(p,)`` or ``(p, k)``.
    """
    out = np.array(M, dtype=float)
    for block in blocks:
        V = block.eigenvectors
        out[block.columns] = (V.T if inverse else V) @ M[block.columns]
    return out


def _rotate_both(blocks: Sequence[PenaltyBlock], M: np.ndarray) -> np.ndarray:
    """``Q M Q'`` for a symmetric ``M`` of shape ``(p, p)``."""
    return _rotate(blocks, _rotate(blocks, M).T)


def _null_dimension(blocks: Sequence[PenaltyBlock], n_columns: int) -> int:
    """``M_p``, the dimension of the summed penalty's null space."""
    return n_columns - sum(block.rank for block in blocks)


def _coefficient_derivatives(S_j, b, factor):
    """``S_j b`` and ``b_j = db/drho_j = -A^-1 S_j b``, each shape ``(m, p)``,
    from the Cholesky ``factor`` of ``A``; ``S_j`` has shape ``(m, p, p)``."""
    S_j_b = S_j @ b
    return S_j_b, -jax.scipy.linalg.cho_solve(factor, S_j_b.T).T


def _block_masks(blocks: Sequence[PenaltyBlock], n_columns: int) -> np.ndarray:
    """1 at each block's columns, shape ``(m, n_columns)``."""
    masks = np.zeros((len(blocks), n_columns))
    for j, block in enumerate(blocks):
        masks[j, block.columns] = 1.0
    return masks


def _inverse_and_edf(factor, H, masks):
    """``A^-1``, ``log|A| / 2``, the edf per block and in all, from the
    Cholesky ``factor`` of ``A``; ``H`` is the information. The edf of a block
    is the trace of its rows and columns of ``A^-1 H``."""
    covariance = jax.scipy.linalg.cho_solve(factor, jnp.eye(H.shape[0]))
    diagonal = jnp.diagonal(covariance @ H)
    half_log_det = jnp.sum(jnp.log(jnp.diagonal(factor[0])))
    return covariance, half_log_det, masks @ diagonal, jnp.sum(diagonal)


def _laplace_summary(A, S, b, masks, loglik, log_pdet, n_null):
    """The Cholesky factor of ``A``, and :func:`_fixed_penalty_summary` on it."""
    factor = jax.scipy.linalg.cho_factor(A, lower=True)
    covariance, half_log_det, edf, edf_total = _inverse_and_edf(factor, A - S, masks)
    score = (
        -loglik
        + 0.5 * b @ S @ b
        - 0.5 * log_pdet
        + half_log_det
        - 0.5 * n_null * _LOG_2PI
    )
    return factor, (score, covariance, edf, edf_total)


@float64_kernel
def _fixed_penalty_summary(A, S, b, masks, loglik, log_pdet, n_null):
    """The REML score at a converged fit, ``A^-1`` and the edf.

    ``V = -l + b' S b / 2 - log|S|_+ / 2 + log|A| / 2 - M_p log(2 pi) / 2``,
    with ``loglik`` the fit's ``l``, ``log_pdet`` ``log|S|_+`` and ``n_null``
    ``M_p``. ``masks`` is :func:`_block_masks`.
    """
    return _laplace_summary(A, S, b, masks, loglik, log_pdet, n_null)[1]


def _require_factored(factored: bool) -> None:
    """Raise ``numpy.linalg.LinAlgError`` unless the derivative kernel could
    factor ``A`` (a converged fit's ``A`` is positive definite, so this is a
    guard: a derivative of NaN would pass on silently)."""
    if not factored:
        raise np.linalg.LinAlgError(
            "A = H + S is not positive definite at the converged fit: its "
            "Cholesky factor is not finite, so the score has no derivatives"
        )


@float64_kernel(static_argnames=("lik",))
def _laplace_derivatives(
    lik, X, Y, aux, weight, S_j, A, S, b, masks, loglik, log_pdet, n_null, ranks
):
    """:func:`_fixed_penalty_summary` and the gradient and Hessian of the
    score in ``rho``, on one Cholesky factor of ``A``.

    ``X``, ``Y`` (shape ``(1, n_bucket)``), ``aux`` and ``weight`` are the
    Newton fit's padded rows; ``S_j`` has shape ``(m, p, p)``, each
    ``exp(rho_j) S_j`` embedded, and ``ranks`` shape ``(m,)``. With ``b_j =
    -A^-1 S_j b``, ``A_j = S_j + dH(X b_j)`` and ``b_jk = -A^-1 (A_k b_j + S_j
    b_k) + delta_jk b_j``:

        dV/drho_j = (b' S_j b - r_j + tr(A^-1 A_j)) / 2
        d2V/drho_j drho_k = (delta_jk b' S_j b + 2 b_k' S_j b
            - tr(A^-1 A_k A^-1 A_j) + tr(A^-1 A_jk)) / 2,

    ``A_jk = d2H(X b_j, X b_k) + dH(X b_jk) + delta_jk S_j``, its trace read by
    ``lik._jax_information_traces`` against ``A^-1``. The ``m`` matrices
    ``dH(X b_j)`` are made one at a time (``jax.lax.map``); the Hessian is
    computed for ``j <= k`` and mirrored. The last output is whether ``A`` and
    its factor are finite (:func:`_require_factored`).
    """
    factor, summary = _laplace_summary(A, S, b, masks, loglik, log_pdet, n_null)
    factored = _factored(A, factor)
    covariance = summary[1]
    y = Y[0]
    m = S_j.shape[0]
    if m == 0:
        return (*summary, jnp.zeros(0), jnp.zeros((0, 0)), factored)
    eta = X @ b
    S_j_b, b_j = _coefficient_derivatives(S_j, b, factor)
    E = X @ b_j.T  # (n_bucket, m): eta's derivative in each rho_j

    def dinformation(j):
        e_j = jax.lax.dynamic_index_in_dim(E, j, axis=1, keepdims=False)
        return lik._jax_dinformation(X, eta, y, aux, weight, e_j)

    A_j = S_j + jax.lax.map(dinformation, jnp.arange(m))
    Ainv_A_j = jax.vmap(lambda M: jax.scipy.linalg.cho_solve(factor, M))(A_j)
    b_S_b = S_j_b @ b
    gradient = 0.5 * (b_S_b - ranks + jnp.trace(Ainv_A_j, axis1=1, axis2=2))

    j, k, diagonal = _upper_pairs(m)
    right = (A_j[k] @ b_j[j][:, :, None])[:, :, 0] + (S_j[j] @ b_j[k][:, :, None])[
        :, :, 0
    ]
    b_jk = -jax.scipy.linalg.cho_solve(factor, right.T).T + jnp.where(
        diagonal[:, None], b_j[j], 0.0
    )
    trace_first, trace_second = lik._jax_information_traces(
        X, eta, y, aux, weight, covariance, X @ b_jk.T, E
    )
    trace_S = jnp.sum(covariance * S_j, axis=(1, 2))  # tr(A^-1 S_j)
    upper = 0.5 * (
        jnp.where(diagonal, b_S_b[j], 0.0)
        + 2.0 * jnp.sum(b_j[k] * S_j_b[j], axis=1)
        - jnp.sum(Ainv_A_j[k] * jnp.swapaxes(Ainv_A_j[j], 1, 2), axis=(1, 2))
        + trace_second[j, k]
        + trace_first
        + jnp.where(diagonal, trace_S[j], 0.0)
    )
    return (*summary, gradient, _mirrored(upper, j, k, m), factored)


@float64_kernel
def _profiled_gaussian(
    X, Y, weight, S, b, masks, log_pdet, n_real, dof, S_j=None, ranks=None
):
    """The Gaussian score with the scale profiled out, at the penalized least
    squares fit ``b`` (:func:`_gaussian_profiled`), on padded rows, with its
    gradient and Hessian in ``rho``.

    ``Y`` has shape ``(1, n_bucket)``. Returns ``X'X + S``, ``D_p``, the
    scale ``D_p / dof``, the score, the covariance ``scale (X'X + S)^-1``, the
    edf per block and in all, and the log-likelihood at that scale (``n_real``
    real rows). Given ``S_j``, shape ``(m, p, p)``, each ``exp(rho_j) S_j``
    embedded, and ``ranks``, shape ``(m,)``, it also returns the gradient,
    the Hessian and whether ``X'X + S`` and its factor are finite, all on one
    Cholesky factor of ``X'X + S``.
    """
    gram = _gram(X, weight) + S
    residual = Y[0] - X @ b
    weighted_rss = (weight * residual) @ residual
    D_p = weighted_rss + b @ (S @ b)
    scale = D_p / dof
    factor = jax.scipy.linalg.cho_factor(gram, lower=True)
    inverse, half_log_det, edf, edf_total = _inverse_and_edf(factor, gram - S, masks)
    score = (
        0.5 * dof * (1.0 + _LOG_2PI + jnp.log(scale)) - 0.5 * log_pdet + half_log_det
    )
    loglik = -weighted_rss / (2.0 * scale) - 0.5 * n_real * jnp.log(
        2.0 * jnp.pi * scale
    )
    summary = (gram, D_p, scale, score, scale * inverse, edf, edf_total, loglik)
    if S_j is None:
        return summary
    factored = _factored(gram, factor)
    m = S_j.shape[0]
    if m == 0:
        return (*summary, jnp.zeros(0), jnp.zeros((0, 0)), factored)
    # D_p has derivatives b' S_j b and delta_jk b' S_j b + 2 b_k' S_j b, as in
    # the Laplace case with the information fixed.
    S_j_b, b_j = _coefficient_derivatives(S_j, b, factor)
    Ainv_S_j = jax.vmap(lambda M: jax.scipy.linalg.cho_solve(factor, M))(S_j)
    D_j = S_j_b @ b
    j, k, diagonal = _upper_pairs(m)
    D_jk = 0.5 * (
        2.0 * jnp.where(diagonal, D_j[j], 0.0)
        + 2.0 * jnp.sum(b_j[k] * S_j_b[j], axis=1)
        + 2.0 * jnp.sum(b_j[j] * S_j_b[k], axis=1)
    )
    trace_j = jnp.trace(Ainv_S_j, axis1=1, axis2=2)
    trace_jk = jnp.sum(Ainv_S_j[k] * jnp.swapaxes(Ainv_S_j[j], 1, 2), axis=(1, 2))
    gradient = 0.5 * (dof * D_j / D_p - ranks + trace_j)
    upper = 0.5 * (
        dof * (D_jk / D_p - D_j[j] * D_j[k] / D_p**2)
        + jnp.where(diagonal, trace_j[j], 0.0)
        - trace_jk
    )
    return (*summary, gradient, _mirrored(upper, j, k, m), factored)


@in_float64
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
    rotated = _eigenbasis(blocks, n_columns)
    if rho.shape != (len(blocks),):
        raise ValueError(f"{rho.size} log penalties for {len(blocks)} blocks")
    # Counted before the rotation, which spreads a value that is not finite
    # over its row's columns of the block.
    _check_design_finite(np.asarray(X))
    # X Q as one product: assigning each block's columns by index took 3 to 6
    # times as long (fancy indexing copies) at the rate model's size.
    X_rotated = X @ _rotate(blocks, np.eye(n_columns))
    start = np.zeros(n_columns) if beta0 is None else np.asarray(beta0, dtype=float)
    session = _reml_in_basis(
        lik,
        X_rotated,
        y,
        aux,
        rotated,
        rho,
        _rotate(blocks, start, inverse=True),
        derivatives,
    )
    fit = replace(
        session.fit,
        beta=_rotate(blocks, session.fit.beta),
        A=_rotate_both(blocks, session.fit.A),
    )
    covariance = (
        None if session.covariance is None else _rotate_both(blocks, session.covariance)
    )
    directions: tuple[FreeDirection, ...] = ()
    if fit.step is not None:
        directions = _directions(rotated, fit.running_away)
        step = _rotate(blocks, fit.step)
        fit = replace(fit, step=step, running_away=running_away_columns(X, step))
    unidentified: tuple[FreeDirection, ...] = ()
    if fit.unidentified is not None:
        named = _named_coordinates(fit.unidentified)
        unidentified = tuple(
            sorted(
                _directions(rotated, named),
                key=lambda d: (d.block is not None, d.block or 0, d.index),
            )
        )
        fit = replace(fit, unidentified=_rotate(blocks, fit.unidentified))
    return replace(
        session,
        fit=fit,
        covariance=covariance,
        running_away_directions=directions,
        unidentified_directions=unidentified,
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
    S_j = _penalty_stack(blocks, rho, n_columns)
    S = _summed(S_j)
    if isinstance(lik, Gaussian):
        return _gaussian_profiled(X, y, blocks, rho, S_j, S, start, derivatives)

    fit, problem = _fit_one(lik, X, y, aux, S, start, free_basis(blocks, n_columns))
    if not fit.converged:
        return RemlScore(np.inf, None, None, fit, 1.0)
    common = (
        fit.A,
        S,
        fit.beta,
        _block_masks(blocks, n_columns),
        np.float64(fit.loglik),
        np.float64(_log_pdet(blocks, rho)),
        np.float64(_null_dimension(blocks, n_columns)),
    )
    if derivatives:
        r = problem.rows
        outputs = jax.device_get(
            _laplace_derivatives(
                lik,
                r.X,
                problem.Y,
                r.aux,
                r.weight,
                S_j,
                *common,
                _ranks(blocks),
            )
        )
        return _score_of(fit, 1.0, outputs[:4], outputs[4:])
    return _score_of(fit, 1.0, jax.device_get(_fixed_penalty_summary(*common)), None)


def _score_of(
    fit: NewtonFit,
    scale: float,
    summary: Sequence[Any],
    derivatives: Sequence[Any] | None,
) -> RemlScore:
    """The :class:`RemlScore` of a kernel's outputs on the host: ``summary``
    the score, the covariance, the edf per block and in all; ``derivatives``
    the gradient, the Hessian and whether ``A`` could be factored
    (:func:`_require_factored`), or None when they were not asked for."""
    score, covariance, edf, edf_total = summary
    gradient = hessian = None
    if derivatives is not None:
        gradient, hessian, factored = derivatives
        _require_factored(bool(factored))
        gradient, hessian = np.array(gradient), np.array(hessian)
    return RemlScore(
        float(score),
        gradient,
        hessian,
        fit,
        scale,
        covariance=np.array(covariance),
        edf=np.array(edf),
        edf_total=float(edf_total),
    )


def _ranks(blocks: Sequence[PenaltyBlock]) -> np.ndarray:
    """Each block's rank, as floats, shape ``(m,)``."""
    return np.array([block.rank for block in blocks], dtype=float)


def _gaussian_profiled(
    X: np.ndarray,
    y: np.ndarray,
    blocks: Sequence[PenaltyBlock],
    rho: np.ndarray,
    S_j: np.ndarray,
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
    the tolerance it is held to, do not depend on the outcome's units. The
    score, its derivatives (:func:`_profiled_gaussian`) and the rest are
    computed on the fit's padded rows; the derivatives need no product with
    the design beyond the Gram.
    """
    n_rows, n_columns = X.shape
    working = float(np.var(y))
    if not working > 0:
        raise ValueError(
            "the Gaussian outcome is constant (variance zero): it has no scale to "
            "profile"
        )
    pls, problem = _fit_one(
        Gaussian(),
        X,
        y,
        working,
        S / working,
        beta0,
        free_basis(blocks, n_columns),
    )
    if not pls.converged:
        return RemlScore(np.inf, None, None, pls, np.nan)
    rows = problem.rows
    # The penalties' stack goes to the device only when the derivatives read it.
    extra = (S_j, _ranks(blocks)) if derivatives else ()
    outputs = jax.device_get(
        _profiled_gaussian(
            rows.X,
            problem.Y,
            rows.weight,
            S,
            pls.beta,
            _block_masks(blocks, n_columns),
            np.float64(_log_pdet(blocks, rho)),
            np.int64(n_rows),
            np.int64(n_rows - _null_dimension(blocks, n_columns)),
            *extra,
        )
    )
    gram, D_p, scale, score, covariance, edf, edf_total, loglik = outputs[:8]
    scale = float(scale)
    fit = replace(
        pls,
        A=np.array(gram) / scale,
        newton_decrement=pls.newton_decrement * working / scale,
        loglik=float(loglik),
    )
    return _score_of(
        fit,
        scale,
        (score, covariance, edf, edf_total),
        outputs[8:] if derivatives else None,
    )


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


def _positions(active: np.ndarray) -> np.ndarray:
    """Each column's index among the active ones (read at the active columns
    only), shape ``(p,)``."""
    return np.cumsum(active) - 1


def _embed(values: np.ndarray, active: np.ndarray, axes: tuple[int, ...]) -> np.ndarray:
    """``values`` of the active columns put back at full width along ``axes``,
    NaN at the inactive columns."""
    shape = [active.size if a in axes else n for a, n in enumerate(values.shape)]
    out = np.full(shape, np.nan)
    out[
        np.ix_(
            *(active if a in axes else np.ones(n, bool) for a, n in enumerate(shape))
        )
    ] = values
    return out


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
    position = _positions(active)
    reduced = [block.at_columns(position[block.columns]) for block in blocks]
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
    covariance = None
    if session.covariance is not None:
        covariance = _embed(session.covariance, active, (0, 1))
    fit = replace(
        session.fit,
        beta=_embed(session.fit.beta, active, (0,)),
        A=_embed(session.fit.A, active, (0, 1)),
    )
    columns = np.flatnonzero(active)

    def in_session(directions: tuple[FreeDirection, ...]) -> tuple[FreeDirection, ...]:
        return tuple(
            d._replace(index=int(columns[d.index])) if d.block is None else d
            for d in directions
        )

    if fit.step is not None:
        fit = replace(
            fit,
            step=_embed(fit.step, active, (0,)),
            running_away=columns[fit.running_away],
        )
    if fit.unidentified is not None:
        fit = replace(fit, unidentified=_embed(fit.unidentified, active, (0,)))
    return replace(
        session,
        fit=fit,
        covariance=covariance,
        running_away_directions=in_session(session.running_away_directions),
        unidentified_directions=in_session(session.unidentified_directions),
    )


def _labeled_score(
    lik: Likelihood,
    data: SessionData,
    index: int,
    blocks: Sequence[PenaltyBlock],
    rho: Sequence[float],
    start: np.ndarray | None,
    derivatives: bool,
) -> RemlScore:
    """:func:`_session_score` of session ``index``, an input error raised again
    with the session's label."""
    try:
        return _session_score(lik, data, blocks, rho, start, derivatives)
    except ValueError as error:
        raise ValueError(f"{_label(data, index)}: {error}") from error


def _pooled_failure(
    pooled: PooledScore, sessions: Sequence[SessionData] | dict[int, SessionData]
) -> str | None:
    """Why a pooled score is infinite, in words (:func:`_failure`), with
    ``sessions`` indexed as ``pooled.failed_session``; None when no session
    failed."""
    i = pooled.failed_session
    return None if i is None else _failure(sessions[i], i, pooled.failed)


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
    starts = [None] * len(sessions)
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
    (its place among all of a search's sessions), whose active masks the
    caller has checked."""
    scores = []
    for index, data, start in zip(indices, sessions, starts, strict=True):
        session = _labeled_score(lik, data, index, blocks, rho, start, derivatives)
        if not np.isfinite(session.score):
            return PooledScore(np.inf, None, None, (), index, session)
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
        self.by_index = dict(zip(indices, sessions, strict=True))
        self.starts: list[np.ndarray | None] = [None] * len(sessions)

    def __call__(self, rho: np.ndarray, derivatives: bool) -> PooledScore:
        pooled = _pooled_reml(
            self.lik,
            list(self.by_index.values()),
            list(self.by_index),
            self.blocks,
            rho,
            self.starts,
            derivatives,
        )
        if np.isfinite(pooled.score):
            self.starts = [s.fit.beta for s in pooled.sessions]
        return pooled

    def failure(self, pooled: PooledScore) -> str | None:
        """Why ``pooled`` is infinite, in words; None when no session failed."""
        return _pooled_failure(pooled, self.by_index)

    def label(self, index: int) -> str:
        """The label of the session at ``index`` among all sessions given."""
        return _label(self.by_index[index], index)


def _grid(m: int) -> list[np.ndarray]:
    """The grid points for one to three terms."""
    lo, hi = LOG_PENALTY_BOUNDS
    if m == 1:
        return [np.array([r]) for r in np.arange(lo, hi + 0.5)]
    return [np.array(point) for point in itertools.product(REML_GRID_POINTS, repeat=m)]


def _grid_start(
    evaluate: _Evaluator, m: int
) -> tuple[np.ndarray, int, Counter[tuple[int, str]]]:
    """The best grid point, the number of points evaluated, and the points
    whose score was not finite counted by (first failing session, status).

    Raises
    ------
    ValueError
        If no grid point gives a finite score; the message names the sessions
        that failed and why.
    """
    best_rho, best_score, n_points = np.full(m, np.nan), np.inf, 0
    failures: Counter[tuple[int, str]] = Counter()

    def visit(rho: np.ndarray) -> None:
        nonlocal best_rho, best_score, n_points
        pooled = evaluate(rho, derivatives=False)
        n_points += 1
        if not np.isfinite(pooled.score):
            failures[pooled.failed_session, pooled.failed_status] += 1
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
            f"{evaluate.label(index)}: {status} at {count} of {n_points} points"
            for (index, status), count in failures.items()
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
    if trial is not None and trial.failed_session is not None:
        reason += f" (the last halving failed in {evaluate.failure(trial)})"
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
        converge: ``REML_MAX_STEPS`` steps without a stationary point (a point
        of negative curvature reached at the last step is not one), an empty
        step or no halving that lowers the score. The message gives the last log penalties, the
        gradient and the steps taken.
    """
    rho = np.array(start, dtype=float)
    current = evaluate(rho, derivatives=True)
    if not np.isfinite(current.score):
        raise ValueError(
            f"no finite REML score at the start {rho}: {evaluate.failure(current)}"
        )
    start_score, n_steps = current.score, 0
    while True:
        curvature = _Curvature.at(rho, current)
        step = curvature.newton_step(current.gradient)
        rounding = _rounding_slack(current.score)
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
        elif small and not probes:  # the last step, taken unless the score rises
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
        session = _labeled_score(lik, data, index, blocks, reference, None, False)
        status = session.fit.status
        if np.isfinite(session.score):
            kept.append(index)
        elif status == NO_FINITE_MAXIMUM:
            set_aside.append((index, session))
        elif status == NOT_IDENTIFIED:
            raise ValueError(
                f"{_label(data, index)}: design not identified at the reference log "
                f"penalties {reference} ({_unidentified_text(session)}); mark the "
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


def _named_coordinates(combinations: np.ndarray) -> np.ndarray:
    """The coordinates that take part in ``combinations``, shape ``(p, k)``
    with orthonormal columns: those with more than ``NAMED_SHARE`` of their
    length in the space the columns span. The length of a coordinate in that
    space is the norm of its row, whichever basis of the space is given."""
    return np.flatnonzero(np.linalg.norm(combinations, axis=1) > NAMED_SHARE)


def _unidentified_text(session: RemlScore) -> str:
    """What a design does not identify, in words: the coordinates the
    likelihood cannot see, else the coordinates collinear to it.

    ``k`` combinations name ``k`` coordinates only when each is one coordinate
    that the likelihood cannot see: ``k`` collinear combinations of columns
    that are not zero take part in more than ``k`` coordinates.
    """
    directions = session.unidentified_directions
    if len(directions) != session.fit.unidentified.shape[1]:
        return "collinear to the likelihood: " + ", ".join(str(d) for d in directions)
    why = "(zero on every row; for the grouped softmax, constant within every group)"
    if all(d.block is None for d in directions):
        return (
            f"columns the likelihood cannot see {why}: {[d.index for d in directions]}"
        )
    names = ", ".join(str(d) for d in directions)
    return f"coordinates the likelihood cannot see {why}: {names}"


class RefitEnd(StrEnum):
    """How :func:`refit_session` ended: with a fit, or why without one
    (``record.NoFit`` takes the members but ``CONVERGED``)."""

    CONVERGED = "converged"
    #: No finite maximum along the intercept.
    ALONG_THE_INTERCEPT = "along_the_intercept"
    #: No finite maximum along a penalty block.
    ALONG_A_PENALTY = "along_a_penalty"
    #: Still no finite maximum after ``REFIT_MAX_ROUNDS`` rounds.
    TOO_MANY_ROUNDS = "too_many_rounds"
    #: Another status than converged or no finite maximum.
    OTHER_STATUS = "other_status"


@dataclass(frozen=True)
class SessionRefit:
    """One session's fit at given log penalties with the columns running away
    made inactive (:func:`refit_session`).

    Attributes
    ----------
    refits : tuple of RemlScore
        Its fits, in order: as given, then after each round of columns made
        inactive.
    left_out : np.ndarray of int, shape (k,)
        The columns made inactive, in the order they were.
    rounds : np.ndarray of int, shape (k,)
        The refit in which each of ``left_out`` was made inactive (1 for the
        first).
    ended : RefitEnd
        How it ended.
    reason : str
        How it ended, in words for a message (with the directions or the
        columns still running away).
    """

    refits: tuple[RemlScore, ...]
    left_out: np.ndarray
    rounds: np.ndarray
    ended: RefitEnd
    reason: str

    @property
    def fitted(self) -> bool:
        """True when its last fit converged."""
        return self.ended == RefitEnd.CONVERGED


#: Why a refit leaves a session without a fit when its fit has no finite
#: maximum along the intercept (:func:`refit_session`).
ALONG_THE_INTERCEPT = (
    "no finite maximum along the intercept: the rows of the reference level, or "
    "of the baseline, hold no event of the kind"
)


def _intercepts(X: np.ndarray) -> np.ndarray:
    """The design columns with one nonzero value on every row (the
    intercept), shape ``(k,)``."""
    if X.shape[0] == 0:
        return np.empty(0, dtype=int)
    same = np.all(X == X[0], axis=0) & (X[0] != 0)
    return np.flatnonzero(same)


def refit_session(
    lik: Likelihood,
    data: SessionData,
    blocks: Sequence[PenaltyBlock],
    rho: Sequence[float],
    *,
    index: int = 0,
) -> SessionRefit:
    """A session's fit at ``rho``, its columns running away made inactive.

    The rule (the penalty search's for a session it sets aside, and a
    per-session fit's at given penalties): the session is fitted at ``rho``;
    while that fit has no finite maximum along unpenalized design columns only
    (no direction through a penalty block), the columns running away are made
    inactive and it is fitted again, at most ``REFIT_MAX_ROUNDS`` times (a
    direction spread over near-collinear columns can name only part of them
    at a time). A direction through a penalty block (a smooth's own line, or
    its penalized coordinates) cannot be left out, and the session stays
    without a fit; so does a fit that ends with any other status. The
    intercept is never made inactive: a session fitted without it is a fit of
    another model. When the columns running away include it, the session
    stays without a fit, with the reason :data:`ALONG_THE_INTERCEPT`, also
    in the last round allowed (no further round could leave it out). The
    intercept is found from the data: a design column with the same nonzero
    value on every row. A grouped-softmax design has none (such a column is
    constant within every group, which the fitter refuses as not
    identified). Neither is
    an error here: the caller decides (the search raises at its reference
    point for a design not identified or a numerical failure, before any
    refit).

    Parameters
    ----------
    lik : Likelihood
    data : SessionData
    blocks : sequence of PenaltyBlock
    rho : sequence of float
    index : int
        The session's place among the caller's, for messages.

    Returns
    -------
    SessionRefit

    Raises
    ------
    ValueError
        If a column of a penalty block is inactive, or the inputs cannot be
        fitted; the message names the session.
    """
    _check_active(data, index, blocks)
    active = (
        np.ones(data.X.shape[1], dtype=bool)
        if data.active is None
        else data.active.copy()
    )
    left_out: list[int] = []
    rounds: list[int] = []
    refits = [_labeled_score(lik, data, index, blocks, rho, None, False)]
    ended, reason = RefitEnd.CONVERGED, ""
    while not refits[-1].fit.converged:
        last = refits[-1]
        if last.fit.status != NO_FINITE_MAXIMUM:
            ended = RefitEnd.OTHER_STATUS
            reason = f"its fit at the chosen log penalties ended as {last.fit.status}"
            break
        along = last.running_away_directions
        if any(d.block is not None for d in along):
            ended = RefitEnd.ALONG_A_PENALTY
            reason = (
                "no finite maximum along a penalty block ("
                + ", ".join(str(d) for d in along)
                + "), whose columns cannot be made inactive"
            )
            break
        if np.intersect1d(last.fit.running_away, _intercepts(data.X)).size:
            ended, reason = RefitEnd.ALONG_THE_INTERCEPT, ALONG_THE_INTERCEPT
            break
        if len(refits) > REFIT_MAX_ROUNDS:
            ended = RefitEnd.TOO_MANY_ROUNDS
            reason = (
                f"no finite maximum after {REFIT_MAX_ROUNDS} rounds of columns "
                f"made inactive (still running away: {last.fit.running_away.tolist()})"
            )
            break
        new = [int(c) for c in last.fit.running_away if active[c]]
        active[new] = False
        left_out.extend(new)
        rounds.extend([len(refits)] * len(new))
        refits.append(
            _labeled_score(
                lik,
                replace(data, active=active.copy()),
                index,
                blocks,
                rho,
                None,
                False,
            )
        )
    if refits[-1].fit.converged:
        reason = "converged at the chosen log penalties" + (
            f" with columns {left_out} inactive" if left_out else ""
        )
    return SessionRefit(
        tuple(refits),
        np.array(left_out, dtype=int),
        np.array(rounds, dtype=int),
        ended,
        reason,
    )


def _refit_set_aside(
    lik: Likelihood,
    sessions: Sequence[SessionData],
    blocks: Sequence[PenaltyBlock],
    rho: np.ndarray,
    set_aside: Sequence[tuple[int, RemlScore]],
) -> tuple[SetAsideSession, ...]:
    """Each set-aside session refitted at ``rho`` (:func:`refit_session`).

    Raises
    ------
    ValueError
        If that share exceeds ``MAX_FAILING_SHARE``; the message names each
        session without a fit, its statuses and why.
    """
    records = tuple(
        SetAsideSession(
            i,
            sessions[i].name,
            score,
            refit_session(lik, sessions[i], blocks, rho, index=i),
        )
        for i, score in set_aside
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


def _pooled_fit(
    rho: np.ndarray,
    pooled: PooledScore,
    blocks: Sequence[PenaltyBlock],
    search: PenaltySearch | None,
) -> PooledFit:
    """The per-session fits, covariances and edf of a pooled score.

    A session's covariance is NaN in the rows and columns of its inactive
    columns; its edf counts the active ones. A session the search set aside
    is put back in its place: its refit at ``rho`` when that converged, else
    NaN values.
    """
    sessions = list(pooled.sessions)
    for left_out in () if search is None else search.set_aside:
        entry = left_out.refits[-1]
        if not left_out.fitted:
            entry = entry.without_fit(len(blocks))
        sessions.insert(left_out.index, entry)
    return PooledFit(rho, pooled.score, pooled.gradient, tuple(sessions), search)


def _by_status(failures: Counter[tuple[int, str]]) -> dict[str, int]:
    """Grid points that failed, counted by status: the sum over the sessions
    that failed first with that status."""
    counts: Counter[str] = Counter()
    for (_, status), n in failures.items():
        counts[status] += n
    return dict(counts)


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
        grid_failures=_by_status(failures),
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
    log penalties (its penalty does not depend on its own data); while that
    fit has no finite maximum along unpenalized design columns only, those
    columns are made inactive and it is fitted again, by :func:`refit_session`
    (its docstring states the rule; a direction through a penalty block or the
    intercept cannot be left out, and the session stays without a fit).
    ``PenaltySearch.set_aside`` records every step. More than ``MAX_FAILING_SHARE`` of the sessions without a fit at the
    end is an error.

    The search starts from the best point of a grid over the log penalties
    (the score can have two optima; the grid is described beside
    ``REML_GRID_POINTS`` in :mod:`.constants`), then takes Newton steps on the
    log penalties with the exact Hessian, an eigenvalue floor, step halving and
    the bounds, to the stopping rule of :func:`_search`.

    Parameters
    ----------
    lik : Likelihood
    sessions : sequence of SessionData
    blocks : sequence of PenaltyBlock, length m >= 1

    Returns
    -------
    PooledFit
        One entry per session given; a set-aside session has its refit, or NaN
        values when it has no fit. The
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
            f"an inner fit did not converge at log penalties {rho}: "
            f"{_pooled_failure(pooled, sessions)}"
        )
    return _pooled_fit(rho, pooled, blocks, None)
