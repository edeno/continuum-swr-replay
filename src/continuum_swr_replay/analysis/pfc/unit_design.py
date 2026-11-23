"""The design every unit of a session shares, in the regression convention's terms.

The units of one session are fit on the same bins (the immobile bins inside a
well visit, ``event_sets.build_visit_fit_mask``) against the same columns; only
the spike counts differ. :func:`build_unit_design` builds those columns once,
with their penalties stated as the shared fitter takes them
(``regression.reml.PenaltyBlock``):

- **The peri-event kernels.** Each kernel is a smooth of lag over the fit
  window: event ``e`` contributes ``f(t - onset_e)`` to the log rate at bin
  ``t``, with ``f = B beta`` on the present cubic B-spline basis
  (``design.event_basis_functions``: knots every 50 ms over +/-250 ms, both
  edge functions dropped so the kernel returns to zero at both ends). The basis
  is the one the shipped models use; its columns are the same. The penalty is
  the second-order difference penalty of the coefficients, ``D2' D2``
  (``regression.pspline.difference_penalty``), the convention's penalty for a
  smooth, in place of the ridge on each coefficient. Its null space, the
  coefficients linear in their index (a plateau and a ramp between the window's
  edges), is not penalized.
- **The marked kernels.** The same basis, each event's copy weighted by its
  mark (``design.convolve_event_basis`` of the weighted onsets), with the same
  penalty: the change of the kernel per unit of the mark.
- **The covariates and baselines** (optional): any terms of the fitter's
  vocabulary built on the pooled rows by the caller (``regression.terms``);
  their penalties are the terms' own. The intercept and the nuisance terms
  (the well, the visit's outcome) are not penalized; a simple coefficient that
  a model tests carries the weak penalty (``terms.log_penalty_from_prior_sd``)
  and is checked by ``support.require_tested_penalized``.

**Open (lead): the kernels' penalty.** :class:`KernelPenalty` names the two
settings: one strength per kernel term chosen by REML pooled over the units of
the session (``"reml_over_units"``, each unit's score summed), or one declared
fixed log penalty for every kernel term, the same in every session
(``"declared"``), the rule of the other families. The design is the same under
both; only how the strengths are set differs. **Open (lead): the width of the
weak penalty** (a prior SD of 1, 2 or 4, :data:`WEAK_PRIOR_SDS`).

Where it differs from the shipped design: the rows (in-visit bins, not the
stop cohort); the kernels' penalty (``D2' D2`` at one strength per kernel,
not ``1 / sigma**2`` on every coefficient), so a kernel's plateau and ramp are
not shrunk; and the baseline columns, which are terms with recorded penalties
and standardizations rather than free columns.

The support rule (:func:`unit_term_support`) reads the design only: a term is
absent from the session's fits, for every unit alike, when fewer than
``support.MIN_UNITS`` visits have a row on a side.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.pfc.design import (
    convolve_event_basis,
    event_basis_functions,
)
from continuum_swr_replay.analysis.pfc.marked_glm import weighted_event_onset_counts
from continuum_swr_replay.analysis.regression.pspline import difference_penalty
from continuum_swr_replay.analysis.regression.reml import (
    PenaltyBlock,
    assemble_penalty,
)
from continuum_swr_replay.analysis.regression.support import (
    MIN_UNITS,
    Removed,
    Support,
    binary_support,
    design_support,
    require_tested_penalized,
)
from continuum_swr_replay.analysis.regression.terms import INTERCEPT, Design

__all__ = [
    "KERNEL_PENALTY_RULES",
    "WEAK_PRIOR_SDS",
    "Kernel",
    "KernelPenalty",
    "UnitDesign",
    "build_unit_design",
    "unit_term_support",
    "without_terms",
]

#: The two settings of the open question on the kernels' penalty.
KERNEL_PENALTY_RULES = ("reml_over_units", "declared")

#: The prior SDs the open width of the weak penalty is chosen among.
WEAK_PRIOR_SDS = (1.0, 2.0, 4.0)

#: The order of the kernels' difference penalty.
KERNEL_PENALTY_ORDER = 2


@dataclass(frozen=True)
class KernelPenalty:
    """How the kernels' penalty strengths are set (open: the lead decides).

    Attributes
    ----------
    rule : {"reml_over_units", "declared"}
        ``"reml_over_units"``: one strength per kernel term, chosen by REML
        pooled over the session's units. ``"declared"``: one fixed log
        penalty for every kernel term, the same in every session.
    log_penalty : float or None
        The declared log penalty; None under ``"reml_over_units"``.
    """

    rule: Literal["reml_over_units", "declared"]
    log_penalty: float | None = None

    def __post_init__(self) -> None:
        if self.rule not in KERNEL_PENALTY_RULES:
            raise ValueError(
                f"rule must be one of {KERNEL_PENALTY_RULES}, not {self.rule!r}"
            )
        declared = self.rule == "declared"
        if declared != (self.log_penalty is not None):
            raise ValueError(
                "a declared kernel penalty needs its log penalty, and REML over "
                "the units takes none"
            )
        if declared and not np.isfinite(self.log_penalty):
            raise ValueError(f"log penalty {self.log_penalty} is not finite")

    def record(self) -> dict:
        """``{"rule": ..., "log_penalty": ...}``, for a fit's record."""
        return {"rule": self.rule, "log_penalty": self.log_penalty}


@dataclass(frozen=True)
class Kernel:
    """One peri-event kernel: an event class, optionally weighted by a mark.

    Attributes
    ----------
    event_class : str
        The key of the event frame its onsets come from.
    mark : str or None
        The event frame's column weighting each event's copy (a standardized
        mark); None for the ordinary kernel.
    """

    event_class: str
    mark: str | None = None

    @property
    def name(self) -> str:
        """``k(<class>)``, or ``k(<class>:<mark>)`` for a marked kernel."""
        inner = (
            self.event_class if self.mark is None else f"{self.event_class}:{self.mark}"
        )
        return f"k({inner})"


@dataclass(frozen=True, eq=False)
class UnitDesign:
    """The design all units of one session share (:func:`build_unit_design`).

    Attributes
    ----------
    X : NDArray, shape (n_fit_bins, p)
        The baseline columns (the intercept first), then each kernel's.
    coef_names : tuple of str
        One per column: the baseline's names, then ``<kernel>.<b>``.
    blocks : tuple of PenaltyBlock
        The baseline's penalty blocks, then one per kernel (``D2' D2``).
    log_penalties : NDArray, shape (n_blocks,)
        Each block's fixed log penalty; NaN for a kernel block whose strength
        REML over the units chooses.
    kernels : tuple of Kernel
    kernel_columns : dict of str to slice
        Each kernel's columns of ``X``, by kernel name.
    penalty : KernelPenalty
    rows : NDArray of bool, shape (n_time,)
        The fit bins on the session's time axis.
    onsets : dict of str to NDArray, shape (n_time,)
        Each kernel's (weighted) onsets on the session's time axis, by kernel
        name: what any curve over the lags is convolved with.
    spline_basis : NDArray, shape (n_lags, n_basis)
    lags : NDArray of int, shape (n_lags,)
        Lags in samples.
    lag_time : NDArray, shape (n_lags,)
        Lags in s.
    baseline : Design or None
        The baseline terms as built on the pooled rows; None for an intercept
        alone.
    baseline_level_rows : dict
        The baseline's level rows on the fit bins (``SessionDesign.level_rows``).
    tested : tuple of str
        The baseline coefficients the model tests (each with the weak penalty).
    """

    X: NDArray
    coef_names: tuple[str, ...]
    blocks: tuple[PenaltyBlock, ...]
    log_penalties: NDArray
    kernels: tuple[Kernel, ...]
    kernel_columns: dict[str, slice]
    penalty: KernelPenalty
    rows: NDArray
    onsets: dict[str, NDArray]
    spline_basis: NDArray
    lags: NDArray
    lag_time: NDArray
    baseline: Design | None = None
    baseline_level_rows: dict = field(default_factory=dict)
    tested: tuple[str, ...] = ()

    @property
    def n_baseline(self) -> int:
        """The baseline's columns, the intercept included."""
        return self.X.shape[1] - sum(
            s.stop - s.start for s in self.kernel_columns.values()
        )

    @property
    def kernel_blocks(self) -> NDArray:
        """The positions in :attr:`blocks` of the kernels' blocks."""
        n_baseline_blocks = len(self.blocks) - len(self.kernels)
        return np.arange(n_baseline_blocks, len(self.blocks))

    def with_kernel_log_penalties(self, rho: Sequence[float]) -> NDArray:
        """All blocks' log penalties, with ``rho`` (one per kernel) at the
        kernels' blocks: the strengths REML over the units chose."""
        out = self.log_penalties.copy()
        out[self.kernel_blocks] = np.asarray(rho, dtype=float)
        return out

    def penalty_matrix(self, log_penalties: Sequence[float] | None = None) -> NDArray:
        """``sum_j exp(rho_j) S_j``, shape ``(p, p)``.

        Parameters
        ----------
        log_penalties : sequence of float, optional
            One per block; :attr:`log_penalties` when None (which must then be
            finite: a declared kernel penalty).
        """
        rho = self.log_penalties if log_penalties is None else np.asarray(log_penalties)
        if not np.isfinite(rho).all():
            raise ValueError(
                "the kernels' strengths are chosen by REML over the units; pass "
                "them (with_kernel_log_penalties)"
            )
        return assemble_penalty(self.blocks, rho, self.X.shape[1])

    def curve(self, beta: NDArray, kernel: str) -> NDArray:
        """A kernel's log-rate curve over the lags, shape ``(n_lags, ...)``.

        Parameters
        ----------
        beta : NDArray, shape (p,) or (p, n_units)
        kernel : str
            A kernel name (:attr:`Kernel.name`).
        """
        return self.spline_basis @ np.asarray(beta)[self.kernel_columns[kernel]]

    def penalty_record(self) -> dict[str, dict]:
        """Term -> how its penalty is set: the baseline's
        (``Design.log_penalty_record``) and each kernel's, ``{"set": "fixed",
        "log_penalty": value}`` or ``{"set": "reml_over_units", "log_penalty":
        None}``."""
        out = {} if self.baseline is None else self.baseline.log_penalty_record()
        for kernel, rho in zip(
            self.kernels, self.log_penalties[self.kernel_blocks], strict=True
        ):
            out[kernel.name] = (
                {"set": "fixed", "log_penalty": float(rho)}
                if np.isfinite(rho)
                else {"set": "reml_over_units", "log_penalty": None}
            )
        return out


def build_unit_design(
    events: Mapping[str, pd.DataFrame],
    fit_mask: pd.Series,
    *,
    kernels: Sequence[Kernel],
    lags: NDArray,
    sampling_frequency: float,
    penalty: KernelPenalty,
    baseline: Design | None = None,
    baseline_rows: pd.DataFrame | None = None,
    tested: Sequence[str] = (),
) -> UnitDesign:
    """The design of one session's unit models.

    Parameters
    ----------
    events : mapping of str to pd.DataFrame
        Each event class's events (``start_time``, s, and each mark column a
        kernel reads), already selected (``event_sets.build_visit_event_sets``).
    fit_mask : pd.Series of bool, shape (n_time,)
        The fit bins, indexed by the session's time axis (s).
    kernels : sequence of Kernel
        The kernels, in column order; names must be unique.
    lags : NDArray of int, shape (n_lags,)
        Contiguous lags in samples (``design.symmetric_lags``).
    sampling_frequency : float
        Hz.
    penalty : KernelPenalty
        How the kernels' strengths are set.
    baseline : Design, optional
        Baseline terms built on the pooled rows (``terms.build_design`` with an
        intercept); None for an intercept alone.
    baseline_rows : pd.DataFrame, optional
        The fit bins' rows the baseline reads, in time order; required with
        ``baseline``. A row with a missing value is the caller's to drop.
    tested : sequence of str
        The baseline coefficients the model tests; each must carry a fixed
        penalty (the weak penalty).

    Returns
    -------
    UnitDesign

    Raises
    ------
    ValueError
        If kernel names repeat, the baseline has no intercept or its rows do
        not match the fit bins, or a tested coefficient has no fixed penalty.
    """
    names = [kernel.name for kernel in kernels]
    if len(set(names)) != len(names):
        raise ValueError(f"kernels repeat: {names}")
    rows = np.asarray(fit_mask, dtype=bool)
    time = fit_mask.index.to_numpy(dtype=float)
    spline_basis, lag_time, _ = event_basis_functions(lags, sampling_frequency)
    n_rows = int(rows.sum())

    if baseline is None:
        if tested:
            raise ValueError(f"tested coefficients {list(tested)} need a baseline")
        X0, coef_names, blocks, rhos, level_rows = (
            np.ones((n_rows, 1)),
            [INTERCEPT],
            [],
            [],
            {},
        )
    else:
        if baseline.intercept_column != 0:
            raise ValueError("the baseline must have the intercept in column 0")
        if baseline_rows is None or len(baseline_rows) != n_rows:
            raise ValueError("baseline_rows must hold one row per fit bin")
        require_tested_penalized(baseline, tested)
        built = baseline.build(baseline_rows)
        X0, _ = baseline.session(baseline_rows)
        coef_names = list(baseline.coef_names)
        blocks = list(baseline.blocks)
        rhos = list(baseline.log_penalties)
        level_rows = built.level_rows

    n_basis = spline_basis.shape[1]
    D2 = difference_penalty(n_basis, KERNEL_PENALTY_ORDER)
    rho_kernel = np.nan if penalty.log_penalty is None else penalty.log_penalty
    columns, onsets, kernel_columns = [X0], {}, {}
    start = X0.shape[1]
    for kernel in kernels:
        frame = events[kernel.event_class]
        weights = (
            None
            if kernel.mark is None
            else pd.to_numeric(frame[kernel.mark]).to_numpy(dtype=float)
        )
        onsets[kernel.name] = weighted_event_onset_counts(frame, time, weights)
        columns.append(
            convolve_event_basis(onsets[kernel.name], spline_basis, lags)[rows]
        )
        kernel_columns[kernel.name] = slice(start, start + n_basis)
        blocks.append(PenaltyBlock(np.arange(start, start + n_basis), D2))
        rhos.append(rho_kernel)
        coef_names += [f"{kernel.name}.{b}" for b in range(n_basis)]
        start += n_basis
    return UnitDesign(
        X=np.concatenate(columns, axis=1),
        coef_names=tuple(coef_names),
        blocks=tuple(blocks),
        log_penalties=np.asarray(rhos, dtype=float),
        kernels=tuple(kernels),
        kernel_columns=kernel_columns,
        penalty=penalty,
        rows=rows,
        onsets=onsets,
        spline_basis=spline_basis,
        lags=np.asarray(lags),
        lag_time=lag_time,
        baseline=baseline,
        baseline_level_rows=level_rows,
        tested=tuple(tested),
    )


def unit_term_support(design: UnitDesign, visit: NDArray) -> dict[str, Support]:
    """Each term's support on the session's fit bins, read from the design only.

    The units of the rule are the visits. A baseline term takes the fitter's
    rule (``support.design_support``). A kernel passes when at least
    ``MIN_UNITS`` visits have a fit bin inside one of its events' windows (a
    non-zero row) and as many a bin outside every window; a marked kernel
    when as many visits have a bin where the weighted rows sum below zero and
    as many above (the basis is non-negative, so the sign is that of the marks
    of the events whose windows cover the bin). The spike counts are never
    read, so the support is the same for every unit of the session.

    Parameters
    ----------
    design : UnitDesign
    visit : NDArray of int, shape (n_fit_bins,)
        Each fit bin's visit, as integer codes from 0.

    Returns
    -------
    dict of str to Support
        Term name -> its support: the baseline's simple terms, then each
        kernel.
    """
    visit = np.asarray(visit)
    out: dict[str, Support] = {}
    if design.baseline is not None:
        out.update(
            design_support(
                design.baseline,
                design.X[:, : design.n_baseline],
                design.baseline_level_rows,
                visit,
            )
        )
    for kernel in design.kernels:
        block = design.X[:, design.kernel_columns[kernel.name]]
        if kernel.mark is None:
            out[kernel.name] = binary_support(
                (np.abs(block).sum(axis=1) > 0).astype(float), visit
            )
        else:
            total = block.sum(axis=1)
            n_low = _n_visits(visit[total < 0])
            n_high = _n_visits(visit[total > 0])
            reason = "" if min(n_low, n_high) >= MIN_UNITS else Removed.SUPPORT
            out[kernel.name] = Support(n_low, n_high, reason)
    return out


def _n_visits(visit: NDArray) -> int:
    """The distinct visits among ``visit``."""
    return int(np.unique(visit).size)


def without_terms(design: UnitDesign, names: Sequence[str]) -> UnitDesign:
    """The design without some kernels (those that failed the support rule).

    A failing kernel is recorded and left out of every unit's fit in the
    session: its columns and its penalty block are removed (a penalized
    column cannot be made inactive). Baseline terms are made inactive by the
    fitter's own rule (``sessions``), not here.

    Raises
    ------
    ValueError
        If a name is not a kernel of the design.
    """
    unknown = sorted(set(names) - set(design.kernel_columns))
    if unknown:
        raise ValueError(f"not kernels of the design: {unknown}")
    keep_kernels = [k for k in design.kernels if k.name not in names]
    n0 = design.n_baseline
    n_basis = design.spline_basis.shape[1]
    columns = [np.arange(n0)] + [
        np.arange(
            design.kernel_columns[k.name].start, design.kernel_columns[k.name].stop
        )
        for k in keep_kernels
    ]
    columns = np.concatenate(columns)
    n_baseline_blocks = len(design.blocks) - len(design.kernels)
    blocks = list(design.blocks[:n_baseline_blocks])
    rhos = list(design.log_penalties[:n_baseline_blocks])
    kernel_columns = {}
    for j, kernel in enumerate(keep_kernels):
        start = n0 + j * n_basis
        kernel_columns[kernel.name] = slice(start, start + n_basis)
        blocks.append(
            PenaltyBlock(
                np.arange(start, start + n_basis),
                difference_penalty(n_basis, KERNEL_PENALTY_ORDER),
            )
        )
        rhos.append(
            design.log_penalties[n_baseline_blocks + design.kernels.index(kernel)]
        )
    return UnitDesign(
        X=np.ascontiguousarray(design.X[:, columns]),
        coef_names=tuple(np.asarray(design.coef_names, dtype=object)[columns]),
        blocks=tuple(blocks),
        log_penalties=np.asarray(rhos, dtype=float),
        kernels=tuple(keep_kernels),
        kernel_columns=kernel_columns,
        penalty=design.penalty,
        rows=design.rows,
        onsets={k.name: design.onsets[k.name] for k in keep_kernels},
        spline_basis=design.spline_basis,
        lags=design.lags,
        lag_time=design.lag_time,
        baseline=design.baseline,
        baseline_level_rows=design.baseline_level_rows,
        tested=design.tested,
    )
