"""Event-mark modulation of the mPFC peri-replay temporal kernels.

The ordinary PFC design gives every event in a class the same temporal basis.
This module adds *marked* copies of that basis: event ``e`` contributes
``z_e * B(t - onset_e)`` where ``z_e`` is a centered, unit-SD event property.
The corresponding coefficient curve is therefore the change in the peri-event
log-rate kernel per within-session, within-class SD of the mark.

The implementation is deliberately additive.  It consumes the same canonical
event frames, fit mask, spline basis and batched Poisson fitter as the shipped
PFC analysis, but it does not alter that analysis's model ladder or artifacts.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.pfc.batched_glm import (
    DEFAULT_NULL_DESIGN_BATCH_SIZE,
    BatchedHeldoutResults,
    PreparedHeldoutResponse,
    fit_poisson_glms,
    make_ridge_penalty,
    prepare_heldout_response,
    score_poisson_designs_heldout,
)
from continuum_swr_replay.analysis.pfc.glm import convolve_event_basis
from continuum_swr_replay.analysis.pfc.heldout import empirical_pvalue
from continuum_swr_replay.data_loaders.utils import parse_nwb_filename
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.session_keys import SESSION_COLUMNS
from continuum_swr_replay.table_io import read_table, resolved_table_path, sha256_file

MARKED_EVENT_CLASSES = ("non_local_no_SWR", "non_local_SWR")
DEFAULT_EVENT_ARM_TABLE_PATH = (
    PROCESSED_DATA_DIR / "replay_content" / "event_arm_table.parquet"
)

CONTENT_MARK_COLUMNS = {
    "arm_fraction": "current_arm_fraction",
    "arm_fraction_posterior": "current_arm_fraction_posterior",
}
EVENT_DURATION_MARK = "log_event_duration_s"

# A within-class content slope estimated from a handful of events is a
# descriptive curve through almost no data, and its label-randomization
# reference has too few distinct permutations to be readable.  Classes below
# this many marked events are refused rather than silently reported.
MIN_MARKED_EVENTS_PER_CLASS = 10


def _block_name(event_class: str, mark: str | None = None) -> str:
    """Canonical design-block name for one class's ordinary or marked kernel.

    Parameters
    ----------
    event_class : str
        Event class the block's onsets come from.
    mark : str, optional
        Mark weighting the block. ``None`` names the ordinary event kernel.

    Returns
    -------
    str
    """
    return f"{event_class}:event" if mark is None else f"{event_class}:{mark}"


class MarkScale(NamedTuple):
    """The within-class reference defining one mark's one-SD unit.

    Attributes
    ----------
    mean : float
        Mean of the mark over its finite events, in the mark's raw units.
    std : float
        Population standard deviation (``ddof=0``) of the mark over its finite
        events, in the mark's raw units. One unit of the standardized
        regressor is this many raw units. Zero is recorded only when a caller
        explicitly permits a constant nuisance mark; its standardized block
        is then identically zero.
    n_events : int
        Finite events the mean and SD were computed from.
    n_missing : int
        Events whose mark is undefined; they are standardized to zero and
        flagged unavailable.
    """

    mean: float
    std: float
    n_events: int
    n_missing: int


class MarkBlockGeometry(NamedTuple):
    """Sparse event/lag destinations for batched permuted mark blocks.

    The three index arrays are the flattened non-zero ``(event, lag)`` pairs of
    one class whose target bin falls inside both the session axis and the fit
    mask, so they all share one length ``n_placements``.

    Attributes
    ----------
    event_indices : NDArray of int, shape (n_placements,)
        Row of the class's event frame each placement comes from.
    lag_indices : NDArray of int, shape (n_placements,)
        Row of ``spline_basis`` each placement contributes.
    row_indices : NDArray of int, shape (n_placements,)
        Fitted-row destination of each placement.
    n_fit_rows : int
        Rows of the fitted design, i.e. ``fit_mask.sum()``.
    event_class : str
        Event class this geometry was resolved for.
    n_events : int
        Events in that class when the geometry was resolved.
    """

    event_indices: NDArray
    lag_indices: NDArray
    row_indices: NDArray
    n_fit_rows: int
    event_class: str
    n_events: int


@dataclass(frozen=True)
class EventKernelBlock:
    """One ordinary or marked temporal-basis block in a design."""

    name: str
    event_class: str
    mark_column: str | None = None


@dataclass(frozen=True)
class KernelDesign:
    """A fit-bin design and the exact columns occupied by each basis block.

    Column 0 is the unpenalized intercept and is owned by no block. Every
    remaining column belongs to exactly one block: the blocks tile columns
    ``1 .. n_features`` contiguously in ``blocks`` order.

    Attributes
    ----------
    matrix : NDArray, shape (n_fit_bins, n_features)
        Intercept column followed by the blocks' basis columns.
    block_slices : dict[str, slice]
        Column span of each block, keyed and ordered by block name.
    blocks : tuple[EventKernelBlock, ...]
        The blocks, in column order.
    """

    matrix: NDArray
    block_slices: dict[str, slice]
    blocks: tuple[EventKernelBlock, ...]

    def __post_init__(self) -> None:
        """Refuse a design whose block spans do not tile its non-intercept columns."""
        names = [block.name for block in self.blocks]
        if list(self.block_slices) != names:
            raise ValueError(
                "block_slices must name the blocks in column order; got "
                f"{list(self.block_slices)} for blocks {names}"
            )
        next_column = 1
        for name in names:
            span = self.block_slices[name]
            if span.step not in (None, 1):
                raise ValueError(f"basis block {name!r} must span contiguous columns")
            if span.start != next_column or span.stop <= span.start:
                raise ValueError(
                    f"basis block {name!r} spans {span.start}:{span.stop}; expected a "
                    f"non-empty span starting at column {next_column}"
                )
            next_column = span.stop
        if next_column != self.matrix.shape[1]:
            raise ValueError(
                f"basis blocks end at column {next_column} but the design has "
                f"{self.matrix.shape[1]} columns (column 0 is the intercept)"
            )


@dataclass(frozen=True)
class MarkedModelLadder:
    """One joint full design and the leave-one-scientific-mark-out designs.

    Attributes
    ----------
    reduced : dict[str, KernelDesign]
        Per event class, the full design minus that class's scientific-mark
        block.
    full : KernelDesign
        The joint design holding every class's ordinary, nuisance and
        scientific block.
    events : dict[str, pd.DataFrame]
        Per event class, the standardized event frame the design was built
        from, carrying ``<mark>_z`` and ``<mark>_available`` columns.
    scales : dict[str, dict[str, MarkScale]]
        Per event class and mark, the centering and scaling reference.
    scientific_mark : str
        Raw name of the mark whose per-class block is dropped in ``reduced``.
    nuisance_marks : tuple[str, ...]
        Raw names of the marks kept in every design.
    classes : tuple[str, ...]
        Event classes, in design-block and ``reduced`` order.
    fit_mask : pd.Series of bool
        Fitted-bin mask indexed by session time, shape ``(n_time,)``.
    spline_basis : NDArray, shape (n_lags, n_basis)
        Basis functions evaluated at ``lags``.
    lags : NDArray of int, shape (n_lags,)
        Lag indices in samples, ascending by one sample.
    """

    reduced: dict[str, KernelDesign]
    full: KernelDesign
    events: dict[str, pd.DataFrame]
    scales: dict[str, dict[str, MarkScale]]
    scientific_mark: str
    nuisance_marks: tuple[str, ...]
    classes: tuple[str, ...]
    fit_mask: pd.Series
    spline_basis: NDArray
    lags: NDArray

    @property
    def scientific_mark_z(self) -> str:
        """Column holding the standardized scientific mark."""
        return f"{self.scientific_mark}_z"


@dataclass(frozen=True)
class MarkedKernelResults:
    """Held-out comparisons, full-data kernels, and shuffle references.

    Attributes
    ----------
    statistics : pd.DataFrame
        One row per unit, with the shared ``marked_full_heldout_deviance``
        column and per-class reduced-model, delta and convergence columns.
    typical_curves : dict[str, NDArray]
        ``typical_curves[c]``, shape ``(n_lags, n_units)``: class ``c``'s
        ordinary log-rate kernel at typical mark values. Non-converged units
        are ``nan``.
    mark_curves : dict[str, NDArray]
        ``mark_curves[c]``, shape ``(n_lags, n_units)``: the change in class
        ``c``'s log-rate kernel per within-session, within-class SD of the
        scientific mark. Non-converged units are ``nan``.
    randomization_reference : dict[str, NDArray]
        ``randomization_reference[c]``, shape ``(n_shuffles, n_units)``:
        held-out improvements from mark-permuted refits of class ``c``.
    diagnostics : dict[str, Any]
        JSON-serializable audit record: mark scales, design widths and ranks,
        shuffle bookkeeping, timings and NaN-channel counts.
    """

    statistics: pd.DataFrame
    typical_curves: dict[str, NDArray]
    mark_curves: dict[str, NDArray]
    randomization_reference: dict[str, NDArray]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class MarkedSessionContext:
    """Canonical inputs needed to fit one session without fitting legacy models.

    Attributes
    ----------
    fit_mask : pd.Series of bool
        Fitted-bin mask indexed by session time, shape ``(n_time,)``.
    counts : NDArray, shape (n_fit_bins, n_units)
        Spike counts on the fitted bins only.
    events : dict[str, pd.DataFrame]
        Per event class, the cohort-restricted events with their attached
        content and duration marks.
    spline_basis : NDArray, shape (n_lags, n_basis)
        Basis functions evaluated at ``lags``.
    lag_time : NDArray, shape (n_lags,)
        Lag of each basis row, in seconds.
    lags : NDArray of int, shape (n_lags,)
        Lag indices in samples, ascending by one sample.
    cohort_info : dict[str, object]
        Bin, event and decode-alignment counts for the session.
    input_provenance : dict[str, object]
        Paths and SHA-256 digests of every canonical table read.
    """

    fit_mask: pd.Series
    counts: NDArray
    events: dict[str, pd.DataFrame]
    spline_basis: NDArray
    lag_time: NDArray
    lags: NDArray
    cohort_info: dict[str, object]
    input_provenance: dict[str, object]


def prepare_marked_session_context(
    nwb_file_name: str,
    epoch_name: str,
    *,
    brain_area: str = "mPFC",
    sampling_frequency: float = 500.0,
    cohort_stops: pd.DataFrame,
    event_arm_table_path: str | Path | None = None,
) -> MarkedSessionContext:
    """Load and align one at-well session without running the legacy ladder.

    This repeats no model fit: the decode and position axes are intersected
    once, spikes are binned once, and only the two replay-class designs needed
    by the marked analysis are built later.  The same canonical loaders and
    cohort helpers used by :func:`pfc.glm.run_session` define every row.

    Parameters
    ----------
    nwb_file_name : str
        Session NWB file, e.g. ``"chimi20200212_.nwb"``.
    epoch_name : str
        Epoch within that file, e.g. ``"02_r1"``.
    brain_area : str
        Sorted-unit area whose spikes are binned.
    sampling_frequency : float
        Sampling rate of the PFC time axis, in Hz.
    cohort_stops : pd.DataFrame
        The canonical stop table restricted to this session, defining the
        at-well fit cohort.
    event_arm_table_path : str or Path, optional
        Event-arm table holding the content marks. Defaults to
        :data:`DEFAULT_EVENT_ARM_TABLE_PATH`.

    Returns
    -------
    MarkedSessionContext
    """
    # These imports open Spyglass/DataJoint resources at import time, so keep
    # them off the pure marked-design import path used by unit tests.
    from continuum_swr_replay.analysis.hippocampus import load_results
    from continuum_swr_replay.analysis.hippocampus.data_aggregation import (
        align_results_to_position,
    )
    from continuum_swr_replay.analysis.pfc.canonical_events import (
        DEFAULT_EVENT_TABLE_PATH,
        canonical_event_provenance,
        load_session_canonical_events,
        load_session_swr_intervals,
        session_data_over_time_path,
    )
    from continuum_swr_replay.analysis.pfc.glm import (
        build_session_event_sets,
        build_session_fit_mask,
        event_basis_functions,
        get_spikecount_per_time_bin,
    )
    from continuum_swr_replay.data_loaders import load_data

    logger = logging.getLogger(
        f"{nwb_file_name.replace('_.nwb', '')}_{epoch_name}_{brain_area}_marked"
    )
    results, _, _ = load_results(
        nwb_file_name,
        epoch_name,
        decoding_brain_area="HPC",
        load_detector=False,
    )
    data = load_data(nwb_file_name, epoch_name)
    results, position_info, alignment = align_results_to_position(
        results, data["position_info"]
    )
    data = {**data, "position_info": position_info}
    time = position_info.index.to_numpy(dtype=float)

    _, fit_mask, bin_counts = build_session_fit_mask(
        position_info,
        cohort_stops=cohort_stops,
        stop_cohort="at_well",
    )
    fit_lags = np.arange(-125, 126, dtype=int)
    canonical_events, event_table_sha256 = load_session_canonical_events(
        nwb_file_name, epoch_name
    )
    swr_intervals, data_over_time_sha256 = load_session_swr_intervals(
        nwb_file_name, epoch_name
    )
    event_times, event_counts = build_session_event_sets(
        data,
        results,
        canonical_events=canonical_events,
        swr_intervals=swr_intervals,
        cohort_stops=cohort_stops,
        stop_cohort="at_well",
        fit_mask=fit_mask,
        basis_half_window_s=float(np.abs(fit_lags).max()) / sampling_frequency,
    )
    marked_events, event_arm_sha256 = load_session_current_arm_marks(
        nwb_file_name,
        epoch_name,
        {event_class: event_times[event_class] for event_class in MARKED_EVENT_CLASSES},
        event_arm_table_path=event_arm_table_path,
    )
    spline_basis, lag_time, _ = event_basis_functions(fit_lags, sampling_frequency)

    spike_times = data.get("spike_times", {}).get(brain_area, [])
    if spike_times:
        counts = np.column_stack(
            [get_spikecount_per_time_bin(neuron, time) for neuron in spike_times]
        ).astype(float)[np.asarray(fit_mask, dtype=bool)]
    else:
        counts = np.empty((int(fit_mask.sum()), 0), dtype=float)
        logger.warning("No %s neurons found for this session", brain_area)

    event_provenance = canonical_event_provenance(
        event_table_path=DEFAULT_EVENT_TABLE_PATH,
        event_table_sha256=event_table_sha256,
        data_over_time_path=session_data_over_time_path(nwb_file_name, epoch_name),
        data_over_time_sha256=data_over_time_sha256,
    )
    return MarkedSessionContext(
        fit_mask=fit_mask,
        counts=counts,
        events=marked_events,
        spline_basis=spline_basis,
        lag_time=lag_time,
        lags=fit_lags,
        cohort_info={**bin_counts, **event_counts, "alignment": alignment},
        input_provenance={
            **event_provenance,
            "event_arm_table_path": str(
                DEFAULT_EVENT_ARM_TABLE_PATH
                if event_arm_table_path is None
                else Path(event_arm_table_path)
            ),
            "event_arm_table_sha256": event_arm_sha256,
        },
    )


def attach_current_arm_marks(
    events: dict[str, pd.DataFrame],
    event_arm_table: pd.DataFrame,
    *,
    classes: tuple[str, ...] = MARKED_EVENT_CLASSES,
) -> dict[str, pd.DataFrame]:
    """Attach current-arm content when defined and duration to every event.

    The event classes are the already cohort-restricted canonical PFC frames.
    Joining by ``event_id`` rather than by time makes the content measurement
    refer to exactly those events. The function deliberately refuses partial
    joins, duplicate current-arm rows, class disagreements, and timing drift.
    An event with no occupancy arm keeps a missing content value; it remains
    eligible for the ordinary event kernel, and
    :func:`standardize_event_marks` is the single place that turns that
    missingness into a ``<mark>_available`` flag.

    Parameters
    ----------
    events : dict[str, pd.DataFrame]
        Per event class, the canonical events with ``event_id``,
        ``start_time`` and ``end_time``.
    event_arm_table : pd.DataFrame
        One session's event-arm rows, one per event and arm.
    classes : tuple[str, ...]
        Event classes to mark; must be unique and present in ``events``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Copies of the input frames, in their original row order, with the
        current-arm content columns of :data:`CONTENT_MARK_COLUMNS` and
        :data:`EVENT_DURATION_MARK` appended.
    """
    required_arm_columns = {
        "event_id",
        "is_current_arm",
        "is_swr",
        "event_start_s",
        "event_end_s",
        "event_duration_s",
        "occupancy_state",
        "at_well",
        "occupancy_well_arm",
        *CONTENT_MARK_COLUMNS,
    }
    missing_columns = sorted(required_arm_columns - set(event_arm_table.columns))
    if missing_columns:
        raise KeyError(
            f"event-arm table is missing required column(s) {missing_columns}"
        )
    if len(classes) == 0 or len(classes) != len(set(classes)):
        raise ValueError("classes must be a non-empty sequence of unique names")

    requested_ids: list[object] = []
    for event_class in classes:
        if event_class not in events:
            raise KeyError(f"event class {event_class!r} is missing")
        frame = events[event_class]
        required_event_columns = {"event_id", "start_time", "end_time"}
        missing = sorted(required_event_columns - set(frame.columns))
        if missing:
            raise KeyError(
                f"event class {event_class!r} is missing column(s) {missing}"
            )
        requested_ids.extend(frame["event_id"].tolist())
    requested_index = pd.Index(requested_ids)
    if requested_index.has_duplicates:
        duplicate = requested_index[requested_index.duplicated()][0]
        raise ValueError(
            f"event_id {duplicate!r} appears in more than one marked event row"
        )

    current_mask = (
        event_arm_table["is_current_arm"]
        .astype("boolean")
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    current = event_arm_table.loc[current_mask].copy()
    current = current.loc[current["event_id"].isin(requested_index)]
    if current["event_id"].duplicated().any():
        duplicate = current.loc[current["event_id"].duplicated(), "event_id"].iloc[0]
        raise ValueError(
            f"event_id {duplicate!r} has more than one current-arm content row"
        )
    metadata_columns = [
        "event_id",
        "is_swr",
        "event_start_s",
        "event_end_s",
        "event_duration_s",
        "occupancy_state",
        "at_well",
        "occupancy_well_arm",
    ]
    metadata = (
        event_arm_table.loc[
            event_arm_table["event_id"].isin(requested_index), metadata_columns
        ]
        .drop_duplicates()
        .copy()
    )
    if metadata["event_id"].duplicated().any():
        duplicate = metadata.loc[metadata["event_id"].duplicated(), "event_id"].iloc[0]
        raise ValueError(
            f"event_id {duplicate!r} has inconsistent event metadata across arm rows"
        )
    missing_ids = requested_index.difference(pd.Index(metadata["event_id"]))
    if len(missing_ids):
        preview = missing_ids[:5].tolist()
        raise ValueError(
            f"{len(missing_ids)} marked event(s) have no event-arm rows; first "
            f"missing event_id(s): {preview}"
        )

    current = current.loc[
        :,
        ["event_id", *CONTENT_MARK_COLUMNS],
    ].rename(columns=CONTENT_MARK_COLUMNS)
    metadata = metadata.rename(
        columns={
            "is_swr": "arm_is_swr",
            "occupancy_state": "arm_occupancy_state",
            "at_well": "arm_at_well",
            "occupancy_well_arm": "arm_occupancy_well_arm",
        }
    ).merge(
        current,
        how="left",
        on="event_id",
        sort=False,
        validate="one_to_one",
    )
    marked: dict[str, pd.DataFrame] = {}
    for event_class in classes:
        frame = events[event_class].copy()
        original_ids = frame["event_id"].to_numpy(copy=True)
        frame = frame.merge(
            metadata,
            how="left",
            on="event_id",
            sort=False,
            validate="one_to_one",
        )
        if not np.array_equal(frame["event_id"].to_numpy(), original_ids):
            raise ValueError(f"current-arm join changed {event_class} event order")
        expected_swr = event_class == "non_local_SWR"
        arm_is_swr = frame["arm_is_swr"].astype("boolean")
        if arm_is_swr.isna().any():
            event_id = frame.loc[arm_is_swr.isna().to_numpy(), "event_id"].iloc[0]
            raise ValueError(
                f"event_id {event_id!r} has a missing event-arm is_swr label"
            )
        if not np.all(arm_is_swr.to_numpy(dtype=bool) == expected_swr):
            raise ValueError(
                f"event-arm is_swr labels disagree with class {event_class!r}"
            )
        for canonical, content in (
            ("start_time", "event_start_s"),
            ("end_time", "event_end_s"),
        ):
            if not np.allclose(
                pd.to_numeric(frame[canonical], errors="coerce"),
                pd.to_numeric(frame[content], errors="coerce"),
                atol=1e-9,
                rtol=0.0,
                equal_nan=False,
            ):
                raise ValueError(
                    f"event-arm timing column {content!r} disagrees with "
                    f"canonical {canonical!r} for class {event_class!r}"
                )
        # The canonical replay table's duration is inclusive of one sampled
        # decode bin. Recomputing ``end - start`` here would shorten every
        # event, with the largest relative error on the shortest events.
        duration = pd.to_numeric(frame["event_duration_s"], errors="coerce").to_numpy(
            float
        )
        if not np.isfinite(duration).all() or np.any(duration <= 0):
            raise ValueError(
                f"class {event_class!r} has non-finite or non-positive durations"
            )
        frame[EVENT_DURATION_MARK] = np.log(duration)
        finite_marks = []
        for output_column in CONTENT_MARK_COLUMNS.values():
            values = pd.to_numeric(frame[output_column], errors="coerce").to_numpy(
                float
            )
            if np.isinf(values).any():
                raise ValueError(
                    f"current-arm mark {output_column!r} is infinite in class "
                    f"{event_class!r}"
                )
            finite_marks.append(np.isfinite(values))
        availability = np.stack(finite_marks)
        if np.any(availability.any(axis=0) & ~availability.all(axis=0)):
            raise ValueError(
                f"current-arm marks are only partially available in class "
                f"{event_class!r}"
            )
        available = availability.all(axis=0)
        wrongly_missing = (~available) & (
            frame["arm_at_well"].fillna(False).astype(bool).to_numpy()
            | frame["arm_occupancy_well_arm"].notna().to_numpy()
        )
        if wrongly_missing.any():
            event_id = frame.loc[wrongly_missing, "event_id"].iloc[0]
            raise ValueError(
                f"event_id {event_id!r} has a defined occupancy well arm but no "
                "current-arm content row"
            )
        marked[event_class] = frame.drop(
            columns=[
                "arm_is_swr",
                "event_start_s",
                "event_end_s",
                "event_duration_s",
                "arm_occupancy_state",
                "arm_at_well",
                "arm_occupancy_well_arm",
            ]
        )
    return marked


def load_session_current_arm_marks(
    nwb_file_name: str,
    epoch_name: str,
    events: dict[str, pd.DataFrame],
    *,
    event_arm_table_path: str | Path | None = None,
) -> tuple[dict[str, pd.DataFrame], str]:
    """Load and attach one session's canonical current-arm content marks.

    Parameters
    ----------
    nwb_file_name : str
        Session NWB file, e.g. ``"chimi20200212_.nwb"``.
    epoch_name : str
        Epoch within that file, e.g. ``"02_r1"``.
    events : dict[str, pd.DataFrame]
        Per event class, the canonical events to mark.
    event_arm_table_path : str or Path, optional
        Event-arm table to read. Defaults to
        :data:`DEFAULT_EVENT_ARM_TABLE_PATH`.

    Returns
    -------
    marked : dict[str, pd.DataFrame]
        ``events`` with the content and duration marks attached.
    event_arm_table_sha256 : str
        Digest of the table the marks were read from.
    """
    path = (
        DEFAULT_EVENT_ARM_TABLE_PATH
        if event_arm_table_path is None
        else Path(event_arm_table_path)
    )
    table = read_table(path)
    missing = [column for column in SESSION_COLUMNS if column not in table]
    if missing:
        raise KeyError(f"event-arm table at {path} is missing column(s) {missing}")
    animal, session_date = parse_nwb_filename(nwb_file_name)
    is_session = (
        table["animal"].eq(animal)
        & table["session_date"].eq(session_date)
        & table["epoch"].eq(epoch_name)
    )
    session = table.loc[is_session]
    if session.empty:
        raise ValueError(
            f"no event-arm rows for ({animal}, {session_date}, {epoch_name}) in {path}"
        )
    return (
        attach_current_arm_marks(events, session),
        sha256_file(resolved_table_path(path)),
    )


def weighted_event_onset_counts(
    events: pd.DataFrame,
    time: NDArray,
    weights: NDArray | None = None,
) -> NDArray:
    """Bin event-onset weights on the PFC time axis.

    This is the weighted analogue of :func:`pfc.glm.event_onset_counts` and
    follows its half-open-bin convention.  Multiple events in one bin add.
    Events outside the sampled frame contribute nothing, matching the ordinary
    onset builder; the canonical session preparation audits those events before
    this function is reached.

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, ...)
        Events carrying ``start_time`` in seconds.
    time : NDArray, shape (n_time,)
        Strictly increasing session time axis, in seconds.
    weights : NDArray, shape (n_events,), optional
        Per-event weight. ``None`` weights every event by one, reproducing
        :func:`pfc.glm.event_onset_counts`.

    Returns
    -------
    NDArray, shape (n_time,)
    """
    axis = np.asarray(time, dtype=float)
    if axis.ndim != 1 or axis.size == 0 or not np.isfinite(axis).all():
        raise ValueError("time must be a non-empty finite one-dimensional axis")
    if axis.size > 1 and not np.all(np.diff(axis) > 0):
        raise ValueError("time must be strictly increasing")
    if "start_time" not in events:
        raise KeyError("events must carry start_time")

    onsets = pd.to_numeric(events["start_time"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(onsets).all():
        raise ValueError("event start_time must be finite")
    if weights is None:
        event_weights = np.ones(len(events), dtype=float)
    else:
        event_weights = np.asarray(weights, dtype=float)
        if event_weights.shape != (len(events),):
            raise ValueError(
                "weights must have one value per event; got "
                f"{event_weights.shape}, expected {(len(events),)}"
            )
        if not np.isfinite(event_weights).all():
            raise ValueError("event weights must be finite")

    inside = (onsets >= axis[0]) & (onsets <= axis[-1])
    indices = np.digitize(onsets[inside], axis[1:])
    return np.bincount(
        indices,
        weights=event_weights[inside],
        minlength=axis.size,
    ).astype(float, copy=False)


def standardize_event_marks(
    events: pd.DataFrame,
    marks: tuple[str, ...],
    *,
    allow_missing: tuple[str, ...] = (),
    allow_constant: tuple[str, ...] = (),
    min_marked_events: int = MIN_MARKED_EVENTS_PER_CLASS,
) -> tuple[pd.DataFrame, dict[str, MarkScale]]:
    """Append within-frame, population-SD z scores for event marks.

    The mean and SD define the scientific unit of the marked kernel.  They are
    computed separately for each session and event class by the ladder builder.
    Infinite marks always fail. A mark with no within-class variation fails
    too, unless it is named in ``allow_constant``, in which case its SD is
    recorded as zero and its standardized block is identically zero -- the
    mark is then carried without contributing anything to the fit. Missing
    values also fail by default. A mark named in ``allow_missing`` is
    standardized over its finite events, while unavailable events receive zero
    *only in the marked regressor* and carry an explicit ``<mark>_available``
    flag. They therefore remain in the ordinary event kernel but contribute
    nothing to a content slope whose predictor is genuinely undefined (for
    example, a mixed stop with no current arm).

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, ...)
        One event class's events. Standardization is over these rows only.
    marks : tuple[str, ...]
        Raw mark columns to standardize.
    allow_missing : tuple[str, ...]
        Marks permitted to be undefined for some events.
    allow_constant : tuple[str, ...]
        Nuisance marks permitted to have zero within-class variation. Their
        standardized values are all zero. Scientific target marks should not
        be included here.
    min_marked_events : int
        Fewest finite events a mark may be standardized from, at least two.

    Returns
    -------
    standardized : pd.DataFrame, shape (n_events, ...)
        A copy of ``events`` with a ``<mark>_z`` and a boolean
        ``<mark>_available`` column appended per mark.
    references : dict[str, MarkScale]
        The centering and scaling reference behind each ``<mark>_z``.
    """
    if min_marked_events < 2:
        raise ValueError(
            "min_marked_events must be at least two; fewer events cannot "
            f"define a within-class slope (got {min_marked_events})"
        )
    out = events.copy()
    if out.empty:
        raise ValueError("an event class must contain at least one event")
    if "event_id" in out and out["event_id"].duplicated().any():
        duplicate = out.loc[out["event_id"].duplicated(), "event_id"].iloc[0]
        raise ValueError(f"event_id {duplicate!r} appears more than once")

    references: dict[str, MarkScale] = {}
    for mark in marks:
        if mark not in out:
            raise KeyError(f"event mark {mark!r} is missing")
        values = pd.to_numeric(out[mark], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(values)
        invalid_nonmissing = out[mark].notna().to_numpy() & ~finite
        if invalid_nonmissing.any() or (mark not in allow_missing and not finite.all()):
            raise ValueError(f"event mark {mark!r} must be finite for every event")
        if finite.sum() < min_marked_events:
            raise ValueError(
                f"event mark {mark!r} has only {int(finite.sum())} finite events; "
                f"at least {min_marked_events} are required to define a "
                "within-class slope"
            )
        mean = float(values[finite].mean())
        std = float(values[finite].std(ddof=0))
        if not np.isfinite(std) or (std <= 0 and mark not in allow_constant):
            raise ValueError(
                f"event mark {mark!r} has no within-class variation (std={std})"
            )
        standardized = np.zeros(len(values), dtype=float)
        if std > 0:
            standardized[finite] = (values[finite] - mean) / std
        out[f"{mark}_z"] = standardized
        out[f"{mark}_available"] = finite
        references[mark] = MarkScale(
            mean=mean,
            std=std,
            n_events=int(finite.sum()),
            n_missing=int((~finite).sum()),
        )
    return out, references


def event_basis_block(
    events: pd.DataFrame,
    time: NDArray,
    spline_basis: NDArray,
    lags: NDArray,
    *,
    mark_column: str | None = None,
) -> NDArray:
    """Return one ordinary or mark-weighted convolutional basis block.

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, ...)
        One event class's events, carrying ``start_time`` in seconds.
    time : NDArray, shape (n_time,)
        Strictly increasing session time axis, in seconds.
    spline_basis : NDArray, shape (n_lags, n_basis)
        Basis functions evaluated at ``lags``.
    lags : NDArray of int, shape (n_lags,)
        Lag indices in samples, ascending by one sample.
    mark_column : str, optional
        Column weighting each event's basis copy. ``None`` gives the ordinary
        unit-weighted kernel.

    Returns
    -------
    NDArray, shape (n_time, n_basis)
    """
    weights = None
    if mark_column is not None:
        if mark_column not in events:
            raise KeyError(f"event mark column {mark_column!r} is missing")
        weights = pd.to_numeric(events[mark_column], errors="coerce").to_numpy(
            dtype=float
        )
    onsets = weighted_event_onset_counts(events, time, weights)
    return convolve_event_basis(onsets, spline_basis, lags)


def build_kernel_design(
    events: dict[str, pd.DataFrame],
    fit_mask: pd.Series,
    spline_basis: NDArray,
    lags: NDArray,
    blocks: tuple[EventKernelBlock, ...],
) -> KernelDesign:
    """Build an intercept plus an ordered collection of event-basis blocks.

    Parameters
    ----------
    events : dict[str, pd.DataFrame]
        Per event class, the events each block convolves.
    fit_mask : pd.Series of bool, shape (n_time,)
        Fitted-bin mask indexed by session time; its index is the time axis.
    spline_basis : NDArray, shape (n_lags, n_basis)
        Basis functions evaluated at ``lags``.
    lags : NDArray of int, shape (n_lags,)
        Lag indices in samples, ascending by one sample.
    blocks : tuple[EventKernelBlock, ...]
        Blocks in column order; their names must be unique.

    Returns
    -------
    KernelDesign
        Design of shape ``(fit_mask.sum(), 1 + len(blocks) * n_basis)``.
    """
    if not isinstance(fit_mask, pd.Series):
        raise TypeError("fit_mask must be a pandas Series carrying the time index")
    rows = np.asarray(fit_mask, dtype=bool)
    time = fit_mask.index.to_numpy(dtype=float)
    if rows.shape != time.shape or not rows.any():
        raise ValueError("fit_mask must be non-empty and aligned to its time index")
    names = [block.name for block in blocks]
    if len(names) != len(set(names)):
        raise ValueError("event-kernel block names must be unique")

    columns = [np.ones((int(rows.sum()), 1), dtype=float)]
    block_slices: dict[str, slice] = {}
    next_column = 1
    n_basis = np.asarray(spline_basis).shape[1]
    for block in blocks:
        if block.event_class not in events:
            raise KeyError(f"event class {block.event_class!r} is missing")
        convolved = event_basis_block(
            events[block.event_class],
            time,
            spline_basis,
            lags,
            mark_column=block.mark_column,
        )[rows]
        if convolved.shape != (int(rows.sum()), n_basis):
            raise ValueError(
                f"basis block {block.name!r} has shape {convolved.shape}; "
                f"expected {(int(rows.sum()), n_basis)}"
            )
        columns.append(convolved)
        block_slices[block.name] = slice(next_column, next_column + n_basis)
        next_column += n_basis
    return KernelDesign(
        matrix=np.concatenate(columns, axis=1),
        block_slices=block_slices,
        blocks=blocks,
    )


def _without_block(design: KernelDesign, block_name: str) -> KernelDesign:
    """Drop one basis block without rebuilding any convolution."""
    if block_name not in design.block_slices:
        raise KeyError(f"basis block {block_name!r} is missing")
    dropped = design.block_slices[block_name]
    keep = np.ones(design.matrix.shape[1], dtype=bool)
    keep[dropped] = False
    blocks = tuple(block for block in design.blocks if block.name != block_name)
    block_slices: dict[str, slice] = {}
    next_column = 1
    for block in blocks:
        width = (
            design.block_slices[block.name].stop - design.block_slices[block.name].start
        )
        block_slices[block.name] = slice(next_column, next_column + width)
        next_column += width
    return KernelDesign(
        matrix=np.ascontiguousarray(design.matrix[:, keep]),
        block_slices=block_slices,
        blocks=blocks,
    )


def build_marked_model_ladder(
    events: dict[str, pd.DataFrame],
    fit_mask: pd.Series,
    spline_basis: NDArray,
    lags: NDArray,
    *,
    scientific_mark: str,
    nuisance_marks: tuple[str, ...] = (),
    classes: tuple[str, ...] = MARKED_EVENT_CLASSES,
    min_marked_events: int = MIN_MARKED_EVENTS_PER_CLASS,
) -> MarkedModelLadder:
    """Build one joint full model and a reduced model for each event class.

    All marks are standardized independently within session and event class.
    The full design contains the ordinary class kernels and every nuisance and
    scientific marked kernel.  Reduced model ``c`` drops only
    ``scientific_mark × c`` while retaining the other class's scientific
    kernel.  Each held-out improvement is therefore the unique contribution
    of that class's content after adjusting for content in the other class,
    and both descriptive kernels come from one common fit.

    Parameters
    ----------
    events : dict[str, pd.DataFrame]
        Per event class, the events carrying every raw mark column.
    fit_mask : pd.Series of bool, shape (n_time,)
        Fitted-bin mask indexed by session time.
    spline_basis : NDArray, shape (n_lags, n_basis)
        Basis functions evaluated at ``lags``.
    lags : NDArray of int, shape (n_lags,)
        Lag indices in samples, ascending by one sample.
    scientific_mark : str
        Mark whose per-class block the reduced designs drop.
    nuisance_marks : tuple[str, ...]
        Marks kept in every design; must not contain ``scientific_mark``.
    classes : tuple[str, ...]
        Event classes, in design-block order; must be unique.
    min_marked_events : int
        Fewest finite events a class may contribute to a marked kernel.

    Returns
    -------
    MarkedModelLadder

    Raises
    ------
    ValueError
        If a class has fewer than ``min_marked_events`` finite events for one
        of its marks; the message names the class, the mark and the count.
    """
    if scientific_mark in nuisance_marks:
        raise ValueError("the scientific mark cannot also be a nuisance mark")
    if len(classes) == 0 or len(classes) != len(set(classes)):
        raise ValueError("classes must be a non-empty sequence of unique names")
    mark_names = (*nuisance_marks, scientific_mark)
    standardized: dict[str, pd.DataFrame] = {}
    scales: dict[str, dict[str, MarkScale]] = {}
    for event_class in classes:
        if event_class not in events:
            raise KeyError(f"event class {event_class!r} is missing")
        try:
            standardized[event_class], scales[event_class] = standardize_event_marks(
                events[event_class],
                mark_names,
                allow_missing=(scientific_mark,),
                min_marked_events=min_marked_events,
            )
        except ValueError as error:
            raise ValueError(f"event class {event_class!r}: {error}") from error

    base_blocks = tuple(
        EventKernelBlock(name=_block_name(event_class), event_class=event_class)
        for event_class in classes
    )
    nuisance_blocks = tuple(
        EventKernelBlock(
            name=_block_name(event_class, mark),
            event_class=event_class,
            mark_column=f"{mark}_z",
        )
        for event_class in classes
        for mark in nuisance_marks
    )
    scientific_blocks = tuple(
        EventKernelBlock(
            name=_block_name(event_class, scientific_mark),
            event_class=event_class,
            mark_column=f"{scientific_mark}_z",
        )
        for event_class in classes
    )
    full_blocks = (*base_blocks, *nuisance_blocks, *scientific_blocks)
    full = build_kernel_design(standardized, fit_mask, spline_basis, lags, full_blocks)

    reduced: dict[str, KernelDesign] = {}
    for event_class in classes:
        reduced[event_class] = _without_block(
            full, _block_name(event_class, scientific_mark)
        )

    return MarkedModelLadder(
        reduced=reduced,
        full=full,
        events=standardized,
        scales=scales,
        scientific_mark=scientific_mark,
        nuisance_marks=nuisance_marks,
        classes=tuple(classes),
        fit_mask=fit_mask,
        spline_basis=np.asarray(spline_basis, dtype=float),
        lags=np.asarray(lags),
    )


def _score_designs(
    designs: NDArray,
    response: PreparedHeldoutResponse,
    penalty_sigma: float,
    *,
    design_batch_size: int,
) -> BatchedHeldoutResults:
    """Score equal-width designs under the marked model's common prior."""
    width = np.asarray(designs).shape[-1]
    return score_poisson_designs_heldout(
        designs,
        response,
        make_ridge_penalty(width, penalty_sigma),
        design_batch_size=design_batch_size,
    )


def permuted_mark_blocks(
    ladder: MarkedModelLadder,
    event_class: str,
    permutations: NDArray,
    *,
    geometry: MarkBlockGeometry | None = None,
) -> NDArray:
    """Build a batch of permuted mark blocks directly on the fitted rows.

    The scalar reference builds one full-session convolution per draw.  This
    version precomputes the event/lag destinations once and reuses them for
    every draw, so no draw pays for a full-session convolution or for the
    zero-filled non-fit portions of the session: each draw is one scatter of
    permuted event weights onto the fitted rows.  It is algebraically
    identical to the scalar form.

    Parameters
    ----------
    ladder : MarkedModelLadder
        The ladder whose standardized marks and basis are permuted.
    event_class : str
        Class whose scientific-mark block is rebuilt.
    permutations : NDArray of int, shape (n_draws, n_events) or (n_events,)
        One permutation of the class's event indices per draw.
    geometry : MarkBlockGeometry, optional
        Precomputed destinations for this class, from
        :func:`_mark_block_geometry`. Resolved here when omitted; a geometry
        built for another class or a different event count is refused.

    Returns
    -------
    NDArray, shape (n_draws, n_fit_bins, n_basis)
    """
    events = ladder.events[event_class]
    draws = np.asarray(permutations)
    if draws.ndim == 1:
        draws = draws[None, :]
    if draws.ndim != 2 or draws.shape[1] != len(events):
        raise ValueError(
            "permutations must have shape (n_draws, n_events); got "
            f"{draws.shape} for {len(events)} events"
        )
    expected = np.arange(len(events))
    if not all(np.array_equal(np.sort(row), expected) for row in draws):
        raise ValueError("every permutation row must contain each event index once")

    if geometry is None:
        geometry = _mark_block_geometry(ladder, event_class)
    elif geometry.event_class != event_class or geometry.n_events != len(events):
        raise ValueError(
            f"geometry was built for class {geometry.event_class!r} with "
            f"{geometry.n_events} events, not {event_class!r} with {len(events)}"
        )

    weights = events[ladder.scientific_mark_z].to_numpy(dtype=float)[draws]
    values = (
        weights[:, geometry.event_indices, None]
        * ladder.spline_basis[geometry.lag_indices][None, :, :]
    )
    blocks = np.zeros(
        (
            draws.shape[0],
            geometry.n_fit_rows,
            ladder.spline_basis.shape[1],
        ),
        dtype=float,
    )
    for draw_index in range(draws.shape[0]):
        np.add.at(blocks[draw_index], geometry.row_indices, values[draw_index])
    return blocks


def _mark_block_geometry(
    ladder: MarkedModelLadder, event_class: str
) -> MarkBlockGeometry:
    """Resolve sparse event/lag pairs to fitted output rows once per class."""
    events = ladder.events[event_class]
    time = ladder.fit_mask.index.to_numpy(dtype=float)
    onsets = pd.to_numeric(events["start_time"], errors="coerce").to_numpy(float)
    inside = (onsets >= time[0]) & (onsets <= time[-1])
    onset_bins = np.digitize(onsets, time[1:])
    positions = onset_bins[:, None] + ladder.lags[None, :]
    valid_position = (positions >= 0) & (positions < len(time)) & inside[:, None]

    fitted_row = np.full(len(time), -1, dtype=np.int64)
    fitted_row[np.asarray(ladder.fit_mask, dtype=bool)] = np.arange(
        int(ladder.fit_mask.sum())
    )
    clipped = np.clip(positions, 0, len(time) - 1)
    destinations = fitted_row[clipped]
    valid = valid_position & (destinations >= 0)
    event_indices, lag_indices = np.nonzero(valid)
    return MarkBlockGeometry(
        event_indices=event_indices,
        lag_indices=lag_indices,
        row_indices=destinations[valid],
        n_fit_rows=int(ladder.fit_mask.sum()),
        event_class=event_class,
        n_events=len(events),
    )


def within_available_mark_permutations(
    rng: np.random.Generator,
    available: NDArray,
    n_draws: int,
) -> NDArray:
    """Permute marked events while leaving undefined-mark events fixed.

    Parameters
    ----------
    rng : np.random.Generator
        Source of the permutations; drawn one row at a time, so a longer call
        extends the stream of a shorter one.
    available : NDArray of bool, shape (n_events,)
        Whether each event's scientific mark is defined.
    n_draws : int
        Permutations to draw.

    Returns
    -------
    NDArray of int, shape (n_draws, n_events)
        Each row is a permutation of ``0 .. n_events - 1`` that moves only the
        positions where ``available`` is True.
    """
    availability = np.asarray(available, dtype=bool)
    if availability.ndim != 1:
        raise ValueError("available must be one-dimensional")
    if n_draws < 0:
        raise ValueError("n_draws must be non-negative")
    permutations = np.repeat(np.arange(len(availability))[None, :], n_draws, axis=0)
    marked = np.flatnonzero(availability)
    for permutation in permutations:
        permutation[marked] = rng.permutation(marked)
    return permutations


def fit_marked_kernel_ladder(
    ladder: MarkedModelLadder,
    counts: NDArray,
    folds: NDArray,
    *,
    penalty_sigma: float,
    n_shuffles: int = 0,
    shuffle_seed: int = 0,
    min_usable_shuffles: int = 1,
    design_batch_size: int = DEFAULT_NULL_DESIGN_BATCH_SIZE,
) -> MarkedKernelResults:
    """Fit marked kernels and compare each with its own reduced design.

    Each class is scored against its own reduced design -- the joint full
    design minus that class's content block -- so the other class's content
    kernel, both ordinary kernels and every nuisance kernel stay in both
    models.  The full design is fit and scored once and shared.

    Shuffles preserve every onset and the exact mark distribution, permuting
    only the scientific mark within its already-separated event class. They
    provide an unrestricted label-randomization reference, not a calibrated
    conditional p-value: content need not be exchangeable with duration,
    occupied arm, or session time. No result here selects neurons for a later
    population summary.

    Parameters
    ----------
    ladder : MarkedModelLadder
        The joint full design and per-class reduced designs to score.
    counts : NDArray, shape (n_fit_bins, n_units) or (n_fit_bins,)
        Spike counts on the ladder's fitted bins.
    folds : NDArray of int, shape (n_fit_bins,)
        Held-out fold label of each fitted bin.
    penalty_sigma : float
        Ridge prior SD applied to every non-intercept column.
    n_shuffles : int
        Mark-permutation draws per class. Zero skips the reference entirely.
    shuffle_seed : int
        Seed of the per-class permutation streams.
    min_usable_shuffles : int
        Fewest finite draws a unit needs for a tail fraction; below it the
        tail fraction is ``nan``.
    design_batch_size : int
        Designs scored per device batch.

    Returns
    -------
    MarkedKernelResults
    """
    if penalty_sigma <= 0 or not np.isfinite(penalty_sigma):
        raise ValueError("penalty_sigma must be finite and positive")
    if n_shuffles < 0:
        raise ValueError("n_shuffles must be non-negative")
    if min_usable_shuffles < 1:
        raise ValueError("min_usable_shuffles must be positive")
    if design_batch_size < 1:
        raise ValueError("design_batch_size must be positive")
    fit_counts = np.asarray(counts, dtype=float)
    if fit_counts.ndim == 1:
        fit_counts = fit_counts[:, None]
    if fit_counts.ndim != 2 or fit_counts.shape[0] != ladder.full.matrix.shape[0]:
        raise ValueError(
            "counts must have shape (n_fit_bins, n_units); got "
            f"{fit_counts.shape} for {ladder.full.matrix.shape[0]} fit bins"
        )
    if not np.isfinite(fit_counts).all() or np.any(fit_counts < 0):
        raise ValueError("counts must be finite and non-negative")
    fold_array = np.asarray(folds)
    if fold_array.shape != (fit_counts.shape[0],):
        raise ValueError("folds must contain one label per fitted time bin")

    response = prepare_heldout_response(fit_counts, fold_array)
    scoring_started = time.perf_counter()
    classes = ladder.classes
    reduced_batch = np.stack([ladder.reduced[name].matrix for name in classes])
    reduced_score = _score_designs(
        reduced_batch,
        response,
        penalty_sigma,
        design_batch_size=design_batch_size,
    )
    full_score = _score_designs(
        ladder.full.matrix[None, :, :],
        response,
        penalty_sigma,
        design_batch_size=design_batch_size,
    )
    observed_scoring_seconds = time.perf_counter() - scoring_started

    n_units = fit_counts.shape[1]
    statistics: dict[str, object] = {}
    typical_curves: dict[str, NDArray] = {}
    mark_curves: dict[str, NDArray] = {}
    randomization_reference: dict[str, NDArray] = {}
    diagnostics: dict[str, Any] = {
        "scales": {
            event_class: {
                mark: reference._asdict() for mark, reference in class_scales.items()
            }
            for event_class, class_scales in ladder.scales.items()
        },
        "n_shuffles": int(n_shuffles),
        "shuffle_seed": int(shuffle_seed),
        "n_usable_shuffles": {},
        "shuffled_design_rank_by_class": {},
        "n_units_nan_heldout_delta_by_class": {},
        "n_units_nan_tail_fraction_by_class": {},
        "n_marked_events_by_class": {
            event_class: int(class_scales[ladder.scientific_mark].n_events)
            for event_class, class_scales in ladder.scales.items()
        },
        # Free permutation of n marked labels has n! distinct draws, which is
        # the finest tail fraction the reference can ever resolve. The cap
        # keeps the record JSON-safe once n! outruns a 64-bit integer.
        "n_distinct_permutations_by_class": {
            event_class: int(
                min(
                    math.factorial(int(class_scales[ladder.scientific_mark].n_events)),
                    10**12,
                )
            )
            for event_class, class_scales in ladder.scales.items()
        },
        "design_widths": {
            "full": int(ladder.full.matrix.shape[1]),
            **{
                f"reduced_without_{event_class}": int(design.matrix.shape[1])
                for event_class, design in ladder.reduced.items()
            },
        },
        "timing_seconds": {
            "observed_heldout_scoring": observed_scoring_seconds,
            "mark_shuffle_scoring_by_class": {},
        },
        "heldout_design_rank_by_fold": {
            "full": np.asarray(full_score.rank[0]).tolist(),
            **{
                f"reduced_without_{event_class}": np.asarray(
                    reduced_score.rank[class_index]
                ).tolist()
                for class_index, event_class in enumerate(classes)
            },
        },
    }

    full_fit_started = time.perf_counter()
    fitted = fit_poisson_glms(
        ladder.full.matrix,
        fit_counts,
        penalty=make_ridge_penalty(ladder.full.matrix.shape[1], penalty_sigma),
        compute_covariance=False,
    )
    diagnostics["timing_seconds"]["full_data_fit"] = (
        time.perf_counter() - full_fit_started
    )
    diagnostics["full_design_rank"] = int(fitted.rank)
    usable = np.asarray(fitted.converged, dtype=bool)
    diagnostics["n_units_full_fit_not_converged"] = int((~usable).sum())
    statistics["marked_full: converged"] = usable
    statistics["marked_full: n_iter"] = np.asarray(fitted.n_iter, dtype=int)
    statistics["marked_full_heldout_deviance"] = full_score.total[0]

    seed_sequence = np.random.SeedSequence(shuffle_seed)
    class_sequences = seed_sequence.spawn(len(classes))
    for class_index, event_class in enumerate(classes):
        geometry = _mark_block_geometry(ladder, event_class)
        observed = reduced_score.total[class_index] - full_score.total[0]
        statistics[f"{event_class}: marked_reduced_heldout_deviance"] = (
            reduced_score.total[class_index]
        )
        statistics[f"{event_class}: marked_heldout_delta_dev"] = observed
        diagnostics["n_units_nan_heldout_delta_by_class"][event_class] = int(
            (~np.isfinite(observed)).sum()
        )
        statistics[f"{event_class}: marked_full_all_folds_converged"] = np.all(
            full_score.converged[0], axis=0
        )
        statistics[f"{event_class}: marked_reduced_all_folds_converged"] = np.all(
            reduced_score.converged[class_index], axis=0
        )
        statistics[f"{event_class}: marked_heldout_delta_dev_folds"] = list(
            (reduced_score.per_fold[class_index] - full_score.per_fold[0]).T
        )

        base_slice = ladder.full.block_slices[_block_name(event_class)]
        mark_slice = ladder.full.block_slices[
            _block_name(event_class, ladder.scientific_mark)
        ]
        base_params = np.asarray(fitted.params[:, base_slice], dtype=float).copy()
        mark_params = np.asarray(fitted.params[:, mark_slice], dtype=float).copy()
        base_params[~usable] = np.nan
        mark_params[~usable] = np.nan
        typical_curves[event_class] = ladder.spline_basis @ base_params.T
        mark_curves[event_class] = ladder.spline_basis @ mark_params.T

        per_class_reference = np.full((n_shuffles, n_units), np.nan)
        shuffled_ranks = np.full((n_shuffles, response.n_folds), -1, dtype=int)
        rng = np.random.default_rng(class_sequences[class_index])
        available = ladder.events[event_class][
            f"{ladder.scientific_mark}_available"
        ].to_numpy(dtype=bool)
        shuffle_started = time.perf_counter()
        for start in range(0, n_shuffles, design_batch_size):
            stop = min(start + design_batch_size, n_shuffles)
            permutations = within_available_mark_permutations(
                rng, available, stop - start
            )
            shuffled_designs = np.repeat(
                ladder.full.matrix[None, :, :], stop - start, axis=0
            )
            shuffled_designs[:, :, mark_slice] = permuted_mark_blocks(
                ladder,
                event_class,
                permutations,
                geometry=geometry,
            )
            shuffled_score = _score_designs(
                shuffled_designs,
                response,
                penalty_sigma,
                design_batch_size=design_batch_size,
            )
            per_class_reference[start:stop] = (
                reduced_score.total[class_index][None, :] - shuffled_score.total
            )
            shuffled_ranks[start:stop] = np.asarray(shuffled_score.rank, dtype=int)
        diagnostics["timing_seconds"]["mark_shuffle_scoring_by_class"][event_class] = (
            time.perf_counter() - shuffle_started
        )
        randomization_reference[event_class] = per_class_reference
        observed_full_ranks = np.asarray(full_score.rank[0], dtype=int)
        diagnostics["shuffled_design_rank_by_class"][event_class] = {
            "min_by_fold": (shuffled_ranks.min(axis=0).tolist() if n_shuffles else []),
            "max_by_fold": (shuffled_ranks.max(axis=0).tolist() if n_shuffles else []),
            "n_draws_different_from_observed_full": int(
                np.any(shuffled_ranks != observed_full_ranks[None, :], axis=1).sum()
            ),
        }
        n_usable = np.isfinite(per_class_reference).sum(axis=0)
        diagnostics["n_usable_shuffles"][event_class] = n_usable.tolist()
        statistics[f"{event_class}: n_usable_mark_shuffles"] = n_usable
        n_nan_tail = 0
        if n_shuffles:
            tail_fraction = empirical_pvalue(
                observed,
                per_class_reference,
                min_usable_draws=min_usable_shuffles,
            )
            statistics[f"{event_class}: marked_label_shuffle_tail_fraction"] = (
                tail_fraction
            )
            n_nan_tail = int((~np.isfinite(tail_fraction)).sum())
        diagnostics["n_units_nan_tail_fraction_by_class"][event_class] = n_nan_tail

    return MarkedKernelResults(
        statistics=pd.DataFrame(statistics),
        typical_curves=typical_curves,
        mark_curves=mark_curves,
        randomization_reference=randomization_reference,
        diagnostics=diagnostics,
    )
