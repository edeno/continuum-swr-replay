"""Per-event features, read on the decode grid by one rule for every inventory.

An interval's **rows** are the decode grid's samples ``t`` with
``event_start_s <= t <= event_end_s``, the grid's last sample left out. Every
feature of an interval is read over its rows and no others, so two
inventories' identical intervals get identical features, whatever detector
found them. An interval with no row (shorter than a sample, off the grid, or
holding only the grid's last sample) has every row feature missing, and its
evidence is unscorable as ``no_row``.

Row ``k`` holds the spikes with ``t_k <= s < t_{k + 1}``, the rule of
``tetrode_spike_counts`` and of the decode. The grid's last sample has no such
span, so it is never a row: an interval ending on it has one row fewer than
the samples it holds (:func:`_grid_rows`). This is the one row rule of every
interval, event or baseline window, and of every reader of rows: the features
here, ``evidence.interval_evidence``, ``selection.baseline_draws`` and the
runner's choice of the rows whose evidence it computes.

A summary over rows (a mean or a maximum) is NaN when any of its rows is NaN,
so a masked sample inside an interval is never skipped silently.

Posterior features come in two namespaces, each read for every inventory's
intervals: ``ss_`` from the state-space decode frame, over the interval's rows,
and ``ms_`` from the memoryless run's stored window statistics, over the
windows whose midpoint lies in the interval. Spatial features of both use one
matrix of track-graph distances between the real place bins, so no junction
placeholder bin enters either.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from types import MappingProxyType

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.detection.memoryless_sequence.detect import (
    CandidateSequences,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.window_statistics import (
    NO_CENTER,
    dispersion_cm,
)
from continuum_swr_replay.analysis.detection.swr_detectors import (
    tetrode_spike_counts,
)
from continuum_swr_replay.analysis.detector_comparison.recount import (
    FRAGMENT_GAPS_S,
)
from continuum_swr_replay.analysis.detector_comparison.schema import STATE_SPACE
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    events_in_stops,
)
from continuum_swr_replay.decoding.results import nonlocal_probability
from continuum_swr_replay.intervals import contiguous_true_runs
from continuum_swr_replay.parameters import DEFAULT_NON_LOCAL_THRESHOLD
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS_DAY

__all__ = [
    "DURATION_STRATA_S",
    "DURATION_STRATUM_LABELS",
    "STOP_CONTEXT_DTYPES",
    "PosteriorTraces",
    "RowTraces",
    "duration_stratum",
    "interval_features",
    "interval_rows",
    "middle_row",
    "ms_features",
    "neighbour_gaps",
    "ss_features",
    "stop_context",
]

DURATION_STRATA_S = (0.0, 0.050, 0.100, 0.200, 0.500, np.inf)
"""Bounds of the duration strata in seconds; each stratum is closed on the
right, so a 50 ms span is in the first and a span over 500 ms in the last."""
DURATION_STRATUM_LABELS = (
    "0-50 ms",
    "50-100 ms",
    "100-200 ms",
    "200-500 ms",
    ">500 ms",
)

STOP_CONTEXT_DTYPES: Mapping[str, object] = MappingProxyType(
    {
        "stop_id": "UInt64",
        "at_well": "boolean",
        "occupancy_state": object,
        "occupancy_well_id": np.float64,
        "current_patch": np.float64,
        "is_rewarded_outcome": np.float64,
    }
)
"""The stop table's columns an event takes from the stop holding its onset,
with the dtype each has; every one can hold a missing value."""


def interval_rows(
    time: NDArray, starts: NDArray, ends: NDArray
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """The samples of each closed interval.

    Every sample with ``start <= t <= end`` is counted, the last one too. On
    the memoryless windows' midpoints these are an interval's windows; on the
    decode grid the rows are :func:`_grid_rows`', which leave out the grid's
    last sample.

    Parameters
    ----------
    time : NDArray, shape (n_time,)
        The grid's sample times in seconds, increasing.
    starts, ends : NDArray, shape (n_intervals,)
        Interval bounds in seconds.

    Returns
    -------
    first_row : NDArray[np.int64], shape (n_intervals,)
        The first sample at or after the start.
    n_rows : NDArray[np.int64], shape (n_intervals,)
        The number of samples with ``start <= t <= end``; 0 where none is.
    """
    time = np.asarray(time, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    first = np.searchsorted(time, starts, side="left")
    after_last = np.searchsorted(time, ends, side="right")
    n_rows = np.maximum(after_last - first, 0)
    return first.astype(np.int64), n_rows.astype(np.int64)


@dataclass(frozen=True)
class RowTraces:
    """One epoch's per-row arrays on the decode grid.

    Attributes
    ----------
    time : NDArray, shape (n_time,)
        The grid's sample times in seconds, increasing.
    spike_counts : NDArray[np.int64], shape (n_time, n_tetrodes)
        Each tetrode's spikes in each row, by ``tetrode_spike_counts``.
    ripple_z : NDArray, shape (n_time,)
        The frame's ``SWR_zscore``, NaN where masked.
    mua_z : NDArray, shape (n_time,)
        The frame's ``MUA_zscore``.
    linear_position : NDArray, shape (n_time,)
        The animal's linear position in cm.
    track_segment_id, patch_id : NDArray[np.int64], shape (n_time,)
        The animal's track segment and patch.
    """

    time: NDArray[np.float64]
    spike_counts: NDArray[np.int64]
    ripple_z: NDArray[np.float64]
    mua_z: NDArray[np.float64]
    linear_position: NDArray[np.float64]
    track_segment_id: NDArray[np.int64]
    patch_id: NDArray[np.int64]

    def __post_init__(self) -> None:
        _check_lengths(self, self.time.shape[0])

    @classmethod
    def from_frame(
        cls, dot: pd.DataFrame, spike_times: Sequence[NDArray]
    ) -> "RowTraces":
        """The traces of a decode frame and its tetrodes' spikes.

        Parameters
        ----------
        dot : pd.DataFrame
            The decode frame, one row per grid sample, its times the index.
        spike_times : sequence of NDArray, each shape (n_spikes,)
            Spike times in seconds, one array per tetrode.
        """
        time = np.asarray(dot.index, dtype=np.float64)
        return cls(
            time=time,
            spike_counts=tetrode_spike_counts(spike_times, time),
            ripple_z=dot["SWR_zscore"].to_numpy(dtype=np.float64),
            mua_z=dot["MUA_zscore"].to_numpy(dtype=np.float64),
            linear_position=dot["linear_position"].to_numpy(dtype=np.float64),
            track_segment_id=dot["track_segment_id"].to_numpy(dtype=np.int64),
            patch_id=dot["patch_id"].to_numpy(dtype=np.int64),
        )


def _check_lengths(traces: "RowTraces | PosteriorTraces", n_time: int) -> None:
    """Raise unless every array field of ``traces`` has ``n_time`` rows."""
    for field in fields(traces):
        length = getattr(traces, field.name).shape[0]
        if length != n_time:
            raise ValueError(f"{field.name} has {length} rows; the grid has {n_time}.")


@dataclass(frozen=True)
class PosteriorTraces:
    """One epoch's state-space posterior summaries on the decode grid.

    Kept beside :class:`RowTraces`, as the spike evidence is, so the features
    that read no posterior need no decode state.

    Attributes
    ----------
    local_prob, no_spike_prob, continuous_prob, fragmented_prob : NDArray
        Each shape (n_time,): the frame's ``Local_prob``, ``No-Spike_prob``,
        ``Non-Local_Continuous_prob`` and ``Non-Local_Fragmented_prob``, as
        float64.
    nonlocal_prob : NDArray, shape (n_time,)
        ``Non-Local_Continuous_prob + Non-Local_Fragmented_prob``, summed and
        kept in the frame's precision (float32), the trace the decode
        compared with the threshold to find its events.
    hpd_cm : NDArray, shape (n_time,)
        The frame's ``hpd``, the size of the posterior's highest-density
        region in cm.
    """

    local_prob: NDArray[np.float64]
    no_spike_prob: NDArray[np.float64]
    continuous_prob: NDArray[np.float64]
    fragmented_prob: NDArray[np.float64]
    nonlocal_prob: NDArray[np.floating]
    hpd_cm: NDArray[np.float64]

    def __post_init__(self) -> None:
        _check_lengths(self, self.local_prob.shape[0])

    @classmethod
    def from_frame(cls, dot: pd.DataFrame) -> "PosteriorTraces":
        """The posterior summaries of a decode frame, one per grid sample.

        Raises
        ------
        ValueError
            A state column is not float32, the decode's precision: the
            non-local sum is compared with the threshold in it, so a tie at
            the threshold reads as the decode read it only there.
        """
        for column in _STATE_COLUMNS:
            if dot[column].dtype != np.float32:
                raise ValueError(
                    f"{column} is {dot[column].dtype}; the decode frame stores "
                    "the state probabilities as float32."
                )
        continuous = dot["Non-Local_Continuous_prob"]
        fragmented = dot["Non-Local_Fragmented_prob"]
        return cls(
            local_prob=dot["Local_prob"].to_numpy(dtype=np.float64),
            no_spike_prob=dot["No-Spike_prob"].to_numpy(dtype=np.float64),
            continuous_prob=continuous.to_numpy(dtype=np.float64),
            fragmented_prob=fragmented.to_numpy(dtype=np.float64),
            nonlocal_prob=nonlocal_probability(dot).to_numpy(),
            hpd_cm=dot["hpd"].to_numpy(dtype=np.float64),
        )


_STATE_COLUMNS = (
    "Local_prob",
    "No-Spike_prob",
    "Non-Local_Continuous_prob",
    "Non-Local_Fragmented_prob",
)
"""The decode frame's state probabilities, float32."""


def middle_row(first: NDArray, n_rows: NDArray | int) -> NDArray[np.int64]:
    """The middle row of each run of ``n_rows`` rows from ``first``, the
    earlier of two: ``first + (n_rows - 1) // 2``.

    Parameters
    ----------
    first : NDArray of int, shape (n,)
    n_rows : NDArray of int, shape (n,), or int

    Returns
    -------
    middle : NDArray[np.int64], shape (n,)
    """
    middle: NDArray[np.int64] = (
        np.asarray(first, dtype=np.int64)
        + (np.asarray(n_rows, dtype=np.int64) - 1) // 2
    )
    return middle


def _grid_rows(
    time: NDArray, intervals: pd.DataFrame
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """The decode-grid rows of each interval: :func:`interval_rows` with the
    grid's last sample left out (the module's row rule).

    Parameters
    ----------
    time : NDArray, shape (n_time,)
        The decode grid's sample times in seconds, increasing.
    intervals : pd.DataFrame
        ``event_start_s`` and ``event_end_s`` of each interval, in seconds.

    Returns
    -------
    first_row : NDArray[np.int64], shape (n_intervals,)
    n_rows : NDArray[np.int64], shape (n_intervals,)
        0 for an interval holding no sample but the grid's last.
    """
    first, n_rows = interval_rows(
        time,
        intervals["event_start_s"].to_numpy(dtype=np.float64),
        intervals["event_end_s"].to_numpy(dtype=np.float64),
    )
    # the last sample holds no span of spikes, so the rows stop before it
    n_rows = np.clip(np.minimum(first + n_rows, time.size - 1) - first, 0, None)
    return first, n_rows


def grid_row_mask(n_time: int, first: NDArray, n_rows: NDArray) -> NDArray[np.bool_]:
    """The rows of every interval, as one mask over the grid.

    Parameters
    ----------
    n_time : int
        The grid's samples.
    first, n_rows : NDArray of int, shape (n_intervals,)
        Each interval's first row and its number of rows (:func:`_grid_rows`).

    Returns
    -------
    NDArray of bool, shape (n_time,)
        True on every row of any interval.
    """
    edges = np.zeros(n_time + 1, dtype=np.int64)
    np.add.at(edges, first, 1)
    np.add.at(edges, first + n_rows, -1)
    mask: NDArray[np.bool_] = np.cumsum(edges[:-1]) > 0
    return mask


def evidence_rows(time: NDArray, events: pd.DataFrame) -> NDArray[np.bool_]:
    """The rows whose evidence is computed: every row of the compared events.

    Parameters
    ----------
    time : NDArray, shape (n_time,)
        The decode grid.
    events : pd.DataFrame
        ``event_start_s`` and ``event_end_s`` of the compared events.

    Returns
    -------
    rows : NDArray of bool, shape (n_time,)
        Never the grid's last sample, which is no row (``features``' row
        rule).
    """
    first, n_rows = _grid_rows(time, events)
    return grid_row_mask(time.size, first, n_rows)


def _over_rows(
    ufunc: np.ufunc, values: NDArray, first: NDArray, n_rows: NDArray
) -> NDArray:
    """``ufunc`` reduced over the rows of each interval that has any.

    An interval may end at the array's last row.

    Returns
    -------
    reduced : NDArray, shape (n_with_rows, *values.shape[1:])
    """
    has_rows = n_rows > 0
    edges = np.column_stack([first[has_rows], first[has_rows] + n_rows[has_rows]])
    if edges.size == 0:
        return np.empty((0, *values.shape[1:]), dtype=values.dtype)
    # reduceat over (start, end) pairs: each even entry is one interval's rows.
    # One padding row lets an end edge equal the length; it lies past every
    # interval's rows, so only a discarded odd entry reads it.
    padded = np.concatenate([values, np.zeros_like(values[:1])], axis=0)
    reduced: NDArray = ufunc.reduceat(padded, edges.ravel(), axis=0)[::2]
    return reduced


def _float_or_nan(reduced: NDArray, has_rows: NDArray) -> NDArray[np.float64]:
    """``reduced`` (one value per interval with rows) spread over every
    interval, NaN where it has no row."""
    values = np.full(has_rows.shape, np.nan)
    values[has_rows] = reduced
    return values


def _int_or_na(reduced: NDArray, has_rows: NDArray) -> pd.arrays.IntegerArray:
    """``reduced`` (one count per interval with rows) spread over every
    interval as Int64, NA where it has no row."""
    values = np.zeros(has_rows.shape, dtype=np.int64)
    values[has_rows] = reduced
    return pd.arrays.IntegerArray(values, ~has_rows)


def interval_features(
    traces: RowTraces, intervals: pd.DataFrame, *, sample_interval_s: float
) -> pd.DataFrame:
    """Population engagement, LFP, MUA and position over each interval's rows.

    Parameters
    ----------
    traces : RowTraces
    intervals : pd.DataFrame
        ``event_start_s`` and ``event_end_s`` of each interval, in seconds.
    sample_interval_s : float
        The grid's step, the time one row stands for.

    Returns
    -------
    features : pd.DataFrame
        One row per interval, on ``intervals``' index: ``first_row``,
        ``n_rows``; ``n_spikes`` and ``n_active_tetrodes`` (tetrodes with a
        spike), ``n_tetrodes``, ``spike_rate_hz`` (``n_spikes / (n_rows *
        sample_interval_s)``); ``ripple_z_max``, ``ripple_z_mean``,
        ``mua_z_max``, ``mua_z_mean`` (NaN when any row is NaN); and
        ``linear_position_cm``, ``track_segment_id``, ``patch_id`` at the
        middle row (the earlier of two). Every feature but ``first_row``,
        ``n_rows`` and ``n_tetrodes`` is missing for an interval with no row.
        The rows are the module's: the grid's last sample is never one.
    """
    first, n_rows = _grid_rows(traces.time, intervals)
    has_rows = n_rows > 0
    per_tetrode = _over_rows(np.add, traces.spike_counts, first, n_rows)
    n_spikes = per_tetrode.sum(axis=1)
    middle = middle_row(first, n_rows)[has_rows]
    n_tetrodes = traces.spike_counts.shape[1]
    columns: dict[str, object] = {
        "first_row": first,
        "n_rows": n_rows,
        "n_spikes": _int_or_na(n_spikes, has_rows),
        "n_active_tetrodes": _int_or_na((per_tetrode > 0).sum(axis=1), has_rows),
        "n_tetrodes": np.full(first.shape, n_tetrodes, dtype=np.int64),
        "spike_rate_hz": _float_or_nan(
            n_spikes / (n_rows[has_rows] * sample_interval_s), has_rows
        ),
    }
    for name, trace in (("ripple_z", traces.ripple_z), ("mua_z", traces.mua_z)):
        # np.maximum and np.add carry a NaN row through to the interval's value
        columns[f"{name}_max"] = _float_or_nan(
            _over_rows(np.maximum, trace, first, n_rows), has_rows
        )
        columns[f"{name}_mean"] = _float_or_nan(
            _over_rows(np.add, trace, first, n_rows) / n_rows[has_rows], has_rows
        )
    columns["linear_position_cm"] = _float_or_nan(
        traces.linear_position[middle], has_rows
    )
    columns["track_segment_id"] = _int_or_na(traces.track_segment_id[middle], has_rows)
    columns["patch_id"] = _int_or_na(traces.patch_id[middle], has_rows)
    return pd.DataFrame(columns, index=intervals.index)


def _real_bins(bins: NDArray, n_bins: int, n_expected: int, name: str) -> NDArray:
    """``bins`` as int64 after checking they index the real place bins.

    -1 marks an unknown bin (``animal_bins``' value, and ``NO_CENTER``).
    """
    bins = np.asarray(bins, dtype=np.int64)
    if bins.shape != (n_expected,):
        raise ValueError(f"{name} has shape {bins.shape}; expected ({n_expected},).")
    if ((bins < -1) | (bins >= n_bins)).any():
        raise ValueError(
            f"{name} holds an index outside the {n_bins} real place bins of the "
            "distance matrix; a junction placeholder bin must never enter."
        )
    return bins


def _pair_distance_cm(
    bin_distance_cm: NDArray, here: NDArray, there: NDArray
) -> NDArray[np.float64]:
    """Track distance from each ``here`` bin to its ``there`` bin; NaN where
    either is unknown (-1)."""
    known = (here >= 0) & (there >= 0)
    distance: NDArray[np.float64] = np.where(
        known, bin_distance_cm[np.maximum(here, 0), np.maximum(there, 0)], np.nan
    )
    return distance


def _extents_cm(
    bins: NDArray, first: NDArray, n_rows: NDArray, bin_distance_cm: NDArray
) -> NDArray[np.float64]:
    """``dispersion_cm`` of each interval's bins (``NO_CENTER`` left out);
    NaN with none."""
    return np.array(
        [
            dispersion_cm(bins[start : start + n], bin_distance_cm) if n else np.nan
            for start, n in zip(first.tolist(), n_rows.tolist(), strict=True)
        ],
        dtype=np.float64,
    )


def _nearest_event_gap_s(intervals: pd.DataFrame, events: pd.DataFrame) -> NDArray:
    """Each interval's end-to-start gap to the nearest of ``events``.

    ``events`` are one detector's, disjoint. 0 where an event overlaps or
    touches the interval; NaN when there is no event.
    """
    starts = intervals["event_start_s"].to_numpy(dtype=np.float64)
    ends = intervals["event_end_s"].to_numpy(dtype=np.float64)
    if len(events) == 0:
        return np.full(starts.shape, np.nan)
    ordered = events.sort_values("event_start_s", kind="stable")
    event_start = ordered["event_start_s"].to_numpy(dtype=np.float64)
    event_end = ordered["event_end_s"].to_numpy(dtype=np.float64)
    # disjoint events: the nearest is the last starting at or before the
    # interval's end, or the first starting after it
    before = np.searchsorted(event_start, ends, side="right") - 1
    after = before + 1
    gap_before = np.where(
        before >= 0, np.maximum(starts - event_end[np.maximum(before, 0)], 0.0), np.inf
    )
    gap_after = np.where(
        after < event_start.size,
        event_start[np.minimum(after, event_start.size - 1)] - ends,
        np.inf,
    )
    gap: NDArray = np.minimum(gap_before, gap_after)
    return gap


def _runs_above(
    above: NDArray[np.bool_], time: NDArray, first: NDArray, n_rows: NDArray
) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
    """The runs of ``above`` inside each interval, each cut at its bounds.

    Returns
    -------
    n_runs : NDArray[np.int64], shape (n_intervals,)
    longest_s : NDArray, shape (n_intervals,)
        The longest run's last sample time less its first; NaN with no run.
    """
    run_start, run_stop = contiguous_true_runs(above).T
    stop = first + n_rows
    # runs are disjoint and ordered: those ending after the interval's first
    # row and starting before its end are its own
    low = np.searchsorted(run_stop, first, side="right")
    high = np.searchsorted(run_start, stop, side="left")
    n_runs = np.where(n_rows > 0, high - low, 0)
    interval = np.repeat(np.arange(first.size), n_runs)
    run = np.repeat(low, n_runs) + (
        np.arange(interval.size) - np.repeat(np.cumsum(n_runs) - n_runs, n_runs)
    )
    cut_first = np.maximum(run_start[run], first[interval])
    cut_last = np.minimum(run_stop[run], stop[interval]) - 1
    longest = np.full(first.shape, np.nan)
    np.fmax.at(longest, interval, time[cut_last] - time[cut_first])
    return n_runs.astype(np.int64), longest


def ss_features(
    traces: RowTraces,
    posterior: PosteriorTraces,
    intervals: pd.DataFrame,
    bin_distance_cm: NDArray,
    map_bin: NDArray,
    animal_bin: NDArray,
    *,
    state_space_events: pd.DataFrame,
) -> pd.DataFrame:
    """The state-space decode's posterior over each interval's rows.

    **Runs.** A run is a stretch of consecutive rows whose non-local
    probability is above ``DEFAULT_NON_LOCAL_THRESHOLD`` (0.8), compared in
    the frame's float32 as the decode compared it (``get_confident_times``;
    a row that reads 0.8 in float32 is not above). Inside an interval a run
    is cut at the interval's bounds. A run is measured as
    ``get_confident_times`` measures an event, its last sample's time less its
    first, so it reads on the scale of the state-space events'
    ``interval_span_s``. ``get_confident_times`` made an event of a run of at
    least ``ripple_detection.core.minimum_sample_count(time, 0.015)`` rows:
    the 15 ms rounded half up to a whole number of samples of the grid's
    median step. 15 ms is 7.5 samples of 2 ms; on the decode grids the median
    step reads 2.0000935 ms (float error at their clock), so the count is 7
    rows, a 12 ms span, where an exact 2 ms step gives 8. So a run inside an
    event no compared state-space event matches is shorter than that,
    continues past the interval's bound, or formed an event that is not
    compared (moving, or set aside).

    Parameters
    ----------
    traces : RowTraces
        The epoch's grid.
    posterior : PosteriorTraces
        On the same grid.
    intervals : pd.DataFrame
        ``event_start_s``, ``event_end_s`` and ``detector`` of each interval,
        of one epoch.
    bin_distance_cm : NDArray, shape (n_bins, n_bins)
        Track-graph distances between the real place bins
        (``interior_bin_distances_cm``).
    map_bin, animal_bin : NDArray of int, shape (n_time,)
        Each row's real bin nearest the MAP estimate and the animal
        (``evidence.animal_bins``, through ``nearest_interior_bins``, so a
        position in a junction gap maps to a real bin); -1 where unknown.
    state_space_events : pd.DataFrame
        The epoch's compared state-space events (``whole_event_speed_eligible``
        true), ``event_start_s`` and ``event_end_s``.

    Returns
    -------
    features : pd.DataFrame
        On ``intervals``' index, over each interval's rows:

        - ``ss_local_prob_mean``, ``ss_no_spike_prob_mean``,
          ``ss_continuous_prob_mean``, ``ss_fragmented_prob_mean``,
          ``ss_hpd_cm_mean``: means;
        - ``ss_nonlocal_prob_max``: the non-local probability's maximum;
        - ``ss_frac_rows_nonlocal``: the rows above the threshold over
          ``n_rows``;
        - ``ss_extent_cm``: ``dispersion_cm`` of the rows' MAP bins;
        - ``ss_distance_cm_mean``: the mean track distance from the MAP bin to
          the animal's bin;
        - ``ss_n_runs_nonlocal`` (Int64) and ``ss_longest_run_nonlocal_s``
          (NaN with no run): the runs inside the interval.

        Each is missing for an interval with no row, and for one holding a
        row where it is unknown: a NaN probability or ``hpd``, or a bin of -1
        for the spatial two. Besides, not read on rows:

        - ``ss_nearest_event_gap_s``: the end-to-start gap to the nearest of
          ``state_space_events``, 0 when one overlaps or touches the interval,
          NaN with none and for the state-space detector's own intervals;
        - ``ss_disagreement``: for every other detector's interval, ``"near"``
          when that gap is at most ``max(FRAGMENT_GAPS_S)`` (50 ms, the bound
          included; gaps compared to the microsecond, as a clock near 1.6e9 s
          reads 25 samples of 2 ms as 50.0002 ms). The fragment recount merges
          only gaps below 50 ms, so a gap of exactly 50 ms is ``"near"`` but
          not merged. Else ``"subthreshold"`` when a run
          lies inside it, else ``"sustained"``; None for the state-space
          detector and where the runs are unknown. It describes an event no
          compared state-space event matches; it does not decide anything.

    Raises
    ------
    ValueError
        An array is not on the grid, a bin is outside the distance matrix, or
        a state-space event is not compared.
    """
    time = traces.time
    _check_lengths(posterior, time.size)
    n_bins = bin_distance_cm.shape[0]
    map_bin = _real_bins(map_bin, n_bins, time.size, "map_bin")
    animal_bin = _real_bins(animal_bin, n_bins, time.size, "animal_bin")
    if not state_space_events["whole_event_speed_eligible"].all():
        raise ValueError(
            "The nearest state-space event is a compared one; pass only the rows "
            "with whole_event_speed_eligible true."
        )
    first, n_rows = _grid_rows(time, intervals)
    has_rows = n_rows > 0

    def mean(values: NDArray) -> NDArray[np.float64]:
        total = _over_rows(np.add, values, first, n_rows)
        return _float_or_nan(total / n_rows[has_rows], has_rows)

    columns: dict[str, object] = {
        f"ss_{name}_prob_mean": mean(values)
        for name, values in (
            ("local", posterior.local_prob),
            ("no_spike", posterior.no_spike_prob),
            ("continuous", posterior.continuous_prob),
            ("fragmented", posterior.fragmented_prob),
        )
    }
    non_local = posterior.nonlocal_prob.astype(np.float64)
    columns["ss_nonlocal_prob_max"] = _float_or_nan(
        _over_rows(np.maximum, non_local, first, n_rows), has_rows
    )
    known = has_rows.copy()
    known[has_rows] = ~_over_rows(np.logical_or, np.isnan(non_local), first, n_rows)
    above = posterior.nonlocal_prob > DEFAULT_NON_LOCAL_THRESHOLD
    n_above = np.zeros(first.shape, dtype=np.int64)
    n_above[has_rows] = _over_rows(np.add, above.astype(np.int64), first, n_rows)
    columns["ss_frac_rows_nonlocal"] = np.where(
        known, n_above / np.maximum(n_rows, 1), np.nan
    )

    unknown_map = np.zeros(first.shape, dtype=bool)
    unknown_map[has_rows] = _over_rows(np.logical_or, map_bin < 0, first, n_rows)
    extent = _extents_cm(map_bin, first, n_rows, bin_distance_cm)
    columns["ss_extent_cm"] = np.where(unknown_map, np.nan, extent)
    columns["ss_distance_cm_mean"] = mean(
        _pair_distance_cm(bin_distance_cm, map_bin, animal_bin)
    )
    columns["ss_hpd_cm_mean"] = mean(posterior.hpd_cm)

    n_runs, longest = _runs_above(above, time, first, n_rows)
    columns["ss_n_runs_nonlocal"] = pd.arrays.IntegerArray(n_runs, ~known)
    columns["ss_longest_run_nonlocal_s"] = np.where(known, longest, np.nan)

    is_ours = (intervals["detector"] == STATE_SPACE).to_numpy()
    gap = np.where(is_ours, np.nan, _nearest_event_gap_s(intervals, state_space_events))
    near = np.round(gap, 6) <= max(FRAGMENT_GAPS_S)
    disagreement = np.where(
        near, "near", np.where(n_runs > 0, "subthreshold", "sustained")
    ).astype(object)
    disagreement[is_ours | (~near & ~known)] = None
    columns["ss_nearest_event_gap_s"] = gap
    columns["ss_disagreement"] = disagreement
    return pd.DataFrame(columns, index=intervals.index)


STATE_SHARE_COLUMNS = (
    "ss_frac_rows_local",
    "ss_frac_rows_no_spike",
    "ss_frac_rows_nonlocal",
    "ss_frac_rows_uncertain",
)
"""The columns :func:`state_row_shares` returns, in order."""


def state_row_shares(
    posterior: PosteriorTraces, first: NDArray, n_rows: NDArray
) -> pd.DataFrame:
    """The share of each interval's rows in which one state is confident.

    A row is Local, No-Spike or non-local when that state's probability is
    above ``DEFAULT_NON_LOCAL_THRESHOLD`` (0.8), compared in float32 as
    :func:`ss_features` compares the non-local sum; the non-local share is
    therefore ``ss_frac_rows_nonlocal``. A row with no state above the
    threshold is uncertain. The threshold is above one half, so no row is in
    two states and the four shares of an interval sum to 1.

    Parameters
    ----------
    posterior : PosteriorTraces
        One epoch's posterior on the decode grid.
    first, n_rows : NDArray of int, shape (n_intervals,)
        Each interval's first grid row and its number of rows, as the
        features table stores them (``first_row``, ``n_rows``).

    Returns
    -------
    shares : pd.DataFrame, shape (n_intervals, 4)
        :data:`STATE_SHARE_COLUMNS`, positionally indexed. NaN for an
        interval with no row, and for one holding a row where a state
        probability is NaN.
    """
    first = np.asarray(first, dtype=np.int64)
    n_rows = np.asarray(n_rows, dtype=np.int64)
    has_rows = n_rows > 0
    threshold = np.float32(DEFAULT_NON_LOCAL_THRESHOLD)
    local = posterior.local_prob.astype(np.float32) > threshold
    no_spike = posterior.no_spike_prob.astype(np.float32) > threshold
    non_local = posterior.nonlocal_prob > DEFAULT_NON_LOCAL_THRESHOLD
    uncertain = ~(local | no_spike | non_local)
    unknown = (
        np.isnan(posterior.local_prob)
        | np.isnan(posterior.no_spike_prob)
        | np.isnan(posterior.nonlocal_prob)
    )
    known = has_rows.copy()
    known[has_rows] = ~_over_rows(np.logical_or, unknown, first, n_rows)

    def share(rows: NDArray[np.bool_]) -> NDArray[np.float64]:
        count = np.zeros(first.shape, dtype=np.int64)
        count[has_rows] = _over_rows(np.add, rows.astype(np.int64), first, n_rows)
        return np.where(known, count / np.maximum(n_rows, 1), np.nan)

    return pd.DataFrame(
        dict(
            zip(
                STATE_SHARE_COLUMNS,
                (share(local), share(no_spike), share(non_local), share(uncertain)),
                strict=True,
            )
        )
    )


def ms_features(
    candidates: CandidateSequences,
    intervals: pd.DataFrame,
    undecoded: NDArray,
    bin_distance_cm: NDArray,
    animal_bin_at_window: NDArray,
) -> pd.DataFrame:
    """The memoryless run's stored window statistics over each interval.

    An interval's windows are those whose midpoint (``midpoint_s``, a grid
    sample) lies in it, both bounds included. The stored centers are after
    the speed gate, so a window whose midpoint speed is not below the
    threshold has none: it is counted as missing, not recomputed. An
    undecoded window (holding a sample of the artifact mask) is counted in
    ``ms_n_windows`` alone and enters no other feature.

    Parameters
    ----------
    candidates : CandidateSequences
        The epoch's stored memoryless run.
    intervals : pd.DataFrame
        ``event_start_s`` and ``event_end_s`` of each interval, in seconds.
    undecoded : NDArray of bool, shape (n_windows,)
        The windows the run left undecoded (``windows_holding`` on the mask).
    bin_distance_cm : NDArray, shape (n_bins, n_bins)
        Track-graph distances between the real place bins, as the run's.
    animal_bin_at_window : NDArray of int, shape (n_windows,)
        The animal's real bin at each window's midpoint sample; -1 where
        unknown.

    Returns
    -------
    features : pd.DataFrame
        On ``intervals``' index: ``ms_n_windows``; over its decoded windows,
        ``ms_n_windows_decoded``, ``ms_n_windows_with_center``
        (``gates.has_center``), ``ms_spread_cm_mean`` (``spread_cm`` over the
        windows whose posterior was usable), ``ms_extent_cm``
        (``dispersion_cm`` of every center, kept or not; a candidate's stored
        ``dispersion_cm`` reads its kept windows alone),
        ``ms_distance_cm_mean`` (the mean track distance from each center to
        the animal's bin, NaN when one of those bins is unknown) and
        ``ms_frac_windows_kept`` (``gates.kept`` over the decoded windows).
        With no window every feature but ``ms_n_windows`` is missing, and a
        mean or extent with nothing to read is NaN.

    Raises
    ------
    ValueError
        ``undecoded`` or ``animal_bin_at_window`` is not one per window, or a
        center or the animal's bin is outside the distance matrix.
    """
    midpoint = candidates.midpoint_s
    n_windows = midpoint.size
    undecoded = np.asarray(undecoded, dtype=bool)
    if undecoded.shape != (n_windows,):
        raise ValueError(
            f"undecoded has shape {undecoded.shape}; there are {n_windows} windows."
        )
    n_bins = bin_distance_cm.shape[0]
    animal_bin = _real_bins(
        animal_bin_at_window, n_bins, n_windows, "animal_bin_at_window"
    )
    gates = candidates.decisions.gates
    decoded = ~undecoded
    with_center = gates.has_center & decoded
    center = np.where(with_center, candidates.decisions.center_index, NO_CENTER)
    center = _real_bins(center, n_bins, n_windows, "center_index")
    usable = gates.posterior_usable & decoded
    first, n_in = interval_rows(
        midpoint,
        intervals["event_start_s"].to_numpy(dtype=np.float64),
        intervals["event_end_s"].to_numpy(dtype=np.float64),
    )
    has_windows = n_in > 0

    def total(values: NDArray) -> NDArray:
        summed = np.zeros(first.shape, dtype=values.dtype)
        summed[has_windows] = _over_rows(np.add, values, first, n_in)
        return summed

    def ratio(numerator: NDArray, denominator: NDArray) -> NDArray[np.float64]:
        return np.where(denominator > 0, numerator / np.maximum(denominator, 1), np.nan)

    n_decoded = total(decoded.astype(np.int64))
    n_with_center = total(with_center.astype(np.int64))
    n_usable = total(usable.astype(np.int64))
    spread = np.where(usable, candidates.decisions.spread_cm, 0.0)
    distance = np.where(
        with_center, _pair_distance_cm(bin_distance_cm, center, animal_bin), 0.0
    )
    return pd.DataFrame(
        {
            "ms_n_windows": n_in,
            "ms_n_windows_decoded": pd.arrays.IntegerArray(n_decoded, ~has_windows),
            "ms_n_windows_with_center": pd.arrays.IntegerArray(
                n_with_center, ~has_windows
            ),
            "ms_spread_cm_mean": ratio(total(spread), n_usable),
            "ms_extent_cm": _extents_cm(center, first, n_in, bin_distance_cm),
            "ms_distance_cm_mean": ratio(total(distance), n_with_center),
            "ms_frac_windows_kept": ratio(
                total((gates.kept & decoded).astype(np.int64)), n_decoded
            ),
        },
        index=intervals.index,
    )


def neighbour_gaps(inventory: pd.DataFrame) -> pd.DataFrame:
    """Each compared event's gaps to its neighbours in its own inventory.

    Neighbours are the events of the same detector in the same epoch. One
    detector's events are disjoint (the inventory schema requires it), so the
    event before in onset order is the one ending last before the onset.

    Parameters
    ----------
    inventory : pd.DataFrame
        Compared events (``whole_event_speed_eligible`` true) in the inventory
        schema, of any detectors and epochs.

    Returns
    -------
    gaps : pd.DataFrame
        On ``inventory``'s index: ``previous_gap_s``, the onset less the end
        of the event before, and ``next_gap_s``, the next event's onset less
        the end; NaN at the epoch's first and last event.

    Raises
    ------
    ValueError
        An event is not compared; its neighbours would not be the compared
        events'.
    """
    if not inventory["whole_event_speed_eligible"].all():
        raise ValueError(
            "Gaps are measured between compared events; pass only the rows with "
            "whole_event_speed_eligible true."
        )
    keys = [*SESSION_COLUMNS_DAY, "detector"]
    ordered = inventory.sort_values([*keys, "event_start_s"], kind="stable")
    groups = ordered.groupby(keys, sort=False)
    gaps = pd.DataFrame(
        {
            "previous_gap_s": ordered["event_start_s"] - groups["event_end_s"].shift(1),
            "next_gap_s": groups["event_start_s"].shift(-1) - ordered["event_end_s"],
        }
    )
    return gaps.reindex(inventory.index)


def stop_context(intervals: pd.DataFrame, stops: pd.DataFrame) -> pd.DataFrame:
    """The stop holding each interval's onset, and the time into it.

    An interval belongs to the stop holding its onset, both bounds of the stop
    included, as ``events_in_stops`` assigns it.

    Parameters
    ----------
    intervals : pd.DataFrame
        ``event_start_s`` of each interval, in seconds.
    stops : pd.DataFrame
        One epoch's rows of the stop table, carrying ``stop_start``,
        ``stop_end``, ``last_well_arrival_time_s`` and the columns of
        ``STOP_CONTEXT_DTYPES``; disjoint, in any order.

    Returns
    -------
    context : pd.DataFrame
        On ``intervals``' index: the columns of ``STOP_CONTEXT_DTYPES``,
        ``time_in_stop_s`` (onset less ``stop_start``) and
        ``time_since_arrival_s`` (onset less ``last_well_arrival_time_s``);
        missing where the onset is in no stop.
    """
    onsets = intervals["event_start_s"].to_numpy(dtype=np.float64)
    in_a_stop = events_in_stops(onsets, stops)
    ordered = stops.sort_values("stop_start", kind="stable").reset_index(drop=True)
    latest_before = (
        np.searchsorted(
            ordered["stop_start"].to_numpy(dtype=np.float64), onsets, side="right"
        )
        - 1
    )
    # disjoint stops: only the latest one beginning at or before the onset can
    # hold it; -1 is no row, so reindex leaves the context missing
    holding = np.where(in_a_stop, latest_before, -1)
    columns = [*STOP_CONTEXT_DTYPES, "stop_start", "last_well_arrival_time_s"]
    taken = (
        ordered[columns]
        .astype(dict(STOP_CONTEXT_DTYPES) | {"stop_start": np.float64})
        .reindex(holding)
    )
    context = taken[list(STOP_CONTEXT_DTYPES)].set_axis(intervals.index)
    context["time_in_stop_s"] = onsets - taken["stop_start"].to_numpy(dtype=np.float64)
    context["time_since_arrival_s"] = onsets - taken[
        "last_well_arrival_time_s"
    ].to_numpy(dtype=np.float64)
    return context


def duration_stratum(interval_span_s: NDArray) -> pd.Categorical:
    """Each span's stratum of ``DURATION_STRATA_S``.

    Spans are rounded to the microsecond first: a span read off timestamps
    near 1.6e9 s carries float error (25 samples of 2 ms read 0.04999995 s),
    and must fall where its whole number of samples falls.

    Parameters
    ----------
    interval_span_s : NDArray, shape (n_intervals,)
        ``event_end_s - event_start_s``, in seconds.

    Returns
    -------
    strata : pd.Categorical, shape (n_intervals,)
        Ordered, with the categories of ``DURATION_STRATUM_LABELS``.
    """
    spans = np.round(np.asarray(interval_span_s, dtype=np.float64), 6)
    strata: pd.Categorical = pd.cut(
        spans,
        bins=list(DURATION_STRATA_S),
        labels=list(DURATION_STRATUM_LABELS),
        right=True,
    )
    return strata
