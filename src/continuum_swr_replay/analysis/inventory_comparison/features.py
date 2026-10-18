"""Per-event features, read on the decode grid by one rule for every inventory.

An interval's **rows** are the decode grid's samples ``t`` with
``event_start_s <= t <= event_end_s``. Every feature of an interval is read
over its rows and no others, so two inventories' identical intervals get
identical features, whatever detector found them. An interval with no row
(shorter than a sample, or off the grid) has every row feature missing.

Row ``k`` holds the spikes with ``t_k <= s < t_{k + 1}``, the rule of
``tetrode_spike_counts`` and of the decode. The grid's last sample has no such
span, so no interval may reach it.

A summary over rows (a mean or a maximum) is NaN when any of its rows is NaN,
so a masked sample inside an interval is never skipped silently.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from types import MappingProxyType

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.hippocampus.swr_inventories import (
    tetrode_spike_counts,
)
from continuum_swr_replay.analysis.replay.cohort_selection import events_in_stops
from continuum_swr_replay.session_keys import SESSION_COLUMNS_DAY

__all__ = [
    "DURATION_STRATA_S",
    "DURATION_STRATUM_LABELS",
    "STOP_CONTEXT_DTYPES",
    "RowTraces",
    "duration_stratum",
    "interval_features",
    "interval_rows",
    "neighbour_gaps",
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
    """The grid rows of each closed interval.

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
        n_time = self.time.shape[0]
        for field in fields(self):
            length = getattr(self, field.name).shape[0]
            if length != n_time:
                raise ValueError(
                    f"{field.name} has {length} rows; the grid has {n_time}."
                )

    @classmethod
    def from_frame(
        cls, dot: pd.DataFrame, spike_times: Sequence[NDArray]
    ) -> "RowTraces":
        """The traces of a decode frame and its tetrodes' spikes.

        Parameters
        ----------
        dot : pd.DataFrame
            The decode frame, one row per grid sample, its times in a ``time``
            column or the index.
        spike_times : sequence of NDArray, each shape (n_spikes,)
            Spike times in seconds, one array per tetrode.
        """
        time = np.asarray(
            dot["time"] if "time" in dot.columns else dot.index, dtype=np.float64
        )
        return cls(
            time=time,
            spike_counts=tetrode_spike_counts(spike_times, time),
            ripple_z=dot["SWR_zscore"].to_numpy(dtype=np.float64),
            mua_z=dot["MUA_zscore"].to_numpy(dtype=np.float64),
            linear_position=dot["linear_position"].to_numpy(dtype=np.float64),
            track_segment_id=dot["track_segment_id"].to_numpy(dtype=np.int64),
            patch_id=dot["patch_id"].to_numpy(dtype=np.int64),
        )


def _over_rows(
    ufunc: np.ufunc, values: NDArray, first: NDArray, n_rows: NDArray
) -> NDArray:
    """``ufunc`` reduced over the rows of each interval that has any.

    Returns
    -------
    reduced : NDArray, shape (n_with_rows, *values.shape[1:])
    """
    has_rows = n_rows > 0
    edges = np.column_stack([first[has_rows], first[has_rows] + n_rows[has_rows]])
    if edges.size == 0:
        return np.empty((0, *values.shape[1:]), dtype=values.dtype)
    # reduceat over (start, end) pairs: each even entry is one interval's rows
    reduced: NDArray = ufunc.reduceat(values, edges.ravel(), axis=0)[::2]
    return reduced


def _float_or_nan(reduced: NDArray, has_rows: NDArray) -> NDArray[np.float64]:
    values = np.full(has_rows.shape, np.nan)
    values[has_rows] = reduced
    return values


def _int_or_na(reduced: NDArray, has_rows: NDArray) -> pd.arrays.IntegerArray:
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

    Raises
    ------
    ValueError
        An interval's rows reach the grid's last sample, which holds no span
        of spikes.
    """
    first, n_rows = interval_rows(
        traces.time,
        intervals["event_start_s"].to_numpy(dtype=np.float64),
        intervals["event_end_s"].to_numpy(dtype=np.float64),
    )
    has_rows = n_rows > 0
    n_time = traces.time.size
    if (has_rows & (first + n_rows >= n_time)).any():
        raise ValueError(
            "An interval reaches the grid's last sample, which holds no span of "
            "spikes and is never read."
        )
    per_tetrode = _over_rows(np.add, traces.spike_counts, first, n_rows)
    n_spikes = per_tetrode.sum(axis=1)
    middle = (first + (n_rows - 1) // 2)[has_rows]
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
