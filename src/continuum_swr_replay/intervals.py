"""Interval algebra: the one home of the project's interval primitives.

Times are seconds. An interval is ``[start, end]``. Two families live here,
with the array and DataFrame signatures their callers use, beside the overlap
predicates, the runs of a boolean mask (:func:`contiguous_true_runs`), a time
grid's sample step (:func:`sample_step`) and the step that is a gap in it
(:func:`max_step_s`), and spike binning.

**Interval sets** (:func:`normalize`, :func:`union`, :func:`intersect`,
:func:`subtract`, :func:`total_duration`, :func:`contains`) work on arrays of
shape ``(n, 2)``. Every function that returns intervals returns a normalized
set: sorted by start, disjoint, with overlapping or touching intervals joined
into one, and with any interval of no duration, or with a missing or inverted
bound, dropped. A shared endpoint therefore intersects in nothing, which is
what a positive-duration overlap rule needs: a single sample observes no time.
Dropping an inverted or non-finite bound is right for the outputs of interval
arithmetic, where an empty intersection appears as an inverted pair; for an
input a caller means to be well formed, ``normalize(..., strict=True)`` raises
instead.

**Closed intervals** (:func:`merge_intervals`, :func:`interval_mask`,
:func:`subtract_intervals`, :func:`masked_seconds`, :func:`pad_intervals`,
:func:`overlaps_intervals`, :func:`covered_seconds`) take intervals as a
``start_time`` / ``end_time`` DataFrame (or start and end arrays), closed: a
time on an endpoint is inside, and intervals that touch merge. This is the
convention of the artifact mask and of excluded time
(``analysis/event_definition/artifact_masks.py``, ``event_definition.eligibility``). They
refuse an interval with a NaN bound or with ``start > end`` (``ValueError``,
naming the rows): such an interval has no closed reading, and merging one
would silently hide its neighbors.

**Overlap conventions.** "Overlap" means four different things in the
project. Each has its own name, so the choice is visible at each caller:

=============  ==============================================  ========  ===========================
convention     predicate                                       touching  zero-length interval
=============  ==============================================  ========  ===========================
closed         :func:`overlaps_intervals`                      overlaps  overlaps any interval it
                                                                         shares a time with
open ``<``     :func:`overlaps_open`                           no        overlaps strictly inside
strict         ``nonlocal_event_rule.strict_overlap_matrix``;   no        overlaps nothing
               the interval sets, through :func:`normalize`
dropped-event  ``event_definition.eligibility.overlaps_dropped_events``  no        a zero-length dropped
                                                                         interval is ignored; a
                                                                         zero-length event strictly
                                                                         inside a dropped one
                                                                         overlaps it
=============  ==============================================  ========  ===========================

Strict overlap is event-level ``is_swr``: both intervals finite with positive
duration, and ``other_end > start`` and ``other_start < end``. The open test is
the same comparison without the duration check; it agrees with strict overlap
on intervals of positive duration. Membership of a time point is closed
(:func:`interval_mask`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = [
    "contains",
    "contiguous_true_runs",
    "min_gap_to_intervals",
    "covered_seconds",
    "decoder_time_bin",
    "get_spikecount_per_time_bin",
    "as_frame",
    "GAP_STEP_RATIO",
    "max_step_s",
    "interval_mask",
    "intersect",
    "masked_seconds",
    "merge_intervals",
    "normalize",
    "overlaps_intervals",
    "overlaps_open",
    "pad_intervals",
    "subtract",
    "subtract_intervals",
    "total_duration",
    "union",
]

# ---------------------------------------------------------------------------
# Interval sets: arrays of shape (n, 2), normalized.
# ---------------------------------------------------------------------------


def _pairs(intervals: NDArray) -> NDArray:
    array = np.asarray(intervals, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 2))
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(
            f"Intervals must be (start, end) pairs; got shape {array.shape}."
        )
    return array


def normalize(intervals: NDArray, *, strict: bool = False) -> NDArray:
    """Sort, join overlapping or touching intervals, and drop empty ones.

    Parameters
    ----------
    intervals : ndarray, shape (n, 2)
    strict : bool, default False
        Raise on an inverted or non-finite bound instead of dropping it, for
        an input that is supposed to be well formed.

    Returns
    -------
    normalized : ndarray, shape (m, 2)

    """
    array = _pairs(intervals)
    if strict and array.size:
        finite = np.isfinite(array).all(axis=1)
        if not finite.all() or (array[finite, 1] < array[finite, 0]).any():
            raise ValueError("An interval has a non-finite or inverted bound.")
    array = array[np.isfinite(array).all(axis=1) & (array[:, 1] > array[:, 0])]
    if array.size == 0:
        return np.empty((0, 2))
    array = array[np.argsort(array[:, 0], kind="stable")]
    # a new interval starts wherever a start lies beyond every earlier end
    reach = np.maximum.accumulate(array[:, 1])
    begins = np.r_[True, array[1:, 0] > reach[:-1]]
    starts = array[begins, 0]
    ends = np.maximum.reduceat(array[:, 1], np.flatnonzero(begins))
    return np.column_stack([starts, ends])


def union(*sets: NDArray) -> NDArray:
    """Every time covered by any of the sets."""
    if not sets:
        return np.empty((0, 2))
    return normalize(np.vstack([_pairs(each) for each in sets]))


def intersect(a: NDArray, b: NDArray) -> NDArray:
    """Every time covered by both sets."""
    a, b = normalize(a), normalize(b)
    if a.size == 0 or b.size == 0:
        return np.empty((0, 2))
    starts = np.maximum(a[:, None, 0], b[None, :, 0])
    ends = np.minimum(a[:, None, 1], b[None, :, 1])
    return normalize(np.column_stack([starts.ravel(), ends.ravel()]))


def subtract(a: NDArray, b: NDArray) -> NDArray:
    """Every time covered by ``a`` and not by ``b``."""
    a, b = normalize(a), normalize(b)
    if a.size == 0 or b.size == 0:
        return a
    # the complement of b, bounded by a's own span so no bound is infinite
    low, high = a[0, 0], a[-1, 1]
    gaps = np.column_stack([np.r_[low, b[:, 1]], np.r_[b[:, 0], high]])
    return intersect(a, gaps)


def total_duration(intervals: NDArray) -> float:
    """Seconds covered, overlap counted once."""
    normalized = normalize(intervals)
    return float((normalized[:, 1] - normalized[:, 0]).sum())


def contains(intervals: NDArray, starts: NDArray, ends: NDArray) -> NDArray:
    """Whether each ``[start, end]`` lies wholly inside the set.

    Parameters
    ----------
    intervals : ndarray, shape (n, 2)
    starts, ends : ndarray, shape (n_events,)

    Returns
    -------
    within : ndarray of bool, shape (n_events,)

    """
    normalized = normalize(intervals)
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    if starts.shape != ends.shape:
        raise ValueError("starts and ends must be the same shape.")
    if (ends < starts).any():
        raise ValueError("An event has end < start; nothing contains it.")
    if normalized.size == 0:
        return np.zeros(starts.shape, dtype=bool)
    index = np.searchsorted(normalized[:, 0], starts, side="right") - 1
    found = index >= 0
    safe = np.clip(index, 0, None)
    within: NDArray = (
        found & (normalized[safe, 0] <= starts) & (ends <= normalized[safe, 1])
    )
    return within


# ---------------------------------------------------------------------------
# Closed intervals: start_time / end_time frames.
# ---------------------------------------------------------------------------


def _require_bounds(
    starts: NDArray[np.float64], ends: NDArray[np.float64], what: str
) -> None:
    """Raise unless every interval has two non-NaN bounds and ``start <= end``.

    Parameters
    ----------
    starts, ends : ndarray, shape (n,)
    what : str
        The function and argument, for the message.

    Raises
    ------
    ValueError
        Naming the offending rows (the first five, and how many in all).
    """
    bad = np.isnan(starts) | np.isnan(ends) | (starts > ends)
    if bad.any():
        rows = np.flatnonzero(bad)
        shown = ", ".join(f"{i}: [{starts[i]!r}, {ends[i]!r}]" for i in rows[:5])
        more = f", and {rows.size - 5} more" if rows.size > 5 else ""
        raise ValueError(
            f"{what}: {rows.size} interval(s) with a NaN bound or start > end "
            f"(row: [start, end]) {shown}{more}"
        )


def merge_intervals(
    starts: NDArray[np.float64], ends: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Merge closed intervals that overlap or touch, in time order.

    Parameters
    ----------
    starts, ends : ndarray, shape (n_intervals,)
        Seconds; ``ends >= starts``.

    Returns
    -------
    ndarray, shape (n_merged, 2)
        Disjoint ``[start, end]`` rows, sorted, with a gap between each.

    Raises
    ------
    ValueError
        If a bound is NaN or a start is after its end.
    """
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    _require_bounds(starts, ends, "merge_intervals")
    if starts.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    order = np.argsort(starts, kind="stable")
    starts, ends = starts[order], ends[order]
    reach = np.maximum.accumulate(ends)
    # closed: a start on the running end touches it, so it merges
    first = np.flatnonzero(np.r_[True, starts[1:] > reach[:-1]])
    return np.column_stack([starts[first], np.maximum.reduceat(ends, first)])


def as_frame(intervals: NDArray[np.float64]) -> pd.DataFrame:
    """An ``(n, 2)`` interval array as a ``start_time`` / ``end_time`` frame.

    Parameters
    ----------
    intervals : NDArray, shape (n_intervals, 2)
        Start and end of each interval, in seconds.

    Returns
    -------
    pd.DataFrame
        Columns ``start_time`` and ``end_time`` (float64, seconds), in the
        array's order.
    """
    return pd.DataFrame(
        {
            "start_time": intervals[:, 0].astype(np.float64),
            "end_time": intervals[:, 1].astype(np.float64),
        }
    )


def interval_mask(
    time: NDArray[np.float64] | pd.Index, intervals: pd.DataFrame
) -> NDArray[np.bool_]:
    """True at each time inside any interval, endpoints included.

    Parameters
    ----------
    time : array-like, shape (n_time,)
        Seconds, in any order.
    intervals : pd.DataFrame
        ``start_time`` and ``end_time`` (seconds), closed; any order, and they
        may overlap.

    Returns
    -------
    ndarray of bool, shape (n_time,)

    Raises
    ------
    ValueError
        If an interval bound is NaN or a start is after its end.
    """
    values = np.asarray(time, dtype=np.float64)
    starts = intervals["start_time"].to_numpy(dtype=np.float64)
    _require_bounds(
        starts, intervals["end_time"].to_numpy(dtype=np.float64), "interval_mask"
    )
    if starts.size == 0:
        return np.zeros(values.shape, dtype=bool)
    order = np.argsort(starts, kind="stable")
    starts = starts[order]
    reach = np.maximum.accumulate(
        intervals["end_time"].to_numpy(dtype=np.float64)[order]
    )
    # the last interval starting at or before each time; inside one iff the
    # furthest end among intervals starting by then reaches the time
    last = np.searchsorted(starts, values, side="right") - 1
    mask: NDArray[np.bool_] = (last >= 0) & (values <= reach[np.maximum(last, 0)])
    return mask


def subtract_intervals(
    stretches: NDArray[np.float64], intervals: pd.DataFrame
) -> NDArray[np.float64]:
    """Closed stretches less closed intervals, as closed stretches.

    A piece's bounds next to a removed interval are the floats just outside
    it (``np.nextafter``), so a time lies in the result exactly when it lies in
    a stretch and in no interval, endpoints included.

    Parameters
    ----------
    stretches : ndarray, shape (n_stretches, 2)
        ``[start, end]`` rows, seconds, closed.
    intervals : pd.DataFrame
        ``start_time`` and ``end_time`` (seconds), closed.

    Returns
    -------
    ndarray, shape (n_pieces, 2)
        The remaining pieces, stretch by stretch, in time order within each.

    Raises
    ------
    ValueError
        If a stretch or interval bound is NaN or a start is after its end.
    """
    stretches = np.asarray(stretches, dtype=np.float64).reshape(-1, 2)
    _require_bounds(stretches[:, 0], stretches[:, 1], "subtract_intervals stretches")
    removed = merge_intervals(
        intervals["start_time"].to_numpy(dtype=np.float64),
        intervals["end_time"].to_numpy(dtype=np.float64),
    )
    if removed.size == 0:
        return stretches.copy()
    before = np.nextafter(removed[:, 0], -np.inf)
    after = np.nextafter(removed[:, 1], np.inf)
    pieces = []
    for start, end in stretches:
        hit = (removed[:, 0] <= end) & (removed[:, 1] >= start)
        lows = np.r_[start, after[hit]]
        highs = np.r_[before[hit], end]
        keep = lows <= highs
        pieces.append(np.column_stack([lows[keep], highs[keep]]))
    return np.concatenate(pieces).reshape(-1, 2)


def masked_seconds(intervals: pd.DataFrame) -> float:
    """Total length of `intervals` in seconds, overlaps counted once.

    Raises ``ValueError`` on a NaN bound or a start after its end
    (:func:`merge_intervals`).
    """
    merged = merge_intervals(
        intervals["start_time"].to_numpy(dtype=np.float64),
        intervals["end_time"].to_numpy(dtype=np.float64),
    )
    return float(np.sum(merged[:, 1] - merged[:, 0]))


def pad_intervals(intervals: pd.DataFrame, distance_s: float) -> pd.DataFrame:
    """The intervals widened by one distance on each side, merged.

    Excluded time is the artifact mask widened here by
    ``event_definition.eligibility.ARTIFACT_REACH_S``; the PFC fit widens excluded time
    by its peri-event basis's half-width. A history window, which looks back
    from a bin, widens the bin instead (``per_bin_design``).

    Parameters
    ----------
    intervals : pd.DataFrame
        ``start_time`` and ``end_time`` (seconds), closed; any order.
    distance_s : float
        Seconds added before each start and after each end.

    Returns
    -------
    pd.DataFrame
        Columns ``start_time`` and ``end_time`` (float64, seconds), closed,
        disjoint and in time order.

    Raises
    ------
    ValueError
        If the distance is negative or not finite, or an interval bound is NaN
        or a start is after its end.
    """
    if not (np.isfinite(distance_s) and distance_s >= 0):
        raise ValueError(f"a padding distance must be finite and >= 0: {distance_s}")
    starts = intervals["start_time"].to_numpy(dtype=np.float64) - distance_s
    ends = intervals["end_time"].to_numpy(dtype=np.float64) + distance_s
    return as_frame(merge_intervals(starts, ends))


def overlaps_intervals(
    starts: NDArray[np.float64], ends: NDArray[np.float64], intervals: pd.DataFrame
) -> NDArray[np.bool_]:
    """True for each closed ``[start, end]`` sharing any time with an interval.

    Both sides are closed, so a stretch that only touches an interval at one
    instant overlaps it.

    Parameters
    ----------
    starts, ends : ndarray, shape (n,)
        Seconds; ``ends >= starts``.
    intervals : pd.DataFrame
        ``start_time`` and ``end_time`` (seconds), closed; any order.

    Returns
    -------
    ndarray of bool, shape (n,)

    Raises
    ------
    ValueError
        If a bound (of a stretch or an interval) is NaN or a start is after
        its end.
    """
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    _require_bounds(starts, ends, "overlaps_intervals stretches")
    merged = merge_intervals(
        intervals["start_time"].to_numpy(dtype=np.float64),
        intervals["end_time"].to_numpy(dtype=np.float64),
    )
    if merged.size == 0:
        return np.zeros(starts.shape, dtype=bool)
    # the last merged interval starting at or before each end; the merged
    # intervals are disjoint and sorted, so an earlier one ends earlier still
    last = np.searchsorted(merged[:, 0], ends, side="right") - 1
    overlaps: NDArray[np.bool_] = (last >= 0) & (
        merged[np.maximum(last, 0), 1] >= starts
    )
    return overlaps


def covered_seconds(
    starts: NDArray[np.float64], ends: NDArray[np.float64], intervals: pd.DataFrame
) -> NDArray[np.float64]:
    """Seconds of each ``[start, end]`` that the intervals cover.

    Overlapping intervals are counted once. Exactly 0.0 where no interval
    reaches a stretch, so ``span - covered_seconds(...)`` is the span itself
    when nothing is masked.

    Parameters
    ----------
    starts, ends : ndarray, shape (n,)
        Seconds; ``ends >= starts``.
    intervals : pd.DataFrame
        ``start_time`` and ``end_time`` (seconds), closed; any order.

    Returns
    -------
    ndarray of float64, shape (n,)

    Raises
    ------
    ValueError
        If a bound (of a stretch or an interval) is NaN or a start is after
        its end.
    """
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    _require_bounds(starts, ends, "covered_seconds stretches")
    merged = merge_intervals(
        intervals["start_time"].to_numpy(dtype=np.float64),
        intervals["end_time"].to_numpy(dtype=np.float64),
    )
    if merged.size == 0:
        return np.zeros(starts.shape, dtype=np.float64)
    lengths = merged[:, 1] - merged[:, 0]
    before = np.concatenate([[0.0], np.cumsum(lengths)])

    def _covered_until(time: NDArray[np.float64]) -> NDArray[np.float64]:
        # masked seconds in (-inf, time]: every interval starting by then, the
        # last of them only up to `time`
        n_started = np.searchsorted(merged[:, 0], time, side="right")
        last = np.maximum(n_started - 1, 0)
        partial = np.clip(time - merged[last, 0], 0.0, lengths[last])
        until: NDArray[np.float64] = np.where(
            n_started > 0, before[last] + partial, 0.0
        )
        return until

    covered: NDArray[np.float64] = _covered_until(ends) - _covered_until(starts)
    return np.where(overlaps_intervals(starts, ends, intervals), covered, 0.0)


# ---------------------------------------------------------------------------
# The open overlap test, runs of a mask, spike binning.
# ---------------------------------------------------------------------------


def overlaps_open(
    start_times: NDArray, end_times: NDArray, starts: NDArray, ends: NDArray
) -> NDArray[np.bool_]:
    """Whether the open test holds, for each query, against any window:
    ``window_start < end_time`` and ``window_end > start_time``.

    Abutting a window exactly is not an overlap. There is no duration check,
    so a zero-length window, or a zero-length ``[start_time, end_time]``,
    strictly inside the other overlaps it (strict overlap would say it overlaps
    nothing; see the conventions table in this module's docstring). The PFC
    fit calls this its half-open test.

    Parameters
    ----------
    start_times, end_times : NDArray, shape (n_queries,)
        Query bounds in seconds.
    starts, ends : NDArray, shape (n_windows,)
        Ascending, non-overlapping window bounds (the columns of
        :func:`merge_intervals`). ``ends`` is then ascending too, so the only
        window that can reach past a query's start among those starting before
        its end is the last of them.

    Returns
    -------
    NDArray of bool, shape (n_queries,)
    """
    start_times = np.asarray(start_times, dtype=np.float64)
    end_times = np.asarray(end_times, dtype=np.float64)
    if len(starts) == 0:
        return np.zeros(start_times.shape, dtype=bool)
    n_starting_before = np.searchsorted(starts, end_times, side="left")
    last_end = np.asarray(ends)[np.maximum(n_starting_before - 1, 0)]
    overlaps: NDArray[np.bool_] = (n_starting_before > 0) & (last_end > start_times)
    return overlaps


def contiguous_true_runs(mask: NDArray) -> NDArray:
    """Half-open ``[start, stop)`` sample bounds of each contiguous True run.

    Parameters
    ----------
    mask : NDArray of bool, shape (n_time,)

    Returns
    -------
    NDArray of int, shape (n_runs, 2)
        One row per run; ``mask[start:stop]`` is all True and is bounded by
        False (or by the array's ends) on both sides.
    """
    flat = np.asarray(mask, dtype=bool)
    if flat.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    padded = np.concatenate([[False], flat, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return edges.reshape(-1, 2).astype(np.int64)


def sample_step(time: NDArray) -> float:
    """The sample step of a time grid: the median of its successive differences.

    Parameters
    ----------
    time : NDArray, shape (n_time,)
        Sample times in seconds, at least two.

    Returns
    -------
    float
        Seconds. The median, so a gap in the grid does not move it.
    """
    return float(np.median(np.diff(time)))


GAP_STEP_RATIO = 1.5
"""A timestamp step over this many sample steps is a gap (``ripple_detection``'s
rule)."""


def max_step_s(sample_step_s: float) -> float:
    """The longest timestamp step that is not a gap in a grid of this step.

    Parameters
    ----------
    sample_step_s : float
        The grid's sample step in seconds: :func:`sample_step`, the median of
        some steps, or a nominal step.

    Returns
    -------
    float
        ``GAP_STEP_RATIO * sample_step_s`` seconds. A step longer than this is
        a gap.
    """
    return GAP_STEP_RATIO * sample_step_s


def min_gap_to_intervals(
    starts: np.ndarray,
    ends: np.ndarray,
    other_starts: np.ndarray,
    other_ends: np.ndarray,
) -> np.ndarray:
    """Minimum temporal gap from each [starts, ends] interval to any other interval.

    0 where they overlap (or touch), inf when there is no other interval.

    Parameters
    ----------
    starts, ends : NDArray, shape (n_intervals,)
    other_starts, other_ends : NDArray, shape (n_other,)

    Returns
    -------
    NDArray, shape (n_intervals,)
        Seconds from each interval to the nearest other one.
    """
    if len(other_starts) == 0:
        return np.full(len(starts), np.inf)
    s = starts[:, None]
    e = ends[:, None]
    os = other_starts[None, :]
    oe = other_ends[None, :]
    # gap = max(0, os - e, s - oe) elementwise; 0 if overlapping
    gap = np.maximum.reduce([np.zeros_like(s + os), os - e, s - oe])
    return gap.min(axis=1)


def decoder_time_bin(spike_times: NDArray, time: NDArray) -> NDArray[np.intp]:
    """The decode bin of each spike, as the decoder assigns it.

    ``np.digitize(spike_times, time[1:-1])``, the rule of ``non_local_detector``
    (``likelihoods.common.get_spikecount_per_time_bin``,
    ``get_spike_time_bin_ind``). With ``n = time.size`` and ``time`` ascending,
    a spike at ``t`` goes to bin

    - ``0`` when ``t < time[1]``, so also every spike before ``time[0]``;
    - ``k`` when ``time[k] <= t < time[k + 1]``, for ``1 <= k <= n - 3``: a spike
      exactly on a bin time ``time[k]`` goes to bin ``k``, the bin that starts
      there;
    - ``n - 2`` when ``t >= time[n - 2]``, so also every spike after
      ``time[-1]`` and a NaN.

    Bin ``n - 1`` is never assigned. Nothing is dropped here: a caller that
    keeps only the spikes in ``[time[0], time[-1]]``, as the decoder does,
    selects them first.

    Parameters
    ----------
    spike_times : NDArray, shape (n_spikes,)
        Spike times in seconds.
    time : NDArray, shape (n_time,)
        The decode's time grid in seconds, ascending, at least two samples.

    Returns
    -------
    NDArray of intp, shape (n_spikes,)
        Each spike's bin, in ``0..n_time - 2``.
    """
    return np.digitize(spike_times, time[1:-1])


def get_spikecount_per_time_bin(spike_times: NDArray, time: NDArray) -> NDArray:
    """Bin spike times into time bins.

    Parameters
    ----------
    spike_times : NDArray, shape (n_spikes,)
        Spike times in seconds.
    time : NDArray, shape (n_time_bins,)
        Time bin starts (left edges) in seconds — i.e. each ``time[i]``
        is the start of bin ``i``, not its center. Matches the
        sample-bin-start convention used by the data loader's
        multiunit pipeline. Spike ``s`` is assigned to bin ``i`` where
        ``time[i] <= s < time[i+1]``; spikes at exactly ``time[-1]``
        fall in the last bin.

    Returns
    -------
    NDArray, shape (n_time_bins,)
        Spike counts per time bin.

    """
    spike_times = spike_times[
        np.logical_and(spike_times >= time[0], spike_times <= time[-1])
    ]
    # ``np.digitize(s, time[1:])`` produces indices ``0..N-1`` against
    # a ``minlength=N`` bincount, so all output bins are reachable.
    # The previous ``time[1:-1]`` form maxed out at index ``N-2``,
    # leaving the final bin unreachable: e.g. with
    # ``time=[0,1,2,3]``, spikes at 2.5 and 3.0 both landed in bin 2
    # and bin 3 stayed at zero.
    return np.bincount(
        np.digitize(spike_times, time[1:]),
        minlength=time.shape[0],
    )
