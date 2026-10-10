"""The artifact mask: the detector's cores, padded from one table.

The detector stores unpadded cores (``detect.detect_artifacts``). One table,
:data:`PADDING_S`, pads them, and :func:`artifact_intervals` gives the one mask
every part of the project uses: the SWR detectors' normalization and event
drops, MUA detection, the decoders (``is_missing`` and the encoding
exclusion), the Widloski runner and every analysis. Artifact time is
unobserved there.

Intervals are closed, ``[start_time, end_time]`` in seconds, the project's
convention: a time on an endpoint is masked. :func:`pad_intervals` is the one
way a stage widens the mask, by the reach of what it reads (a decoded event's
0.3 s, a history window, a peri-event window). Design note:
``docs/design/2026-10-09-artifact-detection.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = [
    "PADDING_S",
    "artifact_intervals",
    "core_class",
    "covered_seconds",
    "interval_mask",
    "masked_seconds",
    "merge_intervals",
    "overlaps_intervals",
    "pad_intervals",
    "subtract_intervals",
]

INTERVAL_COLUMNS = ["start_time", "end_time"]

LFP_SIGNATURES = frozenset({"ref", "ca1"})
"""T signatures read from the LFP: a step on the reference or on CA1."""
SPIKE_SIGNATURES = frozenset({"spk"})
"""T signatures read from spikes alone: a spike line."""
S_SIGNATURE = "S"

PADDING_S: Mapping[str, float] = MappingProxyType(
    {
        # Noise spikes on the affected tetrodes leak up to 1 s past some block
        # edges (detector_report.md §5.1 s_pad, validation/S_edge_spike_rates.csv).
        "S": 1.0,
        # The stored ripple FIR rings after a step: its half-length is 0.267 s,
        # and the ringing above 2 robust z around 425 cores has its 99th
        # percentile at 316 ms; 0.3 s covers 96.8-100 % of held-out cores
        # (followup/justification_report.md §1, c2_padding_heldout.csv). The
        # burst of out-of-distribution spikes after a step reaches +269 ms
        # (detector_report.md §2.3).
        "T_lfp": 0.3,
        # 98.8 % of the out-of-distribution marks after a spike line lie within
        # 100 ms (99th percentile 53 ms; justification_report.md §1).
        "T_spike": 0.1,
    }
)
"""Seconds of padding on each side of a core, by its row: ``S``, ``T_lfp`` (a T
core with an LFP signature, ``ref`` or ``ca1``) and ``T_spike`` (a T core with
the spike-line signature alone). No padding follows a step beyond 0.3 s: R8
reruns the headline results with 1 s masked after each step instead."""


def core_class(cores: pd.DataFrame) -> NDArray[np.str_]:
    """Each core's row of the padding table: ``S``, ``T_lfp`` or ``T_spike``.

    Parameters
    ----------
    cores : pd.DataFrame
        With a ``signatures`` column: ``"S"``, or T signatures joined by ``+``.

    Returns
    -------
    ndarray of str, shape (n_cores,)

    Raises
    ------
    ValueError
        If a signature is not one this table knows.
    """
    rows = []
    for signatures in cores["signatures"].astype(str):
        names = set(signatures.split("+"))
        if names == {S_SIGNATURE}:
            rows.append("S")
        elif names <= LFP_SIGNATURES | SPIKE_SIGNATURES and names & LFP_SIGNATURES:
            rows.append("T_lfp")
        elif names == set(SPIKE_SIGNATURES):
            rows.append("T_spike")
        else:
            raise ValueError(
                f"core signatures {signatures!r} have no row in the padding table"
            )
    return np.asarray(rows, dtype=str)


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
    """
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
    if starts.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    order = np.argsort(starts, kind="stable")
    starts, ends = starts[order], ends[order]
    reach = np.maximum.accumulate(ends)
    # closed: a start on the running end touches it, so it merges
    first = np.flatnonzero(np.r_[True, starts[1:] > reach[:-1]])
    return np.column_stack([starts[first], np.maximum.reduceat(ends, first)])


def _as_frame(intervals: NDArray[np.float64]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start_time": intervals[:, 0].astype(np.float64),
            "end_time": intervals[:, 1].astype(np.float64),
        }
    )


def artifact_intervals(cores: pd.DataFrame, span: tuple[float, float]) -> pd.DataFrame:
    """The artifact mask: the cores padded by :data:`PADDING_S`, merged and
    clipped.

    Parameters
    ----------
    cores : pd.DataFrame
        One epoch's cores, with ``signatures``, ``core_start`` and ``core_end``
        (seconds), as ``detect_artifacts`` returns them.
    span : tuple of float
        The epoch's first and last observed time (seconds); the intervals are
        clipped to it.

    Returns
    -------
    pd.DataFrame
        Columns ``start_time`` and ``end_time`` (float64, seconds), closed,
        disjoint and in time order; no rows when there are no cores.

    Raises
    ------
    ValueError
        If a core's signature has no row in :data:`PADDING_S`.
    """
    pad = np.array([PADDING_S[row] for row in core_class(cores)], dtype=np.float64)
    starts = np.maximum(cores["core_start"].to_numpy(dtype=np.float64) - pad, span[0])
    ends = np.minimum(cores["core_end"].to_numpy(dtype=np.float64) + pad, span[1])
    inside = starts <= ends
    return _as_frame(merge_intervals(starts[inside], ends[inside]))


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
    """
    values = np.asarray(time, dtype=np.float64)
    starts = intervals["start_time"].to_numpy(dtype=np.float64)
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
    """
    stretches = np.asarray(stretches, dtype=np.float64).reshape(-1, 2)
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
    """Total length of `intervals` in seconds, overlaps counted once."""
    merged = merge_intervals(
        intervals["start_time"].to_numpy(dtype=np.float64),
        intervals["end_time"].to_numpy(dtype=np.float64),
    )
    return float(np.sum(merged[:, 1] - merged[:, 0]))


def pad_intervals(
    intervals: pd.DataFrame, before_s: float, after_s: float | None = None
) -> pd.DataFrame:
    """The intervals widened by a distance on each side, merged.

    Every stage that must keep clear of the mask by more than its own time
    widens it here: a decoded event within 0.3 s of the mask, a bin whose
    history window looks back into it, a peri-event window that reaches it.

    Parameters
    ----------
    intervals : pd.DataFrame
        ``start_time`` and ``end_time`` (seconds), closed; any order.
    before_s : float
        Seconds added before each start.
    after_s : float, optional
        Seconds added after each end; `before_s` when None.

    Returns
    -------
    pd.DataFrame
        Columns ``start_time`` and ``end_time`` (float64, seconds), closed,
        disjoint and in time order.

    Raises
    ------
    ValueError
        If a distance is negative or not finite.
    """
    after_s = before_s if after_s is None else after_s
    for distance in (before_s, after_s):
        if not (np.isfinite(distance) and distance >= 0):
            raise ValueError(f"a padding distance must be finite and >= 0: {distance}")
    starts = intervals["start_time"].to_numpy(dtype=np.float64) - before_s
    ends = intervals["end_time"].to_numpy(dtype=np.float64) + after_s
    return _as_frame(merge_intervals(starts, ends))


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
    """
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
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
    """
    starts = np.asarray(starts, dtype=np.float64)
    ends = np.asarray(ends, dtype=np.float64)
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
