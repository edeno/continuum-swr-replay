"""Each consumer's artifact intervals, padded from one table.

The detector stores unpadded cores (``detect.detect_artifacts``). Each consumer
pads them from :data:`CONSUMER_PADDING` and nothing else, so one set of cores
gives every mask in the project:

- ``swr``: the SWR detectors' normalization and event dropping (they read
  the LFP);
- ``decoder``: the decoders' ``is_missing``, the encoding exclusion and MUA
  detection (they read spikes);
- ``analysis``: event eligibility and rate exposure, everything downstream.

Intervals are closed, ``[start_time, end_time]`` in seconds, the project's
convention: a time on an endpoint is masked. Design note:
``docs/design/2026-10-09-artifact-detection.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = [
    "CONSUMER_PADDING",
    "Consumer",
    "artifact_intervals",
    "core_class",
    "interval_mask",
    "masked_seconds",
    "merge_intervals",
    "subtract_intervals",
]

INTERVAL_COLUMNS = ["start_time", "end_time"]

LFP_SIGNATURES = frozenset({"ref", "ca1"})
"""T signatures read from the LFP: a step on the reference or on CA1."""
SPIKE_SIGNATURES = frozenset({"spk"})
"""T signatures read from spikes alone: a spike line."""
S_SIGNATURE = "S"


class Consumer(StrEnum):
    """Who reads a mask, which sets its padding."""

    SWR = "swr"
    """SWR detectors (Kay, Karlsson, Yu, Zugaro): normalization and event
    dropping. They read the LFP."""
    DECODER = "decoder"
    """Decoders (``is_missing``, encoding exclusion) and MUA detection. They read
    spikes."""
    ANALYSIS = "analysis"
    """Event eligibility for every cohort, and rate exposure."""


def _frozen(table: Mapping[Consumer, Mapping[str, tuple[float, float]]]) -> Mapping:
    return MappingProxyType(
        {key: MappingProxyType(dict(row)) for key, row in table.items()}
    )


CONSUMER_PADDING: Mapping[Consumer, Mapping[str, tuple[float, float]]] = _frozen(
    {
        # S 0 s: with gap fill, 0 / 0.5 / 1 s give identical SD / robust SD in all
        # 11 S epochs, and 0.5 s already masks a visual SWR (detector_report.md
        # §5.1 s_pad). T ±0.3 s: consensus ringing, 99th pct 191 ms, max 335 ms;
        # FIR half-length 0.267 s (§5.1 t_pad_lfp). Spike-only ±0.1 s: 99.3 % of
        # excess OOD spikes within ±100 ms (§5.1 t_pad_spk).
        Consumer.SWR: {"S": (0.0, 0.0), "T_lfp": (0.3, 0.3), "T_spike": (0.1, 0.1)},
        # S ±1 s: noise spikes on the affected tetrodes leak up to 1 s past some
        # block edges (§5.1 s_pad decoder, validation/S_edge_spike_rates.csv).
        # T ±0.3 s: the post-step OOD discharge tail reaches +269 ms (§2.3, §6.3).
        Consumer.DECODER: {
            "S": (1.0, 1.0),
            "T_lfp": (0.3, 0.3),
            "T_spike": (0.1, 0.1),
        },
        # The decoder mask plus the 1 s after each LFP step: ripple-band power
        # stays below its random-time level for at least 1 s after a step (§8
        # item 4; Phase C ruling "post-step second").
        Consumer.ANALYSIS: {
            "S": (1.0, 1.0),
            "T_lfp": (0.3, 1.3),
            "T_spike": (0.1, 0.1),
        },
    }
)
"""Seconds of padding ``(before, after)`` a core gets, by consumer and by its
row: ``S``, ``T_lfp`` (a T core with an LFP signature, ``ref`` or ``ca1``) and
``T_spike`` (a T core with the spike-line signature alone)."""


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


def artifact_intervals(
    cores: pd.DataFrame, consumer: Consumer | str, span: tuple[float, float]
) -> pd.DataFrame:
    """One consumer's artifact intervals: padded cores, merged and clipped.

    Parameters
    ----------
    cores : pd.DataFrame
        One epoch's cores, with ``signatures``, ``core_start`` and ``core_end``
        (seconds), as ``detect_artifacts`` returns them.
    consumer : Consumer or str
        ``"swr"``, ``"decoder"`` or ``"analysis"``.
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
        If `consumer` is not in :data:`CONSUMER_PADDING`, or a core's signature
        has no row in it.
    """
    try:
        padding = CONSUMER_PADDING[Consumer(consumer)]
    except ValueError:
        raise ValueError(
            f"unknown artifact consumer {consumer!r}; one of "
            f"{[c.value for c in Consumer]}"
        ) from None
    rows = core_class(cores)
    before = np.array([padding[row][0] for row in rows], dtype=np.float64)
    after = np.array([padding[row][1] for row in rows], dtype=np.float64)
    starts = np.maximum(
        cores["core_start"].to_numpy(dtype=np.float64) - before, span[0]
    )
    ends = np.minimum(cores["core_end"].to_numpy(dtype=np.float64) + after, span[1])
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
