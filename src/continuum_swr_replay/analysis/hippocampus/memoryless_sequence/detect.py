"""Turn per-window statistics into candidate sequences.

The order of the steps follows the original (spec 02, data-flow steps 4-7):
gate each window on its speed, then its posterior spread and its jump to the
next window; collect contiguous runs of surviving windows; merge runs that are
close in both space and time; drop runs shorter than the minimum duration; and
keep the sequences that are long enough and travel far enough to be replay.

Taking the speed gate before the jumps is load-bearing. A window whose
successor was rejected on speed has no forward jump and is itself rejected, so
a sequence ends one window before the first speed-rejected window. Computing
jumps first would let it run one window further, which is enough to move a
candidate across the duration filter. :func:`window_decisions` applies the four
steps in the one correct order so a caller cannot reorder them.

Every threshold is a strict inequality, as in the reference. Ties are
systematic rather than rare here: a 20 cm jump is exactly ten 2 cm bins, and
100 ms and 50 ms are whole multiples of both the original's 5 ms stride and
this project's 4 ms one. The comparisons therefore treat a value within a
relative tolerance of its threshold as equal to it, so float representation
does not decide the outcome.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.statistics import (
    NO_CENTER,
    dispersion_cm,
    forward_jumps_cm,
    window_center_and_spread,
)

__all__ = [
    "CandidateSequences",
    "SequenceAssembly",
    "WindowDecisions",
    "WindowGates",
    "apply_speed_gate",
    "assemble_sequences",
    "contiguous_runs",
    "detect_candidate_sequences",
    "gate_windows",
    "merge_runs",
    "replay_mask",
    "sequence_dispersion_cm",
    "sequence_duration_s",
    "window_decisions",
]

_THRESHOLD_TOLERANCE = 1e-9
"""Relative tolerance at which a value counts as equal to a threshold."""


def _is_below(value: NDArray, threshold: float) -> NDArray:
    """Strictly below ``threshold``, treating a value within tolerance as equal.

    A non-finite value is never below a threshold.
    """
    value = np.asarray(value, dtype=np.float64)
    return (value < threshold) & ~np.isclose(
        value, threshold, rtol=_THRESHOLD_TOLERANCE, atol=0.0
    )


def _is_above(value: NDArray, threshold: float) -> NDArray:
    """Strictly above ``threshold``, treating a value within tolerance as equal."""
    value = np.asarray(value, dtype=np.float64)
    return (value > threshold) & ~np.isclose(
        value, threshold, rtol=_THRESHOLD_TOLERANCE, atol=0.0
    )


@dataclass(frozen=True)
class WindowGates:
    """Which criterion each window met, kept separately so rejections explain themselves.

    Attributes
    ----------
    has_center : ndarray of bool, shape (n_windows,)
        The posterior was usable and the speed gate did not blank it.
    spread_below_threshold : ndarray of bool, shape (n_windows,)
    jump_below_threshold : ndarray of bool, shape (n_windows,)
        False where the forward jump is undefined, which happens when the next
        window has no center.
    kept : ndarray of bool, shape (n_windows,)
        All three together.

    """

    has_center: NDArray
    spread_below_threshold: NDArray
    jump_below_threshold: NDArray
    kept: NDArray


@dataclass(frozen=True)
class WindowDecisions:
    """Per-window statistics and gates in the order the detector computes them."""

    center_index: NDArray
    spread_cm: NDArray
    jump_cm: NDArray
    gates: WindowGates


@dataclass(frozen=True)
class SequenceAssembly:
    """Candidate sequences, with the intermediate stages kept for inspection.

    Attributes
    ----------
    runs : ndarray of int, shape (n_sequences, 2)
        Inclusive first and last window index of each surviving sequence.
    pieces : list of ndarray
        For each sequence, the indices into ``pre_merge_runs`` that it absorbed.
        A sequence with more than one piece spans a merged gap.
    pre_merge_runs : ndarray of int, shape (n_runs, 2)
        Runs before merging, after single-window runs were dropped.
    kept : ndarray of bool, shape (n_windows,)
        The window keep mask the sequences were assembled from. Windows inside
        a merged gap stay False here; they are not decoded positions.

    """

    runs: NDArray
    pieces: list[NDArray]
    pre_merge_runs: NDArray
    kept: NDArray


def apply_speed_gate(
    center_index: NDArray,
    window_speed_cm_per_s: NDArray,
    speed_threshold_cm_per_s: float,
) -> NDArray:
    """Blank the center of every window the animal was not still for.

    The original decodes only speed-eligible windows; blanking the center is
    the equivalent, and it must happen before jumps are computed.

    Parameters
    ----------
    center_index : ndarray of int, shape (n_windows,)
    window_speed_cm_per_s : ndarray, shape (n_windows,)
        Speed interpolated at each window's midpoint. An unknown speed does not
        establish eligibility, so it blanks the center.
    speed_threshold_cm_per_s : float

    Returns
    -------
    center_index : ndarray of int, shape (n_windows,)
        A copy, with ``NO_CENTER`` wherever the window is not eligible.

    """
    center = np.array(center_index, dtype=int)
    speed = np.asarray(window_speed_cm_per_s, dtype=np.float64)
    if speed.shape != center.shape:
        raise ValueError(
            f"center_index has shape {center.shape} but the speed has {speed.shape}."
        )
    center[~_is_below(speed, speed_threshold_cm_per_s)] = NO_CENTER
    return center


def gate_windows(
    center_index: NDArray,
    spread_cm: NDArray,
    jump_cm: NDArray,
    config: MemorylessSequenceConfig,
) -> WindowGates:
    """Apply the per-window criteria, recording each one separately.

    Parameters
    ----------
    center_index : ndarray of int, shape (n_windows,)
        Already gated on speed.
    spread_cm, jump_cm : ndarray, shape (n_windows,)
    config : MemorylessSequenceConfig

    Returns
    -------
    gates : WindowGates

    """
    has_center = np.asarray(center_index, dtype=int) != NO_CENTER
    spread_ok = _is_below(spread_cm, config.spread_threshold_cm)
    jump_ok = _is_below(jump_cm, config.jump_threshold_cm)
    return WindowGates(
        has_center=has_center,
        spread_below_threshold=spread_ok,
        jump_below_threshold=jump_ok,
        kept=has_center & spread_ok & jump_ok,
    )


def window_decisions(
    posterior: NDArray,
    bin_distance_cm: NDArray,
    window_speed_cm_per_s: NDArray,
    config: MemorylessSequenceConfig,
) -> WindowDecisions:
    """Per-window statistics and gates, in the one order that is correct.

    Parameters
    ----------
    posterior : ndarray, shape (n_windows, n_bins)
        Over real place bins, rows summing to one.
    bin_distance_cm : ndarray, shape (n_bins, n_bins)
    window_speed_cm_per_s : ndarray, shape (n_windows,)
        Speed at each window's midpoint.
    config : MemorylessSequenceConfig

    Returns
    -------
    decisions : WindowDecisions

    """
    center, spread = window_center_and_spread(posterior, bin_distance_cm)
    center = apply_speed_gate(
        center, window_speed_cm_per_s, config.speed_threshold_cm_per_s
    )
    jump = forward_jumps_cm(center, bin_distance_cm)
    return WindowDecisions(
        center_index=center,
        spread_cm=spread,
        jump_cm=jump,
        gates=gate_windows(center, spread, jump, config),
    )


def contiguous_runs(kept: NDArray) -> NDArray:
    """Unbroken stretches of kept windows.

    Parameters
    ----------
    kept : ndarray of bool, shape (n_windows,)

    Returns
    -------
    runs : ndarray of int, shape (n_runs, 2)
        Inclusive first and last window index of each run.

    """
    kept = np.asarray(kept, dtype=bool)
    edges = np.diff(np.concatenate(([False], kept, [False])).astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1) - 1
    return np.column_stack([starts, stops]).astype(int)


def merge_runs(
    runs: NDArray,
    center_index: NDArray,
    window_time_s: NDArray,
    bin_distance_cm: NDArray,
    config: MemorylessSequenceConfig,
) -> tuple[NDArray, list[NDArray]]:
    """Join neighboring runs separated by a short gap in both space and time.

    The gap is measured between the facing endpoints: the last window of one
    run and the first window of the next. Merging is transitive, so a chain of
    close runs becomes one sequence.

    Parameters
    ----------
    runs : ndarray of int, shape (n_runs, 2)
        Inclusive bounds, in order.
    center_index : ndarray of int, shape (n_windows,)
    window_time_s : ndarray, shape (n_windows,)
        Window midpoints.
    bin_distance_cm : ndarray, shape (n_bins, n_bins)
    config : MemorylessSequenceConfig

    Returns
    -------
    merged : ndarray of int, shape (n_merged, 2)
    pieces : list of ndarray
        For each merged run, the indices into ``runs`` that it absorbed.

    """
    runs = np.asarray(runs, dtype=int).reshape(-1, 2)
    if len(runs) == 0:
        return runs.copy(), []
    if len(runs) == 1:
        return runs.copy(), [np.array([0])]

    center = np.asarray(center_index, dtype=int)
    time_s = np.asarray(window_time_s, dtype=np.float64)
    distance = np.asarray(bin_distance_cm, dtype=np.float64)

    before, after = runs[:-1, 1], runs[1:, 0]
    gap_cm = distance[center[before], center[after]]
    gap_s = time_s[after] - time_s[before]
    joins = _is_below(gap_cm, config.merge_gap_cm) & _is_below(
        gap_s, config.merge_gap_s
    )

    # a run starts a new sequence when it did not join the run before it
    opens = np.concatenate(([True], ~joins))
    closes = np.concatenate((~joins, [True]))
    first, last = np.flatnonzero(opens), np.flatnonzero(closes)
    merged = np.column_stack([runs[first, 0], runs[last, 1]])
    pieces = [
        np.arange(start, stop + 1) for start, stop in zip(first, last, strict=True)
    ]
    return merged, pieces


def assemble_sequences(
    kept: NDArray,
    center_index: NDArray,
    window_time_s: NDArray,
    bin_distance_cm: NDArray,
    config: MemorylessSequenceConfig,
) -> SequenceAssembly:
    """Collect kept windows into candidate sequences.

    Runs of a single window are dropped first, as in the original: their index
    length is zero. Surviving runs are merged, and then the duration filter
    drops those whose index length is below the stride-derived minimum. Length
    is an index difference throughout, so a run of index length ``n`` holds
    ``n + 1`` windows and spans ``n`` strides.

    Parameters
    ----------
    kept : ndarray of bool, shape (n_windows,)
    center_index : ndarray of int, shape (n_windows,)
    window_time_s : ndarray, shape (n_windows,)
    bin_distance_cm : ndarray, shape (n_bins, n_bins)
    config : MemorylessSequenceConfig

    Returns
    -------
    assembly : SequenceAssembly

    """
    kept = np.asarray(kept, dtype=bool)
    runs = contiguous_runs(kept)
    runs = runs[runs[:, 1] > runs[:, 0]]

    merged, pieces = merge_runs(
        runs, center_index, window_time_s, bin_distance_cm, config
    )
    long_enough = (merged[:, 1] - merged[:, 0]) >= config.minimum_run_index_length
    return SequenceAssembly(
        runs=merged[long_enough],
        pieces=[piece for piece, keep in zip(pieces, long_enough, strict=True) if keep],
        pre_merge_runs=runs,
        kept=kept,
    )


def sequence_duration_s(runs: NDArray, window_time_s: NDArray) -> NDArray:
    """Time from each sequence's first window midpoint to its last.

    The 80 ms window width is not added: the decoding support is a separate
    quantity from the event interval (spec 02, "Window and event timing").

    Parameters
    ----------
    runs : ndarray of int, shape (n_sequences, 2)
    window_time_s : ndarray, shape (n_windows,)

    Returns
    -------
    duration_s : ndarray, shape (n_sequences,)

    """
    runs = np.asarray(runs, dtype=int).reshape(-1, 2)
    time_s = np.asarray(window_time_s, dtype=np.float64)
    return time_s[runs[:, 1]] - time_s[runs[:, 0]]


def sequence_dispersion_cm(
    runs: NDArray,
    kept: NDArray,
    center_index: NDArray,
    bin_distance_cm: NDArray,
) -> NDArray:
    """How far each sequence's centers spread around their own center.

    Only kept windows contribute, matching the original's NaN-masked
    trajectory: windows inside a merged gap were rejected and are not decoded
    positions.

    Parameters
    ----------
    runs : ndarray of int, shape (n_sequences, 2)
    kept : ndarray of bool, shape (n_windows,)
    center_index : ndarray of int, shape (n_windows,)
    bin_distance_cm : ndarray, shape (n_bins, n_bins)

    Returns
    -------
    dispersion_cm : ndarray, shape (n_sequences,)

    """
    runs = np.asarray(runs, dtype=int).reshape(-1, 2)
    kept = np.asarray(kept, dtype=bool)
    center = np.asarray(center_index, dtype=int)

    return np.array(
        [
            dispersion_cm(
                np.where(kept[start : stop + 1], center[start : stop + 1], NO_CENTER),
                bin_distance_cm,
            )
            for start, stop in runs
        ],
        dtype=np.float64,
    ).reshape(len(runs))


def replay_mask(
    duration_s: NDArray,
    dispersion_cm: NDArray,
    config: MemorylessSequenceConfig,
) -> NDArray:
    """Which candidate sequences meet the replay criteria.

    Parameters
    ----------
    duration_s, dispersion_cm : ndarray, shape (n_sequences,)
    config : MemorylessSequenceConfig

    Returns
    -------
    is_replay : ndarray of bool, shape (n_sequences,)

    """
    return _is_above(duration_s, config.minimum_duration_s) & _is_above(
        dispersion_cm, config.dispersion_threshold_cm
    )


@dataclass(frozen=True)
class CandidateSequences:
    """Every sequence one epoch yields, with the evidence for each decision.

    Attributes
    ----------
    runs : ndarray of int, shape (n_sequences, 2)
        Inclusive first and last window index of each sequence.
    duration_s, dispersion_cm : ndarray, shape (n_sequences,)
    is_replay : ndarray of bool, shape (n_sequences,)
        Met both replay criteria. Sequences that did not are kept, because a
        detector's rejections are as much a result as its detections.
    decisions : WindowDecisions
        Per-window centers, spreads, jumps, and gates.
    pieces : list of ndarray
        For each sequence, the indices into ``pre_merge_runs`` it absorbed.
    pre_merge_runs : ndarray of int, shape (n_runs, 2)

    """

    runs: NDArray
    duration_s: NDArray
    dispersion_cm: NDArray
    is_replay: NDArray
    decisions: WindowDecisions
    pieces: list[NDArray]
    pre_merge_runs: NDArray


def detect_candidate_sequences(
    posterior: NDArray,
    midpoint_s: NDArray,
    bin_distance_cm: NDArray,
    window_speed_cm_per_s: NDArray,
    config: MemorylessSequenceConfig,
) -> CandidateSequences:
    """Run every per-window step in order and assemble what survives.

    This is the whole detector downstream of decoding. It exists so the steps
    cannot be composed in another order: the speed gate must reach the centers
    before the jumps are measured, and the runs must be merged before their
    length is tested.

    Parameters
    ----------
    posterior : ndarray, shape (n_windows, n_bins)
        One normalized window per row, over real place bins.
    midpoint_s : ndarray, shape (n_windows,)
        When each window is timestamped.
    bin_distance_cm : ndarray, shape (n_bins, n_bins)
    window_speed_cm_per_s : ndarray, shape (n_windows,)
        Speed at each window's midpoint.
    config : MemorylessSequenceConfig

    Returns
    -------
    candidates : CandidateSequences

    """
    posterior = np.asarray(posterior, dtype=np.float64)
    midpoint_s = np.asarray(midpoint_s, dtype=np.float64)
    if midpoint_s.shape != (posterior.shape[0],):
        raise ValueError(
            f"{posterior.shape[0]} windows carry {midpoint_s.shape} midpoint "
            "times; every window needs exactly one."
        )

    decisions = window_decisions(
        posterior, bin_distance_cm, window_speed_cm_per_s, config
    )
    assembly = assemble_sequences(
        decisions.gates.kept,
        decisions.center_index,
        midpoint_s,
        bin_distance_cm,
        config,
    )
    duration_s = sequence_duration_s(assembly.runs, midpoint_s)
    dispersion_cm = sequence_dispersion_cm(
        assembly.runs, assembly.kept, decisions.center_index, bin_distance_cm
    )
    return CandidateSequences(
        runs=assembly.runs,
        duration_s=duration_s,
        dispersion_cm=dispersion_cm,
        is_replay=replay_mask(duration_s, dispersion_cm, config),
        decisions=decisions,
        pieces=assembly.pieces,
        pre_merge_runs=assembly.pre_merge_runs,
    )
