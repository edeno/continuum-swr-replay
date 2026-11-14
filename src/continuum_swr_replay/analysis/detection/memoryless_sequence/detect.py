"""Turn per-window statistics into candidate sequences.

The order of the steps follows the original: gate each window on its speed,
then its posterior spread and its jump to the next window; collect contiguous
runs of surviving windows; merge runs that are close in both space and time;
drop runs shorter than the minimum duration; and keep the sequences that are
long enough and travel far enough to be replay.

Taking the speed gate before the jumps is load-bearing. A window whose
successor was rejected on speed has no forward jump and is itself rejected, so
a sequence ends one window before the first speed-rejected window. Computing
jumps first would let it run one window further, which is enough to move a
candidate across the duration filter. :func:`window_decisions` applies the four
steps in the one correct order so a caller cannot reorder them.

The per-window, merge and replay thresholds are strict inequalities, as in the
reference. The duration filter in :func:`assemble_sequences` is inclusive
(``>=``), matching the reference's removal of runs with ``lengths < thr``; a
candidate of exactly the minimum duration therefore survives assembly and is
then declined as replay. Ties are systematic rather than rare here: a 20 cm
jump is exactly ten 2 cm bins, and 100 ms is a whole number of strides at both
the original's 5 ms and this project's 4 ms (the 50 ms merge gap can tie only
at 5 ms). The comparisons therefore treat a value within a relative tolerance
of its threshold as equal to it, so float representation does not decide the
outcome.
"""

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from numpy.typing import NDArray

from continuum_swr_replay.analysis.detection.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.window_statistics import (
    NO_CENTER,
    dispersion_cm,
    forward_jumps_cm,
    window_center_and_spread,
)
from continuum_swr_replay.intervals import contiguous_true_runs

__all__ = [
    "CandidateSequences",
    "SequenceAssembly",
    "WindowDecisions",
    "WindowGates",
    "apply_speed_gate",
    "assemble_sequences",
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

_MASS_TOLERANCE = 1e-6
"""How far a usable posterior row may be from summing to one."""


def _is_below(value: NDArray, threshold: float) -> NDArray:
    """Strictly below ``threshold``, treating a value within tolerance as equal.

    NaN and ``+inf`` are never below a threshold.
    """
    value = np.asarray(value, dtype=np.float64)
    return (value < threshold) & ~np.isclose(
        value, threshold, rtol=_THRESHOLD_TOLERANCE, atol=0.0
    )


def _is_above(value: NDArray, threshold: float) -> NDArray:
    """Strictly above ``threshold``, treating a value within tolerance as equal.

    NaN and ``-inf`` are never above a threshold.
    """
    value = np.asarray(value, dtype=np.float64)
    return (value > threshold) & ~np.isclose(
        value, threshold, rtol=_THRESHOLD_TOLERANCE, atol=0.0
    )


def _freeze(instance: Any, **values: Any) -> None:
    """Store coerced values on a frozen dataclass, which cannot assign them itself."""
    for name, value in values.items():
        object.__setattr__(instance, name, value)


def _as_window_arrays(instance: Any, dtype: type) -> None:
    """Coerce every array field to ``dtype`` and require one shared 1-D shape."""
    shapes = {}
    for field in fields(instance):
        if field.type is not NDArray:
            continue
        array: NDArray = np.asarray(getattr(instance, field.name), dtype=dtype)
        if array.ndim != 1:
            raise ValueError(
                f"{field.name} must be one-dimensional, got {array.shape}."
            )
        object.__setattr__(instance, field.name, array)
        shapes[field.name] = array.shape
    if len(set(shapes.values())) > 1:
        raise ValueError(f"Per-window arrays disagree in length: {shapes}.")


@dataclass(frozen=True, eq=False)
class WindowGates:
    """Which criterion each window met, kept separately so rejections explain themselves.

    Attributes
    ----------
    posterior_usable : ndarray of bool, shape (n_windows,)
        The window's posterior gave it a center before any gate was applied.
    speed_eligible : ndarray of bool, shape (n_windows,)
        The speed at the window's midpoint was known and below the threshold.
    spread_below_threshold : ndarray of bool, shape (n_windows,)
    jump_below_threshold : ndarray of bool, shape (n_windows,)
        False where the forward jump is undefined, which happens when the next
        window has no center.

    """

    posterior_usable: NDArray
    speed_eligible: NDArray
    spread_below_threshold: NDArray
    jump_below_threshold: NDArray

    def __post_init__(self) -> None:
        _as_window_arrays(self, bool)

    @property
    def has_center(self) -> NDArray:
        """The posterior was usable and the speed gate did not blank it."""
        has_center: NDArray = self.posterior_usable & self.speed_eligible
        return has_center

    @property
    def kept(self) -> NDArray:
        """Every criterion together: the windows sequences are assembled from."""
        kept: NDArray = (
            self.has_center & self.spread_below_threshold & self.jump_below_threshold
        )
        return kept


@dataclass(frozen=True, eq=False)
class WindowDecisions:
    """Per-window statistics and gates in the order the detector computes them.

    Attributes
    ----------
    center_index : ndarray of int, shape (n_windows,)
        Bin index of each window's center after the speed gate, ``NO_CENTER``
        where the window has none. Agrees with ``gates.has_center``.
    spread_cm : ndarray, shape (n_windows,)
        ``nan`` where the posterior was unusable.
    jump_cm : ndarray, shape (n_windows,)
        Forward jump, ``nan`` where undefined.
    gates : WindowGates

    """

    center_index: NDArray
    spread_cm: NDArray
    jump_cm: NDArray
    gates: WindowGates

    def __post_init__(self) -> None:
        center = np.asarray(self.center_index, dtype=int)
        spread = np.asarray(self.spread_cm, dtype=np.float64)
        jump = np.asarray(self.jump_cm, dtype=np.float64)
        n_windows = self.gates.posterior_usable.shape
        if not center.shape == spread.shape == jump.shape == n_windows:
            raise ValueError(
                "Per-window arrays disagree in length: "
                f"center {center.shape}, spread {spread.shape}, jump {jump.shape}, "
                f"gates {n_windows}."
            )
        if not np.array_equal(center != NO_CENTER, self.gates.has_center):
            raise ValueError(
                "center_index and gates.has_center disagree about which windows "
                "have a center."
            )
        _freeze(self, center_index=center, spread_cm=spread, jump_cm=jump)


@dataclass(frozen=True, eq=False)
class SequenceAssembly:
    """Candidate sequences, with the intermediate stages kept for inspection.

    Attributes
    ----------
    runs : ndarray of int, shape (n_sequences, 2)
        Inclusive first and last window index of each surviving sequence.
    pieces : tuple of ndarray
        For each sequence, the indices into ``pre_merge_runs`` that it absorbed.
        A sequence with more than one piece spans a merged gap.
    pre_merge_runs : ndarray of int, shape (n_runs, 2)
        Runs before merging, after single-window runs were dropped (unless
        ``config.drop_single_window_runs`` is False).
    kept : ndarray of bool, shape (n_windows,)
        The window keep mask the sequences were assembled from. Windows inside
        a merged gap stay False here; they are not decoded positions.

    """

    runs: NDArray
    pieces: tuple[NDArray, ...]
    pre_merge_runs: NDArray
    kept: NDArray

    def __post_init__(self) -> None:
        runs = np.asarray(self.runs, dtype=int).reshape(-1, 2)
        pre_merge = np.asarray(self.pre_merge_runs, dtype=int).reshape(-1, 2)
        kept = np.asarray(self.kept, dtype=bool)
        pieces = tuple(np.asarray(piece, dtype=int) for piece in self.pieces)
        if len(pieces) != len(runs):
            raise ValueError(f"{len(runs)} sequences but {len(pieces)} piece lists.")
        if runs.size and (runs[:, 0] >= runs[:, 1]).any():
            raise ValueError("Every sequence must span at least two windows.")
        if runs.size and runs.max() >= kept.size:
            raise ValueError(
                f"A sequence reaches window {runs.max()} but there are only "
                f"{kept.size} windows."
            )
        for run, piece in zip(runs, pieces, strict=True):
            if piece.size == 0 or not (
                run[0] == pre_merge[piece[0], 0] and run[1] == pre_merge[piece[-1], 1]
            ):
                raise ValueError(
                    f"Sequence {run.tolist()} is not bounded by the pre-merge runs "
                    f"it lists as its pieces, {piece.tolist()}."
                )
        _freeze(self, runs=runs, pieces=pieces, pre_merge_runs=pre_merge, kept=kept)


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
    speed_eligible: NDArray | None = None,
) -> WindowGates:
    """Apply the per-window criteria, recording each one separately.

    Parameters
    ----------
    center_index : ndarray of int, shape (n_windows,)
        Center of each window before the speed gate; ``NO_CENTER`` where the
        posterior gave none.
    spread_cm, jump_cm : ndarray, shape (n_windows,)
        The jump must already reflect the speed gate, since a window whose
        successor was blanked has none.
    config : MemorylessSequenceConfig
    speed_eligible : ndarray of bool, shape (n_windows,), optional
        Which windows the speed gate let through. Every window, when omitted.

    Returns
    -------
    gates : WindowGates

    """
    usable = np.asarray(center_index, dtype=int) != NO_CENTER
    eligible = (
        np.ones_like(usable)
        if speed_eligible is None
        else np.asarray(speed_eligible, dtype=bool)
    )
    return WindowGates(
        posterior_usable=usable,
        speed_eligible=eligible,
        spread_below_threshold=_is_below(spread_cm, config.spread_threshold_cm),
        jump_below_threshold=_is_below(jump_cm, config.jump_threshold_cm),
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
    raw_center, spread = window_center_and_spread(posterior, bin_distance_cm)
    center = apply_speed_gate(
        raw_center, window_speed_cm_per_s, config.speed_threshold_cm_per_s
    )
    jump = forward_jumps_cm(center, bin_distance_cm)
    return WindowDecisions(
        center_index=center,
        spread_cm=spread,
        jump_cm=jump,
        gates=gate_windows(
            raw_center,
            spread,
            jump,
            config,
            speed_eligible=_is_below(
                window_speed_cm_per_s, config.speed_threshold_cm_per_s
            ),
        ),
    )


def merge_runs(
    runs: NDArray,
    center_index: NDArray,
    window_time_s: NDArray,
    bin_distance_cm: NDArray,
    config: MemorylessSequenceConfig,
    unobserved: NDArray | None = None,
) -> tuple[NDArray, list[NDArray]]:
    """Join neighboring runs separated by a short gap in both space and time.

    The gap is measured between the facing endpoints: the last window of one
    run and the first window of the next. Merging is transitive, so a chain of
    close runs becomes one sequence. No merge bridges an ``unobserved``
    window: a candidate never spans one, so the shuffle, which re-decodes a
    candidate's every window, never reads its rows.

    Parameters
    ----------
    runs : ndarray of int, shape (n_runs, 2)
        Inclusive bounds, in order. Every endpoint must have a center.
    center_index : ndarray of int, shape (n_windows,)
    window_time_s : ndarray, shape (n_windows,)
        Window midpoints.
    bin_distance_cm : ndarray, shape (n_bins, n_bins)
    config : MemorylessSequenceConfig
    unobserved : ndarray of bool, shape (n_windows,), optional
        Windows holding a missing row (the artifact mask); None for none.

    Returns
    -------
    merged : ndarray of int, shape (n_merged, 2)
    pieces : list of ndarray
        For each merged run, the indices into ``runs`` that it absorbed.

    Raises
    ------
    ValueError
        A run endpoint has no center. ``NO_CENTER`` is ``-1``, which would
        otherwise index the last bin and measure a gap that does not exist.

    """
    runs = np.asarray(runs, dtype=int).reshape(-1, 2)
    if len(runs) == 0:
        return runs.copy(), []
    center = np.asarray(center_index, dtype=int)
    if (center[runs] == NO_CENTER).any():
        raise ValueError(
            "A run endpoint has no center; runs must be built from windows that "
            "survived the gates."
        )
    time_s = np.asarray(window_time_s, dtype=np.float64)
    distance = np.asarray(bin_distance_cm, dtype=np.float64)

    before, after = runs[:-1, 1], runs[1:, 0]
    gap_cm = distance[center[before], center[after]]
    gap_s = time_s[after] - time_s[before]
    joins = _is_below(gap_cm, config.merge_gap_cm) & _is_below(
        gap_s, config.merge_gap_s
    )
    if unobserved is not None:
        # the windows strictly between the facing endpoints
        count = np.concatenate(([0], np.cumsum(np.asarray(unobserved, dtype=bool))))
        joins &= count[after] == count[before + 1]

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
    unobserved: NDArray | None = None,
) -> SequenceAssembly:
    """Collect kept windows into candidate sequences.

    Runs of a single window are dropped first: as the port defines run length,
    an index difference, theirs is zero. Whether the reference does the same
    depends on its unvendored ``seqle``; ``config.drop_single_window_runs``
    turns the drop off for that sensitivity. Surviving runs are merged, and
    then the duration filter drops those whose index length is below the
    stride-derived minimum. Length is an index difference throughout, so a run
    of index length ``n`` holds ``n + 1`` windows and spans ``n`` strides.

    Parameters
    ----------
    kept : ndarray of bool, shape (n_windows,)
    center_index : ndarray of int, shape (n_windows,)
    window_time_s : ndarray, shape (n_windows,)
    bin_distance_cm : ndarray, shape (n_bins, n_bins)
    config : MemorylessSequenceConfig
    unobserved : ndarray of bool, shape (n_windows,), optional
        Windows no merge may bridge (:func:`merge_runs`).

    Returns
    -------
    assembly : SequenceAssembly

    """
    kept = np.asarray(kept, dtype=bool)
    # first and last window (inclusive) of each unbroken stretch of kept windows
    runs = contiguous_true_runs(kept) - [0, 1]
    if config.drop_single_window_runs:
        runs = runs[runs[:, 1] > runs[:, 0]]

    merged, pieces = merge_runs(
        runs, center_index, window_time_s, bin_distance_cm, config, unobserved
    )
    long_enough = (merged[:, 1] - merged[:, 0]) >= config.minimum_run_index_length
    return SequenceAssembly(
        runs=merged[long_enough],
        pieces=tuple(
            piece for piece, keep in zip(pieces, long_enough, strict=True) if keep
        ),
        pre_merge_runs=runs,
        kept=kept,
    )


def sequence_duration_s(runs: NDArray, window_time_s: NDArray) -> NDArray:
    """Time from each sequence's first window midpoint to its last.

    The 80 ms window width is not added: the decoding support is a separate
    quantity from the event interval.

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
    duration: NDArray = time_s[runs[:, 1]] - time_s[runs[:, 0]]
    return duration


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
    runs: NDArray,
    dispersion_cm: NDArray,
    config: MemorylessSequenceConfig,
) -> NDArray:
    """Which candidate sequences meet the replay criteria.

    Duration is judged in strides, not by subtracting midpoint times: windows
    sit on a uniform grid, so a run's stride count is exact, whereas two
    midpoints on a recording clock of ~1e9 s subtract to within only ~1e-7 s
    of their true span, enough to pass a run of exactly
    ``minimum_duration_s``.

    Parameters
    ----------
    runs : ndarray of int, shape (n_sequences, 2)
        Inclusive first and last window index of each sequence.
    dispersion_cm : ndarray, shape (n_sequences,)
    config : MemorylessSequenceConfig

    Returns
    -------
    is_replay : ndarray of bool, shape (n_sequences,)

    """
    runs = np.asarray(runs, dtype=int).reshape(-1, 2)
    n_strides = runs[:, 1] - runs[:, 0]
    mask: NDArray = _is_above(
        n_strides, config.minimum_duration_s / config.stride_s
    ) & _is_above(dispersion_cm, config.dispersion_threshold_cm)
    return mask


@dataclass(frozen=True, eq=False)
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
    assembly : SequenceAssembly
        The runs before merging and which of them each sequence absorbed.
    midpoint_s : ndarray, shape (n_windows,)
        When each window is timestamped, so durations can be re-derived.
    config : MemorylessSequenceConfig
        The thresholds these decisions were made under, so ``is_replay`` can
        be re-derived and a saved result says what produced it.

    """

    runs: NDArray
    duration_s: NDArray
    dispersion_cm: NDArray
    is_replay: NDArray
    decisions: WindowDecisions
    assembly: SequenceAssembly
    midpoint_s: NDArray
    config: MemorylessSequenceConfig

    def __post_init__(self) -> None:
        runs = np.asarray(self.runs, dtype=int).reshape(-1, 2)
        duration = np.asarray(self.duration_s, dtype=np.float64)
        dispersion = np.asarray(self.dispersion_cm, dtype=np.float64)
        is_replay = np.asarray(self.is_replay, dtype=bool)
        midpoint = np.asarray(self.midpoint_s, dtype=np.float64)
        n_sequences = len(runs)
        if not (
            duration.shape == dispersion.shape == is_replay.shape == (n_sequences,)
        ):
            raise ValueError(
                f"{n_sequences} sequences carry {duration.shape} durations, "
                f"{dispersion.shape} dispersions and {is_replay.shape} verdicts."
            )
        if midpoint.shape != self.decisions.center_index.shape:
            raise ValueError(
                f"{midpoint.shape} midpoints for "
                f"{self.decisions.center_index.shape} windows."
            )
        if not np.array_equal(runs, self.assembly.runs):
            raise ValueError("runs and assembly.runs disagree.")
        if not np.allclose(duration, sequence_duration_s(runs, midpoint)):
            raise ValueError("duration_s does not follow from runs and midpoint_s.")
        if not np.array_equal(is_replay, replay_mask(runs, dispersion, self.config)):
            raise ValueError("is_replay does not follow from the criteria and config.")
        _freeze(
            self,
            runs=runs,
            duration_s=duration,
            dispersion_cm=dispersion,
            is_replay=is_replay,
            midpoint_s=midpoint,
        )

    @property
    def pieces(self) -> tuple[NDArray, ...]:
        """For each sequence, the indices into ``pre_merge_runs`` it absorbed."""
        return self.assembly.pieces

    @property
    def pre_merge_runs(self) -> NDArray:
        """Runs before merging; see ``SequenceAssembly.pre_merge_runs``."""
        return self.assembly.pre_merge_runs


def load_candidates(path: str | Path) -> CandidateSequences:
    """Read a run's ``{epoch_id}_memoryless_candidates.pkl``.

    A file written before the package moved (``docs/history/2026-module-moves.md``)
    names its classes under the former module path and does not load.

    Parameters
    ----------
    path : str or Path
        The candidates file ``detection_inputs.memoryless_output_paths`` names.

    Returns
    -------
    CandidateSequences
    """
    return joblib.load(path)


def detect_candidate_sequences(
    posterior: NDArray,
    midpoint_s: NDArray,
    bin_distance_cm: NDArray,
    window_speed_cm_per_s: NDArray,
    config: MemorylessSequenceConfig,
    unobserved: NDArray | None = None,
) -> CandidateSequences:
    """Run every per-window step in order and assemble what survives.

    This is the whole detector downstream of decoding. It exists so the steps
    cannot be composed in another order: the speed gate must reach the centers
    before the jumps are measured, and the runs must be merged before their
    length is tested.

    Parameters
    ----------
    posterior : ndarray, shape (n_windows, n_bins)
        One normalized window per row, over real place bins. A row that is
        all NaN is a window the decoder could not evaluate.
    midpoint_s : ndarray, shape (n_windows,)
        When each window is timestamped, strictly increasing.
    bin_distance_cm : ndarray, shape (n_bins, n_bins)
    window_speed_cm_per_s : ndarray, shape (n_windows,)
        Speed at each window's midpoint.
    config : MemorylessSequenceConfig
    unobserved : ndarray of bool, shape (n_windows,), optional
        Windows holding a missing row, such as the artifact mask's bins. Their
        posterior rows must be all NaN, so none is kept, and no merge bridges
        one. None for none.

    Returns
    -------
    candidates : CandidateSequences

    Raises
    ------
    ValueError
        The arrays disagree in length, an unobserved window has a posterior,
        the midpoints are not finite and
        increasing, a posterior row mixes finite and non-finite values, or a
        finite posterior row does not sum to one.

    """
    posterior = np.asarray(posterior, dtype=np.float64)
    midpoint_s = np.asarray(midpoint_s, dtype=np.float64)
    if midpoint_s.shape != (posterior.shape[0],):
        raise ValueError(
            f"{posterior.shape[0]} windows carry {midpoint_s.shape} midpoint "
            "times; every window needs exactly one."
        )
    if not np.isfinite(midpoint_s).all():
        raise ValueError("midpoint_s must be finite.")
    if midpoint_s.size > 1 and not (np.diff(midpoint_s) > 0).all():
        raise ValueError("midpoint_s must be strictly increasing.")
    finite = np.isfinite(posterior).all(axis=1)
    unevaluated = np.isnan(posterior).all(axis=1)
    if not (finite | unevaluated).all():
        raise ValueError(
            "A posterior row mixes finite and non-finite values; a window the "
            "decoder could not evaluate must be all NaN."
        )
    mass = posterior[finite].sum(axis=1)
    if not np.allclose(mass, 1.0, rtol=0.0, atol=_MASS_TOLERANCE):
        raise ValueError(
            f"A posterior row sums to {mass[np.argmax(np.abs(mass - 1.0))]}, not "
            "one; normalize the windows first."
        )

    if unobserved is not None:
        unobserved = np.asarray(unobserved, dtype=bool)
        if unobserved.shape != midpoint_s.shape or not unevaluated[unobserved].all():
            raise ValueError(
                "unobserved must flag windows whose posterior rows are all NaN, "
                "one flag per window."
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
        unobserved,
    )
    duration_s = sequence_duration_s(assembly.runs, midpoint_s)
    dispersion = sequence_dispersion_cm(
        assembly.runs, assembly.kept, decisions.center_index, bin_distance_cm
    )
    return CandidateSequences(
        runs=assembly.runs,
        duration_s=duration_s,
        dispersion_cm=dispersion,
        is_replay=replay_mask(assembly.runs, dispersion, config),
        decisions=decisions,
        assembly=assembly,
        midpoint_s=midpoint_s,
        config=config,
    )
