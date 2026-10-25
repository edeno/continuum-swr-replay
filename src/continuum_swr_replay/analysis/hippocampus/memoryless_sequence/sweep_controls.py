"""The memoryless controls' injected conditions, and the truth each is scored on.

The controls (``scripts/detection/sweep_memoryless_controls.py``) and the
held-out check (``scripts/detection/sweep_memoryless_heldout.py``) inject the
same conditions into the sweep's recording (``sweep.build_recording``): a
place-less and a jumping burst, a mid-event jump and a chain of two replays
(:func:`trials`), placed in the recording's immobile stretches clear of its
own events (:func:`usable_intervals`), and read the injected replay's path
(:func:`truth_path`) and the ground its parts cover (:func:`covered_cm`).
"""

import numpy as np
from non_local_detector.simulate.sorted_spikes_simulation import TRACK_HEIGHT

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence import sweep
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.simulation import (
    Gap,
    InjectedActivity,
    Segment,
    immobile_intervals,
    jumping_burst,
    piecewise_replay,
    placeless_burst,
    without,
)

__all__ = [
    "BURST_DURATION_S",
    "CHAIN_GAP_S",
    "CHAIN_PART_S",
    "CHAIN_STARTS_CM",
    "FAR_END_CM",
    "JUMP_CM",
    "N_RECORDINGS",
    "PART_CM",
    "SEGMENT_S",
    "covered_cm",
    "jump_segments",
    "trials",
    "truth_path",
    "usable_intervals",
]

BURST_DURATION_S = (0.08, 0.20, 0.50)
JUMP_CM = (0.0, 20.0, 40.0, 80.0)
SEGMENT_S = (0.12, 0.20)
CHAIN_GAP_S = (0.01, 0.03, 0.06, 0.10, 0.20)
CHAIN_PART_S = 0.15
PART_CM = 60.0

FAR_END_CM = 125.0
"""Where a jump's parts reach nearest the animal, 45 cm from where it rests,
whatever the jump: the second part starts J cm behind the first's end, and the
first starts wherever puts the far end of the two here."""

CHAIN_STARTS_CM = {"chain_continuing": (30.0, 90.0), "chain_elsewhere": (90.0, 10.0)}
"""Where each chain's two parts start; both reach 150 cm at most. Continuing: the
second picks up where the first ended. Elsewhere: the second starts 140 cm
from the first's end."""

N_RECORDINGS = 8
"""Recordings for the no-replay control. The simulator's seed changes the
spikes, not the trajectory, so every recording pauses in the same places and
the control reads local coding at the track's end, over eight draws of it."""


def usable_intervals(recording: dict) -> np.ndarray:
    """The recording's immobile stretches with its own labeled events cut out."""
    return without(
        immobile_intervals(
            recording["time"],
            recording["speed"],
            sweep.CONFIG.speed_threshold_cm_per_s,
            minimum_duration_s=1.0,
        ),
        recording["event_times"],
        margin_s=sweep.PLACEMENT_MARGIN_S,
    )


def truth_path(activity: InjectedActivity):
    """Where the injected activity says the replay is, at any time.

    Before the first sample the first represented place, after the last the
    last; inside, the represented place at the nearest sample, NaN in a
    silent gap.
    """
    time, position = activity.time_s, activity.position_cm
    finite = np.flatnonzero(np.isfinite(position))
    first, last = position[finite[0]], position[finite[-1]]

    def true_position_cm(at):
        at = np.asarray(at, dtype=np.float64)
        nearest = np.clip(np.searchsorted(time, at), 0, time.size - 1)
        previous = np.clip(nearest - 1, 0, time.size - 1)
        closer = np.where(
            np.abs(time[previous] - at) < np.abs(time[nearest] - at), previous, nearest
        )
        inside = position[closer]
        return np.where(at < time[0], first, np.where(at > time[-1], last, inside))

    return true_position_cm


def covered_cm(activity: InjectedActivity) -> float:
    """The ground the activity's parts cover together: the union of their paths."""
    spans = []
    for start_s, end_s in activity.parts:
        here = (activity.time_s >= start_s) & (activity.time_s <= end_s)
        position = activity.position_cm[here]
        position = position[np.isfinite(position)]
        spans.append((position.min(), position.max()))
    covered, reach = 0.0, -np.inf
    for low, high in sorted(spans):
        covered += max(0.0, high - max(low, reach))
        reach = max(reach, high)
    return float(covered)


def jump_segments(segment_s: float, jump_cm: float) -> list[Segment]:
    """Two 60 cm parts, the second starting ``jump_cm`` behind the first's end,
    their far end at ``FAR_END_CM``."""
    first_start = FAR_END_CM - max(PART_CM, 2 * PART_CM - jump_cm)
    return [
        Segment(segment_s, first_start, PART_CM),
        Segment(segment_s, first_start + PART_CM - jump_cm, PART_CM),
    ]


def trials(means: np.ndarray, sampling_frequency: float):
    """Every injected condition: name, grid level, peak rate, builder, and pieces.

    ``pieces`` is the multi-part event's list of segments and gaps, None for a
    burst.
    """
    track_cm = TRACK_HEIGHT

    def burst(builder, duration_s, rate_hz):
        def make(at_s, seed):
            extra = {"seed": seed} if builder is jumping_burst else {}
            return builder(
                duration_s, at_s, sampling_frequency, means, rate_hz, track_cm, **extra
            )

        return make

    def replay(pieces, rate_hz):
        def make(at_s, seed):
            return piecewise_replay(pieces, at_s, sampling_frequency, means, rate_hz)

        return make

    for rate_hz in sweep.PEAK_RATE_HZ:
        for duration_s in BURST_DURATION_S:
            for name, builder in (
                ("placeless_burst", placeless_burst),
                ("jumping_burst", jumping_burst),
            ):
                yield (
                    name,
                    duration_s,
                    rate_hz,
                    burst(builder, duration_s, rate_hz),
                    None,
                )
        for segment_s in SEGMENT_S:
            for jump_cm in JUMP_CM:
                pieces = jump_segments(segment_s, jump_cm)
                yield (
                    f"jump_{1000 * segment_s:.0f}ms",
                    jump_cm,
                    rate_hz,
                    replay(pieces, rate_hz),
                    pieces,
                )
        for condition, (first, second) in CHAIN_STARTS_CM.items():
            for gap_s in CHAIN_GAP_S:
                pieces = [
                    Segment(CHAIN_PART_S, first, PART_CM),
                    Gap(gap_s),
                    Segment(CHAIN_PART_S, second, PART_CM),
                ]
                yield condition, gap_s, rate_hz, replay(pieces, rate_hz), pieces
