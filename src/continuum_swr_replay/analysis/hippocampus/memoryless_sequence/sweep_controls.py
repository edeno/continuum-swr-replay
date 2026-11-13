"""The memoryless controls: injected conditions that are not one clean replay.

The controls ask what each detector reports where there is no replay
(:func:`run_no_replay`: the recording's own immobile stretches, decoded as they
are), and how it carves activity that is not one clean replay
(:func:`run_trials`): a place-less and a jumping burst, a mid-event jump and a
chain of two replays (:func:`trials`), placed in the sweep's recording
(``sweep.build_recording``) clear of its own events (``sweep.usable_intervals``).
Each trial is scored for the four outputs of :data:`DETECTORS`
(:func:`score_trial`) against the injected replay's path (:func:`truth_path`)
and the ground its parts cover (:func:`covered_cm`); each part of a multi-part
event is also decoded alone (:func:`part_alone_spikes`). The held-out check
(``inventory_comparison.heldout_sweep``) injects the same conditions.
"""

import zlib

import numpy as np
from non_local_detector.simulate.sorted_spikes_simulation import (
    SAMPLING_FREQUENCY,
    TRACK_HEIGHT,
)

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence import (
    comparison_simulation,
    sweep,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.scoring import (
    bound_errors,
    part_agreement,
    recovered_parts,
    score_against_injection,
    segmentation,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.simulation import (
    Gap,
    InjectedActivity,
    Segment,
    inject_activity,
    jumping_burst,
    piecewise_replay,
    place_field_means,
    placeless_burst,
    placements,
)

__all__ = [
    "BURST_DURATION_S",
    "CHAIN_GAP_S",
    "CHAIN_PART_S",
    "CHAIN_STARTS_CM",
    "DETECTORS",
    "FAR_END_CM",
    "JUMP_CM",
    "MULTIPART_CONDITIONS",
    "N_RECORDINGS",
    "PART_CM",
    "SEGMENT_S",
    "condition_rng",
    "covered_cm",
    "detector_events",
    "jump_condition",
    "jump_segments",
    "part_alone_spikes",
    "run_detectors",
    "run_no_replay",
    "run_trials",
    "score_trial",
    "trials",
    "truth_path",
]

DETECTORS = (
    "sliding_accepted",
    "sliding_published",
    "state_space_canonical",
    "state_space_continuous",
)
"""The four scored outputs: the sliding window before and after its
significance test, and the state-space detector canonical (continuous plus
fragmented) and continuous-only."""

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


def jump_condition(segment_s: float) -> str:
    """The name of the mid-event jump condition whose parts last ``segment_s``."""
    return f"jump_{1000 * segment_s:.0f}ms"


MULTIPART_CONDITIONS = (*map(jump_condition, SEGMENT_S), *CHAIN_STARTS_CM)
"""The conditions whose events have two parts: the jumps and the chains."""

N_RECORDINGS = 8
"""Recordings for the no-replay control. The simulator's seed changes the
spikes, not the trajectory, so every recording pauses in the same places and
the control reads local coding at the track's end, over eight draws of it."""


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
                    jump_condition(segment_s),
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


def detector_events(sliding: sweep.SlidingWindowRun, state: sweep.StateSpaceRun):
    """The events each of the four scored outputs reports."""
    return {
        "sliding_accepted": [sliding.events[i] for i in sliding.accepted],
        "sliding_published": sweep.published_events(sliding),
        "state_space_canonical": state.canonical,
        "state_space_continuous": state.continuous_only,
    }


def score_trial(
    activity: InjectedActivity, events: dict, state: sweep.StateSpaceRun
) -> list[dict]:
    """One row per scored output: was it called, and how were its parts carved."""
    bounds = activity.bounds
    in_trial = (state.time >= bounds[0]) & (state.time <= bounds[1])
    shares = {
        "continuous_share": float(state.p_continuous[in_trial].mean()),
        "fragmented_share": float(state.p_fragmented[in_trial].mean()),
    }
    rows = []
    for detector, reported in events.items():
        overlapping = [
            event
            for event in reported
            if sweep.overlaps((event.start_s, event.end_s), bounds)
        ]
        row = {"detector": detector, "called": bool(overlapping)} | shares
        if activity.truth == "continuous_replay":
            event_bounds = [(event.start_s, event.end_s) for event in overlapping]
            parts = list(activity.parts)
            truth = truth_path(activity)
            # right about a part: the event is the injection as a whole, and its
            # positions over that part sit on that part's path
            agreement = [
                score_against_injection(
                    event.decoded_time_s,
                    event.decoded_cm,
                    truth,
                    bounds,
                    covered_cm(activity),
                    sweep.MATCH_TOLERANCE_CM,
                ).matches
                & part_agreement(
                    event.decoded_time_s,
                    event.decoded_cm,
                    truth,
                    parts,
                    sweep.MATCH_TOLERANCE_CM,
                )
                for event in overlapping
            ]
            errors = bound_errors(event_bounds, agreement, parts)
            row |= {
                "n_parts": len(parts),
                "n_recovered": int(
                    recovered_parts(event_bounds, agreement, parts).sum()
                ),
                "segmentation": segmentation(event_bounds, agreement, parts),
                "start_errors_s": errors[:, 0].tolist(),
                "end_errors_s": errors[:, 1].tolist(),
            }
        rows.append(row)
    return rows


def run_detectors(recording: dict, spike_times, activity: InjectedActivity, key: str):
    """Both detectors over the activity and a margin either side, from these spikes."""
    window = sweep.decode_window(recording, activity.bounds)
    sliding = sweep.run_sliding_window(recording, spike_times, window, key)
    state = sweep.run_state_space(recording, spike_times, window)
    return sliding, state


def part_alone_spikes(
    original: list[np.ndarray],
    injected: list[np.ndarray],
    part_bounds: tuple[float, float],
) -> list[np.ndarray]:
    """One part of an injected event on its own: the event's own spikes inside
    the part, the recording's own spikes everywhere else.

    Paired with the event, so a part's recovery alone and inside the event
    differ only in what surrounds it, not in a new draw of its spikes.
    """
    start_s, end_s = part_bounds
    return [
        np.sort(
            np.concatenate(
                [
                    before[(before < start_s) | (before > end_s)],
                    after[(after >= start_s) & (after <= end_s)],
                ]
            )
        )
        for before, after in zip(original, injected, strict=True)
    ]


def condition_rng(condition: str, level: float, rate_hz: float) -> np.random.Generator:
    """Placements for one condition, apart from every other's, so adding a
    condition moves no other's."""
    return np.random.default_rng(zlib.crc32(f"{condition}-{level}-{rate_hz}".encode()))


def run_trials(recording: dict, n_seeds: int) -> list[dict]:
    """Every injected condition at every seed, placed as the sweep places replays.

    Each part of a multi-part event is also decoded alone, with exactly the
    spikes it had inside the event and the recording's own spikes everywhere
    else, and its recovery alone is kept beside its recovery inside the event.
    """
    means = place_field_means(comparison_simulation.N_NEURONS, TRACK_HEIGHT)
    intervals = sweep.usable_intervals(recording)
    rows = []
    for condition, level, rate_hz, make, pieces in trials(means, SAMPLING_FREQUENCY):
        span_s = make(0.0, 0).bounds[1]
        at_s = placements(
            intervals,
            span_s,
            sweep.PLACEMENT_MARGIN_S,
            n_seeds,
            condition_rng(condition, level, rate_hz),
        )
        for seed in range(n_seeds):
            activity = make(float(at_s[seed]), seed)
            key = f"{condition}-{level}-rate{rate_hz}-seed{seed}"
            injected = inject_activity(recording["spike_times"], activity, seed)
            sliding, state = run_detectors(recording, injected, activity, key)
            trial = score_trial(activity, detector_events(sliding, state), state)
            if pieces is not None:
                alone = np.zeros(len(DETECTORS))
                for part, ((part_start, _), segment) in enumerate(
                    zip(
                        activity.parts,
                        [piece for piece in pieces if isinstance(piece, Segment)],
                        strict=True,
                    )
                ):
                    single = piecewise_replay(
                        [segment], part_start, SAMPLING_FREQUENCY, means, rate_hz
                    )
                    sliding, state = run_detectors(
                        recording,
                        part_alone_spikes(
                            recording["spike_times"], injected, single.parts[0]
                        ),
                        single,
                        f"{key}-part{part}",
                    )
                    for index, row in enumerate(
                        score_trial(single, detector_events(sliding, state), state)
                    ):
                        alone[index] += row["n_recovered"]
                for index, row in enumerate(trial):
                    row["n_recovered_alone"] = int(alone[index])
            for row in trial:
                rows.append(
                    {
                        "condition": condition,
                        "level": level,
                        "peak_rate_hz": rate_hz,
                        "seed": seed,
                        "truth": activity.truth,
                    }
                    | row
                )
    return rows


def run_no_replay(n_recordings: int) -> list[dict]:
    """Every usable immobile stretch of each recording, decoded as it is."""
    rows = []
    for recording_seed in range(n_recordings):
        recording = sweep.build_recording(seed=recording_seed)
        for index, (start_s, end_s) in enumerate(sweep.usable_intervals(recording)):
            window = sweep.decode_window(recording, (start_s, end_s), margin_s=0.0)
            key = f"no-replay-recording{recording_seed}-stretch{index}"
            sliding = sweep.run_sliding_window(
                recording, recording["spike_times"], window, key
            )
            state = sweep.run_state_space(recording, recording["spike_times"], window)
            for detector, reported in detector_events(sliding, state).items():
                inside = [
                    event
                    for event in reported
                    if start_s <= (event.start_s + event.end_s) / 2 <= end_s
                ]
                rows.append(
                    {
                        "condition": "no_replay",
                        "recording": recording_seed,
                        "stretch": index,
                        "exposure_s": float(end_s - start_s),
                        "detector": detector,
                        "truth": "local",
                        "n_events": len(inside),
                        "event_time_s": float(
                            sum(
                                min(event.end_s, end_s) - max(event.start_s, start_s)
                                for event in inside
                            )
                        ),
                    }
                )
    return rows
