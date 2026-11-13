"""The held-out check on simulated data, where every event's truth is known.

The memoryless sweeps' recording (``memoryless_sequence.sweep``) has its cells
split into discovery and evaluation cells (:func:`folds`). In each fold both
detectors are fitted and run on the discovery cells alone, over every injected
replay of the sweep and every trial of the controls (:func:`trials`,
:func:`run_fold`) and over the controls' no-replay stretches
(:func:`run_no_replay_fold`); the evaluation cells' fields are fitted
separately (:func:`fold_models`). Every event either detector reports is
classed, labelled with what it truly is (:func:`event_truth`,
:func:`matches_injection`) and scored by the held-out rule of
:mod:`~continuum_swr_replay.analysis.detector_comparison.heldout` under both
content estimates (:func:`score_events`). ``heldout.heldout_summary``
summarizes the rows.
"""

import zlib

import numpy as np
from non_local_detector.likelihoods.common import get_spikecount_per_time_bin
from non_local_detector.simulate.sorted_spikes_simulation import (
    SAMPLING_FREQUENCY,
    TRACK_HEIGHT,
)

from continuum_swr_replay.analysis.detection.memoryless_sequence import (
    sweep,
    sweep_controls,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.comparison_simulation import (
    CONFIG,
    N_NEURONS,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.likelihood import (
    with_reference_rate_floor,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.scoring import (
    score_against_injection,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.simulation import (
    Segment,
    inject_activity,
    piecewise_replay,
    place_field_means,
    placements,
)
from continuum_swr_replay.analysis.detector_comparison.heldout import (
    event_classes,
    heldout_score,
    interleaved_folds,
    nearest_window_content,
    poor_coverage_fold,
)

__all__ = [
    "FOLDS",
    "N_PERMUTATIONS",
    "OUR_EVENT_CLASSES",
    "PLACEMENT_SEED",
    "TILING_FOLDS",
    "event_truth",
    "fold_models",
    "folds",
    "matches_injection",
    "permutations_for",
    "run_fold",
    "run_no_replay_fold",
    "score_events",
    "trials",
]

N_PERMUTATIONS = 100
"""Permutations of the evaluation cells' fields per event, for the chance null."""

PLACEMENT_SEED = 2
"""Seeds the placements of a fold's injected trials."""


def folds() -> dict:
    """The discovery and evaluation cells of each fold, by fold name."""
    return interleaved_folds(N_NEURONS) | poor_coverage_fold(N_NEURONS)


FOLDS = tuple(folds())
"""The folds' names, in order: ``("A", "B", "C")``."""

TILING_FOLDS = tuple(interleaved_folds(N_NEURONS))
"""The folds whose evaluation cells tile the track (``A``, ``B``); the other
fold's evaluation cells cover half of it."""

OUR_EVENT_CLASSES = ("ours_only", "shared")
"""The classes of the events our detector reports
(:func:`~continuum_swr_replay.analysis.detector_comparison.heldout.event_classes`):
alone, or overlapping an event of the sliding window."""


def fold_models(discovery: np.ndarray, evaluation: np.ndarray, seed: int = 0):
    """The discovery recording, and the evaluation cells' fields per sample.

    The evaluation fields carry the reference's rate floor, so a spike from an
    evaluation cell nearly silent at a place costs a bounded log penalty rather
    than letting one spike decide an event.
    """
    found = sweep.build_recording(seed=seed, cells=discovery)
    held = sweep.build_recording(seed=seed, cells=evaluation)
    if not np.array_equal(found["place_bin_centers_cm"], held["place_bin_centers_cm"]):
        raise RuntimeError(
            "Discovery and evaluation fits placed their bins differently."
        )
    environment = held["encoding_model"]["environment"]
    interior = np.asarray(environment.is_track_interior_).ravel()
    fields = np.asarray(
        with_reference_rate_floor(held["encoding_model"], CONFIG)["place_fields"],
        dtype=np.float64,
    )[:, interior]
    return found, fields


def trials(means: np.ndarray):
    """The sweep's injected replays and the controls' trials, as the controls
    list them: name, grid level, peak rate, builder, and pieces."""
    sampling_frequency = SAMPLING_FREQUENCY
    for path_cm, midpoint_cm, duration_s, rate_hz in sweep.replay_grid():

        def make(at_s, seed, p=path_cm, m=midpoint_cm, d=duration_s, r=rate_hz):
            return piecewise_replay(
                [Segment(d, m - p / 2, p)],
                at_s,
                sampling_frequency,
                means,
                r,
            )

        yield (
            f"replay_path{path_cm:.0f}_mid{midpoint_cm:.0f}",
            duration_s,
            rate_hz,
            make,
            None,
        )
    yield from sweep_controls.trials(means, sampling_frequency)


def event_truth(event, activity, recorded_events: np.ndarray) -> str:
    """What an event's content truly is, judged by time spent in the activity's parts.

    Overlap is measured against the parts that represent places, not the
    activity's outer bounds: a chain's silent gap represents nothing. Half or
    more of the event inside the parts, the activity's own label, whatever
    local coding it absorbed around them; some but less than half, ``mixed``;
    none, but inside the activity's bounds, ``silent_gap``. Outside the
    activity altogether: ``recorded_event`` if it overlaps one of the
    recording's own labeled events, which the decode window around an
    injection can reach and whose content is real, else ``local``. Whether
    the event also matched the injection is a separate question
    (``matches_injection``).
    """
    duration_s = event.end_s - event.start_s
    inside_s = 0.0
    if activity is not None:
        inside_s = sum(
            max(0.0, min(event.end_s, end_s) - max(event.start_s, start_s))
            for start_s, end_s in activity.parts
        )
        if inside_s <= 0 and sweep.overlaps(
            (event.start_s, event.end_s), activity.bounds
        ):
            return "silent_gap"
    if inside_s <= 0:
        if any(
            sweep.overlaps((event.start_s, event.end_s), (float(a), float(b)))
            for a, b in recorded_events
        ):
            return "recorded_event"
        return "local"
    if inside_s < 0.5 * duration_s:
        return "mixed"
    return activity.truth


def matches_injection(event, activity) -> bool:
    """The event is the injected replay under the sweep's scoring rule."""
    if (
        activity is None
        or activity.truth != "continuous_replay"
        or not sweep.overlaps((event.start_s, event.end_s), activity.bounds)
    ):
        return False
    return score_against_injection(
        event.decoded_time_s,
        event.decoded_cm,
        sweep_controls.truth_path(activity),
        activity.bounds,
        sweep_controls.covered_cm(activity),
        sweep.MATCH_TOLERANCE_CM,
    ).matches


def permutations_for(key: str, n_cells: int) -> list[np.ndarray]:
    """Seeded permutations of the evaluation cells, one stream per event."""
    rng = np.random.default_rng(zlib.crc32(key.encode()))
    return [rng.permutation(n_cells) for _ in range(N_PERMUTATIONS)]


def score_events(
    fold, key, activity, sliding, state, evaluation_spikes, fields, window, recording
) -> list[dict]:
    """Every event either detector reports in one trial, under both estimators."""
    time = state.time
    counts = np.stack(
        [
            get_spikecount_per_time_bin(
                spikes[(spikes >= time[0]) & (spikes <= time[-1])], time
            )
            for spikes in evaluation_spikes
        ],
        axis=1,
    )
    animal_bin = sweep.nearest_bin(recording, np.interp(time, window[0], window[1]))
    ours = state.canonical
    theirs = sweep.published_events(sliding)
    ours_classes, theirs_classes = event_classes(
        [(e.start_s, e.end_s) for e in ours], [(e.start_s, e.end_s) for e in theirs]
    )
    rows = []
    for detector, events, classes in (
        ("state_space", ours, ours_classes),
        ("sliding_window", theirs, theirs_classes),
    ):
        for index, (event, event_class) in enumerate(zip(events, classes, strict=True)):
            bins = (time >= event.start_s) & (time <= event.end_s)
            if not bins.any():
                continue
            contents = {
                "state_space": state.non_local_posterior[bins],
                "sliding_window": nearest_window_content(
                    sliding.posterior, sliding.midpoint_s, time[bins]
                ),
            }
            truth = event_truth(event, activity, recording["event_times"])
            matched = matches_injection(event, activity)
            draws = permutations_for(
                f"{fold}-{key}-{detector}-{index}", fields.shape[0]
            )
            for estimator, content in contents.items():
                score = heldout_score(
                    content, counts[bins], fields, animal_bin[bins], draws
                )
                rows.append(
                    {
                        "fold": fold,
                        "trial": key,
                        "detector": detector,
                        "event": index,
                        "event_class": event_class,
                        "truth": truth,
                        "matches_injection": matched,
                        "estimator": estimator,
                        "duration_s": event.end_s - event.start_s,
                        "spike_term": score.spike_term,
                        "silence_term": score.silence_term,
                        "gain_vs_animal": score.gain_vs_animal,
                        "p_shuffle": score.p_shuffle,
                        "coverage": score.coverage,
                        "n_spikes": score.n_spikes,
                        "supported": score.supported,
                    }
                )
    return rows


def run_fold(fold, discovery, evaluation, n_seeds) -> list[dict]:
    """Every injected trial of :func:`trials`, decoded from this fold's discovery
    cells and scored on its evaluation cells.

    Placements are drawn from one generator seeded :data:`PLACEMENT_SEED`,
    shared by the trials in order.
    """
    recording, fields = fold_models(discovery, evaluation)
    means = place_field_means(N_NEURONS, TRACK_HEIGHT)
    intervals = sweep.usable_intervals(recording)
    rng = np.random.default_rng(PLACEMENT_SEED)
    rows = []
    for condition, level, rate_hz, make, _ in trials(means):
        span_s = make(0.0, 0).bounds[1]
        at_s = placements(intervals, span_s, sweep.PLACEMENT_MARGIN_S, n_seeds, rng)
        for seed in range(n_seeds):
            activity = make(float(at_s[seed]), seed)
            injected = inject_activity(recording["all_spike_times"], activity, seed)
            window = sweep.decode_window(recording, activity.bounds)
            key = f"{condition}-{level}-rate{rate_hz}-seed{seed}"
            found = [injected[cell] for cell in discovery]
            sliding = sweep.run_sliding_window(
                recording, found, window, f"{fold}-{key}"
            )
            state = sweep.run_state_space(recording, found, window)
            for row in score_events(
                fold,
                key,
                activity,
                sliding,
                state,
                [injected[cell] for cell in evaluation],
                fields,
                window,
                recording,
            ):
                rows.append(
                    row
                    | {
                        "condition": condition,
                        "level": level,
                        "peak_rate_hz": rate_hz,
                        "seed": seed,
                    }
                )
    return rows


def run_no_replay_fold(fold, discovery, evaluation, n_recordings) -> list[dict]:
    """The controls' no-replay stretches, decoded from this fold's discovery cells."""
    rows = []
    for recording_seed in range(n_recordings):
        recording, fields = fold_models(discovery, evaluation, seed=recording_seed)
        spike_times = recording["all_spike_times"]
        for index, (start_s, end_s) in enumerate(sweep.usable_intervals(recording)):
            window = sweep.decode_window(recording, (start_s, end_s), margin_s=0.0)
            key = f"no-replay-recording{recording_seed}-stretch{index}"
            found = [spike_times[cell] for cell in discovery]
            sliding = sweep.run_sliding_window(
                recording, found, window, f"{fold}-{key}"
            )
            state = sweep.run_state_space(recording, found, window)
            for row in score_events(
                fold,
                key,
                None,
                sliding,
                state,
                [spike_times[cell] for cell in evaluation],
                fields,
                window,
                recording,
            ):
                rows.append(row | {"condition": "no_replay"})
    return rows
