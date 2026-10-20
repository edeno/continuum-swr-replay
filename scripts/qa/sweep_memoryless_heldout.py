#!/usr/bin/env python
"""Does each detector's content reproduce in cells it never saw?

Held-out evidence on simulated data, where the truth of every event is known,
so the held-out test itself can be calibrated before it is used on recordings.
The simulated recording's 25 cells are split into discovery and evaluation
cells three ways: two interleaved folds, roles swapped, each half covering the
track; and one whose evaluation cells cover only its lower half. In each fold
both detectors are fitted and run on the discovery cells alone, on every
injected replay of the sweep and every trial of the controls; the evaluation
cells' fields are fitted separately, on the same training data.

The no-replay stretches of the controls, eight recordings of the animal
resting with nothing injected, are run the same way in every fold, so held-out
support on pure local coding is measured too.

Every event either detector reports (our non-local events, the sliding window
under the published definition) is classed as shared, ours only or the
comparator's only, labelled with what it truly is, and scored by the one
held-out rule of ``inventory_comparison.heldout``, under both content
estimates: our decoder's non-local posterior and the sliding window's own
posterior, each from the discovery cells. What comes out is how often held-out
support falls on events with spatial content and how often on events without
it, for each class and fold: the error rates held-out results on recordings
will be read with.

Usage:
    python scripts/qa/sweep_memoryless_heldout.py
    python scripts/qa/sweep_memoryless_heldout.py --n-seeds 1 --folds A --out /tmp/pilot.parquet
"""

import argparse
import sys
import time as clock
import zlib
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from non_local_detector.likelihoods.common import (  # noqa: E402
    get_spikecount_per_time_bin,
)

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (  # noqa: E402
    with_reference_rate_floor,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.scoring import (  # noqa: E402
    score_against_injection,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.simulation import (  # noqa: E402
    Segment,
    inject_activity,
    piecewise_replay,
    place_field_means,
    placements,
)
from continuum_swr_replay.analysis.inventory_comparison.heldout import (  # noqa: E402
    COVERAGE_FLOOR,
    event_classes,
    heldout_score,
    interleaved_folds,
    nearest_window_content,
    poor_coverage_fold,
)
from continuum_swr_replay.paths import DETECTION_COMPARISON_DIR
from continuum_swr_replay.tables.io import write_table  # noqa: E402
from scripts.qa import sweep_memoryless_controls as controls  # noqa: E402
from scripts.qa import sweep_memoryless_detection as sweep  # noqa: E402

N_PERMUTATIONS = 100
COVERAGE_SENSITIVITY = (2.0, COVERAGE_FLOOR, 10.0)
PLACEMENT_SEED = 2

OUT_PATH = DETECTION_COMPARISON_DIR / "memoryless_heldout.parquet"
SUMMARY_PATH = Path(
    "docs/design/replay_detection_comparison/detection_heldout_summary.csv"
)


def folds() -> dict:
    return interleaved_folds(sweep.N_NEURONS) | poor_coverage_fold(sweep.N_NEURONS)


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
        with_reference_rate_floor(held["encoding_model"], sweep.CONFIG)["place_fields"],
        dtype=np.float64,
    )[:, interior]
    return found, fields


def trials(means: np.ndarray):
    """The sweep's injected replays and the controls' trials, as the controls
    list them: name, grid level, peak rate, builder, and pieces."""
    sampling_frequency = sweep.SAMPLING_FREQUENCY
    for path_cm in sweep.PATH_CM:
        for midpoint_cm in sweep.MIDPOINT_CM:
            if (
                not 0
                <= midpoint_cm - path_cm / 2
                <= midpoint_cm + path_cm / 2
                <= sweep.TRACK_HEIGHT
            ):
                continue
            for duration_s in sweep.DURATION_S:
                for rate_hz in sweep.PEAK_RATE_HZ:

                    def make(
                        at_s, seed, p=path_cm, m=midpoint_cm, d=duration_s, r=rate_hz
                    ):
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
    yield from controls.trials(means, sampling_frequency)


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
        controls.truth_path(activity),
        activity.bounds,
        controls.covered_cm(activity),
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
    theirs = [sliding.events[i] for i in sliding.accepted if sliding.shuffles[i].passes]
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
    recording, fields = fold_models(discovery, evaluation)
    means = place_field_means(sweep.N_NEURONS, sweep.TRACK_HEIGHT)
    intervals = controls.usable_intervals(recording)
    rng = np.random.default_rng(PLACEMENT_SEED)
    rows = []
    for condition, level, rate_hz, make, _ in trials(means):
        span_s = make(0.0, 0).bounds[1]
        at_s = placements(intervals, span_s, sweep.PLACEMENT_MARGIN_S, n_seeds, rng)
        for seed in range(n_seeds):
            activity = make(float(at_s[seed]), seed)
            injected = inject_activity(recording["all_spike_times"], activity, seed)
            start_s, end_s = activity.bounds
            mask = (recording["time"] >= start_s - sweep.DECODE_MARGIN_S) & (
                recording["time"] <= end_s + sweep.DECODE_MARGIN_S
            )
            window = tuple(recording[k][mask] for k in ("time", "position", "speed"))
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
        for index, (start_s, end_s) in enumerate(controls.usable_intervals(recording)):
            mask = (recording["time"] >= start_s) & (recording["time"] <= end_s)
            window = tuple(recording[k][mask] for k in ("time", "position", "speed"))
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


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """Support rates by fold, estimator, class and truth, and the error rates."""
    rows = []

    def add(summary, measure, fold, estimator, detector, event_class, truth, value):
        rows.append(
            {
                "summary": summary,
                "measure": measure,
                "fold": fold,
                "estimator": estimator,
                "detector": detector,
                "event_class": event_class,
                "truth": truth,
                "value": float(value),
            }
        )

    keys = ["fold", "estimator", "detector"]
    for (fold, estimator, detector, event_class, truth), group in table.groupby(
        [*keys, "event_class", "truth"]
    ):
        where = (fold, estimator, detector, event_class, truth)
        add("by_class", "n_events_all", *where, len(group))
        add(
            "by_class",
            "informative_share",
            *where,
            (group.coverage >= COVERAGE_FLOOR).mean(),
        )
        add("by_class", "median_gain", *where, group.gain_vs_animal.median())
        add("by_class", "median_coverage", *where, group.coverage.median())
        for floor in COVERAGE_SENSITIVITY:
            suffix = "" if floor == COVERAGE_FLOOR else f"_floor{floor:g}"
            informative = group[group.coverage >= floor]
            add("by_class", f"n_events{suffix}", *where, len(informative))
            add(
                "by_class",
                f"support_rate{suffix}",
                *where,
                informative.supported.mean() if len(informative) else np.nan,
            )

    kinds = {
        "false_support_rate": ["local", "no_content", "silent_gap"],
        "support_rate_with_content": ["continuous_replay", "non_local"],
        "support_rate_mixed": ["mixed"],
    }
    for (fold, estimator, detector), group in table.groupby(keys):
        for floor in COVERAGE_SENSITIVITY:
            suffix = "" if floor == COVERAGE_FLOOR else f"_floor{floor:g}"
            informative = group[group.coverage >= floor]
            for measure, truths in kinds.items():
                chosen = informative[informative.truth.isin(truths)]
                where = (fold, estimator, detector, "all", "")
                add("error_rates", f"n_{measure}{suffix}", *where, len(chosen))
                add(
                    "error_rates",
                    f"{measure}{suffix}",
                    *where,
                    chosen.supported.mean() if len(chosen) else np.nan,
                )
            with_content = informative[
                informative.truth.isin(kinds["support_rate_with_content"])
            ]
            add(
                "error_rates",
                f"under_crediting{suffix}",
                fold,
                estimator,
                detector,
                "all",
                "",
                1 - with_content.supported.mean() if len(with_content) else np.nan,
            )

    # the two estimators' verdicts on the same event, among events informative
    # under both
    verdicts = (
        table[table.coverage >= COVERAGE_FLOOR]
        .pivot_table(
            index=["fold", "trial", "detector", "event"],
            columns="estimator",
            values="supported",
            aggfunc="first",
        )
        .reindex(columns=["state_space", "sliding_window"])
        .dropna()
    )
    for fold, group in verdicts.groupby(level="fold"):
        add(
            "agreement",
            "estimator_agreement",
            fold,
            "both",
            "both",
            "all",
            "",
            (group["state_space"] == group["sliding_window"]).mean(),
        )
    summary = pd.DataFrame(rows)
    summary["note"] = ""
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument("--folds", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--n-recordings", type=int, default=controls.N_RECORDINGS)
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args()

    started = clock.time()
    rows = []
    for fold in args.folds:
        discovery, evaluation = folds()[fold]
        rows += run_fold(fold, discovery, evaluation, args.n_seeds)
        rows += run_no_replay_fold(fold, discovery, evaluation, args.n_recordings)
        print(
            f"fold {fold}: {len(rows)} rows, {(clock.time() - started) / 60:.1f} min",
            flush=True,
        )
    table = pd.DataFrame(rows)
    table.attrs = {
        "produced_by": "scripts/qa/sweep_memoryless_heldout.py",
        "produced_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": sweep.git_commit(),
        "n_seeds": args.n_seeds,
        "n_permutations": N_PERMUTATIONS,
        "n_recordings": args.n_recordings,
    }
    write_table(table, args.out)
    summary = summarize(table)
    for measure in ("produced_by", "produced_at", "git_commit"):
        summary.loc[len(summary)] = {
            "summary": "provenance",
            "measure": measure,
            "value": np.nan,
            "note": str(table.attrs[measure]),
        }
    for measure in ("n_seeds", "n_permutations", "n_recordings"):
        summary.loc[len(summary)] = {
            "summary": "provenance",
            "measure": measure,
            "value": float(table.attrs[measure]),
            "note": "",
        }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)
    print(f"\n  raw     -> {args.out}\n  summary -> {args.summary}")
    print(f"  total {(clock.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
