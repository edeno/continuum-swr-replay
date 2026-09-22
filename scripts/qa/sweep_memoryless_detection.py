#!/usr/bin/env python
"""What each detector does to a replay of known shape.

Injects replays of controlled path length, duration, firing rate and distance
from the animal's own place into a simulated recording, and records what the
memoryless sliding-window detector and this project's switching state-space
model each report. Both read the same encoding model: the state-space detector
fits it, and the sliding-window likelihood is handed the same object, so the
two differ in how they use the place fields and not in what the place fields
are.

A detector counts as finding an event only when what it reports is *the
injected event*. Overlapping it in time is not enough: a sequence built from
the local coding beside the injection overlaps it too, and scoring that as a
detection turns the over-call into an apparent detection floor. The rule is
one function, ``scoring.score_against_injection``, applied to every reported
event of either detector that overlaps the injection; the best-scoring one
answers for the cell, so an absorbed sequence that happens to start earlier
cannot hide a true detection behind it.

The point is to separate limits of a method from limits of the data. A replay
that neither detector finds says something about the recording; one that only
the sliding-window detector misses says something about its criteria.

Sorted spikes on a linear track. The clusterless likelihood and the branched
track are not covered here.

Usage:
    python scripts/qa/sweep_memoryless_detection.py
    python scripts/qa/sweep_memoryless_detection.py --n-seeds 2 --out /tmp/pilot.parquet
"""

import argparse
import os
import subprocess
import sys
import time as clock
from datetime import UTC, datetime
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pandas as pd
from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.environment import Environment
from non_local_detector.simulate.sorted_spikes_simulation import (
    SAMPLING_FREQUENCY,
    TRACK_HEIGHT,
    make_simulated_data,
)

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.detect import (
    detect_candidate_sequences,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (
    ENCODING_KEY,
    interior_bin_distances_cm,
    sorted_spikes_epoch_log_likelihood,
    window_midpoints_s,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.scoring import (
    best_scored,
    score_against_injection,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.simulation import (
    InjectedReplay,
    immobile_intervals,
    inject_replay,
    place_field_means,
    placements,
    without,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.statistics import (
    aggregate_windows,
    dispersion_cm,
    normalize_windows,
)
from continuum_swr_replay.parameters import LOCAL_POSITION_STD
from continuum_swr_replay.table_io import write_table

DECODE_BIN_STRIDE = 3
N_NEURONS = 25
CONFIG = MemorylessSequenceConfig(sampling_frequency=500.0)

PATH_CM = (10.0, 20.0, 30.0, 45.0, 60.0, 90.0, 130.0)
DURATION_S = (0.04, 0.06, 0.08, 0.12, 0.20, 0.30, 0.50)
PEAK_RATE_HZ = (20.0, 60.0, 160.0)
MIDPOINT_CM = (30.0, 60.0, 90.0, 120.0, 150.0)
"""Where the represented path is centered, which sets its distance from the animal."""

DECODE_MARGIN_S = 1.0
"""How much time each side of an injected event is decoded.

Wide, so that the window does not cap how long an over-called event can be
reported as. It may reach past the immobile stretch; the speed gate rejects
those windows, which is what it is for."""

PLACEMENT_MARGIN_S = 0.25
"""How much immobile room the injection needs on each side.

Enough for a window of full support plus context, and small enough that the
pieces left after cutting out the recording's own events can still hold an
event."""

NON_LOCAL_PROB_THRESHOLD = 0.8
NON_LOCAL_MINIMUM_DURATION_S = 0.015
"""The project's non-local event definition, as in
``analysis/hippocampus/event_detection.get_confident_times``: non-local
probability above 0.8 for at least 15 ms. The state-space detector is held to
its own published rule here, as the sliding window is held to its."""

SAME_PLACE_CM = 15.0

IN_DOMAIN_PATH_CM = 45.0
IN_DOMAIN_DURATION_S = 0.12
"""The smallest swept path and duration the published replay definition can
accept. Dispersion must exceed 12 cm and a uniform traverse of length L has
dispersion L / sqrt(12), so a path under about 42 cm cannot qualify however
well it is decoded; duration must exceed 100 ms. Three quarters of the grid
lies outside this, by design, to show where each procedure's limit is -- but
the aggregate over the whole grid is not a rate of anything the published
method claims to detect."""

MATCH_TOLERANCE_CM = 20.0
"""How far a detector's decoded position may sit from the injected one.

Generous against a 2 cm grid and a 6 cm smoothing kernel, so it separates
"reported the injected path" from "reported somewhere else entirely" rather
than grading accuracy."""

SUMMARY_PATH = Path(
    "docs/design/replay_detection_comparison/detection_sweep_summary.csv"
)
"""Aggregated results, small enough to keep in the repository.

The raw sweep goes under ``Processed-Data``, which is not tracked. The numbers
quoted in the spec come from this file, so that nobody has to transcribe them
out of a one-off script.
"""


def build_recording():
    """One simulated recording, with both detectors fitted on its non-event data."""
    (
        speed,
        position,
        spike_times,
        time,
        event_times,
        _sampling_frequency,
        is_event,
        _place_fields,
    ) = make_simulated_data(n_neurons=N_NEURONS)
    time, position, speed, is_event = (
        array[::DECODE_BIN_STRIDE] for array in (time, position, speed, is_event)
    )

    is_training = ~is_event

    # the grid both procedures decode on comes from the config, so the jump
    # and spread thresholds in cm and the bins they are measured over cannot
    # drift apart
    detector = NonLocalSortedSpikesDetector(
        environments=[Environment(place_bin_size=CONFIG.place_bin_size_cm)],
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={"position_std": 6.0, "block_size": int(2**8)},
        local_position_std=LOCAL_POSITION_STD,
    )
    # fit, not estimate_parameters. After EM the detector returns an all-zero
    # posterior when predict is given a different time grid from the one it was
    # fitted on, silently, so every sweep cell would read as "not found". The
    # state-space model therefore runs on the library's default discrete
    # transitions: discrete_transition_stickiness is a Dirichlet concentration
    # read only by the EM M-step, so passing one here would change nothing and
    # would read as tuning.
    detector.fit(
        position_time=time,
        position=position,
        spike_times=spike_times,
        is_training=is_training,
    )

    # One set of place fields, not two. The detector's own encoding model holds
    # every entry the sliding-window likelihood needs, so handing it over makes
    # the two procedures read the same object rather than two fits of the same
    # data, which differed by about 4% of peak firing rate.
    encoding_model = detector.encoding_model_[ENCODING_KEY]
    environment = encoding_model["environment"]

    return {
        "time": time,
        "position": position,
        "speed": speed,
        "spike_times": spike_times,
        "event_times": event_times,
        "is_event": is_event,
        "environment": environment,
        "encoding_model": encoding_model,
        "bin_distance_cm": interior_bin_distances_cm(environment),
        "detector": detector,
        "place_bin_centers_cm": np.asarray(environment.place_bin_centers_).ravel()[
            np.asarray(environment.is_track_interior_).ravel()
        ],
    }


def overlaps(reported: tuple[float, float], bounds: tuple[float, float]) -> bool:
    """Strict, positive-duration overlap. The same rule for both detectors."""
    return reported[0] < bounds[1] and bounds[0] < reported[1]


def memoryless_outcome(recording, injected, window, bounds, true_position_cm, path_cm):
    """What the sliding-window detector reports for one injected event."""
    time, position, speed = window
    result = sorted_spikes_epoch_log_likelihood(
        time, position, injected, recording["encoding_model"], CONFIG
    )
    posterior = normalize_windows(
        aggregate_windows(
            result.log_likelihood, CONFIG.window_n_samples, CONFIG.stride_n_samples
        )
    )
    midpoint_s = window_midpoints_s(result.time, CONFIG)
    window_speed = np.interp(midpoint_s, result.time, speed[: result.time.size])
    candidates = detect_candidate_sequences(
        posterior, midpoint_s, recording["bin_distance_cm"], window_speed, CONFIG
    )

    start_s, end_s = bounds
    inside = (midpoint_s >= start_s) & (midpoint_s <= end_s)
    gates = candidates.decisions.gates
    overlapping = [
        i
        for i, (run_start, run_stop) in enumerate(candidates.runs)
        if overlaps((midpoint_s[run_start], midpoint_s[run_stop]), bounds)
    ]
    accepted = [i for i in overlapping if candidates.is_replay[i]]

    def inside_rate(flag):
        return float(flag[inside].mean()) if inside.any() else np.nan

    out = {
        "sliding_found": bool(accepted),
        "sliding_any_candidate": bool(overlapping),
        "sliding_n_candidates": len(overlapping),
        "sliding_n_accepted": len(accepted),
        "sliding_has_center": inside_rate(gates.has_center),
        "sliding_posterior_usable": inside_rate(gates.posterior_usable),
        "sliding_speed_eligible": inside_rate(gates.speed_eligible),
        "sliding_spread_ok": inside_rate(gates.spread_below_threshold),
        "sliding_jump_ok": inside_rate(gates.jump_below_threshold),
        "sliding_best_dispersion_cm": max(
            (candidates.dispersion_cm[i] for i in overlapping), default=np.nan
        ),
        "sliding_best_duration_s": max(
            (candidates.duration_s[i] for i in overlapping), default=np.nan
        ),
        "sliding_matches": False,
        "sliding_position_error_cm": np.nan,
        "sliding_span_cm": np.nan,
    }
    if not accepted:
        return out

    # Every accepted sequence that overlaps the injection is scored, and the
    # best answers for the cell. Two things have to hold for a match. The
    # sequence must sit on the path while the path is being replayed, and it
    # must not cover far more ground than the path does: an absorbed sequence
    # tracks the injection during the injection and then runs off to the
    # animal's own place, which is where its dispersion comes from.
    scored = []
    for i in accepted:
        run_start, run_stop = candidates.runs[i]
        centers = candidates.decisions.center_index[run_start : run_stop + 1]
        times = midpoint_s[run_start : run_stop + 1]
        present = centers >= 0
        decoded_cm = recording["place_bin_centers_cm"][centers[present]]
        scored.append(
            (
                i,
                score_against_injection(
                    times[present],
                    decoded_cm,
                    true_position_cm,
                    bounds,
                    path_cm,
                    MATCH_TOLERANCE_CM,
                ),
                decoded_cm,
                times[present],
            )
        )
    i, score, decoded_cm, times = scored[best_scored([item[1] for item in scored])]
    run_start, run_stop = candidates.runs[i]

    animal_cm = np.interp(times, result.time, position[: result.time.size])
    out |= {
        "sliding_matches": score.matches,
        "sliding_position_error_cm": score.position_error_cm,
        "sliding_span_cm": score.span_cm,
        "sliding_duration_s": float(candidates.duration_s[i]),
        "sliding_dispersion_cm": float(candidates.dispersion_cm[i]),
        "sliding_start_error_s": float(midpoint_s[run_start] - start_s),
        "sliding_end_error_s": float(midpoint_s[run_stop] - end_s),
        "sliding_n_pieces": int(candidates.pieces[i].size),
        "sliding_local_fraction": float(
            np.mean(np.abs(decoded_cm - animal_cm) < SAME_PLACE_CM)
        ),
    }
    return out


def nearest_bin(recording, position_cm: np.ndarray) -> np.ndarray:
    """Interior place bin nearest each decoded position.

    The state-space posterior carries one position axis per state, so its
    argmax is an index into a concatenation. Mapping back to interior bins
    lets the state-space trajectory be measured with the same
    `dispersion_cm` over the same track-graph distances as the sliding
    window's.
    """
    centers = recording["place_bin_centers_cm"]
    return np.abs(centers[None, :] - np.asarray(position_cm)[:, None]).argmin(axis=1)


def non_local_intervals(
    time: np.ndarray, probability: np.ndarray
) -> list[tuple[int, int]]:
    """The state-space detector's events: index bounds of every stretch above
    threshold that lasts the project's minimum duration."""
    above = probability > NON_LOCAL_PROB_THRESHOLD
    edges = np.diff(above.astype(np.int8), prepend=0, append=0)
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    return [
        (a, b)
        for a, b in zip(starts, stops, strict=True)
        if time[b] - time[a] >= NON_LOCAL_MINIMUM_DURATION_S
    ]


def state_space_outcome(recording, injected, window, bounds, true_position_cm, path_cm):
    """What the switching state-space model reports for the same event.

    Scored by the same rule as the sliding window: every reported event that
    overlaps the injection is scored on its own, over its whole extent, and
    the best answers for the cell.
    """
    time, position, _speed = window
    results = recording["detector"].predict(
        position_time=time, position=position, spike_times=injected, time=time
    )
    everything = np.asarray(results.acausal_state_probabilities.values)
    if not np.isfinite(everything).all() or everything.max() <= 0.0:
        raise RuntimeError(
            "The state-space model returned no probability mass. Its state "
            "probabilities must sum to one at every time. A detector fitted with "
            "estimate_parameters does this silently when predict is given another "
            "time grid."
        )
    probability = np.asarray(
        results.acausal_state_probabilities.sel(
            states=["Non-Local Continuous", "Non-Local Fragmented"]
        )
        .sum("states")
        .values
    ).squeeze()

    start_s, end_s = bounds
    hits = [
        (a, b)
        for a, b in non_local_intervals(time, probability)
        if overlaps((time[a], time[b]), bounds)
    ]
    inside = (time >= start_s) & (time <= end_s)

    out = {
        "state_space_found": bool(hits),
        "state_space_n_intervals": len(hits),
        "state_space_peak_probability": (
            float(probability[inside].max()) if inside.any() else np.nan
        ),
        "state_space_matches": False,
        "state_space_replay_matches": False,
        "state_space_position_error_cm": np.nan,
        "state_space_span_cm": np.nan,
        "state_space_dispersion_cm": np.nan,
    }
    if not hits:
        return out

    # where the non-local states put the animal, to score this the same way
    posterior = results["acausal_posterior"]
    is_non_local = np.isin(
        np.asarray(posterior.coords["state"].values),
        ["Non-Local Continuous", "Non-Local Fragmented"],
    )
    bin_position_cm = np.asarray(posterior.coords["position"].values)[is_non_local]
    non_local = np.asarray(posterior.values)[:, is_non_local]

    scored = []
    for a, b in hits:
        decoded_cm = bin_position_cm[np.argmax(non_local[a : b + 1], axis=1)]
        scored.append(
            (
                (a, b),
                score_against_injection(
                    time[a : b + 1],
                    decoded_cm,
                    true_position_cm,
                    bounds,
                    path_cm,
                    MATCH_TOLERANCE_CM,
                ),
                decoded_cm,
            )
        )
    (a, b), score, decoded_cm = scored[best_scored([item[1] for item in scored])]

    out |= {
        "state_space_matches": score.matches,
        "state_space_position_error_cm": score.position_error_cm,
        "state_space_span_cm": score.span_cm,
        "state_space_duration_s": float(time[b] - time[a]),
        "state_space_start_error_s": float(time[a] - start_s),
        "state_space_end_error_s": float(time[b] - end_s),
        "state_space_dispersion_cm": dispersion_cm(
            nearest_bin(recording, decoded_cm), recording["bin_distance_cm"]
        ),
    }
    # and again under the published replay definition the sliding window is
    # held to, so the two rates can be read against each other
    out["state_space_replay_matches"] = bool(
        score.matches
        and out["state_space_duration_s"] > CONFIG.minimum_duration_s
        and out["state_space_dispersion_cm"] > CONFIG.dispersion_threshold_cm
    )
    return out


def run_cell(recording, means, path_cm, midpoint_cm, duration_s, rate_hz, at_s, seed):
    """One combination: inject, decode with both detectors, summarize."""
    injected, bounds = inject_replay(
        recording["spike_times"],
        means,
        InjectedReplay(
            start_cm=midpoint_cm - path_cm / 2,
            path_cm=path_cm,
            duration_s=duration_s,
            peak_rate_hz=rate_hz,
            at_s=at_s,
            seed=seed,
        ),
        sampling_frequency=SAMPLING_FREQUENCY,
    )
    start_s, end_s = bounds
    mask = (recording["time"] >= start_s - DECODE_MARGIN_S) & (
        recording["time"] <= end_s + DECODE_MARGIN_S
    )
    window = (
        recording["time"][mask],
        recording["position"][mask],
        recording["speed"][mask],
    )

    original = np.concatenate(recording["spike_times"])
    added = np.concatenate(injected)
    n_spikes = int(
        ((added >= start_s) & (added <= end_s)).sum()
        - ((original >= start_s) & (original <= end_s) & np.isin(original, added)).sum()
    )
    row = {
        "path_cm": path_cm,
        "midpoint_cm": midpoint_cm,
        "duration_s": duration_s,
        "peak_rate_hz": rate_hz,
        "seed": seed,
        "at_s": at_s,
        "speed_cm_per_s": path_cm / duration_s,
        "distance_from_animal_cm": abs(
            midpoint_cm
            - float(np.interp(at_s, recording["time"], recording["position"]))
        ),
        "n_spikes": n_spikes,
        "spikes_per_window": n_spikes * CONFIG.window_duration_s / duration_s,
    }

    def true_position_cm(at):
        """Where the injected replay says the animal is, at these times."""
        fraction = np.clip((np.asarray(at) - start_s) / (end_s - start_s), 0, 1)
        return midpoint_cm - path_cm / 2 + path_cm * fraction

    row |= memoryless_outcome(
        recording, injected, window, bounds, true_position_cm, path_cm
    )
    row |= state_space_outcome(
        recording, injected, window, bounds, true_position_cm, path_cm
    )
    return row


def git_commit() -> str:
    """The commit this ran from, so a saved table can be tied back to the code."""
    try:
        described = subprocess.run(
            ["git", "describe", "--always", "--dirty", "--abbrev=12"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        print(
            f"warning: git commit unknown ({error}); recording 'unknown'",
            file=sys.stderr,
        )
        return "unknown"
    commit = described.stdout.strip()
    if commit.endswith("-dirty"):
        print(
            f"warning: the working tree has uncommitted changes ({commit}); the "
            "numbers this run saves cannot be tied to a commit",
            file=sys.stderr,
        )
    return commit


def share(numerator: float, denominator: float) -> float:
    """A rate that is undefined, not zero, when nothing was counted."""
    return float(numerator / denominator) if denominator > 0 else np.nan


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """The aggregates the spec quotes, in one long-format table.

    Built here so the numbers in the write-up and the numbers in the sweep
    cannot drift apart.
    """
    found = table[table.sliding_matches & table.state_space_matches].copy()
    found["sliding_duration_ratio"] = found.sliding_duration_s / found.duration_s
    found["state_space_duration_ratio"] = (
        found.state_space_duration_s / found.duration_s
    )
    missed = table[table.state_space_matches & ~table.sliding_matches]

    parts = []

    by_path_duration = (
        table.groupby(["path_cm", "duration_s"])[
            [
                "sliding_found",
                "sliding_matches",
                "sliding_any_candidate",
                "state_space_found",
                "state_space_matches",
            ]
        ]
        .mean()
        .reset_index()
        .melt(id_vars=["path_cm", "duration_s"], var_name="measure", value_name="value")
    )
    by_path_duration["summary"] = "detection_rate_by_path_and_duration"
    parts.append(by_path_duration)

    by_duration = (
        found.groupby("duration_s")[
            ["sliding_duration_ratio", "state_space_duration_ratio"]
        ]
        .median()
        .reset_index()
        .melt(id_vars="duration_s", var_name="measure", value_name="value")
    )
    by_duration["summary"] = "median_duration_ratio_by_duration"
    parts.append(by_duration)

    found["distance_bin_cm"] = pd.cut(
        found.distance_from_animal_cm, [0, 40, 80, 120, 200]
    ).astype(str)
    by_distance = (
        found.groupby("distance_bin_cm", observed=True)[
            [
                "sliding_duration_ratio",
                "sliding_local_fraction",
                "state_space_duration_ratio",
            ]
        ]
        .median()
        .reset_index()
        .melt(id_vars="distance_bin_cm", var_name="measure", value_name="value")
    )
    by_distance["summary"] = "median_by_distance_from_animal"
    parts.append(by_distance)

    overall = pd.DataFrame(
        {
            "measure": [
                "n_cells",
                "sliding_found",
                "sliding_matches",
                "sliding_any_candidate",
                "state_space_found",
                "state_space_matches",
                "state_space_replay_matches",
                "found_by_state_space_only",
                "found_by_sliding_window_only",
                "sliding_accepted_but_not_the_injection",
                "sliding_cells_with_several_accepted_overlaps",
                "missed_with_candidate_below_dispersion",
                "missed_with_no_candidate",
                "missed_median_spread_pass_rate",
            ],
            "value": [
                float(len(table)),
                float(table.sliding_found.mean()),
                float(table.sliding_matches.mean()),
                float(table.sliding_any_candidate.mean()),
                float(table.state_space_found.mean()),
                float(table.state_space_matches.mean()),
                float(table.state_space_replay_matches.mean()),
                float((table.state_space_matches & ~table.sliding_matches).mean()),
                float((table.sliding_matches & ~table.state_space_matches).mean()),
                share(
                    (table.sliding_found & ~table.sliding_matches).sum(),
                    table.sliding_found.sum(),
                ),
                float(table.sliding_n_accepted.gt(1).mean()),
                float(
                    (
                        missed.sliding_n_candidates.gt(0)
                        & missed.sliding_best_dispersion_cm.lt(
                            CONFIG.dispersion_threshold_cm
                        )
                    ).mean()
                ),
                float(missed.sliding_n_candidates.eq(0).mean()),
                float(missed.sliding_spread_ok.median()),
            ],
        }
    )
    overall["summary"] = "overall"
    parts.append(overall)

    in_domain = table[
        (table.path_cm >= IN_DOMAIN_PATH_CM)
        & (table.duration_s >= IN_DOMAIN_DURATION_S)
    ]
    domain = pd.DataFrame(
        {
            "measure": [
                "n_cells",
                "sliding_matches",
                "state_space_matches",
                "state_space_replay_matches",
            ],
            "value": [
                float(len(in_domain)),
                float(in_domain.sliding_matches.mean()),
                float(in_domain.state_space_matches.mean()),
                float(in_domain.state_space_replay_matches.mean()),
            ],
        }
    )
    domain["summary"] = "overall_in_domain"
    parts.append(domain)

    # the grouping column is not called "aggregate": that name shadows
    # DataFrame.aggregate, so frame.aggregate == "x" compares a method and
    # quietly selects nothing
    table = pd.concat(parts, ignore_index=True)
    table["note"] = ""
    return table[
        [
            "summary",
            "measure",
            "path_cm",
            "duration_s",
            "distance_bin_cm",
            "value",
            "note",
        ]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("Processed-Data/replay_detection_comparison")
        / "memoryless_detection_sweep.parquet",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=SUMMARY_PATH,
        help="Where the aggregates go. Point a pilot elsewhere so it does not "
        "overwrite the committed summary.",
    )
    args = parser.parse_args()

    started = clock.time()
    recording = build_recording()
    means = place_field_means(N_NEURONS, TRACK_HEIGHT)
    intervals = immobile_intervals(
        recording["time"],
        recording["speed"],
        CONFIG.speed_threshold_cm_per_s,
        minimum_duration_s=1.0,
    )
    # cut the recording's own labeled events out of the stretches rather than
    # discarding whole stretches. Every stretch holds one, so discarding them
    # leaves nothing, and a silent fallback then kept whichever happened to be
    # last -- one stretch, at one animal position, for every cell of the sweep.
    intervals = without(
        intervals, recording["event_times"], margin_s=PLACEMENT_MARGIN_S
    )
    if len(intervals) == 0:
        raise RuntimeError("No immobile stretch survives excluding the labeled events.")
    print(
        f"setup {clock.time() - started:.1f}s; {len(intervals)} usable stretches "
        f"spanning {np.diff(intervals, axis=1).sum():.1f}s"
    )

    combinations = [
        (path_cm, midpoint_cm, duration_s, rate_hz)
        for path_cm in PATH_CM
        for midpoint_cm in MIDPOINT_CM
        for duration_s in DURATION_S
        for rate_hz in PEAK_RATE_HZ
        if 0.0 <= midpoint_cm - path_cm / 2
        and midpoint_cm + path_cm / 2 <= TRACK_HEIGHT
    ]
    print(
        f"{len(combinations)} combinations x {args.n_seeds} seeds = "
        f"{len(combinations) * args.n_seeds} cells"
    )

    rng = np.random.default_rng(0)
    rows, started = [], clock.time()
    for index, (path_cm, midpoint_cm, duration_s, rate_hz) in enumerate(combinations):
        at_s = placements(intervals, duration_s, PLACEMENT_MARGIN_S, args.n_seeds, rng)
        for seed in range(args.n_seeds):
            rows.append(
                run_cell(
                    recording,
                    means,
                    path_cm,
                    midpoint_cm,
                    duration_s,
                    rate_hz,
                    float(at_s[seed]),
                    seed,
                )
            )
        if index % 25 == 0:
            done = (index + 1) / len(combinations)
            elapsed = clock.time() - started
            print(
                f"  {100 * done:5.1f}%  {elapsed / 60:5.1f} min elapsed, "
                f"{elapsed / max(done, 1e-9) / 60 - elapsed / 60:5.1f} min left",
                flush=True,
            )

    table = pd.DataFrame(rows)
    table.attrs = {
        "produced_by": "scripts/qa/sweep_memoryless_detection.py",
        "produced_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "design_spec": "docs/design/replay_detection_comparison/02-memoryless-sequence.md",
        "n_seeds": args.n_seeds,
        "n_neurons": N_NEURONS,
        "path_cm": list(PATH_CM),
        "duration_s": list(DURATION_S),
        "peak_rate_hz": list(PEAK_RATE_HZ),
        "midpoint_cm": list(MIDPOINT_CM),
        "decode_margin_s": DECODE_MARGIN_S,
        "placement_margin_s": PLACEMENT_MARGIN_S,
        "match_tolerance_cm": MATCH_TOLERANCE_CM,
        "same_place_cm": SAME_PLACE_CM,
        "non_local_prob_threshold": NON_LOCAL_PROB_THRESHOLD,
        "non_local_minimum_duration_s": NON_LOCAL_MINIMUM_DURATION_S,
        "spikes": "sorted",
        "track": "linear",
    }
    write_table(table, args.out)
    summary = summarize(table)
    for key in ("produced_by", "produced_at", "git_commit", "n_cells_total"):
        summary.loc[len(summary)] = {
            "summary": "provenance",
            "measure": key,
            "value": np.nan,
            "note": {
                "produced_by": table.attrs["produced_by"],
                "produced_at": table.attrs["produced_at"],
                "git_commit": table.attrs["git_commit"],
                "n_cells_total": str(len(table)),
            }[key],
        }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)

    print(f"\n{len(table)} cells in {(clock.time() - started) / 60:.1f} min")
    print(f"  raw      -> {args.out}")
    print(f"  summary  -> {args.summary}")
    print(f"\n  sliding window, native criterion : {table.sliding_found.mean():.3f}")
    print(f"  sliding window, and it is the event: {table.sliding_matches.mean():.3f}")
    print(
        f"  sliding window, any candidate      : {table.sliding_any_candidate.mean():.3f}"
    )
    print(
        f"  state space, native criterion      : {table.state_space_found.mean():.3f}"
    )
    print(
        f"  state space, and it is the event   : {table.state_space_matches.mean():.3f}"
    )
    print(
        "  of the sliding window's accepted events, the share that are not the "
        f"injection: {share((table.sliding_found & ~table.sliding_matches).sum(), table.sliding_found.sum()):.3f}"
    )


if __name__ == "__main__":
    main()
