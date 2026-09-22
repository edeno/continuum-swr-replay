#!/usr/bin/env python
"""What each detector does to a replay of known shape.

Injects replays of controlled path length, duration, firing rate and distance
from the animal's own place into a simulated recording, and records what the
memoryless sliding-window detector and this project's switching state-space
model each report. Both read the same encoding model, fitted once.

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
import time as clock
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pandas as pd
from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.environment import Environment
from non_local_detector.likelihoods.sorted_spikes_kde import (
    fit_sorted_spikes_kde_encoding_model,
)
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
    interior_bin_distances_cm,
    sorted_spikes_epoch_log_likelihood,
    window_midpoints_s,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.simulation import (
    immobile_intervals,
    inject_replay,
    place_field_means,
    placements,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.statistics import (
    aggregate_windows,
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

MARGIN_S = 0.5
"""Room left on each side of an injected event, inside one immobile stretch."""

NON_LOCAL_PROB_THRESHOLD = 0.8
DISCRETE_TRANSITION_STICKINESS = np.array([30.0, 100000.0, 30.0, 200.0])
SAME_PLACE_CM = 15.0

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

    environment = Environment(place_bin_size=CONFIG.place_bin_size_cm).fit_place_grid(
        position[:, None]
    )
    is_training = ~is_event
    training_spikes = [
        neuron[np.interp(neuron, time, is_training.astype(float)) > 0.5]
        for neuron in spike_times
    ]
    encoding_model = fit_sorted_spikes_kde_encoding_model(
        position_time=time[is_training],
        position=position[is_training, None],
        spike_times=training_spikes,
        environment=environment,
        sampling_frequency=round(CONFIG.sampling_frequency),
        position_std=6.0,
        disable_progress_bar=True,
    )

    detector = NonLocalSortedSpikesDetector(
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={"position_std": 6.0, "block_size": int(2**8)},
        local_position_std=LOCAL_POSITION_STD,
        discrete_transition_stickiness=DISCRETE_TRANSITION_STICKINESS,
    )
    # fit, not estimate_parameters. After EM the detector returns an all-zero
    # posterior when predict is given a different time grid from the one it was
    # fitted on, silently, so every sweep cell would read as "not found". Plain
    # fit uses the stickiness given above instead of estimating it, which also
    # removes a source of variation between cells.
    detector.fit(
        position_time=time,
        position=position,
        spike_times=spike_times,
        is_training=is_training,
    )

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


def memoryless_outcome(recording, injected, window, bounds):
    """What the sliding-window detector reports for one injected event."""
    time, position = window
    result = sorted_spikes_epoch_log_likelihood(
        time, position, injected, recording["encoding_model"], CONFIG
    )
    posterior = normalize_windows(
        aggregate_windows(
            result.log_likelihood, CONFIG.window_n_samples, CONFIG.stride_n_samples
        )
    )
    midpoint_s = window_midpoints_s(result.time, CONFIG)
    speed = np.interp(
        midpoint_s,
        result.time,
        recording["speed"][recording["window_mask"]][: result.time.size],
    )
    candidates = detect_candidate_sequences(
        posterior, midpoint_s, recording["bin_distance_cm"], speed, CONFIG
    )

    start_s, end_s = bounds
    inside = (midpoint_s >= start_s) & (midpoint_s <= end_s)
    gates = candidates.decisions.gates
    overlapping = [
        i
        for i in range(len(candidates.runs))
        if midpoint_s[candidates.runs[i][0]] < end_s
        and start_s < midpoint_s[candidates.runs[i][1]]
    ]
    accepted = [i for i in overlapping if candidates.is_replay[i]]

    out = {
        "ml_found": bool(accepted),
        "ml_n_candidates": len(overlapping),
        "ml_has_center": float(gates.has_center[inside].mean())
        if inside.any()
        else np.nan,
        "ml_spread_ok": (
            float(gates.spread_below_threshold[inside].mean())
            if inside.any()
            else np.nan
        ),
        "ml_jump_ok": (
            float(gates.jump_below_threshold[inside].mean()) if inside.any() else np.nan
        ),
        "ml_best_dispersion_cm": (
            max((candidates.dispersion_cm[i] for i in overlapping), default=np.nan)
        ),
        "ml_best_duration_s": (
            max((candidates.duration_s[i] for i in overlapping), default=np.nan)
        ),
    }
    if accepted:
        i = accepted[0]
        run_start, run_stop = candidates.runs[i]
        centers = candidates.decisions.center_index[run_start : run_stop + 1]
        present = centers >= 0
        animal_cm = np.interp(
            midpoint_s[run_start : run_stop + 1][present],
            result.time,
            recording["position"][recording["window_mask"]][: result.time.size],
        )
        out |= {
            "ml_duration_s": float(candidates.duration_s[i]),
            "ml_dispersion_cm": float(candidates.dispersion_cm[i]),
            "ml_start_error_s": float(midpoint_s[run_start] - start_s),
            "ml_end_error_s": float(midpoint_s[run_stop] - end_s),
            "ml_n_pieces": int(candidates.pieces[i].size),
            "ml_local_fraction": float(
                np.mean(
                    np.abs(
                        recording["place_bin_centers_cm"][centers[present]] - animal_cm
                    )
                    < SAME_PLACE_CM
                )
            ),
        }
    return out


def state_space_outcome(recording, injected, window, bounds):
    """What the switching state-space model reports for the same event."""
    time, position = window
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
    above = probability > NON_LOCAL_PROB_THRESHOLD
    edges = np.diff(above.astype(np.int8), prepend=0, append=0)
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1

    start_s, end_s = bounds
    hits = [
        (time[a], time[b])
        for a, b in zip(starts, stops, strict=True)
        if time[a] <= end_s and start_s <= time[b]
    ]
    inside = (time >= start_s) & (time <= end_s)
    return {
        "ss_found": bool(hits),
        "ss_n_intervals": len(hits),
        "ss_peak_probability": float(probability[inside].max())
        if inside.any()
        else np.nan,
        "ss_duration_s": float(hits[-1][1] - hits[0][0]) if hits else np.nan,
        "ss_start_error_s": float(hits[0][0] - start_s) if hits else np.nan,
        "ss_end_error_s": float(hits[-1][1] - end_s) if hits else np.nan,
    }


def run_cell(recording, means, path_cm, midpoint_cm, duration_s, rate_hz, at_s, seed):
    """One combination: inject, decode with both detectors, summarize."""
    injected, bounds = inject_replay(
        recording["spike_times"],
        means,
        start_cm=midpoint_cm - path_cm / 2,
        path_cm=path_cm,
        duration_s=duration_s,
        peak_rate_hz=rate_hz,
        at_s=at_s,
        sampling_frequency=SAMPLING_FREQUENCY,
        seed=seed,
    )
    start_s, end_s = bounds
    mask = (recording["time"] >= start_s - MARGIN_S) & (
        recording["time"] <= end_s + MARGIN_S
    )
    recording["window_mask"] = mask
    window = (recording["time"][mask], recording["position"][mask])

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
    row |= memoryless_outcome(recording, injected, window, bounds)
    row |= state_space_outcome(recording, injected, window, bounds)
    return row


def summarize(table: pd.DataFrame) -> pd.DataFrame:
    """The aggregates the spec quotes, in one long-format table.

    Built here so the numbers in the write-up and the numbers in the sweep
    cannot drift apart.
    """
    found = table[table.ml_found & table.ss_found].copy()
    found["ml_duration_ratio"] = found.ml_duration_s / found.duration_s
    found["ss_duration_ratio"] = found.ss_duration_s / found.duration_s
    missed = table[table.ss_found & ~table.ml_found]

    parts = []

    by_path_duration = (
        table.groupby(["path_cm", "duration_s"])[["ml_found", "ss_found"]]
        .mean()
        .reset_index()
        .melt(id_vars=["path_cm", "duration_s"], var_name="measure", value_name="value")
    )
    by_path_duration["summary"] = "detection_rate_by_path_and_duration"
    parts.append(by_path_duration)

    by_duration = (
        found.groupby("duration_s")[["ml_duration_ratio", "ss_duration_ratio"]]
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
            ["ml_duration_ratio", "ml_local_fraction", "ss_duration_ratio"]
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
                "ml_found",
                "ss_found",
                "found_by_state_space_only",
                "found_by_sliding_window_only",
                "missed_with_candidate_below_dispersion",
                "missed_with_no_candidate",
                "missed_median_spread_pass_rate",
            ],
            "value": [
                float(len(table)),
                float(table.ml_found.mean()),
                float(table.ss_found.mean()),
                float((table.ss_found & ~table.ml_found).mean()),
                float((table.ml_found & ~table.ss_found).mean()),
                float(
                    (
                        missed.ml_n_candidates.gt(0)
                        & missed.ml_best_dispersion_cm.lt(
                            CONFIG.dispersion_threshold_cm
                        )
                    ).mean()
                ),
                float(missed.ml_n_candidates.eq(0).mean()),
                float(missed.ml_spread_ok.median()),
            ],
        }
    )
    overall["summary"] = "overall"
    parts.append(overall)

    # the grouping column is not called "aggregate": that name shadows
    # DataFrame.aggregate, so frame.aggregate == "x" compares a method and
    # quietly selects nothing
    return pd.concat(parts, ignore_index=True)[
        ["summary", "measure", "path_cm", "duration_s", "distance_bin_cm", "value"]
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
    # keep the labeled events of the recording out of the way
    for event_start, event_end in recording["event_times"]:
        keep = (intervals[:, 1] < event_start - 0.5) | (
            intervals[:, 0] > event_end + 0.5
        )
        intervals = intervals[keep] if keep.any() else intervals
    print(f"setup {clock.time() - started:.1f}s; {len(intervals)} usable stretches")

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
        at_s = placements(intervals, duration_s, MARGIN_S, args.n_seeds, rng)
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
        "n_seeds": args.n_seeds,
        "n_neurons": N_NEURONS,
        "path_cm": list(PATH_CM),
        "duration_s": list(DURATION_S),
        "peak_rate_hz": list(PEAK_RATE_HZ),
        "midpoint_cm": list(MIDPOINT_CM),
        "margin_s": MARGIN_S,
        "same_place_cm": SAME_PLACE_CM,
        "spikes": "sorted",
        "track": "linear",
    }
    write_table(table, args.out)
    summary = summarize(table)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)

    print(f"\n{len(table)} cells in {(clock.time() - started) / 60:.1f} min")
    print(f"  raw      -> {args.out}")
    print(f"  summary  -> {SUMMARY_PATH}")
    print(f"\n  sliding-window found: {table.ml_found.mean():.3f}")
    print(f"  state-space found:    {table.ss_found.mean():.3f}")
    print(f"  found by ours only:   {(table.ss_found & ~table.ml_found).mean():.3f}")
    print(f"  found by theirs only: {(table.ml_found & ~table.ss_found).mean():.3f}")


if __name__ == "__main__":
    main()
