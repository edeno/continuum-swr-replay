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
one function, ``scoring.score_against_injection``: over the whole reported
event the decoded positions must sit on the injected path (its nearer end,
outside the injection) and must not cover far more ground than the path. It
is applied to every reported event of either detector that overlaps the
injection, and the best-scoring one answers for the cell, so an absorbed
sequence that happens to start earlier cannot hide a true detection behind it.

The point is to separate limits of a method from limits of the data. A replay
that neither detector finds says something about the recording; one that only
the sliding-window detector misses says something about its criteria.

Sorted spikes on a linear track. The clusterless likelihood and the branched
track are not covered here.

Usage:
    python scripts/detection/sweep_memoryless_detection.py
    python scripts/detection/sweep_memoryless_detection.py --n-seeds 2 --out /tmp/pilot.parquet
"""

import argparse
import time as clock
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from non_local_detector.simulate.sorted_spikes_simulation import (
    SAMPLING_FREQUENCY,
    TRACK_HEIGHT,
)

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (
    REFERENCE_RATE_FLOOR_HZ,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.scoring import (
    best_scored,
    score_against_injection,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.shuffle import (
    DEFAULT_N_SHUFFLES,
    SIGNIFICANCE_LEVEL,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.simulation import (
    InjectedReplay,
    immobile_intervals,
    inject_replay,
    place_field_means,
    placements,
    without,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.sweep import (
    CONFIG,
    DECODE_MARGIN_S,
    DURATION_S,
    MATCH_TOLERANCE_CM,
    MIDPOINT_CM,
    N_NEURONS,
    NON_LOCAL_MINIMUM_DURATION_S,
    NON_LOCAL_PROB_THRESHOLD,
    PATH_CM,
    PEAK_RATE_HZ,
    PLACEMENT_MARGIN_S,
    SHUFFLE_RUN_SEED,
    SlidingWindowRun,
    StateSpaceRun,
    build_recording,
    nearest_bin,
    overlaps,
    run_sliding_window,
    run_state_space,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.sweep_summaries import (
    replay_sweep_summary,
    share,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.window_statistics import (
    dispersion_cm,
)
from continuum_swr_replay.paths import detection_comparison_dir
from continuum_swr_replay.runs.device import default_jax_platforms
from continuum_swr_replay.runs.provenance import described_commit, produced_by
from continuum_swr_replay.tables.io import write_table

SAME_PLACE_CM = 15.0

SUMMARY_PATH = Path("paper/reported/inputs/detection/detection_sweep_summary.csv")
"""Aggregated results, small enough to keep in the repository.

The raw sweep goes under ``Processed-Data``, which is not tracked. Quoted
numbers come from this file, so that nobody has to transcribe them out of a
one-off script.
"""


def best_entry(scored: list[tuple]) -> tuple:
    """The scored event that answers for the injection.

    Both detectors score every reported event that overlaps the injection and
    keep whatever else they need beside it; each entry carries its score
    second.
    """
    return scored[best_scored([entry[1] for entry in scored])]


def score_sliding_window(
    run: SlidingWindowRun, bounds, true_position_cm, path_cm, position
) -> dict:
    """What the sliding-window detector reports for one injected event.

    Scored three ways, because the published definition has three parts and
    the reference's detection script applied only some of them.
    ``sliding_matches`` is the whole published definition: a candidate that
    meets the duration and dispersion criteria, is the injected replay, and
    beats the place-cell identity shuffle on both mean spread and mean jump.
    That is what the paper's analyses read, from the corrected
    ``load_replayEvents_cellIDshuffle_sameEvents.m``. The detection script
    itself computed its jump p-values from an all-NaN vector, so every event
    passed that half; ``sliding_matches_spread_only`` is what its own filter
    applied.
    ``sliding_matches_before_shuffle`` leaves the shuffle out.
    """
    candidates, midpoint_s = run.candidates, run.midpoint_s
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
        "sliding_matches_before_shuffle": False,
        "sliding_matches_spread_only": False,
        "sliding_p_spread": np.nan,
        "sliding_p_jump": np.nan,
        "sliding_position_error_cm": np.nan,
        "sliding_span_cm": np.nan,
    }
    if not accepted:
        return out

    # Every accepted sequence that overlaps the injection is scored, and the
    # best answers for the cell. Two things have to hold over the whole
    # sequence for a match: its decoded positions sit on the path, and it does
    # not cover far more ground than the path does. An absorbed sequence
    # tracks the injection during the injection and then sits at the animal's
    # own place, which is where its dispersion comes from.
    scored = [
        (
            i,
            score_against_injection(
                run.events[i].decoded_time_s,
                run.events[i].decoded_cm,
                true_position_cm,
                bounds,
                path_cm,
                MATCH_TOLERANCE_CM,
            ),
        )
        for i in accepted
    ]
    shuffles = run.shuffles

    def any_match(keep) -> bool:
        return any(entry[1].matches for entry in scored if keep(entry[0]))

    i, score = best_entry(scored)
    event = run.events[i]
    animal_cm = np.interp(event.decoded_time_s, run.time, position[: run.time.size])
    out |= {
        "sliding_matches_before_shuffle": score.matches,
        "sliding_matches": any_match(lambda j: shuffles[j].passes),
        "sliding_matches_spread_only": any_match(
            lambda j: shuffles[j].p_spread <= SIGNIFICANCE_LEVEL
        ),
        "sliding_p_spread": shuffles[i].p_spread,
        "sliding_p_jump": shuffles[i].p_jump,
        "sliding_position_error_cm": score.position_error_cm,
        "sliding_span_cm": score.span_cm,
        "sliding_duration_s": float(candidates.duration_s[i]),
        "sliding_dispersion_cm": float(candidates.dispersion_cm[i]),
        "sliding_start_error_s": float(event.start_s - start_s),
        "sliding_end_error_s": float(event.end_s - end_s),
        "sliding_n_pieces": int(candidates.pieces[i].size),
        "sliding_local_fraction": float(
            np.mean(np.abs(event.decoded_cm - animal_cm) < SAME_PLACE_CM)
        ),
    }
    return out


def memoryless_outcome(
    recording, injected, window, bounds, true_position_cm, path_cm, cell_key
):
    """The sliding-window detector run over the window and scored against the event."""
    run = run_sliding_window(recording, injected, window, cell_key)
    return score_sliding_window(run, bounds, true_position_cm, path_cm, window[1])


def score_state_space(
    recording, run: StateSpaceRun, bounds, true_position_cm, path_cm
) -> dict:
    """What the switching state-space model reports for the same event.

    Scored by the same rule as the sliding window: every reported event that
    overlaps the injection is scored on its own, over its whole extent, and
    the best answers for the cell.
    """
    time = run.time
    probability = run.p_continuous + run.p_fragmented
    start_s, end_s = bounds
    hits = [
        event
        for event in run.canonical
        if overlaps((event.start_s, event.end_s), bounds)
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

    scored = []
    for event in hits:
        duration_s = event.end_s - event.start_s
        dispersion = dispersion_cm(
            nearest_bin(recording, event.decoded_cm), recording["bin_distance_cm"]
        )
        score = score_against_injection(
            event.decoded_time_s,
            event.decoded_cm,
            true_position_cm,
            bounds,
            path_cm,
            MATCH_TOLERANCE_CM,
        )
        # and again under the published replay definition the sliding window
        # is held to, so the two rates can be read against each other; judged
        # per event, as the sliding window's accepted set is, before the best
        # is chosen
        meets_replay_definition = bool(
            score.matches
            and duration_s > CONFIG.minimum_duration_s
            and dispersion > CONFIG.dispersion_threshold_cm
        )
        scored.append((event, score, duration_s, dispersion, meets_replay_definition))
    event, score, duration_s, dispersion, _ = best_entry(scored)

    out |= {
        "state_space_matches": score.matches,
        "state_space_replay_matches": any(entry[4] for entry in scored),
        "state_space_position_error_cm": score.position_error_cm,
        "state_space_span_cm": score.span_cm,
        "state_space_duration_s": duration_s,
        "state_space_start_error_s": float(event.start_s - start_s),
        "state_space_end_error_s": float(event.end_s - end_s),
        "state_space_dispersion_cm": dispersion,
    }
    return out


def state_space_outcome(recording, injected, window, bounds, true_position_cm, path_cm):
    """The state-space model run over the window and scored against the event."""
    run = run_state_space(recording, injected, window)
    return score_state_space(recording, run, bounds, true_position_cm, path_cm)


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

    cell_key = (
        f"path{path_cm}-mid{midpoint_cm}-dur{duration_s}-rate{rate_hz}-seed{seed}"
    )
    row |= memoryless_outcome(
        recording, injected, window, bounds, true_position_cm, path_cm, cell_key
    )
    row |= state_space_outcome(
        recording, injected, window, bounds, true_position_cm, path_cm
    )
    return row


def main(argv: Sequence[str] | None = None) -> None:
    # the CPU unless JAX_PLATFORMS names another
    default_jax_platforms("cpu")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument(
        "--out",
        type=Path,
        default=detection_comparison_dir() / "memoryless_detection_sweep.parquet",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=SUMMARY_PATH,
        help="Where the aggregates go. Point a pilot elsewhere so it does not "
        "overwrite the committed summary.",
    )
    args = parser.parse_args(argv)

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
        "produced_by": produced_by(__file__),
        "produced_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": described_commit(),
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
        "shuffle": "place-cell identity (the reference's rateMap(ind_rand, :)); "
        "binned centers, the jump of a pair whose slices hold as many spikes of "
        "each cell set aside (the reference's zero jump on a continuous center), "
        "strict-below count, p <= 0.05",
        "rate_floor_hz": REFERENCE_RATE_FLOOR_HZ,
        "n_shuffles": DEFAULT_N_SHUFFLES,
        "shuffle_run_seed": SHUFFLE_RUN_SEED,
        "track": "linear",
    }
    write_table(table, args.out)
    summary = replay_sweep_summary(table)
    provenance = {
        "produced_by": table.attrs["produced_by"],
        "produced_at": table.attrs["produced_at"],
        "git_commit": table.attrs["git_commit"],
        "n_cells_total": str(len(table)),
    }
    for measure, note in provenance.items():
        summary.loc[len(summary)] = {
            "summary": "provenance",
            "measure": measure,
            "value": np.nan,
            "note": note,
        }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)

    print(f"\n{len(table)} cells in {(clock.time() - started) / 60:.1f} min")
    print(f"  raw      -> {args.out}")
    print(f"  summary  -> {args.summary}")
    print()
    for label, rate in (
        ("sliding window, native criterion", table.sliding_found.mean()),
        ("sliding window, and it is the event", table.sliding_matches.mean()),
        (
            "  the same, before the shuffle",
            table.sliding_matches_before_shuffle.mean(),
        ),
        ("  the same, spread test only", table.sliding_matches_spread_only.mean()),
        ("sliding window, any candidate", table.sliding_any_candidate.mean()),
        ("state space, native criterion", table.state_space_found.mean()),
        ("state space, and it is the event", table.state_space_matches.mean()),
    ):
        print(f"  {label:<35}: {rate:.3f}")
    print(
        "  of the sliding window's accepted events, the share that are not the "
        f"injection: {share((table.sliding_found & ~table.sliding_matches_before_shuffle).sum(), table.sliding_found.sum()):.3f}"
    )


if __name__ == "__main__":
    main()
