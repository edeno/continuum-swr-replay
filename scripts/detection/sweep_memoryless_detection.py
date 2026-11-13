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

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.comparison_simulation import (
    N_NEURONS,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (
    REFERENCE_RATE_FLOOR_HZ,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.shuffle import (
    DEFAULT_N_SHUFFLES,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.sweep import (
    DECODE_MARGIN_S,
    DURATION_S,
    MATCH_TOLERANCE_CM,
    MIDPOINT_CM,
    NON_LOCAL_MINIMUM_DURATION_S,
    NON_LOCAL_PROB_THRESHOLD,
    PATH_CM,
    PEAK_RATE_HZ,
    PLACEMENT_MARGIN_S,
    SHUFFLE_RUN_SEED,
    build_recording,
    replay_grid,
    usable_intervals,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.sweep_replays import (
    SAME_PLACE_CM,
    replay_sweep,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.sweep_summaries import (
    replay_sweep_summary,
    share,
)
from continuum_swr_replay.paths import detection_comparison_dir
from continuum_swr_replay.runs.device import default_jax_platforms
from continuum_swr_replay.runs.provenance import described_commit, produced_by
from continuum_swr_replay.tables.io import write_table

SUMMARY_PATH = Path("paper/reported/inputs/detection/detection_sweep_summary.csv")
"""Aggregated results, small enough to keep in the repository.

The raw sweep goes under ``Processed-Data``, which is not tracked. Quoted
numbers come from this file, so that nobody has to transcribe them out of a
one-off script.
"""


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
    intervals = usable_intervals(recording)
    if len(intervals) == 0:
        raise RuntimeError("No immobile stretch survives excluding the labeled events.")
    print(
        f"setup {clock.time() - started:.1f}s; {len(intervals)} usable stretches "
        f"spanning {np.diff(intervals, axis=1).sum():.1f}s"
    )

    combinations = replay_grid()
    print(
        f"{len(combinations)} combinations x {args.n_seeds} seeds = "
        f"{len(combinations) * args.n_seeds} cells"
    )

    rows, started = [], clock.time()
    for index, cells in enumerate(replay_sweep(recording, intervals, args.n_seeds)):
        rows += cells
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
