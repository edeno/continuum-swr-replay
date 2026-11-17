#!/usr/bin/env python
"""Does each detector's content reproduce in cells it never saw?

Held-out evidence on simulated data, where the truth of every event is known,
so the held-out test itself can be calibrated before it is used on recordings.
The simulated recording's cells are split into discovery and evaluation
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
held-out rule of ``detector_comparison.heldout``, under both content
estimates: our decoder's non-local posterior and the sliding window's own
posterior, each from the discovery cells. What comes out is how often held-out
support falls on events with spatial content and how often on events without
it, for each class and fold: the error rates held-out results on recordings
will be read with.

Usage:
    python scripts/detector_comparison/sweep_memoryless_heldout.py
    python scripts/detector_comparison/sweep_memoryless_heldout.py --n-seeds 1 --folds A --out /tmp/pilot.parquet
"""

import argparse
import time as clock
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.detection.memoryless_sequence import (
    sweep_controls,
)
from continuum_swr_replay.analysis.detector_comparison.heldout import (
    heldout_summary,
)
from continuum_swr_replay.analysis.detector_comparison.heldout_sweep import (
    N_PERMUTATIONS,
    folds,
    run_fold,
    run_no_replay_fold,
)
from continuum_swr_replay.paths import detection_comparison_dir
from continuum_swr_replay.runs.device import default_jax_platforms
from continuum_swr_replay.runs.provenance import described_commit, produced_by
from continuum_swr_replay.tables.io import write_table

SUMMARY_PATH = Path("paper/reported/inputs/detection/detection_heldout_summary.csv")


def main(argv: Sequence[str] | None = None) -> None:
    # the CPU unless JAX_PLATFORMS names another, as the replay sweep
    default_jax_platforms("cpu")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument("--folds", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--n-recordings", type=int, default=sweep_controls.N_RECORDINGS)
    parser.add_argument(
        "--out",
        type=Path,
        default=detection_comparison_dir() / "memoryless_heldout.parquet",
    )
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args(argv)

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
        "produced_by": produced_by(__file__),
        "produced_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": described_commit(),
        "n_seeds": args.n_seeds,
        "n_permutations": N_PERMUTATIONS,
        "n_recordings": args.n_recordings,
    }
    write_table(table, args.out)
    summary = heldout_summary(table)
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
