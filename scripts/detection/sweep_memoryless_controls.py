#!/usr/bin/env python
"""What each detector does with activity that is not one clean replay.

The injected-replay sweep asks how often each detector reports a replay that
is there. This asks the other questions: what each reports where there is no
replay, how it carves a replay that jumps partway or comes in a chain, and
where it puts the bounds of each part. Both detectors, the same recording and
encoding model, the same placements and scoring rule as the sweep:

- no replay: the recording's own immobile stretches, decoded as they are. They
  hold only local coding, so every event reported there is a false positive.
  Eight simulated recordings, the sweep's and seven more, for exposure;
- a place-less burst: every cell fires at one rate unrelated to its field, as
  many spikes as a replay of the same duration and peak rate. No spatial
  content, so a detection is a false positive for either target;
- a jumping burst: the represented place redrawn along the track every 15 ms.
  Non-local representation without a sequence: calling it non-local is right,
  calling it a sequence is not;
- a mid-event jump: two 60 cm parts at constant speed, the second starting J cm
  behind where the first ended, with no gap in time; J = 0 is a continuous
  replay of both parts;
- a chain: two 150 ms, 60 cm replays with a silent gap, the second continuing
  from the first's end or far from it.

Each part is 60 cm and longer than 100 ms, so it can meet the published replay
definition on its own, and each is also decoded alone, with exactly the spikes it
had inside its event and the recording's own spikes around it. Recovery inside
the event and alone is compared both pooled and paired by trial, since pooled
recovery can rise while some trials still lose a part to the jump or the chain.
The
simulated animal rests at the track's 170 cm end, and absorbing local coding
depends on how near it an event comes, so every jump and chain keeps its
parts' far end at the same place, whatever the jump or the gap. Our detector is scored two ways throughout, canonical
(continuous plus fragmented) and continuous-only; the sliding window before and
after its significance test.

Usage:
    python scripts/detection/sweep_memoryless_controls.py
    python scripts/detection/sweep_memoryless_controls.py --n-seeds 1 --out /tmp/pilot.parquet
"""

import argparse
import time as clock
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.detection.memoryless_sequence import sweep
from continuum_swr_replay.analysis.detection.memoryless_sequence.sweep_controls import (
    N_RECORDINGS,
    run_no_replay,
    run_trials,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.sweep_summaries import (
    controls_summary,
)
from continuum_swr_replay.paths import detection_comparison_dir
from continuum_swr_replay.runs.provenance import described_commit, produced_by
from continuum_swr_replay.tables.io import write_table

SUMMARY_PATH = Path("paper/reported/inputs/detection/detection_controls_summary.csv")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=8)
    parser.add_argument("--n-recordings", type=int, default=N_RECORDINGS)
    parser.add_argument(
        "--out",
        type=Path,
        default=detection_comparison_dir() / "memoryless_controls.parquet",
    )
    parser.add_argument("--summary", type=Path, default=SUMMARY_PATH)
    args = parser.parse_args(argv)

    started = clock.time()
    recording = sweep.build_recording()
    trial_table = pd.DataFrame(run_trials(recording, args.n_seeds))
    print(f"{len(trial_table)} trial rows in {(clock.time() - started) / 60:.1f} min")
    no_replay = pd.DataFrame(run_no_replay(args.n_recordings))
    print(f"no-replay stretches done at {(clock.time() - started) / 60:.1f} min")

    table = pd.concat([trial_table, no_replay], ignore_index=True)
    table.attrs = {
        "produced_by": produced_by(__file__),
        "produced_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": described_commit(),
        "n_seeds": args.n_seeds,
        "n_recordings": args.n_recordings,
    }
    write_table(table, args.out)
    summary = controls_summary(trial_table, no_replay)
    for measure in ("produced_by", "produced_at", "git_commit"):
        summary.loc[len(summary)] = {
            "summary": "provenance",
            "measure": measure,
            "value": np.nan,
            "note": str(table.attrs[measure]),
        }
    for measure in ("n_seeds", "n_recordings"):
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
