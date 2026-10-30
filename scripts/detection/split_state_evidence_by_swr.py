#!/usr/bin/env python
"""Held-out spike evidence of Non-Local bouts, split by SWR coupling.

``run_state_evidence.py`` scores each confident decoder-state bout with
held-out tetrodes but does not split the Non-Local bouts by whether they
coincide with an SWR. This runner labels each Non-Local bout of a finished run,
and its matched window, by its full span
(:func:`~continuum_swr_replay.analysis.hippocampus.state_evidence.split_non_local_by_swr`):
"SWR" when some bin of the bout is in an SWR of the decode frame read through
the SWR source (``read_dot_with_swr_source``), else "no SWR". ``bout_statistics``
then runs on each subset as on any state (animal-level t-interval, the paired
bout - matched-window difference) over the primary metrics, their per-held-out-
spike versions and MAP_S's distance from the animal.

Reads every epoch of the run whose ``per_epoch/{epoch}_validation.json`` is
``ok``. Writes, beside the run:

- ``nl_bouts_by_swr.parquet``: the labelled windows;
- ``nl_bouts_by_swr_stats.csv``: the statistics (Part 2 reads it, claim 7);
- ``nl_bouts_by_swr.provenance.json``: the commit, the SWR source and the epochs.

Usage::

    uv run python scripts/detection/split_state_evidence_by_swr.py \\
        Processed-Data/_refresh_20261016/state_evidence
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.events.analysed_time import read_dot_with_swr_source
from continuum_swr_replay.analysis.events.swr_source import swr_source_name
from continuum_swr_replay.analysis.hippocampus.state_evidence import (
    bout_statistics,
    split_non_local_by_swr,
)
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.parameters import ROOT_DIR
from continuum_swr_replay.paths import processed_data_dir
from continuum_swr_replay.runs.io import write_json
from continuum_swr_replay.runs.provenance import produced_by, provenance_record
from continuum_swr_replay.tables.io import read_table, write_table

PRODUCED_BY = produced_by(__file__)
STEM = "nl_bouts_by_swr"
METRICS = (
    "runcv_nl_anim_mark",
    "runcv_nl_null_mark",
    "runcv_nl_anim_mark_ps",
    "runcv_nl_null_mark_ps",
    "runcv_mapS_dist_cm",
    "runcv_ns_expected",
)


def ok_epochs(run_dir: Path) -> list[str]:
    """The run's epochs whose validation record is ``ok``, sorted."""
    return sorted(
        path.name.removesuffix("_validation.json")
        for path in (run_dir / "per_epoch").glob("*_validation.json")
        if json.loads(path.read_text())["status"] == "ok"
    )


def label_epoch(run_dir: Path, epoch_id: str, decode_dir: Path) -> pd.DataFrame:
    """One epoch's Non-Local windows labelled by SWR."""
    windows = read_table(run_dir / "per_epoch" / f"{epoch_id}_windows.parquet")
    frame = read_dot_with_swr_source(decode_paths(epoch_id, decode_dir).data_over_time)
    try:
        return split_non_local_by_swr(windows, frame["SWR_event"].to_numpy() > 0)
    except ValueError as exc:
        raise ValueError(f"{epoch_id}: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="a run_state_evidence.py output")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    decode_dir = processed_data_dir()

    epochs = ok_epochs(run_dir)
    if not epochs:
        raise SystemExit(f"{run_dir}: no epoch with an ok validation record")
    windows = pd.concat(
        [label_epoch(run_dir, e, decode_dir) for e in epochs], ignore_index=True
    )
    states = sorted(windows["state"].unique())
    stats = bout_statistics(windows, metrics=dict.fromkeys(states, METRICS))

    write_table(windows, run_dir / f"{STEM}.parquet")
    stats.to_csv(run_dir / f"{STEM}_stats.csv", index=False)
    write_json(
        run_dir / f"{STEM}.provenance.json",
        {
            "produced_by": PRODUCED_BY,
            "provenance": provenance_record(ROOT_DIR),
            "swr_source": swr_source_name(),
            "decode_dir": str(decode_dir),
            "epochs": epochs,
        },
        indent=1,
    )
    print(f"{len(epochs)} epochs, {len(windows)} windows -> {run_dir / STEM}_stats.csv")


if __name__ == "__main__":
    main()
