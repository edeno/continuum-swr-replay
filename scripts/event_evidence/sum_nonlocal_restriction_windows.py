#!/usr/bin/env python
"""The non-local restriction scores summed over the windows of Figure 2's panel h.

For each epoch of the restriction run (``score_nonlocal_restriction.py``, its
``ok`` records): read its per-bin table, check that its time is the decode
frame's, build the epoch's windows exactly as ``score_tetrode_holdout.py`` does
(``state_evidence.windows.epoch_windows`` on the canonical event table, read
through the SWR source, and the epoch's ``event_participation`` table), sum each
window's bins (``restriction_windows.restriction_window_sums``) and add its
spike count over every tetrode from the epoch's held-out tetrode table
(``restriction_windows.window_spike_counts``, which fails when the two runs'
windows differ). CPU only: pandas and numpy, nothing is decoded or refit.

Writes, under ``processed_data_dir() / EVENT_NONLOCAL_RESTRICTION_WINDOWS_DIR``:

- ``{epoch_id}_nonlocal_restriction_windows.parquet``: one row per window,
  ``epoch_id``, ``animal``, ``window_id``, ``event_id``, ``kind``, ``group``,
  ``s``, ``e``, ``n_bins``, ``n_bins_in_swr``, ``n_bins_scored``,
  ``n_bins_excluded``, ``n_bins_not_finite``, ``duration_s``, the six summed
  scores (float64 nats) and ``n_spikes``;
- ``{epoch_id}_nonlocal_restriction_windows.json``: the epoch's record (the
  windows by kind and group, the windows with a bin in excluded time, outside
  the immobile categories, with a score that is not finite, without spikes, and
  the "no SWR" events with a bin inside an SWR the ``swr`` model used; the time
  step; the digests of the per-bin table, the held-out tetrode table and the
  participation table; provenance);

and ``manifest.json`` gathered from the records. The header records the
digests of the event table, the participation tables, and the restriction and
held-out tetrode runs' manifests; the event table and participation tables must
be the ones the held-out tetrode run read (its manifest's inputs), or the run
stops before any epoch.

Usage::

    uv run --frozen python scripts/event_evidence/sum_nonlocal_restriction_windows.py \\
        --epochs wilbur20210329_02_r1_HPC
    uv run --frozen python scripts/event_evidence/sum_nonlocal_restriction_windows.py \\
        --workers 8
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
import time as clock
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    read_dot_with_swr_source,
    read_swr_table,
)
from continuum_swr_replay.analysis.event_evidence import restriction_windows as rw
from continuum_swr_replay.analysis.event_evidence.state_evidence.windows import (
    epoch_windows,
)
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.decoding.identifiers import parse_epoch_identifier
from continuum_swr_replay.intervals import sample_step
from continuum_swr_replay.parameters import ROOT_DIR
from continuum_swr_replay.paths import (
    EVENT_NONLOCAL_RESTRICTION_DIR,
    EVENT_NONLOCAL_RESTRICTION_WINDOWS_DIR,
    EVENT_PARTICIPATION_DIR,
    EVENT_TETRODE_HOLDOUT_DIR,
    processed_data_dir,
)
from continuum_swr_replay.runs.epoch_records import gather_manifest
from continuum_swr_replay.runs.io import (
    files_sha256,
    json_ready,
    ok_epochs,
    ok_table_paths,
    output_path,
    read_manifest,
    sha256_file,
)
from continuum_swr_replay.runs.provenance import produced_by, provenance_record
from continuum_swr_replay.tables.io import write_table

# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.event_evidence._per_event_run import (  # noqa: E402
    add_run_arguments,
    event_table_input,
    event_table_path,
    run,
    select_epochs,
)

PRODUCED_BY = produced_by(__file__)
OUTPUT_NAME = "nonlocal_restriction_windows"
BINS_NAME = "nonlocal_restriction_bins"
HOLDOUT_NAME = "tetrode_holdout"
PARTICIPATION_NAME = "participation"
HOLDOUT_COLUMNS = [*rw.WINDOW_KEYS, "n_bins", "n_spikes"]


def decode_time(epoch_id: str, data_dir: Path) -> np.ndarray:
    """The epoch's decode frame's time axis, s, shape (n_time,)."""
    frame = read_dot_with_swr_source(decode_paths(epoch_id, data_dir).data_over_time)
    return frame.index.to_numpy(dtype=float)


def run_epoch(epoch_id: str, event_table: Path, data_dir: Path, output: Path) -> dict:
    """Sum one epoch's windows, write its table and return its record.

    Raises
    ------
    ValueError
        If the per-bin table's time is not the decode frame's, or the windows
        are not the held-out tetrode run's.
    """
    started = clock.time()
    bins_path = output_path(
        epoch_id, data_dir / EVENT_NONLOCAL_RESTRICTION_DIR, BINS_NAME
    )
    holdout_path = output_path(
        epoch_id, data_dir / EVENT_TETRODE_HOLDOUT_DIR, HOLDOUT_NAME
    )
    participation_path = output_path(
        epoch_id, data_dir / EVENT_PARTICIPATION_DIR, PARTICIPATION_NAME
    )
    bins = pd.read_parquet(bins_path)
    time = decode_time(epoch_id, data_dir)
    if not np.array_equal(bins["time"].to_numpy(dtype=float), time):
        raise ValueError("the per-bin table's time is not the decode frame's")
    windows = epoch_windows(
        time, read_swr_table(event_table), pd.read_parquet(participation_path)
    )
    table = rw.restriction_window_sums(bins, windows)
    table["n_spikes"] = rw.window_spike_counts(
        pd.read_parquet(holdout_path, columns=HOLDOUT_COLUMNS), windows
    ).to_numpy()
    table.insert(0, "animal", parse_epoch_identifier(epoch_id).animal)
    table.insert(0, "epoch_id", epoch_id)
    write_table(table, output_path(epoch_id, output, OUTPUT_NAME))

    no_swr_events = (table["kind"] == "event") & (table["group"] == "no SWR")
    return json_ready(
        {
            "status": "ok",
            "n_windows": len(table),
            "n_windows_by_kind_group": {
                f"{kind}, {group}": int(n)
                for (kind, group), n in table.groupby(["kind", "group"]).size().items()
            },
            "n_windows_with_excluded_bins": int((table["n_bins_excluded"] > 0).sum()),
            "n_windows_with_unscored_bins": int(
                (table["n_bins_scored"] < table["n_bins"]).sum()
            ),
            "n_windows_not_finite": int((table["n_bins_not_finite"] > 0).sum()),
            "n_windows_without_spikes": int((table["n_spikes"] == 0).sum()),
            "n_no_swr_events_with_in_swr_bins": int(
                (no_swr_events & (table["n_bins_in_swr"] > 0)).sum()
            ),
            "n_in_swr_bins_of_no_swr_events": int(
                table.loc[no_swr_events, "n_bins_in_swr"].sum()
            ),
            "bin_s": sample_step(time),
            "epoch_inputs": {
                name: {"sha256": sha256_file(path), "path": str(path)}
                for name, path in {
                    "bins": bins_path,
                    "tetrode_holdout": holdout_path,
                    "participation": participation_path,
                }.items()
            },
            "seconds": round(clock.time() - started, 1),
        }
    )


def run_input(run_dir: Path) -> dict[str, str]:
    """A run folder as an input: its manifest's digest and where it is."""
    manifest = run_dir / "manifest.json"
    return {"sha256": sha256_file(manifest), "path": str(run_dir)}


def check_holdout_inputs(header_inputs: dict, holdout_manifest: dict) -> None:
    """Fail unless the event table and participation tables are the ones the
    held-out tetrode run read.

    Raises
    ------
    ValueError
        Naming each input whose digest differs.
    """
    differ = [
        name
        for name in ("event_table", "event_participation")
        if header_inputs[name]["sha256"]
        != holdout_manifest["inputs"].get(name, {}).get("sha256")
    ]
    if differ:
        raise ValueError(
            f"not the held-out tetrode run's inputs: {differ}; its windows would "
            "not be these"
        )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_run_arguments(parser)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)

    data_dir = args.data_dir or processed_data_dir()
    output = args.output_dir or data_dir / EVENT_NONLOCAL_RESTRICTION_WINDOWS_DIR
    event_table = event_table_path(data_dir)
    participation_dir = data_dir / EVENT_PARTICIPATION_DIR
    restriction_dir = data_dir / EVENT_NONLOCAL_RESTRICTION_DIR
    holdout_dir = data_dir / EVENT_TETRODE_HOLDOUT_DIR
    header = {
        "produced_by": PRODUCED_BY,
        "provenance": provenance_record(ROOT_DIR),
        "settings": {
            "contrasts": {k: list(v) for k, v in rw.CONTRASTS.items()},
        },
        "inputs": {
            "event_table": event_table_input(event_table),
            "event_participation": {
                "sha256": files_sha256(
                    ok_table_paths(participation_dir, PARTICIPATION_NAME)
                ),
                "path": str(participation_dir),
            },
            "nonlocal_restriction": run_input(restriction_dir),
            "tetrode_holdout": run_input(holdout_dir),
        },
    }
    if args.gather:
        gather_manifest(output, OUTPUT_NAME, header)
        return
    holdout_manifest = read_manifest(holdout_dir)
    check_holdout_inputs(header["inputs"], holdout_manifest)
    every_epoch = sorted(
        set(ok_epochs(read_manifest(restriction_dir)))
        & set(ok_epochs(holdout_manifest))
    )
    run(
        epochs=select_epochs(args.epochs or every_epoch, args.shards, args.n_shards),
        task=run_epoch,
        task_args=(event_table, data_dir, output),
        output=output,
        name=OUTPUT_NAME,
        header=header,
        overwrite=args.overwrite,
        max_workers=args.workers,
        pool_options={"mp_context": multiprocessing.get_context("spawn")},
    )


if __name__ == "__main__":
    main()
