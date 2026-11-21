#!/usr/bin/env python
"""The non-local restriction scores summed over the windows of Figure 2's panel g.

For each epoch of the restriction run (``score_nonlocal_restriction.py``, its
``ok`` records): read its per-bin table, check that its time is the decode
frame's, build the epoch's windows exactly as ``score_tetrode_holdout.py`` does
(``state_evidence.windows.epoch_windows`` on the canonical event table, read
through the SWR source, and the epoch's ``event_participation`` table), sum each
window's bins (``restriction_windows.restriction_window_sums``) and add its
spike count over every tetrode from the epoch's held-out tetrode table
(``restriction_windows.window_spike_counts``, which fails when the two runs'
windows differ by id, event, kind, group or length, or a window lacks a
tetrode's row). An epoch that is ``ok`` in only one of the two runs is left out
and named in a warning. CPU only: pandas and numpy, nothing is decoded or refit.

Writes, under ``processed_data_dir() / EVENT_NONLOCAL_RESTRICTION_WINDOWS_DIR``:

- ``{epoch_id}_nonlocal_restriction_windows.parquet``: one row per window,
  ``epoch_id``, ``animal``, ``window_id``, ``event_id``, ``kind``, ``group``,
  ``s``, ``e``, ``n_bins``, ``n_bins_in_swr``, ``n_bins_immobile``,
  ``n_bins_excluded``, ``n_bins_not_finite``, ``duration_s``, the six summed
  scores (float64 nats), the same over the window's ``immobile_outside_swr``
  bins only (``n_bins_outside_swr``, ``duration_outside_swr_s`` and the six
  ``<score>_outside_swr``; ``restriction_windows.OUTSIDE_SWR_COLUMNS``) and
  ``n_spikes``;
- ``{epoch_id}_nonlocal_restriction_windows.json``: the epoch's record (the
  windows by kind and group, the windows with a bin in excluded time, outside
  the immobile categories, with a score that is not finite, without spikes, the
  "no SWR" events with a bin inside an SWR the ``swr`` model used, and the
  events with no bin outside SWRs; the time
  step; the digests of the per-bin table, the held-out tetrode table and the
  participation table; provenance);

and ``manifest.json`` gathered from the records. The header records the
digests of the event table, the participation tables, and the restriction and
held-out tetrode runs' manifests. The participation tables must be the ones the
held-out tetrode run read (its manifest's inputs), or the run stops before any
epoch: the control windows come from them. An event table other than the one
the held-out tetrode run read does not stop the run: it is logged as a warning
and the header's settings carry a note with both digests
(``restriction_windows.EVENT_TABLE_NOTE``; only then, so a run on the held-out
tetrode run's table keeps its header and resumes), and the events summary marks
every value read from the run pending. Each epoch's windows must still be the
held-out tetrode run's (``restriction_windows.window_spike_counts``).

Usage::

    uv run --frozen python scripts/event_evidence/sum_nonlocal_restriction_windows.py \\
        --epochs wilbur20210329_02_r1_HPC
    uv run --frozen python scripts/event_evidence/sum_nonlocal_restriction_windows.py \\
        --workers 8
"""

from __future__ import annotations

import argparse
import logging
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
HOLDOUT_COLUMNS = [*rw.WINDOW_KEYS, "tetrode", "n_bins", "n_spikes"]


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
            "n_windows_with_bins_not_immobile": int(
                (table["n_bins_immobile"] < table["n_bins"]).sum()
            ),
            "n_windows_not_finite": int((table["n_bins_not_finite"] > 0).sum()),
            "n_windows_without_spikes": int((table["n_spikes"] == 0).sum()),
            "n_no_swr_events_with_in_swr_bins": int(
                (no_swr_events & (table["n_bins_in_swr"] > 0)).sum()
            ),
            "n_in_swr_bins_of_no_swr_events": int(
                table.loc[no_swr_events, "n_bins_in_swr"].sum()
            ),
            "n_events_without_bins_outside_swr": int(
                ((table["kind"] == "event") & (table["n_bins_outside_swr"] == 0)).sum()
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
    """Fail unless the participation tables are the ones the held-out tetrode
    run read: the control windows come from them.

    Raises
    ------
    ValueError
        If the participation tables' digest differs.
    """
    name = "event_participation"
    if header_inputs[name]["sha256"] != holdout_manifest["inputs"].get(name, {}).get(
        "sha256"
    ):
        raise ValueError(
            f"not the held-out tetrode run's inputs: ['{name}']; its windows would "
            "not be these"
        )


def event_table_note(header_inputs: dict, holdout_manifest: dict) -> dict | None:
    """The note the run records when its event table is not the one the
    held-out tetrode run read, or None when it is.

    Returns
    -------
    dict or None
        ``event_table_sha256`` (this run's) and
        ``tetrode_holdout_event_table_sha256`` (the held-out tetrode run's; None
        when its manifest records none), with a sentence saying what differs.
    """
    ours = header_inputs["event_table"]["sha256"]
    theirs = holdout_manifest["inputs"].get("event_table", {}).get("sha256")
    if ours == theirs:
        return None
    return {
        "note": (
            "the event table is not the one the held-out tetrode run read; its "
            "spike counts are matched window by window"
        ),
        "event_table_sha256": ours,
        "tetrode_holdout_event_table_sha256": theirs,
    }


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
    holdout_manifest = read_manifest(holdout_dir)
    note = event_table_note(header["inputs"], holdout_manifest)
    if note is not None:
        logging.getLogger(__name__).warning(
            "%s (this run's sha256 %s, the held-out tetrode run's %s); the run's "
            "values will be pending",
            note["note"],
            note["event_table_sha256"],
            note["tetrode_holdout_event_table_sha256"],
        )
        header["settings"][rw.EVENT_TABLE_NOTE] = note
    if args.gather:
        gather_manifest(output, OUTPUT_NAME, header)
        return
    check_holdout_inputs(header["inputs"], holdout_manifest)
    restriction_ok = set(ok_epochs(read_manifest(restriction_dir)))
    holdout_ok = set(ok_epochs(holdout_manifest))
    in_one_run = sorted(restriction_ok ^ holdout_ok)
    if in_one_run:
        logging.getLogger(__name__).warning(
            "%d epochs are ok in only one of the two runs and are left out: %s",
            len(in_one_run),
            in_one_run,
        )
    every_epoch = sorted(restriction_ok & holdout_ok)
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
