#!/usr/bin/env python
"""Tetrode participation and spike rate of each non-local event, against Local rest.

For every canonical immobile non-local event (``replay_content/event_table.parquet``,
read through the SWR source) and one control window of the same length drawn from
Local rest in the same epoch (``analysis.event_evidence.participation``), count each
HPC tetrode's spikes and write:

- ``{epoch_id}_participation.parquet`` per epoch: one row per window (``kind`` ``"event"`` or
  ``"control"``), with ``event_id``, ``group`` (``"SWR"`` or ``"no SWR"``, the
  event's), ``n_bins``, ``duration_s``, ``n_spikes``, ``n_active_tetrodes``,
  ``n_tetrodes``, ``participation`` and ``spike_rate_hz``;
- ``manifest.json``: the settings, the event table's SHA-256 (``inputs``), the
  code's provenance and the status of each epoch of this invocation.

Spike times are read from the lab's database through ``load_data`` (read only).

Usage::

    uv run --frozen python scripts/event_evidence/measure_event_participation.py \\
        --epochs wilbur20210329_02_r1_HPC
    uv run --frozen python scripts/event_evidence/measure_event_participation.py \\
        --n-workers 8
"""

from __future__ import annotations

import argparse
import logging
import time as clock
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_definition.analysed_time import (
    immobile_analysed_bins,
    read_dot_with_swr_source,
    read_swr_table,
)
from continuum_swr_replay.analysis.event_definition.swr_source import (
    swr_epoch_id_of_session,
)
from continuum_swr_replay.analysis.event_evidence import participation as pt
from continuum_swr_replay.analysis.event_evidence.state_evidence.windows import (
    event_bins,
)
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.decoding.identifiers import parse_epoch_identifier
from continuum_swr_replay.parameters import ROOT_DIR
from continuum_swr_replay.paths import (
    EVENT_PARTICIPATION_DIR,
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.io import output_path, sha256_file, write_json
from continuum_swr_replay.runs.provenance import produced_by, provenance_record
from continuum_swr_replay.tables.io import write_table

PRODUCED_BY = produced_by(__file__)
OUTPUT_NAME = "participation"
"""Each epoch's table is ``{epoch_id}_participation.parquet``."""
SEED = 20261118
"""The control windows' seed, with the epoch id."""

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def epoch_ids(events: pd.DataFrame) -> pd.Series:
    """Each event's epoch id (``swr_source.swr_epoch_id_of_session``)."""
    keys = list(events[["animal", "session_date", "epoch"]].itertuples(index=False))
    ids = {key: swr_epoch_id_of_session(*key) for key in set(keys)}
    return pd.Series([ids[key] for key in keys], index=events.index)


def epoch_events(events: pd.DataFrame, epoch_id: str) -> pd.DataFrame:
    """The table's events of one epoch, by start."""
    mine = events[epoch_ids(events) == epoch_id]
    return mine.sort_values("event_start_s").reset_index(drop=True)


def load_spike_times(epoch_id: str) -> list[np.ndarray]:
    """Each HPC tetrode's sorted spike times (s), from ``load_data`` (read only)."""
    from continuum_swr_replay.data_loaders import load_data
    from continuum_swr_replay.data_types import SessionDataKeys as K

    key = parse_epoch_identifier(epoch_id)
    data = load_data(
        key.nwb_file_name, key.epoch_name, include=frozenset({K.SPIKE_TIMES})
    )
    return [np.sort(np.asarray(t)) for t in data["spike_times"][key.brain_area]]


def epoch_table(epoch_id: str, events: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """One epoch's windows: its events and their Local-rest controls."""
    frame = read_dot_with_swr_source(decode_paths(epoch_id, data_dir).data_over_time)
    time = frame.index.to_numpy(dtype=float)
    bin_s = float(np.median(np.diff(time)))  # for the margin only
    in_event = (frame["SWR_event"].to_numpy() > 0) | (
        frame["non_local_event"].to_numpy() > 0
    )
    local_rest = pt.local_rest_mask(
        frame["Local_prob"].to_numpy(),
        immobile_analysed_bins(frame),
        in_event,
        margin_bins=int(round(pt.MARGIN_S / bin_s)),
    )
    s, e = event_bins(
        time, events["event_start_s"].to_numpy(), events["event_end_s"].to_numpy()
    )
    controls = pt.control_windows(s, e, local_rest, seed=(SEED, epoch_id))
    spike_times = load_spike_times(epoch_id)
    group = np.where(events["is_swr"].to_numpy(dtype=bool), "SWR", "no SWR")
    has_control = controls["s"].to_numpy() >= 0
    parts = []
    for kind, keep, first, last in (
        ("event", np.ones(len(events), dtype=bool), s, e),
        ("control", has_control, controls["s"].to_numpy(), controls["e"].to_numpy()),
    ):
        first, last = first[keep], last[keep]
        start, stop = time[first], pt.window_stops(time, last)
        counts = pt.window_spike_counts(spike_times, start, stop)
        measures = pt.participation_and_rate(counts, stop - start)
        parts.append(
            measures.assign(
                event_id=events["event_id"].to_numpy()[keep],
                animal=events["animal"].to_numpy()[keep],
                epoch_id=epoch_id,
                group=group[keep],
                kind=kind,
                n_bins=last - first + 1,
                start_s=start,
                duration_s=stop - start,
            )
        )
    return pd.concat(parts, ignore_index=True)


def run_epoch(epoch_id: str, event_table: Path, data_dir: Path, output: Path) -> dict:
    """Measure one epoch and write its table; return its status record."""
    started = clock.time()
    events = epoch_events(read_swr_table(event_table), epoch_id)
    table = epoch_table(epoch_id, events, data_dir)
    write_table(table, output_path(epoch_id, output, OUTPUT_NAME))
    return {
        "status": "ok",
        "n_events": len(events),
        "n_without_control": int(len(events) - (table["kind"] == "control").sum()),
        "seconds": round(clock.time() - started, 1),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--epochs", nargs="+", help="epoch ids; default every epoch")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    provenance = provenance_record(ROOT_DIR)
    data_dir = args.data_dir or processed_data_dir()
    output = args.output_dir or data_dir / EVENT_PARTICIPATION_DIR
    output.mkdir(parents=True, exist_ok=True)
    event_table = table_path(
        "event_table", data_dir / replay_content_dir().relative_to(processed_data_dir())
    )
    event_table_sha256 = sha256_file(event_table)
    events = read_swr_table(event_table)
    all_epochs = sorted(epoch_ids(events).unique())
    epochs = args.epochs or all_epochs
    to_run = [
        epoch_id
        for epoch_id in epochs
        if args.overwrite or not output_path(epoch_id, output, OUTPUT_NAME).exists()
    ]
    records: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {
            pool.submit(run_epoch, epoch_id, event_table, data_dir, output): epoch_id
            for epoch_id in to_run
        }
        for k, future in enumerate(as_completed(futures), start=1):
            epoch_id = futures[future]
            try:
                records[epoch_id] = future.result()
            except Exception as error:  # noqa: BLE001 - recorded per epoch
                records[epoch_id] = {"status": "failed", "error": repr(error)}
            logging.info(f"[{k}/{len(to_run)}] {epoch_id}: {records[epoch_id]}")
    write_json(
        output / "manifest.json",
        {
            "produced_by": PRODUCED_BY,
            "provenance": provenance,
            "inputs": {"event_table": {"sha256": event_table_sha256}},
            "settings": {
                "event_table": str(event_table),
                "seed": SEED,
                "local_threshold": pt.LOCAL_THRESHOLD,
                "margin_s": pt.MARGIN_S,
            },
            "epochs": records,
        },
        indent=1,
        sort_keys=True,
    )
    failed = [e for e, r in records.items() if r["status"] != "ok"]
    if failed:
        raise SystemExit(f"{len(failed)} epochs failed: {failed[:5]}")


if __name__ == "__main__":
    main()
