#!/usr/bin/env python
"""Spike likelihood at decoded and actual positions, for every canonical non-local event.

For each event of the event table (``replay_content/event_table.parquet``, read
through the SWR source), over its first to last decode bin, score the decoder's
stored log-likelihood at the decoded position against the animal's measured
position (``analysis.event_evidence.likelihood_advantage``). The spike count that
normalizes ``D_spike`` is the participation run's (``event_participation``, the
same window: the first bin's time to the decode time after the last bin), joined
on ``event_id``; every event of the table must have one.

Writes ``{epoch_id}.parquet`` per epoch (one row per event: ``event_id``,
``animal``, ``epoch_id``, ``group``, ``n_bins``, ``n_scored_bins``, ``D``,
``D_bin``, ``n_spikes``, ``D_spike``) and ``manifest.json``.

Reads only stored files (no database).

Usage::

    uv run --frozen python scripts/event_evidence/score_event_likelihood_advantage.py \\
        --epochs wilbur20210329_02_r1_HPC
    uv run --frozen python scripts/event_evidence/score_event_likelihood_advantage.py \\
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
    read_dot_with_swr_source,
    read_swr_table,
)
from continuum_swr_replay.analysis.event_evidence import likelihood_advantage as la
from continuum_swr_replay.analysis.event_evidence.participation import event_bins
from continuum_swr_replay.analysis.event_evidence.spike_evidence import (
    stored_likelihood_columns,
)
from continuum_swr_replay.analysis.event_evidence.state_evidence.terms import (
    load_encoding,
)
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.decoding.results import open_results
from continuum_swr_replay.parameters import ROOT_DIR
from continuum_swr_replay.paths import (
    EVENT_LIKELIHOOD_ADVANTAGE_DIR,
    EVENT_PARTICIPATION_DIR,
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.io import write_json
from continuum_swr_replay.runs.provenance import produced_by, provenance_record
from continuum_swr_replay.tables.io import write_table

PRODUCED_BY = produced_by(__file__)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def epoch_events(
    event_table: pd.DataFrame, participation: pd.DataFrame
) -> pd.DataFrame:
    """The epoch's events (those of its participation table), with their spikes.

    Raises
    ------
    ValueError
        If an event has no row in the event table or more than one.
    """
    measured = participation.loc[
        participation["kind"] == "event", ["event_id", "n_spikes", "group"]
    ]
    events = event_table.merge(measured, on="event_id", validate="one_to_one")
    if len(events) != len(measured):
        raise ValueError(
            f"{len(measured) - len(events)} measured events are not in the event table"
        )
    return events.sort_values("event_start_s").reset_index(drop=True)


def epoch_scores(epoch_id: str, events: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Score one epoch's events."""
    paths = decode_paths(epoch_id, data_dir)
    frame = read_dot_with_swr_source(paths.data_over_time)
    time = frame.index.to_numpy(dtype=float)
    results = open_results(paths.results, cache=False)
    if not np.array_equal(results["time"].values, time):
        raise ValueError(f"{epoch_id}: results.nc and data_over_time times differ")
    env = load_encoding(epoch_id, data_dir)["env"]
    cols = stored_likelihood_columns(results, env)
    s, e = event_bins(
        time, events["event_start_s"].to_numpy(), events["event_end_s"].to_numpy()
    )
    lengths = e - s + 1
    rows = np.concatenate([np.arange(a, b + 1) for a, b in zip(s, e, strict=True)])
    log_likelihood = (
        results["log_likelihood"].isel(time=rows).values[:, cols].astype(float)
    )
    delta = la.per_bin_advantage(
        log_likelihood,
        frame["map_estimate"].to_numpy()[rows],
        frame["linear_position"].to_numpy()[rows],
        env,
    )
    scores = la.event_scores(delta, lengths, events["n_spikes"].to_numpy())
    return scores.assign(
        event_id=events["event_id"].to_numpy(),
        animal=events["animal"].to_numpy(),
        epoch_id=epoch_id,
        group=events["group"].to_numpy(),
    )


def run_epoch(epoch_id: str, event_table: Path, data_dir: Path, output: Path) -> dict:
    """Score one epoch and write its table; return its status record."""
    started = clock.time()
    participation = pd.read_parquet(
        data_dir / EVENT_PARTICIPATION_DIR / f"{epoch_id}.parquet"
    )
    events = epoch_events(read_swr_table(event_table), participation)
    table = epoch_scores(epoch_id, events, data_dir)
    write_table(table, output / f"{epoch_id}.parquet")
    return {
        "status": "ok",
        "n_events": len(table),
        "n_scored": int(table["D_spike"].notna().sum()),
        "n_partly_scored": int((table["n_scored_bins"] < table["n_bins"]).sum()),
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
    output = args.output_dir or data_dir / EVENT_LIKELIHOOD_ADVANTAGE_DIR
    output.mkdir(parents=True, exist_ok=True)
    event_table = table_path(
        "event_table", data_dir / replay_content_dir().relative_to(processed_data_dir())
    )
    measured = sorted(
        p.stem for p in (data_dir / EVENT_PARTICIPATION_DIR).glob("*_HPC.parquet")
    )
    epochs = args.epochs or measured
    to_run = [
        epoch_id
        for epoch_id in epochs
        if args.overwrite or not (output / f"{epoch_id}.parquet").exists()
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
            "settings": {"event_table": str(event_table)},
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
