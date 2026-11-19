#!/usr/bin/env python
"""Spike likelihood at decoded and actual positions, for every canonical non-local event.

For each event of the event table (``replay_content/event_table.parquet``, read
through the SWR source), over its first to last decode bin, score the decoder's
stored log-likelihood at the decoded position against the animal's measured
position (``analysis.event_evidence.likelihood_advantage``). The spike count that
normalizes ``D_spike`` is the participation run's (``event_participation``, the
same window: the first bin's time to the decode time after the last bin), joined
on ``event_id``; every event of the table must have one.

Writes ``{epoch_id}_likelihood_advantage.parquet`` per epoch (one row per event:
``event_id``, ``animal``, ``epoch_id``, ``group``, ``n_bins``, ``n_scored_bins``,
``D``, ``D_bin``, ``n_spikes``, ``D_spike``), each epoch's record
``{epoch_id}_likelihood_advantage.json`` (with the SHA-256 of the event table and
of the participation tables it read) and ``manifest.json`` gathered from the
records (``runs.epoch_records``). The epochs
are those the participation run's manifest marks ok.

Reads only stored files (no database).

Usage::

    uv run --frozen python scripts/event_evidence/score_event_likelihood_advantage.py \\
        --epochs wilbur20210329_02_r1_HPC
    uv run --frozen python scripts/event_evidence/score_event_likelihood_advantage.py \\
        --n-workers 8
"""

from __future__ import annotations

import argparse
import sys
import time as clock
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_definition.analysed_time import (
    read_dot_with_swr_source,
    read_swr_table,
)
from continuum_swr_replay.analysis.event_evidence import likelihood_advantage as la
from continuum_swr_replay.analysis.event_evidence.spike_evidence import (
    stored_likelihood_columns,
)
from continuum_swr_replay.analysis.event_evidence.state_evidence.terms import (
    load_encoding,
)
from continuum_swr_replay.analysis.event_evidence.state_evidence.windows import (
    event_bins,
)
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.decoding.results import open_results
from continuum_swr_replay.parameters import ROOT_DIR
from continuum_swr_replay.paths import (
    EVENT_LIKELIHOOD_ADVANTAGE_DIR,
    EVENT_PARTICIPATION_DIR,
    processed_data_dir,
)
from continuum_swr_replay.runs.epoch_records import gather_manifest
from continuum_swr_replay.runs.io import (
    files_sha256,
    ok_epochs,
    ok_table_paths,
    output_path,
    read_manifest,
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
OUTPUT_NAME = "likelihood_advantage"
PARTICIPATION_NAME = "participation"


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
        output_path(epoch_id, data_dir / EVENT_PARTICIPATION_DIR, PARTICIPATION_NAME)
    )
    events = epoch_events(read_swr_table(event_table), participation)
    table = epoch_scores(epoch_id, events, data_dir)
    write_table(table, output_path(epoch_id, output, OUTPUT_NAME))
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
    add_run_arguments(parser)
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args(argv)

    data_dir = args.data_dir or processed_data_dir()
    output = args.output_dir or data_dir / EVENT_LIKELIHOOD_ADVANTAGE_DIR
    event_table = event_table_path(data_dir)
    participation_dir = data_dir / EVENT_PARTICIPATION_DIR
    header = {
        "produced_by": PRODUCED_BY,
        "provenance": provenance_record(ROOT_DIR),
        "settings": {},
        "inputs": {
            "event_table": event_table_input(event_table),
            "event_participation": {
                "sha256": files_sha256(
                    ok_table_paths(participation_dir, PARTICIPATION_NAME)
                ),
                "path": str(participation_dir),
            },
        },
    }
    if args.gather:
        gather_manifest(output, OUTPUT_NAME, header)
        return
    every_epoch = ok_epochs(read_manifest(participation_dir))
    run(
        epochs=select_epochs(args.epochs or every_epoch, args.shards, args.n_shards),
        task=run_epoch,
        task_args=(event_table, data_dir, output),
        output=output,
        name=OUTPUT_NAME,
        header=header,
        overwrite=args.overwrite,
        max_workers=args.n_workers,
    )


if __name__ == "__main__":
    main()
