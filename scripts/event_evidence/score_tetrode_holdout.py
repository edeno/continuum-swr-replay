#!/usr/bin/env python
"""Held-out tetrode score of every canonical non-local event and its Local-rest control.

For each epoch (``analysis.event_evidence.tetrode_holdout``): refit the decode's
encodings on five contiguous time blocks, leave each tetrode out of the
decoder's smoother over the whole epoch, and score it at every bin of every
window under the remaining tetrodes' position distribution against local coding
around the animal. The windows are the canonical events
(``replay_content/event_table.parquet``, read through the SWR source; exact
first and last decode bins, their SWR label) and the Local-rest window of each
event's length that the participation run drew (``event_participation``).

Writes ``{epoch_id}_tetrode_holdout.parquet`` per epoch (one row per window and
tetrode: ``window_id`` (the window's number in the epoch), ``event_id`` (the
event's, or for a control the event it matches), ``kind`` (``"event"`` or ``"control"``),
``group``, ``tetrode``, ``n_bins``, ``n_scored_bins``, ``n_spikes``,
``population``, ``local``, ``local_bin``, ``gain``, ``gain_bin``,
``nonlocal_prob``) and ``manifest.json`` (settings, the SHA-256 of the event
table and the participation tables, provenance, each epoch's record).

Spike times and waveform features are read through ``load_data`` (read-only).
Each worker claims one GPU (``runs.device.configure_jax_gpu``).

Usage::

    uv run --frozen python scripts/event_evidence/score_tetrode_holdout.py \\
        --epochs wilbur20210329_02_r1_HPC --gpus 1
    uv run --frozen python scripts/event_evidence/score_tetrode_holdout.py \\
        --gpus 1 2
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import time as clock
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_definition.analysed_time import (
    read_dot_with_swr_source,
    read_swr_table,
)
from continuum_swr_replay.analysis.event_evidence import tetrode_holdout as th
from continuum_swr_replay.analysis.event_evidence.participation import (
    ON_GRID_ATOL_S,
    event_bins,
)
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.decoding.identifiers import parse_epoch_identifier
from continuum_swr_replay.parameters import ROOT_DIR
from continuum_swr_replay.paths import (
    EVENT_PARTICIPATION_DIR,
    EVENT_TETRODE_HOLDOUT_DIR,
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.io import (
    files_sha256,
    ok_epochs,
    ok_table_paths,
    output_path,
    read_manifest,
    sha256_file,
    write_json,
)
from continuum_swr_replay.runs.provenance import produced_by, provenance_record
from continuum_swr_replay.tables.io import write_table

PRODUCED_BY = produced_by(__file__)
OUTPUT_NAME = "tetrode_holdout"
PARTICIPATION_NAME = "participation"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def epoch_windows(
    time: np.ndarray, event_table: pd.DataFrame, participation: pd.DataFrame
) -> pd.DataFrame:
    """The epoch's events and their Local-rest controls as decode-bin windows.

    Raises
    ------
    ValueError
        If an event is not in the event table, or a window is off the decode's
        grid.
    """
    measured = participation[participation["kind"] == "event"][["event_id", "group"]]
    events = event_table.merge(measured, on="event_id", validate="one_to_one")
    if len(events) != len(measured):
        raise ValueError("a measured event is not in the event table")
    s, e = event_bins(
        time, events["event_start_s"].to_numpy(), events["event_end_s"].to_numpy()
    )
    controls = participation[participation["kind"] == "control"]
    cs = np.searchsorted(time, controls["start_s"].to_numpy())
    cs = np.minimum(cs, time.size - 1)
    if not (np.abs(time[cs] - controls["start_s"].to_numpy()) <= ON_GRID_ATOL_S).all():
        raise ValueError("a control window does not start on a decode bin")
    return (
        pd.concat(
            [
                pd.DataFrame(
                    {
                        "event_id": events["event_id"].to_numpy(),
                        "kind": "event",
                        "group": events["group"].to_numpy(),
                        "s": s,
                        "e": e,
                    }
                ),
                pd.DataFrame(
                    {
                        "event_id": controls["event_id"].to_numpy(),
                        "kind": "control",
                        "group": controls["group"].to_numpy(),
                        "s": cs,
                        "e": cs + controls["n_bins"].to_numpy() - 1,
                    }
                ),
            ],
            ignore_index=True,
        )
        .rename_axis("window_id")
        .reset_index()
    )


def run_epoch(epoch_id: str, event_table: Path, data_dir: Path, output: Path) -> dict:
    """Score one epoch and write its table; return its status record."""
    from continuum_swr_replay.analysis.event_evidence.state_evidence.terms import (
        load_detector,
        load_marks_and_training_mask,
    )

    started = clock.time()
    log = logging.getLogger(epoch_id).info
    frame = read_dot_with_swr_source(decode_paths(epoch_id, data_dir).data_over_time)
    time = frame.index.to_numpy(dtype=float)
    participation = pd.read_parquet(
        output_path(epoch_id, data_dir / EVENT_PARTICIPATION_DIR, PARTICIPATION_NAME)
    )
    windows = epoch_windows(time, read_swr_table(event_table), participation)
    inputs = load_marks_and_training_mask(epoch_id)
    if not np.array_equal(inputs["time"], time):
        raise ValueError("the loader's time axis is not the decode's")
    detector = load_detector(epoch_id, data_dir)
    scores, record = th.holdout_scores(
        detector,
        time,
        frame[["head_position_x", "head_position_y"]].to_numpy(dtype=float),
        frame["linear_position"].to_numpy(dtype=float),
        inputs["spike_times"],
        inputs["marks"],
        inputs["is_training"],
        inputs["is_missing"],
        windows,
        log=log,
    )
    columns = windows.set_index("window_id")[["event_id", "kind", "group"]]
    table = scores.join(columns, on="window_id").assign(
        epoch_id=epoch_id, animal=parse_epoch_identifier(epoch_id).animal
    )
    write_table(table, output_path(epoch_id, output, OUTPUT_NAME))
    return {
        "status": "ok",
        "n_windows": len(windows),
        "n_tetrodes": int(scores["tetrode"].nunique()),
        **record,
        "seconds": round(clock.time() - started, 1),
    }


def _claim_gpu(queue) -> None:
    """Worker initializer: claim the next GPU from ``queue``."""
    from continuum_swr_replay.runs.device import configure_jax_gpu

    configure_jax_gpu(gpu=queue.get())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--epochs", nargs="+", help="epoch ids; default every epoch")
    parser.add_argument(
        "--gpus", type=int, nargs="+", required=True, help="one worker per card"
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    provenance = provenance_record(ROOT_DIR)
    data_dir = args.data_dir or processed_data_dir()
    output = args.output_dir or data_dir / EVENT_TETRODE_HOLDOUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    event_table = table_path(
        "event_table", data_dir / replay_content_dir().relative_to(processed_data_dir())
    )
    participation_dir = data_dir / EVENT_PARTICIPATION_DIR
    inputs = {
        "event_table": {"sha256": sha256_file(event_table)},
        "event_participation": {
            "sha256": files_sha256(
                ok_table_paths(participation_dir, PARTICIPATION_NAME)
            )
        },
    }
    epochs = args.epochs or ok_epochs(read_manifest(participation_dir))
    to_run = [
        epoch_id
        for epoch_id in epochs
        if args.overwrite or not output_path(epoch_id, output, OUTPUT_NAME).exists()
    ]
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    for gpu in args.gpus:
        queue.put(gpu)
    records: dict[str, dict] = {}
    with ProcessPoolExecutor(
        max_workers=len(args.gpus),
        mp_context=context,
        initializer=_claim_gpu,
        initargs=(queue,),
    ) as pool:
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
            "inputs": inputs,
            "settings": {
                "event_table": str(event_table),
                "n_folds": th.N_FOLDS,
                "margin_bins": th.MARGIN_BINS,
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
