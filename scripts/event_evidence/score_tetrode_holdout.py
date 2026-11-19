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
``nonlocal_prob``, ``gain_silent``, ``agreement``, ``agreement_silent``,
``uniform``, ``spread_cm``), each epoch's record
``{epoch_id}_tetrode_holdout.json`` (status, settings, the SHA-256 of the event
table and the participation tables, provenance) and ``manifest.json`` gathered
from the records (``runs.epoch_records``).

Spike times and waveform features are read through ``load_data`` (read-only).
Each worker is given one GPU, ``--workers-per-gpu`` to a card, each allowed
that share of its memory, and checks that JAX is on it
(``runs.device.verify_jax_device``).

Several hosts split the epochs with ``--shards``/``--n-shards`` (the epochs,
sorted, whose position modulo ``n`` is one of the host's shards), all writing
into one folder: each epoch has its own record, so a host's gather sees every
record written so far, and the last host to finish gathers them all. A re-run
(another host, more memory per worker) runs only the epochs without an ``ok``
record of the same settings and inputs.

Usage::

    uv run --frozen python scripts/event_evidence/score_tetrode_holdout.py \\
        --epochs wilbur20210329_02_r1_HPC --gpus 1
    uv run --frozen python scripts/event_evidence/score_tetrode_holdout.py \\
        --gpus 1 4 5 --workers-per-gpu 2 --shards 0 1 --n-shards 5
    uv run --frozen python scripts/event_evidence/score_tetrode_holdout.py --gather
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

from continuum_swr_replay.analysis.event_definition.analysed_time import (
    read_dot_with_swr_source,
    read_swr_table,
)
from continuum_swr_replay.analysis.event_evidence import tetrode_holdout as th
from continuum_swr_replay.analysis.event_evidence.state_evidence.windows import (
    event_bins,
)
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.decoding.identifiers import parse_epoch_identifier
from continuum_swr_replay.parameters import ROOT_DIR
from continuum_swr_replay.paths import (
    EVENT_PARTICIPATION_DIR,
    EVENT_TETRODE_HOLDOUT_DIR,
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
OUTPUT_NAME = "tetrode_holdout"
PARTICIPATION_NAME = "participation"


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
    control_start = controls["start_s"].to_numpy()
    cs, _ = event_bins(time, control_start, control_start)
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


def _use_gpu(queue, mem_fraction: float) -> None:
    """Worker initializer: run JAX on the next card from ``queue``.

    Cards are assigned, several workers to a card, so the card is set here
    rather than claimed (``configure_jax_gpu`` allows one process per card).
    """
    import os

    from continuum_swr_replay.runs.device import query_gpus, verify_jax_device

    index = queue.get()
    card = {c.index: c for c in query_gpus()}[index]
    os.environ["CUDA_VISIBLE_DEVICES"] = card.uuid
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(mem_fraction)
    verify_jax_device(
        {
            "jax_backend": "gpu",
            "gpu_index": card.index,
            "gpu_uuid": card.uuid,
            "gpu_name": card.name,
        }
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_run_arguments(parser)
    parser.add_argument("--gpus", type=int, nargs="+", help="the cards to use")
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    args = parser.parse_args(argv)

    data_dir = args.data_dir or processed_data_dir()
    output = args.output_dir or data_dir / EVENT_TETRODE_HOLDOUT_DIR
    event_table = event_table_path(data_dir)
    participation_dir = data_dir / EVENT_PARTICIPATION_DIR
    header = {
        "produced_by": PRODUCED_BY,
        "provenance": provenance_record(ROOT_DIR),
        "settings": {"n_folds": th.N_FOLDS, "margin_bins": th.MARGIN_BINS},
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
    if not args.gpus:
        parser.error("--gpus is required")
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    for gpu in args.gpus:
        for _ in range(args.workers_per_gpu):
            queue.put(gpu)
    every_epoch = ok_epochs(read_manifest(participation_dir))
    run(
        epochs=select_epochs(args.epochs or every_epoch, args.shards, args.n_shards),
        task=run_epoch,
        task_args=(event_table, data_dir, output),
        output=output,
        name=OUTPUT_NAME,
        header=header,
        overwrite=args.overwrite,
        max_workers=len(args.gpus) * args.workers_per_gpu,
        pool_options={
            "mp_context": context,
            "initializer": _use_gpu,
            "initargs": (queue, round(0.9 / args.workers_per_gpu, 3)),
        },
    )


if __name__ == "__main__":
    main()
