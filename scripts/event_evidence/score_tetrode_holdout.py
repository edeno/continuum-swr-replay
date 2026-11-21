#!/usr/bin/env python
"""Held-out tetrode score of every canonical non-local event and its Local-rest control.

For each epoch (``analysis.event_evidence.tetrode_holdout``): refit the decode's
encodings on ``N_FOLDS`` folds of time, leave each tetrode out of the
decoder's smoother over the whole epoch, and score it at every bin of every
window under the remaining tetrodes' position distribution against local coding
around the animal. The windows are the canonical events
(``replay_content/event_table.parquet``, read through the SWR source; exact
first and last decode bins, their SWR label) and the Local-rest window of each
event's length that the participation run drew (``event_participation``).

The encodings are held out by folds of whole trials (``--fold-design trials``,
the default), as in ``score_nonlocal_restriction.py``: a segment runs from one
departure from a well (the trial's ``movement_start_time``, the final poke-out
at its origin) to the next, so the run to a well and the whole stay there are
one segment, and the segments are dealt into ``N_FOLDS`` folds balanced by
destination well (``tetrode_holdout.trial_folds``), with a margin of
``TRIAL_MARGIN_S``. The time before the first trial and after the last trial's
end are segments of their own two classes, and a segment longer than
``MAX_SEGMENT_S`` is cut into equal pieces, each a group of its own, so that
long stays are spread over the folds.
``--fold-design contiguous`` reproduces the earlier design, ``N_FOLDS``
contiguous time blocks with a margin of ``MARGIN_BINS`` bins. The design and its
margin are in the header's settings, so a resume cannot mix designs. The run
made before this option has a header without them, so its records gather or
resume under neither design (by intent: move its folder aside before a rerun).

Writes ``{epoch_id}_tetrode_holdout.parquet`` per epoch (one row per window and
tetrode: ``window_id`` (the window's number in the epoch), ``event_id`` (the
event's, or for a control the event it matches), ``kind`` (``"event"`` or ``"control"``),
``group``, ``tetrode``, ``n_bins``, ``n_scored_bins``, ``n_spikes``,
``population``, ``local``, ``local_bin``, ``gain``, ``gain_bin``,
``nonlocal_prob``, ``gain_silent``, ``agreement``, ``agreement_silent``,
``uniform``, ``spread_cm``), each epoch's record
``{epoch_id}_tetrode_holdout.json`` (status, the fold design and its margin in
bins, the folds' ``n_segments``, ``n_groups`` and ``trial_bin_share`` (None for
``contiguous``), settings, the SHA-256 of the event table and the participation
tables, provenance) and ``manifest.json`` gathered
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
    uv run --frozen python scripts/event_evidence/score_tetrode_holdout.py \\
        --fold-design contiguous --gpus 1
"""

from __future__ import annotations

import argparse
import logging
import sys
import time as clock
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    read_swr_table,
)
from continuum_swr_replay.analysis.event_evidence import tetrode_holdout as th
from continuum_swr_replay.analysis.event_evidence.state_evidence.windows import (
    epoch_windows,
)
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
    add_gpu_arguments,
    add_run_arguments,
    epoch_inputs,
    event_table_input,
    event_table_path,
    gpu_pool,
    run,
    select_epochs,
)

PRODUCED_BY = produced_by(__file__)
OUTPUT_NAME = "tetrode_holdout"
PARTICIPATION_NAME = "participation"


def run_epoch(
    epoch_id: str,
    event_table: Path,
    data_dir: Path,
    output: Path,
    fold_design: str = th.TRIAL_DESIGN,
) -> dict:
    """Score one epoch and write its table; return its status record."""
    started = clock.time()
    log = logging.getLogger(epoch_id).info
    epoch = epoch_inputs(epoch_id, data_dir, trials=fold_design == th.TRIAL_DESIGN)
    folds, margin_bins, fold_counts = th.epoch_folds(
        epoch.time, epoch.loaded.get("trials"), fold_design
    )
    participation = pd.read_parquet(
        output_path(epoch_id, data_dir / EVENT_PARTICIPATION_DIR, PARTICIPATION_NAME)
    )
    windows = epoch_windows(epoch.time, read_swr_table(event_table), participation)
    scores, record = th.holdout_scores(
        epoch.detector,
        epoch.time,
        epoch.frame[["head_position_x", "head_position_y"]].to_numpy(dtype=float),
        epoch.frame["linear_position"].to_numpy(dtype=float),
        epoch.loaded["spike_times"],
        epoch.loaded["marks"],
        epoch.loaded["is_training"],
        epoch.loaded["is_missing"],
        windows,
        folds=folds,
        margin_bins=margin_bins,
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
        "fold_design": fold_design,
        "margin_bins": margin_bins,
        **fold_counts,
        "seconds": round(clock.time() - started, 1),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_run_arguments(parser)
    add_gpu_arguments(parser)
    parser.add_argument(
        "--fold-design",
        choices=th.FOLD_DESIGNS,
        default=th.TRIAL_DESIGN,
        help="whole trials balanced by destination well (default), or the "
        "earlier contiguous time blocks",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_dir or processed_data_dir()
    output = args.output_dir or data_dir / EVENT_TETRODE_HOLDOUT_DIR
    event_table = event_table_path(data_dir)
    participation_dir = data_dir / EVENT_PARTICIPATION_DIR
    header = {
        "produced_by": PRODUCED_BY,
        "provenance": provenance_record(ROOT_DIR),
        "settings": {"n_folds": th.N_FOLDS, **th.fold_settings(args.fold_design)},
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
    max_workers, pool_options = gpu_pool(parser, args)
    every_epoch = ok_epochs(read_manifest(participation_dir))
    run(
        epochs=select_epochs(args.epochs or every_epoch, args.shards, args.n_shards),
        task=run_epoch,
        task_args=(event_table, data_dir, output, args.fold_design),
        output=output,
        name=OUTPUT_NAME,
        header=header,
        overwrite=args.overwrite,
        max_workers=max_workers,
        pool_options=pool_options,
    )


if __name__ == "__main__":
    main()
