#!/usr/bin/env python
"""Held-out tetrode score of every canonical non-local event and its Local-rest control.

For each epoch (``analysis.event_evidence.tetrode_holdout``): refit the decode's
encodings on ``N_FOLDS`` folds of whole trials, leave each tetrode out of the
decoder's smoother over the whole epoch, and score it at every bin of every
window under the remaining tetrodes' position distribution against local coding
around the animal. The windows are the canonical events
(``replay_content/event_table.parquet``, read through the SWR source; exact
first and last decode bins, their SWR label) and the Local-rest window of each
event's length that the participation run drew (``event_participation``).

The encodings are held out by folds of whole trials, as in
``score_nonlocal_restriction.py``: a segment runs from one
departure from a well (the trial's ``movement_start_time``, the final poke-out
at its origin) to the next, so the run to a well and the whole stay there are
one segment, and the segments are dealt into ``N_FOLDS`` folds balanced by
destination well (``regression.folds.trial_folds``), with a margin of
``TRIAL_MARGIN_S``. The time before the first trial and after the last trial's
end are segments of their own two classes, and a segment longer than
``MAX_SEGMENT_S`` is cut into equal pieces, each a group of its own, so that
long stays are spread over the folds. The folds are whole trials because the
animal stays in one patch for long stretches: a place visited in only one
stretch of time would have no data in that stretch's held-out encoding. The
design and its margin are in the header's settings, so a run of another design
does not gather or resume with this one (move its folder aside before a rerun).
The trials are ``load_data``'s (parsed from the DIO events), not the session
cache's, and each epoch's record keeps their digest (``_per_event_run.trials_input``).

Writes ``{epoch_id}_tetrode_holdout.parquet`` per epoch (one row per window and
tetrode: ``window_id`` (the window's number in the epoch), ``event_id`` (the
event's, or for a control the event it matches), ``kind`` (``"event"`` or ``"control"``),
``group``, ``tetrode``, ``n_bins``, ``n_scored_bins``, ``n_spikes``,
``population``, ``local``, ``local_bin``, ``gain``, ``gain_bin``,
``nonlocal_prob``, ``gain_silent``, ``agreement``, ``agreement_silent``,
``uniform``, ``spread_cm``), each epoch's record
``{epoch_id}_tetrode_holdout.json`` (status, the fold design and its margin in
bins, the folds' ``n_segments``, ``n_groups`` and ``trial_bin_share``,
the digest and number of the trials the folds were made from
(``epoch_inputs["trials"]``), settings, the SHA-256 of the event table and the
participation tables, provenance) and ``manifest.json`` gathered from the records
(``runs.epoch_records``).

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
from continuum_swr_replay.analysis.regression.folds import N_FOLDS
from continuum_swr_replay.decoding.identifiers import parse_epoch_identifier
from continuum_swr_replay.parameters import ROOT_DIR
from continuum_swr_replay.paths import (
    EVENT_PARTICIPATION_DIR,
    EVENT_TETRODE_HOLDOUT_DIR,
    PARTICIPATION_NAME,
    TETRODE_HOLDOUT_NAME,
    processed_data_dir,
)
from continuum_swr_replay.runs.epoch_records import gather_manifest
from continuum_swr_replay.runs.io import (
    ok_epochs,
    output_path,
    read_manifest,
)
from continuum_swr_replay.runs.provenance import produced_by, provenance_record
from continuum_swr_replay.tables.io import write_table

# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.event_evidence._per_event_run import (  # noqa: E402
    WINDOW_EVENT_COLUMNS,
    add_gpu_arguments,
    add_run_arguments,
    epoch_inputs,
    event_table_path,
    file_input,
    gpu_pool,
    participation_input,
    run,
    select_epochs,
    trials_input,
)

PRODUCED_BY = produced_by(__file__)


def run_epoch(
    epoch_id: str,
    event_table: Path,
    data_dir: Path,
    output: Path,
) -> dict:
    """Score one epoch and write its table; return its status record."""
    started = clock.time()
    log = logging.getLogger(epoch_id).info
    epoch = epoch_inputs(epoch_id, data_dir, trials=True)
    folds, margin_bins, fold_counts = th.epoch_folds(epoch.time, epoch.loaded["trials"])
    trials = trials_input(epoch.loaded["trials"])
    participation = pd.read_parquet(
        output_path(epoch_id, data_dir / EVENT_PARTICIPATION_DIR, PARTICIPATION_NAME)
    )
    windows = epoch_windows(
        epoch.time,
        read_swr_table(event_table, columns=WINDOW_EVENT_COLUMNS),
        participation,
    )
    # what the epoch needs of the decode frame, copied, so that the frame is
    # not held through the scoring
    head_position = epoch.frame[["head_position_x", "head_position_y"]].to_numpy(
        dtype=float, copy=True
    )
    linear_position = epoch.frame["linear_position"].to_numpy(dtype=float, copy=True)
    time, loaded, detector = epoch.time, epoch.loaded, epoch.detector
    del epoch
    scores, record = th.holdout_scores(
        detector,
        time,
        head_position,
        linear_position,
        loaded["spike_times"],
        loaded["marks"],
        loaded["is_training"],
        loaded["is_missing"],
        windows,
        folds=folds,
        margin_bins=margin_bins,
        log=log,
    )
    columns = windows.set_index("window_id")[["event_id", "kind", "group"]]
    table = scores.join(columns, on="window_id").assign(
        epoch_id=epoch_id, animal=parse_epoch_identifier(epoch_id).animal
    )
    write_table(table, output_path(epoch_id, output, TETRODE_HOLDOUT_NAME))
    return {
        "status": "ok",
        "n_windows": len(windows),
        "n_tetrodes": int(scores["tetrode"].nunique()),
        **record,
        "fold_design": th.TRIAL_DESIGN,
        "margin_bins": margin_bins,
        **fold_counts,
        "epoch_inputs": {"trials": trials},
        "seconds": round(clock.time() - started, 1),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_run_arguments(parser)
    add_gpu_arguments(parser)
    args = parser.parse_args(argv)

    data_dir = args.data_dir or processed_data_dir()
    output = args.output_dir or data_dir / EVENT_TETRODE_HOLDOUT_DIR
    event_table = event_table_path(data_dir)
    participation_dir = data_dir / EVENT_PARTICIPATION_DIR
    header = {
        "produced_by": PRODUCED_BY,
        "provenance": provenance_record(ROOT_DIR),
        "settings": {"n_folds": N_FOLDS, **th.fold_settings()},
        "inputs": {
            "event_table": file_input(event_table),
            "event_participation": participation_input(data_dir),
        },
    }
    if args.gather:
        gather_manifest(output, TETRODE_HOLDOUT_NAME, header)
        return
    max_workers, pool_options = gpu_pool(parser, args)
    every_epoch = ok_epochs(read_manifest(participation_dir))
    run(
        epochs=select_epochs(args.epochs or every_epoch, args.shards, args.n_shards),
        task=run_epoch,
        task_args=(event_table, data_dir, output),
        output=output,
        name=TETRODE_HOLDOUT_NAME,
        header=header,
        overwrite=args.overwrite,
        max_workers=max_workers,
        pool_options=pool_options,
    )


if __name__ == "__main__":
    main()
