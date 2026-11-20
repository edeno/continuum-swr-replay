#!/usr/bin/env python
"""Per-bin scores of the three non-local restriction models on every analyzed epoch.

For each analyzed HPC epoch (``data_loaders.epoch_register.analyzed_epoch_ids``):
the decoders ``none``, ``swr_only`` and ``full``, which differ only in when the
two Non-Local states may be entered, scored at every bin with a likelihood held
out in time (``analysis.event_evidence.nonlocal_restriction``). ``swr_only``
allows the non-local states in every bin inside a detected SWR of the epoch's
SWR source (``swr_source.epoch_source(epoch_id).events``, the Kay inventory),
whether or not the SWR was immobile as a whole event: the model's best case.
That is deliberately not the frame's ``SWR_event`` labels, which keep only the
SWRs the analyzed-time rule keeps.

Each bin is labeled (:func:`bin_categories`): ``excluded`` (in excluded time),
``immobile_in_swr`` / ``immobile_outside_swr`` (analyzed, head speed below the
immobility threshold, inside or outside a source SWR), or ``moving`` (analyzed
and not immobile; a bin without a speed is not immobile, and the record counts
those).

Writes, per epoch, under ``processed_data_dir() / EVENT_NONLOCAL_RESTRICTION_DIR``:

- ``{epoch_id}_nonlocal_restriction.parquet``: the summed scores per category
  (``restriction_sums``: ``category``, ``n_bins``, every model's summed filter
  and two-sided score) with ``epoch_id`` and ``animal``; the table downstream
  statistics read;
- ``{epoch_id}_nonlocal_restriction_bins.parquet``: one row per bin, ``time``,
  the six float32 score columns, ``in_swr``, ``block`` and ``category``, so a
  later per-event or per-ripple-power analysis needs no rerun;
- ``{epoch_id}_nonlocal_restriction.json``: the epoch's record (status,
  ``restriction_scores``' record, bin counts, the per-epoch input digests,
  settings, provenance);

and ``manifest.json`` gathered from the records (``runs.epoch_records``).

An epoch whose models fail a check (a library marginal that disagrees with the
summed filter scores, a non-finite score, or a filtered posterior whose mass is
more than :data:`CAUSAL_MASS_TOLERANCE` from 1) gets status ``failed_check``
and a message naming the failures; its tables are still written, for diagnosis.

Spike times and waveform features are read through ``load_data`` (read-only).
Each worker is given one GPU, ``--workers-per-gpu`` to a card, as in
``score_tetrode_holdout.py``; ``--shards``/``--n-shards`` split the epochs
across hosts writing into one folder.

Usage::

    uv run --frozen python scripts/event_evidence/score_nonlocal_restriction.py \\
        --epochs wilbur20210329_02_r1_HPC --gpus 1
    uv run --frozen python scripts/event_evidence/score_nonlocal_restriction.py \\
        --gpus 1 2 4 5 6 7 8
    uv run --frozen python scripts/event_evidence/score_nonlocal_restriction.py \\
        --gather
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
from numpy.typing import NDArray

from continuum_swr_replay.analysis.event_definition import swr_source
from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    analyzed_bins,
    immobile_analyzed_bins,
    read_dot_with_swr_source,
)
from continuum_swr_replay.analysis.event_evidence import nonlocal_restriction as nr
from continuum_swr_replay.analysis.event_evidence import tetrode_holdout as th
from continuum_swr_replay.data_loaders.epoch_register import analyzed_epoch_ids
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.decoding.identifiers import parse_epoch_identifier
from continuum_swr_replay.intervals import interval_mask
from continuum_swr_replay.parameters import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    ROOT_DIR,
)
from continuum_swr_replay.paths import (
    EVENT_NONLOCAL_RESTRICTION_DIR,
    processed_data_dir,
)
from continuum_swr_replay.runs.epoch_records import gather_manifest
from continuum_swr_replay.runs.io import json_ready, output_path, sha256_file
from continuum_swr_replay.runs.provenance import produced_by, provenance_record
from continuum_swr_replay.tables.io import write_table

# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.event_evidence._per_event_run import (  # noqa: E402
    add_run_arguments,
    run,
    select_epochs,
    use_gpu,
)

PRODUCED_BY = produced_by(__file__)
OUTPUT_NAME = "nonlocal_restriction"
BINS_NAME = "nonlocal_restriction_bins"

CATEGORIES = ("excluded", "immobile_in_swr", "immobile_outside_swr", "moving")
"""Each bin's category (:func:`bin_categories`)."""

CAUSAL_MASS_TOLERANCE = 1e-3
"""Largest allowed distance of a model's filtered posterior mass from 1."""


def bin_categories(
    frame: pd.DataFrame,
    in_swr: NDArray[np.bool_],
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
) -> pd.Categorical:
    """Each bin's category: excluded, immobile in or outside an SWR, or moving.

    Parameters
    ----------
    frame : pd.DataFrame
        A decode frame read through ``analyzed_time.read_dot_with_swr_source``
        (``in_excluded_time`` and ``head_speed``), one row per bin.
    in_swr : ndarray of bool, shape (n_time,)
        Whether each bin lies inside an SWR.
    speed_threshold : float
        cm/s; a bin is immobile when its speed is strictly below it
        (``analyzed_time.immobile_analyzed_bins``).

    Returns
    -------
    pd.Categorical, length n_time
        Categories :data:`CATEGORIES`: ``excluded`` where the bin is not
        analyzed; else ``immobile_in_swr`` / ``immobile_outside_swr`` where it
        is immobile; else ``moving`` (a bin without a speed included).
    """
    in_swr = np.asarray(in_swr, dtype=bool)
    if in_swr.shape != (len(frame),):
        raise ValueError(f"in_swr has shape {in_swr.shape}, not ({len(frame)},)")
    analyzed = analyzed_bins(frame)
    immobile = immobile_analyzed_bins(frame, speed_threshold)
    labels = np.select(
        [~analyzed, immobile & in_swr, immobile],
        ["excluded", "immobile_in_swr", "immobile_outside_swr"],
        default="moving",
    )
    return pd.Categorical(labels, categories=CATEGORIES)


def check_failures(record: dict) -> list[str]:
    """The model checks a ``restriction_scores`` record fails, as messages.

    A model's library marginal must agree with its summed filter scores
    (``marginal_agrees_<model>``), its filtered posterior mass must be within
    :data:`CAUSAL_MASS_TOLERANCE` of 1, and every score must be finite.
    """
    failures = []
    for model in nr.MODELS:
        if not record[f"marginal_agrees_{model}"]:
            failures.append(
                f"{model}: library marginal {record[f'marginal_log_likelihood_{model}']}"
                f" differs from the summed filter scores by "
                f"{record[f'marginal_difference_{model}']}"
            )
        deviation = record[f"causal_mass_deviation_{model}"]
        if not deviation <= CAUSAL_MASS_TOLERANCE:
            failures.append(
                f"{model}: filtered posterior mass is {deviation} from 1 "
                f"(tolerance {CAUSAL_MASS_TOLERANCE})"
            )
    if not record["all_finite"]:
        failures.append("a per-bin score is not finite")
    return failures


def run_epoch(epoch_id: str, data_dir: Path, output: Path) -> dict:
    """Score one epoch, write its two tables and return its record."""
    from continuum_swr_replay.analysis.event_evidence.state_evidence.terms import (
        load_detector,
        load_marks_and_training_mask,
    )

    started = clock.time()
    log = logging.getLogger(epoch_id).info
    paths = decode_paths(epoch_id, data_dir)
    frame = read_dot_with_swr_source(paths.data_over_time)
    time = frame.index.to_numpy(dtype=float)
    inputs = load_marks_and_training_mask(epoch_id)
    if not np.array_equal(inputs["time"], time):
        raise ValueError("the loader's time axis is not the decode's")
    detector = load_detector(epoch_id, data_dir)
    source_events = swr_source.epoch_source(epoch_id).events
    in_swr = interval_mask(time, source_events)
    categories = bin_categories(frame, in_swr)

    scores, record = nr.restriction_scores(
        detector,
        time,
        frame[["head_position_x", "head_position_y"]].to_numpy(dtype=float),
        inputs["spike_times"],
        inputs["marks"],
        inputs["is_training"],
        inputs["is_missing"],
        in_swr,
        log=log,
    )
    animal = parse_epoch_identifier(epoch_id).animal
    sums = (
        nr.restriction_sums(scores, np.asarray(categories))
        .reset_index()
        .assign(epoch_id=epoch_id, animal=animal)
    )
    write_table(sums, output_path(epoch_id, output, OUTPUT_NAME))
    bins_path = write_table(
        scores.assign(category=categories), output_path(epoch_id, output, BINS_NAME)
    )

    analyzed = analyzed_bins(frame)
    immobile = immobile_analyzed_bins(frame)
    in_label = frame[swr_source.SWR_LABEL_COLUMN].to_numpy() > 0
    failures = check_failures(record)
    return json_ready(
        {
            "status": "failed_check" if failures else "ok",
            **({"error": "; ".join(failures)} if failures else {}),
            **record,
            "n_bins": len(frame),
            "n_bins_by_category": {
                c: int(n) for c, n in pd.Series(categories).value_counts().items()
            },
            "n_analyzed_without_speed": int(
                (analyzed & ~np.isfinite(frame["head_speed"].to_numpy())).sum()
            ),
            "n_swr_source_events": len(source_events),
            "n_in_source_swr": int(in_swr.sum()),
            "n_in_swr_label": int(in_label.sum()),
            "n_immobile_in_source_swr": int((immobile & in_swr).sum()),
            "n_immobile_in_swr_label": int((immobile & in_label).sum()),
            "bins_table_bytes": bins_path.stat().st_size,
            "epoch_inputs": {
                "data_over_time": {
                    "sha256": sha256_file(paths.data_over_time),
                    "path": str(paths.data_over_time),
                },
                "detector": {
                    "sha256": sha256_file(paths.detector),
                    "path": str(paths.detector),
                },
                **{
                    f"swr_{name}": entry
                    for name, entry in swr_source.swr_source_files(epoch_id).items()
                },
            },
            "seconds": round(clock.time() - started, 1),
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
    output = args.output_dir or data_dir / EVENT_NONLOCAL_RESTRICTION_DIR
    header = {
        "produced_by": PRODUCED_BY,
        "provenance": provenance_record(ROOT_DIR),
        "settings": {
            "n_folds": nr.N_FOLDS,
            "margin_bins": th.MARGIN_BINS,
            "immobility_threshold_cm_per_s": DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
            "swr_source": swr_source.swr_source_name(),
            "marginal_tolerance": nr.MARGINAL_TOLERANCE,
            "causal_mass_tolerance": CAUSAL_MASS_TOLERANCE,
        },
        # every input is the epoch's own (its decode, detector and SWR files):
        # each record holds their digests under "epoch_inputs"
        "inputs": {},
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
    run(
        epochs=select_epochs(
            args.epochs or analyzed_epoch_ids(), args.shards, args.n_shards
        ),
        task=run_epoch,
        task_args=(data_dir, output),
        output=output,
        name=OUTPUT_NAME,
        header=header,
        overwrite=args.overwrite,
        max_workers=len(args.gpus) * args.workers_per_gpu,
        pool_options={
            "mp_context": context,
            "initializer": use_gpu,
            "initargs": (queue, round(0.9 / args.workers_per_gpu, 3)),
        },
    )


if __name__ == "__main__":
    main()
