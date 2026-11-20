#!/usr/bin/env python
"""Per-bin scores of the three non-local restriction models on every analyzed epoch.

For each analyzed HPC epoch (``data_loaders.epoch_register.analyzed_epoch_ids``):
the decoders ``none``, ``swr`` and ``full``, which differ only in when the two
Non-Local states may be occupied, scored at every bin with a likelihood held
out in time (``analysis.event_evidence.nonlocal_restriction``). ``swr`` is
non-local in every bin inside a detected SWR of the epoch's SWR source
(``swr_source.epoch_source(epoch_id).events``, the Kay inventory), whether or
not the SWR was immobile as a whole event, and local everywhere else. That is
deliberately not the frame's ``SWR_event`` labels, which keep only the SWRs the
analyzed-time rule keeps.

Each bin is labeled (``nonlocal_restriction.bin_categories``): ``excluded``
(in excluded time), ``immobile_in_swr`` / ``immobile_outside_swr`` (analyzed,
head speed below the immobility threshold, inside or outside a source SWR), or
``moving`` (analyzed and not immobile; a bin without a speed is not immobile,
and the record counts those).

Writes, per epoch, under ``processed_data_dir() / EVENT_NONLOCAL_RESTRICTION_DIR``:

- ``{epoch_id}_nonlocal_restriction.parquet``: the summed scores per category
  (``restriction_sums``: ``category``, ``n_bins``, every model's summed filter
  and two-sided score) with ``epoch_id`` and ``animal``; the table downstream
  statistics read;
- ``{epoch_id}_nonlocal_restriction_bins.parquet``: one row per bin, ``time``,
  the six float32 score columns, ``in_swr``, ``block`` and ``category``, so a
  later per-event or per-ripple-power analysis needs no rerun;
- ``{epoch_id}_nonlocal_restriction.json``: the epoch's record (status,
  ``restriction_scores``' record, bin counts, the share of the non-local gain
  outside SWRs over the immobile bins (``share_outside_swr``, both families),
  the per-epoch input digests, settings, provenance);

and ``manifest.json`` gathered from the records (``runs.epoch_records``).

An epoch whose models fail a check (``nonlocal_restriction.check_failures``: a
library marginal that disagrees with the summed filter scores, or a filtered
posterior whose mass is more than ``CAUSAL_MASS_TOLERANCE`` from 1) gets status
``failed_check`` and a message naming the failures; its tables are still
written, for diagnosis.
A labeled bin with a score that is not finite fails the epoch when its sums are
taken (``restriction_sums`` raises), after its per-bin table is written. A
failed or ``failed_check`` epoch can leave tables on disk (and, after an
``--overwrite`` rerun that fails, a sums table older than its bins table), so a
reader takes only the epochs whose record is ``ok`` (``runs.io.ok_table_paths``,
as the other per-epoch outputs are read).

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
import sys
import time as clock
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_definition import swr_source
from continuum_swr_replay.analysis.event_evidence import nonlocal_restriction as nr
from continuum_swr_replay.analysis.event_evidence import tetrode_holdout as th
from continuum_swr_replay.data_loaders.epoch_register import analyzed_epoch_ids
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
    add_gpu_arguments,
    add_run_arguments,
    epoch_inputs,
    gpu_pool,
    run,
    select_epochs,
)

PRODUCED_BY = produced_by(__file__)
OUTPUT_NAME = "nonlocal_restriction"
BINS_NAME = "nonlocal_restriction_bins"


def run_epoch(epoch_id: str, data_dir: Path, output: Path) -> dict:
    """Score one epoch, write its two tables and return its record."""
    started = clock.time()
    log = logging.getLogger(epoch_id).info
    epoch = epoch_inputs(epoch_id, data_dir)
    frame = epoch.frame
    source_events = swr_source.epoch_source(epoch_id).events
    in_swr = interval_mask(epoch.time, source_events)
    in_label = frame[swr_source.SWR_LABEL_COLUMN].to_numpy() > 0
    if (in_label & ~in_swr).any():
        raise ValueError(
            f"{int((in_label & ~in_swr).sum())} bins the frame labels as an SWR "
            "are outside every SWR of the source"
        )
    categories = nr.bin_categories(frame, in_swr)

    scores, record = nr.restriction_scores(
        epoch.detector,
        epoch.time,
        frame[["head_position_x", "head_position_y"]].to_numpy(dtype=float),
        epoch.loaded["spike_times"],
        epoch.loaded["marks"],
        epoch.loaded["is_training"],
        epoch.loaded["is_missing"],
        in_swr,
        log=log,
    )
    bins_path = write_table(
        scores.assign(category=categories), output_path(epoch_id, output, BINS_NAME)
    )
    sums = nr.restriction_sums(scores, np.asarray(categories))
    share = nr.share_outside_swr(sums)
    animal = parse_epoch_identifier(epoch_id).animal
    write_table(
        sums.reset_index().assign(epoch_id=epoch_id, animal=animal),
        output_path(epoch_id, output, OUTPUT_NAME),
    )

    labels = np.asarray(categories)
    analyzed = labels != "excluded"
    immobile = np.isin(labels, [nr.IMMOBILE_IN_SWR, nr.IMMOBILE_OUTSIDE_SWR])
    failures = nr.check_failures(record)
    return json_ready(
        {
            "status": "failed_check" if failures else "ok",
            **({"error": "; ".join(failures)} if failures else {}),
            **record,
            **share,
            "n_bins": len(frame),
            "n_bins_by_category": {
                c: int(n) for c, n in pd.Series(categories).value_counts().items()
            },
            "n_analyzed_without_speed": int(
                (analyzed & ~np.isfinite(frame["head_speed"].to_numpy())).sum()
            ),
            "n_swr_source_events": len(source_events),
            "n_in_swr_label": int(in_label.sum()),
            "n_immobile_in_swr_label": int((immobile & in_label).sum()),
            "bins_table_bytes": bins_path.stat().st_size,
            "epoch_inputs": {
                "data_over_time": {
                    "sha256": sha256_file(epoch.paths.data_over_time),
                    "path": str(epoch.paths.data_over_time),
                },
                "detector": {
                    "sha256": sha256_file(epoch.paths.detector),
                    "path": str(epoch.paths.detector),
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
    add_gpu_arguments(parser)
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
            "models": list(nr.MODELS),
            "marginal_tolerance": nr.MARGINAL_TOLERANCE,
            "causal_mass_tolerance": nr.CAUSAL_MASS_TOLERANCE,
            "predicted_mass_tolerance": nr.PREDICTED_MASS_TOLERANCE,
            "continuous_block_tolerance": nr.CONTINUOUS_BLOCK_TOLERANCE,
        },
        # every input is the epoch's own (its decode, detector and SWR files):
        # each record holds their digests under "epoch_inputs"
        "inputs": {},
    }
    if args.gather:
        gather_manifest(output, OUTPUT_NAME, header)
        return
    max_workers, pool_options = gpu_pool(parser, args)
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
        max_workers=max_workers,
        pool_options=pool_options,
    )


if __name__ == "__main__":
    main()
