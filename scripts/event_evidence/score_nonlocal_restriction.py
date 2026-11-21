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

The likelihood is held out by folds of whole trials (``--fold-design trials``,
the default): a segment runs from one departure from a well (the trial's
``movement_start_time``, the final poke-out at its origin) to the next, so the
run to a well and the whole stay there are one segment, and the segments are
dealt into ``N_FOLDS`` folds balanced by destination well
(``tetrode_holdout.trial_folds``), with a margin of ``TRIAL_MARGIN_S``. The
time before the first trial and after the last trial's end are segments of
their own two classes, and a segment longer than ``MAX_SEGMENT_S`` is cut into
equal pieces, each a group of its own, so that long stays are spread over the
folds.
``--fold-design contiguous`` reproduces the earlier design, ``N_FOLDS``
contiguous time blocks with a margin of ``MARGIN_BINS`` bins. The design and its
margin are in the header's settings, so a resume cannot mix designs.

The whole decoder is held out (``nonlocal_restriction.fold_fit_scores``): per
fold, a detector configured as the canonical one is fit by the library's EM on
the fold's training folds with the fold's bins missing, and the fold's bins are
scored under that fit. The bins table also has the ``full`` model's smoothed
probability of each state (``full_prob_*``) and its most likely non-local place
bin (``full_nonlocal_place_bin``, an index into the track-interior bins), and
the record one entry per fold (``folds``: the fitted discrete transitions and
initial distribution and their largest differences from the canonical, the EM
iterations, convergence and marginal trace, each stage's seconds, the model
checks, the device's peak memory). The header's settings record the fit
(``"fit": "folds"``) and the EM settings; an earlier version of this runner
held out only the encodings, and its header records no ``fit``, so a resume or
a gather cannot mix the two. A fold fit costs about what the canonical decode
costs, five times an epoch.

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
  the six float32 score columns, ``in_swr``, ``block`` (the bin's fold),
  ``place_sampling_s`` (the seconds its fold's encoding was fit on at the
  animal's place bin) and ``category``, so a later per-event or
  per-ripple-power analysis needs no rerun;
- ``{epoch_id}_nonlocal_restriction.json``: the epoch's record (status,
  ``fold_fit_scores``' record, bin counts, the share of the non-local gain
  outside SWRs over the immobile bins (``share_outside_swr``, both families),
  the folds (``n_segments``, ``n_groups``, the smallest and largest fold's share of the
  bins, the margin in bins), the share of the immobile bins whose fold's
  encoding had 0 s and at most 10 s of training data at the animal's place,
  the per-epoch input digests, settings, provenance);

and ``manifest.json`` gathered from the records (``runs.epoch_records``).

An epoch whose models fail a check (``nonlocal_restriction.fold_check_failures``:
a fold whose EM did not converge, or a fold's pass that fails
``check_failures``, such as a library marginal that disagrees with the summed
filter scores, or a filtered posterior whose mass is more than
``CAUSAL_MASS_TOLERANCE`` from 1) gets status
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
    uv run --frozen python scripts/event_evidence/score_nonlocal_restriction.py \\
        --gpus 1 2 4 5 6 7 8 --shards 0 --n-shards 2
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
    NONLOCAL_RESTRICTION_BINS_NAME,
    NONLOCAL_RESTRICTION_NAME,
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

SPARSE_SAMPLING_S = 10.0
"""The record's second sampling share: immobile bins whose fold's encoding had
at most this many seconds of training data at the animal's place."""


def fit_settings() -> dict:
    """The header's settings of the fit: its name and the EM settings of every
    fold's fit (``nonlocal_restriction.em_settings``)."""
    return {
        "fit": nr.FIT,
        "em": nr.em_settings(),
    }


def run_epoch(
    epoch_id: str,
    data_dir: Path,
    output: Path,
    fold_design: str = th.TRIAL_DESIGN,
) -> dict:
    """Score one epoch, write its two tables and return its record."""
    started = clock.time()
    log = logging.getLogger(epoch_id).info
    epoch = epoch_inputs(epoch_id, data_dir, trials=fold_design == th.TRIAL_DESIGN)
    frame = epoch.frame
    folds, margin_bins, fold_counts = th.epoch_folds(
        epoch.time, epoch.loaded.get("trials"), fold_design
    )
    source_events = swr_source.epoch_source(epoch_id).events
    in_swr = interval_mask(epoch.time, source_events)
    in_label = frame[swr_source.SWR_LABEL_COLUMN].to_numpy() > 0
    if (in_label & ~in_swr).any():
        raise ValueError(
            f"{int((in_label & ~in_swr).sum())} bins the frame labels as an SWR "
            "are outside every SWR of the source"
        )
    categories = nr.bin_categories(frame, in_swr)

    scores, record = nr.fold_fit_scores(
        epoch.detector,
        epoch.time,
        frame[["head_position_x", "head_position_y"]].to_numpy(dtype=float),
        epoch.loaded["spike_times"],
        epoch.loaded["marks"],
        epoch.loaded["is_training"],
        epoch.loaded["is_missing"],
        in_swr,
        folds=folds,
        margin_bins=margin_bins,
        linear_position=frame["linear_position"].to_numpy(dtype=float),
        log=log,
    )
    failures = nr.fold_check_failures(record)
    bins_path = write_table(
        scores.assign(category=categories),
        output_path(epoch_id, output, NONLOCAL_RESTRICTION_BINS_NAME),
    )
    sums = nr.restriction_sums(scores, np.asarray(categories))
    share = nr.share_outside_swr(sums)
    animal = parse_epoch_identifier(epoch_id).animal
    write_table(
        sums.reset_index().assign(epoch_id=epoch_id, animal=animal),
        output_path(epoch_id, output, NONLOCAL_RESTRICTION_NAME),
    )

    labels = np.asarray(categories)
    analyzed = labels != "excluded"
    immobile = np.isin(labels, [nr.IMMOBILE_IN_SWR, nr.IMMOBILE_OUTSIDE_SWR])
    sampling = scores["place_sampling_s"].to_numpy()[immobile]
    fold_share = np.bincount(scores["block"], minlength=nr.N_FOLDS) / len(scores)
    return json_ready(
        {
            "status": "failed_check" if failures else "ok",
            **({"error": "; ".join(failures)} if failures else {}),
            **record,
            **share,
            "fit": nr.FIT,
            "fold_design": fold_design,
            "margin_bins": margin_bins,
            **fold_counts,
            "fold_share_min": float(fold_share.min()),
            "fold_share_max": float(fold_share.max()),
            "immobile_share_place_sampling_0s": float(np.mean(sampling == 0)),
            "immobile_share_place_sampling_le_10s": float(
                np.mean(sampling <= SPARSE_SAMPLING_S)
            ),
            "n_immobile_without_place": int(np.isnan(sampling).sum()),
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
    parser.add_argument(
        "--fold-design",
        choices=th.FOLD_DESIGNS,
        default=th.TRIAL_DESIGN,
        help="whole trials balanced by destination well (default), or the "
        "earlier contiguous time blocks",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_dir or processed_data_dir()
    output = args.output_dir or data_dir / EVENT_NONLOCAL_RESTRICTION_DIR
    header = {
        "produced_by": PRODUCED_BY,
        "provenance": provenance_record(ROOT_DIR),
        "settings": {
            "n_folds": nr.N_FOLDS,
            **fit_settings(),
            **th.fold_settings(args.fold_design),
            "sparse_sampling_s": SPARSE_SAMPLING_S,
            "immobility_threshold_cm_per_s": DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
            "swr_source": swr_source.swr_source_name(),
            "models": list(nr.MODELS),
            "filter_identity_tolerance": nr.FILTER_IDENTITY_TOLERANCE,
            "causal_mass_tolerance": nr.CAUSAL_MASS_TOLERANCE,
            "predicted_mass_tolerance": nr.PREDICTED_MASS_TOLERANCE,
            "continuous_block_tolerance": nr.CONTINUOUS_BLOCK_TOLERANCE,
        },
        # every input is the epoch's own (its decode, detector and SWR files):
        # each record holds their digests under "epoch_inputs"
        "inputs": {},
    }
    if args.gather:
        gather_manifest(output, NONLOCAL_RESTRICTION_NAME, header)
        return
    max_workers, pool_options = gpu_pool(parser, args)
    run(
        epochs=select_epochs(
            args.epochs or analyzed_epoch_ids(), args.shards, args.n_shards
        ),
        task=run_epoch,
        task_args=(data_dir, output, args.fold_design),
        output=output,
        name=NONLOCAL_RESTRICTION_NAME,
        header=header,
        overwrite=args.overwrite,
        max_workers=max_workers,
        pool_options=pool_options,
    )


if __name__ == "__main__":
    main()
