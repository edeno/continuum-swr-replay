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

``--fit shared`` (the default) holds out the encoding only and runs the models
with the canonical decoder's transitions and initial conditions
(``nonlocal_restriction.restriction_scores``). ``--fit folds`` holds out the
whole decoder: per fold, a detector configured as the canonical one is fit by
the library's EM on the fold's training folds with the fold's bins missing, and
the fold's bins are scored under that fit
(``nonlocal_restriction.fold_fit_scores``). It writes under
``EVENT_NONLOCAL_RESTRICTION_FOLD_FIT_DIR``; its bins table also has the
``full`` model's smoothed probability of each state (``full_prob_*``) and its
most likely non-local place bin (``full_nonlocal_place_bin``, an index into the
track-interior bins), and its record one entry per fold (``folds``: the fitted
discrete transitions and initial distribution and their largest differences
from the canonical, the EM iterations, convergence and marginal trace, each
stage's seconds, the model checks, the device's peak memory). The fit mode and
the EM settings are in the header's settings (the shared mode's header is the
earlier runs'), so a resume or a gather cannot mix modes. A fold fit costs
about what the canonical decode costs, five times an epoch.

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
  ``restriction_scores``' record, bin counts, the share of the non-local gain
  outside SWRs over the immobile bins (``share_outside_swr``, both families),
  the folds (``n_segments``, ``n_groups``, the smallest and largest fold's share of the
  bins, the margin in bins), the share of the immobile bins whose fold's
  encoding had 0 s and at most 10 s of training data at the animal's place,
  the per-epoch input digests, settings, provenance);

and ``manifest.json`` gathered from the records (``runs.epoch_records``).

An epoch whose models fail a check (``nonlocal_restriction.check_failures``: a
library marginal that disagrees with the summed filter scores, or a filtered
posterior whose mass is more than ``CAUSAL_MASS_TOLERANCE`` from 1; with
``--fit folds``, any fold's pass, ``fold_check_failures``) gets status
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
        --fit folds --gpus 1 2 4 5 6 7 8 --shards 0 --n-shards 2
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
    SAMPLING_FREQUENCY,
)
from continuum_swr_replay.paths import (
    EVENT_NONLOCAL_RESTRICTION_DIR,
    EVENT_NONLOCAL_RESTRICTION_FOLD_FIT_DIR,
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

FOLD_DESIGNS = ("trials", "contiguous")
"""``trials``: whole trials balanced by destination well
(``tetrode_holdout.trial_folds``); ``contiguous``: contiguous time blocks."""

FIT_MODES = ("shared", "folds")
"""``shared``: held-out encodings with the canonical decoder's transitions and
initial conditions (``nonlocal_restriction.restriction_scores``); ``folds``:
the whole decoder refit by EM on each fold's training folds
(``nonlocal_restriction.fold_fit_scores``)."""

OUTPUT_DIRS = {
    "shared": EVENT_NONLOCAL_RESTRICTION_DIR,
    "folds": EVENT_NONLOCAL_RESTRICTION_FOLD_FIT_DIR,
}
"""Each fit mode's default output folder under the processed-data root."""

SPARSE_SAMPLING_S = 10.0
"""The record's second sampling share: immobile bins whose fold's encoding had
at most this many seconds of training data at the animal's place."""


def fold_settings(fold_design: str) -> dict:
    """The header's record of a fold design: its name and margin in seconds;
    for ``trials`` also the segment cap and the scikit-learn version, whose
    ``StratifiedGroupKFold`` heuristic assigns the folds."""
    if fold_design == "trials":
        import sklearn

        return {
            "fold_design": "trials",
            "fold_classes": "to_well",
            "margin_s": th.TRIAL_MARGIN_S,
            "max_segment_s": th.MAX_SEGMENT_S,
            "min_trial_bin_share": th.MIN_TRIAL_BIN_SHARE,
            "sklearn_version": sklearn.__version__,
        }
    if fold_design == "contiguous":
        return {
            "fold_design": "contiguous",
            "margin_s": th.MARGIN_BINS / SAMPLING_FREQUENCY,
            "margin_bins": th.MARGIN_BINS,
        }
    raise ValueError(f"unknown fold design {fold_design!r}, not one of {FOLD_DESIGNS}")


def epoch_folds(
    time: np.ndarray, trials: pd.DataFrame | None, fold_design: str
) -> tuple[np.ndarray | None, int, dict]:
    """Each bin's fold, the margin in bins and the folds' counts under
    ``fold_design``.

    For ``trials``, the segments are ``tetrode_holdout.task_segment_starts``'
    (each trial from its ``movement_start_time`` to the next, of class
    ``to_well``; the time before the first trial and after the last trial's
    ``end_time``, each its own class), cut into pieces of at most
    ``MAX_SEGMENT_S`` (``tetrode_holdout.trial_folds``); ``contiguous`` returns
    no folds (``heldout_terms``' default time blocks) and
    :data:`~.tetrode_holdout.MARGIN_BINS`.

    Returns
    -------
    folds : ndarray of int, shape (n_time,), or None
    margin_bins : int
    counts : dict
        ``n_segments`` (the segments holding a bin), ``n_groups`` (their
        pieces after the cap) and ``trial_bin_share`` (the share of the bins
        from the first departure to the last trial's end), None for
        ``contiguous``.
    """
    if fold_design == "contiguous":
        return (
            None,
            th.MARGIN_BINS,
            {"n_segments": None, "n_groups": None, "trial_bin_share": None},
        )
    if fold_design != "trials":
        raise ValueError(f"unknown fold design {fold_design!r}")
    trials = trials.sort_values("movement_start_time")
    departures = trials["movement_start_time"].to_numpy(dtype=float)
    last_end = float(trials["end_time"].iloc[-1])
    starts, classes = th.task_segment_starts(
        time, departures, trials["to_well"].to_numpy(), last_end
    )
    folds = th.trial_folds(time, starts, classes, nr.N_FOLDS, th.MAX_SEGMENT_S)
    segment = th.trial_segments(time, starts)
    counts = {
        "n_segments": int(np.unique(segment).size),
        "n_groups": int(th.capped_groups(time, segment, th.MAX_SEGMENT_S)[-1]) + 1,
        "trial_bin_share": th.trial_bin_share(time, departures, last_end),
    }
    return folds, th.margin_in_bins(th.TRIAL_MARGIN_S, time), counts


def fit_settings(fit: str) -> dict:
    """The header's settings of a fit mode: none for ``shared``, whose header
    is the earlier runs' (so their records still gather and resume); for
    ``folds`` its name and the EM settings of every fold's fit
    (``nonlocal_restriction.em_settings``), so neither mode resumes or
    gathers the other's records."""
    if fit == "shared":
        return {}
    if fit == "folds":
        return {
            "fit": "folds",
            "em": nr.em_settings(nr.FOLD_FIT_ESTIMATE_ENCODING_MODEL),
        }
    raise ValueError(f"unknown fit mode {fit!r}, not one of {FIT_MODES}")


def run_epoch(
    epoch_id: str,
    data_dir: Path,
    output: Path,
    fold_design: str = "trials",
    fit: str = "shared",
) -> dict:
    """Score one epoch, write its two tables and return its record."""
    if fit not in FIT_MODES:
        raise ValueError(f"unknown fit mode {fit!r}, not one of {FIT_MODES}")
    started = clock.time()
    log = logging.getLogger(epoch_id).info
    epoch = epoch_inputs(epoch_id, data_dir, trials=fold_design == "trials")
    frame = epoch.frame
    folds, margin_bins, fold_counts = epoch_folds(
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

    arguments = (
        epoch.detector,
        epoch.time,
        frame[["head_position_x", "head_position_y"]].to_numpy(dtype=float),
        epoch.loaded["spike_times"],
        epoch.loaded["marks"],
        epoch.loaded["is_training"],
        epoch.loaded["is_missing"],
        in_swr,
    )
    options = {
        "folds": folds,
        "margin_bins": margin_bins,
        "linear_position": frame["linear_position"].to_numpy(dtype=float),
        "log": log,
    }
    if fit == "shared":
        scores, record = nr.restriction_scores(*arguments, **options)
        failures = nr.check_failures(record)
    else:
        scores, record = nr.fold_fit_scores(
            *arguments,
            estimate_encoding_model=nr.FOLD_FIT_ESTIMATE_ENCODING_MODEL,
            **options,
        )
        failures = nr.fold_check_failures(record)
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
    sampling = scores["place_sampling_s"].to_numpy()[immobile]
    fold_share = np.bincount(scores["block"], minlength=nr.N_FOLDS) / len(scores)
    return json_ready(
        {
            "status": "failed_check" if failures else "ok",
            **({"error": "; ".join(failures)} if failures else {}),
            **record,
            **share,
            "fit": fit,
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
        choices=FOLD_DESIGNS,
        default="trials",
        help="whole trials balanced by destination well (default), or the "
        "earlier contiguous time blocks",
    )
    parser.add_argument(
        "--fit",
        choices=FIT_MODES,
        default="shared",
        help="the canonical decoder's transitions with held-out encodings "
        "(default), or the whole decoder refit by EM on each fold's training "
        "folds",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_dir or processed_data_dir()
    output = args.output_dir or data_dir / OUTPUT_DIRS[args.fit]
    header = {
        "produced_by": PRODUCED_BY,
        "provenance": provenance_record(ROOT_DIR),
        "settings": {
            "n_folds": nr.N_FOLDS,
            **fit_settings(args.fit),
            **fold_settings(args.fold_design),
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
        gather_manifest(output, OUTPUT_NAME, header)
        return
    max_workers, pool_options = gpu_pool(parser, args)
    run(
        epochs=select_epochs(
            args.epochs or analyzed_epoch_ids(), args.shards, args.n_shards
        ),
        task=run_epoch,
        task_args=(data_dir, output, args.fold_design, args.fit),
        output=output,
        name=OUTPUT_NAME,
        header=header,
        overwrite=args.overwrite,
        max_workers=max_workers,
        pool_options=pool_options,
    )


if __name__ == "__main__":
    main()
