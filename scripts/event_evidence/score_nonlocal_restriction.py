#!/usr/bin/env python
"""Per-bin scores of the four non-local restriction models on every analyzed epoch.

For each analyzed HPC epoch (``data_loaders.epoch_register.analyzed_epoch_ids``):
the decoders ``none``, ``swr``, ``full`` and ``swr_allowed``
(``nonlocal_restriction.ALL_MODELS``), which differ only in when the two
Non-Local states may be occupied, scored at every bin with a likelihood held
out by folds of whole trials (``analysis.event_evidence.nonlocal_restriction``).
``swr`` is non-local in every bin inside a detected SWR of the epoch's SWR source
(``swr_source.epoch_source(epoch_id).events``, the Kay inventory), whether or
not the SWR was immobile as a whole event, and local everywhere else; ``swr_allowed`` allows every state inside such a bin and
only the local states elsewhere. That is
deliberately not the frame's ``SWR_event`` labels, which keep only the SWRs the
analyzed-time rule keeps.

The likelihood is held out by folds of whole trials: a segment runs from one departure from a well (the trial's
``movement_start_time``, the final poke-out at its origin) to the next, so the
run to a well and the whole stay there are one segment, and the segments are
dealt into ``N_FOLDS`` folds balanced by destination well
(``regression.folds.trial_folds``), with a margin of ``TRIAL_MARGIN_S``. The
time before the first trial and after the last trial's end are segments of
their own two classes, and a segment longer than ``MAX_SEGMENT_S`` is cut into
equal pieces, each a group of its own, so that long stays are spread over the
folds. The folds are whole trials because the animal stays in one patch for
long stretches: a place visited in only one stretch of time would have no data
in that stretch's held-out encoding. The design and its margin are in the
header's settings, so a run of another design does not gather or resume with
this one. The trials are ``load_data``'s (parsed from the DIO events), not the
session cache's, and each epoch's record keeps their digest
(``_per_event_run.trials_input``).

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
(``"fit": "folds"``) and the EM settings, so a resume or a gather cannot mix
runs of another fit. A fold fit costs about what the canonical decode
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
  the eight float32 score columns, ``in_swr``, ``block`` (the bin's fold),
  ``place_sampling_s`` (the seconds its fold's encoding was fit on at the
  animal's place bin) and ``category``, so a later per-event or
  per-ripple-power analysis needs no rerun;
- ``{epoch_id}_nonlocal_restriction.json``: the epoch's record (status,
  ``fold_fit_scores``' record, bin counts, the share of the non-local gain
  outside SWRs over the immobile bins (``share_outside_swr``, both families),
  the folds (``n_segments``, ``n_groups``, the smallest and largest fold's share of the
  bins, the margin in bins), the share of the immobile bins whose fold's
  encoding had 0 s and at most 10 s of training data at the animal's place,
  the per-epoch input digests (the decode frame, the detector, the SWR source's
  files and the trials the folds were made from), settings, provenance);

and ``manifest.json`` gathered from the records (``runs.epoch_records``).

An epoch whose models fail a check (``nonlocal_restriction.fold_check_failures``:
a fold whose EM did not converge, or a fold's pass that fails
``check_failures``: a filtered posterior that is not the library's update of
its predictive posterior, a filtered or predictive posterior whose mass is
farther from 1 than its tolerance, or continuous transitions whose rows do not
sum to 1) gets status ``failed_check`` and a message naming the failures; its
tables are still written, for diagnosis. The difference between the library's
marginal and the summed filter scores is recorded per fold
(``marginal_difference_<model>``), not checked.
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

**Scored from a stored fit** (``--from-stored-fit <run folder>``): the models
of ``--models`` (by default all four) scored with each fold's decoder rebuilt from the stored run's record
(``nonlocal_restriction.stored_fold_fit``: the encoding fit as the fold fit
fits it, the stored discrete transitions and initial distribution, no EM).
Per epoch the stored record must be ``ok`` and of the same inputs (the
SHA-256 of the decode frame, the detector and the SWR source's files, and the
margin in bins) and folds: by the trials' digest where the stored record has
one, otherwise by the folds made now against the
stored per-bin ``block`` at every bin and the folds' ``n_segments`` and
``n_groups`` (``stored_folds_check`` in the record: ``"digest"`` or
``"block_column"``); the tables are those above, with the new
model's columns, and the record's folds also hold
``stored_score_difference``: the largest absolute per-bin difference of each
score scored again (of ``none``, ``swr``, ``full``) from the stored run's, over
the fold's bins, and ``stored_score_difference_max`` its largest; each fold's
fit facts are those of the stored record, so the new run's records hold them
too. A model of the three not in ``--models`` is copied from the stored per-bin
table (``copied_from_stored`` in the record). The header records the models,
the fit's source (``fit_source``: ``"stored"``) and the stored run as an input
(``stored_fit``: its folder and manifest digest, kept as the record of where
the fit came from). It needs ``--output-dir``, which is neither the restriction
run's folder nor the stored run's. The stored restriction run was made this
way, from a run of the three models.

Usage::

    uv run --frozen python scripts/event_evidence/score_nonlocal_restriction.py \\
        --epochs wilbur20210329_02_r1_HPC --gpus 1
    uv run --frozen python scripts/event_evidence/score_nonlocal_restriction.py \\
        --gpus 1 2 4 5 6 7 8
    uv run --frozen python scripts/event_evidence/score_nonlocal_restriction.py \\
        --gather
    uv run --frozen python scripts/event_evidence/score_nonlocal_restriction.py \\
        --gpus 1 2 4 5 6 7 8 --shards 0 --n-shards 2
    uv run --frozen python scripts/event_evidence/score_nonlocal_restriction.py \\
        --from-stored-fit Processed-Data/event_nonlocal_restriction \\
        --output-dir <new folder> --epochs wilbur20210329_02_r1_HPC --gpus 1
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
from continuum_swr_replay.analysis.regression.folds import N_FOLDS
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
from continuum_swr_replay.runs.epoch_records import gather_manifest, read_record
from continuum_swr_replay.runs.io import json_ready, output_path
from continuum_swr_replay.runs.provenance import produced_by, provenance_record
from continuum_swr_replay.tables.io import write_table

# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.event_evidence._per_event_run import (  # noqa: E402
    add_gpu_arguments,
    add_run_arguments,
    epoch_inputs,
    file_input,
    gpu_pool,
    run,
    run_input,
    select_epochs,
    trials_input,
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


STORED_BIN_COLUMNS = ["time", "block", "in_swr", *nr.SCORE_COLUMNS]
"""The stored per-bin table's columns a run from a stored fit reads."""


def current_epoch_inputs(epoch_id: str, paths, trials: pd.DataFrame) -> dict:
    """The epoch's inputs as a record keeps them (``epoch_inputs``)."""
    return {
        "data_over_time": file_input(paths.data_over_time),
        "detector": file_input(paths.detector),
        "trials": trials_input(trials),
        **{
            f"swr_{name}": entry
            for name, entry in swr_source.swr_source_files(epoch_id).items()
        },
    }


def stored_record(
    stored_dir: Path,
    epoch_id: str,
    inputs: dict,
    time: np.ndarray,
    folds: np.ndarray,
    fold_counts: dict,
    margin_bins: int,
) -> tuple[dict, str, pd.DataFrame]:
    """The stored run's record of the epoch, checked to be ``ok`` and of the
    same inputs, how its folds were checked, and its per-bin table.

    Each input's SHA-256 must be the stored record's, and its margin in bins.
    The folds are checked by the trials' digest when the stored record holds
    one (``"digest"``). A record without one is checked on the folds
    themselves (``"block_column"``): the folds made now must be the stored
    per-bin table's ``block`` at every bin of the same time axis, and the
    folds' ``n_segments`` and ``n_groups`` the stored record's, as
    ``scripts/checks/check_stored_folds_reproduce.py`` compares them.

    Parameters
    ----------
    stored_dir : Path
    epoch_id : str
    inputs : dict
        The epoch's inputs now (:func:`current_epoch_inputs`).
    time : ndarray, shape (n_time,)
    folds : ndarray of int, shape (n_time,)
        The folds made now (``tetrode_holdout.epoch_folds``).
    fold_counts : dict
        ``epoch_folds``' counts (``n_segments``, ``n_groups``, ...).
    margin_bins : int

    Returns
    -------
    record : dict
    check : str
        ``"digest"`` or ``"block_column"``.
    bins : pd.DataFrame
        The stored per-bin table's :data:`STORED_BIN_COLUMNS`, read once.

    Raises
    ------
    ValueError
        Naming what differs.
    """
    record = read_record(stored_dir, epoch_id, NONLOCAL_RESTRICTION_NAME)
    if record is None or record.get("status") != "ok":
        raise ValueError(f"the stored run {stored_dir} has no ok record of {epoch_id}")
    stored = record.get("epoch_inputs", {})
    check = "digest" if "trials" in stored else "block_column"
    compared = inputs.keys() | stored.keys()
    if check == "block_column":
        compared -= {"trials"}
    differ = sorted(
        key
        for key in compared
        if {k: v for k, v in inputs.get(key, {}).items() if k != "path"}
        != {k: v for k, v in stored.get(key, {}).items() if k != "path"}
    )
    for key, value in (
        ("margin_bins", margin_bins),
        *(
            (key, fold_counts[key])
            for key in ("n_segments", "n_groups")
            if check == "block_column"
        ),
    ):
        if record.get(key) != value:
            differ.append(key)
    if differ:
        raise ValueError(
            f"the stored record of {epoch_id} was made from other inputs: {differ}"
        )
    bins = pd.read_parquet(
        output_path(epoch_id, stored_dir, NONLOCAL_RESTRICTION_BINS_NAME),
        columns=STORED_BIN_COLUMNS,
    )
    if check == "block_column":
        if len(bins) != len(time) or not np.array_equal(
            bins["time"].to_numpy(dtype=float), time
        ):
            raise ValueError(
                f"the stored per-bin table of {epoch_id} is not on the decode's "
                "time axis"
            )
        moved = np.flatnonzero(bins["block"].to_numpy() != np.asarray(folds))
        if moved.size:
            raise ValueError(
                f"the folds made now differ from the stored run's in "
                f"{moved.size} bins of {epoch_id} (first at bin {int(moved[0])}): "
                "the trials are not the stored run's"
            )
    return record, check, bins


def run_epoch(
    epoch_id: str,
    data_dir: Path,
    output: Path,
    stored_dir: Path | None = None,
    models: tuple[str, ...] = nr.ALL_MODELS,
) -> dict:
    """Score one epoch, write its two tables and return its record.

    With ``stored_dir``, each fold's fit is rebuilt from that run's record of
    the epoch (no EM) and ``models`` are scored; the record compares the
    scores with the stored run's."""
    started = clock.time()
    log = logging.getLogger(epoch_id).info
    epoch = epoch_inputs(epoch_id, data_dir, trials=True)
    frame = epoch.frame
    folds, margin_bins, fold_counts = th.epoch_folds(epoch.time, epoch.loaded["trials"])
    inputs = current_epoch_inputs(epoch_id, epoch.paths, epoch.loaded["trials"])
    stored, folds_check, stored_bins = (
        (None, None, None)
        if stored_dir is None
        else stored_record(
            stored_dir, epoch_id, inputs, epoch.time, folds, fold_counts, margin_bins
        )
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
    # what the epoch needs of the decode frame, copied, so that the frame is
    # not held through the scoring
    head_position = frame[["head_position_x", "head_position_y"]].to_numpy(
        dtype=float, copy=True
    )
    linear_position = frame["linear_position"].to_numpy(dtype=float, copy=True)
    without_speed = ~np.isfinite(frame["head_speed"].to_numpy())
    n_bins = len(frame)
    time, loaded, detector = epoch.time, epoch.loaded, epoch.detector
    del epoch, frame

    from_stored = (
        {}
        if stored is None
        else {"fit": nr.stored_fold_fit(stored["folds"]), "models": models}
    )
    scores, record = nr.fold_fit_scores(
        detector,
        time,
        head_position,
        loaded["spike_times"],
        loaded["marks"],
        loaded["is_training"],
        loaded["is_missing"],
        in_swr,
        folds=folds,
        margin_bins=margin_bins,
        linear_position=linear_position,
        log=log,
        **from_stored,
    )
    failures = nr.fold_check_failures(record, models)
    copied: list[str] = []
    if stored is not None:
        for fold, difference in zip(
            record["folds"],
            nr.stored_score_differences(scores, stored_bins, models),
            strict=True,
        ):
            compared = {
                k: v for k, v in difference.items() if k not in ("fold", "n_bins")
            }
            fold["stored_score_difference"] = compared
            fold["stored_score_difference_max"] = max(compared.values(), default=0.0)
        # a model of the three not scored again is the stored run's
        copied = [c for c in nr.SCORE_COLUMNS if c not in scores.columns]
        for column in copied:
            scores[column] = stored_bins[column].to_numpy()
        order = [*nr.score_columns(nr.table_models(scores.columns))]
        rest = [c for c in scores.columns if c not in order and c != "time"]
        scores = scores[["time", *order, *rest]]
        del stored_bins
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
    fold_share = np.bincount(scores["block"], minlength=N_FOLDS) / len(scores)
    return json_ready(
        {
            "status": "failed_check" if failures else "ok",
            **({"error": "; ".join(failures)} if failures else {}),
            **record,
            **share,
            "fit": nr.FIT,
            **(
                {}
                if stored is None
                else {
                    "models": list(models),
                    "stored_folds_check": folds_check,
                    "copied_from_stored": copied,
                    "stored_score_difference_max": max(
                        f["stored_score_difference_max"] for f in record["folds"]
                    ),
                }
            ),
            "fold_design": th.TRIAL_DESIGN,
            "margin_bins": margin_bins,
            **fold_counts,
            "fold_share_min": float(fold_share.min()),
            "fold_share_max": float(fold_share.max()),
            "immobile_share_place_sampling_0s": float(np.mean(sampling == 0)),
            "immobile_share_place_sampling_le_10s": float(
                np.mean(sampling <= SPARSE_SAMPLING_S)
            ),
            "n_immobile_without_place": int(np.isnan(sampling).sum()),
            "n_bins": n_bins,
            "n_bins_by_category": {
                c: int(n) for c, n in pd.Series(categories).value_counts().items()
            },
            "n_analyzed_without_speed": int((analyzed & without_speed).sum()),
            "n_swr_source_events": len(source_events),
            "n_in_swr_label": int(in_label.sum()),
            "n_immobile_in_swr_label": int((immobile & in_label).sum()),
            "bins_table_bytes": bins_path.stat().st_size,
            "epoch_inputs": inputs,
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
        "--from-stored-fit",
        type=Path,
        default=None,
        help="a restriction run folder: rebuild each fold's fit from its records "
        "(no EM) and score --models; needs --output-dir",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=nr.ALL_MODELS,
        default=None,
        help="with --from-stored-fit, the models to score (full among them); "
        "default all four",
    )
    args = parser.parse_args(argv)

    data_dir = args.data_dir or processed_data_dir()
    stored_dir = args.from_stored_fit
    if stored_dir is None:
        if args.models is not None:
            parser.error("--models needs --from-stored-fit")
        models = nr.ALL_MODELS
        output = args.output_dir or data_dir / EVENT_NONLOCAL_RESTRICTION_DIR
    else:
        models = tuple(m for m in nr.ALL_MODELS if m in (args.models or nr.ALL_MODELS))
        if "full" not in models:
            parser.error("--models must include full")
        if args.output_dir is None:
            parser.error("--from-stored-fit needs --output-dir")
        output = args.output_dir
        refused = {
            (data_dir / EVENT_NONLOCAL_RESTRICTION_DIR).resolve(),
            stored_dir.resolve(),
        }
        if output.resolve() in refused:
            parser.error(
                f"--output-dir {output} is the restriction run or the stored "
                "one; a run from a stored fit writes elsewhere"
            )
    header = {
        "produced_by": PRODUCED_BY,
        "provenance": provenance_record(ROOT_DIR),
        "settings": {
            "n_folds": N_FOLDS,
            **fit_settings(),
            **th.fold_settings(),
            "sparse_sampling_s": SPARSE_SAMPLING_S,
            "immobility_threshold_cm_per_s": DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
            "swr_source": swr_source.swr_source_name(),
            "models": list(models),
            **({} if stored_dir is None else {"fit_source": "stored"}),
            "filter_identity_tolerance": nr.FILTER_IDENTITY_TOLERANCE,
            "causal_mass_tolerance": nr.CAUSAL_MASS_TOLERANCE,
            "predicted_mass_tolerance": nr.PREDICTED_MASS_TOLERANCE,
            "continuous_block_tolerance": nr.CONTINUOUS_BLOCK_TOLERANCE,
        },
        # every input is the epoch's own (its decode, detector and SWR files):
        # each record holds their digests under "epoch_inputs"; a run from a
        # stored fit also names the stored run (its folder and manifest digest)
        "inputs": {} if stored_dir is None else {"stored_fit": run_input(stored_dir)},
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
        task_args=(data_dir, output)
        if stored_dir is None
        else (data_dir, output, stored_dir, models),
        output=output,
        name=NONLOCAL_RESTRICTION_NAME,
        header=header,
        overwrite=args.overwrite,
        max_workers=max_workers,
        pool_options=pool_options,
    )


if __name__ == "__main__":
    main()
