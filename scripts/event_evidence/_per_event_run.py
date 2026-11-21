"""What the per-event runners share: their arguments, the canonical event table,
an epoch's decode inputs, the held-out runs' fold designs, the GPU worker pool,
and the epoch loop that writes each epoch's record and gathers the run's
manifest (``runs.epoch_records``).

Used by ``measure_event_participation.py``, ``score_event_likelihood_advantage.py``,
``score_tetrode_holdout.py``, ``score_nonlocal_restriction.py`` and
``sum_nonlocal_restriction_windows.py``.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_evidence import tetrode_holdout as th
from continuum_swr_replay.decoding.files import DecodePaths, decode_paths
from continuum_swr_replay.parameters import SAMPLING_FREQUENCY
from continuum_swr_replay.paths import (
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.batch import lost_epoch_record
from continuum_swr_replay.runs.epoch_records import (
    epochs_to_run,
    gather_manifest,
    write_record,
)
from continuum_swr_replay.runs.io import sha256_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

FOLD_DESIGNS = ("trials", "contiguous")
"""``trials``: whole trials balanced by destination well
(``tetrode_holdout.trial_folds``); ``contiguous``: contiguous time blocks."""


def event_table_path(data_dir: Path) -> Path:
    """The canonical event table under the processed-data root ``data_dir``."""
    return table_path(
        "event_table", data_dir / replay_content_dir().relative_to(processed_data_dir())
    )


def event_table_input(path: Path) -> dict[str, str]:
    """The event table as a run's input: its digest and where it was read."""
    return {"sha256": sha256_file(path), "path": str(path)}


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """The arguments every per-event runner takes."""
    parser.add_argument("--epochs", nargs="+", help="epoch ids; default every epoch")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--shards",
        type=int,
        nargs="+",
        help="run only the epochs (sorted) whose position modulo --n-shards is "
        "one of these: a host's share of a run split across hosts",
    )
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument(
        "--gather", action="store_true", help="only write the manifest from the records"
    )


def select_epochs(
    epochs: Sequence[str], shards: Sequence[int] | None, n_shards: int
) -> list[str]:
    """The epochs of ``shards`` (all when None), in sorted order."""
    return [
        epoch_id
        for i, epoch_id in enumerate(sorted(epochs))
        if shards is None or i % n_shards in shards
    ]


def use_gpu(queue: multiprocessing.Queue, mem_fraction: float) -> None:
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


def add_gpu_arguments(parser: argparse.ArgumentParser) -> None:
    """``--gpus`` and ``--workers-per-gpu``, the arguments of :func:`gpu_pool`."""
    parser.add_argument("--gpus", type=int, nargs="+", help="the cards to use")
    parser.add_argument("--workers-per-gpu", type=int, default=1)


def gpu_pool(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> tuple[int, dict[str, Any]]:
    """:func:`run`'s ``max_workers`` and ``pool_options`` for ``args.gpus``,
    ``args.workers_per_gpu`` workers to a card, each allowed that share of its
    memory (:func:`use_gpu`); a parser error without ``--gpus``."""
    if not args.gpus:
        parser.error("--gpus is required")
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    for gpu in args.gpus:
        for _ in range(args.workers_per_gpu):
            queue.put(gpu)
    return len(args.gpus) * args.workers_per_gpu, {
        "mp_context": context,
        "initializer": use_gpu,
        "initargs": (queue, round(0.9 / args.workers_per_gpu, 3)),
    }


@dataclass(frozen=True)
class EpochInputs:
    """An epoch's decode frame, its spikes and masks, and its decoder."""

    paths: DecodePaths
    frame: pd.DataFrame
    """The decode frame, read through the SWR source."""
    time: np.ndarray
    """Its time axis, s, shape (n_time,)."""
    loaded: dict
    """``load_marks_and_training_mask``'s spikes, marks and masks."""
    detector: Any


def epoch_inputs(epoch_id: str, data_dir: Path, *, trials: bool = False) -> EpochInputs:
    """Read an epoch's decode frame, spikes, masks and detector; with
    ``trials``, also its trials (``loaded["trials"]``), in the same load.

    Raises
    ------
    ValueError
        If the loader's time axis is not the decode's.
    """
    from continuum_swr_replay.analysis.event_definition.analyzed_time import (
        read_dot_with_swr_source,
    )
    from continuum_swr_replay.analysis.event_evidence.state_evidence.terms import (
        load_detector,
        load_marks_and_training_mask,
    )

    paths = decode_paths(epoch_id, data_dir)
    frame = read_dot_with_swr_source(paths.data_over_time)
    time = frame.index.to_numpy(dtype=float)
    loaded = load_marks_and_training_mask(epoch_id, trials=trials)
    if not np.array_equal(loaded["time"], time):
        raise ValueError("the loader's time axis is not the decode's")
    return EpochInputs(paths, frame, time, loaded, load_detector(epoch_id, data_dir))


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
    folds = th.trial_folds(time, starts, classes, th.N_FOLDS, th.MAX_SEGMENT_S)
    segment = th.trial_segments(time, starts)
    counts = {
        "n_segments": int(np.unique(segment).size),
        "n_groups": int(th.capped_groups(time, segment, th.MAX_SEGMENT_S)[-1]) + 1,
        "trial_bin_share": th.trial_bin_share(time, departures, last_end),
    }
    return folds, th.margin_in_bins(th.TRIAL_MARGIN_S, time), counts


def run(
    *,
    epochs: Sequence[str],
    task: Callable[..., dict],
    task_args: tuple,
    output: Path,
    name: str,
    header: Mapping[str, Any],
    overwrite: bool,
    max_workers: int,
    pool_options: Mapping[str, Any] | None = None,
) -> None:
    """Run ``task(epoch_id, *task_args)`` for each epoch without an ``ok``
    record of this run, write each record as it arrives, then gather the
    manifest; exit with an error if an epoch failed.
    """
    output.mkdir(parents=True, exist_ok=True)
    to_run = epochs_to_run(output, name, epochs, header, overwrite)
    logging.info(f"{len(to_run)} of {len(epochs)} epochs to run")
    failed = []
    with ProcessPoolExecutor(max_workers=max_workers, **(pool_options or {})) as pool:
        futures = {
            pool.submit(task, epoch_id, *task_args): epoch_id for epoch_id in to_run
        }
        for k, future in enumerate(as_completed(futures), start=1):
            epoch_id = futures[future]
            try:
                record = future.result()
            except Exception as error:  # noqa: BLE001 - recorded per epoch
                record = lost_epoch_record(error)
            write_record(output, epoch_id, name, header, record)
            if record["status"] != "ok":
                failed.append(epoch_id)
            shown = {k: v for k, v in record.items() if k != "traceback"}
            logging.info(f"[{k}/{len(to_run)}] {epoch_id}: {shown}")
    gather_manifest(output, name, header)
    if failed:
        raise SystemExit(f"{len(failed)} epochs failed: {failed[:5]}")
