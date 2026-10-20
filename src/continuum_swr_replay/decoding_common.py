"""Shared utilities for decoding scripts.

Contains the common infrastructure used by both ``run_decoding_bandit.py``
(NonLocal 4-state detector) and ``run_decoding_contfrag.py`` (ContFrag
2-state classifier), among it the HPC decode's masks: ``is_missing`` (the
artifact run's mask) and ``is_training`` (that mask and every event of the Kay
inventory left out), both from the paper's SWR source
(``analysis/hippocampus/swr_source.py``). An HPC decode refuses to start unless
the source names both runs (``require_hpc_decode_sources``).
"""

import hashlib
import importlib.metadata
import json
import logging
import logging.config
import os
import re
import resource
import subprocess
import sys
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dask
import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster
from numpy.typing import NDArray

try:
    from dask_cuda import LocalCUDACluster

    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

from continuum_swr_replay.analysis.hippocampus import swr_source
from continuum_swr_replay.analysis.hippocampus.artifacts.masks import PADDING_S
from continuum_swr_replay.data_loaders.epoch_register import epochs_to_run
from continuum_swr_replay.decode_files import decode_paths
from continuum_swr_replay.intervals import interval_mask
from continuum_swr_replay.parameters import (
    BLOCK_SIZE,
    LOCAL_POSITION_STD,
    POSITION_STD,
    PROCESSED_DATA_DIR,
    WAVEFORM_STD,
    TrainingType,
)
from continuum_swr_replay.table_io import sha256_file, write_table
from continuum_swr_replay.types import ClusterlessParams, SortedSpikesParams

_logger = logging.getLogger(__name__)
_warned_cache_name_missing = False
"""Whether ``release_nwb_files`` has said Spyglass's cache name is gone."""


@dataclass(frozen=True)
class DecodingConfig:
    """Encoding-likelihood configuration shared by every detector family.

    ``position_std``, ``waveform_std``, and ``block_size`` configure the
    encoding-model likelihood (the ``*_algorithm_params`` dict) and apply
    to both the NonLocal 4-state detector and the ContFrag 2-state
    classifier. State-specific detector arguments live on subclasses —
    see :class:`NonLocalDecodingConfig`.
    """

    position_std: float = POSITION_STD
    block_size: int = BLOCK_SIZE
    waveform_std: float = WAVEFORM_STD

    def to_clusterless_params(self) -> ClusterlessParams:
        """Get parameters for clusterless decoding (HPC)."""
        return {
            "position_std": self.position_std,
            "waveform_std": self.waveform_std,
            "block_size": self.block_size,
        }

    def to_sorted_spikes_params(self) -> SortedSpikesParams:
        """Get parameters for sorted spikes decoding (mPFC/OFC)."""
        return {
            "position_std": self.position_std,
            "block_size": self.block_size,
        }


@dataclass(frozen=True)
class NonLocalDecodingConfig(DecodingConfig):
    """Decoding config for the NonLocal 4-state detector.

    Adds ``local_position_std``, a detector-level argument that controls
    the Local state's spatial representation. It is passed directly to the
    detector constructor (not through ``to_*_params``). The ContFrag
    2-state classifier has no Local state, so it uses the base
    :class:`DecodingConfig` instead of this subclass.
    """

    local_position_std: float = LOCAL_POSITION_STD


def require_hpc_decode_sources() -> None:
    """Refuse an HPC decode unless the SWR source names both of its runs, and
    its inventory was masked with its artifact run.

    The decode's ``is_missing``, encoding exclusion, MUA detection and
    ``SWR_event`` read ``swr_source.SWR_INVENTORY_RUN_ID`` (the Kay inventory)
    and ``swr_source.ARTIFACT_INTERVALS_RUN_ID`` (the artifact run). Either
    left unset would decode without the masks, so it is refused (fail closed).
    The inventory's manifest names the artifact run the SWR runner masked its
    events with (``settings.artifacts.run_id``; absent from a run made before
    it masked artifacts, such as R3's). An inventory masked with no artifact
    run or another one is refused too: its events, which encoding leaves out,
    would not be the masked inventory's.

    Raises
    ------
    ValueError
        Naming each run id that is unset, or the inventory, the artifact run it
        was masked with and the configured one when those two differ; or when
        the inventory has no manifest.
    """
    unset = [
        f"swr_source.{name}"
        for name in ("SWR_INVENTORY_RUN_ID", "ARTIFACT_INTERVALS_RUN_ID")
        if getattr(swr_source, name) is None
    ]
    if unset:
        raise ValueError(
            f"An HPC decode needs {' and '.join(unset)} set, so that its artifact "
            "mask, encoding exclusion and SWR labels come from named runs; "
            "refusing to decode without them."
        )
    manifest_path = swr_source.swr_inventory_dir() / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"no SWR inventory manifest at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    masked_with = manifest.get("settings", {}).get("artifacts", {}).get("run_id")
    if masked_with != swr_source.ARTIFACT_INTERVALS_RUN_ID:
        recorded = (
            "no artifact run"
            if masked_with is None
            else f"artifact run {masked_with!r}"
        )
        raise ValueError(
            f"The SWR inventory {swr_source.SWR_INVENTORY_RUN_ID!r} was masked with "
            f"{recorded}, but the configured artifact run is "
            f"{swr_source.ARTIFACT_INTERVALS_RUN_ID!r}: rerun the SWR runner with the "
            "configured artifact run and set swr_source.SWR_INVENTORY_RUN_ID to it; "
            "refusing to decode."
        )


def require_saved_decode_masks(attrs: Mapping[str, Any], what: str) -> None:
    """Refuse to resume an HPC decode whose saved results were made under other
    masks.

    A resume reuses ``results.nc`` and rewrites ``data_over_time`` with the
    configured source's SWR labels, so the results must have been decoded with
    the configured Kay inventory and artifact run: their ``decode_masks``
    record (``hpc_decode_masks``, stamped by a fresh save) must name both.

    Parameters
    ----------
    attrs : Mapping
        The saved results' ``attrs``.
    what : str
        What they are, for the message (the file, say).

    Raises
    ------
    ValueError
        If the record is missing (a decode made before the masks) or names
        another inventory or artifact run.
    """
    stamp = attrs.get("decode_masks")
    if stamp is None:
        raise ValueError(
            f"{what} has no decode_masks: it was decoded without the artifact "
            "masks; refusing to resume. Decode it afresh (--overwrite)."
        )
    record = json.loads(stamp)
    configured = {
        "swr_inventory_run_id": swr_source.SWR_INVENTORY_RUN_ID,
        "artifact_run_id": swr_source.ARTIFACT_INTERVALS_RUN_ID,
    }
    saved = {name: record.get(name) for name in configured}
    if saved != configured:
        raise ValueError(
            f"{what} was decoded with {saved}, not the configured {configured}; "
            "refusing to resume. Decode it afresh (--overwrite)."
        )


def hpc_artifact_intervals(nwb_file_name: str, epoch_name: str) -> pd.DataFrame:
    """The epoch's artifact mask, from the configured run.

    Parameters
    ----------
    nwb_file_name, epoch_name : str
        Such as ``"chimi20200225_.nwb"`` and ``"02_r1"``.

    Returns
    -------
    pd.DataFrame
        ``start_time`` and ``end_time`` in seconds, closed, disjoint and in
        time order: the run's cores padded by ``PADDING_S``, the one mask
        every stage uses (``swr_source.artifact_intervals``).

    Raises
    ------
    ValueError
        If a run id is unset (``require_hpc_decode_sources``).
    SwrSourceError
        If the artifact run cannot serve the epoch.
    """
    require_hpc_decode_sources()
    return swr_source.artifact_intervals(
        swr_source.swr_epoch_id(nwb_file_name, epoch_name)
    )


def hpc_ripple_power_inputs(nwb_file_name: str, epoch_name: str) -> dict[str, Any]:
    """The SWR source's ripple-power inputs for ``load_data``, from the
    configured runs (``swr_source.ripple_power_inputs``).

    With them the decode's ``SWR_zscore`` is the Kay detector's consensus over
    the tetrodes rule R-T kept for the configured inventory, z-scored with the
    artifact mask left out and NaN inside it.

    Raises
    ------
    ValueError
        If a run id is unset (``require_hpc_decode_sources``).
    SwrSourceError
        If the inventory or the artifact run cannot serve the epoch.
    """
    require_hpc_decode_sources()
    return swr_source.ripple_power_inputs(
        swr_source.swr_epoch_id(nwb_file_name, epoch_name)
    )


@dataclass(frozen=True)
class HpcDecodeMasks:
    """An HPC decode's masks on its time grid (the position grid)."""

    is_missing: NDArray[np.bool_]
    """Shape (n_time,): bins in an artifact interval, whose
    log-likelihood the decoder sets to 0."""
    is_training: NDArray[np.bool_]
    """Shape (n_time,): bins the encoding model is fitted on."""
    record: dict[str, Any]
    """What they were built from and what they removed, in seconds (JSON)."""


def hpc_decode_masks(
    nwb_file_name: str,
    epoch_name: str,
    position_info: pd.DataFrame,
    artifact_mask: pd.DataFrame,
    training_type: str | TrainingType,
) -> HpcDecodeMasks:
    """Build an HPC decode's ``is_missing`` and ``is_training``.

    - ``is_missing``: the bins (closed) inside an artifact interval.
    - ``is_training``: for ``no_ripple``, the bins outside every event of the
      configured Kay inventory and outside ``is_missing``; for ``all``, the
      bins outside ``is_missing``. Every inventory event counts, at any speed,
      as the stored-ripple rule it replaces excluded every stored ripple: the
      inventory's events passed the detectors' own endpoint speed rule (speed
      at the first and last sample at most 4 cm/s), as the stored ripples did,
      and the SWR runner has already dropped those overlapping the artifact
      mask.

    Parameters
    ----------
    nwb_file_name, epoch_name : str
    position_info : pd.DataFrame
        The decode's position, indexed by time in seconds: the decode grid.
    artifact_mask : pd.DataFrame
        ``hpc_artifact_intervals``' table for the epoch.
    training_type : str or TrainingType
        ``"no_ripple"`` or ``"all"``.

    Returns
    -------
    HpcDecodeMasks
        The masks, and a record of the source's name, both run ids, the
        bins and seconds set missing, the inventory's event count, and the
        bins and seconds removed from encoding (missing ones included).

    Raises
    ------
    ValueError
        If a run id is unset, or `training_type` is not one of the two.
    SwrSourceError
        If the inventory cannot serve the epoch.
    """
    require_hpc_decode_sources()
    training_type = TrainingType(training_type)
    epoch_id = swr_source.swr_epoch_id(nwb_file_name, epoch_name)
    time = position_info.index.to_numpy(dtype=np.float64)
    is_missing = interval_mask(time, artifact_mask)
    if training_type == TrainingType.NO_RIPPLE:
        events = swr_source.read_swr_inventory(epoch_id)
        is_swr = interval_mask(time, events)
        n_events: int | None = len(events)
    else:
        is_swr = np.zeros(time.shape, dtype=bool)
        n_events = None
    is_training = ~is_swr & ~is_missing
    bin_s = float(np.median(np.diff(time)))
    n_missing = int(np.count_nonzero(is_missing))
    n_removed = int(np.count_nonzero(~is_training))
    record = {
        "swr_source": swr_source.swr_source_name(),
        "swr_inventory_run_id": swr_source.SWR_INVENTORY_RUN_ID,
        "artifact_run_id": swr_source.ARTIFACT_INTERVALS_RUN_ID,
        "padding_s": dict(PADDING_S),
        "training_type": training_type.value,
        "n_bins": int(time.size),
        "bin_s": bin_s,
        "n_missing_bins": n_missing,
        "missing_s": n_missing * bin_s,
        "n_swr_events": n_events,
        "n_encoding_removed_bins": n_removed,
        "encoding_removed_s": n_removed * bin_s,
        "encoding_removed_swr_s": int(np.count_nonzero(is_swr)) * bin_s,
    }
    return HpcDecodeMasks(is_missing=is_missing, is_training=is_training, record=record)


class SkipChecksumFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Skipped checksum" not in record.getMessage()


def setup_logger(
    name_logfile: str,
    path_logfile: str | Path,
    *,
    level: int = logging.INFO,
    include_warnings: bool = True,
) -> logging.Logger:
    """Set up a logger that logs to console and a file."""
    log_path = Path(path_logfile)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level_name = logging.getLevelName(level)

    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "skip_checksum": {
                "()": SkipChecksumFilter,
            },
        },
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)-8s - %(message)s",
                "datefmt": "%d-%b-%y %H:%M:%S",
            },
        },
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "filename": str(log_path),
                "mode": "w",
                "encoding": "utf-8",
                "formatter": "standard",
                "filters": ["skip_checksum"],
            },
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "filters": ["skip_checksum"],
            },
        },
        "loggers": {
            name_logfile: {
                "handlers": ["file", "console"],
                "level": level_name,
                "propagate": False,
            },
        },
    }

    if include_warnings:
        logging.captureWarnings(True)
        config["loggers"]["py.warnings"] = {
            "handlers": ["console"],
            "level": level_name,
            "propagate": False,
        }

    logging.config.dictConfig(config)
    return logging.getLogger(name_logfile)


def results_exist(
    save_folder: Path,
    epoch_identifier: str,
    overwrite: bool,
) -> bool:
    """Check if results, detector, and environment files exist."""
    if overwrite:
        return False

    paths = decode_paths(epoch_identifier, save_folder)
    required_files = [paths.results, paths.detector, paths.environment]
    return all(f.exists() for f in required_files)


def write_atomically(path: Path, write: Callable[[Path], object]) -> None:
    """Write `path` through a temporary file beside it, so it is never half-written.

    ``results_exist`` trusts any file that exists, so a crash mid-write must not
    leave a truncated file under the final name.

    Parameters
    ----------
    path : Path
        The final file.
    write : Callable[[Path], object]
        Writes the whole file to the path it is given (for example
        ``detector.save_model``).

    Notes
    -----
    `write` gets ``.{name}.{pid}.tmp`` in the same folder. Only after it returns is
    that file moved onto `path` with ``os.replace``, which is atomic within a
    filesystem. If `write` raises, the temporary file is removed and `path` is left
    as it was. The temporary name does not end in the final suffix, so a
    ``*_environment.pkl``-style glob never picks it up, even after a hard kill.
    """
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, content: object) -> None:
    """Write strict JSON (no NaN or infinity) through a temporary file."""
    text = json.dumps(content, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_atomically(path, lambda temporary: temporary.write_text(text))


def replace_outputs(
    paths: Mapping[str, Path], outputs: Mapping[str, pd.DataFrame | Mapping]
) -> dict[str, dict[str, str]]:
    """Write an epoch's outputs and remove its others; on a failure, remove all.

    The folder then matches the manifest: an output not made this time keeps
    no file from an earlier run, and a write that fails (or is interrupted)
    removes every one of `paths` before re-raising, so no part set is left.

    Parameters
    ----------
    paths : Mapping[str, Path]
        Every file the epoch can write, by output name.
    outputs : Mapping[str, pd.DataFrame | Mapping]
        By name in `paths`: a table, written as parquet with its ``attrs``
        (``table_io.write_table``), or any other content, written as strict
        JSON (``write_json``).

    Returns
    -------
    dict[str, dict[str, str]]
        Each written file's ``path`` and ``sha256``, by name.

    Raises
    ------
    ValueError
        If `outputs` names an output not in `paths` (a misspelt name would
        otherwise drop that output without a word); nothing is written or
        removed.
    """
    unknown = sorted(set(outputs) - set(paths))
    if unknown:
        raise ValueError(
            f"replace_outputs: outputs {unknown} are not among the epoch's files "
            f"{sorted(paths)}; name each output as its output_paths entry does"
        )
    written = {}
    try:
        for name, path in paths.items():
            if name not in outputs:
                path.unlink(missing_ok=True)
                continue
            content = outputs[name]
            if isinstance(content, pd.DataFrame):
                write_table(content, path)
            else:
                write_json(path, content)
            written[name] = {"path": str(path), "sha256": sha256_file(path)}
    except BaseException:
        for path in paths.values():
            path.unlink(missing_ok=True)
        raise
    return written


RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9._-]+")
"""A run id names one folder under its output root: no path separator."""


def parse_run_id(value: str) -> str:
    """Return `value` if it names one run folder: an argparse ``type``.

    Parameters
    ----------
    value : str
        The option's value, such as ``"a1b2c3d4e5f6"``.

    Returns
    -------
    str
        `value`, unchanged.

    Raises
    ------
    ArgumentTypeError
        Unless `value` is letters, digits, ``.``, ``_`` and ``-`` only, and
        not ``.`` or ``..``; argparse reports it as a usage error (exit 2).
    """
    if not RUN_ID_PATTERN.fullmatch(value) or value in {".", ".."}:
        raise ArgumentTypeError(
            f"{value!r} must name one folder: letters, digits, '.', '_' and '-', "
            "and not '.' or '..'."
        )
    return value


def provenance_record(repo: Path) -> dict[str, Any]:
    """Record the code and environment a decode runs from.

    Each decoder logs this once per epoch, after START, as ``PROVENANCE `` plus
    ``json.dumps(record, sort_keys=True)``, and a fresh save stores the same JSON
    in ``results.nc`` as ``attrs["provenance"]``, where it outlives a resume.

    Parameters
    ----------
    repo : Path
        The git work tree the decode runs from. ``uv.lock`` is read from its root.

    Returns
    -------
    dict
        - ``commit``: ``git rev-parse HEAD``.
        - ``dirty``: True when ``git status --porcelain --untracked-files=no``
          lists anything. Untracked files are not counted, because the work
          tree holds untracked files from unrelated work. ``status`` runs with
          ``--no-optional-locks``: by default it refreshes the index and writes
          it back under a lock, and a decode must not write to a live checkout
          that other work commits in.
        - ``uv_lock_sha256``: the sha256 of ``repo / "uv.lock"``.
        - ``ripple_detection_commit``: the git commit ``ripple_detection`` was
          installed from, read from its ``direct_url.json``, or None when that
          names no commit. Its version string comes from the last tag, so it
          does not identify the commit.
        - ``non_local_detector_version``: the installed version.
        - ``cuda_plugin_version``: the installed ``jax-cuda12-plugin`` version,
          or None when it is not installed.
    """

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()

    direct_url = importlib.metadata.distribution("ripple-detection").read_text(
        "direct_url.json"
    )
    vcs_info = json.loads(direct_url).get("vcs_info", {}) if direct_url else {}
    try:
        cuda_plugin_version = importlib.metadata.version("jax-cuda12-plugin")
    except importlib.metadata.PackageNotFoundError:
        cuda_plugin_version = None

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(
            git("--no-optional-locks", "status", "--porcelain", "--untracked-files=no")
        ),
        "uv_lock_sha256": hashlib.sha256((repo / "uv.lock").read_bytes()).hexdigest(),
        "ripple_detection_commit": vcs_info.get("commit_id"),
        "non_local_detector_version": importlib.metadata.version("non-local-detector"),
        "cuda_plugin_version": cuda_plugin_version,
    }


def release_nwb_files() -> int:
    """Close every NWB file Spyglass holds open in this process; return how many.

    Spyglass caches each NWB file it opens, in ``spyglass.utils.nwb_helper_fn``'s
    module-level ``__open_nwb_files``, and never closes one. A long-lived worker
    therefore keeps one file descriptor per distinct file it has read (28 after
    one epoch's full ``load_data``), until it runs out (``[Errno 24] Too many
    open files``; Spyglass issue #1442). This closes them all with Spyglass's
    ``close_nwb_files``.

    Call it only where nothing else in this process can still be reading one of
    those files: data read lazily from a closed file cannot be read. The batch
    runners call it when an epoch ends, on workers that run one epoch at a time.

    It is cleanup, called from a ``finally``, so it never raises: an exception
    there would replace the epoch's own result, and in ``run_batch`` abort the
    whole batch. A failure is logged as a WARNING instead.

    Returns
    -------
    int
        How many files the cache held; all are now closed. -1 when the release
        failed (WARNING logged with the error), or when the count could not be
        read because Spyglass no longer names its cache ``__open_nwb_files``
        (WARNING logged once per process; the files are still closed).
    """
    global _warned_cache_name_missing
    try:
        # Imported here so that importing this module does not need Spyglass.
        from spyglass.utils import nwb_helper_fn

        open_files = vars(nwb_helper_fn).get("__open_nwb_files")
        if open_files is None:
            n_open = -1
            if not _warned_cache_name_missing:
                _warned_cache_name_missing = True
                _logger.warning(
                    "spyglass.utils.nwb_helper_fn has no __open_nwb_files: cannot "
                    "count the NWB files it holds open; closing them anyway"
                )
        else:
            n_open = len(open_files)
        nwb_helper_fn.close_nwb_files()
    except Exception as error:  # noqa: BLE001 - cleanup must not replace the result
        _logger.warning(
            "could not close the NWB files Spyglass held open: %r", error, exc_info=True
        )
        return -1
    return n_open


def raise_open_file_limit(target: int = 65536) -> tuple[int, int]:
    """Raise this process's soft limit on open files to `target`, at most the hard limit.

    A tmux session hands its processes a soft limit of 1024 open files, which a
    batch's long-lived workers can exhaust. Processes started afterwards,
    including dask's worker processes, inherit the raised limit, so call this
    before any cluster starts. A soft limit already at or above the new value is
    left as it is: this never lowers it.

    Parameters
    ----------
    target : int, optional
        The soft limit wanted. It is capped at the hard limit, which an
        unprivileged process cannot raise, unless that is unlimited.

    Returns
    -------
    old_soft : int
    new_soft : int
        The soft limit before and after; equal when it was not raised.
        ``resource.RLIM_INFINITY`` (-1 on Linux) stands for unlimited.
    """
    old_soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    # RLIM_INFINITY is -1 here, so it must not take part in a min or a comparison.
    new_soft = target if hard == resource.RLIM_INFINITY else min(target, hard)
    if old_soft == resource.RLIM_INFINITY or old_soft >= new_soft:
        return old_soft, old_soft
    resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    return old_soft, new_soft


def filter_epoch_info(epoch_info: pd.DataFrame, args: Namespace) -> pd.DataFrame:
    """Filter the epoch_info DataFrame based on command line arguments."""
    is_epoch = np.ones(len(epoch_info), dtype=bool)

    if args.animal is not None:
        is_epoch = np.logical_and(epoch_info.animal == args.animal, is_epoch)

    if args.date is not None:
        is_epoch = np.logical_and(epoch_info.date == args.date, is_epoch)

    if args.epoch is not None:
        epoch = int(args.epoch.split("_")[0])
        is_epoch = np.logical_and(epoch_info.epoch == epoch, is_epoch)

    epoch_info = epoch_info.loc[is_epoch, :]

    if isinstance(epoch_info, pd.Series):
        epoch_info = epoch_info.to_frame().T

    return epoch_info


def add_common_args(parser: ArgumentParser) -> ArgumentParser:
    """Add CLI arguments shared by all decoding scripts."""
    parser.add_argument(
        "--cuda_visible_devices", type=str, default="0,1,2,3,4,5,6,7,8,9"
    )
    parser.add_argument("--local_scratch_dir", type=str, default=".local_scratch")
    parser.add_argument("--brain_area", type=str, default="HPC")
    parser.add_argument("--training_type", type=str, default="no_ripple")
    parser.add_argument("--ripple_detector_name", type=str, default="Kay")
    parser.add_argument("--non_stationary", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--animal", type=str)
    parser.add_argument("--date", type=int)
    parser.add_argument("--epoch", type=str)
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Folder the decodes are written to, and resumed from "
        "(default: Processed-Data). Created if missing.",
    )
    parser.add_argument(
        "--allow_cpu",
        action="store_true",
        help="Decode on a CPU cluster when dask_cuda is not importable, "
        "instead of refusing to start.",
    )
    parser.add_argument(
        "--pending-only",
        action="store_true",
        help="Decode only the epochs the epoch register holds pending a Spyglass "
        "populate (the incremental update, run after the populates), into a "
        "staging --save_dir; the canonical folder is refused. By default the "
        "register's analysed epochs are decoded.",
    )
    return parser


def run_batch(
    run_decode_fn: Callable,
    args: Namespace,
    *,
    extra_kwargs: dict[str, Any] | None = None,
) -> None:
    """Shared main loop: set up cluster, discover sessions, run in parallel.

    Parameters
    ----------
    run_decode_fn : Callable
        The per-session decoding function (e.g., ``run_decode`` from
        the NonLocal or ContFrag script).
    args : Namespace
        Parsed CLI arguments (must have common args from ``add_common_args``).
        ``args.save_dir`` reaches ``run_decode_fn`` resolved against this
        process's working directory, since a dask worker need not share it.
    extra_kwargs : dict, optional
        Additional keyword arguments to pass to ``run_decode_fn`` beyond
        the common ones.

    Notes
    -----
    The epochs are the epoch register's analysed ones, or with ``--pending-only``
    its pending ones alone (``epochs_to_run``), narrowed by ``--animal``,
    ``--date`` and ``--epoch``. Pending epochs are decoded into a staging folder
    and promoted with the register edit that makes them analysed
    (``docs/epoch_register.md``), so ``--pending-only`` refuses the canonical
    ``PROCESSED_DATA_DIR``: a reader of that folder must never meet a pending
    epoch's decode.

    Before the cluster starts, the soft open-file limit is raised
    (``raise_open_file_limit``), so that the worker processes inherit it.
    ``run_decode_fn`` is also given ``release_nwb``: True on the GPU cluster,
    whose workers run one epoch at a time, so each epoch closes the NWB files
    Spyglass opened for it (``release_nwb_files``); False on the CPU cluster,
    whose workers run several epochs at once.

    Raises
    ------
    RuntimeError
        If ``dask_cuda`` is not importable and ``--allow_cpu`` was not given.
        Without this check the batch would silently decode on a CPU cluster.
    ValueError
        If ``--pending-only`` would decode into ``PROCESSED_DATA_DIR``.
    """
    if not HAS_CUDA and not args.allow_cpu:
        raise RuntimeError(
            "dask_cuda is not importable, so this batch would decode on a CPU "
            "cluster. Install dask-cuda by hand (see docs/README.md), or pass "
            "--allow_cpu to decode on CPU."
        )

    if extra_kwargs is None:
        extra_kwargs = {}

    # Resolved once, here, before any worker sees it: a relative --save_dir
    # would otherwise be read against each worker's working directory.
    save_dir = Path(args.save_dir).resolve()
    if args.pending_only and save_dir == Path(PROCESSED_DATA_DIR).resolve():
        raise ValueError(
            "--pending-only decodes the register's pending epochs, which must "
            "not reach the canonical folder before they are analysed: give a "
            f"staging --save_dir, not {save_dir}."
        )

    # Before any cluster starts: a register that does not match Spyglass stops
    # the batch here, with no GPU worker started.
    epoch_info = filter_epoch_info(epochs_to_run(args.pending_only), args)

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"

    # Before the cluster starts: its worker processes inherit this limit.
    old_soft, new_soft = raise_open_file_limit()
    print(f"open-file soft limit {old_soft} -> {new_soft}", flush=True)

    Path(args.local_scratch_dir).mkdir(exist_ok=True, parents=True)

    if HAS_CUDA:
        # One worker per GPU, one thread each: a worker process runs one epoch
        # at a time, so an epoch can close the NWB files Spyglass opened for it
        # without closing a file another epoch is reading.
        cluster = LocalCUDACluster(
            CUDA_VISIBLE_DEVICES=args.cuda_visible_devices,
            local_directory=args.local_scratch_dir,
            threads_per_worker=1,
        )
        release_nwb = True
    else:
        # LocalCluster's default workers run several epochs at once in threads
        # (for example 12 workers x 8 threads on 96 cores), and those threads share
        # Spyglass's per-process cache of open NWB files. Closing it after one
        # epoch could close a file another is still reading, so these epochs
        # leave their files open and rely on the raised open-file limit.
        cluster = LocalCluster(local_directory=args.local_scratch_dir)
        release_nwb = False

    with Client(cluster):
        log_directory = Path.cwd() / "logs"
        log_directory.mkdir(exist_ok=True, parents=True)

        delayed = [
            dask.delayed(run_decode_fn)(
                nwb_file_name=epoch.Index[0],
                epoch_name=epoch.Index[1],
                log_directory=log_directory,
                overwrite=args.overwrite,
                training_type=args.training_type,
                brain_area=args.brain_area,
                non_stationary=args.non_stationary,
                ripple_detector_name=args.ripple_detector_name,
                save_dir=save_dir,
                release_nwb=release_nwb,
                **extra_kwargs,
            )
            for epoch in epoch_info.itertuples()
        ]

        # Per-session run_decode_fn returns the epoch_identifier on
        # failure (after logging the exception) and None on success or
        # graceful skip. dask.compute returns a tuple of these.
        outcomes = dask.compute(*delayed)
        failures = [r for r in outcomes if r is not None]
        n_total = len(outcomes)

        if failures:
            sys.stderr.write(f"\n{len(failures)}/{n_total} session(s) FAILED:\n")
            for ident in failures:
                sys.stderr.write(f"  - {ident}\n")
            sys.stderr.write("See logs/<epoch_identifier>.log for tracebacks.\n")
            sys.exit(1)

        print(f"\nAll {n_total} session(s) completed.")
