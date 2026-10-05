"""Shared utilities for decoding scripts.

Contains the common infrastructure used by both ``run_decoding_bandit.py``
(NonLocal 4-state detector) and ``run_decoding_contfrag.py`` (ContFrag
2-state classifier).
"""

import hashlib
import importlib.metadata
import json
import logging
import logging.config
import os
import resource
import subprocess
import sys
from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import dask
import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster

try:
    from dask_cuda import LocalCUDACluster

    HAS_CUDA = True
except ImportError:
    HAS_CUDA = False

from continuum_swr_replay.data_loaders import get_epoch_info
from continuum_swr_replay.parameters import (
    BLOCK_SIZE,
    LOCAL_POSITION_STD,
    POSITION_STD,
    PROCESSED_DATA_DIR,
    WAVEFORM_STD,
)
from continuum_swr_replay.types import ClusterlessParams, SortedSpikesParams


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

    required_files = [
        save_folder / f"{epoch_identifier}_results.nc",
        save_folder / f"{epoch_identifier}_detector.pkl",
        save_folder / f"{epoch_identifier}_environment.pkl",
    ]
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

    Returns
    -------
    int
        How many files the cache held; all are now closed.
    """
    # Imported here so that importing this module does not need Spyglass.
    from spyglass.utils import nwb_helper_fn

    n_open = len(vars(nwb_helper_fn)["__open_nwb_files"])
    nwb_helper_fn.close_nwb_files()
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
        unprivileged process cannot raise.

    Returns
    -------
    old_soft : int
    new_soft : int
        The soft limit before and after; equal when it was not raised.
    """
    old_soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    new_soft = min(target, hard)
    if old_soft >= new_soft:
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

        epoch_info = get_epoch_info()
        epoch_info = filter_epoch_info(epoch_info, args)

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
