"""Shared utilities for decoding scripts.

Contains the common infrastructure used by both ``run_decoding_bandit.py``
(NonLocal 4-state detector) and ``run_decoding_contfrag.py`` (ContFrag
2-state classifier).
"""

import logging
import logging.config
import os
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
    extra_kwargs : dict, optional
        Additional keyword arguments to pass to ``run_decode_fn`` beyond
        the common ones.
    """
    if extra_kwargs is None:
        extra_kwargs = {}

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"

    Path(args.local_scratch_dir).mkdir(exist_ok=True, parents=True)

    if HAS_CUDA:
        cluster = LocalCUDACluster(
            CUDA_VISIBLE_DEVICES=args.cuda_visible_devices,
            local_directory=args.local_scratch_dir,
        )
    else:
        cluster = LocalCluster(local_directory=args.local_scratch_dir)

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
            sys.stderr.write(
                f"\n{len(failures)}/{n_total} session(s) FAILED:\n"
            )
            for ident in failures:
                sys.stderr.write(f"  - {ident}\n")
            sys.stderr.write(
                "See logs/<epoch_identifier>.log for tracebacks.\n"
            )
            sys.exit(1)

        print(f"\nAll {n_total} session(s) completed.")
