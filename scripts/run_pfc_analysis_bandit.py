"""Batch PFC GLM analysis of neuron modulation during non-local hippocampal events.

This script runs GLM-based analysis of prefrontal cortex (mPFC/OFC) neuron firing
rate modulation around sharp-wave ripples (SWR) and non-local hippocampal events.
For each session, it fits Poisson GLM models with convolutional spline basis functions
to quantify how each neuron's firing rate changes relative to event onsets.

The analysis pipeline per session:
1. Load HPC decoding results to identify non-local events
2. Separate non-local events by SWR overlap
3. Fit GLMs for each neuron comparing models: constant, SWR, non-local, combined
4. Compute theta modulation during running
5. Save per-session neuron statistics and modulation time courses

Requires HPC decoding results to already exist (run run_decoding_bandit.py first).

Usage
-----
# Analyze all mPFC sessions
python run_pfc_analysis_bandit.py --brain_area mPFC

# Analyze specific animal
python run_pfc_analysis_bandit.py --animal wilbur --brain_area mPFC

# Analyze specific session
python run_pfc_analysis_bandit.py --animal wilbur --date 20210329 --epoch 02_r1

# Overwrite existing results
python run_pfc_analysis_bandit.py --brain_area mPFC --overwrite

# Use more Dask workers
python run_pfc_analysis_bandit.py --brain_area mPFC --n_workers 16
"""

import logging
import logging.config
import subprocess
import sys
import warnings
from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from pathlib import Path
from typing import Any

import dask
import numpy as np
import pandas as pd
from dask.distributed import Client, LocalCluster

from continuum_swr_replay.analysis.pfc import run_session
from continuum_swr_replay.data_loaders import get_epoch_info
from continuum_swr_replay.parameters import (
    MAX_SPEED_THRESHOLD_CM_PER_SEC,
    PROCESSED_DATA_DIR,
    SAMPLING_FREQUENCY,
)

warnings.filterwarnings("ignore", message=".*Skipped checksum.*")
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="datajoint")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="hdmf")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pynwb")


class SkipChecksumFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Skipped checksum" not in record.getMessage()


def setup_logger(
    name_logfile: str,
    path_logfile: str | Path,
    *,
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up a logger that logs to console and a file.

    Parameters
    ----------
    name_logfile : str
        Logger name.
    path_logfile : str | Path
        Path to the logfile. Parent dirs will be created if absent.
    level : int
        Logging level (default: INFO).

    Returns
    -------
    logging.Logger
        Configured logger.
    """
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

    logging.config.dictConfig(config)
    return logging.getLogger(name_logfile)


def results_exist(
    save_folder: Path,
    epoch_identifier: str,
    overwrite: bool,
) -> bool:
    """Check if all modulation result files already exist for this session."""
    if overwrite:
        return False
    save_base = save_folder / f"{epoch_identifier}_modulation"
    required_files = [
        save_base.with_suffix(".pkl"),
        save_base.with_suffix(".SWR_modulation.npy"),
        save_base.with_suffix(".no_SWR_modulation.npy"),
        save_base.with_suffix(".non_local_SWR_modulation.npy"),
        save_base.with_suffix(".non_local_no_SWR_modulation.npy"),
    ]
    return all(f.exists() for f in required_files)


def run_analysis(
    nwb_file_name: str,
    epoch_name: str,
    brain_area: str = "mPFC",
    sampling_frequency: int = SAMPLING_FREQUENCY,
    immobility_threshold: float = MAX_SPEED_THRESHOLD_CM_PER_SEC,
    log_directory: str | Path = "logs",
    overwrite: bool = False,
) -> None:
    """Run PFC GLM analysis for a single session.

    Parameters
    ----------
    nwb_file_name : str
        NWB file name (e.g., "wilbur20210329_.nwb")
    epoch_name : str
        Epoch name (e.g., "02_r1")
    brain_area : str
        Brain area to analyze ("mPFC" or "OFC")
    sampling_frequency : int
        Sampling frequency in Hz
    immobility_threshold : float
        Speed threshold for immobility classification (cm/s)
    log_directory : str | Path
        Directory for log files
    overwrite : bool
        Whether to overwrite existing results
    """
    epoch_identifier = (
        f"{nwb_file_name.replace('_.nwb', '')}_{epoch_name}_{brain_area}"
    )
    save_file = PROCESSED_DATA_DIR / f"{epoch_identifier}_modulation.pkl"

    logger = setup_logger(
        name_logfile=epoch_identifier,
        path_logfile=Path(log_directory) / f"{epoch_identifier}.log",
    )

    if results_exist(PROCESSED_DATA_DIR, epoch_identifier, overwrite):
        logger.info(f"Results already exist for {epoch_identifier}. Skipping...")
        return

    logger.info(" START ".center(50, "#"))

    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    ).stdout
    logger.info(f"Git Hash: {git_hash.rstrip()}")
    logger.info(f"brain_area: {brain_area}")
    logger.info(f"sampling_frequency: {sampling_frequency}")
    logger.info(f"immobility_threshold: {immobility_threshold}")

    try:
        logger.info(f"Processing {epoch_identifier}...")
        (
            neuron_stats,
            SWR_modulation,
            no_SWR_modulation,
            non_local_SWR_modulation,
            non_local_no_SWR_modulation,
        ) = run_session(
            nwb_file_name,
            epoch_name,
            brain_area=brain_area,
            sampling_frequency=sampling_frequency,
            immobility_threshold=immobility_threshold,
            run_parallel=False,
        )

        logger.info(f"Analyzed {len(neuron_stats)} neurons")

        neuron_stats.to_pickle(save_file)
        np.save(
            save_file.with_suffix(".SWR_modulation.npy"),
            SWR_modulation,
        )
        np.save(
            save_file.with_suffix(".no_SWR_modulation.npy"),
            no_SWR_modulation,
        )
        np.save(
            save_file.with_suffix(".non_local_SWR_modulation.npy"),
            non_local_SWR_modulation,
        )
        np.save(
            save_file.with_suffix(".non_local_no_SWR_modulation.npy"),
            non_local_no_SWR_modulation,
        )

        logger.info(f"Saved results to {save_file}")
        logger.info("Done!")
    except Exception as e:
        logger.exception(f"Error processing {epoch_identifier}: {e}")


def get_command_line_arguments() -> Namespace:
    parser = ArgumentParser(
        description=__doc__,
        formatter_class=RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--local_scratch_dir", type=str, default=".local_scratch",
        help="Local scratch directory for Dask workers",
    )
    parser.add_argument(
        "--n_workers", type=int, default=32,
        help="Number of Dask workers (default: 32)",
    )
    parser.add_argument(
        "--animal", type=str, default=None,
        help="Filter to specific animal (e.g., wilbur)",
    )
    parser.add_argument(
        "--date", type=int, default=None,
        help="Filter to specific date (e.g., 20210329)",
    )
    parser.add_argument(
        "--epoch", type=str, default=None,
        help="Filter to specific epoch (e.g., 02_r1)",
    )
    parser.add_argument(
        "--brain_area", type=str, default="mPFC",
        help="Brain area to analyze: mPFC or OFC (default: mPFC)",
    )
    parser.add_argument(
        "--log_directory", type=str, default="logs",
        help="Directory for log files (default: logs)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing results",
    )

    return parser.parse_args()


def _filter_epoch_info(epoch_info: pd.DataFrame, args: Namespace) -> pd.DataFrame:
    """Filter epoch_info based on command line arguments.

    Parameters
    ----------
    epoch_info : pd.DataFrame
        All available epochs
    args : Namespace
        Parsed command line arguments

    Returns
    -------
    pd.DataFrame
        Filtered epochs
    """
    is_epoch = np.ones(len(epoch_info), dtype=bool)

    if args.animal is not None:
        is_epoch = np.logical_and(epoch_info.animal == args.animal, is_epoch)

    if args.date is not None:
        is_epoch = np.logical_and(epoch_info.date == args.date, is_epoch)

    if args.epoch is not None:
        is_epoch = np.logical_and(
            epoch_info.index.get_level_values("interval_list_name") == args.epoch,
            is_epoch,
        )

    epoch_info = epoch_info.loc[is_epoch, :]

    if isinstance(epoch_info, pd.Series):
        epoch_info = epoch_info.to_frame().T

    return epoch_info


def _filter_to_sorted_sessions(epoch_info: pd.DataFrame) -> pd.DataFrame:
    """Filter to sessions that have curated spike sorting results.

    Parameters
    ----------
    epoch_info : pd.DataFrame
        Available epochs

    Returns
    -------
    pd.DataFrame
        Epochs with curated spike sorting
    """
    try:
        from spyglass.spikesorting.v0 import CuratedSpikeSorting

        restriction = {
            "preproc_params_name": "default",
            "sort_interval_name": (
                "sleeps_runs_noPrePostTrialTimes raw data valid times"
            ),
            "sorter_params_name": (
                "franklab_probe_ctx_30KHz_115rad_new_mountainsort2"
            ),
            "team_name": "ac_em_xs",
            "sorter": "mountainsort4",
        }

        sorted_nwb_file_names = np.unique(
            (CuratedSpikeSorting & restriction).fetch("nwb_file_name")
        )
        epoch_info = epoch_info[
            epoch_info.index.get_level_values("nwb_file_name").isin(
                sorted_nwb_file_names
            )
        ]
    except ImportError:
        logging.getLogger(__name__).warning(
            "Spyglass not available. Running on all sessions without filtering "
            "by curated spike sorting."
        )

    return epoch_info


def main() -> None:
    """Main entry point for batch PFC GLM analysis."""
    args = get_command_line_arguments()

    Path(args.local_scratch_dir).mkdir(exist_ok=True, parents=True)
    Path(args.log_directory).mkdir(exist_ok=True, parents=True)
    PROCESSED_DATA_DIR.mkdir(exist_ok=True, parents=True)

    epoch_info = get_epoch_info()
    epoch_info = _filter_epoch_info(epoch_info, args)
    epoch_info = _filter_to_sorted_sessions(epoch_info)

    print(f"Processing {len(epoch_info)} sessions for {args.brain_area}...")

    if args.n_workers <= 1 or len(epoch_info) <= 1:
        for epoch in epoch_info.itertuples():
            run_analysis(
                nwb_file_name=epoch.Index[0],
                epoch_name=epoch.Index[1],
                brain_area=args.brain_area,
                sampling_frequency=SAMPLING_FREQUENCY,
                immobility_threshold=MAX_SPEED_THRESHOLD_CM_PER_SEC,
                log_directory=args.log_directory,
                overwrite=args.overwrite,
            )
    else:
        cluster = LocalCluster(
            local_directory=args.local_scratch_dir,
            n_workers=args.n_workers,
        )

        with Client(cluster):
            results = [
                dask.delayed(run_analysis)(
                    nwb_file_name=epoch.Index[0],
                    epoch_name=epoch.Index[1],
                    brain_area=args.brain_area,
                    sampling_frequency=SAMPLING_FREQUENCY,
                    immobility_threshold=MAX_SPEED_THRESHOLD_CM_PER_SEC,
                    log_directory=args.log_directory,
                    overwrite=args.overwrite,
                )
                for epoch in epoch_info.itertuples()
            ]

            dask.compute(*results)


if __name__ == "__main__":
    sys.exit(main())
