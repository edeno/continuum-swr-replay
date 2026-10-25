"""Main script for running continuum neural decoding on spatial bandit task data.

This script performs continuum (non-local) decoding of neural activity to identify
hippocampal replay events during sharp-wave ripples (SWR). It processes sessions
from the spatial bandit task, fitting non-local detectors to distinguish when neural
activity represents the animal's current location versus distant locations (replay).

Usage
-----
# Decode all sessions for an animal
python run_decoding_bandit.py --animal chimi --brain_area HPC

# Decode specific session
python run_decoding_bandit.py --animal chimi --date 20200212 --epoch 02_r1

# With interactive visualization
python run_decoding_bandit.py --animal chimi --create_figurl

# Overwrite existing results
python run_decoding_bandit.py --animal chimi --overwrite

# Write to (and resume from) a staging folder instead of Processed-Data
python run_decoding_bandit.py --animal chimi --save_dir Processed-Data/_staging_redecode_20261003

Output Files
------------
Saved to ``--save_dir`` (default ``paths.processed_data_dir()``):

{epoch_identifier}_results.nc
    Decoding results as xarray Dataset (posterior distributions, state probabilities)
{epoch_identifier}_detector.pkl
    Fitted detector model (can be reloaded for analysis)
{epoch_identifier}_environment.pkl
    Environment configuration (place bins, track graph)
{epoch_identifier}_{threshold}_data_over_time.pkl
    Aggregated time-series data (decoding + behavioral variables)
"""

import logging
import sys
import warnings
from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import xarray as xr

from continuum_swr_replay.analysis.hippocampus import get_data_over_time, load_results
from continuum_swr_replay.data_types import SessionData
from continuum_swr_replay.decoding.config import (
    NonLocalDecodingConfig,
    nonlocal_detector_classes,
)
from continuum_swr_replay.decoding.identifiers import (
    DEFAULT_SWR_DEFINITION,
    DEFAULT_TRAINING_TYPE,
)
from continuum_swr_replay.decoding.transitions import nonlocal_transition_settings
from continuum_swr_replay.parameters import (
    DEFAULT_NON_LOCAL_THRESHOLD,
    SAMPLING_FREQUENCY,
)
from continuum_swr_replay.runs.batch import run_batch
from continuum_swr_replay.runs.cli import add_common_args
from continuum_swr_replay.visualization.figurl import (
    create_interactive_1D_decoding_figurl,
)

# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.decoding import _decode_runner  # noqa: E402

warnings.filterwarnings("ignore")


def nonlocal_detector_kwargs(config: NonLocalDecodingConfig) -> dict[str, Any]:
    """The NonLocal detector's arguments beyond its environment and params.

    The discrete transitions come from ``nonlocal_transition_settings`` (setting
    P30-B, decision R10), never from the library's defaults; the Local state's
    ``local_position_std`` is a detector-level argument.
    """
    return {
        **nonlocal_transition_settings(SAMPLING_FREQUENCY),
        "local_position_std": config.local_position_std,
    }


def generate_figurl(
    logger: logging.Logger,
    data: SessionData,
    results: xr.Dataset,
    data_over_time: pd.DataFrame,
    posterior: xr.DataArray,
    epoch_identifier: str,
    brain_area: str = "HPC",
) -> None:
    """Generate interactive figurl visualization."""
    logger.info("Creating figurl...")
    url = create_interactive_1D_decoding_figurl(
        linear_position=data["position_info"]["linear_position"],
        speed=data["position_info"]["head_speed"],
        head_direction=data["position_info"]["head_orientation"],
        results=results,
        MUA_times=data["hse_times"][brain_area],
        MUA_zscore=data["multiunit_rate_zscore"][brain_area],
        SWR_times=data["ripple_times"],
        SWR_zscore=data["zscored_ripple_consensus_trace"],
        ahead_behind_distance=data_over_time["ahead_behind_distance"],
        hpd=data_over_time["hpd"],
        theta_filtered_lfp=data["theta_filtered_lfp"].iloc[:, 0],
        posterior=posterior,
        view_height=800,
        label=epoch_identifier,
    )
    logger.info(f"Figurl URL: {url}")
    results.attrs["figurl"] = url


RUNNER = _decode_runner.DecodeRunner(
    detector_classes=nonlocal_detector_classes(),
    config_class=NonLocalDecodingConfig,
    detector_kwargs=nonlocal_detector_kwargs,
    aggregate=get_data_over_time,
    load_saved=load_results,
    file_suffix="",
    # Persist the per-timebin log-likelihood: the spike-evidence reality
    # analysis reads it. The ContFrag runner does not.
    hpc_log_likelihood=True,
    figurl=generate_figurl,
    start_banner=" START ",
    decoding_label="Decoding",
)
"""The NonLocal 4-state detector's decode (``_decode_runner.run_decode``)."""


def run_decode(
    nwb_file_name: str,
    epoch_name: str,
    log_directory: str = "",
    overwrite: bool = False,
    create_figurl: bool = False,
    training_type: str = DEFAULT_TRAINING_TYPE,
    non_stationary: bool = False,
    non_local_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
    ripple_detector_name: str = DEFAULT_SWR_DEFINITION,
    brain_area: str = "HPC",
    save_dir: Path | None = None,
    release_nwb: bool = False,
) -> str | None:
    """Run NonLocal decoding for a single session (``_decode_runner.run_decode``).

    Writes ``{epoch_identifier}_results.nc``, ``_detector.pkl``,
    ``_environment.pkl`` and ``_{threshold}_data_over_time.pkl`` to
    ``save_dir``, or on a resume ``data_over_time`` alone. An HPC decode keeps
    its per-bin log-likelihood in ``results.nc``.

    Returns
    -------
    str | None
        ``None`` on success or graceful skip; the ``epoch_identifier``
        on caught exception so ``run_batch`` can aggregate failures.
    """
    return _decode_runner.run_decode(
        RUNNER,
        nwb_file_name,
        epoch_name,
        log_directory=log_directory,
        overwrite=overwrite,
        training_type=training_type,
        non_stationary=non_stationary,
        ripple_detector_name=ripple_detector_name,
        brain_area=brain_area,
        save_dir=save_dir,
        release_nwb=release_nwb,
        non_local_threshold=non_local_threshold,
        create_figurl=create_figurl,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Main entry point for batch decoding analysis."""
    parser = add_common_args(ArgumentParser())
    parser.add_argument(
        "--non_local_threshold", type=float, default=DEFAULT_NON_LOCAL_THRESHOLD
    )
    parser.add_argument("--create_figurl", action="store_true")
    args = _decode_runner.parse_decode_args(parser, argv)

    run_batch(
        run_decode,
        args,
        extra_kwargs={
            "non_local_threshold": args.non_local_threshold,
            "create_figurl": args.create_figurl,
        },
    )


if __name__ == "__main__":
    main()
