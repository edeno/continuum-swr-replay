"""Run Continuous-Fragmented (ContFrag) decoding on spatial bandit task data.

This script fits a 2-state ContFrag classifier to neural activity, classifying
all time as either Continuous (smooth spatial trajectory, RandomWalk transitions)
or Fragmented (disjointed spatial representation, Uniform transitions).

Unlike the NonLocal 4-state detector (run_decoding_bandit.py), the ContFrag
model has no Local or No-Spike states — it decodes spatial position everywhere.
Events for downstream analysis are defined externally via SWR and high-MUA times,
not by model-detected probability thresholding.

Usage
-----
# Decode all sessions for an animal
python run_decoding_contfrag.py --animal chimi --brain_area HPC

# Decode specific session
python run_decoding_contfrag.py --animal chimi --date 20200212 --epoch 02_r1

# With non-stationary model
python run_decoding_contfrag.py --animal chimi --non_stationary

Output Files
------------
Saved to PROCESSED_DATA_DIR with ``_contfrag`` suffix:

{epoch_identifier}_contfrag_results.nc
{epoch_identifier}_contfrag_detector.pkl
{epoch_identifier}_contfrag_environment.pkl
{epoch_identifier}_contfrag_data_over_time.pkl
"""

import logging
import subprocess
import warnings
from argparse import ArgumentParser
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
from non_local_detector import (
    ContFragClusterlessClassifier,
    ContFragSortedSpikesClassifier,
)
from non_local_detector.discrete_state_transitions import DiscreteNonStationaryDiagonal
from non_local_detector.environment import Environment

from continuum_swr_replay.analysis.contfrag import (
    get_contfrag_data_over_time,
    load_contfrag_results,
)
from continuum_swr_replay.data_loaders import (
    get_training_timepoints,
    load_data,
)
from continuum_swr_replay.decoding_common import (
    DecodingConfig,
    add_common_args,
    results_exist,
    run_batch,
    setup_logger,
)
from continuum_swr_replay.parameters import (
    CONTFRAG_DETECTOR_CONFIG,
    PLACE_BIN_SIZE_1D,
    PLACE_BIN_SIZE_2D,
    PROCESSED_DATA_DIR,
)
from continuum_swr_replay.types import SessionData

warnings.filterwarnings("ignore")

# Map string config to actual detector classes
CONTFRAG_DETECTOR_CLASSES = {
    brain_area: (
        (
            ContFragClusterlessClassifier
            if config[0] == "ContFragClusterlessClassifier"
            else ContFragSortedSpikesClassifier
        ),
        config[1],
    )
    for brain_area, config in CONTFRAG_DETECTOR_CONFIG.items()
}


def create_contfrag_detector(
    brain_area: str,
    env: Environment,
    config: DecodingConfig,
    discrete_transition_type,
) -> ContFragClusterlessClassifier | ContFragSortedSpikesClassifier:
    """Factory for creating the appropriate ContFrag classifier."""
    if brain_area not in CONTFRAG_DETECTOR_CLASSES:
        valid_areas = list(CONTFRAG_DETECTOR_CLASSES.keys())
        raise ValueError(
            f"Invalid brain_area '{brain_area}'. Must be one of {valid_areas}"
        )

    detector_class, param_type = CONTFRAG_DETECTOR_CLASSES[brain_area]
    params = getattr(config, f"to_{param_type}_params")()

    param_key = f"{param_type}_algorithm_params"
    return detector_class(
        environments=env,
        **{param_key: params},
        discrete_transition_type=discrete_transition_type,
    )


def perform_decoding(
    logger: logging.Logger,
    spike_times: list,
    spike_waveform_features: list | None,
    position_info: pd.DataFrame,
    linear_edge_order: list,
    linear_edge_spacing: list,
    track_graph: nx.Graph,
    is_training: np.ndarray,
    brain_area: str,
    non_stationary: bool = False,
    is_1d: bool = True,
) -> tuple[
    xr.Dataset,
    ContFragClusterlessClassifier | ContFragSortedSpikesClassifier,
    Environment,
]:
    """Perform ContFrag decoding for a given brain area."""
    if is_1d:
        env = Environment(
            place_bin_size=PLACE_BIN_SIZE_1D,
            track_graph=track_graph,
            edge_order=linear_edge_order,
            edge_spacing=linear_edge_spacing,
        )
    else:
        env = Environment(
            place_bin_size=PLACE_BIN_SIZE_2D,
        )

    config = DecodingConfig()

    if non_stationary:
        discrete_transition_covariate_data = {
            "speed": position_info["head_speed"].to_numpy()
        }
        # ContFrag has 2 states: [Continuous, Fragmented]
        diagonal_values = np.array([0.98, 0.98])
        discrete_transition_type = DiscreteNonStationaryDiagonal(
            diagonal_values=diagonal_values,
        )
    else:
        discrete_transition_covariate_data = None
        discrete_transition_type = None

    logger.info(f"ContFrag decoding {brain_area}...")

    detector = create_contfrag_detector(
        brain_area, env, config, discrete_transition_type
    )

    estimate_params = {
        "position_time": position_info.index.to_numpy(),
        "position": position_info[["head_position_x", "head_position_y"]].to_numpy(),
        "spike_times": spike_times,
        "time": position_info.index.to_numpy(),
        "discrete_transition_covariate_data": discrete_transition_covariate_data,
    }

    if brain_area == "HPC":
        estimate_params["is_training"] = is_training
        if spike_waveform_features is not None:
            estimate_params["spike_waveform_features"] = spike_waveform_features

    results = detector.estimate_parameters(**estimate_params)

    return results, detector, env


def save_results_and_model(
    logger: logging.Logger,
    results: xr.Dataset,
    detector: ContFragClusterlessClassifier | ContFragSortedSpikesClassifier,
    env: Environment,
    data_over_time: pd.DataFrame,
    save_folder: Path,
    epoch_identifier: str,
) -> None:
    """Save ContFrag decoding results."""
    logger.info("Saving results...")
    results.reset_index("state_bins").to_netcdf(
        save_folder / f"{epoch_identifier}_results.nc",
        engine="h5netcdf",
    )
    detector.save_model(save_folder / f"{epoch_identifier}_detector.pkl")
    env.save_environment(save_folder / f"{epoch_identifier}_environment.pkl")

    data_over_time.to_pickle(
        save_folder / f"{epoch_identifier}_data_over_time.pkl"
    )


def organize_results(
    data: SessionData,
    results: xr.Dataset,
    env: Environment,
    nwb_file_name: str,
    epoch_name: str,
    brain_area: str,
) -> tuple[pd.DataFrame, xr.DataArray]:
    """Organize ContFrag decoding results and compute derived measures."""
    time_diffs = np.diff(results.time.values)
    if len(time_diffs) == 0:
        raise ValueError(
            "Time array must have at least 2 elements to compute sampling frequency"
        )
    if np.any(time_diffs <= 0):
        n_bad = np.sum(time_diffs <= 0)
        bad_indices = np.where(time_diffs <= 0)[0][:5]
        bad_values = time_diffs[bad_indices]
        raise ValueError(
            f"Time array must be strictly increasing. "
            f"Found {n_bad}/{len(time_diffs)} non-positive differences. "
            f"First problematic indices: {bad_indices.tolist()}, values: {bad_values.tolist()}"
        )

    median_dt = np.median(time_diffs)
    if median_dt <= 1e-10:
        raise ValueError(
            f"Median time difference too small ({median_dt}), cannot compute sampling frequency"
        )

    sampling_frequency = 1 / median_dt
    return get_contfrag_data_over_time(
        data,
        results,
        env,
        nwb_file_name,
        epoch_name,
        brain_area,
        sampling_frequency=sampling_frequency,
    )


def run_decode(
    nwb_file_name: str,
    epoch_name: str,
    log_directory: str = "",
    overwrite: bool = False,
    training_type: str = "no_ripple",
    non_stationary: bool = False,
    ripple_detector_name: str = "Kay",
    brain_area: str = "HPC",
) -> str | None:
    """Run ContFrag decoding for a single session.

    Returns
    -------
    str | None
        ``None`` on success or graceful skip; the ``epoch_identifier``
        on caught exception so ``run_batch`` can aggregate failures.
    """
    epoch_identifier = (
        f"{nwb_file_name.replace('_.nwb', '')}_{epoch_name}"
        f"_{brain_area}_contfrag"
    )
    if ripple_detector_name != "Kay":
        epoch_identifier += f"_{ripple_detector_name}"
    if non_stationary:
        epoch_identifier += "_non_stationary"

    save_folder = PROCESSED_DATA_DIR

    logger = setup_logger(
        name_logfile=epoch_identifier,
        path_logfile=Path(log_directory) / f"{epoch_identifier}.log",
    )

    try:
        logger.info(" START (ContFrag) ".center(50, "#"))

        git_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            text=True,
        ).stdout
        logger.info(f"Git Hash: {git_hash.rstrip()}")
        logger.info("model: ContFrag (2-state: Continuous, Fragmented)")
        logger.info(f"training type: {training_type}")
        logger.info(f"non_stationary: {non_stationary}")
        logger.info(f"ripple_detector_name: {ripple_detector_name}")
        logger.info(f"brain_area: {brain_area}")
        logger.info("Loading data...")
        data = load_data(
            nwb_file_name,
            epoch_name,
            ripple_detector_name=ripple_detector_name,
        )

        if not data["spike_times"]:
            logger.info("No spikes. Skipping...")
            logger.info("Done!")
            return

        if brain_area not in data["spike_times"]:
            logger.warning(f"No spike data available for {brain_area}. Skipping...")
            return

        if results_exist(save_folder, epoch_identifier, overwrite):
            logger.info(
                f"ContFrag results already exist for {brain_area}. Skipping decoding..."
            )
            results, detector, env = load_contfrag_results(
                nwb_file_name,
                epoch_name,
                brain_area,
                non_stationary=non_stationary,
                ripple_detector_name=ripple_detector_name,
            )
        else:
            is_training = get_training_timepoints(data, training_type=training_type)

            spike_waveform_features = data["spike_waveform_features"].get(
                brain_area, None
            )

            results, detector, env = perform_decoding(
                logger,
                data["spike_times"][brain_area],
                spike_waveform_features,
                data["position_info"],
                data["linear_edge_order"],
                data["linear_edge_spacing"],
                data["track_graph"],
                is_training,
                brain_area=brain_area,
                non_stationary=non_stationary,
            )

        logger.info("Organizing results...")
        data_over_time, posterior = organize_results(
            data, results, env, nwb_file_name, epoch_name, brain_area,
        )

        save_results_and_model(
            logger, results, detector, env,
            data_over_time, save_folder, epoch_identifier,
        )

        logger.info("Done!")
    except Exception as e:
        logger.exception(e)
        return epoch_identifier  # Sentinel: run_batch treats non-None as failure.


def main() -> None:
    """Main entry point for batch ContFrag decoding."""
    parser = add_common_args(
        ArgumentParser(description="Run ContFrag decoding on spatial bandit data")
    )
    args = parser.parse_args()
    run_batch(run_decode, args)


if __name__ == "__main__":
    main()
