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
Saved to ``--save_dir`` (default PROCESSED_DATA_DIR):

{epoch_identifier}_results.nc
    Decoding results as xarray Dataset (posterior distributions, state probabilities)
{epoch_identifier}_detector.pkl
    Fitted detector model (can be reloaded for analysis)
{epoch_identifier}_environment.pkl
    Environment configuration (place bins, track graph)
{epoch_identifier}_{threshold}_data_over_time.pkl
    Aggregated time-series data (decoding + behavioral variables)
"""

import json
import logging
import warnings
from argparse import ArgumentParser
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import xarray as xr
from non_local_detector import NonLocalClusterlessDetector, NonLocalSortedSpikesDetector
from non_local_detector.environment import Environment

from continuum_swr_replay.analysis.hippocampus import get_data_over_time, load_results
from continuum_swr_replay.analysis.hippocampus.swr_source import with_swr_source
from continuum_swr_replay.data_loaders import (
    get_training_timepoints,
    load_data,
)
from continuum_swr_replay.decoding_common import (
    NonLocalDecodingConfig,
    add_common_args,
    hpc_artifact_intervals,
    hpc_decode_masks,
    provenance_record,
    release_nwb_files,
    require_hpc_decode_sources,
    require_saved_decode_masks,
    results_exist,
    run_batch,
    setup_logger,
    write_atomically,
)
from continuum_swr_replay.decoding_transitions import (
    NON_STATIONARY_REFUSAL,
    nonlocal_transition_settings,
)
from continuum_swr_replay.parameters import (
    DETECTOR_CONFIG,
    PLACE_BIN_SIZE_1D,
    PLACE_BIN_SIZE_2D,
    PROCESSED_DATA_DIR,
    ROOT_DIR,
    SAMPLING_FREQUENCY,
)
from continuum_swr_replay.types import SessionData
from continuum_swr_replay.visualization.figurl import (
    create_interactive_1D_decoding_figurl,
)

warnings.filterwarnings("ignore")

# Map string config to actual detector classes
DETECTOR_CLASSES = {
    brain_area: (
        (
            NonLocalClusterlessDetector
            if config[0] == "NonLocalClusterlessDetector"
            else NonLocalSortedSpikesDetector
        ),
        config[1],
    )
    for brain_area, config in DETECTOR_CONFIG.items()
}


def create_detector(
    brain_area: str,
    env: Environment,
    config: NonLocalDecodingConfig,
) -> NonLocalClusterlessDetector | NonLocalSortedSpikesDetector:
    """Factory for creating the appropriate detector for a brain area.

    The discrete transitions come from ``nonlocal_transition_settings`` (setting
    P30-B, decision R10), never from the library's defaults.
    """
    if brain_area not in DETECTOR_CLASSES:
        valid_areas = list(DETECTOR_CLASSES.keys())
        raise ValueError(
            f"Invalid brain_area '{brain_area}'. Must be one of {valid_areas}"
        )

    detector_class, param_type = DETECTOR_CLASSES[brain_area]
    params = getattr(config, f"to_{param_type}_params")()

    param_key = f"{param_type}_algorithm_params"
    return detector_class(
        environments=env,
        **{param_key: params},
        **nonlocal_transition_settings(SAMPLING_FREQUENCY),
        local_position_std=config.local_position_std,
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
    is_missing: np.ndarray | None = None,
) -> tuple[
    xr.Dataset, NonLocalClusterlessDetector | NonLocalSortedSpikesDetector, Environment
]:
    """Perform neural decoding for a given brain area.

    Raises ``ValueError`` when ``non_stationary`` is True: the library's
    non-stationary transitions are disabled (``NON_STATIONARY_REFUSAL``).

    ``is_training`` (shape (n_time,), on the position grid) selects the bins
    the encoding model is fitted on; ``is_missing`` (shape (n_time,), the same
    grid), when given, marks bins whose log-likelihood the decoder sets to 0,
    as an HPC decode's artifact mask does.
    """
    if non_stationary:
        raise ValueError(NON_STATIONARY_REFUSAL)

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

    config = NonLocalDecodingConfig()

    logger.info(f"Decoding {brain_area}...")

    detector = create_detector(brain_area, env, config)

    estimate_params = {
        "position_time": position_info.index.to_numpy(),
        "position": position_info[["head_position_x", "head_position_y"]].to_numpy(),
        "spike_times": spike_times,
        "time": position_info.index.to_numpy(),
        "discrete_transition_covariate_data": None,
    }

    # Apply the training mask (e.g. no_ripple) for every brain area — the
    # sorted-spikes detectors (mPFC/OFC) accept `is_training` too and otherwise
    # silently train on all bins, contrary to the logged training_type.
    estimate_params["is_training"] = is_training
    if is_missing is not None:
        estimate_params["is_missing"] = is_missing
    if brain_area == "HPC":
        if spike_waveform_features is not None:
            estimate_params["spike_waveform_features"] = spike_waveform_features
        # Persist the per-timebin log-likelihood alongside the posterior so the
        # spike-evidence reality analysis (scripts/qa/analyze_event_spike_evidence.py)
        # can read it instead of recomputing the likelihood per event. Adds a
        # (n_time, n_state_bins) `log_likelihood` data_var to results.nc.
        estimate_params["return_outputs"] = "log_likelihood"

    results = detector.estimate_parameters(**estimate_params)

    return results, detector, env


def save_results_and_model(
    logger: logging.Logger,
    results: xr.Dataset,
    detector: NonLocalClusterlessDetector | NonLocalSortedSpikesDetector,
    env: Environment,
    data_over_time: pd.DataFrame,
    save_folder: Path,
    epoch_identifier: str,
    non_local_threshold: float,
    *,
    resumed: bool,
    provenance: str,
    decode_masks: str | None = None,
) -> None:
    """Save decoding results, detector model, environment, and derived data.

    Every file goes through ``write_atomically``, so a crash never leaves a
    truncated file that ``results_exist`` would trust on resume.

    A fresh save stamps ``provenance`` (the JSON of the epoch's PROVENANCE log
    line) into ``results.nc`` as ``attrs["provenance"]``, so the record outlives
    the log, which a resume rewrites, and an HPC decode's ``decode_masks``
    (the JSON of its ``DECODE MASKS`` log line) as ``attrs["decode_masks"]``.
    When ``resumed`` is True the results,
    detector and environment were read back from ``save_folder``, and
    ``results`` is still open lazily on its file there, so only
    ``data_over_time`` is written.
    """
    if resumed:
        logger.info("Resumed: saving data_over_time only...")
    else:
        logger.info("Saving results...")
        stamped = results.reset_index("state_bins")
        stamped.attrs["provenance"] = provenance
        if decode_masks is not None:
            stamped.attrs["decode_masks"] = decode_masks
        write_atomically(
            save_folder / f"{epoch_identifier}_results.nc",
            lambda path: stamped.to_netcdf(path, engine="h5netcdf"),
        )
        write_atomically(
            save_folder / f"{epoch_identifier}_detector.pkl", detector.save_model
        )
        write_atomically(
            save_folder / f"{epoch_identifier}_environment.pkl", env.save_environment
        )

    threshold_int = int(non_local_threshold * 100)
    write_atomically(
        save_folder / f"{epoch_identifier}_{threshold_int}_data_over_time.pkl",
        data_over_time.to_pickle,
    )


def organize_results(
    data: SessionData,
    results: xr.Dataset,
    env: Environment,
    nwb_file_name: str,
    epoch_name: str,
    brain_area: str,
    non_local_threshold: float = 0.8,
) -> tuple[pd.DataFrame, xr.DataArray]:
    """Organize decoding results and compute derived measures.

    For HPC, ``data_over_time`` is then read through the SWR source
    (``swr_source.with_swr_source``): ``SWR_event`` holds the configured Kay
    inventory's labels and ``swr_observed`` its coverage, and the stored
    ripples' labels and counts are gone.
    """
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
            f"Time array must be strictly increasing with positive differences. "
            f"Found {n_bad}/{len(time_diffs)} non-positive differences. "
            f"First problematic indices: {bad_indices.tolist()}, values: {bad_values.tolist()}"
        )

    median_dt = np.median(time_diffs)
    if median_dt <= 1e-10:
        raise ValueError(
            f"Median time difference too small ({median_dt}), cannot compute sampling frequency"
        )

    sampling_frequency = 1 / median_dt
    data_over_time, posterior = get_data_over_time(
        data,
        results,
        env,
        nwb_file_name,
        epoch_name,
        brain_area,
        sampling_frequency=sampling_frequency,
        non_local_threshold=non_local_threshold,
    )
    if brain_area == "HPC":
        # SWR_event from the SWR source, so the decode's provenance is the
        # analyses' source: the stored ripples' labels and counts are replaced.
        # The decode keeps every event it decoded; the analyses' read drops
        # the ones the artifact mask reaches, each with its reason.
        data_over_time = with_swr_source(data_over_time, drop_masked_events=False)

    return data_over_time, posterior


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


def run_decode(
    nwb_file_name: str,
    epoch_name: str,
    log_directory: str = "",
    overwrite: bool = False,
    create_figurl: bool = True,
    training_type: str = "no_ripple",
    non_stationary: bool = False,
    non_local_threshold: float = 0.8,
    ripple_detector_name: str = "Kay",
    brain_area: str = "HPC",
    save_dir: Path = PROCESSED_DATA_DIR,
    release_nwb: bool = False,
) -> str | None:
    """Run NonLocal decoding for a single session.

    Results are written to ``save_dir``. If they are already there (and
    ``overwrite`` is False), the decode resumes: it logs ``RESUMED from existing
    results in <save_dir>``, reads them back from ``save_dir`` and rewrites only
    ``data_over_time``.

    An HPC decode reads the SWR source (``swr_source``) and refuses to start
    unless it names both its Kay inventory and its artifact run
    (``hpc_artifact_intervals``). Its artifact mask, the one every stage uses,
    masks MUA detection (``load_data``) and is its ``is_missing``; its encoding
    leaves the mask and every inventory event out (``hpc_decode_masks``, logged as
    ``DECODE MASKS`` and stamped into ``results.nc``); and its
    ``data_over_time`` carries the source's ``SWR_event``. An HPC resume
    refuses results whose stamped masks name other runs, or none
    (``require_saved_decode_masks``).

    With ``release_nwb``, the epoch ends, pass or fail, by closing the NWB files
    Spyglass holds open in this process (``release_nwb_files``) and logging how
    many. Only for a process that runs one epoch at a time, as ``run_batch``'s
    GPU workers do: another task in the process could still be reading them.

    Returns
    -------
    str | None
        ``None`` on success or graceful skip; the ``epoch_identifier``
        on caught exception so ``run_batch`` can aggregate failures.
    """
    epoch_identifier = f"{nwb_file_name.replace('_.nwb', '')}_{epoch_name}_{brain_area}"
    if ripple_detector_name != "Kay":
        epoch_identifier += f"_{ripple_detector_name}"
    if non_stationary:
        epoch_identifier += "_non_stationary"
    if training_type != "no_ripple":
        epoch_identifier += f"_{training_type}"

    logger = setup_logger(
        name_logfile=epoch_identifier,
        path_logfile=Path(log_directory) / f"{epoch_identifier}.log",
    )

    try:
        logger.info(" START ".center(50, "#"))
        provenance = provenance_record(ROOT_DIR)
        provenance_json = json.dumps(provenance, sort_keys=True)
        logger.info("PROVENANCE " + provenance_json)
        logger.info(f"Git Hash: {provenance['commit']}")
        logger.info(f"training type: {training_type}")
        logger.info(f"non_stationary: {non_stationary}")
        logger.info(f"non_local_threshold: {non_local_threshold}")
        logger.info(f"ripple_detector_name: {ripple_detector_name}")
        logger.info(f"brain_area: {brain_area}")
        logger.info(f"save_dir: {save_dir.resolve()}")
        save_dir.mkdir(parents=True, exist_ok=True)
        artifact_mask = None
        if brain_area == "HPC":
            # fail closed, before anything is loaded: an HPC decode's masks and
            # SWR labels need the SWR source's inventory and artifact runs
            artifact_mask = hpc_artifact_intervals(nwb_file_name, epoch_name)
        logger.info("Loading data...")
        data = load_data(
            nwb_file_name,
            epoch_name,
            ripple_detector_name=ripple_detector_name,
            mua_artifact_intervals=(
                None if artifact_mask is None else {"HPC": artifact_mask}
            ),
        )

        if not data["spike_times"]:
            logger.info("No spikes. Skipping...")
            logger.info("Done!")
            return

        if brain_area not in data["spike_times"]:
            logger.warning(f"No spike data available for {brain_area}. Skipping...")
            return

        decode_masks = None
        resumed = results_exist(save_dir, epoch_identifier, overwrite)
        if resumed:
            logger.info(f"RESUMED from existing results in {save_dir.resolve()}")
            results, detector, env = load_results(
                nwb_file_name,
                epoch_name,
                brain_area,
                non_stationary=non_stationary,
                ripple_detector_name=ripple_detector_name,
                training_type=training_type,
                root=save_dir,
            )
            if artifact_mask is not None:
                # the reused results must come from the configured masks
                require_saved_decode_masks(
                    results.attrs, str(save_dir / f"{epoch_identifier}_results.nc")
                )
        else:
            is_missing = None
            if artifact_mask is None:
                is_training = get_training_timepoints(data, training_type=training_type)
            else:
                masks = hpc_decode_masks(
                    nwb_file_name,
                    epoch_name,
                    data["position_info"],
                    artifact_mask,
                    training_type,
                )
                decode_masks = json.dumps(masks.record, sort_keys=True)
                logger.info("DECODE MASKS " + decode_masks)
                is_training, is_missing = masks.is_training, masks.is_missing

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
                is_missing=is_missing,
            )

        logger.info("Organizing results...")
        data_over_time, posterior = organize_results(
            data,
            results,
            env,
            nwb_file_name,
            epoch_name,
            brain_area,
            non_local_threshold=non_local_threshold,
        )

        if create_figurl:
            generate_figurl(
                logger,
                data,
                results,
                data_over_time,
                posterior,
                epoch_identifier,
                brain_area=brain_area,
            )

        save_results_and_model(
            logger,
            results,
            detector,
            env,
            data_over_time,
            save_dir,
            epoch_identifier,
            non_local_threshold,
            resumed=resumed,
            provenance=provenance_json,
            decode_masks=decode_masks,
        )

        logger.info("Done!")
    except Exception as e:
        logger.exception(e)
        return epoch_identifier  # Sentinel: run_batch treats non-None as failure.
    finally:
        if release_nwb:
            logger.info(f"closed {release_nwb_files()} NWB files Spyglass held open")


def main() -> None:
    """Main entry point for batch decoding analysis."""
    parser = add_common_args(ArgumentParser())
    parser.add_argument("--non_local_threshold", type=float, default=0.8)
    parser.add_argument("--create_figurl", action="store_true")
    args = parser.parse_args()
    if args.non_stationary:
        parser.error(NON_STATIONARY_REFUSAL)
    if args.brain_area == "HPC":
        try:
            require_hpc_decode_sources()
        except ValueError as error:
            parser.error(str(error))

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
