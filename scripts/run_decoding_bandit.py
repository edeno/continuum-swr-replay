"""Main script for running continuum neural decoding on spatial bandit task data.

This script performs continuum (non-local) decoding of neural activity to identify
hippocampal replay events during sharp-wave ripples (SWR). It processes sessions
from the spatial bandit task, fitting non-local detectors to distinguish when neural
activity represents the animal's current location versus distant locations (replay).

The script supports:
- Multi-session batch processing across animals
- GPU-accelerated parallel decoding with Dask CUDA
- Both clusterless (HPC) and sorted spikes (mPFC/OFC) decoding
- Stationary and non-stationary state transition models
- Interactive figurl visualization generation
- Comprehensive logging and reproducibility tracking

Usage
-----
Command-line interface with various options:

# Decode all sessions for an animal
python run_decoding_bandit.py --animal chimi --brain_area HPC

# Decode specific session
python run_decoding_bandit.py --animal chimi --date 20200212 --epoch 02_r1

# With non-stationary model and interactive visualization
python run_decoding_bandit.py --animal chimi --non_stationary --create_figurl

# Overwrite existing results
python run_decoding_bandit.py --animal chimi --overwrite

# Use specific GPUs
python run_decoding_bandit.py --animal chimi --cuda_visible_devices "0,1,2,3"

Workflow
--------
1. Parse command-line arguments to determine which sessions to process
2. Set up GPU cluster (LocalCUDACluster) for parallel processing
3. For each session:
   a. Load data (position, spikes, LFP, task events)
   b. Create environment (place bins, track graph)
   c. Prepare training data (exclude ripples or use all data)
   d. Fit detector model (NonLocalClusterlessDetector or NonLocalSortedSpikesDetector)
   e. Run decoding to get posterior distributions
   f. Compute derived measures (ahead/behind distance, HPD regions)
   g. Save results (NetCDF), models (pickle), and logs
   h. Optionally create interactive figurl visualization
4. Process multiple sessions in parallel using Dask delayed

Output Files
------------
Saved to PROCESSED_DATA_DIR (defined in parameters.py):

{epoch_identifier}_results.nc
    Decoding results as xarray Dataset (posterior distributions, state probabilities)
{epoch_identifier}_detector.pkl
    Fitted detector model (can be reloaded for analysis)
{epoch_identifier}_environment.pkl
    Environment configuration (place bins, track graph)
{epoch_identifier}_{threshold}_data_over_time.pkl
    Aggregated time-series data (decoding + behavioral variables)

Logs written to logs/{epoch_identifier}.log with git hash for reproducibility.

Configuration
-------------
- Detector class: Automatically selected based on brain_area (HPC uses clusterless,
  mPFC/OFC use sorted spikes)
- Place bin size: 2.0 cm for 1D, 3.5 cm for 2D
- Position uncertainty: sqrt(12.5) ≈ 3.536 cm
- Waveform uncertainty: 24.0 µV (clusterless only)
- GPU block size: 1024 for memory management
- Non-local threshold: 0.8 for event classification
- Training type: "no_ripple" (exclude ripples) or "all" (use all data)

GPU and Parallelism
-------------------
- Uses Dask CUDA for GPU cluster management
- XLA_PYTHON_CLIENT_MEM_FRACTION set to 0.99 for GPU memory
- Multiple sessions processed in parallel via dask.compute()
- Each session's decoding uses GPU acceleration (JAX)

Notes
-----
- All time units are in seconds
- Epoch identifiers: {animal}{date}_{epoch}_{brain_area} (e.g., "chimi20200212_02_r1_HPC")
- Results are skipped if files exist unless --overwrite is specified
- Empty spike data is handled gracefully (session skipped with warning)
- Git commit hash is logged for reproducibility

See Also
--------
continuum_swr_replay.data_loaders.bandit_task : Data loading
continuum_swr_replay.analysis.hippocampus : Result analysis and aggregation
non_local_detector : Core decoding algorithms

Examples
--------
# Decode all HPC data for one animal
python run_decoding_bandit.py --animal chimi --brain_area HPC

# Decode single session with figurl
python run_decoding_bandit.py --animal peanut --date 20201125 --epoch 02_r1 \\
    --brain_area HPC --create_figurl

# Non-stationary model with custom GPUs
python run_decoding_bandit.py --animal wilbur --non_stationary \\
    --cuda_visible_devices "2,3"
"""
import logging
import logging.config
import os
import subprocess
import warnings
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import dask
import numpy as np
import pandas as pd
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from non_local_detector import NonLocalClusterlessDetector, NonLocalSortedSpikesDetector
from non_local_detector.environment import Environment
from non_local_detector.models.non_local_model import (
    discrete_transition_type as stationary_discrete_transition_type,
)
from non_local_detector.models.non_local_model import (
    non_stationary_discrete_transition_type,
)

from continuum_swr_replay.analysis.hippocampus import get_data_over_time, load_results
from continuum_swr_replay.data_loaders import (
    get_epoch_info,
    get_training_timepoints,
    load_data,
)
from continuum_swr_replay.parameters import (
    BLOCK_SIZE,
    PLACE_BIN_SIZE_1D,
    PLACE_BIN_SIZE_2D,
    POSITION_STD,
    PROCESSED_DATA_DIR,
    WAVEFORM_STD,
)
from continuum_swr_replay.types import ClusterlessParams, SortedSpikesParams
from continuum_swr_replay.visualization.figurl import (
    create_interactive_1D_decoding_figurl,
)

if TYPE_CHECKING:
    import networkx as nx
    import xarray as xr

    from continuum_swr_replay.types import SessionData

warnings.filterwarnings("ignore")

# Detector configuration by brain area (maps to actual detector classes)
DETECTOR_CLASSES = {
    "HPC": (NonLocalClusterlessDetector, "clusterless"),
    "mPFC": (NonLocalSortedSpikesDetector, "sorted_spikes"),
    "OFC": (NonLocalSortedSpikesDetector, "sorted_spikes"),
}


@dataclass(frozen=True)
class DecodingConfig:
    """Configuration for neural decoding."""

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
    """
    Sets up a logger that logs to console and a file using declarative configuration.

    Parameters
    ----------
    name_logfile : str
        Logger name (and conceptual name of the logfile).
    path_logfile : str
        Path to the logfile. Parent dirs will be created if absent.
    level : int
        Logging level (default: INFO).
    include_warnings : bool
        Route Python warnings to logging (default: True).

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
    brain_area: str,
) -> bool:
    """Check if both results and detector files exist and we're not overwriting."""
    if overwrite:
        return False

    required_files = [
        save_folder / f"{epoch_identifier}_{brain_area}_results.nc",
        save_folder / f"{epoch_identifier}_{brain_area}_detector.pkl",
    ]
    return all(f.exists() for f in required_files)


def create_detector(
    brain_area: str,
    env: Environment,
    config: DecodingConfig,
    discrete_transition_type: str,
) -> NonLocalClusterlessDetector | NonLocalSortedSpikesDetector:
    """Factory for creating the appropriate detector for a brain area.

    Parameters
    ----------
    brain_area : str
        Brain area name (must be 'HPC', 'mPFC', or 'OFC')
    env : Environment
        Environment configuration
    config : DecodingConfig
        Decoding parameters
    discrete_transition_type : str
        Type of transition model

    Returns
    -------
    detector : NonLocalClusterlessDetector | NonLocalSortedSpikesDetector
        Configured detector instance

    Raises
    ------
    ValueError
        If brain_area is not recognized
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
    xr.Dataset, NonLocalClusterlessDetector | NonLocalSortedSpikesDetector, Environment
]:
    """Perform neural decoding for a given brain area.

    Parameters
    ----------
    logger : logging.Logger
    spike_times : list
        List of spike time arrays per unit/tetrode
    spike_waveform_features : Optional[list]
        Waveform features (HPC only)
    position_info : pd.DataFrame
        Position and movement data
    linear_edge_order : list
        Order of track graph edges
    linear_edge_spacing : list
        Spacing of track graph edges
    track_graph : networkx.Graph
        Track graph structure
    is_training : np.ndarray
        Boolean mask for training timepoints
    brain_area : str
        Brain area (HPC, mPFC, or OFC)
    non_stationary : bool
        Use speed-dependent transitions
    is_1d : bool
        Use 1D spatial binning

    Returns
    -------
    tuple
        (results, detector, env)
    """
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
            "speed": position_info["head_speed"].interpolate().to_numpy()
        }
        discrete_transition_type = non_stationary_discrete_transition_type
    else:
        discrete_transition_covariate_data = None
        discrete_transition_type = stationary_discrete_transition_type

    logger.info(f"Decoding {brain_area}...")

    detector = create_detector(brain_area, env, config, discrete_transition_type)

    # Prepare common parameters for all detectors
    estimate_params = {
        "position_time": position_info.index.to_numpy(),
        "position": position_info[["head_position_x", "head_position_y"]].to_numpy(),
        "spike_times": spike_times,
        "time": position_info.index.to_numpy(),
        "discrete_transition_covariate_data": discrete_transition_covariate_data,
    }

    # HPC-specific parameters: is_training and waveform features
    if brain_area == "HPC":
        estimate_params["is_training"] = is_training
        if spike_waveform_features is not None:
            estimate_params["spike_waveform_features"] = spike_waveform_features

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
) -> None:
    """Save decoding results, detector model, environment, and derived data."""
    logger.info("Saving results...")
    detector.save_results(
        results,
        save_folder / f"{epoch_identifier}_results.nc",
    )
    detector.save_model(save_folder / f"{epoch_identifier}_detector.pkl")
    env.save_environment(save_folder / f"{epoch_identifier}_environment.pkl")

    threshold_int = int(non_local_threshold * 100)
    data_over_time.to_pickle(
        save_folder / f"{epoch_identifier}_{threshold_int}_data_over_time.pkl"
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

    Parameters
    ----------
    data : dict
        Loaded session data
    results : xarray.Dataset
        Decoding results
    env : Environment
        Environment configuration
    nwb_file_name : str
        NWB file name
    epoch_name : str
        Epoch name
    brain_area : str
        Brain area
    non_local_threshold : float
        Threshold for non-local classification

    Returns
    -------
    tuple[pd.DataFrame, Any]
        (data_over_time, posterior)
    """
    sampling_frequency = 1 / np.median(np.diff(results.time.values))
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
) -> None:
    # Remove extension from nwb_file_name
    epoch_identifier = f"{nwb_file_name.replace('_.nwb', '')}_{epoch_name}_{brain_area}"
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
        logger.info(" START ".center(50, "#"))

        git_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            text=True,
        ).stdout
        logger.info(f"Git Hash: {git_hash.rstrip()}")
        logger.info(f"training type: {training_type}")
        logger.info(f"non_stationary: {non_stationary}")
        logger.info(f"non_local_threshold: {non_local_threshold}")
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

        # Check if brain area has spike data
        if brain_area not in data["spike_times"]:
            logger.warning(f"No spike data available for {brain_area}. Skipping...")
            return

        if results_exist(
            save_folder,
            epoch_identifier,
            overwrite,
            brain_area,
        ):
            logger.info(f"Results already exist for {brain_area}. Skipping decoding...")
            results, detector, env = load_results(nwb_file_name, epoch_name, brain_area)
        else:
            is_training = get_training_timepoints(data, training_type=training_type)

            # Get spike waveform features if available (HPC only)
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
            save_folder,
            epoch_identifier,
            non_local_threshold,
        )

        logger.info("Done!")
    except Exception as e:
        logger.exception(e)


def get_command_line_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--cuda_visible_devices", type=str, default="0,1,2,3,4,5,6,7,8,9"
    )
    parser.add_argument("--local_scratch_dir", type=str, default=".local_scratch")
    parser.add_argument("--brain_area", type=str, default="HPC")
    parser.add_argument("--training_type", type=str, default="no_ripple")
    parser.add_argument("--ripple_detector_name", type=str, default="Kay")
    parser.add_argument("--non_local_threshold", type=float, default=0.8)
    parser.add_argument("--non_stationary", action="store_true")
    parser.add_argument("--create_figurl", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--animal", type=str)
    parser.add_argument("--date", type=int)
    parser.add_argument("--epoch", type=str)

    return parser.parse_args()


def _filter_epoch_info(epoch_info: pd.DataFrame, args: Namespace) -> pd.DataFrame:
    """Filter the epoch_info DataFrame based on the command line arguments

    Parameters
    ----------
    epoch_info : pd.DataFrame
    args : Namespace

    Returns
    -------
    filtered_epoch_info : pd.DataFrame
    """
    # Initialize a boolean array to keep all rows initially
    is_epoch = np.ones(len(epoch_info), dtype=bool)

    # Apply the animal filter if specified
    if args.animal is not None:
        is_epoch = np.logical_and(epoch_info.animal == args.animal, is_epoch)

    # Apply the date filter if specified
    if args.date is not None:
        is_epoch = np.logical_and(epoch_info.date == args.date, is_epoch)

    # Apply the epoch filter if specified
    if args.epoch is not None:
        epoch = int(args.epoch.split("_")[0])
        is_epoch = np.logical_and(epoch_info.epoch == epoch, is_epoch)

    # Filter the DataFrame based on the combined conditions
    epoch_info = epoch_info.loc[is_epoch, :]

    # Ensure the result is a DataFrame, even if it contains only one row
    if isinstance(epoch_info, pd.Series):
        epoch_info = epoch_info.to_frame().T

    return epoch_info


def main() -> None:
    """Main entry point for batch decoding analysis."""
    args = get_command_line_arguments()

    Path(args.local_scratch_dir).mkdir(exist_ok=True, parents=True)

    cluster = LocalCUDACluster(
        CUDA_VISIBLE_DEVICES=args.cuda_visible_devices,
        local_directory=args.local_scratch_dir,
    )

    with Client(cluster):
        log_directory = Path.cwd() / "logs"
        log_directory.mkdir(exist_ok=True, parents=True)

        epoch_info = get_epoch_info()
        epoch_info = _filter_epoch_info(epoch_info, args)

        # Append the result of the computation into a results list
        results = [
            dask.delayed(run_decode)(
                nwb_file_name=epoch.Index[0],
                epoch_name=epoch.Index[1],
                log_directory=log_directory,
                overwrite=args.overwrite,
                create_figurl=args.create_figurl,
                training_type=args.training_type,
                brain_area=args.brain_area,
                non_local_threshold=args.non_local_threshold,
                non_stationary=args.non_stationary,
            )
            for epoch in epoch_info.itertuples()
        ]

        # Run `dask.compute` on the results list for the code to run
        dask.compute(*results)


if __name__ == "__main__":
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"
    main()
