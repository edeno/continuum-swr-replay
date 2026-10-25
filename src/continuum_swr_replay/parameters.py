"""Global parameters and configuration for neural decoding analysis.

This module provides centralized configuration for the continuum-swr-replay
project, including spatial binning parameters, decoding model parameters,
file paths, and analysis constants.

Constants
---------
SAMPLING_FREQUENCY : int
    Decoding sampling frequency (500 Hz). All temporal analyses use this rate.
DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC : float
    Whole-event speed threshold (4.0 cm/s) shared by Non-Local, SWR, and MUA
    analysis cohorts. Every event sample must be finite and strictly below it.
ROOT_DIR : Path
    Root directory of the project (parent of src/).
PROCESSED_DATA_DIR : Path
    The processed-data root, kept for notebooks and exploratory scripts:
    ``paths.processed_data_dir()``, resolved at each attribute access (``from
    ... import PROCESSED_DATA_DIR`` binds it at that import). Library code and
    scripts call ``paths.processed_data_dir()``.
FIGURE_DIR : Path
    Directory for saving generated figures.
PLACE_BIN_SIZE_1D : float
    Spatial bin size for 1D decoding (2.0 cm).
PLACE_BIN_SIZE_2D : float
    Spatial bin size for 2D decoding (3.5 cm).
POSITION_STD : float
    Spatial uncertainty standard deviation (~3.536 cm, derived from sqrt(12.5)).
WAVEFORM_STD : float
    Waveform feature space standard deviation (24.0 µV) for clusterless decoding.
BLOCK_SIZE : int
    GPU processing block size (1024) for memory management.
LOCAL_POSITION_STD : float
    Gaussian kernel standard deviation (2.5 cm) for the non-local detector's
    Local-state spatial representation. ``> 0`` gives a multi-bin Local state.
PROBABILITY_THRESHOLDS : list
    Probability thresholds for posterior analysis [0.50, 0.80, 0.95].
DEFAULT_NON_LOCAL_THRESHOLD : float
    Default probability threshold (0.8) for classifying non-local events; the
    canonical decode's ``data_over_time`` file is named for it.

Notes
-----
All constants use type hints with Final[] to indicate they should not be
modified.

See Also
--------
continuum_swr_replay.data_types : TypedDict definitions for data structures
continuum_swr_replay.data_loaders.constants : Task-specific constants
continuum_swr_replay.visualization.palettes : COLORS, the event-type colours
continuum_swr_replay.decoding.config : the detector class decoding each brain area
continuum_swr_replay.decoding.identifiers : TrainingType and the decode identifier

Examples
--------
>>> from continuum_swr_replay.parameters import PLACE_BIN_SIZE_1D
>>> PLACE_BIN_SIZE_1D
2.0

"""

from pathlib import Path
from typing import Final

import numpy as np

from continuum_swr_replay.paths import ROOT_DIR, processed_data_dir

# Decoding sampling frequency in samples per second
SAMPLING_FREQUENCY = 500

# Canonical whole-event immobility threshold for Non-Local, SWR, and MUA
# analysis events (cm/s). Every event-bin speed must be strictly below it.
DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC = 4.0

# Data directories and definitions. ROOT_DIR, the project root
# (continuum-swr-replay/, not src/), is defined in paths beside the data roots.
FIGURE_DIR = ROOT_DIR / "paper" / "figures"

# Multiunit High Synchrony Event (HSE) detection parameters
HSE_MINIMUM_DURATION_SEC: Final[float] = 0.015  # 15 ms minimum event duration
HSE_ZSCORE_THRESHOLD: Final[float] = 2.0  # Z-score threshold for event detection
HSE_CLOSE_EVENT_THRESHOLD_SEC: Final[float] = 0.0  # Gap to merge adjacent events (0 = no merging)

# Spatial binning parameters (in cm)
PLACE_BIN_SIZE_1D: Final[float] = 2.0
PLACE_BIN_SIZE_2D: Final[float] = 3.5

# Decoding model parameters
POSITION_STD: Final[float] = float(np.sqrt(12.5))  # ~3.536 cm spatial uncertainty
WAVEFORM_STD: Final[float] = 24.0  # Waveform feature space standard deviation (µV)
BLOCK_SIZE: Final[int] = 2**10  # GPU processing block size (1024)
# Local-state spatial representation: Gaussian kernel std (cm) applied over
# shortest-path track-graph distance, giving the Local state a multi-bin
# (rather than single-bin) position representation in the non-local detector.
LOCAL_POSITION_STD: Final[float] = 2.5

PROBABILITY_THRESHOLDS: Final[list[float]] = [0.50, 0.80, 0.95]

DEFAULT_NON_LOCAL_THRESHOLD: Final[float] = 0.8
"""Default probability threshold for classifying non-local events (0-1)."""


def __getattr__(name: str) -> Path:
    # PROCESSED_DATA_DIR is resolved when read, not bound at import.
    if name == "PROCESSED_DATA_DIR":
        return processed_data_dir()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Sampling and thresholds
    "SAMPLING_FREQUENCY",
    "DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC",
    "PROBABILITY_THRESHOLDS",
    "DEFAULT_NON_LOCAL_THRESHOLD",
    # HSE detection parameters
    "HSE_MINIMUM_DURATION_SEC",
    "HSE_ZSCORE_THRESHOLD",
    "HSE_CLOSE_EVENT_THRESHOLD_SEC",
    # Directories
    "ROOT_DIR",
    "PROCESSED_DATA_DIR",  # noqa: F822 -- resolved by __getattr__
    "FIGURE_DIR",
    # Spatial binning parameters
    "PLACE_BIN_SIZE_1D",
    "PLACE_BIN_SIZE_2D",
    # Decoding model parameters
    "POSITION_STD",
    "WAVEFORM_STD",
    "BLOCK_SIZE",
    "LOCAL_POSITION_STD",
]
