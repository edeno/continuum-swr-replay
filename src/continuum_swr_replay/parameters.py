from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Final

import numpy as np

# Decoding sampling frequency in samples per second
SAMPLING_FREQUENCY = 500

# Maximum speed threshold for filtering non-local events (cm/s)
# Events with speeds above this are likely not genuine replay events
MAX_SPEED_THRESHOLD_CM_PER_SEC = 4.0

# Data directories and definitions
ROOT_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = ROOT_DIR / "Processed-Data"
FIGURE_DIR = ROOT_DIR / "figures"

# Spatial binning parameters (in cm)
PLACE_BIN_SIZE_1D: Final[float] = 2.0
PLACE_BIN_SIZE_2D: Final[float] = 3.5

# Decoding model parameters
POSITION_STD: Final[float] = float(np.sqrt(12.5))  # ~3.536 cm spatial uncertainty
WAVEFORM_STD: Final[float] = 24.0  # Waveform feature space standard deviation (µV)
BLOCK_SIZE: Final[int] = 2**10  # GPU processing block size (1024)

# Detector configuration by brain area
# Maps brain area to (detector_class_name, detector_type_string)
# Actual detector classes must be imported from non_local_detector in analysis scripts
DETECTOR_CONFIG: Final[dict[str, tuple[str, str]]] = {
    "HPC": ("NonLocalClusterlessDetector", "clusterless"),
    "mPFC": ("NonLocalSortedSpikesDetector", "sorted_spikes"),
    "OFC": ("NonLocalSortedSpikesDetector", "sorted_spikes"),
}


@dataclass(frozen=True)
class DecodingParameters:
    """Parameters for neural decoding analysis.

    Attributes
    ----------
    place_bin_size : float
        Spatial bin size in cm for position discretization. Smaller bins
        provide higher spatial resolution but require more computation.
        Default is 2.0 cm for 1D decoding.
    waveform_std : float
        Standard deviation for waveform feature likelihood in µV. Controls
        how tightly spikes must match the waveform features to contribute
        to the posterior. Default is 24.0 µV.
    position_std : float
        Standard deviation for position likelihood in cm. Derived from
        sqrt(12.5) ≈ 3.536 cm, which corresponds to uniform spatial uncertainty.
    block_size : int
        Block size for GPU memory management during decoding. Controls
        how many time bins are processed simultaneously. Must be a power
        of 2 for optimal GPU performance. Default is 1024 (2^10).

    Notes
    -----
    These parameters control the non-local detector's decoding algorithm.
    The geometric mean of state duration is 1 / (1 - p), where:
    - p = 1 - (1 / n_time_steps)
    - n_time_steps should equal half a theta cycle
    - Theta cycles are ~8 Hz or 125 ms per cycle
    - Half a theta cycle is 62.5 ms
    - If timestep is 2 ms, then n_time_steps = 31.25
    - So p = 0.968

    The dataclass is frozen (immutable) to prevent accidental modification
    of these critical analysis parameters.
    """

    place_bin_size: float = PLACE_BIN_SIZE_1D
    waveform_std: float = WAVEFORM_STD
    position_std: float = POSITION_STD
    block_size: int = BLOCK_SIZE


# Singleton instance of decoding parameters
DECODING_PARAMS: Final = DecodingParameters()

PROBABILITY_THRESHOLDS = [0.50, 0.80, 0.95]


class TrainingType(str, Enum):
    """Types of training data selection for decoder.

    Attributes
    ----------
    ALL : str
        Use all time points for training (no exclusions)
    NO_RIPPLE : str
        Exclude ripple times from training data

    Notes
    -----
    Inherits from str to allow direct string comparisons and JSON serialization.
    Use .value to get the underlying string value.

    Examples
    --------
    >>> TrainingType.ALL
    <TrainingType.ALL: 'all'>
    >>> TrainingType.ALL.value
    'all'
    >>> TrainingType.ALL == "all"
    True
    """

    ALL = "all"
    NO_RIPPLE = "no_ripple"

    def __str__(self) -> str:
        """Return string value for easy formatting."""
        return self.value

COLORS = {
    "SWR": "#d95f02",
    "MUA": "#7570b3",
    "Non-Local": "#1b9e77",
    "Local": "#add8e6",  # Soft Blue for Local
    "No-Spike": "#b0b0b0",  # Light Beige for No-Spike
}

__all__ = [
    # Sampling and thresholds
    "SAMPLING_FREQUENCY",
    "MAX_SPEED_THRESHOLD_CM_PER_SEC",
    "PROBABILITY_THRESHOLDS",
    # Directories
    "ROOT_DIR",
    "PROCESSED_DATA_DIR",
    "FIGURE_DIR",
    # Spatial binning parameters
    "PLACE_BIN_SIZE_1D",
    "PLACE_BIN_SIZE_2D",
    # Decoding model parameters
    "POSITION_STD",
    "WAVEFORM_STD",
    "BLOCK_SIZE",
    "DETECTOR_CONFIG",
    # Decoding parameters (dataclass)
    "DecodingParameters",
    "DECODING_PARAMS",
    # Enums
    "TrainingType",
    # Color schemes
    "COLORS",
]
