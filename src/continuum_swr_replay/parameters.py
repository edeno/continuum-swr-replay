from dataclasses import dataclass
from pathlib import Path
from typing import Final

# Decoding sampling frequency in samples per second
SAMPLING_FREQUENCY = 500

# Maximum speed threshold for filtering non-local events (cm/s)
# Events with speeds above this are likely not genuine replay events
MAX_SPEED_THRESHOLD_CM_PER_SEC = 4.0

# Data directories and definitions
ROOT_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = ROOT_DIR / "Processed-Data"
FIGURE_DIR = ROOT_DIR / "figures"


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
        place_bin_size (sqrt(place_bin_size^2 / 12)) for uniform bins.
        Default is 6.0 cm (~sqrt(12.5)).
    block_size : int
        Block size for GPU memory management during decoding. Controls
        how many time bins are processed simultaneously. Must be a power
        of 2 for optimal GPU performance. Default is 4096 (2^12).

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

    place_bin_size: float = 2.0
    waveform_std: float = 24.0
    position_std: float = 6.0
    block_size: int = 4096


# Singleton instance of decoding parameters
DECODING_PARAMS: Final = DecodingParameters()

PROBABILITY_THRESHOLDS = [0.50, 0.80, 0.95]

training_types = [
    "all",
    "no_ripple",
]

COLORS = {
    "SWR": "#d95f02",
    "MUA": "#7570b3",
    "Non-Local": "#1b9e77",
    "Local": "#add8e6",  # Soft Blue for Local
    "No-Spike": "#b0b0b0",  # Light Beige for No-Spike
}
