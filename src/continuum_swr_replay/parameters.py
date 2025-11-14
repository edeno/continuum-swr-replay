from pathlib import Path

# Decoding sampling frequency in samples per second
SAMPLING_FREQUENCY = 500

# Maximum speed threshold for filtering non-local events (cm/s)
# Events with speeds above this are likely not genuine replay events
MAX_SPEED_THRESHOLD_CM_PER_SEC = 4.0

# Data directories and definitions
ROOT_DIR = Path(__file__).parent.parent
PROCESSED_DATA_DIR = ROOT_DIR / "Processed-Data"
FIGURE_DIR = ROOT_DIR / "figures"


"""
1. Geometric mean of duration is 1 / (1 - p)
2. So p = 1 - (1 / n_time_steps).
3. Want `n_time_steps` to equal half a theta cycle.
4. Theta cycles are ~8 Hz or 125 ms per cycle.
5. Half a theta cycle is 62.5 ms.
6. If our timestep is 2 ms, then n_time_steps = 31.25
7. So p = 0.968
"""

decoding_parameters = {
    "place_bin_size": 2.0,
    "waveform_std": 24.0,
    "position_std": 6.0,
    "block_size": int(2**12),
}

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
