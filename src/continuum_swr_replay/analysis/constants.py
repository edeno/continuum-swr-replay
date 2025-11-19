"""Analysis constants and default parameters.

This module defines constants used across analysis and figure generation scripts.
Follows the pattern established in continuum_swr_replay.parameters.

Usage
-----
Import constants as needed::

    from continuum_swr_replay.analysis.constants import (
        DEFAULT_NON_LOCAL_THRESHOLD,
        BINS_DURATION_MS,
    )

Notes
-----
- Color definitions are in continuum_swr_replay.parameters.COLORS
- NWB file lists and other data parameters are in continuum_swr_replay.parameters
- This module focuses on analysis-specific constants shared across figure scripts
"""

import numpy as np

# Event detection thresholds
DEFAULT_NON_LOCAL_THRESHOLD = 0.8
"""Default probability threshold for classifying non-local events (0-1)."""

DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC = 4.0
"""Default speed threshold for classifying immobile periods (cm/s)."""

DEFAULT_SAMPLING_FREQUENCY_HZ = 500.0
"""Default sampling frequency for neural data (Hz)."""

# Histogram bin definitions
# These are shared across multiple figure generation scripts to ensure consistency

BINS_DURATION_MS = np.arange(0, 250, 10)
"""Bin edges for event duration histograms (milliseconds)."""

BINS_PROBABILITY = np.arange(0, 1.05, 0.05)
"""Bin edges for probability histograms (0-1)."""

BINS_DISTANCE_CM = np.arange(0, 230, 2)
"""Bin edges for distance histograms (cm)."""

BINS_HPD_CM = np.arange(0, 250, 10)
"""Bin edges for highest posterior density (HPD) histograms (cm)."""

BINS_SPEED_CM_PER_SEC = np.arange(0, 20, 0.5)
"""Bin edges for speed histograms (cm/s)."""

# Event-triggered analysis
DEFAULT_WINDOW_EVENT_TRIGGERED_SEC = 0.25
"""Default window duration for event-triggered analysis (seconds).

This defines the total window duration centered on event times.
For example, 0.25 seconds means ±0.125 seconds around the event.
"""
