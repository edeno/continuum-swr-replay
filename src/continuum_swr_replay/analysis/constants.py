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
- Color definitions are in continuum_swr_replay.visualization.palettes.COLORS
- NWB file lists and other data parameters are in continuum_swr_replay.parameters
- This module focuses on analysis-specific constants shared across figure scripts

"""

import numpy as np

from continuum_swr_replay.data_loaders.constants import OMISSION_KNOWN_LATENCY_SEC
from continuum_swr_replay.parameters import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC as DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
)
from continuum_swr_replay.parameters import (
    DEFAULT_NON_LOCAL_THRESHOLD as DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.parameters import (
    SAMPLING_FREQUENCY,
)

# Event detection thresholds: DEFAULT_NON_LOCAL_THRESHOLD and
# DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC (the whole-event immobility speed
# threshold, cm/s) are the :mod:`continuum_swr_replay.parameters` values,
# re-exported here for the analysis code that reads them from this module.

# The omission latency itself is a loading-side task constant (see
# ``data_loaders.constants``); the semantics string that reports it is an
# analysis-side provenance field, so it stays here.
OUTCOME_TIME_SEMANTICS: str = (
    "earliest reward onset for rewarded trials; min(destination arrival + "
    f"{OMISSION_KNOWN_LATENCY_SEC:.2f} s, destination final poke-out) for omissions"
)
"""Canonical physical availability rule for a trial's realized outcome."""

# SAMPLING_FREQUENCY is int in parameters.py; cast to float for use in
# arithmetic that expects float division (e.g., duration / sampling_frequency).
DEFAULT_SAMPLING_FREQUENCY_HZ: float = float(SAMPLING_FREQUENCY)
"""Default sampling frequency for neural data (Hz).

Derived from :data:`continuum_swr_replay.parameters.SAMPLING_FREQUENCY`.
"""

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
"""Total window duration for event-triggered analysis (seconds).

The window is centered on the event time, so a value of 0.25 spans
from -0.125 s to +0.125 s relative to the event.
"""

# Spike evidence
REMOTE_CM = 100.0
"""How far (cm along the track graph) a place must be from the animal's to be
remote: the spike-evidence analysis's "remote" event subset
(``scripts/detection/analyze_event_spike_evidence.py``) and the event features'
evidence over remote bins (``inventory_comparison.evidence``,
``scripts/detection/run_event_features.py``)."""
