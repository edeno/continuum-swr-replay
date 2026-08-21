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

from continuum_swr_replay.parameters import (
    MAX_SPEED_THRESHOLD_CM_PER_SEC,
    SAMPLING_FREQUENCY,
)

# Event detection thresholds
DEFAULT_NON_LOCAL_THRESHOLD = 0.8
"""Default probability threshold for classifying non-local events (0-1)."""

# Seconds after destination poke-in by which the absence of reward is
# established on an omission trial.
#
# The rewarded boundary is a stimulus: the pump fires a median 47 ms after
# poke-in (p95 59 ms, p99 80 ms, max 298 ms over 52,605 rewarded trials).
# Omissions have no negative-feedback cue to timestamp, and the previous rule
# used destination poke-out -- a median 880 ms later, and a *behavioural*
# boundary rather than a stimulus one. Gating on when the animal chose to
# leave is circular (dwell length plausibly covaries with replay) and it
# excluded 99% of omission events from every outcome-gated fit, collapsing
# the reward contrast those models exist to estimate.
#
# 0.30 s exceeds every reward latency observed in the corpus, so it cannot
# declare an omission while a pump could still fire, and it retains 99.6% of
# omission pokes with a median 0.58 s of usable dwell. Re-derive from the
# reward-latency distribution if the hardware or task timing changes.
OMISSION_KNOWN_LATENCY_SEC: float = 0.30
OUTCOME_TIME_SEMANTICS: str = (
    "earliest reward onset for rewarded trials; min(destination arrival + "
    f"{OMISSION_KNOWN_LATENCY_SEC:.2f} s, destination departure) for omissions"
)
"""Canonical physical availability rule for a trial's realized outcome."""

DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC: float = float(MAX_SPEED_THRESHOLD_CM_PER_SEC)
"""Default speed threshold for classifying immobile periods (cm/s).

Derived from :data:`continuum_swr_replay.parameters.MAX_SPEED_THRESHOLD_CM_PER_SEC`.
"""

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
