"""Constants the analyses share: windows, bin sizes, cohort and labeling settings.

What belongs here is a setting of an analysis, read by more than one analysis
package or figure. The decode's settings and the event thresholds
(``DEFAULT_NON_LOCAL_THRESHOLD``, ``DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC``) are
in ``continuum_swr_replay.parameters``; the task and recording facts (NWB file
lists, wells, patches, track segments) are in ``data_loaders/constants.py``.
Colors are in ``visualization.palettes.COLORS``.
"""

import numpy as np

from continuum_swr_replay.data_loaders.constants import OMISSION_KNOWN_LATENCY_SEC

# The omission latency itself is a loading-side task constant (see
# ``data_loaders.constants``); the semantics string that reports it is an
# analysis-side provenance field, so it stays here.
OUTCOME_TIME_SEMANTICS: str = (
    "earliest reward onset for rewarded trials; min(destination arrival + "
    f"{OMISSION_KNOWN_LATENCY_SEC:.2f} s, destination final poke-out) for omissions"
)
"""Canonical physical availability rule for a trial's realized outcome."""

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
(``analysis.event_evidence.spike_evidence``) and the event features'
evidence over remote bins (``detector_comparison.evidence``,
``scripts/detector_comparison/run_event_features.py``)."""
