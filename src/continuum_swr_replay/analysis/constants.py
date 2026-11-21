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

# Well-visit context
IN_WELL_VISIT = "in_visit"
"""The ``visit_context`` of an event or stop wholly inside one well visit: from
the animal's first nose-poke in at a reward well to its final poke-out there
(``occupancy_state == "at_well"``). One of the two labels of
:data:`STARTS_IN_WELL_VISIT`, the analyses' subset."""

VISIT_CONTEXTS: tuple[str, ...] = (
    IN_WELL_VISIT,
    "spans_visit_end",
    "spans_visit_start",
    "after_visit_at_well",
    "before_visit_at_well",
    "at_other_well",
    "track_leaf_arm",
    "track_trunk_arm",
    "track_patch_junction",
    "track_center",
    "track_no_position",
    "before_first_visit",
    "no_analyzed_samples",
)
"""Every ``visit_context`` value, in the order a summary lists them.

Made by ``event_tables.visit_context.window_visit_context``:

- ``in_visit``: every sample is inside one well visit.
- ``spans_visit_end`` / ``spans_visit_start``: the window crosses a visit
  boundary; its first sample is inside a visit (it spans a final poke-out) or
  outside one (it spans a first poke-in).
- The rest lie wholly between two visits, placed by the animal's projected
  position at the window's middle sample: ``after_visit_at_well`` (within
  :data:`VISIT_WELL_RADIUS_CM` of the well of the visit just left),
  ``before_visit_at_well`` (of the well of the next visit), ``at_other_well``
  (of any other well); otherwise on the track: ``track_patch_junction`` or
  ``track_center`` (within :data:`VISIT_JUNCTION_RADIUS_CM` of that junction),
  else ``track_leaf_arm`` or ``track_trunk_arm`` by the observed track
  segment, or ``track_no_position`` when the segment is missing.
- ``before_first_visit``: before the epoch's first poke-in.
- ``no_analyzed_samples``: a stop wholly inside excluded time (no event is).
"""

STARTS_IN_WELL_VISIT: tuple[str, ...] = (IN_WELL_VISIT, "spans_visit_end")
"""The ``visit_context`` values of an event or stop that starts inside a well
visit: wholly inside it (``in_visit``), or starting inside it and ending after
its final poke-out (``spans_visit_end``). The one data subset of the analyses
and the default of ``event_definition.cohort_selection.select_visit_context``.

An event belongs to the time when it starts. Keeping only ``in_visit`` would
drop the events that start just before the final poke-out and run past it, so
the rate models would undercount the event starts at the end of a visit (a
counting artifact), and the rate and the content models must analyze the same
events."""

VISIT_WELL_RADIUS_CM = 10.0
"""How near (cm, projected position to the well node) the animal must be to be
next to a well when it is outside a visit (``visit_context``)."""

VISIT_JUNCTION_RADIUS_CM = 10.0
"""How near (cm, projected position to the junction node) the animal must be to
be at a patch junction or the center when it is outside a visit and not next to
a well (``visit_context``)."""
