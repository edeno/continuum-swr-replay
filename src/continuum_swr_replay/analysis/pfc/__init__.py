"""Prefrontal cortex (mPFC/OFC) neural activity analysis.

This module provides tools for analyzing neural activity from prefrontal cortex
regions (medial PFC and orbitofrontal cortex) using generalized linear models (GLM).
The analysis focuses on relating single-unit spiking activity to task variables,
behavioral states, and hippocampal replay events.

Primary use case is modeling PFC unit activity during immobility periods to identify
neurons that modulate their firing during hippocampal replay events, particularly
during sharp-wave ripples (SWR).

Main Functions
--------------
build_session_event_sets
    Build the four event sets (SWR, replay, and the two replay/SWR classes) a
    session's design matrices are keyed on, restricted to the shared stop cohort.
build_session_fit_mask
    The time bins one session's GLMs are fit on: immobility, inside the cohort.
make_matched_control_times
    Draw the duration-matched non-event control windows the engagement test
    compares replay against, from those same fitted bins. Returns a
    ``MatchedControlDraw``: the windows, and the two counts of events that
    could not be given one.
process_neuron
    Fit GLM to relate single neuron activity to task/behavioral covariates.
    Primary entry point for single-unit analysis.

Internal Functions
------------------
_make_design_matrices
    Create design matrices for GLM from task variables and behavioral data.
_fit_immobility_models
    Fit GLM models specifically during immobility periods.

Analysis Workflow
-----------------
1. Load neural data (sorted units from mPFC/OFC)
2. Bin spike times at desired temporal resolution
   (``continuum_swr_replay.intervals.get_spikecount_per_time_bin``)
3. Create design matrices with task variables as predictors
4. Fit GLMs to identify neurons modulated by:
   - Hippocampal replay events
   - Sharp-wave ripples
   - Task variables (position, reward, etc.)
   - Behavioral state (movement vs. immobility)
5. Evaluate model fit and extract coefficients

See Also
--------
continuum_swr_replay.analysis.pfc.glm : GLM implementation details
continuum_swr_replay.data_loaders.spikes : Spike data loading for sorted units
continuum_swr_replay.analysis.hippocampus : Hippocampal replay analysis

Examples
--------
>>> from continuum_swr_replay.analysis.pfc import process_neuron
>>> # Fit GLM for a single neuron
>>> results = process_neuron(spike_times, design_matrix, time_bins)

"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

# Public names are exposed through PEP 562 ``__getattr__`` so that importing a
# leaf submodule (e.g. ``continuum_swr_replay.analysis.pfc.cohort``) does not
# run ``glm``, which imports dask and statsmodels at module top level and,
# through the replay and hippocampus packages, ``non_local_detector`` and JAX.

#: Exported name -> (submodule of this package, attribute it is bound to).
_LAZY_EXPORTS = {
    "MatchedControlDraw": ("glm", "MatchedControlDraw"),
    "build_session_event_sets": ("glm", "build_session_event_sets"),
    "build_session_fit_mask": ("glm", "build_session_fit_mask"),
    "fit_immobility_models": ("glm", "_fit_immobility_models"),
    "make_design_matrices": ("glm", "_make_design_matrices"),
    "make_matched_control_times": ("glm", "make_matched_control_times"),
    "matched_control_seed": ("glm", "matched_control_seed"),
    "process_neuron": ("glm", "process_neuron"),
    "run_session": ("glm", "run_session"),
}


def __getattr__(name: str):
    """Import an exported name from its submodule on first attribute access."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    module = importlib.import_module(f"{__name__}.{module_name}")
    attr = getattr(module, attribute)
    globals()[name] = attr
    return attr


__all__ = [
    "MatchedControlDraw",
    "make_design_matrices",
    "fit_immobility_models",
    "build_session_event_sets",
    "build_session_fit_mask",
    "make_matched_control_times",
    "matched_control_seed",
    "process_neuron",
    "run_session",
]


if TYPE_CHECKING:  # pragma: no cover - types only
    from .glm import (  # noqa: F401
        MatchedControlDraw,
        build_session_event_sets,
        build_session_fit_mask,
        make_matched_control_times,
        matched_control_seed,
        process_neuron,
        run_session,
    )
    from .glm import _fit_immobility_models as fit_immobility_models  # noqa: F401
    from .glm import _make_design_matrices as make_design_matrices  # noqa: F401
