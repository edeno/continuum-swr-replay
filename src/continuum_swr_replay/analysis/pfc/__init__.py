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
get_spikecount_per_time_bin
    Convert spike times to binned spike counts for GLM fitting.

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

from continuum_swr_replay.analysis.pfc.glm import (
    _fit_immobility_models as fit_immobility_models,
)
from continuum_swr_replay.analysis.pfc.glm import (
    _make_design_matrices as make_design_matrices,
)
from continuum_swr_replay.analysis.pfc.glm import (
    MatchedControlDraw,
    build_session_event_sets,
    build_session_fit_mask,
    get_spikecount_per_time_bin,
    make_matched_control_times,
    matched_control_seed,
    process_neuron,
    run_session,
)

__all__ = [
    "MatchedControlDraw",
    "make_design_matrices",
    "fit_immobility_models",
    "build_session_event_sets",
    "build_session_fit_mask",
    "get_spikecount_per_time_bin",
    "make_matched_control_times",
    "matched_control_seed",
    "process_neuron",
    "run_session",
]
