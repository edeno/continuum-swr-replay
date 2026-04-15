"""Hippocampus analysis module for continuum replay during sharp-wave ripples.

This module provides tools for analyzing hippocampal replay events during sharp-wave
ripples (SWR), focusing on continuum (non-local) decoding of spatial position from
clusterless neural activity. It includes functions for loading decoding results,
computing posterior statistics, detecting confident non-local events, and aggregating
data for visualization and statistical analysis.

The analysis pipeline supports identifying when hippocampal activity represents
locations distant from the animal's current position (replay), computing trajectory
statistics (speed, direction), and relating replay events to task variables.

Main Functions
--------------
load_results
    Load saved decoding results, detector models, and environment configurations
    from NetCDF and pickle files. Primary entry point for accessing analysis outputs.
get_data_over_time
    Aggregate decoding results and task variables into time-resolved DataFrames
    for analysis and visualization. Computes derived measures like ahead/behind
    distance and highest posterior density (HPD) regions.

Event Detection
---------------
get_confident_times
    Identify time periods where posterior confidence exceeds a threshold, indicating
    reliable position decoding. Used to filter for high-quality replay events.
get_overlap_times
    Find temporal overlap between different event types (e.g., ripples and confident
    decoding) using interval logic.

Event Analysis
--------------
get_immobile_events
    Filter events to only include those during low-speed periods, ensuring the
    animal is immobile (relevant for replay analysis).
get_event_start_end_times
    Convert event labels to start/end time intervals with durations.
get_event_durations_ms
    Convert event durations from seconds to milliseconds for reporting.

Event-Triggered Analysis
-------------------------
get_signal_windows
    Extract signal windows around event times for peri-event time histogram (PETH)
    analysis. Returns xarray DataArray with dimensions (event, time, signal).

Posterior Analysis
------------------
get_conditional_non_local_posterior
    Extract posterior probability conditioned on being in the non-local state,
    isolating replay-related spatial representations.
get_map_speed
    Compute the speed of the decoded position (MAP estimate) over time, useful for
    characterizing replay trajectory dynamics.
get_posterior_stats
    Calculate summary statistics from the posterior distribution, including mean
    position, variance, and highest posterior density regions.

Typical Workflow
----------------
1. Load decoding results: results, detector, env = load_results(epoch_id)
2. Get aggregated data: data_df = get_data_over_time(results, data, detector, ...)
3. Detect confident events: confident_times = get_confident_times(results, threshold)
4. Analyze replay properties: speed = get_map_speed(results), stats = get_posterior_stats(results)
5. Visualize or perform statistical tests on aggregated data

See Also
--------
continuum_swr_replay.data_loaders.bandit_task : Data loading for decoding
continuum_swr_replay.visualization : Plotting functions for replay analysis

Examples
--------
>>> from continuum_swr_replay.analysis.hippocampus import load_results, get_data_over_time
>>> results, detector, env = load_results("chimi20200212_02_r1_HPC")
>>> data_df = get_data_over_time(results, data, detector, env, threshold=0.80)

"""

from .data_aggregation import (
    compute_swr_nl_event_counts,
    compute_swr_overlap_class_counts,
    get_data_over_time,
    get_event_types,
    load_data_over_time,
)
from .event_analysis import (
    compute_event_timing_offsets,
    get_event_durations_ms,
    get_event_start_end_times,
    get_immobile_events,
)
from .event_detection import (
    classify_swr_overlap,
    get_confident_times,
    get_overlap_times,
    select_typical_event_by_class,
)
from .hpc_raw_stats import (
    DISCRETE_STATE_ORDER,
    RAW_STAT_COLUMNS,
    TRANSITION_MATRIX_COLUMNS,
    apply_raw_stats_flags,
    check_transition_matrix_flags,
    compute_hpc_raw_stats_single,
    extract_discrete_transition_matrix,
)
from .posterior_analysis import (
    get_conditional_non_local_posterior,
    get_map_speed,
    get_posterior_stats,
)
from .quality_metrics import (
    compute_behavior_state_breakdown,
    compute_bout_nonlocal_dominance,
    compute_calibration_curve,
    compute_error_by_track_segment,
    compute_local_decoding_error,
    compute_local_state_collapse_details,
    compute_no_spike_firing_correspondence,
    compute_non_local_during_movement,
    compute_non_local_event_context,
    compute_running_error_tail,
    compute_session_quality_summary,
    compute_state_fraction_during_movement,
)
from .result_loading import load_results
from .triggered_analysis import get_signal_windows

__all__ = [
    "load_results",
    "get_confident_times",
    "get_overlap_times",
    "classify_swr_overlap",
    "select_typical_event_by_class",
    "get_immobile_events",
    "get_event_start_end_times",
    "get_event_durations_ms",
    "compute_event_timing_offsets",
    "get_signal_windows",
    "get_conditional_non_local_posterior",
    "get_map_speed",
    "get_posterior_stats",
    "get_data_over_time",
    "load_data_over_time",
    "get_event_types",
    "compute_swr_overlap_class_counts",
    "compute_swr_nl_event_counts",
    "compute_behavior_state_breakdown",
    "compute_bout_nonlocal_dominance",
    "compute_calibration_curve",
    "compute_error_by_track_segment",
    "compute_local_decoding_error",
    "compute_local_state_collapse_details",
    "compute_no_spike_firing_correspondence",
    "compute_non_local_during_movement",
    "compute_non_local_event_context",
    "compute_running_error_tail",
    "compute_session_quality_summary",
    "compute_state_fraction_during_movement",
    "DISCRETE_STATE_ORDER",
    "RAW_STAT_COLUMNS",
    "TRANSITION_MATRIX_COLUMNS",
    "apply_raw_stats_flags",
    "check_transition_matrix_flags",
    "compute_hpc_raw_stats_single",
    "extract_discrete_transition_matrix",
]
