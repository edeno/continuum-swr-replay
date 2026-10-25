"""Hippocampus-specific analysis functions for replay decoding.

This module provides analysis tools for processing hippocampal decoding results,
including event detection, posterior analysis, and data aggregation. It serves
as the main entry point for hippocampal replay analysis workflows.

Submodules
----------
data_aggregation
    Functions for aggregating decoding results across time and combining
    with behavioral data.
event_analysis
    Tools for analyzing individual replay events and their properties.
posterior_analysis
    Statistical analysis of decoded posterior distributions.
result_loading
    Loading and managing saved decoding results.
triggered_analysis
    Time-locked analysis of signals around events.

Common Functions
----------------
load_results
    Load saved decoding results from disk for further analysis.
get_data_over_time
    Aggregate decoding results into a time-indexed DataFrame.
get_immobile_events
    Find events occurring during animal immobility periods.
get_map_speed
    Compute the speed of the maximum a posteriori (MAP) trajectory,
    characterizing replay trajectory dynamics.
get_posterior_stats
    Calculate summary statistics from the posterior distribution, including mean
    position, variance, and highest posterior density regions.

Typical Workflow
----------------
1. Load decoding results: ``results, detector, env = load_results(nwb_file_name, epoch_name)``
2. Get aggregated data: ``data_df, posterior = get_data_over_time(data, results, env, nwb_file_name, epoch_name, non_local_threshold=0.80)``
3. Detect confident events: ``confident_times = get_confident_times(results, threshold)``
4. Analyze replay properties: ``speed = get_map_speed(results)``, ``stats = get_posterior_stats(...)``
5. Visualize or perform statistical tests on aggregated data

See Also
--------
continuum_swr_replay.data_loaders.bandit_task : Data loading for decoding
continuum_swr_replay.visualization : Plotting functions for replay analysis

Examples
--------
>>> from continuum_swr_replay.analysis.hippocampus import load_results, get_data_over_time
>>> results, detector, env = load_results("chimi20200212_.nwb", "02_r1")
>>> data_df, posterior = get_data_over_time(
...     data, results, env, "chimi20200212_.nwb", "02_r1",
...     non_local_threshold=0.80,
... )
"""

from .data_aggregation import (
    compute_swr_nl_event_counts,
    compute_swr_overlap_class_counts,
    get_data_over_time,
    get_event_types,
    load_data_over_time,
)
from .event_analysis import (
    get_event_durations_ms,
    get_event_start_end_times,
    get_immobile_events,
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
    "DISCRETE_STATE_ORDER",
    "RAW_STAT_COLUMNS",
    "TRANSITION_MATRIX_COLUMNS",
    "apply_raw_stats_flags",
    "check_transition_matrix_flags",
    "compute_behavior_state_breakdown",
    "compute_bout_nonlocal_dominance",
    "compute_calibration_curve",
    "compute_error_by_track_segment",
    "compute_hpc_raw_stats_single",
    "compute_local_decoding_error",
    "compute_local_state_collapse_details",
    "compute_no_spike_firing_correspondence",
    "compute_non_local_during_movement",
    "compute_non_local_event_context",
    "compute_running_error_tail",
    "compute_session_quality_summary",
    "compute_state_fraction_during_movement",
    "compute_swr_nl_event_counts",
    "compute_swr_overlap_class_counts",
    "extract_discrete_transition_matrix",
    "get_conditional_non_local_posterior",
    "get_data_over_time",
    "get_event_durations_ms",
    "get_event_start_end_times",
    "get_event_types",
    "get_immobile_events",
    "get_map_speed",
    "get_posterior_stats",
    "get_signal_windows",
    "load_data_over_time",
    "load_results",
]
