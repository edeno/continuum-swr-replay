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
event_detection
    Functions for detecting non-local replay events using probability thresholds.
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
get_confident_times
    Detect events where decoder confidence exceeds a threshold.
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

Note: public names are exposed through PEP 562 ``__getattr__`` so that
importing a leaf submodule
(e.g. ``continuum_swr_replay.analysis.hippocampus.event_analysis``) does
not eagerly drag in ``data_aggregation``, which itself imports
``continuum_swr_replay.data_loaders.get_labels`` and would open a
DataJoint connection at import time.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_LAZY_EXPORTS = {
    # data_aggregation — pulls in data_loaders.get_labels (Spyglass).
    "compute_swr_nl_event_counts": "data_aggregation",
    "compute_swr_overlap_class_counts": "data_aggregation",
    "get_data_over_time": "data_aggregation",
    "get_event_types": "data_aggregation",
    "load_data_over_time": "data_aggregation",
    # event_analysis — pure pandas/numpy.
    "get_event_durations_ms": "event_analysis",
    "get_event_start_end_times": "event_analysis",
    "get_immobile_events": "event_analysis",
    # event_detection.
    "classify_swr_overlap": "event_detection",
    "classify_swr_overlap_geometry": "event_detection",
    "get_confident_times": "event_detection",
    "get_overlap_times": "event_detection",
    "select_typical_event_by_class": "event_detection",
    # event_table — canonical per-non-local-event classification table.
    "build_classified_nonlocal_event_table": "event_table",
    "intervals_from_labels": "event_table",
    # hpc_raw_stats.
    "DISCRETE_STATE_ORDER": "hpc_raw_stats",
    "RAW_STAT_COLUMNS": "hpc_raw_stats",
    "TRANSITION_MATRIX_COLUMNS": "hpc_raw_stats",
    "apply_raw_stats_flags": "hpc_raw_stats",
    "check_transition_matrix_flags": "hpc_raw_stats",
    "compute_hpc_raw_stats_single": "hpc_raw_stats",
    "extract_discrete_transition_matrix": "hpc_raw_stats",
    # posterior_analysis.
    "get_conditional_non_local_posterior": "posterior_analysis",
    "get_map_speed": "posterior_analysis",
    "get_posterior_stats": "posterior_analysis",
    # quality_metrics.
    "compute_behavior_state_breakdown": "quality_metrics",
    "compute_bout_nonlocal_dominance": "quality_metrics",
    "compute_calibration_curve": "quality_metrics",
    "compute_error_by_track_segment": "quality_metrics",
    "compute_local_decoding_error": "quality_metrics",
    "compute_local_state_collapse_details": "quality_metrics",
    "compute_no_spike_firing_correspondence": "quality_metrics",
    "compute_non_local_during_movement": "quality_metrics",
    "compute_non_local_event_context": "quality_metrics",
    "compute_running_error_tail": "quality_metrics",
    "compute_session_quality_summary": "quality_metrics",
    "compute_state_fraction_during_movement": "quality_metrics",
    # result_loading.
    "load_results": "result_loading",
    # triggered_analysis.
    "get_signal_windows": "triggered_analysis",
}


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{target}")
    attr = getattr(module, name)
    globals()[name] = attr
    return attr


__all__ = sorted(_LAZY_EXPORTS.keys())


if TYPE_CHECKING:  # pragma: no cover - types only
    from .data_aggregation import (  # noqa: F401
        compute_swr_nl_event_counts,
        compute_swr_overlap_class_counts,
        get_data_over_time,
        get_event_types,
        load_data_over_time,
    )
    from .event_analysis import (  # noqa: F401
        get_event_durations_ms,
        get_event_start_end_times,
        get_immobile_events,
    )
    from .event_detection import (  # noqa: F401
        classify_swr_overlap,
        classify_swr_overlap_geometry,
        get_confident_times,
        get_overlap_times,
        select_typical_event_by_class,
    )
    from .event_table import (  # noqa: F401
        build_classified_nonlocal_event_table,
        intervals_from_labels,
    )
    from .hpc_raw_stats import (  # noqa: F401
        DISCRETE_STATE_ORDER,
        RAW_STAT_COLUMNS,
        TRANSITION_MATRIX_COLUMNS,
        apply_raw_stats_flags,
        check_transition_matrix_flags,
        compute_hpc_raw_stats_single,
        extract_discrete_transition_matrix,
    )
    from .posterior_analysis import (  # noqa: F401
        get_conditional_non_local_posterior,
        get_map_speed,
        get_posterior_stats,
    )
    from .quality_metrics import (  # noqa: F401
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
    from .result_loading import load_results  # noqa: F401
    from .triggered_analysis import get_signal_windows  # noqa: F401
