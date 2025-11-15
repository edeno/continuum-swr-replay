"""Hippocampus analysis module for continuum replay during SWR."""

from .data_aggregation import get_data_over_time
from .event_detection import get_confident_times, get_overlap_times
from .posterior_analysis import (
    get_conditional_non_local_posterior,
    get_map_speed,
    get_posterior_stats,
)
from .result_loading import load_results

__all__ = [
    "load_results",
    "get_confident_times",
    "get_overlap_times",
    "get_conditional_non_local_posterior",
    "get_map_speed",
    "get_posterior_stats",
    "get_data_over_time",
]
