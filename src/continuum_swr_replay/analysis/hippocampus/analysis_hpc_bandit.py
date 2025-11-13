"""Hippocampus analysis functions for spatial bandit task.

This module provides a unified interface for hippocampus decoding analysis.
Functions are organized into submodules but re-exported here for backward compatibility.
"""

# Re-export all functions from submodules for backward compatibility
from .data_aggregation import get_data_over_time
from .event_detection import get_confident_times, get_overlap_times
from .posterior_analysis import (
    get_conditional_non_local_posterior,
    get_map_speed,
    get_posterior_stats,
)
from .result_loading import load_results

__all__ = [
    # Result loading
    "load_results",
    # Event detection
    "get_confident_times",
    "get_overlap_times",
    # Posterior analysis
    "get_conditional_non_local_posterior",
    "get_map_speed",
    "get_posterior_stats",
    # Data aggregation
    "get_data_over_time",
]
