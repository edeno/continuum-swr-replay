"""Continuous-Fragmented classifier analysis module.

This module provides tools for analyzing results from the 2-state
Continuous-Fragmented (ContFrag) classifier, which decodes all time as
either smooth trajectory (Continuous state, RandomWalk transitions) or
disjointed spatial representation (Fragmented state, Uniform transitions).

Unlike the NonLocal 4-state detector, the ContFrag model has no Local or
No-Spike states. Events are defined externally via SWR or high-MUA times,
not by model-detected probability thresholding.
"""

from .data_aggregation import get_contfrag_data_over_time, load_contfrag_data_over_time
from .posterior_analysis import get_contfrag_posterior
from .result_loading import load_contfrag_results
from .tables import (
    build_contfrag_event_arm_table,
    build_contfrag_event_table,
    build_contfrag_stop_table,
)

__all__ = [
    "build_contfrag_event_arm_table",
    "build_contfrag_event_table",
    "build_contfrag_stop_table",
    "get_contfrag_data_over_time",
    "get_contfrag_posterior",
    "load_contfrag_data_over_time",
    "load_contfrag_results",
]
