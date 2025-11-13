"""Continuum SWR Replay Analysis Package.

This package provides tools for analyzing neural replay during sharp-wave ripples
in a spatial bandit task. It includes data loading, decoding algorithms, and
visualization utilities.
"""

# Main data loading functions
from .data_loaders import (
    get_epoch_info,
    get_training_timepoints,
    load_data,
)

__version__ = "0.1.0"

__all__ = [
    "load_data",
    "get_epoch_info",
    "get_training_timepoints",
]
