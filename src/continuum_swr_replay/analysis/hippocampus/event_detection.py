"""Detection and analysis of neural events (ripples, non-local states).

This module provides functions for detecting confident neural events from
probability distributions and analyzing event overlap patterns.

Examples
--------
Detect non-local events from decoding results:

>>> from continuum_swr_replay.analysis.hippocampus import get_confident_times
>>> # Combine non-local state probabilities
>>> non_local_prob = (
...     results.acausal_state_probabilities.sel(states="Non-Local Continuous") +
...     results.acausal_state_probabilities.sel(states="Non-Local Fragmented")
... )
>>> # Detect confident events
>>> non_local_times = get_confident_times(
...     non_local_prob,
...     speed=position_info.head_speed.to_numpy(),
...     time=position_info.index.to_numpy(),
...     prob_threshold=0.8,
...     minimum_duration=0.015  # 15 ms minimum
... )
>>> print(f"Detected {len(non_local_times)} non-local events")

Find events that overlap with ripples:

>>> from continuum_swr_replay.analysis.hippocampus import get_overlap_times
>>> overlap_events, no_overlap_events = get_overlap_times(
...     non_local_times, ripple_times, results
... )
>>> print(f"Non-local events with ripples: {len(overlap_events)}")
>>> print(f"Non-local events without ripples: {len(no_overlap_events)}")
"""

import numpy as np
import pandas as pd
import xarray as xr
from ripple_detection.core import segment_boolean_series

from continuum_swr_replay.data_loaders import get_labels


def get_confident_times(
    prob: xr.DataArray,
    speed: np.ndarray | None = None,
    time: np.ndarray | None = None,
    prob_threshold: float = 0.8,
    minimum_duration: float = 0.015,
) -> pd.DataFrame:
    """Extract time intervals where probability exceeds threshold.

    Parameters
    ----------
    prob : xr.DataArray
        Probability of state over time
    speed : np.ndarray | None, optional
        Speed of animal at each time point, by default None
    time : np.ndarray | None, optional
        Time values corresponding to probability samples, by default None
    prob_threshold : float, optional
        Threshold for confident state classification, by default 0.8
    minimum_duration : float, optional
        Minimum duration (seconds) for confident state, by default 0.015

    Returns
    -------
    times : pd.DataFrame
        DataFrame with columns: start_time, end_time, duration, max_speed
        Indexed by event_number (1-based)
    """
    confident_times = np.asarray(
        segment_boolean_series(
            (prob > prob_threshold).to_pandas(),
            minimum_duration=minimum_duration,
        )
    )
    if len(confident_times) == 0:
        confident_times = np.full((0, 2), np.nan)

    # Calculate max speed for each confident time interval (vectorized index computation)
    if speed is not None and time is not None and len(confident_times) > 0:
        # Get bin indices for all intervals at once using searchsorted (vectorized)
        start_indices = np.searchsorted(time, confident_times[:, 0], side="left")
        end_indices = np.searchsorted(time, confident_times[:, 1], side="right")

        # Clip indices to valid range (vectorized)
        n_speed = len(speed)
        start_indices = np.clip(start_indices, 0, n_speed)
        end_indices = np.clip(end_indices, 0, n_speed)

        # Compute max speed per interval
        # Note: grouped max over non-contiguous intervals requires iteration,
        # but index computation above is fully vectorized
        max_speed = np.array(
            [
                np.nanmax(speed[start:end]) if start < end else np.nan
                for start, end in zip(start_indices, end_indices, strict=True)
            ]
        )
    else:
        max_speed = np.full(len(confident_times), np.nan)

    # Create 1-based index for event numbering
    n_times = len(confident_times)
    if n_times > 0:
        index = pd.RangeIndex(start=1, stop=n_times + 1, name="event_number")
    else:
        index = pd.Index([], name="event_number", dtype=int)

    return pd.DataFrame(
        {
            "start_time": confident_times[:, 0],
            "end_time": confident_times[:, 1],
            "duration": confident_times[:, 1] - confident_times[:, 0],
            "max_speed": max_speed,
        },
        index=index,
    )


def get_overlap_times(
    non_local_times: pd.DataFrame, ripple_times: pd.DataFrame, results: xr.Dataset
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find non-local events that overlap with ripple events.

    Parameters
    ----------
    non_local_times : pd.DataFrame
        Time intervals of non-local events
    ripple_times : pd.DataFrame
        Time intervals of ripple events
    results : xr.Dataset
        Decoding results containing time coordinate

    Returns
    -------
    overlap_non_local_times : pd.DataFrame
        Non-local events that overlap with ripples
    no_overlap_non_local_times : pd.DataFrame
        Non-local events that do not overlap with ripples
    """
    non_local_labels = get_labels(non_local_times, results.time)
    ripple_labels = get_labels(ripple_times, results.time)

    overlap_non_local = non_local_labels.loc[
        ripple_labels.event_number > 0
    ].event_number.unique()
    overlap_non_local = overlap_non_local[overlap_non_local > 0]

    non_local_ind = np.arange(len(non_local_times))

    overlap_non_local_times = non_local_times.iloc[
        non_local_ind[np.isin(non_local_ind, overlap_non_local - 1)]
    ]

    no_overlap_non_local_times = non_local_times.iloc[
        non_local_ind[~np.isin(non_local_ind, overlap_non_local - 1)]
    ]

    return overlap_non_local_times, no_overlap_non_local_times
