"""Detection and analysis of neural events (ripples, non-local states)."""


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

    # Calculate max speed for each confident time interval
    if speed is not None and time is not None:
        max_speed = np.array(
            [
                np.nanmax(speed[slice(*np.digitize(time_range, time[1:-1]))])
                for time_range in confident_times
            ]
        )
    else:
        max_speed = np.full(len(confident_times), np.nan)

    n_times = len(confident_times)
    index_array: np.ndarray | list[int] = np.arange(n_times) + 1 if n_times > 0 else []
    index = pd.Index(index_array, name="event_number")
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
