"""Analysis functions for manipulating and characterizing detected events.

This module complements event_detection.py (which detects events from signals)
by providing functions to analyze, filter, and characterize already-detected events.
"""

from __future__ import annotations

import logging
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)


def get_immobile_events(
    data: pd.DataFrame,
    event_col: str,
    speed_col: str = "head_speed",
    speed_threshold: float = 4.0,
    method: Literal["endpoints", "max", "mean"] = "endpoints",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Filter events to only include those during low-speed periods.

    Parameters
    ----------
    data : pd.DataFrame
        Time-indexed data with event labels and speed
    event_col : str
        Column containing event labels (0 = no event, 1+ = event ID)
    speed_col : str
        Column containing speed values (cm/s)
    speed_threshold : float
        Maximum speed threshold (cm/s). Events with speed exactly equal to
        threshold are considered mobile (uses strict < comparison).
    method : {"endpoints", "max", "mean"}
        How to evaluate speed during event:
        - "endpoints": Both start and end must be below threshold (default, most restrictive)
        - "max": Maximum speed during event must be below threshold
        - "mean": Mean speed during event must be below threshold
    group_cols : list[str] | None
        Grouping columns (default: ["animal", "day", "epoch"])

    Returns
    -------
    pd.DataFrame
        Filtered data containing only low-speed events

    Raises
    ------
    ValueError
        If speed_threshold is not positive, method is invalid, or required columns are missing

    Notes
    -----
    The "endpoints" method matches existing patterns in generate_figure02.py
    and is appropriate for ensuring immobility at event boundaries (relevant
    for replay analysis where animal should be still).

    Examples
    --------
    >>> immobile_replay = get_immobile_events(
    ...     data_over_time,
    ...     event_col="non_local_event",
    ...     speed_threshold=4.0,
    ...     method="endpoints"
    ... )

    """
    # Set default group_cols
    if group_cols is None:
        group_cols = ["animal", "day", "epoch"]

    # Validate inputs
    if speed_threshold <= 0:
        raise ValueError(f"speed_threshold must be positive, got {speed_threshold}")
    if method not in ["endpoints", "max", "mean"]:
        raise ValueError(f"method must be 'endpoints', 'max', or 'mean', got {method}")

    if event_col not in data.columns:
        raise ValueError(f"event_col '{event_col}' not found in data")
    if speed_col not in data.columns:
        raise ValueError(f"speed_col '{speed_col}' not found in data")

    def _filter_low_speed(group: pd.DataFrame) -> bool:
        """Filter events based on speed criteria."""
        if len(group) == 0:
            logger.warning("Empty group encountered in speed filtering")
            return False

        if len(group) == 1:
            # Single timepoint: treat as immobile if below threshold
            return bool(group[speed_col].iloc[0] < speed_threshold)

        if method == "endpoints":
            return bool(
                (group[speed_col].iloc[0] < speed_threshold)
                and (group[speed_col].iloc[-1] < speed_threshold)
            )
        elif method == "max":
            return bool(group[speed_col].max() < speed_threshold)
        elif method == "mean":
            return bool(group[speed_col].mean() < speed_threshold)
        return False

    return (
        data.loc[data[event_col] > 0]
        .groupby(group_cols + [event_col])
        .filter(_filter_low_speed)
    )


def get_event_start_end_times(
    data: pd.DataFrame,
    event_col: str,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Extract start and end times for labeled events.

    Converts event labels (0 = no event, 1+ = event ID) to interval format
    with start_time, end_time columns. This is the inverse operation of
    get_labels() from data_loaders.

    Parameters
    ----------
    data : pd.DataFrame
        Time-indexed DataFrame with event labels
    event_col : str
        Column containing event IDs (0 = no event, 1+ = event number)
    group_cols : list[str] | None
        Grouping columns (default: ["animal", "day", "epoch"])

    Returns
    -------
    pd.DataFrame
        Columns: [*group_cols, event_col, start_time, end_time, duration]
        One row per unique event (event_col > 0)

    Raises
    ------
    ValueError
        If event_col is not found in data

    Notes
    -----
    - Duration is in seconds (end_time - start_time)
    - Events with ID 0 (non-events) are automatically excluded
    - Time values come from DataFrame index

    Examples
    --------
    >>> event_times = get_event_start_end_times(
    ...     data_over_time,
    ...     event_col="non_local_event"
    ... )
    >>> print(event_times.head())
       animal       day epoch  non_local_event  start_time  end_time  duration
    0  chimi  20200212  02_r1                1     1234.56   1234.62      0.06

    """
    # Set default group_cols
    if group_cols is None:
        group_cols = ["animal", "day", "epoch"]

    if event_col not in data.columns:
        raise ValueError(f"event_col '{event_col}' not found in data")

    # Extract start/end times for each event. Use ``names="time"`` so the
    # reset column is always called ``"time"`` regardless of whether the
    # input index was unnamed (default ``"index"``) or already named
    # something like ``"time"`` — which is the case for time-indexed
    # ``data_over_time`` frames in real callers.
    event_times = (
        data.reset_index(names="time")
        .groupby(group_cols + [event_col])
        .agg(start_time=("time", "min"), end_time=("time", "max"))
        .reset_index()
    )

    # Remove non-events (event_col == 0)
    event_times = event_times[event_times[event_col] > 0]

    # Compute duration
    event_times["duration"] = event_times["end_time"] - event_times["start_time"]

    return event_times


def get_event_durations_ms(event_times: pd.DataFrame) -> pd.Series:
    """Convert event durations from seconds to milliseconds.

    This is a convenience wrapper for event times from get_confident_times()
    or get_event_start_end_times().

    Parameters
    ----------
    event_times : pd.DataFrame
        Event times with 'duration' column in seconds

    Returns
    -------
    pd.Series
        Event durations in milliseconds

    Raises
    ------
    ValueError
        If 'duration' column is not found in event_times

    Notes
    -----
    For events from get_confident_times(), the duration column is already
    computed accurately as (end_time - start_time). This is more accurate
    than count-based duration estimation.

    Examples
    --------
    >>> event_times = get_event_start_end_times(data, "non_local_event")
    >>> durations_ms = get_event_durations_ms(event_times)
    >>> print(f"Mean duration: {durations_ms.mean():.1f} ms")
    Mean duration: 85.3 ms

    """
    if "duration" not in event_times.columns:
        raise ValueError("event_times must have 'duration' column")

    return event_times["duration"] * 1000.0
