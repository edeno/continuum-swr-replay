"""Analysis functions for manipulating and characterizing detected events.

This module complements events/nonlocal_detection.py (which detects events from signals)
by providing functions to analyze, filter, and characterize already-detected events.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
)
from continuum_swr_replay.analysis.events.immobility import (
    event_speeds_are_immobile,
    intervals_from_labels,
)

logger = logging.getLogger(__name__)


def get_immobile_events(
    data: pd.DataFrame,
    event_col: str,
    speed_col: str = "head_speed",
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    method: Literal["endpoints", "max"] = "max",
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
    method : {"endpoints", "max"}
        How to evaluate speed during event:
        - "endpoints": Both start and end must be below threshold
        - "max": Every speed during the event must be finite and below the
          threshold (default; canonical event-cohort rule)
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
    The default ``"max"`` method is the canonical cohort rule shared by
    Non-Local, SWR, and MUA events. The endpoints mode remains for the
    archived figure scripts and should not define model cohorts.

    Examples
    --------
    >>> immobile_replay = get_immobile_events(
    ...     data_over_time,
    ...     event_col="non_local_event",
    ...     speed_threshold=4.0,
    ...     method="max"
    ... )

    """
    # Set default group_cols
    if group_cols is None:
        group_cols = ["animal", "day", "epoch"]

    # Validate inputs
    if not np.isfinite(speed_threshold) or speed_threshold <= 0:
        raise ValueError(
            f"speed_threshold must be positive and finite, got {speed_threshold}"
        )
    if method not in ["endpoints", "max"]:
        raise ValueError(f"method must be 'endpoints' or 'max', got {method}")

    if event_col not in data.columns:
        raise ValueError(f"event_col '{event_col}' not found in data")
    if speed_col not in data.columns:
        raise ValueError(f"speed_col '{speed_col}' not found in data")

    def _filter_low_speed(group: pd.DataFrame) -> bool:
        """Filter events based on speed criteria."""
        if len(group) == 0:
            logger.warning("Empty group encountered in speed filtering")
            return False

        speed = group[speed_col].to_numpy(dtype=float)
        if not np.isfinite(speed).all():
            return False

        if method == "endpoints":
            return bool((speed[0] < speed_threshold) and (speed[-1] < speed_threshold))
        return event_speeds_are_immobile(speed, speed_threshold=speed_threshold)

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
        If event_col is not found in data, or a session's index is not its
        float, strictly increasing time axis
        (:func:`~continuum_swr_replay.analysis.events.immobility.require_time_index`).

    Notes
    -----
    - Duration is in seconds (end_time - start_time)
    - Events with ID 0 (non-events) are automatically excluded
    - Each session's bounds come from
      :func:`~continuum_swr_replay.analysis.events.immobility.intervals_from_labels`,
      read on the frame's time index
    - Rows are sorted by ``group_cols`` then event ID, with a fresh 0..n-1 index

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

    # One session at a time: ``intervals_from_labels`` reads the bounds off
    # the frame's time index, which is one epoch's time axis.
    columns = [*group_cols, event_col, "start_time", "end_time"]
    active = data.loc[data[event_col] > 0, [*group_cols, event_col]]
    sessions = [
        intervals_from_labels(session[[event_col]], event_col).assign(
            **dict(zip(group_cols, key, strict=True))
        )[columns]
        for key, session in active.groupby(group_cols, sort=True)
    ]
    event_times = (
        pd.concat(sessions, ignore_index=True)
        if sessions
        else pd.DataFrame(columns=columns)
    )
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
