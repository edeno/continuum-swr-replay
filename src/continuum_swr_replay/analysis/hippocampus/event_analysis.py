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


def compute_event_timing_offsets(
    data_over_time: pd.DataFrame,
    reference_event_col: str = "SWR_event",
    target_event_col: str = "non_local_event",
) -> pd.DataFrame:
    """Per-overlap time offsets between two event types.

    For each (reference, target) event pair that shares at least one
    timepoint, computes

        start_diff = target_start - reference_start
        end_diff   = target_end   - reference_end

    Positive ``start_diff`` means the target starts later than the
    reference; negative ``end_diff`` means the target ends earlier.
    For Figure 2, the canonical call is

        offsets = compute_event_timing_offsets(
            data_over_time, "SWR_event", "non_local_event",
        )

    which answers "when, relative to the SWR window, do non-local
    events start and end?" A distribution centered at positive
    ``start_diff`` and negative ``end_diff`` means non-local events
    typically sit inside the SWR window.

    Parameters
    ----------
    data_over_time : pd.DataFrame
        Time-indexed aggregated data with ``animal``, ``day``,
        ``epoch``, and the two event columns.
    reference_event_col : str, default "SWR_event"
        Event column whose boundaries are treated as the reference (t=0).
    target_event_col : str, default "non_local_event"
        Event column whose boundaries are compared to the reference.

    Returns
    -------
    pd.DataFrame
        One row per overlapping (reference_id, target_id) pair with
        columns ``animal``, ``day``, ``epoch``, the two event id
        columns, and ``start_diff`` / ``end_diff`` (seconds).

    Notes
    -----
    If a single reference event overlaps with multiple target events,
    the result includes one row per (reference, target) pair. Same if
    a target event overlaps multiple references. Duplicates are not
    collapsed.

    """
    group_cols = ["animal", "day", "epoch"]
    out_cols = [
        *group_cols,
        reference_event_col,
        target_event_col,
        "start_diff",
        "end_diff",
    ]

    # Precompute per-event start and end times for both event types
    def event_bounds(df: pd.DataFrame, event_col: str) -> pd.DataFrame:
        active = df[df[event_col] > 0]
        if len(active) == 0:
            return pd.DataFrame(
                columns=[*group_cols, event_col, "start_time", "end_time"],
            )
        # Use the time-index of the aggregated data as the canonical time
        bounds = (
            active.reset_index(names="time")
            .groupby([*group_cols, event_col])
            .agg(start_time=("time", "min"), end_time=("time", "max"))
            .reset_index()
        )
        return bounds

    ref_bounds = event_bounds(data_over_time, reference_event_col)
    tgt_bounds = event_bounds(data_over_time, target_event_col)

    # For each reference event, find all target events that share any
    # timepoint during it. We do this by iterating sessions because
    # the per-session event counts are small enough that a vectorised
    # cross-join within each session is cheap.
    rows: list[dict] = []
    key_cols = group_cols
    for key, ref_session in ref_bounds.groupby(key_cols, sort=False):
        tgt_session = tgt_bounds[
            (tgt_bounds[key_cols[0]] == key[0])
            & (tgt_bounds[key_cols[1]] == key[1])
            & (tgt_bounds[key_cols[2]] == key[2])
        ]
        if len(tgt_session) == 0:
            continue
        # Broadcast overlap check: (ref_end >= tgt_start) AND (ref_start <= tgt_end)
        ref_starts = ref_session["start_time"].to_numpy()
        ref_ends = ref_session["end_time"].to_numpy()
        tgt_starts = tgt_session["start_time"].to_numpy()
        tgt_ends = tgt_session["end_time"].to_numpy()

        # Strict positive-duration overlap (matches
        # ``classify_swr_overlap`` / ``get_overlap_times``). A
        # boundary-touch pair shares zero duration and would generate
        # a phantom timing-offset row at start_diff/end_diff = 0; the
        # previous inclusive ``>=`` / ``<=`` form admitted those rows.
        overlap = (
            (ref_ends[:, None] > tgt_starts[None, :])
            & (ref_starts[:, None] < tgt_ends[None, :])
        )
        ref_idx, tgt_idx = overlap.nonzero()
        if ref_idx.size == 0:
            continue

        ref_ids = ref_session[reference_event_col].to_numpy()
        tgt_ids = tgt_session[target_event_col].to_numpy()
        for ri, ti in zip(ref_idx, tgt_idx, strict=True):
            rows.append(
                {
                    group_cols[0]: key[0],
                    group_cols[1]: key[1],
                    group_cols[2]: key[2],
                    reference_event_col: ref_ids[ri],
                    target_event_col: tgt_ids[ti],
                    "start_diff": tgt_starts[ti] - ref_starts[ri],
                    "end_diff": tgt_ends[ti] - ref_ends[ri],
                }
            )

    # Always return a frame with the expected schema, even when no
    # event pairs overlap. Plot helpers (e.g.
    # ``plot_event_timing_offsets``) index ``offsets["start_diff"]``
    # directly and would raise KeyError on a bare empty frame.
    if not rows:
        return pd.DataFrame(columns=out_cols)
    return pd.DataFrame(rows)
