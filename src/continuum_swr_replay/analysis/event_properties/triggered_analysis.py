"""Event-triggered analysis: signal windows around events, and their averages."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import sem

from continuum_swr_replay.analysis.event_definition.analysed_time import (
    excluded_intervals,
)
from continuum_swr_replay.analysis.event_definition.swr_source import (
    swr_epoch_id_of_session,
)
from continuum_swr_replay.analysis.event_properties.event_analysis import (
    get_event_start_end_times,
)
from continuum_swr_replay.intervals import overlaps_intervals
from continuum_swr_replay.parameters import SAMPLING_FREQUENCY
from continuum_swr_replay.tables.columns import require_columns

logger = logging.getLogger(__name__)


def get_signal_windows(
    data: pd.DataFrame,
    event_times: pd.DataFrame,
    signal_cols: list[str],
    time_col: str = "start_time",
    window_sec: float = 2.0,
    sampling_frequency: float = SAMPLING_FREQUENCY,
    event_id_col: str | None = None,
) -> xr.DataArray:
    """Extract signal windows around event times.

    Parameters
    ----------
    data : pd.DataFrame
        Time-indexed DataFrame with signal columns and grouping columns
        Must have columns: animal, day, epoch, {signal_cols}
    event_times : pd.DataFrame
        DataFrame with event times and metadata
        Must have columns: animal, day, epoch, {time_col}
    signal_cols : list[str]
        Names of signal columns to extract from data
    time_col : str
        Column in event_times with event times (typically "start_time" or "end_time")
    window_sec : float
        Total window duration in seconds (centered on event time)
    sampling_frequency : float
        Sampling rate in Hz
    event_id_col : str | None
        Column name in event_times for event IDs (for coordinate labeling)
        If None, sequential integers are used

    Returns
    -------
    xr.DataArray
        Shape: (n_events, n_timepoints, n_signals)
        Dimensions: ("event", "time", "signal")
        Coordinates:
            - event: MultiIndex from event_times if event_id_col provided, else integers
            - time: Time relative to event in seconds (negative = before, positive = after)
            - signal: Signal names from signal_cols

    Raises
    ------
    ValueError
        If required columns are missing or parameters are invalid

    Notes
    -----
    - Excluded time (the artifact mask widened by 0.3 s) is not analysed: an
      event whose window touches the epoch's excluded time
      (``analysed_time.excluded_intervals``; closed) is dropped whole, and its row
      is not returned.
    - Windows are centered on event times
    - Missing data filled with NaN (occurs for events near data boundaries)
    - Events outside the data time range will have partial or fully NaN windows
    - Empty sessions (no events for a given animal/day/epoch) are logged and skipped
    - If ALL sessions have no events, returns an empty DataArray with shape (0, window_len, n_signals)
    - The empty DataArray maintains proper coordinates for time and signal dimensions

    Examples
    --------
    >>> signal_windows = get_signal_windows(
    ...     data_over_time,
    ...     non_local_times,
    ...     signal_cols=["theta_power_zscore", "SWR_zscore", "head_speed"],
    ...     time_col="start_time",
    ...     window_sec=2.0,
    ... )
    >>> # Average across events
    >>> mean_triggered = signal_windows.mean(dim="event")
    >>> # Plot
    >>> mean_triggered.sel(signal="SWR_zscore").plot()

    """
    # Validate inputs
    if sampling_frequency <= 0:
        raise ValueError(
            f"sampling_frequency must be positive, got {sampling_frequency}"
        )
    if window_sec <= 0:
        raise ValueError(f"window_sec must be positive, got {window_sec}")

    require_columns(data, ["animal", "day", "epoch", *signal_cols], "data")
    required_event_cols = ["animal", "day", "epoch", time_col]
    if event_id_col is not None:
        required_event_cols.append(event_id_col)
    require_columns(event_times, required_event_cols, "event_times")

    # Compute window parameters
    half_window_sec = window_sec / 2.0
    dt = 1 / sampling_frequency
    half_window_samples = int(round(half_window_sec / dt))
    window_len = 2 * half_window_samples + 1
    offsets = np.arange(-half_window_samples, half_window_samples + 1)

    all_windows = []
    event_id_list: list[
        tuple[Any, Any, Any, Any] | int
    ] = []  # Mixed type for flexibility

    for (animal, day, epoch), group in data.groupby(["animal", "day", "epoch"]):
        # Get events for this session
        session_events = event_times.loc[
            (event_times.animal == animal)
            & (event_times.day == day)
            & (event_times.epoch == epoch)
        ]

        if len(session_events) == 0:
            logger.debug(f"No events found for {animal}/{day}/{epoch}, skipping")
            continue

        excluded = excluded_intervals(swr_epoch_id_of_session(animal, day, epoch))
        if len(excluded):
            centers = session_events[time_col].to_numpy(dtype=float)
            touches = overlaps_intervals(
                centers - half_window_sec, centers + half_window_sec, excluded
            )
            if touches.any():
                logger.info(
                    "%d of %d windows for %s/%s/%s touch excluded time and are dropped",
                    int(touches.sum()),
                    len(touches),
                    animal,
                    day,
                    epoch,
                )
                session_events = session_events.loc[~touches]
            if len(session_events) == 0:
                continue

        times = session_events[time_col].to_numpy()
        time_index = group.index.values
        time_min, time_max = time_index[0], time_index[-1]

        # Find the *nearest* sample to each event time. ``searchsorted``
        # alone returns the insertion point (the next sample), which
        # for an event at 0.0025 s on a 2 ms grid centers the window
        # on 0.004 s instead of 0.002 s. Compare distances to the
        # neighbours bracketing the insertion point and pick the
        # closer one.
        right = np.searchsorted(time_index, times)
        right_clipped = np.clip(right, 0, len(time_index) - 1)
        left_clipped = np.clip(right - 1, 0, len(time_index) - 1)
        left_d = np.abs(times - time_index[left_clipped])
        right_d = np.abs(times - time_index[right_clipped])
        center_idxs = np.where(left_d <= right_d, left_clipped, right_clipped)

        # Check if events are within extractable range (accounting for window size)
        # Events exactly at boundaries or outside need special handling
        extractable = (times >= time_min + half_window_sec) & (
            times <= time_max - half_window_sec
        )

        # Warn about events with incomplete windows
        if (~extractable).any():
            n_partial = (~extractable).sum()
            logger.warning(
                f"{n_partial} events for {animal}/{day}/{epoch} "
                f"outside extractable range [{time_min + half_window_sec:.2f}, "
                f"{time_max - half_window_sec:.2f}], will have partial windows"
            )

        # Create window indices
        all_window_idxs = center_idxs[:, None] + offsets[None, :]

        # Check bounds - both index bounds AND temporal bounds
        # For events way outside the data range, even "valid" indices might be wrong
        mask = (all_window_idxs >= 0) & (all_window_idxs < len(group))

        # Also check if the actual time difference is reasonable
        # This catches events that are far outside the data range
        for event_idx in range(len(times)):
            center_time = time_index[center_idxs[event_idx]]
            event_time = times[event_idx]
            # If event is more than half window away from closest data point, mask all
            if abs(event_time - center_time) > half_window_sec:
                mask[event_idx, :] = False

        # Initialize windows with NaN
        n_events_session = len(times)
        n_signals = len(signal_cols)
        windows = np.full((n_events_session, window_len, n_signals), np.nan)

        # Fill valid data
        valid_rows, valid_cols = np.where(mask)
        valid_indices = all_window_idxs[mask]

        for i, signal_col in enumerate(signal_cols):
            windows[valid_rows, valid_cols, i] = group[signal_col].to_numpy()[
                valid_indices
            ]

        all_windows.append(windows)

        # Track event IDs
        if event_id_col is not None:
            # Store tuples for MultiIndex
            for _, row in session_events.iterrows():
                event_id_list.append(
                    (row["animal"], row["day"], row["epoch"], row[event_id_col])
                )
        else:
            # Store sequential integers - convert range to list for type compatibility
            event_id_list.extend(
                list(range(len(event_id_list), len(event_id_list) + n_events_session))
            )

    if not all_windows:
        # Return empty array with proper structure
        logger.warning("No events found in any session, returning empty array")
        return xr.DataArray(
            np.empty((0, window_len, len(signal_cols))),
            dims=("event", "time", "signal"),
            coords={
                "event": [],
                "time": offsets * dt,
                "signal": signal_cols,
            },
        )

    # Concatenate all sessions
    all_windows_arr = np.vstack(all_windows)
    time_axis = offsets * dt

    # Create proper index for events
    if event_id_col is not None:
        # Use pandas MultiIndex from tuples
        event_index = pd.MultiIndex.from_tuples(
            event_id_list, names=["animal", "day", "epoch", event_id_col]
        )
    else:
        # Use RangeIndex for consistency with project patterns (see event_definition/nonlocal_detection.py)
        event_index = pd.RangeIndex(start=0, stop=len(event_id_list), name="event")

    return xr.DataArray(
        all_windows_arr,
        dims=("event", "time", "signal"),
        coords={
            "event": event_index,
            "time": time_axis,
            "signal": signal_cols,
        },
    )


def compute_triggered_average_curves(
    data_over_time: pd.DataFrame,
    event_types: dict,
    signals: list[str],
    time_col: str = "start_time",
    window_sec: float = 2.0,
) -> dict:
    """Reduce ``data_over_time`` to event-triggered mean/SEM curves.

    Splits the work that :func:`plot_triggered_averages` used to do inline
    into a small, cacheable product: for each signal, the time axis and the
    across-event mean and SEM for non-SWR (no-overlap) vs SWR-coincident
    (overlap) immobile non-local events. The returned dict is a few KB,
    letting callers cache it instead of the multi-GB ``data_over_time``.

    Parameters
    ----------
    data_over_time : pd.DataFrame
        Aggregated per-timebin dataset.
    event_types : dict
        Output of :func:`get_event_types`.
    signals : list of str
        Signal columns to reduce, in order.
    time_col : str
        Alignment column ('start_time' or 'end_time').
    window_sec : float
        Half-window around each event, in seconds.

    Returns
    -------
    dict
        ``{signal: {"time", "no_overlap_mean", "no_overlap_sem",
        "overlap_mean", "overlap_sem"}}`` with numpy-array values.

    """
    signal_types = list(signals)
    no_overlap_times = get_event_start_end_times(
        event_types["non_local_event_immobile_no_overlap_SWR_event"],
        event_col="non_local_event",
    )
    overlap_times = get_event_start_end_times(
        event_types["non_local_event_immobile_overlap_SWR_event"],
        event_col="non_local_event",
    )
    windows_no = get_signal_windows(
        data=data_over_time,
        event_times=no_overlap_times,
        signal_cols=signal_types,
        time_col=time_col,
        window_sec=window_sec,
        sampling_frequency=SAMPLING_FREQUENCY,
        event_id_col="non_local_event",
    )
    windows_ov = get_signal_windows(
        data=data_over_time,
        event_times=overlap_times,
        signal_cols=signal_types,
        time_col=time_col,
        window_sec=window_sec,
        sampling_frequency=SAMPLING_FREQUENCY,
        event_id_col="non_local_event",
    )
    no_mean_all = windows_no.mean(dim="event")
    ov_mean_all = windows_ov.mean(dim="event")

    curves: dict = {}
    for signal_type in signal_types:
        no_mean = no_mean_all.sel(signal=signal_type)
        curves[signal_type] = {
            "time": no_mean["time"].values,
            "no_overlap_mean": no_mean.values,
            "no_overlap_sem": sem(
                windows_no.sel(signal=signal_type).values,
                axis=0,
                nan_policy="omit",
            ),
            "overlap_mean": ov_mean_all.sel(signal=signal_type).values,
            "overlap_sem": sem(
                windows_ov.sel(signal=signal_type).values,
                axis=0,
                nan_policy="omit",
            ),
        }
    return curves
