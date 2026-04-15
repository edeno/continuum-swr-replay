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
    prob : xr.DataArray, shape (n_time,)
        Probability of state over time.
    speed : np.ndarray, shape (n_time,), optional
        Speed of animal at each time point, by default None.
    time : np.ndarray, shape (n_time,), optional
        Time values in seconds, by default None.
    prob_threshold : float, optional
        Threshold for confident state classification, by default 0.8.
    minimum_duration : float, optional
        Minimum duration (seconds) for confident state, by default 0.015.

    Returns
    -------
    pd.DataFrame, shape (n_events, 4)
        Columns: ``start_time``, ``end_time``, ``duration``, ``max_speed``.
        Indexed by event_number (1-based).

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
        Non-local event intervals with ``start_time``, ``end_time``,
        ``duration``, ``max_speed`` columns. From ``get_confident_times``.
    ripple_times : pd.DataFrame
        Ripple event intervals with ``start_time`` and ``end_time`` columns.
    results : xr.Dataset
        Decoding results; only the ``time`` coordinate is used for
        computing temporal overlap via ``get_labels``.

    Returns
    -------
    overlap_non_local_times : pd.DataFrame
        Subset of ``non_local_times`` that overlap with at least one ripple.
        Same columns as input.
    no_overlap_non_local_times : pd.DataFrame
        Subset of ``non_local_times`` that do not overlap with any ripple.
        Same columns as input.

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


def classify_swr_overlap(
    non_local_events: pd.DataFrame,
    ripple_times: pd.DataFrame,
) -> pd.DataFrame:
    """Label each non-local event by its SWR-overlap geometry.

    Adds an ``overlap_class`` column taking one of three values:

    - ``"contained"`` : the non-local event is fully inside at least one
      SWR event (non-local start >= SWR start AND non-local end <=
      SWR end).
    - ``"extends"``   : the non-local event overlaps at least one SWR
      event but is not fully contained in any of them — it extends
      before or after the SWR window.
    - ``"none"``      : the non-local event does not overlap any SWR
      event.

    Parameters
    ----------
    non_local_events : pd.DataFrame
        Non-local event intervals with ``start_time`` and ``end_time``
        columns (as returned by :func:`get_confident_times`).
    ripple_times : pd.DataFrame
        SWR event intervals with ``start_time`` and ``end_time`` columns.

    Returns
    -------
    pd.DataFrame
        Copy of ``non_local_events`` with an added ``overlap_class``
        column.

    Notes
    -----
    "Contained" and "extends" are decided per non-local event. If a
    non-local event overlaps multiple SWRs, we label it "contained" as
    soon as any one SWR fully contains it; otherwise "extends".

    """
    classes: list[str] = []
    swr_starts = ripple_times["start_time"].to_numpy()
    swr_ends = ripple_times["end_time"].to_numpy()

    for ev in non_local_events.itertuples():
        # Any SWR that overlaps this non-local event at all
        overlap_mask = (swr_ends >= ev.start_time) & (
            swr_starts <= ev.end_time
        )
        if not overlap_mask.any():
            classes.append("none")
            continue
        # Any SWR that fully contains this non-local event?
        contained_mask = (
            overlap_mask
            & (swr_starts <= ev.start_time)
            & (swr_ends >= ev.end_time)
        )
        if contained_mask.any():
            classes.append("contained")
        else:
            classes.append("extends")

    out = non_local_events.copy()
    out["overlap_class"] = classes
    return out


def select_typical_event_by_class(
    classified_events: pd.DataFrame,
    non_local_prob: xr.DataArray,
    peak_threshold: float = 0.9,
) -> dict[str, pd.Series]:
    """Pick one deterministic "typical" event per SWR-overlap class.

    For each overlap class ("contained", "extends", "none"), selects the
    event closest to that class's median duration among events whose peak
    non-local probability meets ``peak_threshold``. If no events in a
    class meet the peak threshold, the rule falls back to the full class.

    Parameters
    ----------
    classified_events : pd.DataFrame
        Non-local event table with an ``overlap_class`` column (from
        :func:`classify_swr_overlap`).
    non_local_prob : xr.DataArray
        Combined non-local state probability time series, used to
        compute per-event peak probability.
    peak_threshold : float, default 0.9
        Minimum peak non-local probability for an event to count as
        "clean" for selection.

    Returns
    -------
    dict
        Mapping ``class -> pd.Series`` (one selected event per class).
        A class with no events is omitted from the dict.

    """
    time_vals = non_local_prob.time.to_numpy()
    prob_vals = non_local_prob.to_numpy().squeeze()

    def peak_prob(start: float, end: float) -> float:
        mask = (time_vals >= start) & (time_vals <= end)
        if not mask.any():
            return 0.0
        return float(prob_vals[mask].max())

    chosen: dict[str, pd.Series] = {}
    for cls in ("contained", "extends", "none"):
        subset = classified_events[
            classified_events["overlap_class"] == cls
        ]
        if len(subset) == 0:
            continue
        subset = subset.copy()
        subset["peak_prob"] = [
            peak_prob(ev.start_time, ev.end_time)
            for ev in subset.itertuples()
        ]
        clean = subset[subset["peak_prob"] >= peak_threshold]
        if len(clean) == 0:
            clean = subset
        median_duration = clean["duration"].median()
        idx = (clean["duration"] - median_duration).abs().idxmin()
        chosen[cls] = clean.loc[idx]
    return chosen
