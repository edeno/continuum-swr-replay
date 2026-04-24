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

    - ``"swr_contained"`` : the non-local event is fully inside at
      least one SWR event (``event.start >= swr.start`` AND
      ``event.end <= swr.end`` for some SWR).
    - ``"swr_partial"``   : the non-local event overlaps at least one
      SWR event but is not fully inside any of them. Covers the
      sub-cases enumerated by
      :func:`classify_swr_overlap_geometry` — ``event_contains_swr``,
      ``single_swr_partial``, ``multi_swr_no_gap``, and
      ``multi_swr_gap``.
    - ``"non_swr"``       : the non-local event does not overlap any
      SWR event.

    These three classes correspond directly to the paper's
    SWR-coincident / non-SWR axis:
    ``SWR-coincident = swr_contained ∪ swr_partial``.

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
    Classification is per-event. If a non-local event overlaps multiple
    SWRs, it is labeled ``"swr_contained"`` as soon as any one SWR
    fully contains it; otherwise ``"swr_partial"``.

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
            classes.append("non_swr")
            continue
        # Any SWR that fully contains this non-local event?
        contained_mask = (
            overlap_mask
            & (swr_starts <= ev.start_time)
            & (swr_ends >= ev.end_time)
        )
        if contained_mask.any():
            classes.append("swr_contained")
        else:
            classes.append("swr_partial")

    out = non_local_events.copy()
    out["overlap_class"] = classes
    return out


def classify_swr_overlap_geometry(
    non_local_events: pd.DataFrame,
    ripple_times: pd.DataFrame,
) -> pd.DataFrame:
    """Fine-grained interval geometry between non-local events and SWRs.

    Extends :func:`classify_swr_overlap` to distinguish sub-cases of the
    overlapping-but-not-contained bucket. Useful for methods-section
    breakdowns and for auditing what the coarser three-way split
    aggregates over.

    Sub-class partition (mutually exclusive, collectively exhaustive):

    - ``"no_overlap"`` — no SWR intersects the event
    - ``"event_contains_swr"`` — the event strictly contains at least
      one SWR (event extends past both boundaries of some overlapping
      SWR)
    - ``"single_swr_partial"`` — the event overlaps exactly one SWR,
      and neither interval contains the other (partial overlap at one
      boundary)
    - ``"multi_swr_no_gap"`` — the event overlaps two or more SWRs and
      every timebin of the event is inside some SWR (tiled by a
      contiguous train of SWRs with no gaps). Expected to be very rare
      because SWR detectors typically enforce a minimum inter-event
      separation.
    - ``"multi_swr_gap"`` — the event overlaps two or more SWRs and at
      least one timebin of the event is outside every SWR (tiled with
      gap(s))
    - ``"swr_contains_event"`` — a single SWR fully contains the event

    The three coarse classes are (in terms of subclasses above):

    - ``"swr_contained"`` = ``swr_contains_event``
    - ``"swr_partial"`` = ``event_contains_swr`` ∪ ``single_swr_partial`` ∪
      ``multi_swr_no_gap`` ∪ ``multi_swr_gap``
    - ``"non_swr"`` = ``no_overlap``

    Display labels for figures / captions: ``"SWR-contained"``,
    ``"SWR-partial"``, ``"non-SWR"``. The binary axis used elsewhere in
    the paper is ``SWR-coincident = SWR-contained ∪ SWR-partial``.

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
        Copy of ``non_local_events`` with five added columns:

        - ``overlap_class`` : coarse label
          (``"non_swr"`` / ``"swr_partial"`` / ``"swr_contained"``)
        - ``overlap_subclass`` : fine label (one of the six strings
          listed above)
        - ``n_swrs_overlapping`` : int, count of SWRs intersecting the
          event
        - ``contains_any_swr`` : bool, True if the event strictly
          contains at least one SWR
        - ``event_has_non_swr_time`` : bool, True if any part of the
          event is outside every SWR

    Notes
    -----
    Boundary-equality conventions: ``"swr_contains_event"`` allows
    boundary equality (``SWR.start == event.start`` is still
    "contained"). ``"event_contains_swr"`` uses strict inequality so a
    boundary-matched SWR is not "strictly inside" the event.
    Containment is decided before the partial-overlap logic, so a
    ``"swr_contains_event"`` row never also appears in a
    partial-overlap subclass count.

    The ``event_has_non_swr_time`` flag is computed by clipping each
    overlapping SWR to the event window, merging the clipped intervals,
    and comparing merged duration to event duration (with 1 μs
    tolerance to absorb floating-point noise).

    """
    swr_starts = ripple_times["start_time"].to_numpy()
    swr_ends = ripple_times["end_time"].to_numpy()

    n_events = len(non_local_events)
    classes: list[str] = [""] * n_events
    subclasses: list[str] = [""] * n_events
    n_overlap_arr = np.zeros(n_events, dtype=int)
    contains_any_swr_arr = np.zeros(n_events, dtype=bool)
    event_has_non_swr_time_arr = np.zeros(n_events, dtype=bool)

    for i, ev in enumerate(non_local_events.itertuples()):
        overlap_mask = (swr_ends >= ev.start_time) & (swr_starts <= ev.end_time)
        n_overlap = int(overlap_mask.sum())
        n_overlap_arr[i] = n_overlap

        if n_overlap == 0:
            classes[i] = "non_swr"
            subclasses[i] = "no_overlap"
            event_has_non_swr_time_arr[i] = True
            continue

        overlapping_starts = swr_starts[overlap_mask]
        overlapping_ends = swr_ends[overlap_mask]

        # Any single SWR fully contains this event?
        containing_mask = (
            (overlapping_starts <= ev.start_time)
            & (overlapping_ends >= ev.end_time)
        )
        if containing_mask.any():
            classes[i] = "swr_contained"
            subclasses[i] = "swr_contains_event"
            # Event sits inside an SWR, so every timebin is SWR-covered.
            event_has_non_swr_time_arr[i] = False
            continue

        # Neither contained. Decide among the partial-overlap subclasses.
        strict_contained_mask = (
            (overlapping_starts > ev.start_time)
            & (overlapping_ends < ev.end_time)
        )
        contains_any_swr = bool(strict_contained_mask.any())
        contains_any_swr_arr[i] = contains_any_swr

        # Does the event have any timebins outside every SWR?
        # Clip each overlapping SWR to [ev.start, ev.end], merge, and
        # compare total merged duration to event duration.
        clipped_starts = np.maximum(overlapping_starts, ev.start_time)
        clipped_ends = np.minimum(overlapping_ends, ev.end_time)
        sort_idx = np.argsort(clipped_starts)
        sorted_starts = clipped_starts[sort_idx]
        sorted_ends = clipped_ends[sort_idx]
        merged_starts = [float(sorted_starts[0])]
        merged_ends = [float(sorted_ends[0])]
        for j in range(1, len(sorted_starts)):
            if sorted_starts[j] <= merged_ends[-1]:
                merged_ends[-1] = max(merged_ends[-1], float(sorted_ends[j]))
            else:
                merged_starts.append(float(sorted_starts[j]))
                merged_ends.append(float(sorted_ends[j]))
        union_duration = sum(
            end - start
            for start, end in zip(merged_starts, merged_ends, strict=True)
        )
        event_duration = ev.end_time - ev.start_time
        event_has_non_swr_time = union_duration < event_duration - 1e-6
        event_has_non_swr_time_arr[i] = event_has_non_swr_time

        classes[i] = "swr_partial"
        if contains_any_swr:
            subclasses[i] = "event_contains_swr"
        elif n_overlap == 1:
            subclasses[i] = "single_swr_partial"
        elif not event_has_non_swr_time:
            subclasses[i] = "multi_swr_no_gap"
        else:
            subclasses[i] = "multi_swr_gap"

    out = non_local_events.copy()
    out["overlap_class"] = classes
    out["overlap_subclass"] = subclasses
    out["n_swrs_overlapping"] = n_overlap_arr
    out["contains_any_swr"] = contains_any_swr_arr
    out["event_has_non_swr_time"] = event_has_non_swr_time_arr
    return out


def select_typical_event_by_class(
    classified_events: pd.DataFrame,
    non_local_prob: xr.DataArray,
    peak_threshold: float = 0.9,
) -> dict[str, pd.Series]:
    """Pick one deterministic "typical" event per SWR-overlap class.

    For each overlap class (``"swr_contained"``, ``"swr_partial"``,
    ``"non_swr"``), selects the event closest to that class's median
    duration among events whose peak non-local probability meets
    ``peak_threshold``. If no events in a class meet the peak threshold,
    the rule falls back to the full class.

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
    for cls in ("swr_contained", "swr_partial", "non_swr"):
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
