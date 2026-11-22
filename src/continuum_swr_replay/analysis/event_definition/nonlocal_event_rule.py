"""The non-local event rule and the SWR-overlap classification of those events.

``get_confident_times`` finds the intervals where a decoded state's probability
stays above a threshold, and ``detect_nonlocal_events`` is the whole rule: those
intervals, kept only when the animal is immobile throughout. The rest of the
module classifies how each event overlaps the SWRs. It detects no ripples
itself. The detector that applies the rule to a decode is
``detection.nonlocal_events``.

Examples
--------
Apply the rule to decoding results:

>>> from continuum_swr_replay.analysis.event_definition.nonlocal_event_rule import get_confident_times
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
...     minimum_duration=0.015  # 7 samples on the decode grid (see below)
... )
>>> print(f"Detected {len(non_local_times)} non-local events")

Find events that overlap with ripples:

>>> from continuum_swr_replay.analysis.event_definition.nonlocal_event_rule import get_overlap_times
>>> overlap_events, no_overlap_events = get_overlap_times(
...     non_local_times, ripple_times
... )
>>> print(f"Non-local events with ripples: {len(overlap_events)}")
>>> print(f"Non-local events without ripples: {len(no_overlap_events)}")

"""

import numpy as np
import pandas as pd
import xarray as xr
from ripple_detection.core import segment_boolean_series

from continuum_swr_replay.analysis.event_definition.immobility import (
    build_speed_eligible_event_cohort,
    intervals_from_labels,
)
from continuum_swr_replay.intervals import total_duration
from continuum_swr_replay.parameters import (
    DEFAULT_NON_LOCAL_THRESHOLD,
    NON_LOCAL_MINIMUM_DURATION_S,
)
from continuum_swr_replay.tables.columns import require_columns

# Interval-arithmetic tolerance for floating-point noise (1 μs). Used in
# classify_swr_overlap_geometry when comparing an event's duration to
# the total duration of its SWR coverage.
INTERVAL_TOLERANCE_SEC = 1e-6

NON_LOCAL_STATES = ("Non-Local Continuous", "Non-Local Fragmented")
"""The decoder states whose probabilities sum to the non-local probability."""


def nonlocal_probability(frame: pd.DataFrame) -> pd.Series:
    """Each row's non-local probability: Non-Local Continuous plus Fragmented.

    Parameters
    ----------
    frame : pd.DataFrame
        Rows of a decode frame, with ``Non-Local_Continuous_prob`` and
        ``Non-Local_Fragmented_prob``.

    Returns
    -------
    pd.Series, shape (n_rows,)
        On ``frame``'s index, in the columns' dtype; NaN where either is NaN.
    """
    return frame["Non-Local_Continuous_prob"] + frame["Non-Local_Fragmented_prob"]


def nonlocal_state_probability(
    acausal_state_probabilities: xr.DataArray,
) -> xr.DataArray:
    """A decode's non-local probability: Non-Local Continuous plus Fragmented.

    The same sum as :func:`nonlocal_probability`, on the decoder's own output
    (``results.acausal_state_probabilities``), before it is written to a frame.

    Parameters
    ----------
    acausal_state_probabilities : xr.DataArray, shape (n_time, n_states)
        With a ``states`` coordinate naming ``Non-Local Continuous`` and
        ``Non-Local Fragmented``.

    Returns
    -------
    xr.DataArray, shape (n_time,)
        NaN where either state's probability is NaN.
    """
    continuous, fragmented = NON_LOCAL_STATES
    return acausal_state_probabilities.sel(
        states=continuous
    ) + acausal_state_probabilities.sel(states=fragmented)


def get_confident_times(
    prob: xr.DataArray,
    speed: np.ndarray | None = None,
    time: np.ndarray | None = None,
    prob_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
    minimum_duration: float = NON_LOCAL_MINIMUM_DURATION_S,
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
        Threshold for confident state classification (strictly above), by
        default ``DEFAULT_NON_LOCAL_THRESHOLD`` (0.8).
    minimum_duration : float, optional
        Minimum duration (seconds) for confident state, by default
        ``NON_LOCAL_MINIMUM_DURATION_S`` (0.015).
        Converted to a sample count, not compared with a timestamp span; see
        Notes.

    Returns
    -------
    pd.DataFrame, shape (n_events, 4)
        Columns: ``start_time``, ``end_time``, ``duration``,
        ``animal_max_speed_cm_s``. Indexed by event_number (1-based).
        ``start_time`` / ``end_time`` are the first and last above-threshold
        samples, so ``duration`` is the first-to-last span.

    Notes
    -----
    The run-length test is ``ripple_detection.core.segment_boolean_series``: a
    run qualifies when it holds at least
    ``minimum_sample_count(time, minimum_duration)`` consecutive samples, i.e.
    ``floor(minimum_duration / median_step + 0.5 + 1e-6)`` (round half up).
    On the 500 Hz decode grid the median step reads 2.0000935 ms (the
    timestamps are quantized at a ~1.6e9 s clock), so 0.015 s is 7.49965
    samples and rounds to 7, not 8. The effective minimum is therefore 7
    samples: a 12 ms ``duration`` (first-to-last span). See
    ``docs/known_issues/nonlocal_minimum_duration_rounding.md``.

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
            "animal_max_speed_cm_s": max_speed,
        },
        index=index,
    )


def detect_nonlocal_events(
    non_local_probability: xr.DataArray,
    speed: np.ndarray,
    time: pd.Index | np.ndarray,
    prob_threshold: float = DEFAULT_NON_LOCAL_THRESHOLD,
    minimum_duration: float = NON_LOCAL_MINIMUM_DURATION_S,
) -> tuple[pd.DataFrame, pd.Series]:
    """The non-local events of a decode: the whole rule, in one place.

    An event is a run of non-local probability above ``prob_threshold`` that
    lasts ``minimum_duration`` (:func:`get_confident_times`) during
    which the animal is immobile throughout: every sampled speed finite and below
    the immobility threshold (``immobility.build_speed_eligible_event_cohort``).
    A run is kept or dropped whole, never clipped.

    Parameters
    ----------
    non_local_probability : xr.DataArray, shape (n_time,)
        P(Non-Local Continuous) + P(Non-Local Fragmented)
        (:func:`nonlocal_state_probability`), one value per row of ``time``.
    speed : np.ndarray, shape (n_time,)
        The animal's speed at each row of ``time``, in cm/s.
    time : pd.Index or np.ndarray, shape (n_time,)
        The time of each row, in seconds. The probability is read on this axis,
        whatever its own ``time`` coordinate holds, and the events are labeled
        on it.
    prob_threshold : float, optional
        The probability an event stays strictly above, by default
        ``DEFAULT_NON_LOCAL_THRESHOLD``.
    minimum_duration : float, optional
        How long a run must last (s, converted to a sample count as
        :func:`get_confident_times` does), by default
        ``NON_LOCAL_MINIMUM_DURATION_S``. Only the event rule's sensitivity
        check (``event_properties.rule_sweep``) gives another.

    Returns
    -------
    events : pd.DataFrame
        One row per event: ``non_local_event`` (its label), ``start_time`` and
        ``end_time`` (its first and last sample) and ``duration``.
    labels : pd.Series, shape (n_time,)
        The event label of each row of ``time``, 0 outside events, named
        ``non_local_event``.
    """
    time_s = np.asarray(time, dtype=float)
    detected = get_confident_times(
        non_local_probability.assign_coords(time=time_s),
        speed,
        time_s,
        prob_threshold,
        minimum_duration,
    )
    return build_speed_eligible_event_cohort(
        detected,
        speed=speed,
        time=time,
        event_col="non_local_event",
        preserve_source_ids=False,
    )


def positive_duration_interval_mask(
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    """Return finite intervals with strictly positive duration."""
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    return np.isfinite(starts) & np.isfinite(ends) & (ends > starts)


def strict_overlap_matrix(
    starts: np.ndarray,
    ends: np.ndarray,
    other_starts: np.ndarray,
    other_ends: np.ndarray,
) -> np.ndarray:
    """Pairwise strict positive-duration overlap between two interval sets.

    This is the one definition of "strict overlap" used by every SWR-overlap
    consumer (:func:`get_overlap_times`, :func:`classify_swr_overlap`,
    :func:`classify_swr_overlap_geometry`, :func:`event_strict_overlap_keys`
    and the classified event table's SWR matcher). The closed and open
    overlap tests are in :mod:`continuum_swr_replay.intervals`
    (``overlaps_intervals``, ``overlaps_open``), whose docstring tabulates the
    conventions.

    Parameters
    ----------
    starts, ends : np.ndarray, shape (n_intervals,)
        Bounds of the primary intervals, in seconds.
    other_starts, other_ends : np.ndarray, shape (n_other,)
        Bounds of the intervals tested against, in seconds.

    Returns
    -------
    np.ndarray of bool, shape (n_intervals, n_other)
        ``True`` where both intervals are finite with strictly positive
        duration (:func:`positive_duration_interval_mask`) and they share
        positive duration: ``other_end > start`` and ``other_start < end``.
        Intervals that only touch at a boundary share zero duration and are
        ``False``; a zero-duration interval overlaps nothing.
    """
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    other_starts = np.asarray(other_starts, dtype=float)
    other_ends = np.asarray(other_ends, dtype=float)
    return (
        positive_duration_interval_mask(starts, ends)[:, None]
        & positive_duration_interval_mask(other_starts, other_ends)[None, :]
        & (other_ends[None, :] > starts[:, None])
        & (other_starts[None, :] < ends[:, None])
    )


def event_strict_overlap_keys(
    data_over_time: pd.DataFrame,
    primary_col: str,
    other_col: str,
    *,
    session_cols: tuple[str, ...] = ("animal", "day", "epoch"),
) -> set[tuple]:
    """Per-(session, primary_id) keys whose interval strictly overlaps any other event.

    For each event of ``primary_col`` (id > 0) in each session, derive
    its [first, last] timestamp interval from the timebin labels
    (:func:`~continuum_swr_replay.analysis.event_definition.immobility.intervals_from_labels`,
    read on the frame's time index) and test strict positive-duration
    overlap (:func:`strict_overlap_matrix`) against every event of
    ``other_col`` in the same session. Returns the set of
    ``(*session_cols, primary_id)`` tuples that pass the strict-overlap
    test. These are the bounds and the predicate the canonical classified
    event table uses, so for Non-Local x SWR the keys are exactly its
    ``is_swr`` events.

    Both sides come from ``data_over_time``, so they share one
    speed-eligible cohort and the overlap counts do not mix populations.

    Events whose labeled span covers a single timebin have zero duration
    and are excluded from both sides of the comparison.

    Why this exists: the inclusive timebin labels written by
    :func:`continuum_swr_replay.data_loaders.utils.get_labels` make a
    boundary timestamp visible to BOTH events when their intervals
    touch (event_a end == event_b start). The label-based
    ``(g[other_col] > 0).any()`` overlap check then counts boundary
    contact as overlap, conflicting with the strict positive-duration
    semantics enforced by :func:`classify_swr_overlap` and
    :func:`get_overlap_times`. This helper yields the strict-semantic
    counterpart usable wherever per-event overlap with another
    detector is needed.
    """
    sess = list(session_cols)

    def _session_intervals(df: pd.DataFrame, ec: str) -> dict[tuple, pd.DataFrame]:
        # Only the label column is passed on, so the bounds always come from
        # the frame's time index (the samples the labels were written on).
        active = df[df[ec] > 0][[*sess, ec]]
        return {
            (key if isinstance(key, tuple) else (key,)): intervals_from_labels(
                grp[[ec]], ec
            )
            for key, grp in active.groupby(sess, sort=False)
        }

    o_intervals = _session_intervals(data_over_time, other_col)
    keys: set[tuple] = set()
    if not o_intervals:
        return keys

    for sk, p_grp in _session_intervals(data_over_time, primary_col).items():
        o_grp = o_intervals.get(sk)
        if o_grp is None:
            continue
        hit = strict_overlap_matrix(
            p_grp["start_time"].to_numpy(),
            p_grp["end_time"].to_numpy(),
            o_grp["start_time"].to_numpy(),
            o_grp["end_time"].to_numpy(),
        ).any(axis=1)
        keys.update((*sk, event_id) for event_id in p_grp[primary_col].to_numpy()[hit])
    return keys


def event_strict_overlap_mask(
    data_over_time: pd.DataFrame,
    primary_col: str,
    other_col: str,
    *,
    session_cols: tuple[str, ...] = ("animal", "day", "epoch"),
) -> pd.Series:
    """Mark every bin of primary events that strictly overlap another event.

    Classification is event-level: when a primary event has any
    positive-duration overlap, all of its labeled bins are ``True``. This is
    deliberately different from a raw per-bin coincidence mask, which splits
    partially overlapping events across both comparison groups.
    """
    require_columns(
        data_over_time, [*session_cols, primary_col, other_col], "data_over_time"
    )

    active = data_over_time[primary_col].to_numpy() > 0
    mask = np.zeros(len(data_over_time), dtype=bool)
    if not active.any():
        return pd.Series(mask, index=data_over_time.index)

    overlap_keys = event_strict_overlap_keys(
        data_over_time,
        primary_col,
        other_col,
        session_cols=session_cols,
    )
    if overlap_keys:
        active_keys = pd.MultiIndex.from_frame(
            data_over_time.loc[active, [*session_cols, primary_col]]
        )
        mask[active] = active_keys.isin(list(overlap_keys))
    return pd.Series(mask, index=data_over_time.index)


def nonlocal_event_bins_by_swr(
    data_over_time: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """The bins of non-local events, split by whether the event overlaps an SWR.

    The split is event-level (:func:`event_strict_overlap_mask`): every bin of
    an event with a positive-duration SWR overlap is in the SWR set, every bin
    of another event in the non-SWR set.

    Parameters
    ----------
    data_over_time : pd.DataFrame
        A frame read through the SWR source (``load_data_over_time``), whose
        ``non_local_event`` labels hold only eligible events (each immobile as
        a whole and above the probability threshold), with ``SWR_event`` and
        the session columns.

    Returns
    -------
    swr, non_swr : pd.Series of bool, shape (n_bins,)
        The bins of SWR-overlapping events, and of the other events.
    """
    active = data_over_time["non_local_event"] > 0
    is_swr_event = event_strict_overlap_mask(
        data_over_time, "non_local_event", "SWR_event"
    )
    return active & is_swr_event, active & ~is_swr_event


def get_overlap_times(
    non_local_times: pd.DataFrame,
    ripple_times: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find non-local events that overlap with ripple events.

    Parameters
    ----------
    non_local_times : pd.DataFrame
        Non-local event intervals with ``start_time``, ``end_time``,
        ``duration``, ``animal_max_speed_cm_s`` columns. From
        ``get_confident_times``.
    ripple_times : pd.DataFrame
        Ripple event intervals with ``start_time`` and ``end_time`` columns.

    Returns
    -------
    overlap_non_local_times : pd.DataFrame
        Subset of ``non_local_times`` that overlap with at least one ripple.
        Same columns as input.
    no_overlap_non_local_times : pd.DataFrame
        Subset of ``non_local_times`` that do not overlap with any ripple.
        Same columns as input.

    Notes
    -----
    Boundary semantics: overlap requires positive-duration intersection.
    A ripple ending exactly at the event's start (or starting exactly
    at the event's end) shares zero duration with the event and is
    classified as no-overlap. This matches
    :func:`classify_swr_overlap` and avoids inflating SWR-coincident
    counts on back-to-back intervals.

    """
    swr_starts = ripple_times["start_time"].to_numpy(dtype=float)
    swr_ends = ripple_times["end_time"].to_numpy(dtype=float)
    ev_starts = non_local_times["start_time"].to_numpy(dtype=float)
    ev_ends = non_local_times["end_time"].to_numpy(dtype=float)

    # For each non-local event, True if any ripple has a strict
    # positive-duration intersection.
    has_overlap = strict_overlap_matrix(ev_starts, ev_ends, swr_starts, swr_ends).any(
        axis=1
    )

    overlap_non_local_times = non_local_times.iloc[has_overlap]
    no_overlap_non_local_times = non_local_times.iloc[~has_overlap]

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

    Boundary semantics: overlap requires *positive-duration* intersection.
    A SWR ending exactly at the event's start time (or starting exactly
    at the event's end time) shares zero positive duration with the event
    and is treated as ``non_swr``. The previous form used inclusive
    inequality (``>=`` / ``<=``) and silently labeled back-to-back
    intervals as ``swr_partial``, inflating SWR-coincident counts.

    """
    classes: list[str] = []
    swr_starts = ripple_times["start_time"].to_numpy(dtype=float)
    swr_ends = ripple_times["end_time"].to_numpy(dtype=float)
    # Strict positive-duration overlap. Boundary contact
    # (swr_end == event.start or swr_start == event.end) shares zero duration
    # with the event and is excluded; a zero- (or negative-) duration event
    # overlaps nothing, so it is non-SWR even if an SWR spans its instant.
    overlaps = strict_overlap_matrix(
        non_local_events["start_time"].to_numpy(dtype=float),
        non_local_events["end_time"].to_numpy(dtype=float),
        swr_starts,
        swr_ends,
    )

    for ev, overlap_mask in zip(non_local_events.itertuples(), overlaps, strict=True):
        if not overlap_mask.any():
            classes.append("non_swr")
            continue
        # Any SWR that fully contains this non-local event? Boundary
        # equality is allowed for containment (an SWR exactly aligned
        # with the event still "contains" it).
        contained_mask = (
            overlap_mask & (swr_starts <= ev.start_time) & (swr_ends >= ev.end_time)
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
    and comparing merged duration to event duration (with
    ``INTERVAL_TOLERANCE_SEC`` to absorb floating-point noise).

    """
    swr_starts = ripple_times["start_time"].to_numpy(dtype=float)
    swr_ends = ripple_times["end_time"].to_numpy(dtype=float)
    # Strict positive-duration overlap (see ``classify_swr_overlap`` for
    # rationale): a SWR that only touches a boundary of the event shares zero
    # duration and is not an overlap. A zero- (or negative-) duration event
    # overlaps nothing, so it is non-SWR regardless of any SWR spanning its
    # instant; without that, the open-interval test would treat a point
    # strictly inside an SWR as contained.
    overlaps = strict_overlap_matrix(
        non_local_events["start_time"].to_numpy(dtype=float),
        non_local_events["end_time"].to_numpy(dtype=float),
        swr_starts,
        swr_ends,
    )

    n_events = len(non_local_events)
    classes: list[str] = [""] * n_events
    subclasses: list[str] = [""] * n_events
    n_overlap_arr = np.zeros(n_events, dtype=int)
    contains_any_swr_arr = np.zeros(n_events, dtype=bool)
    event_has_non_swr_time_arr = np.zeros(n_events, dtype=bool)

    for i, ev in enumerate(non_local_events.itertuples()):
        overlap_mask = overlaps[i]
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
        containing_mask = (overlapping_starts <= ev.start_time) & (
            overlapping_ends >= ev.end_time
        )
        if containing_mask.any():
            classes[i] = "swr_contained"
            subclasses[i] = "swr_contains_event"
            # Event sits inside an SWR, so every timebin is SWR-covered.
            event_has_non_swr_time_arr[i] = False
            continue

        # Neither contained. Decide among the partial-overlap subclasses.
        strict_contained_mask = (overlapping_starts > ev.start_time) & (
            overlapping_ends < ev.end_time
        )
        contains_any_swr = bool(strict_contained_mask.any())
        contains_any_swr_arr[i] = contains_any_swr

        # Does the event have any timebins outside every SWR?
        # Clip each overlapping SWR to [ev.start, ev.end], merge, and
        # compare total merged duration to event duration.
        union_duration = total_duration(
            np.column_stack(
                [
                    np.maximum(overlapping_starts, ev.start_time),
                    np.minimum(overlapping_ends, ev.end_time),
                ]
            )
        )
        event_duration = ev.end_time - ev.start_time
        event_has_non_swr_time = (
            union_duration < event_duration - INTERVAL_TOLERANCE_SEC
        )
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
