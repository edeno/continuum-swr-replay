"""Summary statistics of the immobility stops and of where event onsets fall in them.

Each function takes the canonical stop table (one row per stop) or the event
onset table (one row per event onset, with the stop it falls in) and returns
named numbers. Times are in seconds. A stop is well-anchored when its
``occupancy_state`` is ``at_well`` or ``mixed``
(:func:`~continuum_swr_replay.analysis.event_definition.cohort_selection.select_stop_cohort`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.behavior.design_constants import (
    RATE_BIN_S,
    TMAX,
)
from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    excluded_intervals_by_session,
)
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    select_stop_cohort,
)
from continuum_swr_replay.analysis.replay_models.model_specs import (
    MODEL_PER_BIN_RATE,
    longest_history_window_s,
)
from continuum_swr_replay.analysis.replay_models.per_bin_design import (
    aggregate_to_timebins,
    arrival_spanning_stop_ids,
)
from continuum_swr_replay.intervals import covered_seconds


def stop_cohort_counts(stops: pd.DataFrame) -> dict[str, int]:
    """How the stops divide by where the animal is, and which carry a stay/leave
    outcome.

    Parameters
    ----------
    stops : pd.DataFrame, shape (n_stops, n_columns)
        The stop table, with ``occupancy_state`` (``at_well``, ``mixed`` or
        ``in_transit``) and ``is_stay`` (missing when the stop has no stay/leave
        outcome).

    Returns
    -------
    counts : dict
        - ``n_stops``: the stops.
        - ``n_well_anchored``: stops at a well or mixed.
        - ``n_not_anchored``: the others.
        - ``n_not_anchored_with_stay``: of those, stops with an ``is_stay``.
        - ``n_in_transit``: in-transit stops.
        - ``n_mixed``: mixed stops.
        - ``n_mixed_with_stay``: mixed stops with an ``is_stay``.
        - ``n_at_well_with_stay``: at-well stops with an ``is_stay``.
    """
    anchored, report = select_stop_cohort(stops, cohort="at_well")
    occupancy = stops["occupancy_state"]
    has_stay = stops["is_stay"].notna()
    return {
        "n_stops": len(stops),
        "n_well_anchored": report["n_stops_after"],
        "n_not_anchored": report["n_stops_dropped"],
        "n_not_anchored_with_stay": int(
            has_stay.sum() - anchored["is_stay"].notna().sum()
        ),
        "n_in_transit": int((occupancy == "in_transit").sum()),
        "n_mixed": int((occupancy == "mixed").sum()),
        "n_mixed_with_stay": int((has_stay & (occupancy == "mixed")).sum()),
        "n_at_well_with_stay": int((has_stay & (occupancy == "at_well")).sum()),
    }


def dwell_statistics(stops: pd.DataFrame) -> dict[str, float]:
    """How long the analyzed immobility lasts after a rewarded and an omitted outcome.

    The duration is the stop's analyzed immobility, ``immobility_duration_s``:
    its span, ``stop_end - stop_start``, less the excluded time inside it. It
    is shorter than the span for a stop that holds excluded time, and zero for
    a stop of one sample or wholly inside excluded time.

    Parameters
    ----------
    stops : pd.DataFrame
        Stops with ``is_rewarded_outcome`` (1 rewarded, 0 omitted; a stop
        without a known outcome is left out) and ``immobility_duration_s``.

    Returns
    -------
    statistics : dict
        ``mean_rewarded_s`` and ``mean_omitted_s``, the mean analyzed
        immobility, in seconds, over all the stops with each outcome (those of
        zero duration included), and ``mean_ratio``, rewarded over omitted;
        ``median_rewarded_s``, ``median_omitted_s`` and ``median_ratio``, the
        same of the medians.
    """
    duration = stops.groupby("is_rewarded_outcome")["immobility_duration_s"]
    means, medians = duration.mean(), duration.median()
    return {
        "mean_rewarded_s": means[1.0],
        "mean_omitted_s": means[0.0],
        "mean_ratio": means[1.0] / means[0.0],
        "median_rewarded_s": medians[1.0],
        "median_omitted_s": medians[0.0],
        "median_ratio": medians[1.0] / medians[0.0],
    }


def still_immobile_fraction(stops: pd.DataFrame, at_s: float) -> float:
    """The share of stops still immobile a given time after they began.

    The time is measured from the start of the stop, so excluded time inside a
    stop is not taken out of it.

    Parameters
    ----------
    stops : pd.DataFrame
        Stops with ``stop_start`` and ``stop_end``, in seconds.
    at_s : float
        The time since the stop began, in seconds.

    Returns
    -------
    fraction : float
        The share, of all the stops, of those whose span, ``stop_end -
        stop_start``, is at least ``at_s`` seconds.
    """
    return float((stops["stop_end"] - stops["stop_start"] >= at_s).mean())


def onset_rate_time_course(
    stops: pd.DataFrame,
    onsets: pd.DataFrame,
    window_s: float,
    bin_s: float,
    excluded_intervals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The rate at which events begin at each time since their stop began.

    The count of onsets in each bin of time since the stop began, over the
    seconds of immobility the stops hold in that bin. The clock is the stop's
    own, ``stop_end - stop_start``: a stop covers the bin ``[e, e + bin_s)``
    from ``stop_start + e`` to its end, less the excluded seconds there. Its
    ``immobility_duration_s`` is already net of excluded time, so it is not a
    length on this clock. Fewer, longer stops reach the later bins, which the
    exposure accounts for; :func:`still_immobile_fraction` says how many.

    Parameters
    ----------
    stops : pd.DataFrame
        Stops with the session columns, ``stop_start`` and ``stop_end``, in
        seconds.
    onsets : pd.DataFrame
        The onsets of one population in those stops, with
        ``t_since_stop_onset`` (seconds); one outside ``[0, window_s)`` is
        left out.
    window_s : float
        The time since the stop began that the bins span, in seconds: a whole
        number of bins (up to floating-point rounding, so 4.0 s of 0.1 s bins
        is forty).
    bin_s : float
        The bin width, in seconds.
    excluded_intervals : pd.DataFrame, optional
        Each session's excluded time: the session columns, ``start_time`` and
        ``end_time``, in seconds. None (the default) reads it from the SWR
        source
        (:func:`~continuum_swr_replay.analysis.event_definition.analyzed_time.excluded_intervals_by_session`).

    Returns
    -------
    time_course : pd.DataFrame, shape (n_bins, 3)
        One row per bin, indexed by ``time_s``, the bin's center: ``n_onsets``;
        ``exposure_s``, the seconds of immobility in the bin over all the
        stops; and ``rate``, their ratio in events per second (NaN where no
        stop is immobile).

    Raises
    ------
    ValueError
        If ``window_s`` is not a whole number of bins (the last bin would reach
        past the window).
    """
    n_bins = round(window_s / bin_s)
    if not np.isclose(n_bins * bin_s, window_s):
        raise ValueError(
            f"a window of {window_s} s is not a whole number of bins of {bin_s} s"
        )
    if excluded_intervals is None:
        excluded_intervals = excluded_intervals_by_session(stops)
    t = onsets["t_since_stop_onset"].to_numpy()
    t = t[(t >= 0) & (t < window_s)]
    edges = np.arange(n_bins + 1) * bin_s
    n_onsets, _ = np.histogram(t, edges)
    keys = ["animal", "session_date", "epoch"]
    exposure = np.zeros(len(edges) - 1)
    for session, stops_g in stops.groupby(keys, sort=False):
        session_excluded = excluded_intervals.loc[
            (excluded_intervals[keys] == pd.Series(session, index=keys)).all(axis=1)
        ]
        start = stops_g["stop_start"].to_numpy(dtype=float)
        span = stops_g["stop_end"].to_numpy(dtype=float) - start
        for k, (low, high) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            lo, hi = start + low, start + np.minimum(span, high)
            inside = hi > lo
            exposure[k] += (hi - lo)[inside].sum() - covered_seconds(
                lo[inside], hi[inside], session_excluded
            ).sum()
    return pd.DataFrame(
        {
            "n_onsets": n_onsets,
            "exposure_s": exposure,
            "rate": n_onsets / np.where(exposure > 0, exposure, np.nan),
        },
        index=pd.Index(edges[:-1] + bin_s / 2, name="time_s"),
    )


def outcome_boundary_statistics(
    stops: pd.DataFrame, excluded_intervals: pd.DataFrame | None = None
) -> dict[str, float]:
    """Which stops the per-bin design's outcome-boundary shift can apply to.

    A stop crosses its outcome boundary when the outcome becomes known inside
    it: ``stop_start < outcome_known_time_s <= stop_end``. It spans an arrival
    when the session's next stop arrived at its well inside this one. The
    per-bin design shifts a bin's reward history at the boundary in the
    crossing stops that span no arrival, and only where it still has a bin
    that starts at or past the boundary.

    The bins counted here are the stop-level design's: bins of ``RATE_BIN_S`` within
    ``TMAX`` of the stop's start, less those that excluded time and the
    ladder's history window drop. A fitted population's design also drops the
    bins inside one of its ongoing events, which is not done here, so
    ``n_shifted`` is an upper bound on the stops shifted in each fitted
    population.

    Parameters
    ----------
    stops : pd.DataFrame, shape (n_stops, n_columns)
        The stop table, with the columns
        :func:`~continuum_swr_replay.analysis.replay_models.per_bin_design.aggregate_to_timebins`
        reads (the session columns, ``stop_id``, ``stop_start``, ``stop_end``,
        ``immobility_duration_s``, ``outcome_known_time_s`` and
        ``last_well_arrival_time_s``).
    excluded_intervals : pd.DataFrame, optional
        Each session's excluded time, as ``aggregate_to_timebins`` takes it:
        the session columns, ``start_time`` and ``end_time``, in seconds. None
        (the default) reads it from the SWR source.

    Returns
    -------
    statistics : dict
        - ``n_crossing``, ``crossing_fraction``: stops that cross their outcome
          boundary, and their share of all stops.
        - ``n_shifted``, ``shifted_fraction``: crossing stops that span no
          arrival and keep a bin at or past the boundary, and their share of
          all stops.
        - ``n_spanning_arrival``: stops that contain a later well arrival.
        - ``n_spanning_arrival_crossing``: of those, stops that also cross
          their outcome boundary.
    """
    crosses = (stops["stop_start"] < stops["outcome_known_time_s"]) & (
        stops["outcome_known_time_s"] <= stops["stop_end"]
    )
    spans = stops["stop_id"].isin(arrival_spanning_stop_ids(stops))
    delta = (stops["outcome_known_time_s"] - stops["stop_start"]).to_numpy(float)
    bins = aggregate_to_timebins(
        stops,
        pd.DataFrame({"stop_id": [], "t_since_stop_onset": []}),
        bin_s=RATE_BIN_S,
        tmax=TMAX,
        excluded_intervals=excluded_intervals,
        history_window_s=longest_history_window_s(MODEL_PER_BIN_RATE.values()),
    )
    last_bin = stops["stop_id"].map(bins.groupby("stop_id")["binj"].max())
    # a stop with no bin, or with no outcome time, compares False
    has_bin = last_bin.to_numpy(float) >= np.ceil(delta / RATE_BIN_S)
    shifted = crosses & ~spans & has_bin
    return {
        "n_crossing": int(crosses.sum()),
        "crossing_fraction": float(crosses.mean()),
        "n_shifted": int(shifted.sum()),
        "shifted_fraction": float(shifted.mean()),
        "n_spanning_arrival": int(spans.sum()),
        "n_spanning_arrival_crossing": int((spans & crosses).sum()),
    }


def onset_time_statistics(onsets: pd.DataFrame, window_s: float) -> pd.DataFrame:
    """When, within their stop, the events of each type begin.

    Parameters
    ----------
    onsets : pd.DataFrame, shape (n_onsets, n_columns)
        The event onset table, with ``event_type`` and ``t_since_stop_onset``
        (seconds from the stop's speed-threshold crossing to the event's onset;
        an onset without one is left out).
    window_s : float
        The window from the stop's start, in seconds.

    Returns
    -------
    statistics : pd.DataFrame
        One row per event type: ``within_window_fraction``, the share of its
        onsets earlier than ``window_s``, and ``median_s``, the median onset
        time.
    """
    rows = {}
    for event_type, type_onsets in onsets.groupby("event_type"):
        t = type_onsets["t_since_stop_onset"].dropna()
        rows[event_type] = {
            "within_window_fraction": float((t < window_s).mean()),
            "median_s": t.median(),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("event_type")


def reward_history_incomplete_statistics(
    stops: pd.DataFrame, onsets: pd.DataFrame, event_type: str
) -> dict[str, float]:
    """How much of the data falls in stops without a complete reward history.

    Parameters
    ----------
    stops : pd.DataFrame
        The stop table, with ``stop_id`` (unique) and
        ``reward_history_incomplete`` (1 when the stop has fewer than two known
        outcomes behind it).
    onsets : pd.DataFrame
        The event onset table, with ``event_type`` and ``stop_id``.
    event_type : str
        The event type whose onsets are counted.

    Returns
    -------
    statistics : dict
        ``stop_fraction``, the share of stops flagged incomplete, and
        ``onset_fraction``, the share of the event type's onsets that fall in
        such a stop.
    """
    incomplete = stops.set_index("stop_id")["reward_history_incomplete"]
    onset_stops = onsets.loc[onsets["event_type"] == event_type, "stop_id"]
    return {
        "stop_fraction": float((stops["reward_history_incomplete"] == 1).mean()),
        "onset_fraction": float((onset_stops.map(incomplete) == 1).mean()),
    }
