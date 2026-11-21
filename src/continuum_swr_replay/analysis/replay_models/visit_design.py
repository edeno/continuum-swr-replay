"""The well-visit time-bin design of the event-rate models.

:func:`visit_timebins` cuts every immobility stop of one session into
``RATE_BIN_S`` bins and gives each bin what a rate model of one event kind
reads: its well-visit label, its visit and stretch, its at-risk seconds, the
event starts of the kind it holds, the three visit clocks and the reward
history known at its time. The visits are the session's port experiences.

**Where the bins lie.** A stop's analyzed time is its span
``[stop_start, stop_end]`` less excluded time
(``event_definition.analyzed_time.excluded_intervals``). That time is cut at
the well visits' boundaries (``event_definition.well_visits``: a time is inside
a visit from its first poke-in, included, to its final poke-out, not included),
so a stop that crosses the final poke-out gives bins up to it and a stop that
starts before the first poke-in gives bins from it. Bins are laid from the start
of each piece; the last bin of a piece is partial. A bin therefore lies wholly
inside or wholly outside a visit, and wholly outside excluded time.

**Labels.** A bin inside a visit is ``in_visit`` and carries the visit's
``well_visit_id``. A bin outside every visit carries its stop's
``visit_context``, except that the bins after the final poke-out of a stop that
starts inside a visit (``spans_visit_end``) are ``after_visit_at_well``: the
stop's own label names the time where the stop starts, and these bins are not
in the time the analyses read. So ``select_visit_context(bins)`` keeps exactly
the bins inside the visits (no bin is ``spans_visit_end``), and with them
exactly the events of the default event set
(``analysis.constants.STARTS_IN_WELL_VISIT``): an event belongs to the time
when it starts, and an event starts inside a visit exactly when its start lies
in an in-visit bin. The rate and the content models analyze the same events.

**At-risk time.** A bin's ``exposure`` is its length less the time inside an
ongoing event of the modeled kind; a bin with none left is dropped. Events of
the other kind do not reduce it. An event start is counted in the bin that
holds the at-risk time just before it (a bin is ``(left, right]`` for counting),
so an event that starts on a bin edge is counted in the bin that ends there.

**Timing.** The two clocks that count down (``time_until_visit_end_s``,
``time_until_stretch_end_s``) embed when the animal will next move or leave:
they are future-referencing (``covariate_timing.FUTURE_REFERENCING``), and a
model that uses them is descriptive
(``covariate_timing.assert_predictor_timing(..., descriptive=True)``).

**Reward history.** A bin's ``known_reward_outcome_lag_1..10`` are the
outcomes of the port experiences known at its left edge
(``behavior.covariates.history.lagged_trial_rewards``), most recent first: the
outcome of the visit a bin lies in is lag 1 from the first bin that starts at
or after that outcome became known, also in a stop that began before the
visit's first poke-in.

The module also holds the pieces of the stop-grain design of
``per_bin_design``: the seconds of each bin an interval covers
(:func:`interval_coverage_by_bin`, which :func:`visit_timebins` uses too), the
outcome-boundary shift of a stop's reward lags
(:func:`shift_reward_lags_at_outcome`) and the stops that span a well arrival
by the next stop's arrival clock (:func:`arrival_spanning_stop_ids`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    excluded_intervals_by_session,
)
from continuum_swr_replay.analysis.event_definition.well_visits import (
    well_visit_index,
)
from continuum_swr_replay.intervals import intersect, subtract
from continuum_swr_replay.tables.columns import require_columns
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

from ..behavior.covariates.history import lagged_trial_rewards
from ..behavior.covariates.history._common import _prepare_port_experiences
from ..behavior.design_constants import RATE_BIN_S, REWARD_HISTORY_N_LAGS
from ..constants import IN_WELL_VISIT, STARTS_IN_WELL_VISIT

__all__ = [
    "EVENT_KINDS",
    "VISIT_BIN_COLUMNS",
    "arrival_spanning_stop_ids",
    "interval_coverage_by_bin",
    "shift_reward_lags_at_outcome",
    "visit_timebins",
]

#: The event kinds of the non-local event table, by their ``is_swr`` value.
EVENT_KINDS: dict[str, bool] = {"SWR": True, "No SWR": False}

REWARD_LAG_COLUMNS: tuple[str, ...] = tuple(
    f"known_reward_outcome_lag_{j}" for j in range(1, REWARD_HISTORY_N_LAGS + 1)
)

#: The columns :func:`visit_timebins` returns, in order.
VISIT_BIN_COLUMNS: tuple[str, ...] = (
    *SESSION_COLUMNS,
    "stop_id",
    "visit_context",
    "well_visit_id",
    "visit_stretch_index",
    "bin_start_s",
    "bin_end_s",
    "exposure",
    "count",
    "time_since_visit_start_s",
    "time_until_visit_end_s",
    "time_until_stretch_end_s",
    *REWARD_LAG_COLUMNS,
)

#: Seconds below which two times are one (absolute session times carry about
#: 2e-7 s of rounding; a decode sample is 2 ms). Also the at-risk seconds below
#: which a bin is wholly inside an event.
_TIME_TOLERANCE_S = 1e-5

#: The label of the bins after the final poke-out of a stop that starts inside
#: a visit.
_AFTER_VISIT = "after_visit_at_well"

_STOP_COLUMNS = (
    *SESSION_COLUMNS,
    "stop_id",
    "stop_start",
    "stop_end",
    "visit_context",
    "overlapping_visit_id",
    "visit_first_poke_in_s",
    "visit_last_poke_out_s",
    "visit_stretch_index",
)

_EVENT_COLUMNS = (
    "event_start_s",
    "event_end_s",
    "visit_context",
    "well_visit_id",
    "time_since_visit_start_s",
    "time_until_visit_end_s",
)


def arrival_spanning_stop_ids(stops: pd.DataFrame) -> np.ndarray:
    """``stop_id`` of the bouts that contain a later well arrival.

    A stop spans an arrival iff the NEXT stop of the same session -- ordering
    by ``stop_start`` within ``animal, session_date, epoch`` -- reports a
    ``last_well_arrival_time_s`` strictly inside ``(stop_start, stop_end]`` of
    this bout. Well arrivals are monotone within a session, so only the next
    stop's arrival can fall inside this one and the test is exact. A stop with
    no successor, or whose successor has no arrival clock, does not span.

    Parameters
    ----------
    stops : pd.DataFrame
        Per-stop frame (unique ``stop_id``) carrying ``animal``,
        ``session_date``, ``epoch``, ``stop_start``, ``stop_end`` and
        ``last_well_arrival_time_s``.

    Returns
    -------
    np.ndarray
        The spanning stops' ``stop_id`` values (possibly empty).
    """
    ordered = stops.sort_values([*SESSION_COLUMNS, "stop_start"], kind="stable")
    next_arrival = ordered.groupby(list(SESSION_COLUMNS), sort=False, observed=True)[
        "last_well_arrival_time_s"
    ].shift(-1)
    spans = (next_arrival > ordered["stop_start"]) & (
        next_arrival <= ordered["stop_end"]
    )
    return ordered.loc[spans.fillna(False).to_numpy(), "stop_id"].to_numpy()


def shift_reward_lags_at_outcome(
    lags: np.ndarray,
    completed_trial_reward: np.ndarray,
    stop_start: np.ndarray,
    outcome_known_time: np.ndarray,
    bin_start: np.ndarray,
    spans_arrival: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The known reward lags of each bin, with the outcome that became known.

    A stop's lags are those known at its start. In a bin whose left edge is at
    or after the stop's ``outcome_known_time_s``, for a stop that starts before
    it, the current trip's outcome is known too: the lags move back by one and
    ``completed_trial_reward`` becomes lag 1. A bin that straddles the boundary
    keeps the stop's lags. A stop that spans a well arrival
    (:func:`arrival_spanning_stop_ids`) is not shifted: partway through it the
    completed trip changes, so ``completed_trial_reward`` is not the outcome
    that becomes known.

    Parameters
    ----------
    lags : np.ndarray, shape (n_bins, n_lags)
        Each bin's stop's ``known_reward_outcome_lag_1..n_lags`` (NaN unknown).
    completed_trial_reward : np.ndarray, shape (n_bins,)
        The stop's current-trip outcome (1 rewarded, 0 omitted, NaN none).
    stop_start, outcome_known_time : np.ndarray, shape (n_bins,)
        The stop's start and outcome boundary (absolute s; NaN boundary: none).
    bin_start : np.ndarray, shape (n_bins,)
        Each bin's left edge (absolute s).
    spans_arrival : np.ndarray of bool, shape (n_bins,)
        Whether the bin's stop spans a well arrival.

    Returns
    -------
    shifted : np.ndarray, shape (n_bins, n_lags)
        The lags known at the bin's left edge.
    past_boundary : np.ndarray of bool, shape (n_bins,)
        The bins at or past their stop's boundary, spanning stops included
        (those are the ones left unshifted).
    """
    # NaN boundary -> False, so a stop with no known outcome is untouched;
    # ``stop_start < boundary`` excludes stops already past their outcome at
    # bout start (their bout-start history already contains it).
    past_boundary = (stop_start < outcome_known_time) & (
        bin_start >= outcome_known_time
    )
    shift = past_boundary & ~spans_arrival
    shifted = lags.copy()
    shifted[shift, 1:] = lags[shift, :-1]
    shifted[shift, 0] = completed_trial_reward[shift]
    return shifted, past_boundary


def interval_coverage_by_bin(
    segment: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    horizon: np.ndarray,
    *,
    bin_s: float,
) -> pd.DataFrame:
    """Seconds of each bin of a segment that an interval covers.

    The bins of a segment are ``[j * bin_s, (j + 1) * bin_s)``, ``j = 0, 1,
    ...``, in seconds from the segment's origin, up to ``horizon``.

    Parameters
    ----------
    segment : np.ndarray, shape (n_intervals,)
        The segment each interval lies in (any hashable id).
    start, end : np.ndarray, shape (n_intervals,)
        Each interval's bounds, in seconds from its segment's origin. The part
        before 0 or past ``horizon`` is not counted.
    horizon : np.ndarray, shape (n_intervals,)
        The end of the interval's segment's bins (s from its origin).
    bin_s : float
        Bin width (s).

    Returns
    -------
    pd.DataFrame
        ``segment``, ``binj`` and ``covered_s``: one row per touched bin,
        overlapping intervals summed.
    """
    start = np.maximum(np.asarray(start, dtype=float), 0.0)
    end = np.minimum(np.asarray(end, dtype=float), np.asarray(horizon, dtype=float))
    keep = end > start
    start, end, segment = start[keep], end[keep], np.asarray(segment)[keep]
    first = np.floor(start / bin_s + 1e-9).astype(int)
    last = np.maximum(np.ceil(end / bin_s - 1e-9).astype(int) - 1, first)
    n_bins = last - first + 1
    rep = np.repeat(np.arange(len(start)), n_bins)
    binj = first[rep] + (
        np.arange(len(rep)) - np.repeat(np.cumsum(n_bins) - n_bins, n_bins)
    )
    piece = np.minimum(end[rep], (binj + 1) * bin_s) - np.maximum(
        start[rep], binj * bin_s
    )
    return (
        pd.DataFrame(
            {
                "segment": segment[rep],
                "binj": binj,
                "covered_s": np.clip(piece, 0.0, None),
            }
        )
        .groupby(["segment", "binj"], as_index=False)["covered_s"]
        .sum()
    )


def visit_timebins(
    stop_table: pd.DataFrame,
    events: pd.DataFrame,
    port_experiences: pd.DataFrame,
    *,
    kind: str | None,
    bin_s: float = RATE_BIN_S,
    excluded_intervals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The time bins of one session's stops, for the rate of one event kind.

    Parameters
    ----------
    stop_table : pd.DataFrame
        The stop table's rows of one session (unique ``stop_id``), with
        ``stop_start``, ``stop_end``, ``visit_context`` and the visit columns
        ``overlapping_visit_id``, ``visit_first_poke_in_s``,
        ``visit_last_poke_out_s`` and ``visit_stretch_index``
        (``event_tables.visit_context.stop_visit_columns``).
    events : pd.DataFrame
        The session's events (an event table: ``event_start_s``,
        ``event_end_s``, ``visit_context``, ``well_visit_id``,
        ``time_since_visit_start_s``, ``time_until_visit_end_s``, and
        ``is_swr`` when ``kind`` is given) and the session columns. Rows of
        other sessions are ignored; a non-empty frame must hold rows of this
        session (a session with no events passes an empty frame). Only the
        kind's are counted and reduce the at-risk time; all of them are checked
        against the visits.
    port_experiences : pd.DataFrame
        The session's port experiences (one per visit, ``experience_id`` 0, 1,
        ... in time order), as the table builders take them
        (``event_tables.prepare_data.load_behavioral_context``): the visits and
        the reward outcomes.
    kind : {"SWR", "No SWR"} or None
        The events modeled, by ``is_swr`` (:data:`EVENT_KINDS`); None models
        every event (an event source with one stream, such as the ContFrag
        SWR or MUA events).
    bin_s : float
        Bin width (s); default ``RATE_BIN_S``.
    excluded_intervals : pd.DataFrame, optional
        The session's excluded time: the session columns, ``start_time`` and
        ``end_time`` (absolute s). Rows of other sessions are ignored; a
        non-empty frame must hold rows of this session (a session with no
        excluded time passes an empty frame). None reads it from the SWR
        source (``analyzed_time.excluded_intervals_by_session``).

    Returns
    -------
    pd.DataFrame
        One row per bin with at-risk time, in time order; columns
        :data:`VISIT_BIN_COLUMNS`:

        - ``animal``, ``session_date``, ``epoch``, ``stop_id``: the bin's stop.
        - ``visit_context``: ``in_visit`` inside a visit; outside, the stop's
          label, or ``after_visit_at_well`` after the final poke-out of a stop
          that starts inside a visit.
        - ``well_visit_id`` (``Int64``): the visit the bin lies in; missing
          outside a visit.
        - ``visit_stretch_index`` (``Int64``): the stop's index among the
          stretches of the bin's visit (the stop table's); missing outside a
          visit, and for the bins of a stop in a visit that is not the stop's
          ``overlapping_visit_id`` (the earlier visit of a stop across two).
        - ``bin_start_s``, ``bin_end_s``: the bin's bounds (absolute s); the
          last bin of a piece of analyzed time ends with it.
        - ``exposure``: at-risk seconds, ``bin_end_s - bin_start_s`` less the
          time inside an event of the kind (> 0).
        - ``count``: the starts of the kind's events in the bin.
        - ``time_since_visit_start_s``, ``time_until_visit_end_s``,
          ``time_until_stretch_end_s``: at the bin's center ``c``,
          ``c - first poke-in``, ``final poke-out - c`` and ``min(stop_end,
          final poke-out) - c``; NaN outside a visit.
        - ``known_reward_outcome_lag_1..10``: the outcomes of the port
          experiences known at the bin's left edge, most recent first
          (``lagged_trial_rewards``; NaN where fewer are known); lag 1 is the
          last known outcome.

        ``attrs`` holds ``n_bins_not_at_risk`` (bins wholly inside an event of
        the kind, dropped), ``n_starts_not_at_risk`` (event starts of the kind
        in those bins: an event that starts at the very start of a piece of
        analyzed time has no at-risk time before it, and is not counted;
        ``starts_not_at_risk_s`` lists their start times) and
        ``n_stops_arrival_spanning`` (stops that contain a first poke-in after
        their start, ``stop_start < arrival <= stop_end``).

    Raises
    ------
    ValueError
        If ``stop_table`` holds no stop or more than one session, ``events``
        or ``excluded_intervals`` lacks the session columns or holds rows of
        other sessions only, ``kind`` is unknown, two
        events of the kind overlap, an event of the kind starts outside every
        stop's analyzed time, or an event's label disagrees with its bin's
        (the event and stop tables place the visits differently), or a stop's
        or an event's visit times disagree with the port experiences.

    Notes
    -----
    In-visit exposure plus the time of the kind's events inside in-visit
    analyzed time equals the sum of the stop table's ``in_visit_immobile_s``
    (up to the at-risk floor of 1e-5 s per dropped bin). A stop across two
    visits gives bins in both: those in the earlier visit carry its
    ``well_visit_id`` and no stretch index.
    """
    if kind is not None and kind not in EVENT_KINDS:
        raise ValueError(f"kind must be one of {sorted(EVENT_KINDS)} or None: {kind!r}")
    require_columns(stop_table, list(_STOP_COLUMNS), "stop_table (visit bins)")
    require_columns(
        events,
        [*_EVENT_COLUMNS, *(["is_swr"] if kind is not None else [])],
        "events (visit bins)",
    )
    sessions = stop_table[list(SESSION_COLUMNS)].drop_duplicates()
    if len(sessions) > 1:
        raise ValueError(
            f"visit_timebins builds one session; stop_table holds {len(sessions)}"
        )
    if not len(sessions):
        raise ValueError("visit_timebins needs the stops of one session; got none")
    events = _of_session(events, sessions, "events (visit bins)")
    if excluded_intervals is None:
        excluded_intervals = excluded_intervals_by_session(stop_table)
    excluded = _of_session(
        excluded_intervals, sessions, "excluded_intervals (visit bins)"
    )[["start_time", "end_time"]].to_numpy(dtype=float)

    stops = stop_table.sort_values("stop_start", kind="stable").reset_index(drop=True)
    stop_start = stops["stop_start"].to_numpy(dtype=float)
    stop_end = stops["stop_end"].to_numpy(dtype=float)
    visit_id, arrival, departure = _visit_intervals(port_experiences)
    _check_visit_times(stops, events, arrival, departure)

    # The analyzed time of the stops, cut at the visit boundaries.
    analyzed = subtract(np.column_stack([stop_start, stop_end]), excluded)
    visits = np.column_stack([arrival, departure])
    pieces = np.vstack([intersect(analyzed, visits), subtract(analyzed, visits)])
    pieces = pieces[np.argsort(pieces[:, 0], kind="stable")]
    piece_start, piece_end = pieces[:, 0], pieces[:, 1]
    piece_stop = np.searchsorted(stop_start, piece_start, side="right") - 1
    piece_visit = well_visit_index((piece_start + piece_end) / 2, arrival, departure)

    # Bins laid from each piece's start; the last one ends with the piece.
    length = piece_end - piece_start
    n_bins = np.maximum(np.ceil((length - _TIME_TOLERANCE_S) / bin_s), 1).astype(int)
    first_bin = np.cumsum(n_bins) - n_bins
    bin_piece = np.repeat(np.arange(len(pieces)), n_bins)
    binj = np.arange(len(bin_piece)) - first_bin[bin_piece]
    # Seconds from the piece's start, so the bins' lengths sum to the piece's.
    left = binj * bin_s
    right = np.where(binj == n_bins[bin_piece] - 1, length[bin_piece], left + bin_s)
    bin_start = piece_start[bin_piece] + left
    bin_end = piece_start[bin_piece] + right

    kind_events = (
        events if kind is None else events[events["is_swr"] == EVENT_KINDS[kind]]
    )
    kind_events = kind_events.sort_values("event_start_s", kind="stable")
    ev_start = kind_events["event_start_s"].to_numpy(dtype=float)
    ev_end = kind_events["event_end_s"].to_numpy(dtype=float)
    if np.any(ev_start[1:] < ev_end[:-1] - _TIME_TOLERANCE_S):
        raise ValueError(
            "events of one kind overlap; the at-risk time needs disjoint events"
        )

    covered = _covered_by_bin(
        ev_start, ev_end, piece_start, piece_end, n_bins, first_bin, bin_s=bin_s
    )
    exposure = (right - left) - covered

    # Each start in the bin that holds the at-risk time just before it.
    start_piece = np.searchsorted(piece_start, ev_start, side="right") - 1
    placed = (start_piece >= 0) & (ev_start < piece_end[np.clip(start_piece, 0, None)])
    if not placed.all():
        raise ValueError(
            f"{int((~placed).sum())} event(s) of the kind start outside every "
            "stop's analyzed time (first at "
            f"{ev_start[~placed][0]:.6f} s); the event and stop tables disagree"
        )
    offset = ev_start - piece_start[start_piece]
    start_binj = np.clip(
        np.ceil((offset - _TIME_TOLERANCE_S) / bin_s).astype(int) - 1,
        0,
        n_bins[start_piece] - 1,
    )
    start_bin = first_bin[start_piece] + start_binj
    count = np.bincount(start_bin, minlength=len(bin_start)).astype(float)

    in_visit_start = piece_visit[start_piece] >= 0
    labeled_in = kind_events["visit_context"].isin(STARTS_IN_WELL_VISIT).to_numpy()
    if np.any(in_visit_start != labeled_in):
        raise ValueError(
            f"{int((in_visit_start != labeled_in).sum())} event(s) of the kind "
            "have a visit_context that disagrees with whether their start lies "
            "inside a visit; the event and stop tables place the visits "
            "differently"
        )

    bin_stop = piece_stop[bin_piece]
    bin_visit = piece_visit[bin_piece]
    inside = bin_visit >= 0
    stop_label = stops["visit_context"].to_numpy(dtype=object)[bin_stop]
    label = np.where(
        inside,
        IN_WELL_VISIT,
        np.where(np.isin(stop_label, STARTS_IN_WELL_VISIT), _AFTER_VISIT, stop_label),
    )
    # The bin's visit; index -1 (outside every visit) reads the NaN pad.
    bin_arrival = np.append(arrival, np.nan)[bin_visit]
    bin_departure = np.append(departure, np.nan)[bin_visit]
    bin_visit_id = np.append(visit_id.astype(float), np.nan)[bin_visit]
    # The clocks at the bin's center, from differences of nearby times.
    center = (left + right) / 2
    origin = piece_start[bin_piece]
    since = (origin - bin_arrival) + center
    until = (bin_departure - origin) - center
    until_stretch = (np.minimum(stop_end[bin_stop], bin_departure) - origin) - center
    stop_visit = stops["overlapping_visit_id"].to_numpy(dtype=float, na_value=np.nan)
    own_visit = stop_visit[bin_stop] == bin_visit_id
    stretch = pd.array(
        stops["visit_stretch_index"].to_numpy(dtype=float, na_value=np.nan)[bin_stop],
        dtype="Int64",
    )
    stretch[~own_visit] = pd.NA

    # The outcomes known at each bin's left edge.
    lags = lagged_trial_rewards(bin_start, port_experiences, REWARD_HISTORY_N_LAGS)
    # A stop spans an arrival when a first poke-in lies in (stop_start, stop_end].
    spans_arrival = np.searchsorted(arrival, stop_end, side="right") > np.searchsorted(
        arrival, stop_start, side="right"
    )

    out = pd.DataFrame(
        {
            **{
                column: np.repeat(sessions[column].to_numpy(), len(bin_start))
                for column in SESSION_COLUMNS
            },
            "stop_id": stops["stop_id"].to_numpy()[bin_stop],
            "visit_context": label,
            "well_visit_id": pd.array(bin_visit_id, dtype="Int64"),
            "visit_stretch_index": stretch,
            "bin_start_s": bin_start,
            "bin_end_s": bin_end,
            "exposure": exposure,
            "count": count,
            "time_since_visit_start_s": since,
            "time_until_visit_end_s": until,
            "time_until_stretch_end_s": until_stretch,
            **{column: lags[:, j] for j, column in enumerate(REWARD_LAG_COLUMNS)},
        }
    )
    at_risk = exposure > _TIME_TOLERANCE_S
    lost = ~at_risk[start_bin]
    out = out.loc[at_risk].reset_index(drop=True)
    out.attrs["n_bins_not_at_risk"] = int((~at_risk).sum())
    out.attrs["n_starts_not_at_risk"] = int(lost.sum())
    out.attrs["starts_not_at_risk_s"] = ev_start[lost].tolist()
    out.attrs["n_stops_arrival_spanning"] = int(spans_arrival.sum())
    return out


def _of_session(frame: pd.DataFrame, sessions: pd.DataFrame, name: str) -> pd.DataFrame:
    """The rows of ``frame`` in the one session of ``sessions``.

    The session columns are compared as text, so an integer ``session_date``
    matches its string.

    Raises
    ------
    ValueError
        If ``frame`` lacks a session column, or is not empty but holds no row
        of the session (the frames name the session differently).
    """
    require_columns(frame, list(SESSION_COLUMNS), name)
    key = sessions.iloc[0]
    keep = np.logical_and.reduce(
        [
            frame[column].astype(str).to_numpy() == str(key[column])
            for column in SESSION_COLUMNS
        ]
    )
    if len(frame) and not keep.any():
        raise ValueError(
            f"{name} holds {len(frame)} row(s) and no row of the session "
            f"{tuple(str(key[column]) for column in SESSION_COLUMNS)}; pass an "
            "empty frame for a session with none"
        )
    return frame.loc[keep]


def _visit_intervals(
    port_experiences: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The session's visits, from its port experiences.

    Returns
    -------
    visit_id : np.ndarray of int, shape (n_visits,)
        ``experience_id``, 0, 1, ... in time order.
    arrival, departure : np.ndarray, shape (n_visits,)
        First poke-in and final poke-out (absolute s).
    """
    experiences = _prepare_port_experiences(port_experiences)
    return (
        experiences["experience_id"].to_numpy(dtype=np.int64),
        experiences["arrival_time"].to_numpy(dtype=float),
        experiences["departure_time"].to_numpy(dtype=float),
    )


def _check_visit_times(
    stops: pd.DataFrame,
    events: pd.DataFrame,
    arrival: np.ndarray,
    departure: np.ndarray,
) -> None:
    """Raise unless the stops' and events' visits are the port experiences'.

    A stop's ``overlapping_visit_id`` must name a visit with its
    ``visit_first_poke_in_s`` and ``visit_last_poke_out_s``, and an event's
    ``well_visit_id`` a visit with the poke times its clocks give.
    """
    stop_visit = stops["overlapping_visit_id"].to_numpy(dtype=float, na_value=np.nan)
    event_visit = events["well_visit_id"].to_numpy(dtype=float, na_value=np.nan)
    start = events["event_start_s"].to_numpy(dtype=float)
    checks = {
        "stop": (
            stop_visit,
            stops["visit_first_poke_in_s"].to_numpy(dtype=float),
            stops["visit_last_poke_out_s"].to_numpy(dtype=float),
        ),
        "event": (
            event_visit,
            start - events["time_since_visit_start_s"].to_numpy(dtype=float),
            start + events["time_until_visit_end_s"].to_numpy(dtype=float),
        ),
    }
    for name, (visit, first_in, last_out) in checks.items():
        placed = ~np.isnan(visit)
        index = visit[placed].astype(np.int64)
        known = (index >= 0) & (index < len(arrival))
        bad = ~known
        bad[known] = (
            np.abs(first_in[placed][known] - arrival[index[known]]) > _TIME_TOLERANCE_S
        ) | (
            np.abs(last_out[placed][known] - departure[index[known]])
            > _TIME_TOLERANCE_S
        )
        if bad.any():
            raise ValueError(
                f"{int(bad.sum())} {name}(s) name a visit whose poke times "
                "disagree with the port experiences; the tables and the port "
                "experiences are of different builds"
            )


def _covered_by_bin(
    ev_start: np.ndarray,
    ev_end: np.ndarray,
    piece_start: np.ndarray,
    piece_end: np.ndarray,
    n_bins: np.ndarray,
    first_bin: np.ndarray,
    *,
    bin_s: float,
) -> np.ndarray:
    """Seconds of each bin inside an event, shape (n_bins_total,)."""
    # The pieces each event overlaps: those that end after it starts and start
    # before it ends (pieces are disjoint and in time order).
    lo = np.searchsorted(piece_end, ev_start, side="right")
    hi = np.searchsorted(piece_start, ev_end, side="left")
    n_pairs = np.maximum(hi - lo, 0)
    event = np.repeat(np.arange(len(ev_start)), n_pairs)
    piece = lo[event] + (
        np.arange(len(event)) - np.repeat(np.cumsum(n_pairs) - n_pairs, n_pairs)
    )
    cover = interval_coverage_by_bin(
        piece,
        ev_start[event] - piece_start[piece],
        ev_end[event] - piece_start[piece],
        piece_end[piece] - piece_start[piece],
        bin_s=bin_s,
    )
    segment = cover["segment"].to_numpy(dtype=np.int64)
    # The last bin of a piece runs to the piece's end, past a whole bin.
    binj = np.minimum(cover["binj"].to_numpy(), n_bins[segment] - 1)
    covered = np.zeros(int(n_bins.sum()))
    np.add.at(covered, first_bin[segment] + binj, cover["covered_s"].to_numpy())
    return covered
