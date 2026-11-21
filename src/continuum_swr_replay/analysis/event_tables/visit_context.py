"""Where an event or a stop lies relative to the well visits.

A well visit is one port experience: from the animal's first nose-poke in at a
reward well (``arrival_time``) to its final poke-out there (``departure_time``).
:func:`window_visit_context` labels a time window (an event, or a stop's
analyzed rows) ``in_visit`` when it lies wholly inside one visit, and otherwise
says where it lies: across a visit boundary, before the first visit, or between
two visits, placed by the animal's projected position (next to the well just
left, next to the next well, next to another well, or on the track). The
labels and the radii are in ``analysis.constants`` (:data:`VISIT_CONTEXTS`).

Nothing is dropped here: every event and every stop keeps its row and gets a
label, and ``event_definition.cohort_selection.select_visit_context`` selects
by it.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_definition.well_visits import (
    well_visit_index,
)
from continuum_swr_replay.data_loaders.constants import LEAF_ARMS, TRUNK_ARMS
from continuum_swr_replay.intervals import as_frame, covered_seconds, subtract

from ..behavior.covariates.history._common import _prepare_port_experiences
from ..behavior.spatial_mapping import arm_to_patch, arms_of_wells, stay_indicator
from ..constants import (
    IN_WELL_VISIT,
    STARTS_IN_WELL_VISIT,
    VISIT_JUNCTION_RADIUS_CM,
    VISIT_WELL_RADIUS_CM,
)

__all__ = [
    "EVENT_VISIT_COLUMNS",
    "STOP_VISIT_COLUMNS",
    "VISIT_CONTEXT_COLUMNS",
    "assign_visit_current_arm",
    "event_visit_columns",
    "stop_visit_columns",
    "visit_context_of_samples",
    "window_visit_context",
]

#: The columns :func:`window_visit_context` returns, in table order.
VISIT_CONTEXT_COLUMNS: tuple[str, ...] = (
    "visit_context",
    "well_visit_id",
    "last_completed_well_visit_id",
    "time_since_last_poke_out_s",
    "time_until_next_poke_in_s",
)

#: The stop table's columns of the visit a stop overlaps
#: (:func:`stop_visit_columns`), in table order.
STOP_VISIT_COLUMNS: tuple[str, ...] = (
    "overlapping_visit_id",
    "visit_first_poke_in_s",
    "visit_last_poke_out_s",
    "in_visit_immobile_s",
    "visit_stretch_index",
    "stop_end_kind",
)

#: The event table's visit clocks and trial number (:func:`event_visit_columns`),
#: in table order; carried to the event-arm and event-patch tables.
EVENT_VISIT_COLUMNS: tuple[str, ...] = (
    "time_since_visit_start_s",
    "time_until_visit_end_s",
    "time_until_stretch_end_s",
    "n_visits_arrived",
)

#: Contexts of a window wholly between two visits (or before the first one),
#: the rows whose time since the last poke-out and until the next poke-in are
#: defined.
_OUTSIDE_VISIT_CONTEXTS = (
    "after_visit_at_well",
    "before_visit_at_well",
    "at_other_well",
    "track_leaf_arm",
    "track_trunk_arm",
    "track_patch_junction",
    "track_center",
    "track_no_position",
    "before_first_visit",
)


def _node_positions(track_graph: nx.Graph, nodes: list) -> np.ndarray:
    """``(x, y)`` of each node, NaN for a node without a ``pos``; shape (n, 2)."""
    positions = nx.get_node_attributes(track_graph, "pos")
    return np.asarray(
        [positions.get(node, (np.nan, np.nan)) for node in nodes], dtype=float
    ).reshape(len(nodes), 2)


def _distance_to(points: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Euclidean distance of each point to each target; shape (n_points, n_targets)."""
    return np.linalg.norm(points[:, None, :] - targets[None, :, :], axis=2)


def window_visit_context(
    window_start: np.ndarray,
    window_end: np.ndarray,
    occupancy_state: np.ndarray,
    first_sample_state: np.ndarray,
    reference_position: np.ndarray,
    reference_segment: np.ndarray,
    port_experiences: pd.DataFrame,
    track_graph: nx.Graph,
) -> pd.DataFrame:
    """Label each window by where it lies relative to the well visits.

    Parameters
    ----------
    window_start, window_end : np.ndarray, shape (n_windows,)
        Times (s) of each window's first and last sample; NaN for a window
        with no samples.
    occupancy_state : np.ndarray of str, shape (n_windows,)
        The window's ``occupancy_state`` (``summarize_behavioral_location``):
        ``at_well``, ``mixed``, ``in_transit`` or ``unknown``.
    first_sample_state : np.ndarray of object, shape (n_windows,)
        The per-sample occupancy state of the window's first sample; ``None``
        for a window with no samples.
    reference_position : np.ndarray, shape (n_windows, 2)
        Projected ``(x, y)`` (cm) at the window's middle sample, NaN where
        missing.
    reference_segment : np.ndarray, shape (n_windows,)
        ``track_segment_id`` at the window's middle sample, NaN where missing.
    port_experiences : pd.DataFrame
        The session's port experiences (one per visit, ``experience_id``
        0, 1, ... in time order), as the table builders take them.
    track_graph : nx.Graph
        Session track graph; wells are its degree-one nodes (node id = well
        id), junctions the others, and the center the junction with no well
        neighbor.

    Returns
    -------
    pd.DataFrame, shape (n_windows, 5)
        ``visit_context`` (str, one of ``analysis.constants.VISIT_CONTEXTS``);
        ``well_visit_id`` (``Int64``), the ``experience_id`` of the visit that
        holds the window start, for a window that starts inside a visit
        (``in_visit`` or ``spans_visit_end``,
        ``analysis.constants.STARTS_IN_WELL_VISIT``), missing otherwise;
        ``last_completed_well_visit_id`` (``Int64``), the latest visit whose
        final poke-out is at or before the window start, missing before the
        first; ``time_since_last_poke_out_s``, window start less that
        poke-out, and ``time_until_next_poke_in_s``, the first poke-in after
        the window end less the window end (s). Both times are defined only
        for windows wholly outside a visit (missing for ``in_visit``,
        ``spans_*`` and ``no_analyzed_samples`` rows, and where no such
        poke exists).
    """
    start = np.asarray(window_start, dtype=float)
    end = np.asarray(window_end, dtype=float)
    state = np.asarray(occupancy_state, dtype=object)
    first_state = np.asarray(first_sample_state, dtype=object)
    position = np.asarray(reference_position, dtype=float).reshape(len(start), 2)
    segment = np.asarray(reference_segment, dtype=float)
    n_windows = len(start)

    experiences = _prepare_port_experiences(port_experiences)
    arrivals = experiences["arrival_time"].to_numpy(dtype=float)
    departures = experiences["departure_time"].to_numpy(dtype=float)
    ports = experiences["port"].to_numpy(dtype=float)
    visit_ids = experiences["experience_id"].to_numpy(dtype=np.int64)
    n_visits = len(experiences)

    def _pick(values: np.ndarray, index: np.ndarray, valid: np.ndarray):
        out = np.full(n_windows, np.nan)
        if n_visits:
            out[valid] = values[index[valid]]
        return out

    # A window with no samples has no times, so no visit it could be placed by.
    timed = np.isfinite(start) & np.isfinite(end)
    last_done = np.searchsorted(departures, start, side="right") - 1
    upcoming = np.searchsorted(arrivals, end, side="right")
    has_done = timed & (last_done >= 0)
    has_upcoming = timed & (upcoming < n_visits)
    last_done_well = _pick(ports, last_done, has_done)
    upcoming_well = _pick(ports, upcoming, has_upcoming)

    wells = [node for node in track_graph.nodes if track_graph.degree(node) == 1]
    junctions = [node for node in track_graph.nodes if track_graph.degree(node) > 1]
    center = {
        node
        for node in junctions
        if not any(track_graph.degree(n) == 1 for n in track_graph.neighbors(node))
    }
    well_distance = _distance_to(position, _node_positions(track_graph, wells))
    junction_distance = _distance_to(position, _node_positions(track_graph, junctions))
    well_ids = np.asarray(wells, dtype=float)

    def _near_well(well: np.ndarray) -> np.ndarray:
        near = np.zeros(n_windows, dtype=bool)
        for column, well_id in enumerate(well_ids):
            rows = well == well_id
            near[rows] = well_distance[rows, column] <= VISIT_WELL_RADIUS_CM
        return near

    near_previous = _near_well(last_done_well)
    near_next = _near_well(upcoming_well) & ~near_previous
    near_other = (
        (np.nanmin(well_distance, axis=1, initial=np.inf) <= VISIT_WELL_RADIUS_CM)
        & ~near_previous
        & ~near_next
    )
    nearest_junction = np.full(n_windows, -1, dtype=np.int64)
    finite_junction = np.isfinite(junction_distance).any(axis=1)
    if junctions:
        nearest_junction[finite_junction] = np.asarray(junctions)[
            np.nanargmin(junction_distance[finite_junction], axis=1)
        ]
    at_junction = finite_junction & (
        np.nanmin(junction_distance, axis=1, initial=np.inf) <= VISIT_JUNCTION_RADIUS_CM
    )
    at_center = at_junction & np.isin(nearest_junction, list(center))

    context = np.full(n_windows, "track_no_position", dtype=object)
    finite_segment = np.isfinite(segment)
    context[finite_segment & np.isin(segment, LEAF_ARMS)] = "track_leaf_arm"
    context[finite_segment & np.isin(segment, TRUNK_ARMS)] = "track_trunk_arm"
    context[finite_segment & at_junction] = "track_patch_junction"
    context[finite_segment & at_center] = "track_center"
    context[near_other] = "at_other_well"
    context[near_next] = "before_visit_at_well"
    context[near_previous] = "after_visit_at_well"
    in_transit = state == "in_transit"
    context[~in_transit] = None
    context[state == "unknown"] = "before_first_visit"
    mixed = state == "mixed"
    context[mixed] = np.where(
        first_state[mixed] == "at_well", "spans_visit_end", "spans_visit_start"
    )
    context[state == "at_well"] = IN_WELL_VISIT
    context[pd.isna(first_state)] = "no_analyzed_samples"
    if pd.isna(context).any():
        raise ValueError(
            f"occupancy states {sorted(set(state[pd.isna(context)]))} have no "
            "visit context"
        )

    starts_inside = np.isin(context, STARTS_IN_WELL_VISIT)
    holding = well_visit_index(start, arrivals, departures)
    if (holding[starts_inside] < 0).any():
        raise ValueError(
            "a window labeled as starting inside a visit starts outside every "
            "visit; its occupancy states disagree with the port experiences"
        )
    well_visit_id = pd.array([pd.NA] * n_windows, dtype="Int64")
    well_visit_id[starts_inside] = visit_ids[holding[starts_inside]]
    last_completed = pd.array([pd.NA] * n_windows, dtype="Int64")
    last_completed[has_done] = visit_ids[last_done[has_done]]

    outside = np.isin(context, _OUTSIDE_VISIT_CONTEXTS)
    since = start - _pick(departures, last_done, has_done)
    until = _pick(arrivals, upcoming, has_upcoming) - end
    return pd.DataFrame(
        {
            "visit_context": context,
            "well_visit_id": well_visit_id,
            "last_completed_well_visit_id": last_completed,
            "time_since_last_poke_out_s": np.where(outside, since, np.nan),
            "time_until_next_poke_in_s": np.where(outside, until, np.nan),
        }
    )


def visit_context_of_samples(
    samples: pd.DataFrame,
    sample_window: np.ndarray,
    windows: np.ndarray,
    occupancy_state: np.ndarray,
    port_experiences: pd.DataFrame,
    track_graph: nx.Graph,
) -> pd.DataFrame:
    """:func:`window_visit_context` of windows given by their samples.

    Each window's first and last sample give its times, its first sample the
    side of a boundary it starts on, and its middle sample (the
    ``(n - 1) // 2``-th of its ``n``, in time order) the animal's position.

    Parameters
    ----------
    samples : pd.DataFrame
        Time-indexed (s), sorted samples with the per-sample
        ``occupancy_state`` of ``add_trial_context`` and, when the frame has
        them, ``projected_x_position``, ``projected_y_position`` and
        ``track_segment_id`` (NaN when absent).
    sample_window : np.ndarray, shape (len(samples),)
        The window each sample belongs to.
    windows : np.ndarray, shape (n_windows,)
        The windows to label, in output order. A window with no sample is
        ``no_analyzed_samples``.
    occupancy_state : np.ndarray of str, shape (n_windows,)
        Each window's summarized ``occupancy_state``.
    port_experiences, track_graph
        As :func:`window_visit_context`.

    Returns
    -------
    pd.DataFrame, shape (n_windows, 5)
        :data:`VISIT_CONTEXT_COLUMNS`, one row per entry of ``windows``.
    """
    sample_window = np.asarray(sample_window)
    order = np.argsort(sample_window, kind="stable")
    grouped, first, count = np.unique(
        sample_window[order], return_index=True, return_counts=True
    )
    first_row = order[first]
    middle_row = order[first + (count - 1) // 2]
    last_row = order[first + count - 1]

    windows = np.asarray(windows)
    n_windows = len(windows)
    found = np.clip(np.searchsorted(grouped, windows), 0, max(len(grouped) - 1, 0))
    has_samples = np.zeros(n_windows, dtype=bool)
    if len(grouped):
        has_samples = grouped[found] == windows

    def _at(rows: np.ndarray, values: np.ndarray) -> np.ndarray:
        """``values`` at each window's row; NaN (None for objects) without one."""
        missing = None if values.dtype == object else np.nan
        out = np.full(n_windows, missing, dtype=values.dtype)
        out[has_samples] = values[rows[found[has_samples]]]
        return out

    def _column(name: str) -> np.ndarray:
        if name not in samples.columns:
            return np.full(len(samples), np.nan)
        return samples[name].to_numpy(dtype=float, na_value=np.nan)

    times = samples.index.to_numpy(dtype=float)
    states = samples["occupancy_state"].astype(object).to_numpy()
    position = np.column_stack(
        [
            _at(middle_row, _column("projected_x_position")),
            _at(middle_row, _column("projected_y_position")),
        ]
    )
    return window_visit_context(
        window_start=_at(first_row, times),
        window_end=_at(last_row, times),
        occupancy_state=occupancy_state,
        first_sample_state=_at(first_row, states),
        reference_position=position,
        reference_segment=_at(middle_row, _column("track_segment_id")),
        port_experiences=port_experiences,
        track_graph=track_graph,
    )


def _visit_intervals(port_experiences: pd.DataFrame) -> tuple[np.ndarray, ...]:
    """Each visit's first poke-in, final poke-out and port, in visit order.

    ``experience_id`` runs 0, 1, ... in this order, so a visit id indexes
    these arrays.
    """
    experiences = _prepare_port_experiences(port_experiences)
    return (
        experiences["arrival_time"].to_numpy(dtype=float),
        experiences["departure_time"].to_numpy(dtype=float),
        experiences["port"].to_numpy(dtype=float),
    )


def stop_visit_columns(
    stop_start: np.ndarray,
    stop_end: np.ndarray,
    port_experiences: pd.DataFrame,
    excluded: pd.DataFrame,
) -> pd.DataFrame:
    """The well visit each stop overlaps, its times, and how the stop ends.

    A visit runs from its first poke-in (``arrival_time``, included) to its
    final poke-out (``departure_time``, not included); a stop runs from its
    first to its last immobile sample. The stops of one visit are the stretches
    of immobility between the small movements the animal makes at the well;
    the last of them ends with the animal leaving.

    Parameters
    ----------
    stop_start, stop_end : np.ndarray, shape (n_stops,)
        Each stop's first and last immobile sample (s), in time order.
    port_experiences : pd.DataFrame
        The session's port experiences (one per visit).
    excluded : pd.DataFrame
        The epoch's excluded time (``analyzed_time.excluded_intervals``):
        ``start_time`` and ``end_time``.

    Returns
    -------
    pd.DataFrame, shape (n_stops, 6)
        :data:`STOP_VISIT_COLUMNS`:

        - ``overlapping_visit_id`` (``Int64``): the latest visit the stop
          overlaps at all (the visit it ends in or ended after), so a stop that
          starts before the first poke-in is placed too; missing when the stop
          overlaps no visit. Where ``well_visit_id`` is set, the two are equal.
        - ``visit_first_poke_in_s``, ``visit_last_poke_out_s``: that visit's
          ``arrival_time`` and ``departure_time`` (s); NaN without a visit.
        - ``in_visit_immobile_s``: seconds of the stop inside the well visits,
          less excluded time; 0.0 for a stop outside every visit. A stop that
          crosses the final poke-out counts up to it.
        - ``visit_stretch_index`` (``Int64``): 0, 1, ... in time order for the
          stretches of one visit: the stops placed in it with analyzed seconds
          inside it (outside excluded time, from the first poke-in to the
          final poke-out); missing for every other stop, so also for a stop
          that only touches the visit (one that ends on the first poke-in) or
          lies wholly in excluded time.
        - ``stop_end_kind``: ``"leaves"`` for the visit's last stretch (no
          later stretch of the visit starts before its final poke-out),
          ``"small_movement"`` for its other stretches, ``"outside_visit"`` for
          a stop that is no stretch of a visit.

    Notes
    -----
    A stop overlaps at most one visit when the animal moves between wells. A
    stop that overlaps two (a stop at the start of an epoch, across the first
    visits at the two wells of one patch) is placed in the later one: its
    ``overlapping_visit_id``, poke times, stretch index and end kind are the
    later visit's (a stretch of it only with analyzed seconds in it), the
    earlier visit gets no stretch from it, and its ``in_visit_immobile_s``
    counts its seconds in both.
    """
    start = np.asarray(stop_start, dtype=float)
    end = np.asarray(stop_end, dtype=float)
    n_stops = len(start)
    arrivals, departures, _ = _visit_intervals(port_experiences)

    # The last visit begun by the stop's end; -1 (none) reads the -inf pad.
    latest = np.searchsorted(arrivals, end, side="right") - 1
    overlaps = start < np.append(departures, -np.inf)[latest]
    visit = np.where(overlaps, latest, -1)
    first_poke_in = np.full(n_stops, np.nan)
    last_poke_out = np.full(n_stops, np.nan)
    first_poke_in[overlaps] = arrivals[visit[overlaps]]
    last_poke_out[overlaps] = departures[visit[overlaps]]

    analyzed_visits = as_frame(
        subtract(
            np.column_stack([arrivals, departures]),
            excluded[["start_time", "end_time"]].to_numpy(dtype=float),
        )
    )
    in_visit = covered_seconds(start, end, analyzed_visits)
    # The stop's analyzed seconds inside the visit it is placed in: only a stop
    # with some is one of that visit's stretches.
    lo = np.where(overlaps, np.maximum(start, first_poke_in), 0.0)
    hi = np.where(overlaps, np.minimum(end, last_poke_out), 0.0)
    hi = np.maximum(hi, lo)
    in_own_visit = covered_seconds(lo, hi, analyzed_visits)
    stretches = overlaps & (in_own_visit > 0)

    stretch = (
        pd.Series(start[stretches])
        .groupby(visit[stretches])
        .rank(method="first")
        .to_numpy(dtype=np.int64)
        - 1
    )
    n_stretches = np.bincount(visit[stretches], minlength=len(arrivals))
    stretch_index = pd.array([pd.NA] * n_stops, dtype="Int64")
    stretch_index[stretches] = stretch
    end_kind = np.full(n_stops, "outside_visit", dtype=object)
    end_kind[stretches] = np.where(
        stretch == n_stretches[visit[stretches]] - 1, "leaves", "small_movement"
    )
    overlapping_visit_id = pd.array([pd.NA] * n_stops, dtype="Int64")
    overlapping_visit_id[overlaps] = visit[overlaps]
    return pd.DataFrame(
        {
            "overlapping_visit_id": overlapping_visit_id,
            "visit_first_poke_in_s": first_poke_in,
            "visit_last_poke_out_s": last_poke_out,
            "in_visit_immobile_s": in_visit,
            "visit_stretch_index": stretch_index,
            "stop_end_kind": end_kind,
        }
    )


def event_visit_columns(
    event_start: np.ndarray,
    stop_end: np.ndarray,
    well_visit_id: pd.api.extensions.ExtensionArray | np.ndarray,
    port_experiences: pd.DataFrame,
) -> pd.DataFrame:
    """The visit clocks of each event and the visits arrived at by its start.

    Parameters
    ----------
    event_start : np.ndarray, shape (n_events,)
        Each event's start (s).
    stop_end : np.ndarray, shape (n_events,)
        The last immobile sample (s) of the stop that holds the event.
    well_visit_id : array of Int64, shape (n_events,)
        The visit the event starts in (``window_visit_context``), missing for
        an event that does not start inside a visit.
    port_experiences : pd.DataFrame
        The session's port experiences (one per visit).

    Returns
    -------
    pd.DataFrame, shape (n_events, 4)
        :data:`EVENT_VISIT_COLUMNS`. For an event that starts inside a visit
        (``in_visit`` or ``spans_visit_end``), in seconds:
        ``time_since_visit_start_s`` (event start less the first poke-in),
        ``time_until_visit_end_s`` (the final poke-out less the event start,
        positive), ``time_until_stretch_end_s`` (the end of the stop that holds
        the event, clipped at the final poke-out, less the event start); NaN
        for every other event. ``n_visits_arrived`` (``int64``): the port
        experiences whose first poke-in is at or before the event start, the
        starting port included (the trial number in the epoch, counted in
        visits).
    """
    start = np.asarray(event_start, dtype=float)
    stop_end = np.asarray(stop_end, dtype=float)
    arrivals, departures, _ = _visit_intervals(port_experiences)
    visit_id = pd.array(well_visit_id, dtype="Int64")
    inside = ~np.asarray(visit_id.isna())
    index = visit_id.to_numpy(dtype=np.int64, na_value=0)
    since = np.full(len(start), np.nan)
    until = np.full(len(start), np.nan)
    stretch = np.full(len(start), np.nan)
    since[inside] = start[inside] - arrivals[index[inside]]
    until[inside] = departures[index[inside]] - start[inside]
    stretch[inside] = (
        np.minimum(stop_end[inside], departures[index[inside]]) - start[inside]
    )
    return pd.DataFrame(
        {
            "time_since_visit_start_s": since,
            "time_until_visit_end_s": until,
            "time_until_stretch_end_s": stretch,
            "n_visits_arrived": np.searchsorted(arrivals, start, side="right").astype(
                np.int64
            ),
        }
    )


def assign_visit_current_arm(
    per_event: pd.DataFrame,
    port_experiences: pd.DataFrame,
    track_graph: nx.Graph,
) -> pd.DataFrame:
    """Give an event that crosses the final poke-out its visit's arm and patch.

    The whole-window location of a ``spans_visit_end`` event is ``mixed``, so
    its window gives no current arm. It starts inside a visit, and an event
    belongs to the time when it starts, so its current arm
    (``occupancy_well_arm``) and patch (``current_patch``) are those of the
    visit's well, and ``is_stay`` compares the next arm with that arm. Every
    other event keeps its own. ``occupancy_well_id`` stays the whole-window
    location (missing for these events).

    Parameters
    ----------
    per_event : pd.DataFrame
        One row per event, with ``visit_context``, ``well_visit_id``,
        ``occupancy_well_arm``, ``current_patch``, ``next_arm`` and ``is_stay``.
    port_experiences : pd.DataFrame
        The session's port experiences (one per visit).
    track_graph : nx.Graph
        Session track graph.

    Returns
    -------
    pd.DataFrame
        ``per_event``, changed in place on its ``spans_visit_end`` rows.
    """
    crossing = per_event["visit_context"].eq("spans_visit_end").to_numpy()
    if not crossing.any():
        return per_event
    _, _, ports = _visit_intervals(port_experiences)
    visit_id = per_event["well_visit_id"].to_numpy(dtype=np.int64, na_value=-1)
    arms = arms_of_wells(ports[visit_id[crossing]], track_graph)
    rows = per_event.index[crossing]
    per_event.loc[rows, "occupancy_well_arm"] = arms
    per_event.loc[rows, "current_patch"] = [arm_to_patch(arm) for arm in arms]
    per_event.loc[rows, "is_stay"] = [
        stay_indicator(next_arm, arm)
        for next_arm, arm in zip(
            per_event.loc[rows, "next_arm"].to_numpy(dtype=float), arms, strict=True
        )
    ]
    return per_event
