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

from continuum_swr_replay.data_loaders.constants import LEAF_ARMS, TRUNK_ARMS

from ..behavior.covariates.history._common import _prepare_port_experiences
from ..constants import (
    IN_WELL_VISIT,
    VISIT_JUNCTION_RADIUS_CM,
    VISIT_WELL_RADIUS_CM,
)

__all__ = [
    "VISIT_CONTEXT_COLUMNS",
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
        ``well_visit_id`` (``Int64``), the ``experience_id`` of the visit an
        ``in_visit`` window lies in, missing otherwise;
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
    current = np.searchsorted(arrivals, start, side="right") - 1
    last_done = np.searchsorted(departures, start, side="right") - 1
    upcoming = np.searchsorted(arrivals, end, side="right")
    has_current = timed & (current >= 0)
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

    in_visit = context == IN_WELL_VISIT
    well_visit_id = pd.array([pd.NA] * n_windows, dtype="Int64")
    well_visit_id[in_visit & has_current] = visit_ids[current[in_visit & has_current]]
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
