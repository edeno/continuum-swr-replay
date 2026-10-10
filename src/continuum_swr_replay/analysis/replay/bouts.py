"""Immobility bouts: where a stop begins and ends, and what falls inside one.

The stop grain of every table starts here. A bout is a maximal run of
timebins below the immobility speed threshold; events are placed in one by
timestamp, and their time since the bout began is the point-process clock the
per-bin rate models are built on. ``nearest_well_distance_cm`` is the physical
companion: how far the animal actually was from a reward port while stopped.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
)
from continuum_swr_replay.analysis.hippocampus.artifacts.masks import covered_seconds

__all__ = [
    "assign_bout",
    "bout_exposure_s",
    "immobility_bout_intervals",
    "nearest_well_distance_cm",
    "time_since_stop",
]


def nearest_well_distance_cm(
    dot: pd.DataFrame,
    track_graph: nx.Graph,
) -> np.ndarray:
    """Euclidean distance from each projected position to the nearest well.

    Wells are the degree-one nodes of ``track_graph``.  A malformed track
    graph raises: an all-NaN column is indistinguishable from "the animal
    was never near a well", and would silently empty any downstream
    near-a-well cohort for the whole session.  Missing position *data*
    (absent coordinate columns, individual non-finite rows) is a genuine
    data gap and still yields NaN.

    Parameters
    ----------
    dot : pd.DataFrame
        Timebin frame; uses ``projected_x_position`` / ``projected_y_position``.
    track_graph : nx.Graph
        Track graph whose degree-one nodes are wells, each with a ``pos``
        attribute holding a finite ``(x, y)`` pair.

    Returns
    -------
    np.ndarray, shape (len(dot),)
        Distance in cm, NaN where the position is unavailable.

    Raises
    ------
    ValueError
        If the graph has no degree-one nodes, or any well lacks a finite
        two-element ``pos``.
    """
    n_rows = len(dot)
    coordinate_columns = ("projected_x_position", "projected_y_position")
    if not set(coordinate_columns).issubset(dot.columns):
        return np.full(n_rows, np.nan)

    well_nodes = [node for node in track_graph.nodes if track_graph.degree(node) == 1]
    if not well_nodes:
        raise ValueError(
            "track_graph has no degree-one nodes, so no wells can be located"
        )
    node_positions = nx.get_node_attributes(track_graph, "pos")
    missing_pos = [node for node in well_nodes if node not in node_positions]
    if missing_pos:
        raise ValueError(
            f"track_graph well nodes lack a 'pos' attribute: {missing_pos}"
        )

    try:
        well_positions = np.asarray(
            [node_positions[node] for node in well_nodes], dtype=float
        )
    except (TypeError, ValueError) as err:
        raise ValueError(
            "track_graph well 'pos' attributes are not numeric (x, y) pairs"
        ) from err
    if (
        well_positions.shape != (len(well_nodes), 2)
        or not np.isfinite(well_positions).all()
    ):
        raise ValueError(
            "track_graph well 'pos' attributes must be finite two-element "
            f"(x, y) pairs; got shape {well_positions.shape}"
        )

    positions = dot.loc[:, coordinate_columns].to_numpy(dtype=float)
    output = np.full(n_rows, np.nan)
    valid = np.isfinite(positions).all(axis=1)
    if valid.any():
        displacement = positions[valid, None, :] - well_positions[None, :, :]
        output[valid] = np.linalg.norm(displacement, axis=2).min(axis=1)
    return output


def immobility_bout_intervals(
    dot: pd.DataFrame,
    speed_col: str = "head_speed",
    speed_threshold: float = DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
) -> tuple[np.ndarray, np.ndarray]:
    """Start and end timestamps for every contiguous immobility bout.

    Returns
    -------
    starts : np.ndarray, shape (n_bouts,)
    ends : np.ndarray, shape (n_bouts,)
    """
    imm = (dot[speed_col] < speed_threshold).to_numpy()
    times = dot.index.to_numpy()
    if imm.size == 0 or not imm.any():
        return np.array([]), np.array([])

    padded = np.concatenate(([False], imm, [False]))
    diff = np.diff(padded.astype(np.int8))
    start_idx = np.where(diff == 1)[0]
    end_idx = np.where(diff == -1)[0] - 1
    return times[start_idx], times[end_idx]


def bout_exposure_s(
    bout_starts: np.ndarray, bout_ends: np.ndarray, artifacts: pd.DataFrame
) -> np.ndarray:
    """Each bout's observed immobility: its span less the masked time in it.

    Artifact time is unobserved, so it is no rate's exposure. The bout itself
    (its start, end and ``stop_id``) is not changed; only its exposure falls.

    Parameters
    ----------
    bout_starts, bout_ends : np.ndarray, shape (n_bouts,)
        Seconds, from :func:`immobility_bout_intervals`.
    artifacts : pd.DataFrame
        The epoch's artifact mask: ``start_time`` and ``end_time``, closed.

    Returns
    -------
    np.ndarray, shape (n_bouts,)
        Seconds; ``bout_ends - bout_starts`` itself when nothing is masked.
    """
    span = bout_ends - bout_starts
    if len(artifacts) == 0:
        return span
    exposure: np.ndarray = span - covered_seconds(bout_starts, bout_ends, artifacts)
    return exposure


def time_since_stop(
    event_times: np.ndarray,
    bout_start_times: np.ndarray,
) -> np.ndarray:
    """Seconds from most recent immobility-bout start to each event time."""
    event_times = np.asarray(event_times, dtype=float)
    if bout_start_times.size == 0:
        return np.full_like(event_times, np.nan)
    idx = np.searchsorted(bout_start_times, event_times, side="right") - 1
    idx = np.clip(idx, 0, bout_start_times.size - 1)
    return np.clip(event_times - bout_start_times[idx], 0, None)


def assign_bout(
    event_times: np.ndarray,
    bout_starts: np.ndarray,
    bout_ends: np.ndarray,
) -> np.ndarray:
    """Return the bout index containing each timestamp, or ``-1`` if none."""
    event_times = np.asarray(event_times, dtype=float)
    if bout_starts.size == 0:
        return np.full(event_times.shape, -1, dtype=np.int64)
    idx = np.searchsorted(bout_starts, event_times, side="right") - 1
    clipped = np.clip(idx, 0, bout_starts.size - 1)
    in_bout = (idx >= 0) & (event_times <= bout_ends[clipped])
    return np.where(in_bout, idx, -1).astype(np.int64)
