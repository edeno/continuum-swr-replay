"""Immobility bouts: where a stop begins and ends, and what falls inside one.

The stop grain of every table starts here. A bout is a maximal run of
timebins below the immobility speed threshold; events are placed in one by
timestamp, and their time since the bout began is the point-process clock the
per-bin rate models are built on. ``nearest_well_distance_cm`` is the physical
companion: how far the animal actually was from a reward port while stopped.
``summarize_bout_location`` is where a stop was, over its rows outside excluded
time, and ``bout_start_context`` adds the trial context at the stop's first row;
both stop tables read them.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
)
from continuum_swr_replay.intervals import (
    covered_seconds,
    interval_mask,
)

from ..behavior.covariates.context import summarize_behavioral_location
from ..behavior.spatial_mapping import (
    arm_to_patch,
    arms_of_wells,
    single_arm_from_well,
    stay_indicator_from_patches,
)
from ._common import safe_mode

__all__ = [
    "assign_bout",
    "bout_exposure_s",
    "bout_start_context",
    "immobility_bout_intervals",
    "nearest_well_distance_cm",
    "summarize_bout_location",
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
    bout_starts: np.ndarray, bout_ends: np.ndarray, excluded: pd.DataFrame
) -> np.ndarray:
    """Each bout's analysed immobility: its span less the excluded time in it.

    Excluded time (the artifact mask widened by 0.3 s) is not analysed, so it
    is no rate's exposure. The bout itself (its start, end and ``stop_id``) is
    not changed; only its exposure falls.

    Parameters
    ----------
    bout_starts, bout_ends : np.ndarray, shape (n_bouts,)
        Seconds, from :func:`immobility_bout_intervals`.
    excluded : pd.DataFrame
        The epoch's excluded time (``analysed_time.excluded_intervals``):
        ``start_time`` and ``end_time``, closed.

    Returns
    -------
    np.ndarray, shape (n_bouts,)
        Seconds; ``bout_ends - bout_starts`` itself when nothing is excluded.
    """
    span = bout_ends - bout_starts
    if len(excluded) == 0:
        return span
    exposure: np.ndarray = span - covered_seconds(bout_starts, bout_ends, excluded)
    return exposure


def summarize_bout_location(
    bout_rows: pd.DataFrame,
    n_bouts: int,
    track_graph: nx.Graph,
    excluded: pd.DataFrame,
) -> pd.DataFrame:
    """Where each bout was: location, segment and distance to a well.

    Excluded time (the artifact mask widened by 0.3 s) is not analysed, so each
    summary reads only the bout's rows outside it (closed intervals). A bout
    wholly inside it is ``unknown``, with no well, segment or distance. The
    bout itself is not changed. With no excluded time every row is read.

    Parameters
    ----------
    bout_rows : pd.DataFrame
        The session frame's rows that lie in a bout, time-indexed (seconds),
        with ``_bout_idx`` (0 .. ``n_bouts - 1``, from :func:`assign_bout`),
        ``occupancy_state``, ``occupancy_well_id``,
        ``nearest_well_distance_cm`` and, when the frame has it,
        ``track_segment_id``.
    n_bouts : int
        Number of bouts.
    track_graph : nx.Graph
        Session track graph, for the well's arm.
    excluded : pd.DataFrame
        The epoch's excluded time (``analysed_time.excluded_intervals``):
        ``start_time`` and ``end_time``, closed.

    Returns
    -------
    pd.DataFrame, shape (n_bouts, 8)
        Indexed 0 .. ``n_bouts - 1``: ``occupancy_state``, ``at_well``
        (nullable boolean) and ``occupancy_well_id``
        (``summarize_behavioral_location``); ``modal_observed_track_segment_id``
        (the modal segment, NaN without the column);
        ``nearest_well_distance_cm_min``, ``_mean`` and ``_max`` over the
        finite distances; and ``occupancy_well_arm``, the arm of the well of a
        strict at-well bout, NaN otherwise.
    """
    times = bout_rows.index.to_numpy(dtype=float)
    analysed = bout_rows.loc[~interval_mask(times, excluded)]
    by_bout = dict(tuple(analysed.groupby("_bout_idx", sort=True)))
    no_rows = analysed.iloc[0:0]

    states: list[str] = []
    at_well: list[object] = []
    wells = np.full(n_bouts, np.nan)
    segments = np.full(n_bouts, np.nan)
    distance_min = np.full(n_bouts, np.nan)
    distance_mean = np.full(n_bouts, np.nan)
    distance_max = np.full(n_bouts, np.nan)
    arms = np.full(n_bouts, np.nan)
    for bout in range(n_bouts):
        rows = by_bout.get(bout, no_rows)
        location = summarize_behavioral_location(
            rows[["occupancy_state", "occupancy_well_id"]]
        )
        well = float(location["occupancy_well_id"])
        states.append(str(location["occupancy_state"]))
        at_well.append(location["at_well"])
        wells[bout] = well
        arms[bout] = single_arm_from_well(well, track_graph)
        if "track_segment_id" in rows:
            segments[bout] = safe_mode(rows["track_segment_id"])
        distance = rows["nearest_well_distance_cm"].to_numpy(dtype=float)
        finite_distance = distance[np.isfinite(distance)]
        if finite_distance.size:
            distance_min[bout] = finite_distance.min()
            distance_mean[bout] = finite_distance.mean()
            distance_max[bout] = finite_distance.max()

    return pd.DataFrame(
        {
            "occupancy_state": states,
            "at_well": pd.array(at_well, dtype="boolean"),
            "occupancy_well_id": wells,
            "modal_observed_track_segment_id": segments,
            "nearest_well_distance_cm_min": distance_min,
            "nearest_well_distance_cm_mean": distance_mean,
            "nearest_well_distance_cm_max": distance_max,
            "occupancy_well_arm": arms,
        }
    )


#: Trial-context columns read as-is from each bout's first row.
_FIRST_ROW_COLUMNS = (
    "did_last_completed_trip_switch_patch",
    "patch_switch_fraction_last5_completed_trips",
    "leaky_q_value_diff",
    "is_rewarded_outcome",
)

#: Arm column -> the well column at the bout's first row it is the arm of.
_FIRST_ROW_WELL_ARMS = {
    "prev_arm": "prev_well",
    "next_arm": "next_well",
    "last_rewarded_arm": "last_rewarded_well",
    "completed_trial_arm": "last_visited_well",
}


def bout_start_context(
    bout_rows: pd.DataFrame,
    n_bouts: int,
    track_graph: nx.Graph,
    excluded: pd.DataFrame,
) -> pd.DataFrame:
    """The per-stop context both stop tables share.

    The trial context and the previous, next, last-rewarded and last-visited
    wells are read at each bout's first row as-is, NaN included
    (``first(skipna=False)``): the state at the stop's start, whether or not
    that row is in excluded time. Skipping a leading NaN would surface a later
    row's value and leak the post-boundary trial's context into a bout that
    spans a trial boundary. The location, segment and distance columns are
    :func:`summarize_bout_location`'s, over the bout's rows outside excluded
    time. The patches map from the arms, and ``is_stay`` (will the next trip
    stay in the current patch; post-hoc) from the patches.

    Parameters
    ----------
    bout_rows : pd.DataFrame
        The session frame's rows that lie in a bout, time-indexed (seconds),
        with ``_bout_idx`` (0 .. ``n_bouts - 1``, from :func:`assign_bout`),
        the trial-context columns of ``add_trial_context`` and the columns
        :func:`summarize_bout_location` reads.
    n_bouts : int
        Number of bouts.
    track_graph : nx.Graph
        Session track graph.
    excluded : pd.DataFrame
        The epoch's excluded time (``analysed_time.excluded_intervals``):
        ``start_time`` and ``end_time``, closed.

    Returns
    -------
    pd.DataFrame, shape (n_bouts, 21)
        Indexed 0 .. ``n_bouts - 1``: ``did_last_completed_trip_switch_patch``,
        ``patch_switch_fraction_last5_completed_trips``,
        ``leaky_q_value_diff``, ``is_rewarded_outcome``,
        ``completed_trial_id`` (``Int64``); :func:`summarize_bout_location`'s
        eight columns; ``prev_arm``, ``next_arm``, ``last_rewarded_arm``,
        ``completed_trial_arm``; ``current_patch``, ``prev_patch``,
        ``next_patch`` and ``is_stay``.
    """
    bouts = np.arange(n_bouts)
    first_row = (
        bout_rows.groupby("_bout_idx", sort=True)[
            [*_FIRST_ROW_COLUMNS, "completed_trial_id", *_FIRST_ROW_WELL_ARMS.values()]
        ]
        .first(skipna=False)
        .reindex(bouts)
    )
    context = pd.DataFrame(index=pd.RangeIndex(n_bouts))
    for column in _FIRST_ROW_COLUMNS:
        context[column] = first_row[column].to_numpy()
    context["completed_trial_id"] = pd.array(
        first_row["completed_trial_id"], dtype="Int64"
    )

    location = summarize_bout_location(bout_rows, n_bouts, track_graph, excluded)
    for column in location.columns:
        context[column] = location[column]

    for arm_column, well_column in _FIRST_ROW_WELL_ARMS.items():
        context[arm_column] = arms_of_wells(
            first_row[well_column].to_numpy(dtype=float, na_value=np.nan),
            track_graph,
        )

    current_patch = np.array(
        [arm_to_patch(a) for a in location["occupancy_well_arm"].to_numpy()]
    )
    next_patch = np.array([arm_to_patch(a) for a in context["next_arm"].to_numpy()])
    context["current_patch"] = current_patch
    context["prev_patch"] = [arm_to_patch(a) for a in context["prev_arm"].to_numpy()]
    context["next_patch"] = next_patch
    context["is_stay"] = stay_indicator_from_patches(next_patch, current_patch)
    return context


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
