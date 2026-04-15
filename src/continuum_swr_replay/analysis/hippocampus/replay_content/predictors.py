"""Compute new predictors for the replay content model hierarchy.

New continuous predictors that go beyond the binary indicators in the
original analysis:

- time_since_stop: seconds from immobility onset to event midpoint
- visit_recency: trials since each arm was last visited
- cumulative_reward: rewards received at each arm in last N trials
- log_distance: log of |ahead_behind_distance| per event
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH

from continuum_swr_replay.analysis.replay.spatial_mapping import well_to_arm_set

ALL_ARMS = sorted(TRACK_SEGMENT_TO_PATCH.keys())


# At 500 Hz, timebins are 2ms apart. A gap > 100ms between consecutive
# immobile timebins indicates a new immobile bout (movement intervened).
BOUT_GAP_THRESHOLD_SEC = 0.1


def compute_time_since_stop(
    nl: pd.DataFrame,
    dot: pd.DataFrame | None = None,
    speed_col: str = "head_speed",
    speed_threshold: float = 4.0,
) -> pd.Series:
    """Compute time since immobility onset for each non-local timebin.

    For each non-local timebin, finds the start of the enclosing
    **immobility bout** (when the animal's speed dropped below threshold),
    not just the start of the non-local cluster. This gives the correct
    measure for testing Mallory's temporal dynamics hypothesis.

    If ``dot`` is not provided, falls back to gap-based bout detection
    on the filtered ``nl`` index (less accurate — measures time since
    the non-local cluster started, not since the animal stopped).

    Parameters
    ----------
    nl : pd.DataFrame
        Non-local immobile timebins (already filtered). Must have a
        time-based index.
    dot : pd.DataFrame | None
        Full data_over_time DataFrame (unfiltered). If provided, used to
        find immobility bout boundaries from speed.
    speed_col : str
        Column name for speed in ``dot``.
    speed_threshold : float
        Speed threshold for immobility (cm/s).

    Returns
    -------
    pd.Series
        Time since stop onset (seconds), indexed like nl.
    """
    times = nl.index.values
    if len(times) == 0:
        return pd.Series(dtype=float)

    if dot is not None and speed_col in dot.columns:
        # Find immobility bout starts from the full data
        is_immobile = dot[speed_col] < speed_threshold
        imm_times = dot.index.values
        # Detect transitions: not-immobile → immobile
        imm_arr = is_immobile.values
        transitions = np.where(~imm_arr[:-1] & imm_arr[1:])[0] + 1
        # Also include index 0 if the session starts immobile
        if imm_arr[0]:
            transitions = np.concatenate([[0], transitions])
        bout_start_times_all = imm_times[transitions]

        # For each non-local timebin, find the most recent bout start
        # searchsorted gives the index of the first bout_start > time
        idx = np.searchsorted(bout_start_times_all, times, side="right") - 1
        idx = np.clip(idx, 0, len(bout_start_times_all) - 1)
        result = times - bout_start_times_all[idx]
        # Negative values indicate the non-local event is before any
        # detected immobility start (edge case at session start)
        result = np.clip(result, 0, None)
    else:
        # Fallback: gap-based bout detection on the filtered index
        dt = np.diff(times)
        bout_breaks = np.where(dt > BOUT_GAP_THRESHOLD_SEC)[0] + 1
        bout_starts = np.concatenate([[0], bout_breaks])

        bout_start_times = np.empty(len(times))
        for i in range(len(bout_starts)):
            start_idx = bout_starts[i]
            end_idx = bout_starts[i + 1] if i + 1 < len(bout_starts) else len(times)
            bout_start_times[start_idx:end_idx] = times[start_idx]

        result = times - bout_start_times

    return pd.Series(result, index=nl.index, name="time_since_stop")


def compute_event_time_since_stop(
    events: pd.DataFrame,
    nl: pd.DataFrame,
) -> np.ndarray:
    """Compute time_since_stop for each event (event-level).

    Takes the median time_since_stop across timebins within each event.

    Parameters
    ----------
    events : pd.DataFrame
        Event-level data with ``non_local_event`` column.
    nl : pd.DataFrame
        Non-local immobile timebins with ``non_local_event`` and
        ``time_since_stop`` columns.

    Returns
    -------
    np.ndarray
        One value per event row.
    """
    if "time_since_stop" not in nl.columns:
        nl = nl.copy()
        nl["time_since_stop"] = compute_time_since_stop(nl)

    # Median time_since_stop per event
    event_tss = (
        nl[nl["non_local_event"] > 0]
        .groupby(["animal", "day", "epoch", "non_local_event"])["time_since_stop"]
        .median()
    )

    # Merge with events
    merge_keys = ["animal", "day", "epoch", "non_local_event"]
    merged = events[merge_keys].merge(
        event_tss.reset_index(), on=merge_keys, how="left"
    )
    return merged["time_since_stop"].values


def compute_visit_recency_for_event(
    event_time: float,
    trials: pd.DataFrame,
    track_graph,
    n_default: int | None = None,
) -> dict[int, float]:
    """Compute visit recency (trials since last visit) for each arm at a given time.

    For each of the 9 arms, looks backward through the trial history to find
    the most recent trial where the animal visited one of that arm's wells.

    Parameters
    ----------
    event_time : float
        Timestamp of the event.
    trials : pd.DataFrame
        Trial DataFrame with ``end_time``, ``to_well``, ``from_well``.
    track_graph : networkx.Graph
        Track graph for well-to-arm mapping.
    n_default : int | None
        Value for arms never visited. If None, uses n_trials + 1.

    Returns
    -------
    dict[int, float]
        {arm_id: trials_since_last_visit} for each of the 9 arms.
    """
    n_trials = len(trials)
    if n_default is None:
        n_default = n_trials + 1

    # Trials completed before this event
    past_trials = trials[trials["end_time"] <= event_time]

    # Build well -> set of arms mapping (cached on first call via track_graph)
    # For each past trial, the animal visited both from_well and to_well
    result = {arm: float(n_default) for arm in ALL_ARMS}

    for trial_offset, trial in enumerate(reversed(list(past_trials.itertuples()))):
        trials_ago = trial_offset + 1
        # Wells visited on this trial
        for well_col in ["to_well", "from_well"]:
            well_id = getattr(trial, well_col)
            if np.isnan(well_id):
                continue
            arms = well_to_arm_set(well_id, track_graph)
            for arm in arms:
                if result[arm] == float(n_default):
                    result[arm] = float(trials_ago)

        # Early exit if all arms have been visited
        if all(v != float(n_default) for v in result.values()):
            break

    return result


def compute_cumulative_reward_for_event(
    event_time: float,
    trials: pd.DataFrame,
    track_graph,
    window: int = 10,
) -> dict[int, float]:
    """Compute cumulative reward count for each arm in the last N trials.

    Parameters
    ----------
    event_time : float
        Timestamp of the event.
    trials : pd.DataFrame
        Trial DataFrame with ``end_time``, ``to_well``, ``is_reward``.
    track_graph : networkx.Graph
        Track graph for well-to-arm mapping.
    window : int
        Number of past trials to consider.

    Returns
    -------
    dict[int, float]
        {arm_id: reward_count} for each of the 9 arms.
    """
    # Trials completed before this event
    past_trials = trials[trials["end_time"] <= event_time].tail(window)

    result = dict.fromkeys(ALL_ARMS, 0.0)

    for trial in past_trials.itertuples():
        if not trial.is_reward:
            continue
        well_id = trial.to_well
        if np.isnan(well_id):
            continue
        arms = well_to_arm_set(well_id, track_graph)
        for arm in arms:
            result[arm] += 1.0

    return result


def compute_dist_to_arm(
    current_well: int | float,
    track_graph,
) -> dict[int, float]:
    """Graph shortest-path distance from a well to each arm.

    For each arm (edge), computes the shortest-path distance from the
    animal's current well to the nearest endpoint of that arm. This gives
    an arm-varying continuous predictor that is identifiable in the
    conditional Poisson model.

    Parameters
    ----------
    current_well : int or float
        Well node ID (0-5) where the animal currently is. If NaN, returns
        NaN for all arms.
    track_graph : networkx.Graph
        Track graph with ``edge_id`` attributes on edges and optional
        ``distance`` edge weights.

    Returns
    -------
    dict[int, float]
        ``{arm_id: distance}`` for each of the 9 arms. Distance is 0 for
        arms adjacent to the current well.
    """
    import networkx as nx

    if np.isnan(current_well):
        return dict.fromkeys(ALL_ARMS, np.nan)

    current_well = int(current_well)

    # Compute shortest path lengths from current_well to all nodes.
    # Use edge weights ("distance") if available; fall back to hop count.
    has_weights = any(
        "distance" in d for _, _, d in track_graph.edges(data=True)
    )
    try:
        if has_weights:
            lengths = dict(
                nx.single_source_dijkstra_path_length(
                    track_graph, current_well, weight="distance"
                )
            )
        else:
            lengths = dict(
                nx.single_source_shortest_path_length(track_graph, current_well)
            )
    except nx.NodeNotFound:
        return dict.fromkeys(ALL_ARMS, np.nan)

    # For each arm, find the minimum distance to either endpoint
    arm_to_endpoints: dict[int, list[int]] = {}
    for u, v, data in track_graph.edges(data=True):
        arm_id = data.get("edge_id")
        if arm_id is not None:
            arm_to_endpoints.setdefault(arm_id, []).extend([u, v])

    result = {}
    for arm in ALL_ARMS:
        endpoints = arm_to_endpoints.get(arm, [])
        if not endpoints:
            result[arm] = np.nan
        else:
            dists = [lengths.get(n, np.nan) for n in endpoints]
            valid = [d for d in dists if not np.isnan(d)]
            result[arm] = float(min(valid)) if valid else np.nan

    return result


def add_temporal_interactions(df: pd.DataFrame) -> None:
    """Add time-based interaction columns (in place).

    Creates time_x_current, time_x_prev, and time_x_next from
    time_since_stop.
    """
    if "time_since_stop" in df.columns:
        df["time_x_current"] = df["time_since_stop"] * df["is_current"]
        df["time_x_prev"] = df["time_since_stop"] * df["is_prev"]
        df["time_x_next"] = df["time_since_stop"] * df["is_next"]
