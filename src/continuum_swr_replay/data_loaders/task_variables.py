"""Task-specific derived variables computation."""

import networkx as nx
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def path_progress_for_run_segment(
    pos_info: pd.DataFrame,
    track_graph: nx.Graph,
    start_node: int,
    end_node: int,
) -> np.ndarray:
    """Compute path progress (0-1) along a run segment.

    Parameters
    ----------
    pos_info : pd.DataFrame
        Position info for the segment
    track_graph : nx.Graph
        Track graph
    start_node : int
        Starting well node
    end_node : int
        Ending well node

    Returns
    -------
    np.ndarray
        Path progress values (0 to 1) for each time point
    """
    path = nx.shortest_path(track_graph, source=start_node, target=end_node)
    path_edges = list(zip(path[:-1], path[1:], strict=False))
    path_edge_ids = [track_graph.edges[edge]["edge_id"] for edge in path_edges]

    dist = np.full(pos_info.shape[0], np.nan)
    time = pos_info.index
    add = 0.0

    if len(path_edges) == 0:
        return np.zeros(pos_info.shape[0])

    for (start_node, end_node), edge_id in zip(path_edges, path_edge_ids, strict=False):
        is_edge = pos_info.track_segment_id == edge_id
        proj_pos = pos_info.loc[
            is_edge, ["projected_x_position", "projected_y_position"]
        ].to_numpy()
        start_node_pos = np.array(track_graph.nodes[start_node]["pos"])[None, :]
        dist[is_edge] = np.linalg.norm(proj_pos - start_node_pos, axis=1)
        dist[is_edge] += add
        add += np.linalg.norm(
            np.array(track_graph.nodes[start_node]["pos"])
            - np.array(track_graph.nodes[end_node]["pos"])
        )

    is_nan = np.isnan(dist)
    if np.sum(~is_nan) >= 2:
        interpolater = interp1d(
            time[~is_nan], dist[~is_nan], fill_value="extrapolate", kind="previous"
        )
        dist[is_nan] = interpolater(time[is_nan])

    # Avoid division by zero if start and end nodes are the same
    max_dist = dist.max()
    if max_dist == 0 or np.isnan(max_dist):
        return np.zeros(pos_info.shape[0])
    return np.asarray(dist / max_dist)


def dist_to_goal_per_time(
    track_segment_id,
    projected_x_position,
    projected_y_position,
    goal_node_id,
    track_graph,
    track_segment_id_to_edge,
):
    """Compute distance to goal for a single time point.

    Parameters
    ----------
    track_segment_id : int
        Current track segment ID
    projected_x_position : float
        Current x position
    projected_y_position : float
        Current y position
    goal_node_id : int
        Goal well node ID
    track_graph : nx.Graph
        Track graph
    track_segment_id_to_edge : dict
        Mapping from segment ID to edge tuple

    Returns
    -------
    float
        Distance to goal in graph distance units
    """
    node1, node2 = track_segment_id_to_edge[track_segment_id]
    copy_graph = track_graph.copy()
    copy_graph.add_node("cur_pos", pos=(projected_x_position, projected_y_position))
    x1, y1 = copy_graph.nodes[node1]["pos"]
    x2, y2 = copy_graph.nodes[node2]["pos"]
    x3, y3 = copy_graph.nodes["cur_pos"]["pos"]
    left_distance = np.sqrt((x3 - x1) ** 2 + (y3 - y1) ** 2)
    right_distance = np.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
    nx.add_path(copy_graph, [node1, "cur_pos"], distance=left_distance)
    nx.add_path(copy_graph, ["cur_pos", node2], distance=right_distance)
    return nx.shortest_path_length(
        copy_graph,
        source="cur_pos",
        target=goal_node_id,
        weight="distance",
    )


def dist_to_goal(position_info, trial, track_graph, track_segment_id_to_edge):
    """Compute distance to goal for all time points in a trial.

    Parameters
    ----------
    position_info : pd.DataFrame
        Position info for the trial
    trial : pd.Series
        Trial information with to_well attribute
    track_graph : nx.Graph
        Track graph
    track_segment_id_to_edge : dict
        Mapping from segment ID to edge tuple

    Returns
    -------
    np.ndarray
        Distance to goal for each time point
    """
    return np.array(
        [
            dist_to_goal_per_time(
                row.track_segment_id,
                row.projected_x_position,
                row.projected_y_position,
                trial.to_well,
                track_graph,
                track_segment_id_to_edge,
            )
            for row in position_info.itertuples()
        ]
    )


def turn_direction(trial, position_info, track_graph, track_edges):
    """Determine turn direction sequence for a trial.

    Parameters
    ----------
    trial : pd.Series
        Trial information
    position_info : pd.DataFrame
        Position info for the trial
    track_graph : nx.Graph
        Track graph
    track_edges : np.ndarray
        Array of track edges

    Returns
    -------
    str
        Turn direction sequence (e.g., "left-right-left")
    """
    trial_segment_ids = position_info.loc[
        trial.start_time : trial.end_time
    ].track_segment_id.unique()  # pandas unique is not sorted
    is_bad = (
        position_info.loc[trial.start_time : trial.end_time]
        .groupby("track_segment_id")
        .size()
        < 50
    )
    trial_segment_ids = trial_segment_ids[
        ~np.isin(trial_segment_ids, is_bad.index[is_bad])
    ]

    # Ensure that the trial is in the correct traversal direction
    # track_segment_id does not contain node order information
    trial_nodes = track_edges[trial_segment_ids]

    if trial_nodes[0][0] != trial.from_well:
        trial_nodes = trial_nodes[::-1]

    for ind, (edge1, edge2) in enumerate(zip(trial_nodes[:-1], trial_nodes[1:], strict=False)):
        if edge1[-1] != edge2[0]:
            trial_nodes[ind + 1] = trial_nodes[ind + 1][::-1]

    vectors = np.array(
        [
            np.array(track_graph.nodes[node2]["pos"])
            - np.array(track_graph.nodes[node1]["pos"])
            for (node1, node2) in trial_nodes
        ]
    )
    trial_turns = np.where(
        np.cross(vectors[:-1], vectors[1:], axis=1) > 0, "left", "right"
    )
    return "-".join(trial_turns)


def make_task_variables(
    position_info: pd.DataFrame,
    trials_df: pd.DataFrame,
    track_graph: nx.Graph,
    sampling_frequency: float = 500.0,
    omission_shifts: list[int] | None = None,
    patch_change_shifts: list[int] | None = None,
) -> pd.DataFrame:
    """Compute all task-specific derived variables.

    Parameters
    ----------
    position_info : pd.DataFrame
        Position data
    trials_df : pd.DataFrame
        Trials data
    track_graph : nx.Graph
        Track graph
    sampling_frequency : float, optional
        Sampling frequency in Hz
    omission_shifts : list[int], optional
        Trial shifts for reward omission indicators
    patch_change_shifts : list[int], optional
        Trial shifts for patch change indicators

    Returns
    -------
    pd.DataFrame
        Task variables including:
        - trial_number
        - time_to_goal
        - start_node_id, goal_node_id
        - dist_to_goal
        - path_progress
        - turn_direction
        - is_no_reward_N (for each shift in omission_shifts)
        - is_patch_changeN (for each shift in patch_change_shifts)
    """
    if omission_shifts is None:
        omission_shifts = list(range(1, 6))
    if patch_change_shifts is None:
        patch_change_shifts = list(range(-5, 6))

    time = position_info.index
    n_time = len(time)

    trial_number = np.full((n_time,), np.nan)
    time_to_goal = np.full((n_time), np.nan)
    start_node_id = np.full((n_time), np.nan)
    goal_node_id = np.full((n_time), np.nan)
    dist_to_goal_arr = np.full((n_time), np.nan)
    path_progress = np.full((n_time), np.nan)
    turn_direction_arr = np.full((n_time), pd.NA)
    n_shifts = len(omission_shifts)
    is_prev_reward_omission_pos = np.full((n_time, n_shifts), False)

    track_edges = np.array(track_graph.edges)

    track_segment_id_to_edge = {
        track_graph.edges[edge]["edge_id"]: edge for edge in track_graph.edges
    }
    is_prev_reward_omission = pd.concat(
        [
            (~trials_df.is_reward)
            .shift(shift)
            .fillna(False)
            .rename(f"is_no_reward_{shift}")
            for shift in omission_shifts
        ],
        axis=1,
    )
    is_patch_change = pd.concat(
        [
            trials_df.is_patch_change.shift(shift)
            .fillna(False)
            .rename(f"is_patch_change_{shift}")
            for shift in patch_change_shifts
        ],
        axis=1,
    )
    n_shifts = len(patch_change_shifts)
    is_patch_change_pos = np.full((n_time, n_shifts), False)

    for trial in trials_df.itertuples():
        is_in_trial = (time >= trial.start_time) & (time < trial.end_time)
        from_start_time_to_goal = (time >= trial.start_time) & (
            time < trial.goal_arrival_time
        )
        from_goal_to_next_movement = (time >= trial.goal_arrival_time) & (
            time < trial.end_time
        )
        trial_number[is_in_trial] = trial.Index

        time_to_goal[from_start_time_to_goal] = (
            np.arange(np.sum(from_start_time_to_goal)) / sampling_frequency
        )
        time_to_goal[from_goal_to_next_movement] = 0.0

        start_node_id[is_in_trial] = trial.from_well
        goal_node_id[is_in_trial] = trial.to_well
        turn_direction_arr[is_in_trial] = turn_direction(
            trial, position_info.loc[is_in_trial], track_graph, track_edges
        )
        path_progress[from_start_time_to_goal] = path_progress_for_run_segment(
            position_info.loc[from_start_time_to_goal],
            track_graph,
            trial.from_well,
            trial.to_well,
        )
        path_progress[from_goal_to_next_movement] = 0.0

        dist_to_goal_arr[is_in_trial] = dist_to_goal(
            position_info.loc[is_in_trial],
            trial,
            track_graph,
            track_segment_id_to_edge,
        )
        is_prev_reward_omission_pos[is_in_trial] = is_prev_reward_omission.loc[
            trial.Index
        ].to_numpy()
        is_patch_change_pos[is_in_trial] = is_patch_change.loc[trial.Index].to_numpy()

    return pd.DataFrame(
        {
            "trial_number": trial_number,
            "time_to_goal": time_to_goal,
            "start_node_id": start_node_id,
            "goal_node_id": goal_node_id,
            "dist_to_goal": dist_to_goal_arr,
            "path_progress": path_progress,
            "turn_direction": turn_direction_arr,
            **{
                f"is_no_reward_{shift}": is_reward_omission
                for is_reward_omission, shift in zip(
                    is_prev_reward_omission_pos.T, omission_shifts, strict=False
                )
            },
            **{
                f"is_patch_change{shift}": is_patch_change
                for is_patch_change, shift in zip(
                    is_patch_change_pos.T, patch_change_shifts, strict=False
                )
            },
        },
        index=time,
    ).ffill()
