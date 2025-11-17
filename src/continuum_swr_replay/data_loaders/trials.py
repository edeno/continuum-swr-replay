"""Trial and well visit construction from task events."""

import networkx as nx
import numpy as np
import pandas as pd

from ..types import WellNodeData
from .constants import WELL_PATCH_MAPPING


def get_poke_in_poke_out_times(
    beam_breaks: pd.DataFrame, max_inter_event_interval: float = 1.0
) -> pd.DataFrame:
    """Merge beam break events into poke in/out intervals.

    Parameters
    ----------
    beam_breaks : pd.DataFrame
        Beam break events with start_time and end_time
    max_inter_event_interval : float, optional
        Maximum time between events to be considered part of same poke

    Returns
    -------
    pd.DataFrame
        Merged poke events with start_time, end_time, and duration
    """
    merged_events = []
    for _, df in beam_breaks.groupby("event_name"):
        interevent_interval = (df.start_time.shift(-1) - df.end_time).fillna(0)
        labels = (
            (interevent_interval > max_inter_event_interval).shift(1).fillna(0).cumsum()
        )
        new_events = df.groupby(labels).agg({"start_time": "first", "end_time": "last"})
        merged_events.append(new_events)

    return (
        pd.concat(merged_events)
        .sort_values("start_time")
        .assign(
            duration=lambda x: x.end_time - x.start_time,
        )
    )


def get_run_intervals(poke_times: pd.DataFrame) -> pd.DataFrame:
    """Extract run intervals between pokes.

    Parameters
    ----------
    poke_times : pd.DataFrame
        Poke events with start_time and end_time

    Returns
    -------
    pd.DataFrame
        Run intervals with start_time, end_time, and duration
    """
    return (
        pd.DataFrame(
            {
                "start_time": poke_times["end_time"].to_numpy()[:-1],
                "end_time": poke_times["start_time"].shift(-1).to_numpy()[:-1],
            }
        )
        .dropna()
        .assign(
            duration=lambda x: x.end_time - x.start_time,
        )
    )


def get_start_end_node_id(
    position_info: pd.DataFrame, track_graph: nx.Graph, times: pd.DataFrame
) -> WellNodeData:
    """Determine start and end well nodes for each interval.

    Parameters
    ----------
    position_info : pd.DataFrame
        Position data with projected position columns
    track_graph : nx.Graph
        Track graph with node positions
    times : pd.DataFrame
        Times with start_time and end_time columns

    Returns
    -------
    WellNodeData
        TypedDict with 'from_well' and 'to_well' node ID arrays
    """
    PROJECTED_COLS = ["projected_x_position", "projected_y_position"]

    start_time_pos = position_info.iloc[
        position_info.index.searchsorted(times.start_time.values, side="right")
    ][PROJECTED_COLS].to_numpy()
    end_time_pos = position_info.iloc[
        position_info.index.searchsorted(times.end_time.values, side="right")
    ][PROJECTED_COLS].to_numpy()

    node_attrs = nx.get_node_attributes(track_graph, "pos")
    node_pos = np.array(list(node_attrs.values()))
    node_ids = np.array(list(node_attrs.keys()))

    start_node_ids = node_ids[
        np.argmin(
            np.linalg.norm(start_time_pos[:, None, :] - node_pos[None, :, :], axis=2),
            axis=1,
        )
    ]
    end_node_ids = node_ids[
        np.argmin(
            np.linalg.norm(end_time_pos[:, None, :] - node_pos[None, :, :], axis=2),
            axis=1,
        )
    ]

    return {
        "from_well": start_node_ids,
        "to_well": end_node_ids,
    }


def make_well_visits_df(
    beam_breaks: pd.DataFrame,
    pump_events: pd.DataFrame,
    position_info: pd.DataFrame,
    track_graph: nx.Graph,
) -> pd.DataFrame:
    """Construct well visits DataFrame from beam breaks and pump events.

    Parameters
    ----------
    beam_breaks : pd.DataFrame
        Beam break events
    pump_events : pd.DataFrame
        Reward pump events
    position_info : pd.DataFrame
        Position data
    track_graph : nx.Graph
        Track graph

    Returns
    -------
    pd.DataFrame
        Well visits with trial structure and reward information
    """
    poke_times = get_poke_in_poke_out_times(beam_breaks)
    poke_times = pd.concat(
        (
            poke_times,
            pd.DataFrame(
                {
                    "start_time": [
                        position_info.index[0],
                        position_info.index[-1] - 1e-4,
                    ],
                    "end_time": [
                        position_info.index[0],
                        position_info.index[-1] - 1e-4,
                    ],
                    "duration": [0.0, 0.0],
                },
                index=[0, 0],
            ),
        )
    ).sort_values("start_time")
    poke_times = poke_times.assign(
        **get_start_end_node_id(position_info, track_graph, poke_times)
    ).set_index(["from_well", "to_well"])

    run_times = get_run_intervals(poke_times)
    run_times = run_times.assign(
        **get_start_end_node_id(position_info, track_graph, run_times)
    ).set_index(["from_well", "to_well"])

    well_visits = (
        pd.concat((poke_times, run_times)).sort_values("start_time").reset_index()
    )
    well_visits["is_run"] = well_visits["from_well"] != well_visits["to_well"]
    well_visits["is_reward"] = [
        pump_events.start_time.between(row.start_time, row.end_time).any()
        for row in well_visits.itertuples()
    ]
    well_visits["is_reward"] = (
        well_visits["is_run"] & well_visits["is_reward"].shift(-1)
    ) | well_visits["is_reward"]
    well_visits["from_patch"] = well_visits["from_well"].map(WELL_PATCH_MAPPING)
    well_visits["to_patch"] = well_visits["to_well"].map(WELL_PATCH_MAPPING)
    well_visits["is_patch_change"] = (
        well_visits["from_patch"] != well_visits["to_patch"]
    )

    trial_number = np.full((len(well_visits),), np.nan)
    trial_number[well_visits.is_run] = np.arange(1, well_visits.is_run.sum() + 1)
    trial_number = (
        pd.DataFrame(trial_number)
        .ffill()
        .bfill()
        .astype(int)
        .to_numpy()
        .squeeze()
    )
    well_visits["trial_number"] = trial_number

    return well_visits


def make_trials_df_from_well_visits(well_visits: pd.DataFrame) -> pd.DataFrame:
    """Create trials DataFrame by aggregating well visits.

    Parameters
    ----------
    well_visits : pd.DataFrame
        Well visits data

    Returns
    -------
    pd.DataFrame
        Trials with aggregated information (reward, timing, start/end wells)
    """
    trials = well_visits.groupby("trial_number").agg(
        is_reward=("is_reward", lambda x: x.sum() > 0),
        start_time=("start_time", "min"),
        end_time=("end_time", "max"),
        is_patch_change=("is_patch_change", lambda x: x.sum() > 0),
        from_well=("from_well", "first"),
        to_well=("to_well", "last"),
        from_patch=("from_patch", "first"),
        to_patch=("to_patch", "last"),
        duration=("duration", "sum"),
    )
    goal_time = (
        well_visits.loc[well_visits.from_well != well_visits.to_well]
        .groupby("trial_number")
        .agg(
            goal_arrival_time=("end_time", "min"),
        )
    )
    return trials.merge(goal_time, left_index=True, right_index=True)
