"""Trial and well visit construction from task events.

This module constructs structured trial and well visit DataFrames from raw task events
(beam breaks, pump activations) and position data. It implements the logic for
segmenting continuous behavior into discrete trials and identifying key behavioral
events like well visits, runs, rewards, and patch changes.

Main Functions
--------------
make_well_visits_df
    Primary entry point that constructs the complete well visits DataFrame from
    beam breaks, pump events, position data, and track graph. Combines poke and
    run intervals into a unified behavioral segmentation.
make_trials_df_from_well_visits
    Aggregates well visits into trials by grouping consecutive well visit events
    and computing trial-level summaries (reward status, timing, etc.).

Helper Functions
----------------
get_poke_in_poke_out_times
    Merges consecutive beam break events into poke in/out intervals using a
    maximum inter-event interval threshold.
get_run_intervals
    Extracts movement intervals between consecutive pokes.
get_start_end_node_id
    Determines which wells (graph nodes) the animal is at for each interval
    by finding the nearest node to the projected position.

Data Structure
--------------
Well Visits DataFrame contains:
    - start_time, end_time, duration: Interval timing
    - from_well, to_well: Start and end well node IDs (0-5)
    - from_patch, to_patch: Start and end patch IDs (1-3)
    - is_run: True if movement interval (from_well != to_well)
    - is_reward: True if reward was delivered during this interval
    - reward_on_time: First pump onset in the interval, NaN for no pump
    - is_patch_change: True if movement crossed patch boundaries
    - trial_number: Trial index assignment

Trials DataFrame contains:
    - start_time, end_time, duration: Trial timing (sum of visit durations)
    - goal_arrival_time: Time when animal reached goal well
    - reward_on_time: First pump onset on rewarded trials, else NaN
    - outcome_known_time: reward_on_time for rewards; goal_arrival_time +
      OMISSION_KNOWN_LATENCY_SEC (capped at end_time) for omissions
    - from_well, to_well: Starting and ending well IDs
    - from_patch, to_patch: Starting and ending patch IDs
    - is_reward: True if trial resulted in reward
    - is_patch_change: True if trial crossed patch boundaries

Notes
-----
- Pokes are merged if inter-event interval is below threshold (default 1.0 sec)
- Rewards are associated with runs, not poke intervals
- Trial numbers are forward/backward filled to assign pokes to trials
- Boundary events are added at session start/end for completeness
- Position data must have projected_x_position and projected_y_position columns
- Node positions come from track_graph.nodes[node_id]['pos']

See Also
--------
continuum_swr_replay.data_loaders.events : DIO event loading
continuum_swr_replay.data_loaders.constants : WELL_PATCH_MAPPING
continuum_swr_replay.types : WellNodeData TypedDict

Examples
--------
>>> well_visits = make_well_visits_df(
...     beam_breaks, pump_events, position_info, track_graph
... )
>>> trials = make_trials_df_from_well_visits(well_visits)
>>> trials[["from_well", "to_well", "is_reward"]].head()

"""

import networkx as nx
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.constants import (
    OMISSION_KNOWN_LATENCY_SEC,
)

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
    # Handle empty DataFrame
    if len(beam_breaks) == 0:
        return pd.DataFrame(columns=["start_time", "end_time", "duration"])

    merged_events = []
    for _, df in beam_breaks.groupby("event_name"):
        # ``shift(-1)`` below assumes the rows are in chronological
        # order. Spyglass's ``load_dios`` returns sorted events, but
        # this helper is also called directly on caller-constructed
        # frames and the inter-event-interval / label cumsum would
        # silently produce negative-duration pokes if a row landed
        # out of order.
        df = df.sort_values("start_time").reset_index(drop=True)
        interevent_interval = (df.start_time.shift(-1) - df.end_time).fillna(0)
        labels = (
            (interevent_interval > max_inter_event_interval)
            .shift(1, fill_value=False)
            .cumsum()
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
    position_info: pd.DataFrame,
    track_graph: nx.Graph,
    times: pd.DataFrame,
    well_nodes: list[int] | None = None,
) -> WellNodeData:
    """Determine start and end well nodes for each interval.

    Finds the nearest well node to the animal's projected position at
    the start and end of each interval. Only considers actual well nodes
    (not internal junction nodes) to prevent misassignment.

    Parameters
    ----------
    position_info : pd.DataFrame
        Position data with projected position columns
    track_graph : nx.Graph
        Track graph with node positions
    times : pd.DataFrame
        Times with start_time and end_time columns
    well_nodes : list[int] | None
        Node IDs to consider as wells. If None, uses the well nodes
        from WELL_PATCH_MAPPING if they exist in the graph, otherwise
        falls back to all graph nodes.

    Returns
    -------
    WellNodeData
        TypedDict with 'from_well' and 'to_well' node ID arrays

    """
    projected_cols = ["projected_x_position", "projected_y_position"]

    # We want the position row at-or-before each interval boundary.
    # ``side="right"`` returns the *insertion index after* an exact match,
    # so ``searchsorted([0, 1], 0, side="right") == 1`` — without the
    # ``- 1`` correction we would sample the *next* row, mislabeling
    # ``from_well``/``to_well`` (e.g. for an interval [0, 1] over a
    # position frame at t=0 (well 0) and t=1 (well 1), both endpoints
    # would resolve to row 1 → from_well = to_well = 1). Subtract 1
    # then clamp to the valid index range.
    max_idx = len(position_info) - 1
    start_indices = np.clip(
        position_info.index.searchsorted(times.start_time.values, side="right") - 1,
        0,
        max_idx,
    )
    end_indices = np.clip(
        position_info.index.searchsorted(times.end_time.values, side="right") - 1,
        0,
        max_idx,
    )

    start_time_pos = position_info.iloc[start_indices][projected_cols].to_numpy()
    end_time_pos = position_info.iloc[end_indices][projected_cols].to_numpy()

    node_attrs = nx.get_node_attributes(track_graph, "pos")

    # Determine which nodes to consider as wells
    if well_nodes is not None:
        candidate_ids = sorted(well_nodes)
    else:
        # Default: use WELL_PATCH_MAPPING wells if they exist in the graph,
        # otherwise fall back to all graph nodes (for test compatibility).
        well_mapping_keys = sorted(WELL_PATCH_MAPPING.keys())
        if all(n in node_attrs for n in well_mapping_keys):
            candidate_ids = well_mapping_keys
        else:
            import logging

            logging.warning(
                "WELL_PATCH_MAPPING nodes not found in track graph. "
                "Falling back to all graph nodes for well assignment. "
                "This is expected in tests but not in production."
            )
            candidate_ids = sorted(node_attrs.keys())

    candidate_pos = np.array([node_attrs[n] for n in candidate_ids])
    candidate_arr = np.array(candidate_ids)

    start_node_ids = candidate_arr[
        np.argmin(
            np.linalg.norm(
                start_time_pos[:, None, :] - candidate_pos[None, :, :], axis=2
            ),
            axis=1,
        )
    ]
    end_node_ids = candidate_arr[
        np.argmin(
            np.linalg.norm(
                end_time_pos[:, None, :] - candidate_pos[None, :, :], axis=2
            ),
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

    # Vectorized reward detection: O(n_visits + n_pumps) instead of O(n_visits × n_pumps)
    # Check which well visits contain pump events using binary search
    pump_times_arr = np.sort(pump_events["start_time"].to_numpy(dtype=float))
    if not np.isfinite(pump_times_arr).all():
        raise ValueError("pump event start_time values must be finite")
    visit_starts = well_visits["start_time"].values
    visit_ends = well_visits["end_time"].values

    # Preserve the first physical pump onset inside each visit, not merely the
    # eventual binary label.  Downstream replay covariates need this timestamp
    # to keep goal arrival (destination known) separate from reward delivery
    # (outcome known).  ``NaN`` means no pump occurred in the visit.
    is_reward = np.zeros(len(well_visits), dtype=bool)
    reward_on_time = np.full(len(well_visits), np.nan)

    # For each pump event, find which visit(s) it belongs to
    for pump_time in pump_times_arr:
        # Find visits where start_time <= pump_time <= end_time
        is_in_visit = (visit_starts <= pump_time) & (pump_time <= visit_ends)
        is_reward |= is_in_visit
        first_pump = is_in_visit & ~np.isfinite(reward_on_time)
        reward_on_time[first_pump] = float(pump_time)

    well_visits["is_reward"] = is_reward
    well_visits["reward_on_time"] = reward_on_time
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
        pd.DataFrame(trial_number).ffill().bfill().astype(int).to_numpy().squeeze()
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
    if "reward_on_time" not in well_visits.columns:
        raise KeyError(
            "well_visits must carry reward_on_time; regenerate it with "
            "make_well_visits_df so trial outcomes have a physical timestamp"
        )

    trials = well_visits.groupby("trial_number").agg(
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
    trials = trials.merge(goal_time, left_index=True, right_index=True)

    # The trial's reward is the one delivered at its DESTINATION, so take the
    # first pump at or after goal arrival. A plain min over the trial's visits
    # also sees the origin-well poke -- on the first trial of a session that
    # produced a reward_on_time up to ~38 s BEFORE the arrival it was supposed
    # to follow, which then failed the [arrival, end_time] range check and
    # skipped the whole session.
    arrival = trials["goal_arrival_time"]
    visit_reward = well_visits[["trial_number", "reward_on_time"]].dropna(
        subset=["reward_on_time"]
    )
    visit_reward = visit_reward.join(arrival, on="trial_number")
    at_destination = visit_reward[
        visit_reward["reward_on_time"] >= visit_reward["goal_arrival_time"]
    ]
    trials["reward_on_time"] = (
        at_destination.groupby("trial_number")["reward_on_time"].min()
    )
    # Derive the label from the timestamp so the two cannot disagree. Taking
    # ``any(is_reward)`` over the trial's visits has the same origin-poke
    # contamination the timestamp just shed: it marked trials rewarded whose
    # only pump fired at the well they departed from, leaving a rewarded trial
    # with no destination pump to timestamp.
    trials["is_reward"] = trials["reward_on_time"].notna()
    rewarded = trials["is_reward"].to_numpy(dtype=bool)
    reward_on = trials["reward_on_time"].to_numpy(dtype=float)
    trial_end = trials["end_time"].to_numpy(dtype=float)
    arrival = trials["goal_arrival_time"].to_numpy(dtype=float)

    if (~np.isfinite(reward_on[rewarded])).any():
        bad = tuple(trials.index[rewarded & ~np.isfinite(reward_on)][:10])
        raise ValueError(
            "rewarded trials must have a finite reward_on_time; invalid "
            f"trial_number(s): {bad}"
        )
    if np.isfinite(reward_on[~rewarded]).any():
        bad = tuple(trials.index[(~rewarded) & np.isfinite(reward_on)][:10])
        raise ValueError(
            "omission trials cannot carry reward_on_time; invalid "
            f"trial_number(s): {bad}"
        )

    # Positive feedback is known at the first pump onset. An omission has no
    # cue to timestamp, so it is established once a pump would have fired had
    # the trial been rewarded -- ``goal_arrival_time`` (destination poke-in)
    # plus OMISSION_KNOWN_LATENCY_SEC, which exceeds every observed reward
    # latency. Capped at ``end_time``: leaving is itself conclusive, and the
    # boundary must stay inside the trial.
    outcome_known = np.where(
        rewarded,
        reward_on,
        np.minimum(arrival + OMISSION_KNOWN_LATENCY_SEC, trial_end),
    )
    if (
        not np.isfinite(outcome_known).all()
        or (outcome_known < arrival).any()
        or (outcome_known > trial_end).any()
    ):
        bad_mask = (
            ~np.isfinite(outcome_known)
            | (outcome_known < arrival)
            | (outcome_known > trial_end)
        )
        bad = tuple(trials.index[bad_mask][:10])
        raise ValueError(
            "trial outcome_known_time must be finite and lie between "
            "goal_arrival_time and end_time; invalid trial_number(s): "
            f"{bad}"
        )
    trials["outcome_known_time"] = outcome_known
    return trials
