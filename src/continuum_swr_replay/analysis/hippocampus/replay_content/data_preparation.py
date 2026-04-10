"""Data preparation pipeline for replay content analysis.

Processes all sessions to produce a master event-level and long-format
dataset with all predictors pre-computed. This runs once (~2 hours for
218 sessions due to Spyglass loading) and saves pickles for fast iteration.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)

from .event_collapsing import collapse_to_events, make_long_format_arm
from .predictors import (
    compute_event_log_distance,
    compute_event_time_since_stop,
    compute_time_since_stop,
)
from .spatial_mapping import map_estimate_to_arm, map_estimate_to_patch

logger = logging.getLogger(__name__)


def add_trial_context(
    dot: pd.DataFrame,
    trials: pd.DataFrame,
) -> pd.DataFrame:
    """Add trial-derived context columns to data_over_time.

    Adds per-timepoint columns for previous trial origin well, next trial
    destination well, most recently rewarded well, and task context flags
    (is_after_switch, is_after_omission).

    Parameters
    ----------
    dot : pd.DataFrame
        data_over_time DataFrame (time-indexed).
    trials : pd.DataFrame
        Trial DataFrame with columns: end_time, to_well, from_well,
        to_patch, from_patch, is_reward, is_patch_change.

    Returns
    -------
    pd.DataFrame
        dot with added columns (modified in place and returned).
    """
    dot_times = dot.index.values
    trial_end_times = trials["end_time"].values
    trial_idx = np.searchsorted(trial_end_times, dot_times, side="right")
    n_trials = len(trials)
    to_patches = trials["to_patch"].values
    from_patches = trials["from_patch"].values
    to_wells = trials["to_well"].values
    from_wells = trials["from_well"].values
    is_reward = trials["is_reward"].values
    is_patch_change = trials["is_patch_change"].values

    prev_idx = np.clip(trial_idx - 1, 0, n_trials - 1)
    next_idx = np.clip(trial_idx, 0, n_trials - 1)

    # Patch-level
    dot["prev_trial_from_patch"] = np.where(
        trial_idx > 0, from_patches[prev_idx], np.nan
    )
    dot["next_trial_to_patch"] = np.where(
        trial_idx < n_trials, to_patches[next_idx], np.nan
    )

    # Well-level
    dot["prev_trial_from_well"] = np.where(trial_idx > 0, from_wells[prev_idx], np.nan)
    dot["next_trial_to_well"] = np.where(
        trial_idx < n_trials, to_wells[next_idx], np.nan
    )

    # Most recently rewarded well
    rewarded_end_times = trial_end_times[is_reward]
    rewarded_wells = to_wells[is_reward]
    if len(rewarded_end_times) > 0:
        rew_idx = np.searchsorted(rewarded_end_times, dot_times, side="right") - 1
        rew_idx_clipped = np.clip(rew_idx, 0, None)
        dot["last_rewarded_well"] = np.where(
            rew_idx >= 0, rewarded_wells[rew_idx_clipped], np.nan
        )
    else:
        dot["last_rewarded_well"] = np.nan

    # Task context flags
    dot["is_after_switch"] = np.where(
        trial_idx > 0, is_patch_change[prev_idx].astype(float), np.nan
    )
    dot["is_after_omission"] = np.where(
        trial_idx > 0, (~is_reward[prev_idx]).astype(float), np.nan
    )

    return dot


def prepare_session(
    dot: pd.DataFrame,
    env,
    trials: pd.DataFrame,
    track_graph,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Process one session into event-level and long-format DataFrames.

    Parameters
    ----------
    dot : pd.DataFrame
        Pre-saved data_over_time (from pickle).
    env : Environment
        Decoder environment (from pickle).
    trials : pd.DataFrame
        Trial DataFrame (from Spyglass).
    track_graph : networkx.Graph
        Track graph (from Spyglass).

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame] | None
        (events, long_format) or None if insufficient data.
    """
    if trials is None or len(trials) < 3:
        return None

    # Map decoded positions
    dot["replay_patch"] = map_estimate_to_patch(dot["map_estimate"], env)
    dot["replay_arm"] = map_estimate_to_arm(dot["map_estimate"], env)
    dot["Non-Local_prob"] = (
        dot["Non-Local_Continuous_prob"] + dot["Non-Local_Fragmented_prob"]
    )
    dot["abs_distance"] = dot["ahead_behind_distance"].abs()

    # Add trial context
    dot = add_trial_context(dot, trials)

    # Filter to non-local immobile
    nl = dot[
        (dot["head_speed"] < DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC)
        & (dot["Non-Local_prob"] > DEFAULT_NON_LOCAL_THRESHOLD)
    ].copy()

    if len(nl) < 100:
        return None

    nl["is_swr"] = nl["SWR_event"] > 0

    # Compute time_since_stop using the full dot for immobility bout detection.
    # This gives "time since the animal stopped moving," not just "time since
    # the non-local cluster started."
    nl["time_since_stop"] = compute_time_since_stop(nl, dot=dot)

    # Collapse to events (arm level)
    events = collapse_to_events(nl, level="arm")
    if len(events) < 20:
        return None

    # Add event-level continuous predictors
    events["time_since_stop"] = compute_event_time_since_stop(events, nl)
    events["log_distance"] = compute_event_log_distance(events, nl)

    # Add event midpoint time for visit_recency computation
    event_midpoints = (
        nl[nl["non_local_event"] > 0]
        .groupby(["animal", "day", "epoch", "non_local_event"])
        .apply(
            lambda g: pd.Series({"event_midpoint_time": g.index[len(g) // 2]}),
            include_groups=False,
        )
        .reset_index()
    )
    events = events.merge(
        event_midpoints,
        on=["animal", "day", "epoch", "non_local_event"],
        how="left",
    )

    # Make long format (this filters out events with NaN replay_arm)
    if track_graph is None:
        return None

    # Filter events to only those with valid replay_arm BEFORE long format
    # construction, so events and long stay aligned.
    events = events[events["replay_arm"].notna()].copy()
    if len(events) < 20:
        return None

    long = make_long_format_arm(events, track_graph)

    # Now events and long are aligned: len(long) == len(events) * 9
    n_arms = 9
    assert len(long) == len(events) * n_arms, (
        f"Misaligned: {len(long)} long rows != {len(events)} events * {n_arms}"
    )

    # Broadcast event-level continuous predictors to long format
    for col in ["time_since_stop", "log_distance", "event_midpoint_time"]:
        if col in events.columns:
            long[col] = np.repeat(events[col].values, n_arms)

    # Broadcast session identifiers
    for col in ["animal", "day", "epoch", "non_local_event"]:
        long[col] = np.repeat(events[col].values, n_arms)

    # Add arm_id column
    from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH

    all_arms = sorted(TRACK_SEGMENT_TO_PATCH.keys())
    long["arm_id"] = np.tile(all_arms, len(events))

    return events, long
