"""Data preparation pipeline for replay content analysis.

Processes all sessions to produce a master event-level and long-format
dataset with all predictors pre-computed. This runs once (~2 hours for
218 sessions due to Spyglass loading) and saves pickles for fast iteration.

The primary output is count-based long format for the conditional Poisson
(multinomial-count) model. A binary long format is also produced for
ConditionalLogit robustness checks.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH

from .event_collapsing import (
    collapse_to_events,
    compute_arm_counts,
    make_long_format_arm,
)
from .predictors import (
    add_temporal_interactions,
    compute_cumulative_reward_for_event,
    compute_dist_to_arm,
    compute_event_time_since_stop,
    compute_time_since_stop,
    compute_visit_recency_for_event,
)
from .spatial_mapping import arm_to_well, map_estimate_to_arm, map_estimate_to_patch

logger = logging.getLogger(__name__)

ALL_ARMS = sorted(TRACK_SEGMENT_TO_PATCH.keys())


def add_event_uid(df: pd.DataFrame) -> pd.DataFrame:
    """Add a deterministic integer event ID column.

    Uses a hash of (animal, day, epoch, non_local_event) so that event
    UIDs are stable regardless of row ordering or whether the function is
    called on the full concatenated frame or a per-session subset.

    Hash collisions are possible but negligible for the dataset sizes in
    this project (~50k events). The UIDs are used only as grouping keys
    for ``ConditionalPoisson``, so a collision between events in different
    sessions would merge two unrelated events into one group — detectable
    as an event with 18 rows instead of 9.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: animal, day, epoch, non_local_event.

    Returns
    -------
    pd.DataFrame
        Input with an added ``event_uid`` column (int).
    """
    uid_cols = df[["animal", "day", "epoch", "non_local_event"]]
    df["event_uid"] = pd.util.hash_pandas_object(uid_cols, index=False) % (2**63)
    return df


def add_trial_context(
    dot: pd.DataFrame,
    trials: pd.DataFrame,
) -> pd.DataFrame:
    """Add trial-derived context columns to data_over_time.

    Uses **in-progress trip semantics**: at each timepoint t, the
    "current trip" is the latest trial whose ``start_time`` is at or
    before t. This is the trip the animal is currently completing
    (or has just completed, if t falls in the gap between two trips).

    The trip the animal is currently on tells us:
    - where it came from (``prev_trial_from_well`` = current trip's
      from_well)
    - where it has arrived / will arrive (``prev_trial_to_well`` =
      current trip's to_well — equal to the animal's current well during
      sustained immobility at a destination)
    - whether the trip was a patch switch / omission
      (``is_after_switch``, ``is_after_omission``)

    The "next trip" is the trial immediately following the current one
    in the trial sequence (trial boundaries are contiguous, so we use
    positional indexing rather than a strict time comparison).

    .. note::
        This is a fix relative to the previous behavior, which used
        ``searchsorted(end_times, t)`` and returned the previously
        *completed* trip — incorrect when the animal has already arrived
        at the destination of an in-progress trip whose ``end_time`` is
        still in the future (e.g., during sustained immobility at a
        well, before the trial detector formally closes the trial).

    Parameters
    ----------
    dot : pd.DataFrame
        data_over_time DataFrame (time-indexed).
    trials : pd.DataFrame
        Trial DataFrame with columns: start_time, end_time, to_well,
        from_well, to_patch, from_patch, is_reward, is_patch_change.

    Returns
    -------
    pd.DataFrame
        dot with added columns (modified in place and returned).
    """
    dot_times = dot.index.values
    n_trials = len(trials)
    trial_start_times = trials["start_time"].values
    to_patches = trials["to_patch"].values
    from_patches = trials["from_patch"].values
    to_wells = trials["to_well"].values
    from_wells = trials["from_well"].values
    is_reward = trials["is_reward"].values
    is_patch_change = trials["is_patch_change"].values

    # Current trip = latest trial whose start_time <= dot_time. This is
    # the trip the animal is currently completing or has just finished.
    current_idx = np.searchsorted(trial_start_times, dot_times, side="right") - 1
    valid = current_idx >= 0  # at least one trip has started

    current_clipped = np.clip(current_idx, 0, n_trials - 1)
    next_idx = current_idx + 1
    next_valid = valid & (next_idx < n_trials)
    next_clipped = np.clip(next_idx, 0, n_trials - 1)

    # Patch-level
    dot["prev_trial_from_patch"] = np.where(
        valid, from_patches[current_clipped], np.nan
    )
    dot["next_trial_to_patch"] = np.where(
        next_valid, to_patches[next_clipped], np.nan
    )

    # Well-level
    dot["prev_trial_from_well"] = np.where(
        valid, from_wells[current_clipped], np.nan
    )
    dot["prev_trial_to_well"] = np.where(
        valid, to_wells[current_clipped], np.nan
    )
    dot["next_trial_to_well"] = np.where(
        next_valid, to_wells[next_clipped], np.nan
    )

    # Most recently rewarded well: use end times here because reward is
    # only "delivered" at trial completion. Use the latest rewarded trip
    # whose end_time is at or before t.
    trial_end_times = trials["end_time"].values
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

    # Task context flags: properties of the current (in-progress) trip.
    dot["is_after_switch"] = np.where(
        valid, is_patch_change[current_clipped].astype(float), np.nan
    )
    dot["is_after_omission"] = np.where(
        valid, (~is_reward[current_clipped]).astype(float), np.nan
    )

    return dot


def _add_per_arm_predictors_single_session(
    long_df: pd.DataFrame,
    events: pd.DataFrame,
    trials: pd.DataFrame,
    track_graph,
) -> pd.DataFrame:
    """Add visit_recency, cumulative_reward, and dist_to_arm for one session.

    Parameters
    ----------
    long_df : pd.DataFrame
        Long-format data (9 rows per event) for this session.
    events : pd.DataFrame
        Event-level data for this session.
    trials : pd.DataFrame
        Trials DataFrame for this session.
    track_graph : networkx.Graph
        Track graph for this session.

    Returns
    -------
    pd.DataFrame
        long_df with added columns.
    """
    merge_keys = ["animal", "day", "epoch", "non_local_event"]
    rows = []

    for _, event in events.iterrows():
        event_time = event.get("event_midpoint_time", np.nan)

        # visit_recency and cumulative_reward require event_midpoint_time
        if not np.isnan(event_time):
            vr = compute_visit_recency_for_event(event_time, trials, track_graph)
            cr = compute_cumulative_reward_for_event(event_time, trials, track_graph)
        else:
            vr = dict.fromkeys(ALL_ARMS, np.nan)
            cr = dict.fromkeys(ALL_ARMS, np.nan)

        # dist_to_arm: distance from the animal's current well to each arm.
        # Derive the current well from the modal track_segment_id during the
        # event (the arm the animal is physically on). This is more accurate
        # than prev_trial_to_well when the animal has already arrived at the
        # destination of an in-progress trial that hasn't formally "ended".
        current_arm = event.get("track_segment_id", np.nan)
        current_well = arm_to_well(current_arm, track_graph)
        # Fall back to prev_trial_to_well if track_segment_id is not a leaf arm
        if np.isnan(current_well):
            current_well = event.get("prev_trial_to_well", np.nan)
        dists = compute_dist_to_arm(current_well, track_graph)

        for arm in ALL_ARMS:
            row = {c: event[c] for c in merge_keys}
            row["arm_id"] = arm
            row["visit_recency"] = vr[arm]
            row["cumulative_reward"] = cr[arm]
            row["dist_to_arm"] = dists[arm]
            rows.append(row)

    if not rows:
        for col in ["visit_recency", "cumulative_reward", "dist_to_arm"]:
            long_df[col] = np.nan
        return long_df

    predictor_df = pd.DataFrame(rows)

    long_df = long_df.merge(
        predictor_df,
        on=merge_keys + ["arm_id"],
        how="left",
        suffixes=("", "_new"),
    )

    # Handle case where columns already exist (from a prior run)
    for col in ["visit_recency", "cumulative_reward", "dist_to_arm"]:
        new_col = f"{col}_new"
        if new_col in long_df.columns:
            long_df[col] = long_df[new_col]
            long_df = long_df.drop(columns=[new_col])

    return long_df


def prepare_session(
    dot: pd.DataFrame,
    env,
    trials: pd.DataFrame,
    track_graph,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    """Process one session into event-level and long-format DataFrames.

    Produces both count-based (primary, for ConditionalPoisson) and binary
    (robustness, for ConditionalLogit) long-format outputs. All predictors
    required by the M0-M7 hierarchy are computed here, including per-arm
    continuous predictors (visit_recency, cumulative_reward, dist_to_arm)
    and temporal/SWR/context interaction columns.

    Both long-format outputs include ``event_uid`` for use as the grouping
    variable in conditional models.

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
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None
        (events, long_count, long_binary) or None if insufficient data.
        long_count has ``arm_count`` response for ConditionalPoisson.
        long_binary has ``replay_here`` response for ConditionalLogit.
    """
    if trials is None or len(trials) < 3:
        return None

    if track_graph is None:
        return None

    # Map decoded positions
    dot["replay_patch"] = map_estimate_to_patch(dot["map_estimate"], env)
    dot["replay_arm"] = map_estimate_to_arm(dot["map_estimate"], env)
    dot["Non-Local_prob"] = (
        dot["Non-Local_Continuous_prob"] + dot["Non-Local_Fragmented_prob"]
    )
    dot["abs_distance"] = dot["ahead_behind_distance"].abs()

    # Add trial context (includes prev_trial_to_well for dist_to_arm)
    dot = add_trial_context(dot, trials)

    # Filter to non-local immobile
    nl = dot[
        (dot["head_speed"] < DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC)
        & (dot["Non-Local_prob"] > DEFAULT_NON_LOCAL_THRESHOLD)
    ].copy()

    if len(nl) < 100:
        return None

    nl["is_swr"] = nl["SWR_event"] > 0

    # Compute time_since_stop using the full dot for immobility bout detection
    nl["time_since_stop"] = compute_time_since_stop(nl, dot=dot)

    # Collapse to events (arm level) — still needed for modal arm + context
    events = collapse_to_events(nl, level="arm")
    if len(events) < 20:
        return None

    # Compute per-arm timebin counts
    arm_counts = compute_arm_counts(nl)

    # Add event-level continuous predictors
    events["time_since_stop"] = compute_event_time_since_stop(events, nl)

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

    # Add prev_trial_to_well to events (modal value per event, for dist_to_arm)
    if "prev_trial_to_well" in nl.columns:
        prev_to_well = (
            nl[nl["non_local_event"] > 0]
            .groupby(["animal", "day", "epoch", "non_local_event"])[
                "prev_trial_to_well"
            ]
            .first()
            .reset_index()
        )
        events = events.merge(
            prev_to_well,
            on=["animal", "day", "epoch", "non_local_event"],
            how="left",
        )

    # Merge arm counts into events (for total_count filtering)
    merge_keys = ["animal", "day", "epoch", "non_local_event"]
    events = events.merge(
        arm_counts[merge_keys + ["total_count"]],
        on=merge_keys,
        how="left",
    )

    # Filter: keep events with total_count > 0 (at least one timebin
    # decoded to a real arm, not just junction bins)
    events = events[events["total_count"] > 0].copy()
    if len(events) < 20:
        return None

    n_arms = 9

    # --- Count-based long format (primary: ConditionalPoisson) ---
    long_count = make_long_format_arm(
        events, track_graph, response="count", arm_counts_df=arm_counts
    )

    if len(long_count) != len(events) * n_arms:
        raise ValueError(
            f"Misaligned: {len(long_count)} long rows != {len(events)} * {n_arms}"
        )

    # Broadcast event-level predictors to count long format
    for col in ["time_since_stop", "event_midpoint_time"]:
        if col in events.columns:
            long_count[col] = np.repeat(events[col].values, n_arms)

    # Add per-arm continuous predictors (visit_recency, cumulative_reward,
    # dist_to_arm) and temporal interaction columns
    long_count = _add_per_arm_predictors_single_session(
        long_count, events, trials, track_graph
    )
    add_temporal_interactions(long_count)

    # Add event_uid
    long_count = add_event_uid(long_count)

    # --- Binary long format (robustness: ConditionalLogit) ---
    events_binary = events[events["replay_arm"].notna()].copy()
    long_binary = pd.DataFrame()
    if len(events_binary) >= 20:
        long_binary = make_long_format_arm(
            events_binary, track_graph, response="binary"
        )
        # Broadcast identifiers and predictors
        for col in ["time_since_stop", "event_midpoint_time"]:
            if col in events_binary.columns:
                long_binary[col] = np.repeat(events_binary[col].values, n_arms)
        for col in merge_keys:
            long_binary[col] = np.repeat(events_binary[col].values, n_arms)
        long_binary["arm_id"] = np.tile(ALL_ARMS, len(events_binary))

        # Same per-arm predictors and interactions for robustness path
        long_binary = _add_per_arm_predictors_single_session(
            long_binary, events_binary, trials, track_graph
        )
        add_temporal_interactions(long_binary)

        # Add event_uid
        long_binary = add_event_uid(long_binary)

    return events, long_count, long_binary
