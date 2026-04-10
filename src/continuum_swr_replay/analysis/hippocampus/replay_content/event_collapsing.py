"""Collapse non-local timebins to discrete events and construct long-format data.

Functions for aggregating per-timebin decoding results into per-event summaries,
and for converting event-level data into the long format needed for GLM fitting
(one row per event x spatial unit).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH

from .spatial_mapping import well_to_arm_set


def _modal(series: pd.Series) -> float:
    """Mode of a series, or NaN if empty."""
    valid = series.dropna()
    return valid.mode().iloc[0] if len(valid) > 0 else np.nan


def collapse_to_events(
    nl: pd.DataFrame,
    level: str = "arm",
) -> pd.DataFrame:
    """Collapse non-local timebins to one row per event.

    Parameters
    ----------
    nl : pd.DataFrame
        Non-local immobile timebins with columns including non_local_event,
        SWR_event, and spatial/context columns.
    level : {"patch", "arm"}
        Spatial level for aggregation.

    Returns
    -------
    pd.DataFrame
        One row per event with modal spatial values and onset context flags.
    """
    ev = nl[nl["non_local_event"] > 0]
    if len(ev) == 0:
        return pd.DataFrame()

    gcols = ["animal", "day", "epoch", "non_local_event"]

    common_cols = ["is_swr_flag", "is_after_switch", "is_after_omission"]
    if level == "patch":
        spatial_cols = [
            "replay_patch",
            "patch_id",
            "prev_trial_from_patch",
            "next_trial_to_patch",
            "last_rewarded_patch",
        ]
    else:
        spatial_cols = [
            "replay_arm",
            "track_segment_id",
            "prev_trial_from_well",
            "next_trial_to_well",
            "last_rewarded_well",
        ]

    ev = ev.copy()
    ev["is_swr_flag"] = (ev["SWR_event"] > 0).astype(float)

    all_cols = spatial_cols + common_cols
    available = [c for c in all_cols if c in ev.columns]

    # Context flags use event-onset value (first timebin), not mode
    onset_cols = {"is_after_switch", "is_after_omission"}

    def _summary(g: pd.DataFrame) -> pd.Series:
        result = {}
        for col in available:
            if col == "is_swr_flag":
                result[col] = float((g["SWR_event"] > 0).any())
            elif col in onset_cols:
                result[col] = g[col].iloc[0] if g[col].notna().any() else np.nan
            else:
                result[col] = _modal(g[col])
        return pd.Series(result)

    return (
        ev.groupby(gcols)
        .apply(_summary, include_groups=False)
        .dropna(how="all")
        .reset_index()
    )


def make_long_format_arm(
    event_df: pd.DataFrame,
    track_graph,
) -> pd.DataFrame:
    """Convert events to long format for arm-level model (9 rows per event).

    For each event, creates one row per arm (track segment 0-8). Binary
    indicators mark whether each arm matches the current, previous, next,
    or last-rewarded location.

    Parameters
    ----------
    event_df : pd.DataFrame
        Event-level data from collapse_to_events(level="arm").
    track_graph : networkx.Graph
        Track graph with edge_id attributes on edges.

    Returns
    -------
    pd.DataFrame
        Long-format data with replay_here response and binary predictors.
    """
    # Filter out events with NaN replay_arm (junction bins) — these would
    # produce all-zero replay_here rows, biasing the intercept.
    event_df = event_df[event_df["replay_arm"].notna()].copy()

    all_arms = sorted(set(TRACK_SEGMENT_TO_PATCH.keys()))
    rows = []
    for _, e in event_df.iterrows():
        current_arm = e.get("track_segment_id", np.nan)
        rewarded_arms = well_to_arm_set(
            e.get("last_rewarded_well", np.nan), track_graph
        )
        prev_arms = well_to_arm_set(e.get("prev_trial_from_well", np.nan), track_graph)
        next_arms = well_to_arm_set(e.get("next_trial_to_well", np.nan), track_graph)

        replay_arm = e.get("replay_arm", np.nan)
        for arm in all_arms:
            row = {
                "replay_here": int(replay_arm == arm),
                "is_current": int(current_arm == arm),
                "is_last_rewarded": int(arm in rewarded_arms),
                "is_prev": int(arm in prev_arms),
                "is_next": int(arm in next_arms),
                "is_swr": int(e.get("is_swr_flag", 0)),
                "is_after_switch": float(e.get("is_after_switch", np.nan)),
                "is_after_omission": float(e.get("is_after_omission", np.nan)),
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    add_interaction_columns(df)
    return df


def add_interaction_columns(df: pd.DataFrame) -> None:
    """Add interaction terms to long-format DataFrame (in place).

    Creates switch, omission, and SWR interaction columns from existing
    binary predictors.
    """
    df["switch_x_current"] = df["is_after_switch"] * df["is_current"]
    df["switch_x_prev"] = df["is_after_switch"] * df["is_prev"]
    df["omission_x_rewarded"] = df["is_after_omission"] * df["is_last_rewarded"]
    # SWR interactions
    df["swr_x_current"] = df["is_swr"] * df["is_current"]
    df["swr_x_rewarded"] = df["is_swr"] * df["is_last_rewarded"]
    df["swr_x_prev"] = df["is_swr"] * df["is_prev"]
    if "is_next" in df.columns:
        df["swr_x_next"] = df["is_swr"] * df["is_next"]
    else:
        df["swr_x_next"] = np.zeros(len(df))
