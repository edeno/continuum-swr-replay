"""Collapse non-local timebins to discrete events and construct long-format data.

Functions for aggregating per-timebin decoding results into per-event summaries,
and for converting event-level data into the long format needed for the
conditional Poisson (multinomial-count) model (one row per event x arm).

Two response formats are supported:
- ``"count"``: integer timebin counts per arm (primary, for ConditionalPoisson)
- ``"binary"``: 0/1 modal-arm indicator (robustness, for ConditionalLogit)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH

from .spatial_mapping import well_to_arm_set

ALL_ARMS = sorted(TRACK_SEGMENT_TO_PATCH.keys())


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


def compute_arm_counts(
    nl: pd.DataFrame,
) -> pd.DataFrame:
    """Count decoded timebins per arm for each non-local event.

    For each event, counts how many timebins have their MAP decoded position
    on each of the 9 arms. Timebins on junction edges (replay_arm not in
    ALL_ARMS) are excluded from counts.

    Parameters
    ----------
    nl : pd.DataFrame
        Non-local immobile timebins with ``non_local_event`` and
        ``replay_arm`` columns. Only rows with non_local_event > 0 are used.

    Returns
    -------
    pd.DataFrame
        One row per event with columns: animal, day, epoch, non_local_event,
        arm_count_{arm_id} for each arm, total_count (L_e), plus event-level
        context columns (is_swr_flag, is_after_switch, is_after_omission).
    """
    ev = nl[nl["non_local_event"] > 0].copy()
    if len(ev) == 0:
        return pd.DataFrame()

    gcols = ["animal", "day", "epoch", "non_local_event"]

    # Count timebins per arm (excluding junction bins where replay_arm
    # is not in ALL_ARMS — these are typically edge_id=-1 or NaN)
    arm_set = set(ALL_ARMS)
    ev_valid = ev[ev["replay_arm"].isin(arm_set)]

    counts = (
        ev_valid.groupby(gcols + ["replay_arm"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=ALL_ARMS, fill_value=0)
    )
    counts.columns = [f"arm_count_{a}" for a in ALL_ARMS]
    counts["total_count"] = counts.sum(axis=1)

    result = counts.reset_index()

    # Add event-level context flags
    if "SWR_event" in ev.columns:
        ev["is_swr_flag"] = (ev["SWR_event"] > 0).astype(float)

    context_rows = []
    for keys, g in ev.groupby(gcols):
        row = dict(zip(gcols, keys, strict=True))
        if "is_swr_flag" in ev.columns:
            row["is_swr_flag"] = float(g["is_swr_flag"].any())
        for col in ["is_after_switch", "is_after_omission"]:
            if col in ev.columns:
                row[col] = g[col].iloc[0] if g[col].notna().any() else np.nan
        context_rows.append(row)

    if context_rows:
        context_df = pd.DataFrame(context_rows)
        result = result.merge(context_df, on=gcols, how="left")

    return result


def make_long_format_arm(
    event_df: pd.DataFrame,
    track_graph,
    response: str = "count",
    arm_counts_df: pd.DataFrame | None = None,
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
    response : {"count", "binary"}
        Response variable format:
        - ``"count"``: ``arm_count`` column with integer timebin counts per arm
          (for ConditionalPoisson). Requires ``arm_counts_df``.
        - ``"binary"``: ``replay_here`` column with 0/1 modal-arm indicator
          (for ConditionalLogit robustness check).
    arm_counts_df : pd.DataFrame | None
        Output of ``compute_arm_counts``. Required when ``response="count"``.
        Must have columns ``arm_count_{arm_id}`` for each arm and
        ``total_count`` (L_e).

    Returns
    -------
    pd.DataFrame
        Long-format data with response column and binary predictors.
    """
    if response == "count" and arm_counts_df is None:
        raise ValueError("arm_counts_df is required when response='count'")

    if response == "binary":
        # Filter out events with NaN replay_arm (junction bins)
        event_df = event_df[event_df["replay_arm"].notna()].copy()
    elif response == "count":
        # For count response, keep events with total_count > 0
        merge_keys = ["animal", "day", "epoch", "non_local_event"]
        valid_events = arm_counts_df[arm_counts_df["total_count"] > 0][merge_keys]
        event_df = event_df.merge(valid_events, on=merge_keys, how="inner").copy()

    rows = []
    for _, e in event_df.iterrows():
        current_arm = e.get("track_segment_id", np.nan)
        rewarded_arms = well_to_arm_set(
            e.get("last_rewarded_well", np.nan), track_graph
        )
        prev_arms = well_to_arm_set(e.get("prev_trial_from_well", np.nan), track_graph)
        next_arms = well_to_arm_set(e.get("next_trial_to_well", np.nan), track_graph)

        replay_arm = e.get("replay_arm", np.nan)
        for arm in ALL_ARMS:
            row = {
                "arm_id": arm,
                "is_current": int(current_arm == arm),
                "is_last_rewarded": int(arm in rewarded_arms),
                "is_prev": int(arm in prev_arms),
                "is_next": int(arm in next_arms),
                "is_swr": int(e.get("is_swr_flag", 0)),
                "is_after_switch": float(e.get("is_after_switch", np.nan)),
                "is_after_omission": float(e.get("is_after_omission", np.nan)),
            }
            if response == "binary":
                row["replay_here"] = int(replay_arm == arm)
            rows.append(row)

    df = pd.DataFrame(rows)

    if response == "count":
        merge_keys = ["animal", "day", "epoch", "non_local_event"]
        # Broadcast event identifiers to long format using concat (not
        # np.repeat) to avoid order-dependence on merge reordering.
        n_arms = len(ALL_ARMS)
        id_df = pd.DataFrame(
            np.repeat(event_df[merge_keys].values, n_arms, axis=0),
            columns=merge_keys,
        )
        df = pd.concat([df.reset_index(drop=True), id_df.reset_index(drop=True)], axis=1)

        # Melt arm_counts_df from wide (arm_count_0 ... arm_count_8) to long
        count_cols = [f"arm_count_{a}" for a in ALL_ARMS]
        count_long_df = arm_counts_df[merge_keys + count_cols].melt(
            id_vars=merge_keys,
            value_vars=count_cols,
            var_name="_arm_col",
            value_name="arm_count",
        )
        # Extract arm_id from column name "arm_count_{arm_id}"
        count_long_df["arm_id"] = (
            count_long_df["_arm_col"].str.removeprefix("arm_count_").astype(int)
        )
        count_long_df = count_long_df.drop(columns=["_arm_col"])

        df = df.merge(
            count_long_df,
            on=merge_keys + ["arm_id"],
            how="left",
        )
        df["arm_count"] = df["arm_count"].fillna(0).astype(int)

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
