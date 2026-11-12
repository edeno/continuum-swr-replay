"""Shared derivation of candidate-arm predictors.

Replay and ContFrag have different event detectors and response definitions,
but a covariate with the same canonical name must have one implementation.
This module owns the common nine-arm choice-set expansion after a source has
constructed its event rows and response columns.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders.constants import ALL_ARMS

from .._helpers import (
    arm_to_patch,
    build_well_to_arm_map,
    eq_indicator,
    eq_indicator_or_nan,
)
from ..design_constants import RECENCY_SPLINE_COLUMNS
from ..spatial_mapping import arm_to_well
from .history import (
    attach_arm_uncertainty,
    attach_contingency_value_block,
    center_visit_recency_within_event,
    recent_reward_rate_all_events,
    rolling_reward_count_all_events,
    spline_visit_recency_within_event,
    visit_recency_and_never_visited_all_events,
)
from .history._common import _prepare_port_experiences
from .interactions import attach_interactions
from .spatial import arm_dist_rank, dist_to_arm_from_well


def broadcast_candidate_arms(event_table: pd.DataFrame) -> pd.DataFrame:
    """Repeat each event over the exact canonical nine-arm choice set."""
    if "event_id" not in event_table:
        raise ValueError("event_table is missing event_id")
    if (
        event_table["event_id"].isna().any()
        or event_table["event_id"].duplicated().any()
    ):
        raise ValueError("event_table requires one nonmissing row per event_id")
    n_arms = len(ALL_ARMS)
    output = event_table.loc[event_table.index.repeat(n_arms)].reset_index(drop=True)
    output["candidate_arm"] = np.tile(ALL_ARMS, len(event_table))
    return output


def _distance_and_rank_by_event(
    occupancy_well_arm: np.ndarray, track_graph: nx.Graph
) -> tuple[np.ndarray, np.ndarray]:
    """Graph distance and rank for each event × candidate arm.

    Repeated occupied arms share a cached graph calculation. Unknown occupancy
    yields an all-NaN row rather than a fabricated spatial reference.
    """
    occupancy = np.asarray(occupancy_well_arm, dtype=float)
    n_arms = len(ALL_ARMS)
    distances = np.full((len(occupancy), n_arms), np.nan)
    ranks = np.full_like(distances, np.nan)
    cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for row, occupied_arm in enumerate(occupancy):
        if not np.isfinite(occupied_arm):
            continue
        key = int(occupied_arm)
        if key not in cache:
            occupied_well = arm_to_well(occupied_arm, track_graph)
            distance_by_arm = dist_to_arm_from_well(occupied_well, track_graph)
            distance = np.asarray(
                [distance_by_arm[arm] for arm in ALL_ARMS], dtype=float
            )
            cache[key] = distance, arm_dist_rank(distance)
        distances[row], ranks[row] = cache[key]
    return distances, ranks


def _history_by_event(
    event_times: np.ndarray,
    port_experiences: pd.DataFrame,
    track_graph: nx.Graph,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute every common arm-history family in one event-aligned pass."""
    times = np.asarray(event_times, dtype=float)
    n_arms = len(ALL_ARMS)
    shape = (len(times), n_arms)
    recency = np.full(shape, np.nan)
    never_visited = np.full(shape, np.nan)
    reward_count = np.full(shape, np.nan)
    reward_fraction = np.full(shape, np.nan)
    finite = np.isfinite(times)
    experiences = _prepare_port_experiences(port_experiences)
    if experiences.empty or not finite.any():
        return recency, never_visited, reward_count, reward_fraction

    well_to_arm = build_well_to_arm_map(track_graph)
    query_times = times[finite]
    recency[finite], never_visited[finite] = visit_recency_and_never_visited_all_events(
        query_times,
        experiences,
        well_to_arm,
    )
    reward_count[finite] = rolling_reward_count_all_events(
        query_times,
        experiences,
        well_to_arm,
        window=10,
    )
    reward_fraction[finite] = recent_reward_rate_all_events(
        query_times,
        experiences,
        well_to_arm,
        k=10,
    )
    return recency, never_visited, reward_count, reward_fraction


def attach_candidate_arm_covariates(
    frame: pd.DataFrame,
    *,
    event_table: pd.DataFrame,
    port_experiences: pd.DataFrame,
    track_graph: nx.Graph,
    contingency_stop_id: np.ndarray,
    contingency: pd.DataFrame | None,
    include_belief_variance: bool,
) -> pd.DataFrame:
    """Attach the common candidate-arm predictors and interactions.

    Parameters
    ----------
    frame
        Event-major, candidate-minor frame containing exactly nine canonical
        candidate rows per event plus source-specific response columns.
    event_table
        One row per event in the same order used to construct ``frame``.
    port_experiences, track_graph
        Canonical reward-port experience history and task graph.
    contingency_stop_id
        Qvals join key for each candidate row. Replay uses its canonical
        ``stop_id``; ContFrag supplies the corresponding plain Replay stop key.
    contingency
        Canonical qvals table, or ``None`` when values are unavailable.
    include_belief_variance
        Whether this table's published schema carries candidate-arm Beta
        variance and its SWR interaction.

    Returns
    -------
    pandas.DataFrame
        A copy with all shared candidate-arm predictors attached.
    """
    required = {
        "event_id",
        "candidate_arm",
        "occupancy_well_arm",
        "prev_arm",
        "next_arm",
        "last_rewarded_arm",
        "current_patch",
        "event_start_s",
        "time_since_stop_start_s",
        "is_swr",
        "is_mua",
        "is_rewarded_outcome",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"candidate-arm input is missing columns {missing}")
    if len(frame) != len(event_table) * len(ALL_ARMS):
        raise ValueError(
            "candidate-arm input must contain exactly nine rows per event; "
            f"got {len(frame)} rows for {len(event_table)} events"
        )
    expected_event_ids = np.repeat(event_table["event_id"].to_numpy(), len(ALL_ARMS))
    expected_arms = np.tile(ALL_ARMS, len(event_table))
    if not np.array_equal(frame["event_id"].to_numpy(), expected_event_ids):
        raise ValueError("candidate-arm rows are not aligned to event_table order")
    if not np.array_equal(frame["candidate_arm"].to_numpy(), expected_arms):
        raise ValueError("candidate-arm rows are not in canonical arm order")
    contingency_stop_id = np.asarray(contingency_stop_id)
    if contingency_stop_id.shape != (len(frame),):
        raise ValueError(
            "contingency_stop_id must have one value per candidate row; "
            f"got {contingency_stop_id.shape}, expected {(len(frame),)}"
        )

    output = frame.copy()
    arm = output["candidate_arm"].to_numpy()
    current = output["occupancy_well_arm"].to_numpy()
    output["is_current_arm"] = eq_indicator_or_nan(arm, current)
    output["is_prev_arm"] = eq_indicator(arm, output["prev_arm"].to_numpy())
    output["is_next_arm"] = eq_indicator(arm, output["next_arm"].to_numpy())
    output["is_last_rewarded_arm"] = eq_indicator(
        arm,
        output["last_rewarded_arm"].to_numpy(),
    )
    output["is_last_rewarded_noncurrent_arm"] = output[
        "is_last_rewarded_arm"
    ].to_numpy() * (1.0 - output["is_current_arm"].to_numpy())
    output["arm_patch_id"] = np.asarray([arm_to_patch(value) for value in arm])
    output["is_same_patch_as_current"] = eq_indicator_or_nan(
        output["arm_patch_id"].to_numpy(),
        output["current_patch"].to_numpy(),
    )

    distances, ranks = _distance_and_rank_by_event(
        event_table["occupancy_well_arm"].to_numpy(),
        track_graph,
    )
    recency, never_visited, reward_count, reward_fraction = _history_by_event(
        event_table["event_start_s"].to_numpy(dtype=float),
        port_experiences,
        track_graph,
    )
    output["distance_to_candidate_arm_cm"] = distances.reshape(-1)
    output["candidate_arm_distance_rank"] = ranks.reshape(-1)
    output["visit_recency_trials"] = recency.reshape(-1)
    output["is_candidate_arm_never_visited"] = never_visited.reshape(-1)
    output["visit_recency_trials_event_leaf_centered"] = (
        center_visit_recency_within_event(recency, never_visited).reshape(-1)
    )
    basis = spline_visit_recency_within_event(
        output["visit_recency_trials"].to_numpy(),
        output["is_candidate_arm_never_visited"].to_numpy(),
        output["is_current_arm"].to_numpy(),
        output["is_prev_arm"].to_numpy(),
        arm,
        output["event_id"].to_numpy(),
    )
    for index, column in enumerate(RECENCY_SPLINE_COLUMNS):
        output[column] = basis[:, index]
    output["reward_count_candidate_arm_last10_known_trials"] = reward_count.reshape(-1)
    output["reward_fraction_candidate_arm_last10_visits"] = reward_fraction.reshape(-1)

    (
        output["task_reward_probability_candidate_arm"],
        output["reward_belief_mean_candidate_arm_preoutcome"],
    ) = attach_contingency_value_block(
        arm,
        contingency_stop_id,
        contingency,
    )
    if include_belief_variance:
        output["reward_belief_variance_candidate_arm_preoutcome"] = (
            attach_arm_uncertainty(
                arm,
                contingency_stop_id,
                contingency,
            )
        )

    rewarded = output["is_rewarded_outcome"].to_numpy(dtype=float)

    # Interaction (product) predictors. Each ``a_x_b`` column is ``a * b`` so a
    # model coefficient on it tests whether the effect of ``a`` differs by ``b``
    # (equivalently, whether ``b``'s effect differs by ``a``). The factors, and
    # the question each family answers, are declared in ``interactions.py``.
    # Columns are read in their own dtype, the event flags as integers, so a
    # product of integer factors stays integer.
    products = [
        "is_current_arm_x_is_last_rewarded_arm",
        "time_since_stop_start_s_x_is_current_arm",
        "time_since_stop_start_s_x_is_prev_arm",
        "time_since_stop_start_s_x_is_next_arm",
        "swr_x_is_current_arm",
        "swr_x_is_prev_arm",
        "swr_x_is_next_arm",
        "swr_x_is_last_rewarded_arm",
        "swr_x_reward_fraction_candidate_arm_last10_visits",
        "swr_x_visit_recency_trials_event_leaf_centered",
        *(f"swr_x_{column}" for column in RECENCY_SPLINE_COLUMNS),
        "swr_x_is_candidate_arm_never_visited",
        "swr_x_is_last_rewarded_noncurrent_arm",
    ]
    if include_belief_variance:
        products.append("swr_x_reward_belief_variance_candidate_arm_preoutcome")
    products += [
        "mua_x_is_current_arm",
        "mua_x_is_prev_arm",
        "mua_x_is_next_arm",
        "mua_x_is_last_rewarded_arm",
        "is_omitted_outcome_x_is_last_rewarded_arm",
    ]
    attach_interactions(
        output,
        products,
        dtype=None,
        factor_values={
            "is_swr": output["is_swr"].astype(int).to_numpy(),
            "is_mua": output["is_mua"].astype(int).to_numpy(),
            "is_omitted_outcome": np.where(
                np.isfinite(rewarded), 1.0 - rewarded, np.nan
            ),
        },
    )
    return output
