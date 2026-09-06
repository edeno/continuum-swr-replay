"""Shared prep helpers, the arm standardization column list, and canonical
column-order constants for the per-grain table builders.

The three prep helpers and ``EVENT_ARM_STANDARDIZED_COLS`` are genuinely
cross-grain (used by more than one builder module). The per-grain column-order
tuples (``_EVENT_TABLE_*`` etc.) each serve one grain, but live here so the
``tables.py`` shim can re-export them for prepare_data's empty-session
short-circuit and the parity/provenance tests without importing every grain
module."""

from __future__ import annotations

import pandas as pd

from continuum_swr_replay.analysis.hippocampus.event_immobility import (
    EventDefinitionError,
    filter_event_labels_by_speed,
    validate_nonlocal_event_probability_contract,
    whole_event_immobility_mask,
)
from continuum_swr_replay.data_loaders.constants import (
    ALL_ARMS,
)

from .covariate_registry import stamp_covariate_schema
from .model_specs import (
    N_REWARD_LAGS,
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_INCOMPLETE,
)
from .spatial_mapping import map_estimate_to_arm
from .trajectory import (
    NONLOCAL_ONLY_COLUMNS,
    SHARED_TRAJECTORY_COLUMNS,
)

# Continuous per-arm covariates standardized to pooled-GLOBAL unit SD as ``<col>_z``.
# The build fills NaN placeholders per session; the real global z is computed once in
# ``prepare_data.main`` after the all-session concat (see analysis.replay.standardization).
EVENT_ARM_STANDARDIZED_COLS = (
    "reward_belief_mean_candidate_arm_preoutcome",
    "reward_fraction_candidate_arm_last10_visits",
    "task_reward_probability_candidate_arm",
    "reward_belief_variance_candidate_arm_preoutcome",
)


def _prepare_dot(dot: pd.DataFrame, env) -> pd.DataFrame:
    """Return a copy of ``dot`` annotated with ``decoded_arm`` and ``nonlocal_prob``.

    ``decoded_arm`` comes from :func:`map_estimate_to_arm` (junction bins
    excluded). ``nonlocal_prob`` is the sum of the two non-local state
    posteriors — this is the ``w_t`` that drives event filtering and
    per-event weight sums.
    """
    dot = dot.copy()
    dot["decoded_arm"] = map_estimate_to_arm(dot["map_estimate"].to_numpy(), env)
    dot["nonlocal_prob"] = (
        dot["Non-Local_Continuous_prob"].to_numpy()
        + dot["Non-Local_Fragmented_prob"].to_numpy()
    )
    return dot


def _in_event_mask(
    dot: pd.DataFrame,
    speed_threshold: float,
    nonlocal_threshold: float,
) -> pd.Series:
    """Return complete bins of Non-Local labels that satisfy their contract.

    Event labels are created upstream from probability-thresholded intervals
    and whole-event immobility. This function validates that contract and then
    returns complete labeled events. It never clips individual bins, because
    doing so can split one event across stops and silently alter its duration.
    """
    active = dot["non_local_event"].to_numpy() > 0
    if not active.any():
        return pd.Series(False, index=dot.index)

    # Validate the detector-definition contract first. A stale label can
    # violate both probability and speed; reporting the upstream probability
    # violation consistently across replay and time-bin builders makes the
    # regeneration cause unambiguous.
    validate_nonlocal_event_probability_contract(
        dot,
        probability_cols=("nonlocal_prob",),
        probability_threshold=nonlocal_threshold,
    )

    immobile = whole_event_immobility_mask(
        dot,
        "non_local_event",
        speed_threshold=speed_threshold,
    ).to_numpy()
    invalid_speed = active & ~immobile
    if invalid_speed.any():
        invalid_ids = sorted(
            pd.unique(dot.loc[invalid_speed, "non_local_event"]).tolist()
        )
        raise EventDefinitionError(
            "non_local_event labels violate the whole-event immobility "
            f"contract (all speeds finite and < {speed_threshold:g} cm/s); "
            f"invalid event labels: {invalid_ids[:10]}. Regenerate the "
            "data-over-time file with the unified event definition."
        )

    return pd.Series(active, index=dot.index)


def _filter_external_event_labels(
    dot: pd.DataFrame,
    speed_threshold: float,
) -> pd.DataFrame:
    """Apply the shared atomic speed rule to SWR/MUA labels when present."""
    event_cols = [column for column in ("SWR_event", "MUA_event") if column in dot]
    if not event_cols:
        return dot
    return filter_event_labels_by_speed(
        dot,
        event_cols,
        speed_threshold=speed_threshold,
    )


# ── Public API ──


#: Canonical column order for ``event_table``.
#:
#: Kept central so the empty-session short-circuit returns the right shape and
#: so downstream consumers can rely on column order.
_EVENT_TABLE_BUILD_COLUMNS: tuple[str, ...] = (
    "animal",
    "session_date",
    "epoch",
    "non_local_event",
    "event_id",
    "time_since_stop_start_s",
    "time_since_last_well_arrival_s",
    "time_since_last_reward_s",
    "occupancy_state",
    "at_well",
    "occupancy_well_id",
    "modal_observed_track_segment_id",
    "nearest_well_distance_cm_min",
    "nearest_well_distance_cm_mean",
    "nearest_well_distance_cm_max",
    "is_swr",
    "is_mua",
    "did_last_completed_trip_switch_patch",
    "is_rewarded_outcome",
    "is_stay",
    "occupancy_well_arm",
    "completed_trial_id",
    "prev_arm",
    "next_arm",
    "last_rewarded_arm",
    "current_patch",
    "prev_patch",
    "next_patch",
    "n_nonlocal_bins",
    "nonlocal_probability_sum",
    "posterior_peak_mean",
    "posterior_entropy_mean",
    "posterior_hpd_arm_count_mean_coverage_50pct",
    "posterior_hpd_arm_count_mean_coverage_80pct",
    "posterior_hpd_arm_count_mean_coverage_95pct",
    "n_undecoded_bins",
    "n_unique_arms",
    "stop_idx",
    "stop_id",
    # Compatibility timing aliases + SWR classification from
    # build_classified_nonlocal_event_table. Atomic label validation makes
    # these timings identical to event_start/end/duration/midpoint above.
    "event_start_s",
    "event_end_s",
    "event_duration_s",
    "event_midpoint_s",
    "animal_max_speed_cm_s",
    "animal_mean_speed_cm_s",
    "animal_start_speed_cm_s",
    "animal_end_speed_cm_s",
    "whole_event_speed_eligible",
    "endpoints_speed_below_thresh",
    "peak_nonlocal_prob",
    "overlap_class",
    "overlap_subclass",
    "n_swrs_overlapping",
    "matched_swr_event",
    "matched_swr_start",
    "matched_swr_end",
    # Trajectory rollups (shared with ContFrag) + Non-Local-only state means.
    *SHARED_TRAJECTORY_COLUMNS,
    *NONLOCAL_ONLY_COLUMNS,
)

_EVENT_TABLE_COLUMNS: tuple[str, ...] = tuple(
    stamp_covariate_schema(
        pd.DataFrame(columns=_EVENT_TABLE_BUILD_COLUMNS), table="event"
    ).columns
)


#: Canonical column order for ``stop_table``.
_STOP_TABLE_BUILD_COLUMNS: tuple[str, ...] = (
    "animal",
    "session_date",
    "epoch",
    "stop_idx",
    "stop_id",
    "stop_start",
    "stop_end",
    "immobility_duration_s",
    "reward_fraction_last10_known_trials",
    "is_swr_stop",
    "n_swr_events",
    "is_mua_stop",
    "n_mua_events",
    "has_replay",
    "n_selected_events",
    "did_last_completed_trip_switch_patch",
    "is_rewarded_outcome",
    "completed_trial_reward",
    "completed_trial_id",
    "completed_trial_arm",
    "patch_switch_fraction_last5_completed_trips",
    "leaky_q_value_diff",
    "is_stay",
    "occupancy_state",
    "at_well",
    "occupancy_well_id",
    "modal_observed_track_segment_id",
    "nearest_well_distance_cm_min",
    "nearest_well_distance_cm_mean",
    "nearest_well_distance_cm_max",
    "occupancy_well_arm",
    "prev_arm",
    "next_arm",
    "last_rewarded_arm",
    "current_patch",
    "prev_patch",
    "next_patch",
    *[f"known_reward_outcome_lag_{j}" for j in range(1, N_REWARD_LAGS + 1)],
    *REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_INCOMPLETE,
    "last_well_arrival_time_s",
    "outcome_known_time_s",
)

_STOP_TABLE_COLUMNS: tuple[str, ...] = tuple(
    stamp_covariate_schema(
        pd.DataFrame(columns=_STOP_TABLE_BUILD_COLUMNS), table="stop"
    ).columns
)


#: Canonical column order for ``event_arm_table``.
_EVENT_ARM_TABLE_BUILD_COLUMNS: tuple[str, ...] = (
    # Canonical persisted column order (broadcast event-level fields,
    # per-arm responses/predictors, interaction products, then the
    # standardized ``_z`` placeholders filled globally in prepare_data).
    "animal",
    "session_date",
    "epoch",
    "non_local_event",
    "event_id",
    "stop_idx",
    "stop_id",
    "event_start_s",
    "event_end_s",
    "event_duration_s",
    "time_since_stop_start_s",
    "time_since_last_well_arrival_s",
    "time_since_last_reward_s",
    "occupancy_state",
    "at_well",
    "occupancy_well_id",
    "modal_observed_track_segment_id",
    "nearest_well_distance_cm_min",
    "nearest_well_distance_cm_mean",
    "nearest_well_distance_cm_max",
    "is_swr",
    "is_mua",
    "did_last_completed_trip_switch_patch",
    "is_rewarded_outcome",
    "is_stay",
    "occupancy_well_arm",
    "prev_arm",
    "next_arm",
    "last_rewarded_arm",
    "current_patch",
    "prev_patch",
    "next_patch",
    "n_nonlocal_bins",
    "candidate_arm",
    "arm_patch_id",
    "arm_count_hard",
    "arm_count_soft",
    "arm_posterior_mass",
    "arm_fraction",
    "arm_fraction_posterior",
    "is_current_arm",
    "is_prev_arm",
    "is_next_arm",
    "is_last_rewarded_arm",
    "is_last_rewarded_noncurrent_arm",
    "is_same_patch_as_current",
    "distance_to_candidate_arm_cm",
    "candidate_arm_distance_rank",
    "visit_recency_trials",
    "visit_recency_trials_event_leaf_centered",
    "visit_recency_trials_event_leaf_centered_spline_1",
    "visit_recency_trials_event_leaf_centered_spline_2",
    "visit_recency_trials_event_leaf_centered_spline_3",
    "is_candidate_arm_never_visited",
    "reward_count_candidate_arm_last10_known_trials",
    "reward_fraction_candidate_arm_last10_visits",
    "task_reward_probability_candidate_arm",
    "reward_belief_mean_candidate_arm_preoutcome",
    "reward_belief_variance_candidate_arm_preoutcome",
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
    "swr_x_visit_recency_trials_event_leaf_centered_spline_1",
    "swr_x_visit_recency_trials_event_leaf_centered_spline_2",
    "swr_x_visit_recency_trials_event_leaf_centered_spline_3",
    "swr_x_is_candidate_arm_never_visited",
    "swr_x_is_last_rewarded_noncurrent_arm",
    "swr_x_reward_belief_variance_candidate_arm_preoutcome",
    "mua_x_is_current_arm",
    "mua_x_is_prev_arm",
    "mua_x_is_next_arm",
    "mua_x_is_last_rewarded_arm",
    "is_omitted_outcome_x_is_last_rewarded_arm",
    *[f"{c}_z" for c in EVENT_ARM_STANDARDIZED_COLS],
)

_EVENT_ARM_TABLE_COLUMNS: tuple[str, ...] = tuple(
    stamp_covariate_schema(
        pd.DataFrame(columns=_EVENT_ARM_TABLE_BUILD_COLUMNS), table="event_arm"
    ).columns
)


#: Canonical column order for ``event_sequence_table``.
_EVENT_SEQUENCE_TABLE_BUILD_COLUMNS: tuple[str, ...] = (
    "animal",
    "session_date",
    "epoch",
    "non_local_event",
    "event_id",
    "t_in_event",
    "time",
    "nonlocal_prob",
    "decoded_arm_map",
    *(f"arm_posterior_{a}" for a in ALL_ARMS),
)

_EVENT_SEQUENCE_TABLE_COLUMNS: tuple[str, ...] = tuple(
    stamp_covariate_schema(
        pd.DataFrame(columns=_EVENT_SEQUENCE_TABLE_BUILD_COLUMNS),
        table="event_sequence",
    ).columns
)
