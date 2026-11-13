"""Shared parts of the per-grain table builders.

- The prep helpers and ``EVENT_ARM_STANDARDIZED_COLS``, used by more than one
  builder module.
- The per-grain column-order tuples (``_EVENT_TABLE_*`` etc.). Each serves one
  grain, but lives here so prepare_data's empty-session short-circuits and the
  parity/provenance tests can import them without importing every grain module.
- The posterior alignment (``align_posterior_to_dot``), the NaN-strict reducers
  (``safe_mode``, ``first_row_floats``, ``strict_mean``, ``strict_sum``) and the
  response-matrix alignment (``align_count_matrix``,
  ``assert_event_label_alignment``, ``assert_single_session_alignment``) that
  the NonLocal builders here and the ContFrag builder (``contfrag/tables.py``)
  both run; the choice analyses read the strict reducers too.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_definition.immobility import (
    validate_nonlocal_event_probability_contract,
    validate_whole_event_immobility_contract,
)
from continuum_swr_replay.data_loaders.constants import (
    ALL_ARMS,
)
from continuum_swr_replay.decoding.results import nonlocal_probability
from continuum_swr_replay.parameters import DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC
from continuum_swr_replay.tables.columns import require_columns
from continuum_swr_replay.tables.covariate_registry import stamp_covariate_schema

from ..behavior.design_constants import (
    N_REWARD_LAGS,
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_INCOMPLETE,
)
from ..behavior.spatial_mapping import map_estimate_to_arm
from .trajectory import (
    NONLOCAL_ONLY_COLUMNS,
    SHARED_TRAJECTORY_COLUMNS,
)

# Continuous per-arm covariates standardized to pooled-GLOBAL unit SD as ``<col>_z``.
# The build fills NaN placeholders per session; the real global z is computed once in
# ``prepare_data.standardize_event_arm`` after the all-session concat (see tables.standardization).
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
    dot["nonlocal_prob"] = nonlocal_probability(dot).to_numpy()
    return dot


def _in_event_mask(dot: pd.DataFrame, nonlocal_threshold: float) -> pd.Series:
    """Return complete bins of Non-Local labels that satisfy their contract.

    Event labels are created upstream from probability-thresholded intervals
    and whole-event immobility, and the read (``analysed_time.with_swr_source``)
    refuses a frame that breaks either. This guard checks the same contract
    again for a builder called on a frame read another way, then returns
    complete labeled events. It never clips individual bins, because doing so
    can split one event across stops and silently alter its duration.
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

    validate_whole_event_immobility_contract(
        dot,
        "non_local_event",
        speed_threshold=DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    )

    return pd.Series(active, index=dot.index)


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
    # Canonical event timing + speed summaries + SWR classification, all from
    # build_classified_nonlocal_event_table (its event_start/end/duration/
    # midpoint_time renamed to the *_s names; see _merge_classified_event_facts).
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


# ── Posterior alignment ──


def align_posterior_to_dot(
    conditional_posterior,
    dot: pd.DataFrame,
) -> np.ndarray:
    """Return a ``(T_dot, n_position_bins)`` posterior array aligned to ``dot.index``."""
    post_vals = np.asarray(conditional_posterior.values)
    if post_vals.shape[0] != len(dot):
        raise ValueError(
            f"conditional_posterior time axis has {post_vals.shape[0]} "
            f"entries but dot has {len(dot)}; the two must be aligned "
            "1-to-1 (both come from the same decoder run)."
        )
    if "time" in conditional_posterior.coords:
        post_times = np.asarray(conditional_posterior["time"].values, dtype=float)
        drift = float(np.abs(post_times - dot.index.to_numpy(dtype=float)).max())
        if drift > 1e-6:
            raise ValueError(
                f"conditional_posterior time axis drifts from dot.index "
                f"by up to {drift:.3g}s — the two must share the same "
                "time origin, not just the same length."
            )
    return post_vals


# ── NaN-strict reducers ──


def safe_mode(series: pd.Series) -> float:
    """Mode of a series (first value on tie), ``NaN`` if empty or all NaN."""
    valid = series.dropna()
    if len(valid) == 0:
        return np.nan
    return float(valid.mode().iloc[0])


def first_row_floats(
    groups: pd.api.typing.DataFrameGroupBy, columns: Sequence[str]
) -> dict[str, np.ndarray]:
    """Each group's first-row value of each column, as a float, NaN kept.

    Trial-derived context columns are NaN at every timebin before the first
    trial start (see ``add_trial_context``). A NaN-skipping first would skip
    those leading NaNs and return the next trial's context, leaking future
    state into a group whose first row had no trial active, so the first row
    is read as-is (``first(skipna=False)``). The later rows are not checked
    against it.

    Parameters
    ----------
    groups : DataFrameGroupBy
        The frame grouped by event or bout.
    columns : Sequence[str]
        Numeric, boolean or nullable columns to read.

    Returns
    -------
    dict[str, np.ndarray]
        Column -> array of shape ``(n_groups,)``, in the groups' order.
    """
    first = groups[list(columns)].first(skipna=False)
    return {
        column: first[column].to_numpy(dtype=float, na_value=np.nan)
        for column in columns
    }


def strict_mean(series: pd.Series) -> float:
    """``Series.mean`` with ``skipna=False``.

    pandas' default ``"mean"`` aggregator silently drops NaN, so a
    groupby reduction over a column that contains a mix of finite and
    NaN values emits the mean of the surviving rows. For Stage-5/6
    stop-level / arm-level aggregations that's a per-event survivorship
    bias: a stop with one finite event and one NaN event reports the
    finite event's value while ``n_events`` still counts both. Use
    this aggregator instead so any NaN propagates to the output.
    """
    return float(series.mean(skipna=False))


def strict_sum(series: pd.Series) -> float:
    """``Series.sum`` with ``skipna=False``."""
    return float(series.sum(skipna=False))


# ── Response-matrix alignment ──


def align_count_matrix(
    counts: np.ndarray,
    ids_sorted: np.ndarray,
    target_ids: np.ndarray,
    n_arms: int,
) -> np.ndarray:
    """Reindex a ``(n_events_src, n_arms)`` count matrix to ``target_ids``.

    An id in ``target_ids`` with no row in ``ids_sorted`` is an event that
    contributed no counted rows -- every one of its in-event timebins failed to
    attribute to a real arm. Its output row is filled, and **the fill value
    depends on the input dtype**: integer counts fill ``0`` (a real count of
    zero) and float counts fill ``NaN`` (unknown). That is deliberate --
    ``arm_count_hard`` genuinely observed zero real-arm bins, while
    ``arm_fraction``'s 0/0 must stay undefined rather than become 0 or +inf --
    but it is subtle enough to be worth stating, since the two paths differ
    only by ``counts.dtype``.

    On the shipped corpus no event triggers the fill (``n_junction_bins`` is 0
    for all 65,913), so this path is exercised only by tests.
    """
    mapping = {int(i): k for k, i in enumerate(ids_sorted)}
    fill = np.nan if np.issubdtype(counts.dtype, np.floating) else 0
    out = np.full((target_ids.size, n_arms), fill, dtype=counts.dtype)
    for k, tid in enumerate(target_ids):
        src_idx = mapping.get(int(tid))
        if src_idx is not None:
            out[k] = counts[src_idx]
    return out


def assert_event_label_alignment(
    source_ids: np.ndarray,
    target_ids: np.ndarray,
    *,
    source_name: str,
) -> None:
    """Require one exact event-label set before response matrices are aligned.

    ``align_count_matrix`` may legitimately fill a hard-count row when every bin
    of a known event maps to a junction. It must not, however, conceal an event
    missing from the source frame or an extra event from another/stale build.
    Callers therefore pass the labels from *all* qualified source bins here,
    before selecting only real-arm bins for counting.
    """
    source = np.asarray(source_ids, dtype=float)
    target = np.asarray(target_ids, dtype=float)
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError(
            f"event-label alignment failed for {source_name}: labels must be finite"
        )
    if (
        not np.equal(source, np.floor(source)).all()
        or not np.equal(target, np.floor(target)).all()
    ):
        raise ValueError(
            f"event-label alignment failed for {source_name}: labels must be integral"
        )
    if len(np.unique(target)) != len(target):
        raise ValueError(
            "event-label alignment failed: target event table contains duplicate labels"
        )
    source_set = set(source.astype(np.int64))
    target_set = set(target.astype(np.int64))
    if source_set != target_set:
        missing = sorted(target_set - source_set)[:10]
        extra = sorted(source_set - target_set)[:10]
        raise ValueError(
            f"event-label alignment failed for {source_name}: missing target labels "
            f"{missing}, extra source labels {extra}. The source frame and event "
            "table must come from the same session and build."
        )


def assert_single_session_alignment(
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    source_name: str,
    session_cols: tuple[str, ...] = ("animal", "day", "epoch"),
    target_session_cols: tuple[str, ...] | None = None,
) -> None:
    """Require source and target frames to name the same single session.

    Event labels are session-local integers. Exact label-set equality alone
    cannot detect a stale source frame from another session whose labels happen
    to overlap, so builders validate the natural session key before aligning
    response matrices.
    """
    target_cols = target_session_cols or session_cols
    if len(target_cols) != len(session_cols):
        raise ValueError("source and target session keys must have equal arity")
    require_columns(source, session_cols, f"session alignment source {source_name}")
    require_columns(target, target_cols, f"session alignment target of {source_name}")
    if (
        source.loc[:, list(session_cols)].isna().any().any()
        or target.loc[:, list(target_cols)].isna().any().any()
    ):
        raise ValueError(
            f"session alignment for {source_name} requires nonmissing natural keys"
        )

    source_keys = source.loc[:, list(session_cols)].drop_duplicates()
    target_keys = target.loc[:, list(target_cols)].drop_duplicates()
    if len(source_keys) != 1 or len(target_keys) != 1:
        raise ValueError(
            f"session alignment for {source_name} requires exactly one session "
            f"per frame; found source={len(source_keys)}, target={len(target_keys)}"
        )
    source_key = tuple(source_keys.iloc[0].tolist())
    target_key = tuple(target_keys.iloc[0].tolist())
    if source_key != target_key:
        raise ValueError(
            f"session alignment failed for {source_name}: source={source_key}, "
            f"target={target_key}"
        )
