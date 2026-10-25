"""The interaction (product) design columns, each declared once with its factors.

:data:`INTERACTION_FACTORS` maps every product column a model ladder fits to the
design columns it multiplies, in the order its name spells them. It is the one
answer to "what is this product made of?":

- ``contrast_aggregation`` builds the content contrast ladder's fit-time
  products from it (:func:`attach_interactions`);
- ``covariate_timing`` resolves a product to the timing atoms of its factors;
- ``cohort_audit`` decides from its factors whether a product is NaN on the
  qvals gap.

The ``swr_x_*`` products of the canonical event-arm and event-patch tables are
declared here too, so their timing and qvals status come from the same place,
but those tables still build them with their own code, in ``candidate_arm.py``
and ``tables_arm_patch.py``. The tables' other products (``mua_x_*``,
``time_since_stop_start_s_x_*``, ``is_current_arm_x_is_last_rewarded_arm``,
``is_omitted_outcome_x_is_last_rewarded_arm``) are fitted by no ladder and are
not declared, so the timing check refuses them until they are.
A product name spells its factors joined by ``_x_``, with ``swr`` standing for
``is_swr``.

A factor is a design column. Its source covariate (:func:`factor_source`) is
the column with the fit-time ``_z`` standardization removed, then mapped through
:data:`TRANSFORM_SOURCES` when it is a fit-time transform of a catalogued
covariate.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

#: Event length on a log scale, z-scored at fit time as the event-length control.
LOG_EVENT_LENGTH = "log_n_nonlocal_bins"
#: Within-stop time on a log1p scale, z-scored at fit time as the duration control.
LOG1P_STOP_TIME = "log1p_time_since_stop_start_s"

#: Fit-time transform column -> the catalogued covariate it is computed from.
TRANSFORM_SOURCES: dict[str, str] = {
    LOG_EVENT_LENGTH: "n_nonlocal_bins",
    LOG1P_STOP_TIME: "time_since_stop_start_s",
}

#: Product column -> the design columns it multiplies, in name order.
INTERACTION_FACTORS: dict[str, tuple[str, ...]] = {
    # --- Built at fit time on the content contrast arm cohort. ---
    # State modulation of the current-arm effect.
    "rpe_postoutcome_z_x_is_current_arm": ("rpe_postoutcome_z", "is_current_arm"),
    "relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm": (
        "relative_value_switch_minus_stay_preoutcome_z",
        "is_current_arm",
    ),
    "reward_belief_differential_entropy_leaf_mean_preoutcome_z_x_is_current_arm": (
        "reward_belief_differential_entropy_leaf_mean_preoutcome_z",
        "is_current_arm",
    ),
    "reward_belief_mean_update_l1_all_ports_postoutcome_z_x_is_current_arm": (
        "reward_belief_mean_update_l1_all_ports_postoutcome_z",
        "is_current_arm",
    ),
    # Task context.
    "periswitch_index_trials_z_x_is_current_arm": (
        "periswitch_index_trials_z",
        "is_current_arm",
    ),
    "is_periswitch_post_x_is_current_arm": ("is_periswitch_post", "is_current_arm"),
    "periswitch_index_trials_z_x_is_periswitch_post_x_is_current_arm": (
        "periswitch_index_trials_z",
        "is_periswitch_post",
        "is_current_arm",
    ),
    "trials_since_block_change_z_x_is_current_arm": (
        "trials_since_block_change_z",
        "is_current_arm",
    ),
    "patch_switch_fraction_last5_completed_trips_z_x_is_current_arm": (
        "patch_switch_fraction_last5_completed_trips_z",
        "is_current_arm",
    ),
    # Within-stop-time duration control: z(log1p t) : is_current.
    "log1p_time_since_stop_start_s_z_x_is_current_arm": (
        f"{LOG1P_STOP_TIME}_z",
        "is_current_arm",
    ),
    # Event-length control: z(log n bins) : is_current / : is_prev.
    "log_n_nonlocal_bins_z_x_is_current_arm": (
        f"{LOG_EVENT_LENGTH}_z",
        "is_current_arm",
    ),
    "log_n_nonlocal_bins_z_x_is_prev_arm": (f"{LOG_EVENT_LENGTH}_z", "is_prev_arm"),
    # SWR products of a base covariate, in its canonical spelling.
    "swr_x_reward_belief_mean_candidate_arm_preoutcome": (
        "is_swr",
        "reward_belief_mean_candidate_arm_preoutcome",
    ),
    "swr_x_reward_belief_variance_candidate_arm_preoutcome_z": (
        "is_swr",
        "reward_belief_variance_candidate_arm_preoutcome_z",
    ),
    # --- Built at fit time on the content contrast patch cohort. ---
    "periswitch_index_trials_z_x_is_current_patch": (
        "periswitch_index_trials_z",
        "is_current_patch",
    ),
    "log1p_time_since_stop_start_s_z_x_is_current_patch": (
        f"{LOG1P_STOP_TIME}_z",
        "is_current_patch",
    ),
    "log_n_nonlocal_bins_z_x_is_current_patch": (
        f"{LOG_EVENT_LENGTH}_z",
        "is_current_patch",
    ),
    "log_n_nonlocal_bins_z_x_is_prev_patch": (
        f"{LOG_EVENT_LENGTH}_z",
        "is_prev_patch",
    ),
    "patch_switch_fraction_last5_completed_trips_z_x_is_current_patch": (
        "patch_switch_fraction_last5_completed_trips_z",
        "is_current_patch",
    ),
    # Also a canonical event-patch column; the fit-time build overwrites it.
    "swr_x_is_current_patch": ("is_swr", "is_current_patch"),
    # --- Canonical event-patch table products (tables_arm_patch.py). ---
    "swr_x_is_last_rewarded_noncurrent_patch": (
        "is_swr",
        "is_last_rewarded_noncurrent_patch",
    ),
    "swr_x_is_prev_patch": ("is_swr", "is_prev_patch"),
    # --- Canonical event-arm table products (candidate_arm.py). ---
    "swr_x_is_current_arm": ("is_swr", "is_current_arm"),
    "swr_x_is_prev_arm": ("is_swr", "is_prev_arm"),
    "swr_x_is_next_arm": ("is_swr", "is_next_arm"),
    "swr_x_is_last_rewarded_arm": ("is_swr", "is_last_rewarded_arm"),
    "swr_x_is_last_rewarded_noncurrent_arm": (
        "is_swr",
        "is_last_rewarded_noncurrent_arm",
    ),
    "swr_x_reward_fraction_candidate_arm_last10_visits": (
        "is_swr",
        "reward_fraction_candidate_arm_last10_visits",
    ),
    "swr_x_visit_recency_trials_event_leaf_centered": (
        "is_swr",
        "visit_recency_trials_event_leaf_centered",
    ),
    "swr_x_visit_recency_trials_event_leaf_centered_spline_1": (
        "is_swr",
        "visit_recency_trials_event_leaf_centered_spline_1",
    ),
    "swr_x_visit_recency_trials_event_leaf_centered_spline_2": (
        "is_swr",
        "visit_recency_trials_event_leaf_centered_spline_2",
    ),
    "swr_x_visit_recency_trials_event_leaf_centered_spline_3": (
        "is_swr",
        "visit_recency_trials_event_leaf_centered_spline_3",
    ),
    "swr_x_is_candidate_arm_never_visited": (
        "is_swr",
        "is_candidate_arm_never_visited",
    ),
    "swr_x_reward_belief_variance_candidate_arm_preoutcome": (
        "is_swr",
        "reward_belief_variance_candidate_arm_preoutcome",
    ),
}


def factor_source(factor: str) -> str:
    """The catalogued covariate a product factor is measured from.

    Parameters
    ----------
    factor : str
        A factor column of :data:`INTERACTION_FACTORS`.

    Returns
    -------
    str
        ``factor`` without its ``_z`` suffix, mapped through
        :data:`TRANSFORM_SOURCES` when it is a fit-time transform.
    """
    base = factor.removesuffix("_z")
    return TRANSFORM_SOURCES.get(base, base)


def attach_interactions(frame: pd.DataFrame, names: Iterable[str]) -> pd.DataFrame:
    """Write each product column in ``names`` onto ``frame``, in order.

    Each product is the float64 elementwise product of its declared factors,
    taken left to right; a NaN in any factor makes it NaN.

    Parameters
    ----------
    frame : pandas.DataFrame
        Design frame carrying every factor of every product in ``names``.
        Modified in place.
    names : iterable of str
        Keys of :data:`INTERACTION_FACTORS`.

    Returns
    -------
    pandas.DataFrame
        ``frame``, with the products appended (or overwritten where present).

    Raises
    ------
    KeyError
        If a name is not declared, or one of its factors is absent.
    """
    for name in names:
        first, *rest = INTERACTION_FACTORS[name]
        product = frame[first].to_numpy(dtype=float)
        for factor in rest:
            product = product * frame[factor].to_numpy(dtype=float)
        frame[name] = product
    return frame
