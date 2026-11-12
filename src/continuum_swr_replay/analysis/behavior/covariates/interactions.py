"""The interaction (product) design columns, each declared once with its factors.

:data:`INTERACTION_FACTORS` maps every product column the package builds to the
design columns it multiplies, in the order its name spells them. It is the one
answer to "what is this product made of?":

- the builders write every product from it (:func:`attach_interactions`):
  ``contrast_aggregation`` its fit-time products, ``candidate_arm.py`` the
  event-arm table's (Replay and ContFrag), ``tables_arm_patch.py`` the
  event-patch table's, ``contfrag/tables.py`` the ContFrag-only ones, and
  ``simulation.py`` the simulator's;
- ``covariate_timing`` resolves a product to the timing atoms of its factors;
- ``cohort_audit`` decides from its factors whether a product is NaN on the
  qvals gap.

A product name spells its factors joined by ``_x_``, with ``swr`` standing for
``is_swr`` and ``mua`` for ``is_mua``; the two ContFrag continuous-score
products keep their older names (:data:`LEGACY_PRODUCT_NAMES`).

Products no ladder fits are in :data:`UNFITTED_PRODUCTS`. ``covariate_timing``
does not resolve them, so the timing check refuses them as unclassified until
one is reviewed and taken out of that set.

A factor is a design column. Its source covariate (:func:`factor_source`) is
the column with the fit-time ``_z`` standardization removed, then mapped through
:data:`TRANSFORM_SOURCES` when it is a fit-time transform of a catalogued
covariate.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import numpy.typing as npt
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
    # Also an event-patch table column; the fit-time build overwrites it.
    "swr_x_is_current_patch": ("is_swr", "is_current_patch"),
    # --- Event-patch table (tables_arm_patch.py). ---
    "swr_x_is_last_rewarded_noncurrent_patch": (
        "is_swr",
        "is_last_rewarded_noncurrent_patch",
    ),
    "swr_x_is_prev_patch": ("is_swr", "is_prev_patch"),
    # --- Event-arm table (candidate_arm.py), Replay and ContFrag. ---
    # Does the current arm being also the last-rewarded arm change its
    # representation beyond the two main effects?
    "is_current_arm_x_is_last_rewarded_arm": (
        "is_current_arm",
        "is_last_rewarded_arm",
    ),
    # Does representation of the current / previous / next arm change over
    # the stop?
    "time_since_stop_start_s_x_is_current_arm": (
        "time_since_stop_start_s",
        "is_current_arm",
    ),
    "time_since_stop_start_s_x_is_prev_arm": (
        "time_since_stop_start_s",
        "is_prev_arm",
    ),
    "time_since_stop_start_s_x_is_next_arm": (
        "time_since_stop_start_s",
        "is_next_arm",
    ),
    # Does SWR replay weight each arm differently from non-SWR non-local
    # activity?
    "swr_x_is_current_arm": ("is_swr", "is_current_arm"),
    "swr_x_is_prev_arm": ("is_swr", "is_prev_arm"),
    "swr_x_is_next_arm": ("is_swr", "is_next_arm"),
    "swr_x_is_last_rewarded_arm": ("is_swr", "is_last_rewarded_arm"),
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
    # SWR-specific arm baselines of the pooled contrast rungs (fit-time only):
    # without them the arms share one baseline across SWR and non-SWR events,
    # and SWR's shift of content onto the trunks loads on the swr_x terms.
    **{
        f"swr_x_is_candidate_arm_{k}": ("is_swr", f"is_candidate_arm_{k}")
        for k in range(8)
    },
    "swr_x_is_last_rewarded_noncurrent_arm": (
        "is_swr",
        "is_last_rewarded_noncurrent_arm",
    ),
    "swr_x_reward_belief_variance_candidate_arm_preoutcome": (
        "is_swr",
        "reward_belief_variance_candidate_arm_preoutcome",
    ),
    # The MUA mirror: a control cohort against the SWR terms.
    "mua_x_is_current_arm": ("is_mua", "is_current_arm"),
    "mua_x_is_prev_arm": ("is_mua", "is_prev_arm"),
    "mua_x_is_next_arm": ("is_mua", "is_next_arm"),
    "mua_x_is_last_rewarded_arm": ("is_mua", "is_last_rewarded_arm"),
    # Does an omitted outcome change representation of the last-rewarded arm?
    "is_omitted_outcome_x_is_last_rewarded_arm": (
        "is_omitted_outcome",
        "is_last_rewarded_arm",
    ),
    # --- ContFrag event-arm table only (contfrag/tables.py). ---
    "continuous_replay_score_mean_x_is_current_arm": (
        "continuous_prob_mean",
        "is_current_arm",
    ),
    "continuous_replay_score_mean_x_is_prev_arm": (
        "continuous_prob_mean",
        "is_prev_arm",
    ),
    "continuous_replay_score_mean_x_is_next_arm": (
        "continuous_prob_mean",
        "is_next_arm",
    ),
    "continuous_x_last_rewarded": ("continuous_prob_mean", "is_last_rewarded_arm"),
}

#: Products whose name does not spell its factors: ContFrag's continuous-score
#: products, named before the convention (``continuous_replay_score_mean`` and
#: ``continuous`` stand for the ``continuous_prob_mean`` column).
LEGACY_PRODUCT_NAMES = frozenset(
    {
        "continuous_replay_score_mean_x_is_current_arm",
        "continuous_replay_score_mean_x_is_prev_arm",
        "continuous_replay_score_mean_x_is_next_arm",
        "continuous_x_last_rewarded",
    }
)

#: Products no ladder fits. Their timing has not been reviewed, so
#: ``covariate_timing`` leaves them unresolved and the timing check refuses
#: them; fitting one starts by taking it out of this set.
UNFITTED_PRODUCTS = frozenset(
    {
        "is_current_arm_x_is_last_rewarded_arm",
        "time_since_stop_start_s_x_is_current_arm",
        "time_since_stop_start_s_x_is_prev_arm",
        "time_since_stop_start_s_x_is_next_arm",
        "mua_x_is_current_arm",
        "mua_x_is_prev_arm",
        "mua_x_is_next_arm",
        "mua_x_is_last_rewarded_arm",
        "is_omitted_outcome_x_is_last_rewarded_arm",
        *LEGACY_PRODUCT_NAMES,
    }
)


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


def interaction_product(
    name: str, factor_values: Mapping[str, npt.ArrayLike]
) -> np.ndarray:
    """The product ``name``: its declared factors multiplied left to right.

    Parameters
    ----------
    name : str
        A key of :data:`INTERACTION_FACTORS`.
    factor_values : Mapping[str, array_like]
        Factor name -> values, all of one shape.

    Returns
    -------
    numpy.ndarray
        The elementwise product, in the dtype numpy promotes the factors to; a
        NaN in any factor makes it NaN.
    """
    first, *rest = INTERACTION_FACTORS[name]
    product = np.asarray(factor_values[first])
    for factor in rest:
        product = product * np.asarray(factor_values[factor])
    return product


class _FactorColumns(Mapping[str, np.ndarray]):
    """A frame's columns as arrays, with some factors supplied by the caller."""

    def __init__(self, frame, dtype, supplied):
        self._frame, self._dtype, self._supplied = frame, dtype, supplied

    def __getitem__(self, factor: str) -> np.ndarray:
        if factor in self._supplied:
            return np.asarray(self._supplied[factor])
        return self._frame[factor].to_numpy(dtype=self._dtype)

    def __iter__(self):
        return iter(self._frame.columns)

    def __len__(self) -> int:
        return len(self._frame.columns)


def attach_interactions(
    frame: pd.DataFrame,
    names: Iterable[str],
    *,
    dtype: npt.DTypeLike | None = float,
    factor_values: Mapping[str, npt.ArrayLike] | None = None,
) -> pd.DataFrame:
    """Write each product column in ``names`` onto ``frame``, in order.

    Parameters
    ----------
    frame : pandas.DataFrame
        Design frame carrying every factor of every product in ``names`` that
        ``factor_values`` does not supply. Modified in place.
    names : iterable of str
        Keys of :data:`INTERACTION_FACTORS`.
    dtype : dtype or None, optional
        What each factor column is read as. ``float`` (the default) gives a
        float64 product; ``None`` reads every column in its own dtype, so a
        product of integer factors stays integer.
    factor_values : Mapping[str, array_like], optional
        Factor name -> values used in place of the frame's column (or for a
        factor the frame does not carry), as given; ``dtype`` does not apply.

    Returns
    -------
    pandas.DataFrame
        ``frame``, with the products appended (or overwritten where present).

    Raises
    ------
    KeyError
        If a name is not declared, or one of its factors is absent.
    """
    columns = _FactorColumns(frame, dtype, factor_values or {})
    for name in names:
        frame[name] = interaction_product(name, columns)
    return frame
