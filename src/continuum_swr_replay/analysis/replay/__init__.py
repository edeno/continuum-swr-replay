"""Replay content analysis package.

Houses the conditional Poisson fitter, model hierarchy, spatial mapping,
simulation utilities, and model comparison/checking helpers used across
the replay content analysis pipeline.
"""

from ._helpers import (
    POPULATIONS,
    add_arm_fixed_effects,
    add_trial_context,
    aggregate_to_timebins,
    attach_is_swr_to_onsets,
    behavioral_location_at_times,
    nearest_well_distance_cm,
    raised_cosine_history_basis,
    single_arm_from_well,
    summarize_behavioral_location,
    trial_context_at_times,
    trial_cutoff_times,
)
from .choice_prediction import (
    build_choice_dataset,
    build_residualized_choice_dataset,
    fit_choice_model,
    predict_arm_fractions,
)
from .cohort_selection import freeze_common_cohort
from .covariates import UNCERTAINTY_COVARIATES, add_uncertainty_covariates
from .fitting import (
    fit_conditional_logit,
    fit_conditional_poisson,
    fit_null_model_conditional,
)
from .learning import (
    StopSummaryConfig,
    fit_is_stay_logistic_per_session,
    fit_is_stay_pooled_gee,
    stop_replay_summary,
)
from .model_checking import (
    compute_pearson_residuals,
    cross_validate_conditional_poisson,
)
from .model_specs import (
    ARM_FE_PREDICTORS,
    BELIEF_DEPENDENT_CONTENT_MODELS,
    BIN_S,
    MODEL_CHOICE,
    MODEL_HIERARCHY,
    MODEL_PATCH,
    MODEL_PATCH_MUA,
    MODEL_PATCH_SWITCH,
    MODEL_PATCH_SWR,
    MODEL_PER_BIN_RATE,
    MODEL_PER_STOP_RATE,
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    N_REWARD_LAGS,
    SPLINE_KNOTS,
    STOP_COVARIATES,
    TMAX,
    ModelSpec,
    PerBinRateModelSpec,
    default_content_model_names,
)
from .per_bin_rate_model import (
    cross_validate_per_bin_rate_model,
    fit_per_bin_rate_model,
    per_bin_rate_gof,
)
from .per_stop_rate_model import PER_STOP_RATE_OUTCOMES, fit_per_stop_rate_model
from .retrospective import (
    arm_path_well_to_well,
    at_well_event_cohort,
    chosen_arm_fraction,
    chosen_patch_fraction,
    compute_retrospective_metrics,
    contains_path_to_next,
    ends_at_next,
    ends_at_next_patch,
    retrospective_alignment_with_shuffle,
)
from .shuffles import (
    circular_time_shuffle,
    next_arm_circular_shift,
    stop_label_shuffle,
)
from .spatial import arm_dist_rank, dist_to_arm_from_well
from .switch_analysis import (
    build_control_aligned_table,
    build_switch_aligned_table,
    stop_patch_replay,
)
from .tables import (
    build_event_arm_table,
    build_event_patch_table,
    build_event_sequence_table,
    build_event_table,
    build_stop_table,
)

__all__ = [
    "ARM_FE_PREDICTORS",
    "BELIEF_DEPENDENT_CONTENT_MODELS",
    "BIN_S",
    "MODEL_CHOICE",
    "MODEL_HIERARCHY",
    "MODEL_PATCH",
    "MODEL_PATCH_MUA",
    "MODEL_PATCH_SWR",
    "MODEL_PATCH_SWITCH",
    "MODEL_PER_BIN_RATE",
    "MODEL_PER_STOP_RATE",
    "N_HISTORY_BASIS",
    "N_HISTORY_LAGS",
    "N_REWARD_LAGS",
    "PER_STOP_RATE_OUTCOMES",
    "POPULATIONS",
    "SPLINE_KNOTS",
    "STOP_COVARIATES",
    "TMAX",
    "UNCERTAINTY_COVARIATES",
    "ModelSpec",
    "PerBinRateModelSpec",
    "StopSummaryConfig",
    "add_arm_fixed_effects",
    "add_trial_context",
    "add_uncertainty_covariates",
    "aggregate_to_timebins",
    "arm_dist_rank",
    "arm_path_well_to_well",
    "at_well_event_cohort",
    "attach_is_swr_to_onsets",
    "behavioral_location_at_times",
    "build_choice_dataset",
    "build_control_aligned_table",
    "build_event_arm_table",
    "build_event_patch_table",
    "build_event_sequence_table",
    "build_event_table",
    "build_residualized_choice_dataset",
    "build_stop_table",
    "build_switch_aligned_table",
    "chosen_arm_fraction",
    "chosen_patch_fraction",
    "circular_time_shuffle",
    "compute_pearson_residuals",
    "compute_retrospective_metrics",
    "contains_path_to_next",
    "cross_validate_conditional_poisson",
    "cross_validate_per_bin_rate_model",
    "dist_to_arm_from_well",
    "default_content_model_names",
    "ends_at_next",
    "ends_at_next_patch",
    "fit_choice_model",
    "fit_conditional_logit",
    "fit_conditional_poisson",
    "fit_is_stay_logistic_per_session",
    "fit_is_stay_pooled_gee",
    "fit_null_model_conditional",
    "fit_per_bin_rate_model",
    "fit_per_stop_rate_model",
    "freeze_common_cohort",
    "next_arm_circular_shift",
    "nearest_well_distance_cm",
    "per_bin_rate_gof",
    "raised_cosine_history_basis",
    "single_arm_from_well",
    "predict_arm_fractions",
    "retrospective_alignment_with_shuffle",
    "stop_label_shuffle",
    "stop_patch_replay",
    "stop_replay_summary",
    "summarize_behavioral_location",
    "trial_context_at_times",
    "trial_cutoff_times",
]
