"""Replay content analysis package.

Houses the conditional Poisson fitter, model hierarchy, spatial mapping,
simulation utilities, and model comparison/checking helpers used across
the replay content analysis pipeline.
"""

from ._helpers import add_arm_fixed_effects
from .choice_prediction import (
    build_choice_dataset,
    build_residualized_choice_dataset,
    fit_choice_model,
    predict_arm_fractions,
)
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
    MODEL_HIERARCHY,
    MODEL_PATCH,
    MODEL_PATCH_SWITCH,
    ModelSpec,
)
from .rate_model import fit_rate_model
from .retrospective import (
    arm_path_well_to_well,
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
    "MODEL_HIERARCHY",
    "MODEL_PATCH",
    "MODEL_PATCH_SWITCH",
    "ModelSpec",
    "StopSummaryConfig",
    "add_arm_fixed_effects",
    "arm_dist_rank",
    "arm_path_well_to_well",
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
    "dist_to_arm_from_well",
    "ends_at_next",
    "ends_at_next_patch",
    "fit_choice_model",
    "fit_conditional_logit",
    "fit_conditional_poisson",
    "fit_is_stay_logistic_per_session",
    "fit_is_stay_pooled_gee",
    "fit_null_model_conditional",
    "fit_rate_model",
    "next_arm_circular_shift",
    "predict_arm_fractions",
    "retrospective_alignment_with_shuffle",
    "stop_label_shuffle",
    "stop_patch_replay",
    "stop_replay_summary",
]
