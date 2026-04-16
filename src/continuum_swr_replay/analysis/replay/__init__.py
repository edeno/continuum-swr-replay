"""Replay content analysis package.

Houses the conditional Poisson fitter, model hierarchy, spatial mapping,
simulation utilities, and model comparison/checking helpers used across
the replay content analysis pipeline.
"""

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
from .model_checking import (
    compute_pearson_residuals,
    cross_validate_conditional_poisson,
)
from .model_specs import (
    MODEL_HIERARCHY,
    MODEL_PATCH,
    MODEL_PATCH_SWITCH,
    ModelSpec,
)
from .rate_model import fit_rate_model
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
    "MODEL_HIERARCHY",
    "MODEL_PATCH",
    "MODEL_PATCH_SWITCH",
    "ModelSpec",
    "arm_dist_rank",
    "build_choice_dataset",
    "build_event_arm_table",
    "build_event_patch_table",
    "build_event_sequence_table",
    "build_event_table",
    "build_residualized_choice_dataset",
    "build_control_aligned_table",
    "build_stop_table",
    "build_switch_aligned_table",
    "circular_time_shuffle",
    "compute_pearson_residuals",
    "cross_validate_conditional_poisson",
    "dist_to_arm_from_well",
    "fit_choice_model",
    "fit_conditional_logit",
    "fit_conditional_poisson",
    "fit_null_model_conditional",
    "fit_rate_model",
    "next_arm_circular_shift",
    "predict_arm_fractions",
    "stop_label_shuffle",
    "stop_patch_replay",
]
