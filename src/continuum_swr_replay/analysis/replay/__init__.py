"""Replay content analysis package.

Houses the conditional Poisson fitter, model hierarchy, spatial mapping,
simulation utilities, and model comparison/checking helpers used across
the replay content analysis pipeline.
"""

from ._helpers import (
    add_arm_fixed_effects,
    single_arm_from_well,
)
from .choice_prediction import (
    build_choice_dataset,
    fit_choice_model,
)
from .cohort_selection import (
    STOP_COHORT_ARTIFACT_SUFFIX,
    STOP_COHORT_RULES,
    STOP_COHORTS,
    freeze_common_cohort,
    select_event_cohort,
    select_stop_cohort,
    stop_cohort_suffix,
)
from .covariates import (
    UNCERTAINTY_COVARIATES,
    add_trial_context,
    add_uncertainty_covariates,
    behavioral_location_from_port_experiences,
    summarize_behavioral_location,
    trial_arrival_times,
    trial_context_at_times,
    trial_outcome_times,
)
from .fitting import fit_conditional_poisson
from .model_specs import (
    ARRIVAL_CLOCK_COLUMN,
    BIN_S,
    MODEL_CHOICE,
    MODEL_PATCH,
    MODEL_PATCH_MUA,
    MODEL_PATCH_SWITCH,
    MODEL_PATCH_SWR,
    MODEL_PER_BIN_RATE,
    MODEL_PER_STOP_RATE,
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    N_REWARD_LAGS,
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_N_BASIS,
    STOP_COVARIATES,
    TMAX,
    TMAX_ARRIVAL,
    PerBinRateModelSpec,
)
from .per_bin_design import (
    POPULATIONS,
    aggregate_to_timebins,
    attach_is_swr_to_onsets,
    raised_cosine_history_basis,
)
from .per_bin_rate_model import (
    apply_arrival_window,
    cross_validate_per_bin_rate_model,
    fit_per_bin_rate_model,
    freeze_per_bin_cohort,
    per_bin_rate_gof,
)
from .per_stop_rate_model import fit_per_stop_rate_model
from .retrospective import (
    at_well_event_cohort,
    retrospective_alignment_with_shuffle,
)
from .switch_analysis import build_switch_aligned_table
from .tables_arm_patch import build_event_patch_table
from .tables_stop import build_stop_table

# Deferred rather than imported eagerly: the Dirichlet-multinomial fitter pulls
# in JAX and switches it to float64 at import. Both effects are process-wide, so
# they are paid by callers that actually use the estimator rather than by every
# consumer of this package (a GPU decoding run in the same process would
# otherwise inherit float64).
_LAZY_EXPORTS = {
    "cross_validate_dirichlet_multinomial": ".dirichlet_multinomial",
    "event_loglik": ".dirichlet_multinomial",
    "fit_dirichlet_multinomial": ".dirichlet_multinomial",
    "multinomial_log_constant": ".dirichlet_multinomial",
}


def __getattr__(name: str):
    """Import the deferred estimator exports on first attribute access."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module_name, __name__), name)


__all__ = [
    "BIN_S",
    "MODEL_CHOICE",
    "MODEL_PATCH",
    "MODEL_PATCH_MUA",
    "MODEL_PATCH_SWR",
    "MODEL_PATCH_SWITCH",
    "MODEL_PER_BIN_RATE",
    "MODEL_PER_STOP_RATE",
    "N_HISTORY_BASIS",
    "N_HISTORY_LAGS",
    "N_REWARD_LAGS",
    "POPULATIONS",
    "REWARD_HISTORY_COLUMNS",
    "REWARD_HISTORY_N_BASIS",
    "ARRIVAL_CLOCK_COLUMN",
    "STOP_COHORTS",
    "STOP_COHORT_ARTIFACT_SUFFIX",
    "STOP_COHORT_RULES",
    "STOP_COVARIATES",
    "TMAX",
    "TMAX_ARRIVAL",
    "UNCERTAINTY_COVARIATES",
    "PerBinRateModelSpec",
    "add_arm_fixed_effects",
    "add_trial_context",
    "add_uncertainty_covariates",
    "aggregate_to_timebins",
    "at_well_event_cohort",
    "attach_is_swr_to_onsets",
    "behavioral_location_from_port_experiences",
    "build_choice_dataset",
    "build_event_patch_table",
    "build_stop_table",
    "build_switch_aligned_table",
    "cross_validate_dirichlet_multinomial",
    "cross_validate_per_bin_rate_model",
    "event_loglik",
    "fit_choice_model",
    "fit_conditional_poisson",
    "fit_dirichlet_multinomial",
    "fit_per_bin_rate_model",
    "fit_per_stop_rate_model",
    "freeze_common_cohort",
    "apply_arrival_window",
    "freeze_per_bin_cohort",
    "multinomial_log_constant",
    "per_bin_rate_gof",
    "raised_cosine_history_basis",
    "single_arm_from_well",
    "retrospective_alignment_with_shuffle",
    "select_event_cohort",
    "select_stop_cohort",
    "stop_cohort_suffix",
    "summarize_behavioral_location",
    "trial_arrival_times",
    "trial_context_at_times",
    "trial_outcome_times",
]
