"""Replay content analysis package.

Houses the conditional Poisson fitter, model hierarchy, spatial mapping,
simulation utilities, and model comparison/checking helpers used across
the replay content analysis pipeline.

Note: public names are exposed through PEP 562 ``__getattr__``, so importing a
leaf submodule (e.g. ``analysis.replay.design_constants``) does not run the
whole package. The eager re-exports used to: they pulled in
``non_local_detector``, JAX and statsmodels through ``tables_arm_patch`` and
the fitters, so every consumer of a row key or a cohort rule paid ~4.5 s and
loaded JAX. A name is imported from its submodule on first access.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

#: Exported name -> the submodule (relative to this package) that defines it.
_LAZY_EXPORTS = {
    # _helpers.
    "add_arm_fixed_effects": "_helpers",
    "single_arm_from_well": "_helpers",
    # choice_prediction.
    "build_choice_dataset": "choice_prediction",
    "fit_choice_model": "choice_prediction",
    # covariates.
    "UNCERTAINTY_COVARIATES": "covariates",
    "add_trial_context": "covariates",
    "add_uncertainty_covariates": "covariates",
    "behavioral_location_from_port_experiences": "covariates",
    "summarize_behavioral_location": "covariates",
    "trial_arrival_times": "covariates",
    "trial_context_at_times": "covariates",
    "trial_outcome_times": "covariates",
    # dirichlet_multinomial. It switches JAX to float64 at import, a
    # process-wide effect paid only by callers that use the estimator (a GPU
    # decoding run in the same process would otherwise inherit float64).
    "cross_validate_dirichlet_multinomial": "dirichlet_multinomial",
    "event_loglik": "dirichlet_multinomial",
    "fit_dirichlet_multinomial": "dirichlet_multinomial",
    "multinomial_log_constant": "dirichlet_multinomial",
    # fitting.
    "fit_conditional_poisson": "fitting",
    # model_specs.
    "ARRIVAL_CLOCK_COLUMN": "model_specs",
    "BIN_S": "model_specs",
    "MODEL_CHOICE": "model_specs",
    "MODEL_PATCH": "model_specs",
    "MODEL_PATCH_MUA": "model_specs",
    "MODEL_PATCH_SWITCH": "model_specs",
    "MODEL_PATCH_SWR": "model_specs",
    "MODEL_PER_BIN_RATE": "model_specs",
    "MODEL_PER_STOP_RATE": "model_specs",
    "N_HISTORY_BASIS": "model_specs",
    "N_HISTORY_LAGS": "model_specs",
    "N_REWARD_LAGS": "model_specs",
    "REWARD_HISTORY_COLUMNS": "model_specs",
    "REWARD_HISTORY_N_BASIS": "model_specs",
    "STOP_COVARIATES": "model_specs",
    "TMAX": "model_specs",
    "TMAX_ARRIVAL": "model_specs",
    "PerBinRateModelSpec": "model_specs",
    # per_bin_design.
    "NO_EXCLUDED_TIME": "per_bin_design",
    "POPULATIONS": "per_bin_design",
    "aggregate_to_timebins": "per_bin_design",
    "attach_is_swr_to_onsets": "per_bin_design",
    "raised_cosine_history_basis": "per_bin_design",
    # per_bin_rate_model.
    "apply_arrival_window": "per_bin_rate_model",
    "cross_validate_per_bin_rate_model": "per_bin_rate_model",
    "fit_per_bin_rate_model": "per_bin_rate_model",
    "freeze_per_bin_cohort": "per_bin_rate_model",
    "per_bin_rate_gof": "per_bin_rate_model",
    # per_stop_rate_model.
    "fit_per_stop_rate_model": "per_stop_rate_model",
    # retrospective.
    "at_well_event_cohort": "retrospective",
    "retrospective_alignment_with_shuffle": "retrospective",
    # switch_analysis.
    "build_switch_aligned_table": "switch_analysis",
    # tables_arm_patch.
    "build_event_patch_table": "tables_arm_patch",
    # tables_stop.
    "build_stop_table": "tables_stop",
}


def __getattr__(name: str):
    """Import an exported name from its submodule on first attribute access."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f"{__name__}.{target}")
    attr = getattr(module, name)
    globals()[name] = attr
    return attr


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
    "NO_EXCLUDED_TIME",
    "POPULATIONS",
    "REWARD_HISTORY_COLUMNS",
    "REWARD_HISTORY_N_BASIS",
    "ARRIVAL_CLOCK_COLUMN",
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
    "apply_arrival_window",
    "freeze_per_bin_cohort",
    "multinomial_log_constant",
    "per_bin_rate_gof",
    "raised_cosine_history_basis",
    "single_arm_from_well",
    "retrospective_alignment_with_shuffle",
    "summarize_behavioral_location",
    "trial_arrival_times",
    "trial_context_at_times",
    "trial_outcome_times",
]


if TYPE_CHECKING:  # pragma: no cover - types only
    from ._helpers import add_arm_fixed_effects, single_arm_from_well  # noqa: F401
    from .choice_prediction import build_choice_dataset, fit_choice_model  # noqa: F401
    from .covariates import (  # noqa: F401
        UNCERTAINTY_COVARIATES,
        add_trial_context,
        add_uncertainty_covariates,
        behavioral_location_from_port_experiences,
        summarize_behavioral_location,
        trial_arrival_times,
        trial_context_at_times,
        trial_outcome_times,
    )
    from .dirichlet_multinomial import (  # noqa: F401
        cross_validate_dirichlet_multinomial,
        event_loglik,
        fit_dirichlet_multinomial,
        multinomial_log_constant,
    )
    from .fitting import fit_conditional_poisson  # noqa: F401
    from .model_specs import (  # noqa: F401
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
    from .per_bin_design import (  # noqa: F401
        NO_EXCLUDED_TIME,
        POPULATIONS,
        aggregate_to_timebins,
        attach_is_swr_to_onsets,
        raised_cosine_history_basis,
    )
    from .per_bin_rate_model import (  # noqa: F401
        apply_arrival_window,
        cross_validate_per_bin_rate_model,
        fit_per_bin_rate_model,
        freeze_per_bin_cohort,
        per_bin_rate_gof,
    )
    from .per_stop_rate_model import fit_per_stop_rate_model  # noqa: F401
    from .retrospective import (  # noqa: F401
        at_well_event_cohort,
        retrospective_alignment_with_shuffle,
    )
    from .switch_analysis import build_switch_aligned_table  # noqa: F401
    from .tables_arm_patch import build_event_patch_table  # noqa: F401
    from .tables_stop import build_stop_table  # noqa: F401
