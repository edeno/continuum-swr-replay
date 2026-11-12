"""The events and the task: covariates, tables, and the rate, content and choice
analyses fit on them.

Subpackages, each importing only those listed before it and the modules at
this level:

- ``covariates``: the behavioral covariates, and the behavioral-model values
  (``qvals_table``) they join.
- ``choice``: next-choice prediction, stay/leave, switch-aligned content and
  retrospective alignment.
- ``tables``: the canonical event, arm, patch, stop and time-bin tables and
  their contracts.
- ``rate`` and ``content``: the event-rate and content models and their
  statistics. Neither imports the other.

The modules at this level are what several subpackages share: the model
ladders (``model_specs``) and design constants, the per-bin design and its
bases, the fitters and their diagnostics, the spatial lookups, the fit
provenance, the cohort audits, the covariate timing, the contrast
aggregation and the across-session coefficient statistics. Of these,
``cohort_audit``, ``covariate_timing`` and ``contrast_aggregation`` read
``covariates``, and ``coefficient_statistics`` reaches it only through
``contrast_aggregation`` (so ``covariates`` imports none of the four); the
others import no subpackage.

Note: public names are exposed through PEP 562 ``__getattr__``, so importing a
leaf submodule (e.g. ``analysis.behavior.design_constants``) does not run the
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
    "build_choice_dataset": "choice.choice_prediction",
    "fit_choice_model": "choice.choice_prediction",
    # covariates.
    "UNCERTAINTY_COVARIATES": "covariates",
    "add_trial_context": "covariates",
    "add_uncertainty_covariates": "covariates",
    "behavioral_location_from_port_experiences": "covariates",
    "summarize_behavioral_location": "covariates",
    "trial_arrival_times": "covariates",
    "trial_context_at_times": "covariates",
    "trial_outcome_times": "covariates",
    # dirichlet_multinomial. Its JAX calls run in float64 inside
    # ``jax.enable_x64``, scoped to each call; the process default is untouched.
    "cross_validate_dirichlet_multinomial": "content.dirichlet_multinomial",
    "event_loglik": "content.dirichlet_multinomial",
    "fit_dirichlet_multinomial": "content.dirichlet_multinomial",
    "multinomial_log_constant": "content.dirichlet_multinomial",
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
    "apply_arrival_window": "rate.per_bin_rate_model",
    "cross_validate_per_bin_rate_model": "rate.per_bin_rate_model",
    "fit_per_bin_rate_model": "rate.per_bin_rate_model",
    "freeze_per_bin_cohort": "rate.per_bin_rate_model",
    "per_bin_rate_gof": "rate.per_bin_rate_model",
    # per_stop_rate_model.
    "fit_per_stop_rate_model": "rate.per_stop_rate_model",
    # retrospective.
    "at_well_event_cohort": "choice.retrospective",
    "retrospective_alignment_with_shuffle": "choice.retrospective",
    # switch_analysis.
    "build_switch_aligned_table": "choice.switch_analysis",
    # tables_arm_patch.
    "build_event_patch_table": "tables.tables_arm_patch",
    # tables_stop.
    "build_stop_table": "tables.tables_stop",
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


__all__ = sorted(_LAZY_EXPORTS)


if TYPE_CHECKING:  # pragma: no cover - types only
    from ._helpers import add_arm_fixed_effects, single_arm_from_well  # noqa: F401
    from .choice.choice_prediction import (  # noqa: F401
        build_choice_dataset,
        fit_choice_model,
    )
    from .choice.retrospective import (  # noqa: F401
        at_well_event_cohort,
        retrospective_alignment_with_shuffle,
    )
    from .choice.switch_analysis import build_switch_aligned_table  # noqa: F401
    from .content.dirichlet_multinomial import (  # noqa: F401
        cross_validate_dirichlet_multinomial,
        event_loglik,
        fit_dirichlet_multinomial,
        multinomial_log_constant,
    )
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
    from .rate.per_bin_rate_model import (  # noqa: F401
        apply_arrival_window,
        cross_validate_per_bin_rate_model,
        fit_per_bin_rate_model,
        freeze_per_bin_cohort,
        per_bin_rate_gof,
    )
    from .rate.per_stop_rate_model import fit_per_stop_rate_model  # noqa: F401
    from .tables.tables_arm_patch import build_event_patch_table  # noqa: F401
    from .tables.tables_stop import build_stop_table  # noqa: F401
