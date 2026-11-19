"""Traceable public API for behavioral clocks, context, and history covariates.

The submodules encode an explicit dependency direction: physical trial clocks
feed time-local context and behavioral history; generic table-building helpers
do not own or re-export these scientific definitions.

Reading order for a newcomer tracing how a covariate is derived:
1. ``docs/covariate_dictionary.md`` — the prose scientific companion.
2. ``definitions.py`` — the machine-readable catalog: find any covariate by
   name; its ``producer`` field names the code that computes/attaches it.
3. Raw behavioral inputs are built in ``data_loaders/trials.py``
   (``make_well_visits_df`` -> ``make_trials_df_from_well_visits`` ->
   ``make_port_experiences_df``).
4. ``clocks.py`` owns arrival-vs-outcome time; ``context.py`` owns per-timebin
   trial context; the ``history/`` package owns behavioral history, split by
   concern into ``reward_history`` (rolling reward counts / recent reward
   rates), ``visit_recency`` (how recently / whether an arm was visited),
   ``value_uncertainty`` (leaky-Q value history, model-free uncertainty),
   ``value_block`` (the precomputed qvals-artifact joins) and ``diagnostics``
   (offline learner, no cataloged covariate);
   ``candidate_arm.py`` is the shared nine-arm predictor block.
5. Pre/post-outcome timing lives in ``../covariate_timing.py``.
6. ``interactions.py`` declares the factors of every interaction product a
   ladder fits; the fit-time products are built from it, and timing and the
   qvals gap read it.
"""

from . import candidate_arm, clocks, context, definitions, history
from .candidate_arm import (
    attach_candidate_arm_covariates,
    broadcast_candidate_arms,
)
from .clocks import (
    trial_arrival_times,
    trial_outcome_times,
)
from .context import (
    add_trial_context,
    behavioral_location_from_port_experiences,
    summarize_behavioral_location,
    trial_context_at_times,
)
from .definitions import (
    COVARIATE_DEFINITIONS,
    POST_CHOICE_COVARIATES,
    Availability,
    CovariateDefinition,
    QueryClock,
    definition_for,
    definitions_frame,
)
from .history import (
    CANONICAL_LEAKY_Q_ALPHA,
    CANONICAL_SWITCH_WINDOW,
    UNCERTAINTY_COVARIATES,
    add_uncertainty_covariates,
    arm_never_visited_all_events,
    assert_qvals_aligned,
    attach_arm_uncertainty,
    attach_contingency_value_block,
    canonical_port_value_history,
    canonical_trial_switch_history,
    center_visit_recency_within_event,
    lagged_trial_rewards,
    load_contingency,
    q_all_arm,
    recent_reward_rate_all_events,
    recent_trial_reward_rate,
    reward_basis_from_lags,
    reward_history_basis_features,
    rolling_reward_count_all_events,
    spline_visit_recency_within_event,
    visit_recency_all_events,
    visit_recency_and_never_visited_all_events,
)

__all__ = [
    "CANONICAL_LEAKY_Q_ALPHA",
    "CANONICAL_SWITCH_WINDOW",
    "COVARIATE_DEFINITIONS",
    "POST_CHOICE_COVARIATES",
    "UNCERTAINTY_COVARIATES",
    "Availability",
    "CovariateDefinition",
    "QueryClock",
    "add_trial_context",
    "add_uncertainty_covariates",
    "arm_never_visited_all_events",
    "assert_qvals_aligned",
    "attach_arm_uncertainty",
    "attach_candidate_arm_covariates",
    "attach_contingency_value_block",
    "behavioral_location_from_port_experiences",
    "broadcast_candidate_arms",
    "candidate_arm",
    "canonical_port_value_history",
    "canonical_trial_switch_history",
    "center_visit_recency_within_event",
    "clocks",
    "context",
    "definition_for",
    "definitions",
    "definitions_frame",
    "history",
    "lagged_trial_rewards",
    "load_contingency",
    "q_all_arm",
    "recent_reward_rate_all_events",
    "recent_trial_reward_rate",
    "reward_basis_from_lags",
    "reward_history_basis_features",
    "rolling_reward_count_all_events",
    "spline_visit_recency_within_event",
    "summarize_behavioral_location",
    "trial_arrival_times",
    "trial_context_at_times",
    "trial_outcome_times",
    "visit_recency_all_events",
    "visit_recency_and_never_visited_all_events",
]
