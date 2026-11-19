"""History covariates, organized by concern.

Reading order — each concern is its own module:

  * reward history — :mod:`reward_history` (rolling reward counts, recent
    reward rates over an arm's last visits, stop-level reward histories);
  * visit recency — :mod:`visit_recency` (how recently, and whether ever, each
    arm was visited, at event time);
  * value & uncertainty — :mod:`value_uncertainty` (the canonical leaky-Q value
    history and the model-free behavioral-uncertainty covariates, computed from
    the behavioral stream);
  * value-block join — :mod:`value_block` (joins the precomputed ``qvals_table``
    artifact: candidate-arm reward probability, belief mean, belief variance);
  * diagnostics — :mod:`diagnostics` (a per-animal fitted-alpha leaky-Q learner
    used only for offline model checking; it produces no cataloged covariate);
  * shared plumbing — :mod:`_common` (port-experience validation).

Only vectorized, all-events public names are re-exported here. Private helpers
are imported from their owning concern submodule (for example
``history._common._prepare_port_experiences``), not off this package.
"""

from __future__ import annotations

from .diagnostics import (
    q_all_arm,
)
from .reward_history import (
    lagged_trial_rewards,
    recent_reward_rate_all_events,
    recent_trial_reward_rate,
    reward_basis_from_lags,
    reward_history_basis_features,
    rolling_reward_count_all_events,
)
from .value_block import (
    assert_qvals_aligned,
    attach_arm_uncertainty,
    attach_contingency_value_block,
    load_contingency,
)
from .value_uncertainty import (
    CANONICAL_LEAKY_Q_ALPHA,
    CANONICAL_SWITCH_WINDOW,
    UNCERTAINTY_COVARIATES,
    add_uncertainty_covariates,
    canonical_port_value_history,
    canonical_trial_switch_history,
)
from .visit_recency import (
    arm_never_visited_all_events,
    center_visit_recency_within_event,
    spline_visit_recency_within_event,
    visit_recency_all_events,
    visit_recency_and_never_visited_all_events,
)

__all__ = [
    "CANONICAL_LEAKY_Q_ALPHA",
    "CANONICAL_SWITCH_WINDOW",
    "UNCERTAINTY_COVARIATES",
    "add_uncertainty_covariates",
    "arm_never_visited_all_events",
    "assert_qvals_aligned",
    "attach_arm_uncertainty",
    "attach_contingency_value_block",
    "canonical_port_value_history",
    "canonical_trial_switch_history",
    "center_visit_recency_within_event",
    "lagged_trial_rewards",
    "load_contingency",
    "q_all_arm",
    "recent_reward_rate_all_events",
    "recent_trial_reward_rate",
    "reward_basis_from_lags",
    "reward_history_basis_features",
    "rolling_reward_count_all_events",
    "spline_visit_recency_within_event",
    "visit_recency_all_events",
    "visit_recency_and_never_visited_all_events",
]
