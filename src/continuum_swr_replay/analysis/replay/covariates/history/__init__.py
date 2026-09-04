"""History covariates, organized by concern.

Reading order — each concern is its own module:

  * reward history — :mod:`reward_history` (rolling reward counts, recent
    reward rates over an arm's last visits, stop-level reward histories);
  * visit recency — :mod:`visit_recency` (how recently, and whether ever, each
    arm was visited, at event time and as per-arm scalar views);
  * value & uncertainty — :mod:`value_uncertainty` (the canonical leaky-Q value
    history and the model-free behavioral-uncertainty covariates, computed from
    the behavioral stream);
  * value-block join — :mod:`value_block` (joins the precomputed ``qvals_table``
    artifact: candidate-arm reward probability, belief mean, belief variance);
  * diagnostics — :mod:`diagnostics` (a per-animal fitted-alpha leaky-Q learner
    used only for offline model checking; it produces no catalogued covariate);
  * shared plumbing — :mod:`_common` (port-experience validation).

This package replaces the former single ``history.py`` module; every *public*
name that module exposed is re-exported here, so ``covariates.history.X`` and
``from ...covariates.history import X`` keep resolving unchanged for the public
API. Private helpers that had a single caller (e.g. ``_completed_visit_context``)
now live in their concern submodule rather than on this package.
"""

from __future__ import annotations

from ._common import _prepare_port_experiences as _prepare_port_experiences
from .diagnostics import (
    _fit_alpha as _fit_alpha,
)
from .diagnostics import (
    _q_chosen as _q_chosen,
)
from .diagnostics import (
    q_all_arm,
)
from .reward_history import (
    _observed_port_outcomes as _observed_port_outcomes,
)
from .reward_history import (
    lagged_trial_rewards,
    recent_reward_rate_all_events,
    recent_reward_rate_per_arm,
    recent_trial_reward_rate,
    rolling_reward_count_all_events,
    rolling_reward_count_per_arm,
)
from .value_block import (
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
from .value_uncertainty import (
    _leaky_q_value_diff as _leaky_q_value_diff,
)
from .value_uncertainty import (
    _omission_streak as _omission_streak,
)
from .visit_recency import (
    arm_never_visited_all_events,
    arm_never_visited_per_arm,
    center_visit_recency_within_event,
    spline_visit_recency_within_event,
    visit_recency_all_events,
    visit_recency_and_never_visited_all_events,
    visit_recency_per_arm,
)

__all__ = [
    "CANONICAL_LEAKY_Q_ALPHA",
    "CANONICAL_SWITCH_WINDOW",
    "UNCERTAINTY_COVARIATES",
    "add_uncertainty_covariates",
    "arm_never_visited_all_events",
    "arm_never_visited_per_arm",
    "attach_arm_uncertainty",
    "attach_contingency_value_block",
    "canonical_port_value_history",
    "canonical_trial_switch_history",
    "center_visit_recency_within_event",
    "lagged_trial_rewards",
    "load_contingency",
    "q_all_arm",
    "recent_reward_rate_all_events",
    "recent_reward_rate_per_arm",
    "recent_trial_reward_rate",
    "rolling_reward_count_all_events",
    "rolling_reward_count_per_arm",
    "spline_visit_recency_within_event",
    "visit_recency_all_events",
    "visit_recency_and_never_visited_all_events",
    "visit_recency_per_arm",
]
