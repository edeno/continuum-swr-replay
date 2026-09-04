"""Guard that the history package preserves the former module's public surface.

``history.py`` became a ``history/`` package split by concern. Every name that
was importable from ``covariates.history`` (and re-exported through
``covariates``) must still resolve, so no downstream importer breaks.
"""

from __future__ import annotations

import importlib

# The complete set of names external code imports from covariates.history,
# including the private helpers pulled by name (candidate_arm, context, tests).
_HISTORY_SURFACE = (
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
    "_prepare_port_experiences",
    "_observed_port_outcomes",
    "_leaky_q_value_diff",
    "_omission_streak",
    "_q_chosen",
    "_fit_alpha",
)

_HISTORY = "continuum_swr_replay.analysis.replay.covariates.history"
_COVARIATES = "continuum_swr_replay.analysis.replay.covariates"


def test_history_package_reexports_every_prior_name():
    """Every prior ``covariates.history.X`` name still resolves off the package."""
    history = importlib.import_module(_HISTORY)
    missing = [name for name in _HISTORY_SURFACE if not hasattr(history, name)]
    assert not missing, f"history package no longer exposes: {missing}"


def test_history_is_a_package_split_by_concern():
    """The concern submodules import cleanly and hold the expected owners."""
    reward = importlib.import_module(f"{_HISTORY}.reward_history")
    recency = importlib.import_module(f"{_HISTORY}.visit_recency")
    value = importlib.import_module(f"{_HISTORY}.value_uncertainty")
    diagnostics = importlib.import_module(f"{_HISTORY}.diagnostics")
    common = importlib.import_module(f"{_HISTORY}._common")

    assert hasattr(reward, "rolling_reward_count_all_events")
    assert hasattr(recency, "visit_recency_all_events")
    assert hasattr(value, "canonical_port_value_history")
    assert hasattr(diagnostics, "q_all_arm")
    assert hasattr(common, "_prepare_port_experiences")


def test_history_names_still_resolve_through_covariates_package():
    """The public names re-exported through ``covariates`` are unchanged objects."""
    history = importlib.import_module(_HISTORY)
    covariates = importlib.import_module(_COVARIATES)
    for name in ("rolling_reward_count_all_events", "visit_recency_all_events",
                 "canonical_port_value_history", "q_all_arm"):
        assert getattr(covariates, name) is getattr(history, name), name
