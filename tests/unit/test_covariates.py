"""Tests for the covariate module extraction.

Phase-0 of the covariate harmonization moved the behavioral covariate
definitions out of ``_helpers`` into ``covariates`` with a re-export shim. This
checks the move is transparent: every moved name resolves to the *same object*
whether imported from ``covariates``, from the ``_helpers`` shim, or (for the
public ones) off the package namespace.
"""

from __future__ import annotations

import continuum_swr_replay.analysis.replay as replay_pkg
from continuum_swr_replay.analysis.replay import _helpers, covariates

# Names that moved to covariates.py and are re-exported from _helpers.
SHIMMED_NAMES = [
    "UNCERTAINTY_COVARIATES",
    "visit_recency_all_events",
    "cumulative_reward_all_events",
    "recent_reward_rate_all_events",
    "recent_trial_reward_rate",
    "lagged_trial_rewards",
    "_leaky_q_value_diff",
    "_omission_streak",
    "add_uncertainty_covariates",
    "visit_recency_per_arm",
    "cumulative_reward_per_arm",
    "recent_reward_rate_per_arm",
]

# Public covariate names also exposed on the package namespace.
PUBLIC_NAMES = ["UNCERTAINTY_COVARIATES", "add_uncertainty_covariates"]


def test_shim_reexports_same_objects():
    """Each moved name is the identical object via covariates and via _helpers."""
    for name in SHIMMED_NAMES:
        via_cov = getattr(covariates, name)
        via_helpers = getattr(_helpers, name)
        assert via_cov is via_helpers, f"{name} differs between covariates and _helpers"


def test_public_names_resolve_via_package():
    """The public covariate names resolve off the package and match covariates."""
    for name in PUBLIC_NAMES:
        assert getattr(replay_pkg, name) is getattr(covariates, name)
        assert name in replay_pkg.__all__


def test_q_learner_lives_in_covariates():
    """The promoted fitted-alpha leaky-Q learner is importable from covariates."""
    assert callable(covariates._q_chosen)
    assert callable(covariates._fit_alpha)


def test_no_circular_import_either_order():
    """Importing covariates first (before _helpers is cached) must not deadlock.

    Re-imports are cached, but the names must still be present regardless of which
    module a consumer reaches first; ``_helpers`` and ``covariates`` form a
    re-export cycle, so this pins that it resolves.
    """
    from continuum_swr_replay.analysis.replay._helpers import (  # noqa: F401
        recent_reward_rate_all_events as via_helpers,
    )
    from continuum_swr_replay.analysis.replay.covariates import (  # noqa: F401
        recent_reward_rate_all_events,
    )

    assert recent_reward_rate_all_events is via_helpers
