"""Tests for the covariate module extraction.

Phase-0 of the covariate harmonization moved the behavioral covariate
definitions out of ``_helpers`` into ``covariates`` with a re-export shim. This
checks the move is transparent: every moved name resolves to the *same object*
whether imported from ``covariates``, from the ``_helpers`` shim, or (for the
public ones) off the package namespace.
"""

from __future__ import annotations

import subprocess
import sys

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


def test_covariates_imports_standalone_no_deadlock():
    """Importing ``covariates`` first in a *fresh* interpreter must not deadlock.

    ``_helpers`` and ``covariates`` form a re-export cycle (covariates imports
    ``_trial_cutoff_times`` / ``build_well_to_arm_map`` back from ``_helpers``).
    A subprocess is required to make ``covariates`` the entry import: within this
    test process both modules are already cached, so an inline import would only
    re-assert identity, not the load order. The shim resolves because importing
    the submodule first runs the package ``__init__``, which loads ``_helpers``
    before ``.covariates``.
    """
    code = (
        "from continuum_swr_replay.analysis.replay.covariates import "
        "recent_reward_rate_all_events as a;"
        "from continuum_swr_replay.analysis.replay._helpers import "
        "recent_reward_rate_all_events as b;"
        "assert a is b"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
