"""Unit conventions of the shared contrast aggregation's ``std_factor``.

Three mutually exclusive labels, in the order ``std_factor`` resolves them:
binary per-category predictors, the reward-history basis features (weighted
reward fractions, so their coefficients are per unit of the fraction), and
continuous covariates scaled by their pooled SD.
"""

from __future__ import annotations

from continuum_swr_replay.analysis.replay.contrast_aggregation import std_factor
from continuum_swr_replay.analysis.replay.model_specs import REWARD_HISTORY_COLUMNS

SD = {"rpe_postoutcome": 2.0, "reward_history_rcos_0": 0.25}


def test_binary_predictor_is_per_category() -> None:
    assert std_factor("is_current_arm", SD) == (1.0, "per_category")
    assert std_factor("swr_x_is_current_arm", SD) == (1.0, "per_category")


def test_reward_history_basis_is_per_unit_even_when_an_sd_is_available() -> None:
    """The features are fractions in [0, 1]; a stray SD entry must not rescale them."""
    for column in REWARD_HISTORY_COLUMNS:
        assert std_factor(column, SD) == (1.0, "per_unit")
        assert std_factor(f"is_swr:{column}", SD) == (1.0, "per_unit")


def test_continuous_covariate_is_scaled_by_its_pooled_sd() -> None:
    assert std_factor("rpe_postoutcome", SD) == (2.0, "per_SD")
    assert std_factor("is_swr:rpe_postoutcome", SD) == (2.0, "per_SD")
    assert std_factor("swr_x_rpe_postoutcome", SD) == (2.0, "per_SD")


def test_unknown_term_is_assumed_z_scored_at_fit_time() -> None:
    assert std_factor("rpe_postoutcome_z_x_is_current_arm", SD) == (1.0, "per_SD")
