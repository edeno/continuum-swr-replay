"""Unit conventions and session-drop rule of the shared contrast aggregation.

``std_factor`` resolves three mutually exclusive labels, in order: binary
per-category predictors, the reward-history basis features (weighted reward
fractions, so their coefficients are per unit of the fraction), and continuous
covariates scaled by their pooled SD.

``aggregate_contrast`` drops a session fit for a term only when THAT term is
unidentified: the fitters name the separated terms on the effect scale, with
``|coef| < 10`` as a backstop. Precision alone is not a drop criterion.
"""

from __future__ import annotations

import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.contrast_aggregation import (
    aggregate_contrast,
    std_factor,
)
from continuum_swr_replay.analysis.replay.model_specs import REWARD_HISTORY_COLUMNS

SD = {"rpe_postoutcome": 2.0, "reward_history_rcos_0": 0.25}

# Eight identified per-session coefficients; median 0.315.
REWARD_COEFS = [0.30, 0.35, 0.25, 0.40, 0.20, 0.45, 0.28, 0.33]


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


def _session_fits(**overrides) -> pd.DataFrame:
    """Eight identified sessions across two animals for one (model, population)."""
    n = 8
    fits = pd.DataFrame(
        {
            "model": ["G5_rpe"] * n,
            "population": ["non_local"] * n,
            "animal": ["a"] * 4 + ["b"] * 4,
            "coef_rpe_postoutcome": [0.30, 0.35, 0.25, 0.40, 0.20, 0.45, 0.28, 0.33],
            "bse_rpe_postoutcome": [0.10] * n,
            "pval_rpe_postoutcome": [0.01] * n,
            "separated": [False] * n,
            "converged": [True] * n,
        }
    )
    for column, values in overrides.items():
        fits[column] = values
    return fits


SPECS = [("G5_rpe", "non_local", "rpe_postoutcome")]


def test_imprecise_but_identified_session_is_kept() -> None:
    """A near-zero coefficient with an ordinary SE is a small effect, not separation.

    coef 0.01 with SE 0.5 satisfies the retired ``SE > 10 * |coef|`` rule, which
    dropped every session whose effect happened to be near zero -- selection on
    the estimate, biasing the surviving median away from zero.
    """
    fits = _session_fits()
    fits.loc[0, "coef_rpe_postoutcome"] = 0.01
    fits.loc[0, "bse_rpe_postoutcome"] = 0.5

    out = aggregate_contrast(fits, SPECS, level="rate")

    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_sessions"] == 8
    assert row["n_dropped"] == 0
    assert row["coef_session_median"] == pytest.approx(0.305)


def test_separated_session_is_dropped() -> None:
    """Result frames written before the term-level flag carry only ``separated``;
    for those the fit-level flag remains the identifiability rule."""
    fits = _session_fits()
    fits.loc[0, "separated"] = True

    out = aggregate_contrast(fits, SPECS, level="rate")

    row = out.iloc[0]
    assert row["n_sessions"] == 7
    assert row["n_dropped"] == 1
    assert row["n_sessions_flagged_separated"] == 1
    assert row["coef_session_median"] == pytest.approx(0.33)


def test_runaway_coefficient_is_still_dropped_by_the_backstop() -> None:
    """|coef| >= 10 remains a backstop for fits with no separation flag."""
    fits = _session_fits()
    fits.loc[0, "coef_rpe_postoutcome"] = 42.0

    out = aggregate_contrast(fits, SPECS, level="rate")

    row = out.iloc[0]
    assert row["n_sessions"] == 7
    assert row["n_dropped"] == 1


def _two_term_fits() -> pd.DataFrame:
    """Eight sessions carrying both a self-history and a reward-basis coefficient."""
    n = 8
    return pd.DataFrame(
        {
            "model": ["S1_rew"] * n,
            "population": ["contrast"] * n,
            "animal": ["a"] * 4 + ["b"] * 4,
            "coef_hist_rcos_2": [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
            "bse_hist_rcos_2": [0.10] * n,
            "pval_hist_rcos_2": [0.01] * n,
            "coef_reward_history_rcos_0": REWARD_COEFS,
            "bse_reward_history_rcos_0": [0.10] * n,
            "pval_reward_history_rcos_0": [0.01] * n,
            "separated": [False] * n,
            "converged": [True] * n,
            "separated_terms": [""] * n,
        }
    )


def test_separation_drops_only_the_term_it_holds_for() -> None:
    """A session separated on ``hist_rcos_2`` still contributes its reward coefficient.

    The fit-level flag dropped the session for every term, which is what emptied
    the audited is_swr-contrast table.
    """
    fits = _two_term_fits()
    fits.loc[0, "separated"] = True
    fits.loc[0, "separated_terms"] = "hist_rcos_2"

    out = aggregate_contrast(
        fits,
        [
            ("S1_rew", "contrast", "hist_rcos_2"),
            ("S1_rew", "contrast", "reward_history_rcos_0"),
        ],
        level="rate",
    ).set_index("term")

    assert out.loc["hist_rcos_2", "n_sessions"] == 7
    assert out.loc["hist_rcos_2", "n_dropped"] == 1
    assert out.loc["hist_rcos_2", "coef_session_median"] == pytest.approx(0.50)
    assert out.loc["reward_history_rcos_0", "n_sessions"] == 8
    assert out.loc["reward_history_rcos_0", "n_dropped"] == 0
    assert out.loc["reward_history_rcos_0", "coef_session_median"] == pytest.approx(
        0.315
    )
    # The fit-level diagnostic still counts the flagged fit, for both terms.
    assert set(out["n_sessions_flagged_separated"]) == {1}


def test_is_swr_product_is_dropped_when_either_factor_separates() -> None:
    """``is_swr:<c>`` is only estimable where both of its factors are."""
    n = 12
    fits = pd.DataFrame(
        {
            "model": ["S4_value"] * n,
            "population": ["contrast"] * n,
            "animal": ["a"] * 6 + ["b"] * 6,
            "coef_is_swr:q_arm": [0.1 * (i + 1) for i in range(n)],
            "bse_is_swr:q_arm": [0.10] * n,
            "pval_is_swr:q_arm": [0.01] * n,
            "separated": [False] * n,
            "converged": [True] * n,
            # Session 1 separated on the covariate, session 2 on the indicator.
            "separated_terms": ["", "q_arm", "is_swr"] + [""] * (n - 3),
        }
    )

    out = aggregate_contrast(
        fits, [("S4_value", "contrast", "is_swr:q_arm")], level="rate"
    )

    assert out.iloc[0]["n_sessions"] == 10
    assert out.iloc[0]["n_dropped"] == 2
    # 0.2 and 0.3 are gone; the remaining ten run 0.1, 0.4, 0.5, ..., 1.2.
    assert out.iloc[0]["coef_session_median"] == pytest.approx(0.75)


def test_unrelated_separated_term_never_drops_a_session() -> None:
    """The nuisance/self-history terms that flagged 55% of fits are inert here."""
    fits = _two_term_fits()
    fits["separated"] = True
    fits["separated_terms"] = "hist_rcos_0|hist_rcos_1"

    out = aggregate_contrast(
        fits, [("S1_rew", "contrast", "reward_history_rcos_0")], level="rate"
    )

    assert out.iloc[0]["n_sessions"] == 8
    assert out.iloc[0]["n_dropped"] == 0
    assert out.iloc[0]["n_sessions_flagged_separated"] == 8
