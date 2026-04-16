"""Interaction recovery and model-selection accuracy tests.

These tests use ``simulate_with_event_covariates`` to generate data
with a known generative truth that includes interaction terms, then ask:

1. **Interaction recovery**: when the truth has a non-zero interaction
   coefficient (e.g., a SWR-specific current-arm preference), can the
   conditional Poisson recover its sign and magnitude?
2. **Model-selection accuracy**: when the truth is M2 (only base
   indicators active), do AIC/BIC pick M2 over the larger M3-M5?
3. **Selection power**: when the truth IS M3 (reward binary active),
   does AIC/BIC correctly upgrade from M2 to M3?

These are the validations that determine whether the M0–M8 hierarchy
comparisons reported in the paper actually identify the right
generative story.
"""

from __future__ import annotations

import pytest

from continuum_swr_replay.analysis.replay.fitting import (
    fit_conditional_poisson,
)
from continuum_swr_replay.analysis.replay.model_specs import (
    MODEL_HIERARCHY,
)
from continuum_swr_replay.analysis.replay.simulation import (
    simulate_with_event_covariates,
)

# ── Interaction recovery ──


class TestInteractionRecovery:
    """Each interaction column is identifiable when its base predictor is
    also in the model. We test recovery of one interaction at a time, with
    the matching base effect held at a non-zero baseline."""

    @pytest.mark.parametrize(
        "interaction,base",
        [
            ("swr_x_current", "is_current"),
            ("switch_x_prev", "is_prev"),
            ("current_x_last_rewarded", "is_last_rewarded"),
            ("omission_x_last_rewarded", "is_last_rewarded"),
        ],
    )
    def test_recover_single_interaction(self, interaction, base):
        truth = {base: 0.4, interaction: 0.6}
        df = simulate_with_event_covariates(
            truth, n_events=600, L_e=30, seed=hash(interaction) % 10000
        )
        fit = fit_conditional_poisson(df, predictors=list(truth))
        assert fit is not None
        for p, val in truth.items():
            z = (fit[f"coef_{p}"] - val) / fit[f"bse_{p}"]
            assert abs(z) < 3.5, (
                f"{p}: coef={fit[f'coef_{p}']:.3f} truth={val} z={z:.2f}"
            )


class TestVIFForInteractions:
    """Interactions are products of a base predictor and an event-level
    scalar — collinearity with the base is a real risk. Verify VIF stays
    bounded under typical event-level covariate distributions."""

    def test_vif_under_five(self):
        truth = {
            "is_current": 0.5,
            "is_prev": 0.3,
            "swr_x_current": 0.4,
            "switch_x_prev": 0.2,
        }
        df = simulate_with_event_covariates(truth, n_events=600, L_e=30, seed=99)
        fit = fit_conditional_poisson(df, predictors=list(truth))
        assert fit is not None
        for p in truth:
            vif = fit.get(f"vif_{p}")
            assert vif is not None, f"missing VIF for {p}"
            assert vif < 5.0, f"{p}: VIF={vif:.2f} suggests collinearity"


# ── Model selection ──


def _fit_hierarchy(df, models):
    """Fit each named model in MODEL_HIERARCHY on df and return the
    {model_name: fit_dict} mapping (excluding any model that fails)."""
    fits = {}
    for name in models:
        spec = MODEL_HIERARCHY[name]
        fit = fit_conditional_poisson(df, predictors=list(spec.predictors))
        if fit is not None:
            fit["n_params"] = len(spec.predictors)
            fits[name] = fit
    return fits


def _selection_winner(fits, criterion):
    """Return the model name with the best (smallest) criterion value."""
    return min(fits, key=lambda name: fits[name][criterion])


class TestSelectionPicksTruthM2:
    """When data are generated from M2 (spatial indicators only), BIC
    should not significantly prefer the larger models."""

    def test_bic_does_not_prefer_overfit(self):
        truth = {
            "is_current": 0.6,
            "is_prev": 0.7,
            # Everything beyond M2 set to 0
            "is_last_rewarded": 0.0,
            "current_x_last_rewarded": 0.0,
            "visit_recency": 0.0,
            "cumulative_reward": 0.0,
            "dist_to_arm": 0.0,
        }
        df = simulate_with_event_covariates(truth, n_events=800, L_e=25, seed=7)
        fits = _fit_hierarchy(df, ["M0", "M1", "M2", "M3", "M4", "M5"])
        winner = _selection_winner(fits, "bic")
        assert winner in {"M1", "M2"}, (
            f"BIC picked {winner}; expected M1 or M2. "
            f"BICs: {[(n, round(f['bic'], 2)) for n, f in fits.items()]}"
        )


class TestSelectionUpgradesWhenTruthIsLarger:
    """When data are generated from M3 (reward binary active), both
    AIC and BIC should prefer M3 (or larger) over M2."""

    def test_selection_finds_reward_predictor(self):
        truth = {
            "is_current": 0.5,
            "is_prev": 0.4,
            "is_last_rewarded": 0.6,
            "current_x_last_rewarded": -0.3,
            # Continuous reward history at 0 — not the truth
            "visit_recency": 0.0,
            "cumulative_reward": 0.0,
        }
        df = simulate_with_event_covariates(truth, n_events=800, L_e=25, seed=42)
        fits = _fit_hierarchy(df, ["M0", "M1", "M2", "M3", "M4", "M5"])
        for criterion in ["aic", "bic"]:
            winner = _selection_winner(fits, criterion)
            # Should pick M3+ because the reward binary is active.
            assert winner not in {"M0", "M1", "M2"}, (
                f"{criterion.upper()} picked {winner}; expected M3+. "
                f"{criterion}s: {[(n, round(f[criterion], 2)) for n, f in fits.items()]}"
            )

    def test_m3_strictly_better_than_m2_by_lrt(self):
        truth = {
            "is_current": 0.5,
            "is_prev": 0.4,
            "cumulative_reward": 0.6,
        }
        df = simulate_with_event_covariates(truth, n_events=800, L_e=25, seed=42)
        fits = _fit_hierarchy(df, ["M2", "M3"])
        # LRT: 2*(llf_M3 - llf_M2) should exceed chi^2 threshold
        lr_stat = 2 * (fits["M3"]["llf"] - fits["M2"]["llf"])
        assert lr_stat > 6, (
            f"LR statistic M3 vs M2 = {lr_stat:.2f}; "
            f"expected > 6 for the active cumulative_reward to register"
        )


class TestSelectionUnderNull:
    """When truth is M0 (no effects), BIC should prefer simpler models."""

    def test_bic_picks_null_when_truth_is_null(self):
        truth = {
            "is_current": 0.0,
            "is_prev": 0.0,
            "is_last_rewarded": 0.0,
            "current_x_last_rewarded": 0.0,
        }
        df = simulate_with_event_covariates(truth, n_events=500, L_e=20, seed=123)
        fits = _fit_hierarchy(df, ["M0", "M1", "M2", "M3"])
        winner = _selection_winner(fits, "bic")
        assert winner == "M0", (
            f"BIC picked {winner}; expected M0. "
            f"BICs: {[(n, round(f['bic'], 2)) for n, f in fits.items()]}"
        )
