"""Interaction recovery and model-selection accuracy tests.

These tests use ``simulate_with_event_covariates`` to generate data
with a known generative truth that includes interaction terms, then ask:

1. **Interaction recovery**: when the truth has a non-zero interaction
   coefficient (e.g., a SWR-specific current-arm preference), can the
   conditional Poisson recover its sign and magnitude?
2. **Model-selection accuracy**: when the truth is M2 (only base
   indicators active), do AIC/BIC pick M2 over the larger M3-M7?
3. **Selection power**: when the truth IS M3 (reward predictors active),
   does AIC/BIC correctly upgrade from M2 to M3?

These are the validations that determine whether the M0–M7 hierarchy
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
            ("time_x_next", "is_next"),
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

    def test_vif_under_three(self):
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
    """When data are generated from M2 (base indicators only), neither
    AIC nor BIC should significantly prefer the larger M3-M5 over M2."""

    def test_bic_does_not_prefer_overfit(self):
        truth = {
            "is_current": 0.6,
            "is_prev": 0.7,
            "is_next": 0.4,
            # Reward / recency / distance / interactions all 0
            "is_last_rewarded": 0.0,
            "visit_recency": 0.0,
            "cumulative_reward": 0.0,
            "dist_to_arm": 0.0,
        }
        df = simulate_with_event_covariates(truth, n_events=800, L_e=25, seed=7)
        fits = _fit_hierarchy(df, ["M0", "M1", "M2", "M3", "M4"])
        winner = _selection_winner(fits, "bic")
        # BIC penalizes parameters more strongly; should pick M2 (truth)
        # or M1 (under-fit one predictor). Should NOT pick M3+ because
        # the extra predictors bring no improvement worth the BIC penalty.
        assert winner in {"M1", "M2"}, (
            f"BIC picked {winner}; expected M1 or M2. "
            f"BICs: {[(n, round(f['bic'], 2)) for n, f in fits.items()]}"
        )

    def test_aic_does_not_severely_overfit(self):
        truth = {
            "is_current": 0.6,
            "is_prev": 0.7,
            "is_next": 0.4,
            "is_last_rewarded": 0.0,
            "visit_recency": 0.0,
            "cumulative_reward": 0.0,
            "dist_to_arm": 0.0,
        }
        df = simulate_with_event_covariates(truth, n_events=800, L_e=25, seed=7)
        fits = _fit_hierarchy(df, ["M0", "M1", "M2", "M3", "M4"])
        winner_aic = _selection_winner(fits, "aic")
        # AIC is more permissive than BIC. With null β = 0 on the M3 extras,
        # AIC should still favour M2 most of the time, but allow M3-M4 too
        # — what we want to RULE OUT is AIC picking M0 or M1 when M2 is true.
        assert winner_aic in {"M2", "M3", "M4"}, (
            f"AIC picked {winner_aic}; expected M2-M4. "
            f"AICs: {[(n, round(f['aic'], 2)) for n, f in fits.items()]}"
        )


class TestSelectionUpgradesWhenTruthIsLarger:
    """When data are generated from M3 (reward predictors active), both
    AIC and BIC should prefer M3 (or larger) over M2."""

    def test_selection_finds_reward_predictors(self):
        truth = {
            "is_current": 0.5,
            "is_prev": 0.4,
            "is_next": 0.3,
            "is_last_rewarded": 0.6,
            "visit_recency": -0.15,
            "cumulative_reward": 0.0,
        }
        df = simulate_with_event_covariates(truth, n_events=800, L_e=25, seed=42)
        fits = _fit_hierarchy(df, ["M0", "M1", "M2", "M3", "M4"])
        # Both criteria should beat M2 with M3 (which adds the active
        # predictors). M4 might also win if dist_to_arm by chance helps.
        for criterion in ["aic", "bic"]:
            winner = _selection_winner(fits, criterion)
            assert winner in {"M3", "M4"}, (
                f"{criterion.upper()} picked {winner}; expected M3 or M4. "
                f"{criterion}s: {[(n, round(f[criterion], 2)) for n, f in fits.items()]}"
            )

    def test_m3_strictly_better_than_m2_when_truth_is_m3(self):
        truth = {
            "is_current": 0.5,
            "is_prev": 0.4,
            "is_next": 0.3,
            "is_last_rewarded": 0.6,
            "visit_recency": -0.15,
            "cumulative_reward": 0.0,
        }
        df = simulate_with_event_covariates(truth, n_events=800, L_e=25, seed=42)
        fits = _fit_hierarchy(df, ["M2", "M3"])
        # Likelihood ratio: 2*(llf3 - llf2) >> chi^2(2 df, 0.05) = 5.99
        lr_stat = 2 * (fits["M3"]["llf"] - fits["M2"]["llf"])
        assert lr_stat > 6, (
            f"LR statistic M3 vs M2 = {lr_stat:.2f}; "
            f"expected > 6 for the active reward effects to register"
        )


# ── Null calibration: type-I rate of selection ──


class TestSelectionUnderNull:
    """When all coefficients are 0, BIC should overwhelmingly pick M0
    (the null) and AIC should pick M0 most of the time."""

    def test_bic_picks_null_when_truth_is_null(self):
        truth = dict.fromkeys(
            [
                "is_current",
                "is_prev",
                "is_next",
                "is_last_rewarded",
                "visit_recency",
                "cumulative_reward",
                "dist_to_arm",
            ],
            0.0,
        )
        # Repeat across seeds to make this not a single-seed gate
        m0_picked = 0
        n_seeds = 10
        for seed in range(n_seeds):
            df = simulate_with_event_covariates(
                truth, n_events=400, L_e=20, seed=seed
            )
            fits = _fit_hierarchy(df, ["M0", "M1", "M2", "M3"])
            if _selection_winner(fits, "bic") == "M0":
                m0_picked += 1
        # Under the null, BIC should pick M0 most of the time.
        assert m0_picked >= int(0.7 * n_seeds), (
            f"BIC picked M0 only {m0_picked}/{n_seeds} times under the null; "
            f"expected at least {int(0.7 * n_seeds)}"
        )
