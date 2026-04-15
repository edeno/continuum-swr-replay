"""Tier-A simulation tests: does fit_conditional_poisson recover known β?

For each scenario we generate synthetic per-arm timebin counts from a known
softmax-multinomial generative model and check that
``fit_conditional_poisson`` returns coefficients within a few standard
errors of the truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from continuum_swr_replay.analysis.hippocampus.replay_content.fitting import (
    fit_conditional_poisson,
)
from continuum_swr_replay.analysis.hippocampus.replay_content.simulation import (
    DEFAULT_PREDICTORS,
    make_synthetic_env,
    simulate_autocorrelated_counts,
    simulate_counts,
    simulate_full_pipeline,
    simulate_hard_vs_soft_counts,
    simulate_posterior_bumps,
)


def _recover(beta_true, *, n_events=400, L_e=25, seed=0):
    df = simulate_counts(beta_true, n_events=n_events, L_e=L_e, seed=seed)
    fit = fit_conditional_poisson(df, predictors=list(beta_true))
    assert fit is not None, "fit returned None"
    return fit


def _zscore(fit, name, truth):
    return (fit[f"coef_{name}"] - truth) / fit[f"bse_{name}"]


class TestSingleEffectRecovery:
    """One non-zero coefficient at a time — sign and magnitude must recover."""

    @pytest.mark.parametrize(
        "predictor,truth",
        [
            ("is_current", 1.0),
            ("is_prev", 0.8),
            ("is_next", 0.8),
            ("is_last_rewarded", 0.6),
        ],
    )
    def test_single_predictor(self, predictor, truth):
        beta = {p: (truth if p == predictor else 0.0) for p in DEFAULT_PREDICTORS}
        fit = _recover(beta, n_events=400, L_e=25, seed=42)
        z = _zscore(fit, predictor, truth)
        assert abs(z) < 3, (
            f"{predictor}: coef={fit[f'coef_{predictor}']:.3f} "
            f"truth={truth}, z={z:.2f}"
        )


class TestNullScenario:
    """Under β = 0, fitted coefficients should not differ from zero."""

    def test_null_coefficients_near_zero(self):
        beta = dict.fromkeys(DEFAULT_PREDICTORS, 0.0)
        fit = _recover(beta, n_events=400, L_e=25, seed=7)
        for p in DEFAULT_PREDICTORS:
            z = fit[f"coef_{p}"] / fit[f"bse_{p}"]
            assert abs(z) < 3, f"{p}: |z|={abs(z):.2f} under null"


class TestMixedScenario:
    """Past + future + reward, all non-zero, all should be recovered."""

    def test_joint_recovery(self):
        beta = {
            "is_current": 0.5,
            "is_prev": 0.7,
            "is_next": -0.4,
            "is_last_rewarded": 0.3,
        }
        fit = _recover(beta, n_events=600, L_e=30, seed=11)
        for p, truth in beta.items():
            z = _zscore(fit, p, truth)
            assert abs(z) < 3, (
                f"{p}: coef={fit[f'coef_{p}']:.3f} truth={truth} z={z:.2f}"
            )


class TestPowerScales:
    """Bigger n_events × L_e → tighter SE on the same truth."""

    def test_se_shrinks_with_more_data(self):
        beta = {p: (1.0 if p == "is_current" else 0.0) for p in DEFAULT_PREDICTORS}
        fit_small = _recover(beta, n_events=100, L_e=10, seed=3)
        fit_large = _recover(beta, n_events=800, L_e=40, seed=3)
        assert fit_large["bse_is_current"] < fit_small["bse_is_current"]


class TestContinuousPredictor:
    """Continuous arm-varying predictor (visit_recency) recovers."""

    def test_visit_recency(self):
        # negative coefficient: more-recently-visited arms get more replay
        beta = {"is_current": 0.4, "visit_recency": -0.15}
        fit = _recover(beta, n_events=500, L_e=30, seed=5)
        z = _zscore(fit, "visit_recency", -0.15)
        assert abs(z) < 3, f"visit_recency z={z:.2f}"


# ── Tier C tests: posterior/MAP-space simulation ──


class TestSyntheticEnv:
    """Synthetic env shape and bin layout sanity checks."""

    def test_default_env_has_nine_arms(self):
        env = make_synthetic_env()
        assert set(env.arm_extents) == set(range(9))
        bin_df = env.place_bin_centers_nodes_df_
        real_bins = bin_df[bin_df["edge_id"] >= 0]
        # 25 bins per arm × 9 arms
        assert len(real_bins) == 9 * 25
        # 4 junction bins between each adjacent arm pair (8 gaps)
        junction_bins = bin_df[bin_df["edge_id"] == -1]
        assert len(junction_bins) == 8 * 4

    def test_bin_centers_increase_monotonically(self):
        env = make_synthetic_env()
        lp = env.place_bin_centers_nodes_df_["linear_position"].values
        assert (np.diff(lp) > 0).all()


class TestTierCRecoverySigmaZero:
    """At sigma=0 every MAP lands cleanly in its chosen arm — Tier C should
    behave like Tier A (no spatial-smearing artifact)."""

    def test_recovery_matches_truth(self):
        beta = {"is_current": 0.8, "is_prev": 0.5}
        long_df, diag = simulate_posterior_bumps(
            beta, n_events=400, L_e=25, sigma=0.0, seed=17
        )
        assert diag["fraction_on_chosen"] == pytest.approx(1.0)
        fit = fit_conditional_poisson(long_df, predictors=list(beta))
        assert fit is not None
        for p, truth in beta.items():
            z = (fit[f"coef_{p}"] - truth) / fit[f"bse_{p}"]
            assert abs(z) < 3, f"{p} z={z:.2f}"


class TestSpatialSmearingAttenuates:
    """At high sigma, a chosen arm's MAPs leak to neighboring arms whose
    is_current = 0. This should attenuate the recovered indicator
    coefficient relative to truth — a real-pipeline artifact that Tier A
    cannot expose."""

    def test_indicator_coef_shrinks_with_sigma(self):
        beta = {"is_current": 1.5}
        # sigma=0: clean recovery
        df_clean, diag_clean = simulate_posterior_bumps(
            beta, n_events=600, L_e=30, sigma=0.0, seed=23
        )
        fit_clean = fit_conditional_poisson(df_clean, predictors=["is_current"])
        # sigma=20 (arm length 50): substantial leakage to neighbors
        df_smeared, diag_smeared = simulate_posterior_bumps(
            beta, n_events=600, L_e=30, sigma=20.0, seed=23
        )
        fit_smeared = fit_conditional_poisson(
            df_smeared, predictors=["is_current"]
        )
        assert diag_smeared["fraction_on_chosen"] < diag_clean["fraction_on_chosen"]
        # attenuation: smeared coef strictly smaller than clean coef
        assert (
            fit_smeared["coef_is_current"] < fit_clean["coef_is_current"]
        ), (
            f"clean={fit_clean['coef_is_current']:.3f} "
            f"smeared={fit_smeared['coef_is_current']:.3f}"
        )
        # And meaningfully less than truth
        assert fit_smeared["coef_is_current"] < beta["is_current"]


# ── Full-pipeline tests: real compute_arm_counts → make_long_format_arm chain ──


class TestFullPipelineRecovery:
    """End-to-end simulation through the production transformation chain."""

    def test_pipeline_produces_expected_shape(self):
        beta = {"is_current": 0.5}
        long_df, events, diag = simulate_full_pipeline(
            beta, n_events=50, L_e=20, sigma=0.0, seed=1
        )
        assert diag["n_events_used"] > 40
        # 9 rows per surviving event
        assert len(long_df) == diag["n_events_used"] * 9
        assert {"event_uid", "arm_id", "arm_count", "is_current"}.issubset(
            long_df.columns
        )

    def test_recovers_past_arm_influence(self):
        """If replays preferentially visit the previously-departed arm, the
        full pipeline should recover a positive is_prev coefficient."""
        beta = {"is_current": 0.0, "is_prev": 0.8, "is_next": 0.0}
        long_df, _, _ = simulate_full_pipeline(
            beta, n_events=400, L_e=25, sigma=0.0, seed=11
        )
        fit = fit_conditional_poisson(long_df, predictors=list(beta))
        assert fit is not None
        z_prev = (fit["coef_is_prev"] - 0.8) / fit["bse_is_prev"]
        assert abs(z_prev) < 3, f"is_prev z={z_prev:.2f}"
        # Other coefficients shouldn't spuriously inflate
        assert abs(fit["coef_is_current"] / fit["bse_is_current"]) < 3
        assert abs(fit["coef_is_next"] / fit["bse_is_next"]) < 3

    def test_recovers_future_arm_influence(self):
        """Symmetric: prospective replay (toward next arm) should recover."""
        beta = {"is_current": 0.0, "is_prev": 0.0, "is_next": 0.8}
        long_df, _, _ = simulate_full_pipeline(
            beta, n_events=400, L_e=25, sigma=0.0, seed=12
        )
        fit = fit_conditional_poisson(long_df, predictors=list(beta))
        assert fit is not None
        z_next = (fit["coef_is_next"] - 0.8) / fit["bse_is_next"]
        assert abs(z_next) < 3, f"is_next z={z_next:.2f}"

    def test_distinguishes_past_from_future(self):
        """Both effects present and unequal — model should separate them.

        Each coefficient must recover within 3·SE of truth (per-coefficient
        gate). We deliberately do NOT assert a strict ordering across
        coefficients on a single seed — small true-β gaps (e.g. 0.4 vs 0.3)
        have non-trivial probability of inverting under the joint sampling
        distribution even when each coefficient is well-estimated.
        """
        beta = {"is_current": 0.3, "is_prev": 0.9, "is_next": 0.4}
        long_df, _, _ = simulate_full_pipeline(
            beta, n_events=600, L_e=30, sigma=0.0, seed=13
        )
        fit = fit_conditional_poisson(long_df, predictors=list(beta))
        assert fit is not None
        for p, truth in beta.items():
            z = (fit[f"coef_{p}"] - truth) / fit[f"bse_{p}"]
            assert abs(z) < 3, (
                f"{p}: coef={fit[f'coef_{p}']:.3f} truth={truth} z={z:.2f}"
            )
        # The largest true effect must remain the largest fitted effect —
        # this is a much weaker (and well-powered) discrimination check
        # than ranking all three coefficients.
        assert fit["coef_is_prev"] > max(
            fit["coef_is_current"], fit["coef_is_next"]
        ), (
            f"is_prev (truth=0.9) failed to dominate: "
            f"prev={fit['coef_is_prev']:.3f} "
            f"current={fit['coef_is_current']:.3f} "
            f"next={fit['coef_is_next']:.3f}"
        )


# ── Tier B tests: SE calibration under within-event autocorrelation ──


def _replicated_fits(simulator, beta, n_replicates, **kwargs):
    """Run ``simulator`` over ``n_replicates`` seeds and return the
    arrays (coefs, bses) for each predictor."""
    predictor = next(iter(beta))
    coefs, bses = [], []
    for s in range(n_replicates):
        df = simulator(beta, seed=s, **kwargs)
        fit = fit_conditional_poisson(df, predictors=list(beta))
        if fit is None:
            continue
        coefs.append(fit[f"coef_{predictor}"])
        bses.append(fit[f"bse_{predictor}"])
    return np.asarray(coefs), np.asarray(bses)


class TestAutocorrelationDoesNotBiasPointEstimate:
    """Marginal arm distribution under the lazy Markov chain still equals
    softmax(Xβ), so β̂ should be unbiased even at high rho."""

    def test_mean_coef_close_to_truth_with_high_rho(self):
        beta = {"is_current": 0.7}
        coefs, _ = _replicated_fits(
            simulate_autocorrelated_counts,
            beta,
            n_replicates=40,
            n_events=150,
            L_e=20,
            rho=0.7,
        )
        sem = float(np.std(coefs, ddof=1) / np.sqrt(len(coefs)))
        assert abs(coefs.mean() - 0.7) < 3 * sem, (
            f"mean={coefs.mean():.3f} truth=0.7 sem={sem:.3f}"
        )


class TestSEUnderCoverageUnderAutocorrelation:
    """The main Tier-B finding: with rho > 0, the conditional Poisson's
    nominal ``bse`` is *narrower* than the empirical SE of β̂ across
    replicates. CIs computed from ``bse`` will therefore under-cover and
    p-values will be anti-conservative.

    We compare two regimes — iid (rho=0) and strongly correlated (rho=0.8)
    — using the same n_events, L_e, and β. If the conditional Poisson's
    SE were robust to autocorrelation, the calibration ratio
    mean(bse) / empirical_se would be ~1 in both cases. We expect it to
    stay near 1 at rho=0 and to drop below 1 at rho=0.8.
    """

    def test_calibration_degrades_with_autocorrelation(self):
        beta = {"is_current": 0.6}
        common = dict(n_events=120, L_e=20, n_replicates=40)

        coefs_iid, bses_iid = _replicated_fits(
            simulate_autocorrelated_counts, beta, rho=0.0, **common
        )
        coefs_corr, bses_corr = _replicated_fits(
            simulate_autocorrelated_counts, beta, rho=0.8, **common
        )

        emp_se_iid = float(coefs_iid.std(ddof=1))
        emp_se_corr = float(coefs_corr.std(ddof=1))
        ratio_iid = float(bses_iid.mean() / emp_se_iid)
        ratio_corr = float(bses_corr.mean() / emp_se_corr)

        # Sanity: iid case should be near 1 (allow generous tolerance for
        # 40 replicates).
        assert 0.7 < ratio_iid < 1.4, (
            f"iid calibration ratio {ratio_iid:.2f} should be ~1"
        )
        # Main finding: autocorrelated case should have a noticeably
        # narrower nominal bse than the empirical spread of β̂.
        assert ratio_corr < ratio_iid * 0.85, (
            f"Expected SE under-coverage under rho=0.8: "
            f"ratio_iid={ratio_iid:.2f} vs ratio_corr={ratio_corr:.2f}. "
            f"emp_se_iid={emp_se_iid:.3f} emp_se_corr={emp_se_corr:.3f} "
            f"mean_bse_iid={bses_iid.mean():.3f} "
            f"mean_bse_corr={bses_corr.mean():.3f}"
        )


# ── MAP-collapse vs full-posterior comparison ──


class TestHardVsSoftAgreeAtSigmaZero:
    """At sigma=0 every bump is a delta on the chosen arm — hard and soft
    should give numerically identical β̂."""

    def test_match_at_sigma_zero(self):
        beta = {"is_current": 0.6}
        hard, soft, _ = simulate_hard_vs_soft_counts(
            beta, n_events=300, L_e=20, sigma=0.0, seed=37
        )
        fit_hard = fit_conditional_poisson(hard, predictors=list(beta))
        fit_soft = fit_conditional_poisson(soft, predictors=list(beta))
        assert fit_hard is not None and fit_soft is not None
        assert fit_hard["coef_is_current"] == pytest.approx(
            fit_soft["coef_is_current"], rel=1e-3
        )


class TestHardVsSoftBothAttenuateUnderSmearing:
    """Hard MAP and soft posterior averaging both attenuate, but in
    *different* directions and amounts.

    Empirical finding (recorded here as the test): under a Gaussian-bump
    decoder posterior with σ comparable to half the arm length, the hard
    MAP path actually attenuates *less* than the soft path — because MAP
    is a sharp summary that concentrates on the bump's mode, while the
    soft path correctly preserves diffuse tail mass and thereby spreads
    counts more uniformly across arms.

    Implication: contrary to the naive intuition that "using the full
    posterior would obviously be better", swapping MAP-collapse for soft
    counts in the production pipeline would *worsen* the attenuation of
    indicator coefficients under realistic σ. Whether this generalises to
    other generative models (e.g., bimodal posteriors that MAP picks
    asymmetrically) is open and worth checking.
    """

    def test_both_attenuate_and_record_relative_magnitudes(self):
        truth = 1.5
        beta = {"is_current": truth}
        hard, soft, _ = simulate_hard_vs_soft_counts(
            beta, n_events=600, L_e=30, sigma=20.0, seed=41
        )
        fit_hard = fit_conditional_poisson(hard, predictors=list(beta))
        fit_soft = fit_conditional_poisson(soft, predictors=list(beta))
        assert fit_hard is not None and fit_soft is not None
        # Both paths attenuate (point estimate strictly below truth).
        assert fit_hard["coef_is_current"] < truth
        assert fit_soft["coef_is_current"] < truth
        # And the attenuation directions are not equal — there is a real
        # gap between the two methods on the same data.
        assert fit_hard["coef_is_current"] != pytest.approx(
            fit_soft["coef_is_current"], rel=0.02
        ), (
            f"hard β̂ = {fit_hard['coef_is_current']:.3f}, "
            f"soft β̂ = {fit_soft['coef_is_current']:.3f}, truth = {truth}"
        )


class TestUnbiasedAcrossSeeds:
    """Mean of repeated fits should be close to the truth (unbiasedness)."""

    def test_mean_coef_close_to_truth(self):
        beta = {"is_current": 0.6}
        coefs = []
        for seed in range(20):
            df = simulate_counts(beta, n_events=200, L_e=20, seed=seed)
            fit = fit_conditional_poisson(df, predictors=["is_current"])
            assert fit is not None
            coefs.append(fit["coef_is_current"])
        mean = float(np.mean(coefs))
        sem = float(np.std(coefs, ddof=1) / np.sqrt(len(coefs)))
        # mean within ~3 SEM of truth
        assert abs(mean - 0.6) < 3 * sem, (
            f"mean={mean:.3f} truth=0.6 sem={sem:.3f}"
        )
