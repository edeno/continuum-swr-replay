"""Tier-A simulation tests: does fit_conditional_poisson recover known β?

For each scenario we generate synthetic per-arm timebin counts from a known
softmax-multinomial generative model and check that
``fit_conditional_poisson`` returns coefficients within a few standard
errors of the truth.
"""

from __future__ import annotations

import numpy as np
import pytest

from continuum_swr_replay.analysis.replay_models.content.simulation import (
    DEFAULT_PREDICTORS,
    _markov_arm_sequence,
    make_synthetic_env,
    simulate_autocorrelated_counts,
    simulate_counts,
    simulate_dirichlet_multinomial_counts,
    simulate_full_pipeline,
    simulate_hard_vs_soft_counts,
    simulate_posterior_bumps,
    simulate_with_event_covariates,
)
from continuum_swr_replay.analysis.replay_models.fitting import (
    fit_conditional_poisson,
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
            ("is_current_arm", 1.0),
            ("is_prev_arm", 0.8),
            ("is_next_arm", 0.8),
            ("is_last_rewarded_arm", 0.6),
        ],
    )
    def test_single_predictor(self, predictor, truth):
        beta = {p: (truth if p == predictor else 0.0) for p in DEFAULT_PREDICTORS}
        fit = _recover(beta, n_events=400, L_e=25, seed=42)
        z = _zscore(fit, predictor, truth)
        assert abs(z) < 3, (
            f"{predictor}: coef={fit[f'coef_{predictor}']:.3f} truth={truth}, z={z:.2f}"
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
            "is_current_arm": 0.5,
            "is_prev_arm": 0.7,
            "is_next_arm": -0.4,
            "is_last_rewarded_arm": 0.3,
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
        beta = {p: (1.0 if p == "is_current_arm" else 0.0) for p in DEFAULT_PREDICTORS}
        fit_small = _recover(beta, n_events=100, L_e=10, seed=3)
        fit_large = _recover(beta, n_events=800, L_e=40, seed=3)
        assert fit_large["bse_is_current_arm"] < fit_small["bse_is_current_arm"]


class TestContinuousPredictor:
    """Continuous arm-varying predictor (visit_recency) recovers."""

    def test_visit_recency(self):
        # negative coefficient: more-recently-visited arms get more replay
        beta = {"is_current_arm": 0.4, "visit_recency_trials": -0.15}
        fit = _recover(beta, n_events=500, L_e=30, seed=5)
        z = _zscore(fit, "visit_recency_trials", -0.15)
        assert abs(z) < 3, f"visit_recency z={z:.2f}"

    def test_novelty_is_coherent_with_recency_and_swr_interaction(self):
        df = simulate_with_event_covariates(
            {
                "visit_recency_trials": 0.0,
                "is_candidate_arm_never_visited": 0.0,
                "swr_x_is_candidate_arm_never_visited": 0.0,
            },
            n_events=40,
            seed=17,
        )
        assert df["is_candidate_arm_never_visited"].eq(1.0).any()
        assert df["is_candidate_arm_never_visited"].eq(0.0).any()
        for _, event in df.groupby("event_id"):
            event = event.set_index("arm_id")
            np.testing.assert_array_equal(
                event.loc[[6, 7, 8], "is_candidate_arm_never_visited"], 0.0
            )
            novelty = event.loc[range(6), "is_candidate_arm_never_visited"]
            assert novelty.isin([0.0, 1.0]).all()
            if novelty.eq(1.0).any() and novelty.eq(0.0).any():
                assert (
                    event.loc[
                        novelty.index[novelty.eq(1.0)], "visit_recency_trials"
                    ].min()
                    > event.loc[
                        novelty.index[novelty.eq(0.0)], "visit_recency_trials"
                    ].max()
                )
        np.testing.assert_array_equal(
            df["swr_x_is_candidate_arm_never_visited"],
            df["is_swr"] * df["is_candidate_arm_never_visited"],
        )


class TestFractionalResponse:
    """Fitter must accept non-integer responses without silent truncation.

    ``arm_count_hard_hard`` is integer, but the robustness responses
    ``arm_count_hard_soft`` and ``arm_posterior_mass`` are fractional. Before
    the fix, ``fit_conditional_poisson`` cast the response to int, which
    silently floored soft responses < 1 to zero and destroyed information.
    """

    def test_integer_and_float_responses_agree(self):
        """Casting integer counts to float must not change β̂."""
        beta = {"is_current_arm": 0.8, "is_prev_arm": 0.4}
        df = simulate_counts(beta, n_events=300, L_e=25, seed=101)
        fit_int = fit_conditional_poisson(df, predictors=list(beta))

        df_float = df.copy()
        df_float["arm_count_hard"] = df_float["arm_count_hard"].astype(float)
        fit_float = fit_conditional_poisson(df_float, predictors=list(beta))

        assert fit_int is not None and fit_float is not None
        for p in beta:
            assert fit_int[f"coef_{p}"] == pytest.approx(
                fit_float[f"coef_{p}"], rel=1e-9
            )
            assert fit_int[f"bse_{p}"] == pytest.approx(fit_float[f"bse_{p}"], rel=1e-9)

    def test_subunit_fractional_response_is_not_floored(self):
        """Responses in (0, 1) must influence the fit, not be floored to 0.

        Scaling all counts by a constant < 1 yields a likelihood that is a
        positive-scalar multiple of the integer case, so β̂ is unchanged
        and ``bse`` scales by 1/sqrt(scale). If the old ``.astype(int)``
        truncation were still in effect, every response would floor to 0,
        the all-zero event filter would drop every event, and the fit
        would return None.
        """
        beta = {"is_current_arm": 1.0}
        df = simulate_counts(beta, n_events=300, L_e=25, seed=102)

        scale = 0.3
        df_frac = df.copy()
        df_frac["arm_count_hard"] = df_frac["arm_count_hard"].astype(float) * scale

        fit_int = fit_conditional_poisson(df, predictors=list(beta))
        fit_frac = fit_conditional_poisson(df_frac, predictors=list(beta))

        assert fit_int is not None
        assert fit_frac is not None, (
            "fit returned None — fractional responses were likely floored"
        )
        # β̂ is mathematically invariant to a positive scalar rescaling of
        # the response; numerical match is at the optimizer's convergence
        # tolerance, not machine epsilon.
        assert fit_frac["coef_is_current_arm"] == pytest.approx(
            fit_int["coef_is_current_arm"], rel=1e-4
        )
        # bse scales by 1/sqrt(scale) because the information matrix scales
        # linearly with the response.
        expected_bse = fit_int["bse_is_current_arm"] / np.sqrt(scale)
        assert fit_frac["bse_is_current_arm"] == pytest.approx(expected_bse, rel=1e-2)


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
        beta = {"is_current_arm": 0.8, "is_prev_arm": 0.5}
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
        beta = {"is_current_arm": 1.5}
        # sigma=0: clean recovery
        df_clean, diag_clean = simulate_posterior_bumps(
            beta, n_events=600, L_e=30, sigma=0.0, seed=23
        )
        fit_clean = fit_conditional_poisson(df_clean, predictors=["is_current_arm"])
        # sigma=20 (arm length 50): substantial leakage to neighbors
        df_smeared, diag_smeared = simulate_posterior_bumps(
            beta, n_events=600, L_e=30, sigma=20.0, seed=23
        )
        fit_smeared = fit_conditional_poisson(df_smeared, predictors=["is_current_arm"])
        assert diag_smeared["fraction_on_chosen"] < diag_clean["fraction_on_chosen"]
        # attenuation: smeared coef strictly smaller than clean coef
        assert fit_smeared["coef_is_current_arm"] < fit_clean["coef_is_current_arm"], (
            f"clean={fit_clean['coef_is_current_arm']:.3f} "
            f"smeared={fit_smeared['coef_is_current_arm']:.3f}"
        )
        # And meaningfully less than truth
        assert fit_smeared["coef_is_current_arm"] < beta["is_current_arm"]


# ── Full-pipeline tests: production map_estimate_to_arm → arm_count_hard ──


class TestFullPipelineRecovery:
    """Simulated events through the production MAP-to-arm mapping and hard arm
    count (`simulate_full_pipeline`), then the production fitter.
    """

    def test_pipeline_produces_expected_shape(self):
        beta = {"is_current_arm": 0.5}
        long_df, diag = simulate_full_pipeline(
            beta, n_events=50, L_e=20, sigma=0.0, seed=1
        )
        assert diag["n_events_used"] > 40
        # 9 rows per surviving event
        assert len(long_df) == diag["n_events_used"] * 9
        assert {"event_id", "arm_id", "arm_count_hard", "is_current_arm"}.issubset(
            long_df.columns
        )

    def test_recovers_past_arm_influence(self):
        """If replays preferentially visit the previously-departed arm, the
        full pipeline should recover a positive is_prev coefficient."""
        beta = {"is_current_arm": 0.0, "is_prev_arm": 0.8, "is_next_arm": 0.0}
        long_df, _ = simulate_full_pipeline(
            beta, n_events=400, L_e=25, sigma=0.0, seed=11
        )
        fit = fit_conditional_poisson(long_df, predictors=list(beta))
        assert fit is not None
        z_prev = (fit["coef_is_prev_arm"] - 0.8) / fit["bse_is_prev_arm"]
        assert abs(z_prev) < 3, f"is_prev z={z_prev:.2f}"
        # Other coefficients shouldn't spuriously inflate
        assert abs(fit["coef_is_current_arm"] / fit["bse_is_current_arm"]) < 3
        assert abs(fit["coef_is_next_arm"] / fit["bse_is_next_arm"]) < 3

    def test_recovers_future_arm_influence(self):
        """Symmetric: prospective replay (toward next arm) should recover."""
        beta = {"is_current_arm": 0.0, "is_prev_arm": 0.0, "is_next_arm": 0.8}
        long_df, _ = simulate_full_pipeline(
            beta, n_events=400, L_e=25, sigma=0.0, seed=12
        )
        fit = fit_conditional_poisson(long_df, predictors=list(beta))
        assert fit is not None
        z_next = (fit["coef_is_next_arm"] - 0.8) / fit["bse_is_next_arm"]
        assert abs(z_next) < 3, f"is_next z={z_next:.2f}"

    def test_distinguishes_past_from_future(self):
        """Both effects present and unequal — model should separate them.

        Each coefficient must recover within 3·SE of truth (per-coefficient
        gate). We deliberately do NOT assert a strict ordering across
        coefficients on a single seed — small true-β gaps (e.g. 0.4 vs 0.3)
        have non-trivial probability of inverting under the joint sampling
        distribution even when each coefficient is well-estimated.
        """
        beta = {"is_current_arm": 0.3, "is_prev_arm": 0.9, "is_next_arm": 0.4}
        long_df, _ = simulate_full_pipeline(
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
        assert fit["coef_is_prev_arm"] > max(
            fit["coef_is_current_arm"], fit["coef_is_next_arm"]
        ), (
            f"is_prev (truth=0.9) failed to dominate: "
            f"prev={fit['coef_is_prev_arm']:.3f} "
            f"current={fit['coef_is_current_arm']:.3f} "
            f"next={fit['coef_is_next_arm']:.3f}"
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


class TestLazyMarkovChainOverdispersesCounts:
    """Pin the direction of the variance change documented in
    ``simulation.py`` module + ``simulate_autocorrelated_counts`` docs.

    The lazy Markov chain ``X_{t+1} = X_t (w.p. rho) else fresh ~ pi``
    has the same marginal pi as iid sampling, but positive autocorrelation
    INFLATES the variance of the per-event arm count S_i = sum_t I(X_t = i)
    by a factor approaching ``(1 + rho) / (1 - rho)`` for large L. The
    docstring previously claimed the variance was *smaller* (which would
    be underdispersion), and the conclusion that ``bse`` is too narrow
    requires the opposite — overdispersion. This test fails if anyone
    rewires the chain in a way that inverts the variance direction OR if
    the docstring claim is reverted.

    Empirical reference (pi=[0.3, 0.7], L=20, 5000 trials):
    - iid (rho=0)  variance ≈ 4.07 (matches analytic Multinomial 4.20)
    - rho=0.8      variance ≈ 30.3 (~7× larger, consistent with
      asymptotic factor (1+0.8)/(1-0.8) = 9 attenuated by finite L).
    """

    def test_high_rho_variance_strictly_exceeds_iid_and_multinomial(self):
        rng = np.random.default_rng(0)
        pi = np.array([0.3, 0.7])
        L = 20
        n_trials = 2000

        def _var_arm0(rho: float) -> float:
            counts = np.empty(n_trials)
            for i in range(n_trials):
                seq = _markov_arm_sequence(pi, rho, L, rng)
                counts[i] = (seq == 0).sum()
            return float(counts.var(ddof=1))

        var_iid = _var_arm0(0.0)
        var_corr = _var_arm0(0.8)
        multinom_var = L * pi[0] * (1 - pi[0])  # = 4.2

        # iid case must match multinomial within ~25% (sampling noise).
        assert 0.75 * multinom_var < var_iid < 1.25 * multinom_var, (
            f"iid variance {var_iid:.2f} should match Multinomial "
            f"{multinom_var:.2f}; sanity check failed."
        )
        # Main pin: rho=0.8 must produce strictly LARGER count
        # variance than iid, by a wide margin (analytic factor ~9).
        assert var_corr > 3.0 * var_iid, (
            f"Expected overdispersion under rho=0.8: var_corr="
            f"{var_corr:.2f} should be ≫ var_iid={var_iid:.2f} "
            "(analytic ratio ≈ 9). If this fails the docstring's "
            "'larger / overdispersion' claim and the bse-too-narrow "
            "story would be reversed."
        )


class TestAutocorrelationDoesNotBiasPointEstimate:
    """Marginal arm distribution under the lazy Markov chain still equals
    softmax(Xβ), so β̂ should be unbiased even at high rho."""

    @pytest.mark.slow
    def test_mean_coef_close_to_truth_with_high_rho(self):
        beta = {"is_current_arm": 0.7}
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

    @pytest.mark.slow
    def test_calibration_degrades_with_autocorrelation(self):
        beta = {"is_current_arm": 0.6}
        common = {"n_events": 120, "L_e": 20, "n_replicates": 40}

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
        beta = {"is_current_arm": 0.6}
        hard, soft, _ = simulate_hard_vs_soft_counts(
            beta, n_events=300, L_e=20, sigma=0.0, seed=37
        )
        fit_hard = fit_conditional_poisson(hard, predictors=list(beta))
        fit_soft = fit_conditional_poisson(soft, predictors=list(beta))
        assert fit_hard is not None and fit_soft is not None
        assert fit_hard["coef_is_current_arm"] == pytest.approx(
            fit_soft["coef_is_current_arm"], rel=1e-3
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
    indicator coefficients under realistic σ. Whether this generalizes to
    other generative models (e.g., bimodal posteriors that MAP picks
    asymmetrically) is open and worth checking.
    """

    def test_both_attenuate_and_record_relative_magnitudes(self):
        truth = 1.5
        beta = {"is_current_arm": truth}
        hard, soft, _ = simulate_hard_vs_soft_counts(
            beta, n_events=600, L_e=30, sigma=20.0, seed=41
        )
        fit_hard = fit_conditional_poisson(hard, predictors=list(beta))
        fit_soft = fit_conditional_poisson(soft, predictors=list(beta))
        assert fit_hard is not None and fit_soft is not None
        # Both paths attenuate (point estimate strictly below truth).
        assert fit_hard["coef_is_current_arm"] < truth
        assert fit_soft["coef_is_current_arm"] < truth
        # The empirical claim in the docstring is *directional*: under
        # this generative regime, hard MAP attenuates LESS than soft
        # posterior averaging, so ``hard_β̂`` sits between ``soft_β̂``
        # and ``truth``. A test that only required the two estimates
        # to differ would still pass if a regression flipped the
        # ordering — exactly the case the docstring's "swapping MAP
        # for soft would *worsen* attenuation" claim hinges on.
        assert fit_hard["coef_is_current_arm"] > fit_soft["coef_is_current_arm"], (
            f"hard β̂ = {fit_hard['coef_is_current_arm']:.3f}, "
            f"soft β̂ = {fit_soft['coef_is_current_arm']:.3f}, "
            f"truth = {truth} — hard MAP must attenuate less than soft "
            "posterior averaging under this regime, otherwise the "
            "test's docstring claim no longer holds."
        )


class TestUnbiasedAcrossSeeds:
    """Mean of repeated fits should be close to the truth (unbiasedness)."""

    def test_mean_coef_close_to_truth(self):
        beta = {"is_current_arm": 0.6}
        coefs = []
        for seed in range(20):
            df = simulate_counts(beta, n_events=200, L_e=20, seed=seed)
            fit = fit_conditional_poisson(df, predictors=["is_current_arm"])
            assert fit is not None
            coefs.append(fit["coef_is_current_arm"])
        mean = float(np.mean(coefs))
        sem = float(np.std(coefs, ddof=1) / np.sqrt(len(coefs)))
        # mean within ~3 SEM of truth
        assert abs(mean - 0.6) < 3 * sem, f"mean={mean:.3f} truth=0.6 sem={sem:.3f}"


class TestDirichletMultinomialSimulator:
    """The overdispersed simulator: same softmax mean, correlated within-event bins."""

    def test_long_format_and_totals(self):
        df = simulate_dirichlet_multinomial_counts(
            {"is_current_arm": 1.0},
            log_phi_intercept=np.log(3.0),
            n_events=40,
            L_e=32,
            seed=0,
        )
        assert {"event_id", "candidate_arm", "arm_count_hard", "arm_fraction"} <= set(
            df.columns
        )
        assert len(df) == 40 * 9
        totals = df.groupby("event_id")["arm_count_hard"].sum()
        assert (totals == 32).all()
        fractions = df.groupby("event_id")["arm_fraction"].sum()
        np.testing.assert_allclose(fractions.to_numpy(), 1.0)
        # Every canonical predictor the multinomial simulator builds is present.
        assert {"is_swr", "is_prev_arm", "swr_x_is_current_arm"} <= set(df.columns)

    def test_p_true_is_returned_and_is_a_distribution(self):
        df = simulate_dirichlet_multinomial_counts(
            {"is_current_arm": 1.0},
            log_phi_intercept=np.log(3.0),
            n_events=25,
            L_e=20,
            seed=1,
        )
        p_true = df.attrs["p_true"]
        assert p_true.shape == (25, 9)
        np.testing.assert_allclose(p_true.sum(axis=1), 1.0)
        assert (p_true >= 0).all()

    def test_seed_is_reproducible(self):
        kwargs = {"log_phi_intercept": np.log(3.0), "n_events": 20, "L_e": 16}
        a = simulate_dirichlet_multinomial_counts(
            {"is_current_arm": 1.0}, seed=5, **kwargs
        )
        b = simulate_dirichlet_multinomial_counts(
            {"is_current_arm": 1.0}, seed=5, **kwargs
        )
        c = simulate_dirichlet_multinomial_counts(
            {"is_current_arm": 1.0}, seed=6, **kwargs
        )
        np.testing.assert_array_equal(a["arm_count_hard"], b["arm_count_hard"])
        assert not np.array_equal(a["arm_count_hard"], c["arm_count_hard"])

    def test_small_phi_overdisperses_relative_to_multinomial(self):
        """Small phi spreads per-event counts far beyond the multinomial variance.

        Large phi is the multinomial limit, so its count variance must land near
        the analytic ``L·p(1-p)`` while small phi lands well above it.
        """
        beta = {"is_current_arm": 1.0}
        L_e = 32
        variances = {}
        for phi in (1.5, 1e4):
            df = simulate_dirichlet_multinomial_counts(
                beta, log_phi_intercept=np.log(phi), n_events=2000, L_e=L_e, seed=3
            )
            current = df.loc[df["is_current_arm"] == 1.0, "arm_count_hard"].to_numpy()
            variances[phi] = float(np.var(current, ddof=1))
            if phi == 1e4:
                p = float(np.mean(current)) / L_e
                multinomial_var = L_e * p * (1 - p)

        assert variances[1e4] == pytest.approx(multinomial_var, rel=0.15)
        assert variances[1.5] > 3.0 * variances[1e4], variances

    def test_is_swr_shifts_concentration(self):
        """A negative ``log_phi_is_swr`` makes SWR events the more concentrated ones."""
        df = simulate_dirichlet_multinomial_counts(
            {"is_current_arm": 1.0},
            log_phi_intercept=np.log(20.0),
            log_phi_is_swr=-3.0,
            n_events=800,
            L_e=32,
            seed=4,
        )
        peak = df.groupby("event_id")["arm_fraction"].max()
        is_swr = df.groupby("event_id")["is_swr"].first()
        swr_peak = peak[is_swr == 1.0].mean()
        non_swr_peak = peak[is_swr == 0.0].mean()
        assert swr_peak > non_swr_peak + 0.05, (swr_peak, non_swr_peak)

    def test_unknown_predictor_rejected(self):
        with pytest.raises(ValueError, match="does not know how to build"):
            simulate_dirichlet_multinomial_counts(
                {"not_a_real_column": 1.0}, log_phi_intercept=0.0, n_events=10, seed=0
            )


class TestSharedDesignSamplerIsSeedPinned:
    """The two simulators draw one design from one RNG, in a fixed order.

    Both public simulators call the same private column sampler as their first
    RNG consumer, so a reordering of the draws inside it silently changes every
    seeded fixture in the suite -- including the recovery tests, whose tolerances
    are stated against these exact designs. These values pin the draw order.
    """

    GOLDEN = {
        "is_current_arm": [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "is_prev_arm": [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        "visit_recency_trials": [4.0, 7.0, 3.0, 1.0, 5.0, 2.0, 0.0, 0.0, 0.0],
        "distance_to_candidate_arm_cm": [
            72.901512,
            92.742393,
            96.792619,
            1.470630,
            86.364009,
            98.119504,
            95.721018,
            14.876401,
            97.262881,
        ],
        "is_swr": [0.0] * 9,
        "time_since_stop_start_s": [1.208379] * 9,
    }

    def _first_event(self, df):
        return df[df["event_id"] == 0].reset_index(drop=True)

    def test_multinomial_simulator_draws_the_pinned_design(self):
        df = self._first_event(
            simulate_with_event_covariates(
                {"is_current_arm": 0.5}, n_events=3, L_e=8, seed=0
            )
        )
        for column, expected in self.GOLDEN.items():
            np.testing.assert_allclose(
                df[column].to_numpy(dtype=float), expected, atol=1e-6, err_msg=column
            )
        assert df["arm_count_hard"].tolist() == [0, 0, 1, 1, 3, 1, 0, 1, 1]

    def test_overdispersed_simulator_shares_that_design(self):
        df = self._first_event(
            simulate_dirichlet_multinomial_counts(
                {"is_current_arm": 0.5},
                log_phi_intercept=1.0,
                n_events=3,
                L_e=8,
                seed=0,
            )
        )
        for column, expected in self.GOLDEN.items():
            np.testing.assert_allclose(
                df[column].to_numpy(dtype=float), expected, atol=1e-6, err_msg=column
            )
