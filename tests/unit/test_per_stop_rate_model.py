"""Tests for analysis/replay/per_stop_rate_model.py.

The Poisson rate model is the Stage 2 paper claim: do behavioral
contexts change the NUMBER of replay events per stop, independently
of replay content. A sign error in the ``log(immobility_time)``
offset, a botched Poisson fit, or a silent inclusion of zero-duration
stops would all corrupt the conclusions.

Pin recovery on synthetic data where the truth is known.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.per_stop_rate_model import (
    cross_validate_per_stop_rate_model,
    fit_per_stop_rate_model,
)


def _simulate_stops(
    *,
    n_stops: int,
    log_intercept: float,
    coefs: dict[str, float] | None = None,
    duration_range: tuple[float, float] = (5.0, 30.0),
    seed: int = 0,
) -> pd.DataFrame:
    """Simulate a stop table with known per-stop event rates.

    For each stop i: ``immobility_time_i ~ Uniform(duration_range)``,
    rate ``λ_i = exp(log_intercept + Σ_k β_k * x_{i,k})``,
    ``n_events_i ~ Poisson(λ_i * immobility_time_i)``.

    Predictor values are i.i.d. Bernoulli(0.5) — uncorrelated across
    predictors, so coefficient recovery is identifiable.
    """
    rng = np.random.default_rng(seed)
    coefs = coefs or {}

    immobility_time = rng.uniform(*duration_range, size=n_stops)
    log_lambda = np.full(n_stops, log_intercept, dtype=float)

    columns: dict[str, np.ndarray] = {}
    for name, beta in coefs.items():
        x = rng.integers(0, 2, size=n_stops).astype(float)
        columns[name] = x
        log_lambda += beta * x

    expected_count = np.exp(log_lambda) * immobility_time
    n_events = rng.poisson(expected_count)

    df = pd.DataFrame(
        {
            "n_events": n_events,
            "immobility_time": immobility_time,
            **columns,
        }
    )
    return df


# ── Intercept-only recovery ──


class TestInterceptOnlyRecovery:
    def test_recovers_known_baseline_rate(self):
        """With no predictors, exp(intercept) = mean rate per second
        of immobility. Generate at λ = 2 events/sec, fit intercept
        only, and check exp(coef_intercept) is within a few SE of 2.0."""
        true_rate = 2.0
        df = _simulate_stops(
            n_stops=400, log_intercept=np.log(true_rate), seed=42
        )
        fit = fit_per_stop_rate_model(df, predictors=[])
        assert fit is not None
        recovered_rate = fit["rr_intercept"]
        z = (np.log(recovered_rate) - np.log(true_rate)) / fit["bse_intercept"]
        assert abs(z) < 3, (
            f"intercept rate {recovered_rate:.3f} (truth {true_rate}, z={z:.2f})"
        )

    def test_rr_equals_exp_coef(self):
        """``rr_*`` columns must literally be ``exp(coef_*)``."""
        df = _simulate_stops(n_stops=200, log_intercept=np.log(1.0), seed=0)
        fit = fit_per_stop_rate_model(df, predictors=[])
        assert fit["rr_intercept"] == pytest.approx(
            np.exp(fit["coef_intercept"])
        )

    def test_bic_uses_log_likelihood_formula(self):
        """BIC should be on the same LLF scale as AIC/model comparisons."""
        df = _simulate_stops(n_stops=100, log_intercept=np.log(1.0), seed=0)
        fit = fit_per_stop_rate_model(df, predictors=[])
        expected = fit["n_params"] * np.log(fit["n_stops"]) - 2 * fit["llf"]
        assert fit["bic"] == pytest.approx(expected)


# ── Coefficient recovery with predictors ──


class TestPredictorCoefficientRecovery:
    def test_recovers_known_binary_coefficient(self):
        """When the truth is rate 1.0 with x=0 and rate 2.0 with x=1,
        the recovered β_1 should be ln(2) ≈ 0.693 within a few SE."""
        df = _simulate_stops(
            n_stops=600,
            log_intercept=0.0,
            coefs={"is_after_reward": np.log(2.0)},
            seed=11,
        )
        fit = fit_per_stop_rate_model(df, predictors=["is_after_reward"])
        assert fit is not None
        truth = np.log(2.0)
        z = (fit["coef_is_after_reward"] - truth) / fit["bse_is_after_reward"]
        assert abs(z) < 3, (
            f"is_after_reward coef {fit['coef_is_after_reward']:.3f} "
            f"(truth {truth:.3f}, z={z:.2f})"
        )
        # rate-ratio column matches
        assert fit["rr_is_after_reward"] == pytest.approx(
            np.exp(fit["coef_is_after_reward"])
        )

    def test_null_predictor_recovers_zero_coefficient(self):
        """When the truth has no predictor effect, β̂ should be
        consistent with zero (z within ±3)."""
        df = _simulate_stops(
            n_stops=600,
            log_intercept=np.log(1.5),
            coefs={"is_after_omission": 0.0},
            seed=23,
        )
        fit = fit_per_stop_rate_model(df, predictors=["is_after_omission"])
        assert fit is not None
        z = fit["coef_is_after_omission"] / fit["bse_is_after_omission"]
        assert abs(z) < 3, (
            f"null predictor recovered as significant: z={z:.2f}"
        )

    def test_recovers_two_predictors_simultaneously(self):
        """With two independent binary predictors, both coefficients
        should be recovered when fit jointly."""
        df = _simulate_stops(
            n_stops=800,
            log_intercept=0.0,
            coefs={
                "is_after_reward": np.log(2.0),
                "is_after_switch": np.log(0.5),
            },
            seed=7,
        )
        fit = fit_per_stop_rate_model(
            df, predictors=["is_after_reward", "is_after_switch"]
        )
        assert fit is not None
        for name, truth in [
            ("is_after_reward", np.log(2.0)),
            ("is_after_switch", np.log(0.5)),
        ]:
            z = (fit[f"coef_{name}"] - truth) / fit[f"bse_{name}"]
            assert abs(z) < 3.5, (
                f"{name}: coef {fit[f'coef_{name}']:.3f} (truth {truth:.3f}, z={z:.2f})"
            )


# ── Offset semantics ──


class TestImmobilityTimeOffset:
    def test_doubling_immobility_time_does_not_change_rate_estimate(self):
        """The offset is ``log(immobility_time)``: the model fits a
        rate per second, not a count. Doubling every stop's
        immobility_time should leave the recovered rate unchanged
        (and double the recovered raw counts)."""
        true_rate = 1.5
        df_a = _simulate_stops(
            n_stops=400, log_intercept=np.log(true_rate),
            duration_range=(10.0, 10.0), seed=0,
        )
        df_b = df_a.copy()
        df_b["immobility_time"] *= 2.0
        # Counts also need to double under the new "true" exposure for
        # the rate to stay the same; resimulate with the doubled time.
        df_b = _simulate_stops(
            n_stops=400, log_intercept=np.log(true_rate),
            duration_range=(20.0, 20.0), seed=0,
        )
        fit_a = fit_per_stop_rate_model(df_a, predictors=[])
        fit_b = fit_per_stop_rate_model(df_b, predictors=[])
        # Both should recover ~true_rate within tolerance.
        for fit in (fit_a, fit_b):
            z = (np.log(fit["rr_intercept"]) - np.log(true_rate)) / fit["bse_intercept"]
            assert abs(z) < 3

    def test_zero_duration_stops_excluded(self):
        """Stops with immobility_time == 0 would give log(0) = -inf as
        an offset and crash the GLM. They must be excluded silently."""
        df = _simulate_stops(n_stops=100, log_intercept=0.0, seed=0)
        df.loc[:9, "immobility_time"] = 0.0  # 10 zero-duration stops
        fit = fit_per_stop_rate_model(df, predictors=[])
        assert fit is not None
        assert fit["n_stops"] == 90  # 100 - 10 zero-duration

    def test_negative_immobility_time_excluded(self):
        """Negative offsets are excluded by the same ``> 0`` filter."""
        df = _simulate_stops(n_stops=100, log_intercept=0.0, seed=0)
        df.loc[:4, "immobility_time"] = -1.0
        fit = fit_per_stop_rate_model(df, predictors=[])
        assert fit["n_stops"] == 95


# ── Edge cases ──


class TestEdgeCases:
    def test_returns_none_below_min_stops(self):
        df = _simulate_stops(n_stops=15, log_intercept=0.0, seed=0)
        fit = fit_per_stop_rate_model(df, predictors=[], min_stops=20)
        assert fit is None

    def test_missing_predictor_column_returns_none(self):
        df = _simulate_stops(n_stops=100, log_intercept=0.0, seed=0)
        fit = fit_per_stop_rate_model(df, predictors=["nonexistent_column"])
        assert fit is None

    def test_constant_predictor_returns_none(self):
        """Regression: ``sm.add_constant`` is a no-op when the input
        already has a constant column, so the returned ``X`` has
        ``len(predictors)`` columns instead of
        ``len(predictors) + 1``. The post-fit
        ``zip(col_names, result.params, strict=True)`` then raises
        ``ValueError`` after the GLM has already been fit. The fix
        rejects rank-deficient designs (any predictor constant within
        the surviving rows) up front and returns ``None``."""
        df = _simulate_stops(n_stops=100, log_intercept=0.0, seed=0)
        # Force the predictor to be constant within the stop subset
        # that survives the offset > 0 filter.
        df["is_after_reward"] = 1.0
        fit = fit_per_stop_rate_model(df, predictors=["is_after_reward"])
        assert fit is None, (
            f"fit_per_stop_rate_model returned {fit!r} on constant predictor; "
            "should return None instead of crashing in zip(strict=True) "
            "after the GLM has already been fit."
        )

    def test_one_constant_one_varying_predictor_returns_none(self):
        """Mixed case: even if some predictors vary, a constant one
        in the same call still makes the design rank-deficient
        against the intercept and crashes the post-fit zip."""
        df = _simulate_stops(
            n_stops=120,
            log_intercept=0.0,
            coefs={"is_after_omission": 0.5},
            seed=0,
        )
        df["is_after_reward"] = 1.0
        fit = fit_per_stop_rate_model(
            df, predictors=["is_after_reward", "is_after_omission"]
        )
        assert fit is None

    def test_collinear_predictors_returns_none(self):
        """Regression: a per-column constancy check misses designs
        like ``x2 = 1 - x1`` — neither predictor is individually
        constant, but ``[1, x1, x2]`` is rank-2-not-3 and the GLM
        silently pseudo-inverts to return finite coefficients/p-values
        for an unidentified model. The full-rank check catches it."""
        df = _simulate_stops(n_stops=200, log_intercept=0.0, seed=0)
        rng = np.random.default_rng(1)
        df["x1"] = rng.integers(0, 2, len(df)).astype(float)
        df["x2"] = 1.0 - df["x1"].to_numpy()
        fit = fit_per_stop_rate_model(df, predictors=["x1", "x2"])
        assert fit is None, (
            f"fit_per_stop_rate_model returned {fit!r} on collinear "
            "predictors x1 + x2 = 1; the full-rank check should "
            "reject this before the GLM produces a non-unique "
            "solution."
        )

    def test_duplicated_predictors_returns_none(self):
        """Regression: identical predictor columns also trip the
        rank check (``[1, a, a]`` has rank 2 < 3)."""
        df = _simulate_stops(n_stops=200, log_intercept=0.0, seed=0)
        rng = np.random.default_rng(2)
        df["a"] = rng.integers(0, 2, len(df)).astype(float)
        df["b"] = df["a"].to_numpy().copy()
        fit = fit_per_stop_rate_model(df, predictors=["a", "b"])
        assert fit is None

    def test_independent_predictors_still_fit(self):
        """Sanity: the new rank check must NOT reject identifiable
        designs."""
        df = _simulate_stops(
            n_stops=200,
            log_intercept=0.0,
            coefs={"is_after_omission": 0.5, "is_after_reward": 0.3},
            seed=0,
        )
        fit = fit_per_stop_rate_model(
            df, predictors=["is_after_omission", "is_after_reward"]
        )
        assert fit is not None
        assert "coef_is_after_omission" in fit
        assert "coef_is_after_reward" in fit

    def test_missing_response_column_returns_none(self):
        df = _simulate_stops(n_stops=100, log_intercept=0.0, seed=0)
        df = df.drop(columns=["n_events"])
        fit = fit_per_stop_rate_model(df, predictors=[])
        assert fit is None

    def test_nan_rows_dropped_before_fit(self):
        """Rows with NaN in any required column are dropped (rather
        than crashing the GLM with 'invalid value' errors)."""
        df = _simulate_stops(
            n_stops=100, log_intercept=0.0,
            coefs={"is_after_reward": 0.0}, seed=0,
        )
        df.loc[:9, "is_after_reward"] = np.nan
        fit = fit_per_stop_rate_model(df, predictors=["is_after_reward"])
        assert fit is not None
        assert fit["n_stops"] == 90

    def test_n_events_total_matches_input_after_filtering(self):
        df = _simulate_stops(n_stops=200, log_intercept=0.0, seed=5)
        fit = fit_per_stop_rate_model(df, predictors=[])
        assert fit["n_events_total"] == int(df["n_events"].sum())


# ── Cross-validation ──


class TestCrossValidateRateModel:
    def test_returns_cv_keys(self):
        df = _simulate_stops(n_stops=300, log_intercept=0.0, seed=1)
        cv = cross_validate_per_stop_rate_model(df, predictors=[])
        assert cv is not None
        assert set(cv) >= {"cv_ll_per_stop_mean", "cv_ll_per_stop_std", "cv_n_folds"}
        assert cv["cv_n_folds"] == 5
        assert np.isfinite(cv["cv_ll_per_stop_mean"])

    def test_informative_predictor_improves_held_out_ll(self):
        # A strong true effect should give higher held-out LL than the null.
        df = _simulate_stops(
            n_stops=600, log_intercept=-1.0, coefs={"x": 1.5}, seed=2
        )
        null = cross_validate_per_stop_rate_model(df, predictors=[])
        full = cross_validate_per_stop_rate_model(df, predictors=["x"])
        assert null is not None and full is not None
        assert full["cv_ll_per_stop_mean"] > null["cv_ll_per_stop_mean"]

    def test_too_few_stops_returns_none(self):
        df = _simulate_stops(n_stops=8, log_intercept=0.0, seed=3)
        assert cross_validate_per_stop_rate_model(df, predictors=[]) is None

    def test_missing_column_returns_none(self):
        df = _simulate_stops(n_stops=200, log_intercept=0.0, seed=4)
        assert cross_validate_per_stop_rate_model(df, predictors=["not_a_column"]) is None
