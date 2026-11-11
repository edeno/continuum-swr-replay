"""Threshold / event-inclusion robustness tests.

The real pipeline applies two filters before constructing arm counts:
- ``Non-Local_prob > 0.8`` (event detector threshold)
- ``head_speed < 4 cm/s`` (immobility)

Threshold choice changes which timebins enter the GLM. In simulation we
ask the closely-related questions:

1. **Noise contamination**: when "noise" events (uniform over arms, no
   signal) are mixed in with real signal events, how much does β̂
   attenuate? This proxies the effect of letting in spurious events
   from a too-permissive threshold.
2. **Minimum-count filter**: dropping low-L_e events should not bias
   β̂ — it should only inflate SE. Confirms the conditional Poisson
   handles event-size heterogeneity correctly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay.fitting import (
    fit_conditional_poisson,
)
from continuum_swr_replay.analysis.replay.simulation import (
    simulate_counts,
)


def _mix_signal_and_noise(beta_signal, n_signal, n_noise, L_e, seed):
    """Pool ``n_signal`` events generated under ``beta_signal`` with
    ``n_noise`` events generated under all-zero β (uniform over arms).
    All events have globally unique ``event_id``s."""
    sig = simulate_counts(beta_signal, n_events=n_signal, L_e=L_e, seed=seed)
    noise_beta = dict.fromkeys(beta_signal, 0.0)
    noi = simulate_counts(noise_beta, n_events=n_noise, L_e=L_e, seed=seed + 1_000_000)
    noi["event_id"] = noi["event_id"].astype(int) + n_signal
    return pd.concat([sig, noi], ignore_index=True)


class TestNoiseContamination:
    """Adding pure-noise events attenuates β̂ toward 0 — the inferential
    cost of a too-permissive event detector."""

    def test_attenuation_increases_with_noise_fraction(self):
        beta = {"is_current_arm": 1.0}
        df_clean = simulate_counts(beta, n_events=400, L_e=20, seed=0)
        df_50pct_noise = _mix_signal_and_noise(beta, 400, 400, L_e=20, seed=0)
        df_75pct_noise = _mix_signal_and_noise(beta, 400, 1200, L_e=20, seed=0)

        fit_clean = fit_conditional_poisson(df_clean, predictors=list(beta))
        fit_50 = fit_conditional_poisson(df_50pct_noise, predictors=list(beta))
        fit_75 = fit_conditional_poisson(df_75pct_noise, predictors=list(beta))

        # Monotonic attenuation
        assert (
            fit_clean["coef_is_current_arm"]
            > fit_50["coef_is_current_arm"]
            > fit_75["coef_is_current_arm"]
            > 0
        ), (
            f"clean={fit_clean['coef_is_current_arm']:.3f}, "
            f"50%={fit_50['coef_is_current_arm']:.3f}, "
            f"75%={fit_75['coef_is_current_arm']:.3f}"
        )

    def test_pure_noise_recovers_zero(self):
        """Sanity: with 100% noise, fitted β should be indistinguishable
        from 0 (this is also the null calibration check we already have
        in the simulation suite, repeated here for the threshold story)."""
        beta = {"is_current_arm": 0.0, "is_prev_arm": 0.0}
        df = simulate_counts(beta, n_events=600, L_e=20, seed=1)
        fit = fit_conditional_poisson(df, predictors=list(beta))
        assert fit is not None
        for p in beta:
            z = fit[f"coef_{p}"] / fit[f"bse_{p}"]
            assert abs(z) < 3, f"{p} z={z:.2f} under all-noise input"


class TestEventSizeFilter:
    """Dropping low-L_e events should NOT bias β̂. The conditional
    likelihood weights events by their information content automatically
    — small events contribute less, but they don't bias the estimate."""

    def test_dropping_short_events_does_not_bias(self):
        """Generate events with mixed L_e (some short, some long), then
        drop the short ones. β̂ from the filtered fit should be close to
        β̂ from the full fit."""
        beta = {"is_current_arm": 0.7}
        rng = np.random.default_rng(13)
        n_events = 600
        # Mix of short and long events
        L_e = rng.choice([5, 30], size=n_events, p=[0.5, 0.5])
        df_full = simulate_counts(beta, n_events=n_events, L_e=L_e, seed=13)

        # Keep only events with total_count > 10 (drops the short ones)
        totals = df_full.groupby("event_id")["arm_count_hard"].sum()
        kept_events = totals[totals > 10].index
        df_filtered = df_full[df_full["event_id"].isin(kept_events)].copy()

        fit_full = fit_conditional_poisson(df_full, predictors=list(beta))
        fit_filtered = fit_conditional_poisson(df_filtered, predictors=list(beta))
        assert fit_full is not None and fit_filtered is not None

        # Both should be close to truth, and to each other.
        z_diff = (
            fit_full["coef_is_current_arm"] - fit_filtered["coef_is_current_arm"]
        ) / np.hypot(fit_full["bse_is_current_arm"], fit_filtered["bse_is_current_arm"])
        assert abs(z_diff) < 3, (
            f"filter changed coef by {z_diff:.2f} σ: "
            f"full={fit_full['coef_is_current_arm']:.3f}, "
            f"filtered={fit_filtered['coef_is_current_arm']:.3f}"
        )

    def test_filtering_only_inflates_se(self):
        """Filtering reduces information, so SE should grow (not shrink)."""
        beta = {"is_current_arm": 0.7}
        n_events = 600
        L_e = np.full(n_events, 20)
        df_full = simulate_counts(beta, n_events=n_events, L_e=L_e, seed=29)

        # Keep first half — equivalent to a permissive vs strict threshold
        kept = df_full["event_id"] < n_events // 2
        df_strict = df_full[kept].copy()

        fit_full = fit_conditional_poisson(df_full, predictors=list(beta))
        fit_strict = fit_conditional_poisson(df_strict, predictors=list(beta))
        assert fit_strict["bse_is_current_arm"] > fit_full["bse_is_current_arm"]


class TestSweepThresholdSensitivityCurve:
    """Smoke test: produce the attenuation curve as noise fraction grows.
    Useful to keep around as a self-documenting witness that the test
    suite measures the right quantity."""

    def test_curve_is_monotonic(self):
        beta = {"is_current_arm": 1.0}
        n_signal = 400
        L_e = 20
        coefs = []
        for n_noise in [0, 200, 400, 800, 1600]:
            if n_noise == 0:
                df = simulate_counts(beta, n_events=n_signal, L_e=L_e, seed=2)
            else:
                df = _mix_signal_and_noise(beta, n_signal, n_noise, L_e=L_e, seed=2)
            fit = fit_conditional_poisson(df, predictors=list(beta))
            coefs.append(fit["coef_is_current_arm"])
        # Curve should be monotonically decreasing.
        diffs = np.diff(coefs)
        assert (diffs <= 0).all(), (
            f"Attenuation curve should be monotonic; got coefs = {coefs}"
        )
        # And at the extreme of 80% noise we should still get a positive
        # — but materially attenuated — coefficient.
        assert 0 < coefs[-1] < 0.5 * coefs[0]
