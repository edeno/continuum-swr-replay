"""Invariant and correctness tests for analysis/replay/uncertainty.py.

Checks the three per-event decoder-confidence summaries against known
limiting cases (delta posterior, uniform posterior) and the HPD-width
ordering required by the plan:
``posterior_hpd_width_50 ≤ _80 ≤ _95``.
"""

from __future__ import annotations

import numpy as np
import pytest

from continuum_swr_replay.analysis.replay.uncertainty import (
    posterior_entropy_mean,
    posterior_hpd_width,
    posterior_hpd_widths,
    posterior_peak_mean,
)


def _delta_posterior(T: int, n_arms: int, arm: int) -> np.ndarray:
    """Posterior fully concentrated on ``arm`` at every timebin."""
    p = np.zeros((T, n_arms))
    p[:, arm] = 1.0
    return p


def _uniform_posterior(T: int, n_arms: int) -> np.ndarray:
    return np.full((T, n_arms), 1.0 / n_arms)


# ── posterior_peak_mean ──


class TestPosteriorPeakMean:
    def test_delta_posterior_peak_is_one(self):
        arm_post = _delta_posterior(T=5, n_arms=9, arm=3)
        w = np.array([0.9, 0.85, 0.95, 0.81, 0.9])
        event_id = np.array([0, 0, 0, 0, 0])
        _, peak = posterior_peak_mean(arm_post, w, event_id)
        np.testing.assert_allclose(peak, 1.0)

    def test_uniform_posterior_peak_is_one_over_n_arms(self):
        arm_post = _uniform_posterior(T=4, n_arms=9)
        w = np.ones(4)
        event_id = np.array([0, 0, 0, 0])
        _, peak = posterior_peak_mean(arm_post, w, event_id)
        np.testing.assert_allclose(peak, 1 / 9)

    def test_range(self):
        """Peak always in [1/n_arms, 1]."""
        rng = np.random.default_rng(0)
        T, n_arms = 50, 9
        arm_post = rng.dirichlet(np.ones(n_arms), size=T)
        w = rng.uniform(0.5, 1.0, size=T)
        event_id = rng.integers(0, 5, size=T)
        _, peak = posterior_peak_mean(arm_post, w, event_id)
        assert np.all(peak >= 1 / n_arms - 1e-9)
        assert np.all(peak <= 1.0 + 1e-9)


# ── posterior_entropy_mean ──


class TestPosteriorEntropyMean:
    def test_delta_posterior_entropy_is_zero(self):
        arm_post = _delta_posterior(T=5, n_arms=9, arm=2)
        w = np.array([0.9, 0.85, 0.95, 0.81, 0.9])
        event_id = np.zeros(5, dtype=int)
        _, entropy = posterior_entropy_mean(arm_post, w, event_id)
        np.testing.assert_allclose(entropy, 0.0, atol=1e-12)

    def test_uniform_posterior_entropy_is_log_n(self):
        n_arms = 9
        arm_post = _uniform_posterior(T=4, n_arms=n_arms)
        w = np.ones(4)
        event_id = np.zeros(4, dtype=int)
        _, entropy = posterior_entropy_mean(arm_post, w, event_id)
        np.testing.assert_allclose(entropy, np.log(n_arms))

    def test_entropy_nonnegative(self):
        rng = np.random.default_rng(1)
        arm_post = rng.dirichlet(np.ones(9), size=30)
        w = rng.uniform(0.5, 1.0, size=30)
        event_id = rng.integers(0, 4, size=30)
        _, entropy = posterior_entropy_mean(arm_post, w, event_id)
        assert np.all(entropy >= -1e-12)


# ── posterior_hpd_width ──


class TestPosteriorHpdWidth:
    def test_delta_posterior_width_is_one(self):
        arm_post = _delta_posterior(T=5, n_arms=9, arm=4)
        w = np.array([0.9, 0.85, 0.95, 0.81, 0.9])
        event_id = np.zeros(5, dtype=int)
        for cov in (0.5, 0.8, 0.95):
            _, width = posterior_hpd_width(arm_post, w, event_id, coverage=cov)
            np.testing.assert_allclose(width, 1.0)

    def test_uniform_posterior_width_is_n_arms_at_all_coverages(self):
        # Tie-inclusive HPD (matches non_local_detector's definition):
        # every arm has the same probability under a uniform posterior,
        # so every arm lies at or above the density threshold at every
        # coverage — the HPD set is always all 9 arms.
        n_arms = 9
        arm_post = _uniform_posterior(T=4, n_arms=n_arms)
        w = np.ones(4)
        event_id = np.zeros(4, dtype=int)
        for cov in (0.5, 0.8, 0.95):
            _, width = posterior_hpd_width(arm_post, w, event_id, coverage=cov)
            np.testing.assert_allclose(width, n_arms)

    def test_hpd_ordering_invariant(self):
        """Plan requirement: width_50 ≤ width_80 ≤ width_95 per event."""
        rng = np.random.default_rng(7)
        T, n_arms = 60, 9
        arm_post = rng.dirichlet(np.ones(n_arms) * 0.5, size=T)
        w = rng.uniform(0.3, 1.0, size=T)
        event_id = rng.integers(0, 8, size=T)
        _, widths = posterior_hpd_widths(
            arm_post, w, event_id, coverages=(0.5, 0.8, 0.95)
        )
        assert np.all(widths[0.5] <= widths[0.8] + 1e-12)
        assert np.all(widths[0.8] <= widths[0.95] + 1e-12)

    def test_invalid_coverage_raises(self):
        arm_post = _uniform_posterior(T=2, n_arms=3)
        w = np.ones(2)
        event_id = np.zeros(2, dtype=int)
        with pytest.raises(ValueError, match="coverage must be in"):
            posterior_hpd_width(arm_post, w, event_id, coverage=0.0)
        with pytest.raises(ValueError, match="coverage must be in"):
            posterior_hpd_width(arm_post, w, event_id, coverage=1.5)


# ── Two-event mix ──


class TestMultipleEvents:
    def test_event_ordering_matches_sorted_unique(self):
        arm_post = _delta_posterior(T=4, n_arms=3, arm=0)
        w = np.ones(4)
        event_id = np.array([10, 10, 5, 5])
        event_ids, peak = posterior_peak_mean(arm_post, w, event_id)
        np.testing.assert_array_equal(event_ids, [5, 10])
        np.testing.assert_allclose(peak, [1.0, 1.0])
