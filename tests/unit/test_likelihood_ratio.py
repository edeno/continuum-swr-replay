"""Tests for the likelihood ratio test used in GLM model comparison.

This is the statistical test that determines whether neural activity is
modulated by theta phase or SWR events. Getting this wrong would invalidate
the paper's core findings about modulation.
"""

from __future__ import annotations

import pytest
import scipy.stats

from continuum_swr_replay.analysis.pfc.glm import likelihood_ratio_test


class TestLikelihoodRatioTest:
    """Test likelihood ratio test correctness."""

    def test_identical_models_give_zero_ratio(self):
        """Identical deviances should give zero likelihood ratio."""
        lr, p_val = likelihood_ratio_test(
            deviance_full=100.0,
            deviance_restricted=100.0,
            degrees_of_freedom_full=95,
            degrees_of_freedom_restricted=97,
        )

        assert lr == 0.0
        assert p_val == 1.0

    def test_better_full_model_gives_positive_ratio(self):
        """Full model with lower deviance should give positive ratio."""
        lr, p_val = likelihood_ratio_test(
            deviance_full=80.0,
            deviance_restricted=100.0,
            degrees_of_freedom_full=95,
            degrees_of_freedom_restricted=97,
        )

        assert lr == 20.0
        assert 0 <= p_val <= 1

    def test_known_p_value(self):
        """Compare against scipy.stats.chi2.sf directly."""
        lr, p_val = likelihood_ratio_test(
            deviance_full=90.0,
            deviance_restricted=100.0,
            degrees_of_freedom_full=95,
            degrees_of_freedom_restricted=98,
        )

        expected_lr = 10.0
        expected_df = 3
        expected_p = scipy.stats.chi2.sf(expected_lr, df=expected_df)

        assert lr == pytest.approx(expected_lr)
        assert p_val == pytest.approx(expected_p)

    def test_large_difference_gives_small_p_value(self):
        """Large deviance difference should give very small p-value."""
        lr, p_val = likelihood_ratio_test(
            deviance_full=50.0,
            deviance_restricted=100.0,
            degrees_of_freedom_full=95,
            degrees_of_freedom_restricted=97,
        )

        assert lr == 50.0
        assert p_val < 0.001

    def test_single_degree_of_freedom_difference(self):
        """Test with 1 df difference (common for adding one predictor)."""
        lr, p_val = likelihood_ratio_test(
            deviance_full=98.0,
            deviance_restricted=100.0,
            degrees_of_freedom_full=96,
            degrees_of_freedom_restricted=97,
        )

        expected_p = scipy.stats.chi2.sf(2.0, df=1)
        assert p_val == pytest.approx(expected_p)

    def test_degrees_of_freedom_computed_correctly(self):
        """df should be restricted - full (number of added parameters)."""
        # 3 extra parameters (e.g., intercept + cos + sin)
        lr, p_val = likelihood_ratio_test(
            deviance_full=90.0,
            deviance_restricted=100.0,
            degrees_of_freedom_full=94,
            degrees_of_freedom_restricted=97,
        )

        # Chi-square with df=3
        expected_p = scipy.stats.chi2.sf(10.0, df=3)
        assert p_val == pytest.approx(expected_p)
