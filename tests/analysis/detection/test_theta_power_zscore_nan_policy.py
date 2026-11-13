"""Regression test for the theta-power z-score NaN contract.

Both data-aggregation paths (HPC + ContFrag) feed
``_nearest``-interpolated theta power into ``scipy.stats.zscore``.
The default ``nan_policy="propagate"`` returns all-NaN whenever any
input value is NaN — silently suppressing theta event classification
downstream in :func:`quality_metrics.compute_non_local_event_context`.

Both paths route through the shared
:func:`continuum_swr_replay.analysis.detection.nonlocal_events.theta_power_zscore`
helper, so this test exercises the helper's behavior directly. A
regression that drops the kwarg, or replaces the helper, fails the
behavioral assertion below — not a source-grep that could pass while
the runtime path still produced NaN.
"""

from __future__ import annotations

import numpy as np

from continuum_swr_replay.analysis.detection.nonlocal_events import (
    theta_power_zscore,
)


class TestThetaPowerZScore:
    def test_finite_inputs_match_plain_zscore(self):
        """When no NaN is present the helper matches a plain z-score."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = theta_power_zscore(x)
        np.testing.assert_allclose(out, (x - x.mean()) / x.std(), atol=1e-12)

    def test_partial_nan_input_keeps_finite_z_scores(self):
        """The user's hazard: a single NaN in the trace must not zero
        out the entire output."""
        out = theta_power_zscore(np.array([1.0, 2.0, np.nan, 4.0]))
        finite = ~np.isnan(out)
        assert finite.sum() == 3, (
            f"Expected 3 finite z-scores, got {out.tolist()}. The "
            "default scipy.stats.zscore nan_policy='propagate' "
            "produces all-NaN; the helper must keep nan_policy='omit'."
        )
        # The three finite z-scores are computed against the
        # NaN-omitted mean/std, so they sum to ~0 and have unit
        # population std.
        assert np.isclose(out[finite].mean(), 0.0, atol=1e-12)

    def test_all_nan_input_returns_all_nan(self):
        """Sanity: with no finite samples there's nothing to z-score
        against, so NaN propagates."""
        out = theta_power_zscore(np.full(5, np.nan))
        assert np.all(np.isnan(out))

    def test_no_finite_values_when_only_one_finite_sample(self):
        """Single-finite-sample case: variance is undefined, so the
        finite slot must NOT carry an informative z-score (it would
        be a divide-by-zero or 0/0 = NaN). The originally-NaN slots
        stay NaN.

        Scipy's exact value at the lone finite slot has varied across
        versions (NaN under ddof=1, 0.0 under ddof=0 on some
        releases). The contract downstream is: nothing here should
        be a finite, informative z-score that would fool the
        ``compute_non_local_event_context`` theta classifier into
        treating a single sample as a real distribution.
        """
        out = theta_power_zscore(np.array([np.nan, 3.0, np.nan, np.nan]))
        assert np.isnan(out[0]) and np.isnan(out[2]) and np.isnan(out[3])
        assert not np.isfinite(out[1]), (
            f"single-finite-sample slot returned a finite z-score "
            f"({out[1]}); variance is undefined here, so downstream "
            "code that thresholds on ``|z| > k`` must not see a "
            "real number."
        )

    def test_returns_ndarray(self):
        """Helper returns a plain ndarray — downstream xarray/pandas
        wrapping is the caller's responsibility."""
        out = theta_power_zscore(np.array([1.0, 2.0, 3.0]))
        assert isinstance(out, np.ndarray)
