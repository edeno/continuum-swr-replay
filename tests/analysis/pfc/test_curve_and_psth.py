"""One unit's kernel with its pointwise standard error, and its rate around events,
on inputs small enough to check by hand."""

from __future__ import annotations

import numpy as np

from continuum_swr_replay.analysis.pfc.modulation import (
    curve_standard_error,
    peri_event_rate,
)


def test_the_standard_error_takes_the_off_diagonal_covariance():
    # Two basis functions; lag 0 reads the first, lag 1 their sum, lag 2 their
    # difference. With variances 4 and 1 and covariance 1.5:
    #   lag 0: 4;  lag 1: 4 + 1 + 2*1.5 = 8;  lag 2: 4 + 1 - 2*1.5 = 2.
    basis = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, -1.0]])
    params = np.array([0.5, -0.25])
    covariance = np.array([[4.0, 1.5], [1.5, 1.0]])

    curve, standard_error = curve_standard_error(basis, params, covariance)

    np.testing.assert_allclose(curve, [0.5, 0.25, 0.75])
    np.testing.assert_allclose(standard_error, np.sqrt([4.0, 8.0, 2.0]))


def test_a_variance_rounded_below_zero_is_a_standard_error_of_zero():
    basis = np.array([[1.0, -1.0]])
    covariance = np.array([[1.0, 1.0 + 1e-12], [1.0 + 1e-12, 1.0]])

    _, standard_error = curve_standard_error(basis, np.zeros(2), covariance)

    assert standard_error[0] == 0.0


def test_the_rate_counts_each_spike_at_its_lag_from_every_event_near_it():
    # Bins of 0.125 s over lags -0.25 to 0.25 (times exact in binary). Event at
    # 1.0: spikes at 0.8125 (lag -0.1875), 1.0625 (0.0625) and 1.25 (0.25, the
    # last lag, included). Event at 1.5: the spike at 1.25 again (-0.25, the
    # first lag, included) and 1.6875 (0.1875). The spike at 3.0 is near none.
    spikes = np.array([0.8125, 1.0625, 1.25, 1.6875, 3.0])

    centers, rate = peri_event_rate(spikes, np.array([1.0, 1.5]), (-0.25, 0.25), 0.125)

    np.testing.assert_allclose(centers, [-0.1875, -0.0625, 0.0625, 0.1875])
    # counts 2, 0, 1, 2 over 2 events of 0.125 s
    np.testing.assert_allclose(rate, [8.0, 0.0, 4.0, 8.0])
