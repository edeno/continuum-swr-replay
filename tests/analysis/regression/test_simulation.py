"""The shared fitter's intervals and tests on simulated rate data.

Each data set is 350 synthetic stops in 15 ms bins (lognormal durations,
median 1 s after a reward and 0.3 s after none, half rewarded) with Poisson
events at a log rate ``f(s) + b * rewarded`` over the time ``s`` since the stop
began, ``s`` below 4 s, about 300 events in all. The counts and exposures are
summed over bins in the same 15 ms of ``s`` and the same reward, which leaves
the likelihood unchanged up to a constant. The model is an intercept, the 0/1
reward and a cubic P-spline of ``s`` on 15 basis functions over [0, 4] with
its penalty chosen by REML.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.stats

from continuum_swr_replay.analysis.regression.likelihoods import Poisson
from continuum_swr_replay.analysis.regression.pspline import (
    PSplineBasis,
    sum_to_zero_constraint,
)
from continuum_swr_replay.analysis.regression.reml import (
    PenaltyBlock,
    SessionData,
    select_penalties,
)

BIN_S = 0.015
CLOCK_MAX_S = 4.0
N_STOPS = 350
N_EVENTS = 300
N_DATA_SETS = 1000
N_BASIS = 15
LOG_RATE_RATIO = np.log(1.3)
#: Truths of the log rate over the clock (up to a constant): a fourfold rise
#: over the first second, and none.
SHAPES = {
    "rise": lambda s: np.log(4.0) * (1.0 - np.exp(-s / 0.5)),
    "flat": lambda s: np.zeros_like(s),
}


def _simulate(
    shape: str, log_rate_ratio: float, rng: np.random.Generator
) -> tuple[SessionData, np.ndarray]:
    """One data set's summed cells, their design (intercept, reward, smooth)
    and the smooth's penalty, shape ``(N_BASIS - 1, N_BASIS - 1)``."""
    rewarded = rng.random(N_STOPS) < 0.5
    duration = np.where(rewarded, 1.0, 0.3) * np.exp(rng.normal(size=N_STOPS))
    duration = np.maximum(duration, BIN_S)
    n_bins = np.ceil(duration / BIN_S).astype(int)
    stop = np.repeat(np.arange(N_STOPS), n_bins)
    k = np.arange(n_bins.sum()) - np.repeat(np.cumsum(n_bins) - n_bins, n_bins)
    start = k * BIN_S
    end = np.minimum(start + BIN_S, duration[stop])
    s, exposure, reward = 0.5 * (start + end), end - start, rewarded[stop]
    keep = s < CLOCK_MAX_S
    s, exposure, reward, k = s[keep], exposure[keep], reward[keep], k[keep]
    expected = exposure * np.exp(SHAPES[shape](s) + log_rate_ratio * reward)
    y = rng.poisson(N_EVENTS * expected / expected.sum())

    _, cell = np.unique(2 * k + reward, return_inverse=True)
    cell_y = np.bincount(cell, weights=y)
    cell_exposure = np.bincount(cell, weights=exposure)
    cell_s = np.bincount(cell, weights=s * exposure) / cell_exposure
    cell_reward = np.bincount(cell, weights=reward) / np.bincount(cell)

    basis = PSplineBasis(0.0, CLOCK_MAX_S, N_BASIS)
    B = basis.design(cell_s)
    Z = sum_to_zero_constraint(B)
    X = np.column_stack([np.ones(cell_s.size), cell_reward, B @ Z])
    return SessionData(X, cell_y, np.log(cell_exposure)), Z.T @ basis.penalty() @ Z


def _fits(shape: str, log_rate_ratio: float, seed: int) -> np.ndarray:
    """The reward coefficient, its SE and the smooth's edf over the data sets.

    Returns
    -------
    np.ndarray, shape (N_DATA_SETS, 3)
    """
    out = np.empty((N_DATA_SETS, 3))
    for i, child in enumerate(np.random.SeedSequence(seed).spawn(N_DATA_SETS)):
        data, penalty = _simulate(shape, log_rate_ratio, np.random.default_rng(child))
        blocks = [PenaltyBlock(np.arange(2, 2 + penalty.shape[0]), penalty)]
        result = select_penalties(Poisson(), [data], blocks)
        assert result.search.converged
        session = result.sessions[0]
        out[i] = session.fit.beta[1], np.sqrt(session.covariance[1, 1]), session.edf[0]
    return out


@pytest.mark.slow
def test_coverage_and_size() -> None:
    """The 95 % interval of the reward coefficient covers the planted 1.3-fold
    effect in 93-97 % of 1,000 data sets with the rise truth; with a flat truth and
    no reward effect the coefficient is significant at 5 % in 3-7 %, and the
    smooth shrinks to a line (edf below 2.5) in at least 90 %."""
    z = scipy.stats.norm.ppf(0.975)
    rise = _fits("rise", LOG_RATE_RATIO, seed=20261120)
    flat = _fits("flat", 0.0, seed=20261121)

    coverage = np.mean(np.abs(rise[:, 0] - LOG_RATE_RATIO) <= z * rise[:, 1])
    size = np.mean(np.abs(flat[:, 0]) > z * flat[:, 1])
    shrunk = np.mean(flat[:, 2] < 2.5)
    print(f"coverage {coverage:.3f}, size {size:.3f}, flat edf < 2.5 {shrunk:.3f}")

    assert 0.93 <= coverage <= 0.97
    assert 0.03 <= size <= 0.07
    assert shrunk >= 0.90
