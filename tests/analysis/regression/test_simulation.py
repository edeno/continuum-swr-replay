"""The shared fitter's intervals and tests on simulated rate data.

Each data set is the mgcv comparison's stop bins (``stop_bins``: 350 synthetic
stops in 15 ms bins, lognormal durations, median 1 s after a reward and 0.3 s
after none, half rewarded) with Poisson events at a log rate
``f(s) + b * rewarded`` over the time ``s`` since the stop began, ``s`` below
4 s, about 300 events in all. The counts and exposures are
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
from continuum_swr_replay.analysis.regression.pspline import PSplineBasis
from continuum_swr_replay.analysis.regression.reml import (
    PenaltyBlock,
    SessionData,
    select_penalties,
)
from scripts.checks.compare_regression_to_mgcv import HI as CLOCK_MAX_S
from scripts.checks.compare_regression_to_mgcv import stop_bins

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
) -> tuple[SessionData, np.ndarray, np.ndarray]:
    """One data set's summed cells, their design (intercept, reward, smooth),
    the smooth's penalty, shape ``(N_BASIS - 1, N_BASIS - 1)``, and the
    cells' clock, shape ``(n_cells,)``."""
    bins = stop_bins(rng)
    s, exposure = bins["s"].to_numpy(), bins["expo"].to_numpy()
    reward, k = bins["rew"].to_numpy(), bins["k"].to_numpy()
    expected = exposure * np.exp(SHAPES[shape](s) + log_rate_ratio * reward)
    y = rng.poisson(N_EVENTS * expected / expected.sum())

    _, cell = np.unique(2 * k + reward, return_inverse=True)
    cell_y = np.bincount(cell, weights=y)
    cell_exposure = np.bincount(cell, weights=exposure)
    cell_s = np.bincount(cell, weights=s * exposure) / cell_exposure
    cell_reward = np.bincount(cell, weights=reward) / np.bincount(cell)

    design, penalty, _ = PSplineBasis(0.0, CLOCK_MAX_S, N_BASIS).constrained(cell_s)
    X = np.column_stack([np.ones(cell_s.size), cell_reward, design])
    data = SessionData(X, cell_y, np.log(cell_exposure))
    return data, penalty, cell_s


def _fits(shape: str, log_rate_ratio: float, seed: int) -> np.ndarray:
    """Per data set: the reward coefficient, its SE, the smooth's edf, and the
    smooth's across-the-function coverage.

    The across-the-function coverage (Wood's) is the share of the cells at
    which the smooth's 95 % Bayesian interval, ``f_hat +- 1.96 se`` from the
    covariance ``A^-1``, covers the truth. The fitted smooth sums to zero over
    the cells (its constraint), so the truth it estimates is ``f(s)`` less its
    mean over the cells.

    Returns
    -------
    np.ndarray, shape (N_DATA_SETS, 4)
    """
    z = scipy.stats.norm.ppf(0.975)
    out = np.empty((N_DATA_SETS, 4))
    for i, child in enumerate(np.random.SeedSequence(seed).spawn(N_DATA_SETS)):
        data, penalty, clock = _simulate(
            shape, log_rate_ratio, np.random.default_rng(child)
        )
        columns = np.arange(2, 2 + penalty.shape[0])
        blocks = [PenaltyBlock(columns, penalty)]
        result = select_penalties(Poisson(), [data], blocks)
        session = result.sessions[0]
        C = data.X[:, columns]
        curve = C @ session.fit.beta[columns]
        covariance = session.covariance[np.ix_(columns, columns)]
        se = np.sqrt(np.einsum("ij,jk,ik->i", C, covariance, C))
        truth = SHAPES[shape](clock)
        covered = np.abs(curve - (truth - truth.mean())) <= z * se
        out[i] = (
            session.fit.beta[1],
            np.sqrt(session.covariance[1, 1]),
            session.edf[0],
            covered.mean(),
        )
    return out


@pytest.mark.slow
def test_coverage_and_size() -> None:
    """The 95 % interval of the reward coefficient covers the planted 1.3-fold
    effect in 93-97 % of 1,000 data sets with the rise truth, and the smooth's
    95 % Bayesian interval covers the centered rise at 92-98 % of the cells on
    average (Wood's across-the-function coverage); with a flat truth and no
    reward effect the coefficient is significant at 5 % in 3-7 %, and the
    smooth shrinks to a line (edf below 2.5) in at least 90 %."""
    z = scipy.stats.norm.ppf(0.975)
    rise = _fits("rise", LOG_RATE_RATIO, seed=20261120)
    flat = _fits("flat", 0.0, seed=20261121)

    coverage = np.mean(np.abs(rise[:, 0] - LOG_RATE_RATIO) <= z * rise[:, 1])
    size = np.mean(np.abs(flat[:, 0]) > z * flat[:, 1])
    shrunk = np.mean(flat[:, 2] < 2.5)
    across = np.mean(rise[:, 3])
    print(
        f"coverage {coverage:.3f}, size {size:.3f}, flat edf < 2.5 {shrunk:.3f}, "
        f"across-the-function coverage {across:.3f}"
    )

    assert 0.93 <= coverage <= 0.97
    assert 0.03 <= size <= 0.07
    assert shrunk >= 0.90
    assert 0.92 <= across <= 0.98
