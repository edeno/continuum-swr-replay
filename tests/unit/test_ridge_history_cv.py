"""The ridge cross-validated self vs self+cross history comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from continuum_swr_replay.analysis.replay.duration_history_simulation import (
    SCENARIOS,
    simulate_pooled_design,
)
from continuum_swr_replay.analysis.replay.model_specs import (
    MODEL_CROSS_HISTORY_DURATION_RATE,
    MODEL_CROSS_HISTORY_LONG_RATE,
)
from continuum_swr_replay.analysis.replay.ridge_history_cv import (
    assign_stop_folds,
    compare_cross_history,
    fit_ridge_poisson,
    ridge_design,
)

LADDER = MODEL_CROSS_HISTORY_LONG_RATE


@pytest.fixture(scope="module")
def cross_design() -> pd.DataFrame:
    """Non-SWR rate raised 5x 50 ms after an SWR-coupled event ends."""
    return simulate_pooled_design(
        SCENARIOS["d_cross_swr_to_nonswr"],
        response="no_swr",
        ladder=LADDER,
        n_sessions=3,
        n_stops=200,
        seed=11,
    )


@pytest.fixture(scope="module")
def null_design() -> pd.DataFrame:
    """Self-history only (an onset-locked rhythm); no cross dependence."""
    return simulate_pooled_design(
        SCENARIOS["c_onset_rhythm"],
        response="no_swr",
        ladder=LADDER,
        n_sessions=3,
        n_stops=200,
        seed=11,
    )


def test_the_long_pair_differs_from_the_shipped_rungs_only_in_the_cross_window():
    self_long, cross_long = LADDER.values()
    shipped = MODEL_CROSS_HISTORY_DURATION_RATE
    assert self_long.formula() == shipped["Mself_dur"].formula()

    def non_cross(spec):
        return [c for c in spec.design_columns() if not c.startswith("cross_")]

    assert non_cross(cross_long) == non_cross(shipped["Mselfcross_dur"])
    assert cross_long.history_lags() == 67
    assert cross_long.cross_history_basis == 6


def test_zero_penalty_reproduces_statsmodels(cross_design):
    spec = LADDER["Mselfcross_long_dur"]
    X, _, _ = ridge_design(cross_design, spec)
    y = cross_design["count"].to_numpy(dtype=float)
    offset = np.log(cross_design["exposure"].to_numpy(dtype=float))
    fit = fit_ridge_poisson(X, y, offset, np.zeros(X.shape[1]))
    ref = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset).fit()
    assert fit.converged
    np.testing.assert_allclose(fit.params, ref.params, atol=1e-6)


def test_the_penalised_fit_is_a_stationary_point(cross_design):
    spec = LADDER["Mselfcross_long_dur"]
    X, _, penalized = ridge_design(cross_design, spec)
    y = cross_design["count"].to_numpy(dtype=float)
    offset = np.log(cross_design["exposure"].to_numpy(dtype=float))
    lam = np.where(penalized, 4.0, 0.0)
    fit = fit_ridge_poisson(X, y, offset, lam)
    grad = X.T @ (y - np.exp(X @ fit.params + offset)) - lam * fit.params
    assert fit.converged
    assert np.max(np.abs(grad)) < 1e-5
    # Only the history columns are penalised.
    assert penalized.sum() == 4 + 4 + 6 + 6


def test_folds_are_whole_stops_and_every_session_is_in_every_fold(cross_design):
    folds = assign_stop_folds(cross_design, n_folds=5, seed=3)
    frame = cross_design[["session", "stop_id"]].assign(fold=folds)
    assert (frame.groupby("stop_id")["fold"].nunique() == 1).all()
    assert (frame.groupby("session")["fold"].nunique() == 5).all()
    np.testing.assert_array_equal(
        folds, assign_stop_folds(cross_design, n_folds=5, seed=3)
    )


def test_held_out_gain_finds_a_real_cross_effect_and_not_a_self_only_one(
    cross_design, null_design
):
    grid = (0.25, 1.0, 4.0)
    real = compare_cross_history(cross_design, LADDER, sigma_grid=grid, n_folds=3)
    null = compare_cross_history(null_design, LADDER, sigma_grid=grid, n_folds=3)
    assert real["gain_per_event"] > 0
    assert real["n_folds_positive"] >= 2
    assert null["gain_per_event"] < real["gain_per_event"] / 5
    # The end-locked cross kernel peaks where the truth does (50 ms).
    kernel = real["models"]["Mselfcross_long_dur"]["kernels"]["cross_offset"]["log_rr"]
    assert 2 <= int(np.argmax(kernel)) + 1 <= 5
