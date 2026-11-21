"""The penalized Newton fit against hand cases and the content fitter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.optimize

from continuum_swr_replay.analysis.regression.constants import NEWTON_DECREMENT_TOL
from continuum_swr_replay.analysis.regression.likelihoods import (
    Bernoulli,
    GroupedSoftmax,
    Poisson,
)
from continuum_swr_replay.analysis.regression.newton import NewtonFit, penalized_newton
from continuum_swr_replay.analysis.replay_models.fitting import fit_conditional_poisson


def test_hand_case_poisson_two_parameters() -> None:
    """Intercept + one 0/1 covariate: the log rates of the two groups."""
    x = np.array([0, 0, 0, 0, 1, 1, 1], dtype=float)
    exposure = np.array([1.0, 2.0, 0.5, 1.5, 1.0, 3.0, 2.0])
    y = np.array([2, 3, 0, 4, 5, 9, 7], dtype=float)
    X = np.column_stack([np.ones_like(x), x])

    fit = penalized_newton(
        Poisson(), X, y, np.log(exposure), np.zeros((2, 2)), np.zeros(2)
    )

    rate_0 = y[x == 0].sum() / exposure[x == 0].sum()  # 9 / 5
    rate_1 = y[x == 1].sum() / exposure[x == 1].sum()  # 21 / 6
    assert isinstance(fit, NewtonFit)
    assert fit.converged
    np.testing.assert_allclose(
        fit.beta, [np.log(rate_0), np.log(rate_1 / rate_0)], rtol=0, atol=1e-10
    )
    # The returned A is the information at the optimum: diag of the expected counts.
    mu = exposure * np.exp(X @ fit.beta)
    np.testing.assert_allclose(fit.A, (X.T * mu) @ X, rtol=1e-12)
    expected_loglik = y @ np.log(mu) - mu.sum()
    assert fit.loglik == pytest.approx(expected_loglik, rel=1e-12)


def test_hand_case_ridge_one_parameter() -> None:
    """One coefficient with a quadratic penalty: the root of its stationarity.

    ``sum_i x_i (y_i - e_i exp(x_i b)) - lam b = 0``.
    """
    x = np.array([0.5, -1.0, 2.0, 1.0, 0.0, 1.5])
    exposure = np.array([1.0, 2.0, 1.0, 0.5, 1.0, 2.0])
    y = np.array([1, 0, 6, 2, 1, 5], dtype=float)
    lam = 3.0

    def stationarity(b: float) -> float:
        return float(x @ (y - exposure * np.exp(x * b)) - lam * b)

    root = scipy.optimize.brentq(stationarity, -5.0, 5.0, xtol=1e-14, rtol=1e-15)
    fit = penalized_newton(
        Poisson(), x[:, None], y, np.log(exposure), np.array([[lam]]), np.zeros(1)
    )

    assert fit.converged
    assert fit.beta[0] == pytest.approx(root, abs=1e-10)
    # A = H + S at the optimum.
    hessian = (x**2 * exposure * np.exp(x * fit.beta[0])).sum()
    assert fit.A[0, 0] == pytest.approx(hessian + lam, rel=1e-12)


def _fractional_content_data(n_events: int = 80, n_arms: int = 9) -> pd.DataFrame:
    """Events with a share on each arm, a continuous covariate varying within
    events, and an arm indicator; seeded."""
    rng = np.random.default_rng(3)
    event_id = np.repeat(np.arange(n_events), n_arms)
    arm = np.tile(np.arange(n_arms), n_events)
    x = rng.normal(size=n_events * n_arms)
    is_arm_2 = (arm == 2).astype(float)
    eta = 0.8 * x - 0.6 * is_arm_2
    weight = np.exp(eta).reshape(n_events, n_arms)
    # Noisy shares: a Dirichlet around the softmax probabilities.
    probability = weight / weight.sum(axis=1, keepdims=True)
    shares = np.vstack([rng.dirichlet(20.0 * p) for p in probability])
    frame = pd.DataFrame(
        {
            "event_id": event_id,
            "arm_fraction": shares.ravel(),
            "x": x,
            "is_arm_2": is_arm_2,
        }
    )
    # Rows out of event order, as a table may hold them.
    return frame.sample(frac=1.0, random_state=0).reset_index(drop=True)


def test_softmax_equals_conditional_poisson() -> None:
    """The grouped softmax on event shares is the content fitter's likelihood."""
    data = _fractional_content_data()
    predictors = ["x", "is_arm_2"]
    reference = fit_conditional_poisson(
        data, predictors, response_col="arm_fraction", event_col="event_id"
    )
    assert reference is not None and reference["converged"]

    ordered = data.sort_values("event_id", kind="stable")
    X = ordered[predictors].to_numpy(float)
    fit = penalized_newton(
        GroupedSoftmax(),
        X,
        ordered["arm_fraction"].to_numpy(float),
        ordered["event_id"].to_numpy(),
        np.zeros((2, 2)),
        np.zeros(2),
    )

    assert fit.converged
    expected = np.array([reference[f"coef_{name}"] for name in predictors])
    np.testing.assert_allclose(fit.beta, expected, rtol=1e-6)
    assert fit.loglik == pytest.approx(reference["llf"], rel=1e-9)
    # The model SEs are the inverse information at the optimum, as there.
    se = np.sqrt(np.diag(np.linalg.inv(fit.A)))
    expected_se = np.array([reference[f"bse_{name}"] for name in predictors])
    np.testing.assert_allclose(se, expected_se, rtol=1e-6)


def _logistic_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(5)
    x = rng.normal(size=200)
    X = np.column_stack([np.ones_like(x), x])
    y = (rng.uniform(size=x.size) < 1 / (1 + np.exp(-(0.3 + 1.2 * x)))).astype(float)
    return X, y


def _decrement(lik, X, y, aux, S, beta) -> float:
    eta = X @ beta
    g = X.T @ lik.score_eta(eta, y, aux) - S @ beta
    A = lik.information(X, eta, y, aux) + S
    return 0.5 * float(g @ np.linalg.solve(A, g))


def test_decrement_rule_and_halving() -> None:
    """The fit stops on the decrement; a saturated start converges by the cap."""
    X, y = _logistic_data()
    S = np.diag([0.0, 0.5])
    from_zero = penalized_newton(Bernoulli(), X, y, None, S, np.zeros(2))
    assert from_zero.converged
    assert from_zero.newton_decrement < NEWTON_DECREMENT_TOL
    # The recorded decrement is the one at the returned coefficients.
    assert from_zero.newton_decrement == pytest.approx(
        _decrement(Bernoulli(), X, y, None, S, from_zero.beta), rel=1e-6, abs=1e-15
    )

    # From (200, -300) the probabilities saturate: the information is tiny
    # and the raw step about 1e4 long. Capped at 10 and halved, the steps
    # bring the fit back to the same optimum.
    start = np.array([200.0, -300.0])
    saturated = penalized_newton(Bernoulli(), X, y, None, S, start)
    assert saturated.converged
    assert saturated.newton_decrement < NEWTON_DECREMENT_TOL
    np.testing.assert_allclose(saturated.beta, from_zero.beta, rtol=0, atol=1e-9)
    assert saturated.n_iter > from_zero.n_iter

    # Without the cap, halving alone cannot: the same start stops short.
    uncapped = penalized_newton(Bernoulli(), X, y, None, S, start, max_step=np.inf)
    assert not uncapped.converged


def test_stops_unconverged_rather_than_raising() -> None:
    """A singular unpenalized design and too few steps both report not converged."""
    X, y = _logistic_data()
    duplicated = np.column_stack([X, X[:, 1]])
    singular = penalized_newton(
        Bernoulli(), duplicated, y, None, np.zeros((3, 3)), np.zeros(3)
    )
    assert not singular.converged

    short = penalized_newton(
        Bernoulli(), X, y, None, np.zeros((2, 2)), np.array([25.0, -30.0]), max_iter=2
    )
    assert not short.converged
    assert short.n_iter == 2
