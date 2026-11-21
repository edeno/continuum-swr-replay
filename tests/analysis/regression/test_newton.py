"""The penalized Newton fit against hand cases and the content fitter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scipy.optimize

from continuum_swr_replay.analysis.regression import newton
from continuum_swr_replay.analysis.regression.constants import (
    NEWTON_DECREMENT_TOL,
    NO_MAXIMUM_ETA_STEP,
)
from continuum_swr_replay.analysis.regression.likelihoods import (
    Bernoulli,
    Gaussian,
    GroupedSoftmax,
    Poisson,
)
from continuum_swr_replay.analysis.regression.newton import (
    CONVERGED,
    HALVING_FAILED,
    NO_FINITE_MAXIMUM,
    NOT_IDENTIFIED,
    NOT_POSITIVE_DEFINITE,
    STEP_LIMIT,
    NewtonFit,
    penalized_newton,
)
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
    assert singular.status == NOT_IDENTIFIED and not singular.converged

    max_iter = 2
    short = penalized_newton(
        Bernoulli(),
        X,
        y,
        None,
        np.zeros((2, 2)),
        np.array([25.0, -30.0]),
        max_iter=max_iter,
    )
    assert short.status == STEP_LIMIT and not short.converged
    assert short.n_iter == max_iter


def test_a_start_that_overflows_reports_not_converged() -> None:
    """A start whose expected counts overflow gives an infinite A, not an error."""
    X = np.array([[1.0], [2.0]])
    with np.errstate(over="ignore", invalid="ignore"):
        fit = penalized_newton(
            Poisson(),
            X,
            np.array([1.0, 2.0]),
            np.zeros(2),
            np.zeros((1, 1)),
            np.array([800.0]),
        )
    assert not fit.converged
    assert fit.n_iter == 0


def test_last_step_below_the_objective_rounding_is_taken() -> None:
    """From a start whose predicted gain is below the rounding of F, the fit
    still lands on the optimum: F cannot tell the points apart, the decrement
    can."""
    rng = np.random.default_rng(12)
    X = np.column_stack([np.ones(400), rng.normal(size=(400, 4))])
    offset = np.log(rng.uniform(0.5, 2.0, size=400))
    y = rng.poisson(np.exp(1.0 + 0.3 * X[:, 1] + offset)).astype(float)
    S = np.diag([0.0, 1.0, 1.0, 1.0, 1.0])
    lik = Poisson()
    optimum = penalized_newton(lik, X, y, offset, S, np.zeros(5)).beta
    for _ in range(3):  # polish by plain Newton steps
        eta = X @ optimum
        g = X.T @ lik.score_eta(eta, y, offset) - S @ optimum
        optimum = optimum + np.linalg.solve(lik.information(X, eta, y, offset) + S, g)
    A = lik.information(X, X @ optimum, y, offset) + S
    F = -lik.loglik(X @ optimum, y, offset) + 0.5 * optimum @ S @ optimum
    # A start off the optimum along A's weakest direction, with a predicted
    # gain delta' A delta / 2 a tenth of the rounding of F.
    gain = 0.1 * abs(F) * np.finfo(float).eps
    eigenvalues, vectors = np.linalg.eigh(A)
    delta = vectors[:, 0] * np.sqrt(2 * gain / eigenvalues[0])

    fit = penalized_newton(lik, X, y, offset, S, optimum + delta)

    assert fit.converged
    assert np.max(np.abs(delta)) > 1e-9
    np.testing.assert_allclose(fit.beta, optimum, rtol=0, atol=1e-12)


def _separated_cases() -> dict[str, tuple]:
    """Fits whose likelihood has no finite maximum along one direction.

    - a logistic 0/1 column whose rows are all y = 1;
    - a Poisson 0/1 cell with no events;
    - a logistic continuous column that quasi-separates the outcome (y = 1
      above zero, y = 0 below, both at zero), its other values at least 0.5
      from zero.
    """
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    high = (x > 1.5).astype(float)
    y = (rng.uniform(size=300) < 0.3).astype(float)
    y[high == 1] = 1.0
    logistic = (Bernoulli(), np.column_stack([np.ones(300), x, high]), y, None)
    counts = rng.poisson(1.0, size=300).astype(float)
    counts[high == 1] = 0.0
    poisson = (Poisson(), np.column_stack([np.ones(300), high]), counts, np.zeros(300))
    away = rng.choice([-1.0, 1.0], size=200) * rng.uniform(0.5, 2.0, size=200)
    z = np.r_[away, np.zeros(10)]
    quasi_y = np.r_[(z[:200] > 0).astype(float), np.tile([0.0, 1.0], 5)]
    quasi = (Bernoulli(), np.column_stack([np.ones(210), z]), quasi_y, None)
    return {"logistic_cell": logistic, "poisson_cell": poisson, "quasi": quasi}


@pytest.mark.parametrize(
    ("name", "running_away"),
    [("logistic_cell", [2]), ("poisson_cell", [1]), ("quasi", [1])],
)
def test_separation_has_no_finite_maximum(name: str, running_away: list[int]) -> None:
    """The decrement goes to zero while a coefficient runs to infinity: the fit
    is not converged, says why, and names the column running away (the 0/1
    cell, the separating column; not the intercept)."""
    lik, X, y, aux = _separated_cases()[name]
    p = X.shape[1]
    fit = penalized_newton(lik, X, y, aux, np.zeros((p, p)), np.zeros(p))
    assert not fit.converged
    assert fit.status == NO_FINITE_MAXIMUM
    assert fit.running_away.tolist() == running_away
    assert np.max(np.abs(X @ fit.step)) > NO_MAXIMUM_ETA_STEP


def test_a_converged_fit_has_no_running_away_columns() -> None:
    X, y = _logistic_data()
    fit = penalized_newton(Bernoulli(), X, y, None, np.zeros((2, 2)), np.zeros(2))
    assert fit.converged
    assert fit.step is None and fit.running_away.size == 0


def test_running_away_columns_by_their_part_of_the_step() -> None:
    """Column ``j``'s part is ``max_i |X_ij step_j|``; those above
    ``NO_MAXIMUM_ETA_STEP`` (0.1), the largest first."""
    X = np.array([[1.0, 0.0, 2.0], [1.0, 1.0, -1.0]])
    step = np.array([0.05, -1.0, 0.5])  # parts 0.05, 1.0, 1.0
    np.testing.assert_array_equal(newton.running_away_columns(X, step), [1, 2])
    np.testing.assert_array_equal(
        newton.running_away_columns(X, np.array([0.2, 0.0, 0.0])), [0]
    )


def test_each_status() -> None:
    """Every way the fit ends has its own status."""
    X, y = _logistic_data()
    S = np.zeros((2, 2))
    fit = penalized_newton(Bernoulli(), X, y, None, S, np.zeros(2))
    assert fit.converged and fit.status == CONVERGED

    short = penalized_newton(
        Bernoulli(), X, y, None, S, np.array([25.0, -30.0]), max_iter=2
    )
    assert short.status == STEP_LIMIT and not short.converged

    with np.errstate(over="ignore", invalid="ignore"):
        overflow = penalized_newton(
            Poisson(),
            np.array([[1.0], [2.0]]),
            np.array([1.0, 2.0]),
            np.zeros(2),
            np.zeros((1, 1)),
            np.array([800.0]),
        )
    assert overflow.status == NOT_POSITIVE_DEFINITE

    class _Uphill(Bernoulli):
        """The Bernoulli with the score's sign flipped: no step lowers F."""

        def score_eta(self, eta, y, aux):
            return -super().score_eta(eta, y, aux)

    uphill = penalized_newton(_Uphill(), X, y, None, S, np.array([0.3, 1.0]))
    assert uphill.status == HALVING_FAILED and not uphill.converged


@pytest.mark.parametrize("seed", range(20))
def test_a_collinear_design_is_not_fitted(seed: int) -> None:
    """A duplicated column (up to its scale) is refused before any step,
    whatever the rounding of the information's Cholesky factor (on these 20
    seeds it succeeds at the start on 11)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=300)
    X = np.column_stack([np.ones(300), x, 3.0 * x])
    y = rng.poisson(np.exp(0.5 + np.sin(3.0 * x))).astype(float)
    fit = penalized_newton(
        Poisson(), X, y, np.zeros(300), np.zeros((3, 3)), np.zeros(3)
    )
    assert fit.status == NOT_IDENTIFIED and not fit.converged
    # The one combination it cannot identify is x against 3 x.
    assert fit.unidentified.shape == (3, 1)
    np.testing.assert_array_equal(np.abs(fit.unidentified[:, 0]) > 0.1, [0, 1, 1])


@pytest.mark.parametrize("to", [0.45, 0.3])
def test_last_step_never_raises_f_nor_lands_unconverged(to: float) -> None:
    """``F(b) = exp(b) - b`` (one Poisson count, optimum 0). From ``b = -0.5``
    a move to 0.45 lowers the decrement (0.128 to 0.103) but raises ``F``
    (0.107 to 0.118 above its minimum); a move to 0.3 lowers ``F`` but lands at
    a decrement of 0.045, above the tolerance. Neither is taken."""
    X, y, offset, S = np.ones((1, 1)), np.ones(1), np.zeros(1), np.zeros((1, 1))
    lik = Poisson()
    beta = np.array([-0.5])
    eta = X @ beta
    g = X.T @ lik.score_eta(eta, y, offset)
    A = lik.information(X, eta, y, offset)
    decrement = 0.5 * float(g @ np.linalg.solve(A, g))
    fit = NewtonFit(beta, A, CONVERGED, decrement, 3, lik.loglik(eta, y, offset))

    F = -fit.loglik  # no penalty
    moved = newton._last_step(lik, X, y, offset, S, fit, F, np.array([to + 0.5]))

    assert moved is fit


def _inputs(family: str) -> tuple:
    """A small valid fit's likelihood, design, response and aux."""
    X = np.column_stack([np.ones(6), np.arange(6.0)])
    if family == "poisson":
        return Poisson(), X, np.array([0.0, 1, 2, 1, 3, 2]), np.zeros(6)
    if family == "bernoulli":
        return Bernoulli(), X, np.array([0.0, 1, 0, 1, 1, 0]), None
    if family == "gaussian":
        return Gaussian(), X, np.array([0.1, 0.5, 0.3, 0.9, 1.2, 1.0]), 1.0
    shares = np.array([0.2, 0.3, 0.5, 0.6, 0.4, 0.0])
    return GroupedSoftmax(), X[:, 1:], shares, np.array([0, 0, 0, 1, 1, 1])


def _changed(family: str, part: str, index: int, value: float) -> tuple:
    lik, X, y, aux = _inputs(family)
    X, y = X.copy(), y.copy()
    aux = None if aux is None else np.array(aux, dtype=float)
    if part == "X":
        X[index, 0] = value
    elif part == "y":
        y[index] = value
    elif part == "aux" and family == "gaussian":
        aux = value
    else:
        aux[index] = value
    return lik, X, y, aux


@pytest.mark.parametrize(
    ("family", "part", "index", "value", "message"),
    [
        ("poisson", "X", 2, np.nan, "design.*not finite"),
        ("bernoulli", "X", 0, np.inf, "design.*not finite"),
        ("poisson", "y", 1, np.nan, "response.*not finite"),
        ("gaussian", "y", 4, np.inf, "response.*not finite"),
        ("poisson", "y", 3, -1.0, "Poisson.*below zero"),
        ("bernoulli", "y", 2, 0.5, r"Bernoulli.*\{0, 1\}"),
        ("poisson", "aux", 1, np.nan, "offset.*not finite"),
        ("poisson", "aux", 4, -np.inf, "zero exposure.*drop"),
        ("gaussian", "aux", 0, 0.0, "scale.*positive"),
        ("grouped_softmax", "y", 1, -0.3, "shares.*negative"),
        ("grouped_softmax", "aux", 2, np.nan, "group ids.*not finite"),
    ],
)
def test_inputs_are_checked(
    family: str, part: str, index: int, value: float, message: str
) -> None:
    """A fit refuses inputs it cannot fit, and says what is wrong."""
    lik, X, y, aux = _changed(family, part, index, value)
    p = X.shape[1]
    with pytest.raises(ValueError, match=message):
        penalized_newton(lik, X, y, aux, np.zeros((p, p)), np.zeros(p))


def test_a_group_whose_shares_sum_to_zero_is_refused() -> None:
    lik, X, y, groups = _inputs("grouped_softmax")
    y = y.copy()
    y[3:] = 0.0
    with pytest.raises(ValueError, match="group.*sum to zero"):
        penalized_newton(lik, X, y, groups, np.zeros((1, 1)), np.zeros(1))


def test_a_finite_maximum_far_away_is_a_step_limit_not_a_separation() -> None:
    """A Poisson slope of 2e3 on a column of units 1e-3: every step is cut to
    the cap of 10 and lowers ``F``, but the full step shrinks as the fit
    approaches the optimum, so a fit that runs out of steps on the way is
    "step limit reached", not "no finite maximum"."""
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 1e-3, 400)
    X = np.column_stack([np.ones(400), x])
    y = rng.poisson(np.exp(0.2 + 2e3 * x)).astype(float)
    S, start = np.zeros((2, 2)), np.zeros(2)
    short = penalized_newton(Poisson(), X, y, np.zeros(400), S, start, max_iter=40)
    assert short.status == STEP_LIMIT
    full = penalized_newton(Poisson(), X, y, np.zeros(400), S, start, max_iter=400)
    assert full.converged
    assert full.beta[1] == pytest.approx(2e3, rel=0.1)


def test_a_softmax_design_constant_within_groups_is_not_identified() -> None:
    """The grouped softmax sees only variation within a group: two columns
    whose difference is constant within every group (as with a session's
    role columns that together mark every candidate) cannot be told apart,
    and the fit is refused as not identified, not left to fail as an
    information that is not positive definite."""
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(50), 4)
    x = rng.normal(size=200)
    level = np.repeat(rng.normal(size=50), 4)
    X = np.column_stack([x, x + level])
    y = rng.dirichlet(np.ones(4), size=50).ravel()
    fit = penalized_newton(
        GroupedSoftmax(), X, y, groups, np.zeros((2, 2)), np.zeros(2)
    )
    assert fit.status == NOT_IDENTIFIED
