"""The penalized Newton fit against hand cases and the content fitter."""

from __future__ import annotations

import dataclasses
import math
import re

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
import scipy.optimize

from continuum_swr_replay.analysis.regression import newton, reml, runtime
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
from continuum_swr_replay.analysis.regression.padding import (
    ROW_BUCKET,
    SMALLEST_ROW_BUCKET,
    bucket_size,
)
from continuum_swr_replay.analysis.regression.reml import (
    PenaltyBlock,
    reml_and_derivatives,
)
from continuum_swr_replay.analysis.regression.runtime import COMPILE_LOG
from continuum_swr_replay.analysis.replay_models.fitting import fit_conditional_poisson

from .conftest import family_kernel, jax_family


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
    g = X.T @ family_kernel(lik, "score", None, eta, y, aux) - S @ beta
    A = family_kernel(lik, "information", X, eta, y, aux) + S
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
        g = X.T @ family_kernel(lik, "score", None, eta, y, offset) - S @ optimum
        information = family_kernel(lik, "information", X, eta, y, offset)
        optimum = optimum + np.linalg.solve(information + S, g)
    A = family_kernel(lik, "information", X, X @ optimum, y, offset) + S
    F = (
        -family_kernel(lik, "loglik", None, X @ optimum, y, offset)
        + 0.5 * optimum @ S @ optimum
    )
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
    # parts 0.05, 1.0, 1.2: the largest first
    np.testing.assert_array_equal(
        newton.running_away_columns(X, np.array([0.05, -1.0, 0.6])), [2, 1]
    )


@pytest.mark.parametrize("larger", [1, 2])
def test_running_away_columns_equal_but_for_rounding_keep_column_order(
    larger: int,
) -> None:
    """Two columns of one direction have parts that differ in the last digit
    only, and which comes out larger depends on the machine's linear-algebra
    kernel: they are a tie, in column order, whichever is larger."""
    X = np.array([[1.0, 0.0, 2.0], [1.0, 1.0, -1.0]])
    step = np.array([0.05, -1.0, 0.5])  # parts 0.05, 1.0, 1.0
    step[larger] *= 1.0 + 4e-16
    part = np.max(np.abs(X * step), axis=0)
    assert part[larger] > part[3 - larger]
    np.testing.assert_array_equal(newton.running_away_columns(X, step), [1, 2])


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
        """The Bernoulli with the score's sign flipped (in the function the
        kernels call): no step lowers F."""

        def _jax_score(self, eta, y, aux, weight):
            return -super()._jax_score(eta, y, aux, weight)

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
    beta = np.array([[-0.5]])
    with jax.enable_x64(True):
        problem = newton._Problem.of(lik, X, y, offset, S)
        F, loglik, eta = problem.evaluate(beta)
        r = problem.iterate(beta, eta, F, newton.MAX_STEP_LENGTH)
        assert r["decrement"][0] == pytest.approx(0.128, abs=1e-3)
        r["step"] = np.array([[to + 0.5]])
        fit = newton._last_step(
            problem, beta, r, np.array([True]), 3, newton.MAX_STEP_LENGTH
        )(0)

    # The fit stays where it was, with A and the decrement there by hand:
    # g = 1 - exp(-0.5), A = exp(-0.5), decrement g^2 / (2 A).
    g, A = 1.0 - np.exp(-0.5), np.exp(-0.5)
    assert (fit.status, fit.n_iter) == (CONVERGED, 3)
    np.testing.assert_array_equal(fit.beta, [-0.5])
    np.testing.assert_allclose(fit.A, [[A]], rtol=1e-15)
    assert fit.newton_decrement == pytest.approx(0.5 * g**2 / A, rel=1e-14)
    assert fit.loglik == pytest.approx(-0.5 - np.exp(-0.5), rel=1e-15)


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


def test_sessions_of_ten_sizes_compile_once_per_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ten fits with ten row counts in four row buckets (512, 2,048, 4,096 and
    8,192 rows) trace each Newton kernel exactly once per bucket. The family
    is a class made here, so no earlier test can have traced its kernels."""

    class _Counted(Poisson):
        pass

    monkeypatch.setattr(runtime, "LIVE_PROGRAM_LIMIT", 10**9)  # no clear here
    sizes = [300, 500, 1500, 2500, 3500, 4000, 4100, 5000, 6500, 8190]
    buckets = {bucket_size(n, ROW_BUCKET, SMALLEST_ROW_BUCKET) for n in sizes}
    assert buckets == {512, 2048, 4096, 8192}
    COMPILE_LOG.clear()
    for n in sizes:
        rng = np.random.default_rng(n)
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        y = rng.poisson(np.exp(0.2 + 0.3 * X[:, 1])).astype(float)
        fit = penalized_newton(
            _Counted(), X, y, np.zeros(n), np.zeros((2, 2)), np.zeros(2)
        )
        assert fit.converged
    for kernel in ("_evaluate", "_iterate"):
        assert COMPILE_LOG.count(kernel) == len(buckets), kernel
        assert {sig[1][0] for sig in COMPILE_LOG.signatures(kernel)} == {
            (b, 2) for b in buckets
        }


def _batch_against_alone(lik, X, responses, aux) -> list[NewtonFit]:
    """Fit ``responses`` on one design in one batch and each alone, assert
    each batch fit equals its fit alone (statuses and step counts exactly,
    coefficients within the references' rule), and return the fits alone."""
    p = X.shape[1]
    S, start = np.zeros((p, p)), np.zeros(p)
    alone = [penalized_newton(lik, X, r, aux, S, start) for r in responses]
    with jax.enable_x64(True):
        problems = [newton._Problem.of(lik, X, r, aux, S) for r in responses]
        Y = jax.device_put(np.concatenate([np.asarray(q.Y) for q in problems]))
        batch = newton._fit(
            dataclasses.replace(problems[0], Y=Y),
            np.zeros((len(responses), p)),
            None,
            newton.NEWTON_MAX_STEPS,
            newton.MAX_STEP_LENGTH,
        )
    for got, want in zip(batch, alone, strict=True):
        assert (got.status, got.n_iter) == (want.status, want.n_iter)
        np.testing.assert_allclose(got.beta, want.beta, rtol=1e-9, atol=1e-11)
    return alone


def test_a_batch_of_fits_equals_each_fit_alone() -> None:
    """Two fits on one design in one batch, one converging and one separated
    (it stops with no finite maximum, later): each ends as it does alone, so
    the linear predictor carried for the fits still stepping is theirs."""
    X, y = _logistic_data()
    separated = (X[:, 1] > 0).astype(float)
    alone = _batch_against_alone(Bernoulli(), X, [y, separated], None)
    assert [f.status for f in alone] == [CONVERGED, NO_FINITE_MAXIMUM]
    assert alone[0].n_iter < alone[1].n_iter


class _Fenced(Poisson):
    """Poisson whose log-likelihood is minus infinity where a real row's
    linear predictor passes 1: a fit whose optimum lies beyond fails its
    halving search."""

    def _jax_loglik(self, eta, y, aux, weight):
        fenced = jnp.any((weight > 0) & (eta > 1.0))
        return jnp.where(fenced, -jnp.inf, super()._jax_loglik(eta, y, aux, weight))


class _Cracked(Poisson):
    """Poisson whose information is NaN where a real row's linear predictor
    passes 1: a fit that steps beyond loses a positive definite ``A``."""

    def _jax_information(self, X, eta, y, aux, weight):
        cracked = jnp.any((weight > 0) & (eta > 1.0))
        information = super()._jax_information(X, eta, y, aux, weight)
        return jnp.where(cracked, jnp.nan, information)


def _poisson_pair(seed: int) -> tuple:
    """One design and two responses: rates near 1 (an optimum at eta near 0)
    and near e^2.5 (beyond 1)."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(200), 0.1 * rng.normal(size=200)])
    return X, [rng.poisson(1.0, 200).astype(float), rng.poisson(12.0, 200) * 1.0]


def test_a_batch_where_one_fit_fails_its_halving_equals_each_fit_alone(
    monkeypatch,
) -> None:
    """With one halving allowed, the fit whose optimum is beyond the fence
    fails its search while the other converges, in the batch as alone."""
    monkeypatch.setattr(newton, "MAX_STEP_HALVINGS", 1)
    X, responses = _poisson_pair(11)
    alone = _batch_against_alone(_Fenced(), X, responses, np.zeros(200))
    assert [f.status for f in alone] == [CONVERGED, HALVING_FAILED]


def test_a_batch_where_one_fit_loses_a_positive_definite_a_equals_each_alone() -> None:
    """The fit that steps past eta = 1 stops with ``A`` not positive definite
    after a step or more, while the other converges, in the batch as alone."""
    X, responses = _poisson_pair(12)
    alone = _batch_against_alone(_Cracked(), X, responses, np.zeros(200))
    assert [f.status for f in alone] == [CONVERGED, NOT_POSITIVE_DEFINITE]
    assert alone[1].n_iter >= 1


def test_the_carried_linear_predictor_stays_near_x_beta(monkeypatch) -> None:
    """A slow fit (a far start, steps cut to 0.5, 45 iterations): the linear
    predictor carried from step to step equals ``X @ beta`` at the last
    iteration within 8 eps of the largest ``|X @ beta|`` along the path (2.5
    measured; the rounding of the large early predictors is carried, so it is
    51 eps of the final ``max |eta|``)."""
    seen = []
    iterate = newton._Problem.iterate

    def recorded(self, beta, eta, F, cap):
        seen.append((beta[0].copy(), np.asarray(eta)[0].copy()))
        return iterate(self, beta, eta, F, cap)

    monkeypatch.setattr(newton._Problem, "iterate", recorded)
    rng = np.random.default_rng(3)
    n, p = 2000, 6
    X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
    rate = np.exp(0.3 + X[:, 1:] @ rng.normal(scale=0.3, size=p - 1))
    y = rng.poisson(rate).astype(float)
    start = rng.normal(scale=3.0, size=p)
    fit = penalized_newton(
        Poisson(), X, y, np.zeros(n), np.zeros((p, p)), start, max_step=0.5
    )
    assert fit.converged and fit.n_iter >= 40
    # The last call is the last step's landing, from a fresh evaluation; the
    # one before it is the last iteration on the carried predictor.
    assert len(seen) == fit.n_iter + 1
    beta, eta = seen[-2]
    largest = max(np.max(np.abs(X @ b)) for b, _ in seen)
    gap = np.max(np.abs(eta[:n] - X @ beta))
    assert gap <= 8 * np.finfo(float).eps * largest


def _transposes_of_rows(compiled_text: str, n_rows: int) -> list[str]:
    """The ``transpose`` instructions of an optimized HLO program whose shape
    has ``n_rows`` as a dimension."""
    found = []
    for line in compiled_text.splitlines():
        shape = re.search(r"= \w+\[([\d,]*)\]\S* transpose\(", line)
        if shape and str(n_rows) in shape.group(1).split(","):
            found.append(line.strip()[:160])
    return found


@pytest.mark.skipif(
    jax.default_backend() != "cpu", reason="reads the CPU's compiled program"
)
@pytest.mark.parametrize("family", ["poisson", "bernoulli", "gaussian", "softmax"])
def test_no_kernel_copies_the_transposed_design(family: str) -> None:
    """A canary for a copy of the transposed design (padded rows x columns)
    in the compiled Newton iteration, information and REML derivative kernels,
    on the CPU.
    ``X.T @ score`` and ``(X.T * w) @ X`` compiled to a ``transpose`` of the
    design (then a ``copy`` to another layout), about half of an iteration at
    the rate model's size; XLA folds ``X.T @ (w * X)`` into the product.

    It matches only an instruction named ``transpose`` whose printed shape has
    the padded row count: XLA could write the same copy as a ``copy`` with
    another layout, which it would miss, and it is not a test of every form
    (inside the iteration's program ``dot_general(w * X, X)`` compiled to a
    ``transpose`` of the weighted design, which it catches, but compiled alone
    it had none). The jaxpr cannot be read instead: ``X.T @ (w * X)`` has a
    ``transpose`` there that XLA removes."""
    rng = np.random.default_rng(0)
    n, p = 5000, 5
    X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
    if family == "softmax":
        lik, aux = GroupedSoftmax(), np.repeat(np.arange(n // 5), 5)
        y = np.tile([0.4, 0.3, 0.1, 0.1, 0.1], n // 5)
    else:
        lik = {"poisson": Poisson(), "bernoulli": Bernoulli(), "gaussian": Gaussian()}[
            family
        ]
        aux = {"poisson": np.zeros(n), "bernoulli": None, "gaussian": 1.0}[family]
        y = rng.integers(0, 2, size=n).astype(float)
    with jax.enable_x64(True):
        problem = newton._Problem.of(lik, X, y, aux, np.eye(p))
        r = problem.rows
        n_bucket = r.X.shape[0]
        beta, F = np.zeros((1, p)), np.zeros(1)
        eta = np.zeros(n_bucket)
        programs = {
            "_iterate": newton._iterate.lower(
                lik,
                r.X,
                problem.Y,
                r.aux,
                r.weight,
                problem.S_device,
                beta,
                eta[None],
                F,
                np.float64(1.0),
                np.float64(NEWTON_DECREMENT_TOL),
                np.int64(3),
            ),
            "_jax_information": jax_family.lower(
                lik, "information", r.X, eta, problem.Y[0], r.aux, r.weight
            ),
            "_jax_dinformation": jax_family.lower(
                lik, "dinformation", r.X, eta, problem.Y[0], r.aux, r.weight, eta
            ),
            "_jax_d2information": jax_family.lower(
                lik, "d2information", r.X, eta, problem.Y[0], r.aux, r.weight, eta, eta
            ),
        }
        # REML's kernel of the score and its derivatives in two log penalties.
        S_j = np.zeros((2, p, p))
        S_j[0, 1:3, 1:3] = S_j[1, 3:, 3:] = np.eye(2)
        masks, f = (S_j.diagonal(axis1=1, axis2=2) > 0) * 1.0, np.float64(0.0)
        b, ranks = np.zeros(p), np.full(2, 2.0)
        if family == "gaussian":
            programs["_profiled_gaussian"] = reml._profiled_gaussian.lower(
                r.X,
                problem.Y,
                r.weight,
                np.eye(p),
                b,
                masks,
                f,
                np.int64(n),
                np.int64(n - 1),
                S_j,
                ranks,
            )
        else:
            programs["_laplace_derivatives"] = reml._laplace_derivatives.lower(
                lik,
                r.X,
                problem.Y,
                r.aux,
                r.weight,
                S_j,
                np.eye(p),
                np.eye(p),
                b,
                masks,
                f,
                f,
                f,
                ranks,
            )
        for name, lowered in programs.items():
            text = lowered.compile().as_text()
            assert _transposes_of_rows(text, n_bucket) == [], name


def test_a_patched_rule_constant_takes_effect_after_the_kernel_compiled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rule constants are arguments of the compiled kernel, read at each
    call: with no halvings allowed the same fit fails at its first step, and
    with a loose decrement tolerance (and the step allowed to move the linear
    predictor) it stops after the one last step, without a new trace."""
    monkeypatch.setattr(runtime, "LIVE_PROGRAM_LIMIT", 10**9)  # no clear here
    X, y = _logistic_data()
    S = np.zeros((2, 2))
    start = np.array([0.3, 1.0])
    fit = penalized_newton(Bernoulli(), X, y, None, S, start)
    assert fit.converged and fit.n_iter > 2
    COMPILE_LOG.clear()

    monkeypatch.setattr(newton, "MAX_STEP_HALVINGS", 0)
    halted = penalized_newton(Bernoulli(), X, y, None, S, start)
    assert (halted.status, halted.n_iter) == (HALVING_FAILED, 0)

    monkeypatch.setattr(newton, "MAX_STEP_HALVINGS", 30)
    monkeypatch.setattr(newton, "NEWTON_DECREMENT_TOL", 1e6)
    monkeypatch.setattr(newton, "NO_MAXIMUM_ETA_STEP", 1e6)
    early = penalized_newton(Bernoulli(), X, y, None, S, start)
    assert early.converged and early.n_iter == 1
    assert COMPILE_LOG.count() == 0


# Two fits whose last step the NumPy fitter (before the JAX one) refused: its
# sum of the log-likelihood moved F by more than the step's allowance,
# 8 eps |F|, between two points a decrement of about 1e-15 apart. The inputs
# are those of the review's probes (fixed seeds). The step is checked against
# an exact sum: math.fsum over each row's term, each term from a linear
# predictor summed with math.fsum.


def _exact_objective(family: str, X, y, aux, S, beta) -> float:
    """``F = -l + beta' S beta / 2`` with every sum exact (``math.fsum``)."""
    eta = np.array(
        [math.fsum(x * b for x, b in zip(row, beta, strict=True)) for row in X]
    )
    if family == "poisson":
        terms = y * (eta + aux) - np.exp(eta + aux)
    else:
        terms = y * eta - np.logaddexp(0.0, eta)
    penalty = math.fsum((0.5 * beta[:, None] * S * beta[None, :]).ravel())
    return -math.fsum(terms) + penalty


def _bernoulli_probe():
    """The Bernoulli case (n 2,500, p 6, a penalty of 1 on columns 1 to 5)."""
    rng = np.random.default_rng(1165)
    p = int(rng.integers(2, 9))
    n = int(rng.choice([40, 150, 600, 2500, 5000]))
    X = np.column_stack(
        [np.ones(n), rng.normal(size=(n, p - 1)) * rng.choice([0.3, 1, 3])]
    )
    beta = rng.normal(scale=rng.choice([0.2, 0.8, 1.5]), size=p)
    lam = rng.choice([0, 0.01, 1.0, 50.0])
    S = np.zeros((p, p))
    S[1:, 1:] = lam * np.eye(p - 1)
    start = np.zeros(p) if rng.random() < 0.7 else rng.normal(scale=0.5, size=p)
    eta = X @ beta
    y = (rng.random(n) < 1 / (1 + np.exp(-eta))).astype(float)
    assert (n, p, lam) == (2500, 6, 1.0)
    return X, y, S, start


def _poisson_probe():
    """The Poisson REML case (n 4,096, p 6, blocks [1, 2, 3] and [4, 5]),
    drawn after the probe's three families at 60 rows from the same seed."""
    rng = np.random.default_rng(5)
    p = 6
    for n, family in [
        (60, "poisson"),
        (60, "bernoulli"),
        (60, "gaussian"),
        (60, "softmax"),
        (4096, "poisson"),
    ]:
        X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
        beta = rng.normal(scale=0.4, size=p)
        eta = X @ beta
        if family == "poisson":
            offset = rng.normal(scale=0.3, size=n)
            y = rng.poisson(np.exp(eta + offset)).astype(float)
        elif family == "bernoulli":
            rng.random(n)
        elif family == "gaussian":
            rng.normal(size=n)
        else:
            sizes = rng.integers(2, 8, size=max(n // 5, 3))
            X = rng.normal(size=(int(sizes.sum()), p))
            w = np.exp(X @ beta)
            for s, z in zip(np.r_[0, np.cumsum(sizes)[:-1]], sizes, strict=True):
                rng.choice(z, p=w[s : s + z] / w[s : s + z].sum())
    blocks = [
        PenaltyBlock(np.array([1, 2, 3]), np.diag([1.0, 2.0, 3.0])),
        PenaltyBlock(np.array([4, 5]), np.eye(2)),
    ]
    return X, y, offset, blocks


def _last_steps(monkeypatch: pytest.MonkeyPatch) -> list:
    """Record each call of ``newton._last_step`` (of a batch of one): its
    problem, coefficients, full step and iteration, and, once the fit is
    read, whether the step was taken."""
    calls = []
    last_step = newton._last_step

    def recorded(problem, beta, r, members, n_iter, cap):
        call = {"problem": problem, "beta": beta[0].copy(), "n_iter": n_iter}
        call["step"] = r["step"][0].copy()
        calls.append(call)
        fit_of = last_step(problem, beta, r, members, n_iter, cap)

        def fit(b):
            out = fit_of(b)
            call["taken"] = out.n_iter == n_iter + 1
            return out

        return fit

    monkeypatch.setattr(newton, "_last_step", recorded)
    return calls


def _exact_change(family: str, call) -> tuple[float, float]:
    """The change of the exact F along a recorded last step, and the rule's
    allowance ``8 eps max(1, |F|)``, both in eps |F|."""
    problem, beta, step = call["problem"], call["beta"], call["step"]
    X, n = problem.X, problem.rows.n_real
    y, aux = np.asarray(problem.Y[0])[:n], problem.aux
    before = _exact_objective(family, X, y, aux, problem.S, beta)
    after = _exact_objective(family, X, y, aux, problem.S, beta + step)
    unit = np.finfo(float).eps * abs(before)
    return (after - before) / unit, 8.0 * np.finfo(float).eps * max(
        1.0, abs(before)
    ) / unit


def _assert_the_decision_fits_the_exact_sum(family: str, call) -> None:
    """A step taken does not raise the exact F by more than the allowance, and
    a step refused does not lower it by more than the allowance."""
    change, allowance = _exact_change(family, call)
    assert not (call["taken"] and change > allowance), change
    assert not (not call["taken"] and change < -allowance), change


def test_a_last_step_the_numpy_sum_refused_is_decided_by_the_exact_sum(
    monkeypatch,
) -> None:
    """Bernoulli: the NumPy fitter refused the last step (its F rose 16.8 eps
    |F| with it, beyond the allowance of 8); under the exact sum it moves F
    by -0.99 eps |F|. Whatever the platform's rounding decides, the decision
    must agree with the exact sum. (On the CPU where this was written the step
    is taken, 9 steps to 10; that count is not asserted.)"""
    X, y, S, start = _bernoulli_probe()
    calls = _last_steps(monkeypatch)
    fit = penalized_newton(Bernoulli(), X, y, None, S, start)
    assert fit.status == CONVERGED and "taken" in calls[-1]
    if calls[-1]["taken"]:
        np.testing.assert_array_equal(fit.beta, calls[-1]["beta"] + calls[-1]["step"])
    _assert_the_decision_fits_the_exact_sum("bernoulli", calls[-1])


def test_a_reml_fits_last_step_is_decided_by_the_exact_sum(monkeypatch) -> None:
    """Poisson REML score: the NumPy fitter's inner fit refused its last step,
    which lowers the exact F by about 4 eps |F|; the decision must agree with
    the exact sum. (Taken on the CPU where this was written, 4 steps to 5;
    not asserted.)"""
    X, y, offset, blocks = _poisson_probe()
    calls = _last_steps(monkeypatch)
    score = reml_and_derivatives(Poisson(), X, y, offset, blocks, [0.5, -0.3])
    assert score.fit.status == CONVERGED and "taken" in calls[-1]
    _assert_the_decision_fits_the_exact_sum("poisson", calls[-1])


@pytest.mark.parametrize(
    ("lik", "aux"), [(Poisson(), np.empty(0)), (Bernoulli(), None), (Gaussian(), 1.0)]
)
def test_no_rows_is_a_design_not_identified(lik, aux) -> None:
    """No rows identify no coefficient: the status of the NumPy fitter, no
    step, log-likelihood 0."""
    fit = penalized_newton(
        lik, np.empty((0, 2)), np.empty(0), aux, np.zeros((2, 2)), np.zeros(2)
    )
    assert (fit.status, fit.n_iter, fit.loglik) == (NOT_IDENTIFIED, 0, 0.0)
    np.testing.assert_array_equal(fit.beta, [0.0, 0.0])


@pytest.mark.parametrize(
    ("lik", "aux"), [(Poisson(), np.empty(0)), (Bernoulli(), None), (Gaussian(), 1.0)]
)
def test_no_rows_with_a_full_penalty_converge_to_the_penaltys_minimum(lik, aux) -> None:
    """With no rows and a penalty of full rank every coefficient is identified
    by the penalty alone: the optimum of ``beta' S beta / 2`` is 0, whatever
    the start, with ``A = S``, a log-likelihood of 0 and a decrement of 0."""
    S = np.array([[2.0, 0.5], [0.5, 1.0]])
    fit = penalized_newton(
        lik, np.empty((0, 2)), np.empty(0), aux, S, np.array([0.7, -1.2])
    )
    assert fit.status == CONVERGED
    np.testing.assert_allclose(fit.beta, [0.0, 0.0], atol=1e-15)
    np.testing.assert_allclose(fit.A, S, rtol=1e-15)
    assert fit.loglik == 0.0 and fit.newton_decrement < 1e-30


@pytest.mark.parametrize(
    "change",
    [
        {"S": np.float32},
        {"S": int},
        {"X": np.float32},
        {"X": int},
        {"y": np.float32},
        {"y": int},
        {"offset": np.float32},
        {"offset": int},
        {"start": np.float32},
        {"start": int},
    ],
    ids=lambda change: "-".join(f"{k}_{v.__name__}" for k, v in change.items()),
)
def test_inputs_of_other_dtypes_fit_as_float64(change) -> None:
    """Integer and float32 inputs are fitted in float64: the same fit as the
    same values given as float64. The float32 inputs hold values float32
    rounds (0.1, 1/3), so a fit computed in float32 would differ."""
    rng = np.random.default_rng(4)
    if change.get(next(iter(change))) is np.float32:
        inputs = {
            "X": np.column_stack([np.ones(30), rng.uniform(0.0, 3.0, size=30)]),
            "y": rng.poisson(2.0, size=30) + 1.0 / 3.0,
            "offset": 0.1 * rng.integers(-5, 6, size=30) + 1.0 / 3.0,
            "S": np.diag([0.0, 1.0 / 3.0]),
            "start": np.array([0.1, 1.0 / 3.0]),
        }
    else:
        inputs = {
            "X": np.column_stack([np.ones(30), rng.integers(0, 3, size=30)]),
            "y": rng.poisson(2.0, size=30),
            "offset": rng.integers(-1, 2, size=30),
            "S": np.diag([0, 1]),
            "start": np.zeros(2, dtype=int),
        }
    given = {k: np.asarray(v, dtype=change.get(k, float)) for k, v in inputs.items()}
    as_float64 = {k: np.asarray(v, dtype=float) for k, v in given.items()}
    for k, kind in change.items():
        if kind is np.float32:  # float32 changed the values
            assert not np.array_equal(as_float64[k], inputs[k])

    def fit(a):
        return penalized_newton(
            Poisson(), a["X"], a["y"], a["offset"], a["S"], a["start"]
        )

    expected, got = fit(as_float64), fit(given)
    assert expected.converged
    assert (got.status, got.n_iter) == (expected.status, expected.n_iter)
    np.testing.assert_array_equal(got.beta, expected.beta)
    np.testing.assert_array_equal(got.A, expected.A)


def test_a_fit_owns_its_arrays() -> None:
    """``A`` and ``step`` are copies, not views of the batch's arrays."""
    lik, X, y, aux = _separated_cases()["logistic_cell"]
    p = X.shape[1]
    fit = penalized_newton(lik, X, y, aux, np.zeros((p, p)), np.zeros(p))
    assert fit.status == NO_FINITE_MAXIMUM
    assert fit.A.base is None and fit.step.base is None and fit.beta.base is None
