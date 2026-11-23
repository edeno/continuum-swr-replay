"""The likelihoods' derivatives against central finite differences, and
their draws.

Each level is checked against a finite difference of the level below it, and
the lowest against the log-likelihood itself: ``score_eta`` is the gradient of
``loglik`` in ``eta``; ``information`` is minus the derivative of the
coefficient gradient ``X' score_eta(X beta)`` in ``beta``; ``dinformation`` is
the derivative of ``information`` along a direction of ``eta``; and
``d2information`` the derivative of ``dinformation`` along a second one.
The draws (``draw``) have the family's mean.
"""

from __future__ import annotations

import numpy as np
import pytest

from continuum_swr_replay.analysis.regression.likelihoods import (
    Bernoulli,
    Gaussian,
    GroupedSoftmax,
    Poisson,
    draw,
)

TOLERANCE = 1e-6
N_ROWS = 14
N_COLUMNS = 3


def _relative_error(analytic: np.ndarray, numeric: np.ndarray) -> float:
    """``||analytic - numeric|| / ||numeric||``; zero when both are zero."""
    gap = float(np.linalg.norm(np.asarray(analytic) - np.asarray(numeric)))
    return gap / float(np.linalg.norm(numeric)) if gap > 0 else 0.0


def _case(name: str):
    """A likelihood, design, response, aux and linear predictor, seeded."""
    rng = np.random.default_rng(7)
    X = rng.normal(scale=0.6, size=(N_ROWS, N_COLUMNS))
    eta = X @ rng.normal(scale=0.5, size=N_COLUMNS)
    if name == "poisson":
        offset = np.log(rng.uniform(0.2, 2.0, size=N_ROWS))
        y = rng.poisson(np.exp(eta + offset)).astype(float)
        return Poisson(), X, y, offset, eta
    if name == "bernoulli":
        y = (rng.uniform(size=N_ROWS) < 0.4).astype(float)
        return Bernoulli(), X, y, None, eta
    if name == "gaussian":
        y = eta + rng.normal(size=N_ROWS)
        return Gaussian(), X, y, 0.7, eta
    # Groups of sizes 3, 2, 5 and 4, sorted by id.
    groups = np.repeat([0, 1, 2, 3], [3, 2, 5, 4])
    if name == "softmax_fractions":
        y = np.concatenate([rng.dirichlet(np.ones(k)) for k in (3, 2, 5, 4)])
    else:  # counts: each group's total differs from one
        y = rng.poisson(2.0, size=N_ROWS).astype(float)
        y[0] += 1.0  # no group is empty
        y[3] += 1.0
        y[5] += 1.0
        y[10] += 1.0
    return GroupedSoftmax(), X, y, groups, eta


CASES = ["poisson", "bernoulli", "gaussian", "softmax_fractions", "softmax_counts"]


@pytest.mark.parametrize("name", CASES)
def test_derivatives_match_finite_differences(name: str) -> None:
    lik, X, y, aux, eta = _case(name)
    rng = np.random.default_rng(11)
    d1 = rng.normal(size=N_ROWS)
    d2 = rng.normal(size=N_ROWS)

    # score_eta: the gradient of loglik in eta.
    h = 1e-5
    eye = np.eye(N_ROWS)
    numeric_score = np.array(
        [
            (lik.loglik(eta + h * e, y, aux) - lik.loglik(eta - h * e, y, aux))
            / (2 * h)
            for e in eye
        ]
    )
    assert _relative_error(lik.score_eta(eta, y, aux), numeric_score) < TOLERANCE

    # information: minus the beta-derivative of X' score_eta(X beta).
    beta = np.linalg.lstsq(X, eta, rcond=None)[0]
    eta = X @ beta  # on the column space, so eta is X beta exactly

    def gradient(b: np.ndarray) -> np.ndarray:
        return X.T @ lik.score_eta(X @ b, y, aux)

    h = 1e-5
    numeric_information = -np.column_stack(
        [
            (gradient(beta + h * e) - gradient(beta - h * e)) / (2 * h)
            for e in np.eye(N_COLUMNS)
        ]
    )
    information = lik.information(X, eta, y, aux)
    assert information.shape == (N_COLUMNS, N_COLUMNS)
    assert _relative_error(information, numeric_information) < TOLERANCE

    # dinformation: the derivative of information along d1.
    h = 1e-5
    numeric_d = (
        lik.information(X, eta + h * d1, y, aux)
        - lik.information(X, eta - h * d1, y, aux)
    ) / (2 * h)
    assert _relative_error(lik.dinformation(X, eta, y, aux, d1), numeric_d) < TOLERANCE

    # d2information: the derivative of dinformation(d1) along d2.
    numeric_d2 = (
        lik.dinformation(X, eta + h * d2, y, aux, d1)
        - lik.dinformation(X, eta - h * d2, y, aux, d1)
    ) / (2 * h)
    analytic_d2 = lik.d2information(X, eta, y, aux, d1, d2)
    assert _relative_error(analytic_d2, numeric_d2) < TOLERANCE
    # Symmetric in its two directions.
    assert np.allclose(analytic_d2, lik.d2information(X, eta, y, aux, d2, d1))


def test_softmax_information_is_minus_the_hessian_of_loglik() -> None:
    """The grouped information equals a direct second difference of loglik."""
    lik, X, y, groups, eta = _case("softmax_counts")
    beta = np.linalg.lstsq(X, eta, rcond=None)[0]
    h = 1e-4
    eye = np.eye(N_COLUMNS)

    def loglik(b: np.ndarray) -> float:
        return lik.loglik(X @ b, y, groups)

    hessian = np.array(
        [
            [
                (
                    loglik(beta + h * (ei + ej))
                    - loglik(beta + h * (ei - ej))
                    - loglik(beta - h * (ei - ej))
                    + loglik(beta - h * (ei + ej))
                )
                / (4 * h * h)
                for ej in eye
            ]
            for ei in eye
        ]
    )
    information = lik.information(X, X @ beta, y, groups)
    assert _relative_error(information, -hessian) < 1e-5


def test_softmax_rejects_unsorted_groups() -> None:
    lik, X, y, groups, eta = _case("softmax_fractions")
    with pytest.raises(ValueError, match="sorted"):
        lik.loglik(eta, y, groups[::-1])


def test_softmax_is_invariant_to_a_constant_within_each_group() -> None:
    """A group-level constant cancels: it is not identified, as the class says."""
    lik, X, y, groups, eta = _case("softmax_fractions")
    shift = np.array([3.0, -1.0, 0.5, 20.0])[groups]
    assert np.isclose(lik.loglik(eta + shift, y, groups), lik.loglik(eta, y, groups))


N_DRAWS = 20_000


@pytest.mark.parametrize(
    "name", ["poisson", "bernoulli", "gaussian", "softmax_one", "softmax_shares"]
)
def test_draws_have_the_family_mean(name: str) -> None:
    """At a fixed ``eta``, the mean of 20,000 draws of each row is the
    family's mean within 4 standard errors (the draws' SD over sqrt(20,000)):
    ``exp(eta + offset)``, ``expit(eta)``, ``eta``, and the softmax within each
    group. A share draw sums to one in every group."""
    rng = np.random.default_rng(11)
    eta = np.array([-1.2, 0.3, 0.9, -0.4, 1.5, 0.0, -2.0])
    groups = np.array([0, 0, 0, 1, 1, 2, 2])
    if name == "poisson":
        lik, aux, n_trials = Poisson(), np.log([0.5, 1.0, 2.0, 3.0, 0.2, 1.0, 4.0]), 1
        mean = np.exp(eta + aux)
    elif name == "bernoulli":
        lik, aux, n_trials = Bernoulli(), None, 1
        mean = 1.0 / (1.0 + np.exp(-eta))
    elif name == "gaussian":
        lik, aux, n_trials = Gaussian(), 0.7, 1
        mean = eta
    else:
        lik, aux = GroupedSoftmax(), groups
        n_trials = 1 if name == "softmax_one" else 5
        weight = np.exp(eta)
        mean = weight / np.bincount(groups, weights=weight)[groups]
    draws = np.stack(
        [draw(lik, eta, aux, rng, n_trials=n_trials) for _ in range(N_DRAWS)]
    )
    assert draws.shape == (N_DRAWS, eta.size)
    se = draws.std(axis=0, ddof=1) / np.sqrt(N_DRAWS)
    assert np.all(np.abs(draws.mean(axis=0) - mean) < 4.0 * se)
    if isinstance(lik, GroupedSoftmax):
        totals = np.stack([np.bincount(groups, weights=row) for row in draws])
        np.testing.assert_allclose(totals, 1.0, rtol=0, atol=1e-12)
        assert set(np.unique(draws * n_trials)) <= set(range(n_trials + 1))


def test_a_draw_refuses_an_unknown_family() -> None:
    """A likelihood without a draw is refused by name."""

    class Other:
        name = "other"

    with pytest.raises(ValueError, match="no draw for the likelihood 'other'"):
        draw(Other(), np.zeros(2), None, np.random.default_rng(0))
