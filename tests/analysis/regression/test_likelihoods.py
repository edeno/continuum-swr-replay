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
ETA = np.array([-1.2, 0.3, 0.9, -0.4, 1.5, 0.0, -2.0])
GROUPS = np.array([0, 0, 0, 1, 1, 2, 2])
#: The softmax's noise models: (n_trials, concentration).
SOFTMAX = {
    "categorical": (1, None),
    "multinomial": (5, None),
    "multinomial_per_group": (np.array([2, 4, 7]), None),
    "dirichlet_multinomial": (np.array([2, 4, 7]), 3.0),
    "dirichlet_multinomial_per_group": (6, np.array([0.5, 3.0, 20.0])),
}


def _softmax_draws(
    name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """20,000 draws of a softmax noise model at ``ETA``, the softmax ``p``
    and each row's ``n`` and concentration (inf without one)."""
    n_trials, concentration = SOFTMAX[name]
    rng = np.random.default_rng(13)
    draws = np.stack(
        [
            draw(
                GroupedSoftmax(),
                ETA,
                GROUPS,
                rng,
                n_trials=n_trials,
                concentration=concentration,
            )
            for _ in range(N_DRAWS)
        ]
    )
    weight = np.exp(ETA)
    p = weight / np.bincount(GROUPS, weights=weight)[GROUPS]
    n = np.broadcast_to(np.asarray(n_trials), (3,))[GROUPS]
    a = (
        np.full(3, np.inf)
        if concentration is None
        else np.broadcast_to(np.asarray(concentration, dtype=float), (3,))
    )
    return draws, p, n, a[GROUPS]


@pytest.mark.parametrize("name", ["poisson", "bernoulli", "gaussian"])
def test_draws_have_the_family_mean(name: str) -> None:
    """At a fixed ``eta``, the mean of 20,000 draws of each row is the
    family's mean within 4 standard errors (the draws' SD over sqrt(20,000)):
    ``exp(eta + offset)``, ``expit(eta)`` and ``eta``. The softmax's are
    tested below, for each noise model."""
    rng = np.random.default_rng(11)
    if name == "poisson":
        lik, aux = Poisson(), np.log([0.5, 1.0, 2.0, 3.0, 0.2, 1.0, 4.0])
        mean = np.exp(ETA + aux)
    elif name == "bernoulli":
        lik, aux = Bernoulli(), None
        mean = 1.0 / (1.0 + np.exp(-ETA))
    else:
        lik, aux = Gaussian(), 0.7
        mean = ETA
    draws = np.stack([draw(lik, ETA, aux, rng) for _ in range(N_DRAWS)])
    assert draws.shape == (N_DRAWS, ETA.size)
    se = draws.std(axis=0, ddof=1) / np.sqrt(N_DRAWS)
    assert np.all(np.abs(draws.mean(axis=0) - mean) < 4.0 * se)


@pytest.mark.parametrize("name", list(SOFTMAX))
def test_softmax_draws_have_the_model_mean_and_variance(name: str) -> None:
    """Each noise model of the grouped softmax: the mean share of each row is
    the softmax ``p`` within 4 standard errors, and its variance is ``p (1 -
    p) (n + a) / (n (1 + a))`` (``a`` infinite without a concentration:
    ``p (1 - p) / n``) within 4 standard errors of the sample variance. The
    shares of each group sum to one and are counts over ``n``."""
    draws, p, n, a = _softmax_draws(name)
    se = draws.std(axis=0, ddof=1) / np.sqrt(N_DRAWS)
    assert np.all(np.abs(draws.mean(axis=0) - p) < 4.0 * se)
    with np.errstate(invalid="ignore"):
        ratio = np.where(np.isinf(a), 1.0 / n, (n + a) / (n * (1.0 + a)))
    variance = p * (1 - p) * ratio
    squared = (draws - p) ** 2
    variance_se = squared.std(axis=0, ddof=1) / np.sqrt(N_DRAWS)
    assert np.all(np.abs(squared.mean(axis=0) - variance) < 4.0 * variance_se)
    totals = np.stack([np.bincount(GROUPS, weights=row) for row in draws])
    np.testing.assert_allclose(totals, 1.0, rtol=0, atol=1e-12)
    counts = draws * n
    np.testing.assert_allclose(counts, np.round(counts), rtol=0, atol=1e-9)


def test_overdispersion_widens_the_shares() -> None:
    """At the same ``n``, a smaller concentration gives a larger variance,
    and a very large one gives the multinomial's."""
    rng = np.random.default_rng(3)

    def variance(concentration):
        draws = np.stack(
            [
                draw(
                    GroupedSoftmax(),
                    ETA,
                    GROUPS,
                    rng,
                    n_trials=10,
                    concentration=concentration,
                )
                for _ in range(5000)
            ]
        )
        return draws.var(axis=0)[0]

    small, large, none = variance(1.0), variance(1e6), variance(None)
    assert small > 2 * none
    assert large == pytest.approx(none, rel=0.15)


def test_softmax_draw_refuses_bad_noise_parameters() -> None:
    """A number of choices below one or not whole, a concentration not
    positive, and a per-group value of the wrong length are refused."""
    rng = np.random.default_rng(0)
    for kwargs, match in (
        ({"n_trials": 0}, "whole numbers of at least one"),
        ({"n_trials": 2.5}, "whole numbers of at least one"),
        ({"n_trials": np.array([1, 2])}, "one per group"),
        ({"n_trials": 3, "concentration": 0.0}, "finite and positive"),
    ):
        with pytest.raises(ValueError, match=match):
            draw(GroupedSoftmax(), ETA, GROUPS, rng, **kwargs)


def test_softmax_draw_refuses_a_group_without_weight() -> None:
    """A group whose every ``eta`` is minus infinity (every share zero) has
    no share to draw: it is refused, not given to its first row."""
    eta = np.array([0.1, 0.2, -np.inf, -np.inf])
    with pytest.raises(ValueError, match="group 1 has no finite eta"):
        draw(GroupedSoftmax(), eta, np.array([0, 0, 1, 1]), np.random.default_rng(0))


def test_a_draw_refuses_an_unknown_family() -> None:
    """A likelihood without a draw is refused by name."""

    class Other:
        name = "other"

    with pytest.raises(ValueError, match="no draw for the likelihood 'other'"):
        draw(Other(), np.zeros(2), None, np.random.default_rng(0))
