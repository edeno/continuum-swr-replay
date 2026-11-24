"""The likelihoods' derivatives against central finite differences, and
their draws.

Each level of a family's JAX functions (``_jax_<name>``, called on NumPy rows
by ``conftest.family_kernel``) is checked against a finite difference of the
level below it, and the lowest against the log-likelihood itself: the score is
the gradient of the log-likelihood in ``eta``; the information is minus the
derivative of the coefficient gradient ``X' score(X beta)`` in ``beta``;
``dinformation`` is the derivative of the information along a direction of
``eta``; and ``d2information`` the derivative of ``dinformation`` along a
second one. The hand-written derivatives also equal JAX's automatic
derivatives of the log-likelihood. The draws (``draw``) have the family's
mean.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from continuum_swr_replay.analysis.regression.likelihoods import (
    Bernoulli,
    Gaussian,
    GroupedSoftmax,
    Poisson,
    draw,
    mean,
)
from continuum_swr_replay.analysis.regression.newton import penalized_newton
from continuum_swr_replay.analysis.regression.padding import pad_values

from .conftest import family_kernel, numeric_gradient, numeric_hessian, relative_error

TOLERANCE = 1e-6
N_ROWS = 14
N_COLUMNS = 3


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

    def loglik(e: np.ndarray) -> float:
        return family_kernel(lik, "loglik", None, e, y, aux)

    def score(e: np.ndarray) -> np.ndarray:
        return family_kernel(lik, "score", None, e, y, aux)

    def information(e: np.ndarray) -> np.ndarray:
        return family_kernel(lik, "information", X, e, y, aux)

    def dinformation(e: np.ndarray, d: np.ndarray) -> np.ndarray:
        return family_kernel(lik, "dinformation", X, e, y, aux, d)

    def d2information(e: np.ndarray, d: np.ndarray, d_: np.ndarray) -> np.ndarray:
        return family_kernel(lik, "d2information", X, e, y, aux, d, d_)

    # The score: the gradient of the log-likelihood in eta.
    numeric_score = numeric_gradient(loglik, eta, 1e-5)
    assert relative_error(score(eta), numeric_score) < TOLERANCE

    # The information: minus the beta-derivative of X' score(X beta).
    beta = np.linalg.lstsq(X, eta, rcond=None)[0]
    eta = X @ beta  # on the column space, so eta is X beta exactly

    def gradient(b: np.ndarray) -> np.ndarray:
        return X.T @ score(X @ b)

    numeric_information = -np.column_stack(numeric_gradient(gradient, beta, 1e-5))
    assert information(eta).shape == (N_COLUMNS, N_COLUMNS)
    assert relative_error(information(eta), numeric_information) < TOLERANCE

    # dinformation: the derivative of the information along d1.
    h = 1e-5
    numeric_d = (information(eta + h * d1) - information(eta - h * d1)) / (2 * h)
    assert relative_error(dinformation(eta, d1), numeric_d) < TOLERANCE

    # d2information: the derivative of dinformation(d1) along d2.
    numeric_d2 = (dinformation(eta + h * d2, d1) - dinformation(eta - h * d2, d1)) / (
        2 * h
    )
    analytic_d2 = d2information(eta, d1, d2)
    assert relative_error(analytic_d2, numeric_d2) < TOLERANCE
    # Symmetric in its two directions.
    assert np.allclose(analytic_d2, d2information(eta, d2, d1))


AUTODIFF_RTOL = 1e-12
"""Relative to the largest entry of each quantity (the Gaussian's derivatives
of the information are exactly zero)."""


def _assert_close(hand, autodiff) -> None:
    hand, autodiff = np.asarray(hand), np.asarray(autodiff)
    scale = np.max(np.abs(autodiff))
    np.testing.assert_allclose(hand, autodiff, rtol=0, atol=AUTODIFF_RTOL * scale)


@pytest.mark.parametrize("name", CASES)
def test_hand_derivatives_equal_autodiff(name: str) -> None:
    """On the padded rows the kernels see: the score is ``jax.grad`` of the
    log-likelihood in ``eta``, the information minus ``jax.hessian`` of it in
    ``beta``, ``dinformation`` the ``jax.jvp`` of the information along a
    direction of ``eta``, and ``d2information`` the nested ``jvp``."""
    lik, X, y, aux, _ = _case(name)
    rng = np.random.default_rng(13)
    beta = rng.normal(scale=0.5, size=N_COLUMNS)
    rows = lik.pad(X, y, aux)
    d1, d2 = (
        pad_values(rng.normal(size=N_ROWS), N_ROWS, rows.y.size) for _ in range(2)
    )
    with jax.enable_x64(True):
        Xp, yp, w = jnp.asarray(rows.X), jnp.asarray(rows.y), jnp.asarray(rows.weight)
        a = jax.tree_util.tree_map(jnp.asarray, rows.aux)
        eta = Xp @ beta

        def loglik_eta(e):
            return lik._jax_loglik(e, yp, a, w)

        def information(e):
            return lik._jax_information(Xp, e, yp, a, w)

        def dinformation(e):
            return jax.jvp(information, (e,), (d1,))[1]

        _assert_close(lik._jax_score(eta, yp, a, w), jax.grad(loglik_eta)(eta))
        _assert_close(
            information(eta), -jax.hessian(lambda b: loglik_eta(Xp @ b))(beta)
        )
        _assert_close(lik._jax_dinformation(Xp, eta, yp, a, w, d1), dinformation(eta))
        _assert_close(
            lik._jax_d2information(Xp, eta, yp, a, w, d1, d2),
            jax.jvp(dinformation, (eta,), (d2,))[1],
        )


@pytest.mark.parametrize("name", CASES)
def test_information_traces_equal_the_traces_of_the_derivatives(name: str) -> None:
    """``_jax_information_traces`` against a symmetric ``M``: each trace
    equals ``sum(M * dinformation)`` (or ``d2information``) of that direction,
    made in full, on the padded rows; for the diagonal families it is read
    from ``h_i = x_i' M x_i`` without making them. For the grouped softmax
    the function is the generic one, which runs the same loops as this test:
    there it checks only the symmetry of the second traces; the check of the
    traces' values is of the diagonal families' shortcut."""
    lik, X, y, aux, _ = _case(name)
    rng = np.random.default_rng(17)
    beta = rng.normal(scale=0.5, size=N_COLUMNS)
    rows = lik.pad(X, y, aux)
    n = rows.y.size
    D, E = (
        np.column_stack(
            [pad_values(rng.normal(size=N_ROWS), N_ROWS, n) for _ in range(k)]
        )
        for k in (3, 2)
    )
    root = rng.normal(size=(N_COLUMNS, N_COLUMNS))
    M = root @ root.T
    with jax.enable_x64(True):
        Xp, yp, w = jnp.asarray(rows.X), jnp.asarray(rows.y), jnp.asarray(rows.weight)
        a = jax.tree_util.tree_map(jnp.asarray, rows.aux)
        eta = Xp @ beta
        first, second = lik._jax_information_traces(Xp, eta, yp, a, w, M, D, E)
        want_first = [
            jnp.sum(M * lik._jax_dinformation(Xp, eta, yp, a, w, D[:, i]))
            for i in range(3)
        ]
        want_second = [
            [
                jnp.sum(M * lik._jax_d2information(Xp, eta, yp, a, w, E[:, j], E[:, k]))
                for k in range(2)
            ]
            for j in range(2)
        ]
    _assert_close(first, np.array(want_first))
    _assert_close(second, np.array(want_second))
    np.testing.assert_array_equal(second, second.T)


def test_softmax_information_is_minus_the_hessian_of_loglik() -> None:
    """The grouped information equals a direct second difference of the
    log-likelihood."""
    lik, X, y, groups, eta = _case("softmax_counts")
    beta = np.linalg.lstsq(X, eta, rcond=None)[0]

    def loglik(b: np.ndarray) -> float:
        return family_kernel(lik, "loglik", None, X @ b, y, groups)

    hessian = numeric_hessian(loglik, beta, 1e-4)
    information = family_kernel(lik, "information", X, X @ beta, y, groups)
    assert relative_error(information, -hessian) < 1e-5


def test_softmax_rejects_unsorted_groups() -> None:
    lik, X, y, groups, eta = _case("softmax_fractions")
    with pytest.raises(ValueError, match="sorted"):
        family_kernel(lik, "loglik", None, eta, y, groups[::-1])


def test_softmax_is_invariant_to_a_constant_within_each_group() -> None:
    """A group-level constant cancels: it is not identified, as the class says."""
    lik, X, y, groups, eta = _case("softmax_fractions")
    shift = np.array([3.0, -1.0, 0.5, 20.0])[groups]
    assert np.isclose(
        family_kernel(lik, "loglik", None, eta + shift, y, groups),
        family_kernel(lik, "loglik", None, eta, y, groups),
    )


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


DIAGONAL = [("poisson", np.empty(0)), ("bernoulli", None), ("gaussian", 0.7)]
FAMILIES = {"poisson": Poisson, "bernoulli": Bernoulli, "gaussian": Gaussian}


@pytest.mark.parametrize(("name", "aux"), DIAGONAL)
def test_no_rows_give_zeros(name, aux) -> None:
    """With no rows the log-likelihood is 0, the score empty and the
    information and its derivatives zero, as the NumPy formulas gave."""
    lik, X, none = FAMILIES[name](), np.empty((0, 2)), np.empty(0)
    assert family_kernel(lik, "loglik", None, none, none, aux) == 0.0
    assert family_kernel(lik, "score", None, none, none, aux).shape == (0,)
    for information in (
        family_kernel(lik, "information", X, none, none, aux),
        family_kernel(lik, "dinformation", X, none, none, aux, none),
        family_kernel(lik, "d2information", X, none, none, aux, none, none),
    ):
        np.testing.assert_array_equal(information, np.zeros((2, 2)))


def test_a_softmax_without_rows_is_refused() -> None:
    none = np.empty(0)
    with pytest.raises(ValueError, match="no rows"):
        family_kernel(GroupedSoftmax(), "loglik", None, none, none, np.empty(0, int))


@pytest.mark.parametrize("length", [5, 2 * N_ROWS])
def test_a_vector_of_another_length_is_refused(length: int) -> None:
    """``eta`` needs one value per row; the error names the argument and both
    lengths."""
    lik, X, y, aux, eta = _case("poisson")
    wrong = np.resize(eta, length)
    with pytest.raises(ValueError, match=rf"eta of shape \({length},\) for 14 rows"):
        lik.weighted_design(X, y, aux, wrong)


@pytest.mark.parametrize("name", ["poisson", "bernoulli", "softmax_counts"])
def test_a_per_row_vector_with_two_axes_is_refused(name: str) -> None:
    """``eta``, ``y`` and a per-row offset take one axis: a column of shape
    (n, 1) broadcast against the padded rows and gave a wrong number. The
    error names the argument and its shape."""
    lik, X, y, aux, eta = _case(name)
    column = rf"of shape \({N_ROWS}, 1\): one value per row, on one axis"
    with pytest.raises(ValueError, match="eta " + column):
        lik.weighted_design(X, y, aux, eta[:, None])
    with pytest.raises(ValueError, match="y " + column):
        lik.weighted_design(X, y[:, None], aux, eta)
    if name == "poisson":
        with pytest.raises(ValueError, match="offset " + column):
            lik.weighted_design(X, y, aux[:, None], eta)


def test_a_single_poisson_offset_is_every_rows() -> None:
    """A scalar offset is the offset of every row when the rows are padded;
    None is refused (``TypeError``)."""
    lik, X, y, _, eta = _case("poisson")
    for name, design, offset in (
        ("loglik", None, 0.5),
        ("score", None, 0.5),
        ("information", X, 0.0),
    ):
        np.testing.assert_array_equal(
            family_kernel(lik, name, design, eta, y, offset),
            family_kernel(lik, name, design, eta, y, np.full(N_ROWS, offset)),
        )
    np.testing.assert_array_equal(
        lik.weighted_design(X, y, 0.0, eta),
        lik.weighted_design(X, y, np.zeros(N_ROWS), eta),
    )
    with pytest.raises(TypeError, match="offset is None"):
        lik.pad(X, y, None)


def test_a_family_class_that_can_keep_state_is_refused_before_any_call() -> None:
    """A class with an ``__init__`` or a cached property could put a parameter
    on its objects: it is refused at the first call, whatever was read
    before, not only once the attribute exists."""

    class _Cached(Poisson):
        @functools.cached_property
        def scale(self) -> float:
            return 2.0

    class _Initialized(Poisson):
        def __init__(self) -> None:
            pass

    lik, X, y, aux, eta = _case("poisson")
    for family in (_Cached(), _Initialized()):
        with pytest.raises(TypeError, match="can keep state on its objects"):
            family.weighted_design(X, y, aux, eta)
        with pytest.raises(TypeError, match="can keep state on its objects"):
            hash(family)


def test_a_family_with_an_attribute_of_its_own_is_refused() -> None:
    """The kernels are keyed on the family's class, so an attribute set on
    the object would be ignored by a kernel compiled for another object of
    the class: such an object is refused."""

    class _Scaled(Poisson):
        pass

    lik, X, y, aux, eta = _case("poisson")
    scaled = _Scaled()
    np.testing.assert_array_equal(
        scaled.weighted_design(X, y, aux, eta), lik.weighted_design(X, y, aux, eta)
    )
    scaled.scale = 2.0
    with pytest.raises(TypeError, match=r"keeps \['scale'\] on the object"):
        scaled.weighted_design(X, y, aux, eta)
    with pytest.raises(TypeError, match=r"keeps \['scale'\] on the object"):
        penalized_newton(scaled, X, y, aux, np.zeros((3, 3)), np.zeros(3))


def test_the_mean_of_each_family() -> None:
    """The expected outcome at a linear predictor, the mean each draw has:
    ``exp(eta + offset)``, ``expit(eta)``, ``eta``, and under the grouped
    softmax each group's shares."""
    eta = np.array([0.2, -1.0, 0.5, 1.5, 0.0])
    offset = np.log(np.array([1.0, 2.0, 0.5, 1.0, 3.0]))
    np.testing.assert_allclose(
        mean(Poisson(), eta, offset), np.exp(eta) * np.exp(offset)
    )
    np.testing.assert_allclose(mean(Bernoulli(), eta, None), 1 / (1 + np.exp(-eta)))
    np.testing.assert_array_equal(mean(Gaussian(), eta, None), eta)
    groups = np.array([0, 0, 1, 1, 1])
    shares = mean(GroupedSoftmax(), eta, groups)
    first = np.exp(eta[:2]) / np.exp(eta[:2]).sum()
    second = np.exp(eta[2:]) / np.exp(eta[2:]).sum()
    np.testing.assert_allclose(shares, np.r_[first, second], rtol=1e-15)
