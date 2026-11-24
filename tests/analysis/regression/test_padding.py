"""``regression.padding``: padded rows and groups change no likelihood,
gradient or information, and give no NaN.

The likelihoods' kernels run on padded rows (each row's terms times its
weight) and on the same rows unpadded (every weight 1): the log-likelihood,
the coefficient gradient, the information and its two directional
derivatives agree, and every value is finite, also for a softmax with groups
that have no rows. So do REML's derivative kernels (the score, ``A^-1``, the
edf, and the gradient and Hessian in the log penalties). The profiled
Gaussian's scale and score count the real rows only.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from continuum_swr_replay.analysis.regression import likelihoods, reml
from continuum_swr_replay.analysis.regression.likelihoods import (
    Bernoulli,
    Gaussian,
    GroupedSoftmax,
    Groups,
    Poisson,
)
from continuum_swr_replay.analysis.regression.padding import (
    GROUP_BUCKET,
    ROW_BUCKET,
    SMALLEST_ROW_BUCKET,
    SOFTMAX_ROW_BUCKET,
    bucket_size,
    pad_groups,
    pad_rows,
    pad_values,
)

from .conftest import family_kernel

RTOL = 1e-12
"""Relative to the largest entry of each quantity: an entry that is 0 but for
rounding (a gradient at a column's sum) cannot be compared on its own scale."""
PHI = 1.7
N_COLUMNS = 5


def _design(rng: np.random.Generator, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.column_stack([np.ones(n_rows), rng.normal(size=(n_rows, N_COLUMNS - 1))])
    beta = rng.normal(scale=0.4, size=N_COLUMNS)
    return X, beta


def _diagonal_case(family: str, n_rows: int = 300):
    """``(lik, X, beta, y, aux)`` of one family, with fixed seeds."""
    rng = np.random.default_rng(7)
    X, beta = _design(rng, n_rows)
    eta = X @ beta
    if family == "poisson":
        offset = rng.normal(scale=0.5, size=n_rows)
        return Poisson(), X, beta, rng.poisson(np.exp(eta + offset)) * 1.0, offset
    if family == "bernoulli":
        y = (rng.random(n_rows) < 1 / (1 + np.exp(-eta))) * 1.0
        return Bernoulli(), X, beta, y, None
    return Gaussian(), X, beta, eta + rng.normal(size=n_rows), PHI


def _softmax_case(n_groups: int, seed: int = 3):
    """``(X, beta, y, group ids)``: 2 to 9 rows per group, shares summing to 1,
    group ids sorted but not 0 to G - 1."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(2, 10, size=n_groups)
    ids = np.repeat(np.sort(rng.choice(10 * n_groups, n_groups, replace=False)), sizes)
    X, beta = _design(rng, ids.size)
    y = rng.random(ids.size)
    y /= np.repeat(np.add.reduceat(y, np.r_[0, np.cumsum(sizes)[:-1]]), sizes)
    return X, beta, y, ids


def _kernels(lik, X, y, aux, weight, beta, d1, d2):
    """``(loglik, gradient, information, dinformation, d2information)`` from
    the family's kernels on the given rows; every per-row value finite."""
    with jax.enable_x64(True):
        X, y, weight = jnp.asarray(X), jnp.asarray(y), jnp.asarray(weight)
        aux = jax.tree_util.tree_map(jnp.asarray, aux)
        eta = X @ beta
        if isinstance(aux, Groups):
            _, norm, p, total = lik._jax_parts(eta, y, aux, weight)
            assert bool(jnp.all(jnp.isfinite(p)))
            assert bool(jnp.all(jnp.isfinite(total)))
        score = lik._jax_score(eta, y, aux, weight)
        assert bool(jnp.all(jnp.isfinite(score)))
        return [
            np.asarray(v)
            for v in (
                lik._jax_loglik(eta, y, aux, weight),
                X.T @ score,
                lik._jax_information(X, eta, y, aux, weight),
                lik._jax_dinformation(X, eta, y, aux, weight, d1),
                lik._jax_d2information(X, eta, y, aux, weight, d1, d2),
            )
        ]


def _assert_equal(padded, reference):
    for got, want in zip(padded, reference, strict=True):
        assert np.all(np.isfinite(got))
        scale = np.max(np.abs(want))
        np.testing.assert_allclose(got, want, rtol=0, atol=RTOL * scale)


def test_bucket_size_is_a_power_of_two_below_the_step_then_the_next_multiple():
    sizes = (1, 3, 4, 5, 8, 9, 16, 17)
    assert [bucket_size(n, 16, 4) for n in sizes] == [4, 4, 4, 8, 8, 16, 16, 32]
    assert [bucket_size(n, 12, 1) for n in (1, 3, 7, 9, 12, 13)] == [
        1,
        4,
        8,
        12,
        12,
        24,
    ]
    assert [bucket_size(n, 4, 4) for n in (1, 3, 4, 5, 8, 9)] == [4, 4, 4, 8, 8, 12]
    for bad in [(0, 4, 4), (1, 0, 4), (1, 4, 0)]:
        with pytest.raises(ValueError):
            bucket_size(*bad)


def test_the_row_buckets_of_each_family():
    """Diagonal families: 256, 512, 1,024, 2,048, then multiples of 4,096; the
    softmax's rows: 256, 512, then multiples of 1,024; its groups: multiples
    of 64."""
    assert (ROW_BUCKET, SOFTMAX_ROW_BUCKET, SMALLEST_ROW_BUCKET) == (4096, 1024, 256)
    rows = (1, 256, 257, 600, 2049, 4096, 4097, 36724)
    assert [bucket_size(n, ROW_BUCKET, SMALLEST_ROW_BUCKET) for n in rows] == [
        256,
        256,
        512,
        1024,
        4096,
        4096,
        8192,
        36864,
    ]
    assert [bucket_size(n, SOFTMAX_ROW_BUCKET, SMALLEST_ROW_BUCKET) for n in rows] == [
        256,
        256,
        512,
        1024,
        3072,
        4096,
        5120,
        36864,
    ]
    assert [pad_rows(np.zeros((n, 1)), np.zeros(n)).y.size for n in (0, 300)] == [
        256,
        512,
    ]


def test_padded_rows_follow_the_real_ones_with_zeros():
    rng = np.random.default_rng(0)
    X, y, offset = rng.normal(size=(5, 2)), rng.random(5), rng.normal(size=5)

    padded = pad_rows(X, y, offset, step=4)

    assert padded.X.shape == (8, 2) and padded.n_real == 5
    np.testing.assert_array_equal(padded.X[:5], X)
    np.testing.assert_array_equal(padded.y[:5], y)
    np.testing.assert_array_equal(padded.offset[:5], offset)
    np.testing.assert_array_equal(padded.weight, [1, 1, 1, 1, 1, 0, 0, 0])
    assert not padded.X[5:].any() and not padded.y[5:].any()
    assert not padded.offset[5:].any()
    with pytest.raises(ValueError, match="one row each"):
        pad_rows(X, y[:4])


def test_padded_groups_number_the_groups_and_put_padding_in_one_more():
    X = np.ones((5, 1))
    padded = pad_groups(X, np.ones(5), np.array([10, 10, 12, 30, 30]), row_step=4)

    np.testing.assert_array_equal(padded.group, [0, 0, 1, 2, 2, 3, 3, 3])
    assert (padded.n_real_groups, padded.n_groups) == (3, GROUP_BUCKET)
    np.testing.assert_array_equal(padded.weight, [1] * 5 + [0] * 3)
    with pytest.raises(ValueError, match="sorted"):
        pad_groups(X, np.ones(5), np.array([1, 0, 0, 2, 2]))


@pytest.mark.parametrize("family", ["poisson", "bernoulli", "gaussian"])
@pytest.mark.parametrize("step", [ROW_BUCKET, 300, 7])
def test_a_padded_diagonal_family_equals_the_unpadded_fit(family, step):
    """``step = 300`` pads nothing; 7 pads 1 row; the bucket pads 212 (to 512)."""
    lik, X, beta, y, aux = _diagonal_case(family)
    rng = np.random.default_rng(1)
    d1, d2 = rng.normal(size=(2, y.size))
    padded = pad_rows(X, y, aux if family == "poisson" else None, step=step)
    padded_aux = padded.offset if family == "poisson" else aux
    n = padded.y.size

    _assert_equal(
        _kernels(
            lik,
            padded.X,
            padded.y,
            padded_aux,
            padded.weight,
            beta,
            pad_values(d1, y.size, n),
            pad_values(d2, y.size, n),
        ),
        _kernels(lik, X, y, aux, np.ones(y.size), beta, d1, d2),
    )


@pytest.mark.parametrize(
    ("n_groups", "row_step"),
    [
        (40, SOFTMAX_ROW_BUCKET),  # a padding group and empty groups after it
        (63, SOFTMAX_ROW_BUCKET),  # the padding group is the last of the bucket
        (40, None),  # rows fill the bucket: the padding group has no rows
    ],
)
def test_a_padded_softmax_equals_the_unpadded_fit(n_groups, row_step):
    """Every case has a group with no rows (``n_g = 0``, ``sum exp`` 0): the
    groups after the padding group, or the padding group of the unpadded
    rows. The guarded ``log`` keeps every value finite."""
    X, beta, y, ids = _softmax_case(n_groups)
    padded = pad_groups(X, y, ids, row_step=row_step or y.size)
    lik = GroupedSoftmax()
    n_padded_rows = padded.y.size - padded.n_real
    assert (n_padded_rows == 0) == (row_step is None)
    assert padded.n_groups > padded.n_real_groups
    rng = np.random.default_rng(2)
    d1, d2 = rng.normal(size=(2, y.size))
    n = padded.y.size
    unpadded = pad_groups(X, y, ids, row_step=y.size, group_step=1)
    assert unpadded.n_groups == n_groups + 1  # one padding group, with no rows

    _assert_equal(
        _kernels(
            lik,
            padded.X,
            padded.y,
            Groups(padded.group, padded.n_groups),
            padded.weight,
            beta,
            pad_values(d1, y.size, n),
            pad_values(d2, y.size, n),
        ),
        _kernels(
            lik,
            X,
            y,
            Groups(unpadded.group, n_groups),
            np.ones(y.size),
            beta,
            d1,
            d2,
        ),
    )


def _numpy_reference(family, X, beta, y, aux):
    """``(loglik, gradient, information)`` on the unpadded rows, from the
    formulas, in plain NumPy."""
    eta = X @ beta
    if family == "poisson":
        mu = np.exp(eta + aux)
        loglik, score, w = y @ (eta + aux) - mu.sum(), y - mu, mu
    elif family == "bernoulli":
        mu = 1.0 / (1.0 + np.exp(-eta))
        loglik = y @ eta - np.log1p(np.exp(eta)).sum()
        score, w = y - mu, mu * (1.0 - mu)
    elif family == "gaussian":
        r = y - eta
        loglik = -(r @ r) / (2 * aux) - 0.5 * y.size * np.log(2 * np.pi * aux)
        score, w = r / aux, np.full(y.size, 1.0 / aux)
    else:  # grouped softmax, aux the group ids
        starts = np.flatnonzero(np.r_[True, aux[1:] != aux[:-1]])
        sizes = np.diff(starts, append=y.size)
        e = np.exp(eta)
        p = e / np.repeat(np.add.reduceat(e, starts), sizes)
        total = np.add.reduceat(y, starts)
        loglik = y @ np.log(p)
        score = y - np.repeat(total, sizes) * p
        Xp = np.add.reduceat(p[:, None] * X, starts)
        information = (X.T * (np.repeat(total, sizes) * p)) @ X - (Xp.T * total) @ Xp
        return loglik, X.T @ score, information
    return loglik, X.T @ score, (X.T * w) @ X


@pytest.mark.parametrize("family", ["poisson", "bernoulli", "gaussian", "softmax"])
def test_the_padded_kernels_equal_the_numpy_formulas(family):
    """The kernels on padded rows against the log-likelihood, gradient and
    information written here in NumPy on the unpadded rows."""
    rng = np.random.default_rng(5)
    if family == "softmax":
        X, beta, y, ids = _softmax_case(40)
        padded = pad_groups(X, y, ids)
        lik, aux, padded_aux = (
            GroupedSoftmax(),
            ids,
            Groups(padded.group, padded.n_groups),
        )
    else:
        lik, X, beta, y, aux = _diagonal_case(family)
        padded = pad_rows(X, y, aux if family == "poisson" else None)
        padded_aux = padded.offset if family == "poisson" else aux
    d = pad_values(rng.normal(size=y.size), y.size, padded.y.size)
    got = _kernels(lik, padded.X, padded.y, padded_aux, padded.weight, beta, d, d)
    _assert_equal(got[:3], _numpy_reference(family, X, beta, y, aux))


@pytest.mark.parametrize("family", ["poisson", "bernoulli", "gaussian", "softmax"])
def test_junk_in_the_padded_rows_changes_nothing(family):
    """Padded rows filled with junk (a random design, a response of 5, an
    offset of 2; for the softmax, in the padding group) change no quantity:
    each padded row is weighted 0."""
    rng = np.random.default_rng(6)
    if family == "softmax":
        X, beta, y, ids = _softmax_case(40)
        padded = pad_groups(X, y, ids)
        lik, aux = GroupedSoftmax(), Groups(padded.group, padded.n_groups)
    else:
        lik, X, beta, y, aux = _diagonal_case(family)
        padded = pad_rows(X, y, aux if family == "poisson" else None)
        aux = padded.offset if family == "poisson" else aux
    n, n_bucket = padded.n_real, padded.y.size
    d1, d2 = (pad_values(rng.normal(size=n), n, n_bucket) for _ in range(2))
    clean = _kernels(lik, padded.X, padded.y, aux, padded.weight, beta, d1, d2)

    junk_X, junk_y = padded.X.copy(), padded.y.copy()
    junk_X[n:] = rng.normal(size=(n_bucket - n, N_COLUMNS))
    junk_y[n:] = 5.0
    if family == "poisson":
        aux = aux.copy()
        aux[n:] = 2.0
    junk_d1, junk_d2 = d1.copy(), d2.copy()
    junk_d1[n:], junk_d2[n:] = rng.normal(size=(2, n_bucket - n))
    junk = _kernels(lik, junk_X, junk_y, aux, padded.weight, beta, junk_d1, junk_d2)
    for got, want in zip(junk, clean, strict=True):
        np.testing.assert_array_equal(got, want)


def test_the_public_method_pads_and_returns_the_real_rows():
    """The weighted design, and the score on padded rows, have one value per
    real row."""
    lik, X, beta, y, aux = _diagonal_case("poisson")
    eta = X @ beta
    score = family_kernel(lik, "score", None, eta, y, aux)
    assert score.shape == y.shape
    np.testing.assert_allclose(score, y - np.exp(eta + aux), rtol=1e-14)
    weighted = lik.weighted_design(X, y, aux, eta)
    assert weighted.shape == X.shape


@pytest.mark.parametrize("step", [ROW_BUCKET, 7])
def test_the_profiled_gaussian_counts_the_real_rows(step):
    """The scale ``D_p / (n - M_p)``, the score and the log-likelihood at the
    scale do not move with the padding."""
    lik, X, beta, y, _ = _diagonal_case("gaussian")
    S_j = np.stack([np.diag([0.0, 2.0, 2.0, 0.0, 0.0]), np.diag([0.0] * 3 + [3.0] * 2)])
    masks = (S_j.diagonal(axis1=1, axis2=2) > 0) * 1.0

    def summary(padded):
        with jax.enable_x64(True):
            return jax.device_get(
                reml._profiled_gaussian(
                    padded.X,
                    padded.y[None],
                    padded.weight,
                    S_j.sum(axis=0),
                    beta,
                    masks,
                    np.float64(0.0),
                    np.int64(y.size),
                    np.int64(y.size - 1),
                    S_j,
                    np.array([2.0, 2.0]),
                )
            )

    _assert_equal(summary(pad_rows(X, y, step=step)), summary(pad_rows(X, y, step=300)))
    assert likelihoods.Gaussian.quadratic


def _two_penalties() -> tuple[np.ndarray, np.ndarray]:
    """Two penalties, on columns 1-2 and 3-4, shape ``(2, p, p)``, and their
    column masks, shape ``(2, p)``."""
    S_j = np.zeros((2, N_COLUMNS, N_COLUMNS))
    S_j[0, 1:3, 1:3] = [[1.6, -0.8], [-0.8, 1.6]]
    S_j[1, 3:, 3:] = 1.5 * np.eye(2)
    return S_j, (S_j.diagonal(axis1=1, axis2=2) > 0) * 1.0


def _reml_derivatives(lik, X, y, aux, weight, beta, A):
    """The outputs of REML's derivative kernel at ``beta`` on the given rows
    (score, covariance, edf, its total, gradient, Hessian), with the given
    ``A``, so that only the rows differ between two calls."""
    S_j, masks = _two_penalties()
    with jax.enable_x64(True):
        aux = jax.tree_util.tree_map(jnp.asarray, aux)
        outputs = reml._laplace_derivatives(
            lik,
            jnp.asarray(X),
            jnp.asarray(y)[None],
            aux,
            jnp.asarray(weight),
            S_j,
            A,
            S_j.sum(axis=0),
            beta,
            masks,
            np.float64(-3.0),
            np.float64(0.5),
            np.float64(1.0),
            np.array([2.0, 2.0]),
        )
        return [np.asarray(v) for v in outputs]


def _derivative_cases(family: str, step: int | None):
    """``(lik, beta, A, unpadded, padded)``, each set of rows ``(X, y, aux,
    weight)``. ``step`` is the row step; for the softmax None fills the row
    bucket, so that the padding group has no rows."""
    if family == "softmax":
        X, beta, y, ids = _softmax_case(40)
        lik = GroupedSoftmax()
        padded = pad_groups(X, y, ids, row_step=step or y.size)
        unpadded = pad_groups(X, y, ids, row_step=y.size, group_step=1)
        rows = (X, y, Groups(unpadded.group, 40), np.ones(y.size))
        aux = Groups(padded.group, padded.n_groups)
        information = family_kernel(lik, "information", X, X @ beta, y, ids)
    else:
        lik, X, beta, y, aux = _diagonal_case(family)
        padded = pad_rows(X, y, aux if family == "poisson" else None, step=step)
        rows = (X, y, aux, np.ones(y.size))
        information = family_kernel(lik, "information", X, X @ beta, y, aux)
        aux = padded.offset if family == "poisson" else aux
    # The first column is constant, which the softmax cannot see: a unit on
    # its diagonal keeps A positive definite.
    A = information + _two_penalties()[0].sum(axis=0) + np.diag(np.eye(N_COLUMNS)[0])
    return lik, beta, A, rows, (padded.X, padded.y, aux, padded.weight)


@pytest.mark.parametrize(
    ("family", "step"),
    [
        ("poisson", ROW_BUCKET),
        ("poisson", 7),
        ("bernoulli", ROW_BUCKET),
        ("bernoulli", 7),
        ("softmax", SOFTMAX_ROW_BUCKET),
        ("softmax", None),
    ],
)
def test_the_reml_derivatives_do_not_move_with_the_padding(family, step):
    """REML's derivative kernel on padded rows equals it on the rows unpadded,
    for the score, ``A^-1``, the edf and the gradient and Hessian in ``rho``;
    the softmax's cases have groups with no rows."""
    lik, beta, A, rows, padded = _derivative_cases(family, step)
    _assert_equal(
        _reml_derivatives(lik, *padded, beta, A),
        _reml_derivatives(lik, *rows, beta, A),
    )


@pytest.mark.parametrize("family", ["poisson", "bernoulli", "softmax", "gaussian"])
def test_junk_in_the_padded_rows_changes_no_reml_derivative(family):
    """Padded rows filled with junk (a random design, a response of 5, an
    offset of 2) leave REML's derivative kernels exactly as they were: the
    Laplace families' and the profiled Gaussian's."""
    rng = np.random.default_rng(8)
    if family == "gaussian":
        lik, X, beta, y, _ = _diagonal_case(family)
        padded = pad_rows(X, y)
        rows = [padded.X, padded.y, None, padded.weight]
        S_j, masks = _two_penalties()

        def outputs(X, y, aux, weight):
            with jax.enable_x64(True):
                return jax.device_get(
                    reml._profiled_gaussian(
                        X,
                        y[None],
                        weight,
                        S_j.sum(axis=0),
                        beta,
                        masks,
                        np.float64(0.0),
                        np.int64(padded.n_real),
                        np.int64(padded.n_real - 1),
                        S_j,
                        np.array([2.0, 2.0]),
                    )
                )
    else:
        lik, beta, A, _, rows = _derivative_cases(family, ROW_BUCKET)
        rows = list(rows)

        def outputs(X, y, aux, weight):
            return _reml_derivatives(lik, X, y, aux, weight, beta, A)

    clean = outputs(*rows)
    X, y, aux, weight = rows
    n, n_bucket = int(weight.sum()), y.size
    assert n < n_bucket
    X, y = X.copy(), y.copy()
    X[n:] = rng.normal(size=(n_bucket - n, N_COLUMNS))
    y[n:] = 5.0
    if family == "poisson":
        aux = aux.copy()
        aux[n:] = 2.0
    for got, want in zip(outputs(X, y, aux, weight), clean, strict=True):
        np.testing.assert_array_equal(got, want)


def test_an_offset_of_minus_infinity_would_give_nan():
    """Why a padded row's offset is 0 and its weight carries the mask: a zero
    weight on an offset of minus infinity is ``0 * -inf``."""
    with jax.enable_x64(True):
        offset = jnp.array([0.0, -jnp.inf])
        weight, y, eta = jnp.array([1.0, 0.0]), jnp.zeros(2), jnp.zeros(2)
        assert jnp.isnan(jnp.sum(weight * (y * (eta + offset) - jnp.exp(eta + offset))))
        padded = pad_rows(np.zeros((1, 1)), np.zeros(1), np.zeros(1), step=2)
        offset = jnp.asarray(padded.offset)
        terms = jnp.asarray(padded.y) * offset - jnp.exp(offset)
        assert jnp.isfinite(jnp.sum(jnp.asarray(padded.weight) * terms))
