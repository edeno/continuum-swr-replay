"""``regression.padding``: padded rows and groups change no likelihood,
gradient or information, and give no NaN.

The likelihoods' kernels run on padded rows (each row's terms times its
weight) and on the same rows unpadded (every weight 1): the log-likelihood,
the coefficient gradient, the information and its two directional
derivatives agree, and every value is finite, also for a softmax with groups
that have no rows. The profiled Gaussian's scale and score count the real rows
only.
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
    SOFTMAX_ROW_BUCKET,
    bucket_size,
    pad_groups,
    pad_rows,
    pad_values,
)

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


def test_bucket_size_is_the_next_multiple():
    assert [bucket_size(n, 4) for n in (1, 3, 4, 5, 8, 9)] == [4, 4, 4, 8, 8, 12]
    with pytest.raises(ValueError):
        bucket_size(0, 4)


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
    assert ROW_BUCKET == 4096
    with pytest.raises(ValueError, match="one row each"):
        pad_rows(X, y[:4])


def test_padded_groups_number_the_groups_and_put_padding_in_one_more():
    X = np.ones((5, 1))
    padded = pad_groups(X, np.ones(5), np.array([10, 10, 12, 30, 30]), row_step=4)

    np.testing.assert_array_equal(padded.group, [0, 0, 1, 2, 2, 3, 3, 3])
    assert (padded.n_real_groups, padded.n_groups) == (3, GROUP_BUCKET)
    np.testing.assert_array_equal(padded.weight, [1] * 5 + [0] * 3)
    assert SOFTMAX_ROW_BUCKET == 1024
    with pytest.raises(ValueError, match="sorted"):
        pad_groups(X, np.ones(5), np.array([1, 0, 0, 2, 2]))


@pytest.mark.parametrize("family", ["poisson", "bernoulli", "gaussian"])
@pytest.mark.parametrize("step", [ROW_BUCKET, 300, 7])
def test_a_padded_diagonal_family_equals_the_unpadded_fit(family, step):
    """``step = 300`` pads nothing; 7 pads 1 row; the bucket pads 3,796."""
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
            pad_values(d1, n),
            pad_values(d2, n),
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
            pad_values(d1, n),
            pad_values(d2, n),
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


def test_the_public_methods_pad_and_return_the_real_rows():
    """A public method pads to the bucket and returns one value per real row."""
    lik, X, beta, y, aux = _diagonal_case("poisson")
    eta = X @ beta
    score = lik.score_eta(eta, y, aux)
    assert score.shape == y.shape
    np.testing.assert_allclose(score, y - np.exp(eta + aux), rtol=1e-14)
    weighted = lik.weighted_design(X, y, aux, eta)
    assert weighted.shape == X.shape


@pytest.mark.parametrize("step", [ROW_BUCKET, 7])
def test_the_profiled_gaussian_counts_the_real_rows(step):
    """The scale ``D_p / (n - M_p)``, the score and the log-likelihood at the
    scale do not move with the padding."""
    lik, X, beta, y, _ = _diagonal_case("gaussian")
    S = np.diag([0.0, 2.0, 2.0, 2.0, 2.0])
    masks = np.zeros((1, N_COLUMNS))
    masks[0, 1:] = 1.0

    def summary(padded):
        with jax.enable_x64(True):
            return jax.device_get(
                reml._profiled_gaussian(
                    padded.X,
                    padded.y,
                    padded.weight,
                    S,
                    beta,
                    masks,
                    np.float64(0.0),
                    np.int64(y.size),
                    np.int64(y.size - 1),
                )
            )

    _assert_equal(summary(pad_rows(X, y, step=step)), summary(pad_rows(X, y, step=300)))
    assert likelihoods.Gaussian.quadratic


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
