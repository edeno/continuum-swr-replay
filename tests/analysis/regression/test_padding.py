"""``regression.padding``: padded rows and groups change no likelihood,
gradient or information, and give no NaN.

The reference is the present NumPy likelihood code on the unpadded rows. Two
checks: the NumPy code itself on the padded rows (a design row of zeros adds
nothing to a gradient or an information; for the grouped softmax nothing at
all), and a weighted JAX form of each family on the padded rows, the form a
kernel takes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from continuum_swr_replay.analysis.regression.likelihoods import (
    Bernoulli,
    Gaussian,
    GroupedSoftmax,
    Poisson,
)
from continuum_swr_replay.analysis.regression.padding import (
    GROUP_BUCKET,
    ROW_BUCKET,
    SOFTMAX_ROW_BUCKET,
    bucket_size,
    pad_groups,
    pad_rows,
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


# The weighted JAX form of each family: each row's term times its weight.


def _diagonal_jax(family, X, eta, y, aux, weight):
    """``(loglik, gradient, information)`` with each row weighted."""
    if family == "poisson":
        mu = jnp.exp(eta + aux)
        terms = y * (eta + aux) - mu
        score, w = y - mu, mu
    elif family == "bernoulli":
        mu = jax.nn.sigmoid(eta)
        terms = y * eta - jnp.logaddexp(0.0, eta)
        score, w = y - mu, mu * (1 - mu)
    else:
        terms = -((y - eta) ** 2) / (2 * aux) - 0.5 * jnp.log(2 * jnp.pi * aux)
        score, w = (y - eta) / aux, jnp.full_like(eta, 1 / aux)
    for values in (terms, score, w):
        assert bool(jnp.all(jnp.isfinite(values)))
    return (
        jnp.sum(weight * terms),
        X.T @ (weight * score),
        (X.T * (weight * w)) @ X,
    )


def _softmax_jax(X, eta, y, group, weight, n_groups):
    """``(loglik, gradient, information)`` of the grouped softmax by segment
    sums, with the ``n_g log(sum exp)`` term guarded where ``n_g = 0``."""

    def per_group(values):
        return jax.ops.segment_sum(
            values, group, num_segments=n_groups, indices_are_sorted=True
        )

    top = jax.ops.segment_max(
        eta, group, num_segments=n_groups, indices_are_sorted=True
    )
    shifted = eta - top[group]
    norm = per_group(jnp.exp(shifted))
    total = per_group(weight * y)
    p = jnp.exp(shifted) / norm[group]
    for values in (shifted, p):
        assert bool(jnp.all(jnp.isfinite(values)))
    group_term = jnp.where(total > 0, total * jnp.log(jnp.where(norm > 0, norm, 1)), 0)
    loglik = jnp.sum(weight * y * shifted) - jnp.sum(group_term)
    score = weight * (y - total[group] * p)
    Xp = per_group((weight * p)[:, None] * X)
    information = (X.T * (weight * total[group] * p)) @ X - (Xp.T * total) @ Xp
    return loglik, X.T @ score, information


def _numpy_reference(lik, X, beta, y, aux):
    eta = X @ beta
    return (
        lik.loglik(eta, y, aux),
        X.T @ lik.score_eta(eta, y, aux),
        lik.information(X, eta, y, aux),
    )


def _assert_equal(padded, reference):
    for got, want in zip(padded, reference, strict=True):
        scale = np.max(np.abs(want))
        np.testing.assert_allclose(np.asarray(got), want, rtol=0, atol=RTOL * scale)


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
def test_a_padded_diagonal_family_equals_the_unpadded_numpy_fit(family, step):
    """``step = 300`` pads nothing; 7 pads 1 row; the bucket pads 3,796."""
    lik, X, beta, y, aux = _diagonal_case(family)
    padded = pad_rows(X, y, aux if family == "poisson" else None, step=step)
    padded_aux = padded.offset if family == "poisson" else aux
    reference = _numpy_reference(lik, X, beta, y, aux)

    with jax.enable_x64(True):
        _assert_equal(
            _diagonal_jax(
                family,
                jnp.asarray(padded.X),
                jnp.asarray(padded.X @ beta),
                jnp.asarray(padded.y),
                None if padded_aux is None else jnp.asarray(padded_aux),
                jnp.asarray(padded.weight),
            ),
            reference,
        )

    # The NumPy code itself: the zero design rows add nothing to the gradient
    # or to the information and its derivatives (the log-likelihood needs the
    # weights).
    eta, padded_eta = X @ beta, padded.X @ beta
    rng = np.random.default_rng(1)
    d1, d2 = rng.normal(size=(2, padded.y.size))
    _assert_equal(
        [
            padded.X.T @ lik.score_eta(padded_eta, padded.y, padded_aux),
            lik.information(padded.X, padded_eta, padded.y, padded_aux),
            lik.dinformation(padded.X, padded_eta, padded.y, padded_aux, d1),
            lik.d2information(padded.X, padded_eta, padded.y, padded_aux, d1, d2),
        ],
        [
            reference[1],
            reference[2],
            lik.dinformation(X, eta, y, aux, d1[: y.size]),
            lik.d2information(X, eta, y, aux, d1[: y.size], d2[: y.size]),
        ],
    )


@pytest.mark.parametrize(
    ("n_groups", "row_step"),
    [
        (40, SOFTMAX_ROW_BUCKET),  # a padding group and empty groups after it
        (63, SOFTMAX_ROW_BUCKET),  # the padding group is the last of the bucket
        (40, None),  # rows fill the bucket: the padding group has no rows
    ],
)
def test_a_padded_softmax_equals_the_unpadded_numpy_fit(n_groups, row_step):
    X, beta, y, ids = _softmax_case(n_groups)
    padded = pad_groups(X, y, ids, row_step=row_step or y.size)
    lik = GroupedSoftmax()
    reference = _numpy_reference(lik, X, beta, y, ids)
    n_padded_rows = padded.y.size - padded.n_real
    assert (n_padded_rows == 0) == (row_step is None)
    assert padded.n_groups > padded.n_real_groups

    with jax.enable_x64(True):
        _assert_equal(
            _softmax_jax(
                jnp.asarray(padded.X),
                jnp.asarray(padded.X @ beta),
                jnp.asarray(padded.y),
                jnp.asarray(padded.group),
                jnp.asarray(padded.weight),
                padded.n_groups,
            ),
            reference,
        )

    # The NumPy code itself, on the padded rows: the padding group has a total
    # of 0, so it adds nothing to any term.
    eta, padded_eta = X @ beta, padded.X @ beta
    rng = np.random.default_rng(2)
    d1, d2 = rng.normal(size=(2, padded.y.size))
    args = (padded_eta, padded.y, padded.group)
    _assert_equal(
        [
            lik.loglik(*args),
            padded.X.T @ lik.score_eta(*args),
            lik.information(padded.X, *args),
            lik.dinformation(padded.X, *args, d1),
            lik.d2information(padded.X, *args, d1, d2),
        ],
        [
            *reference,
            lik.dinformation(X, eta, y, ids, d1[: y.size]),
            lik.d2information(X, eta, y, ids, d1[: y.size], d2[: y.size]),
        ],
    )


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
