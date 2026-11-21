"""The cubic P-spline basis, its difference penalty and the sum-to-zero constraint."""

from __future__ import annotations

import numpy as np
import pytest

from continuum_swr_replay.analysis.regression.pspline import (
    PSplineBasis,
    difference_penalty,
    penalty_rank,
    sum_to_zero_constraint,
)

BASIS = PSplineBasis(lo=0.0, hi=4.0, n_basis=10)


def test_knots_are_equally_spaced_around_the_range() -> None:
    knots = BASIS.knots
    assert knots.size == BASIS.n_basis + BASIS.degree + 1
    np.testing.assert_allclose(np.diff(knots), 4.0 / 7)  # 10 - 3 intervals
    # The range is the span of the inner knots.
    assert knots[BASIS.degree] == pytest.approx(0.0)
    assert knots[BASIS.n_basis] == pytest.approx(4.0)


def test_from_spacing_puts_knots_that_far_apart() -> None:
    basis = PSplineBasis.from_spacing(0.0, 20.0, 0.5)
    assert basis.n_basis == 40 + 3
    np.testing.assert_allclose(np.diff(basis.knots), 0.5)
    with pytest.raises(ValueError, match="whole number"):
        PSplineBasis.from_spacing(0.0, 1.0, 0.3)


def test_partition_of_unity_inside_the_range() -> None:
    x = np.linspace(0.0, 4.0, 401)  # both ends included
    B = BASIS.design(x)
    assert B.shape == (401, 10)
    assert np.all(B >= 0)
    np.testing.assert_allclose(B.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_clamped_outside_the_range() -> None:
    """Below ``lo`` a row is the row at ``lo``; above ``hi``, the row at ``hi``."""
    B = BASIS.design(np.array([-3.0, -1e-9, 0.0, 4.0, 4.0 + 1e-9, 50.0]))
    np.testing.assert_array_equal(B[0], B[2])
    np.testing.assert_array_equal(B[1], B[2])
    np.testing.assert_array_equal(B[4], B[3])
    np.testing.assert_array_equal(B[5], B[3])
    np.testing.assert_allclose(B.sum(axis=1), 1.0, rtol=0, atol=1e-12)


def test_penalty_null_space_is_the_linear_functions() -> None:
    P = BASIS.penalty()
    np.testing.assert_array_equal(P, difference_penalty(10, order=2))
    index = np.arange(10, dtype=float)
    np.testing.assert_allclose(P @ np.ones(10), 0.0, atol=1e-12)
    np.testing.assert_allclose(P @ index, 0.0, atol=1e-12)
    assert penalty_rank(P) == 10 - 2
    # Coefficients linear in the index draw a straight line in x: on equally
    # spaced knots, sum_j (a + b j) B_j(x) is a + b' x.
    x = np.linspace(0.0, 4.0, 81)
    curve = BASIS.design(x) @ (0.3 + 1.7 * index)
    slope, intercept = np.polyfit(x, curve, 1)
    np.testing.assert_allclose(curve, intercept + slope * x, rtol=0, atol=1e-10)
    # Any coefficient vector outside that space is penalized.
    rng = np.random.default_rng(0)
    wiggly = rng.normal(size=10)
    residual = wiggly - np.polyval(np.polyfit(index, wiggly, 1), index)
    assert residual @ P @ residual > 1e-3


def test_constraint_removes_the_constant() -> None:
    rng = np.random.default_rng(1)
    rows = BASIS.design(rng.uniform(-1.0, 5.0, size=300))
    Z = sum_to_zero_constraint(rows)
    assert Z.shape == (10, 9)
    np.testing.assert_allclose(Z.T @ Z, np.eye(9), atol=1e-12)
    # Every constrained curve sums to zero over the given rows.
    np.testing.assert_allclose(rows.sum(axis=0) @ Z, 0.0, atol=1e-10)
    # The basis alone holds the constant (it sums to one), so beside an
    # intercept it is rank deficient; constrained, it is not.
    intercept = np.ones((rows.shape[0], 1))
    assert np.linalg.matrix_rank(np.hstack([intercept, rows])) == 10
    assert np.linalg.matrix_rank(np.hstack([intercept, rows @ Z])) == 10
    # The constant is not a constrained curve: a curve summing to zero over
    # n rows is at least sqrt(n) from the constant one.
    n_rows = rows.shape[0]
    coef, *_ = np.linalg.lstsq(rows @ Z, np.ones(n_rows), rcond=None)
    assert np.linalg.norm(rows @ Z @ coef - 1.0) == pytest.approx(np.sqrt(n_rows))
    # The constrained penalty keeps one null direction: the centered line.
    assert penalty_rank(Z.T @ BASIS.penalty() @ Z) == 10 - 2


def test_constrained_smooth_is_the_basis_times_the_constraint() -> None:
    """``constrained`` gives ``B Z``, ``Z' P Z`` and ``Z``, with the constraint
    computed on the rows given: each column of the design sums to zero over
    them, and the penalty keeps the centered line as its null space."""
    x = np.random.default_rng(2).uniform(0.0, 4.0, size=200)
    design, penalty, Z = BASIS.constrained(x)
    B = BASIS.design(x)
    np.testing.assert_array_equal(Z, sum_to_zero_constraint(B))
    np.testing.assert_array_equal(design, B @ Z)
    np.testing.assert_array_equal(penalty, Z.T @ BASIS.penalty() @ Z)
    assert design.shape == (200, 9) and penalty.shape == (9, 9)
    np.testing.assert_allclose(design.sum(axis=0), 0.0, atol=1e-10)
    assert penalty_rank(penalty) == 10 - 2


def test_no_rows_give_an_empty_design() -> None:
    """The basis at no rows is an empty design of the basis's width (scipy's
    design matrix refuses an empty input)."""
    assert PSplineBasis(0.0, 1.0, 8).design(np.empty(0)).shape == (0, 8)
