"""Linear functionals of a fit: range-mean curve contrasts and coefficient differences."""

from __future__ import annotations

import numpy as np
import pytest

from continuum_swr_replay.analysis.regression.contrasts import (
    coefficient_contrast,
    evaluate_contrast,
    range_grid,
    range_mean_contrast,
)
from continuum_swr_replay.analysis.regression.pspline import PSplineBasis

BASIS = PSplineBasis(0.0, 5.0, 8)
Z = BASIS.constrained(np.linspace(0.0, 5.0, 300)).constraint
N_COEF = 1 + Z.shape[1] + 1  # intercept, the smooth, one slope
TERM_RANGE = (BASIS.lo, BASIS.hi)


def curve_rows(x: np.ndarray) -> np.ndarray:
    """The smooth's rows of the full coefficient vector: columns 1 .. 7."""
    rows = np.zeros((np.size(x), N_COEF))
    rows[:, 1:-1] = BASIS.design(x) @ Z
    return rows


def a_fit(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    beta = rng.normal(size=N_COEF)
    root = rng.normal(size=(N_COEF, N_COEF))
    return beta, root @ root.T / N_COEF + 0.1 * np.eye(N_COEF)


def test_range_grid_includes_both_ends_and_a_point_is_one_value() -> None:
    np.testing.assert_allclose(range_grid(2.0, 4.0), np.linspace(2.0, 4.0, 201))
    np.testing.assert_allclose(range_grid(0.0, 0.25), np.arange(26) * 0.01)
    np.testing.assert_array_equal(range_grid(np.log(20), np.log(20)), [np.log(20)])
    with pytest.raises(ValueError, match="whole number"):
        range_grid(0.0, 0.255)
    with pytest.raises(ValueError, match="ends before"):
        range_grid(1.0, 0.5)


def test_range_mean_contrast_is_the_difference_of_grid_means() -> None:
    beta, _ = a_fit(0)
    c = range_mean_contrast(curve_rows, (2.0, 4.0), (0.0, 1.0), term_range=TERM_RANGE)
    curve = lambda x: curve_rows(x) @ beta  # noqa: E731
    expected = np.mean([curve(np.array([x]))[0] for x in np.linspace(2, 4, 201)])
    expected -= np.mean([curve(np.array([x]))[0] for x in np.linspace(0, 1, 101)])
    assert c @ beta == pytest.approx(expected, rel=1e-12)
    # Only the smooth's coefficients enter.
    assert c[0] == 0.0 and c[-1] == 0.0


def test_curve_contrast_variance() -> None:
    """The exact variance of a range-mean contrast is the variance of the
    contrast over draws of the coefficients from N(beta, V)."""
    beta, V = a_fit(1)
    c = range_mean_contrast(curve_rows, (0.0, 0.25), (1.0, 2.0), term_range=TERM_RANGE)
    result = evaluate_contrast(c, beta, V)
    draws = np.random.default_rng(2).multivariate_normal(beta, V, size=100_000)
    values = draws @ c
    assert result.estimate == pytest.approx(c @ beta)
    assert result.variance == pytest.approx(values.var(), rel=0.02)
    assert result.se == pytest.approx(np.sqrt(result.variance))


def test_coefficient_contrast_takes_named_differences() -> None:
    names = ["intercept", "role[current]", "role[previous]", "value"]
    c = coefficient_contrast(names, plus=["role[current]"], minus=["role[previous]"])
    np.testing.assert_array_equal(c, [0.0, 1.0, -1.0, 0.0])
    beta = np.array([0.3, 1.0, 0.25, -2.0])
    V = np.diag([1.0, 0.5, 0.25, 9.0])
    V[1, 2] = V[2, 1] = 0.1
    result = evaluate_contrast(c, beta, V)
    assert result.estimate == pytest.approx(0.75)
    assert result.variance == pytest.approx(0.5 + 0.25 - 0.2)
    with pytest.raises(ValueError, match="not a coefficient"):
        coefficient_contrast(names, plus=["role[next]"])
    with pytest.raises(ValueError, match="more than once"):
        coefficient_contrast(names, plus=["value"], minus=["value"])


def test_a_left_out_coefficient_spoils_only_the_contrasts_that_use_it() -> None:
    beta, V = a_fit(3)
    beta[-1] = np.nan
    V[-1, :] = V[:, -1] = np.nan
    c = range_mean_contrast(curve_rows, (2.0, 4.0), (0.0, 1.0), term_range=TERM_RANGE)
    result = evaluate_contrast(c, beta, V)
    assert np.isfinite(result.estimate) and np.isfinite(result.variance)
    slope = np.zeros(N_COEF)
    slope[-1] = 1.0
    left_out = evaluate_contrast(slope, beta, V)
    assert np.isnan(left_out.estimate) and np.isnan(left_out.variance)


def test_a_contrast_must_match_the_coefficients() -> None:
    beta, V = a_fit(4)
    with pytest.raises(ValueError, match="coefficients"):
        evaluate_contrast(np.ones(N_COEF + 1), beta, V)


def test_a_range_outside_the_terms_range_is_refused() -> None:
    """Beyond its range a clamped smooth is flat: a mean over (5, 6) of a term
    on [0, 5] would silently be the curve at 5."""
    for first in ((5.0, 6.0), (-0.5, 1.0), (5.01, 5.01)):
        with pytest.raises(ValueError, match="outside the term's range"):
            range_mean_contrast(curve_rows, first, (0.0, 1.0), term_range=TERM_RANGE)
    with pytest.raises(ValueError, match="outside the term's range"):
        range_mean_contrast(curve_rows, (0.0, 1.0), (4.0, 6.0), term_range=TERM_RANGE)
    # Both ends of the term's range are inside it.
    range_mean_contrast(curve_rows, (0.0, 0.0), (5.0, 5.0), term_range=TERM_RANGE)
