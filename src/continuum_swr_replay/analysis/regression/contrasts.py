"""Linear functionals of a fit: a scalar ``c' beta`` and its exact variance.

Every scalar estimand of a fit is a linear functional of its coefficients, so
its variance in a session is exact: ``c' V c``, with ``V`` the fit's Bayesian
covariance ``(H + S)^-1``. Two kinds are built here:

- a curve contrast (:func:`range_mean_contrast`): the mean of a curve over one
  range of its variable minus its mean over another, each mean taken over an
  equally spaced grid (:func:`range_grid`, step :data:`GRID_STEP`). A range
  whose two ends are equal is one point, so a difference of the curve at two
  points is a curve contrast too;
- a coefficient difference (:func:`coefficient_contrast`): a sum of named
  coefficients less another sum.

A curve contrast is given the term's range and refuses a range outside it:
beyond its range a clamped smooth is flat, so a mean there would silently be
the curve at the end. A curve enters only through ``curve_rows``, the
function that maps values of its variable to the rows of the full coefficient
vector that give the curve there. The basis that made them, and how its freedom was set, do not matter
here.

A coefficient left out of a session (NaN in its fit) makes NaN every contrast
that uses it, and no other.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

import numpy as np

__all__ = [
    "GRID_STEP",
    "Contrast",
    "coefficient_contrast",
    "evaluate_contrast",
    "range_grid",
    "range_mean_contrast",
]

GRID_STEP = 0.01
"""The grid step of a range mean, in the curve variable's units (0.01 s on a
clock, 0.01 on a log scale)."""


class Contrast(NamedTuple):
    """The estimate and exact variance of ``c' beta``.

    Attributes
    ----------
    estimate : float
        ``c' beta``.
    variance : float
        ``c' V c``.
    """

    estimate: float
    variance: float

    @property
    def se(self) -> float:
        """The standard error, ``sqrt(variance)``."""
        return float(np.sqrt(self.variance))


def range_grid(lo: float, hi: float, step: float = GRID_STEP) -> np.ndarray:
    """The equally spaced grid from ``lo`` to ``hi``, both ends included.

    Parameters
    ----------
    lo, hi : float
        The range. ``lo == hi`` is one point.
    step : float
        The spacing; it must split the range into a whole number of steps.

    Returns
    -------
    np.ndarray, shape (n_points,)

    Raises
    ------
    ValueError
        If ``hi < lo`` or ``step`` does not split the range into a whole
        number of steps.
    """
    if hi < lo:
        raise ValueError(f"the range ends before it starts: [{lo}, {hi}]")
    n_steps = (hi - lo) / step
    if not np.isclose(n_steps, round(n_steps), rtol=0.0, atol=1e-6):
        raise ValueError(
            f"step {step} does not split [{lo}, {hi}] into a whole number of steps"
        )
    return np.linspace(lo, hi, int(round(n_steps)) + 1)


def range_mean_contrast(
    curve_rows: Callable[[np.ndarray], np.ndarray],
    first: tuple[float, float],
    second: tuple[float, float],
    *,
    term_range: tuple[float, float],
    step: float = GRID_STEP,
) -> np.ndarray:
    """The ``c`` of a curve's mean over ``first`` less its mean over ``second``.

    With ``f(x) = curve_rows(x) @ beta``, ``c @ beta`` is the mean of ``f``
    over :func:`range_grid` of ``first`` less its mean over that of
    ``second``. Both ranges must lie inside the term's range: beyond it a
    clamped smooth is flat, so a mean there would be the curve at the end of
    the range.

    Parameters
    ----------
    curve_rows : callable
        Maps values of the curve's variable, shape ``(n_points,)``, to the
        rows of the full coefficient vector, shape ``(n_points, n_coef)``.
    first, second : tuple of float
        The two ranges ``(lo, hi)``.
    term_range : tuple of float
        The range ``(lo, hi)`` the term's basis covers.
    step : float
        The grid step.

    Returns
    -------
    np.ndarray, shape (n_coef,)

    Raises
    ------
    ValueError
        If a range lies outside ``term_range``.
    """
    lo, hi = term_range
    tolerance = 1e-9 * (hi - lo)
    for name, (a, b) in (("first", first), ("second", second)):
        if a < lo - tolerance or b > hi + tolerance:
            raise ValueError(
                f"the {name} range [{a}, {b}] lies outside the term's range [{lo}, {hi}]"
            )
    return curve_rows(range_grid(*first, step)).mean(axis=0) - curve_rows(
        range_grid(*second, step)
    ).mean(axis=0)


def coefficient_contrast(
    names: Sequence[str], plus: Sequence[str], minus: Sequence[str] = ()
) -> np.ndarray:
    """The ``c`` of the sum of ``plus`` coefficients less the sum of ``minus``.

    Parameters
    ----------
    names : sequence of str, length n_coef
        The fit's coefficient names, in order.
    plus, minus : sequence of str
        Names in ``names``; each at most once over both.

    Returns
    -------
    np.ndarray, shape (n_coef,)

    Raises
    ------
    ValueError
        If a name is not a coefficient or is given more than once.
    """
    given = [*plus, *minus]
    if len(set(given)) < len(given):
        raise ValueError(f"a coefficient is given more than once: {given}")
    index = {name: i for i, name in enumerate(names)}
    missing = [name for name in given if name not in index]
    if missing:
        raise ValueError(f"{missing} is not a coefficient of {list(names)}")
    c = np.zeros(len(names))
    c[[index[name] for name in plus]] = 1.0
    c[[index[name] for name in minus]] = -1.0
    return c


def evaluate_contrast(
    c: np.ndarray, beta: np.ndarray, covariance: np.ndarray
) -> Contrast:
    """The estimate ``c' beta`` and its variance ``c' V c``.

    Only the coefficients with a non-zero entry in ``c`` are read, so a
    coefficient left out of the session (NaN) spoils only a contrast that uses
    it.

    Parameters
    ----------
    c : np.ndarray, shape (n_coef,)
    beta : np.ndarray, shape (n_coef,)
        The fit's coefficients.
    covariance : np.ndarray, shape (n_coef, n_coef)
        Their covariance, ``(H + S)^-1``.

    Returns
    -------
    Contrast
        NaN estimate and variance when a coefficient it uses is NaN.

    Raises
    ------
    ValueError
        If the shapes do not match.
    """
    c = np.asarray(c, dtype=float)
    beta = np.asarray(beta, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    n = beta.size
    if c.shape != (n,) or covariance.shape != (n, n):
        raise ValueError(
            f"a contrast of shape {c.shape} and a covariance of shape "
            f"{covariance.shape} for {n} coefficients"
        )
    used = np.flatnonzero(c)
    c_used = c[used]
    estimate = float(c_used @ beta[used])
    variance = float(c_used @ covariance[np.ix_(used, used)] @ c_used)
    return Contrast(estimate, variance)
