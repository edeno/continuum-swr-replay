"""Simultaneous 95 % bands for a curve on a fixed grid.

A band of this module holds the whole curve with 95 % probability, not each
point alone. Two forms:

- one fit's curve (:func:`simultaneous_band`): the curve is ``f = C beta`` on
  the grid, ``C`` the rows of the coefficient vector that give the curve at
  each grid point. Its pointwise standard error is
  ``se_g = sqrt(diag(C V C'))``, ``V = (H + S)^-1`` the fit's covariance. With
  ``z ~ N(0, V)`` drawn :data:`BAND_DRAWS` times from a stored seed, the
  critical value ``c`` is the 0.95 quantile of ``max_g |C z|_g / se_g``, and
  the band is ``f +/- c se``;
- the animal-balanced curve across sessions (:func:`across_session_band`):
  the per-session curves on one grid are taken session -> recording day ->
  animal as a mean of means, with nested bootstrap draws of that curve
  (``shared_statistics.animal_day_session_bootstrap``). ``sd_g`` is the draws'
  standard deviation, ``c`` the 0.95 quantile of
  ``max_g |draw_g - mean_g| / sd_g``, and the band is
  ``estimate +/- c sd_g k`` with ``k = t(0.975, A - 1) sqrt(A / (A - 1)) /
  z(0.975)``: the small-sample correction over the ``A`` animals that
  ``shared_statistics.bootstrap_interval`` applies to a scalar.

Neither form depends on the basis that made the curve or on how its freedom
was set: the first takes ``C``, the coefficients and their covariance, the
second the per-session curve values.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

from continuum_swr_replay.analysis.shared_statistics import (
    animal_day_session_bootstrap,
)

__all__ = [
    "BAND_DRAWS",
    "BAND_LEVEL",
    "Band",
    "across_session_band",
    "simultaneous_band",
]

BAND_DRAWS = 10_000
"""Draws of a fit's coefficients behind one fit's band."""

BAND_LEVEL = 0.95
"""The probability that the band holds the whole curve."""


class Band(NamedTuple):
    """A simultaneous band on a grid.

    Attributes
    ----------
    estimate : np.ndarray, shape (n_points,)
        The curve.
    low, high : np.ndarray, shape (n_points,)
        The band.
    se : np.ndarray, shape (n_points,)
        The pointwise standard error (one fit's) or the draws' standard
        deviation (across sessions).
    critical : float
        The critical value ``c``.
    n_draws : int
        The draws behind ``critical``.
    seed : int
        Their seed.
    """

    estimate: np.ndarray
    low: np.ndarray
    high: np.ndarray
    se: np.ndarray
    critical: float
    n_draws: int
    seed: int


def simultaneous_band(
    curve_matrix: np.ndarray,
    beta: np.ndarray,
    covariance: np.ndarray,
    *,
    seed: int,
    n_draws: int = BAND_DRAWS,
) -> Band:
    """One fit's simultaneous band for the curve ``curve_matrix @ beta``.

    Parameters
    ----------
    curve_matrix : np.ndarray, shape (n_points, n_coef)
        ``C``: the rows of the coefficient vector that give the curve at each
        grid point.
    beta : np.ndarray, shape (n_coef,)
        The fit's coefficients.
    covariance : np.ndarray, shape (n_coef, n_coef)
        Their covariance, ``(H + S)^-1``.
    seed : int
        The seed of the draws (``numpy.random.default_rng(seed)``).
    n_draws : int
        Draws of ``z ~ N(0, V)``.

    Returns
    -------
    Band

    Raises
    ------
    ValueError
        If the curve uses a coefficient left out of the fit (NaN), or its
        standard error is zero at a grid point.
    """
    C = np.asarray(curve_matrix, dtype=float)
    used = np.flatnonzero(np.any(C != 0.0, axis=0))
    beta_used = np.asarray(beta, dtype=float)[used]
    V = np.asarray(covariance, dtype=float)[np.ix_(used, used)]
    if not (np.isfinite(beta_used).all() and np.isfinite(V).all()):
        raise ValueError("the curve uses a coefficient left out of the fit")
    C = C[:, used]
    eigenvalues, eigenvectors = np.linalg.eigh((V + V.T) / 2)
    # C z with z = U sqrt(lambda) e, e standard normal: draws of N(0, C V C').
    root = C @ (eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None)))
    se = np.sqrt(np.einsum("gr,gr->g", root, root))
    if not (se > 0).all():
        raise ValueError("the curve's standard error is zero at a grid point")
    rng = np.random.default_rng(seed)
    deviations = rng.standard_normal((n_draws, root.shape[1])) @ root.T
    critical = float(np.quantile(np.max(np.abs(deviations) / se, axis=1), BAND_LEVEL))
    estimate = C @ beta_used
    return Band(
        estimate,
        estimate - critical * se,
        estimate + critical * se,
        se,
        critical,
        n_draws,
        seed,
    )


def across_session_band(
    curves: np.ndarray,
    animal: np.ndarray,
    day: np.ndarray,
    *,
    n_draws: int,
    seed: int,
) -> Band:
    """The simultaneous band of the animal-balanced curve across sessions.

    Parameters
    ----------
    curves : np.ndarray, shape (n_sessions, n_points)
        Each session's curve on one grid. A session with a value that is not
        finite is left out.
    animal, day : np.ndarray, shape (n_sessions,)
        Each session's animal and recording day (a day label names a day
        within its animal).
    n_draws : int
        Nested bootstrap replicates.
    seed : int
        Their seed.

    Returns
    -------
    Band
        ``se`` is the draws' standard deviation at each grid point.

    Raises
    ------
    ValueError
        If fewer than two animals have a session, or the draws do not vary at
        a grid point.
    """
    boot = animal_day_session_bootstrap(curves, animal, day, n_draws=n_draws, seed=seed)
    n_animals = boot.n_animals
    if n_animals < 2:
        raise ValueError(f"{n_animals} animal: a band needs at least two")
    sd = boot.draws.std(axis=0, ddof=1)
    if not (sd > 0).all():
        raise ValueError("the draws do not vary at a grid point")
    deviation = np.max(np.abs(boot.draws - boot.draws.mean(axis=0)) / sd, axis=1)
    critical = float(np.quantile(deviation, BAND_LEVEL))
    tail = (1 + BAND_LEVEL) / 2
    k = (
        student_t.ppf(tail, n_animals - 1)
        * np.sqrt(n_animals / (n_animals - 1))
        / norm.ppf(tail)
    )
    half = critical * sd * k
    return Band(
        boot.estimate,
        boot.estimate - half,
        boot.estimate + half,
        sd,
        critical,
        n_draws,
        seed,
    )
