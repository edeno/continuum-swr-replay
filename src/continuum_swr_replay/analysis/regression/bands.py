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
  (animals, then their sessions;
  ``shared_statistics.animal_balanced_bootstrap``). With ``sd_g`` the draws'
  standard deviation, the band is ``estimate +/- c sd_g sqrt(A / (A - 1))``,
  the scale of ``shared_statistics.bootstrap_interval``'s scalar interval
  over the ``A`` animals with ``c`` in place of ``t(0.975, A - 1)``. ``c`` is
  the 0.95 quantile of a pivot simulated at the level of the animals: ``A``
  independent Gaussian curves with the grid correlation of the draws, and
  ``max_g |their mean| / (their sample SD / sqrt(A))``, drawn
  :data:`PIVOT_DRAWS` times from the stored seed. At one grid point this pivot
  is ``|t|`` with ``A - 1`` degrees of freedom, so the band is the scalar
  interval. With five animals the draws' own maximum cannot carry the tail
  that the t correction supplies (a quantile of it under-covered the whole
  curve: 0.80 to 0.90 in simulation).

Neither form depends on the basis that made the curve or on how its freedom
was set: the first takes ``C``, the coefficients and their covariance, the
second the per-session curve values.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from continuum_swr_replay.analysis.shared_statistics import (
    animal_balanced_bootstrap,
)

__all__ = [
    "BAND_DRAWS",
    "BAND_LEVEL",
    "PIVOT_DRAWS",
    "Band",
    "across_session_band",
    "simultaneous_band",
]

BAND_DRAWS = 10_000
"""Draws of a fit's coefficients behind one fit's band."""

BAND_LEVEL = 0.95
"""The probability that the band holds the whole curve."""

PIVOT_DRAWS = 10_000
"""Draws of the animal-level pivot behind the across-session band."""


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


def _animal_pivot_quantile(
    correlation: np.ndarray, n_animals: int, rng: np.random.Generator
) -> float:
    """The 0.95 quantile of ``max_g |mean| / (sd / sqrt(A))`` over ``A``
    independent Gaussian curves with the grid correlation given.

    Parameters
    ----------
    correlation : np.ndarray, shape (n_points, n_points)
        The grid correlation; negative eigenvalues are clipped to zero.
    n_animals : int
        ``A``.
    rng : np.random.Generator

    Returns
    -------
    float
    """
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    root = eigenvectors * np.sqrt(np.clip(eigenvalues, 0.0, None))
    maxima = []
    for start in range(0, PIVOT_DRAWS, 1000):
        n = min(1000, PIVOT_DRAWS - start)
        curves = rng.standard_normal((n, n_animals, root.shape[1])) @ root.T
        mean = curves.mean(axis=1)
        se = curves.std(axis=1, ddof=1) / np.sqrt(n_animals)
        maxima.append(np.max(np.abs(mean) / se, axis=1))
    return float(np.quantile(np.concatenate(maxima), BAND_LEVEL))


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
        The seed of the bootstrap draws (``animal_balanced_bootstrap``) and,
        as ``numpy.random.default_rng((seed, 1))``, of the pivot's draws.

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
    boot = animal_balanced_bootstrap(curves, animal, day, n_draws=n_draws, seed=seed)
    n_animals = boot.n_animals
    if n_animals < 2:
        raise ValueError(f"{n_animals} animal: a band needs at least two")
    sd = boot.draws.std(axis=0, ddof=1)
    if not (sd > 0).all():
        raise ValueError("the draws do not vary at a grid point")
    critical = _animal_pivot_quantile(
        np.corrcoef(boot.draws, rowvar=False).reshape(sd.size, sd.size),
        n_animals,
        np.random.default_rng((seed, 1)),
    )
    half = critical * sd * np.sqrt(n_animals / (n_animals - 1))
    return Band(
        boot.estimate,
        boot.estimate - half,
        boot.estimate + half,
        sd,
        critical,
        n_draws,
        seed,
    )
