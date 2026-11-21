"""Simultaneous bands for a curve: one fit's, and the animal-balanced curve's."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from scipy.stats import t as student_t

from continuum_swr_replay.analysis import shared_statistics as stats
from continuum_swr_replay.analysis.regression.bands import (
    BAND_DRAWS,
    across_session_band,
    simultaneous_band,
)
from continuum_swr_replay.analysis.regression.likelihoods import Poisson
from continuum_swr_replay.analysis.regression.newton import penalized_newton
from continuum_swr_replay.analysis.regression.pspline import PSplineBasis

BASIS = PSplineBasis(0.0, 4.0, 8)
GRID = np.linspace(0.0, 4.0, 101)
ROWS = np.random.default_rng(20261121).uniform(0.0, 4.0, 1500)
Z = BASIS.constrained(ROWS).constraint  # the smooth sums to zero over the rows


def _curve_matrix(x: np.ndarray) -> np.ndarray:
    """An intercept and a sum-to-zero smooth: the curve is the smooth's part."""
    return np.hstack([np.zeros((x.size, 1)), BASIS.design(x) @ Z])


def test_the_band_is_the_curve_plus_minus_a_critical_value_of_its_pointwise_se() -> (
    None
):
    rng = np.random.default_rng(0)
    p = BASIS.n_basis
    beta = rng.normal(size=p)
    root = rng.normal(size=(p, p))
    V = root @ root.T / p + 0.05 * np.eye(p)
    C = _curve_matrix(GRID)
    band = simultaneous_band(C, beta, V, seed=3)
    np.testing.assert_allclose(band.estimate, C @ beta)
    np.testing.assert_allclose(band.se, np.sqrt(np.diag(C @ V @ C.T)))
    np.testing.assert_allclose(band.high - band.estimate, band.critical * band.se)
    np.testing.assert_allclose(band.estimate - band.low, band.critical * band.se)
    # Wider than pointwise, narrower than Bonferroni over the grid.
    assert norm.ppf(0.975) < band.critical < norm.ppf(1 - 0.025 / GRID.size)
    assert band.seed == 3 and band.n_draws == BAND_DRAWS
    again = simultaneous_band(C, beta, V, seed=3)
    assert again.critical == band.critical


def test_a_band_refuses_a_curve_on_a_left_out_coefficient() -> None:
    beta = np.ones(BASIS.n_basis)
    V = np.eye(BASIS.n_basis)
    beta[3] = np.nan
    with pytest.raises(ValueError, match="left out"):
        simultaneous_band(_curve_matrix(GRID), beta, V, seed=0)


@pytest.mark.slow
def test_simultaneous_band_coverage() -> None:
    """Over 500 simulated Poisson fits of a known curve, the band covers the
    whole true curve in 93 to 97 % of them, and the pointwise band less often."""
    rng = np.random.default_rng(1)
    X = np.hstack([np.ones((ROWS.size, 1)), BASIS.design(ROWS) @ Z])
    # The true coefficients: the least-squares fit of a curve in the basis.
    G = np.hstack([np.ones((GRID.size, 1)), BASIS.design(GRID) @ Z])
    beta_true = np.linalg.lstsq(G, 0.5 + np.sin(1.5 * GRID), rcond=None)[0]
    curve_true = _curve_matrix(GRID) @ beta_true
    offset = np.full(ROWS.size, np.log(0.5))
    C = _curve_matrix(GRID)
    whole, pointwise, criticals = 0, 0, []
    n_fits = 500
    for i in range(n_fits):
        y = rng.poisson(np.exp(X @ beta_true + offset)).astype(float)
        fit = penalized_newton(
            Poisson(), X, y, offset, np.zeros((X.shape[1],) * 2), np.zeros(X.shape[1])
        )
        assert fit.converged
        band = simultaneous_band(C, fit.beta, np.linalg.inv(fit.A), seed=i)
        criticals.append(band.critical)
        whole += bool(
            np.all(np.abs(band.estimate - curve_true) <= band.critical * band.se)
        )
        z = norm.ppf(0.975)
        pointwise += bool(np.all(np.abs(band.estimate - curve_true) <= z * band.se))
    print(
        f"simultaneous {whole / n_fits:.3f}, pointwise {pointwise / n_fits:.3f}, "
        f"critical median {np.median(criticals):.3f}"
    )
    assert 0.93 <= whole / n_fits <= 0.97
    assert pointwise < whole


def _sessions(n_points: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Five animals of three days of two sessions; smooth animal offsets."""
    rng = np.random.default_rng(seed)
    n_animals, n_days, n_sessions = 5, 3, 2
    animal = np.repeat(np.arange(n_animals), n_days * n_sessions)
    day = np.repeat(np.arange(n_animals * n_days), n_sessions)
    grid = np.linspace(0.0, 4.0, n_points)
    offsets = rng.normal(0.0, 0.5, n_animals)
    curves = np.sin(grid)[None, :] + offsets[animal, None]
    curves = curves + rng.normal(0.0, 0.3, (animal.size, n_points))
    return curves, animal, day


def test_the_across_session_band_scales_the_bootstrap_sd_by_an_animal_level_pivot() -> (
    None
):
    curves, animal, day = _sessions(GRID.size, 1)
    band = across_session_band(curves, animal, day, n_draws=2000, seed=5)
    boot = stats.animal_balanced_bootstrap(curves, animal, day, n_draws=2000, seed=5)
    np.testing.assert_allclose(band.estimate, boot.estimate)
    sd = boot.draws.std(axis=0, ddof=1)
    np.testing.assert_allclose(band.se, sd)
    np.testing.assert_allclose(
        band.high - band.estimate, band.critical * sd * np.sqrt(5 / 4)
    )
    # A whole curve needs more than one point's t quantile.
    assert band.critical > student_t.ppf(0.975, 4)
    assert band.seed == 5 and band.n_draws == 2000
    again = across_session_band(curves, animal, day, n_draws=2000, seed=5)
    assert again.critical == band.critical


@pytest.mark.parametrize("n_animals", [5, 2])
def test_at_one_grid_point_the_band_is_the_scalar_interval(n_animals: int) -> None:
    """With one grid point the pivot is t with A - 1 degrees of freedom, so
    the band is estimand_summary's interval up to the pivot's simulation error
    (its 0.95 quantile from 10,000 draws: about 2 % for A = 5, 4 % for A = 2)."""
    curves, animal, day = _sessions(1, 2)
    keep = animal < n_animals
    curves, animal, day = curves[keep], animal[keep], day[keep]
    band = across_session_band(curves, animal, day, n_draws=2000, seed=9)
    t = student_t.ppf(0.975, n_animals - 1)
    assert band.critical == pytest.approx(
        t, rel=3 * (0.02 if n_animals == 5 else 0.045)
    )
    labels = pd.Series(np.arange(animal.size))
    summary = stats.estimand_summary(
        pd.Series(curves[:, 0]),
        pd.Series(animal),
        pd.Series(day),
        labels,
        n_draws=2000,
        seed=9,
    )
    assert band.estimate[0] == pytest.approx(summary["estimate"])
    assert band.high[0] - band.estimate[0] == pytest.approx(
        summary["high"] - summary["estimate"], rel=0.15
    )
    assert (band.high[0] - band.estimate[0]) / band.critical == pytest.approx(
        (summary["high"] - summary["estimate"]) / t
    )
