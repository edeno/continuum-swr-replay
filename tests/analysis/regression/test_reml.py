"""REML: its derivatives, the penalty search, pooling and the Gaussian scale."""

from __future__ import annotations

import json

import numpy as np
import pytest
import scipy.optimize

from continuum_swr_replay.analysis.regression.constants import (
    LOG_PENALTY_BOUNDS,
    NULL_SPACE_EDF_TOL,
)
from continuum_swr_replay.analysis.regression.likelihoods import (
    Bernoulli,
    Gaussian,
    GroupedSoftmax,
    Poisson,
)
from continuum_swr_replay.analysis.regression.newton import penalized_newton
from continuum_swr_replay.analysis.regression.pspline import (
    PSplineBasis,
    sum_to_zero_constraint,
)
from continuum_swr_replay.analysis.regression.reml import (
    PenaltyBlock,
    SessionData,
    assemble_penalty,
    fit_at_penalties,
    pooled_reml,
    refine_penalties,
    reml_and_derivatives,
    select_penalties,
)
from scripts.checks.compare_regression_to_mgcv import (
    DATA_SETS,
    FIXTURE_DIR,
    agreement,
    fit_ours,
    make_data,
)

FAMILIES = ("poisson", "bernoulli", "gaussian", "grouped_softmax")
N_BASIS = 7  # 6 constrained columns per smooth


def _relative_error(analytic: np.ndarray, numeric: np.ndarray) -> float:
    """``||analytic - numeric|| / ||numeric||``."""
    return float(np.linalg.norm(analytic - numeric) / np.linalg.norm(numeric))


def _smooth(x: np.ndarray, n_basis: int = N_BASIS) -> tuple[np.ndarray, np.ndarray]:
    """The constrained P-spline design of ``x`` over [0, 1] and its penalty."""
    basis = PSplineBasis(0.0, 1.0, n_basis)
    B = basis.design(x)
    Z = sum_to_zero_constraint(B)
    return B @ Z, Z.T @ basis.penalty() @ Z


def _session(
    family: str, n_smooths: int, seed: int, n_rows: int = 240
) -> tuple[object, SessionData, list[PenaltyBlock]]:
    """A likelihood, one seeded data set with ``n_smooths`` smooths, and its blocks.

    Every family but the grouped softmax has an intercept in column 0; the
    softmax has groups of four rows and no group-level column.
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(n_smooths, n_rows))
    truth = sum(0.7 * np.sin(2 * np.pi * (j + 1) * x[j] / 2) for j in range(n_smooths))
    parts = [] if family == "grouped_softmax" else [np.ones((n_rows, 1))]
    blocks = []
    start = len(parts)
    for j in range(n_smooths):
        design, penalty = _smooth(x[j])
        parts.append(design)
        blocks.append(PenaltyBlock(np.arange(start, start + design.shape[1]), penalty))
        start += design.shape[1]
    X = np.hstack(parts)
    if family == "poisson":
        offset = np.log(rng.uniform(0.5, 2.0, size=n_rows))
        y = rng.poisson(np.exp(0.7 + truth + offset)).astype(float)
        return Poisson(), SessionData(X, y, offset), blocks
    if family == "bernoulli":
        y = (rng.uniform(size=n_rows) < 1 / (1 + np.exp(-truth))).astype(float)
        return Bernoulli(), SessionData(X, y), blocks
    if family == "gaussian":
        y = 1.0 + truth + rng.normal(scale=0.5, size=n_rows)
        return Gaussian(), SessionData(X, y), blocks
    groups = np.repeat(np.arange(n_rows // 4), 4)
    weight = np.exp(truth).reshape(-1, 4)
    p = weight / weight.sum(axis=1, keepdims=True)
    y = np.concatenate([rng.dirichlet(30.0 * row) for row in p])
    return GroupedSoftmax(), SessionData(X, y, groups), blocks


def _score(lik, data, blocks, rho, beta0=None):
    return reml_and_derivatives(lik, data.X, data.y, data.aux, blocks, rho, beta0)


@pytest.mark.parametrize("n_smooths", [1, 2, 3])
@pytest.mark.parametrize("family", FAMILIES)
def test_gradient_and_hessian_finite_differences(family: str, n_smooths: int) -> None:
    """The gradient and Hessian are central differences of V.

    Every fit starts from the optimum at ``rho`` (as the search warm-starts),
    so each lands on its own optimum to rounding: from a cold start the fit
    stops within about 1e-8 of it, which the score inherits through
    ``log|A|`` and the differences divide by the step.
    """
    lik, data, blocks = _session(family, n_smooths, seed=11)
    rho = np.array([0.5, 1.5, -0.5])[:n_smooths]
    warm = _score(lik, data, blocks, rho).fit.beta
    at = _score(lik, data, blocks, rho, warm)
    assert np.isfinite(at.score) and at.fit.converged

    def V(r: np.ndarray) -> float:
        return _score(lik, data, blocks, r, warm).score

    unit = np.eye(n_smooths)
    h = 1e-4  # first differences
    numeric_gradient = np.array(
        [(V(rho + h * e) - V(rho - h * e)) / (2 * h) for e in unit]
    )
    h = 1e-3  # second differences
    numeric_hessian = np.array(
        [
            [
                (
                    V(rho + h * (ej + ek))
                    - V(rho + h * (ej - ek))
                    - V(rho - h * (ej - ek))
                    + V(rho - h * (ej + ek))
                )
                / (4 * h * h)
                for ek in unit
            ]
            for ej in unit
        ]
    )

    assert _relative_error(at.gradient, numeric_gradient) < 1e-5
    assert _relative_error(at.hessian, numeric_hessian) < 1e-5
    np.testing.assert_array_equal(at.hessian, at.hessian.T)


def test_block_rank_is_the_penalty_rank() -> None:
    """The constrained second-difference penalty has rank n_basis - 2."""
    _, data, blocks = _session("poisson", 1, seed=1)
    assert blocks[0].rank == N_BASIS - 2
    eigenvalues = np.linalg.eigvalsh(blocks[0].S)
    positive = eigenvalues[eigenvalues > 1e-9 * eigenvalues.max()]
    assert blocks[0].log_pdet == pytest.approx(np.log(positive).sum(), rel=1e-12)


def test_overlapping_blocks_are_refused() -> None:
    lik, data, blocks = _session("poisson", 2, seed=2)
    overlapping = [blocks[0], PenaltyBlock(blocks[1].columns - 1, blocks[1].S)]
    with pytest.raises(ValueError, match="overlap"):
        reml_and_derivatives(lik, data.X, data.y, data.aux, overlapping, [0.0, 0.0])
    outside = [PenaltyBlock(blocks[1].columns + 5, blocks[1].S)]
    with pytest.raises(ValueError, match="outside"):
        pooled_reml(lik, [data], outside, [0.0])


def test_fixed_penalties_skip_selection() -> None:
    """At given log penalties: the Newton fit at that penalty, and its edf and covariance."""
    lik, data, blocks = _session("poisson", 2, seed=3)
    rho = np.array([1.0, 3.0])
    result = fit_at_penalties(lik, [data], blocks, rho)

    assert result.search is None
    np.testing.assert_array_equal(result.rho, rho)
    S = assemble_penalty(blocks, rho, data.X.shape[1])
    direct = penalized_newton(lik, data.X, data.y, data.aux, S, np.zeros(S.shape[0]))
    session = result.sessions[0]
    np.testing.assert_allclose(session.fit.beta, direct.beta, rtol=0, atol=1e-10)
    np.testing.assert_allclose(session.covariance, np.linalg.inv(direct.A), rtol=1e-10)
    F = np.linalg.solve(direct.A, direct.A - S)  # A^-1 H
    expected_edf = [np.trace(F[np.ix_(b.columns, b.columns)]) for b in blocks]
    np.testing.assert_allclose(session.edf, expected_edf, rtol=1e-10)
    assert result.score == pytest.approx(
        pooled_reml(lik, [data], blocks, rho).score, rel=1e-12
    )


def test_no_finite_score_raises() -> None:
    """A column the data cannot identify fails every inner fit: no fallback."""
    lik, data, blocks = _session("poisson", 1, seed=4)
    X = np.column_stack([data.X, np.zeros(data.X.shape[0])])
    with pytest.raises(ValueError, match="no finite REML score"):
        select_penalties(lik, [SessionData(X, data.y, data.aux)], blocks)


def test_pooled_score_is_the_sum_and_a_failed_session_is_infinite() -> None:
    lik, first, blocks = _session("poisson", 2, seed=5)
    _, second, _ = _session("poisson", 2, seed=6)
    rho = np.array([0.0, 2.0])
    pooled = pooled_reml(lik, [first, second], blocks, rho)
    parts = [_score(lik, d, blocks, rho) for d in (first, second)]
    assert pooled.score == pytest.approx(sum(p.score for p in parts), rel=1e-12)
    np.testing.assert_allclose(pooled.gradient, sum(p.gradient for p in parts))
    np.testing.assert_allclose(pooled.hessian, sum(p.hessian for p in parts))

    broken = SessionData(
        np.column_stack([second.X, np.zeros(second.X.shape[0])]),
        second.y,
        second.aux,
    )
    blocks_wide = blocks  # the extra column is unpenalized
    failed = pooled_reml(lik, [first, broken], blocks_wide, rho)
    assert failed.score == np.inf


def _one_clock_counts(
    n_events: int, seed: int
) -> tuple[SessionData, list[PenaltyBlock]]:
    """Poisson counts over time since a stop began, with a flat truth.

    350 stops of lognormal duration (median 1 s after a reward, 0.3 s
    after none), 15 ms bins over the first 4 s, the counts and exposures
    summed over bins at the same time (which leaves the likelihood unchanged
    up to a constant); a cubic P-spline with 15 basis functions over [0, 4].
    """
    rng = np.random.default_rng(seed)
    bin_s, n_stops = 0.015, 350
    rewarded = rng.random(n_stops) < 0.5
    duration = np.where(rewarded, 1.0, 0.3) * np.exp(rng.normal(size=n_stops))
    duration = np.maximum(duration, bin_s)
    n_bins = np.ceil(duration / bin_s).astype(int)
    stop = np.repeat(np.arange(n_stops), n_bins)
    k = np.arange(n_bins.sum()) - np.repeat(np.cumsum(n_bins) - n_bins, n_bins)
    start = k * bin_s
    end = np.minimum(start + bin_s, duration[stop])
    exposure, clock = end - start, 0.5 * (start + end)
    keep = clock < 4.0
    exposure, clock, k = exposure[keep], clock[keep], k[keep]
    y = rng.poisson(n_events / exposure.sum() * exposure).astype(float)
    _, cell = np.unique(k, return_inverse=True)
    cell_exposure = np.bincount(cell, weights=exposure)
    cell_clock = np.bincount(cell, weights=clock * exposure) / cell_exposure
    basis = PSplineBasis(0.0, 4.0, 15)
    B = basis.design(cell_clock)
    Z = sum_to_zero_constraint(B)
    X = np.column_stack([np.ones(cell_clock.size), B @ Z])
    data = SessionData(X, np.bincount(cell, weights=y), np.log(cell_exposure))
    return data, [PenaltyBlock(np.arange(1, 15), Z.T @ basis.penalty() @ Z)]


def test_grid_start_finds_better_optimum() -> None:
    """The REML score has two optima here: a wiggly one near log penalty 3.5
    (edf 2.5) and the straight line at the upper bound. Newton from 0 stops
    at the wiggly one; the grid start finds the line, whose score is lower."""
    data, blocks = _one_clock_counts(n_events=100, seed=36)
    local = refine_penalties(Poisson(), [data], blocks, [0.0])
    selected = select_penalties(Poisson(), [data], blocks)

    assert local.search.converged and selected.search.converged
    assert 2.0 < local.rho[0] < 5.0
    assert local.sessions[0].edf[0] > 2.0
    np.testing.assert_array_equal(selected.search.at_boundary, [True])
    assert selected.sessions[0].edf[0] == pytest.approx(1.0, abs=1e-5)
    assert selected.score < local.score - 0.1
    assert selected.search.n_grid_points == 31


def test_pooled_invariant_to_duplication() -> None:
    """Duplicating every session leaves the selected penalties unchanged."""
    lik, first, blocks = _session("poisson", 2, seed=7)
    _, second, _ = _session("poisson", 2, seed=8)
    once = select_penalties(lik, [first, second], blocks)
    twice = select_penalties(lik, [first, second, first, second], blocks)
    assert once.search.converged and twice.search.converged
    np.testing.assert_allclose(twice.rho, once.rho, rtol=0, atol=1e-6)
    assert twice.score == pytest.approx(2 * once.score, rel=1e-10)
    assert once.search.n_grid_points == 25  # the product of five points per term


def test_more_than_three_terms_use_one_term_grids() -> None:
    """Four terms: two passes of one-term integer grids (2 x 4 x 31 points),
    then Newton to a stationary point."""
    lik, data, blocks = _session("poisson", 4, seed=10, n_rows=400)
    result = select_penalties(lik, [data], blocks)
    assert result.search.n_grid_points == 2 * 4 * 31
    assert result.search.converged
    free = ~result.search.at_boundary
    assert np.max(np.abs(result.gradient[free])) < 1e-6 * max(1.0, abs(result.score))
    assert result.score <= result.search.start_score


def test_gaussian_scale_matches_closed_form() -> None:
    """The profiled scale minimizes the Laplace score at the selected penalty,
    and the fit is penalized least squares."""
    lik, data, blocks = _session("gaussian", 1, seed=9, n_rows=200)
    result = select_penalties(lik, [data], blocks)
    session = result.sessions[0]
    X, y = data.X, data.y
    n, p = X.shape
    S_w = assemble_penalty(
        blocks, result.rho, p
    )  # on the residual-sum-of-squares scale

    pls = np.linalg.solve(X.T @ X + S_w, X.T @ y)
    np.testing.assert_allclose(session.fit.beta, pls, rtol=1e-10, atol=1e-12)

    null_dim = p - sum(b.rank for b in blocks)
    log_pdet_w = sum(
        b.rank * r + b.log_pdet for b, r in zip(blocks, result.rho, strict=True)
    )

    def laplace_score(log_phi: float) -> float:
        """The Laplace REML score at scale phi with the penalty S_w / phi."""
        phi = np.exp(log_phi)
        residual = y - X @ pls
        minus_loglik = residual @ residual / (2 * phi) + n / 2 * np.log(2 * np.pi * phi)
        log_pdet = log_pdet_w - (p - null_dim) * log_phi
        _, log_det_a = np.linalg.slogdet((X.T @ X + S_w) / phi)
        return float(
            minus_loglik
            + pls @ S_w @ pls / (2 * phi)
            - log_pdet / 2
            + log_det_a / 2
            - null_dim / 2 * np.log(2 * np.pi)
        )

    best = scipy.optimize.minimize_scalar(laplace_score, bracket=(-3.0, 0.0), tol=1e-12)
    assert session.scale == pytest.approx(np.exp(best.x), rel=1e-6)
    assert session.score == pytest.approx(
        laplace_score(np.log(session.scale)), rel=1e-12
    )
    np.testing.assert_allclose(
        session.covariance,
        session.scale * np.linalg.inv(X.T @ X + S_w),
        rtol=1e-10,
    )
    # The search stopped at a stationary point of the profiled score.
    assert result.search.converged
    assert np.max(np.abs(result.gradient)) < 1e-6 * max(1.0, abs(result.score))


def test_gaussian_scale_is_profiled_not_given() -> None:
    lik, data, blocks = _session("gaussian", 1, seed=9)
    with pytest.raises(ValueError, match="profiled"):
        reml_and_derivatives(lik, data.X, data.y, 0.5, blocks, [0.0])


def test_at_boundary() -> None:
    """A straight-line truth goes to the upper bound (edf 1, the line); a wiggly
    truth on the same rows and noise stays inside."""
    rng = np.random.default_rng(0)
    n_rows = 400
    x = rng.uniform(size=n_rows)
    design, penalty = _smooth(x)
    X = np.column_stack([np.ones(n_rows), design])
    noise = rng.normal(size=n_rows)
    blocks = [PenaltyBlock(np.arange(1, 7), penalty)]

    line = select_penalties(Gaussian(), [SessionData(X, 0.8 * x + noise)], blocks)
    wiggle = select_penalties(
        Gaussian(), [SessionData(X, np.sin(2 * np.pi * x) + noise)], blocks
    )

    np.testing.assert_array_equal(line.search.at_boundary, [True])
    assert line.rho[0] == LOG_PENALTY_BOUNDS[1]
    assert line.sessions[0].edf[0] == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_array_equal(wiggle.search.at_boundary, [False])
    assert wiggle.search.converged and wiggle.search.n_outer_steps >= 1
    assert 3.0 < wiggle.sessions[0].edf[0] < 5.0


def test_shrunk_to_null_space() -> None:
    """A straight-line truth whose search stops on the plateau, short of the
    bound, is flagged by its edf (one, the line); at_boundary misses it. A
    curved truth on the same rows is not flagged."""
    rng = np.random.default_rng(2)
    n_rows = 400
    x = rng.uniform(size=n_rows)
    basis = PSplineBasis(0.0, 1.0, 10)
    B = basis.design(x)
    Z = sum_to_zero_constraint(B)
    X = np.column_stack([np.ones(n_rows), B @ Z])
    blocks = [PenaltyBlock(np.arange(1, 10), Z.T @ basis.penalty() @ Z)]
    line_y = (x - 0.5) + rng.normal(size=n_rows)
    curve_y = np.sin(2 * np.pi * x) + rng.normal(size=n_rows)

    line = select_penalties(Gaussian(), [SessionData(X, line_y)], blocks)
    curve = select_penalties(Gaussian(), [SessionData(X, curve_y)], blocks)

    assert line.search.converged and curve.search.converged
    assert line.sessions[0].edf[0] - 1.0 <= NULL_SPACE_EDF_TOL
    np.testing.assert_array_equal(line.sessions[0].in_null_space, [True])
    np.testing.assert_array_equal(line.search.at_boundary, [False])
    assert curve.sessions[0].edf[0] > 3.0
    np.testing.assert_array_equal(curve.sessions[0].in_null_space, [False])


#: Largest difference from mgcv allowed per quantity (``agreement``): absolute
#: for the log penalty, edf, REML score and curves, relative for the 0/1
#: coefficient, its standard error and the scale. The fixtures agree to about
#: 2e-8 in the log penalty and edf, 1e-11 in the score and 6e-10 in the curves.
MGCV_TOLERANCE = {
    "log_penalty": 1e-6,
    "edf": 1e-6,
    "reml": 1e-8,
    "curve": 1e-7,
    "se_curve": 1e-7,
    "coef_rew": 1e-7,
    "se_rew": 1e-7,
    "scale": 1e-7,
}


@pytest.mark.parametrize("name", list(DATA_SETS))
def test_matches_mgcv_fixtures(name: str) -> None:
    """The shared fitter's REML fit equals mgcv's on the committed fixtures."""
    mgcv = json.loads((FIXTURE_DIR / f"{name}.json").read_text())
    frame = make_data(name)
    assert len(frame) == mgcv["n_rows"]
    assert frame["y"].sum() == pytest.approx(mgcv["y_sum"], rel=1e-12)

    ours = fit_ours(name, frame)
    differences = agreement(name, ours, mgcv)

    assert ours["converged"] and mgcv["converged"]
    assert not ours["in_null_space"].any()
    expected_keys = {"log_penalty", "edf", "reml", "curve", "se_curve"}
    if name.startswith("poisson"):
        expected_keys |= {"coef_rew", "se_rew"}
    if name == "gaussian_one_smooth":
        expected_keys |= {"scale"}
    assert set(differences) == expected_keys
    for key, value in differences.items():
        assert value < MGCV_TOLERANCE[key], f"{key}: {value:.2e}"
