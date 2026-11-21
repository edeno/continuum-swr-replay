"""REML: its derivatives, the penalty search, pooling and the Gaussian scale."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
import scipy.optimize

from continuum_swr_replay.analysis.regression import reml
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
from continuum_swr_replay.analysis.regression.newton import (
    NO_FINITE_MAXIMUM,
    NOT_IDENTIFIED,
    STEP_LIMIT,
    penalized_newton,
)
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
from scripts.checks import compare_regression_to_neurospatial as neurospatial
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
    assert result.gradient is None  # not computed: nothing uses it
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
    np.testing.assert_allclose(twice.rho, once.rho, rtol=0, atol=1e-6)
    assert twice.score == pytest.approx(2 * once.score, rel=1e-10)
    assert once.search.n_grid_points == 25  # the product of five points per term


def test_more_than_three_terms_use_one_term_grids() -> None:
    """Four terms: two passes of one-term integer grids (2 x 4 x 31 points),
    then Newton to a stationary point."""
    lik, data, blocks = _session("poisson", 4, seed=10, n_rows=400)
    result = select_penalties(lik, [data], blocks)
    assert result.search.n_grid_points == 2 * 4 * 31
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
    # The intercept's covariance with the sum-to-zero smooth is zero but for
    # rounding (about 1e-18 beside entries near 1e-2), which differs between
    # the two computations: those entries are compared to 1e-14 of the largest.
    expected = session.scale * np.linalg.inv(X.T @ X + S_w)
    np.testing.assert_allclose(
        session.covariance,
        expected,
        rtol=1e-10,
        atol=1e-14 * np.max(np.abs(expected)),
    )
    # The search stopped at a stationary point of the profiled score.
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
    assert wiggle.search.n_outer_steps >= 1
    assert 3.0 < wiggle.sessions[0].edf[0] < 5.0


def test_shrunk_to_null_space() -> None:
    """A straight-line truth whose search stops on the plateau, short of the
    bound, is flagged by its edf (one, the line); at_boundary misses it. A
    curved truth on the same rows is not flagged.

    The score of the line falls toward its limit at the bound as
    ``c exp(-rho)``; Newton from the grid's center climbs the plateau and
    stops once the gain left, about ``|gradient|``, is below the decrement
    rule's reach (near log penalty 17.5 here). The grid start would begin at
    the bound itself, the lowest grid point of a score that falls to it."""
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

    line = refine_penalties(Gaussian(), [SessionData(X, line_y)], blocks, [6.0])
    curve = select_penalties(Gaussian(), [SessionData(X, curve_y)], blocks)

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

    assert mgcv["converged"]
    assert not ours["in_null_space"].any()
    expected_keys = {"log_penalty", "edf", "reml", "curve", "se_curve"}
    if name.startswith("poisson"):
        expected_keys |= {"coef_rew", "se_rew"}
    if name == "gaussian_one_smooth":
        expected_keys |= {"scale"}
    assert set(differences) == expected_keys
    for key, value in differences.items():
        assert value < MGCV_TOLERANCE[key], f"{key}: {value:.2e}"


def test_one_penalty_matches_neurospatial_fixture() -> None:
    """One penalty shared by three units: the score, the selected strength and
    the curves equal neurospatial's REML on the committed fixture.

    neurospatial's score leaves out terms that do not depend on the strength
    (the likelihood's offset term, ``log|S|_+`` at unit strength, the
    ``2 pi`` constant), so the two scores are compared after removing them.
    Its own selection stops at ``xatol = 1e-3`` in the log strength; the
    fixture also holds its objective's minimum found at ``xatol = 1e-10``.
    """
    fixture = json.loads(neurospatial.FIXTURE.read_text())
    data = neurospatial.make_data()
    assert data["counts"].sum() == fixture["counts_sum"]
    basis = PSplineBasis(0.0, 1.0, neurospatial.N_BASIS)
    np.testing.assert_allclose(
        basis.design(data["x"]), data["basis"], rtol=0, atol=1e-14
    )
    X = data["basis"] @ data["rotation"]
    offset = np.log(data["occupancy"])
    sessions = [
        SessionData(X, data["counts"][:, unit], offset)
        for unit in range(data["counts"].shape[1])
    ]
    blocks = [PenaltyBlock(np.arange(X.shape[1]), np.diag(data["weights"]))]
    assert blocks[0].rank == neurospatial.N_BASIS - 2

    constant = len(sessions) * (
        -0.5 * blocks[0].log_pdet
        - 0.5 * (X.shape[1] - blocks[0].rank) * np.log(2 * np.pi)
    ) - float((data["counts"] * offset[:, None]).sum())
    for rho, theirs in zip(
        fixture["score_grid_log_penalty"], fixture["score_grid_objective"], strict=True
    ):
        ours = pooled_reml(Poisson(), sessions, blocks, [rho], derivatives=False)
        assert ours.score - constant == pytest.approx(theirs, abs=1e-7)

    result = select_penalties(Poisson(), sessions, blocks)
    assert result.rho[0] == pytest.approx(fixture["tight_log_penalty"], abs=1e-6)
    assert abs(result.rho[0] - fixture["select_log_penalty"]) < fixture["select_xatol"]
    log_rate = np.column_stack([X @ session.fit.beta for session in result.sessions])
    np.testing.assert_allclose(log_rate, fixture["log_rate"], rtol=0, atol=1e-6)


def test_gaussian_is_invariant_to_the_outcome_scale() -> None:
    """An outcome scaled by 1, 1e3, 1e6 and 1e9: the Newton fit takes one exact step
    to the same curve up to the scale, and the selected log penalty (on the
    residual-sum-of-squares scale, as mgcv's ``sp``) is the same, with every
    grid point finite."""
    rng = np.random.default_rng(1)
    n_rows = 400
    s = rng.uniform(0.0, 4.0, n_rows)
    basis = PSplineBasis(0.0, 4.0, 15)
    B = basis.design(s)
    Z = sum_to_zero_constraint(B)
    X = np.column_stack([np.ones(n_rows), B @ Z])
    blocks = [PenaltyBlock(np.arange(1, 15), Z.T @ basis.penalty() @ Z)]
    base = np.sin(2.0 * s) + rng.normal(scale=0.5, size=n_rows)
    S = assemble_penalty(blocks, [2.0], X.shape[1])

    unit = penalized_newton(Gaussian(), X, base, 1.0, S, np.zeros(X.shape[1]))
    reference = select_penalties(Gaussian(), [SessionData(X, base)], blocks)
    for scale in (1.0, 1e3, 1e6, 1e9):
        fit = penalized_newton(
            Gaussian(), X, scale * base, 1.0, S, np.zeros(X.shape[1]), max_iter=1
        )
        assert fit.converged
        np.testing.assert_allclose(fit.beta / scale, unit.beta, rtol=1e-10)

        result = select_penalties(Gaussian(), [SessionData(X, scale * base)], blocks)
        assert result.search.n_grid_not_finite == 0
        np.testing.assert_allclose(result.rho, reference.rho, rtol=0, atol=1e-9)
        np.testing.assert_allclose(
            result.sessions[0].fit.beta / scale,
            reference.sessions[0].fit.beta,
            rtol=1e-8,
        )


@pytest.mark.parametrize("rho", [-10.0, 0.0, 10.0, 20.0])
def test_a_linear_term_beside_its_smooth_is_not_identified(rho: float) -> None:
    """A linear column of ``x`` beside a P-spline of ``x``: the spline's
    penalty leaves the straight line free, so the design cannot tell the two
    apart, at every penalty strength."""
    rng = np.random.default_rng(0)
    x = rng.uniform(size=500)
    design, penalty = _smooth(x, n_basis=10)
    X = np.column_stack([np.ones(500), x, design])
    y = rng.poisson(np.exp(0.5 + np.sin(3.0 * x))).astype(float)
    blocks = [PenaltyBlock(np.arange(2, 11), penalty)]
    score = reml_and_derivatives(Poisson(), X, y, np.zeros(500), blocks, [rho])
    assert score.score == np.inf
    assert score.fit.status == NOT_IDENTIFIED
    with pytest.raises(ValueError, match="session 0: design not identified"):
        fit_at_penalties(Poisson(), [SessionData(X, y, np.zeros(500))], blocks, [rho])


def test_an_unconverged_search_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A search that runs out of steps, or whose step no halving accepts, is
    an error that gives the last log penalties, the gradient and the steps."""
    lik, data, blocks = _session("poisson", 2, seed=7)
    monkeypatch.setattr(reml, "REML_MAX_STEPS", 1)
    with pytest.raises(ValueError, match=r"not converge.*1 step.*gradient"):
        refine_penalties(lik, [data], blocks, [-5.0, 15.0])
    with pytest.raises(ValueError, match=r"not converge.*1 step.*gradient"):
        select_penalties(lik, [data], blocks)
    monkeypatch.setattr(reml, "REML_MAX_STEPS", 50)
    monkeypatch.setattr(reml, "MAX_STEP_HALVINGS", 0)
    with pytest.raises(ValueError, match=r"no halving.*0 steps.*gradient"):
        refine_penalties(lik, [data], blocks, [-5.0, 15.0])


def _with_zero_column(data: SessionData, name: str | None = None) -> SessionData:
    """``data`` with an unpenalized column that is zero on every row, last."""
    X = np.column_stack([data.X, np.zeros(data.X.shape[0])])
    return SessionData(X, data.y, data.aux, name=name)


def test_a_failing_session_is_named() -> None:
    """The pooled score records the first session whose fit failed and why,
    and every error from a pooled fit names it."""
    lik, first, blocks = _session("poisson", 1, seed=4)
    _, second, _ = _session("poisson", 1, seed=5)
    broken = _with_zero_column(second, name="j16_20210710_02")
    pooled = pooled_reml(lik, [first, broken], blocks, [0.0])
    assert pooled.score == np.inf
    assert pooled.failed_session == 1
    assert pooled.failed_status == NOT_IDENTIFIED
    message = r"session 1 \(j16_20210710_02\).*design not identified"
    with pytest.raises(ValueError, match=r"1 of 2 sessions.*" + message):
        select_penalties(lik, [first, broken], blocks)
    with pytest.raises(ValueError, match=message):
        refine_penalties(lik, [first, broken], blocks, [0.0])
    with pytest.raises(ValueError, match=message):
        fit_at_penalties(lik, [first, broken], blocks, [0.0])
    unnamed = _with_zero_column(second)
    with pytest.raises(ValueError, match=r"session 0: design not identified"):
        fit_at_penalties(lik, [unnamed, first], blocks, [0.0])


def test_grid_points_that_fail_are_counted_by_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session whose inner fit fails at the weakest penalties only: the
    search goes on from the finite points and records how many failed and why."""
    lik, data, blocks = _session("poisson", 1, seed=4)
    newton_fit = reml.penalized_newton

    def fails_when_weak(*args, **kwargs):
        fit = newton_fit(*args, **kwargs)
        # The penalty's largest eigenvalue, whatever basis it is given in.
        strength = np.linalg.eigvalsh(args[4])[-1] / blocks[0].eigenvalues[-1]
        weak = strength < np.exp(-4.5)
        return replace(fit, status=STEP_LIMIT) if weak else fit

    monkeypatch.setattr(reml, "penalized_newton", fails_when_weak)
    result = select_penalties(lik, [data], blocks)
    assert result.search.n_grid_not_finite == 6  # log penalties -10 to -5
    assert result.search.grid_failures == {STEP_LIMIT: 6}


def test_an_inactive_column_is_left_out_of_its_session() -> None:
    """A session without support for an unpenalized 0/1 column marks it
    inactive: its fit, score and derivatives are those of its design built
    without the column, and the column's coefficient and covariance are NaN."""
    lik, first, blocks = _session("poisson", 2, seed=5)
    _, second, _ = _session("poisson", 2, seed=6)
    rng = np.random.default_rng(0)
    n_rows = first.X.shape[0]

    def with_indicator(data: SessionData, indicator: np.ndarray) -> SessionData:
        X = np.column_stack([data.X[:, :1], indicator, data.X[:, 1:]])
        return SessionData(X, data.y, data.aux)

    shifted = [PenaltyBlock(b.columns + 1, b.S) for b in blocks]
    first_full = with_indicator(first, (rng.uniform(size=n_rows) < 0.5).astype(float))
    active = np.ones(first_full.X.shape[1], dtype=bool)
    active[1] = False
    second_inactive = replace(
        with_indicator(second, np.zeros(n_rows)), active=active, name="no reward"
    )

    rho = np.array([0.5, 2.0])
    pooled = pooled_reml(lik, [first_full, second_inactive], shifted, rho)
    one = _score(lik, first_full, shifted, rho)
    two = _score(lik, second, blocks, rho)  # the design without the column
    assert pooled.score == pytest.approx(one.score + two.score, rel=1e-12)
    np.testing.assert_allclose(pooled.gradient, one.gradient + two.gradient)
    np.testing.assert_allclose(pooled.hessian, one.hessian + two.hessian)

    result = fit_at_penalties(lik, [first_full, second_inactive], shifted, rho)
    session = result.sessions[1]
    reference = fit_at_penalties(lik, [second], blocks, rho).sessions[0]
    assert np.isnan(session.fit.beta[1])
    np.testing.assert_allclose(session.fit.beta[active], reference.fit.beta)
    assert np.all(np.isnan(session.covariance[1])) and np.all(
        np.isnan(session.covariance[:, 1])
    )
    np.testing.assert_allclose(
        session.covariance[np.ix_(active, active)], reference.covariance
    )
    np.testing.assert_allclose(session.edf, reference.edf)
    assert session.edf_total == pytest.approx(reference.edf_total)

    selected = select_penalties(lik, [first_full, second_inactive], shifted)
    assert np.isnan(selected.sessions[1].fit.beta[1])
    assert np.all(np.isfinite(selected.sessions[0].fit.beta))


def test_an_inactive_penalized_column_is_refused() -> None:
    """The penalty identifies a smooth's columns: none may be inactive."""
    lik, data, blocks = _session("poisson", 1, seed=5)
    active = np.ones(data.X.shape[1], dtype=bool)
    active[blocks[0].columns[2]] = False
    with pytest.raises(ValueError, match=r"session 0.*column 3.*penalty"):
        pooled_reml(lik, [replace(data, active=active)], blocks, [0.0])


def test_a_penalty_that_is_not_symmetric_positive_semidefinite_is_refused() -> None:
    _, _, blocks = _session("poisson", 1, seed=1)
    S = blocks[0].S.copy()
    asymmetric = S.copy()
    asymmetric[0, 1] += 1e-3
    with pytest.raises(ValueError, match="not symmetric"):
        PenaltyBlock(blocks[0].columns, asymmetric)
    with pytest.raises(ValueError, match="not positive semidefinite"):
        PenaltyBlock(blocks[0].columns, S - 0.01 * np.eye(S.shape[0]))
    # Rounding is not refused.
    PenaltyBlock(blocks[0].columns, S + 1e-15 * np.triu(np.ones_like(S), 1))


def test_an_input_error_names_the_session() -> None:
    lik, first, blocks = _session("poisson", 1, seed=4)
    _, second, _ = _session("poisson", 1, seed=5)
    offset = second.aux.copy()
    offset[7] = -np.inf
    bad = SessionData(second.X, second.y, offset, name="peanut_02")
    with pytest.raises(ValueError, match=r"session 1 \(peanut_02\).*zero exposure"):
        select_penalties(lik, [first, bad], blocks)


def test_softmax_penalty_selection_is_stationary() -> None:
    """The grouped softmax's search converges, and at its optimum the central
    differences of the score agree with the analytic gradient (both near zero)."""
    rng = np.random.default_rng(1)
    n_events, n_arms = 300, 9
    n_rows = n_events * n_arms
    s = rng.uniform(0.0, 4.0, n_rows)
    basis = PSplineBasis(0.0, 4.0, 12)
    B = basis.design(s)
    Z = sum_to_zero_constraint(B)
    X = np.column_stack([B @ Z, rng.normal(size=n_rows)])
    eta = np.sin(1.5 * s) + 0.4 * X[:, -1]
    groups = np.repeat(np.arange(n_events), n_arms)
    weight = np.exp(eta).reshape(-1, n_arms)
    p = weight / weight.sum(axis=1, keepdims=True)
    y = np.concatenate([rng.dirichlet(10.0 * row) for row in p])
    data = SessionData(X, y, groups)
    blocks = [PenaltyBlock(np.arange(11), Z.T @ basis.penalty() @ Z)]

    result = select_penalties(GroupedSoftmax(), [data], blocks)

    assert not result.search.at_boundary.any()
    assert 3.0 < result.sessions[0].edf[0] < 8.0
    warm = result.sessions[0].fit.beta
    h = 1e-4
    numeric = (
        _score(GroupedSoftmax(), data, blocks, result.rho + h, warm).score
        - _score(GroupedSoftmax(), data, blocks, result.rho - h, warm).score
    ) / (2 * h)
    assert abs(result.gradient[0]) < 1e-6
    assert abs(numeric) < 1e-6
    assert abs(numeric - result.gradient[0]) < 1e-6


def test_three_penalties_over_several_sessions() -> None:
    """Three smooths shared by six Poisson sessions of 4,000 rows: the grid
    (125 points) and Newton reach a stationary point with a positive definite
    Hessian, and a search started elsewhere ends there too."""
    sessions = []
    for seed in range(6):
        lik, data, blocks = _session("poisson", 3, seed=40 + seed, n_rows=4000)
        sessions.append(data)
    selected = select_penalties(lik, sessions, blocks)
    assert selected.search.n_grid_points == 125
    assert selected.search.n_grid_not_finite == 0
    assert not selected.search.at_boundary.any()
    pooled = pooled_reml(lik, sessions, blocks, selected.rho)
    assert np.all(np.linalg.eigvalsh(pooled.hessian) > 0)
    decrement = 0.5 * pooled.gradient @ np.linalg.solve(pooled.hessian, pooled.gradient)
    assert decrement < 1e-8
    refined = refine_penalties(lik, sessions, blocks, selected.rho + 1.0)
    np.testing.assert_allclose(refined.rho, selected.rho, rtol=0, atol=1e-6)


def test_no_penalty_fit_at_penalties() -> None:
    """With no penalty block: the unpenalized Newton fit, its score
    ``-l + log|H| / 2 - p log(2 pi) / 2`` and empty edf."""
    lik, data, _ = _session("poisson", 1, seed=3)
    result = fit_at_penalties(lik, [data], [], [])
    session = result.sessions[0]
    p = data.X.shape[1]
    direct = penalized_newton(
        lik, data.X, data.y, data.aux, np.zeros((p, p)), np.zeros(p)
    )
    np.testing.assert_allclose(session.fit.beta, direct.beta, rtol=0, atol=1e-10)
    np.testing.assert_allclose(session.covariance, np.linalg.inv(direct.A), rtol=1e-10)
    _, log_det = np.linalg.slogdet(direct.A)
    expected = -direct.loglik + 0.5 * log_det - 0.5 * p * np.log(2 * np.pi)
    assert result.score == pytest.approx(expected, rel=1e-12)
    assert session.edf.shape == (0,)
    assert session.edf_total == pytest.approx(p, rel=1e-10)


def _content_session(
    seed: int, n_events: int = 140, n_arms: int = 6
) -> tuple[SessionData, list[PenaltyBlock]]:
    """A grouped-softmax session of the content model's shape.

    Each event has ``n_arms`` candidate rows and one 0/1 response; a cubic
    P-spline of a clock in [0, 5] (10 basis functions, 9 after the
    sum-to-zero constraint) with a linear truth of slope -1.2, so the fit at
    a strong penalty has null-space coefficients of several units, and two
    unpenalized 0/1 role columns (the first row of each event; one other row).
    """
    rng = np.random.default_rng(seed)
    n_rows = n_events * n_arms
    x = rng.uniform(0.0, 5.0, n_rows)
    basis = PSplineBasis(0.0, 5.0, 10)
    B = basis.design(x)
    Z = sum_to_zero_constraint(B)
    role = np.zeros((n_rows, 2))
    first = np.arange(n_events) * n_arms
    role[first, 0] = 1.0
    role[first + 1 + rng.integers(0, n_arms - 1, n_events), 1] = 1.0
    X = np.column_stack([B @ Z, role])
    eta = -1.2 * x + role @ np.array([1.0, -0.5])
    weight = np.exp(eta).reshape(n_events, n_arms)
    p = weight / weight.sum(axis=1, keepdims=True)
    y = np.zeros((n_events, n_arms))
    y[np.arange(n_events), [rng.choice(n_arms, p=row) for row in p]] = 1.0
    groups = np.repeat(np.arange(n_events), n_arms)
    blocks = [PenaltyBlock(np.arange(9), Z.T @ basis.penalty() @ Z)]
    return SessionData(X, y.ravel(), groups), blocks


def test_inner_fits_converge_at_strong_penalties() -> None:
    """At log penalties 15 to 20 every inner fit converges, and at 20 the fit
    is the fit in the penalty's null space (the smooth a straight line) to
    1e-6 in the linear predictor; the difference is of order exp(-20).

    Computed with the penalty matrix as given, ``b' S b`` loses
    ``eps |S| |b|^2`` to rounding, more than the gain of the last Newton steps,
    and some of these fits stopped as "step halving failed"."""
    for seed in range(10):
        data, blocks = _content_session(seed)
        for rho in np.arange(15.0, 21.0):
            pooled = pooled_reml(GroupedSoftmax(), [data], blocks, [rho])
            assert np.isfinite(pooled.score), (seed, rho, pooled.failure)
        strong = pooled.sessions[0].fit
        null = blocks[0].null_space
        X_null = np.column_stack([data.X[:, :9] @ null, data.X[:, 9:]])
        p = X_null.shape[1]
        line = penalized_newton(
            GroupedSoftmax(),
            X_null,
            data.y,
            data.aux,
            np.zeros((p, p)),
            np.zeros(p),
        )
        assert line.converged
        np.testing.assert_allclose(
            data.X @ strong.beta, X_null @ line.beta, rtol=0, atol=1e-6
        )


def test_a_flat_pooled_score_ends_converged_in_the_null_space() -> None:
    """Forty content sessions whose clock truth is a straight line: the pooled
    score falls toward its limit at the bound by less than 1e-6 from log
    penalty 14 on. The search ends converged, every grid point finite, at a
    point where each session's smooth is the line (edf 1, ``in_null_space``),
    and the score is within the decrement rule's reach of its limit."""
    sessions = []
    for seed in range(40):
        data, blocks = _content_session(seed)
        sessions.append(data)
    result = select_penalties(GroupedSoftmax(), sessions, blocks)

    assert result.search.n_grid_not_finite == 0
    assert result.rho[0] > 15.0
    for session in result.sessions:
        assert session.edf[0] == pytest.approx(1.0, abs=1e-5)
        np.testing.assert_array_equal(session.in_null_space, [True])
    limit = pooled_reml(GroupedSoftmax(), sessions, blocks, [LOG_PENALTY_BOUNDS[1]])
    # Warm and cold inner starts differ by the inner fits' accuracy (6e-10 here).
    assert abs(result.score - limit.score) < 1e-6


def test_a_gain_below_the_rounding_of_the_score_ends_the_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the decrement's tolerance at zero, the search still stops: once the
    gain the next step predicts is below the rounding of the pooled score,
    ``8 eps max(1, |V|)``, no halving can show it, and the point is the optimum
    to rounding."""
    lik, data, blocks = _session("poisson", 2, seed=7)
    reference = select_penalties(lik, [data], blocks)
    monkeypatch.setattr(reml, "REML_DECREMENT_TOL", 0.0)
    result = select_penalties(lik, [data], blocks)
    np.testing.assert_allclose(result.rho, reference.rho, rtol=0, atol=1e-6)
    assert result.score <= reference.score + 8 * np.finfo(float).eps * abs(
        reference.score
    )


def _with_arm_columns(
    data: SessionData, seed: int, separated: bool, n_arms: int = 6
) -> SessionData:
    """``data`` (a :func:`_content_session`) with two unpenalized 0/1 columns
    on the second and third rows of events, an arm and a role.

    Separated: the arm marks the second row of the first 40 events, never
    chosen (a chosen second row is moved to the first), and the third row of
    the next 40, the role marks those same third rows. Each column has the
    response on part of its rows, so a rule on one column at a time passes
    both; the direction "arm down, role up" lowers only rows never chosen,
    and the likelihood has no finite maximum along it.
    """
    rng = np.random.default_rng(seed)
    n_events = data.X.shape[0] // n_arms
    y = data.y.reshape(n_events, n_arms).copy()
    arm = np.zeros((n_events, n_arms))
    role = np.zeros((n_events, n_arms))
    if separated:
        y[:40, 0] += y[:40, 1]
        y[:40, 1] = 0.0
        arm[:40, 1] = 1.0
        arm[40:80, 2] = 1.0
        role[40:80, 2] = 1.0
    else:
        arm[rng.uniform(size=n_events) < 0.5, 2] = 1.0
        role[rng.uniform(size=n_events) < 0.3, 2] = 1.0
    X = np.column_stack([data.X, arm.ravel(), role.ravel()])
    return SessionData(X, y.ravel(), data.aux, name=f"s{seed}")


def test_separated_sessions_are_set_aside_before_the_search() -> None:
    """One of ten sessions has no finite maximum (an arm column with a role
    column separates): it fails at the reference point (every log penalty at
    the grid center) and is set aside, named, with its status and the columns
    running away; the search over the other nine is the search without it.
    Three of ten (30 %, above ``MAX_FAILING_SHARE``) is an error naming each."""
    sessions = []
    for seed in range(10):
        data, blocks = _content_session(seed)
        sessions.append(_with_arm_columns(data, seed, separated=seed == 4))
    arm, role = 11, 12

    pooled = pooled_reml(GroupedSoftmax(), sessions, blocks, [6.0])
    assert pooled.failed_session == 4
    assert pooled.failure == (
        "session 4 (s4): no finite maximum (columns running away: [11, 12])"
    )

    result = select_penalties(GroupedSoftmax(), sessions, blocks)
    (left_out,) = result.search.set_aside
    assert (left_out.index, left_out.name, left_out.status) == (
        4,
        "s4",
        NO_FINITE_MAXIMUM,
    )
    assert sorted(left_out.running_away.tolist()) == [arm, role]
    assert result.search.share_set_aside == pytest.approx(0.1)
    np.testing.assert_array_equal(result.search.reference_rho, [6.0])
    assert len(result.sessions) == 10
    assert np.all(np.isnan(result.sessions[4].fit.beta))
    assert np.isnan(result.sessions[4].edf[0])
    assert result.sessions[4].fit.status == NO_FINITE_MAXIMUM

    without = select_penalties(GroupedSoftmax(), sessions[:4] + sessions[5:], blocks)
    np.testing.assert_array_equal(result.rho, without.rho)
    assert result.score == without.score
    assert without.search.set_aside == ()
    for kept, alone in zip(
        result.sessions[:4] + result.sessions[5:], without.sessions, strict=True
    ):
        np.testing.assert_array_equal(kept.fit.beta, alone.fit.beta)

    # Leaving the running-away columns out of the session gives it a fit.
    active = np.ones(sessions[4].X.shape[1], dtype=bool)
    active[left_out.running_away] = False
    repaired = replace(sessions[4], active=active)
    assert np.isfinite(pooled_reml(GroupedSoftmax(), [repaired], blocks, [6.0]).score)

    many = [
        _with_arm_columns(_content_session(seed)[0], seed, separated=seed in (1, 4, 7))
        for seed in range(10)
    ]
    message = (
        r"3 of 10 sessions \(30%, more than MAX_FAILING_SHARE = 20%\): "
        r"session 1 \(s1\): no finite maximum \(columns running away: \[1[12], "
        r"1[12]\]\); session 4 \(s4\).*; session 7 \(s7\)"
    )
    with pytest.raises(ValueError, match=message):
        select_penalties(GroupedSoftmax(), many, blocks)
