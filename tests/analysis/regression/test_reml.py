"""REML: its derivatives, the penalty search, pooling and the Gaussian scale."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.optimize

from continuum_swr_replay.analysis.regression import reml, runtime
from continuum_swr_replay.analysis.regression.constants import (
    LOG_PENALTY_BOUNDS,
    NULL_SPACE_EDF_TOL,
    REFIT_MAX_ROUNDS,
)
from continuum_swr_replay.analysis.regression.likelihoods import (
    Bernoulli,
    Gaussian,
    GroupedSoftmax,
    Poisson,
)
from continuum_swr_replay.analysis.regression.newton import (
    CONVERGED,
    HALVING_FAILED,
    NO_FINITE_MAXIMUM,
    NOT_IDENTIFIED,
    STEP_LIMIT,
    penalized_newton,
)
from continuum_swr_replay.analysis.regression.padding import Groups
from continuum_swr_replay.analysis.regression.pspline import PSplineBasis
from continuum_swr_replay.analysis.regression.reml import (
    PenaltyBlock,
    SessionData,
    assemble_penalty,
    fit_at_penalties,
    in_null_space,
    pooled_reml,
    refine_penalties,
    refit_session,
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
    stop_bins,
)

from .conftest import (
    draw_response,
    numeric_gradient,
    numeric_hessian,
    relative_error,
    stack_smooths,
)

FAMILIES = ("poisson", "bernoulli", "gaussian", "grouped_softmax")
N_BASIS = 7  # 6 constrained columns per smooth


def _share_without_fit(search) -> float:
    """The share of the sessions given (in the choice and set aside) that end
    without a fit: set aside and not fitted by their refits."""
    n_sessions = search.in_choice.size + len(search.set_aside)
    return sum(not r.fitted for r in search.set_aside) / n_sessions


def _smooth(x: np.ndarray, n_basis: int = N_BASIS) -> tuple[np.ndarray, np.ndarray]:
    """The constrained P-spline design of ``x`` over [0, 1] and its penalty."""
    design, penalty, _ = PSplineBasis(0.0, 1.0, n_basis).constrained(x)
    return design, penalty


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
    X, blocks = stack_smooths(*parts, *(_smooth(x[j]) for j in range(n_smooths)))
    if family == "poisson":
        offset = np.log(rng.uniform(0.5, 2.0, size=n_rows))
        y = draw_response(rng, "poisson", 0.7 + truth + offset)
        return Poisson(), SessionData(X, y, offset), blocks
    if family == "bernoulli":
        y = draw_response(rng, "bernoulli", truth)
        return Bernoulli(), SessionData(X, y), blocks
    if family == "gaussian":
        y = 1.0 + truth + rng.normal(scale=0.5, size=n_rows)
        return Gaussian(), SessionData(X, y), blocks
    groups = np.repeat(np.arange(n_rows // 4), 4)
    y = draw_response(rng, "shares", truth, total=30.0)
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

    gradient = numeric_gradient(V, rho, 1e-4)
    hessian = numeric_hessian(V, rho, 1e-3)

    assert relative_error(at.gradient, gradient) < 1e-5
    assert relative_error(at.hessian, hessian) < 1e-5
    np.testing.assert_array_equal(at.hessian, at.hessian.T)


def _score_in_jax(lik, data: SessionData, blocks, beta):
    """The REML score as a JAX function of ``rho``, on the unpadded rows, from
    the log-likelihood alone: the information is ``-jax.hessian`` of it, and
    the fit ``b(rho)`` is four Newton steps from ``beta``, the fit at the
    point of differentiation. A Newton step's derivative in its start is zero
    at the optimum, so the steps' derivatives in ``rho`` there are the
    implicit derivatives of the optimum, to second order after two steps. The
    Gaussian's profiled score is written in closed form. It is written in the
    penalties' eigenbasis, as the fitter computes: with ``S`` as given, the
    rounding of its null space times ``exp(rho)`` moved the score's
    derivatives by up to 8e-7 of themselves at log penalties 8 to 12. Called
    with JAX's 64-bit types on."""
    p = data.X.shape[1]
    Q = reml._rotate(blocks, np.eye(p))
    blocks = reml._eigenbasis(blocks, p)
    X, y = jnp.asarray(data.X @ Q), jnp.asarray(data.y)
    beta = Q.T @ beta
    units = jnp.asarray(reml._penalty_stack(blocks, np.zeros(len(blocks)), p))
    ranks = jnp.asarray([b.rank for b in blocks], dtype=float)
    constant = sum(b.log_pdet for b in blocks)
    n_null = p - sum(b.rank for b in blocks)

    def penalty(rho):
        return jnp.tensordot(jnp.exp(rho), units, axes=1)

    if isinstance(lik, Gaussian):

        def V(rho):
            A = X.T @ X + penalty(rho)
            b = jnp.linalg.solve(A, X.T @ y)
            residual = y - X @ b
            dof = y.size - n_null
            D_p = residual @ residual + b @ penalty(rho) @ b
            return (
                0.5 * dof * (1.0 + jnp.log(2 * jnp.pi) + jnp.log(D_p / dof))
                - 0.5 * (ranks @ rho + constant)
                + 0.5 * jnp.linalg.slogdet(A)[1]
            )

        return V
    aux = data.aux
    if isinstance(lik, GroupedSoftmax):
        aux = Groups(jnp.asarray(data.aux), int(data.aux.max()) + 1)
    weight = jnp.ones(y.size)

    def loglik(b):
        return lik._jax_loglik(X @ b, y, aux, weight)

    def V(rho):
        S = penalty(rho)
        b = jax.lax.stop_gradient(jnp.asarray(beta))
        for _ in range(4):
            A = -jax.hessian(loglik)(b) + S
            b = b + jnp.linalg.solve(A, jax.grad(loglik)(b) - S @ b)
        A = -jax.hessian(loglik)(b) + S
        return (
            -loglik(b)
            + 0.5 * b @ S @ b
            - 0.5 * (ranks @ rho + constant)
            + 0.5 * jnp.linalg.slogdet(A)[1]
            - 0.5 * n_null * jnp.log(2 * jnp.pi)
        )

    return V


def _with_a_ridge(family: str, seed: int):
    """Two smooths (penalties with a null space) and a third block, a ridge on
    three more columns (no null space), on 120 rows."""
    lik, data, blocks = _session(family, 2, seed=seed, n_rows=120)
    rng = np.random.default_rng(seed + 100)
    p = data.X.shape[1]
    X = np.column_stack([data.X, rng.normal(scale=0.5, size=(data.y.size, 3))])
    ridge = PenaltyBlock(np.arange(p, p + 3), np.eye(3))
    return lik, replace(data, X=X), [*blocks, ridge]


@pytest.mark.parametrize(
    "rho", [[0.5, 2.0, 1.0], [8.0, 12.0, 10.0]], ids=["moderate", "strong"]
)
@pytest.mark.parametrize("n_blocks", [1, 2, 3])
@pytest.mark.parametrize("family", FAMILIES)
def test_gradient_and_hessian_equal_autodiff_of_the_score(
    family: str, n_blocks: int, rho: list[float]
) -> None:
    """The hand-written gradient and Hessian in ``rho`` (the information's
    directional derivatives, the traces against ``A^-1``, the profiled
    Gaussian's own) equal ``jax.grad`` and ``jax.hessian`` of the score written
    in JAX (:func:`_score_in_jax`), on 120 rows, unpadded: one, two and three
    penalties (the third a ridge, with no null space), at moderate and strong
    log penalties. The
    finite differences of the score check the same derivatives on the padded
    rows (:func:`test_gradient_and_hessian_finite_differences`)."""
    if n_blocks == 3:
        lik, data, blocks = _with_a_ridge(family, seed=5)
    else:
        lik, data, blocks = _session(family, n_blocks, seed=5, n_rows=120)
    rho = np.array(rho[:n_blocks])
    # Warm-started at its own fit, so that it lands on the optimum to rounding
    # (from a cold start one softmax fit stopped 2e-13 of the score short).
    at = _score(lik, data, blocks, rho, _score(lik, data, blocks, rho).fit.beta)
    assert at.fit.converged
    with jax.enable_x64(True):
        V = _score_in_jax(lik, data, blocks, at.fit.beta)
        rho_jax = jnp.asarray(rho)
        score = float(V(rho_jax))
        gradient = np.asarray(jax.grad(V)(rho_jax))
        hessian = np.asarray(jax.hessian(V)(rho_jax))
    assert score == pytest.approx(at.score, rel=1e-12)
    # 5e-11 of the largest entry, at least 1e-14: up to 7e-12 of it measured
    # (the Bernoulli's Hessian at the strong penalties), most below 1e-13; the
    # softmax's one smooth at the strong penalties, whose gradient is 9e-6
    # (a score near its limit in the null space), differed by 9e-16.
    for got, want in [(at.gradient, gradient), (at.hessian, hessian)]:
        floor = max(5e-11 * np.abs(want).max(), 1e-14)
        np.testing.assert_allclose(got, want, rtol=0, atol=floor)


@pytest.mark.parametrize("family", FAMILIES)
def test_no_penalty_blocks_give_empty_derivatives(family: str) -> None:
    """With no penalty blocks the score is the fit's, the gradient has shape
    ``(0,)`` and the Hessian ``(0, 0)``, through ``reml_and_derivatives`` and
    ``pooled_reml``, with or without the derivatives."""
    lik, data, _ = _session(family, 1, seed=1)
    alone = reml_and_derivatives(
        lik, data.X, data.y, data.aux, [], np.array([]), derivatives=False
    )
    at = reml_and_derivatives(lik, data.X, data.y, data.aux, [], np.array([]))
    assert at.fit.converged and np.isfinite(at.score)
    assert at.score == alone.score and alone.gradient is None
    assert at.gradient.shape == (0,) and at.gradient.dtype == np.float64
    assert at.hessian.shape == (0, 0) and at.hessian.dtype == np.float64
    np.testing.assert_array_equal(at.covariance, alone.covariance)
    pooled = pooled_reml(lik, [data], [], [])
    assert pooled.score == at.score
    assert pooled.gradient.shape == (0,) and pooled.hessian.shape == (0, 0)


@pytest.mark.parametrize("family", FAMILIES)
def test_a_design_not_finite_is_counted_before_it_is_rotated(family: str) -> None:
    """One infinite and one NaN entry are counted as two: the rotation into
    the eigenbasis would spread each over the columns of its block."""
    lik, data, blocks = _session(family, 2, seed=3)
    X = data.X.copy()
    X[0, blocks[0].columns[0]] = np.inf
    X[5, blocks[1].columns[2]] = np.nan
    with pytest.raises(ValueError, match="design not finite in 2 entries"):
        reml_and_derivatives(lik, X, data.y, data.aux, blocks, [0.0, 0.0])


@pytest.mark.parametrize("family", ["poisson", "bernoulli", "grouped_softmax"])
def test_a_converged_fit_whose_a_is_not_positive_definite_raises(
    family: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A converged fit whose ``A`` cannot be factored (forced here) stops the
    derivatives with ``LinAlgError`` rather than passing NaN on. The
    Gaussian makes its ``A`` from the rows (the test below)."""
    lik, data, blocks = _session(family, 1, seed=2)
    newton_fit = reml._fit_one

    def cracked(*args, **kwargs):
        fit, problem = newton_fit(*args, **kwargs)
        return replace(fit, A=-np.eye(fit.A.shape[0])), problem

    monkeypatch.setattr(reml, "_fit_one", cracked)
    with pytest.raises(np.linalg.LinAlgError, match="not positive definite"):
        _score(lik, data, blocks, [0.5])


def test_the_profiled_gaussian_flags_a_gram_it_cannot_factor() -> None:
    """The profiled Gaussian's kernel returns whether ``X'X + S`` was
    factored; the host raises on False (as for the Laplace families)."""
    _, data, blocks = _session("gaussian", 1, seed=2)
    p = data.X.shape[1]
    S_j = reml._penalty_stack(blocks[:1], np.zeros(1), p)
    rows = Gaussian().pad(data.X, data.y, None)
    with jax.enable_x64(True):

        def factored(S):
            outputs = reml._profiled_gaussian(
                rows.X,
                rows.y[None],
                rows.weight,
                S,
                np.zeros(p),
                np.ones((1, p)),
                np.float64(0.0),
                np.int64(data.y.size),
                np.int64(data.y.size - 2),
                S_j,
                np.array([float(blocks[0].rank)]),
            )
            return bool(outputs[-1])

        assert factored(S_j[0])
        assert not factored(-1e6 * np.eye(p))
    with pytest.raises(np.linalg.LinAlgError, match="not positive definite"):
        reml._require_factored(False)


@pytest.mark.parametrize("family", ["bernoulli", "grouped_softmax"])
def test_an_inner_fit_without_a_finite_maximum_has_no_derivatives(
    family: str,
) -> None:
    """An unpenalized column that separates the outcome: the inner fit has no
    finite maximum, the score is infinite and there is no gradient or
    Hessian."""
    lik, data, blocks = _session(family, 1, seed=6)
    if family == "bernoulli":
        separating = data.y.copy()
    else:  # the largest share of each group of four
        shares = data.y.reshape(-1, 4)
        separating = (shares == shares.max(axis=1, keepdims=True)).ravel() * 1.0
        data = replace(data, y=separating)
    X = np.column_stack([data.X, separating])
    at = reml_and_derivatives(lik, X, data.y, data.aux, blocks, [1.0])
    assert at.fit.status == NO_FINITE_MAXIMUM
    assert at.score == np.inf and at.gradient is None and at.hessian is None


def test_block_rank_is_the_penalty_rank() -> None:
    """The constrained second-difference penalty has rank n_basis - 2."""
    _, data, blocks = _session("poisson", 1, seed=1)
    assert blocks[0].rank == N_BASIS - 2
    eigenvalues = np.linalg.eigvalsh(blocks[0].S)
    positive = eigenvalues[eigenvalues > 1e-9 * eigenvalues.max()]
    assert blocks[0].log_pdet == pytest.approx(np.log(positive).sum(), rel=1e-12)


def test_a_block_moved_to_other_columns_keeps_its_size() -> None:
    """``at_columns`` copies the decomposition to other columns and refuses a
    column set of another length, as building the block again would."""
    block = PenaltyBlock(np.arange(2, 5), np.diag([1.0, 2.0, 0.0]))
    moved = block.at_columns(np.array([0, 3, 7]))
    np.testing.assert_array_equal(moved.columns, [0, 3, 7])
    assert moved.S is block.S and moved.rank == block.rank == 2
    with pytest.raises(ValueError, match="the penalty has 3 columns"):
        block.at_columns(np.array([0, 3]))


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
    with pytest.raises(ValueError, match="design not identified.*cannot see"):
        select_penalties(lik, [SessionData(X, data.y, data.aux)], blocks)


def test_a_session_whose_design_is_not_identified_raises() -> None:
    """Ten sessions, one with a 0/1 column that is zero on every row (a term
    the session has no support for): the search does not set it aside but
    raises, naming the session, the column and the remedy."""
    rng = np.random.default_rng(0)
    basis = PSplineBasis(0.0, 4.0, 10)
    sessions = []
    for index in range(10):
        x = rng.uniform(0.0, 4.0, 600)
        design, penalty, _ = basis.constrained(x)
        has_reward = index != 3  # session 3 has no rewarded row
        reward = (rng.uniform(size=600) < 0.5).astype(float) * has_reward
        X, blocks = stack_smooths(np.ones(600), reward, (design, penalty))
        y = draw_response(rng, "poisson", -1.0 + np.sin(x) + 0.3 * reward)
        sessions.append(SessionData(X, y, np.zeros(600), name=f"s{index}"))
    message = (
        r"session 3 \(s3\): design not identified .*cannot see .*: "
        r"\[1\]\).*inactive \(SessionData.active\)"
    )
    with pytest.raises(ValueError, match=message):
        select_penalties(Poisson(), sessions, blocks)


def test_a_numerical_failure_at_the_reference_point_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A session whose fit at the reference point ends as "step halving
    failed" (a numerical failure, not a lack of support) is not set aside:
    the search raises, naming it, though it is one session of ten."""
    sessions = []
    for seed in range(10):
        lik, data, blocks = _session("poisson", 1, seed=4 + seed)
        sessions.append(data)
    sessions[1] = replace(sessions[1], name="peanut_02")
    newton_fit = reml._fit_one

    def fails_for_one(*args, **kwargs):
        fit, problem = newton_fit(*args, **kwargs)
        mine = args[2] is sessions[1].y
        return (replace(fit, status=HALVING_FAILED) if mine else fit), problem

    monkeypatch.setattr(reml, "_fit_one", fails_for_one)
    with pytest.raises(ValueError, match=r"session 1 \(peanut_02\): step halving"):
        select_penalties(lik, sessions, blocks)


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

    The mgcv comparison's stop bins (:func:`stop_bins`: 350 stops of
    lognormal duration, 15 ms bins over the first 4 s), the counts and exposures
    summed over bins at the same time (which leaves the likelihood unchanged
    up to a constant); a cubic P-spline with 15 basis functions over [0, 4].
    """
    rng = np.random.default_rng(seed)
    bins = stop_bins(rng)
    exposure, clock = bins["expo"].to_numpy(), bins["s"].to_numpy()
    y = rng.poisson(n_events / exposure.sum() * exposure).astype(float)
    _, cell = np.unique(bins["k"].to_numpy(), return_inverse=True)
    cell_exposure = np.bincount(cell, weights=exposure)
    cell_clock = np.bincount(cell, weights=clock * exposure) / cell_exposure
    design, penalty, _ = PSplineBasis(0.0, 4.0, 15).constrained(cell_clock)
    X, blocks = stack_smooths(np.ones(cell_clock.size), (design, penalty))
    data = SessionData(X, np.bincount(cell, weights=y), np.log(cell_exposure))
    return data, blocks


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
    X, blocks = stack_smooths(np.ones(n_rows), _smooth(x))
    noise = rng.normal(size=n_rows)

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
    X, blocks = stack_smooths(np.ones(n_rows), _smooth(x, n_basis=10))
    line_y = (x - 0.5) + rng.normal(size=n_rows)
    curve_y = np.sin(2 * np.pi * x) + rng.normal(size=n_rows)

    line = refine_penalties(Gaussian(), [SessionData(X, line_y)], blocks, [6.0])
    curve = select_penalties(Gaussian(), [SessionData(X, curve_y)], blocks)

    assert line.sessions[0].edf[0] - 1.0 <= NULL_SPACE_EDF_TOL
    np.testing.assert_array_equal(in_null_space(line.sessions[0], blocks), [True])
    np.testing.assert_array_equal(line.search.at_boundary, [False])
    assert curve.sessions[0].edf[0] > 3.0
    np.testing.assert_array_equal(in_null_space(curve.sessions[0], blocks), [False])


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
    design, penalty, _ = PSplineBasis(0.0, 4.0, 15).constrained(s)
    X, blocks = stack_smooths(np.ones(n_rows), (design, penalty))
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
        assert sum(result.search.grid_failures.values()) == 0
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
    X, blocks = stack_smooths(np.ones(500), x, _smooth(x, n_basis=10))
    y = draw_response(rng, "poisson", 0.5 + np.sin(3.0 * x))
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


def test_the_coordinates_named_do_not_depend_on_the_basis() -> None:
    """Two bases of one space of combinations name the same coordinates: those
    with more than a tenth of their length in the space."""
    first = np.zeros((8, 2))
    first[[0, 1], 0] = 1.0
    first[[2, 3], 1] = 1.0
    first[4], first[5] = [0.13, 0.13], [0.13, -0.13]  # below a tenth in each vector
    first[6], first[7] = [0.04, 0.04], [0.04, -0.04]  # below a tenth in the space
    first /= np.linalg.norm(first, axis=0)
    turn = np.array([[1.0, -1.0], [1.0, 1.0]]) / np.sqrt(2)
    second = first @ turn  # coordinates 4 and 5 are now each in one vector
    for basis in (first, second):
        np.testing.assert_allclose(basis.T @ basis, np.eye(2), atol=1e-12)
    assert np.abs(first[[4, 5]]).max() < 0.1 < np.linalg.norm(first[4])
    np.testing.assert_array_equal(reml._named_coordinates(first), np.arange(6))
    np.testing.assert_array_equal(reml._named_coordinates(second), np.arange(6))


def test_a_smooth_with_no_values_is_named_as_not_seen() -> None:
    """A session whose smooth columns are zero on every row: the likelihood
    cannot see the smooth's unpenalized direction, and the error says so."""
    lik, data, blocks = _session("poisson", 1, seed=4)
    X = data.X.copy()
    X[:, blocks[0].columns] = 0.0
    with pytest.raises(ValueError, match=r"cannot see .*block 0's null space"):
        select_penalties(lik, [data, replace(data, X=X)], blocks)


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
    with pytest.raises(ValueError, match=message + r".*cannot see .*: \[7\]"):
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
    newton_fit = reml._fit_one

    def fails_when_weak(*args, **kwargs):
        fit, problem = newton_fit(*args, **kwargs)
        # The penalty's largest eigenvalue, whatever basis it is given in.
        strength = np.linalg.eigvalsh(args[4])[-1] / blocks[0].eigenvalues[-1]
        weak = strength < np.exp(-4.5)
        return (replace(fit, status=STEP_LIMIT) if weak else fit), problem

    monkeypatch.setattr(reml, "_fit_one", fails_when_weak)
    result = select_penalties(lik, [data], blocks)
    assert sum(result.search.grid_failures.values()) == 6  # log penalties -10 to -5
    assert result.search.grid_failures == {STEP_LIMIT: 6}


def test_grid_failures_of_two_sessions_with_one_status_are_summed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two sessions that fail at different grid points with the same status:
    the count for that status is the sum over both, not the count of one."""
    lik, first, blocks = _session("poisson", 1, seed=4)
    _, second, _ = _session("poisson", 1, seed=5)
    newton_fit = reml._fit_one

    def fails_at_one_end(*args, **kwargs):
        fit, problem = newton_fit(*args, **kwargs)
        strength = np.linalg.eigvalsh(args[4])[-1] / blocks[0].eigenvalues[-1]
        if args[2] is first.y:
            fails = strength < np.exp(-4.5)  # log penalties -10 to -5
        else:
            fails = strength > np.exp(17.5)  # log penalties 18 to 20
        return (replace(fit, status=STEP_LIMIT) if fails else fit), problem

    monkeypatch.setattr(reml, "_fit_one", fails_at_one_end)
    search = select_penalties(lik, [first, second], blocks).search
    assert search.grid_failures == {STEP_LIMIT: 9}
    assert sum(search.grid_failures.values()) == 9


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
    design, penalty, _ = PSplineBasis(0.0, 4.0, 12).constrained(s)
    X = np.column_stack([design, rng.normal(size=n_rows)])
    eta = np.sin(1.5 * s) + 0.4 * X[:, -1]
    groups = np.repeat(np.arange(n_events), n_arms)
    y = draw_response(rng, "shares", eta, total=10.0, group_size=n_arms)
    data = SessionData(X, y, groups)
    blocks = [PenaltyBlock(np.arange(11), penalty)]

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
    assert sum(selected.search.grid_failures.values()) == 0
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
    seed: int, n_events: int = 140, n_arms: int = 6, by_the_clock: bool = False
) -> tuple[SessionData, list[PenaltyBlock]]:
    """A grouped-softmax session of the content model's shape.

    Each event has ``n_arms`` candidate rows and one 0/1 response; a cubic
    P-spline of a clock in [0, 5] (10 basis functions, 9 after the
    sum-to-zero constraint) with a linear truth of slope -1.2, so the fit at
    a strong penalty has null-space coefficients of several units, and two
    unpenalized 0/1 role columns (the first row of each event; one other row).
    ``by_the_clock``: each event's response is on its row of smallest clock,
    which the smooth's unpenalized line separates.
    """
    rng = np.random.default_rng(seed)
    n_rows = n_events * n_arms
    x = rng.uniform(0.0, 5.0, n_rows)
    design, penalty, _ = PSplineBasis(0.0, 5.0, 10).constrained(x)
    role = np.zeros((n_rows, 2))
    first = np.arange(n_events) * n_arms
    role[first, 0] = 1.0
    role[first + 1 + rng.integers(0, n_arms - 1, n_events), 1] = 1.0
    X = np.column_stack([design, role])
    eta = -1.2 * x + role @ np.array([1.0, -0.5])
    weight = np.exp(eta).reshape(n_events, n_arms)
    p = weight / weight.sum(axis=1, keepdims=True)
    y = np.zeros((n_events, n_arms))
    y[np.arange(n_events), [rng.choice(n_arms, p=row) for row in p]] = 1.0
    if by_the_clock:
        y = np.zeros((n_events, n_arms))
        y[np.arange(n_events), np.argmin(x.reshape(n_events, n_arms), axis=1)] = 1.0
    groups = np.repeat(np.arange(n_events), n_arms)
    blocks = [PenaltyBlock(np.arange(9), penalty)]
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
            assert np.isfinite(pooled.score), (seed, rho, pooled.failed_status)
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

    assert sum(result.search.grid_failures.values()) == 0
    assert result.rho[0] > 15.0
    for session in result.sessions:
        assert session.edf[0] == pytest.approx(1.0, abs=1e-5)
        np.testing.assert_array_equal(in_null_space(session, blocks), [True])
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
    the grid center) and is set aside, so the penalty choice is the one made
    without it. After the search it is fitted at the chosen penalties with
    its running-away columns inactive: its fit is that of the session with
    those columns inactive, flagged as not in the penalty choice."""
    sessions = []
    for seed in range(10):
        data, blocks = _content_session(seed)
        sessions.append(_with_arm_columns(data, seed, separated=seed == 4))
    arm, role = 11, 12

    pooled = pooled_reml(GroupedSoftmax(), sessions, blocks, [6.0])
    assert pooled.failed_session == 4
    assert pooled.failed_status == NO_FINITE_MAXIMUM
    failure = (
        "session 4 (s4): no finite maximum (columns running away: [11, 12]; "
        "along column 11, column 12)"
    )
    with pytest.raises(ValueError, match=re.escape(failure)):
        fit_at_penalties(GroupedSoftmax(), sessions, blocks, [6.0])

    result = select_penalties(GroupedSoftmax(), sessions, blocks)
    (left_out,) = result.search.set_aside
    assert (left_out.index, left_out.name) == (4, "s4")
    assert left_out.reference.fit.status == NO_FINITE_MAXIMUM
    assert sorted(left_out.reference.fit.running_away.tolist()) == [arm, role]
    assert [r.fit.status for r in left_out.refits] == [NO_FINITE_MAXIMUM, CONVERGED]
    assert sorted(left_out.left_out.tolist()) == [arm, role]
    assert left_out.fitted and "inactive" in left_out.reason
    np.testing.assert_array_equal(result.search.reference_rho, [6.0])
    np.testing.assert_array_equal(result.search.in_choice, [0, 1, 2, 3, 5, 6, 7, 8, 9])
    assert _share_without_fit(result.search) == 0.0

    without = select_penalties(GroupedSoftmax(), sessions[:4] + sessions[5:], blocks)
    np.testing.assert_array_equal(result.rho, without.rho)
    assert result.score == without.score
    for kept, alone in zip(
        result.sessions[:4] + result.sessions[5:], without.sessions, strict=True
    ):
        np.testing.assert_array_equal(kept.fit.beta, alone.fit.beta)

    refitted = result.sessions[4]
    active = np.ones(sessions[4].X.shape[1], dtype=bool)
    active[[arm, role]] = False
    direct = fit_at_penalties(
        GroupedSoftmax(), [replace(sessions[4], active=active)], blocks, result.rho
    ).sessions[0]
    np.testing.assert_array_equal(refitted.fit.beta, direct.fit.beta)
    np.testing.assert_array_equal(refitted.edf, direct.edf)
    assert np.all(np.isnan(refitted.fit.beta[[arm, role]]))


def test_the_search_refits_a_set_aside_session_by_refit_session() -> None:
    """The search's refit of a set-aside session is :func:`refit_session` at
    the chosen penalties: the same fits, columns left out, rounds and reason;
    the columns are left out in round 1."""
    sessions = []
    for seed in range(10):
        data, blocks = _content_session(seed)
        sessions.append(_with_arm_columns(data, seed, separated=seed == 4))
    result = select_penalties(GroupedSoftmax(), sessions, blocks)
    (set_aside,) = result.search.set_aside
    refit = refit_session(GroupedSoftmax(), sessions[4], blocks, result.rho, index=4)
    assert refit.fitted and refit.reason == set_aside.reason
    np.testing.assert_array_equal(refit.left_out, set_aside.left_out)
    np.testing.assert_array_equal(refit.rounds, [1, 1])
    assert len(refit.refits) == len(set_aside.refits)
    for one, two in zip(refit.refits, set_aside.refits, strict=True):
        np.testing.assert_array_equal(one.fit.beta, two.fit.beta)
        assert one.fit.status == two.fit.status


def test_refit_session_counts_its_rounds_and_reports_a_block() -> None:
    """Rounds: a direction spread over near-copies of a 0/1 column is left out
    over several rounds, each column with the round that left it out. A
    separation by a smooth's own line ends without a fit, with the reason, and
    is not an error."""
    rng = np.random.default_rng(0)
    basis = PSplineBasis(0.0, 4.0, 10)
    x = rng.uniform(0.0, 4.0, 600)
    design, penalty, _ = basis.constrained(x)
    indicator = (rng.uniform(size=600) < 0.2).astype(float)
    copies = [indicator * (1.0 + 0.05 * rng.normal(size=600)) for _ in range(10)]
    X, blocks = stack_smooths(np.ones(600), (design, penalty), *copies)
    y = draw_response(rng, "bernoulli", np.sin(x))
    y[indicator == 1] = 1.0
    with np.errstate(over="ignore"):
        refit = refit_session(Bernoulli(), SessionData(X, y), blocks, [6.0])
    assert refit.fitted and len(refit.refits) > 2
    assert refit.ended == reml.RefitEnd.CONVERGED
    assert refit.rounds.size == refit.left_out.size
    assert refit.rounds[0] == 1 and refit.rounds[-1] == len(refit.refits) - 1
    assert np.all(np.diff(refit.rounds) >= 0)

    data, blocks = _content_session(6, by_the_clock=True)
    line = refit_session(GroupedSoftmax(), data, blocks, [6.0])
    assert not line.fitted and line.left_out.size == 0
    assert line.ended == reml.RefitEnd.ALONG_A_PENALTY
    assert "block 0's null space" in line.reason


def _rate_session_with_levels(
    seed: int, empty_reference: bool
) -> tuple[SessionData, np.ndarray]:
    """A Poisson session of 40 visits of 30 bins: an intercept, two level
    columns (the reference level is the first 10 visits), a slope and a
    P-spline of 7 columns. ``empty_reference``: no event on the reference
    level's rows, so the fit has no finite maximum along the intercept."""
    rng = np.random.default_rng(seed)
    visit = np.repeat(np.arange(40), 30)
    level = np.where(visit < 10, 0, np.where(visit < 25, 1, 2))
    x = rng.normal(size=40)[visit]
    t = np.tile(np.arange(30) * 0.1, 40)
    count = rng.poisson(0.2, visit.size).astype(float)
    if empty_reference:
        count[level == 0] = 0.0
    smooth = PSplineBasis(0.0, 3.0, 8).constrained(t)
    X = np.column_stack([np.ones(visit.size), level == 1, level == 2, x, smooth.design])
    offset = np.full(visit.size, np.log(0.1))
    return SessionData(X, count, offset, name=f"s{seed}"), smooth.penalty


def test_the_search_never_refits_a_session_without_its_intercept() -> None:
    """Six Poisson sessions, one whose reference level holds no event: the
    search sets it aside and leaves it without a fit, with the intercept
    reason and NaN coefficients; it does not fit it with the intercept left
    out. The chosen penalty and the other sessions' fits are as before."""
    sessions, penalty = [], None
    for seed in range(6):
        data, penalty = _rate_session_with_levels(seed, empty_reference=seed == 0)
        sessions.append(data)
    blocks = [PenaltyBlock(np.arange(4, 11), penalty)]
    result = select_penalties(Poisson(), sessions, blocks)
    (left_out,) = result.search.set_aside
    assert left_out.index == 0 and not left_out.fitted
    assert left_out.left_out.size == 0
    assert left_out.reason.startswith("no finite maximum along the intercept")
    assert left_out.refits[-1].fit.status == NO_FINITE_MAXIMUM
    assert np.isnan(result.sessions[0].fit.beta).all()
    assert _share_without_fit(result.search) == pytest.approx(1 / 6)
    np.testing.assert_array_equal(result.rho, [20.0])
    assert result.sessions[1].fit.beta[3] == pytest.approx(
        -0.1197006834685946, rel=1e-12
    )


def test_refit_session_does_not_make_the_intercept_inactive() -> None:
    """A session whose fit has no finite maximum along the intercept (the
    column with one nonzero value on every row) is left without a fit, with
    that reason, and no column is made inactive."""
    data, penalty = _rate_session_with_levels(0, empty_reference=True)
    blocks = [PenaltyBlock(np.arange(4, 11), penalty)]
    refit = refit_session(Poisson(), data, blocks, [6.0])
    assert not refit.fitted and refit.left_out.size == 0
    assert len(refit.refits) == 1
    assert 0 in refit.refits[0].fit.running_away
    assert refit.ended == reml.RefitEnd.ALONG_THE_INTERCEPT
    assert refit.reason == reml.ALONG_THE_INTERCEPT
    # Doubling the intercept column is still an intercept.
    doubled = replace(data, X=np.column_stack([2.0 * data.X[:, :1], data.X[:, 1:]]))
    assert refit_session(Poisson(), doubled, blocks, [6.0]).ended == (
        reml.RefitEnd.ALONG_THE_INTERCEPT
    )


def test_the_intercept_is_named_before_the_rounds_run_out(monkeypatch) -> None:
    """Scripted fits: each round runs away along one new column, and the fit
    of the last round allowed along the intercept. The refit ends along the
    intercept, which no further round could leave out, not after too many
    rounds."""

    def running_away(columns: list[int]) -> SimpleNamespace:
        fit = SimpleNamespace(
            converged=False, status=NO_FINITE_MAXIMUM, running_away=np.array(columns)
        )
        directions = tuple(reml.FreeDirection(None, c, True) for c in columns)
        return SimpleNamespace(fit=fit, running_away_directions=directions)

    fits = iter(
        [running_away([k]) for k in range(1, REFIT_MAX_ROUNDS + 1)]
        + [running_away([0])]
    )
    monkeypatch.setattr(reml, "_labeled_score", lambda *args: next(fits))
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(40), rng.normal(size=(40, REFIT_MAX_ROUNDS))])
    refit = refit_session(Poisson(), SessionData(X, np.ones(40), np.zeros(40)), [], [])
    assert len(refit.refits) == REFIT_MAX_ROUNDS + 1
    assert refit.ended == reml.RefitEnd.ALONG_THE_INTERCEPT
    assert refit.left_out.tolist() == list(range(1, REFIT_MAX_ROUNDS + 1))


def test_a_separation_by_a_smooths_line_stays_without_a_fit() -> None:
    """A content session whose responses are each event's row of smallest
    clock is separated by the smooth's own line, which cannot be made
    inactive: it is set aside, refitted at the chosen penalties without
    success, and left without a fit, named, with the reason in words. Three
    of ten such sessions (30 %, above ``MAX_FAILING_SHARE``) is an error
    naming each."""
    sessions = []
    for seed in range(10):
        data, blocks = _content_session(seed, by_the_clock=seed == 6)
        sessions.append(replace(data, name=f"s{seed}"))
    result = select_penalties(GroupedSoftmax(), sessions, blocks)
    (left_out,) = result.search.set_aside
    assert left_out.index == 6 and not left_out.fitted
    assert left_out.refits[-1].fit.status == NO_FINITE_MAXIMUM
    assert left_out.left_out.size == 0
    assert "block 0's null space" in left_out.reason
    assert _share_without_fit(result.search) == pytest.approx(0.1)
    assert np.all(np.isnan(result.sessions[6].fit.beta))
    assert 6 not in result.search.in_choice

    many = []
    for seed in range(10):
        data, _ = _content_session(seed, by_the_clock=seed in (1, 4, 7))
        many.append(replace(data, name=f"s{seed}"))
    message = (
        r"3 of 10 sessions \(30%, more than MAX_FAILING_SHARE = 20%\) have no fit"
        r".*session 1 \(s1\): no finite maximum.*block 0's null space.*"
        r"session 4 \(s4\).*session 7 \(s7\)"
    )
    with pytest.raises(ValueError, match=message):
        select_penalties(GroupedSoftmax(), many, blocks)


def test_a_direction_spread_over_near_copies_is_left_out_in_rounds() -> None:
    """Ten near-copies of a 0/1 column (each the indicator times a little
    noise), all of whose rows have y = 1 in one session: the first fit names
    only part of the separating direction, and the refit leaves out the
    columns named, round after round, until the session has a fit."""
    rng = np.random.default_rng(0)
    basis = PSplineBasis(0.0, 4.0, 10)
    sessions = []
    for seed in range(6):
        x = rng.uniform(0.0, 4.0, 600)
        design, penalty, _ = basis.constrained(x)
        indicator = (rng.uniform(size=600) < 0.2).astype(float)
        copies = [indicator * (1.0 + 0.05 * rng.normal(size=600)) for _ in range(10)]
        X, blocks = stack_smooths(np.ones(600), (design, penalty), *copies)
        y = draw_response(rng, "bernoulli", np.sin(x))
        if seed == 2:
            y[indicator == 1] = 1.0
        sessions.append(SessionData(X, y, name=f"s{seed}"))
    with np.errstate(over="ignore"):
        result = select_penalties(Bernoulli(), sessions, blocks)
    (left_out,) = result.search.set_aside
    assert left_out.index == 2 and left_out.fitted
    assert len(left_out.refits) > 2  # more than one round
    assert set(left_out.left_out.tolist()) <= set(range(10, 20))
    assert len(set(left_out.left_out.tolist())) == left_out.left_out.size
    assert _share_without_fit(result.search) == 0.0


def _two_smooth_session(seed: int) -> tuple[SessionData, list[PenaltyBlock]]:
    """:func:`_content_session` with a second smooth: a P-spline of a share in
    [0, 1] (8 basis functions, 7 after the constraint) with no effect."""
    data, blocks = _content_session(seed)
    rng = np.random.default_rng(1000 + seed)
    share = rng.uniform(0.0, 1.0, data.X.shape[0])
    design, penalty = _smooth(share, n_basis=8)
    X = np.column_stack([data.X[:, :9], design, data.X[:, 9:]])
    second = PenaltyBlock(np.arange(9, 16), penalty)
    return SessionData(X, data.y, data.aux), [*blocks, second]


def test_two_smooths_in_their_null_spaces_return() -> None:
    """Twenty content sessions with two smooths whose truths are a line and
    zero: the score is flat toward the bound in both log penalties. The search
    returns, every grid point finite, each term's edf that of a line in every
    session and both flagged ``in_null_space``."""
    sessions = []
    for seed in range(20):
        data, blocks = _two_smooth_session(seed)
        sessions.append(data)
    result = select_penalties(GroupedSoftmax(), sessions, blocks)

    assert sum(result.search.grid_failures.values()) == 0
    assert result.search.set_aside == ()
    assert np.all(result.rho > 15.0)
    for session in result.sessions:
        np.testing.assert_allclose(session.edf, [1.0, 1.0], rtol=0, atol=1e-5)
        np.testing.assert_array_equal(in_null_space(session, blocks), [True, True])


def _stack(parts: list[SessionData], n_shared: int = 9) -> SessionData:
    """Sessions stacked into one: their first ``n_shared`` columns shared (the
    smooth), the rest one set per session (zero on the other sessions' rows),
    the group ids made consecutive."""
    n_own = parts[0].X.shape[1] - n_shared
    n_rows = sum(part.X.shape[0] for part in parts)
    X = np.zeros((n_rows, n_shared + n_own * len(parts)))
    groups, row, offset = [], 0, 0
    for k, part in enumerate(parts):
        rows = slice(row, row + part.X.shape[0])
        X[rows, :n_shared] = part.X[:, :n_shared]
        X[rows, n_shared + k * n_own : n_shared + (k + 1) * n_own] = part.X[
            :, n_shared:
        ]
        groups.append(part.aux + offset)
        row += part.X.shape[0]
        offset = groups[-1][-1] + 1
    y = np.concatenate([part.y for part in parts])
    return SessionData(X, y, np.concatenate(groups))


def test_a_stack_with_inactive_columns_is_the_stack_without_them() -> None:
    """Five animals, each a stack of three content sessions with four columns
    of their own (two roles, an arm and a third role). One session of the
    third animal is separated: that animal's fit has no finite maximum and
    names that session's arm and role columns, and the animal is set aside.
    Marking those columns inactive gives the stack built without them (score,
    derivatives, coefficients, covariance and edf), and the search keeps the
    animal."""
    _, blocks = _content_session(0)
    animals = []
    for animal in range(5):
        parts = []
        for k in range(3):
            seed = 3 * animal + k
            data, _ = _content_session(seed)
            separated = (animal, k) == (2, 1)
            parts.append(_with_arm_columns(data, seed, separated=separated))
        animals.append(_stack(parts))
    own_columns = 9 + 4 * 1 + np.arange(4)  # the separated session's columns
    arm, role = own_columns[2], own_columns[3]

    separated = pooled_reml(GroupedSoftmax(), [animals[2]], blocks, [6.0])
    assert separated.failed_status == NO_FINITE_MAXIMUM
    reference = select_penalties(GroupedSoftmax(), animals, blocks)
    (left_out,) = reference.search.set_aside
    assert left_out.index == 2
    assert sorted(left_out.reference.fit.running_away.tolist()) == [arm, role]
    assert left_out.fitted
    assert sorted(left_out.left_out.tolist()) == [arm, role]

    active = np.ones(animals[2].X.shape[1], dtype=bool)
    active[[arm, role]] = False
    masked = replace(animals[2], active=active)
    built = SessionData(animals[2].X[:, active], animals[2].y, animals[2].aux)
    for rho in (0.0, 6.0, 18.0):
        one = pooled_reml(GroupedSoftmax(), [masked], blocks, [rho])
        two = pooled_reml(GroupedSoftmax(), [built], blocks, [rho])
        assert one.score == pytest.approx(two.score, rel=1e-12)
        np.testing.assert_allclose(one.gradient, two.gradient, rtol=1e-9)
        np.testing.assert_allclose(one.hessian, two.hessian, rtol=1e-9)
        fit, alone = one.sessions[0], two.sessions[0]
        assert np.all(np.isnan(fit.fit.beta[~active]))
        np.testing.assert_allclose(fit.fit.beta[active], alone.fit.beta, atol=1e-10)
        np.testing.assert_allclose(
            fit.covariance[np.ix_(active, active)], alone.covariance, rtol=1e-9
        )
        np.testing.assert_allclose(fit.edf, alone.edf, rtol=1e-10)

    repaired = select_penalties(
        GroupedSoftmax(), [*animals[:2], masked, *animals[3:]], blocks
    )
    assert repaired.search.set_aside == ()
    assert np.all(np.isfinite(repaired.sessions[2].fit.beta[active]))


@pytest.mark.parametrize("rho", [0.0, 6.0, 20.0])
def test_a_separation_by_a_smooths_line_is_named_as_its_null_space(rho: float) -> None:
    """A 0/1 outcome that is a step in ``x`` beside a P-spline of ``x``: the
    smooth's unpenalized straight line separates it at every penalty. Each
    step is cut to the step cap, so the fit runs out of steps; it is reported
    as no finite maximum, running away along the block's null space and the
    intercept, which says that the smooth's own line is in the direction."""
    rng = np.random.default_rng(0)
    x = rng.uniform(0.0, 4.0, 300)
    design, penalty, _ = PSplineBasis(0.0, 4.0, 10).constrained(x)
    X, blocks = stack_smooths(np.ones(300), (design, penalty))
    y = (x > 2.0).astype(float)
    with np.errstate(over="ignore"):
        score = reml_and_derivatives(Bernoulli(), X, y, None, blocks, [rho])
    assert score.fit.status == NO_FINITE_MAXIMUM
    assert set(score.running_away_directions) == {
        reml.FreeDirection(0, 0, True),
        reml.FreeDirection(None, 0, True),
    }


class _Quadratic:
    """A pooled score given by a formula, for the search alone."""

    def __init__(self, function) -> None:
        self.function = function

    def __call__(self, rho: np.ndarray, derivatives: bool) -> reml.PooledScore:
        score, gradient, hessian = self.function(np.asarray(rho, dtype=float))
        return reml.PooledScore(score, gradient, hessian, ())


@pytest.mark.parametrize("weak", [1e-5, 1e-6, 1e-7])
def test_a_stiff_term_does_not_set_the_floor_of_a_weak_one(weak: float) -> None:
    """``V = 500 (r1 - 3)^2 + a (r2 - 12)^2`` from (3, 4): the weak term moves
    as it would alone. With the eigenvalue floor taken over the whole Hessian
    (1e-3 here), ``a`` of 1e-5 and 1e-6 crawled for 50 steps and raised, and
    1e-7 stopped at once, eight units short. Curvature 2e-7 is below the
    floor of 1e-6 that a term alone has too: that term stops where the gain
    left is within the flat-score rule's reach."""

    def score(r: np.ndarray):
        value = 5e4 + 500 * (r[0] - 3) ** 2 + weak * (r[1] - 12) ** 2
        gradient = np.array([1000 * (r[0] - 3), 2 * weak * (r[1] - 12)])
        return value, gradient, np.diag([1000.0, 2 * weak])

    def alone(r: np.ndarray):
        value = 5e4 + weak * (r[0] - 12) ** 2
        return value, np.array([2 * weak * (r[0] - 12)]), np.array([[2 * weak]])

    rho, _, _, _ = reml._search(_Quadratic(score), np.array([3.0, 4.0]))
    single, _, _, _ = reml._search(_Quadratic(alone), np.array([4.0]))
    assert rho[0] == pytest.approx(3.0, abs=1e-9)
    assert rho[1] == pytest.approx(single[0], abs=1e-6)
    assert weak * (rho[1] - 12) ** 2 < 1e-7


def test_the_search_does_not_stop_where_the_curvature_is_negative() -> None:
    """``V = 100 - (r - 5)^2 + 0.01 (r - 5)^4`` has a maximum at 5 and minima at
    ``5 +- sqrt(50)``. Started 1e-5 from the maximum, the decrement (with the
    curvature made positive) is below the tolerance; the search steps along
    the negative curvature instead and ends at a minimum."""

    def score(r: np.ndarray):
        d = r[0] - 5
        value = 100 - d**2 + 0.01 * d**4
        return value, np.array([-2 * d + 0.04 * d**3]), np.array([[-2 + 0.12 * d**2]])

    rho, current, _, _ = reml._search(_Quadratic(score), np.array([5.00001]))
    assert abs(rho[0] - 5) == pytest.approx(np.sqrt(50), abs=1e-6)
    assert current.hessian[0, 0] > 0


def test_the_last_allowed_step_does_not_end_where_the_curvature_is_negative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The double well of the test above, started 1e-5 from its maximum with no
    step left (``REML_MAX_STEPS`` steps taken): the decrement is small but the
    curvature negative, so the point is not a stationary point the search may
    stop at, and the search raises instead of returning the maximum."""

    def score(r: np.ndarray):
        d = r[0] - 5
        value = 100 - d**2 + 0.01 * d**4
        return value, np.array([-2 * d + 0.04 * d**3]), np.array([[-2 + 0.12 * d**2]])

    monkeypatch.setattr(reml, "REML_MAX_STEPS", 0)
    with pytest.raises(ValueError, match="no stationary point in 0 steps"):
        reml._search(_Quadratic(score), np.array([5.00001]))


def test_a_constant_gaussian_outcome_is_named() -> None:
    """A Gaussian outcome with no variance has no scale to profile: the error
    says the outcome is constant."""
    _, data, blocks = _session("gaussian", 1, seed=9)
    constant = replace(data, y=np.full(data.y.shape, 2.5), name="flat")
    with pytest.raises(ValueError, match=r"session 0 \(flat\).*outcome is constant"):
        pooled_reml(Gaussian(), [constant], blocks, [0.0])


@pytest.mark.parametrize("rho", [15.0, 17.5, 20.0])
def test_gradient_and_hessian_at_strong_penalties(rho: float) -> None:
    """At log penalties 15 to 20 the analytic gradient and Hessian of the
    two-smooth content score equal Richardson-extrapolated central
    differences of the score (steps 0.05 and 0.025, warm-started at the
    fit): the derivatives hold in the eigenbasis where the score is a
    shallow ``c exp(-rho)``."""
    data, blocks = _two_smooth_session(3)
    point = np.array([rho, rho - 2.0])
    at = _score(GroupedSoftmax(), data, blocks, point)
    warm = at.fit.beta

    def value(r: np.ndarray) -> float:
        return reml_and_derivatives(
            GroupedSoftmax(),
            data.X,
            data.y,
            data.aux,
            blocks,
            r,
            warm,
            derivatives=False,
        ).score

    def gradient(h: float) -> np.ndarray:
        return numeric_gradient(value, point, h)

    def hessian(h: float) -> np.ndarray:
        return numeric_hessian(value, point, h)

    extrapolated_gradient = (4 * gradient(0.025) - gradient(0.05)) / 3
    extrapolated_hessian = (4 * hessian(0.025) - hessian(0.05)) / 3
    np.testing.assert_allclose(
        at.gradient, extrapolated_gradient, rtol=1e-4, atol=1e-11
    )
    np.testing.assert_allclose(at.hessian, extrapolated_hessian, rtol=1e-3, atol=1e-10)


def test_a_softmax_design_collinear_within_events_is_named() -> None:
    """A content session with a column that equals its first role column plus
    a constant per event: the softmax cannot tell the two apart. The search
    raises as "design not identified", naming the two columns."""
    sessions = []
    for seed in range(5):
        data, blocks = _content_session(seed)
        rng = np.random.default_rng(seed)
        if seed == 2:
            level = np.repeat(rng.normal(size=data.X.shape[0] // 6), 6)
            extra = data.X[:, 9] + level
        else:
            extra = rng.normal(size=data.X.shape[0])
        X = np.column_stack([data.X, extra])
        sessions.append(SessionData(X, data.y, data.aux, name=f"s{seed}"))
    message = (
        r"session 2 \(s2\): design not identified .*collinear to the likelihood: "
        r"column 9, column 11\)"
    )
    with pytest.raises(ValueError, match=message):
        select_penalties(GroupedSoftmax(), sessions, blocks)


def test_a_search_clears_the_caches_at_most_once_at_its_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With the program limit patched to 0, every outermost call of the fitter
    clears JAX's caches at its entry: a penalty search, an outer entry, clears
    once, never between two of its evaluations."""
    lik, data, blocks = _session("poisson", 2, 3)
    clears = []
    monkeypatch.setattr(runtime, "LIVE_PROGRAM_LIMIT", 0)
    monkeypatch.setattr(jax, "clear_caches", lambda: clears.append(True))
    select_penalties(lik, [data], blocks)
    assert len(clears) == 1
