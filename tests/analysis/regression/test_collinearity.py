"""Collinearity of a fit's terms: VIF, mgcv's concurvity, the weighting."""

from __future__ import annotations

import json

import numpy as np
import pytest

from continuum_swr_replay.analysis.regression.collinearity import (
    MAX_EXPLAINED,
    concurvity,
    separable,
    term_collinearity,
    weighted_design,
)
from continuum_swr_replay.analysis.regression.likelihoods import (
    Bernoulli,
    Gaussian,
    GroupedSoftmax,
    Poisson,
)
from continuum_swr_replay.analysis.regression.pspline import PSplineBasis
from continuum_swr_replay.analysis.regression.reml import (
    PenaltyBlock,
    SessionData,
    select_penalties,
)
from scripts.checks.compare_regression_to_mgcv import (
    DATA_SETS,
    FIXTURE_DIR,
    HI,
    LO,
    N_BASIS,
    SMOOTHS,
    make_data,
)


def _poisson_design(seed: int = 0, n: int = 800):
    """An intercept, a smooth of 7 columns and a column ``z`` correlated with
    the smooth's variable, with Poisson counts and an offset."""
    rng = np.random.default_rng(seed)
    t = rng.uniform(0.0, 4.0, n)
    z = np.sin(t) + 0.5 * rng.normal(size=n) + 2.0
    smooth = PSplineBasis(0.0, 4.0, 8).constrained(t)
    X = np.column_stack([np.ones(n), smooth.design, z])
    offset = np.log(rng.uniform(0.5, 2.0, n))
    beta = np.r_[-1.0, 0.3 * rng.normal(size=7), 0.4]
    y = rng.poisson(np.exp(X @ beta + offset)).astype(float)
    return X, y, offset, beta, {"s(t)": np.arange(1, 8), "z": np.array([8])}


def test_single_column_equals_r2() -> None:
    """For a term of one column, observed, worst and estimate all equal the R^2
    of an OLS of the weighted column on the other weighted columns, measured
    beyond the intercept; VIF = 1 / (1 - R^2)."""
    X, y, offset, beta, terms = _poisson_design()
    weighted = weighted_design(Poisson(), X, y, offset, X @ beta)
    measures = concurvity(weighted, terms, beta, intercept=0)["z"]

    column, rest = weighted[:, 8], weighted[:, :8]
    residual = column - rest @ np.linalg.lstsq(rest, column, rcond=None)[0]
    one = weighted[:, :1]
    around = column - one @ np.linalg.lstsq(one, column, rcond=None)[0]
    r2 = 1.0 - residual @ residual / (around @ around)
    assert 0.2 < r2 < 0.9
    for key in ("observed", "worst", "estimate"):
        assert measures[key] == pytest.approx(r2, rel=1e-10)

    record = term_collinearity(Poisson(), X, y, offset, beta, terms, intercept=0)
    assert record["z"]["r2"] == pytest.approx(r2, rel=1e-10)
    assert record["z"]["vif"] == pytest.approx(1.0 / (1.0 - r2), rel=1e-10)
    assert set(record["s(t)"]) == {"observed", "worst", "estimate"}


def test_the_intercept_does_not_explain_a_column() -> None:
    """A 0/1 column independent of the rest, with mean 0.85, has an R^2 near
    zero, not near 0.85: the intercept is projected out first."""
    rng = np.random.default_rng(1)
    n = 2000
    X = np.column_stack(
        [np.ones(n), rng.normal(size=n), (rng.uniform(size=n) < 0.85).astype(float)]
    )
    terms = {"x": np.array([1]), "r": np.array([2])}
    measures = concurvity(X, terms, np.array([0.0, 1.0, 1.0]), intercept=0)
    assert measures["r"]["estimate"] < 0.01


@pytest.mark.parametrize("family", ["poisson", "bernoulli", "gaussian", "softmax"])
def test_softmax_weighting(family: str) -> None:
    """The weighted cross-product equals the likelihood's information at the
    fit, for the grouped softmax (each group centered at its probabilities and
    weighted by its total) and the diagonal families."""
    rng = np.random.default_rng(2)
    n = 240
    X = rng.normal(size=(n, 4))
    eta = X @ np.array([0.5, -0.3, 0.2, 0.1])
    if family == "softmax":
        lik, aux = GroupedSoftmax(), np.repeat(np.arange(n // 4), 4)
        y = rng.uniform(0.0, 3.0, n)
        information = lik.information(X, eta, y, aux)
    elif family == "gaussian":
        lik, aux, y = Gaussian(), None, rng.normal(size=n)
        information = lik.information(X, eta, y, 1.0)
    else:
        aux = np.log(rng.uniform(0.5, 2.0, n)) if family == "poisson" else None
        lik = Poisson() if family == "poisson" else Bernoulli()
        y = (
            rng.poisson(1.0, n).astype(float)
            if family == "poisson"
            else (eta > 0) * 1.0
        )
        information = lik.information(X, eta, y, aux)
    weighted = weighted_design(lik, X, y, aux, eta)
    np.testing.assert_allclose(
        weighted.T @ weighted, information, rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("name", list(DATA_SETS))
def test_matches_mgcv_concurvity(name: str) -> None:
    """On the mgcv fixtures (unit weights, as mgcv computes it), each smooth's
    observed, worst and estimate equal mgcv's ``concurvity(full = TRUE)`` (to
    3.5e-12 when written; the bound is 1e-8)."""
    mgcv = json.loads((FIXTURE_DIR / f"{name}.json").read_text())["concurvity"]
    frame = make_data(name)
    basis = PSplineBasis(LO, HI, N_BASIS)
    poisson = name.startswith("poisson")
    parts = [np.ones((len(frame), 1))]
    if poisson:
        parts.append(frame[["rew"]].to_numpy(dtype=float))
    terms = {"para_rest": np.arange(1, len(parts))} if poisson else {}
    blocks = []
    for variable in SMOOTHS[name]:
        design, penalty, _ = basis.constrained(frame[variable].to_numpy(dtype=float))
        start = sum(p.shape[1] for p in parts)
        terms[f"s({variable})"] = np.arange(start, start + design.shape[1])
        blocks.append(PenaltyBlock(terms[f"s({variable})"], penalty))
        parts.append(design)
    X = np.hstack(parts)
    y = frame["y"].to_numpy(dtype=float)
    if poisson:
        lik, data = Poisson(), SessionData(X, y, np.log(frame["expo"].to_numpy()))
    elif name == "bernoulli_one_smooth":
        lik, data = Bernoulli(), SessionData(X, y)
    else:
        lik, data = Gaussian(), SessionData(X, y)
    beta = select_penalties(lik, [data], blocks).sessions[0].fit.beta
    ours = concurvity(X, terms, beta, intercept=0)
    for variable in SMOOTHS[name]:
        k = mgcv["terms"].index(f"s({variable})")
        for key in ("observed", "worst", "estimate"):
            assert ours[f"s({variable})"][key] == pytest.approx(
                mgcv[key][k], abs=1e-8
            ), (variable, key)


def test_an_inactive_column_is_left_out() -> None:
    """A column whose coefficient is NaN is not in the design measured, and a
    term with no column left is not measured."""
    X, y, offset, beta, terms = _poisson_design()
    extra = np.column_stack([X, np.random.default_rng(3).normal(size=len(y))])
    with_nan = np.r_[beta, np.nan]
    terms = {**terms, "w": np.array([9])}
    record = term_collinearity(
        Poisson(), extra, y, offset, with_nan, terms, intercept=0
    )
    alone = term_collinearity(
        Poisson(), X, y, offset, beta, {**terms, "w": np.array([], int)}, intercept=0
    )
    assert "w" not in record
    assert record == alone


def test_separable() -> None:
    """A term is separable below ``MAX_EXPLAINED``: ``r2`` for one column,
    ``estimate`` for several."""
    record = {
        "a": {"r2": MAX_EXPLAINED - 0.01, "vif": 4.8},
        "b": {"r2": MAX_EXPLAINED + 0.01, "vif": 5.3},
        "c": {"observed": 0.99, "worst": 0.99, "estimate": 0.5},
        "d": {"observed": 0.1, "worst": 0.9, "estimate": MAX_EXPLAINED},
    }
    assert separable(record) == {"a": True, "b": False, "c": True, "d": False}


def test_a_column_zero_in_the_session_does_not_change_the_measures() -> None:
    """A smooth's column that is zero on every row of a session (a basis
    function beyond the session's values, identified by the penalty) adds a
    direction the data do not see: the measures equal those without it."""
    X, y, offset, beta, terms = _poisson_design()
    padded = np.column_stack([X[:, :8], np.zeros(len(y)), X[:, 8:]])
    padded_beta = np.r_[beta[:8], 0.0, beta[8:]]
    padded_terms = {"s(t)": np.arange(1, 9), "z": np.array([9])}
    weighted = weighted_design(Poisson(), X, y, offset, X @ beta)
    padded_weighted = weighted_design(
        Poisson(), padded, y, offset, padded @ padded_beta
    )
    one = concurvity(weighted, terms, beta, intercept=0)
    two = concurvity(padded_weighted, padded_terms, padded_beta, intercept=0)
    for name in terms:
        for key in ("observed", "worst", "estimate"):
            assert two[name][key] == pytest.approx(one[name][key], rel=1e-9)
