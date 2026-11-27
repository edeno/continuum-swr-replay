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

from .conftest import draw_response, family_kernel


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
    y = draw_response(rng, "poisson", X @ beta + offset)
    return X, y, offset, beta, {"s(t)": np.arange(1, 8), "z": np.array([8])}


def test_single_column_equals_r2() -> None:
    """For a term of one column, observed, worst and estimate all equal the R^2
    of an OLS of the weighted column on the other weighted columns, measured
    beyond the intercept; VIF = 1 / (1 - R^2)."""
    X, y, offset, beta, terms = _poisson_design()
    weighted = weighted_design(Poisson(), X, y, offset, X @ beta)
    measures = concurvity(weighted, terms, beta, intercept=[0])["z"]

    column, rest = weighted[:, 8], weighted[:, :8]
    residual = column - rest @ np.linalg.lstsq(rest, column, rcond=None)[0]
    one = weighted[:, :1]
    around = column - one @ np.linalg.lstsq(one, column, rcond=None)[0]
    r2 = 1.0 - residual @ residual / (around @ around)
    assert 0.2 < r2 < 0.9
    for key in ("observed", "worst", "estimate"):
        assert measures[key] == pytest.approx(r2, rel=1e-10)

    record = term_collinearity(Poisson(), X, y, offset, beta, terms, intercept=[0])
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
    measures = concurvity(X, terms, np.array([0.0, 1.0, 1.0]), intercept=[0])
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
        information = family_kernel(lik, "information", X, eta, y, aux)
    elif family == "gaussian":
        lik, aux, y = Gaussian(), None, rng.normal(size=n)
        information = family_kernel(lik, "information", X, eta, y, 1.0)
    else:
        aux = np.log(rng.uniform(0.5, 2.0, n)) if family == "poisson" else None
        lik = Poisson() if family == "poisson" else Bernoulli()
        y = (
            rng.poisson(1.0, n).astype(float)
            if family == "poisson"
            else (eta > 0) * 1.0
        )
        information = family_kernel(lik, "information", X, eta, y, aux)
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
    ours = concurvity(X, terms, beta, intercept=[0])
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
        Poisson(), extra, y, offset, with_nan, terms, intercept=[0]
    )
    alone = term_collinearity(
        Poisson(), X, y, offset, beta, {**terms, "w": np.array([], int)}, intercept=[0]
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
    one = concurvity(weighted, terms, beta, intercept=[0])
    two = concurvity(padded_weighted, padded_terms, padded_beta, intercept=[0])
    for name in terms:
        for key in ("observed", "worst", "estimate"):
            assert two[name][key] == pytest.approx(one[name][key], rel=1e-9)


def test_a_column_that_does_not_vary_is_not_measured() -> None:
    """A column that is twice the intercept, or zero, has nothing left once the
    intercept is projected out: it is not measured (NaN, with the reason), not
    read from rounding. An exact copy of other columns has r2 1 and an
    infinite VIF."""
    X, y, offset, beta, terms = _poisson_design()
    n = len(y)
    flat = np.column_stack([X, np.full(n, 2.0), np.zeros(n), X[:, 8] + X[:, 1]])
    flat_beta = np.r_[beta, 0.0, 0.0, 0.1]
    flat_terms = {
        **terms,
        "c": np.array([9]),
        "zero": np.array([10]),
        "copy": np.array([11]),
    }
    record = term_collinearity(
        Poisson(), flat, y, offset, flat_beta, flat_terms, intercept=[0]
    )
    for name in ("c", "zero"):
        assert np.isnan(record[name]["r2"]) and np.isnan(record[name]["vif"])
        assert "does not vary" in record[name]["not_measured"]
    assert record["copy"]["r2"] == pytest.approx(1.0)
    assert record["copy"]["vif"] == np.inf
    assert separable(record)["c"] is False


# --- the baseline copies, projected out one session at a time ------------------------


def _unit(kind: str, n_sessions: int, seed: int = 0, shared_intercept: bool = False):
    """A fit of several sessions with baseline copies: each session's rows in
    turn, its copies zero on the other sessions' rows (Poisson: an intercept
    copy per session, or a shared intercept and copies of a two-level
    category; softmax: copies of three coded arms of four), then four shared
    columns (one of them partly explained by the copies, two by each other).

    Returns
    -------
    lik, X, y, aux, beta, terms, copies, sessions
        ``copies`` the columns projected out (the shared intercept first),
        ``sessions`` each session's rows and copy columns.
    """
    rng = np.random.default_rng(seed)
    n_arms, n_events = 4, 40
    per = n_events * n_arms if kind == "softmax" else 300
    session = np.repeat(np.arange(n_sessions), per)
    n = session.size
    if kind == "softmax":
        arm = np.tile(np.arange(n_arms), n_sessions * n_events)
        coded = [
            (arm == a) & (session == s) for a in (1, 2, 3) for s in range(n_sessions)
        ]
        intercept = []
    else:
        level = rng.uniform(size=n) < 0.4
        coded = (
            [level & (session == s) for s in range(n_sessions)]
            if shared_intercept
            else [session == s for s in range(n_sessions)]
        )
        intercept = [np.ones(n)] if shared_intercept else []
    copies_X = np.column_stack(coded).astype(float)
    shared = rng.normal(size=(n, 4))
    shared[:, 0] += 0.5 * copies_X[:, 0]
    shared[:, 2] = 0.6 * shared[:, 1] + 0.4 * rng.normal(size=n)
    X = np.column_stack([*intercept, copies_X, shared])
    k0 = len(intercept)
    k = copies_X.shape[1]
    beta = rng.normal(scale=0.2, size=X.shape[1])
    if kind == "softmax":
        lik, aux = GroupedSoftmax(), np.repeat(np.arange(n_sessions * n_events), n_arms)
        eta = (X @ beta).reshape(-1, n_arms)
        p = np.exp(eta) / np.exp(eta).sum(axis=1, keepdims=True)
        y = np.concatenate([rng.dirichlet(5 * r) for r in p])
    else:
        lik, aux = Poisson(), np.zeros(n)
        beta[: k0 + k] -= 1.0
        y = rng.poisson(np.exp(X @ beta)).astype(float)
    start = k0 + k
    terms = {
        "a": np.array([start]),
        "b": np.array([start + 1]),
        "cd": np.arange(start + 2, start + 4),
    }
    columns = (
        np.arange(k0, k0 + k).reshape(-1, n_sessions) if kind == "softmax" else None
    )
    sessions = []
    for s in range(n_sessions):
        own = columns[:, s].tolist() if kind == "softmax" else [k0 + s]
        sessions.append((slice(s * per, (s + 1) * per), own))
    return lik, X, y, aux, beta, terms, [*range(k0), *range(k0, k0 + k)], sessions


@pytest.mark.parametrize(
    ("kind", "n_sessions", "shared_intercept"),
    [
        ("poisson", 3, False),
        ("poisson", 12, False),
        ("poisson", 4, True),
        ("softmax", 3, False),
        ("softmax", 10, False),
    ],
)
def test_copies_by_session_equal_the_sequential_projection(
    kind, n_sessions, shared_intercept
) -> None:
    """Projecting each session's copies out of its own rows (the range of
    its copies taken out, a QR of what is left, the blocks stacked and
    reduced) gives the measures of the
    sequential projection of every copy out of the whole design, to 1e-10, on
    a day-like and an animal-like unit, under the Poisson (intercept copies;
    a shared intercept with category copies) and the grouped softmax."""
    lik, X, y, aux, beta, terms, copies, sessions = _unit(
        kind, n_sessions, shared_intercept=shared_intercept
    )
    oracle = term_collinearity(lik, X, y, aux, beta, terms, intercept=copies)
    by_session = term_collinearity(
        lik, X, y, aux, beta, terms, intercept=copies, session_copies=sessions
    )
    assert set(by_session) == set(oracle) == {"a", "b", "cd"}
    for name, measures in oracle.items():
        for measure, value in measures.items():
            assert by_session[name][measure] == pytest.approx(
                value, rel=1e-10, abs=1e-10
            )
    assert oracle["cd"]["estimate"] > 0.1  # the measures are not all trivial


def test_a_copy_with_nothing_left_is_skipped_by_session() -> None:
    """A copy that repeats another copy of its session has nothing left once
    that one is projected out. The projection by session gives the measures of
    the design without the repeated copy. The sequential projection divides
    the rounding left of the copy by its norm and projects out that arbitrary
    direction too (NaN when the norm is exactly zero), so its measures are
    off."""
    lik, X, y, aux, beta, terms, copies, sessions = _unit("poisson", 3)
    twice = np.column_stack([X, X[:, 1]])
    beta_twice = np.r_[beta, 0.0]
    extra = twice.shape[1] - 1
    with_repeat = [
        (rows, own + [extra] if own == [1] else own) for rows, own in sessions
    ]
    got = term_collinearity(
        lik, twice, y, aux, beta_twice, terms, intercept=[*copies, extra],
        session_copies=with_repeat,
    )  # fmt: skip
    want = term_collinearity(
        lik, X, y, aux, beta, terms, intercept=copies, session_copies=sessions
    )
    for name in want:
        for measure, value in want[name].items():
            assert got[name][measure] == pytest.approx(value, rel=1e-10, abs=1e-12)
    sequential = term_collinearity(
        lik, twice, y, aux, beta_twice, terms, intercept=[*copies, extra]
    )
    off = sequential["a"]["r2"]
    assert np.isnan(off) or abs(off - want["a"]["r2"]) > 1e-6 * want["a"]["r2"]
