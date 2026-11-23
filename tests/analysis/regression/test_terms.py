"""Model terms, the design they build, standardization and the design check."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.regression.likelihoods import (
    Bernoulli,
    GroupedSoftmax,
    Poisson,
)
from continuum_swr_replay.analysis.regression.pspline import PSplineBasis
from continuum_swr_replay.analysis.regression.reml import (
    PenaltyBlock,
    SessionData,
    fit_at_penalties,
)
from continuum_swr_replay.analysis.regression.terms import (
    SMALLEST_COLUMN_SIZE,
    Category,
    DesignTerm,
    LinearFunctional,
    Offset,
    Slope,
    Smooth,
    _refuse_small,
    _small_terms,
    build_design,
    log_penalty_from_prior_sd,
    prior_sd_from_log_penalty,
    standardization,
)

from .conftest import N_LAGS, ROLE, ROLES, draw_response

LAGS = tuple(range(1, N_LAGS + 1))
LAG_COLUMNS = tuple(f"lag_{lag}" for lag in LAGS)


def _rate_terms(penalty: float | str = 2.0) -> list:
    return [
        Smooth("time_since_s", 0.0, 20.0, 12, penalty),
        Slope("last_rewarded", "none"),
        Slope("x", "none"),
        LinearFunctional(LAG_COLUMNS, LAGS, 0.0, float(np.log(10)), 8, penalty),
        Offset("exposure"),
    ]


def _content_terms() -> list:
    return [
        ROLE,
        Slope(
            "reward_fraction", "none", by=ROLE, levels=("current", "previous", "other")
        ),
        Smooth("clock_s", 0.0, 20.0, 8, 4.0, by=ROLE, levels=("current",)),
    ]


def test_smooth_penalty_is_required() -> None:
    """A smooth takes its penalty as given: no default, ``"none"`` or a log
    penalty in the bounds."""
    with pytest.raises(TypeError):
        Smooth("x", 0.0, 1.0, 8)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="penalty"):
        Smooth("x", 0.0, 1.0, 8, "reml")
    with pytest.raises(ValueError, match="outside the bounds"):
        Smooth("x", 0.0, 1.0, 8, 50.0)


def test_simple_term_penalty_is_required() -> None:
    """A slope and a category take their penalty as a smooth does: no
    default, ``"none"`` or a log penalty in the bounds."""
    with pytest.raises(TypeError):
        Slope("x")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        Category("g", ("a", "b"), reference="a")  # type: ignore[call-arg]
    for bad in ("reml", True):
        with pytest.raises(ValueError, match="penalty"):
            Slope("x", bad)
        with pytest.raises(ValueError, match="penalty"):
            Category("g", ("a", "b"), reference="a", penalty=bad)
    with pytest.raises(ValueError, match="outside the bounds"):
        Slope("x", -11.0)


def test_prior_sd_and_log_penalty() -> None:
    """A Normal prior of SD ``s`` on a coefficient is the log penalty
    ``log(1 / s**2)``, and back; an SD that is not positive and finite is
    refused."""
    for sd in (0.5, 1.0, 2.0, 4.0):
        rho = log_penalty_from_prior_sd(sd)
        assert rho == pytest.approx(np.log(1.0 / sd**2), abs=1e-15)
        assert prior_sd_from_log_penalty(rho) == pytest.approx(sd, rel=1e-15)
    assert log_penalty_from_prior_sd(1.0) == 0.0
    for bad in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="prior SD"):
            log_penalty_from_prior_sd(bad)


def test_penalized_simple_terms_get_identity_blocks(content_sessions) -> None:
    """A penalized slope, a penalized slope by a category (one block per
    level) and a penalized category (one block per coded level) each get the
    identity on their own column at their log penalty, in column order; an
    unpenalized simple term has no block; the record names every term's
    penalty."""
    pooled = pd.concat(content_sessions.values(), ignore_index=True)
    pooled["noise"] = np.random.default_rng(0).normal(size=len(pooled))
    role = Category("role", ROLES, reference="other", penalty=-1.0)
    design = build_design(
        [
            role,
            Slope("reward_fraction", -2.0, by=role, levels=("current", "previous")),
            Slope("clock_s", 0.5),
            Slope("noise", "none"),
            Smooth("clock_s", 0.0, 20.0, 8, 4.0, by=role, levels=("current",)),
        ],
        pooled,
        intercept=False,
    )
    penalized = [
        "role[current]",
        "role[previous]",
        "role[trunk]",
        "reward_fraction:role=current",
        "reward_fraction:role=previous",
        "clock_s",
    ]
    assert [t.name for t in design.terms if t.block is not None] == [
        *penalized,
        "s(clock_s):role=current",
    ]
    assert design.by_name["noise"].block is None
    for k, name in enumerate(penalized):
        block = design.blocks[design.by_name[name].block]
        assert design.by_name[name].block == k
        np.testing.assert_array_equal(block.columns, design.by_name[name].columns)
        np.testing.assert_array_equal(block.S, np.eye(1))
    np.testing.assert_array_equal(
        design.log_penalties, [-1.0, -1.0, -1.0, -2.0, -2.0, 0.5, 4.0]
    )
    assert design.log_penalty_record() == {
        "role": {"set": "fixed", "log_penalty": -1.0},
        "reward_fraction:role=current": {"set": "fixed", "log_penalty": -2.0},
        "reward_fraction:role=previous": {"set": "fixed", "log_penalty": -2.0},
        "clock_s": {"set": "fixed", "log_penalty": 0.5},
        "noise": {"set": "none", "log_penalty": None},
        "s(clock_s):role=current": {"set": "fixed", "log_penalty": 4.0},
    }


def test_the_penalty_acts_in_the_reported_unit(content_sessions) -> None:
    """A penalized continuous slope reads its covariate per pooled SD, so the
    penalty is on the coefficient per SD: the covariate given in units 1000
    times larger builds the same columns, to rounding."""
    pooled = pd.concat(content_sessions.values(), ignore_index=True)
    terms = [
        Slope("clock_s", 0.0),
        Slope("clock_s", 0.0, by=ROLE, levels=("current",)),
    ]
    X, _ = build_design(terms, pooled, intercept=False).session(pooled)
    big = pooled.assign(clock_s=pooled["clock_s"] * 1000.0)
    X_big, _ = build_design(terms, big, intercept=False).session(big)
    np.testing.assert_allclose(X_big, X, rtol=1e-12, atol=1e-13)
    clock = pooled["clock_s"].to_numpy()
    np.testing.assert_allclose(X[:, 0], clock / np.std(clock), rtol=1e-14)


def test_fixed_knots_and_pooled_constraint(rate_sessions) -> None:
    """Two sessions with different ranges get the same knots and the same
    constraint (computed once on the pooled rows), and a session's curve equals
    the pooled design's curve at the same coefficients."""
    frames = list(rate_sessions.values())
    short = frames[0][frames[0]["time_since_s"] < 1.0].reset_index(drop=True)
    long = frames[1]
    pooled = pd.concat([short, long], ignore_index=True)
    design = build_design(
        [Smooth("time_since_s", 0.0, 20.0, 12, 2.0)], pooled, intercept=True
    )

    basis = PSplineBasis(0.0, 20.0, 12)
    Z = basis.constrained(pooled["time_since_s"].to_numpy()).constraint
    record = design.basis_record()["s(time_since_s)"]
    np.testing.assert_array_equal(record["knots"], basis.knots)
    np.testing.assert_array_equal(record["constraint"], Z)
    # The constraint is the pooled one, not either session's.
    alone = basis.constrained(short["time_since_s"].to_numpy()).constraint
    assert not np.allclose(np.abs(alone), np.abs(Z))

    beta = np.random.default_rng(0).normal(size=12)
    X_pooled, _ = design.session(pooled)
    X_short, _ = design.session(short)
    X_long, _ = design.session(long)
    np.testing.assert_allclose(
        X_short @ beta, (X_pooled @ beta)[: len(short)], atol=1e-14
    )
    np.testing.assert_allclose(
        X_long @ beta, (X_pooled @ beta)[len(short) :], atol=1e-14
    )
    np.testing.assert_allclose(X_short[:, 1:], basis.design(short["time_since_s"]) @ Z)


def test_clamp_beyond_range(rate_sessions) -> None:
    """A value above ``hi`` gives the basis row at ``hi``, below ``lo`` the row
    at ``lo``."""
    pooled = next(iter(rate_sessions.values()))
    design = build_design(
        [Smooth("time_since_s", 0.0, 3.0, 8, "none")], pooled, intercept=False
    )
    rows = pd.DataFrame({"time_since_s": [3.0, 7.5, 0.0, -1.0]})
    X, offset = design.session(rows)
    assert offset is None
    # Rows of one matrix product: equal but for the last digit on a machine
    # whose kernel treats rows differently by their place (the basis rows
    # themselves are equal bit for bit, test_pspline).
    np.testing.assert_allclose(X[1], X[0], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(X[3], X[2], rtol=1e-12, atol=1e-14)
    assert design.blocks == ()


def test_linear_functional_matches_explicit_sum(rate_sessions) -> None:
    """Model D's column ``b`` is ``sum_l B_b(log l) r_l``, summed by a loop."""
    rows = next(iter(rate_sessions.values())).iloc[:50].reset_index(drop=True)
    term = LinearFunctional(LAG_COLUMNS, LAGS, 0.0, float(np.log(10)), 8, 1.0)
    design = build_design([term], rows, intercept=False)
    X, _ = design.session(rows)
    basis = PSplineBasis(0.0, float(np.log(10)), 8)
    expected = np.zeros((len(rows), 8))
    for i in range(len(rows)):
        for b in range(8):
            for lag, column in zip(LAGS, LAG_COLUMNS, strict=True):
                at = basis.design(np.array([np.log(lag)]))[0, b]
                expected[i, b] += at * rows.loc[i, column]
    np.testing.assert_allclose(X, expected, atol=1e-14)
    (block,) = design.blocks
    np.testing.assert_array_equal(block.columns, np.arange(8))
    np.testing.assert_array_equal(design.log_penalties, [1.0])
    assert design.basis_record()["lf(lag_1..lag_10)"]["constraint"] is None


def test_standardization_ddof0_and_centering(content_sessions) -> None:
    """The pooled SD has ``ddof = 0``; a slope in an interaction is centered at
    the pooled mean of the rows it reads, a slope alone is not, and a 0/1
    column is neither centered nor scaled."""
    values = np.array([0.0, 1.0, 3.0, 4.0])
    scale = standardization(values)
    assert scale.sd == pytest.approx(np.std(values, ddof=0))
    assert scale.reference == pytest.approx(2.0)
    assert standardization(np.array([0.0, 1.0, 1.0])) is None

    pooled = pd.concat(content_sessions.values(), ignore_index=True)
    design = build_design(_content_terms(), pooled, intercept=False)
    leaf = pooled["role"] != "trunk"
    fraction = pooled.loc[leaf, "reward_fraction"].to_numpy()
    record = design.standardization_record()
    assert record == {
        "reward_fraction": {
            "sd": pytest.approx(np.std(fraction, ddof=0)),
            "ddof": 0,
            "reference": pytest.approx(np.mean(fraction)),
            "centered": [
                "reward_fraction:role=current",
                "reward_fraction:role=previous",
                "reward_fraction:role=other",
            ],
        }
    }
    X, _ = design.session(pooled)
    j = design.coef_names.index("reward_fraction:role=current")
    current = (pooled["role"] == "current").to_numpy()
    expected = (pooled["reward_fraction"] - fraction.mean()) / np.std(fraction)
    np.testing.assert_allclose(X[current, j], expected[current])
    assert np.all(X[~current, j] == 0.0)

    rate = pd.concat([pd.DataFrame({"r": [0.0, 1.0, 1.0, 0.0], "x": values})])
    by = Category("r", (0.0, 1.0), reference=0.0, penalty="none")
    built = build_design(
        [Slope("r", "none"), Slope("x", "none", by=by)], rate, intercept=True
    )
    assert built.coef_names == ("(Intercept)", "r", "x:r=0.0", "x:r=1.0")
    X, _ = built.session(rate)
    np.testing.assert_array_equal(X[:, 1], rate["r"])  # 0/1: as it is


def test_one_scale_per_covariate_across_its_terms(content_sessions) -> None:
    """Two slopes on one covariate (one per group of roles; or one alone and
    one by a role) share one SD and one reference, computed over the pooled
    rows of every level any of them reads; only the slopes by a category are
    centered, and the record names them."""
    pooled = pd.concat(content_sessions.values(), ignore_index=True)
    pooled.loc[pooled["role"] == "previous", "reward_fraction"] *= 0.3
    design = build_design(
        [
            ROLE,
            Slope("reward_fraction", "none", by=ROLE, levels=("current",)),
            Slope("reward_fraction", "none", by=ROLE, levels=("previous", "other")),
        ],
        pooled,
        intercept=False,
    )
    leaf = pooled.loc[pooled["role"] != "trunk", "reward_fraction"]
    (record,) = design.standardization_record().values()
    assert record["sd"] == pytest.approx(np.std(leaf, ddof=0))
    assert record["reference"] == pytest.approx(np.mean(leaf))
    X, _ = design.session(pooled)
    for level in ("current", "previous"):
        rows = (pooled["role"] == level).to_numpy()
        j = design.coef_names.index(f"reward_fraction:role={level}")
        expected = (pooled.loc[rows, "reward_fraction"] - leaf.mean()) / np.std(leaf)
        np.testing.assert_allclose(X[rows, j], expected)

    both = build_design(
        [
            ROLE,
            Slope("clock_s", "none"),
            Slope("clock_s", "none", by=ROLE, levels=("current",)),
        ],
        pooled,
        intercept=False,
    )
    clock = pooled["clock_s"]
    record = both.standardization_record()["clock_s"]
    assert record["sd"] == pytest.approx(np.std(clock, ddof=0))
    assert record["centered"] == ["clock_s:role=current"]
    X, _ = both.session(pooled)
    np.testing.assert_allclose(
        X[:, both.coef_names.index("clock_s")], clock / np.std(clock)
    )


def test_category_and_by_terms(content_sessions) -> None:
    """A category has one column per level but the reference; a term by a
    category is zero off its levels, whose values it does not read (the
    trunk's missing reward fraction); a value outside the levels is refused."""
    pooled = pd.concat(content_sessions.values(), ignore_index=True)
    design = build_design(_content_terms(), pooled, intercept=False)
    assert design.coef_names[:3] == ("role[current]", "role[previous]", "role[trunk]")
    assert [t.level for t in design.terms[:3]] == [
        ("role", "current"),
        ("role", "previous"),
        ("role", "trunk"),
    ]
    assert [t.kind for t in design.terms[:4]] == ["level", "level", "level", "slope"]
    assert design.terms[3].level == ("role", "current")
    # The terms fitted for a level leave a session with it.
    assert design.by_name["role[current]"].dependents == (
        "reward_fraction:role=current",
        "s(clock_s):role=current",
    )
    assert design.by_name["role[trunk]"].dependents == ()
    assert not design.missing_rows(pooled).any()
    X, _ = design.session(pooled)
    smooth = design.term_columns["s(clock_s):role=current"]
    assert np.all(X[(pooled["role"] != "current").to_numpy()][:, smooth] == 0.0)
    (block,) = design.blocks
    np.testing.assert_array_equal(block.columns, smooth)

    bad = pooled.copy()
    bad.loc[0, "role"] = "unknown"
    with pytest.raises(ValueError, match="values outside the levels"):
        design.session(bad)


def test_missing_values_are_found_and_refused(rate_sessions) -> None:
    """A row with a value the design reads missing (or an exposure that is not
    positive) is found by ``missing_rows`` and refused by ``session``."""
    rows = next(iter(rate_sessions.values())).copy()
    design = build_design(_rate_terms(), rows, intercept=True)
    rows.loc[3, "lag_4"] = np.nan
    rows.loc[5, "exposure"] = 0.0
    np.testing.assert_array_equal(np.flatnonzero(design.missing_rows(rows)), [3, 5])
    with pytest.raises(ValueError, match="2 rows with missing values"):
        design.session(rows)


def _scaled_session(family: str, scale: float) -> tuple[object, SessionData, list]:
    """A simulated session of 600 rows (an intercept, three N(0,1) columns, a
    P-spline of 7 columns) with its first N(0,1) column multiplied by
    ``scale``."""
    rng = np.random.default_rng(0)
    n = 600
    x = rng.normal(size=(n, 3))
    smooth = PSplineBasis(0.0, 10.0, 8).constrained(rng.uniform(0.0, 10.0, n))
    X = np.column_stack([np.ones(n), x, smooth.design])
    beta = np.r_[-0.5, 0.8, -0.6, 0.4, 0.3 * rng.normal(size=7)]
    if family == "softmax":
        X, beta = X[:, 1:], beta[1:]
        groups = np.repeat(np.arange(n // 4), 4)
        y = draw_response(rng, "counts", X @ beta, total=5)
        lik, aux, column = GroupedSoftmax(), groups, 0
    elif family == "poisson":
        offset = np.log(rng.uniform(0.5, 2.0, n))
        y = draw_response(rng, "poisson", X @ beta + offset)
        lik, aux, column = Poisson(), offset, 1
    else:
        y = draw_response(rng, "bernoulli", X @ beta)
        lik, aux, column = Bernoulli(), None, 1
    X = X.copy()
    X[:, column] *= scale
    block = PenaltyBlock(np.arange(X.shape[1] - 7, X.shape[1]), smooth.penalty)
    return lik, SessionData(X, y, aux), [block]


@pytest.mark.parametrize("family", ["poisson", "bernoulli", "softmax"])
def test_tiny_column_is_refused(family: str) -> None:
    """A column multiplied by 1e-4 makes the fit end at the step limit, and the
    design check refuses it, naming it; multiplied by 1e-2 it is accepted and
    the fit converges."""
    lik, tiny, blocks = _scaled_session(family, 1e-4)
    with pytest.raises(ValueError, match="step limit reached"):
        fit_at_penalties(lik, [tiny], blocks, [2.0])
    p = tiny.X.shape[1]
    terms = [DesignTerm(f"c{j}", f"c{j}", "slope", np.array([j])) for j in range(p - 7)]
    terms.append(DesignTerm("smooth", "smooth", "smooth", np.arange(p - 7, p)))
    column = 0 if family == "softmax" else 1
    with pytest.raises(ValueError, match=re.escape(f"c{column} (")):
        _refuse_small(_small_terms(tiny.X, terms))

    lik, small, blocks = _scaled_session(family, 1e-2)
    _refuse_small(_small_terms(small.X, terms))
    assert fit_at_penalties(lik, [small], blocks, [2.0]).sessions[0].fit.converged
    # A column zero on every row is left to the support rule, not refused.
    zero = np.column_stack([small.X, np.zeros(len(small.y))])
    _refuse_small(
        _small_terms(zero, [*terms, DesignTerm("z", "z", "slope", np.array([p]))])
    )


def test_a_smooth_is_measured_as_one_term(rate_sessions) -> None:
    """In a session that covers a small part of a smooth's range one of its
    columns is below the threshold, but the term is not: the check measures
    the smooth's columns together, and refuses the term only when all are
    small."""
    frames = list(rate_sessions.values())
    short = frames[0][frames[0]["time_since_s"] < 1.0].reset_index(drop=True)
    pooled = pd.concat([short, frames[1]], ignore_index=True)
    design = build_design(
        [Smooth("time_since_s", 0.0, 20.0, 12, 2.0)], pooled, intercept=True
    )
    X, _ = design.session(short)
    assert np.max(np.abs(X[:, 1:]), axis=0).min() < SMALLEST_COLUMN_SIZE
    assert design.small_terms(X) == {}
    assert list(design.small_terms(X * 1e-3)) == ["s(time_since_s)"]
    with pytest.raises(ValueError, match=re.escape("s(time_since_s) (")):
        _refuse_small(design.small_terms(X * 1e-3))


def test_every_built_column_passes_the_design_check(
    rate_sessions, content_sessions
) -> None:
    """Every column the terms build (0/1 terms, pooled-SD slopes, constrained
    spline columns, model D's functional, a smooth by a category) is at least
    ``SMALLEST_COLUMN_SIZE`` in every synthetic session (or zero)."""
    for sessions, terms, intercept in (
        (rate_sessions, _rate_terms(), True),
        (content_sessions, _content_terms(), False),
    ):
        pooled = pd.concat(sessions.values(), ignore_index=True)
        design = build_design(terms, pooled, intercept=intercept)
        for rows in sessions.values():
            X, _ = design.session(rows)
            size = np.max(np.abs(X), axis=0)
            for term in design.terms:
                largest = size[term.columns].max()
                assert largest == 0.0 or largest >= SMALLEST_COLUMN_SIZE, term.name


def test_a_missing_category_value_is_a_missing_row(content_sessions) -> None:
    """A category value that is missing (None or NaN) makes its row missing,
    for the category's own columns and for the terms by it; a value outside
    the levels is still refused."""
    pooled = pd.concat(content_sessions.values(), ignore_index=True)
    design = build_design(_content_terms(), pooled, intercept=False)
    rows = next(iter(content_sessions.values())).copy()
    rows["role"] = rows["role"].astype(object)
    rows.loc[0, "role"] = None
    rows.loc[5, "role"] = np.nan
    np.testing.assert_array_equal(np.flatnonzero(design.missing_rows(rows)), [0, 5])
    by_only = build_design(
        [Slope("clock_s", "none", by=ROLE)],
        pooled,
        intercept=False,
    )
    np.testing.assert_array_equal(np.flatnonzero(by_only.missing_rows(rows)), [0, 5])
    rows.loc[1, "role"] = "unknown"
    with pytest.raises(ValueError, match="values outside the levels"):
        design.missing_rows(rows)


def test_a_column_too_small_on_the_pooled_rows_is_refused_at_construction(
    rate_sessions,
) -> None:
    """A term whose columns are all below the threshold on the pooled rows is
    the caller's error (a units problem): building the design raises, naming
    the term; in one session the same is left to the support rule."""
    pooled = pd.concat(rate_sessions.values(), ignore_index=True)
    tiny = pooled.assign(**{c: pooled[c] * 1e-4 for c in LAG_COLUMNS})
    term = LinearFunctional(LAG_COLUMNS, LAGS, 0.0, float(np.log(10)), 8, 1.0)
    with pytest.raises(ValueError, match=re.escape("lf(lag_1..lag_10) (")):
        build_design([term], tiny, intercept=True)
    # A missing value in one pooled row does not hide the check: it is made
    # on the rows that are not missing.
    tiny.loc[0, "lag_3"] = np.nan
    with pytest.raises(ValueError, match=re.escape("lf(lag_1..lag_10) (")):
        build_design([term], tiny, intercept=True)
    design = build_design([term], pooled, intercept=True)
    X, _ = design.session(tiny.iloc[1:101])
    assert np.abs(X[:, 1:]).max() < SMALLEST_COLUMN_SIZE
    assert design.small_terms(X) == {
        "lf(lag_1..lag_10)": pytest.approx(np.abs(X[:, 1:]).max())
    }


def test_a_missing_level_is_not_the_reference(content_sessions) -> None:
    """A category with no term by it: a missing value makes its row missing
    (NaN in the level columns), not a row of the reference level (zeros)."""
    pooled = pd.concat(content_sessions.values(), ignore_index=True)
    design = build_design(
        [ROLE, Slope("clock_s", "none")],
        pooled,
        intercept=False,
    )
    rows = next(iter(content_sessions.values())).copy()
    rows["role"] = rows["role"].astype(object)
    rows.loc[2, "role"] = None
    np.testing.assert_array_equal(np.flatnonzero(design.missing_rows(rows)), [2])
    X = design.build(rows).X
    assert np.isnan(X[2, :3]).all()


def test_a_slope_is_measured_after_its_standardization(rate_sessions) -> None:
    """A slope's units do not reach the design: its raw values scaled by 1e-6
    build, and give the same standardized column (to rounding). A
    standardized slope is still measured, because it can be small on the
    pooled rows that are not missing: when it shares its scale with a slope
    of the covariate on another level, and when its spread lies on rows
    missing for another term. Both are refused, naming the term."""
    pooled = pd.concat(rate_sessions.values(), ignore_index=True)
    by = Category("last_rewarded", (0.0, 1.0), reference=0.0, penalty="none")
    for terms in ([Slope("x", "none")], [by, Slope("x", "none", by=by)]):
        tiny = pooled.assign(x=pooled["x"] * 1e-6)
        X, _ = build_design(terms, pooled, intercept=True).session(pooled)
        X_tiny, _ = build_design(terms, tiny, intercept=True).session(tiny)
        np.testing.assert_allclose(X_tiny, X, rtol=1e-12, atol=1e-15)

    rng = np.random.default_rng(0)
    level = np.where(np.arange(400) < 200, "a", "b")
    spread = 5.0 + 100.0 * np.where(np.arange(400) % 2 == 0, 1.0, -1.0)
    x = np.where(level == "a", 5.0 + 1e-4 * rng.normal(size=400), spread)
    rows = pd.DataFrame({"g": level, "x": x})
    g = Category("g", ("a", "b"), reference="a", penalty="none")
    shared = [
        g,
        Slope("x", "none", by=g, levels=("a",)),
        Slope("x", "none", by=g, levels=("b",)),
    ]
    with pytest.raises(ValueError, match=re.escape("x:g=a (")):
        build_design(shared, rows, intercept=True)

    z = 1e-3 * rng.normal(size=400)
    z[:5] = [50.0, -50.0, 40.0, -40.0, 30.0]
    h = np.where(np.arange(400) % 2 == 0, "u", "v").astype(object)
    h[:5] = None  # the rows holding z's spread are missing
    rows = pd.DataFrame({"z": z, "h": h})
    with pytest.raises(ValueError, match=re.escape("z (")):
        build_design(
            [
                Category("h", ("u", "v"), reference="u", penalty="none"),
                Slope("z", "none"),
            ],
            rows,
            intercept=True,
        )
