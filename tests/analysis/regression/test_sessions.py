"""The per-session fitting loop: support, penalized simple terms, sessions
without a finite maximum, records."""

from __future__ import annotations

import jax
import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq

from continuum_swr_replay.analysis.regression import runtime
from continuum_swr_replay.analysis.regression.calibration import log_penalty_for_edf
from continuum_swr_replay.analysis.regression.constants import LOG_PENALTY_BOUNDS
from continuum_swr_replay.analysis.regression.contrasts import evaluate_contrast
from continuum_swr_replay.analysis.regression.likelihoods import (
    Gaussian,
    GroupedSoftmax,
    Poisson,
)
from continuum_swr_replay.analysis.regression.record import NoFit
from continuum_swr_replay.analysis.regression.reml import SessionData, fit_at_penalties
from continuum_swr_replay.analysis.regression.sessions import fit_sessions
from continuum_swr_replay.analysis.regression.simulation import draw_data_set
from continuum_swr_replay.analysis.regression.support import Removed
from continuum_swr_replay.analysis.regression.terms import (
    SMALLEST_COLUMN_SIZE,
    Category,
    Offset,
    Slope,
    Smooth,
    build_design,
    log_penalty_from_prior_sd,
)
from continuum_swr_replay.analysis.shared_statistics import animal_day_session_values
from scripts.checks import simulate_declared_penalty as sim

from .conftest import ROLE, by_session, cohort, rate_rows

N_ARMS = 6


def _choice_rows(seed: int, separated: bool, n_events: int = 140) -> pd.DataFrame:
    """A content session of one-hot choices among six candidates: a clock with
    a linear effect and two 0/1 columns ``a`` and ``b`` on the second and third
    rows of some events. Separated: ``a`` marks the second row of the first 40
    events, never chosen, and the third row of the next 40, which ``b`` marks
    too; each column alone has choices on its rows, but the direction "a down,
    b up" lowers only rows never chosen, so the fit has no finite maximum."""
    rng = np.random.default_rng(seed)
    clock = rng.uniform(0.0, 5.0, (n_events, N_ARMS))
    a = np.zeros((n_events, N_ARMS))
    b = np.zeros((n_events, N_ARMS))
    if separated:
        a[:40, 1] = 1.0
        a[40:80, 2] = 1.0
        b[40:80, 2] = 1.0
    else:
        a[rng.uniform(size=n_events) < 0.5, 2] = 1.0
        b[rng.uniform(size=n_events) < 0.3, 3] = 1.0
    eta = -1.2 * clock + 0.5 * a - 0.3 * b
    p = np.exp(eta) / np.exp(eta).sum(axis=1, keepdims=True)
    y = np.zeros((n_events, N_ARMS))
    y[np.arange(n_events), [rng.choice(N_ARMS, p=row) for row in p]] = 1.0
    if separated:
        y[:40, 0] += y[:40, 1]
        y[:40, 1] = 0.0
    return pd.DataFrame(
        {
            "animal": "sim",
            "session_date": f"2026{seed:04d}",
            "epoch": "02_r1",
            "event_id": np.repeat(np.arange(n_events), N_ARMS),
            "y": y.ravel(),
            "clock": clock.ravel(),
            "a": a.ravel(),
            "b": b.ravel(),
        }
    )


def _fit(design, sessions, lik, **kw):
    return fit_sessions(
        design,
        sessions,
        lik,
        model="test",
        kind="SWR",
        table_sha256={"t": "0" * 64},
        **kw,
    )


def _rate_design(pooled: pd.DataFrame, *categories: Category):
    """The small rate design: ``categories``, a slope of ``x`` and the
    exposure's offset, with an intercept."""
    return build_design(
        [*categories, Slope("x", "none"), Offset("exposure")], pooled, intercept=True
    )


def test_a_separated_session_is_kept_with_its_terms_missing() -> None:
    """At a given penalty, one of six sessions has no finite maximum along two
    0/1 columns that each pass the support rule: ``fit_at_penalties`` raises
    on it, the loop fits every session, the separated one with those columns
    left out (NaN, with the round), the others equal to their fits alone."""
    sessions = by_session(
        [_choice_rows(seed, separated=seed == 3) for seed in range(6)]
    )
    pooled = pd.concat(sessions.values(), ignore_index=True)
    design = build_design(
        [Smooth("clock", 0.0, 5.0, 10, 2.0), Slope("a", "none"), Slope("b", "none")],
        pooled,
        intercept=False,
    )
    records = _fit(design, sessions, GroupedSoftmax(), response="y", unit="event_id")

    for (key, rows), record in zip(sessions.items(), records, strict=True):
        X, _ = design.session(rows)
        data = SessionData(X, rows["y"].to_numpy(), rows["event_id"].to_numpy())
        assert record.status == "converged"
        assert all(s["passes"] for s in record.support.values())
        if key[1] == "20260003":
            with pytest.raises(ValueError, match="no finite maximum"):
                fit_at_penalties(
                    GroupedSoftmax(), [data], design.blocks, design.log_penalties
                )
            assert record.left_out == {
                "a": Removed.NO_FINITE_MAXIMUM,
                "b": Removed.NO_FINITE_MAXIMUM,
            }
            assert record.left_out_round == {"a": 1, "b": 1}
            assert np.isnan(record.coef[9:]).all()
            assert np.isnan(np.asarray(record.vcov)[9:]).all()
            assert np.isfinite(record.coef[:9]).all()
            assert record.no_fit == ""
            assert record.edf["a"] == 0.0 and set(record.collinearity) == {"s(clock)"}
            continue
        alone = fit_at_penalties(
            GroupedSoftmax(), [data], design.blocks, design.log_penalties
        ).sessions[0]
        np.testing.assert_array_equal(record.coef, alone.fit.beta)
        np.testing.assert_array_equal(record.vcov, alone.covariance)
        assert record.left_out == {} and record.no_fit == ""
        assert record.edf["s(clock)"] == pytest.approx(alone.edf[0])


def _role_rows(seed: int, n_previous: int, never_chosen: bool = False) -> pd.DataFrame:
    """Content rows (conftest) with the previous arm a candidate of only the
    first ``n_previous`` events; ``never_chosen``: no share on it."""
    from .conftest import content_rows

    rows = content_rows(seed)
    previous = rows["role"] == "previous"
    rows = rows[~previous | (rows["event_id"] < n_previous)].copy()
    if never_chosen:
        rows.loc[rows["role"] == "previous", "share"] = 0.0
    return rows.reset_index(drop=True)


def _role_design(pooled: pd.DataFrame):
    return build_design(
        [
            ROLE,
            Slope(
                "reward_fraction",
                "none",
                by=ROLE,
                levels=("current", "previous", "other"),
            ),
            Smooth("clock_s", 0.0, 20.0, 8, 4.0),
        ],
        pooled,
        intercept=False,
    )


def test_an_inactive_role_drops_its_rows() -> None:
    """A session whose previous arm is a candidate in 3 events fails the
    support rule for that role: its rows are dropped (counted), and the fit is
    that of the session without those rows and columns."""
    frames = [_role_rows(0, n_previous=3), *(_role_rows(s, 150) for s in (1, 2))]
    sessions = by_session(frames)
    pooled = pd.concat(sessions.values(), ignore_index=True)
    design = _role_design(pooled)
    record = _fit(
        design, sessions, GroupedSoftmax(), response="share", unit="event_id"
    )[0]

    # A dropped level keeps the counts that made it fail.
    assert record.support["role[previous]"]["n_high"] == 3
    # The levels are decided first: the slope by the role leaves with its rows.
    assert record.left_out == {
        "role[previous]": Removed.SUPPORT,
        "reward_fraction:role=previous": Removed.WITH_LEVEL,
    }
    assert record.left_out_round == {}
    assert record.cohort_counts["rows_dropped"] == {"role[previous]": 3}
    assert record.n_rows == len(frames[0]) - 3

    kept = frames[0][frames[0]["role"] != "previous"].reset_index(drop=True)
    X, _ = design.session(kept)
    active = ~np.isin(design.coef_names, list(record.left_out))
    data = SessionData(
        X, kept["share"].to_numpy(), kept["event_id"].to_numpy(), active=active
    )
    alone = fit_at_penalties(
        GroupedSoftmax(), [data], design.blocks, design.log_penalties
    )
    # Not bit for bit: the loop builds the design on every row and then drops
    # rows, and here it is built on the rows kept. A row of the smooth is a row
    # of a matrix product, whose last digit depends, on some machines (the AVX2
    # kernels of the linear-algebra library), on where the row lies in the
    # matrix. A fit on other rows or columns differs in the first digits.
    # NaN (an inactive column) must be at the same places.
    np.testing.assert_allclose(
        record.coef, alone.sessions[0].fit.beta, rtol=1e-10, atol=1e-12
    )


def test_a_role_without_a_finite_maximum_drops_its_rows() -> None:
    """A role that passes the support rule but holds no share in a session has
    no finite maximum along its column: the column is left out, and, as the
    support rule would, the role's rows are dropped and the slope by it leaves
    with them."""
    frames = [_role_rows(0, 150, never_chosen=True), _role_rows(1, 150)]
    sessions = by_session(frames)
    design = _role_design(pd.concat(sessions.values(), ignore_index=True))
    record = _fit(
        design, sessions, GroupedSoftmax(), response="share", unit="event_id"
    )[0]

    assert record.status == "converged"
    # A level dropped for no finite maximum keeps its counts from before.
    assert record.support["role[previous]"]["n_high"] == 150
    assert record.left_out["role[previous]"] == Removed.NO_FINITE_MAXIMUM
    assert record.left_out_round == {"role[previous]": 1}
    assert record.left_out["reward_fraction:role=previous"] == Removed.WITH_LEVEL
    assert record.cohort_counts["rows_dropped"] == {"role[previous]": 150}
    assert record.n_rows == len(frames[0]) - 150


def test_rate_sessions_are_fitted_and_recorded(rate_sessions) -> None:
    """Poisson rate sessions with an offset: every session converges, equals
    its fit alone, and records what the likelihood saw, the edf of every term,
    the collinearity, the basis and the standardization."""
    pooled = pd.concat(rate_sessions.values(), ignore_index=True)
    design = build_design(
        [
            Smooth("time_since_s", 0.0, 20.0, 12, 2.0),
            Slope("last_rewarded", "none"),
            Slope("x", "none"),
            Offset("exposure"),
        ],
        pooled,
        intercept=True,
    )
    records = _fit(
        design, rate_sessions, Poisson(), response="count", unit="well_visit_id"
    )
    for rows, record in zip(rate_sessions.values(), records, strict=True):
        X, offset = design.session(rows)
        data = SessionData(X, rows["count"].to_numpy(), offset)
        alone = fit_at_penalties(Poisson(), [data], design.blocks, design.log_penalties)
        np.testing.assert_array_equal(record.coef, alone.sessions[0].fit.beta)
        assert record.likelihood == "poisson" and record.status == "converged"
        assert record.n_events == rows["count"].sum()
        assert record.n_units == rows["well_visit_id"].nunique()
        assert set(record.edf) == {"s(time_since_s)", "last_rewarded", "x"}
        assert record.edf["x"] == 1.0
        assert set(record.collinearity["x"]) == {"r2", "vif"}
        assert set(record.collinearity["s(time_since_s)"]) == {
            "observed",
            "worst",
            "estimate",
        }
        assert record.log_penalty == {
            "s(time_since_s)": {"set": "fixed", "log_penalty": 2.0},
            "last_rewarded": {"set": "none", "log_penalty": None},
            "x": {"set": "none", "log_penalty": None},
        }
        assert set(record.standardization) == {"x"}
        assert record.cohort_counts["rows_missing_values"] == 0


def test_the_program_bound_holds_between_the_sessions_of_one_loop(
    monkeypatch: pytest.MonkeyPatch, rate_sessions
) -> None:
    """With the program limit patched to 0, JAX's caches are cleared before
    every session's fit (each is an outermost call of the fitter), so the
    live programs stay bounded during a loop over a whole cohort; the records
    are the ones the loop gives without the bound."""
    pooled = pd.concat(rate_sessions.values(), ignore_index=True)
    design = _rate_design(pooled)
    fit = {"response": "count", "unit": "well_visit_id"}
    unbounded = _fit(design, rate_sessions, Poisson(), **fit)
    clears = []
    clear = jax.clear_caches
    monkeypatch.setattr(runtime, "LIVE_PROGRAM_LIMIT", 0)
    monkeypatch.setattr(jax, "clear_caches", lambda: (clears.append(True), clear()))
    bounded = _fit(design, rate_sessions, Poisson(), **fit)
    assert len(clears) >= len(rate_sessions)
    for got, want in zip(bounded, unbounded, strict=True):
        assert (got.status, got.n_iter) == (want.status, want.n_iter)
        assert got.coef == want.coef
        assert got.vcov == want.vcov


def test_a_design_not_identified_at_the_first_fit_names_the_session(
    rate_sessions,
) -> None:
    """Two copies of one slope both pass the support rule; the first fit finds
    the design not identified, an error that names the session."""
    sessions = {}
    for key, rows in rate_sessions.items():
        sessions[key] = rows.assign(x_copy=rows["x"])
    pooled = pd.concat(sessions.values(), ignore_index=True)
    design = build_design(
        [Slope("x", "none"), Slope("x_copy", "none"), Offset("exposure")],
        pooled,
        intercept=True,
    )
    with pytest.raises(
        ValueError, match=r"session sim/20260000/02_r1: .*not identified"
    ):
        _fit(design, sessions, Poisson(), response="count", unit="well_visit_id")


def test_missing_rows_are_dropped_and_counted(rate_sessions) -> None:
    """Rows with a missing value the design reads are dropped before the fit
    and counted."""
    key, rows = next(iter(rate_sessions.items()))
    rows = rows.copy()
    rows.loc[[2, 7], "x"] = np.nan
    pooled = pd.concat(rate_sessions.values(), ignore_index=True)
    design = _rate_design(pooled)
    (record,) = _fit(
        design, {key: rows}, Poisson(), response="count", unit="well_visit_id"
    )
    assert record.cohort_counts["rows_missing_values"] == 2
    assert record.n_rows == len(rows) - 2


def test_a_session_with_no_rows_left_is_recorded_not_fitted(rate_sessions) -> None:
    """A session given with no rows, or whose rows are all missing, or with no
    events, is not fitted: one record says why, and the loop goes on."""
    pooled = pd.concat(rate_sessions.values(), ignore_index=True)
    design = _rate_design(pooled)
    keys = list(rate_sessions)
    frames = list(rate_sessions.values())
    missing = frames[1].assign(x=np.nan)
    quiet = frames[2].assign(count=0.0)
    sessions = {
        keys[0]: frames[0].iloc[:0],
        keys[1]: missing,
        keys[2]: quiet,
        keys[3]: frames[3],
    }
    records = _fit(design, sessions, Poisson(), response="count", unit="well_visit_id")
    assert [r.status for r in records] == ["not fitted"] * 3 + ["converged"]
    assert records[0].no_fit == NoFit.NO_ROWS
    assert records[1].cohort_counts["rows_missing_values"] == len(missing)
    assert records[1].no_fit == NoFit.NO_ROWS
    assert records[2].no_fit == NoFit.NO_EVENTS
    assert records[3].no_fit == ""
    for record in records[:3]:
        assert np.isnan(record.coef).all() and record.edf == {}


def _grouped_rate(rows: pd.DataFrame, b_visits: np.ndarray) -> pd.DataFrame:
    """Rate rows with a category ``grp``: level ``b`` on the given visits,
    the reference ``a`` on the others."""
    return rows.assign(grp=np.where(np.isin(rows["well_visit_id"], b_visits), "b", "a"))


def test_a_level_passes_on_its_own_count(rate_sessions) -> None:
    """A level with many units passes however thin the rest is: a session
    whose reference level holds 4 visits keeps every row (the rule for a level
    is one-sided)."""
    frames = list(rate_sessions.values())
    keys = list(rate_sessions)
    sessions = {
        keys[0]: _grouped_rate(frames[0], np.arange(4, 200)),
        keys[1]: _grouped_rate(frames[1], np.arange(0, 200, 2)),
    }
    pooled = pd.concat(sessions.values(), ignore_index=True)
    grp = Category("grp", ("a", "b"), reference="a", penalty="none")
    design = _rate_design(pooled, grp)
    record = _fit(design, sessions, Poisson(), response="count", unit="well_visit_id")[
        0
    ]
    assert record.support["grp[b]"]["passes"]
    assert record.support["grp[b]"]["n_low"] < 5
    assert record.left_out == {} and record.cohort_counts["rows_dropped"] == {}
    assert record.n_rows == len(sessions[keys[0]])


def test_support_is_counted_again_on_the_rows_kept(content_sessions) -> None:
    """A 0/1 slope that is 1 on the previous arm's rows of 3 events and the
    current arm's rows of 3 others passes on all rows of a session (6 events),
    but the previous arm is too thin and its rows are dropped: the slope is
    counted again on the rows kept (3 events), fails there, and the record
    holds the counts the fit used."""
    frames = []
    for rows in content_sessions.values():
        on_previous = (rows["role"] == "previous") & (rows["event_id"] < 3)
        on_current = (rows["role"] == "current") & rows["event_id"].between(3, 5)
        frames.append(rows.assign(flag=(on_previous | on_current) * 1.0))
    first = frames[0]
    keep = (first["role"] != "previous") | (first["event_id"] < 3)
    frames[0] = first[keep].reset_index(drop=True)
    sessions = by_session(frames)
    pooled = pd.concat(sessions.values(), ignore_index=True)
    design = build_design(
        [ROLE, Slope("flag", "none"), Smooth("clock_s", 0.0, 20.0, 8, 4.0)],
        pooled,
        intercept=False,
    )
    record = _fit(
        design, sessions, GroupedSoftmax(), response="share", unit="event_id"
    )[0]
    assert record.left_out == {
        "role[previous]": Removed.SUPPORT,
        "flag": Removed.SUPPORT,
    }
    assert record.cohort_counts["rows_dropped"] == {"role[previous]": 3}
    assert record.support["flag"]["n_high"] == 3
    assert not record.support["flag"]["passes"]
    assert record.support["role[previous]"]["n_high"] == 3
    assert record.status == "converged"


def test_a_session_left_without_events_by_the_row_rule_is_not_fitted(
    rate_sessions,
) -> None:
    """A thin level that holds every event: its rows are dropped, no event is
    left, and the session gets a not-fitted record; the loop goes on."""
    frames = list(rate_sessions.values())
    keys = list(rate_sessions)
    thin = _grouped_rate(frames[0], np.arange(3))
    thin.loc[thin["grp"] == "a", "count"] = 0.0
    sessions = {keys[0]: thin, keys[1]: _grouped_rate(frames[1], np.arange(0, 200, 2))}
    pooled = pd.concat(sessions.values(), ignore_index=True)
    grp = Category("grp", ("a", "b"), reference="a", penalty="none")
    design = _rate_design(pooled, grp)
    first, second = _fit(
        design, sessions, Poisson(), response="count", unit="well_visit_id"
    )
    assert first.status == "not fitted" and first.no_fit == NoFit.NO_EVENTS
    # The support rule reads rows, not events: x has the rows of many visits
    # on each side of its median on the rows kept, so it passes.
    assert first.left_out == {"grp[b]": Removed.SUPPORT}
    assert first.support["x"]["passes"]
    assert second.status == "converged"


def test_a_column_too_small_in_one_session_fails_its_support(content_sessions) -> None:
    """A slope centered at its pooled mean is nearly constant at that mean in
    one session: its column there is below the threshold, so it fails the
    support rule in that session (inactive, with the reason recorded), and the
    loop fits every session."""
    from .conftest import content_rows

    frames = list(content_sessions.values())[:2]
    pooled = pd.concat(frames, ignore_index=True)
    design = build_design(
        [
            ROLE,
            Slope(
                "reward_fraction",
                "none",
                by=ROLE,
                levels=("current", "previous", "other"),
            ),
        ],
        pooled,
        intercept=False,
    )
    mean = design.standardization_record()["reward_fraction"]["reference"]
    flat = content_rows(7)
    flat.loc[flat["role"] == "previous", "reward_fraction"] = mean + 0.001
    sessions = {**by_session(frames), ("sim", "flat", "02_r1"): flat}
    records = _fit(
        design, sessions, GroupedSoftmax(), response="share", unit="event_id"
    )
    assert [r.status for r in records] == ["converged"] * 3
    last = records[-1]
    assert last.left_out == {"reward_fraction:role=previous": Removed.TOO_SMALL}
    entry = last.support["reward_fraction:role=previous"]
    assert not entry["passes"] and entry["reason"] == Removed.TOO_SMALL
    assert 0.0 < entry["size"] < SMALLEST_COLUMN_SIZE


def test_a_session_without_reference_rows_names_the_session_and_columns(
    rate_sessions,
) -> None:
    """A session with no rows of a category's reference level: the intercept
    and the levels' columns are collinear, and the fit is refused, naming the
    session, the category, its reference level and those coefficients (the
    reference level must have rows in every fit)."""
    frames = list(rate_sessions.values())
    keys = list(rate_sessions)
    level = {0: "a", 1: "b", 2: "c"}
    sessions = {}
    for key, rows in zip(keys[:3], frames[:3], strict=True):
        visit = rows["well_visit_id"]
        sessions[key] = rows.assign(grp=(visit % 3).map(level))
    first = sessions[keys[0]]
    sessions[keys[0]] = first.assign(grp=np.where(first["well_visit_id"] % 2, "b", "c"))
    pooled = pd.concat(sessions.values(), ignore_index=True)
    grp = Category("grp", ("a", "b", "c"), reference="a", penalty="none")
    design = _rate_design(pooled, grp)
    message = (
        r"session sim/20260000/02_r1: the category 'grp' has no rows of its "
        r"reference level 'a'.*grp\[b\], grp\[c\]"
    )
    with pytest.raises(ValueError, match=message):
        _fit(design, sessions, Poisson(), response="count", unit="well_visit_id")


def _visits_at(levels: list[str], date: str = "20260101") -> pd.DataFrame:
    """Rate rows of one bin per visit with a category ``level``: one visit at
    each of ``levels``, Poisson counts of mean 2, exposure 1."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "animal": "sim",
            "session_date": date,
            "epoch": "02_r1",
            "visit": np.arange(len(levels)),
            "level": levels,
            "count": rng.poisson(2.0, len(levels)).astype(float),
            "exposure": 1.0,
        }
    )


def test_a_penalized_category_without_reference_rows_is_refused() -> None:
    """Every visit at level b of a penalized category whose reference is a:
    the data identify only the intercept plus b, and the penalty alone would
    split them (b at 0 with the prior SD as its standard error). The fit is
    refused, naming the session, the category, its reference level and its
    level's coefficient."""
    frame = _visits_at(["b"] * 12)
    level = Category("level", ("a", "b"), "a", log_penalty_from_prior_sd(2.0))
    design = build_design([level, Offset("exposure")], frame, intercept=True)
    with pytest.raises(
        ValueError,
        match=r"session sim/20260101/02_r1: the category 'level' has no rows of its "
        r"reference level 'a'.*level\[b\]",
    ):
        _fit(
            design,
            {("sim", "20260101", "02_r1"): frame},
            Poisson(),
            response="count",
            unit="visit",
        )


def test_a_softmax_session_without_the_reference_role_is_refused() -> None:
    """Under the grouped softmax, a session whose events have no row of the
    penalized role category's reference: the levels' columns are collinear
    with one another, and the fit is refused, naming that session."""
    from .conftest import content_rows

    second = content_rows(1)
    frames = [content_rows(0), second[second["role"] != "other"]]
    sessions = by_session(frames)
    weak = log_penalty_from_prior_sd(2.0)
    role = Category("role", ROLE.levels, reference="other", penalty=weak)
    design = build_design(
        [role, Slope("clock_s", weak)],
        pd.concat(sessions.values(), ignore_index=True),
        intercept=False,
    )
    with pytest.raises(
        ValueError,
        match=r"session sim/20260001/02_r1: the category 'role' has no rows of its "
        r"reference level 'other'.*role\[current\], role\[previous\], role\[trunk\]",
    ):
        _fit(design, sessions, GroupedSoftmax(), response="share", unit="event_id")


def test_a_shared_category_needs_its_reference_in_the_fit_only() -> None:
    """A category that is not a baseline is shared by the sessions of a fit:
    the fit identifies it when one of its sessions has rows of the reference
    level, so a fit by day whose second session has none is fitted."""
    sessions = {
        ("sim", "20260101", "02_r1"): _visits_at(["a", "b"] * 15),
        ("sim", "20260101", "04_r2"): _visits_at(["b"] * 12),
    }
    level = Category("level", ("a", "b"), "a", log_penalty_from_prior_sd(2.0))
    design = build_design(
        [level, Offset("exposure")],
        pd.concat(sessions.values(), ignore_index=True),
        intercept=True,
        baseline=("(Intercept)",),
    )
    (record,) = _fit(
        design, sessions, Poisson(), response="count", unit="visit", fit_by="day"
    )
    assert record.status == "converged" and record.left_out == {}


def test_events_without_response_are_counted_apart_from_those_emptied() -> None:
    """Under the softmax, an event given with no response and an event whose
    response was all on dropped rows are counted under two names."""
    frames = [_role_rows(0, n_previous=3), *(_role_rows(s, 150) for s in (1, 2))]
    first = frames[0]
    first.loc[first["event_id"] == 10, "share"] = 0.0
    for event in (0, 1):
        rows = first["event_id"] == event
        first.loc[rows, "share"] = (first.loc[rows, "role"] == "previous") * 1.0
    sessions = by_session(frames)
    design = _role_design(pd.concat(sessions.values(), ignore_index=True))
    record = _fit(
        design, sessions, GroupedSoftmax(), response="share", unit="event_id"
    )[0]
    assert record.cohort_counts["events_without_response"] == 1
    assert record.cohort_counts["events_emptied_by_dropped_rows"] == 2
    assert record.n_units == 150 - 3


def _median_case() -> pd.DataFrame:
    """15 visits of 21 bins, one event each: x per visit and a category, whose
    level C (3 visits, all at x = 2) is too thin. On all rows the median of x
    is 2 and no visit is above it, so x fails; once C's rows are dropped the
    median moves below 2 and x passes."""
    spec = (
        [("A", 0.0)] * 3
        + [("B", 0.0)] * 3
        + [("A", 1.0)]
        + [("A", 2.0)] * 3
        + [("B", 2.0)] * 2
        + [("C", 2.0)] * 3
    )
    rows = [
        {
            "animal": "sim",
            "session_date": "1",
            "epoch": "02_r1",
            "visit": visit,
            "cat": level,
            "x": x,
            "exposure": 0.1,
            "count": 1.0 if b == 0 else 0.0,
        }
        for visit, (level, x) in enumerate(spec)
        for b in range(21)
    ]
    return pd.DataFrame(rows)


@pytest.mark.parametrize("category_first", [True, False])
def test_a_term_is_judged_once_on_the_final_rows(category_first: bool) -> None:
    """The levels are decided first (C is thin, its rows are dropped), then
    every other term once on the rows kept: x, failing on all rows, passes
    there and is fitted. Its left_out entry and its passes flag agree, the
    dropped level keeps the counts that made it fail, and the result does not
    depend on the order of the terms."""
    rows = _median_case()
    cat = Category("cat", ("A", "B", "C"), reference="A", penalty="none")
    terms = [cat, Slope("x", "none"), Offset("exposure")]
    if not category_first:
        terms = [Slope("x", "none"), Offset("exposure"), cat]
    design = build_design(terms, rows, intercept=True)
    (record,) = _fit(
        design, {("sim", "1", "02_r1"): rows}, Poisson(), response="count", unit="visit"
    )
    assert record.status == "converged"
    assert record.left_out == {"cat[C]": Removed.SUPPORT}
    assert record.support["x"]["passes"]
    assert np.isfinite(record.coef[record.coef_names.index("x")])
    assert record.support["cat[C]"]["n_high"] == 3
    assert not record.support["cat[C]"]["passes"]
    for column, why in record.left_out.items():
        if why == Removed.SUPPORT:
            assert not record.support[column]["passes"]
    for column, entry in record.support.items():
        if not entry["passes"]:
            assert column in record.left_out


def _empty_reference_rows() -> pd.DataFrame:
    """40 visits of 30 bins with a category: the reference level A (10 visits)
    holds no event, B and C do; a slope x per visit."""
    rng = np.random.default_rng(1)
    rows = []
    for visit in range(40):
        level = "A" if visit < 10 else ("B" if visit < 25 else "C")
        x = rng.normal()
        for _ in range(30):
            count = 0.0 if level == "A" else float(rng.poisson(0.2))
            rows.append(
                {"visit": visit, "cat": level, "x": x, "exposure": 0.1, "count": count}
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("shift", [0.0, 1e4, 1e5])
def test_a_shift_common_to_every_candidate_leaves_no_column_out(shift: float) -> None:
    """Events of two candidates with shares (0.2, 0.8) and a slope of 0 and 2
    plus ``shift``, which is common to an event's candidates and cancels in
    its softmax: the session is fitted, nothing left out, at ``log(4) / 2``
    per SD (the slope's SD is 1)."""
    n_events = 100
    rows = pd.DataFrame(
        {
            "animal": "sim",
            "session_date": "20260101",
            "epoch": "02_r1",
            "event_id": np.repeat(np.arange(n_events), 2),
            "share": np.tile([0.2, 0.8], n_events),
            "x": np.tile([0.0, 2.0], n_events) + shift,
        }
    )
    design = build_design([Slope("x", "none")], rows, intercept=False)
    (record,) = _fit(
        design,
        by_session([rows]),
        GroupedSoftmax(),
        response="share",
        unit="event_id",
    )
    assert record.status == "converged"
    assert record.left_out == {} and record.no_fit == ""
    np.testing.assert_allclose(record.coef, [np.log(4.0) / 2.0], rtol=0, atol=1e-9)


def test_no_finite_maximum_along_the_intercept_keeps_the_fit_tried() -> None:
    """The reference level has 10 visits and no event, two other levels have
    events: the fit has no finite maximum along the intercept. Nothing is
    dropped or made inactive. A fit was tried, so the session is not "not
    fitted": its record keeps the last fit's status, its true counts and NaN
    coefficients, with the cause; the loop goes on."""
    frame = _empty_reference_rows()
    other = frame.assign(count=np.random.default_rng(2).poisson(0.2, len(frame)) * 1.0)
    design = build_design(
        [
            Category("cat", ("A", "B", "C"), reference="A", penalty="none"),
            Slope("x", "none"),
            Offset("exposure"),
        ],
        frame,
        intercept=True,
    )
    first, second = _fit(
        design,
        {("sim", "1", "02_r1"): frame, ("sim", "2", "02_r1"): other},
        Poisson(),
        response="count",
        unit="visit",
    )
    assert first.status == "no finite maximum"
    assert first.no_fit == NoFit.ALONG_THE_INTERCEPT
    assert first.n_rows == len(frame) and first.n_units == 40
    assert first.n_events == frame["count"].sum() > 0
    assert first.n_iter > 0 and first.edf == {} and first.collinearity == {}
    assert first.left_out == {} and first.cohort_counts["rows_dropped"] == {}
    assert np.isnan(first.coef).all()
    assert second.status == "converged"


def test_a_fit_along_a_penalty_direction_keeps_the_fit_tried() -> None:
    """A content session whose response is each event's row of smallest clock
    has no finite maximum along the smooth's own line, which cannot be left
    out: its record keeps the last fit's status and its true counts, with NaN
    coefficients and the cause."""
    from .conftest import content_rows

    frames = [content_rows(seed) for seed in range(3)]
    first = frames[0]
    smallest = first.groupby("event_id")["clock_s"].transform("min")
    first["share"] = (first["clock_s"] == smallest) * 1.0
    sessions = by_session(frames)
    design = build_design(
        [ROLE, Smooth("clock_s", 0.0, 20.0, 8, 6.0)],
        pd.concat(sessions.values(), ignore_index=True),
        intercept=False,
    )
    record = _fit(
        design, sessions, GroupedSoftmax(), response="share", unit="event_id"
    )[0]
    assert record.status == "no finite maximum"
    assert record.no_fit == NoFit.ALONG_A_PENALTY
    assert (record.n_rows, record.n_units, record.n_events) == (600, 150, 150.0)
    assert np.isnan(record.coef).all() and record.edf == {}


def test_an_unpenalized_term_leaves_whole() -> None:
    """An unpenalized smooth whose rows in one session hold no share above a
    clock of 17 s: the refit leaves out only part of its columns. The whole
    term leaves the session (no finite maximum, the round of its first
    columns), and the session is fitted again without it: the fit is that
    of the session with every column of the smooth inactive."""
    from .conftest import content_rows

    frames = [content_rows(seed) for seed in range(4)]
    gap = frames[2]
    gap.loc[gap["clock_s"] > 17.0, "share"] = 0.0
    sessions = by_session(frames)
    design = build_design(
        [ROLE, Smooth("clock_s", 0.0, 20.0, 14, "none")],
        pd.concat(sessions.values(), ignore_index=True),
        intercept=False,
    )
    record = _fit(
        design, sessions, GroupedSoftmax(), response="share", unit="event_id"
    )[2]
    smooth = [c for c in design.coef_names if c.startswith("s(clock_s).")]
    assert record.status == "converged"
    assert record.left_out == dict.fromkeys(smooth, Removed.NO_FINITE_MAXIMUM)
    assert record.left_out_round == dict.fromkeys(smooth, 1)
    assert record.edf["s(clock_s)"] == 0.0

    X, _ = design.session(gap)
    active = ~np.isin(design.coef_names, smooth)
    data = SessionData(
        X, gap["share"].to_numpy(), gap["event_id"].to_numpy(), active=active
    )
    alone = fit_at_penalties(GroupedSoftmax(), [data], [], [])
    np.testing.assert_array_equal(record.coef, alone.sessions[0].fit.beta)


def test_an_offset_needs_the_poisson_likelihood(
    rate_sessions, content_sessions
) -> None:
    """An Offset term is refused under any other likelihood: a Bernoulli on
    the rate rows, a grouped softmax on the content rows."""
    from continuum_swr_replay.analysis.regression.likelihoods import Bernoulli

    rates = {k: r.assign(y=(r["count"] > 0) * 1.0) for k, r in rate_sessions.items()}
    design = _rate_design(pd.concat(rates.values(), ignore_index=True))
    with pytest.raises(ValueError, match="an Offset term needs the Poisson"):
        _fit(design, rates, Bernoulli(), response="y", unit="well_visit_id")

    content = {k: r.assign(exposure=1.0) for k, r in content_sessions.items()}
    design = build_design(
        [ROLE, Offset("exposure")],
        pd.concat(content.values(), ignore_index=True),
        intercept=False,
    )
    with pytest.raises(ValueError, match="an Offset term needs the Poisson"):
        _fit(design, content, GroupedSoftmax(), response="share", unit="event_id")


@pytest.mark.parametrize("family", ["poisson", "bernoulli", "gaussian", "softmax"])
def test_a_missing_unit_id_is_refused(family, rate_sessions, content_sessions) -> None:
    """A row whose unit id (an event id under the softmax) is missing means a
    broken table: fit_sessions refuses it, naming the session, the unit
    column and the rows, before any design is built."""
    from continuum_swr_replay.analysis.regression.likelihoods import (
        Bernoulli,
        Gaussian,
    )

    if family == "softmax":
        sessions, unit, response = dict(content_sessions), "event_id", "share"
        design = build_design(
            [ROLE], pd.concat(sessions.values(), ignore_index=True), intercept=False
        )
        lik = GroupedSoftmax()
    else:
        sessions = {
            k: r.assign(y=(r["count"] > 0) * 1.0) for k, r in rate_sessions.items()
        }
        unit, response = "well_visit_id", "count" if family == "poisson" else "y"
        terms = [Slope("x", "none")] + (
            [Offset("exposure")] if family == "poisson" else []
        )
        design = build_design(
            terms, pd.concat(sessions.values(), ignore_index=True), intercept=True
        )
        lik = {"poisson": Poisson(), "bernoulli": Bernoulli(), "gaussian": Gaussian()}[
            family
        ]
    key = list(sessions)[1]
    broken = sessions[key].astype({unit: float})
    broken.loc[[0, 3, 5], unit] = np.nan
    sessions[key] = broken
    name = "/".join(key)
    with pytest.raises(
        ValueError, match=rf"session {name}: 3 rows with a missing {unit}"
    ):
        _fit(design, sessions, lik, response=response, unit=unit)


def test_a_fit_not_kept_records_what_the_refit_left_out(monkeypatch) -> None:
    """A refit (scripted) that leaves x out in round 1 and then runs away
    along the intercept: the session has no kept fit, and its record lists
    what the refit had left out before it ended, with the round, as a record
    of what was tried; no rows are dropped for it."""
    from dataclasses import replace

    from continuum_swr_replay.analysis.regression import sessions as loop

    real = loop.refit_session

    def scripted(lik, data, blocks, rho, **kw):
        return replace(
            real(lik, data, blocks, rho, **kw),
            left_out=np.array([3]),
            rounds=np.array([1]),
        )

    monkeypatch.setattr(loop, "refit_session", scripted)
    frame = _empty_reference_rows()
    design = _rate_design(
        frame, Category("cat", ("A", "B", "C"), reference="A", penalty="none")
    )
    assert design.coef_names[3] == "x"
    (record,) = _fit(
        design,
        {("sim", "1", "02_r1"): frame},
        Poisson(),
        response="count",
        unit="visit",
    )
    assert record.status == "no finite maximum"
    assert record.no_fit == NoFit.ALONG_THE_INTERCEPT
    assert record.left_out == {"x": Removed.NO_FINITE_MAXIMUM}
    assert record.left_out_round == {"x": 1}
    assert record.cohort_counts["rows_dropped"] == {}
    assert record.n_rows == len(frame) and np.isnan(record.coef).all()


def test_a_term_that_is_not_a_level_drops_no_rows(rate_sessions) -> None:
    """A 0/1 slope on two visits fails the support rule and is made
    inactive; the category's levels pass. No row is dropped for the slope:
    the fit sees every row."""
    keys, frames = list(rate_sessions), list(rate_sessions.values())
    sessions = {}
    for key, rows in zip(keys[:2], frames[:2], strict=True):
        rows = _grouped_rate(rows, np.arange(0, 200, 2))
        sessions[key] = rows.assign(flag=rows["well_visit_id"].isin([0, 1]) * 1.0)
    pooled = pd.concat(sessions.values(), ignore_index=True)
    grp = Category("grp", ("a", "b"), reference="a", penalty="none")
    design = build_design(
        [grp, Slope("flag", "none"), Slope("x", "none"), Offset("exposure")],
        pooled,
        intercept=True,
    )
    record = _fit(design, sessions, Poisson(), response="count", unit="well_visit_id")[
        0
    ]
    assert record.left_out == {"flag": Removed.SUPPORT}
    assert record.cohort_counts["rows_dropped"] == {}
    assert record.n_rows == len(sessions[keys[0]])


@pytest.mark.slow
def test_the_declared_penalty_recovers_a_planted_curve() -> None:
    """30 simulated sessions (5 animals of 2 days of 3 sessions, about 120
    events each, real-sized visit bins) with a threefold rise over the first
    3 s after arrival: at the log penalty that gives a median edf of 6
    (``calibration.log_penalty_for_edf``), every session's fit converges and
    records its penalty as fixed, and the animal-balanced Q1 contrast (the
    curve 2 to 4 s after arrival less the first second) is the planted one
    less the penalty's shrinkage, to three SDs of the estimate over data sets
    of this size. ``contrast_vector`` itself is tested exactly in
    ``test_simulate_declared_penalty.py``."""
    config = sim.Config(days_per_animal=(2,) * 5, sessions_per_day=(3, 3))
    sessions = sim.cohort(config, seed=7)
    pooled = pd.concat(sessions.values(), ignore_index=True)
    data = draw_data_set(
        sim.model_at("rise", pooled, 0.0),
        sessions,
        sim.scenarios()[0],
        np.random.default_rng(8),
    )
    assert len(data) == 30
    found = log_penalty_for_edf(
        sim.design_at(pooled, 0.0),
        data,
        Poisson(),
        term=sim.TERM,
        target_edf=6.0,
        response="count",
        unit="cell",
    )
    design = sim.design_at(pooled, found.log_penalty)
    records = _fit(design, data, Poisson(), response="count", unit="cell")

    assert all(r.status == "converged" for r in records)
    assert all(
        r.log_penalty == {sim.TERM: {"set": "fixed", "log_penalty": found.log_penalty}}
        for r in records
    )
    assert np.median([r.edf[sim.TERM] for r in records]) == pytest.approx(6.0, abs=0.01)
    c = sim.contrast_vector(design)
    estimates = pd.Series(
        [
            evaluate_contrast(c, np.array(r.coef), np.array(r.vcov)).estimate
            for r in records
        ]
    )
    per_animal = animal_day_session_values(
        estimates, *(pd.Series([key[k] for key in sessions]) for k in range(3))
    )
    planted = sim.planted_contrast("rise")
    print(f"planted {planted:.4f}, estimate {per_animal.mean():.4f}")
    # Over 40 data sets of this cohort at this penalty (data seeds 100 to
    # 139), the estimate less the planted value had mean -0.055 (the
    # penalty's shrinkage) and SD 0.048: the bound is that mean to three SDs,
    # -0.20 to +0.09. This seed's is +0.020; with the curve at 0 in place of
    # the reference window (the mean over 0 to 1 s) it is +0.285.
    assert per_animal.mean() - planted == pytest.approx(-0.055, abs=0.145)


def _two_cells(n_low: tuple[float, float], n_high: tuple[float, float]) -> dict:
    """A rate session of 20 visits, one row each: visits 0-9 with ``flag`` 0,
    10-19 with ``flag`` 1. ``n_low`` and ``n_high`` are each side's (events,
    exposure), spread over its visits; the likelihood sees only the totals,
    so the fit is that of two cells."""
    events = np.zeros(20)
    events[0], events[10] = n_low[0], n_high[0]
    exposure = np.r_[np.full(10, n_low[1] / 10), np.full(10, n_high[1] / 10)]
    rows = pd.DataFrame(
        {
            "visit": np.arange(20),
            "flag": (np.arange(20) >= 10) * 1.0,
            "exposure": exposure,
            "count": events,
        }
    )
    return {("sim", "1", "02_r1"): rows}


def _two_cell_fit(sessions: dict, penalty) -> object:
    rows = next(iter(sessions.values()))
    design = build_design(
        [Slope("flag", penalty), Offset("exposure")], rows, intercept=True
    )
    (record,) = _fit(design, sessions, Poisson(), response="count", unit="visit")
    return record


def _penalized_two_cells(y0: float, e0: float, y1: float, e1: float, lam: float):
    """The two-cell model's penalized maximum, solved in one dimension.

    The score equations of ``-l + lam b**2 / 2`` with ``l`` the Poisson
    log-likelihood of the cells (``exp(a)`` on the 0 side, ``exp(a + b)`` on
    the 1 side) are ``y0 + y1 = e0 exp(a) + e1 exp(a + b)`` and ``y1 = e1
    exp(a + b) + lam b``. So ``e0 exp(a) = y0 + lam b``, and ``b`` is the root
    of ``g(b) = y1 - lam b - e1 exp(b) (y0 + lam b) / e0``, with
    ``y0 + lam b > 0``."""

    def g(b: float) -> float:
        return y1 - lam * b - e1 * np.exp(b) * (y0 + lam * b) / e0

    lo = -y0 / lam * (1 - 1e-12) if y0 > 0 else 1e-300
    lo = max(lo, -60.0)
    b = brentq(g, lo, 60.0, xtol=1e-14, rtol=1e-15)
    return np.log((y0 + lam * b) / e0), b


@pytest.mark.parametrize(
    "cells",
    [
        ((30.0, 12.0), (18.0, 5.0)),
        ((30.0, 12.0), (0.0, 5.0)),
        ((0.0, 12.0), (9.0, 5.0)),
    ],
    ids=["counts", "no_event_on_1", "no_event_on_0"],
)
def test_a_penalized_binary_slope_is_the_penalized_maximum(cells) -> None:
    """Two cells, an intercept and one penalized 0/1 coefficient: the fit
    equals the root of the penalized score equations to 1e-8, for ordinary
    counts and for no event on one side (a finite value, the term in the
    fit; the support rule reads rows, so it passes)."""
    (y0, e0), (y1, e1) = cells
    for rho in (-2.0, 0.0, 1.5):
        record = _two_cell_fit(_two_cells(*cells), rho)
        a, b = _penalized_two_cells(y0, e0, y1, e1, np.exp(rho))
        assert record.status == "converged" and record.left_out == {}
        assert record.support["flag"]["passes"]
        assert record.coef == pytest.approx([a, b], abs=1e-8)
        assert record.log_penalty == {"flag": {"set": "fixed", "log_penalty": rho}}
        assert 0.0 < record.edf["flag"] < 1.0


#: mgcv 1.9-4 (R 4.5.3, the ``csr-mgcv`` environment) on the two-cell data of
#: :func:`_two_cells`. mgcv's penalty on the deviance, ``sp b**2``, is the
#: fitter's ``exp(rho) b**2 / 2`` on the log-likelihood. (cells, rho) ->
#: (intercept, flag, edf of flag), as this R script printed them (rerun
#: 2026-11-23, the same digits)::
#:
#:     library(mgcv)
#:     fit <- function(y0, e0, y1, e1, rho) {
#:       d <- data.frame(y = c(y0, rep(0, 9), y1, rep(0, 9)),
#:                       e = c(rep(e0 / 10, 10), rep(e1 / 10, 10)),
#:                       flag = rep(c(0, 1), each = 10))
#:       m <- gam(y ~ flag + offset(log(e)), family = poisson, data = d,
#:                paraPen = list(flag = list(diag(1), sp = exp(rho))),
#:                control = gam.control(epsilon = 1e-14))
#:       cat(sprintf("%.12f %.12f %.10f\n", coef(m)[1], coef(m)[2], m$edf[2]))
#:     }
#:     fit(30, 12, 18, 5, -2.0)  # counts
#:     fit(30, 12, 18, 5, 1.5)   # counts
#:     fit(30, 12, 0, 5, 0.0)    # no_event_on_1
#:     fit(0, 12, 9, 5, 1.5)     # no_event_on_0
MGCV_PARAPEN = {
    ("counts", -2.0): (0.917914817698, 0.360306342817, 0.9881004000),
    ("counts", 1.5): (0.954357044809, 0.259724286112, 0.7092165189),
    ("no_event_on_1", 0.0): (0.852710720856, -1.848028978946, 0.6342607833),
    ("no_event_on_0", 1.5): (-1.024568629085, 0.961114260769, 0.3338322237),
}
TWO_CELLS = {
    "counts": ((30.0, 12.0), (18.0, 5.0)),
    "no_event_on_1": ((30.0, 12.0), (0.0, 5.0)),
    "no_event_on_0": ((0.0, 12.0), (9.0, 5.0)),
}


@pytest.mark.parametrize("case", list(MGCV_PARAPEN), ids=str)
def test_a_penalized_binary_slope_matches_mgcv(case) -> None:
    """Outside check: the intercept, the penalized 0/1 coefficient and its
    edf equal mgcv's ``paraPen`` fit at the same fixed strength, to 1e-8."""
    cells, rho = case
    record = _two_cell_fit(_two_cells(*TWO_CELLS[cells]), rho)
    a, b, edf = MGCV_PARAPEN[case]
    assert record.coef == pytest.approx([a, b], abs=1e-8)
    assert record.edf["flag"] == pytest.approx(edf, abs=1e-8)


def test_the_penalty_moves_the_estimate_toward_zero() -> None:
    """At the weakest log penalty allowed the estimate is the plain maximum
    likelihood ``log((y1 / e1) / (y0 / e0))``; a stronger penalty moves it
    toward zero, monotonically, and lowers its edf."""
    cells = ((30.0, 12.0), (18.0, 5.0))
    plain = np.log((18.0 / 5.0) / (30.0 / 12.0))
    weakest = _two_cell_fit(_two_cells(*cells), LOG_PENALTY_BOUNDS[0])
    assert weakest.coef[1] == pytest.approx(plain, abs=1e-4)
    assert weakest.edf["flag"] == pytest.approx(1.0, abs=1e-4)
    path = [
        _two_cell_fit(_two_cells(*cells), rho)
        for rho in (LOG_PENALTY_BOUNDS[0], -4.0, -1.0, 0.0, 2.0, 5.0, 10.0)
    ]
    b = np.array([r.coef[1] for r in path])
    edf = np.array([r.edf["flag"] for r in path])
    assert (b > 0).all() and (np.diff(b) < 0).all() and b[-1] < 1e-3
    assert (np.diff(edf) < 0).all()


def test_an_unpenalized_side_without_events_is_left_out() -> None:
    """The same session with the 0/1 term unpenalized: it passes the support
    rule (rows on both sides), but no event on one side means no finite
    maximum along it, and the existing rule leaves it out."""
    record = _two_cell_fit(_two_cells((30.0, 12.0), (0.0, 5.0)), "none")
    assert record.status == "converged"
    assert record.support["flag"]["passes"]
    assert record.left_out == {"flag": Removed.NO_FINITE_MAXIMUM}
    assert np.isnan(record.coef[1]) and record.edf["flag"] == 0.0


def test_an_unpenalized_level_with_one_event_is_fitted() -> None:
    """A rate session with a level ``b`` on 10 of 60 visits and one event
    there, the category unpenalized. The support rule counts rows, so the
    level passes (the event-based rule, which counted units with an event,
    left it out); its maximum is finite, so the refit keeps it. Its value
    and standard error are those of the two-cell maximum likelihood:
    ``log((y_b / e_b) / (y_a / e_a))`` and ``sqrt(1 / y_a + 1 / y_b)``."""
    rows = rate_rows(0)
    rows["g"] = np.where(rows["well_visit_id"] % 6 == 0, "b", "a")
    on_b = rows["g"] == "b"
    rows.loc[on_b, "count"] = 0.0
    rows.loc[rows.index[on_b][0], "count"] = 1.0
    design = build_design(
        [Category("g", ("a", "b"), reference="a", penalty="none"), Offset("exposure")],
        rows,
        intercept=True,
    )
    (record,) = _fit(
        design, by_session([rows]), Poisson(), response="count", unit="well_visit_id"
    )
    y_a, e_a = rows.loc[~on_b, ["count", "exposure"]].sum()
    y_b, e_b = rows.loc[on_b, ["count", "exposure"]].sum()
    assert record.status == "converged" and record.left_out == {}
    assert record.support["g[b]"] == {
        "n_low": 50,
        "n_high": 10,
        "passes": True,
        "reason": "",
        "size": None,
    }
    assert record.coef == pytest.approx(
        [np.log(y_a / e_a), np.log((y_b / e_b) / (y_a / e_a))], abs=1e-8
    )
    assert np.sqrt(record.vcov[1][1]) == pytest.approx(
        np.sqrt(1 / y_a + 1 / y_b), rel=1e-6
    )
    assert record.coef[1] == pytest.approx(-3.67, abs=0.005)
    assert np.sqrt(record.vcov[1][1]) == pytest.approx(1.0, abs=0.005)


def test_penalized_terms_record_their_edf(content_sessions) -> None:
    """A penalized slope, a penalized slope by a category and a penalized
    category under the grouped softmax: each penalized coefficient's edf is
    in the record and equals ``[(H + S)^-1 H]_jj``, with the information
    ``H = sum_g X_g' (diag(p_g) - p_g p_g') X_g`` (each event's shares sum
    to one) computed here from the record's coefficients and ``S = I`` (every
    log penalty 0), to 1e-8; the category's term sums its levels'."""
    pooled = pd.concat(content_sessions.values(), ignore_index=True)
    role = Category("role", ROLE.levels, reference="other", penalty=0.0)
    design = build_design(
        [
            role,
            Slope("reward_fraction", 0.0, by=role, levels=("current", "previous")),
            Slope("clock_s", 0.0),
        ],
        pooled,
        intercept=False,
    )
    records = _fit(
        design, content_sessions, GroupedSoftmax(), response="share", unit="event_id"
    )
    for rows, record in zip(content_sessions.values(), records, strict=True):
        assert record.status == "converged" and record.left_out == {}
        X, _ = design.session(rows)
        assert X.shape == (record.n_rows, len(design.coef_names))
        assert np.isfinite(X).all()
        eta = X @ np.array(record.coef)
        H = np.zeros((X.shape[1], X.shape[1]))
        for event in np.unique(rows["event_id"]):
            on = (rows["event_id"] == event).to_numpy()
            p = np.exp(eta[on] - eta[on].max())
            p /= p.sum()
            n = rows.loc[on, "share"].sum()
            H += n * X[on].T @ (np.diag(p) - np.outer(p, p)) @ X[on]
        influence = np.diag(np.linalg.solve(H + np.eye(X.shape[1]), H))
        by_name = dict(zip(record.coef_names, influence, strict=True))
        names = [
            "role[current]",
            "role[previous]",
            "role[trunk]",
            "reward_fraction:role=current",
            "reward_fraction:role=previous",
            "clock_s",
        ]
        assert set(record.coef_names) == set(names)
        for name in names:
            assert record.edf[name] == pytest.approx(by_name[name], abs=1e-8), name
        assert record.edf["role"] == pytest.approx(
            sum(
                record.edf[f"role[{level}]"]
                for level in ("current", "previous", "trunk")
            )
        )


def _gaussian_sessions() -> dict:
    """Two rate sessions with a Gaussian response, a three-level category
    ``g`` (by visit) and ``x_big``, a covariate of mean near 100 and SD near
    50."""
    rng = np.random.default_rng(1)
    frames = []
    for seed in range(2):
        rows = rate_rows(seed)
        visit = rows["well_visit_id"]
        rows["g"] = np.select([visit % 3 == 0, visit % 3 == 1], ["a", "b"], "c")
        rows["x_big"] = 50.0 * rows["x"] + 100.0
        rows["y"] = rng.normal(size=len(rows)) + 0.3 * rows["x"]
        frames.append(rows)
    return by_session(frames)


#: The log penalties of the Gaussian checks: the category's levels, the
#: plain slope of ``x_big``, the slope of ``x`` by ``g`` and the 0/1 slope.
RHO_G, RHO_BIG, RHO_BY, RHO_01 = 3.0, 1.3, -0.4, 2.0


def _gaussian_fit(sessions: dict):
    pooled = pd.concat(sessions.values(), ignore_index=True)
    g = Category("g", ("a", "b", "c"), reference="a", penalty=RHO_G)
    design = build_design(
        [
            g,
            Slope("x_big", RHO_BIG),
            Slope("x", RHO_BY, by=g, levels=("b", "c")),
            Slope("last_rewarded", RHO_01),
        ],
        pooled,
        intercept=True,
    )
    records = _fit(design, sessions, Gaussian(), response="y", unit="well_visit_id")
    return pooled, records


def _ridge_by_hand(rows: pd.DataFrame, pooled: pd.DataFrame):
    """The design built by hand in the reported unit, and its penalty:
    ``x_big`` divided by its pooled SD (not centered); ``x`` by ``g``
    centered at its pooled mean over the rows of levels b and c and divided
    by their pooled SD (ddof 0); the 0/1 slope as it is."""
    on_bc = pooled["g"].isin(["b", "c"])
    mean_x, sd_x = pooled.loc[on_bc, "x"].mean(), pooled.loc[on_bc, "x"].std(ddof=0)
    is_b = (rows["g"] == "b").to_numpy(float)
    is_c = (rows["g"] == "c").to_numpy(float)
    centered = (rows["x"].to_numpy() - mean_x) / sd_x
    X = np.column_stack(
        [
            np.ones(len(rows)),
            is_b,
            is_c,
            rows["x_big"].to_numpy() / pooled["x_big"].std(ddof=0),
            is_b * centered,
            is_c * centered,
            rows["last_rewarded"].to_numpy(),
        ]
    )
    lam = np.exp([-np.inf, RHO_G, RHO_G, RHO_BIG, RHO_BY, RHO_BY, RHO_01])
    return X, np.diag(lam)


def test_a_gaussian_penalized_fit_is_the_ridge_solution() -> None:
    """The penalty acts in the reported unit: under the Gaussian, a penalized
    category, a penalized slope of a covariate far from mean 0 and SD 1, a
    penalized slope by the category and a penalized 0/1 slope equal the
    ridge solution ``(X'X + diag(exp(rho))) b = X'y`` with ``X`` built by
    hand (pooled SD, by-category centering), to 1e-10."""
    sessions = _gaussian_sessions()
    pooled, records = _gaussian_fit(sessions)
    assert pooled["x_big"].mean() > 50 and pooled["x_big"].std(ddof=0) > 40
    for rows, record in zip(sessions.values(), records, strict=True):
        X, S = _ridge_by_hand(rows, pooled)
        b = np.linalg.solve(X.T @ X + S, X.T @ rows["y"].to_numpy())
        assert record.status == "converged" and record.left_out == {}
        assert record.coef_names == [
            "(Intercept)",
            "g[b]",
            "g[c]",
            "x_big",
            "x:g=b",
            "x:g=c",
            "last_rewarded",
        ]
        assert np.abs(np.array(record.coef) - b).max() < 1e-10


def test_a_penalized_category_edf_is_the_sum_of_its_levels() -> None:
    """The record's ``edf`` holds each level of a penalized category by its
    coefficient name and their sum by the category's name (a total over all
    entries would count the levels twice). Each level's edf, computed from
    the fit alone, is the diagonal element of the influence of the penalized
    information, ``[(X'X + S)^-1 X'X]_jj`` under the Gaussian; the
    category's is the sum of its levels', to 1e-10."""
    sessions = _gaussian_sessions()
    pooled, records = _gaussian_fit(sessions)
    for rows, record in zip(sessions.values(), records, strict=True):
        X, S = _ridge_by_hand(rows, pooled)
        gram = X.T @ X
        influence = np.diag(np.linalg.solve(gram + S, gram))
        by_name = dict(zip(record.coef_names, influence, strict=True))
        for name in ("g[b]", "g[c]", "x_big", "x:g=b", "x:g=c", "last_rewarded"):
            assert record.edf[name] == pytest.approx(by_name[name], abs=1e-10)
        assert record.edf["g"] == pytest.approx(
            by_name["g[b]"] + by_name["g[c]"], abs=1e-10
        )
        assert 0.5 < by_name["g[b]"] < 0.99


def _penalized_poisson_by_hand(X, y, log_offset, lam) -> np.ndarray:
    """The maximum of ``l - sum(lam b**2) / 2`` (Poisson with a log offset)
    by plain Newton steps, written here apart from the fitter."""
    b = np.zeros(X.shape[1])
    for _ in range(100):
        mu = np.exp(X @ b + log_offset)
        step = np.linalg.solve(
            X.T @ (mu[:, None] * X) + np.diag(lam), X.T @ (y - mu) - lam * b
        )
        b += step
        if np.abs(step).max() < 1e-14:
            break
    return b


def test_a_penalized_level_that_fails_support_leaves_with_its_dependents() -> None:
    """A penalized category ``g`` whose level ``c`` has rows of only 3 visits
    in one session, with penalized slopes by ``g`` (one on ``b`` and ``c``,
    one on ``a`` and ``c``) and two penalized plain slopes. In that session
    the level fails support, its rows are dropped and the slopes for ``c``
    leave with it (edf 0, coefficient NaN). The rest of the fit equals a
    penalized Poisson fit made here without ``c``'s rows and columns, each
    column at its declared log penalty, to 1e-8: the blocks of the dropped
    columns are removed and the other log penalties stay aligned."""
    frames = [rate_rows(seed) for seed in range(3)]
    for frame in frames:
        visit = frame["well_visit_id"]
        frame["g"] = np.select([visit % 3 == 0, visit % 3 == 1], ["a", "b"], "c")
    first = frames[0]
    first["g"] = np.where(first["well_visit_id"] % 2 == 0, "a", "b")
    first.loc[first["well_visit_id"].isin([2, 14, 26]), "g"] = "c"
    sessions = by_session(frames)
    pooled = pd.concat(sessions.values(), ignore_index=True)
    rho = {
        "g[b]": 0.7,
        "g[c]": 0.7,
        "x": 1.3,
        "last_rewarded:g=b": -0.4,
        "last_rewarded:g=c": -0.4,
        "time_since_s:g=a": 2.0,
        "time_since_s:g=c": 2.0,
        "time_until_s": 0.5,
    }
    g = Category("g", ("a", "b", "c"), reference="a", penalty=0.7)
    design = build_design(
        [
            g,
            Slope("x", 1.3),
            Slope("last_rewarded", -0.4, by=g, levels=("b", "c")),
            Slope("time_since_s", 2.0, by=g, levels=("a", "c")),
            Slope("time_until_s", 0.5),
            Offset("exposure"),
        ],
        pooled,
        intercept=True,
    )
    records = _fit(design, sessions, Poisson(), response="count", unit="well_visit_id")
    record = records[0]
    first = next(iter(sessions.values()))
    on_c = (first["g"] == "c").to_numpy()
    assert record.status == "converged"
    assert record.support["g[c]"]["n_high"] == 3
    assert record.left_out == {
        "g[c]": Removed.SUPPORT,
        "last_rewarded:g=c": Removed.WITH_LEVEL,
        "time_since_s:g=c": Removed.WITH_LEVEL,
    }
    assert record.cohort_counts["rows_dropped"] == {"g[c]": int(on_c.sum())}
    assert record.n_rows == int((~on_c).sum())
    for name in record.left_out:
        assert record.edf[name] == 0.0
        assert np.isnan(record.coef[record.coef_names.index(name)])
    assert record.edf["g"] == pytest.approx(record.edf["g[b]"])
    j_c = record.coef_names.index("g[c]")
    assert all(np.isfinite(r.coef[j_c]) for r in records[1:])  # c on 20 visits

    X, _ = design.session(first)
    keep = [j for j, n in enumerate(design.coef_names) if n not in record.left_out]
    names = [design.coef_names[j] for j in keep]
    lam = np.array([0.0 if n == "(Intercept)" else np.exp(rho[n]) for n in names])
    b = _penalized_poisson_by_hand(
        X[~on_c][:, keep],
        first["count"].to_numpy()[~on_c],
        np.log(first["exposure"].to_numpy()[~on_c]),
        lam,
    )
    assert np.abs(np.array(record.coef)[keep] - b).max() < 1e-8


def test_every_sparse_session_has_its_penalized_term() -> None:
    """Six simulated rate sessions in which the 0/1 term is 1 on 6 of 60
    visits and has few events there (none in some sessions): with a weak
    penalty every session has a finite value for the term, and none leaves
    it out for support."""
    rng = np.random.default_rng(3)
    frames = []
    for seed in range(6):
        visit = np.repeat(np.arange(60), 10)
        flag = (visit < 6) * 1.0
        exposure = np.full(visit.size, 0.05)
        rate = np.exp(np.log(2.0) + 0.5 * flag)
        count = rng.poisson(rate * exposure).astype(float)
        if seed in (1, 4):
            count[flag == 1] = 0.0
        frames.append(
            pd.DataFrame(
                {
                    "animal": "sim",
                    "session_date": f"2026{seed:04d}",
                    "epoch": "02_r1",
                    "visit": visit,
                    "flag": flag,
                    "exposure": exposure,
                    "count": count,
                }
            )
        )
    sessions = by_session(frames)
    pooled = pd.concat(sessions.values(), ignore_index=True)
    design = build_design(
        [Slope("flag", log_penalty_from_prior_sd(2.0)), Offset("exposure")],
        pooled,
        intercept=True,
    )
    records = _fit(design, sessions, Poisson(), response="count", unit="visit")
    for record in records:
        assert record.status == "converged"
        assert record.left_out == {}
        assert record.support["flag"] == {
            "n_low": 54,
            "n_high": 6,
            "passes": True,
            "reason": "",
            "size": None,
        }
        assert np.isfinite(record.coef[1])
    assert any(f.loc[f["flag"] == 1, "count"].sum() == 0 for f in frames)


def test_a_penalized_term_without_rows_on_a_side_fails_its_support() -> None:
    """The support rule holds for a penalized term too: a 0/1 term with rows
    of 3 visits on its 1 side fails it, is inactive in the fit (its block is
    not fitted), and the session's fit converges."""
    sessions = _two_cells((30.0, 12.0), (18.0, 5.0))
    rows = next(iter(sessions.values()))
    rows["flag"] = (rows["visit"] >= 17) * 1.0
    record = _two_cell_fit(sessions, 0.0)
    assert record.status == "converged"
    assert record.left_out == {"flag": Removed.SUPPORT}
    assert record.support["flag"]["n_high"] == 3
    assert np.isnan(record.coef[1]) and record.edf["flag"] == 0.0


# --- the unit of the fit: grouping, the record, baselines --------------------------


def _day_sessions(n: int = 2, day: str = "20260101", animal: str = "sim") -> dict:
    """``n`` rate sessions of one recording day (seeds 0 to n - 1): every
    session's visit ids start at 0."""
    return cohort(
        rate_rows,
        animals=[animal],
        days=[day],
        epochs=[f"0{2 * s + 2}_r{s + 1}" for s in range(n)],
        seed=lambda a, d, e: e,
    )


def _rate_fit(design, sessions, **kw):
    """The records of a rate fit (Poisson; ``count`` of ``well_visit_id``)."""
    return _fit(
        design, sessions, Poisson(), response="count", unit="well_visit_id", **kw
    )


def _small_rate_design(pooled, baseline=()):
    return build_design(
        [Slope("last_rewarded", log_penalty_from_prior_sd(2.0)), Offset("exposure")],
        pooled,
        intercept=True,
        baseline=baseline,
    )


def test_visits_of_two_sessions_are_two_units() -> None:
    """Two sessions of one day whose visit ids are the same numbers: at the
    day unit the units and the support counts are the sums of the two
    sessions' own."""
    sessions = _day_sessions()
    design = _small_rate_design(
        pd.concat(sessions.values(), ignore_index=True), baseline=("(Intercept)",)
    )
    alone = _rate_fit(design, sessions)
    (day,) = _rate_fit(design, sessions, fit_by="day")
    assert day.n_units == sum(r.n_units for r in alone) == 120
    for side in ("n_low", "n_high"):
        assert day.support["last_rewarded"][side] == sum(
            r.support["last_rewarded"][side] for r in alone
        )


def test_a_larger_unit_needs_a_baseline() -> None:
    """At a day or an animal unit a design without a baseline would fit one
    intercept for all the unit's sessions: refused at entry. The equivalence
    of a unit's fit with a design built by hand is in
    ``test_unit_fit_equals_a_hand_built_indicator_design`` (Poisson) and
    ``test_softmax_unit_fit_equals_an_independent_maximization``."""
    sessions = _day_sessions(2)
    design = _small_rate_design(pd.concat(sessions.values(), ignore_index=True))
    for fit_by in ("day", "animal"):
        with pytest.raises(
            ValueError, match=r"build_design\(\.\.\., baseline=\.\.\.\)"
        ):
            _rate_fit(design, sessions, fit_by=fit_by)
    assert len(_rate_fit(design, sessions)) == 2


def test_record_holds_its_sessions() -> None:
    """``sessions`` lists each session of the fit, in order, with the rows,
    units and events the fit saw of it and its rows removed; the counts sum
    to the record's. At the animal unit the key is the animal alone."""
    sessions = {
        **_day_sessions(2, "20260101"),
        **_day_sessions(1, "20260102"),
    }
    first = next(iter(sessions))
    sessions[first] = sessions[first].assign(
        last_rewarded=sessions[first]["last_rewarded"].where(sessions[first].index >= 7)
    )
    pooled = pd.concat(sessions.values(), ignore_index=True).dropna()
    design = _small_rate_design(pooled, baseline=("(Intercept)",))
    (record,) = _rate_fit(design, sessions, fit_by="animal")
    assert (record.animal, record.session_date, record.epoch) == ("sim", "", "")
    assert [
        (e["animal"], e["session_date"], e["epoch"]) for e in record.sessions
    ] == list(sessions)
    for entry, rows in zip(record.sessions, sessions.values(), strict=True):
        missing = 7 if rows is sessions[first] else 0
        assert entry["rows_missing_values"] == missing
        assert entry["n_rows"] == len(rows) - missing
        assert entry["n_events"] == rows["count"].to_numpy()[missing:].sum()
        assert entry["rows_dropped"] == {}
    for field in ("n_rows", "n_units", "n_events"):
        assert sum(e[field] for e in record.sessions) == getattr(record, field)
    assert record.cohort_counts["rows_missing_values"] == 7
    # Each session's baseline copies, as data.
    assert [e["baseline"] for e in record.sessions] == [
        [f"(Intercept)@{'/'.join(key)}"] for key in sessions
    ]


def test_units_in_the_order_of_their_first_session() -> None:
    sessions = {
        ("b", "20260101", "02_r1"): rate_rows(0),
        ("a", "20260101", "02_r1"): rate_rows(1),
        ("b", "20260102", "02_r1"): rate_rows(2),
    }
    design = _small_rate_design(
        pd.concat(sessions.values(), ignore_index=True), baseline=("(Intercept)",)
    )
    records = _rate_fit(design, sessions, fit_by="animal")
    assert [r.animal for r in records] == ["b", "a"]
    assert [len(r.sessions) for r in records] == [2, 1]
    with pytest.raises(ValueError, match="fit_by"):
        _rate_fit(design, sessions, fit_by="week")


def test_rate_baselines_match_separate_intercepts() -> None:
    """Poisson: k sessions sharing only a slope. Each session's intercept is
    the root of its own score equation at the shared slope, ``log(sum y /
    sum exposure exp(b x))`` on its rows, to 1e-8."""
    sessions = _day_sessions(4)
    design = _small_rate_design(
        pd.concat(sessions.values(), ignore_index=True), baseline=("(Intercept)",)
    )
    (record,) = _rate_fit(design, sessions, fit_by="day")
    assert record.status == "converged"
    coef = dict(zip(record.coef_names, record.coef, strict=True))
    b = coef["last_rewarded"]
    for key, rows in sessions.items():
        by_hand = np.log(
            rows["count"].sum()
            / (rows["exposure"] * np.exp(b * rows["last_rewarded"])).sum()
        )
        assert coef[f"(Intercept)@{'/'.join(key)}"] == pytest.approx(by_hand, abs=1e-8)
    assert "(Intercept)" not in record.edf and set(record.edf) == {"last_rewarded"}
    assert set(record.collinearity) == {"last_rewarded"}


def test_unit_fit_equals_a_hand_built_indicator_design() -> None:
    """The baseline copies are the indicator columns built by hand: the fit
    of one session-indicator column per session and the shared slope, by
    ``fit_at_penalties``, gives the same coefficients and covariance, to
    1e-8."""
    sessions = _day_sessions(3)
    pooled = pd.concat(sessions.values(), ignore_index=True)
    design = _small_rate_design(pooled, baseline=("(Intercept)",))
    (record,) = _rate_fit(design, sessions, fit_by="day")
    sizes = [len(rows) for rows in sessions.values()]
    session = np.repeat(np.arange(3), sizes)
    X = np.column_stack(
        [*(session == s for s in range(3)), pooled["last_rewarded"].to_numpy()]
    ).astype(float)
    data = SessionData(
        X, pooled["count"].to_numpy(), np.log(pooled["exposure"].to_numpy())
    )
    block = design.blocks[0]
    by_hand = fit_at_penalties(
        Poisson(),
        [data],
        [type(block)(np.array([3]), block.S)],
        design.log_penalties,
    ).sessions[0]
    np.testing.assert_allclose(record.coef, by_hand.fit.beta, rtol=0, atol=1e-8)
    np.testing.assert_allclose(record.vcov, by_hand.covariance, rtol=0, atol=1e-8)


def test_session_without_support_dropped_and_counted() -> None:
    """A session with three visits: its intercept copy fails the support rule
    (fewer than five units), so its rows leave the unit's fit, counted under
    the copy's name, and the other sessions are fitted."""
    sessions = _day_sessions(3)
    thin = list(sessions)[1]
    sessions[thin] = sessions[thin][sessions[thin]["well_visit_id"] < 3]
    design = _small_rate_design(
        pd.concat(sessions.values(), ignore_index=True), baseline=("(Intercept)",)
    )
    (record,) = _rate_fit(design, sessions, fit_by="day")
    copy = f"(Intercept)@{'/'.join(thin)}"
    assert record.status == "converged"
    assert record.left_out == {copy: Removed.SUPPORT}
    assert record.support[copy]["n_high"] == 3
    assert record.cohort_counts["rows_dropped"] == {copy: len(sessions[thin])}
    entry = record.sessions[1]
    assert entry["n_rows"] == 0 and entry["rows_dropped"] == {copy: len(sessions[thin])}
    assert record.n_rows == sum(len(r) for k, r in sessions.items() if k != thin)
    assert np.isnan(record.coef[record.coef_names.index(copy)])


def test_a_session_without_events_is_dropped_from_the_unit() -> None:
    """A session with no event: its intercept copy has no finite maximum, the
    refit leaves it out and its rows are dropped, counted; at the session unit
    the same session is not fitted."""
    sessions = _day_sessions(3)
    empty = list(sessions)[2]
    sessions[empty] = sessions[empty].assign(count=0.0)
    design = _small_rate_design(
        pd.concat(sessions.values(), ignore_index=True), baseline=("(Intercept)",)
    )
    (record,) = _rate_fit(design, sessions, fit_by="day")
    copy = f"(Intercept)@{'/'.join(empty)}"
    assert record.status == "converged"
    assert record.left_out[copy] == Removed.NO_FINITE_MAXIMUM
    assert record.cohort_counts["rows_dropped"][copy] == len(sessions[empty])
    alone = _rate_fit(design, sessions)
    assert alone[2].no_fit == NoFit.NO_EVENTS


def _arm_sessions(n: int = 3, n_events: int = 60, n_arms: int = 4) -> dict:
    """Content sessions of one day: each event has a row per arm, a share per
    row and a 0/1 column ``flag``."""
    out = {}
    for s in range(n):
        rng = np.random.default_rng(50 + s)
        arm = np.tile(np.arange(n_arms), n_events)
        event = np.repeat(np.arange(n_events), n_arms) + 1000 * s
        flag = (rng.uniform(size=arm.size) < 0.3).astype(float)
        eta = 0.4 * arm * (s - 1) + 0.6 * flag
        w = np.exp(eta).reshape(n_events, n_arms)
        share = np.concatenate([rng.dirichlet(10 * r / r.sum()) for r in w])
        key = ("sim", "20260101", f"0{2 * s + 2}_r1")
        out[key] = pd.DataFrame(
            {"event_id": event, "arm": arm, "flag": flag, "share": share}
        )
    return out


def _arm_design(pooled):
    return build_design(
        [
            Category("arm", (0, 1, 2, 3), reference=0, penalty="none"),
            Slope("flag", log_penalty_from_prior_sd(2.0)),
        ],
        pooled,
        intercept=False,
        baseline=("arm",),
    )


def test_reference_level_per_session_copy() -> None:
    """A session whose rows lack an arm the others have: its copy of that arm
    fails the support rule with no rows, and the unit fits. Each session keeps
    the reference arm 0."""
    sessions = _arm_sessions()
    last = list(sessions)[2]
    sessions[last] = sessions[last][sessions[last]["arm"] != 3]
    design = _arm_design(pd.concat(sessions.values(), ignore_index=True))
    (record,) = _fit(
        design, sessions, GroupedSoftmax(), response="share", unit="event_id",
        fit_by="day",
    )  # fmt: skip
    copy = f"arm[3]@{'/'.join(last)}"
    assert record.status == "converged"
    assert record.left_out == {copy: Removed.SUPPORT}
    assert record.support[copy]["n_high"] == 0
    assert record.sessions[2]["baseline"] == [
        f"arm[{a}]@{'/'.join(last)}" for a in (1, 2, 3)
    ]
    assert set(record.collinearity) == {"flag"} and set(record.edf) == {"flag"}


def test_a_session_without_the_reference_arm_is_an_error_naming_the_fit() -> None:
    """A baseline category's reference level must have rows in every session
    of the fit: the copies of a session without its reference arm are not
    identified, and the fit is refused, naming the fit, the session and those
    copies."""
    sessions = _arm_sessions()
    middle = list(sessions)[1]
    sessions[middle] = sessions[middle][sessions[middle]["arm"] != 0]
    design = _arm_design(pd.concat(sessions.values(), ignore_index=True))
    with pytest.raises(
        ValueError,
        match=r"day sim/20260101: the category 'arm' has no rows of its reference "
        r"level 0 in session sim/20260101/04_r1.*arm\[1\]@sim/20260101/04_r1",
    ):
        _fit(
            design, sessions, GroupedSoftmax(), response="share", unit="event_id",
            fit_by="day",
        )  # fmt: skip


def _weighted_r2(W: np.ndarray, projected: list[int], term: int, others: list[int]):
    """The R^2 of column ``term`` of ``W`` on ``others`` after the columns
    ``projected`` are projected out by one QR: the collinearity record's
    measure, computed directly."""
    Q = np.linalg.qr(W[:, projected])[0]
    P = W - Q @ (Q.T @ W)
    B = np.linalg.qr(P[:, others])[0]
    a = P[:, term]
    return float((B.T @ a) @ (B.T @ a) / (a @ a))


def test_a_shared_intercept_with_per_session_category_copies() -> None:
    """Poisson, three sessions: a shared intercept, a category copied for
    each session, a weakly penalized slope and an unpenalized one. The fit
    equals the design built by hand, to 1e-8; the edf and the collinearity
    record hold only the shared slopes, the collinearity with the intercept
    and every copy projected out; each session lists its two copies."""
    sessions = {}
    for key, rows in _day_sessions(3).items():
        visit = rows["well_visit_id"].to_numpy()
        sessions[key] = rows.assign(cat=np.array(["A", "B", "C"])[visit % 3])
    pooled = pd.concat(sessions.values(), ignore_index=True)
    weak = log_penalty_from_prior_sd(2.0)
    design = build_design(
        [
            Category("cat", ("A", "B", "C"), reference="A", penalty="none"),
            Slope("last_rewarded", weak),
            Slope("x", "none"),
            Offset("exposure"),
        ],
        pooled,
        intercept=True,
        baseline=("cat",),
    )
    (record,) = _rate_fit(design, sessions, fit_by="day")
    assert record.status == "converged"
    keys = ["/".join(k) for k in sessions]
    assert record.coef_names == [
        "(Intercept)",
        *(f"cat[B]@{k}" for k in keys),
        *(f"cat[C]@{k}" for k in keys),
        "last_rewarded",
        "x",
    ]
    session = np.repeat(np.arange(3), [len(r) for r in sessions.values()])
    cat = pooled["cat"].to_numpy()
    X = np.column_stack(
        [
            np.ones(len(pooled)),
            *((cat == "B") & (session == s) for s in range(3)),
            *((cat == "C") & (session == s) for s in range(3)),
            pooled["last_rewarded"].to_numpy(),
            pooled["x"].to_numpy() / pooled["x"].std(ddof=0),
        ]
    ).astype(float)
    y, offset = pooled["count"].to_numpy(), np.log(pooled["exposure"].to_numpy())
    block = design.blocks[0]
    by_hand = fit_at_penalties(
        Poisson(),
        [SessionData(X, y, offset)],
        [type(block)(np.array([7]), block.S)],
        design.log_penalties,
    ).sessions[0]
    np.testing.assert_allclose(record.coef, by_hand.fit.beta, rtol=0, atol=1e-8)
    np.testing.assert_allclose(record.vcov, by_hand.covariance, rtol=0, atol=1e-8)
    assert set(record.edf) == {"last_rewarded", "x"} and record.edf["x"] == 1.0
    assert set(record.collinearity) == {"last_rewarded", "x"}
    W = X * np.sqrt(np.exp(X @ np.asarray(record.coef) + offset))[:, None]
    projected = list(range(7))
    assert record.collinearity["last_rewarded"]["r2"] == pytest.approx(
        _weighted_r2(W, projected, 7, [8]), abs=1e-10
    )
    assert [e["baseline"] for e in record.sessions] == [
        [f"cat[B]@{k}", f"cat[C]@{k}"] for k in keys
    ]


def _same_event_ids(n_sessions: int = 3, n_arms: int = 4) -> dict:
    """Content sessions whose event ids are the same numbers (0, 1, ...) in
    every session, rows shuffled; per-session arm effects and a 0/1 flag. In
    the second session one event has no response."""
    out = {}
    for s in range(n_sessions):
        rng = np.random.default_rng(70 + s)
        n_events = 30 + 10 * s
        arm = np.tile(np.arange(n_arms), n_events)
        event = np.repeat(np.arange(n_events), n_arms)
        flag = (rng.uniform(size=arm.size) < 0.4).astype(float)
        eta = np.array([0.0, 0.5, -0.3, 0.2])[arm] * (s + 1) + 0.7 * flag
        w = np.exp(eta).reshape(n_events, n_arms)
        share = np.concatenate([rng.dirichlet(8 * r / r.sum()) for r in w])
        if s == 1:
            share[event == 3] = 0.0
        key = ("sim", "20260101", f"0{2 * s + 2}_r1")
        frame = pd.DataFrame(
            {"event_id": event, "arm": arm, "flag": flag, "share": share}
        )
        out[key] = frame.iloc[rng.permutation(arm.size)].reset_index(drop=True)
    return out


def test_softmax_unit_fit_equals_an_independent_maximization() -> None:
    """The grouped softmax with the same event ids in every session: each
    session's events are its own groups. The unit's coefficients and
    covariance equal a direct maximization of the multinomial log-likelihood
    over per-session groups, to 1e-8; the events count once per session; the
    per-session event counts sum to the unit's."""
    from scipy.optimize import minimize

    sessions = _same_event_ids()
    weak = log_penalty_from_prior_sd(2.0)
    design = build_design(
        [
            Category("arm", (0, 1, 2, 3), reference=0, penalty="none"),
            Slope("flag", weak),
        ],
        pd.concat(sessions.values(), ignore_index=True),
        intercept=False,
        baseline=("arm",),
    )
    (record,) = _fit(
        design, sessions, GroupedSoftmax(), response="share", unit="event_id",
        fit_by="day",
    )  # fmt: skip
    assert record.status == "converged"
    Xs, ys, gs, start = [], [], [], 0
    for s, frame in enumerate(sessions.values()):
        frame = frame.sort_values("event_id", kind="stable")
        cols = np.zeros((len(frame), 10))
        for a in (1, 2, 3):
            cols[:, (a - 1) * 3 + s] = frame["arm"].to_numpy() == a
        cols[:, -1] = frame["flag"].to_numpy()
        Xs.append(cols)
        ys.append(frame["share"].to_numpy())
        gs.append(frame["event_id"].to_numpy() + start)
        start += frame["event_id"].max() + 1
    X, y, g = np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)
    lam, n_groups = float(np.exp(weak)), g.max() + 1
    total = np.bincount(g, weights=y)

    def parts(b):
        eta = X @ b
        top = np.full(n_groups, -np.inf)
        np.maximum.at(top, g, eta)
        e = np.exp(eta - top[g])
        sums = np.bincount(g, weights=e)
        return eta, e / sums[g], np.log(sums) + top

    def objective(b):
        eta, p, lse = parts(b)
        value = -(y @ eta - total @ lse - 0.5 * lam * b[-1] ** 2)
        gradient = -(X.T @ (y - total[g] * p))
        gradient[-1] += lam * b[-1]
        return value, gradient

    b = minimize(objective, np.zeros(10), jac=True, method="BFGS",
                 options={"gtol": 1e-12, "maxiter": 10_000}).x  # fmt: skip
    for _ in range(20):
        _, p, _ = parts(b)
        H = np.zeros((10, 10))
        for k in range(n_groups):
            rows = g == k
            Xg, pg = X[rows], p[rows]
            H += total[k] * (Xg.T @ (Xg * pg[:, None]) - np.outer(Xg.T @ pg, Xg.T @ pg))
        H[-1, -1] += lam
        b = b - np.linalg.solve(H, objective(b)[1])
    names = [
        f"arm[{a}]@sim/20260101/0{2 * s + 2}_r1" for a in (1, 2, 3) for s in range(3)
    ]
    order = [record.coef_names.index(n) for n in [*names, "flag"]]
    np.testing.assert_allclose(np.asarray(record.coef)[order], b, rtol=0, atol=1e-8)
    np.testing.assert_allclose(
        np.asarray(record.vcov)[np.ix_(order, order)],
        np.linalg.inv(H),
        rtol=0,
        atol=1e-8,
    )
    assert record.n_units == 30 + 40 + 50 - 1 == record.n_events
    for field in ("n_units", "n_events", "events_without_response"):
        got = [e[field] for e in record.sessions]
        unit = getattr(record, field, record.cohort_counts.get(field))
        assert sum(got) == unit, field
    assert [e["events_without_response"] for e in record.sessions] == [0, 1, 0]


@pytest.mark.parametrize("cause", ["no_events", "support"])
@pytest.mark.parametrize("place", ["first", "last"])
def test_a_dropped_session_leaves_the_shared_fit_unchanged(cause, place) -> None:
    """A session dropped from the unit (no event, or three visits) leaves the
    shared coefficients those of the fit without it, to the last digits,
    whether it is the first or the last of the unit."""
    sessions = _day_sessions(4)
    keys = list(sessions)
    rows = sessions[keys[1]]
    dropped = (
        rows.assign(count=0.0)
        if cause == "no_events"
        else rows[rows["well_visit_id"] < 3]
    )
    others = {k: sessions[k] for k in keys if k != keys[1]}
    design = build_design(
        [
            Slope("last_rewarded", log_penalty_from_prior_sd(2.0)),
            Slope("x", "none"),
            Smooth("time_since_s", 0.0, 20.0, 8, 2.0),
            Offset("exposure"),
        ],
        pd.concat(sessions.values(), ignore_index=True),
        intercept=True,
        baseline=("(Intercept)",),
    )
    given = (
        {keys[1]: dropped, **others}
        if place == "first"
        else {**others, keys[1]: dropped}
    )
    (without,) = _rate_fit(design, others, fit_by="day")
    (with_it,) = _rate_fit(design, given, fit_by="day")

    def shared(record):
        names = zip(record.coef_names, record.coef, strict=True)
        return {n: c for n, c in names if "@" not in n}

    # Not bit for bit: with the session's rows dropped the fit reads a copy of
    # the kept rows, and a threaded linear-algebra library (OpenBLAS on four
    # threads, as on CI) may sum a matrix product over it in another order,
    # which moves the last digit. A fit that kept the session's rows differs in
    # the first digits.
    got, expected = shared(with_it), shared(without)
    assert list(got) == list(expected)
    np.testing.assert_allclose(
        list(got.values()), list(expected.values()), rtol=1e-10, atol=1e-12
    )
    copy = f"(Intercept)@{'/'.join(keys[1])}"
    assert with_it.cohort_counts["rows_dropped"] == {copy: len(dropped)}
