"""The per-session fitting loop: support, sessions without a finite maximum, records."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.regression.calibration import log_penalty_for_edf
from continuum_swr_replay.analysis.regression.contrasts import evaluate_contrast
from continuum_swr_replay.analysis.regression.likelihoods import GroupedSoftmax, Poisson
from continuum_swr_replay.analysis.regression.record import NoFit
from continuum_swr_replay.analysis.regression.reml import SessionData, fit_at_penalties
from continuum_swr_replay.analysis.regression.sessions import fit_sessions
from continuum_swr_replay.analysis.regression.support import Removed
from continuum_swr_replay.analysis.regression.terms import (
    SMALLEST_COLUMN_SIZE,
    Category,
    Offset,
    Slope,
    Smooth,
    build_design,
)
from continuum_swr_replay.analysis.shared_statistics import animal_day_session_values
from scripts.checks import simulate_declared_penalty as sim

from .conftest import ROLE, by_session

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
        [*categories, Slope("x"), Offset("exposure")], pooled, intercept=True
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
        [Smooth("clock", 0.0, 5.0, 10, 2.0), Slope("a"), Slope("b")],
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
            Slope("reward_fraction", by=ROLE, levels=("current", "previous", "other")),
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
            Slope("last_rewarded"),
            Slope("x"),
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
            "s(time_since_s)": {"set": "fixed", "log_penalty": 2.0}
        }
        assert set(record.standardization) == {"x"}
        assert record.cohort_counts["rows_missing_values"] == 0


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
        [Slope("x"), Slope("x_copy"), Offset("exposure")], pooled, intercept=True
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
    grp = Category("grp", ("a", "b"), reference="a")
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
        [ROLE, Slope("flag"), Smooth("clock_s", 0.0, 20.0, 8, 4.0)],
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
    grp = Category("grp", ("a", "b"), reference="a")
    design = _rate_design(pooled, grp)
    first, second = _fit(
        design, sessions, Poisson(), response="count", unit="well_visit_id"
    )
    assert first.status == "not fitted" and first.no_fit == NoFit.NO_EVENTS
    # The slope has its events on the level's rows only, so it fails too.
    assert first.left_out == {"grp[b]": Removed.SUPPORT, "x": Removed.SUPPORT}
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
            Slope("reward_fraction", by=ROLE, levels=("current", "previous", "other")),
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
    and the levels' columns are collinear, and the error names the session and
    those coefficients (the reference level must have rows in every
    session)."""
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
    grp = Category("grp", ("a", "b", "c"), reference="a")
    design = _rate_design(pooled, grp)
    message = (
        r"session sim/20260000/02_r1: at its first fit its design is not identified"
        r".*\(Intercept\), grp\[b\], grp\[c\]"
    )
    with pytest.raises(ValueError, match=message):
        _fit(design, sessions, Poisson(), response="count", unit="well_visit_id")


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
    cat = Category("cat", ("A", "B", "C"), reference="A")
    terms = [cat, Slope("x"), Offset("exposure")]
    if not category_first:
        terms = [Slope("x"), Offset("exposure"), cat]
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
            Category("cat", ("A", "B", "C"), reference="A"),
            Slope("x"),
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
        terms = [Slope("x")] + ([Offset("exposure")] if family == "poisson" else [])
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
    design = _rate_design(frame, Category("cat", ("A", "B", "C"), reference="A"))
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
    grp = Category("grp", ("a", "b"), reference="a")
    design = build_design(
        [grp, Slope("flag"), Slope("x"), Offset("exposure")], pooled, intercept=True
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
    data = sim.simulate(sessions, "rise", np.random.default_rng(8))
    assert len(data) == 30
    pooled = pd.concat(data.values(), ignore_index=True)
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
        estimates,
        pd.Series([s.animal for s in sessions]),
        pd.Series([s.day for s in sessions]),
        pd.Series([s.key[2] for s in sessions]),
    )
    planted = sim.planted_contrast("rise")
    print(f"planted {planted:.4f}, estimate {per_animal.mean():.4f}")
    # Over 40 data sets of this cohort at this penalty (data seeds 100 to
    # 139), the estimate less the planted value had mean -0.055 (the
    # penalty's shrinkage) and SD 0.048: the bound is that mean to three SDs,
    # -0.20 to +0.09. This seed's is +0.020; with the curve at 0 in place of
    # the reference window (the mean over 0 to 1 s) it is +0.285.
    assert per_animal.mean() - planted == pytest.approx(-0.055, abs=0.145)
