"""The per-session fitting loop: support, sessions without a finite maximum, records."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.regression.likelihoods import GroupedSoftmax, Poisson
from continuum_swr_replay.analysis.regression.reml import SessionData, fit_at_penalties
from continuum_swr_replay.analysis.regression.sessions import fit_sessions
from continuum_swr_replay.analysis.regression.terms import (
    Category,
    Offset,
    Slope,
    Smooth,
    build_design,
)

from .conftest import ROLES, by_session

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
                "a": "no finite maximum (round 1)",
                "b": "no finite maximum (round 1)",
            }
            assert np.isnan(record.coef[9:]).all()
            assert np.isnan(np.asarray(record.vcov)[9:]).all()
            assert np.isfinite(record.coef[:9]).all()
            assert "inactive" in record.fit_reason
            assert record.edf["a"] == 0.0 and set(record.collinearity) == {"s(clock)"}
            continue
        alone = fit_at_penalties(
            GroupedSoftmax(), [data], design.blocks, design.log_penalties
        ).sessions[0]
        np.testing.assert_array_equal(record.coef, alone.fit.beta)
        np.testing.assert_array_equal(record.vcov, alone.covariance)
        assert record.left_out == {} and record.fit_reason == ""
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
    role = Category("role", ROLES, reference="other")
    return build_design(
        [
            role,
            Slope("reward_fraction", by=role, levels=("current", "previous", "other")),
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

    assert record.support["role[previous]"]["n_high"] == 3
    # The slope by the role fails its own support too (3 events).
    assert record.left_out == {
        "role[previous]": "support",
        "reward_fraction:role=previous": "support",
    }
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
    np.testing.assert_array_equal(record.coef, alone.sessions[0].fit.beta)


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
    assert record.support["role[previous]"]["passes"]
    assert record.left_out["role[previous]"].startswith("no finite maximum (round ")
    assert record.left_out["reward_fraction:role=previous"] == (
        "rows of role[previous] dropped"
    )
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
    design = build_design([Slope("x"), Offset("exposure")], pooled, intercept=True)
    (record,) = _fit(
        design, {key: rows}, Poisson(), response="count", unit="well_visit_id"
    )
    assert record.cohort_counts["rows_missing_values"] == 2
    assert record.n_rows == len(rows) - 2


def test_a_session_with_no_rows_left_is_recorded_not_fitted(rate_sessions) -> None:
    """A session given with no rows, or whose rows are all missing, or with no
    events, is not fitted: one record says why, and the loop goes on."""
    pooled = pd.concat(rate_sessions.values(), ignore_index=True)
    design = build_design([Slope("x"), Offset("exposure")], pooled, intercept=True)
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
    assert "no rows" in records[0].fit_reason
    assert records[1].cohort_counts["rows_missing_values"] == len(missing)
    assert "no rows" in records[1].fit_reason
    assert "no events" in records[2].fit_reason
    for record in records[:3]:
        assert np.isnan(record.coef).all() and record.edf == {}
