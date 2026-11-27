"""The visit rate model's estimands (``rate/visit_rate_statistics.py``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.regression.newton import CONVERGED
from continuum_swr_replay.analysis.replay_models.declared_tests import DIFFERENCE
from continuum_swr_replay.analysis.replay_models.rate import visit_rate as vr
from continuum_swr_replay.analysis.replay_models.rate import (
    visit_rate_scenarios as scenarios,
)
from continuum_swr_replay.analysis.replay_models.rate.visit_rate_statistics import (
    lag_curve_band,
    lag_curves,
    session_estimands,
    summarize_estimands,
    third_clock_q7a,
)
from continuum_swr_replay.analysis.shared_statistics import estimand_summary

CLOCKS = scenarios.PlantedClocks(
    arrival=lambda s: 1.5 * (1.0 - np.exp(-np.asarray(s) / 1.5)),
    leaving=lambda u: -1.0 * np.exp(-np.asarray(u) / 0.3),
)
SETTINGS = vr.VisitRateSettings(fit_by="session", include_stretch_clock=False)


@pytest.fixture
def cohort(make_visit_rows, cohort_of):
    """Three sessions with counts from a constant ratio of 1.3, and a fourth
    with no event."""
    rng = np.random.default_rng(5)
    out = {}
    for i in range(3):
        key = (f"rat{i}", "20200101", "02_r1")
        rows = make_visit_rows(key, n_visits=60, seed=60 + i)
        truth = scenarios.constant_ratio(rows, CLOCKS, log_ratio=np.log(1.3), total=500)
        rows["count"] = rng.poisson(truth.expected).astype(float)
        out[key] = cohort_of(rows, "SWR")
    key = ("rat0", "20200102", "02_r1")
    out[key] = cohort_of(make_visit_rows(key, n_visits=20, seed=70), "SWR")
    return out


def _at_clock(
    rows: pd.DataFrame, clock: str, lo: float, hi: float, rewarded: float = 0.0
) -> pd.DataFrame:
    """One row of the session repeated at clock values from ``lo`` to ``hi``
    in steps of 0.01 s, the last outcome set to ``rewarded``, every other
    column held."""
    grid = np.arange(round(lo * 100), round(hi * 100) + 1) / 100.0
    frame = rows.iloc[[0] * grid.size].reset_index(drop=True)
    frame[clock] = grid
    frame[vr.REWARD] = rewarded
    return frame


def test_estimands_are_linear_functionals(cohort) -> None:
    """Each declared estimand is a difference of mean linear predictors of
    the full design (``Design.session``), every other column held, so that
    every other term cancels: Q1 at the clock values of its windows after an
    omission, Q7a at those of its windows, Q2 and Q2-onset after a reward
    less after an omission at the same times since arrival over their
    windows; each variance is ``c' V c``. The session without an event has
    no fit and no value."""
    fit = vr.fit_visit_rate(
        cohort, model="A", kind="SWR", settings=SETTINGS, allow_provisional="test"
    )
    values = (
        session_estimands(fit)
        .set_index(["animal", "session_date", "estimand_id"])
        .sort_index()
    )
    pairs = {
        "Q1": ((vr.ARRIVAL, (2.0, 4.0), 0.0), (vr.ARRIVAL, (0.0, 1.0), 0.0)),
        "Q7a": ((vr.LEAVING, (0.0, 0.25), 0.0), (vr.LEAVING, (1.0, 2.0), 0.0)),
        "Q2": ((vr.ARRIVAL, (4.0, 10.0), 1.0), (vr.ARRIVAL, (4.0, 10.0), 0.0)),
        "Q2-onset": ((vr.ARRIVAL, (2.0, 4.0), 1.0), (vr.ARRIVAL, (2.0, 4.0), 0.0)),
    }
    for rows, record in zip(cohort.values(), fit.records, strict=True):
        at = (record.animal, record.session_date)
        if record.status != CONVERGED:
            assert record.no_fit == "no_events"
            assert values.loc[at, "estimate"].isna().all()
            continue
        beta = np.asarray(record.coef)
        vcov = np.asarray(record.vcov)
        session_rows = rows.rows["A"]
        for estimand_id, (first, second) in pairs.items():
            X_first, _ = fit.design.session(
                _at_clock(session_rows, first[0], *first[1], first[2])
            )
            X_second, _ = fit.design.session(
                _at_clock(session_rows, second[0], *second[1], second[2])
            )
            contrast = X_first.mean(axis=0) - X_second.mean(axis=0)
            row = values.loc[(*at, estimand_id)]
            assert row["estimate"] == pytest.approx(contrast @ beta, rel=1e-9)
            variance = contrast @ vcov @ contrast
            assert row["variance"] == pytest.approx(variance, rel=1e-9)
            assert row["se"] == pytest.approx(np.sqrt(variance))
    assert sum(r.status == CONVERGED for r in fit.records) == 3


def _values(kind: str, seed: int) -> pd.DataFrame:
    """Hand-made session values: 5 animals, 2 days, 1 or 2 sessions a day."""
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(5):
        for d in range(2):
            for e in range(1 + (a + d) % 2):
                for estimand_id in vr.RATE_ESTIMANDS:
                    rows.append(
                        {
                            "animal": f"rat{a}",
                            "session_date": f"2020010{d}",
                            "epoch": f"0{2 * e + 2}_r{e + 1}",
                            "kind": kind,
                            "estimand_id": estimand_id,
                            "estimate": rng.normal(0.3, 0.2),
                            "variance": 0.01,
                            "se": 0.1,
                        }
                    )
    return pd.DataFrame(rows)


def test_difference_between_kinds() -> None:
    """Each kind is ``estimand_summary`` of its session values; the
    difference is ``estimand_summary`` of the per-session differences,
    over the sessions where both kinds have a value."""
    swr, no_swr = _values("SWR", 0), _values("No SWR", 1)
    no_swr.loc[0, "estimate"] = np.nan  # one session of Q1 without a No SWR value
    summary = summarize_estimands(
        pd.concat([swr, no_swr], ignore_index=True), n_draws=200, seed=3
    ).set_index(["estimand_id", "tested"])
    assert len(summary) == 3 * len(vr.RATE_ESTIMANDS)
    keys = ["animal", "session_date", "epoch"]
    for estimand_id in vr.RATE_ESTIMANDS:
        a = swr[swr["estimand_id"] == estimand_id].set_index(keys)["estimate"]
        b = no_swr[no_swr["estimand_id"] == estimand_id].set_index(keys)["estimate"]
        for tested, values in (("SWR", a), ("No SWR", b), (DIFFERENCE, a - b)):
            values = values.reset_index()
            expected = estimand_summary(
                values["estimate"],
                values["animal"],
                values["session_date"],
                values["epoch"],
                n_draws=200,
                seed=3,
            )
            got = summary.loc[(estimand_id, tested)]
            for name, value in expected.items():
                assert got[name] == pytest.approx(value), (estimand_id, tested, name)
    assert summary.loc[("Q1", DIFFERENCE), "n_sessions"] == len(a) - 1
    with pytest.raises(ValueError, match="No SWR"):
        summarize_estimands(swr, n_draws=10, seed=0)


def test_lag_curves_and_their_band(cohort) -> None:
    fit = vr.fit_visit_rate(
        cohort, model="D", kind="SWR", settings=SETTINGS, allow_provisional="test"
    )
    curves = lag_curves(fit, seed=0)
    assert len(curves) == len(fit.records) * len(vr.LAGS)
    fitted = curves.dropna(subset=["estimate"])
    assert len(fitted) == 3 * len(vr.LAGS)
    assert (fitted["low"] < fitted["estimate"]).all()
    assert (fitted["estimate"] < fitted["high"]).all()
    band = lag_curve_band(curves, n_draws=200, seed=1)
    assert band.estimate.shape == (len(vr.LAGS),)
    assert np.all(band.low < band.high)
    with pytest.raises(ValueError, match="model A"):
        session_estimands(fit)


def test_q7a_with_and_without_the_third_clock(make_visit_rows, cohort_of) -> None:
    """A drop on the leaving clock only, in visits of one or two stretches:
    Q7a of each fit is its own session value, the difference is with less
    without, and each summary is ``estimand_summary`` of its column."""
    rng = np.random.default_rng(8)
    cohort = {}
    for i in range(3):
        key = (f"rat{i}", "20200101", "02_r1")
        rows = make_visit_rows(key, n_visits=80, seed=80 + i)
        truth = scenarios.drop_before_leaving(
            rows, CLOCKS.arrival, drop=1.0, total=800.0
        )
        rows["count"] = rng.poisson(truth.expected).astype(float)
        cohort[key] = cohort_of(rows, "SWR")
    fits = {
        include: vr.fit_visit_rate(
            cohort,
            model="A",
            kind="SWR",
            settings=vr.VisitRateSettings(
                fit_by="session", include_stretch_clock=include
            ),
            allow_provisional="test",
        )
        for include in (True, False)
    }
    compared = third_clock_q7a(fits[True], fits[False], n_draws=200, seed=2)
    sessions = compared.sessions
    assert len(sessions) == 3 and (sessions["kind"] == "SWR").all()
    for include, column in ((True, "with_stretch"), (False, "without_stretch")):
        q7a = session_estimands(fits[include])
        q7a = q7a[q7a["estimand_id"] == "Q7a"]["estimate"].to_numpy()
        assert sessions[column].to_numpy() == pytest.approx(q7a)
    assert sessions["difference"].to_numpy() == pytest.approx(
        sessions["with_stretch"] - sessions["without_stretch"]
    )
    summary = compared.summary.set_index("value")
    for column in ("with_stretch", "without_stretch", "difference"):
        expected = estimand_summary(
            sessions[column],
            sessions["animal"],
            sessions["session_date"],
            sessions["epoch"],
            n_draws=200,
            seed=2,
        )
        for name, value in expected.items():
            assert summary.loc[column, name] == pytest.approx(value, nan_ok=True)
    with pytest.raises(ValueError, match="does not"):
        third_clock_q7a(fits[False], fits[True], n_draws=10, seed=0)


def test_a_record_of_a_larger_unit_is_refused(cohort) -> None:
    """Until the statistics read the unit of the fit, a record of a day or an
    animal is refused, so that it is never reduced as a session."""
    for model, read in (
        ("A", session_estimands),
        ("D", lambda f: lag_curves(f, seed=0)),
    ):
        fit = vr.fit_visit_rate(
            cohort, model=model, kind="SWR", settings=SETTINGS, allow_provisional="test"
        )
        day = fit._replace(
            records=[
                r.__class__(**{**r.__dict__, "fit_by": "day"}) for r in fit.records
            ]
        )
        with pytest.raises(ValueError, match="fit_by"):
            read(day)
