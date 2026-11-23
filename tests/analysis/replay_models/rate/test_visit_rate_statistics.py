"""The visit rate model's estimands (``rate/visit_rate_statistics.py``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.regression.contrasts import range_grid
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
)
from continuum_swr_replay.analysis.shared_statistics import estimand_summary

CLOCKS = scenarios.PlantedClocks(
    arrival=lambda s: 1.5 * (1.0 - np.exp(-np.asarray(s) / 1.5)),
    leaving=lambda u: -1.0 * np.exp(-np.asarray(u) / 0.3),
)
SETTINGS = vr.VisitRateSettings(include_stretch_clock=False)


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


def test_estimands_are_linear_functionals(cohort) -> None:
    """Q1 and Q7a are the grid means of the fitted smooths over their
    windows; Q2 is the reward coefficient; each variance is ``c' V c``. The
    session without an event has no fit and no value."""
    fit = vr.fit_visit_rate(cohort, model="A", kind="SWR", settings=SETTINGS)
    values = (
        session_estimands(fit)
        .set_index(["animal", "session_date", "estimand_id"])
        .sort_index()
    )
    c = vr.declared_contrasts(fit.design, SETTINGS)
    arrival = vr.curve_rows(fit.design, f"s({vr.ARRIVAL})")
    leaving = vr.curve_rows(fit.design, f"s({vr.LEAVING})")
    for record in fit.records:
        at = (record.animal, record.session_date)
        if record.status != CONVERGED:
            assert record.no_fit == "no_events"
            assert values.loc[at, "estimate"].isna().all()
            continue
        beta = np.asarray(record.coef)
        vcov = np.asarray(record.vcov)
        q1 = (arrival(range_grid(2.0, 4.0)) @ beta).mean() - (
            arrival(range_grid(0.0, 1.0)) @ beta
        ).mean()
        q7a = (leaving(range_grid(0.0, 0.25)) @ beta).mean() - (
            leaving(range_grid(1.0, 2.0)) @ beta
        ).mean()
        q2 = beta[fit.design.coef_names.index(vr.REWARD)]
        for estimand_id, expected in (("Q1", q1), ("Q2", q2), ("Q7a", q7a)):
            row = values.loc[(*at, estimand_id)]
            assert row["estimate"] == pytest.approx(expected, rel=1e-10)
            variance = c[estimand_id] @ vcov @ c[estimand_id]
            assert row["variance"] == pytest.approx(variance, rel=1e-10)
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
    fit = vr.fit_visit_rate(cohort, model="D", kind="SWR", settings=SETTINGS)
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
