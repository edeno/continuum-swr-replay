"""The visit rate model's estimands (``rate/visit_rate_statistics.py``)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.regression.contrasts import evaluate_contrast
from continuum_swr_replay.analysis.regression.newton import CONVERGED
from continuum_swr_replay.analysis.regression.units import unit_summary
from continuum_swr_replay.analysis.replay_models.declared_tests import DIFFERENCE
from continuum_swr_replay.analysis.replay_models.rate import visit_rate as vr
from continuum_swr_replay.analysis.replay_models.rate import (
    visit_rate_scenarios as scenarios,
)
from continuum_swr_replay.analysis.replay_models.rate.visit_rate_statistics import (
    difference_curve_band,
    difference_curves,
    lag_curve_band,
    lag_curves,
    summarize_estimands,
    unit_estimands,
)
from continuum_swr_replay.analysis.shared_statistics import estimand_summary

CLOCKS = scenarios.PlantedClocks(
    arrival=lambda s: 1.5 * (1.0 - np.exp(-np.asarray(s) / 1.5)),
    leaving=lambda u: -1.0 * np.exp(-np.asarray(u) / 0.3),
)
SETTINGS = vr.VisitRateSettings(fit_by="session")


@pytest.fixture
def cohort(make_visit_rows, cohort_of):
    """Three sessions with counts from a constant ratio of 1.3, and a fourth
    with no event."""
    rng = np.random.default_rng(5)
    out = {}
    for i in range(3):
        key = (f"rat{i}", "20200101", "02_r1")
        rows = make_visit_rows(key, n_visits=60, seed=60 + i, length_s=(1.5, 12.0))
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
        unit_estimands(fit)
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
                            "fit_by": "session",
                            "kind": kind,
                            "estimand_id": estimand_id,
                            "estimate": rng.normal(0.3, 0.2),
                            "variance": 0.01,
                            "se": 0.1,
                            "supported": True,
                            "settings": "same",
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
    declared = [e for e in vr.RATE_ESTIMANDS if vr.is_declared_test(e, "session")]
    assert len(summary) == 3 * len(declared)
    keys = ["animal", "session_date", "epoch"]
    for estimand_id in declared:
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
        unit_estimands(fit)


def _two_days(make_visit_rows, cohort_of) -> dict:
    """Two animals of two days, each day two sessions, counts from a constant
    ratio of 1.3."""
    rng = np.random.default_rng(9)
    out = {}
    for a in range(2):
        for d in range(2):
            for epoch in ("02_r1", "04_r2"):
                key = (f"rat{a}", f"2020010{d}", epoch)
                rows = make_visit_rows(
                    key, n_visits=50, seed=len(out) + 90, length_s=(1.5, 12.0)
                )
                truth = scenarios.constant_ratio(
                    rows, CLOCKS, log_ratio=np.log(1.3), total=400
                )
                rows["count"] = rng.poisson(truth.expected).astype(float)
                out[key] = cohort_of(rows, "SWR")
    return out


@pytest.mark.parametrize(
    ("fit_by", "n_units"), [("session", 8), ("day", 4), ("animal", 2)]
)
def test_unit_estimands_at_each_unit(
    make_visit_rows, cohort_of, fit_by, n_units
) -> None:
    """One row per unit and estimand, keyed by the unit; each value is the
    declared contrast mapped onto the unit's coefficients by name (the
    baseline copies are not read), with its window support beside it; the
    summary is ``unit_summary`` at the unit."""
    cohort = _two_days(make_visit_rows, cohort_of)
    settings = vr.VisitRateSettings(fit_by=fit_by)
    fit = vr.fit_visit_rate(
        cohort, model="A", kind="SWR", settings=settings, allow_provisional="test"
    )
    values = unit_estimands(fit)
    assert len(values) == n_units * len(vr.RATE_ESTIMANDS)
    assert (values["fit_by"] == fit_by).all()
    c = vr.declared_contrasts(fit.design, settings)
    record = fit.records[-1]
    got = values[
        (values["animal"] == record.animal)
        & (values["session_date"] == record.session_date)
        & (values["epoch"] == record.epoch)
    ].set_index("estimand_id")
    for estimand_id, contrast in c.items():
        on_fit = fit.design.contrast_on(contrast, record.coef_names)
        expected = evaluate_contrast(
            on_fit, np.asarray(record.coef), np.asarray(record.vcov)
        )
        assert got.loc[estimand_id, "estimate"] == pytest.approx(
            expected.estimate, rel=1e-10
        )
    support = record.cohort_counts[vr.WINDOW_SUPPORT]
    assert got.loc["Q2", "supported"] == support["Q2"]["supported"]
    assert np.isnan(got.loc["Q7a", "visits_omitted"])
    both = pd.concat(
        [values, values.assign(kind="No SWR", estimate=values["estimate"] - 0.1)]
    )
    summary = summarize_estimands(both, n_draws=100, seed=1).set_index(
        ["estimand_id", "tested"]
    )
    assert (summary["fit_by"] == fit_by).all()
    q7a = values[values["estimand_id"] == "Q7a"]
    columns = {"session": [0, 1, 2], "day": [0, 1], "animal": [0]}[fit_by]
    keys = q7a.iloc[:, columns]
    index = (
        pd.Index(keys.iloc[:, 0].to_numpy(), name="animal")
        if fit_by == "animal"
        else pd.MultiIndex.from_frame(keys)
    )
    expected = unit_summary(
        pd.Series(q7a["estimate"].to_numpy(), index=index), fit_by, n_draws=100, seed=1
    )
    assert summary.loc[("Q7a", "SWR"), "estimate"] == pytest.approx(
        expected["estimate"]
    )
    assert summary.loc[("Q7a", DIFFERENCE), "estimate"] == pytest.approx(0.1)
    # The window tests are declared per animal (the twenty-sixth round): at
    # another unit the summary leaves them out; the unit table keeps them.
    windows = {"Q2", "Q2-onset"}
    summarized = set(summary.index.get_level_values("estimand_id"))
    assert (windows <= summarized) == (fit_by == "animal")
    assert not (windows & summarized) or fit_by == "animal"
    assert windows <= set(values["estimand_id"])


def test_values_of_mixed_units_are_refused(make_visit_rows, cohort_of) -> None:
    """Records of a day and of an animal are not reduced together; a fit
    whose records are of another unit than its setting is refused."""
    cohort = _two_days(make_visit_rows, cohort_of)
    tables = []
    for fit_by in ("day", "animal"):
        settings = vr.VisitRateSettings(fit_by=fit_by)
        fit = vr.fit_visit_rate(
            cohort, model="A", kind="SWR", settings=settings, allow_provisional="test"
        )
        tables.append(unit_estimands(fit))
    mixed = pd.concat(tables)
    mixed = pd.concat([mixed, mixed.assign(kind="No SWR")])
    with pytest.raises(ValueError, match="one unit of the fit"):
        summarize_estimands(mixed, n_draws=10, seed=0)
    relabeled = fit._replace(
        records=[r.__class__(**{**r.__dict__, "fit_by": "day"}) for r in fit.records]
    )
    with pytest.raises(ValueError, match="fit_by"):
        unit_estimands(relabeled)


def test_an_unsupported_window_is_marked_and_left_out(
    make_visit_rows, cohort_of
) -> None:
    """Visits after an omission of 2.5 to 5.9 s: many reach Q2's window, none
    lasts past 6 s. Q2's window is read in parts of at most 2 s (4 to 6, 6 to
    8, 8 to 10), and its 6 to 8 s part has no visit after an omission: Q2 is
    unsupported, its estimate and standard error NaN, its counts kept per
    part; it is left out of the reduction, which counts it. The onset test
    (2 to 4 s) stays supported."""
    rng = np.random.default_rng(11)
    cohort = {}
    for i in range(3):
        key = (f"rat{i}", "20200101", "02_r1")
        rows = make_visit_rows(
            key,
            n_visits=80,
            seed=110 + i,
            length_s=(1.5, 12.0),
            omission_length_s=(2.5, 5.9),
        )
        truth = scenarios.constant_ratio(rows, CLOCKS, log_ratio=0.3, total=500)
        rows["count"] = rng.poisson(truth.expected).astype(float)
        cohort[key] = cohort_of(rows, "SWR")
    settings = vr.VisitRateSettings(fit_by="session")
    fit = vr.fit_visit_rate(
        cohort, model="A", kind="SWR", settings=settings, allow_provisional="test"
    )
    values = unit_estimands(fit).set_index(["animal", "estimand_id"])
    for animal in ("rat0", "rat1", "rat2"):
        q2 = values.loc[(animal, "Q2")]
        assert not q2["supported"] and q2["visits_omitted"] == 0
        assert q2["at_risk_s_omitted"] == 0.0 and q2["visits_rewarded"] >= 5
        assert np.isnan(q2["estimate"]) and np.isnan(q2["se"])
        onset = values.loc[(animal, "Q2-onset")]
        assert onset["supported"] and np.isfinite(onset["estimate"])
    support = fit.records[0].cohort_counts[vr.WINDOW_SUPPORT]
    parts = [(r["outcome"], tuple(r["range_s"])) for r in support["Q2"]["reads"]]
    assert parts == [
        (1, (4.0, 6.0)),
        (1, (6.0, 8.0)),
        (1, (8.0, 10.0)),
        (0, (4.0, 6.0)),
        (0, (6.0, 8.0)),
        (0, (8.0, 10.0)),
    ]
    omitted = {
        tuple(r["range_s"]): r["visits"]
        for r in support["Q2"]["reads"]
        if r["outcome"] == 0
    }
    assert omitted[(4.0, 6.0)] >= 5 and omitted[(6.0, 8.0)] == 0
    rows = cohort[("rat0", "20200101", "02_r1")].rows["A"]
    s = rows[vr.ARRIVAL].to_numpy()
    on = (rows[vr.REWARD] == 0.0).to_numpy() & (s >= 4.0) & (s < 6.0)
    assert omitted[(4.0, 6.0)] == rows.loc[on, "well_visit_id"].nunique()
    # Read as three animals (one session each), where the window tests are
    # declared: Q2 is left out of the reduction and counted.
    table = unit_estimands(fit).assign(fit_by="animal", session_date="", epoch="")
    both = pd.concat([table, table.assign(kind="No SWR")])
    summary = summarize_estimands(both, n_draws=100, seed=0).set_index(
        ["estimand_id", "tested"]
    )
    assert summary.loc[("Q2-onset", "SWR"), "n_units_unsupported"] == 0
    assert summary.loc[("Q2", "SWR"), "n_units_unsupported"] == 3
    assert summary.loc[("Q2", "SWR"), "n_units"] == 0
    assert np.isnan(summary.loc[("Q2", "SWR"), "estimate"])


def test_the_difference_curve_and_its_band(cohort) -> None:
    """The difference curve of each unit is the outcome coefficient plus the
    rewarded arrival curve less the omitted one on the grid, inside its
    simultaneous band; across units it has a band; at the window's grid its
    mean is Q2."""
    fit = vr.fit_visit_rate(
        cohort, model="A", kind="SWR", settings=SETTINGS, allow_provisional="test"
    )
    curves = difference_curves(fit, seed=0)
    grid = vr.difference_grid(SETTINGS)
    assert len(curves) == len(fit.records) * grid.size
    fitted = curves.dropna(subset=["estimate"])
    assert (fitted["low"] <= fitted["estimate"]).all()
    assert (fitted["estimate"] <= fitted["high"]).all()
    record = fit.records[0]
    C = vr.difference_curve_rows(fit.design, grid)
    first = curves[
        (curves["animal"] == record.animal)
        & (curves["session_date"] == record.session_date)
    ]["estimate"].to_numpy()
    assert first == pytest.approx(C @ np.asarray(record.coef), rel=1e-10)
    q2 = unit_estimands(fit).set_index(["animal", "session_date", "estimand_id"])
    window = (grid >= 4.0 - 1e-9) & (grid <= 10.0 + 1e-9)
    # The 0.1 s grid's mean over the window is near the 0.01 s grid's (Q2).
    assert first[window].mean() == pytest.approx(
        q2.loc[(record.animal, record.session_date, "Q2"), "estimate"], abs=0.02
    )
    band = difference_curve_band(fitted, n_draws=200, seed=1)
    assert band.estimate.shape == grid.shape and np.all(band.low < band.high)


def _animal_values(kind: str, supported: list[bool]) -> pd.DataFrame:
    """Hand-made Q2 values of five animals at the animal unit."""
    return pd.DataFrame(
        {
            "animal": [f"rat{a}" for a in range(5)],
            "session_date": "",
            "epoch": "",
            "fit_by": "animal",
            "kind": kind,
            "estimand_id": "Q2",
            "estimate": np.linspace(0.1, 0.5, 5),
            "variance": 0.01,
            "se": 0.1,
            "supported": supported,
            "settings": "same",
        }
    )


def test_the_kind_difference_counts_the_units_either_kind_lacks() -> None:
    """The difference SWR minus No SWR leaves out a unit where either kind's
    value lacks support, and counts it."""
    swr = _animal_values("SWR", [True, False, True, True, True])
    no_swr = _animal_values("No SWR", [True, True, False, True, True])
    summary = summarize_estimands(
        pd.concat([swr, no_swr]), n_draws=100, seed=0
    ).set_index(["estimand_id", "tested"])
    difference = summary.loc[("Q2", DIFFERENCE)]
    assert difference["n_units_unsupported"] == 2
    assert difference["n_units"] == 3


def test_tables_of_other_settings_are_refused(cohort) -> None:
    """Each unit table carries its fit's settings; tables of other settings
    (another prior SD here) are not reduced together."""
    tables = []
    for sd in (2.0, 1.0):
        settings = dataclasses.replace(SETTINGS, weak_prior_sd=sd)
        fit = vr.fit_visit_rate(
            cohort, model="A", kind="SWR", settings=settings, allow_provisional="test"
        )
        tables.append(unit_estimands(fit))
    assert tables[0]["settings"].nunique() == 1
    assert tables[0]["settings"].iloc[0] != tables[1]["settings"].iloc[0]
    with pytest.raises(ValueError, match="settings"):
        summarize_estimands(
            pd.concat([tables[0], tables[1].assign(kind="No SWR")]), n_draws=10, seed=0
        )
    summarize_estimands(
        pd.concat([tables[0], tables[0].assign(kind="No SWR")]), n_draws=10, seed=0
    )


def test_equal_settings_have_one_id() -> None:
    """``settings_id`` is canonical: a prior SD of 2.0, np.float64(2.0) or 2,
    a mapping in another key order, a cap of 10 or 10.0 give one id; a
    setting that differs in any field gives another."""
    from continuum_swr_replay.analysis.replay_models.rate.visit_rate_statistics import (
        settings_id,
    )

    base = vr.VisitRateSettings(max_time_since_arrival_s=10.0)
    record = {"tables": {"a": "x", "b": "y"}, "n": 1}
    calibrated = {
        ("SWR", vr.LEAVING): vr.CalibratedPenalty(1.5, 10.0, record),
        ("SWR", vr.OMITTED_ARRIVAL_TERM): vr.CalibratedPenalty(0.5, 15.0, record),
    }
    reordered = {k: calibrated[k] for k in reversed(list(calibrated))}
    reordered_record = {"n": 1, "tables": {"b": "y", "a": "x"}}
    pairs = [
        (
            dataclasses.replace(base, weak_prior_sd=2.0),
            dataclasses.replace(base, weak_prior_sd=np.float64(2.0)),
        ),
        (
            dataclasses.replace(base, weak_prior_sd=2.0),
            dataclasses.replace(base, weak_prior_sd=2),
        ),
        (
            dataclasses.replace(base, calibrated_penalties=calibrated),
            dataclasses.replace(
                base,
                calibrated_penalties={
                    k: vr.CalibratedPenalty(
                        p.log_penalty, p.target_edf, reordered_record
                    )
                    for k, p in reordered.items()
                },
            ),
        ),
        (
            vr.VisitRateSettings(max_time_since_arrival_s=10.0),
            vr.VisitRateSettings(max_time_since_arrival_s=10),
        ),
    ]
    for a, b in pairs:
        assert settings_id(a) == settings_id(b)
        assert len(settings_id(a)) == 16
    others = [
        dataclasses.replace(base, weak_prior_sd=1.0),
        dataclasses.replace(base, fit_by="animal"),
        dataclasses.replace(base, reward_form="coefficient"),
        dataclasses.replace(base, q1_reference_s=(1.0, 2.0)),
        dataclasses.replace(base, max_time_since_arrival_s=None),
        dataclasses.replace(base, log_penalties={vr.LEAVING: 2.0}),
        dataclasses.replace(base, calibrated_penalties=calibrated),
    ]
    ids = {settings_id(s) for s in [base, *others]}
    assert len(ids) == len(others) + 1
