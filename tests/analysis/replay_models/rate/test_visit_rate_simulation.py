"""The visit rate model's scenario list (``rate/visit_rate_simulation.py``).

The quick configuration (a synthetic cohort of 5 animals of 2 sessions, 2 data
sets) runs through the engine; one test plants a coefficient inside the model
on a larger synthetic cohort. Every number here is a simulation number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.regression import simulation as engine
from continuum_swr_replay.analysis.regression.sessions import fit_sessions
from continuum_swr_replay.analysis.regression.simulation import (
    InsideScenario,
    OutsideScenario,
    Simulation,
)
from continuum_swr_replay.analysis.replay_models.rate import visit_rate as vr
from continuum_swr_replay.analysis.replay_models.rate import (
    visit_rate_scenarios as sc,
)
from continuum_swr_replay.analysis.replay_models.rate import (
    visit_rate_simulation as vs,
)


@pytest.fixture(scope="module")
def quick() -> Simulation:
    return vs.simulation(True, 1)


@pytest.fixture(scope="module")
def quick_result(quick):
    return engine.run(quick)


def test_the_quick_list_runs_through_the_engine(quick, quick_result) -> None:
    """Every scenario of the kind, every model and estimand has a row; the
    session-level and animal-level models are fitted to the same data sets;
    every fit converged and every unit has a value."""
    summary = quick_result.summary
    assert list(dict.fromkeys(summary["scenario"])) == list(vs.SCENARIOS["SWR"])
    assert set(summary["model"]) == {"session_sd2", "animal"}
    assert set(summary["estimand"]) == {"Q1", "Q1b", "Q2", "Q7a"}
    assert (summary["share_not_converged"] == 0.0).all()
    assert (summary["share_without_value"] == 0.0).all()
    assert (summary["n_data_sets_with_estimate"] == 2).all()
    animal_q2 = quick_result.sessions.query("model == 'animal' and estimand == 'Q2'")
    assert animal_q2["session_date"].isna().all()
    assert len(animal_q2) == len(vs.SCENARIOS["SWR"]) * 2 * 5
    assert (animal_q2["n_sessions"] == 2).all()
    table = vs.table(quick_result, quick)
    assert {"kind", "target_edf", "level", "prior_sd", "truth_omission_side"} <= set(
        table.columns
    )
    assert quick.attrs["allow_provisional"] == "simulation"


def test_the_same_seed_gives_the_same_table(quick, quick_result) -> None:
    """A second run of one scenario gives its rows of the run of every
    scenario: the same seed, the same data sets."""
    again = engine.run(quick, scenarios=["constant_ratio"]).summary
    first = quick_result.summary.query("scenario == 'constant_ratio'")
    pd.testing.assert_frame_equal(
        again.reset_index(drop=True), first.reset_index(drop=True)
    )


def test_the_planted_values(quick) -> None:
    """A constant ratio is the truth of every level; a later onset has no
    planted ratio; the differing ratios' session truth is the reduction of
    the sessions' own ratios, and the animal truth lies within their range."""
    truths = quick.attrs["truths"]
    log_ratio = vs.LOG_RATIO["SWR"]
    for level in ("session", "animal", "omission_side"):
        assert truths["constant_ratio"][level]["Q2"] == pytest.approx(log_ratio)
        assert truths["no_effect"][level]["Q2"] == 0.0
        assert np.isnan(truths["later_onset"][level]["Q2"])
    keys = pd.DataFrame(
        list(quick.sessions), columns=["animal", "session_date", "epoch"]
    )
    ratios = sc.session_log_ratios(
        keys, mean=log_ratio, animal_sd=0.2, session_sd=0.3, seed=vs.DIFFERING_SEED
    )
    per_animal = pd.Series(ratios).groupby(level=0).mean()
    assert truths["differing_ratio"]["session"]["Q2"] == pytest.approx(
        per_animal.mean()
    )
    assert (
        min(ratios.values())
        < truths["differing_ratio"]["animal"]["Q2"]
        < max(ratios.values())
    )
    assert truths["drop_before_leaving"]["session"]["Q7a"] == pytest.approx(-vs.DROP)


def test_a_planted_coefficient_inside_the_model_is_recovered() -> None:
    """Ten synthetic sessions of 80 visits and about 2,000 events each, the
    rate flat on the clocks and 0.6 higher after a reward, planted on the
    session-level model's design: Q2 of the session-level model and of the
    animal-level model are within 3 standard errors of 0.6 over four data
    sets (a simulation tolerance); Q1 and Q7a are near 0."""
    sessions = {}
    for a in range(5):
        for e in range(2):
            key = (f"animal{a}", f"2026{e + 1:04d}", "02_r1")
            rows = vs.synthetic_visit_rows(key, n_visits=80, seed=100 + 2 * a + e)
            sessions[key] = rows
    pooled = pd.concat(sessions.values(), ignore_index=True)
    models = vs._models("SWR", pooled, vs.Config(prior_sds=(2.0,)), None)
    seconds = np.mean([rows[vr.EXPOSURE].sum() for rows in sessions.values()])
    inside = InsideScenario(
        "inside", {"(Intercept)": float(np.log(2000.0 / seconds)), vr.REWARD: 0.6}
    )
    outside = OutsideScenario(
        "outside", _constant, planted={"Q2": 0.6, "Q1": 0.0, "Q1b": 0.0, "Q7a": 0.0}
    )
    result = engine.run(Simulation(models, sessions, [inside, outside], 4, 200, 3))
    table = result.summary.set_index(["scenario", "model", "estimand"])
    for row in (
        table.loc[("inside", "session_sd2", "Q2")],
        table.loc[("outside", "animal", "Q2")],
        table.loc[("outside", "session_sd2", "Q2")],
    ):
        assert row["planted"] == pytest.approx(0.6)
        assert abs(row["bias"]) < 3 * row["sd"] / 2 + 0.01
    for estimand in ("Q1", "Q7a"):
        assert abs(table.loc[("outside", "animal", estimand), "mean_estimate"]) < 0.15


def _constant(sessions, rng):
    """About 2,000 events a session, flat on the clocks, 0.6 higher after a
    reward."""
    out = {}
    for key, rows in sessions.items():
        log_rate = 0.6 * rows[vr.REWARD].to_numpy()
        mean = rows[vr.EXPOSURE].to_numpy() * np.exp(log_rate)
        out[key] = mean * 2000.0 / mean.sum()
    return out


def test_sessions_spread_over_each_animal() -> None:
    keys = [("a", f"{d}", "02_r1") for d in range(10)] + [("b", "1", "02_r1")]
    picked = vs.spread_sessions(keys, 3)
    assert picked == [
        ("a", "0", "02_r1"),
        ("a", "4", "02_r1"),
        ("a", "9", "02_r1"),
        ("b", "1", "02_r1"),
    ]


def test_a_smoothed_curve_follows_its_bins() -> None:
    """Binned counts from a known log rate: the smoothed curve is within 0.1
    of it on the grid's interior (a simulation tolerance)."""
    rng = np.random.default_rng(0)
    x = np.arange(0.125, 10.0, 0.25)
    truth = np.log(0.5) + np.log(4.0) * (1 - np.exp(-x / 1.5))
    at_risk = np.full(x.size, 2000.0)
    starts = rng.poisson(at_risk * np.exp(truth)).astype(float)
    curve = vs.smoothed_log_rate(starts, at_risk, x, 0.0, 10.0, 0.5, 0.0)
    inner = (x > 0.5) & (x < 9.5)
    assert np.max(np.abs(curve(x[inner]) - truth[inner])) < 0.1


# --- The joint run of both kinds ---------------------------------------------------


@pytest.fixture(scope="module")
def joint() -> Simulation:
    return vs.simulation(True, 1, {"kind": "both"})


@pytest.fixture(scope="module")
def joint_result(joint):
    return engine.run(joint)


def _kind_rows(rows: pd.DataFrame, kind: str) -> pd.DataFrame:
    return rows[rows[vs.KIND_COLUMN] == kind].reset_index(drop=True)


def test_the_joint_rows_hold_each_kind_with_its_own_planted_rate(joint) -> None:
    """Each session's rows are its SWR rows, then its No SWR rows, each with
    the expected counts of its own kind's scenario: under ``swr_ratio`` the
    SWR rows carry the constant ratio of 2.5 and the No SWR rows no effect."""
    cohorts = {kind: vs._synthetic_cohort(kind) for kind in vr.EVENT_KINDS}
    for key, rows in joint.sessions.items():
        for kind, scenario in zip(
            ("SWR", "No SWR"), vs.JOINT_SCENARIOS["swr_ratio"], strict=True
        ):
            own = cohorts[kind][key].rows["A"]
            part = _kind_rows(rows, kind)
            assert len(part) == len(own)
            planted = vs._scenario_truth(
                scenario,
                own,
                vs.QUICK_CLOCKS,
                kind,
                float(own[vr.RESPONSE].sum()),
                (0.0, 1.0),
                {},
                None,
            )[0]
            np.testing.assert_allclose(
                part["expected_swr_ratio"].to_numpy(), planted.expected, rtol=1e-12
            )


def test_the_joint_fit_gives_each_kind_the_values_of_its_own_run(joint) -> None:
    """On one drawn data set, the joint fit's values of each kind in each
    session (estimate, standard error, status, cause) are those the
    single-kind run's model gives on that kind's rows of the data set, by the
    engine's own per-session path, exactly."""
    model = joint.models[0]
    scenario = next(s for s in joint.scenarios if s.name == "swr_ratio")
    data = engine.draw_data_set(
        model, joint.sessions, scenario, np.random.default_rng(7)
    )
    units = model.fit(data).set_index(["estimand", "animal", "session_date", "epoch"])
    for kind in ("SWR", "No SWR"):
        part = {key: _kind_rows(rows, kind) for key, rows in data.items()}
        pooled = pd.concat(
            [vs._synthetic_cohort(kind)[k].rows["A"] for k in part], ignore_index=True
        )
        (single,) = vs._models(
            kind,
            pooled,
            vs.Config(kind=kind, prior_sds=(2.0,), animal_level=False),
            None,
        )
        records = fit_sessions(
            single.design,
            part,
            single.likelihood,
            model=single.name,
            kind=kind,
            response=single.response,
            unit=single.unit,
            table_sha256={},
        )
        rows, _ = engine._session_rows(single, records, {})
        for row in rows:
            if row["estimand"] not in vs.JOINT_ESTIMANDS:
                continue
            name = vs.joint_estimand(row["estimand"], kind)
            got = units.loc[(name, row["animal"], row["session_date"], row["epoch"])]
            assert got["estimate"] == row["estimate"]
            assert got["se"] == row["se"]
            assert got["status"] == row["status"]
            assert got["without_value"] == row["without_value"]


def test_the_difference_is_the_difference_of_the_session_values(
    joint, joint_result
) -> None:
    """Each session's SWR minus No SWR value is the difference of its two
    values, its standard error the root of the summed variances; the data
    set's estimate of the difference is the difference of the two estimates
    (the reduction is a mean of means and every session has both values); the
    planted difference is the difference of the planted values."""
    sessions = joint_result.sessions
    keys = ["scenario", "model", "data_set", "animal", "session_date", "epoch"]
    for estimand in vs.JOINT_ESTIMANDS:
        by = {
            kind: sessions[sessions["estimand"] == vs.joint_estimand(estimand, kind)]
            .set_index(keys)
            .sort_index()
            for kind in (*vr.EVENT_KINDS, vs.DIFFERENCE)
        }
        swr, no_swr, diff = by["SWR"], by["No SWR"], by[vs.DIFFERENCE]
        np.testing.assert_array_equal(
            diff["estimate"], swr["estimate"] - no_swr["estimate"]
        )
        np.testing.assert_allclose(
            diff["se"], np.hypot(swr["se"], no_swr["se"]), rtol=1e-12
        )
    sets = joint_result.data_sets.set_index(
        ["scenario", "model", "data_set", "estimand"]
    )
    for (scenario, model, data_set), _ in sets.groupby(level=[0, 1, 2]):
        for estimand in vs.JOINT_ESTIMANDS:
            at = {
                kind: sets.loc[
                    (scenario, model, data_set, vs.joint_estimand(estimand, kind))
                ]
                for kind in (*vr.EVENT_KINDS, vs.DIFFERENCE)
            }
            assert at[vs.DIFFERENCE]["estimate"] == pytest.approx(
                at["SWR"]["estimate"] - at["No SWR"]["estimate"], abs=1e-12
            )
            planted = at["SWR"]["planted"] - at["No SWR"]["planted"]
            if np.isnan(planted):
                assert np.isnan(at[vs.DIFFERENCE]["planted"])
            else:
                assert at[vs.DIFFERENCE]["planted"] == pytest.approx(planted, abs=1e-12)


def test_the_joint_truths_are_the_single_kind_truths(joint) -> None:
    """Each kind's planted value in the joint run is that of the single-kind
    run of its scenario on the same rows and seeds; a later onset has no Q2,
    so neither has the difference."""
    single = {
        kind: vs.simulation(True, 1, {"kind": kind}).attrs["truths"]
        for kind in vr.EVENT_KINDS
    }
    truths = joint.attrs["truths"]
    for name, scenarios in vs.JOINT_SCENARIOS.items():
        for kind, scenario in zip(("SWR", "No SWR"), scenarios, strict=True):
            for estimand in vs.JOINT_ESTIMANDS:
                expected = single[kind][scenario]["session"][estimand]
                got = truths[name][vs.joint_estimand(estimand, kind)]
                if np.isnan(expected):
                    assert np.isnan(got)
                else:
                    assert got == pytest.approx(expected, abs=1e-12)
    assert np.isnan(truths["swr_later_onset"][vs.joint_estimand("Q2", vs.DIFFERENCE)])
    assert truths["both_ratios"][
        vs.joint_estimand("Q2", vs.DIFFERENCE)
    ] == pytest.approx(vs.LOG_RATIO["SWR"] - vs.LOG_RATIO["No SWR"])


def test_the_joint_run_recovers_the_swr_ratio(joint_result) -> None:
    """With a ratio of 2.5 planted in SWR only, the SWR Q2 is far above the
    No SWR Q2 in every data set (a simulation tolerance: 10 synthetic sessions
    of about 150 events), every fit converged and every session has a value."""
    sets = joint_result.data_sets.query("scenario == 'swr_ratio'").set_index(
        ["data_set", "estimand"]
    )["estimate"]
    for data_set in range(2):
        swr = sets.loc[(data_set, vs.joint_estimand("Q2", "SWR"))]
        no_swr = sets.loc[(data_set, vs.joint_estimand("Q2", "No SWR"))]
        assert swr > 0.5
        assert abs(no_swr) < 0.4
    summary = joint_result.summary
    assert (summary["share_not_converged"] == 0.0).all()
    assert (summary["share_without_value"] == 0.0).all()
    assert list(dict.fromkeys(summary["scenario"])) == list(vs.JOINT_SCENARIOS)
