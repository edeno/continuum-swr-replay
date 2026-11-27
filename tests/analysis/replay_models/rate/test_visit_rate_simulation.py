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
    every fit converged and every unit has a value."""
    summary = quick_result.summary
    assert list(dict.fromkeys(summary["scenario"])) == list(vs.SCENARIOS["SWR"])
    assert set(summary["model"]) == {"session_sd2"}
    assert set(summary["estimand"]) == {"Q1", "Q1b", "Q2", "Q2-onset", "Q7a"}
    assert (summary["share_not_converged"] == 0.0).all()
    assert (summary["share_without_value"] == 0.0).all()
    assert (summary["n_data_sets_with_estimate"] == 2).all()
    table = vs.table(quick_result, quick)
    assert {"kind", "target_edf", "prior_sd", "truth_by_outcome"} <= set(table.columns)
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
    """A constant ratio is the truth of every window and of the coefficient;
    a later onset has no planted coefficient but real window truths; the
    differing ratios' truth is the reduction of the sessions' own ratios."""
    truths = quick.attrs["truths"]
    log_ratio = vs.LOG_RATIO["SWR"]
    coef = vr.SECONDARY_ESTIMAND
    for name, value in (("constant_ratio", log_ratio), ("no_effect", 0.0)):
        assert truths[name]["pooled"][coef] == pytest.approx(value)
        for window in ("Q2", "Q2-onset"):
            assert truths[name]["by_outcome"][window] == pytest.approx(value)
    assert np.isnan(truths["later_onset"]["pooled"][coef])
    assert truths["later_onset"]["by_outcome"]["Q2-onset"] > 0.0
    keys = pd.DataFrame(
        list(quick.sessions), columns=["animal", "session_date", "epoch"]
    )
    ratios = sc.session_log_ratios(
        keys, mean=log_ratio, animal_sd=0.2, session_sd=0.3, seed=vs.DIFFERING_SEED
    )
    per_animal = pd.Series(ratios).groupby(level=0).mean()
    assert truths["differing_ratio"]["by_outcome"]["Q2"] == pytest.approx(
        per_animal.mean()
    )
    assert truths["drop_before_leaving"]["by_outcome"]["Q7a"] == pytest.approx(-vs.DROP)


def test_a_planted_coefficient_inside_the_model_is_recovered() -> None:
    """Ten synthetic sessions of 80 visits and about 2,000 events each, the
    rate flat on the clocks and 0.6 higher after a reward, planted on the
    session-level model's design: Q2 of the session-level model is within 3
    standard errors of 0.6 over four data sets (a simulation tolerance)."""
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
        "outside",
        _constant,
        planted={"Q2": 0.6, "Q2-onset": 0.6, "Q1": 0.0, "Q1b": 0.0, "Q7a": 0.0},
    )
    result = engine.run(Simulation(models, sessions, [inside, outside], 4, 200, 3))
    table = result.summary.set_index(["scenario", "model", "estimand"])
    for row in (
        table.loc[("inside", "session_sd2", "Q2")],
        table.loc[("outside", "session_sd2", "Q2")],
    ):
        assert row["planted"] == pytest.approx(0.6)
        assert abs(row["bias"]) < 3 * row["sd"] / 2 + 0.01


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


def test_the_rate_table_truth_is_flagged_as_read_from_the_real_counts(
    quick, quick_result
) -> None:
    """The rate table's planted values are a projection of a tabulation of the
    rows' counts (the real counts on the real cohort): its rows of the table
    carry ``truth_from_real_counts`` and the run's ``attrs`` say so; no other
    scenario's do."""
    table = vs.table(quick_result, quick)
    flagged = table.groupby("scenario")["truth_from_real_counts"].agg(["all", "any"])
    assert flagged.loc["rate_table"].all()
    assert not flagged.drop(index="rate_table")["any"].any()
    assert list(quick.attrs["truth_from_real_counts"]["scenarios"]) == ["rate_table"]
