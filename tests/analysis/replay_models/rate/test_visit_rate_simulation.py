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
    assert {"kind", "reward_form", "calibrate", "prior_sd", "truth_by_outcome"} <= set(
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
    config = vs.Config(prior_sds=(2.0,), fit_by=("session",))
    models = vs._models("SWR", pooled, config, {"session": None})
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


#: A small cohort of two sessions a day, for the units.
SMALL = (3, 2, 2, 30, 120.0, (1.0, 12.0), (0.5, 8.0))


@pytest.fixture(scope="module")
def at_every_unit() -> Simulation:
    return vs.simulation(
        False,
        1,
        {"synthetic": SMALL, "n_data_sets": 2, "n_draws": 50, "prior_sds": (2.0,)},
    )


def test_models_at_each_unit_on_the_same_draws(at_every_unit) -> None:
    """One model per unit (session, day, animal), named for it, of one
    design; the engine fits each on the same data sets, and each unit's rows
    are its fits."""
    models = at_every_unit.models
    assert [m.name for m in models] == ["session_sd2", "day_sd2", "animal_sd2"]
    assert [m.fit_by for m in models] == ["session", "day", "animal"]
    assert len({m.design.coef_names for m in models}) == 1
    result = engine.run(at_every_unit, scenarios=["constant_ratio"])
    units = result.sessions.groupby("model")["fit_by"].agg(["first", "size"])
    n_estimands = len(models[0].estimands)
    assert units.loc["session_sd2", "size"] == 12 * 2 * n_estimands
    assert units.loc["day_sd2", "size"] == 6 * 2 * n_estimands
    assert units.loc["animal_sd2", "size"] == 3 * 2 * n_estimands
    summary = result.summary.set_index(["model", "estimand"])
    assert summary.loc[("animal_sd2", "Q2"), "planted"] == pytest.approx(
        vs.LOG_RATIO["SWR"]
    )
    assert np.isfinite(summary.loc[("animal_sd2", "Q2"), "planted_unit"])


def test_the_window_truths_of_a_later_onset(at_every_unit) -> None:
    """The later onset's window truths are the plain means over each window
    of the planted difference ``f(s) - f(max(s - delay, 0))``, in closed
    form: nonzero over 2 to 4 s, zero over 4 to 10 s (the planted arrival
    curve is flat from 3 s); the coefficient has no truth (no ratio)."""
    truth = at_every_unit.attrs["truths"]["later_onset"]

    def difference(s):
        s = np.asarray(s)
        return vs.QUICK_CLOCKS.arrival(s) - vs.QUICK_CLOCKS.arrival(
            np.maximum(s - vs.DELAY_S, 0.0)
        )

    for name, (lo, hi) in (("Q2", (4.0, 10.0)), ("Q2-onset", (2.0, 4.0))):
        grid = np.linspace(lo, hi, round((hi - lo) * 100) + 1)
        assert truth["by_outcome"][name] == pytest.approx(np.mean(difference(grid)))
    assert truth["by_outcome"]["Q2"] == 0.0
    assert truth["by_outcome"]["Q2-onset"] > 0.01
    assert np.isnan(truth["pooled"][vr.SECONDARY_ESTIMAND])


def test_the_coefficient_form_is_a_run_of_its_own() -> None:
    """The secondary value's run: the coefficient form at each unit, its one
    estimand, planted from the coefficient's projection."""
    run = vs.simulation(
        False,
        1,
        {
            "synthetic": SMALL,
            "reward_form": "coefficient",
            "n_data_sets": 2,
            "n_draws": 50,
            "prior_sds": (2.0,),
        },
    )
    assert {tuple(m.estimands) for m in run.models} == {(vr.SECONDARY_ESTIMAND,)}
    constant = run.scenarios[vs.SCENARIOS["SWR"].index("constant_ratio")]
    model = run.models[0]
    assert engine.planted_value(
        constant, model, vr.SECONDARY_ESTIMAND
    ) == pytest.approx(vs.LOG_RATIO["SWR"])
    with pytest.raises(ValueError, match="reward_form"):
        vs.Config(reward_form="time_varying")


def test_the_curves_are_calibrated_at_the_unit() -> None:
    """Each curve of the form is calibrated at the unit to its target: the
    median edf over the unit's fits (three animals) reaches the target, and
    the record names the unit."""
    cohort = vs._synthetic_cohort("SWR", SMALL)
    settings = vr.VisitRateSettings(fit_by="animal")
    targets = {
        vr.OMITTED_ARRIVAL_TERM: 5.0,
        vr.REWARDED_ARRIVAL_TERM: 5.0,
        vr.LEAVING: 3.0,
    }
    found = vs.calibrate_clocks(
        cohort, "SWR", settings, vs.QUICK_CLOCKS, {"rows": "synthetic"}, targets=targets
    )
    assert set(found) == {("SWR", c) for c in settings.curves("A")}
    for (_, curve), penalty in found.items():
        last = penalty.rows["rounds"][-1][curve]
        assert penalty.target_edf == targets[curve]
        assert last["median_edf"] == pytest.approx(targets[curve], abs=0.01)
        assert last["n_units_used"] == 3
        assert penalty.rows["fit_by"] == "animal"
