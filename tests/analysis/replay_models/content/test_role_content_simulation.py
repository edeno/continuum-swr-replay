"""The role content models' scenario list for the simulation engine."""

from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.regression import simulation as engine
from continuum_swr_replay.analysis.replay_models.content import role_content as rc
from continuum_swr_replay.analysis.replay_models.content import (
    role_content_scenarios as sc,
)
from continuum_swr_replay.analysis.replay_models.content import (
    role_content_simulation as rsim,
)
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS
from scripts.checks import run_simulations

from .role_tables import synthetic_event_arm_table

SESSIONS = (
    ("a1", "20200101", "02_r1"),
    ("a1", "20200102", "02_r1"),
    ("a2", "20200103", "04_r2"),
    ("a3", "20200104", "02_r1"),
)


def _table() -> pd.DataFrame:
    table = synthetic_event_arm_table(SESSIONS, n_visits=80, swr_share=0.5, seed=5)
    rng = np.random.default_rng(0)
    events = table["event_id"].unique()
    bins = pd.Series(rng.integers(7, 60, events.size).astype(float), index=events)
    return table.assign(n_nonlocal_bins=table["event_id"].map(bins))


@pytest.fixture(scope="module")
def cohort() -> rsim.SimulationCohort:
    return rsim.cohort_from_table(_table())


def test_the_cohort_holds_no_response_and_every_events_bins(cohort) -> None:
    assert rc.RESPONSE not in cohort.rows.columns
    per_event = cohort.rows.groupby(rc.UNIT)[rsim.N_BINS].agg(["min", "max"])
    assert (per_event["min"] == per_event["max"]).all()
    assert np.isfinite(cohort.rows[rsim.N_BINS]).all()
    assert set(cohort.data_truths) == {"SWR", "No SWR"}


def test_the_prior_sds_are_the_declared_sd_then_its_pair() -> None:
    assert rsim.PRIOR_SDS == (rc.WEAK_PRIOR_SD, *rc.WEAK_PRIOR_SD_SENSITIVITY)
    assert rsim.PRIOR_SDS == (8.0, 4.0, 16.0)


def test_the_quick_run_goes_through_the_runner(monkeypatch) -> None:
    monkeypatch.setattr(rsim, "load_table", _table)
    table = run_simulations.run("role_content", quick=True)
    assert table.attrs["scenarios"] == ["null_role_content", "role_current_large"]
    assert table.attrs["content_model"] == "role_content"
    assert table.attrs["kind"] == "No SWR"
    assert set(table["estimand"]) == {"Q3a", "Q3b", "Q4a", "Q4b", "Q7b"}
    names = [rsim.model_name(u, rc.WEAK_PRIOR_SD) for u in rsim.FIT_BYS]
    assert sorted(set(table["model"])) == sorted(names)
    planted = table.set_index(["model", "scenario", "estimand"])["planted"]
    for name in names:
        assert planted[(name, "role_current_large", "Q3a")] == pytest.approx(1.0)
        assert planted[(name, "null_role_content", "Q3a")] == 0.0
    assert (table["share_not_converged"] == 0).all()


def test_a_scenario_keeps_its_place_and_its_data(cohort) -> None:
    """A run that computes one scenario's shares draws the data sets of a
    run that computes them all."""
    kwargs = {"model": "role_content", "kind": "No SWR", "prior_sds": (8.0,),
              "n_data_sets": 2, "n_draws": 50}  # fmt: skip
    every = rsim.build_simulation(cohort, **kwargs)
    one = rsim.build_simulation(cohort, scenarios=["role_previous_small"], **kwargs)
    assert [s.name for s in every.scenarios] == [s.name for s in one.scenarios]
    a = engine.run(every, 1, ["role_previous_small"]).data_sets
    b = engine.run(one, 1, ["role_previous_small"]).data_sets
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(ValueError, match="not computed"):
        engine.run(one, 1, ["null_role_content"])


def test_shares_and_planted_values_are_the_scenarios(cohort) -> None:
    run = rsim.build_simulation(
        cohort, model="role_content_uncertainty", kind="SWR", prior_sds=(4.0, 8.0),
        scenarios=["recency_curve_large"], n_data_sets=2, n_draws=50,
    )  # fmt: skip
    assert [m.name for m in run.models] == [
        f"{u}_sd{sd}" for u in ("session", "day", "animal") for sd in (4, 8)
    ]
    assert [m.fit_by for m in run.models] == [
        u for u in ("session", "day", "animal") for _ in (4, 8)
    ]
    scenario = next(s for s in run.scenarios if s.name == "recency_curve_large")
    shares = scenario.expected(run.sessions, np.random.default_rng(0))
    for key, rows in run.sessions.items():
        total = pd.Series(shares[key]).groupby(rows[rc.UNIT].to_numpy()).sum()
        np.testing.assert_allclose(total, 1.0)
    truth = sc.Truth(
        arm=sc.arm_effects(),
        curves={(rc.RECENCY, rc.REFERENCE_ROLE): sc.recency_curve(1.0)},
    )
    reference = sc.ScenarioReference(
        cohort.model_rows("role_content_uncertainty"), cohort.reference_log_duration
    )
    q5a = sc.planted_values(truth, reference)["Q5a"]
    assert scenario.planted["Q5a"] == pytest.approx(q5a)
    low5 = sc.planted_values(
        truth, reference, settings=rc.RoleContentSettings(q5a_trials=(20.0, 5.0))
    )["Q5a"]
    assert scenario.planted[rsim.Q5A_LOW5] == pytest.approx(low5)
    assert 0 < low5 < q5a
    # A run can be sent to a process pool.
    pickle.dumps(run)


def test_every_listed_scenario_builds(cohort) -> None:
    for model in rc.MODELS:
        names = [n for n, _ in rsim.scenario_list(cohort, model, "unadjusted", "SWR")]
        assert names[0] == f"null_{model}"
        assert {"from_data", "spread"} <= set(names)
    adjusted = rsim.scenario_list(cohort, "role_content", "length_adjusted", "SWR")
    assert [n for n, _ in adjusted] == ["length_current", "length_value"]


def test_a_thin_run_keeps_that_many_events(cohort) -> None:
    run = rsim.build_simulation(
        cohort, model="role_content", kind="No SWR", prior_sds=(8.0,),
        scenarios=["null_role_content"], thin=12, n_data_sets=2, n_draws=50,
    )  # fmt: skip
    assert all(rows[rc.UNIT].nunique() <= 12 for rows in run.sessions.values())


def _one_arm_share(data, sessions) -> float:
    shares = pd.concat(
        [rows.assign(y=data[key][rc.RESPONSE]) for key, rows in sessions.items()]
    )
    return float((shares.groupby(rc.UNIT)["y"].max() >= 1 - 1e-9).mean())


def test_the_calibrated_noise_draws_its_stated_share_of_one_draw_events(
    cohort,
) -> None:
    """Under the calibrated clustering an event is one draw with probability
    pi: with a large concentration and many bins, the rest almost never put
    all their bins on one arm, so the one-arm share is about pi."""
    clustering = {"No SWR": ((np.inf, 0.6, 1e4),)}
    run = rsim.build_simulation(
        cohort, model="role_content", kind="No SWR", noise=rsim.CALIBRATED_NOISE,
        prior_sds=(8.0,), scenarios=["null_role_content"], n_data_sets=1,
        n_draws=50, clustering=clustering,
    )  # fmt: skip
    for rows in run.sessions.values():
        per_event = rows.groupby(rc.UNIT)[[rsim.NOISE_TRIALS, rsim.NOISE_CONCENTRATION]]
        assert (per_event.nunique() == 1).all().all()
        trials = per_event.first()[rsim.NOISE_TRIALS]
        bins = rows.groupby(rc.UNIT)[rsim.N_BINS].first()
        assert ((trials == 1) | (trials == bins)).all()
    events = pd.concat(
        [
            rows.groupby(rc.UNIT)[rsim.NOISE_TRIALS].first()
            for rows in run.sessions.values()
        ]
    )
    assert (events == 1).mean() == pytest.approx(0.6, abs=0.05)
    place = [s.name for s in run.scenarios].index("null_role_content")
    data = engine.draw_data_set(
        run.models[0], run.sessions, run.scenarios[place],
        np.random.default_rng(engine.data_seed(run.seed, place, 0)),
    )  # fmt: skip
    share = _one_arm_share(data, run.sessions)
    assert share == pytest.approx(0.6, abs=0.05)
    assert share >= (events == 1).mean()


def test_the_calibrated_noise_takes_each_brackets_values(cohort) -> None:
    """Events are given the pi and the concentration of their length
    bracket (upper edges exclusive)."""
    clustering = {"SWR": ((30.0, 1.0, 2.0), (np.inf, 0.0, 5.0))}
    run = rsim.build_simulation(
        cohort, model="role_content", kind="SWR", noise=rsim.CALIBRATED_NOISE,
        prior_sds=(8.0,), scenarios=["null_role_content"], n_data_sets=1,
        n_draws=50, clustering=clustering,
    )  # fmt: skip
    rows = pd.concat(run.sessions.values())
    short = rows[rsim.N_BINS] < 30
    assert (rows.loc[short, rsim.NOISE_TRIALS] == 1).all()
    assert (rows.loc[short, rsim.NOISE_CONCENTRATION] == 2.0).all()
    assert (rows.loc[~short, rsim.NOISE_TRIALS] == rows.loc[~short, rsim.N_BINS]).all()
    assert (rows.loc[~short, rsim.NOISE_CONCENTRATION] == 5.0).all()


@pytest.mark.parametrize("noise", ["categorical", "multinomial", "dirichlet_10"])
def test_each_noise_draws_counts_over_the_events_bins(cohort, noise) -> None:
    """A share is a count over the event's choices: 0 or 1 for one draw,
    a whole number over the event's bins otherwise; an event's shares sum to
    one."""
    run = rsim.build_simulation(
        cohort, model="role_content", kind="No SWR", noise=noise, prior_sds=(8.0,),
        scenarios=["role_current_large"], n_data_sets=1, n_draws=50,
    )  # fmt: skip
    place = [s.name for s in run.scenarios].index("role_current_large")
    data = engine.draw_data_set(
        run.models[0], run.sessions, run.scenarios[place],
        np.random.default_rng(engine.data_seed(run.seed, place, 0)),
    )  # fmt: skip
    for key, rows in run.sessions.items():
        y = data[key][rc.RESPONSE].to_numpy()
        n = 1.0 if noise == "categorical" else rows[rsim.N_BINS].to_numpy()
        np.testing.assert_allclose(y * n, np.round(y * n), atol=1e-9)
        total = pd.Series(y).groupby(rows[rc.UNIT].to_numpy()).sum()
        np.testing.assert_allclose(total, 1.0)


@pytest.mark.parametrize("a", rsim.CONCENTRATIONS)
def test_the_dirichlet_noise_has_the_mean_and_variance_its_docstring_states(
    a,
) -> None:
    """At 24 bins the Dirichlet-multinomial's mean share is the planted
    probability and its variance that of 24 independent bins times
    (24 + a) / (1 + a) (the docstring of CONCENTRATIONS)."""
    from continuum_swr_replay.analysis.regression.likelihoods import (
        GroupedSoftmax,
        draw,
    )

    n_bins, n_groups = 24, 20000
    p = np.array([0.4, 0.2, 0.15, 0.1, 0.05, 0.05, 0.03, 0.01, 0.01])
    groups = np.repeat(np.arange(n_groups), p.size)
    eta = np.tile(np.log(p), n_groups)
    rng = np.random.default_rng(3)
    y = draw(GroupedSoftmax(), eta, groups, rng, n_trials=n_bins, concentration=a)
    y = y.reshape(n_groups, p.size)
    np.testing.assert_allclose(y.mean(axis=0), p, atol=0.01)
    multiple = y.var(axis=0) / (p * (1 - p) / n_bins)
    np.testing.assert_allclose(multiple[:4], (n_bins + a) / (1 + a), rtol=0.06)


def _big_cohort() -> rsim.SimulationCohort:
    """Four sessions of many events, each of many bins (nearly noiseless
    shares), the time until leaving spread over its whole range."""
    table = synthetic_event_arm_table(
        SESSIONS, n_visits=400, events_per_visit=3.0, swr_share=0.5, seed=5
    )
    events = table["event_id"].unique()
    until = pd.Series(np.random.default_rng(0).uniform(0, 3.0, events.size), events)
    return rsim.cohort_from_table(
        table.assign(
            n_nonlocal_bins=5000.0,
            time_until_visit_end_s=table["event_id"].map(until),
        )
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("model", "variant", "scenario", "estimand"),
    [
        ("role_content", "unadjusted", "role_current_large", "Q3a"),
        ("role_content", "unadjusted", "role_previous_large", "Q3b"),
        ("role_content", "unadjusted", "reward_curve_current_large", "Q4a"),
        ("role_content", "unadjusted", "value_other_leaf_large", "Q4b"),
        ("role_content", "unadjusted", "leaving_curve_large", "Q7b"),
        ("role_content_uncertainty", "unadjusted", "recency_curve_large", "Q5a"),
        ("role_content", "length_adjusted", "length_current", "Q3a"),
    ],
)
def test_the_planted_values_are_what_the_engine_recovers(
    model, variant, scenario, estimand
) -> None:
    """On large sessions with loose penalties the fit's estimate of each
    planted estimand is its planted value, at every unit of the fit."""
    penalties = {rc.REWARD_FRACTION: -8.0, rc.RECENCY: -8.0, rc.LEAVING: -6.0}
    run = rsim.build_simulation(
        _big_cohort(), model=model, kind="No SWR", variant=variant,
        prior_sds=(16.0,), smooth_log_penalties=penalties, scenarios=[scenario],
        n_data_sets=2, n_draws=50,
    )  # fmt: skip
    summary = engine.run(run, 1, [scenario]).summary.set_index(["model", "estimand"])
    for unit in rsim.FIT_BYS:
        row = summary.loc[(rsim.model_name(unit, 16.0), estimand)]
        assert abs(row["planted"]) > 0.1
        assert row["mean_estimate"] == pytest.approx(row["planted"], rel=0.1)


def test_truth_from_data_gives_the_known_levels_and_band_curves() -> None:
    """Levels are the log ratio of a role's mean share to the other-leaf
    role's; a band's curve value is the log of its shrunk mean over the
    role's mean; a covariate with one value gives a flat zero curve."""
    n = 20
    current = pd.DataFrame(
        {
            rc.ROLE: "current",
            rc.LEAVING: np.r_[np.full(n, 0.25), np.full(n, 2.0)],
            rc.RESPONSE: np.r_[np.full(n, 0.6), np.full(n, 0.2)],
        }
    )
    previous = pd.DataFrame({rc.ROLE: "previous", rc.RESPONSE: np.full(2 * n, 0.2)})
    other = pd.DataFrame({rc.ROLE: rc.REFERENCE_ROLE, rc.RESPONSE: np.full(4 * n, 0.1)})
    rows = pd.concat([current, previous, other], ignore_index=True).assign(
        animal="a1", **{rc.REWARD_FRACTION: 0.5, rc.RECENCY: np.log(10.0)}
    )
    rows[rc.LEAVING] = rows[rc.LEAVING].fillna(1.0)
    (truth,) = sc.truth_from_data(rows).values()
    assert truth.role["current"] == pytest.approx(np.log(4.0))
    assert truth.role["previous"] == pytest.approx(np.log(2.0))
    leaving = truth.curves[(rc.LEAVING, "current")]
    k = sc.DATA_PSEUDO_EVENTS
    low = np.log((n * 0.6 + k * 0.4) / (n + k) / 0.4)
    high = np.log((n * 0.2 + k * 0.4) / (n + k) / 0.4)
    np.testing.assert_allclose(leaving(np.array([0.25, 1.0, 2.25])), [low, 0.0, high])
    flat = truth.curves[(rc.REWARD_FRACTION, "current")]
    np.testing.assert_allclose(flat(np.linspace(0, 1, 5)), 0.0, atol=1e-12)
    assert truth.arm == {}


def test_sessions_per_animal_keeps_evenly_spaced_sessions() -> None:
    sessions = (
        ("a1", "20200101", "02_r1"),
        ("a1", "20200102", "02_r1"),
        ("a1", "20200103", "02_r1"),
        ("a2", "20200104", "02_r1"),
    )
    table = synthetic_event_arm_table(sessions, n_visits=20, seed=1).assign(
        n_nonlocal_bins=10.0
    )
    kept = rsim.cohort_from_table(table, sessions_per_animal=2).rows
    found = set(map(tuple, kept[list(SESSION_COLUMNS)].drop_duplicates().values))
    assert found == {sessions[0], sessions[2], sessions[3]}
    one = rsim.cohort_from_table(table, sessions_per_animal=1).rows
    assert set(one["animal"]) == {"a1", "a2"}
    assert one.groupby("animal")["session_date"].nunique().eq(1).all()


def test_the_unadjusted_length_scenario_has_no_planted_value(cohort) -> None:
    run = rsim.build_simulation(
        cohort, model="role_content", kind="No SWR", prior_sds=(8.0,),
        scenarios=["length_current_unadjusted"], n_data_sets=1, n_draws=50,
    )  # fmt: skip
    scenario = next(s for s in run.scenarios if s.name == "length_current_unadjusted")
    assert np.isnan(scenario.planted["Q3a"])


@pytest.mark.slow
def test_calibrate_smooth_penalties_reaches_its_targets(cohort) -> None:
    targets = {rc.REWARD_FRACTION: 2.5, rc.LEAVING: 3.0}
    found = rsim.calibrate_smooth_penalties(cohort, targets=targets)
    assert set(found) == set(targets)
    for covariate, entry in found.items():
        assert entry["target_edf"] == targets[covariate]
        assert entry["median_edf"] == pytest.approx(targets[covariate], abs=0.05)
        assert entry["term"] == f"s({covariate}):{rc.ROLE}=current"
        assert np.isfinite(entry["log_penalty"])
        assert entry["n_sessions_used"] >= 1
