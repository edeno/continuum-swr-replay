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


def test_the_quick_run_goes_through_the_runner(monkeypatch) -> None:
    monkeypatch.setattr(rsim, "load_table", _table)
    table = run_simulations.run("role_content", quick=True)
    assert table.attrs["scenarios"] == ["null_role_content", "role_current_large"]
    assert table.attrs["content_model"] == "role_content"
    assert table.attrs["kind"] == "No SWR"
    assert set(table["estimand"]) == {"Q3a", "Q3b", "Q4a", "Q4b", "Q7b"}
    planted = table.set_index(["scenario", "estimand"])["planted"]
    assert planted[("role_current_large", "Q3a")] == pytest.approx(1.0)
    assert planted[("null_role_content", "Q3a")] == 0.0
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
    assert [m.name for m in run.models] == ["sd4", "sd8"]
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
