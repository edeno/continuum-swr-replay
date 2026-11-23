"""The planted truths of the role content models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay_models.content import role_content as rc
from continuum_swr_replay.analysis.replay_models.content import (
    role_content_scenarios as sc,
)
from continuum_swr_replay.analysis.replay_models.content import (
    uncertainty_scenarios as us,
)
from continuum_swr_replay.analysis.shared_statistics import animal_day_session_values

from .role_tables import draw_shares, synthetic_event_arm_table

SESSIONS = (
    ("a1", "20200101", "02_r1"),
    ("a1", "20200101", "04_r2"),
    ("a2", "20200103", "02_r1"),
)
CONTENT_IDS = {e.estimand_id for m in rc.MODELS for e in rc.model_estimands(m)}


@pytest.fixture(scope="module")
def rows() -> pd.DataFrame:
    table = synthetic_event_arm_table(SESSIONS, n_visits=100, swr_share=0.0, seed=2)
    cohort = rc.role_content_cohort(table)
    return cohort.rows.assign(_l0=cohort.reference_log_duration)


@pytest.fixture(scope="module")
def reference(rows) -> sc.ScenarioReference:
    return sc.ScenarioReference(rows, float(rows["_l0"].iloc[0]))


def test_expected_shares_are_a_softmax_per_event(rows, reference) -> None:
    truth = sc.Truth(arm={1: 0.5}, role={"current": 1.0})
    share = sc.expected_shares(rows, truth, reference)
    totals = pd.Series(share).groupby(rows[rc.UNIT].to_numpy()).sum()
    np.testing.assert_allclose(totals, 1.0)
    event = rows[rc.UNIT].iloc[0]
    on = (rows[rc.UNIT] == event).to_numpy()
    eta = (rows.loc[on, rc.ARM] == 1) * 0.5 + (rows.loc[on, rc.ROLE] == "current") * 1.0
    np.testing.assert_allclose(share[on], np.exp(eta) / np.exp(eta).sum())


def test_curves_and_slopes_are_centered_as_the_model_centers_them(
    rows, reference
) -> None:
    curve = sc.recency_curve(1.0)
    truth = sc.Truth(
        curves={(rc.RECENCY, rc.REFERENCE_ROLE): curve},
        slopes={(rc.VALUE, "current"): 0.5},
    )
    eta = sc._linear_predictor(rows, truth, reference)
    other = (rows[rc.ROLE] == rc.REFERENCE_ROLE).to_numpy()
    leaf = rows[rc.ROLE].isin(rc.LEAF_ROLES).to_numpy()
    current = (rows[rc.ROLE] == "current").to_numpy()
    # The recency curve has mean 0 over the pooled other-leaf rows.
    assert eta[other].mean() == pytest.approx(0.0, abs=1e-12)
    # The slope is per pooled SD over the leaf rows, centered at their mean.
    value = rows[rc.VALUE].to_numpy()
    expected = 0.5 * (value[current] - value[leaf].mean()) / value[leaf].std()
    np.testing.assert_allclose(eta[current], expected)


def test_planted_values_by_hand(rows, reference) -> None:
    truth = sc.Truth(
        role={"current": 0.7, "previous": -0.2},
        slopes={(rc.VALUE, rc.REFERENCE_ROLE): 0.3, (rc.BELIEF_VARIANCE, rc.REFERENCE_ROLE): 0.1},
        curves={
            (rc.REWARD_FRACTION, "current"): lambda x: 2.0 * x,
            (rc.RECENCY, rc.REFERENCE_ROLE): lambda x: x,
            (rc.LEAVING, "current"): lambda u: -u,
        },
    )  # fmt: skip
    planted = sc.planted_values(truth, reference)
    current = rows.loc[rows[rc.ROLE] == "current", rc.REWARD_FRACTION]
    assert set(planted) == CONTENT_IDS
    assert planted["Q3a"] == 0.7 and planted["Q3b"] == -0.2
    assert planted["Q4b"] == 0.3 and planted["Q5b-var"] == 0.1 and planted["Q5b-n"] == 0
    assert planted["Q4a"] == pytest.approx(2.0 * current.std(ddof=0))
    assert planted["Q5a"] == pytest.approx(np.log(20.0) - np.log(3.0))
    assert planted["Q7b"] == pytest.approx(-(0.125 - 1.5))
    full = sc.planted_values(
        truth, reference, settings=rc.RoleContentSettings(q4a_reading="full_range")
    )
    assert full["Q4a"] == pytest.approx(2.0)


def test_the_scenario_list_is_complete_and_named() -> None:
    scenarios = sc.role_content_scenarios() + us.uncertainty_scenarios()
    names = [s.name for s in scenarios]
    assert len(names) == len(set(names))
    for s in scenarios:
        assert s.model in rc.MODELS and s.variant in rc.VARIANTS
        assert s.failure and set(s.validates) <= CONTENT_IDS and s.validates
    single = {s.name for s in scenarios}
    for size in ("small", "large"):
        for name in ("role_current", "role_previous", "value_other_leaf", "value_current",
                     "reward_curve_current", "reward_curve_previous", "leaving_curve",
                     "recency_curve", "belief_variance_other_leaf",
                     "sample_count_other_leaf"):  # fmt: skip
            assert f"{name}_{size}" in single
    assert {"recency_only", "belief_variance", "sample_count", "trial_drift",
            "reward_curvature", "length_current", "length_value", "large_effects"} <= single  # fmt: skip


def test_null_plants_zero_and_single_effects_plant_one(reference) -> None:
    by_name = {s.name: s for s in sc.role_content_scenarios()}
    key = SESSIONS[0]
    null = sc.planted_values(by_name["null_role_content"].truth_for(key), reference)
    assert all(v == 0 for v in null.values())
    planted = sc.planted_values(
        by_name["role_previous_large"].truth_for(key), reference
    )
    assert planted["Q3b"] == sc.SIZES["role"]["large"]
    assert all(v == 0 for k, v in planted.items() if k != "Q3b")
    planted = sc.planted_values(
        by_name["sample_count_other_leaf_small"].truth_for(key), reference
    )
    assert planted["Q5b-n"] == sc.SIZES["slope"]["small"]
    assert all(v == 0 for k, v in planted.items() if k != "Q5b-n")


def test_uncertainty_truths_plant_no_measure_effect_where_they_should_not(
    reference,
) -> None:
    for s in us.uncertainty_scenarios():
        planted = sc.planted_values(s.truth_for(SESSIONS[0]), reference)
        assert planted["Q5a"] > 0.3
        if s.name != "belief_variance":
            assert planted["Q5b-var"] == 0
        if s.name != "sample_count":
            assert planted["Q5b-n"] == 0


def test_truth_from_data_recovers_the_role_ratio(rows, reference) -> None:
    truth = sc.Truth(role={"current": 1.0, "trunk": -1.0})
    drawn = pd.concat(
        [
            draw_shares(part, sc.expected_shares(part, truth, reference), 200, i)
            for i, (_, part) in enumerate(rc.session_rows(rows).items())
        ],
        ignore_index=True,
    )
    truths = sc.truth_from_data(drawn)
    assert set(truths) == {"a1", "a2"}
    for animal, data_truth in truths.items():
        assert data_truth.role["current"] == pytest.approx(1.0, abs=0.15), animal
        planted = sc.planted_values(data_truth, reference)
        assert np.isfinite(list(planted.values())).all()
        assert planted["Q4b"] == 0
        share = sc.expected_shares(drawn, data_truth, reference)
        assert np.isfinite(share).all()


def test_spread_truths_target_the_animal_balanced_mean(reference) -> None:
    keys = [*SESSIONS, ("a2", "20200104", "02_r1")]
    mean = sc.Truth(role={"current": 0.5}, slopes={(rc.VALUE, rc.REFERENCE_ROLE): 0.2})
    truths = sc.spread_truths(keys, mean, animal_sd=0.3, session_sd=0.1, seed=4)
    assert truths == sc.spread_truths(keys, mean, animal_sd=0.3, session_sd=0.1, seed=4)
    scenario = sc.Scenario(
        "spread", "role_content", "unadjusted", ("Q3a",), "x", truths.__getitem__
    )
    planted = scenario.planted(keys, reference)
    target = sc.planted_target(planted)
    values = planted["Q3a"]
    labels = [planted[c] for c in rc.SESSION_COLUMNS]
    expected = animal_day_session_values(values, *labels).mean()
    assert target["Q3a"] == pytest.approx(expected)
    # a1's two sessions share a day: the day's mean, then the animal's.
    a1 = planted.loc[planted["animal"] == "a1", "Q3a"].mean()
    a2 = planted.loc[planted["animal"] == "a2", "Q3a"].mean()
    assert target["Q3a"] == pytest.approx((a1 + a2) / 2)
    assert planted["Q3a"].std() > 0


def test_thin_events_keeps_whole_events(rows) -> None:
    part = next(iter(rc.session_rows(rows).values()))
    thin = sc.thin_events(part, 20, seed=0)
    assert thin[rc.UNIT].nunique() == 20 and len(thin) == 180
    assert sc.thin_events(part, 10**6, seed=0) is part


def test_every_scenario_gives_shares_on_real_shaped_rows(rows, reference) -> None:
    scenarios = sc.role_content_scenarios() + us.uncertainty_scenarios()
    sessions = rc.session_rows(rows)
    for s in scenarios:
        for key, part in sessions.items():
            share = s.expected_shares(key, part, reference)
            assert np.isfinite(share).all() and (share > 0).all(), s.name
        planted = s.planted(list(sessions), reference)
        assert len(planted) == len(sessions)
