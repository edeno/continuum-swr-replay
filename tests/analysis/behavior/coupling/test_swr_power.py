"""The per-event SWR-power model: its post-outcome gate, its fit, its cohort record
and the statistics of the power's distribution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior.cohort_audit import (
    COHORT_COLUMNS,
    DROP_FIELDS,
    write_cohort_audit,
)
from continuum_swr_replay.analysis.behavior.coupling import swr_power as mod

SC = ["animal", "session_date", "epoch"]


def _reconciles(rec: dict) -> bool:
    return rec["n_in"] - sum(rec[f] for f in DROP_FIELDS) == rec["n_modeled"]


def test_swr_power_records_split_qvals_and_own():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "animal": np.repeat(["a", "b"], n // 2),
            "session_date": 1,
            "epoch": "e",
            **{c: rng.normal(size=n) for c in mod.PREDICTORS},
            mod.POWER_COL: rng.normal(size=n),
            # Every event starts after its outcome, so the post-outcome gate
            # is a no-op here and the split under test is the NaN attribution.
            "event_start_s": 10.0,
            "outcome_known_time": 5.0,
            "trial_reward": np.tile([0.0, 1.0], n // 2),
        }
    )
    df.loc[0:9, "reward_belief_mean_chosen_arm_preoutcome"] = np.nan
    df.loc[100:104, "known_reward_outcome_lag_1"] = np.nan  # 5 own-predictor NaN -> own
    # The events each session keeps: 100 less its 10 qvals / 5 own NaN rows.
    coefs = pd.DataFrame(
        [
            {"animal": "a", "session_date": 1, "epoch": "e", "n_events": 90},
            {"animal": "b", "session_date": 1, "epoch": "e", "n_events": 95},
        ]
    )
    recs = mod.power_cohort_records(df, coefs)
    assert all(_reconciles(r) for r in recs)
    r = recs[0]
    assert r["n_drop_qvals_gap"] == 10
    assert r["n_drop_own_predictor_missing"] == 5
    assert r["n_in"] == n
    assert r["n_modeled"] == 185
    assert r["n_excluded_session"] == 0


def test_swr_power_n_modeled_is_the_fitters_count():
    """``n_modeled`` read from the fit equals the old recount of kept events.

    The record used to recount the fitted sessions' complete, post-outcome,
    outcome-balanced events; it now sums the fitter's ``n_events``. The frame
    exercises every way an event or session leaves the fit: qvals and own NaN,
    a pre-outcome event, a session under 30 events, and a session left with
    one outcome.
    """
    rng = np.random.default_rng(3)
    sizes = {"a": 60, "b": 50, "c": 20, "d": 40}
    n = sum(sizes.values())
    df = pd.DataFrame(
        {
            "animal": np.repeat(list(sizes), list(sizes.values())),
            "session_date": 1,
            "epoch": "e",
            **{c: rng.normal(size=n) for c in mod.PREDICTORS},
            mod.POWER_COL: rng.normal(size=n),
            "event_start_s": 10.0,
            "outcome_known_time": 5.0,
            "trial_reward": np.tile([0.0, 1.0], n // 2),
        }
    )
    df.loc[0:4, "reward_belief_mean_chosen_arm_preoutcome"] = np.nan  # qvals gap
    df.loc[60:62, "known_reward_outcome_lag_1"] = np.nan  # own
    df.loc[63:66, "event_start_s"] = 1.0  # pre-outcome
    # Session d keeps only rewarded events: unbalanced, so not fit.
    d_rows = df.index[df.animal == "d"]
    df.loc[d_rows[df.loc[d_rows, "trial_reward"] == 0.0], "event_start_s"] = 1.0

    coefs = mod.fit_power_model(df)
    assert set(coefs.animal) == {"a", "b"}  # c: < 30 events; d: one outcome

    # The old recount, kept here as the reference.
    _, _, selected, _ = mod.balanced_post_outcome_selection(df)
    qp = [c for c in mod._QVALS_PREDICTORS if c in df.columns]
    op = [c for c in (*mod.PREDICTORS, mod.POWER_COL, "event_start_s") if c not in qp]
    complete = np.isfinite(df[qp + op].to_numpy(dtype=float)).all(axis=1)
    kept = df[complete & selected].groupby(SC, sort=False).size()
    fit_keys = {tuple(r) for r in coefs[SC].to_numpy()}
    old_n_modeled = int(sum(int(kept.get(k, 0)) for k in fit_keys))

    rec = mod.power_cohort_records(df, coefs)[0]
    assert rec["n_modeled"] == old_n_modeled == 55 + 43
    assert _reconciles(rec)
    assert mod.power_cohort_records(df, coefs.iloc[0:0])[0]["n_modeled"] == 0


def test_swr_power_excluded_session_splits_into_its_three_gates(tmp_path):
    """The two written counts plus the <30-event remainder are ``n_excluded_session``.

    Complete events (no qvals or own NaN) leave the fit by exactly one of: a
    pre-outcome start, a session left with one outcome, or a session with fewer
    than 30 outcome-balanced events. The CSV carries the first two after the
    canonical columns and no longer carries the gate's constant fields.
    """
    rng = np.random.default_rng(3)
    sizes = {"a": 60, "b": 50, "c": 20, "d": 40}
    n = sum(sizes.values())
    df = pd.DataFrame(
        {
            "animal": np.repeat(list(sizes), list(sizes.values())),
            "session_date": 1,
            "epoch": "e",
            **{c: rng.normal(size=n) for c in mod.PREDICTORS},
            mod.POWER_COL: rng.normal(size=n),
            "event_start_s": 10.0,
            "outcome_known_time": 5.0,
            "trial_reward": np.tile([0.0, 1.0], n // 2),
        }
    )
    df.loc[0:4, "reward_belief_mean_chosen_arm_preoutcome"] = np.nan  # qvals gap
    df.loc[60:62, "known_reward_outcome_lag_1"] = np.nan  # own
    df.loc[63:66, "event_start_s"] = 1.0  # 4 pre-outcome
    # Session d's 20 omissions start pre-outcome; its 20 rewarded are unbalanced.
    d_rows = df.index[df.animal == "d"]
    df.loc[d_rows[df.loc[d_rows, "trial_reward"] == 0.0], "event_start_s"] = 1.0

    coefs = mod.fit_power_model(df)
    rec = mod.power_cohort_records(df, coefs)[0]
    _, _, selected, _ = mod.balanced_post_outcome_selection(df)
    session_gate = int(selected.sum()) - rec["n_modeled"]

    assert rec["n_events_excluded_pre_outcome"] == 4 + 20
    assert rec["n_events_excluded_outcome_imbalance"] == 20
    assert session_gate == 20  # session c
    assert rec["n_excluded_session"] == (
        rec["n_events_excluded_pre_outcome"]
        + rec["n_events_excluded_outcome_imbalance"]
        + session_gate
    )

    out = tmp_path / "swr_power_cohort_audit.csv"
    write_cohort_audit([rec], out, extra_columns=mod.COHORT_EXTRA_COLUMNS)
    written = pd.read_csv(out)
    assert list(written.columns) == [*COHORT_COLUMNS, *mod.COHORT_EXTRA_COLUMNS]
    assert written["n_events_excluded_pre_outcome"].iloc[0] == 24
    for gone in (
        "post_outcome_gate",
        "post_outcome_gate_rule",
        "outcome_time_semantics",
    ):
        assert gone not in rec


def test_swr_power_missing_timing_is_attributed_not_absorbed():
    """Both timing columns land in named buckets, never the residual.

    ``outcome_known_time`` arrives on the qvals left join, so its
    absence is a qvals gap; ``event_start`` is an event-table column,
    so its absence is an own-predictor gap. A physically pre-outcome
    event is neither -- it is a real event the gate excludes.
    """
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "animal": np.repeat(["a", "b"], n // 2),
            "session_date": 1,
            "epoch": "e",
            **{c: rng.normal(size=n) for c in mod.PREDICTORS},
            mod.POWER_COL: rng.normal(size=n),
            "event_start_s": 10.0,
            "outcome_known_time": 5.0,
            "trial_reward": np.tile([0.0, 1.0], n // 2),
        }
    )
    df.loc[0:6, "outcome_known_time"] = np.nan  # 7 -> qvals_gap
    df.loc[50:52, "event_start_s"] = np.nan  # 3 -> own
    df.loc[100:103, "event_start_s"] = 1.0  # 4 physically pre-outcome
    coefs = pd.DataFrame(
        [
            {"animal": "a", "session_date": 1, "epoch": "e", "n_events": 90},
            {"animal": "b", "session_date": 1, "epoch": "e", "n_events": 96},
        ]
    )
    r = mod.power_cohort_records(df, coefs)[0]
    assert _reconciles(r)
    assert r["n_drop_qvals_gap"] == 7
    assert r["n_drop_own_predictor_missing"] == 3
    assert r["n_events_excluded_pre_outcome"] == 4


def test_swr_power_gate_requires_timing_columns():
    """The gate names the missing columns instead of raising KeyError."""
    df = pd.DataFrame(
        {
            "animal": ["a"],
            "session_date": [1],
            "epoch": ["e"],
            **{c: [0.0] for c in mod.PREDICTORS},
            mod.POWER_COL: [0.0],
        }
    )
    with pytest.raises(ValueError, match="event_start_s"):
        mod.power_cohort_records(df, pd.DataFrame())


def test_swr_power_gate_treats_equality_as_eligible():
    """An event starting exactly at the outcome boundary is kept."""
    df = pd.DataFrame(
        {
            "animal": ["a", "a"],
            "session_date": 1,
            "epoch": "e",
            "event_start_s": [5.0, 4.999],
            "outcome_known_time": [5.0, 5.0],
            "trial_reward": [1.0, 0.0],
        }
    )
    finite, eligible = mod.post_outcome_masks(df)
    assert finite.all()
    assert eligible.tolist() == [True, False]


def test_swr_power_gate_suppresses_session_when_one_outcome_vanishes():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "animal": ["a"] * 4,
            "session_date": 1,
            "epoch": "e",
            **{column: rng.normal(size=4) for column in mod.PREDICTORS},
            mod.POWER_COL: rng.normal(size=4),
            "event_start_s": [5.0, 6.0, 4.0, 4.5],
            "outcome_known_time": [5.0] * 4,
            "trial_reward": [1.0, 1.0, 0.0, 0.0],
        }
    )

    gated, audit = mod.gate_events_after_outcome(df)

    assert gated.empty
    row = audit.iloc[0]
    assert row["n_rewarded_events_after_post_outcome_gate"] == 2
    assert row["n_omission_events_after_post_outcome_gate"] == 0
    assert not bool(row["outcome_balance_valid"])
    assert row["n_events_after_outcome_balance_guard"] == 0


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_swr_power_complete_case_guard_blocks_rewarded_only_fit(invalid_value):
    """A non-finite sole omission cannot disappear inside the OLS fit."""
    rng = np.random.default_rng(2)
    n_rewarded = 31
    n = n_rewarded + 1
    df = pd.DataFrame(
        {
            "animal": ["a"] * n,
            "session_date": 1,
            "epoch": "e",
            **{column: rng.normal(size=n) for column in mod.PREDICTORS},
            mod.POWER_COL: rng.normal(size=n),
            "event_start_s": 10.0,
            "outcome_known_time": 5.0,
            "trial_reward": [1.0] * n_rewarded + [0.0],
        }
    )
    # The chosen-arm belief is qvals-derived, so this row must remain a
    # mechanism-B qvals gap in the cohort audit even though it also triggers
    # the post-complete-case outcome-balance guard.
    df.loc[n - 1, "reward_belief_mean_chosen_arm_preoutcome"] = invalid_value

    coefs = mod.fit_power_model(df)

    assert coefs.empty
    gate = pd.DataFrame(coefs.attrs["post_outcome_gate_audit"]).iloc[0]
    assert gate["n_rewarded_events_before_complete_case"] == n_rewarded
    assert gate["n_omission_events_before_complete_case"] == 1
    assert gate["n_rewarded_events_after_complete_case"] == n_rewarded
    assert gate["n_omission_events_after_complete_case"] == 0
    assert gate["n_events_excluded_incomplete_fit"] == 1
    assert not bool(gate["outcome_balance_valid"])
    assert gate["outcome_balance_status"] == "missing_after_complete_case:omission"
    assert mod.power_cohort_records(df, coefs)[0]["n_drop_qvals_gap"] == 1


def test_power_distribution_statistics_by_class():
    """Percentiles over every event; mean and sample SD per class, NaN left out."""
    power = pd.Series([0.0, 1.0, 2.0, 3.0, 10.0, np.nan], index=[5, 4, 3, 2, 1, 0])
    is_swr = pd.Series([False, False, False, True, True, True], index=power.index)

    stats = mod.power_distribution_statistics(power.iloc[:5], is_swr.iloc[:5])

    # Linear interpolation over 0, 1, 2, 3, 10: the 25th is 1, the 75th 3.
    np.testing.assert_allclose(stats["percentiles"], [0.2, 1.0, 2.0, 3.0, 8.6])
    assert stats["n_events"] == 5
    assert stats["non_swr"] == {"n": 3, "mean": 1.0, "sd": 1.0}
    # Mean 6.5, not the median; SD over n - 1.
    assert stats["swr"]["n"] == 2
    assert stats["swr"]["mean"] == 6.5
    assert stats["swr"]["sd"] == pytest.approx(np.sqrt(24.5))
    # A NaN power counts in n but not in the class mean.
    with_nan = mod.power_distribution_statistics(power, is_swr)
    assert with_nan["swr"]["n"] == 3
    assert with_nan["swr"]["mean"] == 6.5
    assert np.isnan(with_nan["percentiles"]).all()


def test_power_distribution_sarle_is_the_hippocampus_coefficient():
    from continuum_swr_replay.analysis.hippocampus.class_or_continuum import (
        sarle_bimodality,
    )

    rng = np.random.default_rng(0)
    power = pd.Series(np.r_[rng.normal(-2, 0.3, 40), rng.normal(2, 0.3, 60)])
    is_swr = power > 0
    stats = mod.power_distribution_statistics(power, is_swr)
    assert stats["sarle_bimodality"] == sarle_bimodality(power.to_numpy())
    assert stats["non_swr"]["n"] == 40


def test_duration_power_correlation_drops_incomplete_events():
    duration = pd.Series([1.0, 2.0, 4.0, np.nan, 8.0])
    power = pd.Series([1.0, 2.0, 3.0, 100.0, np.nan])

    corr = mod.duration_power_correlation(duration, power)

    # Only the first three events have both: power rises linearly with log2
    # duration (0, 1, 2 -> 1, 2, 3), less than linearly with duration.
    assert corr["log_duration_corr"] == pytest.approx(1.0)
    assert corr["corr"] == pytest.approx(np.corrcoef([1, 2, 4], [1, 2, 3])[0, 1])
    assert corr["corr"] < 1.0


def _power_stops() -> pd.DataFrame:
    """Two stops of one session, each after a completed trip, valid for
    ``add_uncertainty_covariates``."""
    stops = pd.DataFrame(
        {
            "animal": "a",
            "session_date": 1,
            "epoch": "02_r1",
            "stop_idx": [0, 1],
            "stop_id": [10, 11],
            "is_stay": [1.0, 0.0],
            "is_rewarded_outcome": [1.0, 0.0],
            "completed_trial_reward": [1.0, 0.0],
            "occupancy_well_arm": [0.0, 1.0],
            "completed_trial_id": pd.array([0, 1], dtype="Int64"),
            "completed_trial_arm": [0.0, 1.0],
            "patch_switch_fraction_last5_completed_trips": [0.0, 0.0],
            "leaky_q_value_diff": [0.0, 0.0],
            "prev_arm": [0.0, 1.0],
            "did_last_completed_trip_switch_patch": [0.0, 0.0],
        }
    )
    for lag in range(1, 11):
        stops[f"known_reward_outcome_lag_{lag}"] = np.nan
    stops["known_reward_outcome_lag_1"] = [1.0, 0.0]
    return stops


def test_each_event_gets_its_stops_covariates_and_qvals_row():
    # Events 1 and 2 at stop 10, event 3 at stop 11; only stop 10 has a qvals row.
    events = pd.DataFrame(
        {
            "event_id": [3, 1, 2],
            "stop_id": [11, 10, 10],
            "ripple_power_z": [0.3, 0.1, 0.2],
        }
    )
    qvals = pd.DataFrame({"stop_id": [10], **{c: [7.0] for c in mod._QVALS_PREDICTORS}})

    frame = mod.assemble_power_frame(events, _power_stops(), qvals)

    assert list(frame["event_id"]) == [3, 1, 2]
    assert list(frame["known_reward_outcome_lag_1"]) == [0.0, 1.0, 1.0]
    assert list(frame.columns) == [
        *events.columns,
        *mod.STOP_COVARIATES,
        *mod._QVALS_PREDICTORS,
    ]
    for column in mod._QVALS_PREDICTORS:
        assert frame[column].isna().tolist() == [True, False, False]


def test_a_duplicated_stop_raises_instead_of_multiplying_events():
    events = pd.DataFrame({"event_id": [1], "stop_id": [10]})
    qvals = pd.DataFrame(
        {"stop_id": [10, 10], **{c: [1.0, 2.0] for c in mod._QVALS_PREDICTORS}}
    )
    with pytest.raises(pd.errors.MergeError):
        mod.assemble_power_frame(events, _power_stops(), qvals)
