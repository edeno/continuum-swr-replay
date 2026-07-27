"""Cohort-audit emitter: reconciliation, mutually-exclusive drop buckets, flags."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.cohort_audit import (
    classify_drops,
    cohort_record,
    drop_profile,
    event_drop_buckets,
    records_from_common_cohort_provenance,
    term_diagnostics,
    write_cohort_audit,
)


def test_reconciles():
    rec = cohort_record(
        "M3_armreward", "all", "event",
        n_in=100, n_modeled=80,
        sessions_in=10, sessions_modeled=9, animals_modeled=5,
        n_drop_qvals_gap=5, n_drop_own_predictor_missing=12, n_drop_zero_response=3,
    )
    assert rec["n_modeled"] == 80
    assert rec["n_drop_qvals_gap"] == 5


def test_mismatch_raises():
    with pytest.raises(AssertionError, match="does not reconcile"):
        cohort_record(
            "M3", "all", "event",
            n_in=100, n_modeled=80,  # 100 - 10 != 80
            sessions_in=10, sessions_modeled=9, animals_modeled=5,
            n_drop_own_predictor_missing=10,
        )


def test_classify_drops_mutually_exclusive():
    # unit 0 fails BOTH qvals-gap and own-predictor-missing -> counted once, in the
    # higher-priority qvals-gap bucket.
    qvals = np.array([True, False, False, False])
    own = np.array([True, True, False, False])
    zero = np.array([False, False, True, False])
    counts = classify_drops({
        "n_drop_qvals_gap": qvals,
        "n_drop_own_predictor_missing": own,
        "n_drop_zero_response": zero,
    })
    assert counts["n_drop_qvals_gap"] == 1
    assert counts["n_drop_own_predictor_missing"] == 1  # unit 1 only (unit 0 taken)
    assert counts["n_drop_zero_response"] == 1
    # Σ buckets == total distinct dropped units (0,1,2) == 3
    assert sum(counts.values()) == 3


def test_classify_then_reconcile_closes():
    counts = classify_drops({
        "n_drop_qvals_gap": np.array([True, False, False, False, False]),
        "n_drop_own_predictor_missing": np.array([True, True, False, False, False]),
    })
    rec = cohort_record(
        "M4", "all", "event",
        n_in=5, n_modeled=5 - sum(counts.values()),
        sessions_in=2, sessions_modeled=2, animals_modeled=2, **counts,
    )
    assert rec["n_modeled"] == 3


def test_common_cohort_bucket():
    # In common mode an event whose OWN predictors are present but a later rung's
    # predictor is NaN must land in n_drop_common_cohort_other, not own-missing.
    rec = cohort_record(
        "M0_null", "all", "event", cohort_mode="common",
        n_in=50, n_modeled=44,
        sessions_in=5, sessions_modeled=5, animals_modeled=3,
        n_drop_common_cohort_other=6,
    )
    assert rec["n_drop_common_cohort_other"] == 6
    assert rec["n_drop_own_predictor_missing"] == 0


def test_flags_do_not_reconcile():
    # Nonzero convergence/separation flags must NOT enter the reconciliation: the row
    # still closes on the drop buckets alone.
    rec = cohort_record(
        "M3", "swr", "event",
        n_in=40, n_modeled=35,
        sessions_in=8, sessions_modeled=8, animals_modeled=5,
        n_drop_own_predictor_missing=5,
        n_sessions_flagged_nonconverged=2, n_sessions_flagged_separated=3,
    )
    assert rec["n_sessions_flagged_nonconverged"] == 2
    assert rec["n_sessions_flagged_separated"] == 3
    assert rec["n_modeled"] == 35


def test_drop_profile_early_session():
    dropped = pd.DataFrame({
        "time_since_stop": [0.1, 0.2, 0.15, 0.3],
        "reward_outcome": [0, 0, 0, 1],
    })
    prof = drop_profile(dropped)
    assert "median_tss=0.17" in prof or "median_tss=0.18" in prof
    assert "rewarded_frac=0.250" in prof


def test_drop_profile_empty():
    assert drop_profile(pd.DataFrame()) == "no drops"


def test_write_cohort_audit_column_order(tmp_path):
    recs = [
        cohort_record(
            "M0", "all", "event", n_in=10, n_modeled=10,
            sessions_in=1, sessions_modeled=1, animals_modeled=1,
        )
    ]
    out = tmp_path / "run_cohort_audit.csv"
    df = write_cohort_audit(recs, out)
    assert out.exists()
    reread = pd.read_csv(out)
    assert list(reread.columns)[:5] == ["model", "population", "grain", "cohort_mode", "n_in"]
    assert "n_sessions_flagged_separated" in reread.columns
    assert df["n_modeled"].iloc[0] == 10


def _long(events_cols: dict, n_arms: int = 3) -> pd.DataFrame:
    """Build a long event-arm frame: events_cols maps event_id -> per-column arm values.

    Each value is a list of length n_arms (one per arm); the response is 'y'.
    """
    rows = []
    for eid, cols in events_cols.items():
        for a in range(n_arms):
            row = {"event_id": eid, "arm": a}
            for c, vals in cols.items():
                row[c] = vals[a]
            rows.append(row)
    return pd.DataFrame(rows)


def test_event_drop_buckets_common_cohort_and_qvals():
    # event 1: own predictor 'own' NaN -> own_predictor_missing
    # event 2: a value covariate q_arm NaN -> qvals_gap (priority over own/common)
    # event 3: only the OTHER rung's predictor 'other' NaN -> common_cohort_other
    # event 4: complete but response all zero -> zero_response
    # event 5: clean -> modeled
    df = _long({
        1: {"y": [1, 0, 0], "own": [float("nan"), 1.0, 1.0], "other": [1, 1, 1],
            "q_arm": [0.1, 0.2, 0.3]},
        2: {"y": [1, 0, 0], "own": [1.0, 1.0, 1.0], "other": [1, 1, 1],
            "q_arm": [float("nan"), 0.2, 0.3]},
        3: {"y": [1, 0, 0], "own": [1.0, 1.0, 1.0], "other": [float("nan"), 1, 1],
            "q_arm": [0.1, 0.2, 0.3]},
        4: {"y": [0, 0, 0], "own": [1.0, 1.0, 1.0], "other": [1, 1, 1],
            "q_arm": [0.1, 0.2, 0.3]},
        5: {"y": [1, 0, 0], "own": [1.0, 1.0, 1.0], "other": [1, 1, 1],
            "q_arm": [0.1, 0.2, 0.3]},
    })
    buckets = event_drop_buckets(
        df, event_col="event_id", response_col="y",
        own_predictors=["own", "q_arm"],  # this rung uses own + value
        union_predictors=["own", "other", "q_arm"],
    )
    assert buckets["n_drop_qvals_gap"] == 1  # event 2
    assert buckets["n_drop_own_predictor_missing"] == 1  # event 1
    assert buckets["n_drop_common_cohort_other"] == 1  # event 3 (other rung's predictor)
    assert buckets["n_drop_zero_response"] == 1  # event 4
    # Reconciliation for this rung: 5 events in, 4 dropped, 1 modeled.
    n_in = 5
    rec = cohort_record(
        "M_own_value", "all", "event", cohort_mode="common",
        n_in=n_in, n_modeled=1, sessions_in=1, sessions_modeled=1, animals_modeled=1,
        n_drop_qvals_gap=buckets["n_drop_qvals_gap"],
        n_drop_own_predictor_missing=buckets["n_drop_own_predictor_missing"],
        n_drop_common_cohort_other=buckets["n_drop_common_cohort_other"],
        n_drop_zero_response=buckets["n_drop_zero_response"],
    )
    assert rec["n_modeled"] == 1


def test_event_drop_buckets_null_rung_uses_common_bucket():
    """For a rung with NO own predictors (e.g. M0), a non-qvals union drop is
    attributed to common_cohort_other, never own_predictor_missing."""
    df = _long({
        1: {"y": [1, 0, 0], "other": [float("nan"), 1, 1]},
        2: {"y": [1, 0, 0], "other": [1, 1, 1]},
    })
    buckets = event_drop_buckets(
        df, event_col="event_id", response_col="y",
        own_predictors=[], union_predictors=["other"],
    )
    assert buckets["n_drop_own_predictor_missing"] == 0
    assert buckets["n_drop_common_cohort_other"] == 1


def test_event_drop_buckets_qvals_gap_only_for_own_value_covariate():
    """C2 regression: a base rung that does NOT use q_arm must NOT report a
    qvals_gap when a *later* rung's q_arm is the NaN cause — it is common-cohort."""
    df = _long({
        1: {"y": [1, 0, 0], "own": [1.0, 1.0, 1.0], "q_arm": [float("nan"), 0.2, 0.3]},
        2: {"y": [1, 0, 0], "own": [1.0, 1.0, 1.0], "q_arm": [0.1, 0.2, 0.3]},
    })
    # Base rung uses only 'own'; q_arm belongs to another rung in the union.
    base = event_drop_buckets(
        df, event_col="event_id", response_col="y",
        own_predictors=["own"], union_predictors=["own", "q_arm"],
    )
    assert base["n_drop_qvals_gap"] == 0  # not this rung's value covariate
    assert base["n_drop_common_cohort_other"] == 1  # event 1, dropped for the value rung

    # The value rung that DOES use q_arm attributes the same event to qvals_gap.
    value = event_drop_buckets(
        df, event_col="event_id", response_col="y",
        own_predictors=["own", "q_arm"], union_predictors=["own", "q_arm"],
    )
    assert value["n_drop_qvals_gap"] == 1
    assert value["n_drop_common_cohort_other"] == 0


def test_records_from_common_cohort_provenance():
    """The choice/comparison native provenance maps to a reconciling standardized row."""
    prov = pd.DataFrame([
        {"animal": "a", "day": 1, "epoch": "e", "n_units_before": 50,
         "n_units_after": 40, "n_missing_units": 5, "n_invalid_choice_units": 3,
         "n_nonpositive_offset_units": 0, "n_zero_response_units": 2, "status": "fit"},
        {"animal": "b", "day": 2, "epoch": "e", "n_units_before": 30,
         "n_units_after": 0, "n_missing_units": 0, "n_invalid_choice_units": 0,
         "n_nonpositive_offset_units": 0, "n_zero_response_units": 0,
         "status": "too_few_units"},
    ])
    rec = records_from_common_cohort_provenance(prov, "choice", "all")
    assert rec["n_in"] == 80
    assert rec["n_drop_own_predictor_missing"] == 8  # missing 5 + invalid-choice 3
    assert rec["n_drop_zero_response"] == 2
    assert rec["n_modeled"] == 40  # only the fit session's surviving units
    assert rec["n_excluded_session"] == 30  # the too_few session's 30 units
    assert rec["n_sessions_modeled"] == 1


def test_records_from_provenance_empty_returns_none():
    assert records_from_common_cohort_provenance(pd.DataFrame(), "choice", "all") is None


def test_term_diagnostics_counts_quasiseparation():
    # Two terms; term "a" has one quasi-separated session (|coef|>=10).
    res = pd.DataFrame({
        "model": ["C1"] * 6,
        "population": ["all"] * 6,
        "animal": ["x", "x", "y", "x", "y", "y"],
        "coef_a": [0.5, 12.0, -0.3, np.nan, 0.1, 0.2],
        "coef_b": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    })
    td = term_diagnostics(res, [("C1", "all", "a"), ("C1", "all", "b")])
    a = td[td.term == "a"].iloc[0]
    assert a["n_sessions_fit"] == 5  # one NaN excluded
    assert a["n_sessions_quasiseparated"] == 1
    assert a["n_sessions_used"] == 4
    assert a["n_animals_used"] == 2
    b = td[td.term == "b"].iloc[0]
    assert b["n_sessions_quasiseparated"] == 0
    assert b["n_sessions_used"] == 6
