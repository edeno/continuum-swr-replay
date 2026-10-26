"""Cohort-audit emitter: reconciliation, mutually-exclusive drop buckets, flags."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay.cohort_audit import (
    COHORT_COLUMNS,
    classify_drops,
    cohort_record,
    drop_profile,
    escape_note_field,
    estimator_skip_note,
    event_drop_buckets,
    records_from_common_cohort_provenance,
    term_diagnostics,
    write_cohort_audit,
)


def test_reconciles():
    rec = cohort_record(
        "M3_armreward",
        "all",
        "event",
        n_in=100,
        n_modeled=80,
        sessions_in=10,
        sessions_modeled=9,
        animals_modeled=5,
        n_drop_qvals_gap=5,
        n_drop_own_predictor_missing=12,
        n_drop_zero_response=3,
    )
    assert rec["n_modeled"] == 80
    assert rec["n_drop_qvals_gap"] == 5


def test_mismatch_raises():
    with pytest.raises(AssertionError, match="does not reconcile"):
        cohort_record(
            "M3",
            "all",
            "event",
            n_in=100,
            n_modeled=80,  # 100 - 10 != 80
            sessions_in=10,
            sessions_modeled=9,
            animals_modeled=5,
            n_drop_own_predictor_missing=10,
        )


def test_classify_drops_mutually_exclusive():
    # unit 0 fails BOTH qvals-gap and own-predictor-missing -> counted once, in the
    # higher-priority qvals-gap bucket.
    qvals = np.array([True, False, False, False])
    own = np.array([True, True, False, False])
    zero = np.array([False, False, True, False])
    counts = classify_drops(
        {
            "n_drop_qvals_gap": qvals,
            "n_drop_own_predictor_missing": own,
            "n_drop_zero_response": zero,
        }
    )
    assert counts["n_drop_qvals_gap"] == 1
    assert counts["n_drop_own_predictor_missing"] == 1  # unit 1 only (unit 0 taken)
    assert counts["n_drop_zero_response"] == 1
    # Σ buckets == total distinct dropped units (0,1,2) == 3
    assert sum(counts.values()) == 3


def test_classify_then_reconcile_closes():
    counts = classify_drops(
        {
            "n_drop_qvals_gap": np.array([True, False, False, False, False]),
            "n_drop_own_predictor_missing": np.array([True, True, False, False, False]),
        }
    )
    rec = cohort_record(
        "M4",
        "all",
        "event",
        n_in=5,
        n_modeled=5 - sum(counts.values()),
        sessions_in=2,
        sessions_modeled=2,
        animals_modeled=2,
        **counts,
    )
    assert rec["n_modeled"] == 3


def test_common_cohort_bucket():
    # In common mode an event whose OWN predictors are present but a later rung's
    # predictor is NaN must land in n_drop_common_cohort_other, not own-missing.
    rec = cohort_record(
        "M0_null",
        "all",
        "event",
        cohort_mode="common",
        n_in=50,
        n_modeled=44,
        sessions_in=5,
        sessions_modeled=5,
        animals_modeled=3,
        n_drop_common_cohort_other=6,
    )
    assert rec["n_drop_common_cohort_other"] == 6
    assert rec["n_drop_own_predictor_missing"] == 0


def test_flags_do_not_reconcile():
    # Nonzero convergence/separation flags must NOT enter the reconciliation: the row
    # still closes on the drop buckets alone.
    rec = cohort_record(
        "M3",
        "swr",
        "event",
        n_in=40,
        n_modeled=35,
        sessions_in=8,
        sessions_modeled=8,
        animals_modeled=5,
        n_drop_own_predictor_missing=5,
        n_sessions_flagged_nonconverged=2,
        n_sessions_flagged_separated=3,
    )
    assert rec["n_sessions_flagged_nonconverged"] == 2
    assert rec["n_sessions_flagged_separated"] == 3
    assert rec["n_modeled"] == 35


def test_drop_profile_early_session():
    dropped = pd.DataFrame(
        {
            "time_since_stop_start_s": [0.1, 0.2, 0.15, 0.3],
            "is_rewarded_outcome": [0, 0, 0, 1],
        }
    )
    prof = drop_profile(dropped)
    assert "median_tss=0.17" in prof or "median_tss=0.18" in prof
    assert "rewarded_frac=0.250" in prof


def test_drop_profile_empty():
    assert drop_profile(pd.DataFrame()) == "no drops"


def test_write_cohort_audit_column_order(tmp_path):
    recs = [
        cohort_record(
            "M0",
            "all",
            "event",
            n_in=10,
            n_modeled=10,
            sessions_in=1,
            sessions_modeled=1,
            animals_modeled=1,
        )
    ]
    out = tmp_path / "run_cohort_audit.csv"
    df = write_cohort_audit(recs, out)
    assert out.exists()
    reread = pd.read_csv(out)
    assert list(reread.columns)[:5] == [
        "model",
        "population",
        "grain",
        "cohort_mode",
        "n_in",
    ]
    assert "n_sessions_flagged_separated" in reread.columns
    assert df["n_modeled"].iloc[0] == 10


def test_write_cohort_audit_extra_columns_follow_the_canonical_ones(tmp_path):
    rec = cohort_record(
        "M0",
        "all",
        "event",
        n_in=10,
        n_modeled=10,
        sessions_in=1,
        sessions_modeled=1,
        animals_modeled=1,
    )
    rec.update(n_b=2, n_a=1, not_written=3)
    out = tmp_path / "run_cohort_audit.csv"
    write_cohort_audit([rec], out, extra_columns=("n_b", "n_a"))
    assert list(pd.read_csv(out).columns) == [*COHORT_COLUMNS, "n_b", "n_a"]


def _long(events_cols: dict, n_arms: int = 3) -> pd.DataFrame:
    """Build a long event-arm frame: events_cols maps event_id -> per-column arm values.

    Each value is a list of length n_arms (one per arm); the response is 'y'.
    """
    rows = []
    for eid, cols in events_cols.items():
        for a in range(n_arms):
            row = {"event_id": eid, "candidate_arm": a}
            for c, vals in cols.items():
                row[c] = vals[a]
            rows.append(row)
    return pd.DataFrame(rows)


def test_event_drop_buckets_common_cohort_and_qvals():
    # event 1: own predictor 'own' NaN -> own_predictor_missing
    # event 2: a value covariate q_arm NaN on a stop with no qvals row -> qvals_gap
    #          (priority over own/common)
    # event 3: only the OTHER rung's predictor 'other' NaN -> common_cohort_other
    # event 4: complete but response all zero -> zero_response
    # event 5: clean -> modeled
    df = _long(
        {
            1: {
                "has_qvals_row": [True, True, True],
                "y": [1, 0, 0],
                "own": [float("nan"), 1.0, 1.0],
                "other": [1, 1, 1],
                "reward_belief_mean_candidate_arm_preoutcome": [0.1, 0.2, 0.3],
            },
            2: {
                "has_qvals_row": [False, False, False],
                "y": [1, 0, 0],
                "own": [1.0, 1.0, 1.0],
                "other": [1, 1, 1],
                "reward_belief_mean_candidate_arm_preoutcome": [float("nan"), 0.2, 0.3],
            },
            3: {
                "has_qvals_row": [True, True, True],
                "y": [1, 0, 0],
                "own": [1.0, 1.0, 1.0],
                "other": [float("nan"), 1, 1],
                "reward_belief_mean_candidate_arm_preoutcome": [0.1, 0.2, 0.3],
            },
            4: {
                "has_qvals_row": [True, True, True],
                "y": [0, 0, 0],
                "own": [1.0, 1.0, 1.0],
                "other": [1, 1, 1],
                "reward_belief_mean_candidate_arm_preoutcome": [0.1, 0.2, 0.3],
            },
            5: {
                "has_qvals_row": [True, True, True],
                "y": [1, 0, 0],
                "own": [1.0, 1.0, 1.0],
                "other": [1, 1, 1],
                "reward_belief_mean_candidate_arm_preoutcome": [0.1, 0.2, 0.3],
            },
        }
    )
    buckets = event_drop_buckets(
        df,
        event_col="event_id",
        response_col="y",
        own_predictors=[
            "own",
            "reward_belief_mean_candidate_arm_preoutcome",
        ],  # this rung uses own + value
        union_predictors=[
            "own",
            "other",
            "reward_belief_mean_candidate_arm_preoutcome",
        ],
    )
    assert buckets["n_drop_qvals_gap"] == 1  # event 2
    assert buckets["n_drop_own_predictor_missing"] == 1  # event 1
    assert (
        buckets["n_drop_common_cohort_other"] == 1
    )  # event 3 (other rung's predictor)
    assert buckets["n_drop_zero_response"] == 1  # event 4
    # Reconciliation for this rung: 5 events in, 4 dropped, 1 modeled.
    n_in = 5
    rec = cohort_record(
        "M_own_value",
        "all",
        "event",
        cohort_mode="common",
        n_in=n_in,
        n_modeled=1,
        sessions_in=1,
        sessions_modeled=1,
        animals_modeled=1,
        n_drop_qvals_gap=buckets["n_drop_qvals_gap"],
        n_drop_own_predictor_missing=buckets["n_drop_own_predictor_missing"],
        n_drop_common_cohort_other=buckets["n_drop_common_cohort_other"],
        n_drop_zero_response=buckets["n_drop_zero_response"],
    )
    assert rec["n_modeled"] == 1


def test_event_drop_buckets_null_rung_uses_common_bucket():
    """For a rung with NO own predictors (e.g. M0), a non-qvals union drop is
    attributed to common_cohort_other, never own_predictor_missing."""
    df = _long(
        {
            1: {"y": [1, 0, 0], "other": [float("nan"), 1, 1]},
            2: {"y": [1, 0, 0], "other": [1, 1, 1]},
        }
    )
    buckets = event_drop_buckets(
        df,
        event_col="event_id",
        response_col="y",
        own_predictors=[],
        union_predictors=["other"],
    )
    assert buckets["n_drop_own_predictor_missing"] == 0
    assert buckets["n_drop_common_cohort_other"] == 1


def test_event_drop_buckets_qvals_gap_only_for_own_value_covariate():
    """C2 regression: a base rung that does NOT use q_arm must NOT report a
    qvals_gap when a *later* rung's q_arm is the NaN cause — it is common-cohort."""
    df = _long(
        {
            1: {
                "has_qvals_row": [False, False, False],
                "y": [1, 0, 0],
                "own": [1.0, 1.0, 1.0],
                "reward_belief_mean_candidate_arm_preoutcome": [float("nan"), 0.2, 0.3],
            },
            2: {
                "has_qvals_row": [True, True, True],
                "y": [1, 0, 0],
                "own": [1.0, 1.0, 1.0],
                "reward_belief_mean_candidate_arm_preoutcome": [0.1, 0.2, 0.3],
            },
        }
    )
    # Base rung uses only 'own'; q_arm belongs to another rung in the union.
    base = event_drop_buckets(
        df,
        event_col="event_id",
        response_col="y",
        own_predictors=["own"],
        union_predictors=["own", "reward_belief_mean_candidate_arm_preoutcome"],
    )
    assert base["n_drop_qvals_gap"] == 0  # not this rung's value covariate
    assert (
        base["n_drop_common_cohort_other"] == 1
    )  # event 1, dropped for the value rung

    # The value rung that DOES use q_arm attributes the same event to qvals_gap.
    value = event_drop_buckets(
        df,
        event_col="event_id",
        response_col="y",
        own_predictors=["own", "reward_belief_mean_candidate_arm_preoutcome"],
        union_predictors=["own", "reward_belief_mean_candidate_arm_preoutcome"],
    )
    assert value["n_drop_qvals_gap"] == 1
    assert value["n_drop_common_cohort_other"] == 0


def test_every_contrast_belief_predictor_is_booked_to_the_qvals_gap():
    """The per-arm belief mean and variance share the qvals tail's NaN mask, so a
    CS6 / C13 drop of a tail event is a qvals gap, not own-predictor missingness."""
    from continuum_swr_replay.analysis.replay.cohort_audit import QVALS_GAP_COLS
    from continuum_swr_replay.analysis.replay.model_specs import (
        SWR_CONTRAST_CONTENT_ARM_LADDER,
    )

    belief = {
        predictor
        for predictors, _scope in SWR_CONTRAST_CONTENT_ARM_LADDER.values()
        for predictor in predictors
        if "reward_belief" in predictor and "candidate_arm" in predictor
    }

    assert "swr_x_reward_belief_variance_candidate_arm_preoutcome_z" in belief
    assert all(predictor in QVALS_GAP_COLS for predictor in belief)


@pytest.mark.parametrize(
    "column",
    [
        "task_reward_probability_candidate_arm",
        "reward_belief_variance_candidate_arm_preoutcome_z",
        "swr_x_reward_belief_mean_candidate_arm_preoutcome",
        "reward_belief_mean_candidate_arm_preoutcome_patch",
        "reward_belief_mean_candidate_arm_preoutcome_patch_z",
        "reward_belief_mean_candidate_arm_preoutcome_z_patch",
        "rpe_postoutcome_z_x_is_current_arm",
        "trials_since_block_change_z_x_is_current_arm",
        "periswitch_index_trials_z_x_is_current_patch",
        "periswitch_index_trials_z_x_is_periswitch_post_x_is_current_arm",
    ],
)
def test_qvals_gap_includes_every_variant_of_a_gated_covariate(column):
    """``_z``, ``swr_x_``, ``_patch`` and ``_x_`` products of a gated base are in."""
    from continuum_swr_replay.analysis.replay.cohort_audit import QVALS_GAP_COLS

    assert column in QVALS_GAP_COLS


@pytest.mark.parametrize(
    "column",
    [
        # qvals columns that are also NaN on stops that HAVE a qvals row (masked
        # next to a behavioral-model trial gap): such a drop is not a qvals gap.
        "relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm",
        "reward_belief_mean_update_l1_all_ports_postoutcome_z_x_is_current_arm",
        # not from qvals_table at all
        "reward_fraction_candidate_arm_last10_visits",
        "swr_x_is_current_arm",
        "is_current_patch",
        None,
    ],
)
def test_qvals_gap_excludes_columns_missing_for_other_reasons(column):
    from continuum_swr_replay.analysis.replay.cohort_audit import QVALS_GAP_COLS

    assert column not in QVALS_GAP_COLS


def test_every_qvals_produced_covariate_declares_its_gap_status():
    """A new qvals covariate must be classified, not silently left out of the gap.

    Every catalogued covariate whose producer is the qvals pipeline is
    ``qvals_sourced`` (and row-gated or not) or named here with the reason it
    is not.
    """
    from continuum_swr_replay.analysis.replay.covariates.definitions import (
        COVARIATE_DEFINITIONS,
        QVALS_ROW_GATED_COVARIATES,
        QVALS_SOURCED_COVARIATES,
    )

    sourced_not_gated = {
        # masked on qvals rows next to a behavioral-model trial gap
        "relative_value_switch_minus_stay_preoutcome",
        "reward_belief_mean_update_l1_all_ports_postoutcome",
        "reward_belief_mean_update_l1_alternative_patches_postoutcome",
    }
    not_sourced = {
        # carried on stops with no qvals row too; NaN until the outcome is known
        "is_rewarded_outcome",
        "is_omitted_outcome",
    }
    qvals_produced = {
        name
        for name, definition in COVARIATE_DEFINITIONS.items()
        if "qvals" in definition.producer
    }
    assert QVALS_ROW_GATED_COVARIATES <= QVALS_SOURCED_COVARIATES <= qvals_produced
    assert qvals_produced - not_sourced == QVALS_SOURCED_COVARIATES
    assert QVALS_SOURCED_COVARIATES - sourced_not_gated == QVALS_ROW_GATED_COVARIATES


@pytest.mark.parametrize(
    ("column", "sourced"),
    [
        ("relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm", True),
        ("reward_belief_mean_update_l1_all_ports_postoutcome_z_x_is_current_arm", True),
        ("rpe_postoutcome_z_x_is_current_arm", True),
        ("task_reward_probability_candidate_arm_patch", True),
        ("is_rewarded_outcome", False),
        ("reward_fraction_candidate_arm_last10_visits", False),
        ("swr_x_is_current_arm", False),
        (None, False),
    ],
)
def test_qvals_sourced_columns_include_the_masked_qvals_columns(column, sourced):
    from continuum_swr_replay.analysis.replay.cohort_audit import (
        is_qvals_sourced_column,
    )

    assert is_qvals_sourced_column(column) is sourced


def _masked_value_frame() -> pd.DataFrame:
    """Three events of a C7-style rung (one masked qvals scalar, ``v``).

    event 1: ``v`` NaN, stop has no qvals row       -> qvals gap
    event 2: ``v`` NaN, stop HAS a qvals row (masked) -> own-predictor missing
    event 3: clean                                   -> modeled
    """
    v = "relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm"
    return _long(
        {
            1: {
                "y": [1, 0, 0],
                "is_current_arm": [1.0, 0.0, 0.0],
                v: [float("nan")] * 3,
                "has_qvals_row": [False] * 3,
            },
            2: {
                "y": [1, 0, 0],
                "is_current_arm": [1.0, 0.0, 0.0],
                v: [float("nan")] * 3,
                "has_qvals_row": [True] * 3,
            },
            3: {
                "y": [1, 0, 0],
                "is_current_arm": [1.0, 0.0, 0.0],
                v: [0.5, 0.0, 0.0],
                "has_qvals_row": [True] * 3,
            },
        }
    )


def test_event_drop_buckets_books_the_qvals_gap_per_row():
    """A masked qvals column on a mapped stop is own-predictor missingness."""
    frame = _masked_value_frame()
    predictors = [
        "is_current_arm",
        "relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm",
    ]

    buckets = event_drop_buckets(
        frame,
        event_col="event_id",
        response_col="y",
        own_predictors=predictors,
        union_predictors=predictors,
    )

    assert buckets["n_drop_qvals_gap"] == 1
    assert buckets["n_drop_own_predictor_missing"] == 1
    assert buckets["dropped_event_ids"] == {1, 2}


def test_prepare_session_books_the_qvals_gap_per_row(monkeypatch):
    from continuum_swr_replay.analysis.replay import contrast_aggregation

    monkeypatch.setattr(contrast_aggregation, "MIN_EVENTS", 1)
    predictors = [
        "is_current_arm",
        "relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm",
    ]

    surv, prep = contrast_aggregation.prepare_session(
        _masked_value_frame(), predictors, response_col="y"
    )

    assert (prep["n_qvals"], prep["n_own"], prep["n_surv"]) == (1, 1, 1)
    assert set(surv["event_id"]) == {3}


def test_qvals_gap_needs_the_row_flag_only_for_a_qvals_predictor():
    from continuum_swr_replay.analysis.replay.cohort_audit import (
        qvals_gap_event_ids,
    )

    frame = _masked_value_frame().drop(columns="has_qvals_row")
    assert qvals_gap_event_ids(frame, ["is_current_arm"]) == set()
    with pytest.raises(KeyError, match="has_qvals_row"):
        qvals_gap_event_ids(
            frame,
            ["relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm"],
        )
    with pytest.raises(TypeError, match="boolean"):
        qvals_gap_event_ids(
            frame.assign(has_qvals_row=1.0),
            ["relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm"],
        )


def test_attach_has_qvals_row_reads_a_row_gated_base_column():
    from continuum_swr_replay.analysis.replay.cohort_audit import attach_has_qvals_row

    frame = _long(
        {
            1: {"task_reward_probability_candidate_arm": [float("nan")] * 3},
            2: {"task_reward_probability_candidate_arm": [0.5, 0.2, 0.0]},
        }
    )
    out = attach_has_qvals_row(
        frame, from_column="task_reward_probability_candidate_arm"
    )
    assert out.groupby("event_id")["has_qvals_row"].first().to_dict() == {
        1: False,
        2: True,
    }
    assert out["has_qvals_row"].dtype == bool

    # Not row-gated, a product, absent, or NaN on only part of an event: refused.
    with pytest.raises(ValueError, match="row-gated"):
        attach_has_qvals_row(
            frame.assign(relative_value_switch_minus_stay_preoutcome=0.0),
            from_column="relative_value_switch_minus_stay_preoutcome",
        )
    with pytest.raises(ValueError, match="row-gated"):
        attach_has_qvals_row(frame, from_column="rpe_postoutcome_z_x_is_current_arm")
    with pytest.raises(KeyError, match="not in the frame"):
        attach_has_qvals_row(frame, from_column="rpe_postoutcome")
    partial = frame.copy()
    partial.loc[3, "task_reward_probability_candidate_arm"] = float("nan")
    with pytest.raises(ValueError, match="some but not all rows"):
        attach_has_qvals_row(
            partial, from_column="task_reward_probability_candidate_arm"
        )


def test_records_from_common_cohort_provenance():
    """The choice/comparison native provenance maps to a reconciling standardized row."""
    prov = pd.DataFrame(
        [
            {
                "animal": "a",
                "day": 1,
                "epoch": "e",
                "n_units_before": 50,
                "n_units_after": 40,
                "n_missing_units": 5,
                "n_invalid_choice_units": 3,
                "n_nonfinite_units": 1,
                "n_nonpositive_offset_units": 0,
                "n_zero_response_units": 1,
                "status": "fit",
            },
            {
                "animal": "b",
                "day": 2,
                "epoch": "e",
                "n_units_before": 30,
                "n_units_after": 0,
                "n_missing_units": 0,
                "n_invalid_choice_units": 0,
                "n_nonpositive_offset_units": 0,
                "n_zero_response_units": 0,
                "status": "too_few_units",
            },
        ]
    )
    rec = records_from_common_cohort_provenance(prov, "choice", "all")
    assert rec["n_in"] == 80
    assert rec["n_drop_own_predictor_missing"] == 8  # missing 5 + invalid-choice 3
    assert rec["n_drop_nonfinite"] == 1
    assert rec["n_drop_zero_response"] == 1
    assert rec["n_modeled"] == 40  # only the fit session's surviving units
    assert rec["n_excluded_session"] == 30  # the too_few session's 30 units
    assert rec["n_sessions_modeled"] == 1


def test_event_drop_buckets_attributes_infinity_explicitly():
    frame = pd.DataFrame(
        {
            "event_id": [1, 1, 2, 2],
            "y": [1, 0, 1, 0],
            "x": [0.0, 0.0, np.inf, 1.0],
        }
    )

    buckets = event_drop_buckets(
        frame,
        event_col="event_id",
        response_col="y",
        own_predictors=["x"],
        union_predictors=["x"],
    )

    assert buckets["n_drop_nonfinite"] == 1
    assert buckets["dropped_event_ids"] == {2}


def test_records_from_provenance_empty_returns_none():
    assert (
        records_from_common_cohort_provenance(pd.DataFrame(), "choice", "all") is None
    )


def test_term_diagnostics_counts_quasiseparation():
    # Two terms; term "a" has one quasi-separated session (|coef|>=10).
    res = pd.DataFrame(
        {
            "model": ["C1"] * 6,
            "population": ["all"] * 6,
            "animal": ["x", "x", "y", "x", "y", "y"],
            "coef_a": [0.5, 12.0, -0.3, np.nan, 0.1, 0.2],
            "coef_b": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    td = term_diagnostics(res, [("C1", "all", "a"), ("C1", "all", "b")])
    a = td[td.term == "a"].iloc[0]
    assert a["n_sessions_fit"] == 5  # one NaN excluded
    assert a["n_sessions_quasiseparated"] == 1
    assert a["n_sessions_used"] == 4
    assert a["n_animals_used"] == 2
    b = td[td.term == "b"].iloc[0]
    assert b["n_sessions_quasiseparated"] == 0
    assert b["n_sessions_used"] == 6


def test_an_embedded_message_cannot_impersonate_a_note_separator():
    """A fitter's refusal text is quoted, not authored: it may carry ``,``/``;``."""
    message = (
        "dm_design:within-event design is rank deficient: at least one "
        "predictor is constant within every event (or collinear) in "
        "['is_current_arm', 'is_prev_arm']"
    )
    cleaned = escape_note_field(message)

    assert ";" not in cleaned
    assert "," not in cleaned
    # Separators become spaces, never nothing: the words must not run together.
    assert "is_current_arm'  'is_prev_arm'" not in cleaned
    assert "is_current_arm' 'is_prev_arm'" in cleaned
    assert "rank deficient" in cleaned


def test_skip_note_summarizes_classes_and_names_only_the_refusals():
    """One policy for every runner: classes summarized, refusals named."""
    note = estimator_skip_note(
        [
            ("chimi/20200212/02_r1", "fit_none"),
            ("j16/20210714/04_r2", "dm_design:rank deficient, in ['a', 'b']"),
            ("senor/20201027/02_r1", "pinned_fit_none"),
        ],
        {"dm_design", "pinned_fit_none"},
    )

    assert note.startswith("no fit: dm_design, fit_none, pinned_fit_none")
    assert "j16/20210714/04_r2" in note
    assert "senor/20201027/02_r1" in note
    # A routine short session is counted by class, not named.
    assert "chimi/20200212/02_r1" not in note
    # The refusal's own text carries none of the note's separators.
    assert "rank deficient in ['a' 'b']" in note


def test_skip_note_is_empty_when_nothing_was_skipped():
    assert estimator_skip_note([], {"dm_design"}) == ""
