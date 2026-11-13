"""A block of models fit per session on one frozen complete-case cohort."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay_models.common_cohort import (
    fit_per_session_common_cohort,
    freeze_session_cohort,
    hash_unit_ids,
)

MODELS = {
    "M0": SimpleNamespace(predictors=()),
    "M1": SimpleNamespace(predictors=("x",)),
}


def _two_event_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "animal": ["a"] * 6,
            "session_date": [1] * 6,
            "epoch": ["e"] * 6,
            "event_id": [1, 1, 1, 2, 2, 2],
            "patch_count_hard": [1, 0, 0, 0, 1, 0],
            "x": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        }
    )


def test_common_cohort_drops_whole_units_for_union_predictors_and_zero_response():
    table = pd.DataFrame(
        {
            "animal": ["a"] * 9,
            "session_date": [1] * 9,
            "epoch": ["e"] * 9,
            "event_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "patch_count_hard": [1, 0, 0, 0, 0, 0, 1, 0, 0],
            "x": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, float("nan"), 2.0],
        }
    )

    cohort, provenance = freeze_session_cohort(
        table,
        keys=("a", 1, "e"),
        analysis="test",
        models=MODELS,
        unit_col="event_id",
        required_base=["patch_count_hard"],
        response_col="patch_count_hard",
        drop_zero_response=True,
        min_units=1,
    )

    assert tuple(cohort["event_id"].unique()) == (1,)
    assert provenance["included_ids"] == (1,)
    assert provenance["included_ids_hash"] == hash_unit_ids((1,))
    assert provenance["n_units_before"] == 3
    assert provenance["n_missing_units"] == 1
    assert provenance["n_zero_response_units"] == 1
    assert provenance["n_units_after"] == 1
    assert provenance["n_rows_after"] == 3
    assert provenance["status"] == "ok"


def test_a_cohort_below_the_minimum_is_too_small():
    _cohort, provenance = freeze_session_cohort(
        _two_event_table(),
        keys=("a", 1, "e"),
        analysis="test",
        models=MODELS,
        unit_col="event_id",
        required_base=["patch_count_hard"],
        response_col="patch_count_hard",
        drop_zero_response=True,
        min_units=3,
    )
    assert provenance["n_units_after"] == 2
    assert provenance["status"] == "too_few_units"


def test_the_unit_hash_is_of_the_sorted_ids_one_per_line():
    assert hash_unit_ids((1, 2)) == hashlib.sha256(b"1\n2\n").hexdigest()
    assert hash_unit_ids(()) == hashlib.sha256(b"").hexdigest()


def test_common_cohort_success_tags_all_models_with_same_hash():
    def fake_fit(group, spec):
        return {
            "n_events": group["event_id"].nunique(),
            "n_params": len(spec.predictors),
        }

    fits, provenance = fit_per_session_common_cohort(
        _two_event_table(),
        MODELS,
        fake_fit,
        analysis="test",
        unit_col="event_id",
        required_base=["patch_count_hard"],
        response_col="patch_count_hard",
        drop_zero_response=True,
        min_units=1,
    )

    hash0 = fits["M0"].loc[0, "cohort_included_ids_hash"]
    hash1 = fits["M1"].loc[0, "cohort_included_ids_hash"]
    assert hash0 == hash1 == provenance.loc[0, "included_ids_hash"]
    assert fits["M0"].loc[0, "cohort_n_units"] == 2
    assert fits["M1"].loc[0, "cohort_n_units"] == 2
    assert provenance.loc[0, "status"] == "fit"


def test_common_cohort_drops_session_if_any_model_fails():
    def fake_fit(group, spec):
        if spec.predictors:
            return None
        return {"n_events": group["event_id"].nunique()}

    fits, provenance = fit_per_session_common_cohort(
        _two_event_table(),
        MODELS,
        fake_fit,
        analysis="test",
        unit_col="event_id",
        required_base=["patch_count_hard"],
        response_col="patch_count_hard",
        drop_zero_response=True,
        min_units=1,
    )

    assert fits["M0"].empty
    assert fits["M1"].empty
    assert provenance.loc[0, "status"] == "fit_failed"
    assert provenance.loc[0, "failed_models"] == ("M1",)


def test_a_missing_column_stops_the_block_before_any_session_is_fit():
    table = pd.DataFrame(
        {
            "animal": ["a"] * 6,
            "session_date": [1] * 6,
            "epoch": ["e1"] * 3 + ["e2"] * 3,
            "event_id": [1, 1, 1, 2, 2, 2],
            "patch_count_hard": [1, 0, 0, 0, 1, 0],
        }
    )
    calls = []

    def fake_fit(group, spec):
        calls.append(spec)
        return {"n_events": group["event_id"].nunique()}

    with pytest.raises(ValueError, match=r"test.*\['x'\]"):
        fit_per_session_common_cohort(
            table,
            MODELS,
            fake_fit,
            analysis="test",
            unit_col="event_id",
            required_base=["patch_count_hard"],
            response_col="patch_count_hard",
            drop_zero_response=True,
            min_units=1,
        )
    assert calls == []
