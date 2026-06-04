"""Regression tests for event-source comparison fairness rules."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "compare_event_source_models.py"


def _load_compare_script():
    spec = importlib.util.spec_from_file_location(
        "compare_event_source_models", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_restricted_patch_models_drop_swr_interactions():
    compare = _load_compare_script()

    assert tuple(compare._patch_models_for_source("SWR")) == ("P0", "P1", "P2", "P3")
    assert tuple(compare._patch_models_for_source("MUA")) == ("P0", "P1", "P2", "P3")
    assert "P4" not in compare._patch_models_for_source("SWR")
    assert "P4" in compare._patch_models_for_source("non_local")


def test_source_restricted_switch_models_drop_swr_interactions():
    compare = _load_compare_script()

    assert tuple(compare._patch_switch_models_for_source("SWR")) == (
        "PS0",
        "PS1",
        "PS2",
    )
    assert tuple(compare._patch_switch_models_for_source("MUA")) == (
        "PS0",
        "PS1",
        "PS2",
    )
    assert "PS3" not in compare._patch_switch_models_for_source("SWR")
    assert "PS3" in compare._patch_switch_models_for_source("non_local")


def test_swr_rate_models_drop_response_leaking_r3():
    compare = _load_compare_script()

    assert tuple(compare._rate_models_for_source("SWR")) == ("R0", "R1", "R2")
    assert "R3" not in compare._rate_models_for_source("SWR")
    assert "R3" in compare._rate_models_for_source("MUA")
    assert "R3" in compare._rate_models_for_source("non_local")


def test_choice_swr_split_is_separate_and_skipped_for_swr_source():
    compare = _load_compare_script()

    assert tuple(compare._choice_models_for_source("SWR")) == (
        "C_replay_only",
        "C_behavior",
        "C_full",
    )
    assert compare._choice_swr_split_models_for_source("SWR") == {}
    assert tuple(compare._choice_swr_split_models_for_source("non_local")) == (
        "C_behavior",
        "C_full_swr_split",
    )


def test_non_local_uses_effective_hpc_area_for_outputs(tmp_path):
    compare = _load_compare_script()

    assert compare._effective_brain_area("non_local", "mPFC") == "HPC"
    out_path = compare._comparison_output_path(tmp_path, "non_local", "mPFC")
    assert out_path.name == "non_local_HPC_comparison_fits.pkl"


def test_comparison_metadata_records_event_definition_semantics():
    compare = _load_compare_script()

    metadata = compare._comparison_metadata("SWR", "mPFC")
    assert metadata["event_source"] == "SWR"
    assert metadata["brain_area"] == "mPFC"
    assert metadata["same_identified_events"] is False
    assert metadata["rate_models"] == ("R0", "R1", "R2")


def test_common_cohort_drops_whole_units_for_union_predictors_and_zero_response():
    compare = _load_compare_script()
    models = {
        "M0": SimpleNamespace(predictors=()),
        "M1": SimpleNamespace(predictors=("x",)),
    }
    table = pd.DataFrame(
        {
            "animal": ["a"] * 9,
            "day": [1] * 9,
            "epoch": ["e"] * 9,
            "event_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "patch_count_hard": [1, 0, 0, 0, 0, 0, 1, 0, 0],
            "x": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 0.0, float("nan"), 2.0],
        }
    )

    cohort, provenance = compare._freeze_common_cohort(
        table,
        keys=("a", 1, "e"),
        analysis="test",
        models=models,
        unit_col="event_id",
        required_base=["patch_count_hard"],
        response_col="patch_count_hard",
        drop_zero_response=True,
        min_units=1,
    )

    assert tuple(cohort["event_id"].unique()) == (1,)
    assert provenance["included_ids"] == (1,)
    assert provenance["n_missing_units"] == 1
    assert provenance["n_zero_response_units"] == 1
    assert provenance["n_units_after"] == 1
    assert provenance["n_rows_after"] == 3


def test_common_cohort_success_tags_all_models_with_same_hash():
    compare = _load_compare_script()
    models = {
        "M0": SimpleNamespace(predictors=()),
        "M1": SimpleNamespace(predictors=("x",)),
    }
    table = pd.DataFrame(
        {
            "animal": ["a"] * 6,
            "day": [1] * 6,
            "epoch": ["e"] * 6,
            "event_id": [1, 1, 1, 2, 2, 2],
            "patch_count_hard": [1, 0, 0, 0, 1, 0],
            "x": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        }
    )

    def fake_fit(group, spec):
        return {
            "n_events": group["event_id"].nunique(),
            "n_params": len(spec.predictors),
        }

    fits, provenance = compare._fit_per_session_common_cohort(
        table,
        models,
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
    compare = _load_compare_script()
    models = {
        "M0": SimpleNamespace(predictors=()),
        "M1": SimpleNamespace(predictors=("x",)),
    }
    table = pd.DataFrame(
        {
            "animal": ["a"] * 6,
            "day": [1] * 6,
            "epoch": ["e"] * 6,
            "event_id": [1, 1, 1, 2, 2, 2],
            "patch_count_hard": [1, 0, 0, 0, 1, 0],
            "x": [0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
        }
    )

    def fake_fit(group, spec):  # noqa: ARG001
        if spec.predictors:
            return None
        return {"n_events": group["event_id"].nunique()}

    fits, provenance = compare._fit_per_session_common_cohort(
        table,
        models,
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
