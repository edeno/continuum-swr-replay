"""Regression tests for event-source comparison fairness rules."""

from __future__ import annotations

import importlib.util
from pathlib import Path

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
