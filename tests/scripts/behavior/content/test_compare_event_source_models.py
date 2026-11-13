"""The event-source comparison runner: its paths and metadata.

The rate, patch and choice blocks and the common-cohort fit they share are
tested in ``tests/analysis/behavior/`` (``rate/test_per_stop_rate_model.py``,
``test_common_cohort.py``, ``content/test_session_fits.py``,
``choice/test_choice_prediction.py``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = (
    REPO_ROOT / "scripts" / "behavior" / "content" / "compare_event_source_models.py"
)


def _load_compare_script():
    spec = importlib.util.spec_from_file_location(
        "compare_event_source_models", SCRIPT_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rate_models_drop_response_leaking_is_swr_stop():
    # The is_swr_stop term (R3) is dropped from MODEL_PER_STOP_RATE for every source:
    # it is tautological with a per-stop event count that overlaps SWRs.
    compare = _load_compare_script()

    # The null plus the reward-history basis rung; the response-leaking
    # is_swr_stop term (R3) stays dropped for every source.
    assert tuple(compare._RATE_MODELS) == ("R0", "R1")
    assert "R3" not in compare._RATE_MODELS


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
    assert metadata["rate_models"] == ("R0", "R1")
