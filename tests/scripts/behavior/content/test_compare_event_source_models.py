"""The event-source comparison runner: its rate block, paths and metadata.

The patch and choice blocks and the common-cohort fit they share are tested in
``tests/analysis/behavior/`` (``test_common_cohort.py``, ``content/test_session_fits.py``,
``choice/test_choice_prediction.py``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.behavior.design_constants import (
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_INCOMPLETE,
)

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


def _rate_stop_table(n_stops: int = 40, seed: int = 0) -> pd.DataFrame:
    """One session's complete per-stop table for the rate block.

    The first three stops carry an incomplete history (zero-filled features,
    flag 1), like the epoch prefix of the real tables.
    """
    rng = np.random.default_rng(seed)
    history = rng.uniform(0.0, 1.0, size=(n_stops, len(REWARD_HISTORY_COLUMNS)))
    history[:3] = 0.0
    incomplete = np.zeros(n_stops)
    incomplete[:3] = 1.0
    duration = rng.uniform(1.0, 10.0, size=n_stops)
    table = pd.DataFrame(
        {
            "animal": "a",
            "session_date": 1,
            "epoch": "02_r1",
            "stop_id": np.arange(n_stops),
            "n_selected_events": rng.poisson(0.3 * duration),
            "immobility_duration_s": duration,
        }
    )
    for j, column in enumerate(REWARD_HISTORY_COLUMNS):
        table[column] = history[:, j]
    table[REWARD_HISTORY_INCOMPLETE] = incomplete
    return table


def test_rate_block_fits_a_complete_stop_table():
    compare = _load_compare_script()

    fits, provenance = compare.fit_per_stop_rate_hierarchy(_rate_stop_table(), "SWR")

    assert provenance["status"].tolist() == ["fit"]
    assert tuple(fits) == ("R0", "R1")
    for fit in fits.values():
        assert len(fit) == 1
        assert fit.loc[0, "cohort_n_units"] == 40


def test_rate_block_refuses_a_stop_table_without_a_ladder_column():
    """R6 fit 0 of 242 sessions this way and still exited 0."""
    compare = _load_compare_script()
    stops = _rate_stop_table().drop(columns="reward_history_rcos_0")

    with pytest.raises(ValueError, match="rate_hierarchy") as raised:
        compare.fit_per_stop_rate_hierarchy(stops, "SWR")

    assert "reward_history_rcos_0" in str(raised.value)
    assert "'SWR'" in str(raised.value)
