"""Behavioral tests for the key-aligned harmonization parity gate."""

from __future__ import annotations

import pandas as pd

from continuum_swr_replay.analysis.replay.covariate_registry import (
    finalize_covariate_schema,
)
from scripts.exploratory import _verify_harmonization_parity as parity


def _stop_frame(stop_indices: list[int], n_events: list[int]) -> pd.DataFrame:
    """A canonical stop frame.

    ``finalize_covariate_schema`` translates the ``qvals`` grain only, so a
    stop-grain fixture has to be spelled canonically to reach the stamp.
    """
    return pd.DataFrame(
        {
            "animal": "rat",
            "session_date": 1,
            "epoch": "01_r1",
            "stop_idx": stop_indices,
            "stop_id": [f"stop-{index}" for index in stop_indices],
            "n_selected_events": n_events,
        }
    )


def _write_pair(
    tmp_path,
    baseline: pd.DataFrame,
    regenerated: pd.DataFrame,
):
    baseline_path = tmp_path / "baseline.pkl"
    regenerated_path = tmp_path / "regenerated.pkl"
    baseline.to_pickle(baseline_path)
    finalize_covariate_schema(regenerated, table="stop").to_pickle(regenerated_path)
    return baseline_path, regenerated_path


def test_verify_table_rejects_row_loss_even_when_new_is_exact_subset(tmp_path):
    backup_path, new_path = _write_pair(
        tmp_path,
        _stop_frame([0, 1, 2], [4, 5, 6]),
        _stop_frame([1, 2], [5, 6]),
    )

    assert not parity.verify_table("replay stop", backup_path, new_path, table="stop")


def test_verify_table_accepts_identical_keyed_cohort(tmp_path):
    frame = _stop_frame([0, 1, 2], [4, 5, 6])
    backup_path, new_path = _write_pair(tmp_path, frame, frame.copy())

    assert parity.verify_table("replay stop", backup_path, new_path, table="stop")


def test_verify_table_detects_changed_core_value_on_shared_key(tmp_path):
    backup_path, new_path = _write_pair(
        tmp_path,
        _stop_frame([0, 1], [4, 5]),
        _stop_frame([0, 1], [4, 99]),
    )

    assert not parity.verify_table("replay stop", backup_path, new_path, table="stop")


def test_main_fails_when_no_table_pair_was_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(
        parity,
        "TABLES",
        [
            (
                "replay stop",
                "stop",
                tmp_path / "missing-old.pkl",
                tmp_path / "missing-new.pkl",
            )
        ],
    )

    assert parity.main() == 1
