"""Behavioral tests for the key-aligned harmonization parity gate."""

from __future__ import annotations

import pandas as pd

from scripts.exploratory import _verify_harmonization_parity as parity


def _stop_frame(stop_indices: list[int], n_events: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "animal": "rat",
            "day": 1,
            "epoch": "01_r1",
            "stop_idx": stop_indices,
            "stop_id": [f"stop-{index}" for index in stop_indices],
            "n_events": n_events,
        }
    )


def test_verify_table_rejects_row_loss_even_when_new_is_exact_subset(tmp_path):
    backup_path = tmp_path / "backup.pkl"
    new_path = tmp_path / "new.pkl"
    _stop_frame([0, 1, 2], [4, 5, 6]).to_pickle(backup_path)
    _stop_frame([1, 2], [5, 6]).to_pickle(new_path)

    assert not parity.verify_table("replay stop", backup_path, new_path)


def test_verify_table_accepts_identical_keyed_cohort(tmp_path):
    backup_path = tmp_path / "backup.pkl"
    new_path = tmp_path / "new.pkl"
    _stop_frame([0, 1, 2], [4, 5, 6]).to_pickle(backup_path)
    _stop_frame([0, 1, 2], [4, 5, 6]).to_pickle(new_path)

    assert parity.verify_table("replay stop", backup_path, new_path)


def test_verify_table_detects_changed_core_value_on_shared_key(tmp_path):
    backup_path = tmp_path / "backup.pkl"
    new_path = tmp_path / "new.pkl"
    _stop_frame([0, 1], [4, 5]).to_pickle(backup_path)
    _stop_frame([0, 1], [4, 99]).to_pickle(new_path)

    assert not parity.verify_table("replay stop", backup_path, new_path)


def test_main_fails_when_no_table_pair_was_checked(tmp_path, monkeypatch):
    monkeypatch.setattr(
        parity,
        "TABLES",
        [("replay stop", tmp_path / "missing-old.pkl", tmp_path / "missing-new.pkl")],
    )

    assert parity.main() == 1
