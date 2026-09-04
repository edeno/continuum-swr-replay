"""Tests for the shared keyed table-parity core.

`diff_table` is the one implementation of "does this table match that table, row
for row" used by both the regeneration manifest's baseline gate and the golden-
baseline refactor check. These pin the properties both callers rely on: exact
value drift is caught, NaN-vs-None does not false-positive, structural drift
raises, and a declared drift column is allowed through.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.table_io import write_table
from continuum_swr_replay.table_parity import compare_table_roots, diff_table


def _stop_frame(value: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "animal": ["chimi", "chimi"],
            "session_date": [20200212, 20200212],
            "epoch": ["02_r1", "02_r1"],
            "stop_id": [10, 11],
            "n_selected_events": [1.0, value],
        }
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [1, 2, 3],
            "value": [1.0, 2.0, np.nan],
            "label": ["a", "b", None],
        }
    )


def test_identical_frames_report_no_drift():
    report = diff_table(_frame(), _frame(), keys=["event_id"], label="t")
    assert report == {}


def test_nan_vs_none_does_not_false_positive():
    a = _frame()
    b = _frame()
    # object column: one side stores the missing entry as np.nan, the other None
    b["label"] = np.array(["a", "b", np.nan], dtype=object)
    assert diff_table(a, b, keys=["event_id"], label="t") == {}


def test_value_drift_in_undeclared_column_raises():
    a = _frame()
    b = _frame()
    b.loc[0, "value"] = 1.5
    with pytest.raises(ValueError, match="undeclared"):
        diff_table(a, b, keys=["event_id"], label="t")


def test_value_drift_passes_when_declared():
    a = _frame()
    b = _frame()
    b.loc[0, "value"] = 1.5
    report = diff_table(
        a, b, keys=["event_id"], label="t",
        expected_drift_columns=frozenset({"value"}),
    )
    assert "value" in report


def test_row_count_drift_raises():
    a = _frame()
    b = _frame().iloc[:2]
    with pytest.raises(ValueError, match="rows"):
        diff_table(a, b, keys=["event_id"], label="t")


def test_column_set_drift_raises():
    a = _frame()
    b = _frame().drop(columns=["label"])
    with pytest.raises(ValueError, match="column set"):
        diff_table(a, b, keys=["event_id"], label="t")


def test_key_set_drift_raises():
    a = _frame()
    b = _frame()
    b.loc[2, "event_id"] = 99
    with pytest.raises(ValueError, match="key set"):
        diff_table(a, b, keys=["event_id"], label="t")


def test_atol_zero_catches_tiny_numeric_drift():
    a = _frame()
    b = _frame()
    b.loc[0, "value"] = 1.0 + 1e-12
    with pytest.raises(ValueError, match="undeclared"):
        diff_table(a, b, keys=["event_id"], label="t", atol=0.0)


def test_compare_table_roots_detects_drift(tmp_path):
    """Root-level compare keys each table by its registered row key."""
    base = tmp_path / "base"
    cand = tmp_path / "cand"
    base.mkdir()
    cand.mkdir()
    write_table(_stop_frame(2.0), base / "stop_table")
    write_table(_stop_frame(2.0), cand / "stop_table")
    # identical -> no drift
    report = compare_table_roots(cand, base, ("stop_table",), atol=0.0)
    assert report["stop_table"] == {}
    # mutate one value in the candidate -> raises on the stop row key
    write_table(_stop_frame(9.0), cand / "stop_table")
    with pytest.raises(ValueError, match="undeclared"):
        compare_table_roots(cand, base, ("stop_table",), atol=0.0)
