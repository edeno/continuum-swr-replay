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
        a,
        b,
        keys=["event_id"],
        label="t",
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


def test_declared_added_column_passes_and_is_not_compared():
    """A candidate that gained a column matches a baseline that predates it."""
    baseline = _frame()
    candidate = _frame()
    candidate["reward_history_rcos_0"] = [0.1, 0.2, 0.3]
    report = diff_table(
        candidate,
        baseline,
        keys=["event_id"],
        label="t",
        expected_added_columns=("reward_history_rcos_0",),
        atol=0.0,
    )
    assert report == {}


def test_undeclared_added_column_raises():
    baseline = _frame()
    candidate = _frame()
    candidate["reward_history_rcos_0"] = [0.1, 0.2, 0.3]
    with pytest.raises(ValueError, match="column set differs"):
        diff_table(candidate, baseline, keys=["event_id"], label="t")


def test_declaring_an_addition_does_not_hide_drift_or_a_missing_column():
    """The declaration excuses exactly the named column and nothing else."""
    baseline = _frame()
    drifted = _frame()
    drifted["reward_history_rcos_0"] = [0.1, 0.2, 0.3]
    drifted.loc[0, "value"] = 1.5
    with pytest.raises(ValueError, match="undeclared"):
        diff_table(
            drifted,
            baseline,
            keys=["event_id"],
            label="t",
            expected_added_columns=("reward_history_rcos_0",),
            atol=0.0,
        )

    also_added = _frame()
    also_added["reward_history_rcos_0"] = [0.1, 0.2, 0.3]
    also_added["unexpected"] = [1, 2, 3]
    with pytest.raises(ValueError, match="candidate-only=\\['unexpected'\\]"):
        diff_table(
            also_added,
            baseline,
            keys=["event_id"],
            label="t",
            expected_added_columns=("reward_history_rcos_0",),
        )

    dropped = _frame().drop(columns=["label"])
    with pytest.raises(ValueError, match="baseline-only=\\['label'\\]"):
        diff_table(
            dropped,
            baseline,
            keys=["event_id"],
            label="t",
            expected_added_columns=("reward_history_rcos_0",),
        )


def test_declaring_a_column_the_candidate_lacks_is_harmless():
    """One global declaration can cover a gate where only one table gained it."""
    report = diff_table(
        _frame(),
        _frame(),
        keys=["event_id"],
        label="t",
        expected_added_columns=("reward_history_rcos_0",),
        atol=0.0,
    )
    assert report == {}


def _uint64_frame(last: int) -> pd.DataFrame:
    """Frame whose uint64 column holds values above 2**53 (stop_id is a hash)."""
    return pd.DataFrame(
        {
            "event_id": [1, 2],
            "stop_id": pd.array([8393993873049741010, last], dtype="uint64"),
        }
    )


def test_uint64_drift_of_one_unit_above_2_53_is_caught():
    """Integer columns are compared exactly, not through a lossy float cast."""
    baseline_value = 8393993873049741011
    assert baseline_value > 2**53
    # The float64 images of the two values are identical, so a float comparison
    # would report zero difference.
    assert float(baseline_value) == float(baseline_value + 1)
    with pytest.raises(ValueError, match="undeclared"):
        diff_table(
            _uint64_frame(baseline_value + 1),
            _uint64_frame(baseline_value),
            keys=["event_id"],
            label="t",
            atol=0.0,
        )


def test_identical_uint64_columns_report_no_drift():
    report = diff_table(
        _uint64_frame(8393993873049741011),
        _uint64_frame(8393993873049741011),
        keys=["event_id"],
        label="t",
        atol=0.0,
    )
    assert report == {}


def test_nullable_integer_missing_values_compare_equal():
    """Int64 with pd.NA must not raise, and NA-vs-NA must compare equal."""
    frame = pd.DataFrame(
        {"event_id": [1, 2], "trial": pd.array([7, None], dtype="Int64")}
    )
    assert diff_table(frame, frame.copy(), keys=["event_id"], label="t") == {}
    changed = frame.copy()
    changed["trial"] = pd.array([8, None], dtype="Int64")
    with pytest.raises(ValueError, match="undeclared"):
        diff_table(changed, frame, keys=["event_id"], label="t")
