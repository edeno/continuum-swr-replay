"""ContFrag output schemas carry event-source provenance."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLES_PATH = REPO_ROOT / "src/continuum_swr_replay/analysis/contfrag/tables.py"


def _load_column_helpers() -> dict[str, object]:
    from continuum_swr_replay.analysis.replay.trajectory import (
        SHARED_TRAJECTORY_COLUMNS,
    )

    src = TABLES_PATH.read_text()
    tree = ast.parse(src)
    names = {
        "_EVENT_SOURCE_TO_COL",
        "_CONTFRAG_TRAJECTORY_COLUMNS",
        "_resolve_event_col",
        "_contfrag_event_table_columns",
        "_contfrag_stop_table_columns",
        "_contfrag_event_arm_table_columns",
    }
    # Seed the dependency the trajectory-column constant is derived from.
    ns: dict[str, object] = {"SHARED_TRAJECTORY_COLUMNS": SHARED_TRAJECTORY_COLUMNS}

    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in names for t in node.targets
        ):
            wrapper = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(wrapper)
            exec(compile(wrapper, "tables.py", "exec"), ns)  # noqa: S102
        if isinstance(node, ast.FunctionDef) and node.name in names:
            wrapper = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(wrapper)
            exec(compile(wrapper, "tables.py", "exec"), ns)  # noqa: S102
    return ns


def test_contfrag_table_schemas_include_source_and_area_provenance():
    helpers = _load_column_helpers()

    event_cols = helpers["_contfrag_event_table_columns"]("SWR")
    stop_cols = helpers["_contfrag_stop_table_columns"]()
    event_arm_cols = helpers["_contfrag_event_arm_table_columns"]("MUA")

    assert event_cols[:5] == [
        "animal",
        "day",
        "epoch",
        "event_source",
        "brain_area",
    ]
    assert stop_cols[:5] == [
        "animal",
        "day",
        "epoch",
        "event_source",
        "brain_area",
    ]
    assert event_arm_cols[:5] == [
        "animal",
        "day",
        "epoch",
        "event_source",
        "brain_area",
    ]


def test_contfrag_stop_table_schema_is_pinned_exactly():
    """The full ContFrag stop-table column list, pinned in order.

    The prefix assertion above only covers the five provenance columns, so a
    dropped or reordered column anywhere after them is invisible: deleting
    ``immobility_time`` — the exposure offset every rate model divides by —
    leaves the whole contfrag suite green. Pin the entire list in order so any
    schema edit has to be deliberate.
    """
    stop_cols = _load_column_helpers()["_contfrag_stop_table_columns"]()

    assert list(stop_cols) == [
        "animal",
        "day",
        "epoch",
        "event_source",
        "brain_area",
        "stop_idx",
        "stop_id",
        "stop_start",
        "stop_end",
        "immobility_time",
        "reward_rate_trial_k10",
        "is_swr_stop",
        "is_mua_stop",
        "has_replay",
        "n_events",
        "is_after_switch",
        "is_after_omission",
        "is_after_reward",
        "is_after_stay",
        "reward_outcome",
        "is_stay",
        "current_arm",
        "prev_arm",
        "next_arm",
        "last_rewarded_arm",
        "current_patch",
        "prev_patch",
        "next_patch",
    ]
