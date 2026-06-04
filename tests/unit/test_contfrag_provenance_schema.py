"""ContFrag output schemas carry event-source provenance."""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLES_PATH = REPO_ROOT / "src/continuum_swr_replay/analysis/contfrag/tables.py"


def _load_column_helpers() -> dict[str, object]:
    src = TABLES_PATH.read_text()
    tree = ast.parse(src)
    names = {
        "_EVENT_SOURCE_TO_COL",
        "_resolve_event_col",
        "_contfrag_event_table_columns",
        "_contfrag_stop_table_columns",
        "_contfrag_event_arm_table_columns",
    }
    ns: dict[str, object] = {}

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
