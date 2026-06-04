"""Regression test: ``prepare_replay_content_data.py`` always writes
``event_sequence_table.pkl``, even when no session contributed
sequence rows.

Stage 5B (the retrospective-alignment step) does
``pd.read_pickle(INPUT_DIR / "event_sequence_table.pkl")``
unconditionally. The previous prep code returned ``None`` for the
aggregated sequence frame and skipped the save when no posterior
was available, which crashed Stage 5B with
``FileNotFoundError``. The fix builds an empty schema-complete
frame in that branch so Stage 5B can read a real file and the
existing "empty sequence table is legitimate" path in
``contains_path_to_next`` then handles it cleanly.

The contract the prep script must keep:

1. The aggregator never assigns ``event_sequence_table_all = None``
   (otherwise the saver loop's ``if frame is None: continue``
   branch silently drops the file).
2. The save loop tuple actually includes
   ``("event_sequence_table", event_sequence_table_all)`` — without
   it, no fallback frame is enough.
3. The empty-fallback path uses the canonical schema from
   :func:`_event_sequence_table_columns` so Stage 5B finds the
   columns it expects when ``read_pickle`` returns 0 rows.

These three preconditions are checked here through AST inspection
(structural) plus a behavioral roundtrip of the empty schema frame
(pickle → read_pickle preserves the canonical column order).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.replay.tables import (
    _event_sequence_table_columns,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _main_function_ast() -> ast.FunctionDef:
    src = (SCRIPTS_DIR / "prepare_replay_content_data.py").read_text()
    tree = ast.parse(src)
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )


def test_event_sequence_table_all_never_assigned_None_in_main():
    main_fn = _main_function_ast()
    bad = []
    for stmt in ast.walk(main_fn):
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "event_sequence_table_all"
                ):
                    rhs_dump = ast.dump(stmt.value)
                    if (
                        "Constant(value=None" in rhs_dump
                        or "NameConstant(value=None" in rhs_dump
                    ):
                        bad.append(rhs_dump)
    assert not bad, (
        "event_sequence_table_all is still being assigned None in "
        f"main() — Stage 5B will crash on read_pickle. RHS samples: "
        f"{bad}"
    )


def test_save_loop_includes_event_sequence_table_tuple():
    """The saver iterates a hard-coded ``(name, frame)`` tuple list.
    A regression that drops the ``event_sequence_table`` entry
    leaves Stage 5B without a file to read, regardless of how
    correct the empty-fallback frame is."""
    main_fn = _main_function_ast()
    found = False
    for node in ast.walk(main_fn):
        if not isinstance(node, ast.For):
            continue
        if not isinstance(node.iter, ast.Tuple):
            continue
        for elt in node.iter.elts:
            if not isinstance(elt, ast.Tuple) or len(elt.elts) != 2:
                continue
            name_node, frame_node = elt.elts
            if (
                isinstance(name_node, ast.Constant)
                and name_node.value == "event_sequence_table"
                and isinstance(frame_node, ast.Name)
                and frame_node.id == "event_sequence_table_all"
            ):
                found = True
                break
    assert found, (
        "The saver-loop tuple in main() no longer contains "
        '("event_sequence_table", event_sequence_table_all). '
        "Stage 5B's read_pickle will fail with FileNotFoundError."
    )


def test_empty_fallback_uses_canonical_sequence_schema():
    """The else-branch when no session emitted sequence rows must
    build the empty frame with ``_event_sequence_table_columns()``
    so the saved file has the canonical column order. A regression
    that constructs an empty ``pd.DataFrame()`` instead writes a
    0-column frame; Stage 5B then ``KeyError``s on the expected
    fields."""
    # The else-branch must construct a DataFrame whose columns kwarg
    # is the helper output. AST-walk the conditional that gates on
    # ``all_event_sequence_tables`` and inspect its else body.
    main_fn = _main_function_ast()
    found_canonical_call = False
    for node in ast.walk(main_fn):
        if not isinstance(node, ast.If):
            continue
        if (
            isinstance(node.test, ast.Name)
            and node.test.id == "all_event_sequence_tables"
        ):
            for else_stmt in node.orelse:
                for sub in ast.walk(else_stmt):
                    if (
                        isinstance(sub, ast.Call)
                        and isinstance(sub.func, ast.Name)
                        and sub.func.id == "_event_sequence_table_columns"
                    ):
                        found_canonical_call = True
    assert found_canonical_call, (
        "the empty-sequence-table fallback in main() no longer "
        "calls ``_event_sequence_table_columns()`` to populate "
        "DataFrame columns — Stage 5B will read a 0-column frame "
        "and KeyError on its expected fields."
    )


def test_empty_canonical_frame_roundtrips_through_pickle(tmp_path):
    """Behavioral: an empty frame with the canonical schema
    survives pickle → read_pickle with column order intact. This is
    what Stage 5B receives when no session contributed sequences,
    so the column set is the contract Stage 5B relies on."""
    cols = _event_sequence_table_columns()
    df = pd.DataFrame(columns=cols)
    path = tmp_path / "event_sequence_table.pkl"
    df.to_pickle(path)
    loaded = pd.read_pickle(path)
    assert list(loaded.columns) == cols, (
        f"empty schema frame did not roundtrip cleanly: "
        f"{list(loaded.columns)} != {cols}"
    )
    assert len(loaded) == 0
    # Sanity: required Stage 5B fields are present.
    for required in ("event_id", "decoded_arm_map", "nonlocal_prob"):
        assert required in loaded.columns
