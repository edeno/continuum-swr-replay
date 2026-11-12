"""Regression test: ``prepare_replay_content_data.py`` always writes
``event_sequence_table.pkl``, even when no session contributed
sequence rows.

The retrospective-alignment pass does
``read_table(INPUT_DIR / "event_sequence_table.pkl")``
unconditionally. The previous prep code returned ``None`` for the
aggregated sequence frame and skipped the save when no posterior
was available, which crashed the retrospective pass with
``FileNotFoundError``. The fix builds an empty schema-complete
frame in that branch so the retrospective pass can read a real file and the
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
   :data:`_EVENT_SEQUENCE_TABLE_COLUMNS` so the retrospective pass finds the
   columns it expects when the reader returns 0 rows.

These three preconditions are checked here through AST inspection
(structural) plus a behavioral roundtrip of the empty schema frame
(write_table → read_table preserves the canonical column order).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.behavior.tables._tables_common import (
    _EVENT_SEQUENCE_TABLE_COLUMNS,
)
from continuum_swr_replay.tables.io import read_table, write_table

REPO_ROOT = Path(__file__).resolve().parents[4]
# The build lives in the package; its command line is
# scripts/tables/prepare_replay_content_data.py.
PREPARE_DATA = (
    REPO_ROOT / "src/continuum_swr_replay/analysis/behavior/tables/prepare_data.py"
)


def _main_function_ast() -> ast.Module:
    """Parse the whole prepare_data module.

    The assembly/publish logic these tests inspect lives in the extracted stage
    helpers (``_assemble_canonical_tables`` / ``_publish_canonical_tables``), not
    in ``main`` itself, so walk the module rather than a single function.
    """
    return ast.parse(PREPARE_DATA.read_text())


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
        f"main() — the retrospective pass will crash on read. RHS samples: "
        f"{bad}"
    )


def test_save_loop_includes_event_sequence_table_tuple():
    """The saver iterates a hard-coded ``(name, frame)`` tuple list.
    A regression that drops the ``event_sequence_table`` entry
    leaves the retrospective pass without a file to read, regardless of how
    correct the empty-fallback frame is."""
    module = _main_function_ast()
    found = False
    # The publish spec is a ``(basename, table_key, frame)`` tuple (basename
    # first, frame last), passed to ``_publish_canonical_tables``. Find it
    # wherever it is built in the module.
    for node in ast.walk(module):
        if not isinstance(node, ast.Tuple) or len(node.elts) < 2:
            continue
        name_node, frame_node = node.elts[0], node.elts[-1]
        if (
            isinstance(name_node, ast.Constant)
            and name_node.value == "event_sequence_table"
            and isinstance(frame_node, ast.Name)
            and frame_node.id == "event_sequence_table_all"
        ):
            found = True
            break
    assert found, (
        "The publish spec no longer pairs "
        '("event_sequence_table", ..., event_sequence_table_all); '
        "the retrospective pass's read will fail with FileNotFoundError."
    )


def test_empty_fallback_uses_canonical_sequence_schema():
    """The else-branch when no session emitted sequence rows must
    build the empty frame from ``_EVENT_SEQUENCE_TABLE_COLUMNS``
    so the saved file has the canonical column order. A regression
    that constructs an empty ``pd.DataFrame()`` instead writes a
    0-column frame; the retrospective pass then ``KeyError``s on the expected
    fields."""
    # The else-branch must construct a DataFrame whose columns kwarg is
    # built from the canonical constant. AST-walk the conditional that
    # gates on ``all_event_sequence_tables`` and inspect its else body.
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
                        isinstance(sub, ast.Name)
                        and sub.id == "_EVENT_SEQUENCE_TABLE_COLUMNS"
                    ):
                        found_canonical_call = True
    assert found_canonical_call, (
        "the empty-sequence-table fallback in main() no longer "
        "references ``_EVENT_SEQUENCE_TABLE_COLUMNS`` to populate "
        "DataFrame columns — the retrospective pass will read a 0-column frame "
        "and KeyError on its expected fields."
    )


def test_empty_canonical_frame_roundtrips_through_storage(tmp_path):
    """Behavioral: an empty frame with the canonical schema
    survives write_table → read_table with column order intact. This is
    what the retrospective pass receives when no session contributed sequences,
    so the column set is the contract the retrospective pass relies on."""
    cols = list(_EVENT_SEQUENCE_TABLE_COLUMNS)
    df = pd.DataFrame(columns=cols)
    path = write_table(df, tmp_path / "event_sequence_table")
    loaded = read_table(path)
    assert list(loaded.columns) == cols, (
        f"empty schema frame did not roundtrip cleanly: "
        f"{list(loaded.columns)} != {cols}"
    )
    assert len(loaded) == 0
    # Sanity: required the retrospective pass fields are present.
    for required in ("event_id", "decoded_arm_map", "nonlocal_prob"):
        assert required in loaded.columns
