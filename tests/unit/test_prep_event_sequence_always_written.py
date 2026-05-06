"""Regression test: ``prepare_replay_content_data.py`` always writes
``event_sequence_table.pkl``, even when no session contributed
sequence rows.

Stage 5B (the retrospective-alignment step) does
``pd.read_pickle(INPUT_DIR / "event_sequence_table.pkl")``
unconditionally. The previous prep code returned ``None`` for the
aggregated sequence frame and skipped the save when no posterior
was available, which crashed Stage 5B with
``FileNotFoundError``. The fix writes an empty schema-complete
frame so Stage 5B can read a real file and the existing
"empty sequence table is legitimate" path in
``contains_path_to_next`` then handles it cleanly.

This test reads the prep script's source and asserts that the
``event_sequence_table_all`` aggregator is never assigned to
``None`` — i.e., the saver loop's ``if frame is None: continue``
branch can no longer skip the sequence table.
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def test_event_sequence_table_all_never_assigned_None_in_main():
    src = (SCRIPTS_DIR / "prepare_replay_content_data.py").read_text()
    tree = ast.parse(src)
    main_fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    # Walk the assignments inside main(); flag any ``event_sequence_table_all = None``
    # (or a ternary whose else branch is ``None``).
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


def test_canonical_sequence_schema_helper_is_imported_or_available():
    """The empty-frame fallback uses ``_event_sequence_table_columns``
    so the saved file has the canonical column order. This pins that
    pathway by checking the helper is referenced in the prep script."""
    src = (SCRIPTS_DIR / "prepare_replay_content_data.py").read_text()
    assert "_event_sequence_table_columns" in src, (
        "the prep script no longer references "
        "``_event_sequence_table_columns``; the empty-sequence fallback "
        "must use the canonical schema so Stage 5B's read_pickle finds "
        "the expected columns."
    )
