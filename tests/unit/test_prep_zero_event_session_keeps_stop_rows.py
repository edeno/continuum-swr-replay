"""Regression test: ``_process_session`` must not drop zero-event
sessions before building the stop table.

Both prep scripts used to do::

    event_table = build_event_table(...)
    if len(event_table) == 0:
        return None
    stop_table = build_stop_table(...)

That short-circuit dropped sessions whose immobility bouts had no
non-local events — even though the stop table's documented
contract explicitly includes "zero-event bouts" and the rate model
fits ``n_events`` per stop over a denominator that must include
zero-event stops.

This test extracts the prep-script ``_process_session`` AST and
verifies that the call to ``build_(_contfrag)_stop_table`` is
*not* gated behind a ``return None`` on empty ``event_table``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def _load_function(rel_path: str, name: str) -> ast.FunctionDef:
    """Return the AST FunctionDef for ``name`` defined in ``rel_path``."""
    src = (SCRIPTS_DIR / rel_path).read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AttributeError(f"{name!r} not found in {rel_path}")


def _stop_call_index_and_event_return_index(
    fn: ast.FunctionDef, stop_call_name: str
) -> tuple[int | None, int | None]:
    """Locate the (first occurrence of) ``stop_call_name`` and the index
    of any ``return None`` immediately gated by ``len(event_table) == 0``.
    Indices are statement-position within the function body."""
    stop_idx: int | None = None
    early_return_idx: int | None = None
    for stmt in ast.walk(fn):
        # Find a call to the stop-table builder.
        if (
            isinstance(stmt, ast.Call)
            and isinstance(stmt.func, ast.Name)
            and stmt.func.id == stop_call_name
        ):
            # Find the enclosing statement's position.
            for j, top in enumerate(fn.body):
                for sub in ast.walk(top):
                    if sub is stmt:
                        if stop_idx is None or j < stop_idx:
                            stop_idx = j
        if isinstance(stmt, ast.If):
            cond_src = ast.dump(stmt.test)
            if (
                "event_table" in cond_src
                and ("Eq" in cond_src or "== 0" in cond_src)
                and any(
                    isinstance(s, ast.Return) and s.value is not None
                    and isinstance(s.value, ast.Constant) and s.value.value is None
                    for s in stmt.body
                )
            ):
                # Position of this If in fn.body
                for j, top in enumerate(fn.body):
                    if top is stmt:
                        early_return_idx = j
    return stop_idx, early_return_idx


@pytest.mark.parametrize(
    ("script", "stop_builder"),
    [
        ("prepare_replay_content_data.py", "build_stop_table"),
        ("prepare_contfrag_replay_content_data.py", "build_contfrag_stop_table"),
    ],
)
def test_stop_table_built_before_any_early_return_on_empty_event_table(
    script, stop_builder
):
    fn = _load_function(script, "_process_session")
    stop_idx, early_idx = _stop_call_index_and_event_return_index(fn, stop_builder)
    assert stop_idx is not None, (
        f"{script}: did not find a call to {stop_builder} in _process_session"
    )
    if early_idx is not None:
        # If a ``return None`` on empty event_table still exists,
        # the stop-table call must come BEFORE it. Otherwise zero-
        # event sessions are silently dropped.
        assert stop_idx < early_idx, (
            f"{script}: {stop_builder} (stmt #{stop_idx}) is built AFTER "
            f"the empty-event_table return (stmt #{early_idx}) — that "
            "drops zero-event sessions before their stop bouts are "
            "saved."
        )
