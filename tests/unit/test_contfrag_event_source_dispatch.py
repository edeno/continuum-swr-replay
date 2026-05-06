"""Regression test for strict ``event_source`` dispatch in contfrag tables.

The previous form ``"SWR_event" if event_source == "SWR" else "MUA_event"``
silently routed any non-``"SWR"`` value (including typos like
``"swr"`` / ``"Swr"`` / ``"ripple"``) to the MUA column, so direct
calls into the public table builders could quietly produce a table
keyed against the wrong event source. The new ``_resolve_event_col``
helper raises ``ValueError`` for anything outside ``{"SWR", "MUA"}``.

The dispatch helper is loaded by ``ast`` so the test stays offline
regardless of any future module-top imports in
``analysis/contfrag/tables.py``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLES_PATH = (
    REPO_ROOT / "src/continuum_swr_replay/analysis/contfrag/tables.py"
)


def _load_helper(name: str):
    src = TABLES_PATH.read_text()
    tree = ast.parse(src)
    # Pre-load module constants the helper closes over.
    ns: dict = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_EVENT_SOURCE_TO_COL"
            for t in node.targets
        ):
            wrapper = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(wrapper)
            exec(compile(wrapper, "tables.py", "exec"), ns)  # noqa: S102
            break
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            wrapper = ast.Module(body=[node], type_ignores=[])
            ast.fix_missing_locations(wrapper)
            exec(compile(wrapper, "tables.py", "exec"), ns)  # noqa: S102
            return ns[name]
    raise AttributeError(f"{name!r} not found")


_resolve_event_col = _load_helper("_resolve_event_col")


class TestResolveEventCol:
    def test_swr_resolves_to_swr_event_column(self):
        assert _resolve_event_col("SWR") == "SWR_event"

    def test_mua_resolves_to_mua_event_column(self):
        assert _resolve_event_col("MUA") == "MUA_event"

    @pytest.mark.parametrize("bad", ["swr", "Swr", "mua", "ripple", "", "SWR "])
    def test_typos_raise_value_error(self, bad):
        with pytest.raises(ValueError, match="event_source must be"):
            _resolve_event_col(bad)
