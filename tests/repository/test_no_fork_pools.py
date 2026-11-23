"""No code asks for a ``fork`` pool.

A process forked from one in which JAX's threads run can deadlock, so a pool
that may compute with JAX starts its processes by spawn
(``runs.threads.pool_context``). This scans ``src/`` and ``scripts/`` for a
call of ``get_context("fork")`` or ``set_start_method("fork")``.
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests.support._imports import REPO, SRC

FORK_CALLS = frozenset({"get_context", "set_start_method"})


def fork_calls(path: Path) -> list[int]:
    """Lines of `path` that call one of :data:`FORK_CALLS` with ``"fork"``."""
    lines = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", getattr(node.func, "id", None))
        arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
        if name in FORK_CALLS and any(
            isinstance(a, ast.Constant) and a.value == "fork" for a in arguments
        ):
            lines.append(node.lineno)
    return lines


def test_no_code_starts_a_fork_pool():
    files = [*SRC.rglob("*.py"), *(REPO / "scripts").rglob("*.py")]
    assert len(files) > 100
    found = {
        str(path.relative_to(REPO)): lines
        for path in files
        if (lines := fork_calls(path))
    }
    assert found == {}


def test_the_scan_sees_a_fork_call(tmp_path):
    path = tmp_path / "pool.py"
    path.write_text(
        "import multiprocessing\n"
        "multiprocessing.get_context('spawn')\n"
        "multiprocessing.get_context('fork').Pool(2)\n"
        "multiprocessing.set_start_method(method='fork')\n"
    )
    assert fork_calls(path) == [3, 4]
