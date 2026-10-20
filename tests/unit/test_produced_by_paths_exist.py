"""Every recorded producer or design-spec path names a tracked file.

Runners stamp their outputs with the script that wrote them (``PRODUCED_BY``)
and the design note they implement (``DESIGN_SPEC``). A script move that leaves
the string behind records a path that no longer exists, so a reader of the
output cannot find the code. This test reads every ``*.py`` under ``scripts/``
and ``src/`` with ``ast`` and checks each string literal that is

- assigned to a name ``PRODUCED_BY`` or ``DESIGN_SPEC``, or
- the value of a ``"produced_by"`` or ``"design_spec"`` key in a dict literal,

against ``git ls-files``. Values that are not literals (``PRODUCED_BY`` itself,
``provenance["produced_by"]``) are not strings and are skipped.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ROOTS = (REPO / "scripts", REPO / "src")
NAMES = frozenset({"PRODUCED_BY", "DESIGN_SPEC"})
KEYS = frozenset({"produced_by", "design_spec"})


def _tracked() -> frozenset[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, capture_output=True, check=True, text=True
    ).stdout
    return frozenset(p for p in out.split("\0") if p)


def _literal(node: ast.expr | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _recorded_paths(source: str, filename: str) -> list[tuple[int, str]]:
    """``(line, path)`` for every recorded producer/spec literal in ``source``."""
    found = []
    for node in ast.walk(ast.parse(source, filename=filename)):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = _literal(node.value)
            if value is not None and any(
                isinstance(t, ast.Name) and t.id in NAMES for t in targets
            ):
                found.append((node.lineno, value))
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if _literal(key) in KEYS and _literal(value) is not None:
                    found.append((value.lineno, _literal(value)))
    return found


def _all_recorded() -> list[tuple[str, int, str]]:
    return [
        (str(path.relative_to(REPO)), line, recorded)
        for root in ROOTS
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
        for line, recorded in _recorded_paths(path.read_text(), str(path))
    ]


def test_every_recorded_path_is_a_tracked_file():
    recorded = _all_recorded()
    # the runners that stamp their outputs; an empty scan would pass vacuously
    assert len(recorded) >= 20, recorded

    tracked = _tracked()
    missing = [entry for entry in recorded if entry[2] not in tracked]

    assert missing == [], (
        "These record a producer or design-spec path that is not a tracked file "
        f"(a moved script must record its new path): {missing}"
    )


def test_the_scan_finds_names_and_dict_keys():
    source = (
        'PRODUCED_BY = "scripts/a.py"\n'
        'DESIGN_SPEC: str = "docs/b.md"\n'
        "def f(provenance):\n"
        '    return {"produced_by": "scripts/c.py", "design_spec": PRODUCED_BY,\n'
        '            "other": "scripts/d.py", "produced_by_x": provenance["produced_by"]}\n'
    )

    assert [path for _, path in _recorded_paths(source, "probe.py")] == [
        "scripts/a.py",
        "docs/b.md",
        "scripts/c.py",
    ]
