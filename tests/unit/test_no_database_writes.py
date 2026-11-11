"""No code in the repository writes to the lab's Spyglass database.

The analyses only read Spyglass (``docs/development.md``, "Data Integration with
Spyglass"). This test parses every Python file under ``src/`` and ``scripts/``
with ``ast`` and fails on a call to a DataJoint write method or schema creation.
pandas' ``insert`` and ``drop`` share names with nothing listed here, so a
DataFrame call is not mistaken for a database write; DataJoint's plain
``delete`` and ``insert`` are not listed for the same reason, and their
Spyglass forms (``cautious_delete``, ``insert1``) are.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

#: DataJoint / Spyglass methods that write to the database.
WRITE_METHODS = frozenset(
    {
        "populate",
        "insert1",
        "insert_selection",
        "cautious_delete",
        "super_delete",
        "delete_quick",
        "drop_quick",
        "alter",
        "update1",
    }
)


def _writes(path: Path) -> list[str]:
    """``line: call`` for each database write in one file."""
    found = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        attr = node.func.attr
        target = node.func.value
        if attr in WRITE_METHODS:
            found.append(f"{node.lineno}: .{attr}(")
        elif attr == "schema" and isinstance(target, ast.Name) and target.id == "dj":
            found.append(f"{node.lineno}: dj.schema(")
    return found


def test_the_rule_catches_writes(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(
        "import datajoint as dj\n"
        "schema = dj.schema('x')\n"
        "Table.populate(key)\n"
        "Table().insert1(row)\n"
        "(Table & key).cautious_delete()\n"
        "frame.insert(0, 'a', 1)\n"
        "frame.drop(columns='a')\n"
    )
    assert _writes(sample) == [
        "2: dj.schema(",
        "3: .populate(",
        "4: .insert1(",
        "5: .cautious_delete(",
    ]


@pytest.mark.parametrize("root", ["src", "scripts"])
def test_no_database_writes(root):
    offenders = {
        str(path.relative_to(REPO)): writes
        for path in sorted((REPO / root).rglob("*.py"))
        if (writes := _writes(path))
    }
    assert not offenders, (
        f"database writes in {root}/: {offenders}. The analyses only read Spyglass; "
        "a write needs the data owner's approval and belongs to a documented lab "
        "operation (docs/development.md)."
    )
