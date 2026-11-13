"""No supported code reads a canonical table with ``pd.read_parquet``.

A canonical table carries its SWR-source stamp in the parquet metadata.
``tables.io.read_table`` restores it and ``analysed_time.read_swr_table`` also
checks it; ``pd.read_parquet`` drops it, so a table read that way cannot be
checked against the configured SWR source. This test parses every Python file
under ``src/`` and the supported ``scripts/`` with ``ast`` and fails on a
``read_parquet`` call whose first argument names a canonical table: a
``..._table`` name of ``paths.CANONICAL_TABLE_BASENAMES`` or a
``table_path(...)`` call. ``scripts/exploratory/`` is exempt.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from continuum_swr_replay.paths import CANONICAL_TABLE_BASENAMES

REPO = Path(__file__).resolve().parents[2]

#: A first argument that names a canonical table; ``onset_table`` is how the
#: summaries spell ``event_onset_table``.
CANONICAL = re.compile(
    r"\b(" + "|".join((*CANONICAL_TABLE_BASENAMES, "onset_table")) + r")\b|table_path\("
)


def _raw_reads(path: Path) -> list[str]:
    """``line: argument`` for each ``read_parquet`` of a canonical table in one file."""
    found = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if not (isinstance(node, ast.Call) and node.args):
            continue
        name = getattr(node.func, "attr", getattr(node.func, "id", None))
        argument = ast.unparse(node.args[0])
        if name == "read_parquet" and CANONICAL.search(argument):
            found.append(f"{node.lineno}: {argument}")
    return found


def _supported_files() -> list[Path]:
    files = [*(REPO / "src").rglob("*.py"), *(REPO / "scripts").rglob("*.py")]
    return sorted(p for p in files if "exploratory" not in p.parts)


def test_matcher_flags_a_raw_read(tmp_path: Path) -> None:
    source = tmp_path / "example.py"
    source.write_text(
        "import pandas as pd\n"
        "a = pd.read_parquet(inputs.event_table)\n"
        "b = pd.read_parquet(table_path('stop_table', root))\n"
        "c = pd.read_parquet(folder / 'sessions.parquet')\n"
        "d = read_swr_table(inputs.event_table)\n"
    )
    assert _raw_reads(source) == [
        "2: inputs.event_table",
        "3: table_path('stop_table', root)",
    ]


@pytest.mark.parametrize(
    "path", _supported_files(), ids=lambda p: str(p.relative_to(REPO))
)
def test_no_raw_read_of_a_canonical_table(path: Path) -> None:
    assert not _raw_reads(path), (
        f"{path.relative_to(REPO)} reads a canonical table with read_parquet; "
        "use analysis.event_definition.analysed_time.read_swr_table (or tables.io.read_table "
        "for qvals)"
    )
