"""``docs/regression_pipeline_map.md`` must match the tree it describes.

The map is generated from the source with ``ast``. A hand-maintained map rots
silently; this test runs the generator's ``--check`` mode so a renamed, added,
or deleted public symbol fails here instead of misleading a reader.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "qa" / "write_pipeline_map.py"


def test_committed_pipeline_map_matches_the_tree() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "docs/regression_pipeline_map.md is stale; regenerate with "
        "`uv run python scripts/qa/write_pipeline_map.py`.\n"
        f"{result.stdout}\n{result.stderr}"
    )
