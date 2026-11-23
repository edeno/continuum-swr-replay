#!/usr/bin/env python
"""Write the shared regression fitter's reference outputs, the fixtures its tests
compare it with.

Runs every case of ``scripts/checks/_regression_reference.py`` (each builds its
inputs from a fixed seed and runs one public entry of
``continuum_swr_replay.analysis.regression``) and writes one file per group,
``tests/analysis/regression/data/numpy_reference/<group>.json``: the commit the
outputs were made at, the command, the versions of Python, NumPy and SciPy, and
the outputs of each case, one case per line. Floats are written exactly
(``repr``); a float that is not finite as a string (strict JSON).

``tests/analysis/regression/test_reference_fixtures.py`` runs the same cases and
compares their outputs with these files by the rule of
``_regression_reference.differences``: a float within ``1e-11 + 1e-9 |b|`` of
its reference ``b``, every status, reason, step count, column left out and count
equal. The references are of the NumPy fitter, so that a fitter that replaces
its solver can be checked against it.

Writing into the fixture folder needs a checkout whose tracked files equal its
commit, so the commit recorded is the code that made the outputs; a trial build
goes elsewhere with ``--out-dir``. Run it with one BLAS thread, as the stored
files were made:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
        uv run --frozen python scripts/checks/write_regression_reference.py
    uv run --frozen python scripts/checks/write_regression_reference.py \\
        --out-dir /tmp/reference --group newton
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import scipy

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from continuum_swr_replay.runs.provenance import git_commit, git_dirty  # noqa: E402
from continuum_swr_replay.tables.io import write_atomically  # noqa: E402
from scripts.checks import _regression_reference as reference  # noqa: E402

FIXTURE_DIR = REPO / "tests" / "analysis" / "regression" / "data" / "numpy_reference"
COMMAND = "python scripts/checks/write_regression_reference.py"


def group_text(group: str, commit: str, command: str) -> str:
    """The file of one group: its header and one case per line, strict JSON."""
    header = {
        "commit": commit,
        "command": command,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    lines = [f"{json.dumps(k)}: {json.dumps(v)}," for k, v in header.items()]
    cases = [
        f"{json.dumps(name)}: "
        + json.dumps(reference.run(group, name), allow_nan=False)
        for name in reference.GROUPS[group]
    ]
    return "{\n" + "\n".join(lines) + '\n"cases": {\n' + ",\n".join(cases) + "\n}\n}\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=FIXTURE_DIR,
        help="Where the files are written (default: the test fixtures).",
    )
    parser.add_argument(
        "--group",
        action="append",
        choices=list(reference.GROUPS),
        help="Write only this group (repeatable; default: every group).",
    )
    args = parser.parse_args(argv)
    if args.out_dir.resolve() == FIXTURE_DIR.resolve() and git_dirty(REPO):
        parser.error(
            "tracked files differ from the commit: commit first, or write a trial "
            "build with --out-dir"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    commit = git_commit(REPO) + ("-dirty" if git_dirty(REPO) else "")
    command = " ".join([COMMAND, *(argv if argv is not None else sys.argv[1:])])
    for group in args.group or reference.GROUPS:
        text = group_text(group, commit, command)
        json.loads(text)  # the file is valid JSON
        path = args.out_dir / f"{group}.json"
        write_atomically(path, lambda tmp, text=text: tmp.write_text(text))
        print(f"{path}: {len(reference.GROUPS[group])} cases, {len(text):,} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
