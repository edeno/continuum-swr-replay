"""Write one summary file, paper/reported/<name>.json, from outputs the analyses already wrote.

Every number the project reports is a value in one of these files
(``continuum_swr_replay.reported``). This reads the analyses' outputs and refits
nothing. Afterwards run
``scripts/reported/emit_reported_values.py`` to write the values into the
LaTeX macros and the docs.

Usage::

    uv run --frozen python scripts/reported/summarize.py --summary events
    uv run --frozen python scripts/reported/summarize.py --summary survey --output-dir /tmp/check

The summary records the code that built it (``reported/code_record.py``): every
package module loaded during the build (the summary's builder and this script's own
imports), this script, and the package's data files, with their digests; the
commit; the digest of ``uv.lock``; the diff of the tracked files and ``uv.lock``
against the commit; and the names of files git does not track. A build from
uncommitted code is allowed and recorded; ``scripts/reported/check_release.py``
flags it at a milestone. Only the requested summary's builder,
``reported.<name>``, is imported, so other builders' modules are not recorded.
"""

from __future__ import annotations

import argparse
import importlib
import time
from collections.abc import Sequence
from pathlib import Path

from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported.code_record import (
    build_files,
    is_record,
    try_code_record,
)
from continuum_swr_replay.reported.values import SUMMARIES
from continuum_swr_replay.runs.provenance import described_commit, produced_by

SUMMARY_DIR = ROOT_DIR / "paper" / "reported"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summary", required=True, choices=list(SUMMARIES))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SUMMARY_DIR,
        help="where to write <name>.json (default: paper/reported)",
    )
    args = parser.parse_args(argv)
    started = time.time()
    builder = importlib.import_module(f"continuum_swr_replay.reported.{args.summary}")
    summary = builder.summarize(
        builder.default_inputs(), produced_by(__file__), described_commit()
    )
    summary.code = try_code_record(ROOT_DIR, build_files(ROOT_DIR), started)
    path = args.output_dir / f"{args.summary}.json"
    summary.write(path)
    n_pending = sum(v.pending is not None for v in summary.values.values())
    code = summary.code
    if not is_record(code):
        note = f"with no code record ({code['error']})"
    else:
        warnings = []
        if code["diff"]:
            warnings.append("from uncommitted code")
        if code["untracked"]:
            warnings.append(f"from untracked files {', '.join(code['untracked'])}")
        if code["modified_during_build"]:
            warnings.append(
                "with files modified during the build "
                + ", ".join(code["modified_during_build"])
            )
        note = f"with a code record of {len(code['files'])} files"
        note += "".join(f", {w}" for w in warnings)
    print(f"wrote {len(summary.values)} values ({n_pending} pending) to {path}, {note}")


if __name__ == "__main__":
    main()
