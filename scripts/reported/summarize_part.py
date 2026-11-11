"""Write one part's summary file, paper/reported/part{n}.json, from outputs the analyses already wrote.

Every number the project reports is a value in one of these files
(``continuum_swr_replay.reported``; design
``docs/design/2026-10-26-reported-values-design.md``). This reads the
analyses' outputs and refits nothing. Afterwards run
``scripts/reported/emit_reported_values.py`` to write the values into the
LaTeX macros and the docs.

Usage::

    uv run python scripts/reported/summarize_part.py --part 2

The summary records the code that built it: every package module the part's
builder loaded and the package's data files, with their digests, the commit, the
digest of ``uv.lock``, and the diff of those files and the ones git does not track
when they differ from the commit (``reported/code_record.py``). A build from
uncommitted code is allowed and recorded; ``scripts/reported/check_release.py``
flags it at a milestone. Only the requested part's builder is imported, so the
record lists only the modules that part uses.
"""

from __future__ import annotations

import argparse
import importlib
import time
from collections.abc import Sequence

from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported.code_record import build_files, try_code_record
from continuum_swr_replay.runs.provenance import described_commit, produced_by

SUMMARY_DIR = ROOT_DIR / "paper" / "reported"
PARTS = (0, 1, 2, 3, 4)
"""The parts with a summary; part ``n`` is built by ``reported.part{n}``."""


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", type=int, required=True, choices=PARTS)
    args = parser.parse_args(argv)
    started = time.time()
    builder = importlib.import_module(f"continuum_swr_replay.reported.part{args.part}")
    summary = builder.summarize(
        builder.default_inputs(), produced_by(__file__), described_commit()
    )
    summary.code = try_code_record(ROOT_DIR, build_files(ROOT_DIR), started)
    path = SUMMARY_DIR / f"part{args.part}.json"
    summary.write(path)
    n_pending = sum(v.pending is not None for v in summary.values.values())
    code = summary.code
    if "error" in code:
        note = "with no code record"
    else:
        uncommitted = ", from uncommitted code" if code["uncommitted"] else ""
        note = f"with a code record of {len(code['files'])} files{uncommitted}"
    print(f"wrote {len(summary.values)} values ({n_pending} pending) to {path}, {note}")


if __name__ == "__main__":
    main()
