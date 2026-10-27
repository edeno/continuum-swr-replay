"""Write one part's summary file, paper/reported/part{n}.json, from outputs the analyses already wrote.

Every number the project reports is a value in one of these files
(``continuum_swr_replay.reported``; design
``docs/design/2026-10-26-reported-values-design.md``). This reads the
analyses' outputs and refits nothing. Afterwards run
``scripts/reported/emit_reported_values.py`` to write the values into the
LaTeX macros and the docs.

Usage::

    uv run python scripts/reported/summarize_part.py --part 2
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported import part2, part4
from continuum_swr_replay.runs.provenance import described_commit, produced_by

SUMMARY_DIR = ROOT_DIR / "paper" / "reported"
BUILDERS = {
    2: lambda by, commit: part2.summarize(part2.default_inputs(), by, commit),
    4: lambda by, commit: part4.summarize(part4.default_inputs(), by, commit),
}
"""The parts with a summary so far."""


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--part", type=int, required=True, choices=sorted(BUILDERS))
    args = parser.parse_args(argv)
    summary = BUILDERS[args.part](produced_by(__file__), described_commit())
    path = SUMMARY_DIR / f"part{args.part}.json"
    summary.write(path)
    n_pending = sum(v.pending is not None for v in summary.values.values())
    print(f"wrote {len(summary.values)} values ({n_pending} pending) to {path}")


if __name__ == "__main__":
    main()
