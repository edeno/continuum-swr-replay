"""Write one summary file, paper/reported/<name>.json, from outputs the analyses already wrote.

Every number the project reports is a value in one of these files
(``continuum_swr_replay.reported``). This reads the analyses' outputs and refits
nothing. Afterwards run
``scripts/reported/emit_reported_values.py`` to write the values into the
LaTeX macros and the docs.

Usage::

    uv run --frozen python scripts/reported/summarize.py --summary events
    uv run --frozen python scripts/reported/summarize.py --summary survey --output-dir /tmp/check

The summary records the files of the checkout that built it
(``reported/code_record.py``): every package module loaded during the build (the
summary's builder, what it imports, and this script's own imports), this script,
the package's data files and the inputs the repository tracks, with their
digests; the commit; and the digest of ``uv.lock``. The commit the summary names
is that record's (``code_record.commit_label``).

A summary in ``paper/reported/`` is built from committed files: when any of
those files, or ``uv.lock``, differs from the commit, nothing is written there.
Commit first, or pass ``--output-dir`` for a trial build, which records the files
that differ.
"""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence
from pathlib import Path

from continuum_swr_replay.paths import ROOT_DIR, SUMMARY_DIR
from continuum_swr_replay.reported.code_record import (
    build_files,
    code_record,
    tracked,
    uncommitted,
)
from continuum_swr_replay.reported.values import SUMMARIES
from continuum_swr_replay.runs.provenance import git_commit, produced_by


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
    canonical = args.output_dir.resolve() == SUMMARY_DIR.resolve() or (
        args.output_dir.exists()
        and SUMMARY_DIR.exists()
        and args.output_dir.samefile(SUMMARY_DIR)
    )

    def refuse_uncommitted(files: list[str]) -> None:
        """Exit when a summary for ``paper/reported/`` would be built from
        ``files`` that differ from the commit."""
        if canonical and files:
            parser.exit(
                1,
                f"not written: a summary in {SUMMARY_DIR.relative_to(ROOT_DIR)} is "
                "built from committed files, and these differ from the commit: "
                f"{', '.join(files)}. Commit them, or pass "
                "--output-dir for a trial build.\n",
            )

    builder = importlib.import_module(f"continuum_swr_replay.reported.{args.summary}")
    # Checked before the build, so uncommitted code costs no build.
    head = git_commit(ROOT_DIR)
    refuse_uncommitted(uncommitted(ROOT_DIR, build_files(ROOT_DIR)))
    summary = builder.summarize(builder.default_inputs(), produced_by(__file__))
    inputs = tracked(
        ROOT_DIR,
        [
            record["path"]
            for record in summary.inputs
            if "n_files" not in record and not Path(record["path"]).is_absolute()
        ],
    )
    summary.code = code_record(
        ROOT_DIR, [*build_files(ROOT_DIR), *(ROOT_DIR / i for i in inputs)]
    )
    refuse_uncommitted(summary.code["uncommitted"])
    if summary.code["head"] != head:
        parser.exit(
            1,
            "not written: the checkout moved to another commit while the summary "
            "was built; rebuild\n",
        )
    path = args.output_dir / f"{args.summary}.json"
    summary.write(path)
    n_pending = sum(v.pending is not None for v in summary.values.values())
    note = f"with a record of {len(summary.code['files'])} files"
    if summary.code["uncommitted"]:
        note += f", from uncommitted files ({', '.join(summary.code['uncommitted'])})"
    print(f"wrote {len(summary.values)} values ({n_pending} pending) to {path}, {note}")


if __name__ == "__main__":
    main()
