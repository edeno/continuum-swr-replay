"""Check the summaries before a milestone: present, built from committed code, current with it and with their inputs, and tagged with known questions.

Run at a milestone (a draft going to the co-authors, submission, a release), not on
every commit: during development a summary may lag the code, and
``emit_reported_values.py`` only reports it. Run it where the processed data is, so
the inputs can be re-checked. Exits 1, naming each summary to rebuild and why, when:

- a summary's ``<name>.json`` is missing, or another ``.json`` is there;
- a summary has no code record;
- it was built from uncommitted files (a trial build put among the summaries);
- its recorded files (its code, and the inputs the repository tracks), or
  ``uv.lock``, have changed since;
- a recorded single-file input has changed since, or is not on this machine
  (``--code-only`` skips the inputs, and says so);
- it has values with tag problems (the question list does not know a tag, or the
  question does not list the figure);
- the manuscript cites one of its values that is pending;
- ``paper/reported_values.tex`` is not what the summaries give (edited by hand,
  or not regenerated after a rebuild).

Which values the manuscript cites is read from ``paper/main.tex`` and the other
``.tex`` files under ``paper/``, by their macros; how many is listed as a note. Sets of input files are not re-checked
(their records name a label, not the builder's selection); they are noted too.

Usage::

    uv run --frozen python scripts/reported/check_release.py
    uv run --frozen python scripts/reported/check_release.py --code-only
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported.emit import manuscript_source, tex_file
from continuum_swr_replay.reported.release import release_problems
from continuum_swr_replay.reported.values import read_summaries

SUMMARY_DIR = ROOT_DIR / "paper" / "reported"
MANUSCRIPT = ROOT_DIR / "paper" / "main.tex"
TEX_PATH = ROOT_DIR / "paper" / "reported_values.tex"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="check the code records and tags, not the inputs",
    )
    args = parser.parse_args(argv)
    try:
        summaries = read_summaries(SUMMARY_DIR)
    except ValueError as error:
        print(error)
        return 1
    problems, notes = release_problems(
        summaries,
        ROOT_DIR,
        manuscript_source(MANUSCRIPT, TEX_PATH),
        check_inputs=not args.code_only,
    )
    macros_current = TEX_PATH.read_text() == tex_file(summaries)
    for name in summaries:
        print(f"{name}.json:")
        for problem in problems.get(name, []):
            print(f"  - {problem}")
        for note in notes.get(name, []):
            print(f"  (note) {note}")
    if not macros_current:
        print(
            f"{TEX_PATH.relative_to(ROOT_DIR)} is not what the summaries give; "
            "run `make reported-values`"
        )
    if problems:
        print(
            f"{len(problems)} of {len(summaries)} summaries with problems; rebuild a "
            "stale one with scripts/reported/summarize.py, then `make reported-values`"
        )
    if problems or not macros_current:
        return 1
    checked = (
        "code, tags, citations and macros"
        if args.code_only
        else "code, tags, citations, macros and single-file inputs"
    )
    print(f"all {len(summaries)} summaries are current: {checked} checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
