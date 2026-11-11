"""Check the summaries before a milestone: present, built from committed code, current with it and with their inputs, and tagged with known questions.

Run at a milestone (a draft going to the co-authors, submission, a release), not on
every commit: during development a summary may lag the code, and
``emit_reported_values.py`` only reports it. Run it where the processed data is, so
the inputs can be re-checked. Exits 1, naming each summary to rebuild and why, when:

- a summary's ``<name>.json`` is missing, or another ``.json`` is there;
- a summary has no code record (migrated from an earlier layout instead of
  rebuilt, or git could not answer);
- it was built from uncommitted or untracked code, or its files changed while it
  was built;
- its recorded code, or ``uv.lock``, has changed since;
- a recorded single-file input has changed since, or is not on this machine
  (``--code-only`` skips the inputs, and says so);
- it has values with tag problems (the question list does not know a tag, or the
  question does not list the figure).

Sets of input files are not re-checked (their records name a label, not the
builder's selection); they are listed as notes.

Usage::

    uv run python scripts/reported/check_release.py
    uv run python scripts/reported/check_release.py --code-only
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported.release import release_problems
from continuum_swr_replay.reported.values import read_summaries

SUMMARY_DIR = ROOT_DIR / "paper" / "reported"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--code-only",
        action="store_true",
        help="check the code records and tags, not the inputs",
    )
    args = parser.parse_args(argv)
    try:
        summaries = read_summaries(SUMMARY_DIR, check_tags=False)
    except ValueError as error:
        print(error)
        return 1
    problems, notes = release_problems(
        summaries, ROOT_DIR, check_inputs=not args.code_only
    )
    for name in summaries:
        if name not in problems and name not in notes:
            continue
        print(f"{name}.json:")
        for problem in problems.get(name, []):
            print(f"  - {problem}")
        for note in notes.get(name, []):
            print(f"  (note) {note}")
    if problems:
        print(
            f"{len(problems)} of {len(summaries)} summaries to rebuild "
            "(scripts/reported/summarize.py), then `make reported-values`"
        )
        return 1
    checked = "code and tags" if args.code_only else "code, tags and single-file inputs"
    print(f"all {len(summaries)} summaries are current: {checked} checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
