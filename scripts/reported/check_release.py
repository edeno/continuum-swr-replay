"""Check the summaries before a milestone: built from committed code, current with it, and tagged with known questions.

Run at a milestone (a draft going to the co-authors, submission, a release), not on
every commit: during development a summary may lag the code, and
``emit_reported_values.py`` only reports it. Exits 1, naming each part to rebuild
and why, when any summary:

- has no code record (built before records were kept);
- was built from uncommitted code;
- was built from code that has changed since, or under another ``uv.lock``;
- has values whose question or figure tags the question list does not know.

Usage::

    uv run python scripts/reported/check_release.py
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported.release import release_problems
from continuum_swr_replay.reported.values import read_summary

SUMMARY_DIR = ROOT_DIR / "paper" / "reported"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    summaries = {
        s.part: s
        for s in (
            read_summary(p, check_tags=False)
            for p in sorted(SUMMARY_DIR.glob("part*.json"))
        )
    }
    problems = release_problems(summaries, ROOT_DIR)
    for part, found in problems.items():
        print(f"part{part}.json:")
        for problem in found:
            print(f"  - {problem}")
    if problems:
        print(
            f"{len(problems)} of {len(summaries)} summaries to rebuild "
            "(scripts/reported/summarize_part.py), then `make reported-values`"
        )
        return 1
    print(f"all {len(summaries)} summaries are current")
    return 0


if __name__ == "__main__":
    sys.exit(main())
