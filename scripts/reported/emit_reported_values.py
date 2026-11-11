"""Write the reported values into paper/reported_values.tex and the docs' inline markers.

Reads every summary file in ``paper/reported/`` (``summarize.py`` writes
them), writes the LaTeX macros and ``paper/reported/questions.md`` (every question
of ``reported/questions.py`` with the values that answer it), and rewrites the
number inside every
``<!--v:<summary>.key-->...<!--/v-->`` marker in the markdown under ``paper/``
and ``docs/``. Every number it changes is listed on the screen and in
``paper/reported/changes.md`` (a dated section is appended for each emit that
changes numbers), so a replaced number is never silent.

Everything that can fail runs before anything is written: the summaries are
read (exactly one ``<name>.json`` per summary), ``questions.md`` is rendered, each
summary's staleness is worked out and every doc's markers are filled in memory.
Only then are the files written and the changes logged, so a replaced number is
never left unlisted by a crash.

A summary written before the question list changed its tags is still emitted: the
number of values with tag problems is printed for each summary, and each one is
listed in ``questions.md``, until the summary is rebuilt. A summary whose code has
changed since it was built is named too, but never refused: ``check_release.py``
is the check that fails, at a milestone.

Usage::

    uv run python scripts/reported/emit_reported_values.py
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date

from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported.emit import emit, marked_docs, questions_view
from continuum_swr_replay.reported.release import staleness
from continuum_swr_replay.reported.values import read_summaries, summary_tag_problems

SUMMARY_DIR = ROOT_DIR / "paper" / "reported"
TEX_PATH = ROOT_DIR / "paper" / "reported_values.tex"
CHANGES_PATH = SUMMARY_DIR / "changes.md"
QUESTIONS_PATH = SUMMARY_DIR / "questions.md"
DOC_ROOTS = (ROOT_DIR / "paper", ROOT_DIR / "docs")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    summaries = read_summaries(SUMMARY_DIR, check_tags=False)
    view = questions_view(summaries)
    notices = []
    for name, summary in summaries.items():
        problems = summary_tag_problems(summary)
        if problems:
            notices.append(
                f"  {name}.json: {len(problems)} values with tag problems "
                "(listed in questions.md; rebuild the summary)"
            )
        notices += [f"  {name}.json: {r}" for r in staleness(summary, ROOT_DIR)]
    docs = marked_docs(DOC_ROOTS)
    changes = emit(summaries, TEX_PATH, docs, ROOT_DIR)
    if changes:
        if not CHANGES_PATH.exists():
            CHANGES_PATH.write_text(
                "# Reported values the emitter changed\n\n"
                "Each section is one emit that changed numbers: the text that was in each\n"
                "marker, and the computed value now there. Newest last.\n"
            )
        section = [
            "",
            f"## {date.today().isoformat()}",
            "",
            "| Doc | Key | Was | Now |",
            "|---|---|---|---|",
            *(f"| `{c.doc}` | `{c.key}` | {c.old} | {c.new} |" for c in changes),
            "",
        ]
        with CHANGES_PATH.open("a") as f:
            f.write("\n".join(section))
    QUESTIONS_PATH.write_text(view)
    print(
        f"wrote {TEX_PATH.relative_to(ROOT_DIR)} and "
        f"{QUESTIONS_PATH.relative_to(ROOT_DIR)}; filled {len(docs)} docs"
    )
    for notice in notices:
        print(notice)
    for c in changes:
        print(f"  {c.doc}: {c.key}: {c.old!r} -> {c.new!r}")
    print(
        f"{len(changes)} numbers changed; listed in {CHANGES_PATH.relative_to(ROOT_DIR)}"
    )


if __name__ == "__main__":
    main()
