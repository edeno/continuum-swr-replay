"""Write the reported values into paper/reported_values.tex and the docs' inline markers.

Reads every summary file in ``paper/reported/`` (``summarize_part.py`` writes
them), writes the LaTeX macros and ``paper/reported/questions.md`` (every question
of ``reported/questions.py`` with the values that answer it), and rewrites the
number inside every
``<!--v:part{n}.key-->...<!--/v-->`` marker in the markdown under ``paper/``
and ``docs/``. Every number it changes is listed on the screen and in
``paper/reported/changes.md`` (a dated section is appended for each emit that
changes numbers), so a replaced number is never silent.

A summary written before the question list changed its tags is still emitted; the
values whose tags the list does not know are printed, and listed in
``questions.md``, until the summary is rebuilt.

Usage::

    uv run python scripts/reported/emit_reported_values.py
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date

from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported.emit import emit, marked_docs, questions_view
from continuum_swr_replay.reported.values import read_summary, summary_tag_problems

SUMMARY_DIR = ROOT_DIR / "paper" / "reported"
TEX_PATH = ROOT_DIR / "paper" / "reported_values.tex"
CHANGES_PATH = SUMMARY_DIR / "changes.md"
QUESTIONS_PATH = SUMMARY_DIR / "questions.md"
DOC_ROOTS = (ROOT_DIR / "paper", ROOT_DIR / "docs")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    summaries = {
        s.part: s
        for s in (
            read_summary(p, check_tags=False)
            for p in sorted(SUMMARY_DIR.glob("part*.json"))
        )
    }
    docs = marked_docs(DOC_ROOTS)
    changes = emit(summaries, TEX_PATH, docs, ROOT_DIR)
    QUESTIONS_PATH.write_text(questions_view(summaries))
    print(
        f"wrote {TEX_PATH.relative_to(ROOT_DIR)} and "
        f"{QUESTIONS_PATH.relative_to(ROOT_DIR)}; filled {len(docs)} docs"
    )
    for part, summary in sorted(summaries.items()):
        problems = summary_tag_problems(summary)
        if problems:
            print(
                f"  part{part}.json: {len(problems)} values with tags the question "
                "list does not know (rebuild the summary)"
            )
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
    for c in changes:
        print(f"  {c.doc}: {c.key}: {c.old!r} -> {c.new!r}")
    print(
        f"{len(changes)} numbers changed; listed in {CHANGES_PATH.relative_to(ROOT_DIR)}"
    )


if __name__ == "__main__":
    main()
