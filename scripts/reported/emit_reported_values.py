"""Write the reported values into paper/reported_values.tex and the docs' inline markers.

Reads every summary file in ``paper/reported/`` (``summarize_part.py`` writes
them), writes the LaTeX macros, and rewrites the number inside every
``<!--v:part{n}.key-->...<!--/v-->`` marker in the markdown under ``paper/``
and ``docs/``. Every number it changes is listed on the screen and in
``paper/reported/changes.md``, so a replaced number is never silent.

Usage::

    uv run python scripts/reported/emit_reported_values.py
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import date

from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported.emit import emit, marked_docs
from continuum_swr_replay.reported.values import read_summary

SUMMARY_DIR = ROOT_DIR / "paper" / "reported"
TEX_PATH = ROOT_DIR / "paper" / "reported_values.tex"
CHANGES_PATH = SUMMARY_DIR / "changes.md"
DOC_ROOTS = (ROOT_DIR / "paper", ROOT_DIR / "docs")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    summaries = {
        s.part: s
        for s in (read_summary(p) for p in sorted(SUMMARY_DIR.glob("part*.json")))
    }
    docs = marked_docs(DOC_ROOTS)
    changes = emit(summaries, TEX_PATH, docs, ROOT_DIR)
    print(f"wrote {TEX_PATH.relative_to(ROOT_DIR)}; filled {len(docs)} docs")
    lines = [
        f"# Reported values changed by the last emit ({date.today().isoformat()})",
        "",
        "Each row is a number the emitter replaced in a doc: the text that was",
        "there, and the computed value now there.",
        "",
        "| Doc | Key | Was | Now |",
        "|---|---|---|---|",
        *(f"| `{c.doc}` | `{c.key}` | {c.old} | {c.new} |" for c in changes),
        "",
    ]
    CHANGES_PATH.write_text("\n".join(lines))
    for c in changes:
        print(f"  {c.doc}: {c.key}: {c.old!r} -> {c.new!r}")
    print(
        f"{len(changes)} numbers changed; listed in {CHANGES_PATH.relative_to(ROOT_DIR)}"
    )


if __name__ == "__main__":
    main()
