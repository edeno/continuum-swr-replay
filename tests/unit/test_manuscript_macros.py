"""Every reported-value macro the manuscript uses must be defined.

``paper/main.tex`` writes result numbers as macros (``\\PartTwoNEvents``) that
``paper/reported_values.tex`` defines; the emitter writes that file from the
part summaries. A macro the emitter no longer writes -- a renamed or removed
summary key -- would only show up as a LaTeX error, or not at all in a
non-stop build. This test reads both files as text, so it needs no LaTeX.
"""

from __future__ import annotations

import re
from pathlib import Path

PAPER = Path(__file__).resolve().parents[2] / "paper"

_DEFINED = re.compile(r"\\newcommand\{\\(Part(?:One|Two|Three|Four)[A-Za-z]+)\}")
_USED = re.compile(r"\\(Part(?:One|Two|Three|Four)[A-Za-z]+)")
_COMMENT = re.compile(r"(?<!\\)%.*$", re.MULTILINE)


def defined_macros(reported_values_tex: str) -> set[str]:
    """Names of the reported-value macros a ``reported_values.tex`` defines."""
    return set(_DEFINED.findall(reported_values_tex))


def used_macros(manuscript_tex: str) -> set[str]:
    """Names of the reported-value macros a manuscript uses outside comments."""
    return set(_USED.findall(_COMMENT.sub("", manuscript_tex)))


def test_used_and_defined_macros_are_found():
    reported = (
        "\\newcommand{\\PartOneSweepReplays}{4,536}  % part1.sweep_replays\n"
        "\\newcommand{\\PartFourUnits}{\\ensuremath{-}0.50}  % part4.units\n"
    )
    manuscript = (
        "We simulated \\PartOneSweepReplays{} replays and \\PartTwoMissing{}.\n"
        "% \\PartThreeCommentedOut is in a comment\n"
        "A literal 50\\% stays \\PartFourUnits.\n"
    )
    assert defined_macros(reported) == {"PartOneSweepReplays", "PartFourUnits"}
    assert used_macros(manuscript) == {
        "PartOneSweepReplays",
        "PartTwoMissing",
        "PartFourUnits",
    }


def test_manuscript_reads_the_reported_values():
    manuscript = (PAPER / "main.tex").read_text()
    assert re.search(r"^\\input\{reported_values(\.tex)?\}", manuscript, re.M)


def test_every_macro_the_manuscript_uses_is_defined():
    defined = defined_macros((PAPER / "reported_values.tex").read_text())
    used = used_macros((PAPER / "main.tex").read_text())
    assert defined, "reported_values.tex defines no reported-value macros"
    missing = sorted(used - defined)
    assert not missing, (
        f"main.tex uses macros reported_values.tex does not define: {missing}. "
        "Rebuild the summaries and run `make reported-values`, or fix the name."
    )
