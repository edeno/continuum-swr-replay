"""The manuscript's reported-value macros are defined and hold current values.

``paper/main.tex`` writes result numbers as macros (``\\PartZeroBandLowMode``)
that ``paper/reported_values.tex`` defines; the emitter writes that file, the
values index and the docs' inline values from the part summaries. A macro the
emitter no longer writes -- a renamed or removed summary key -- would only show
up as a LaTeX error, or not at all in a non-stop build; a summary rebuilt
without rerunning the emitter would leave the old number in the PDF. These
tests read the files as text, so they need no LaTeX and no lab data.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from continuum_swr_replay.reported import emit, values

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"

_DEFINED = re.compile(r"\\newcommand\{\\(Part(?:Zero|One|Two|Three|Four)[A-Za-z]+)\}")
_USED = re.compile(r"\\(Part(?:Zero|One|Two|Three|Four)[A-Za-z]+)")
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


def test_emitted_files_hold_the_committed_summaries():
    summaries = {
        s.part: s
        for s in map(values.read_summary, sorted(PAPER.glob("reported/part*.json")))
    }
    assert set(summaries) == set(values.PARTS)
    stale = "is stale: run `make reported-values`"
    assert (PAPER / "reported_values.tex").read_text() == emit.tex_file(summaries), (
        f"paper/reported_values.tex {stale}"
    )
    assert (PAPER / "reported" / "values_index.md").read_text() == (
        emit.values_index(summaries)
    ), f"paper/reported/values_index.md {stale}"
    by_key = values.values_by_key(summaries)
    docs = emit.marked_docs((PAPER, ROOT / "docs"))
    assert PAPER / "README.md" in docs
    for doc in docs:
        text = doc.read_text()
        filled, _ = emit.fill_markers(text, by_key, str(doc.relative_to(ROOT)))
        assert filled == text, f"{doc.relative_to(ROOT)} {stale}"


def _value(key: str, value: float, pending: str | None = None) -> values.Value:
    return values.Value(
        key=key,
        value=value,
        kind="fraction",
        style="pctnum",
        unit="",
        description="d",
        claims=("c",),
        used_in=("paper",),
        source="s",
        pending=pending,
    )


def test_markers_are_rewritten_outside_code_only():
    by_key = {"part0.share": _value("share", 0.421)}
    text = (
        "Stale <!--v:part0.share-->99<!--/v--> and current "
        "<!--v:part0.share-->42<!--/v-->; "
        "an example `<!--v:part0.share-->7<!--/v-->` in code."
    )
    filled, changes = emit.fill_markers(text, by_key, "doc.md")
    assert filled == (
        "Stale <!--v:part0.share-->42<!--/v--> and current "
        "<!--v:part0.share-->42<!--/v-->; "
        "an example `<!--v:part0.share-->7<!--/v-->` in code."
    )
    assert changes == [emit.Change("doc.md", "part0.share", "99", "42")]


def test_a_marker_naming_no_summary_key_is_an_error():
    with pytest.raises(KeyError, match="part0.gone"):
        emit.fill_markers("<!--v:part0.gone-->1<!--/v-->", {}, "doc.md")


def test_a_pending_value_is_flagged_in_the_macros_and_the_index():
    summary = values.Summary(part=0, produced_by="by", commit="c")
    summary.add(_value("share", 0.5, pending="untracked script"))
    assert "[PENDING: untracked script]" in emit.tex_file({0: summary})
    assert "pending: untracked script" in emit.values_index({0: summary})
