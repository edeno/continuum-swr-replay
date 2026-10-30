"""``docs/models_table.md`` must enumerate exactly the ladders in ``model_specs``.

The catalog is a hand-written mirror of the executable specs, so it drifts
silently: rungs get added to ``model_specs.py`` and never reach the doc, and
retired rungs linger there long after the code drops them.

The machine-readable contract (also stated at the top of the doc): every section
that mirrors an executable ladder carries **exactly one** fenced block whose info
string is ``rungs``. Inside such a block, a line that starts in column 0 and is
not a ``#`` comment begins with a rung name -- its first whitespace-delimited
token. Indented lines are continuations and blank lines are ignored. The section
is identified by the nearest preceding ATX heading, mapped to its ladder by
:data:`EXPECTED`.

Point ``MODELS_TABLE_PATH`` at another copy of the doc to check that copy
instead (used to check a historical revision).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from continuum_swr_replay.analysis.replay import model_specs as specs

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = Path(
    os.environ.get("MODELS_TABLE_PATH", REPO_ROOT / "docs" / "models_table.md")
)

_HEADING = re.compile(r"^(#{2,6})\s+(.*?)\s*$")
_FENCE_OPEN = re.compile(r"^```rungs\s*$")
_FENCE_ANY = re.compile(r"^```")

# Heading text -> the executable rung names that heading must enumerate.
EXPECTED: dict[str, frozenset[str]] = {
    "Per-bin rate — Family B": frozenset(specs.MODEL_PER_BIN_RATE),
    "MU uncertainty — per-bin Poisson (auxiliary)": frozenset(
        specs.MODEL_UNCERTAINTY_RATE
    ),
    "Cross-history coupling — per-bin Poisson (auxiliary)": frozenset(
        specs.MODEL_CROSS_HISTORY_RATE
    ),
    "Cross-history coupling, duration-aware — per-bin Poisson (auxiliary)": (
        frozenset(specs.MODEL_CROSS_HISTORY_DURATION_RATE)
    ),
    "Cross-history coupling, long window — ridge cross-validated (auxiliary)": (
        frozenset(specs.MODEL_CROSS_HISTORY_LONG_RATE)
    ),
    "Per-stop rate — Family A": frozenset(specs.MODEL_PER_STOP_RATE),
    "Content — arm": frozenset(specs.MODEL_HIERARCHY),
    "Content — patch": frozenset(specs.MODEL_PATCH),
    "ContFrag content — arm": frozenset(specs.MODEL_HIERARCHY_SWR),
    "ContFrag content — patch": frozenset(specs.MODEL_PATCH_SWR),
    "Post-switch patch": frozenset(specs.MODEL_PATCH_SWITCH),
    "Choice": frozenset(specs.MODEL_CHOICE),
    "Rate `is_swr` contrast — stratified (G)": frozenset(
        specs.SWR_CONTRAST_RATE_G_SPECS
    ),
    "Rate `is_swr` contrast — pooled (S)": frozenset(specs.SWR_CONTRAST_RATE_S_SPECS),
    "Content `is_swr` contrast — arm (C / CS)": frozenset(
        specs.SWR_CONTRAST_CONTENT_ARM_LADDER
    ),
    "Content `is_swr` contrast — patch (CP / CPS)": frozenset(
        specs.SWR_CONTRAST_CONTENT_PATCH_LADDER
    ),
    "Leaf-only contrast rungs": specs.LEAF_ONLY_CONTRAST_RUNGS,
    "Descriptive rungs": specs.DESCRIPTIVE_CONTENT_RUNGS | specs.DESCRIPTIVE_RATE_RUNGS,
    "Post-outcome cohort rungs": specs.POST_OUTCOME_RATE_RUNGS,
}

# Families whose rungs also carry a per-rung markdown table row (formula /
# question / key term). A row dropped from the table is drift the ``rungs``
# block alone cannot catch.
TABLE_FAMILIES: tuple[str, ...] = (
    "Rate `is_swr` contrast — stratified (G)",
    "Rate `is_swr` contrast — pooled (S)",
    "Content `is_swr` contrast — arm (C / CS)",
    "Content `is_swr` contrast — patch (CP / CPS)",
)

# Covariate names retired from the pipeline. The catalog describes the models as
# they are fitted today, so none of these may appear in it; the migration record
# in docs/design/ is where they still live.
RETIRED_NAMES: tuple[str, ...] = (
    "is_last_rewarded_elsewhere",
    "swr_x_is_last_rewarded_elsewhere",
    "is_after_omission",
    "swr_x_q_arm",
    "arm_uncertainty",
    "recent_switch_rate",
    "reward_lag1",
    "q_arm",
)


def _read_doc() -> str:
    if not DOC_PATH.is_file():
        pytest.fail(f"models_table doc not found at {DOC_PATH}")
    return DOC_PATH.read_text(encoding="utf-8")


def _parse_rungs_blocks(text: str) -> dict[str, list[list[str]]]:
    """Map heading text -> the rung-name lists of each ``rungs`` block under it."""
    blocks: dict[str, list[list[str]]] = {}
    heading = ""
    names: list[str] | None = None
    in_other_fence = False
    for line in text.splitlines():
        if in_other_fence:
            # Skip to the close so headings inside a code block are not
            # mistaken for section headings.
            in_other_fence = not _FENCE_ANY.match(line)
            continue
        if names is not None:
            if _FENCE_ANY.match(line):
                blocks.setdefault(heading, []).append(names)
                names = None
            elif line.strip() and not line.startswith((" ", "\t", "#")):
                names.append(line.split()[0])
            continue
        if _FENCE_OPEN.match(line):
            names = []
            continue
        if _FENCE_ANY.match(line):
            in_other_fence = True
            continue
        match = _HEADING.match(line)
        if match is not None:
            heading = match.group(2)
    assert names is None, f"unterminated ```rungs block under heading {heading!r}"
    assert not in_other_fence, f"unterminated code fence under heading {heading!r}"
    return blocks


def _section_text(text: str, heading: str) -> str:
    """The lines of ``heading``'s own section, excluding nested code fences' headings.

    A section runs from its heading to the next heading at the same or a
    shallower level, so a per-rung table row is only credited to the family it
    is written under -- a row under a neighbouring family cannot satisfy this
    family's check.
    """
    lines = text.splitlines()
    start: int | None = None
    level = 0
    in_fence = False
    collected: list[str] = []
    for index, line in enumerate(lines):
        if _FENCE_ANY.match(line):
            in_fence = not in_fence
            if start is not None:
                collected.append(line)
            continue
        match = None if in_fence else _HEADING.match(line)
        if start is None:
            if match is not None and match.group(2) == heading:
                start = index
                level = len(match.group(1))
            continue
        if match is not None and len(match.group(1)) <= level:
            break
        collected.append(line)
    if start is None:
        pytest.fail(f"{DOC_PATH.name}: no heading {heading!r}")
    return "\n".join(collected)


@pytest.fixture(scope="module")
def blocks() -> dict[str, list[list[str]]]:
    return _parse_rungs_blocks(_read_doc())


@pytest.mark.parametrize("heading", sorted(EXPECTED))
def test_each_family_has_exactly_one_rungs_block(
    blocks: dict[str, list[list[str]]], heading: str
) -> None:
    found = blocks.get(heading, [])
    assert len(found) == 1, (
        f"{DOC_PATH.name}: heading {heading!r} must carry exactly one "
        f"```rungs block; found {len(found)}."
    )


def test_no_unmapped_rungs_blocks(blocks: dict[str, list[list[str]]]) -> None:
    """Every ``rungs`` block belongs to a heading this test knows how to check."""
    unmapped = sorted(set(blocks) - set(EXPECTED))
    assert not unmapped, (
        f"{DOC_PATH.name}: ```rungs blocks under unmapped headings {unmapped}. "
        "Add the heading to EXPECTED with its ladder, or use a plain fence."
    )


@pytest.mark.parametrize("heading", sorted(EXPECTED))
def test_rungs_block_mirrors_the_executable_ladder(
    blocks: dict[str, list[list[str]]], heading: str
) -> None:
    found = blocks.get(heading, [])
    if len(found) != 1:
        pytest.skip("covered by test_each_family_has_exactly_one_rungs_block")
    listed = found[0]
    assert len(listed) == len(set(listed)), (
        f"{DOC_PATH.name}: duplicate rung names under {heading!r}."
    )
    expected = EXPECTED[heading]
    missing = sorted(expected - set(listed))
    extra = sorted(set(listed) - expected)
    assert not missing and not extra, (
        f"{DOC_PATH.name}: {heading!r} does not mirror model_specs.py — "
        f"missing {missing}, undocumented-in-specs {extra}."
    )


def test_swr_and_mua_contfrag_ladders_are_the_same_rungs() -> None:
    """The doc documents one ContFrag ladder per grain; the specs must agree."""
    assert set(specs.MODEL_HIERARCHY_SWR) == set(specs.MODEL_HIERARCHY_MUA)
    assert set(specs.MODEL_PATCH_SWR) == set(specs.MODEL_PATCH_MUA)


@pytest.mark.parametrize("heading", TABLE_FAMILIES)
def test_table_families_carry_a_row_per_rung(heading: str) -> None:
    section = _section_text(_read_doc(), heading)
    rows = {
        match.group(1)
        for match in re.finditer(r"^\|\s*`([^`]+)`\s*\|", section, flags=re.MULTILINE)
    }
    missing = sorted(EXPECTED[heading] - rows)
    assert not missing, (
        f"{DOC_PATH.name}: {heading!r} rungs with no markdown table row: {missing}."
    )


@pytest.mark.parametrize("name", RETIRED_NAMES)
def test_retired_covariate_names_are_absent(name: str) -> None:
    text = _read_doc()
    hits = [
        line
        for line in text.splitlines()
        if re.search(rf"\b{re.escape(name)}\b", line) is not None
    ]
    assert not hits, (
        f"{DOC_PATH.name}: retired covariate name {name!r} appears in the "
        f"catalog: {hits[:3]}"
    )
