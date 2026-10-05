"""``docs/epoch_register.md`` must mirror the epoch register it documents.

The doc restates the module's rule, its reason codes with their definitions,
the counts per status and reason, and every excluded and pending entry with its
evidence. ``scripts/qa/write_epoch_register_doc.py`` generates those parts from
``data_loaders/epoch_register.py`` and its ``epoch_register.csv``; the first
test fails when the committed doc is not the generator's output, and the rest
read the doc itself, so they also hold the generator to the register.

The machine-readable contract: the rule is the blockquote under ``## The rule``;
``## Reason codes`` holds one table row per code
(``| `CODE` | status | definition |``); ``## Counts`` starts with one row per
status and reason, then a total; ``## Excluded and pending epochs`` holds one row
per session and evidence (``| session | epochs | status | `REASON` | evidence |``),
its epochs comma-separated, ``*`` for a session-wide entry. A literal ``|``
inside a cell is written ``\\|``.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

import scripts.qa.write_epoch_register_doc as generator
from continuum_swr_replay.data_loaders import epoch_register
from continuum_swr_replay.data_loaders.epoch_register import (
    REASON_DEFINITIONS,
    RULE,
    Reason,
    Status,
    read_register,
)

DOC_PATH = Path(__file__).resolve().parents[2] / "docs" / "epoch_register.md"
_HEADING = re.compile(r"^##\s+(.*?)\s*$")
_CELL_SPLIT = re.compile(r"(?<!\\)\|")


def _sections() -> dict[str, list[str]]:
    """Each ``##`` heading's lines, up to the next ``##`` heading."""
    if not DOC_PATH.is_file():
        pytest.fail(f"epoch register doc not found at {DOC_PATH}")
    sections: dict[str, list[str]] = {}
    heading = ""
    for line in DOC_PATH.read_text(encoding="utf-8").splitlines():
        match = _HEADING.match(line)
        if match is not None:
            heading = match.group(1)
            sections[heading] = []
        elif heading:
            sections[heading].append(line)
    return sections


def _rows(lines: list[str]) -> list[list[str]]:
    """The body rows of the section's first table, as unescaped, stripped cells."""
    table: list[str] = []
    for line in lines:
        if line.startswith("|"):
            table.append(line)
        elif table:
            break  # the first table ends at its first non-table line
    rows = []
    for line in table[2:]:  # past the header and its separator
        cells = _CELL_SPLIT.split(line.strip())[1:-1]
        rows.append([cell.strip().replace("\\|", "|") for cell in cells])
    return rows


def _code(cell: str) -> str:
    return cell.strip("`")


def _words(text: str) -> str:
    return " ".join(text.split())


def test_the_committed_doc_is_the_generators_output():
    """Regenerate with ``uv run python scripts/qa/write_epoch_register_doc.py``."""
    committed = DOC_PATH.read_text(encoding="utf-8")

    assert generator.render(committed) == committed


def test_the_doc_states_the_rule_verbatim():
    quoted = " ".join(
        line.lstrip(">").strip()
        for line in _sections()["The rule"]
        if line.startswith(">")
    )
    assert _words(quoted) == _words(RULE)


def test_the_doc_lists_every_reason_code_with_its_definition():
    rows = _rows(_sections()["Reason codes"])
    listed = {_code(code): (status, definition) for code, status, definition in rows}

    assert len(rows) == len(listed)
    assert listed == {
        reason.value: (
            epoch_register.reason_status(reason).value,
            REASON_DEFINITIONS[reason],
        )
        for reason in Reason
    }


def test_the_doc_counts_each_status_and_reason():
    rows = _rows(_sections()["Counts"])
    *body, total = rows
    register = read_register()
    counts = Counter(zip(register["status"], register["reason"], strict=True))

    assert {
        (status, "" if reason == "—" else _code(reason)): int(n)
        for status, reason, n in body
    } == dict(counts)
    assert int(total[-1].strip("*")) == len(register)


def test_the_doc_lists_every_excluded_and_pending_entry_with_its_evidence():
    listed = set()
    for session, epochs, status, reason, evidence in _rows(
        _sections()["Excluded and pending epochs"]
    ):
        for epoch in epochs.split(","):
            listed.add((session, epoch.strip(), status, _code(reason), evidence))
    register = read_register().reset_index()
    unanalysed = register[register["status"] != Status.ANALYSED]

    assert listed == set(unanalysed.itertuples(index=False, name=None))
