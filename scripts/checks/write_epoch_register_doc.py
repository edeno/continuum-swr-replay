#!/usr/bin/env python
"""Regenerate the parts of ``docs/epoch_register.md`` that mirror the register.

The doc's prose is hand-written. Its rule, reason codes, counts and table of
excluded and pending epochs are generated from ``data_loaders/epoch_register.py``
and ``epoch_register.csv``, each between a pair of markers::

    <!-- BEGIN GENERATED: name -->
    ...
    <!-- END GENERATED: name -->

so that a register edit (a pending epoch flipped to ``analyzed``, say) needs only
this script to bring the doc back in line.

Usage:
    uv run python scripts/checks/write_epoch_register_doc.py            # rewrite
    uv run python scripts/checks/write_epoch_register_doc.py --check    # fail on drift
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
import textwrap
from collections import Counter
from pathlib import Path

import pandas as pd

from continuum_swr_replay.data_loaders.epoch_register import (
    REASON_DEFINITIONS,
    RULE,
    SESSION_WIDE,
    Reason,
    Status,
    read_register,
    reason_status,
)
from continuum_swr_replay.decoding.identifiers import parse_nwb_filename

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "epoch_register.md"

REASON_ORDER: tuple[Reason, ...] = (
    Reason.NO_NWB,
    Reason.NO_RUN_EPOCH,
    Reason.DEPLETION_TASK,
    Reason.NOT_INGESTED,
    Reason.NO_TRIAL_INTERVALS,
    Reason.PENDING_POPULATE,
    Reason.MISSING_INPUT,
    Reason.SUSTAINED_NOISE,
)
"""The rule's order, which is also the precedence when several reasons hold."""

_BLOCK = re.compile(
    r"(<!-- BEGIN GENERATED: (?P<name>[a-z-]+) -->\n)(.*?)(<!-- END GENERATED: "
    r"(?P=name) -->)",
    re.DOTALL,
)


def _cell(text: str) -> str:
    """A table cell: a literal ``|`` is escaped."""
    return text.replace("|", "\\|")


def rule() -> str:
    return "\n".join(
        textwrap.wrap(RULE, width=84, initial_indent="> ", subsequent_indent="> ")
    )


def reason_codes() -> str:
    assert set(REASON_ORDER) == set(Reason), "REASON_ORDER must list every code"
    lines = ["| code | status | definition |", "|---|---|---|"]
    lines += [
        f"| `{reason}` | {reason_status(reason)} | {REASON_DEFINITIONS[reason]} |"
        for reason in REASON_ORDER
    ]
    return "\n".join(lines)


def counts(register: pd.DataFrame) -> str:
    by_reason = Counter(zip(register["status"], register["reason"], strict=True))
    lines = ["| status | reason | entries |", "|---|---|---|"]
    lines.append(f"| analyzed | — | {by_reason[(Status.ANALYZED.value, '')]} |")
    for reason in REASON_ORDER:
        n = by_reason.get((reason_status(reason).value, reason.value), 0)
        if n:
            lines.append(f"| {reason_status(reason)} | `{reason}` | {n} |")
    lines.append(f"| **total** | | **{len(register)}** |")

    session_wide = register.index.get_level_values("epoch") == SESSION_WIDE
    n_wide = int(session_wide.sum())
    lines += [
        "",
        f"An entry is one run epoch, except the {n_wide} session-wide `*` entries "
        f"of sessions with no known run epoch: {len(register) - n_wide} run epochs "
        f"and {n_wide} session-wide entries in all.",
        "",
        "| animal | analyzed | pending |",
        "|---|---|---|",
    ]
    animal = [
        parse_nwb_filename(name)[0] for name in register.index.get_level_values(0)
    ]
    per_animal = (
        register.assign(animal=animal)
        .groupby("animal")["status"]
        .value_counts()
        .unstack(fill_value=0)
    )
    for name, row in per_animal.iterrows():
        lines.append(
            f"| {name} | {row.get(Status.ANALYZED.value, 0)} | "
            f"{row.get(Status.PENDING.value, 0)} |"
        )
    n_analyzed = by_reason[(Status.ANALYZED.value, "")]
    n_pending = by_reason.get((Status.PENDING.value, Reason.PENDING_POPULATE.value), 0)
    lines.append(f"| **all** | **{n_analyzed}** | **{n_pending}** |")
    return "\n".join(lines)


def entries(register: pd.DataFrame) -> str:
    """One row per session, status, reason and evidence, its epochs listed."""
    rank = {reason.value: i for i, reason in enumerate(REASON_ORDER)}
    unanalyzed = register[register["status"] != Status.ANALYZED].reset_index()
    rows = []
    for (name, status, reason, evidence), group in unanalyzed.groupby(
        ["nwb_file_name", "status", "reason", "evidence"]
    ):
        epochs = sorted(group["epoch"])
        rows.append((rank[reason], name, epochs[0], epochs, status, reason, evidence))
    lines = [
        "| session | epochs | status | reason | evidence |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {name} | {', '.join(epochs)} | {status} | `{reason}` | {_cell(evidence)} |"
        for _, name, _, epochs, status, reason, evidence in sorted(rows)
    ]
    return "\n".join(lines)


def render(text: str) -> str:
    """`text` with every generated block rewritten from the register."""
    register = read_register()
    blocks = {
        "rule": rule(),
        "reason-codes": reason_codes(),
        "counts": counts(register),
        "entries": entries(register),
    }
    found = {match.group("name") for match in _BLOCK.finditer(text)}
    if found != set(blocks):
        raise ValueError(
            f"{DOC} has generated blocks {sorted(found)}, not {sorted(blocks)}"
        )
    return _BLOCK.sub(
        lambda match: f"{match.group(1)}{blocks[match.group('name')]}\n{match.group(4)}",
        text,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit 1 if the committed doc differs from the register.",
    )
    args = parser.parse_args(argv)

    committed = DOC.read_text(encoding="utf-8")
    generated = render(committed)
    if not args.check:
        DOC.write_text(generated, encoding="utf-8")
        print(f"wrote {DOC.relative_to(REPO)}")
        return 0
    if committed == generated:
        print(f"{DOC.relative_to(REPO)} is current")
        return 0
    sys.stdout.writelines(
        difflib.unified_diff(
            committed.splitlines(keepends=True),
            generated.splitlines(keepends=True),
            fromfile=f"{DOC.relative_to(REPO)} (committed)",
            tofile=f"{DOC.relative_to(REPO)} (regenerated)",
        )
    )
    print(
        "\nthe epoch register doc is stale -- rerun "
        "`uv run python scripts/checks/write_epoch_register_doc.py`",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
