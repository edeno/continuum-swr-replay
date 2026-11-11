"""Convert the migrated detection summary to the current question ids, without rebuilding it.

Run once, then deleted: the summary is replaced when it is rebuilt on the cluster
(``TODO.md``), and git keeps the script. Earlier steps of the same migration were
earlier versions of this script; each step's entry in the summary's ``migrated``
record names its commit.

This step follows the outline's move of the simulated comparison with the
Widloski & Foster detector into Part 1, beside Figure 1:

- question 2a.3 (on simulated data) becomes 1.1;
- the later questions of section 2a move up: 2a.4 (the pilot, on one real epoch)
  becomes 2a.3, 2a.5 becomes 2a.4, 2a.6 becomes 2a.5.

Only the detection summary has values for these questions. Every value's number,
kind, style, unit, description, source and pending note is copied unchanged, and
checked to be unchanged before anything is written; writing checks each value's
tags against the question list.

Usage::

    uv run --frozen python scripts/reported/migrate_summaries.py
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported.values import Summary, Value
from continuum_swr_replay.runs.provenance import git

SUMMARY_DIR = ROOT_DIR / "paper" / "reported"
SCRIPT = "scripts/reported/migrate_summaries.py"

IDS = {"2a.3": "1.1", "2a.4": "2a.3", "2a.5": "2a.4", "2a.6": "2a.5"}
"""The question ids that change; every other id stays."""

_KEPT = ("value", "kind", "style", "unit", "description", "source", "pending")


def migrate(script_commit: str) -> Summary:
    """The detection summary with the current question ids, its values unchanged."""
    payload = json.loads((SUMMARY_DIR / "detection.json").read_text())
    migrated = payload["migrated"]
    migrated["steps"].append(
        {
            "change": "the simulated comparison moved to Part 1 (question 1.1) and "
            "the later questions of section 2a renumbered",
            "script_commit": script_commit,
            "ran_at": script_commit,
        }
    )
    summary = Summary(
        name="detection",
        produced_by=payload["produced_by"],
        commit=payload["commit"],
        inputs=payload["inputs"],
        code=None,
        migrated=migrated,
    )
    for key, fields in payload["values"].items():
        summary.add(
            Value(
                key=key,
                questions=tuple(IDS.get(q, q) for q in fields["questions"]),
                used_in=tuple(fields["used_in"]),
                **{field: fields[field] for field in _KEPT},
            )
        )
    written = json.loads(summary.to_json())
    for key, fields in payload["values"].items():
        assert all(written["values"][key][f] == fields[f] for f in _KEPT), key
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    if git(ROOT_DIR, "status", "--porcelain", "--", SCRIPT):
        raise SystemExit(f"{SCRIPT} has uncommitted changes; commit it first")
    script_commit = git(ROOT_DIR, "log", "-1", "--format=%H", "--", SCRIPT)
    summary = migrate(script_commit)
    summary.write(SUMMARY_DIR / "detection.json")
    print(f"detection.json: {len(summary.values)} values")


if __name__ == "__main__":
    main()
