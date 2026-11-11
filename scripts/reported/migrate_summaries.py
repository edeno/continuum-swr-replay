"""Convert the summaries that need the cluster to rebuild into the current layout, without rebuilding them.

Run once, then deleted: the summaries it converts are replaced when they are
rebuilt on the cluster (``TODO.md``), and git keeps the script. The survey summary
needs no lab data, so it is rebuilt, not migrated.

This step follows the question list's merges and renumbering:

- questions 2b.3 and 2b.5 merged into 2b.2 and 2b.3, and the later ids in sections
  2a and 2b renumbered (:data:`IDS`);
- the detection summary's two pilot values, from one real epoch, answer the new
  question 2a.4 instead of 2a.3 (which is about simulated data);
- the supplementary figures' ids are their output folders' names
  (``figure: memoryless_comparison``).

It also corrects each summary's ``migrated`` record into one entry per step, each
naming the commit that holds the version of this script that ran. The earlier two
steps ran from an uncommitted copy of the script; their entries say so.

Every value's number, kind, style, unit, description, source and pending note is
copied unchanged, and checked to be unchanged before anything is written; each
value is built with its tags checked against the question list.

Usage::

    uv run --frozen python scripts/reported/migrate_summaries.py
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported.values import Summary, Value
from continuum_swr_replay.runs.provenance import git, git_commit

SUMMARY_DIR = ROOT_DIR / "paper" / "reported"
SCRIPT = "scripts/reported/migrate_summaries.py"
MIGRATED = ("detection", "events", "behavior", "mpfc")

IDS = {
    "2a.4": "2a.5",
    "2a.5": "2a.6",
    "2b.3": "2b.2",
    "2b.4": "2b.3",
    "2b.5": "2b.3",
    "2b.6": "2b.4",
    "2b.7": "2b.5",
    "2b.8": "2b.6",
    "2b.9": "2b.7",
}
"""The question ids that change; every other id stays."""

PILOT = ("pilot_",)
"""The detection summary's keys from the one real pilot epoch: question 2a.4."""

USES = {"figure: memoryless comparison": "figure: memoryless_comparison"}
"""The ``used_in`` tags that change."""

EARLIER_STEPS = [
    {
        "change": "renamed from part{n}.json to {name}.json (schema 1 to 4)",
        "script_commit": "6e81598c43fc764b4431944e33630a31458115c1",
        "ran_at": "eb1f98092c89ee24a7a40ca72238a6d3e307003e, from an uncommitted copy",
    },
    {
        "change": "claim numbers mapped to question ids, and tags to the question "
        "list's decisions (schema 4 to 5)",
        "script_commit": "b895716f9c48565db933f90417484b88710d4742",
        "ran_at": "6e81598c43fc764b4431944e33630a31458115c1, from an uncommitted copy",
    },
]
"""The two steps before this one, with the commits that hold the script that ran."""

_KEPT = ("value", "kind", "style", "unit", "description", "source", "pending")


def _questions(name: str, key: str, questions: list[str]) -> tuple[str, ...]:
    if name == "detection" and key.startswith(PILOT):
        return tuple("2a.4" if q == "2a.3" else q for q in questions)
    out: list[str] = []
    for q in questions:
        new = IDS.get(q, q)
        if new not in out:
            out.append(new)
    return tuple(out)


def migrate(name: str, number: int, script_commit: str) -> Summary:
    """Summary ``name`` with the current question ids, its values unchanged."""
    payload = json.loads((SUMMARY_DIR / f"{name}.json").read_text())
    steps = [
        {**step, "change": step["change"].format(n=number, name=name)}
        for step in EARLIER_STEPS
    ]
    steps.append(
        {
            "change": "questions merged and renumbered, the pilot values moved to "
            "2a.4, figure ids named for their folders",
            "script_commit": script_commit,
            "ran_at": script_commit,
        }
    )
    summary = Summary(
        name=name,
        produced_by=payload["produced_by"],
        commit=payload["commit"],
        inputs=payload["inputs"],
        code=None,
        migrated={
            "from": payload["migrated"]["from"],
            "by": SCRIPT,
            "steps": steps,
        },
    )
    for key, fields in payload["values"].items():
        summary.add(
            Value(
                key=key,
                questions=_questions(name, key, fields["questions"]),
                used_in=tuple(USES.get(u, u) for u in fields["used_in"]),
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
    assert git_commit(ROOT_DIR)
    numbers = {"detection": 1, "events": 2, "behavior": 3, "mpfc": 4}
    summaries = {name: migrate(name, numbers[name], script_commit) for name in MIGRATED}
    for name, summary in summaries.items():
        summary.write(SUMMARY_DIR / f"{name}.json")
        print(f"{name}.json: {len(summary.values)} values")


if __name__ == "__main__":
    main()
