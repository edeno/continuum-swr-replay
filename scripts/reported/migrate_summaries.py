"""Convert the summaries that need the cluster to rebuild into the current layout, without rebuilding them.

Run once, then deleted when those summaries are rebuilt (``TODO.md``). The survey
summary needs no lab data, so it is rebuilt (``summarize.py --summary survey``),
not migrated.

The four summaries were first renamed from ``part1.json``-``part4.json`` (the
layout keyed by part number; schema 4). This converts them to schema 5: each value
names the ``questions`` it answers by their ids in the outline, and its tags follow
the decisions the builders already encode:

- every claim number becomes its question's id (:data:`IDS`);
- the detection summary's held-out values no longer answer question 2c.7 (remote
  events), and the events summary's remote-group spike-support values do;
- the figure tags that named no figure point at Figure 2 or the supplement, and
  the values only a document used are tagged ``supplement``.

Every value's number, kind, style, unit, description, source and pending note is
copied unchanged, and checked to be unchanged before anything is written. Each
value is built with its tags checked, so a tag the question list does not accept
stops the conversion. The summary's ``migrated`` record says what was done.

Usage::

    uv run python scripts/reported/migrate_summaries.py
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported.questions import QUESTIONS
from continuum_swr_replay.reported.values import SCHEMA_VERSION, Summary, Value
from continuum_swr_replay.runs.provenance import git_commit

SUMMARY_DIR = ROOT_DIR / "paper" / "reported"

MIGRATED = ("detection", "events", "behavior", "mpfc")

IDS = {q.claim: q.id for q in QUESTIONS if q.claim and q.claim != "12"}
"""Each claim number to its question's id."""
IDS |= {"12": "2a.3", "synthesis Layer 2": "3.8"}

USES = {
    "figure: main, left out": "figure: 2",
    "figure: main, other side of the overlap": "figure: 2",
    "figure: spike support": "figure: 2",
    "figure: 3, content occupancy": "figure: 2",
    "figure: one class or two": None,
    "docs: replay_rate_content_swr_synthesis": "supplement",
    "docs: rate_models": "supplement",
}
"""The ``used_in`` tags that change, and what each becomes (None: dropped)."""

REMOTE = ("swr_remote_", "non_swr_remote_")
"""The events summary's remote-group keys, which also answer question 2c.7."""

_KEPT = ("value", "kind", "style", "unit", "description", "source", "pending")


def _questions(name: str, key: str, claims: list[str]) -> tuple[str, ...]:
    ids = [IDS[c] for c in claims]
    if name == "detection":
        ids = [q for q in ids if q != "2c.7"]
    if name == "events" and key.startswith(REMOTE) and "2c.7" not in ids:
        ids.append("2c.7")
    return tuple(ids)


def _uses(used_in: list[str]) -> tuple[str, ...]:
    out: list[str] = []
    for tag in used_in:
        new = USES.get(tag, tag)
        if new is not None and new not in out:
            out.append(new)
    return tuple(out)


def migrate(name: str, commit: str) -> Summary:
    """Summary ``name`` in the current layout, its values unchanged."""
    payload = json.loads((SUMMARY_DIR / f"{name}.json").read_text())
    assert payload["schema_version"] == 4, (name, payload["schema_version"])
    migrated = payload["migrated"]
    migrated["changes"].append(
        "claim numbers mapped to question ids, and tags to the question list's "
        f"decisions (schema 4 to 5, at {commit})"
    )
    summary = Summary(
        name=name,
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
                questions=_questions(name, key, fields["claims"]),
                used_in=_uses(fields["used_in"]),
                **{field: fields[field] for field in _KEPT},
            )
        )
    written = json.loads(summary.to_json())
    assert written["schema_version"] == SCHEMA_VERSION
    for key, fields in payload["values"].items():
        assert all(written["values"][key][f] == fields[f] for f in _KEPT), key
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    commit = git_commit(ROOT_DIR)
    summaries = {name: migrate(name, commit) for name in MIGRATED}
    for name, summary in summaries.items():
        summary.write(SUMMARY_DIR / f"{name}.json")
        print(f"{name}.json: {len(summary.values)} values")


if __name__ == "__main__":
    main()
