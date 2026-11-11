"""Convert the summaries that need the cluster to rebuild into the current layout, without rebuilding them.

Run once, then deleted when those summaries are rebuilt (``TODO.md``). The survey
summary needs no lab data, so it is rebuilt (``summarize.py --summary survey``),
not migrated.

Each summary is read from its old file, written as ``paper/reported/<name>.json``
through :class:`~continuum_swr_replay.reported.values.Summary` (so the layout is the
one a build writes), and the old file removed. Every value's number, kind, style,
unit, description, source and pending note is copied unchanged, and checked to be
unchanged before anything is written. The summary records what was done in
``migrated``: the file it came from, the commit, this script, and each change. A
migrated summary has no code record, so ``make release-check`` names it until it
is rebuilt.

Usage::

    uv run python scripts/reported/migrate_summaries.py
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported.values import Summary, Value
from continuum_swr_replay.runs.provenance import git_commit

SUMMARY_DIR = ROOT_DIR / "paper" / "reported"

RENAMED = {1: "detection", 2: "events", 3: "behavior", 4: "mpfc"}
"""The summaries keyed by part number, and the name each now has."""

_KEPT = ("value", "kind", "style", "unit", "description", "source", "pending")
"""The fields of a value a migration must not change."""


def migrate(number: int, name: str, commit: str) -> Summary:
    """The summary in ``part{number}.json`` as summary ``name``, its values
    unchanged."""
    path = SUMMARY_DIR / f"part{number}.json"
    payload = json.loads(path.read_text())
    summary = Summary(
        name=name,
        produced_by=payload["produced_by"],
        commit=payload["commit"],
        inputs=payload["inputs"],
        code=None,
        migrated={
            "from": f"paper/reported/part{number}.json",
            "schema_version": payload["schema_version"],
            "commit": commit,
            "by": "scripts/reported/migrate_summaries.py",
            "changes": [f"renamed from part{number}.json to {name}.json"],
        },
    )
    for key, fields in payload["values"].items():
        summary.add(
            Value(
                key=key,
                claims=tuple(fields["claims"]),
                used_in=tuple(fields["used_in"]),
                check_tags=False,
                **{field: fields[field] for field in _KEPT},
            )
        )
    for key, fields in payload["values"].items():
        written = json.loads(summary.to_json())["values"][key]
        assert all(written[f] == fields[f] for f in _KEPT), key
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args(argv)
    commit = git_commit(ROOT_DIR)
    for number, name in RENAMED.items():
        summary = migrate(number, name, commit)
        summary.write(SUMMARY_DIR / f"{name}.json")
        (SUMMARY_DIR / f"part{number}.json").unlink()
        print(f"part{number}.json -> {name}.json: {len(summary.values)} values")


if __name__ == "__main__":
    main()
