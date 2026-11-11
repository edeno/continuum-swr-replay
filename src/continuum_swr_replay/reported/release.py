"""What stands between the summaries and a release: the checks ``make
release-check`` runs at a milestone (a draft going to the co-authors, submission,
a release), and never on every commit."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from continuum_swr_replay.reported.code_record import code_changes
from continuum_swr_replay.reported.values import Summary, summary_tag_problems


def staleness(summary: Summary, repo: Path) -> list[str]:
    """Why ``summary`` may not match the code now; empty when it does.

    A summary with no code record, one built from uncommitted code, and one whose
    recorded modules (or ``uv.lock``) have changed since are each named.
    """
    if summary.code is None:
        return ["no code record: built before code records were kept"]
    problems = []
    if summary.code["uncommitted"]:
        problems.append("built from uncommitted code (the diff is in the record)")
    changed = code_changes(summary.code, repo)
    if changed:
        problems.append(f"code changed since the build: {', '.join(changed)}")
    return problems


def release_problems(
    summaries: Mapping[int, Summary], repo: Path
) -> dict[int, list[str]]:
    """Each part with something to fix before a release: its staleness
    (:func:`staleness`) and its values whose tags the question list does not know."""
    problems = {}
    for part, summary in sorted(summaries.items()):
        found = staleness(summary, repo)
        tags = summary_tag_problems(summary)
        if tags:
            noun = "value" if len(tags) == 1 else "values"
            found.append(
                f"{len(tags)} {noun} with tags the question list does not know"
            )
        if found:
            problems[part] = found
    return problems
