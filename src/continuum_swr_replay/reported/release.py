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

    A summary with no code record (built before records were kept, or where git
    could not say), one built from uncommitted or untracked code, one whose files
    changed while it was built, and one whose recorded files (or ``uv.lock``) have
    changed since are each named.
    """
    code = summary.code
    if code is None:
        return ["no code record"]
    if "error" in code:
        return [f"no code record: {code['error']}"]
    problems = []
    if code["untracked"]:
        problems.append(
            f"built from files git does not track: {', '.join(code['untracked'])}"
        )
    elif code["uncommitted"]:
        problems.append("built from uncommitted code (the diff is in the record)")
    if code["modified_during_build"]:
        problems.append(
            "files modified while it was built: "
            + ", ".join(code["modified_during_build"])
        )
    changed = code_changes(code, repo)
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
