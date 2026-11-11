"""What stands between the summaries and a release: the checks ``make
release-check`` runs at a milestone (a draft going to the co-authors, submission,
a release), and never on every commit."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from continuum_swr_replay.reported.code_record import (
    code_changes,
    input_changes,
    is_record,
)
from continuum_swr_replay.reported.emit import cited_values
from continuum_swr_replay.reported.values import Summary, summary_tag_problems


def staleness(summary: Summary, repo: Path) -> list[str]:
    """Why ``summary``'s code record does not vouch for it now; empty when it does.

    Each is named: no code record (built before records were kept, or git could
    not answer, with its message), a record that lists no files, files git does
    not track, a tracked diff, files modified while it was built, and recorded
    files (or ``uv.lock``) that changed since.
    """
    code = summary.code
    if summary.migrated is not None:
        return [
            f"migrated from {summary.migrated['from']}, not rebuilt: no code record"
        ]
    if code is None:
        return ["no code record"]
    if not is_record(code):
        return [f"no code record: {code['error']}"]
    problems = []
    if not code["files"]:
        problems.append("the code record lists no files")
    if code["untracked"]:
        problems.append(
            f"built from files git does not track: {', '.join(code['untracked'])}"
        )
    if code["diff"]:
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


def input_problems(summary: Summary, root: Path) -> tuple[list[str], list[str]]:
    """What the recorded inputs say: problems (an input changed since the build,
    or is not on this machine) and notes (sets of files that cannot be re-checked
    from their records)."""
    found = input_changes(summary.inputs, root)
    problems = [f"input changed since the build: {p}" for p in found["changed"]]
    problems += [f"input not on this machine: {p}" for p in found["missing"]]
    notes = [f"input file set not re-checked: {p}" for p in found["unchecked"]]
    return problems, notes


def release_problems(
    summaries: Mapping[str, Summary],
    repo: Path,
    manuscript: str,
    *,
    check_inputs: bool = True,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Each summary with something to fix before a release, and each one's notes.

    The problems are its staleness (:func:`staleness`), its values with tag
    problems, its pending values that ``manuscript`` (the LaTeX source) cites,
    and, with ``check_inputs``, its recorded single-file inputs that changed or
    are absent (:func:`input_problems`). The notes give how many of its values
    the manuscript cites, and name what could not be checked: the input file
    sets, or every input when ``check_inputs`` is off.
    """
    problems: dict[str, list[str]] = {}
    notes: dict[str, list[str]] = {}
    cited = cited_values(summaries, manuscript)
    for name, summary in summaries.items():
        found = staleness(summary, repo)
        tags = summary_tag_problems(summary)
        if tags:
            noun = "value" if len(tags) == 1 else "values"
            found.append(f"{len(tags)} {noun} with tag problems: {', '.join(tags)}")
        pending = [key for key in cited[name] if summary.values[key].pending]
        if pending:
            found.append(f"the manuscript cites pending values: {', '.join(pending)}")
        notes[name] = [
            f"the manuscript cites {len(cited[name])} of {len(summary.values)} values"
        ]
        if check_inputs:
            input_found, input_notes = input_problems(summary, repo)
            found += input_found
            notes[name] += input_notes
        else:
            notes[name].append(
                f"{len(summary.inputs)} inputs not checked (--code-only)"
            )
        if found:
            problems[name] = found
    return problems, notes
