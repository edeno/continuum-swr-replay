"""What stands between the summaries and a release: the checks ``make
release-check`` runs at a milestone (a draft going to the co-authors, submission,
a release), and never on every commit."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from continuum_swr_replay.reported.code_record import code_changes
from continuum_swr_replay.reported.emit import cited_values
from continuum_swr_replay.reported.values import Summary, summary_tag_problems
from continuum_swr_replay.runs.io import sha256_file


def staleness(summary: Summary, repo: Path) -> list[str]:
    """Why ``summary``'s code record does not vouch for it now; empty when it does.

    Each is named: no code record, a record that lists no files, a build from
    uncommitted code (a trial build put among the summaries), and recorded
    files (or ``uv.lock``) that changed since.
    """
    code = summary.code
    if code is None:
        return ["no code record"]
    problems = []
    if not code["files"]:
        problems.append("the code record lists no files")
    if code["uncommitted"]:
        problems.append(
            f"built from uncommitted code: {', '.join(code['uncommitted'])}"
        )
    changed = code_changes(code, repo)
    if changed:
        problems.append(f"code changed since the build: {', '.join(changed)}")
    return problems


def input_problems(
    summary: Summary, root: Path, hashes: dict[Path, str] | None = None
) -> tuple[list[str], list[str]]:
    """What the recorded inputs say: problems (a single file whose sha256 differs
    now, or that is not on this machine) and notes (sets of files, records with
    ``n_files``: they name a label, not the selection the builder made, so they
    cannot be re-checked without rebuilding).

    ``hashes`` holds the sha256 of each file already read, and is filled in, so
    that an input several summaries record is read once."""
    hashes = {} if hashes is None else hashes
    problems, notes = [], []
    for record in summary.inputs:
        path = record["path"]
        file = root / path  # an absolute path stays as it is
        if "n_files" in record:
            notes.append(f"input file set not re-checked: {path}")
        elif not file.is_file():
            problems.append(f"input not on this machine: {path}")
        else:
            if file not in hashes:
                hashes[file] = sha256_file(file)
            if hashes[file] != record["sha256"]:
                problems.append(f"input changed since the build: {path}")
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
    hashes: dict[Path, str] = {}
    for name, summary in summaries.items():
        found = staleness(summary, repo)
        tags = summary_tag_problems(summary)
        if tags:
            noun = "value" if len(tags) == 1 else "values"
            found.append(f"{len(tags)} {noun} with tag problems: {', '.join(tags)}")
        pending = [
            key for key in cited[name] if summary.values[key].pending is not None
        ]
        if pending:
            found.append(f"the manuscript cites pending values: {', '.join(pending)}")
        notes[name] = [
            f"the manuscript cites {len(cited[name])} of {len(summary.values)} values"
        ]
        if check_inputs:
            input_found, input_notes = input_problems(summary, repo, hashes)
            found += input_found
            notes[name] += input_notes
        else:
            notes[name].append(
                f"{len(summary.inputs)} inputs not checked (--code-only)"
            )
        if found:
            problems[name] = found
    return problems, notes
