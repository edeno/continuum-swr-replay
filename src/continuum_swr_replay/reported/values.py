"""The reported-value record and the summary file that holds one part's values."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import InitVar, asdict, dataclass, field
from pathlib import Path
from typing import Any

from continuum_swr_replay.reported.code_record import CodeRecord, CodeRecordError
from continuum_swr_replay.reported.questions import tag_problems

SCHEMA_VERSION = 5
"""The layout of a summary file. Bump it when a field is added or renamed.
Version 5 names the summary (``summary``, a key of :data:`SUMMARIES`), tags each
value with the ``questions`` it answers, holds ``code``, the record of the code
that built it (:class:`~continuum_swr_replay.reported.code_record.CodeRecord`),
and, for a summary converted from an earlier layout, ``migrated``."""

SUMMARIES = {
    "survey": "the published replay studies the introduction summarizes "
    "(Supplementary Table S1)",
    "detection": "the detector: the held-out tetrode test of each decoder state, "
    "and the comparison with the Widloski & Foster detector on simulated replays",
    "events": "the events: their overlap with SWRs and the other detectors, their "
    "spike support, and their properties",
    "behavior": "the events and the task: event rate and represented content",
    "mpfc": "how mPFC units respond to the events",
}
"""Every summary file, ``paper/reported/<name>.json``, by name, in order. A summary
is named for what it holds, not for a part of the paper: a value's part is its
question's (:mod:`~continuum_swr_replay.reported.questions`)."""

KINDS = ("count", "fraction", "quantity")
"""A count is exact; a fraction is a share in [0, 1]; a quantity has a unit."""

STYLES = ("count", "pct", "pctnum", "sig2", "sig2s", "int", "dec2", "pval")
"""How a value is printed (:mod:`continuum_swr_replay.reported.formatting`)."""

_KEY = re.compile(r"[a-z]+(_[a-z]+)*")


@dataclass(frozen=True)
class Value:
    """One reported number and where it comes from.

    Attributes
    ----------
    key : str
        Lower-case words joined by underscores, unique within the part. No
        digits: the key becomes a LaTeX macro name.
    value : float or int
        At full precision. Never NaN or infinite.
    kind : str
        One of :data:`KINDS`.
    style : str
        The default print style, one of :data:`STYLES`.
    unit : str
        The unit of a quantity ("ms", "cm", "nats per spike", ...); empty for
        counts and fractions.
    description : str
        What the number is, precisely enough to recompute it.
    questions : tuple of str
        The ids of the questions it answers
        (:data:`~continuum_swr_replay.reported.questions.QUESTIONS`).
    used_in : tuple of str
        "paper", "supplement", or a figure,
        ``"figure: <id>"`` or ``"figure: <id><panel>"``
        (:data:`~continuum_swr_replay.reported.questions.FIGURES`).
    source : str
        The file, run or table it was computed from, and the selection.
    pending : str or None
        Why the value may not be quoted yet (for example a value from an
        untracked script), or None.
    check_tags : bool
        Refuse ``questions`` or ``used_in`` that do not match the question list
        (:func:`~continuum_swr_replay.reported.questions.tag_problems`). Only
        :func:`read_summary` turns it off, to read a summary built before the list
        changed. Not stored: a value carries no record of whether it was checked.
    """

    key: str
    value: float | int
    kind: str
    style: str
    unit: str
    description: str
    questions: tuple[str, ...]
    used_in: tuple[str, ...]
    source: str
    pending: str | None = None
    check_tags: InitVar[bool] = True

    def __post_init__(self, check_tags: bool) -> None:
        if not _KEY.fullmatch(self.key):
            raise ValueError(f"key {self.key!r}: lower-case words and underscores only")
        if self.kind not in KINDS:
            raise ValueError(f"{self.key}: kind {self.kind!r} not in {KINDS}")
        if self.style not in STYLES:
            raise ValueError(f"{self.key}: style {self.style!r} not in {STYLES}")
        if not math.isfinite(float(self.value)):
            raise ValueError(f"{self.key}: value {self.value} is not finite")
        if self.kind == "count" and float(self.value) != int(self.value):
            raise ValueError(f"{self.key}: a count must be whole, got {self.value}")
        if self.kind == "fraction" and not 0.0 <= float(self.value) <= 1.0:
            raise ValueError(
                f"{self.key}: a fraction must be in [0, 1], got {self.value}"
            )
        if not self.questions or not self.used_in:
            raise ValueError(f"{self.key}: name its questions and where it is used")
        if check_tags:
            problems = tag_problems(self.questions, self.used_in)
            if problems:
                raise ValueError(f"{self.key}: {'; '.join(problems)}")


@dataclass
class Summary:
    """One summary's values, the provenance of the inputs they were read from, and
    the code that built them.

    Attributes
    ----------
    name : str
        A key of :data:`SUMMARIES`.
    code : CodeRecord, CodeRecordError or None
        The record of the code that built it
        (:class:`~continuum_swr_replay.reported.code_record.CodeRecord`); a
        :class:`~continuum_swr_replay.reported.code_record.CodeRecordError` when
        git could not answer; None when no build recorded it (a summary that was
        migrated, or one a builder returned before ``summarize.py`` stamped it).
    migrated : dict or None
        For a summary converted from an earlier layout instead of rebuilt: what it
        was converted from, at which commit, by what, and what changed. A rebuild
        replaces it with a summary that has none.
    """

    name: str
    produced_by: str
    commit: str
    inputs: list[dict[str, Any]] = field(default_factory=list)
    values: dict[str, Value] = field(default_factory=dict)
    code: CodeRecord | CodeRecordError | None = None
    migrated: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.name not in SUMMARIES:
            raise ValueError(f"no summary {self.name!r}; one of {list(SUMMARIES)}")

    def add(self, *values: Value) -> None:
        for value in values:
            if value.key in self.values:
                raise ValueError(f"{self.name}: duplicate key {value.key}")
            self.values[value.key] = value

    def to_json(self) -> str:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "summary": self.name,
            "title": SUMMARIES[self.name],
            "produced_by": self.produced_by,
            "commit": self.commit,
            "code": self.code,
            "migrated": self.migrated,
            "inputs": self.inputs,
            "values": {
                key: {k: v for k, v in asdict(value).items() if k != "key"}
                for key, value in sorted(self.values.items())
            },
        }
        return json.dumps(payload, indent=1, sort_keys=False, allow_nan=False) + "\n"

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.to_json())
        tmp.replace(path)


def read_summary(path: Path, *, check_tags: bool = True) -> Summary:
    """A summary file read back, its values as :class:`Value`.

    Parameters
    ----------
    path : Path
        A summary file, ``<name>.json``.
    check_tags : bool
        Check each value's question and figure tags
        (:func:`~continuum_swr_replay.reported.questions.tag_problems`). False reads a
        summary built before the question list changed, so that its numbers can
        still be emitted; :func:`summary_tag_problems` then lists what to fix.

    Raises
    ------
    ValueError
        If the schema is not :data:`SCHEMA_VERSION`, or the file lacks ``code``
        or ``migrated``.
    """
    payload = json.loads(path.read_text())
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema {payload['schema_version']}, expected {SCHEMA_VERSION}"
        )
    missing = {"code", "migrated"} - set(payload)
    if missing:
        raise ValueError(f"{path}: no {', '.join(sorted(missing))}")
    summary = Summary(
        name=payload["summary"],
        produced_by=payload["produced_by"],
        commit=payload["commit"],
        inputs=payload["inputs"],
        code=payload["code"],
        migrated=payload["migrated"],
    )
    for key, fields in payload["values"].items():
        fields = dict(fields)
        fields["questions"] = tuple(fields["questions"])
        fields["used_in"] = tuple(fields["used_in"])
        summary.add(Value(key=key, check_tags=check_tags, **fields))
    return summary


def read_summaries(directory: Path, *, check_tags: bool = True) -> dict[str, Summary]:
    """Every summary in ``directory``, by name, in the order of :data:`SUMMARIES`:
    exactly one ``<name>.json`` for each, each saying it is that summary.

    Raises
    ------
    ValueError
        If a summary's file is missing, another ``.json`` is there, or a file's
        ``summary`` is not the one its name gives.
    """
    expected = {f"{name}.json": name for name in SUMMARIES}
    found = {path.name for path in directory.glob("*.json")}
    if found != set(expected):
        missing = sorted(set(expected) - found)
        extra = sorted(found - set(expected))
        raise ValueError(
            f"{directory}: summaries missing {missing or 'none'}, "
            f"unexpected {extra or 'none'}"
        )
    summaries = {}
    for file, name in expected.items():
        summary = read_summary(directory / file, check_tags=check_tags)
        if summary.name != name:
            raise ValueError(f"{directory / file}: holds summary {summary.name}")
        summaries[name] = summary
    return summaries


def summary_tag_problems(summary: Summary) -> dict[str, list[str]]:
    """Each value of ``summary`` whose tags do not match the question list, with
    what is wrong (:func:`~continuum_swr_replay.reported.questions.tag_problems`)."""
    problems = {
        key: tag_problems(value.questions, value.used_in)
        for key, value in summary.values.items()
    }
    return {key: found for key, found in problems.items() if found}


def _shown(path: Path, root: Path) -> str:
    """``path`` relative to ``root`` when it is under it, before or after
    symlinks are followed (a worktree's ``Processed-Data`` is often a link)."""
    for candidate, base in (
        (path.absolute(), root.absolute()),
        (path.resolve(), root.resolve()),
    ):
        if candidate.is_relative_to(base):
            return str(candidate.relative_to(base))
    return str(path)


def file_record(path: Path, root: Path) -> dict[str, Any]:
    """An input file's path (relative to ``root`` when under it) and sha256."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": _shown(path, root), "sha256": digest}


def files_record(
    name: str, paths: Iterable[Path], root: Path, note: str = ""
) -> dict[str, Any]:
    """Many input files as one record: their count and one sha256 over each
    file's name and sha256, in sorted order."""
    digest = hashlib.sha256()
    paths = sorted(paths)
    for path in paths:
        digest.update(path.name.encode())
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    folder = _shown(paths[0].parent, root) if paths else ""
    record: dict[str, Any] = {
        "path": f"{folder}/{name}",
        "n_files": len(paths),
        "sha256": digest.hexdigest(),
    }
    if note:
        record["note"] = note
    return record


def values_by_key(summaries: Mapping[str, Summary]) -> dict[str, Value]:
    """Every value of every summary, keyed ``<summary>.<key>``."""
    return {
        f"{name}.{key}": value
        for name, summary in summaries.items()
        for key, value in summary.values.items()
    }
