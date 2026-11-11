"""The reported-value record and the summary file that holds one builder's values."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported.code_record import CodeRecord, commit_label
from continuum_swr_replay.reported.questions import tag_problems

SCHEMA_VERSION = 6
"""The layout of a summary file. Bump it when a field is added or renamed.
Version 6: each value records its ``function``, the builder function that
made it, and the ``figures`` that show it. Whether the manuscript cites a
value is not recorded: it is read from the manuscript
(:func:`~continuum_swr_replay.reported.emit.cited_macros`)."""

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
class Tags:
    """Which questions a group of values answers, and which figures show them:
    what a builder declares once, in its ``TAGS`` table, for each group of values.

    Attributes
    ----------
    questions : tuple of str
        Question ids (:data:`~continuum_swr_replay.reported.questions.QUESTIONS`).
    figures : tuple of str
        Figures or lettered panels (``"2"``, ``"2b"``, ``"memoryless_comparison"``;
        :data:`~continuum_swr_replay.reported.questions.FIGURES`); empty for
        values no figure shows.
    """

    questions: tuple[str, ...]
    figures: tuple[str, ...] = ()

    def __or__(self, other: Tags) -> Tags:
        """The questions and figures of both, in order, each once."""
        return Tags(
            tuple(dict.fromkeys((*self.questions, *other.questions))),
            tuple(dict.fromkeys((*self.figures, *other.figures))),
        )


@dataclass(frozen=True)
class Value:
    """One reported number and where it comes from.

    Attributes
    ----------
    key : str
        Lower-case words joined by underscores, unique within the summary. No
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
    figures : tuple of str
        The figures or lettered panels that show it (``"2"``, ``"2b"``;
        :data:`~continuum_swr_replay.reported.questions.FIGURES`); empty when
        none does.
    source : str
        The file, run or table it was computed from, and the selection.
    function : str
        The function that made it, in its summary's builder module
        (``event_cohort_values`` of ``reported/events.py``). That function
        names and describes the value; the statistic is computed by the
        ``analysis`` function it calls.
    pending : str or None
        Why the value may not be quoted yet (for example a value from an
        untracked script), or None.

    Its tags are checked against the question list when its summary is written
    (:meth:`Summary.write`), not here, so a summary built before the list changed
    can still be read.
    """

    key: str
    value: float | int
    kind: str
    style: str
    unit: str
    description: str
    questions: tuple[str, ...]
    figures: tuple[str, ...]
    source: str
    function: str
    pending: str | None = None

    def __post_init__(self) -> None:
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
        if not self.questions:
            raise ValueError(f"{self.key}: name the questions it answers")
        if not self.function.isidentifier():
            raise ValueError(
                f"{self.key}: function {self.function!r} is not a function's name"
            )
        if self.pending is not None and not self.pending.strip():
            raise ValueError(f"{self.key}: a pending note must say why")
        if "\n" in self.description + (self.pending or ""):
            raise ValueError(
                f"{self.key}: its description and pending note are single lines"
            )


def value_factory(
    function: Callable[..., Any],
    source: str,
    tags: Tags,
    pending: str | None = None,
) -> Callable[..., Value]:
    """A maker of the values one builder function names and describes.

    Parameters
    ----------
    function : callable
        The builder function the values are made in. Its name is recorded as
        each value's ``function``, so a value can be traced to its code.
    source : str
        The file, run or table the values are computed from.
    tags : Tags
        The questions the values answer and the figures that show them.
    pending : str, optional
        Why the values may not be quoted yet.

    Returns
    -------
    make : callable
        ``make(key, value, kind, style, description, unit="", *, tags=None,
        pending=None)``, a :class:`Value`. ``tags`` and ``pending`` replace the
        shared ones for that value. A whole count is stored as an int, a
        fraction or a quantity as a float.
    """
    shared_tags, shared_pending = tags, pending

    def make(
        key: str,
        value: float | int,
        kind: str,
        style: str,
        description: str,
        unit: str = "",
        *,
        tags: Tags | None = None,
        pending: str | None = None,
    ) -> Value:
        if kind != "count":
            value = float(value)
        elif float(value).is_integer():
            value = int(value)
        chosen = tags or shared_tags
        return Value(
            key=key,
            value=value,
            kind=kind,
            style=style,
            unit=unit,
            description=description,
            questions=chosen.questions,
            figures=chosen.figures,
            source=source,
            function=function.__name__,
            pending=shared_pending if pending is None else pending,
        )

    return make


@dataclass
class Summary:
    """One summary's values, the provenance of the inputs they were read from, and
    the code that built them.

    Attributes
    ----------
    name : str
        A key of :data:`SUMMARIES`.
    produced_by : str
        The script that built it, relative to the repository.
    inputs : list of dict
        One record per input file (:func:`file_record`) or set of files
        (:func:`files_record`).
    values : dict of str to Value
        The values, by key.
    code : CodeRecord or None
        The record of the files of the checkout that built it
        (:class:`~continuum_swr_replay.reported.code_record.CodeRecord`); None
        for a summary a builder returned, before ``summarize.py`` records it.
    """

    name: str
    produced_by: str
    inputs: list[dict[str, Any]] = field(default_factory=list)
    values: dict[str, Value] = field(default_factory=dict)
    code: CodeRecord | None = None

    def __post_init__(self) -> None:
        if self.name not in SUMMARIES:
            raise ValueError(f"no summary {self.name!r}; one of {list(SUMMARIES)}")

    @property
    def commit(self) -> str:
        """The commit it was built at, from its code record
        (:func:`~continuum_swr_replay.reported.code_record.commit_label`); empty
        before ``summarize.py`` records it."""
        return commit_label(self.code) if self.code is not None else ""

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
            "inputs": self.inputs,
            "values": {
                key: {k: v for k, v in asdict(value).items() if k != "key"}
                for key, value in sorted(self.values.items())
            },
        }
        return json.dumps(payload, indent=1, sort_keys=False, allow_nan=False) + "\n"

    def write(self, path: Path) -> None:
        """Write the summary to ``path``.

        Raises
        ------
        ValueError
            If a value's tags do not match the question list
            (:func:`summary_tag_problems`); nothing is written.
        """
        problems = summary_tag_problems(self)
        if problems:
            listed = "; ".join(f"{k}: {', '.join(p)}" for k, p in problems.items())
            raise ValueError(f"{self.name}: values with tag problems: {listed}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(self.to_json())
        tmp.replace(path)


def pend(
    values: Iterable[Value],
    note: str | None,
    selected: Callable[[Value], bool] = lambda value: True,
) -> list[Value]:
    """``values`` with ``note`` added to the pending note of the selected ones.

    The one rule for a value read from a stale output (one made from other
    tables than the current ones): it is kept and marked pending with the
    reason, never dropped and never a reason to refuse the build. The release
    check refuses a pending value the manuscript cites.
    """
    if note is None:
        return list(values)
    return [
        replace(v, pending=note if v.pending is None else f"{v.pending}; {note}")
        if selected(v)
        else v
        for v in values
    ]


def read_summary(path: Path) -> Summary:
    """A summary file read back, its values as :class:`Value`. Its tags are not
    checked: :func:`summary_tag_problems` lists any that the question list no
    longer accepts.

    Parameters
    ----------
    path : Path
        A summary file, ``<name>.json``.

    Raises
    ------
    ValueError
        If the schema is not :data:`SCHEMA_VERSION`, or the file lacks ``code``.
    """
    payload = json.loads(path.read_text())
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema {payload['schema_version']}, expected {SCHEMA_VERSION}"
        )
    if "code" not in payload:
        raise ValueError(f"{path}: no code")
    summary = Summary(
        name=payload["summary"],
        produced_by=payload["produced_by"],
        inputs=payload["inputs"],
        code=payload["code"],
    )
    for key, fields in payload["values"].items():
        fields = dict(fields)
        fields["questions"] = tuple(fields["questions"])
        fields["figures"] = tuple(fields["figures"])
        summary.add(Value(key=key, **fields))
    return summary


def read_summaries(directory: Path) -> dict[str, Summary]:
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
        summary = read_summary(directory / file)
        if summary.name != name:
            raise ValueError(f"{directory / file}: holds summary {summary.name}")
        summaries[name] = summary
    return summaries


def summary_tag_problems(summary: Summary) -> dict[str, list[str]]:
    """Each value of ``summary`` whose tags do not match the question list, with
    what is wrong (:func:`~continuum_swr_replay.reported.questions.tag_problems`)."""
    problems = {
        key: tag_problems(value.questions, value.figures)
        for key, value in summary.values.items()
    }
    return {key: found for key, found in problems.items() if found}


def rel(path: Path) -> str:
    """``path`` relative to the repository when it is under it: how a builder
    names an input in a value's ``source``."""
    return (
        str(path.relative_to(ROOT_DIR)) if path.is_relative_to(ROOT_DIR) else str(path)
    )


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
