"""The reported-value record and the summary file that holds one part's values."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
"""The layout of a summary file. Bump it when a field is added or renamed."""

PARTS = {1: "the model", 2: "the events", 3: "behavior", 4: "other brain areas"}
"""The paper's four parts (``docs/analysis_map.md``)."""

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
    claims : tuple of str
        The claims-table rows it supports.
    used_in : tuple of str
        "paper", "supplement", or "figure: <name>".
    source : str
        The file, run or table it was computed from, and the selection.
    pending : str or None
        Why the value may not be quoted yet (for example the Widloski pilot
        checks), or None.
    """

    key: str
    value: float | int
    kind: str
    style: str
    unit: str
    description: str
    claims: tuple[str, ...]
    used_in: tuple[str, ...]
    source: str
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
        if not self.claims or not self.used_in:
            raise ValueError(f"{self.key}: name its claims and where it is used")


@dataclass
class Summary:
    """One part's values and the provenance of the inputs they were read from."""

    part: int
    produced_by: str
    commit: str
    inputs: list[dict[str, Any]] = field(default_factory=list)
    values: dict[str, Value] = field(default_factory=dict)

    def add(self, *values: Value) -> None:
        for value in values:
            if value.key in self.values:
                raise ValueError(f"part {self.part}: duplicate key {value.key}")
            self.values[value.key] = value

    def to_json(self) -> str:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "part": self.part,
            "title": PARTS[self.part],
            "produced_by": self.produced_by,
            "commit": self.commit,
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


def read_summary(path: Path) -> Summary:
    """A summary file read back, its values as :class:`Value`."""
    payload = json.loads(path.read_text())
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: schema {payload['schema_version']}, expected {SCHEMA_VERSION}"
        )
    summary = Summary(
        part=payload["part"],
        produced_by=payload["produced_by"],
        commit=payload["commit"],
        inputs=payload["inputs"],
    )
    for key, fields in payload["values"].items():
        fields = dict(fields)
        fields["claims"] = tuple(fields["claims"])
        fields["used_in"] = tuple(fields["used_in"])
        summary.add(Value(key=key, **fields))
    return summary


def file_record(path: Path, root: Path) -> dict[str, Any]:
    """An input file's path (relative to ``root`` when under it) and sha256."""
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    try:
        shown = str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        shown = str(path)
    return {"path": shown, "sha256": digest}


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
    try:
        folder = str(paths[0].parent.resolve().relative_to(root.resolve()))
    except (ValueError, IndexError):
        folder = str(paths[0].parent) if paths else ""
    record: dict[str, Any] = {
        "path": f"{folder}/{name}",
        "n_files": len(paths),
        "sha256": digest.hexdigest(),
    }
    if note:
        record["note"] = note
    return record


def values_by_key(summaries: Mapping[int, Summary]) -> dict[str, Value]:
    """Every value of every summary, keyed ``part{n}.{key}``."""
    return {
        f"part{part}.{key}": value
        for part, summary in summaries.items()
        for key, value in summary.values.items()
    }
