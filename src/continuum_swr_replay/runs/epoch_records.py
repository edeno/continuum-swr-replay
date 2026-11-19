"""A run whose epochs each write their own record, and whose manifest gathers them.

Each epoch's record, ``{epoch_id}_{name}.json``, sits beside its table
``{epoch_id}_{name}.parquet`` (``runs.io.output_path``) and carries the run's
``settings``, its ``inputs`` (each input's ``sha256``, and where it was read)
and the code's ``provenance``, with the epoch's ``status``. The run's
``manifest.json`` is gathered from the records (:func:`gather_manifest`), so a
run resumed, re-run for its failed epochs, or split across hosts writing into
one folder needs no manifest of its own: an epoch with an ``ok`` record made
under the same settings and input digests is not run again
(:func:`epochs_to_run`), and the gather refuses records made under others.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from continuum_swr_replay.runs.io import write_json

MANIFEST = "manifest.json"


def record_path(output_dir: Path, epoch_id: str, name: str) -> Path:
    """Where an epoch's record of table ``name`` is."""
    return output_dir / f"{epoch_id}_{name}.json"


def _digests(inputs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Each input's SHA-256: what makes two runs' inputs the same (their paths
    differ between checkouts)."""
    return {key: value.get("sha256") for key, value in inputs.items()}


def same_run(record: Mapping[str, Any], header: Mapping[str, Any]) -> bool:
    """Whether ``record`` was made under ``header``'s settings and inputs."""
    return record.get("settings") == header["settings"] and _digests(
        record.get("inputs", {})
    ) == _digests(header["inputs"])


def read_record(output_dir: Path, epoch_id: str, name: str) -> dict | None:
    """An epoch's record, or None when it has none."""
    path = record_path(output_dir, epoch_id, name)
    return json.loads(path.read_text()) if path.exists() else None


def write_record(
    output_dir: Path,
    epoch_id: str,
    name: str,
    header: Mapping[str, Any],
    record: Mapping[str, Any],
) -> None:
    """Write an epoch's record: ``record`` with the run's settings, inputs and
    provenance (``header``)."""
    write_json(
        record_path(output_dir, epoch_id, name),
        {
            **record,
            "settings": header["settings"],
            "inputs": header["inputs"],
            "provenance": header["provenance"],
        },
        indent=1,
        sort_keys=True,
    )


def epochs_to_run(
    output_dir: Path,
    name: str,
    epochs: Iterable[str],
    header: Mapping[str, Any],
    overwrite: bool,
) -> list[str]:
    """The epochs to run, in order: all with ``overwrite``, else those without
    an ``ok`` record made under ``header``'s settings and inputs."""
    to_run = []
    for epoch_id in epochs:
        record = read_record(output_dir, epoch_id, name)
        done = (
            record is not None
            and record.get("status") == "ok"
            and same_run(record, header)
        )
        if overwrite or not done:
            to_run.append(epoch_id)
    return to_run


def gather_manifest(
    output_dir: Path, name: str, header: Mapping[str, Any]
) -> dict[str, Any]:
    """Write ``manifest.json`` from every epoch's record in ``output_dir``.

    The manifest's ``epochs`` are the records (each with its own
    ``provenance``); its ``settings`` and ``inputs`` are ``header``'s, and its
    ``provenance`` lists every distinct one the records hold.

    Raises
    ------
    ValueError
        If a record was made under other settings or inputs than ``header``'s.
    """
    suffix = f"_{name}.json"
    records = {
        path.name[: -len(suffix)]: json.loads(path.read_text())
        for path in sorted(output_dir.glob(f"*{suffix}"))
    }
    other = sorted(e for e, record in records.items() if not same_run(record, header))
    if other:
        raise ValueError(
            f"{len(other)} records in {output_dir} were made under other settings "
            f"or inputs: {other[:3]}"
        )
    provenances: list[dict] = []
    for record in records.values():
        if record["provenance"] not in provenances:
            provenances.append(record["provenance"])
    manifest = {
        "produced_by": header["produced_by"],
        "settings": header["settings"],
        "inputs": header["inputs"],
        "provenance": provenances,
        "epochs": records,
    }
    write_json(output_dir / MANIFEST, manifest, indent=1, sort_keys=True)
    return manifest
