"""Whether an output was made from the current inputs.

An output records the SHA-256 of each input it read: a measurement run in its
``manifest.json`` (:func:`recorded_sha256`), a PFC bundle or session table in
its provenance record under the names :func:`table_hashes` gives. A reader
hashes the current inputs and counts the records that name another digest
(:func:`stale_counts`); the stop-context fits and the marked-family bundles
each have a reader of their own records (:func:`stop_context_stale_counts`,
:func:`marked_family_stale_counts`). A fitted model's sidecar is read by
``runs.fit_provenance``.

The functions here read and compare; the summary builders write the pending
notes from what they return.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from continuum_swr_replay.runs.io import sha256_file

__all__ = [
    "marked_family_stale_counts",
    "read_output_manifest",
    "recorded_sha256",
    "stale_counts",
    "stop_context_stale_counts",
    "table_hashes",
]


def read_output_manifest(folder: Path) -> dict:
    """The ``manifest.json`` a measurement wrote in ``folder``, as recorded.

    Unlike ``runs.io.read_manifest`` it asks for no per-epoch record: a
    measurement over the whole cohort (a bootstrap, a reference) keeps none.
    """
    return json.loads((folder / "manifest.json").read_text())


def recorded_sha256(manifest: Mapping[str, Any], name: str) -> str | None:
    """The SHA-256 a manifest records for input ``name``, or None.

    A manifest records it under ``name`` or ``inputs.<name>``, as
    ``{"sha256": ...}``.
    """
    return (manifest.get(name) or manifest.get("inputs", {}).get(name) or {}).get(
        "sha256"
    )


def stale_counts(
    records: Iterable[Mapping[str, Any]], current: Mapping[str, str]
) -> dict[str, int]:
    """For each input of ``current``, how many ``records`` record another SHA-256.

    Parameters
    ----------
    records : iterable of Mapping
        Each output's record: input name -> the SHA-256 it was made from. A
        record without the name counts as another digest.
    current : Mapping of str to str
        Input name -> the current file's SHA-256 (:func:`table_hashes`).

    Returns
    -------
    dict of str to int
        Input name -> the number of records that differ, in ``current``'s order.
    """
    records = list(records)
    return {
        name: sum(record.get(name) != sha for record in records)
        for name, sha in current.items()
    }


def table_hashes(event_table: Path, stop_table: Path) -> dict[str, str]:
    """The SHA-256 of the canonical event and stop tables, under the names a
    PFC bundle's or session table's provenance records them by."""
    return {
        "canonical_event_table_sha256": sha256_file(event_table),
        "stop_table_sha256": sha256_file(stop_table),
    }


def stop_context_stale_counts(
    results: Path, bundle_dir: Path, current: Mapping[str, str], n_cohort: int
) -> dict[str, int]:
    """How many of the stop-context fit's inputs differ from the current files.

    The fit's ``results.provenance.json`` records the SHA-256 of every canonical
    bundle and cached session table it read; each session table's provenance
    records the event and stop tables it was assembled from
    (``input_provenance``). A bundle or session table that is missing, or whose
    file differs (matched by file name), counts as differing.

    Parameters
    ----------
    results : Path
        The fit's ``results.csv``.
    bundle_dir : Path
        The folder of the canonical bundles.
    current : Mapping of str to str
        The current tables' digests (:func:`table_hashes`).
    n_cohort : int
        The number of PFC-cohort epochs the fit must cover.

    Returns
    -------
    dict of str to int
        ``bundle_sha256``, ``session_table_sha256`` and each name of
        ``current`` -> the number of sessions that differ.

    Raises
    ------
    ValueError
        If the fit does not cover the ``n_cohort`` PFC epochs.
    """
    provenance = json.loads(results.with_name("results.provenance.json").read_text())
    if len(provenance["bundles"]) != n_cohort:
        raise ValueError(
            f"{results}: fit on {len(provenance['bundles'])} sessions, not the "
            f"{n_cohort} of the PFC cohort"
        )
    sessions_dir = results.parent / "sessions"
    tables = {
        Path(path).name: sha for path, sha in provenance["session_tables"].items()
    }
    records = [
        json.loads((sessions_dir / name).with_suffix(".provenance.json").read_text())
        for name in tables
    ]

    def differs(path: Path, sha: str) -> bool:
        return not path.exists() or sha256_file(path) != sha

    return {
        "bundle_sha256": sum(
            differs(bundle_dir / Path(path).name, sha)
            for path, sha in provenance["bundles"].items()
        ),
        "session_table_sha256": sum(
            differs(sessions_dir / name, sha) for name, sha in tables.items()
        ),
        **stale_counts((r["input_provenance"] for r in records), current),
    }


def marked_family_stale_counts(
    summary_path: Path, current: Mapping[str, str], n_cohort: int
) -> tuple[dict[str, int], int]:
    """How many marked-family bundles differ from the current inputs.

    Reads every bundle sidecar (``*_marked_family.provenance.json``) beside the
    summary, wherever it lives: each records the event and stop tables its fit
    read. A bundle file written after the summary also counts, under
    ``"bundle written after the summary"``.

    Parameters
    ----------
    summary_path : Path
        The marked-family summary.
    current : Mapping of str to str
        The current tables' digests (:func:`table_hashes`).
    n_cohort : int
        The number of PFC-cohort epochs the sidecars must cover.

    Returns
    -------
    counts : dict of str to int
        Each name of ``current``, then ``"bundle written after the summary"``
        -> the number of bundles that differ.
    n_sidecars : int
        The number of sidecars read.

    Raises
    ------
    ValueError
        If the sidecars do not cover the ``n_cohort`` PFC epochs.
    """
    folder = summary_path.parent
    sidecars = [
        json.loads(path.read_text())
        for path in sorted(folder.glob("*_marked_family.provenance.json"))
    ]
    sessions = {(s["nwb_file_name"], s["epoch_name"]) for s in sidecars}
    if len(sessions) != n_cohort:
        raise ValueError(
            f"{folder}: marked-family bundles of {len(sessions)} sessions, not the "
            f"{n_cohort} of the PFC cohort"
        )
    written = summary_path.stat().st_mtime
    counts = {
        **stale_counts(sidecars, current),
        "bundle written after the summary": sum(
            path.stat().st_mtime > written
            for path in folder.glob("*_marked_family.pkl")
        ),
    }
    return counts, len(sidecars)
