"""The artifact run folder: one cores file per epoch, and the run's manifest.

A run folder, ``<root>/<run_id>/``, holds:

- ``{epoch_id}_artifact_intervals.parquet``: the epoch's unpadded cores and
  their evidence (:data:`CORE_COLUMNS`), one row per core; no rows for an
  epoch without artifacts. The file never holds padded intervals: the
  reader pads the cores into the mask (``artifact_masks.artifact_intervals``).
- ``manifest.json``: the run's settings and one record per epoch, with its
  ``status``, its span (``span_s``, the first and last LFP sample the detector
  read) and its file's path and sha256 (``outputs.cores``).

``scripts/detection/run_artifact_detection.py`` writes it; ``swr_source`` reads
it through :func:`read_epoch_cores`, which serves an epoch only from a manifest
that names the folder, holds the epoch as ``ok``, and records the sha256 the
file still has.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_definition.artifact_masks import (
    artifact_intervals as padded_cores,
)
from continuum_swr_replay.runs import io as runs_io
from continuum_swr_replay.runs.io import sha256_file
from continuum_swr_replay.tables.io import write_table

__all__ = [
    "CORES_SUFFIX",
    "CORE_COLUMNS",
    "ArtifactRunError",
    "EpochCores",
    "cores_path",
    "empty_cores",
    "epoch_artifact_intervals",
    "make_cores",
    "read_epoch_cores",
    "read_manifest",
    "validate_cores",
    "write_cores",
]

CORES_SUFFIX = "_artifact_intervals.parquet"

CORE_COLUMNS: Mapping[str, str] = {
    "kind": "object",
    "signatures": "object",
    "core_start": "float64",
    "core_end": "float64",
    "spec_max": "float64",
    "n_windows": "float64",
    "ref_j5_max": "float64",
    "ca1_frac_max": "float64",
    "spk_frac_max": "float64",
    "spk_ood_max": "float64",
}
"""A cores file's columns and dtypes, in order.

- ``kind``: ``"S"`` (sustained ripple-band noise) or ``"T"`` (a transient).
- ``signatures``: ``"S"``, or the T signatures that fired, sorted and joined
  by ``+``: ``ca1``, ``ref`` and ``spk``.
- ``core_start``, ``core_end``: the unpadded core, seconds, closed.
- S evidence: ``spec_max``, the largest channel specificity in the block, and
  ``n_windows``, its 1-s windows (gap-filled ones included).
- T evidence: ``ref_j5_max`` (largest reference |J5|, robust units),
  ``ca1_frac_max`` (largest fraction of CA1 channels with a persistent step at
  one sample), ``spk_frac_max`` (largest fraction of tetrodes in one spike-line
  bin) and ``spk_ood_max`` (largest OOD fraction among those bins).

Evidence a core's kind or signatures do not produce is NaN.
"""

_KINDS = frozenset({"S", "T"})


class ArtifactRunError(ValueError):
    """An artifact run folder cannot serve an epoch: no manifest, another run's
    manifest, an epoch it did not run or that failed, a missing or changed file,
    or malformed cores."""


@dataclass(frozen=True)
class EpochCores:
    """One epoch's cores and the span they were detected over."""

    cores: pd.DataFrame
    """Columns :data:`CORE_COLUMNS`, one row per core, in time order."""
    span: tuple[float, float]
    """The first and last LFP sample time the detector read, seconds."""


def cores_path(run_dir: Path, epoch_id: str) -> Path:
    """``run_dir / {epoch_id}_artifact_intervals.parquet``."""
    return Path(run_dir) / f"{epoch_id}{CORES_SUFFIX}"


def empty_cores() -> pd.DataFrame:
    """A cores table with no rows, in :data:`CORE_COLUMNS`."""
    return pd.DataFrame(
        {name: pd.Series(dtype=dtype) for name, dtype in CORE_COLUMNS.items()}
    )


def make_cores(columns: Mapping[str, Any] | pd.DataFrame) -> pd.DataFrame:
    """A cores table from some of its columns; the others are NaN.

    Parameters
    ----------
    columns : mapping or pd.DataFrame
        Columns of :data:`CORE_COLUMNS` (at least ``kind``, ``signatures``,
        ``core_start`` and ``core_end`` for a valid table).

    Returns
    -------
    pd.DataFrame
        In :data:`CORE_COLUMNS`' order and dtypes.
    """
    frame = pd.DataFrame(columns)
    return frame.reindex(columns=list(CORE_COLUMNS)).astype(dict(CORE_COLUMNS))


def validate_cores(cores: pd.DataFrame, what: str) -> pd.DataFrame:
    """`cores` in :data:`CORE_COLUMNS`' order and dtypes, or raise.

    Parameters
    ----------
    cores : pd.DataFrame
    what : str
        What the table is, for the message (a path, say).

    Raises
    ------
    ArtifactRunError
        If a column is missing, a kind is not S or T, a core bound is not
        finite, or a core ends before it starts.
    """
    missing = [name for name in CORE_COLUMNS if name not in cores.columns]
    if missing:
        raise ArtifactRunError(f"{what} lacks the cores columns {missing}")
    out = cores.loc[:, list(CORE_COLUMNS)].astype(dict(CORE_COLUMNS))
    kinds = set(out["kind"])
    if not kinds <= _KINDS:
        raise ArtifactRunError(
            f"{what}: a core's kind is {sorted(kinds - _KINDS)}, not S or T"
        )
    starts = out["core_start"].to_numpy()
    ends = out["core_end"].to_numpy()
    if not (np.isfinite(starts).all() and np.isfinite(ends).all()):
        raise ArtifactRunError(f"{what}: a core bound is not finite")
    if (ends < starts).any():
        raise ArtifactRunError(f"{what}: a core ends before it starts")
    return out.reset_index(drop=True)


def write_cores(cores: pd.DataFrame, path: Path, *, stamp: Mapping[str, Any]) -> str:
    """Write one epoch's cores atomically; return the file's sha256.

    Parameters
    ----------
    cores : pd.DataFrame
        Columns :data:`CORE_COLUMNS`; no rows for an epoch without artifacts.
    path : Path
        Usually :func:`cores_path`.
    stamp : mapping
        Kept as the table's ``attrs`` (in the parquet metadata), such as the
        run id, epoch id, span and commit.

    Raises
    ------
    ArtifactRunError
        If the cores are malformed (:func:`validate_cores`); nothing is written.
    """
    table = validate_cores(cores, str(path))
    table.attrs = dict(stamp)
    written = write_table(table, path)
    return sha256_file(written)


def read_manifest(run_dir: Path) -> dict[str, Any]:
    """The run folder's manifest, checked to name the folder
    (``runs.io.read_manifest``).

    Raises
    ------
    ArtifactRunError
        If there is no readable manifest, or it is another run's.
    """
    try:
        return runs_io.read_manifest(Path(run_dir), names_folder=True)
    except ValueError as error:
        raise ArtifactRunError(f"artifact run: {error}") from error


def read_epoch_cores(run_dir: Path, epoch_id: str) -> EpochCores:
    """One epoch's cores and span, checked against the run's manifest.

    Parameters
    ----------
    run_dir : Path
        An artifact run folder.
    epoch_id : str
        Such as ``"chimi20200225_02_r1_HPC"``.

    Raises
    ------
    ArtifactRunError
        If the manifest is missing or another run's, does not hold the epoch,
        holds it with a status other than ``ok``, or without a span; if the
        file is missing (an epoch without artifacts has an empty file, never a
        missing one) or its sha256 is not the recorded one; or if the cores
        are malformed.
    """
    manifest = read_manifest(run_dir)
    record = manifest.get("epochs", {}).get(epoch_id)
    name = Path(run_dir).name
    if record is None:
        raise ArtifactRunError(
            f"{epoch_id} is not in the artifact run {name}; run the artifact "
            "runner for it"
        )
    if record.get("status") != "ok":
        why = record.get("error") or record.get("reason") or "no reason recorded"
        raise ArtifactRunError(
            f"{epoch_id} is {record.get('status')!r} in the artifact run {name} "
            f"({str(why)[:200]})"
        )
    path = cores_path(run_dir, epoch_id)
    if not path.exists():
        raise ArtifactRunError(
            f"{epoch_id}: no artifact intervals at {path}; an epoch without "
            "artifacts needs an empty file, not a missing one"
        )
    try:
        runs_io.recorded_output(path, record, "cores")
    except ValueError as error:
        raise ArtifactRunError(f"{epoch_id}: {error}") from error
    span = record.get("span_s")
    if span is None or len(span) != 2:
        raise ArtifactRunError(f"{epoch_id}: the artifact run records no span")
    cores = validate_cores(pd.read_parquet(path), str(path))
    return EpochCores(cores=cores, span=(float(span[0]), float(span[1])))


def epoch_artifact_intervals(
    artifact_dir: Path | None, identifier: str
) -> pd.DataFrame:
    """The epoch's artifact mask, from an artifact run folder.

    The run's cores for the epoch padded by ``PADDING_S``, merged and clipped to
    the run's span, closed: the mask every stage uses.

    Parameters
    ----------
    artifact_dir : Path or None
        The artifact run folder; None when no run is configured.
    identifier : str
        The epoch's id, such as ``"chimi20200225_02_r1_HPC"``.

    Returns
    -------
    pd.DataFrame
        ``start_time`` and ``end_time`` in seconds; no rows when no run is
        configured or the epoch has no artifacts.

    Raises
    ------
    ArtifactRunError
        If the run cannot serve the epoch (:func:`read_epoch_cores`).
    """
    if artifact_dir is None:
        return pd.DataFrame({"start_time": [], "end_time": []}, dtype=np.float64)
    epoch = read_epoch_cores(artifact_dir, identifier)
    return padded_cores(epoch.cores, epoch.span)
