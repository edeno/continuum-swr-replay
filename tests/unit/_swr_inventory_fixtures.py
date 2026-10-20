"""Synthetic SWR inventories, written as the SWR runner writes them.

``write_swr_inventory`` makes a run folder under a ``tmp_path``: a
``manifest.json`` with one record per epoch and, for each served epoch, its
``{epoch_id}_kay_events.parquet`` (indexed by ``event_number`` 1..n) and
``{epoch_id}_swr_observed.parquet``, with the sha256 of each in the record, as
``run_swr_detection.py`` writes them. ``write_artifact_run`` writes an artifact
run folder as ``run_artifact_detection.py`` writes it (one cores file per epoch
and a manifest), which ``swr_inventory.artifact_intervals`` reads, and
``use_swr_source`` points the source's configured constants at both. Offline:
nothing here touches Spyglass or ``Processed-Data``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.events.artifact_intervals import (
    cores_path,
    make_cores,
    write_cores,
    write_manifest,
)
from continuum_swr_replay.runs.io import sha256_file

RUN_ID = "testrun0001"
ARTIFACT_RUN_ID = "testartifacts01"


def kay_events(intervals: Sequence[tuple[float, float]]) -> pd.DataFrame:
    """A Kay event table for `intervals`, as ``Kay_ripple_detector`` returns it."""
    starts = np.array([start for start, _ in intervals], dtype=float)
    ends = np.array([end for _, end in intervals], dtype=float)
    return pd.DataFrame(
        {
            "start_time": starts,
            "end_time": ends,
            "duration": ends - starts,
            "max_zscore": np.full(starts.size, 3.0),
            "peak_time": (starts + ends) / 2,
        },
        index=pd.Index(np.arange(1, starts.size + 1), name="event_number"),
    )


def write_swr_inventory(
    root: Path,
    events: Mapping[str, Sequence[tuple[float, float]]],
    *,
    observed: Mapping[str, Sequence[tuple[float, float]]] | None = None,
    records: Mapping[str, dict] | None = None,
    run_id: str = RUN_ID,
    artifact_run_id: str | None = None,
) -> Path:
    """Write a run folder; return its path.

    Parameters
    ----------
    root : Path
        The folder that holds run folders (``SWR_INVENTORY_ROOT``'s stand-in).
    events : mapping
        Epoch id -> Kay event intervals ``(start, end)`` in seconds; each such
        epoch is ``ok``.
    observed : mapping, optional
        Epoch id -> stored-band stretches; ``[(0, 1e6)]`` by default.
    records : mapping, optional
        Epoch id -> a manifest record written as given (a failed or skipped
        epoch, say), with no files.
    run_id : str
    artifact_run_id : str, optional
        The artifact run the inventory was masked with, written as the SWR
        runner writes it (``settings.artifacts.run_id``); no settings when
        None, as in a run made before the runner masked artifacts.
    """
    directory = root / run_id
    directory.mkdir(parents=True, exist_ok=True)
    manifest_epochs: dict[str, dict] = {}
    for epoch_id, intervals in events.items():
        events_path = directory / f"{epoch_id}_kay_events.parquet"
        kay_events(intervals).to_parquet(events_path)
        stretches = (observed or {}).get(epoch_id, [(0.0, 1e6)])
        observed_path = directory / f"{epoch_id}_swr_observed.parquet"
        pd.DataFrame(
            {
                "scope": ["stored_band"] * len(stretches) + ["zugaro"],
                "start_s": [float(s) for s, _ in stretches] + [0.0],
                "end_s": [float(e) for _, e in stretches] + [1e6],
            }
        ).to_parquet(observed_path)
        manifest_epochs[epoch_id] = {
            "status": "ok",
            "detectors": {"kay": "ok", "karlsson": "ok", "yu": "ok", "zugaro": "ok"},
            # the detecting electrodes and the band's rate, as the runner records
            "electrode_ids": [0, 4, 8],
            "sampling": {"nominal": 1000.0, "measured": 1000.0},
            "position_span_s": [0.0, 1e6],
            "outputs": {
                "kay": {"path": str(events_path), "sha256": sha256_file(events_path)},
                "swr_observed": {
                    "path": str(observed_path),
                    "sha256": sha256_file(observed_path),
                },
            },
        }
    for epoch_id, record in (records or {}).items():
        manifest_epochs[epoch_id] = record
    manifest: dict = {"run_id": run_id, "epochs": manifest_epochs}
    if artifact_run_id is not None:
        manifest["settings"] = {"artifacts": {"run_id": artifact_run_id}}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return directory


def artifact_cores(cores: Sequence[tuple[str, float, float]]) -> pd.DataFrame:
    """A cores table from ``(signatures, core_start, core_end)``, kind S or T,
    with no evidence."""
    return make_cores(
        {
            "kind": ["S" if sig == "S" else "T" for sig, _, _ in cores],
            "signatures": [sig for sig, _, _ in cores],
            "core_start": [float(start) for _, start, _ in cores],
            "core_end": [float(end) for _, _, end in cores],
        }
    )


def write_artifact_run(
    root: Path,
    cores: Mapping[str, Sequence[tuple[str, float, float]]],
    *,
    span: tuple[float, float] = (0.0, 1e6),
    run_id: str = ARTIFACT_RUN_ID,
) -> Path:
    """Write an artifact run folder, one cores file per epoch; return its path.

    Parameters
    ----------
    root : Path
        The folder that holds run folders (``ARTIFACT_INTERVALS_DIR``'s
        stand-in).
    cores : mapping
        Epoch id -> its unpadded cores ``(signatures, core_start, core_end)``,
        such as ``("spk", 104.11, 104.11)``; each epoch is ``ok``.
    span : tuple of float
        Every epoch's span, which the mask's intervals are clipped to.
    run_id : str
    """
    directory = root / run_id
    directory.mkdir(parents=True, exist_ok=True)
    records = {}
    for epoch_id, epoch_cores in cores.items():
        path = cores_path(directory, epoch_id)
        sha = write_cores(artifact_cores(epoch_cores), path, stamp={})
        records[epoch_id] = {
            "status": "ok",
            "span_s": [float(span[0]), float(span[1])],
            "outputs": {"cores": {"path": str(path), "sha256": sha}},
        }
    write_manifest(directory, {"run_id": run_id, "epochs": records})
    return directory


def use_swr_source(monkeypatch, inventory: Path, artifacts: Path | None = None) -> None:
    """Point the SWR source at a written inventory run folder (and artifact run
    folder, if given), as its configured constants would."""
    from continuum_swr_replay.analysis.events import swr_inventory

    monkeypatch.setattr(swr_inventory, "SWR_INVENTORY_ROOT", inventory.parent)
    monkeypatch.setattr(swr_inventory, "SWR_INVENTORY_RUN_ID", inventory.name)
    if artifacts is not None:
        monkeypatch.setattr(swr_inventory, "ARTIFACT_INTERVALS_DIR", artifacts.parent)
        monkeypatch.setattr(swr_inventory, "ARTIFACT_INTERVALS_RUN_ID", artifacts.name)
