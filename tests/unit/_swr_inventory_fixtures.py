"""Synthetic SWR inventories, written as the SWR runner writes them.

``write_swr_inventory`` makes a run folder under a ``tmp_path``: a
``manifest.json`` with one record per epoch and, for each served epoch, its
``{epoch_id}_kay_events.parquet`` (indexed by ``event_number`` 1..n) and
``{epoch_id}_swr_observed.parquet``, with the sha256 of each in the record, as
``run_swr_detection.py`` writes them. Offline: nothing here touches Spyglass or
``Processed-Data``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.table_io import sha256_file

RUN_ID = "testrun0001"


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
    (directory / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "epochs": manifest_epochs}, indent=1)
    )
    return directory
