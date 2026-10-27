"""Measure slow- and fast-gamma power during each non-local event, and during running and quiet rest.

Is a non-local event at rest accompanied by a rise in slow or fast gamma
(``analysis.hippocampus.gamma_power``)? Per epoch of the SWR run that
``swr_source`` pins:

1. the referenced raw CA1 LFP of the electrodes rule R-T kept
   (``get_ripple_band_inputs``; the run's manifest lists them);
2. each band's power, averaged over those tetrodes, z-scored over the decode
   span outside excluded time, and read onto the decode grid;
3. each immobile non-local event's mean (the canonical event table's events),
   and the epoch's running and quiet-rest levels (``brain_state_reference``).

Writes ``Processed-Data/event_gamma/{epoch_id}_events.parquet`` per epoch,
``reference.parquet`` (one row per epoch) and ``manifest.json``. A failing
epoch is recorded and the rest run; a rerun skips the epochs already written
unless ``--overwrite``.

Usage::

    uv run python scripts/detection/measure_event_gamma.py --n-workers 8
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.events import analysed_time, swr_source
from continuum_swr_replay.analysis.hippocampus.brain_state_reference import (
    brain_state_reference,
)
from continuum_swr_replay.analysis.hippocampus.gamma_power import (
    FILTER_ORDER,
    GAMMA_BANDS,
    band_power,
    event_means,
    zscore_on,
)
from continuum_swr_replay.analysis.inventory_comparison import inventory_run
from continuum_swr_replay.data_loaders.lfp import get_ripple_band_inputs
from continuum_swr_replay.decoding.identifiers import format_epoch_identifier
from continuum_swr_replay.intervals import interval_mask
from continuum_swr_replay.paths import (
    processed_data_dir,
    replay_content_dir,
    swr_inventory_root,
    table_path,
)
from continuum_swr_replay.runs.provenance import described_commit, produced_by
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

KEYS = list(SESSION_COLUMNS)


def out_dir() -> Path:
    return processed_data_dir() / "event_gamma"


def epoch_gamma(
    identifier: str, record: dict[str, Any], events: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """One epoch's events with their mean gamma, and its reference levels."""
    inputs = get_ripple_band_inputs(record["nwb_file_name"], record["epoch_name"])
    keep = np.isin(inputs["electrode_ids"], np.asarray(record["electrode_ids"]))
    if keep.sum() != len(record["electrode_ids"]):
        raise ValueError("the band lacks some of the SWR run's kept electrodes")
    lfp_time = np.asarray(inputs["time"], dtype=np.float64)
    lfp = inputs["raw_referenced"][:, keep]
    sampling_frequency = float(record["sampling"]["measured"])

    dot = analysed_time.with_swr_source(
        pd.read_pickle(inventory_run.decode_path(identifier))
    )
    grid_time, _ = inventory_run.speed_trace(dot)
    excluded = analysed_time.excluded_intervals(identifier)
    in_span = (lfp_time >= grid_time[0]) & (lfp_time <= grid_time[-1])
    zscore_keep = in_span & ~interval_mask(lfp_time, excluded)

    columns = {}
    for word, band in GAMMA_BANDS.items():
        power = zscore_on(band_power(lfp, sampling_frequency, band), zscore_keep)
        on_grid = np.interp(grid_time, lfp_time, power)
        dot[f"{word}_z"] = on_grid
        columns[f"{word}_z"] = event_means(
            grid_time,
            on_grid,
            events["event_start_s"].to_numpy(dtype=np.float64),
            events["event_end_s"].to_numpy(dtype=np.float64),
        )
    reference = brain_state_reference(dot, {word: f"{word}_z" for word in GAMMA_BANDS})
    table = events[
        [*KEYS, "event_id", "event_start_s", "event_end_s", "is_swr"]
    ].assign(**columns)
    return table.reset_index(drop=True), {
        "epoch_id": identifier,
        "animal": str(events["animal"].iloc[0]),
        "session_date": int(events["session_date"].iloc[0]),
        "epoch": str(events["epoch"].iloc[0]),
        "n_kept_tetrodes": int(keep.sum()),
        "sampling_frequency": sampling_frequency,
        **reference,
    }


def _run(
    job: tuple[str, dict[str, Any], pd.DataFrame, bool],
) -> tuple[str, dict[str, Any]]:
    identifier, record, events, overwrite = job
    path = out_dir() / f"{identifier}_events.parquet"
    ref_path = out_dir() / f"{identifier}_reference.json"
    if path.exists() and ref_path.exists() and not overwrite:
        return identifier, {"status": "ok", "reused": True}
    started = time.perf_counter()
    try:
        table, reference = epoch_gamma(identifier, record, events)
    except (ValueError, KeyError, OSError) as error:
        return identifier, {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
    table.to_parquet(path, index=False)
    ref_path.write_text(json.dumps(reference, default=float) + "\n")
    return identifier, {"status": "ok", "wall_clock_s": time.perf_counter() - started}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--epoch-id", action="append", help="only these epochs")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    swr_manifest = json.loads(
        (
            swr_inventory_root() / swr_source.SWR_INVENTORY_RUN_ID / "manifest.json"
        ).read_text()
    )
    events = pd.read_parquet(
        table_path("event_table", replay_content_dir()),
        columns=[*KEYS, "event_id", "event_start_s", "event_end_s", "is_swr"],
    )
    by_epoch = {
        format_epoch_identifier(f"{animal}{day}_.nwb", epoch, "HPC"): frame
        for (animal, day, epoch), frame in events.groupby(KEYS)
    }
    jobs = [
        (identifier, record, by_epoch[identifier], args.overwrite)
        for identifier, record in sorted(swr_manifest["epochs"].items())
        if record.get("status") == "ok"
        and identifier in by_epoch
        and (not args.epoch_id or identifier in args.epoch_id)
    ]
    out_dir().mkdir(parents=True, exist_ok=True)
    print(f"{len(jobs)} epochs")
    epochs: dict[str, Any] = {}
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        for identifier, result in pool.map(_run, jobs):
            epochs[identifier] = result
            print(
                f"  {identifier}: {result['status']} {result.get('error', '')}",
                flush=True,
            )
    references = [
        json.loads((out_dir() / f"{i}_reference.json").read_text())
        for i, r in sorted(epochs.items())
        if r["status"] == "ok"
    ]
    pd.DataFrame(references).to_parquet(out_dir() / "reference.parquet", index=False)
    (out_dir() / "manifest.json").write_text(
        json.dumps(
            {
                "produced_by": produced_by(__file__),
                "commit": described_commit(),
                "swr_run": swr_source.SWR_INVENTORY_RUN_ID,
                "bands_hz": GAMMA_BANDS,
                "filter": f"Butterworth order {FILTER_ORDER}, zero phase (sosfiltfilt)",
                "power": "squared Hilbert envelope, mean over the kept CA1 tetrodes",
                "zscore": "over the decode span outside excluded time",
                "epochs": epochs,
            },
            indent=1,
        )
        + "\n"
    )
    n_failed = sum(r["status"] != "ok" for r in epochs.values())
    print(f"done: {len(epochs) - n_failed} ok, {n_failed} failed")


if __name__ == "__main__":
    main()
