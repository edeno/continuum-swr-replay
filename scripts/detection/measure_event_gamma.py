"""Measure slow- and fast-gamma power during each non-local event, and during running and quiet rest.

Is a non-local event at rest accompanied by a rise in slow or fast gamma
(``analysis.hippocampus.gamma_power``)? Per epoch of the SWR run that
``swr_source`` pins:

1. the referenced raw CA1 LFP of the electrodes rule R-T kept
   (``get_ripple_band_inputs``; the run's manifest lists them);
2. each band's power, averaged over those tetrodes, z-scored over the decode
   span outside excluded time, and read onto the decode grid;
3. each immobile non-local event's mean (the canonical event table's events),
   and the epoch's running and quiet-rest levels (``brain_state_reference``);
4. the same on band power itself (not z), and each class's change against the
   epoch's quiet rest in dB: ``10 log10(mean power over the class's event
   samples / mean power over quiet rest)`` (``gamma_power.power_change_db``).

Writes ``Processed-Data/<--out-name>/{epoch_id}_events.parquet`` per epoch,
``reference.parquet`` (one row per epoch) and ``manifest.json``. Each epoch's
``{epoch_id}_reference.json`` and the manifest record the SHA-256 of the
canonical event table the events were read from. A failing epoch is recorded
and the rest run; a rerun skips the epochs already written from the same event
table unless ``--overwrite``.

Usage::

    uv run python scripts/detection/measure_event_gamma.py --n-workers 8 \\
        [--out-name event_gamma]
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
    regime_masks,
)
from continuum_swr_replay.analysis.hippocampus.gamma_power import (
    FILTER_ORDER,
    GAMMA_BANDS,
    band_power,
    event_means,
    power_change_db,
    samples_in_intervals,
    zscore_on,
)
from continuum_swr_replay.analysis.inventory_comparison import inventory_run
from continuum_swr_replay.data_loaders.lfp import get_ripple_band_inputs
from continuum_swr_replay.decoding.identifiers import format_epoch_identifier
from continuum_swr_replay.intervals import interval_mask
from continuum_swr_replay.paths import (
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.io import sha256_file
from continuum_swr_replay.runs.provenance import described_commit, produced_by
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

KEYS = list(SESSION_COLUMNS)


def out_dir(name: str = "event_gamma") -> Path:
    return processed_data_dir() / name


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

    starts = events["event_start_s"].to_numpy(dtype=np.float64)
    ends = events["event_end_s"].to_numpy(dtype=np.float64)
    is_swr = events["is_swr"].to_numpy(dtype=bool)
    rest = regime_masks(dot)["rest"]
    in_class = {
        "swr": samples_in_intervals(grid_time, starts[is_swr], ends[is_swr]),
        "non_swr": samples_in_intervals(grid_time, starts[~is_swr], ends[~is_swr]),
    }
    columns = {}
    change_db = {}
    for word, band in GAMMA_BANDS.items():
        power = band_power(lfp, sampling_frequency, band)
        on_grid = np.interp(grid_time, lfp_time, zscore_on(power, zscore_keep))
        dot[f"{word}_z"] = on_grid
        columns[f"{word}_z"] = event_means(grid_time, on_grid, starts, ends)
        power_on_grid = np.interp(grid_time, lfp_time, power)
        dot[f"{word}_power"] = power_on_grid
        columns[f"{word}_power"] = event_means(grid_time, power_on_grid, starts, ends)
        for group, mask in in_class.items():
            change_db[f"{word}_db_{group}"] = power_change_db(power_on_grid, mask, rest)
    reference = brain_state_reference(
        dot,
        {
            **{word: f"{word}_z" for word in GAMMA_BANDS},
            **{f"{word}_power": f"{word}_power" for word in GAMMA_BANDS},
        },
    )
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
        **change_db,
    }


def epochs_to_measure(
    events: pd.DataFrame,
) -> list[tuple[str, dict[str, Any], pd.DataFrame]]:
    """Each epoch of the event table, with its SWR-inventory record and its events.

    The epochs are the ones the SWR source served the table's SWRs from, so
    every one has a record: ``ok``, or ``partial`` with a Kay inventory
    (``swr_source.swr_inventory_record``, the source's one rule).

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, n_columns)
        The canonical event table's rows, with the session keys.

    Returns
    -------
    list of (str, dict, pd.DataFrame)
        ``(epoch_id, record, the epoch's events)``, sorted by epoch id.

    Raises
    ------
    swr_source.SwrSourceError
        If the SWR source cannot serve an epoch of the table.
    """
    by_epoch = {
        format_epoch_identifier(f"{animal}{day}_.nwb", epoch, "HPC"): frame
        for (animal, day, epoch), frame in events.groupby(KEYS)
    }
    return [
        (identifier, swr_source.swr_inventory_record(identifier), frame)
        for identifier, frame in sorted(by_epoch.items())
    ]


def _written_from(ref_path: Path) -> str | None:
    """The event-table SHA-256 an epoch's written reference records, if any."""
    if not ref_path.exists():
        return None
    return json.loads(ref_path.read_text()).get("event_table_sha256")


def _run(
    job: tuple[str, dict[str, Any], pd.DataFrame, bool, str, str],
) -> tuple[str, dict[str, Any]]:
    identifier, record, events, overwrite, name, event_table_sha256 = job
    path = out_dir(name) / f"{identifier}_events.parquet"
    ref_path = out_dir(name) / f"{identifier}_reference.json"
    # An epoch written from another event table describes other events.
    if (
        path.exists()
        and _written_from(ref_path) == event_table_sha256
        and not overwrite
    ):
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
    reference["event_table_sha256"] = event_table_sha256
    ref_path.write_text(json.dumps(reference, default=float) + "\n")
    return identifier, {"status": "ok", "wall_clock_s": time.perf_counter() - started}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--epoch-id", action="append", help="only these epochs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--out-name", default="event_gamma")
    args = parser.parse_args(argv)

    event_table = table_path("event_table", replay_content_dir())
    event_table_sha256 = sha256_file(event_table)
    events = pd.read_parquet(
        event_table,
        columns=[*KEYS, "event_id", "event_start_s", "event_end_s", "is_swr"],
    )
    jobs = [
        (
            identifier,
            record,
            frame,
            args.overwrite,
            args.out_name,
            event_table_sha256,
        )
        for identifier, record, frame in epochs_to_measure(events)
        if not args.epoch_id or identifier in args.epoch_id
    ]
    out_dir(args.out_name).mkdir(parents=True, exist_ok=True)
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
        json.loads((out_dir(args.out_name) / f"{i}_reference.json").read_text())
        for i, r in sorted(epochs.items())
        if r["status"] == "ok"
    ]
    pd.DataFrame(references).to_parquet(
        out_dir(args.out_name) / "reference.parquet", index=False
    )
    (out_dir(args.out_name) / "manifest.json").write_text(
        json.dumps(
            {
                "produced_by": produced_by(__file__),
                "commit": described_commit(),
                "swr_run": swr_source.SWR_INVENTORY_RUN_ID,
                "event_table": {"path": str(event_table), "sha256": event_table_sha256},
                "bands_hz": GAMMA_BANDS,
                "filter": f"Butterworth order {FILTER_ORDER}, zero phase (sosfiltfilt)",
                "power": "squared Hilbert envelope, mean over the kept CA1 tetrodes",
                "zscore": "over the decode span outside excluded time",
                "change_db": "10 log10 of the mean band power over each class's event "
                "samples over the mean over quiet rest (below 4 cm/s, in no non-local, "
                "SWR or MUA event, outside excluded time), on the decode grid",
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
