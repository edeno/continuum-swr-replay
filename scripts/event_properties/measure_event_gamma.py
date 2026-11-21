"""Measure slow- and fast-gamma power during each non-local event, and during running and quiet rest.

Is a non-local event at rest accompanied by a rise in slow or fast gamma
(``analysis.event_properties.gamma_power``)? Per epoch of the SWR run that
``swr_source`` pins:

1. the referenced raw CA1 LFP of the electrodes rule R-T kept
   (``get_ripple_band_inputs``; the run's manifest lists them);
2. each band's power, averaged over those tetrodes, z-scored over the decode
   span outside excluded time, and read onto the decode grid;
3. each immobile non-local event's mean (the canonical event table's events
   that start inside a well visit, ``cohort_selection.select_visit_context``),
   and the epoch's running and quiet-rest levels (``brain_state_reference``;
   quiet rest is the rest inside a well visit, the epoch's port experiences
   read from the session cache: ``event_tables.prepare_data.well_visit_times``);
4. the level of quiet-rest and of running windows as long as each class's
   events, ``--n-windows`` drawn per event (``gamma_power.matched_level``): an
   event's mean over its bins is compared with these, not with the median of
   single bins, which a mean over a window of skewed power exceeds when nothing
   differs;
5. the same on band power itself (not z), and each class's change against the
   epoch's quiet rest in dB: ``10 log10(mean power over the class's event
   samples / mean power over quiet rest)`` (``gamma_power.power_change_db``).

Writes ``Processed-Data/<--out-name>/{epoch_id}_events.parquet`` per epoch,
``{epoch_id}_gamma_z.parquet`` (each band's z-scored power at every bin of the
decode grid, so that gamma around the events can be drawn without the LFP),
``reference.parquet`` (one row per epoch) and ``manifest.json``. Each epoch's
``{epoch_id}_reference.json`` and the manifest record the SHA-256 of the
canonical event table the events were read from and the subset they are
(``cohort_selection.ANALYZED_SUBSET``). A failing epoch is recorded and the rest run; a rerun skips
the epochs already written from the same event table and subset unless
``--overwrite``.

Usage::

    uv run python scripts/event_properties/measure_event_gamma.py --n-workers 8 \\
        [--out-name event_gamma]
"""

from __future__ import annotations

import argparse
import json
import time
import zlib
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.detector_comparison import inventory_run
from continuum_swr_replay.analysis.event_definition import analyzed_time, swr_source
from continuum_swr_replay.analysis.event_definition.analyzed_time import read_swr_table
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    ANALYZED_SUBSET,
    count_visit_context,
    select_visit_context,
)
from continuum_swr_replay.analysis.event_definition.well_visits import in_well_visit
from continuum_swr_replay.analysis.event_properties.gamma_power import (
    FILTER_ORDER,
    GAMMA_BANDS,
)
from continuum_swr_replay.analysis.event_properties.gamma_power import (
    epoch_gamma as epoch_gamma_measures,
)
from continuum_swr_replay.analysis.event_tables.prepare_data import well_visit_times
from continuum_swr_replay.data_loaders.lfp import get_ripple_band_inputs
from continuum_swr_replay.paths import (
    EVENT_GAMMA_DIR,
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.io import sha256_file, write_json
from continuum_swr_replay.runs.provenance import described_commit, produced_by
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

KEYS = list(SESSION_COLUMNS)

SEED = 20261120
"""Seeds the matched windows, with the epoch's id."""


def out_dir(name: str = EVENT_GAMMA_DIR) -> Path:
    return processed_data_dir() / name


def epoch_gamma(
    identifier: str, record: dict[str, Any], events: pd.DataFrame, n_windows: int
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """One epoch's events with their mean gamma, its reference levels, and its
    gamma at every bin of the decode grid (``gamma_power.epoch_gamma`` on the
    epoch's LFP and decode frame)."""
    inputs = get_ripple_band_inputs(record["nwb_file_name"], record["epoch_name"])
    keep = np.isin(inputs["electrode_ids"], np.asarray(record["electrode_ids"]))
    if keep.sum() != len(record["electrode_ids"]):
        raise ValueError("the band lacks some of the SWR run's kept electrodes")
    lfp_time = np.asarray(inputs["time"], dtype=np.float64)
    lfp = inputs["raw_referenced"][:, keep]
    sampling_frequency = float(record["sampling"]["measured"])

    dot = analyzed_time.with_swr_source(
        pd.read_pickle(inventory_run.decode_path(identifier))
    )
    grid_time, _ = inventory_run.speed_trace(dot)
    in_visit = in_well_visit(grid_time, *well_visit_times(identifier))
    columns, reference, on_grid = epoch_gamma_measures(
        lfp,
        lfp_time,
        sampling_frequency,
        dot,
        grid_time,
        analyzed_time.excluded_intervals(identifier),
        events,
        in_visit,
        n_windows,
        np.random.default_rng([SEED, zlib.crc32(identifier.encode())]),
    )
    table = events[
        [*KEYS, "event_id", "event_start_s", "event_end_s", "is_swr"]
    ].assign(**columns)
    per_bin = pd.DataFrame(
        {"time": grid_time}
        | {name: values.astype(np.float32) for name, values in on_grid.items()}
    )
    return (
        table.reset_index(drop=True),
        {
            "epoch_id": identifier,
            "animal": str(events["animal"].iloc[0]),
            "session_date": int(events["session_date"].iloc[0]),
            "epoch": str(events["epoch"].iloc[0]),
            "n_kept_tetrodes": int(keep.sum()),
            "sampling_frequency": sampling_frequency,
            **reference,
        },
        per_bin,
    )


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
        swr_source.swr_epoch_id_of_session(animal, day, epoch): frame
        for (animal, day, epoch), frame in events.groupby(KEYS)
    }
    return [
        (identifier, swr_source.swr_inventory_record(identifier), frame)
        for identifier, frame in sorted(by_epoch.items())
    ]


def _written_from(ref_path: Path) -> tuple[str | None, str | None]:
    """The event-table SHA-256 and the subset an epoch's written reference records."""
    if not ref_path.exists():
        return None, None
    reference = json.loads(ref_path.read_text())
    return reference.get("event_table_sha256"), reference.get("subset")


def _run(
    job: tuple[str, dict[str, Any], pd.DataFrame, bool, str, str, int],
) -> tuple[str, dict[str, Any]]:
    identifier, record, events, overwrite, name, event_table_sha256, n_windows = job
    path = out_dir(name) / f"{identifier}_events.parquet"
    ref_path = out_dir(name) / f"{identifier}_reference.json"
    per_bin_path = out_dir(name) / f"{identifier}_gamma_z.parquet"
    # An epoch written from another event table or subset describes other events.
    if (
        path.exists()
        and per_bin_path.exists()
        and _written_from(ref_path) == (event_table_sha256, ANALYZED_SUBSET)
        and not overwrite
    ):
        return identifier, {"status": "ok", "reused": True}
    started = time.perf_counter()
    try:
        table, reference, per_bin = epoch_gamma(identifier, record, events, n_windows)
    except (ValueError, KeyError, OSError) as error:
        return identifier, {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
    table.to_parquet(path, index=False)
    per_bin.to_parquet(per_bin_path, index=False)
    reference["event_table_sha256"] = event_table_sha256
    reference["subset"] = ANALYZED_SUBSET
    write_json(ref_path, reference, default=float, final_newline=True)
    return identifier, {"status": "ok", "wall_clock_s": time.perf_counter() - started}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--epoch-id", action="append", help="only these epochs")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--out-name", default=EVENT_GAMMA_DIR)
    parser.add_argument(
        "--n-windows",
        type=int,
        default=100,
        help="quiet-rest and running windows drawn per event",
    )
    args = parser.parse_args(argv)

    event_table = table_path("event_table", replay_content_dir())
    event_table_sha256 = sha256_file(event_table)
    table = read_swr_table(
        event_table,
        columns=[
            *KEYS,
            "event_id",
            "event_start_s",
            "event_end_s",
            "is_swr",
            "visit_context",
        ],
    )
    events = select_visit_context(table)
    jobs = [
        (
            identifier,
            record,
            frame,
            args.overwrite,
            args.out_name,
            event_table_sha256,
            args.n_windows,
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
    write_json(
        out_dir(args.out_name) / "manifest.json",
        {
            "produced_by": produced_by(__file__),
            "commit": described_commit(),
            "swr_run": swr_source.SWR_INVENTORY_RUN_ID,
            "event_table": {"path": str(event_table), "sha256": event_table_sha256},
            "subset": ANALYZED_SUBSET,
            "visit_context_counts": {
                label: int(n) for label, n in count_visit_context(table)["n"].items()
            },
            "bands_hz": GAMMA_BANDS,
            "filter": f"Butterworth order {FILTER_ORDER}, zero phase (sosfiltfilt)",
            "power": "squared Hilbert envelope, mean over the kept CA1 tetrodes",
            "zscore": "over the decode span outside excluded time",
            "change_db": "10 log10 of the mean band power over each class's event "
            "samples over the mean over quiet rest (below 4 cm/s, in no non-local, "
            "SWR or MUA event, inside a well visit, outside excluded time), on the "
            "decode grid",
            "matched_windows": "for each class of events, the median over its events "
            "of the mean z over a quiet-rest (running) window as long as the event, "
            "averaged over the draws; windows drawn uniformly over every window of "
            "the regime that long",
            "n_windows": args.n_windows,
            "seed": SEED,
            "per_bin": "{epoch_id}_gamma_z.parquet: each band's z-scored power at "
            "every bin of the decode grid (float32)",
            "epochs": epochs,
        },
        indent=1,
        final_newline=True,
    )
    n_failed = sum(r["status"] != "ok" for r in epochs.values())
    print(f"done: {len(epochs) - n_failed} ok, {n_failed} failed")


if __name__ == "__main__":
    main()
