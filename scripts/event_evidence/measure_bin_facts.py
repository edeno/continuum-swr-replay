#!/usr/bin/env python
"""Each decode bin's spikes over every tetrode and whether it is inside a well visit.

For every epoch of the non-local restriction run (``score_nonlocal_restriction.py``,
its ``ok`` records): load the epoch's spikes (``state_evidence.terms.load_marks_and_training_mask``,
which reads the lab's database through ``load_data``, read only), check that the
loader's time axis is the decode frame's, and read the epoch's well visits from
the session cache of the table builders only (``event_tables.prepare_data.well_visit_times``;
an epoch without a current cache entry fails instead of being fetched). Then
write the per-bin facts (``event_evidence.bin_facts.bin_facts``: spikes binned
as the decoder bins them, and ``in_well_visit``). The restriction run's record
counts no spikes (the fold fit's likelihood is the library's, which returns no
count), so nothing here ties the spikes to that run; the window run
(``sum_nonlocal_restriction_windows.py``) ties them to the held-out tetrode
run's, window by window, and their time to the restriction run's per-bin table.
No event table is read, and every bin is kept: the facts are not made
over a subset, so the run records none. CPU only.

Writes, under ``processed_data_dir() / EVENT_BIN_FACTS_DIR``:

- ``{epoch_id}_bin_facts.parquet``: one row per decode bin, ``time`` (s),
  ``n_spikes`` (int32) and ``in_well_visit`` (bool);
- ``{epoch_id}_bin_facts.json``: the epoch's record (bins, spikes, well visits,
  bins and spikes inside them, the digests of the decode frame and the
  restriction run's record; provenance);

and ``manifest.json`` gathered from the records. The inputs record the
session-cache entries the visits were read from
(``prepare_data.session_cache_record``, over every epoch of the restriction run),
so the events summary and Figure 2 can tell a run made from other visits.

Usage::

    JAX_PLATFORMS=cpu uv run --frozen python scripts/event_evidence/measure_bin_facts.py \\
        --epochs wilbur20210329_02_r1_HPC
    JAX_PLATFORMS=cpu uv run --frozen python scripts/event_evidence/measure_bin_facts.py \\
        --workers 6
"""

from __future__ import annotations

import argparse
import multiprocessing
import sys
import time as clock
from pathlib import Path

import numpy as np

from continuum_swr_replay.analysis.event_evidence.bin_facts import bin_facts
from continuum_swr_replay.analysis.event_tables.prepare_data import (
    session_cache_record,
    well_visit_times,
)
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.parameters import ROOT_DIR
from continuum_swr_replay.paths import (
    BIN_FACTS_NAME,
    EVENT_BIN_FACTS_DIR,
    EVENT_NONLOCAL_RESTRICTION_DIR,
    NONLOCAL_RESTRICTION_NAME,
    processed_data_dir,
)
from continuum_swr_replay.runs.epoch_records import gather_manifest
from continuum_swr_replay.runs.io import (
    json_ready,
    ok_epochs,
    output_path,
    read_manifest,
    sha256_file,
)
from continuum_swr_replay.runs.provenance import produced_by, provenance_record
from continuum_swr_replay.tables.io import write_table

# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.event_evidence._per_event_run import (  # noqa: E402
    add_run_arguments,
    run,
    select_epochs,
)

PRODUCED_BY = produced_by(__file__)


def decode_time(epoch_id: str, data_dir: Path) -> np.ndarray:
    """The epoch's decode frame's time axis, s, shape (n_time,)."""
    from continuum_swr_replay.analysis.event_definition.analyzed_time import (
        read_dot_with_swr_source,
    )

    frame = read_dot_with_swr_source(decode_paths(epoch_id, data_dir).data_over_time)
    return frame.index.to_numpy(dtype=float)


def run_epoch(
    epoch_id: str, data_dir: Path, restriction_dir: Path, output: Path
) -> dict:
    """Measure one epoch, write its table and return its record.

    Raises
    ------
    ValueError
        If the loader's time axis is not the decode frame's.
    """
    from continuum_swr_replay.analysis.event_evidence.state_evidence.terms import (
        load_marks_and_training_mask,
    )

    started = clock.time()
    time = decode_time(epoch_id, data_dir)
    loaded = load_marks_and_training_mask(epoch_id)
    if not np.array_equal(np.asarray(loaded["time"], dtype=float), time):
        raise ValueError("the loader's time axis is not the decode frame's")
    visits = well_visit_times(epoch_id)
    table = bin_facts(time, loaded["spike_times"], visits)
    n_spikes = int(table["n_spikes"].sum())
    restriction_record = output_path(
        epoch_id, restriction_dir, NONLOCAL_RESTRICTION_NAME
    ).with_suffix(".json")
    write_table(table, output_path(epoch_id, output, BIN_FACTS_NAME))
    inside = table["in_well_visit"].to_numpy()
    return json_ready(
        {
            "status": "ok",
            "n_bins": len(table),
            "n_spikes": n_spikes,
            "n_well_visits": len(visits[0]),
            "n_bins_in_well_visit": int(inside.sum()),
            "n_spikes_in_well_visit": int(table.loc[inside, "n_spikes"].sum()),
            "epoch_inputs": {
                name: {"sha256": sha256_file(path), "path": str(path)}
                for name, path in {
                    "data_over_time": decode_paths(epoch_id, data_dir).data_over_time,
                    "restriction_record": restriction_record,
                }.items()
            },
            "seconds": round(clock.time() - started, 1),
        }
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_run_arguments(parser)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args(argv)

    data_dir = args.data_dir or processed_data_dir()
    output = args.output_dir or data_dir / EVENT_BIN_FACTS_DIR
    restriction_dir = data_dir / EVENT_NONLOCAL_RESTRICTION_DIR
    every_epoch = sorted(ok_epochs(read_manifest(restriction_dir)))
    header = {
        "produced_by": PRODUCED_BY,
        "provenance": provenance_record(ROOT_DIR),
        "settings": {"binning": "np.digitize(t, time[1:-1]), time[0] <= t <= time[-1]"},
        "inputs": {"session_cache": session_cache_record(every_epoch)},
    }
    if args.gather:
        gather_manifest(output, BIN_FACTS_NAME, header)
        return
    run(
        epochs=select_epochs(args.epochs or every_epoch, args.shards, args.n_shards),
        task=run_epoch,
        task_args=(data_dir, restriction_dir, output),
        output=output,
        name=BIN_FACTS_NAME,
        header=header,
        overwrite=args.overwrite,
        max_workers=args.workers,
        pool_options={"mp_context": multiprocessing.get_context("spawn")},
    )


if __name__ == "__main__":
    main()
