#!/usr/bin/env python
"""The rank of the tetrodes' summed rate at each non-local event's decoded place.

For every analyzed epoch with events in the canonical event table
(``replay_content/event_table.parquet``, read through the SWR source): read the
encoding model the decode saved (``{epoch_id}_detector.pkl``,
``decoding.encoding_model.load_clusterless_encoding_model``), take its summed
ground intensity (the expected spikes per time bin over all tetrodes) at the
track's real places (``content_place_rate.real_places``: interior bins off the
junction placeholders), and give each event the rank of the rate at the places
nearest its replay start and end positions
(``content_place_rate.epoch_place_rates``). Writes, under
``processed_data_dir() / EVENT_CONTENT_PLACE_RATE_DIR``:

- ``{epoch_id}_content_place_rate.parquet``: one row per event of the table in
  the epoch, every visit context (the readers select the events,
  ``cohort_selection.select_events_in``): ``event_id``, ``epoch_id``,
  ``animal``, ``group`` (``"SWR"`` or ``"no SWR"``), ``place_rate_rank``,
  ``place_rate_ratio``, ``start_place_rate_rank`` and ``end_place_rate_rank``;
- ``{epoch_id}_content_place_rate.json``: the epoch's record (events, real
  places, interior junction bins left out, events without a rank, the ground
  intensity's smallest, median and largest value over the real places, the
  detector file's digest; provenance);

and ``manifest.json`` gathered from the records. Nothing is read from the lab's
database. CPU only.

Usage::

    JAX_PLATFORMS=cpu uv run --frozen python \\
        scripts/event_properties/measure_content_place_rate.py \\
        --epochs wilbur20210402_02_r1_HPC
    JAX_PLATFORMS=cpu uv run --frozen python \\
        scripts/event_properties/measure_content_place_rate.py --workers 8
"""

from __future__ import annotations

import argparse
import sys
import time as clock
from pathlib import Path

import numpy as np

from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    read_swr_table,
)
from continuum_swr_replay.analysis.event_definition.swr_source import (
    event_epoch_ids,
    event_table_epochs,
)
from continuum_swr_replay.analysis.event_evidence.state_evidence.windows import (
    SWR_LABELS,
)
from continuum_swr_replay.analysis.event_properties import content_place_rate as cpr
from continuum_swr_replay.data_loaders.epoch_register import analyzed_epoch_ids
from continuum_swr_replay.decoding.encoding_model import (
    load_clusterless_encoding_model,
)
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.parameters import ROOT_DIR
from continuum_swr_replay.paths import (
    CONTENT_PLACE_RATE_NAME,
    EVENT_CONTENT_PLACE_RATE_DIR,
    processed_data_dir,
)
from continuum_swr_replay.runs.epoch_records import gather_manifest
from continuum_swr_replay.runs.io import json_ready, output_path
from continuum_swr_replay.runs.provenance import produced_by, provenance_record
from continuum_swr_replay.tables.io import write_table

# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.event_evidence._per_event_run import (  # noqa: E402
    add_cpu_arguments,
    add_run_arguments,
    cpu_pool,
    event_table_path,
    file_input,
    run,
    select_epochs,
)

PRODUCED_BY = produced_by(__file__)
EVENT_COLUMNS = [
    "animal",
    "session_date",
    "epoch",
    "event_id",
    "is_swr",
    *cpr.POSITION_COLUMNS,
]
"""The event table's columns the run reads."""


def run_epoch(epoch_id: str, event_table: Path, data_dir: Path, output: Path) -> dict:
    """Rank one epoch's events, write its table and return its record."""
    started = clock.time()
    events = read_swr_table(event_table, columns=EVENT_COLUMNS)
    events = events[event_epoch_ids(events) == epoch_id].reset_index(drop=True)
    detector = decode_paths(epoch_id, data_dir).detector
    model = load_clusterless_encoding_model(detector)
    environment = model["environment"]
    bins = environment.place_bin_centers_nodes_df_
    interior = np.asarray(environment.is_track_interior_, dtype=bool).ravel()
    centers, rates = cpr.real_places(
        np.asarray(model["summed_ground_process_intensity"]),
        bins["linear_position"].to_numpy(),
        interior,
        bins["edge_id"].to_numpy(),
    )
    ranked = cpr.epoch_place_rates(events, centers, rates)
    table = ranked.assign(
        event_id=events["event_id"],
        epoch_id=epoch_id,
        animal=events["animal"],
        group=np.where(
            events["is_swr"].to_numpy(dtype=bool), SWR_LABELS[True], SWR_LABELS[False]
        ),
    )[
        [
            "event_id",
            "epoch_id",
            "animal",
            "group",
            "place_rate_rank",
            "place_rate_ratio",
            *cpr.RANK_COLUMNS,
        ]
    ]
    write_table(table, output_path(epoch_id, output, CONTENT_PLACE_RATE_NAME))
    return json_ready(
        {
            "status": "ok",
            "n_events": len(table),
            "n_places": len(centers),
            "n_interior_junction_bins": int(
                (bins["edge_id"].to_numpy()[interior] < 0).sum()
            ),
            "n_events_without_rank": int(table["place_rate_rank"].isna().sum()),
            "ground_intensity_min": float(rates.min()),
            "ground_intensity_median": float(np.median(rates)),
            "ground_intensity_max": float(rates.max()),
            "epoch_inputs": {"detector": file_input(detector)},
            "seconds": round(clock.time() - started, 1),
        }
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_run_arguments(parser)
    add_cpu_arguments(parser, workers=8)
    args = parser.parse_args(argv)

    data_dir = args.data_dir or processed_data_dir()
    output = args.output_dir or data_dir / EVENT_CONTENT_PLACE_RATE_DIR
    event_table = event_table_path(data_dir)
    every_epoch = event_table_epochs(
        read_swr_table(event_table, columns=EVENT_COLUMNS[:3]), analyzed_epoch_ids()
    )
    header = {
        "produced_by": PRODUCED_BY,
        "provenance": provenance_record(ROOT_DIR),
        "settings": {
            "places": "the environment's track-interior place bins with "
            "edge_id >= 0; summed_ground_process_intensity at each",
            "rank": "position among the places ordered by rate (ties in place "
            "order), over n_places - 1",
            "event_place": "the place nearest each of "
            f"{list(cpr.POSITION_COLUMNS)}; the event's rank and ratio are the "
            "means over the positions present, the ratio over the median rate",
        },
        "inputs": {"event_table": file_input(event_table)},
    }
    if args.gather:
        gather_manifest(output, CONTENT_PLACE_RATE_NAME, header)
        return
    max_workers, pool_options = cpu_pool(args)
    run(
        epochs=select_epochs(args.epochs or every_epoch, args.shards, args.n_shards),
        task=run_epoch,
        task_args=(event_table, data_dir, output),
        output=output,
        name=CONTENT_PLACE_RATE_NAME,
        header=header,
        overwrite=args.overwrite,
        max_workers=max_workers,
        pool_options=pool_options,
    )


if __name__ == "__main__":
    main()
