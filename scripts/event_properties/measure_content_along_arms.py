"""Place every event time bin's content on its arm of the track.

For every time bin of the canonical events (``event_sequence_table``), its
epoch's decode MAP position (``map_estimate``, read through the SWR source; the
decode's sha256 must be the one the event features run recorded) placed on the
track by ``analysis/event_properties/content_along_arms.arm_positions``: its arm
and the fraction of the arm behind it (1 on a leaf arm is its end at the well);
the content's track distance to the nearest well
(``content_along_arms.distance_to_nearest_well``) and to the animal
(``|ahead_behind_distance|``).

One table per epoch, ``Processed-Data/content_along_arms/<epoch>.parquet``, and
a ``manifest.json``.

Usage::

    uv run python scripts/event_properties/measure_content_along_arms.py \\
        --n-workers 12 [--epochs chimi20200212_02_r1_HPC]
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.detector_comparison.feature_run import (
    STATE_SPACE_FEATURES_RUN,
)
from continuum_swr_replay.analysis.event_definition import analyzed_time
from continuum_swr_replay.analysis.event_definition.analyzed_time import read_swr_table
from continuum_swr_replay.analysis.event_definition.swr_source import (
    swr_epoch_id_of_session,
)
from continuum_swr_replay.analysis.event_properties.content_along_arms import (
    arm_positions,
    check_arm_orientation,
    distance_to_nearest_well,
)
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.paths import (
    CONTENT_ALONG_ARMS_DIR,
    event_features_root,
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.io import read_manifest, sha256_file, write_json
from continuum_swr_replay.runs.provenance import described_commit, produced_by
from continuum_swr_replay.tables.io import write_table


def measure_epoch(job: tuple[str, str, pd.DataFrame]) -> pd.DataFrame:
    """One epoch's event bins placed on the track.

    ``bins`` holds the epoch's event bins (``event_id``, ``time``). Every bin
    must be a row of the decode frame with a finite MAP position.
    """
    identifier, expected_sha256, bins = job
    paths = decode_paths(identifier, processed_data_dir())
    dot = analyzed_time.read_verified_dot_with_swr_source(
        paths.data_over_time, expected_sha256
    )
    env = pd.read_pickle(paths.environment)
    edge_order = [tuple(edge) for edge in env.edge_order]
    check_arm_orientation(edge_order, env.track_graph)
    rows = dot.index.get_indexer(bins["time"].to_numpy())
    if (rows < 0).any():
        raise ValueError(
            f"{identifier}: {(rows < 0).sum()} event bins not in the frame"
        )
    try:
        placed = arm_positions(
            dot["map_estimate"].to_numpy()[rows], env.place_bin_centers_nodes_df_
        )
    except ValueError as error:
        raise ValueError(f"{identifier}: {error}") from error
    return pd.DataFrame(
        {
            "event_id": bins["event_id"].to_numpy(),
            "epoch_id": identifier,
            "decode_sha256": expected_sha256,
            "time": bins["time"].to_numpy(),
            "arm": placed["arm"].to_numpy(),
            "fraction": placed["fraction"].to_numpy(),
            "arm_length_cm": placed["arm_length_cm"].to_numpy(),
            "distance_to_well_cm": distance_to_nearest_well(
                placed["arm"].to_numpy(),
                placed["fraction"].to_numpy(),
                env.track_graph,
                edge_order,
            ),
            "distance_to_animal_cm": np.abs(
                dot["ahead_behind_distance"].to_numpy()[rows]
            ),
        }
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--epochs", nargs="*")
    args = parser.parse_args(argv)
    manifest = read_manifest(event_features_root() / STATE_SPACE_FEATURES_RUN)
    event_table = table_path("event_table", replay_content_dir())
    sequence_table = table_path("event_sequence_table", replay_content_dir())
    events = read_swr_table(event_table, columns=["event_id"])
    bins = pd.read_parquet(
        sequence_table, columns=["animal", "session_date", "epoch", "event_id", "time"]
    )
    if set(bins["event_id"].unique()) != set(events["event_id"]):
        raise ValueError("the sequence table's events are not the event table's")
    sessions = bins[["animal", "session_date", "epoch"]].drop_duplicates()
    sessions["epoch_id"] = [
        swr_epoch_id_of_session(animal, day, epoch)
        for animal, day, epoch in sessions.itertuples(index=False)
    ]
    bins = bins.merge(sessions, on=["animal", "session_date", "epoch"], validate="m:1")
    by_epoch = dict(tuple(bins.groupby("epoch_id")))
    identifiers = sorted(by_epoch)
    if args.epochs:
        identifiers = [i for i in identifiers if i in set(args.epochs)]
    jobs = []
    for identifier in identifiers:
        record = manifest["epochs"][identifier]
        if record["status"] != "ok":
            raise ValueError(f"{identifier}: event features status {record['status']}")
        jobs.append((identifier, record["decode"]["sha256"], by_epoch[identifier]))
    folder = processed_data_dir() / CONTENT_ALONG_ARMS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        for table in pool.map(measure_epoch, jobs):
            identifier = table.epoch_id.iloc[0]
            write_table(table, folder / f"{identifier}.parquet")
            print(f"{identifier}: {len(table)} bins", flush=True)
    if not args.epochs:
        write_json(
            folder / "manifest.json",
            {
                "produced_by": produced_by(__file__),
                "commit": described_commit(),
                "event_features_run": STATE_SPACE_FEATURES_RUN,
                "event_table": {
                    "path": str(event_table),
                    "sha256": sha256_file(event_table),
                },
                "event_sequence_table": {
                    "path": str(sequence_table),
                    "sha256": sha256_file(sequence_table),
                },
                "n_epochs": len(jobs),
                "n_bins": len(bins),
                "position": "the decode's map_estimate at each event bin, on its "
                "nearest real place bin (edge_id >= 0); fraction 0 at the arm's first "
                "node in the edge order, 1 at its second (a leaf arm's end, the well node)",
                "distance_to_well": "track-graph distance to the nearest degree-1 "
                "node, out through either end of the arm (graph edge lengths)",
                "distance_to_animal": "|ahead_behind_distance| of the decode frame",
            },
            indent=1,
            final_newline=True,
        )


if __name__ == "__main__":
    main()
