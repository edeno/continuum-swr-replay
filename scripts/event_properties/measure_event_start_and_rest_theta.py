"""Measure each event's start distance and the theta-like share of duration-matched quiet rest.

For every immobile non-local event of the canonical event table, from its
epoch's decode frame (read through the SWR source; the decode's sha256 must be
the one the event features run recorded) and environment: the start distance
(track-graph distance from the animal to the event's first decoded position)
and the share of ``--n-windows`` duration-matched windows of quiet rest whose
theta is theta-like. The measures are
``analysis/event_properties/start_and_rest_theta.py``'s; this script selects the
epochs, reads the frames and writes the tables.

One table per epoch, ``Processed-Data/event_start_rest_theta/<epoch>.parquet``,
and a ``manifest.json``.

Usage::

    uv run python scripts/event_properties/measure_event_start_and_rest_theta.py \\
        --n-workers 12 [--epochs chimi20200212_02_r1_HPC]
"""

from __future__ import annotations

import argparse
import zlib
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.detector_comparison.feature_run import (
    STATE_SPACE_FEATURES_RUN,
)
from continuum_swr_replay.analysis.event_definition import analysed_time
from continuum_swr_replay.analysis.event_definition.analysed_time import read_swr_table
from continuum_swr_replay.analysis.event_definition.swr_source import (
    swr_epoch_id_of_session,
)
from continuum_swr_replay.analysis.event_properties.start_and_rest_theta import (
    start_and_rest_theta,
)
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.paths import (
    EVENT_START_REST_THETA_DIR,
    event_features_root,
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.io import read_manifest, sha256_file, write_json
from continuum_swr_replay.runs.provenance import described_commit, produced_by

SEED = 20261028


def measure_epoch(job: tuple[str, str, pd.DataFrame, int]) -> pd.DataFrame:
    """One epoch's events: start distance and duration-matched rest theta.

    The decode frame is read through the SWR source, refused unless its
    sha256 is ``expected_sha256``; the measures are
    ``start_and_rest_theta.start_and_rest_theta``'s, with the epoch and the
    decode's hash beside each event.
    """
    identifier, expected_sha256, events, n_windows = job
    paths = decode_paths(identifier, processed_data_dir())
    dot = analysed_time.read_verified_dot_with_swr_source(
        paths.data_over_time, expected_sha256
    )
    env = pd.read_pickle(paths.environment)
    rng = np.random.default_rng([SEED, zlib.crc32(identifier.encode())])
    try:
        table = start_and_rest_theta(dot, env, events, n_windows, rng)
    except ValueError as error:
        raise ValueError(f"{identifier}: {error}") from error
    table.insert(1, "epoch_id", identifier)
    table.insert(2, "decode_sha256", expected_sha256)
    return table


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--n-windows", type=int, default=10)
    parser.add_argument("--epochs", nargs="*")
    args = parser.parse_args(argv)
    manifest = read_manifest(event_features_root() / STATE_SPACE_FEATURES_RUN)
    event_table = table_path("event_table", replay_content_dir())
    events = read_swr_table(
        event_table,
        columns=["event_id", "animal", "session_date", "epoch", "non_local_event"],
    )
    events["epoch_id"] = [
        swr_epoch_id_of_session(animal, day, epoch)
        for animal, day, epoch in zip(
            events.animal, events.session_date, events.epoch, strict=True
        )
    ]
    by_epoch = dict(tuple(events.groupby("epoch_id")))
    identifiers = sorted(by_epoch)
    if args.epochs:
        identifiers = [i for i in identifiers if i in set(args.epochs)]
    jobs = []
    for identifier in identifiers:
        record = manifest["epochs"][identifier]
        if record["status"] != "ok":
            raise ValueError(f"{identifier}: event features status {record['status']}")
        jobs.append(
            (
                identifier,
                record["decode"]["sha256"],
                by_epoch[identifier],
                args.n_windows,
            )
        )
    folder = processed_data_dir() / EVENT_START_REST_THETA_DIR
    folder.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        for table in pool.map(measure_epoch, jobs):
            identifier = table.epoch_id.iloc[0]
            tmp = folder / f"{identifier}.parquet.tmp"
            table.to_parquet(tmp, index=False)
            tmp.replace(folder / f"{identifier}.parquet")
            print(f"{identifier}: {len(table)} events", flush=True)
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
                "n_epochs": len(jobs),
                "n_windows": args.n_windows,
                "seed": SEED,
                "start_distance": "track-graph distance (env.distance_between_nodes_) "
                "from the animal's linear position to the MAP at the event's first "
                "bin with a finite MAP, both on their nearest real place bins",
                "rest": "head_speed below 4 cm/s, in no non-local, SWR or MUA event, "
                "outside excluded time; windows of the event's number of bins, "
                "uniform over every such window of the epoch",
                "theta_like": "window mean of theta_power_zscore at or above the "
                "epoch's median running theta (above 10 cm/s)",
            },
            indent=1,
            final_newline=True,
        )


if __name__ == "__main__":
    main()
