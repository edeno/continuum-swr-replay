"""Place every event's posterior mass on the arms of the track.

For every canonical event (``event_sequence_table``'s time bins, as
``measure_content_along_arms.py`` reads them), its epoch's posterior over the
two non-local states at each of its time bins (the decode's ``results.nc``),
summed over the states and renormalized over the real place bins: the event's
position distribution at that time bin. Each place's mass is summed over the
event's time bins, apart for the time bins at which the place is far from the
animal and those at which it is near
(``analysis/event_properties/content_along_arms.event_place_mass``). A place is
a real place bin of the environment, on its arm as
``content_along_arms.arm_positions`` puts a position there, with its track
distance to the nearest well; its distance to the animal is the environment's
shortest path between the two place bins (``distance_between_nodes_``).

Where ``measure_content_along_arms.py`` gives a time bin's content to its one
most probable place (the MAP), this gives it to every place by its probability.

One table per epoch, ``Processed-Data/content_mass_along_arms/<epoch>.parquet``,
and a ``manifest.json``.

Usage::

    uv run python scripts/event_properties/measure_content_mass_along_arms.py \\
        --n-workers 8 [--epochs chimi20200212_02_r1_HPC]
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.decode_quality.posterior_analysis import (
    map_estimate_to_map_nodes,
)
from continuum_swr_replay.analysis.event_definition import analyzed_time
from continuum_swr_replay.analysis.event_definition.analyzed_time import read_swr_table
from continuum_swr_replay.analysis.event_definition.nonlocal_event_rule import (
    NON_LOCAL_STATES,
)
from continuum_swr_replay.analysis.event_definition.swr_source import (
    swr_epoch_id_of_session,
)
from continuum_swr_replay.analysis.event_properties.content_along_arms import (
    arm_positions,
    check_arm_orientation,
    distance_to_nearest_well,
    event_place_mass,
)
from continuum_swr_replay.analysis.event_properties.event_statistics import FAR_CM
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.decoding.results import open_results
from continuum_swr_replay.paths import (
    CONTENT_MASS_ALONG_ARMS_DIR,
    STATE_SPACE_FEATURES_RUN,
    event_features_root,
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.io import read_manifest, sha256_file, write_json
from continuum_swr_replay.runs.provenance import described_commit, produced_by
from continuum_swr_replay.runs.threads import pool_context
from continuum_swr_replay.tables.io import write_atomically


def measure_epoch(job: tuple[str, str, pd.DataFrame]) -> pd.DataFrame:
    """One epoch's events: the posterior mass of each at each place.

    ``bins`` holds the epoch's event bins (``event_id``, ``time``). Every bin
    must be a row of the decode frame and of ``results.nc``, with the animal's
    position known.
    """
    identifier, expected_sha256, bins = job
    bins = bins.sort_values(["event_id", "time"])
    paths = decode_paths(identifier, processed_data_dir())
    dot = analyzed_time.read_verified_dot_with_swr_source(
        paths.data_over_time, expected_sha256
    )
    environment = pd.read_pickle(paths.environment)
    edge_order = [tuple(edge) for edge in environment.edge_order]
    check_arm_orientation(edge_order, environment.track_graph)
    rows = dot.index.get_indexer(bins["time"].to_numpy())
    if (rows < 0).any():
        raise ValueError(
            f"{identifier}: {(rows < 0).sum()} event bins not in the frame"
        )

    nodes = environment.place_bin_centers_nodes_df_
    real = nodes["node_id"].to_numpy() != -1
    real_nodes = nodes["node_id"].to_numpy()[real]
    read_order = np.argsort(rows)
    with open_results(paths.results, cache=False) as results:
        if not np.allclose(results["time"].to_numpy()[rows], bins["time"].to_numpy()):
            raise ValueError(
                f"{identifier}: results.nc is not on the frame's time bins"
            )
        mass = np.zeros((rows.size, int(real.sum())), dtype=np.float64)
        for state in NON_LOCAL_STATES:
            posterior = results["acausal_posterior"].sel(state=state)
            place = posterior["position"].to_numpy()
            if not np.allclose(
                place, nodes["linear_position"].to_numpy(), equal_nan=True
            ):
                raise ValueError(
                    f"{identifier}: the posterior's bins are not the environment's"
                )
            mass[read_order] += np.nan_to_num(
                posterior.isel(time=rows[read_order]).to_numpy()[:, real]
            )
    total = mass.sum(axis=1, keepdims=True)
    if (total <= 0).any():
        raise ValueError(f"{identifier}: an event bin has no non-local posterior mass")
    mass /= total

    animal_node, known = map_estimate_to_map_nodes(
        dot["linear_position"].to_numpy(dtype=np.float64)[rows], nodes
    )
    if not known.all():
        raise ValueError(
            f"{identifier}: the animal's position is missing at an event bin"
        )
    between = environment.distance_between_nodes_
    here, which = np.unique(animal_node, return_inverse=True)
    from_animal = np.array([[between[a][b] for b in real_nodes] for a in here])
    table = event_place_mass(
        bins["event_id"].to_numpy(), mass, from_animal[which], FAR_CM
    )

    try:
        placed = arm_positions(nodes["linear_position"].to_numpy()[real], nodes)
    except ValueError as error:
        raise ValueError(f"{identifier}: {error}") from error
    placed["distance_to_well_cm"] = distance_to_nearest_well(
        placed["arm"].to_numpy(),
        placed["fraction"].to_numpy(),
        environment.track_graph,
        edge_order,
    )
    table = table.join(placed, on="place").drop(columns="place")
    table.insert(1, "epoch_id", identifier)
    table.insert(2, "decode_sha256", expected_sha256)
    return table.astype({"weight": np.float32, "distance_to_animal_cm": np.float32})


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--epochs", nargs="*")
    parser.add_argument("--output-dir", type=Path, help="write here (a trial run)")
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
        jobs.append(
            (
                identifier,
                record["decode"]["sha256"],
                by_epoch[identifier][["event_id", "time"]],
            )
        )
    folder = args.output_dir or processed_data_dir() / CONTENT_MASS_ALONG_ARMS_DIR
    folder.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(
        max_workers=args.n_workers, mp_context=pool_context()
    ) as pool:
        for table in pool.map(measure_epoch, jobs):
            identifier = table.epoch_id.iloc[0]
            write_atomically(
                folder / f"{identifier}.parquet",
                lambda path, table=table: table.to_parquet(path, index=False),
            )
            print(f"{identifier}: {table.event_id.nunique()} events", flush=True)
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
                "weight": "the posterior of the non-local states "
                f"({', '.join(NON_LOCAL_STATES)}) summed over the states and "
                "renormalized over the real place bins at each event time bin, summed "
                "over the event's time bins; one row per event, place bin and side "
                f"(far: more than {FAR_CM:g} cm from the animal along the track at "
                "that time bin), so an event's weights sum to its number of time bins",
                "place": "a real place bin (node_id not -1), on its arm as the "
                "content-along-arms run places a position; distance_to_well the "
                "track-graph distance to the nearest degree-1 node",
                "distance_to_animal": "the environment's shortest path between the "
                "animal's place bin and the place (distance_between_nodes_), the "
                "mass-weighted mean over the time bins of the side",
            },
            indent=1,
            final_newline=True,
        )


if __name__ == "__main__":
    main()
