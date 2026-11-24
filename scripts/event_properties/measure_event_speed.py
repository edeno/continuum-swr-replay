"""Measure how fast each event's content moves along the track, two ways.

For every immobile non-local event of the canonical event table that starts
inside a well visit (``cohort_selection.select_visit_context``), from its
epoch's decode frame (read through the SWR source; the decode's sha256 must be
the one the event features run recorded), its ``results.nc`` and its
environment: the MAP line-fit speed and the transport-slope speed
(``analysis/event_properties/event_speed.py``, which has the definitions). An
event's time bins are the frame's bins that carry its label.

One table per epoch, ``Processed-Data/event_speed/<epoch>.parquet``, and a
``manifest.json``.

Usage::

    uv run python scripts/event_properties/measure_event_speed.py \\
        --n-workers 8 [--epochs chimi20200212_02_r1_HPC]
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
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    ANALYZED_SUBSET,
    count_visit_context,
    select_visit_context,
)
from continuum_swr_replay.analysis.event_definition.swr_source import (
    swr_epoch_id_of_session,
)
from continuum_swr_replay.analysis.event_properties.event_speed import (
    CONTINUOUS_PROBABILITY,
    MIN_LAGS,
    TRANSPORT_LAG_BINS,
    event_speeds,
)
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.decoding.results import open_results
from continuum_swr_replay.parameters import SAMPLING_FREQUENCY
from continuum_swr_replay.paths import (
    EVENT_SPEED_DIR,
    event_features_root,
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.io import read_manifest, sha256_file, write_json
from continuum_swr_replay.runs.provenance import described_commit, produced_by

CONTINUOUS_STATE = "Non-Local Continuous"


def measure_epoch(job: tuple[str, str, pd.DataFrame]) -> pd.DataFrame:
    """One epoch's events and their two speeds.

    The decode frame is read through the SWR source, refused unless its sha256
    is ``expected_sha256``; the Continuous-state posterior is read from the
    decode's ``results.nc`` at the events' time bins, on the environment's
    real place bins.
    """
    identifier, expected_sha256, events = job
    paths = decode_paths(identifier, processed_data_dir())
    dot = analyzed_time.read_verified_dot_with_swr_source(
        paths.data_over_time, expected_sha256
    )
    environment = pd.read_pickle(paths.environment)
    labels = dot["non_local_event"].to_numpy()
    event_of_label = pd.Series(
        events.event_id.to_numpy(), index=events.non_local_event.to_numpy()
    )
    in_event = np.flatnonzero(np.isin(labels, event_of_label.index))
    # each event's bins together, in time order; events in the order of the frame
    rows = in_event[np.argsort(labels[in_event], kind="stable")]
    event_id = event_of_label.loc[labels[rows]].to_numpy()
    if set(event_id) != set(events.event_id):
        raise ValueError(f"{identifier}: an event's label has no bin in the frame")

    nodes = environment.place_bin_centers_nodes_df_
    real = nodes["node_id"].to_numpy() != -1
    read_order = np.argsort(rows)
    with open_results(paths.results, cache=False) as results:
        time = results["time"].to_numpy()
        frame_time = dot.index.to_numpy()
        if time.size != frame_time.size or not np.allclose(
            time[rows], frame_time[rows]
        ):
            raise ValueError(
                f"{identifier}: results.nc is not on the frame's time bins"
            )
        posterior = results["acausal_posterior"].sel(state=CONTINUOUS_STATE)
        place = posterior["position"].to_numpy()
        if not np.allclose(place, nodes["linear_position"].to_numpy(), equal_nan=True):
            raise ValueError(
                f"{identifier}: the posterior's bins are not the environment's"
            )
        continuous = np.empty((rows.size, int(real.sum())), dtype=np.float64)
        continuous[read_order] = np.nan_to_num(
            posterior.isel(time=rows[read_order]).to_numpy()[:, real]
        )
    try:
        table = event_speeds(
            event_id,
            rows,
            dot["map_estimate"].to_numpy(dtype=np.float64)[rows],
            continuous,
            environment,
            1 / SAMPLING_FREQUENCY,
        )
    except ValueError as error:
        raise ValueError(f"{identifier}: {error}") from error
    table.insert(1, "epoch_id", identifier)
    table.insert(2, "decode_sha256", expected_sha256)
    return table


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--epochs", nargs="*")
    parser.add_argument("--output-dir", help="write here instead (a trial run)")
    args = parser.parse_args(argv)
    manifest = read_manifest(event_features_root() / STATE_SPACE_FEATURES_RUN)
    event_table = table_path("event_table", replay_content_dir())
    table = read_swr_table(
        event_table,
        columns=[
            "event_id",
            "animal",
            "session_date",
            "epoch",
            "non_local_event",
            "visit_context",
        ],
    )
    events = select_visit_context(table)
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
        jobs.append((identifier, record["decode"]["sha256"], by_epoch[identifier]))
    folder = processed_data_dir() / EVENT_SPEED_DIR
    if args.output_dir:
        from pathlib import Path

        folder = Path(args.output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        for measured in pool.map(measure_epoch, jobs):
            identifier = measured.epoch_id.iloc[0]
            tmp = folder / f"{identifier}.parquet.tmp"
            measured.to_parquet(tmp, index=False)
            tmp.replace(folder / f"{identifier}.parquet")
            print(f"{identifier}: {len(measured)} events", flush=True)
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
                "subset": ANALYZED_SUBSET,
                "visit_context_counts": {
                    label: int(n)
                    for label, n in count_visit_context(table)["n"].items()
                },
                "n_epochs": len(jobs),
                "n_events": len(events),
                "map_line_fit_speed": "the slope, at least 0, of the track-graph "
                "distance (env.distance_between_nodes_) from the event's first MAP "
                "position to its MAP position at each time bin, against time",
                "transport_slope_speed": "the slope, at least 0, of the mean earth "
                "mover's distance on the track graph between the renormalized "
                f"{CONTINUOUS_STATE} posteriors of two time bins a lag apart, "
                f"against the lag, over lags of {TRANSPORT_LAG_BINS} time bins "
                f"({MIN_LAGS} at least), on the time bins whose state probability "
                f"is above {CONTINUOUS_PROBABILITY}",
            },
            indent=1,
            final_newline=True,
        )


if __name__ == "__main__":
    main()
