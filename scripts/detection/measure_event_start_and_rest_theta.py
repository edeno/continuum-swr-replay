"""Measure each event's start distance and the theta-like share of duration-matched quiet rest.

For every immobile non-local event of the canonical event table, from its
epoch's decode frame (read through the SWR source; the decode's sha256 must be
the one the event features run recorded) and environment:

- **start distance**: track-graph distance from the animal's position to the
  event's first decoded position (the MAP at the event's first bin with a
  finite MAP), both on real place bins, never a junction placeholder
  (``posterior_analysis.track_distance_cm``);
- **duration-matched quiet rest**: ``--n-windows`` windows of quiet rest
  (below 4 cm/s, in no non-local, SWR or MUA event, outside excluded time:
  ``brain_state_reference.regime_masks``) with as many bins as the event, drawn
  uniformly over every such window of the epoch; each window's theta is the
  mean of ``theta_power_zscore`` over its bins, as an event's ``theta_z`` is,
  and it is theta-like when it reaches the epoch's median running theta
  (``brain_state_reference``). The event's ``rest_theta_like_share`` is the
  share of its windows that are theta-like.

One table per epoch, ``Processed-Data/event_start_rest_theta/<epoch>.parquet``,
and a ``manifest.json``.

Usage::

    uv run python scripts/detection/measure_event_start_and_rest_theta.py \\
        --n-workers 12 [--epochs chimi20200212_02_r1_HPC]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import pickle
import zlib
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.events import analysed_time
from continuum_swr_replay.analysis.hippocampus.brain_state_reference import (
    brain_state_reference,
    matched_window_means,
    regime_masks,
)
from continuum_swr_replay.analysis.hippocampus.posterior_analysis import (
    track_distance_cm,
)
from continuum_swr_replay.decoding.files import decode_paths
from continuum_swr_replay.decoding.identifiers import format_epoch_identifier
from continuum_swr_replay.paths import (
    event_features_root,
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.reported.events import EVENT_FEATURES_RUN
from continuum_swr_replay.runs.io import read_manifest
from continuum_swr_replay.runs.provenance import described_commit, produced_by

OUT_NAME = "event_start_rest_theta"
SEED = 20261028


def measure_epoch(job: tuple[str, str, pd.DataFrame, int]) -> pd.DataFrame:
    """One epoch's events: start distance and duration-matched rest theta."""
    identifier, expected_sha256, events, n_windows = job
    paths = decode_paths(identifier, processed_data_dir())
    raw = paths.data_over_time.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    if sha256 != expected_sha256:
        raise ValueError(f"{identifier}: decode sha256 {sha256} is not the run's")
    dot = analysed_time.with_swr_source(pickle.load(io.BytesIO(raw)))
    env = pd.read_pickle(paths.environment)

    labels = dot["non_local_event"].to_numpy()
    order = np.argsort(labels, kind="stable")
    bounds = np.searchsorted(labels[order], events.non_local_event.to_numpy())
    ends = np.searchsorted(
        labels[order], events.non_local_event.to_numpy(), side="right"
    )
    n_rows = ends - bounds
    if (n_rows == 0).any():
        raise ValueError(f"{identifier}: an event's label has no bin in the frame")
    map_estimate = dot["map_estimate"].to_numpy(dtype=np.float64)
    animal = dot["linear_position"].to_numpy(dtype=np.float64)
    theta = dot["theta_power_zscore"].to_numpy(dtype=np.float64)

    first_decoded = np.full(len(events), -1, dtype=np.int64)
    theta_mean = np.full(len(events), np.nan)
    contiguous = np.zeros(len(events), dtype=bool)
    for k, (a, b) in enumerate(zip(bounds, ends, strict=True)):
        rows = np.sort(order[a:b])
        contiguous[k] = rows[-1] - rows[0] + 1 == rows.size
        finite = rows[np.isfinite(map_estimate[rows])]
        if finite.size:
            first_decoded[k] = finite[0]
        theta_mean[k] = np.nanmean(theta[rows])
    has_map = first_decoded >= 0
    start = np.full(len(events), np.nan)
    start[has_map] = track_distance_cm(
        animal[first_decoded[has_map]],
        map_estimate[first_decoded[has_map]],
        env.place_bin_centers_nodes_df_,
        env.distance_between_nodes_,
    )

    rest = regime_masks(dot)["rest"]
    reference = brain_state_reference(dot)
    rng = np.random.default_rng([SEED, zlib.crc32(identifier.encode())])
    windows = matched_window_means(theta, rest, n_rows, n_windows, rng)
    valid = np.isfinite(windows)
    above = windows >= reference["theta_running_median"]
    with np.errstate(invalid="ignore"):
        share = np.where(
            valid.any(axis=1), (above & valid).sum(axis=1) / valid.sum(axis=1), np.nan
        )
    return pd.DataFrame(
        {
            "event_id": events.event_id.to_numpy(),
            "epoch_id": identifier,
            "decode_sha256": sha256,
            "n_bins": n_rows,
            "bins_contiguous": contiguous,
            "start_distance_cm": start,
            "theta_mean": theta_mean,
            "theta_running_median": reference["theta_running_median"],
            "theta_rest_median": reference["theta_rest_median"],
            "rest_windows": valid.sum(axis=1),
            "rest_theta_like_share": share,
            "rest_theta_mean": np.nanmean(np.where(valid, windows, np.nan), axis=1)
            if valid.any()
            else np.nan,
        }
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--n-windows", type=int, default=10)
    parser.add_argument("--epochs", nargs="*")
    args = parser.parse_args(argv)
    manifest = read_manifest(event_features_root() / EVENT_FEATURES_RUN)
    event_table = table_path("event_table", replay_content_dir())
    events = pd.read_parquet(
        event_table,
        columns=["event_id", "animal", "session_date", "epoch", "non_local_event"],
    )
    events["epoch_id"] = [
        format_epoch_identifier(f"{animal}{day}_.nwb", epoch, "HPC")
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
    folder = processed_data_dir() / OUT_NAME
    folder.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        for table in pool.map(measure_epoch, jobs):
            identifier = table.epoch_id.iloc[0]
            tmp = folder / f"{identifier}.parquet.tmp"
            table.to_parquet(tmp, index=False)
            tmp.replace(folder / f"{identifier}.parquet")
            print(f"{identifier}: {len(table)} events", flush=True)
    if not args.epochs:
        (folder / "manifest.json").write_text(
            json.dumps(
                {
                    "produced_by": produced_by(__file__),
                    "commit": described_commit(),
                    "event_features_run": EVENT_FEATURES_RUN,
                    "event_table": {
                        "path": str(event_table),
                        "sha256": hashlib.sha256(event_table.read_bytes()).hexdigest(),
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
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
