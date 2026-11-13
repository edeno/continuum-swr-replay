"""Add each compared event's Local, No-Spike, non-local and uncertain shares to an event features run.

The event features run stores, for every compared event of the nine
inventories, the share of its decode-grid rows above the non-local threshold
(``ss_frac_rows_nonlocal``) but not the shares of the other states. This pass
reads each ``ok`` epoch's ``features`` table (its ``first_row`` and ``n_rows``)
and the same decode the run read, and writes
:func:`~continuum_swr_replay.analysis.detector_comparison.features.state_row_shares`
per event beside the run's tables, as ``{epoch_id}_state_shares.parquet``.

Per epoch it refuses to write unless:

- the decode's sha256 is the one the run's manifest records;
- the decode grid has the number of rows the run recorded;
- the recomputed non-local share equals the stored ``ss_frac_rows_nonlocal``
  exactly, wherever both are known.

A ``state_shares_manifest.json`` beside them records the commit and each
epoch's status. A failing epoch is recorded and the rest run; a rerun skips the
epochs already ``ok`` unless ``--overwrite`` is given.

Usage::

    uv run python scripts/detector_comparison/add_state_shares.py \\
        --event-features-run 30cf1cc432e9 --n-workers 8
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.detector_comparison import inventory_run
from continuum_swr_replay.analysis.detector_comparison.features import (
    STATE_SHARE_COLUMNS,
    PosteriorTraces,
    state_row_shares,
)
from continuum_swr_replay.analysis.event_definition import analysed_time
from continuum_swr_replay.paths import event_features_root
from continuum_swr_replay.runs.io import (
    ok_epochs,
    output_path,
    read_manifest,
    write_json,
)
from continuum_swr_replay.runs.provenance import described_commit, produced_by

MANIFEST_NAME = "state_shares_manifest.json"
KEY_COLUMNS = ("animal", "day", "epoch", "detector", "event_key")


def shares_path(run_dir: Path, identifier: str) -> Path:
    """Where one epoch's state shares are written."""
    return run_dir / f"{identifier}_state_shares.parquet"


def epoch_state_shares(
    run_dir: Path, identifier: str, record: dict[str, Any]
) -> pd.DataFrame:
    """One epoch's events with their four state shares.

    Raises
    ------
    ValueError
        The decode is not the one the run read, its grid is not the run's, or
        the recomputed non-local share differs from the stored one.
    """
    dot = analysed_time.read_verified_dot_with_swr_source(
        inventory_run.decode_path(identifier), record["decode"]["sha256"]
    )
    if len(dot) != record["n_grid_rows"]:
        raise ValueError(
            f"the decode grid has {len(dot)} rows; the run read {record['n_grid_rows']}"
        )
    features = pd.read_parquet(
        output_path(identifier, run_dir, "features"),
        columns=[*KEY_COLUMNS, "first_row", "n_rows", "ss_frac_rows_nonlocal"],
    )
    shares = state_row_shares(
        PosteriorTraces.from_frame(dot),
        features["first_row"].to_numpy(dtype=np.int64),
        features["n_rows"].to_numpy(dtype=np.int64),
    )
    stored = features["ss_frac_rows_nonlocal"].to_numpy(dtype=np.float64)
    recomputed = shares["ss_frac_rows_nonlocal"].to_numpy()
    both = np.isfinite(stored) & np.isfinite(recomputed)
    if not np.array_equal(stored[both], recomputed[both]):
        n_differ = int((stored[both] != recomputed[both]).sum())
        raise ValueError(
            f"{n_differ} events' recomputed non-local share differs from the stored one"
        )
    return pd.concat(
        [features[list(KEY_COLUMNS)].reset_index(drop=True), shares], axis=1
    )


def _run_epoch(args: tuple[Path, str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    run_dir, identifier, record = args
    try:
        table = epoch_state_shares(run_dir, identifier, record)
    except (ValueError, OSError, KeyError) as error:
        return identifier, {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
        }
    path = shares_path(run_dir, identifier)
    tmp = path.with_suffix(".parquet.tmp")
    table.to_parquet(tmp, index=False)
    tmp.replace(path)
    known = table[list(STATE_SHARE_COLUMNS)].notna().all(axis=1)
    return identifier, {
        "status": "ok",
        "n_events": int(len(table)),
        "n_known": int(known.sum()),
        "max_share_sum_error": float(
            np.abs(table.loc[known, list(STATE_SHARE_COLUMNS)].sum(axis=1) - 1.0).max()
        )
        if known.any()
        else None,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--event-features-run", required=True)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    run_dir = event_features_root() / args.event_features_run
    run_manifest = read_manifest(run_dir)
    manifest_path = run_dir / MANIFEST_NAME
    previous = (
        json.loads(manifest_path.read_text())["epochs"]
        if manifest_path.exists()
        else {}
    )
    ok = ok_epochs(run_manifest)
    todo = [
        i
        for i in ok
        if args.overwrite
        or previous.get(i, {}).get("status") != "ok"
        or not shares_path(run_dir, i).exists()
    ]
    print(f"{len(ok)} ok epochs in run {args.event_features_run}; {len(todo)} to do")
    jobs = [(run_dir, i, run_manifest["epochs"][i]) for i in todo]
    epochs = {i: previous[i] for i in ok if i in previous and i not in todo}
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        for identifier, record in pool.map(_run_epoch, jobs):
            epochs[identifier] = record
            print(f"  {identifier}: {record['status']} {record.get('error', '')}")
    manifest = {
        "produced_by": produced_by(__file__),
        "commit": described_commit(),
        "event_features_run": args.event_features_run,
        "columns": list(STATE_SHARE_COLUMNS),
        "rule": "a row is in a state when its probability is above 0.8 (float32); "
        "uncertain when none is",
        "epochs": dict(sorted(epochs.items())),
    }
    write_json(manifest_path, manifest, indent=1, final_newline=True)
    n_failed = sum(r["status"] != "ok" for r in epochs.values())
    print(f"done: {len(epochs) - n_failed} ok, {n_failed} failed -> {manifest_path}")
    if n_failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
