"""Measure each epoch's theta, ripple and MUA levels during running and during quiet rest.

The references an event's ``theta_z``, ``ripple_power_z`` and ``mua_rate_z`` are
read against (``analysis.event_properties.brain_state_reference``): is a non-local
event at rest a theta-like event, or a quiet-rest one? One row per epoch, for
the ``ok`` epochs of an event features run, from the same decodes (each
decode's sha256 must be the one the run recorded), written to
``Processed-Data/brain_state_reference/reference.parquet`` with a
``manifest.json`` beside it.

Usage::

    uv run python scripts/detection/measure_brain_state_reference.py \\
        --event-features-run 30cf1cc432e9 --n-workers 8
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import pandas as pd

from continuum_swr_replay.analysis.event_properties.brain_state_reference import (
    brain_state_reference,
)
from continuum_swr_replay.analysis.events import analysed_time
from continuum_swr_replay.analysis.inventory_comparison import inventory_run
from continuum_swr_replay.paths import (
    BRAIN_STATE_REFERENCE_DIR,
    event_features_root,
    processed_data_dir,
)
from continuum_swr_replay.runs.io import ok_epochs, read_manifest, write_json
from continuum_swr_replay.runs.provenance import described_commit, produced_by


def _epoch(job: tuple[str, str]) -> dict[str, Any]:
    identifier, expected_sha256 = job
    dot = analysed_time.read_verified_dot_with_swr_source(
        inventory_run.decode_path(identifier), expected_sha256
    )
    animal, day, epoch = (dot[c].iloc[0] for c in ("animal", "day", "epoch"))
    return {
        "epoch_id": identifier,
        "animal": str(animal),
        "session_date": int(day),
        "epoch": str(epoch),
        "decode_sha256": expected_sha256,
        **brain_state_reference(dot),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--event-features-run", required=True)
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args(argv)
    manifest = read_manifest(event_features_root() / args.event_features_run)
    jobs = [
        (identifier, manifest["epochs"][identifier]["decode"]["sha256"])
        for identifier in ok_epochs(manifest)
    ]
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        rows = list(pool.map(_epoch, jobs))
    out_dir = processed_data_dir() / BRAIN_STATE_REFERENCE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    tmp = out_dir / "reference.parquet.tmp"
    table.to_parquet(tmp, index=False)
    tmp.replace(out_dir / "reference.parquet")
    write_json(
        out_dir / "manifest.json",
        {
            "produced_by": produced_by(__file__),
            "commit": described_commit(),
            "event_features_run": args.event_features_run,
            "n_epochs": len(table),
            "running": "head_speed above 10 cm/s, outside excluded time",
            "rest": "head_speed below 4 cm/s, in no non-local, SWR or MUA event, "
            "outside excluded time",
        },
        indent=1,
        final_newline=True,
    )
    print(f"wrote {len(table)} epochs to {out_dir / 'reference.parquet'}")


if __name__ == "__main__":
    main()
