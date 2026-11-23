"""Count each epoch's immobile time bins inside the well visits by non-local probability.

How the immobile time is spread over the decoder's non-local probability
(Continuous plus Fragmented), for the panel of Figure 2's supplement that shows
the time is almost all near 0 or near 1. The bins are the immobile ones outside
excluded time (``analyzed_time.immobile_analyzed_bins``) inside a well visit
(first poke-in to final poke-out, ``event_definition.well_visits.in_well_visit``,
the visits read from the session cache:
``event_tables.prepare_data.well_visit_times``), counted in 50 equal
probability bins (``analysis.decode_quality.nonlocal_probability.probability_counts``).
For the ``ok`` epochs of an event features run, from the same decodes (each
decode's sha256 must be the one the run recorded), written to
``Processed-Data/nonlocal_probability/counts.parquet`` (or ``--output-dir``)
with a ``manifest.json`` beside it that records each epoch's status and bins,
and the session-cache entries the visits were read from
(``prepare_data.session_cache_record``).

Usage::

    uv run python scripts/decoding/measure_nonlocal_probability.py \\
        --event-features-run e2331f976fd1 --n-workers 8
    uv run python scripts/decoding/measure_nonlocal_probability.py \\
        --event-features-run e2331f976fd1 --epochs chimi20200212_02_r1_HPC \\
        --output-dir /tmp/check
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.decode_quality.nonlocal_probability import (
    PROBABILITY_BIN_EDGES,
    probability_counts,
)
from continuum_swr_replay.analysis.detector_comparison import inventory_run
from continuum_swr_replay.analysis.event_definition import analyzed_time
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    ANALYZED_SUBSET,
)
from continuum_swr_replay.analysis.event_definition.well_visits import in_well_visit
from continuum_swr_replay.analysis.event_tables.prepare_data import (
    session_cache_record,
    well_visit_times,
)
from continuum_swr_replay.parameters import DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC
from continuum_swr_replay.paths import (
    NONLOCAL_PROBABILITY_DIR,
    event_features_root,
    processed_data_dir,
)
from continuum_swr_replay.runs.io import ok_epochs, read_manifest, write_json
from continuum_swr_replay.runs.provenance import described_commit, produced_by
from continuum_swr_replay.tables.io import write_atomically


def _epoch(job: tuple[str, str]) -> tuple[pd.DataFrame, dict]:
    """One epoch's counts, and its record: the bins counted."""
    identifier, expected_sha256 = job
    dot = analyzed_time.read_verified_dot_with_swr_source(
        inventory_run.decode_path(identifier), expected_sha256
    )
    animal, day, epoch = (dot[c].iloc[0] for c in ("animal", "day", "epoch"))
    in_visit = in_well_visit(dot.index.to_numpy(), *well_visit_times(identifier))
    counts = probability_counts(dot, in_visit)
    counts.insert(0, "decode_sha256", expected_sha256)
    counts.insert(0, "epoch", str(epoch))
    counts.insert(0, "session_date", int(day))
    counts.insert(0, "animal", str(animal))
    counts.insert(0, "epoch_id", identifier)
    return counts, {"status": "ok", "n_bins": int(counts["n_bins"].sum())}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--event-features-run", required=True)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument(
        "--epochs", nargs="+", help="only these epochs (a smoke test; default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=f"default: Processed-Data/{NONLOCAL_PROBABILITY_DIR}",
    )
    args = parser.parse_args(argv)
    manifest = read_manifest(event_features_root() / args.event_features_run)
    jobs = [
        (identifier, manifest["epochs"][identifier]["decode"]["sha256"])
        for identifier in ok_epochs(manifest)
        if args.epochs is None or identifier in args.epochs
    ]
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        tables, records = zip(*pool.map(_epoch, jobs), strict=True)
    out_dir = args.output_dir or processed_data_dir() / NONLOCAL_PROBABILITY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.concat(tables, ignore_index=True)
    write_atomically(
        out_dir / "counts.parquet", lambda path: table.to_parquet(path, index=False)
    )
    write_json(
        out_dir / "manifest.json",
        {
            "produced_by": produced_by(__file__),
            "commit": described_commit(),
            "event_features_run": args.event_features_run,
            "subset": ANALYZED_SUBSET,
            "session_cache": session_cache_record(identifier for identifier, _ in jobs),
            "n_epochs": len(tables),
            "bins": f"head_speed below {DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC:g} "
            "cm/s, inside a well visit (first poke-in to final poke-out), outside "
            "excluded time",
            "probability": "non-local probability (Continuous plus Fragmented)",
            "probability_bins": [float(edge) for edge in PROBABILITY_BIN_EDGES],
            "epochs": {
                identifier: record
                for (identifier, _), record in zip(jobs, records, strict=True)
            },
        },
        indent=1,
        final_newline=True,
    )
    print(f"wrote {len(tables)} epochs to {out_dir / 'counts.parquet'}")


if __name__ == "__main__":
    main()
