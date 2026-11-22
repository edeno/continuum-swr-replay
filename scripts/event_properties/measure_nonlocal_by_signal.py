"""Count each epoch's immobile time bins by ripple power, by MUA and by non-local probability.

Where the non-local time sits against the SWR threshold and against MUA: how
much of it has ripple power below the SWR threshold, and the share of bins that
are non-local at each power (Figure 2i), and the same for MUA (Figure 2's
supplement), over the immobile time inside the well visits (the epoch's port
experiences, read from the session cache:
``event_tables.prepare_data.well_visit_times``). One pass over each decode
frame counts, for each signal, the bins by signal bin and non-local probability
bin, 0.8 an edge (``analysis.event_properties.nonlocal_by_signal.rest_signal_counts``),
so every count the figures and the events summary read is a sum over one
table. The manifest records each epoch's immobile bins kept and left out as
outside every visit, and the session-cache entries the visits were read from
(``prepare_data.session_cache_record``). For the ``ok`` epochs of an event
features run, from the same decodes (each decode's sha256 must be the one the
run recorded), written to ``Processed-Data/nonlocal_by_signal/counts.parquet``
(or ``--output-dir``) with a ``manifest.json`` beside it.

Usage::

    uv run python scripts/event_properties/measure_nonlocal_by_signal.py \\
        --event-features-run e2331f976fd1 --n-workers 8
    uv run python scripts/event_properties/measure_nonlocal_by_signal.py \\
        --event-features-run e2331f976fd1 --epochs chimi20200212_02_r1_HPC \\
        --output-dir /tmp/check
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.detector_comparison import inventory_run
from continuum_swr_replay.analysis.event_definition import analyzed_time
from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    immobile_analyzed_bins,
)
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    ANALYZED_SUBSET,
)
from continuum_swr_replay.analysis.event_definition.well_visits import in_well_visit
from continuum_swr_replay.analysis.event_properties.class_or_continuum import (
    KAY_ZSCORE_THRESHOLD,
)
from continuum_swr_replay.analysis.event_properties.nonlocal_by_signal import (
    MUA_LEVEL_Z,
    PROBABILITY_EDGES,
    SIGNALS,
    rest_signal_counts,
)
from continuum_swr_replay.analysis.event_tables.prepare_data import (
    session_cache_record,
    well_visit_times,
)
from continuum_swr_replay.parameters import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.paths import (
    NONLOCAL_BY_SIGNAL_DIR,
    event_features_root,
    processed_data_dir,
)
from continuum_swr_replay.runs.io import ok_epochs, read_manifest, write_json
from continuum_swr_replay.runs.provenance import described_commit, produced_by


def _epoch(job: tuple[str, str]) -> tuple[pd.DataFrame, dict[str, int]]:
    """One epoch's counts, and its immobile bins kept and left out by the visits."""
    identifier, expected_sha256 = job
    dot = analyzed_time.read_verified_dot_with_swr_source(
        inventory_run.decode_path(identifier), expected_sha256
    )
    animal, day, epoch = (dot[c].iloc[0] for c in ("animal", "day", "epoch"))
    in_visit = in_well_visit(dot.index.to_numpy(), *well_visit_times(identifier))
    immobile = immobile_analyzed_bins(dot)
    counts = rest_signal_counts(dot, in_visit)
    counts.insert(0, "decode_sha256", expected_sha256)
    counts.insert(0, "epoch", str(epoch))
    counts.insert(0, "session_date", int(day))
    counts.insert(0, "animal", str(animal))
    counts.insert(0, "epoch_id", identifier)
    return counts, {
        "n_immobile_bins_in_visit": int((immobile & in_visit).sum()),
        "n_immobile_bins_outside_visit": int((immobile & ~in_visit).sum()),
    }


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
        help=f"default: Processed-Data/{NONLOCAL_BY_SIGNAL_DIR}",
    )
    args = parser.parse_args(argv)
    manifest = read_manifest(event_features_root() / args.event_features_run)
    jobs = [
        (identifier, manifest["epochs"][identifier]["decode"]["sha256"])
        for identifier in ok_epochs(manifest)
        if args.epochs is None or identifier in args.epochs
    ]
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        tables, kept = zip(*pool.map(_epoch, jobs), strict=True)
    out_dir = args.output_dir or processed_data_dir() / NONLOCAL_BY_SIGNAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.concat(tables, ignore_index=True)
    tmp = out_dir / "counts.parquet.tmp"
    table.to_parquet(tmp, index=False)
    tmp.replace(out_dir / "counts.parquet")
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
            "nonlocal": "non-local probability (Continuous plus Fragmented) above "
            f"{DEFAULT_NON_LOCAL_THRESHOLD:g}",
            "signals": {name: column for name, (column, _) in SIGNALS.items()},
            "signal_bins": "0.25 z wide from -2 to 10 with an open bin at each "
            f"end; the Kay threshold ({KAY_ZSCORE_THRESHOLD:g} z) and the MUA "
            f"level ({MUA_LEVEL_Z:g} z) are edges",
            "probability_bins": [float(edge) for edge in PROBABILITY_EDGES],
            "epochs": {
                identifier: record
                for (identifier, _), record in zip(jobs, kept, strict=True)
            },
        },
        indent=1,
        final_newline=True,
    )
    print(f"wrote {len(tables)} epochs to {out_dir / 'counts.parquet'}")


if __name__ == "__main__":
    main()
