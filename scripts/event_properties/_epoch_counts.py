"""The skeleton the per-epoch counts runners share.

``measure_nonlocal_by_signal.py``, ``measure_state_by_signal.py`` (here) and
``scripts/decoding/measure_nonlocal_probability.py`` each count every ``ok``
epoch of an event features run from its decode (each decode's sha256 must be
the one the run recorded), over the time inside the well visits read from the
session cache (``event_tables.prepare_data.well_visit_times``), and write one
``counts.parquet`` with a ``manifest.json`` beside it. :func:`run` does that;
each script gives its own measurement of one epoch, its output folder and the
manifest keys that say what it counted.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

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
from continuum_swr_replay.paths import event_features_root, processed_data_dir
from continuum_swr_replay.runs.io import ok_epochs, read_manifest, write_json
from continuum_swr_replay.runs.provenance import described_commit
from continuum_swr_replay.tables.io import write_atomically

Measure = Callable[[pd.DataFrame, np.ndarray], tuple[pd.DataFrame, dict]]
"""One epoch's measurement: given its decode frame and whether each bin is
inside a well visit, its counts and its manifest record."""

BINS = (
    f"head_speed below {DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC:g} cm/s, inside a "
    "well visit (first poke-in to final poke-out), outside excluded time"
)
"""The manifest's ``bins``: the time every counts run counts."""


def _epoch(job: tuple[str, str], measure: Measure) -> tuple[pd.DataFrame, dict]:
    """One epoch's counts, after its ``epoch_id``, ``animal``,
    ``session_date``, ``epoch`` and ``decode_sha256``, and its record."""
    identifier, expected_sha256 = job
    dot = analyzed_time.read_verified_dot_with_swr_source(
        inventory_run.decode_path(identifier), expected_sha256
    )
    animal, day, epoch = (dot[c].iloc[0] for c in ("animal", "day", "epoch"))
    in_visit = in_well_visit(dot.index.to_numpy(), *well_visit_times(identifier))
    counts, record = measure(dot, in_visit)
    counts.insert(0, "decode_sha256", expected_sha256)
    counts.insert(0, "epoch", str(epoch))
    counts.insert(0, "session_date", int(day))
    counts.insert(0, "animal", str(animal))
    counts.insert(0, "epoch_id", identifier)
    return counts, record


def run(
    argv: Sequence[str] | None,
    *,
    description: str,
    produced_by: str,
    output_name: str,
    measure: Measure,
    counted: Mapping[str, object],
) -> None:
    """Parse the runner's arguments, count every epoch and write the counts and
    the manifest.

    Parameters
    ----------
    argv : sequence of str or None
        The command line (None: ``sys.argv``).
    description : str
        The runner's one-line description.
    produced_by : str
        The runner's ``runs.provenance.produced_by``.
    output_name : str
        The output folder's name under ``Processed-Data`` (``--output-dir``
        replaces it).
    measure : Measure
        One epoch's measurement (a module-level function, sent to the worker
        processes).
    counted : mapping
        The manifest's keys after ``bins``, in order: what the counts are by.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--event-features-run", required=True)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument(
        "--epochs", nargs="+", help="only these epochs (a smoke test; default: all)"
    )
    parser.add_argument(
        "--output-dir", type=Path, help=f"default: Processed-Data/{output_name}"
    )
    args = parser.parse_args(argv)
    manifest = read_manifest(event_features_root() / args.event_features_run)
    jobs = [
        (identifier, manifest["epochs"][identifier]["decode"]["sha256"])
        for identifier in ok_epochs(manifest)
        if args.epochs is None or identifier in args.epochs
    ]
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        tables, records = zip(
            *pool.map(partial(_epoch, measure=measure), jobs), strict=True
        )
    out_dir = args.output_dir or processed_data_dir() / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.concat(tables, ignore_index=True)
    write_atomically(
        out_dir / "counts.parquet", lambda path: table.to_parquet(path, index=False)
    )
    write_json(
        out_dir / "manifest.json",
        {
            "produced_by": produced_by,
            "commit": described_commit(),
            "event_features_run": args.event_features_run,
            "subset": ANALYZED_SUBSET,
            "session_cache": session_cache_record(identifier for identifier, _ in jobs),
            "n_epochs": len(tables),
            "bins": BINS,
            **counted,
            "epochs": {
                identifier: record
                for (identifier, _), record in zip(jobs, records, strict=True)
            },
        },
        indent=1,
        final_newline=True,
    )
    print(f"wrote {len(tables)} epochs to {out_dir / 'counts.parquet'}")
