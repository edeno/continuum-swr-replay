"""Measure each epoch's non-local detector exposure, and its part inside the well visits.

The rate at which an analysis that starts from another inventory leaves non-local
events out (the events summary's left-out values, question 2b.2) is per hour of
rest inside the well visits, as the events are those that start inside one. The
exposure is the inventory comparison's (the time the non-local detector observed,
less excluded time and the events set aside, inside the still intervals), rebuilt
from the tables the inventory run stored and each decode frame's speed trace
(``nonlocal_comparison.exposure_in_well_visits``); a rebuilt exposure that is not
the stored one is refused. Its part inside the visits is its intersection with the
epoch's visits (the session cache's port experiences:
``event_tables.prepare_data.well_visit_times``).

For the ``ok`` epochs of an event features run, from the decodes it recorded
(each decode's sha256 must be the one the run recorded): one row per epoch in
``Processed-Data/<paths.EXPOSURE_IN_VISITS_DIR>/exposure.parquet``,
with a ``manifest.json`` recording the subset and the session-cache entries the
visits were read from (``prepare_data.session_cache_record``).

Usage::

    uv run python scripts/detector_comparison/measure_exposure_in_well_visits.py \\
        --event-features-run e2331f976fd1 --n-workers 4
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.detector_comparison import inventory_run
from continuum_swr_replay.analysis.detector_comparison.nonlocal_comparison import (
    exposure_in_well_visits,
)
from continuum_swr_replay.analysis.detector_comparison.schema import STATE_SPACE
from continuum_swr_replay.analysis.event_definition import analyzed_time
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    ANALYZED_SUBSET,
)
from continuum_swr_replay.analysis.event_tables.prepare_data import (
    session_cache_record,
    well_visit_times,
)
from continuum_swr_replay.paths import (
    EXPOSURE_IN_VISITS_DIR,
    event_features_root,
    inventory_comparison_root,
    processed_data_dir,
)
from continuum_swr_replay.runs.io import (
    ok_epochs,
    output_path,
    read_manifest,
    write_json,
)
from continuum_swr_replay.runs.provenance import described_commit, produced_by
from continuum_swr_replay.tables.io import read_table

EXPOSURE_ATOL_S = 1e-6
"""How far a rebuilt exposure may be from the stored one (s)."""


def _epoch(job: tuple[str, str, Path]) -> dict[str, Any]:
    """One epoch's exposure and its part inside the visits.

    Raises
    ------
    ValueError
        If the rebuilt exposure is not the inventory run's stored one.
    """
    identifier, expected_sha256, inventory_dir = job
    dot = analyzed_time.read_verified_dot_with_swr_source(
        inventory_run.decode_path(identifier), expected_sha256
    )
    time, speed = inventory_run.speed_trace(dot)
    del dot
    observations, set_aside, counts = (
        read_table(output_path(identifier, inventory_dir, name))
        for name in ("observations", "set_aside", "counts")
    )
    exposure_s, in_visit_s = exposure_in_well_visits(
        observations, set_aside, time, speed, *well_visit_times(identifier)
    )
    (stored_s,) = counts.loc[counts["detector"] == STATE_SPACE, "exposure_s"]
    if not np.isclose(exposure_s, stored_s, rtol=0.0, atol=EXPOSURE_ATOL_S):
        raise ValueError(
            f"{identifier}: the rebuilt exposure ({exposure_s} s) is not the "
            f"inventory run's ({stored_s} s)"
        )
    animal, day, epoch = observations[["animal", "day", "epoch"]].iloc[0]
    return {
        "epoch_id": identifier,
        "animal": str(animal),
        "day": int(day),
        "epoch": str(epoch),
        "decode_sha256": expected_sha256,
        "exposure_s": exposure_s,
        "exposure_in_visit_s": in_visit_s,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--event-features-run", required=True)
    parser.add_argument("--n-workers", type=int, default=1)
    args = parser.parse_args(argv)
    manifest = read_manifest(event_features_root() / args.event_features_run)
    inventory = manifest["settings"]["input_runs"]["inventory"]
    inventory_dir = inventory_comparison_root() / inventory
    jobs = [
        (identifier, manifest["epochs"][identifier]["decode"]["sha256"], inventory_dir)
        for identifier in ok_epochs(manifest)
    ]
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        rows = list(pool.map(_epoch, jobs))
    out_dir = processed_data_dir() / EXPOSURE_IN_VISITS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    tmp = out_dir / "exposure.parquet.tmp"
    table.to_parquet(tmp, index=False)
    tmp.replace(out_dir / "exposure.parquet")
    write_json(
        out_dir / "manifest.json",
        {
            "produced_by": produced_by(__file__),
            "commit": described_commit(),
            "event_features_run": args.event_features_run,
            "inventory_run": inventory,
            "subset": ANALYZED_SUBSET,
            "session_cache": session_cache_record(table["epoch_id"]),
            "n_epochs": len(table),
            "exposure": "the inventory comparison's exposure of the non-local "
            "detector (observed, less excluded time and the events set aside, "
            "inside the still intervals), rebuilt from its stored tables and "
            "equal to its stored exposure_s",
            "exposure_in_visit": "its intersection with the well visits (first "
            "poke-in to final poke-out)",
        },
        indent=1,
        final_newline=True,
    )
    print(
        f"wrote {len(table)} epochs to {out_dir / 'exposure.parquet'}: "
        f"{table['exposure_in_visit_s'].sum() / 3600:.3f} of "
        f"{table['exposure_s'].sum() / 3600:.3f} h inside a visit"
    )


if __name__ == "__main__":
    main()
