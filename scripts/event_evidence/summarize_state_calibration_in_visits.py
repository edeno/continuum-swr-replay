"""Summarize the held-out state calibration over the immobile windows inside the well visits.

``run_state_evidence.py`` scored random 10-bin calibration windows with held-out
tetrodes over all of each epoch's immobile time, before the analyses were
restricted to the time inside a well visit. This runner reads its stored
per-epoch windows, keeps the immobile calibration windows that start inside a
well visit (``state_evidence.calibration_windows_in_well_visits``, each epoch's
visits from ``event_tables.prepare_data.well_visit_times``), and summarizes
them with the module's own statistics (``state_evidence.calibration_statistics``
over ``IN_VISIT_CALIBRATION_METRICS``: the animal-level estimate and its nested
bootstrap interval per probability bin). Nothing is decoded or scored again.

Reads every epoch the run's ``manifest.json`` marks ``ok``. Writes to
``Processed-Data/state_calibration_in_visits/`` (or ``--output-dir``):

- ``calibration_bins.csv`` and ``calibration_trends.csv``: the statistics;
- ``manifest.json``: the run read (its folder and the sha256 of its manifest),
  the subset (``cohort_selection.ANALYZED_SUBSET``), the session-cache entries
  the visits were read from (``prepare_data.session_cache_record``) and each
  epoch's immobile calibration windows kept and left out, and the kept ones
  with no scored spike for each metric (per spike, those have no value).

Usage::

    uv run python scripts/event_evidence/summarize_state_calibration_in_visits.py
    uv run python scripts/event_evidence/summarize_state_calibration_in_visits.py \\
        --output-dir /tmp/check
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    ANALYZED_SUBSET,
)
from continuum_swr_replay.analysis.event_evidence.state_evidence import (
    IN_VISIT_CALIBRATION_METRICS,
    calibration_statistics,
    calibration_windows_in_well_visits,
)
from continuum_swr_replay.analysis.event_tables.prepare_data import (
    session_cache_record,
    well_visit_times,
)
from continuum_swr_replay.paths import (
    STATE_CALIBRATION_IN_VISITS_DIR,
    STATE_EVIDENCE_DIR,
    STATE_EVIDENCE_RUN,
    processed_data_dir,
)
from continuum_swr_replay.runs.io import (
    ok_epochs,
    read_manifest,
    sha256_file,
    write_json,
)
from continuum_swr_replay.runs.provenance import described_commit, produced_by

COLUMNS = [
    "epoch_id",
    "animal",
    "kind",
    "regime",
    "start_time",
    *(f"p_{state}" for state in IN_VISIT_CALIBRATION_METRICS),
    *sorted({m for metrics in IN_VISIT_CALIBRATION_METRICS.values() for m in metrics}),
]
"""The stored windows' columns read."""


def epoch_windows(run_dir: Path, epoch_id: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """One epoch's immobile calibration windows inside the well visits, how many
    were kept and left out, and how many of the kept ones have no scored spike
    for each metric (they enter no statistic)."""
    windows = pd.read_parquet(
        run_dir / "per_epoch" / f"{epoch_id}_windows.parquet", columns=COLUMNS
    )
    kept = calibration_windows_in_well_visits(windows, *well_visit_times(epoch_id))
    n_immobile = int(
        ((windows["kind"] == "calib") & (windows["regime"] == "immobile")).sum()
    )
    counts = {"n_kept": len(kept), "n_left_out": n_immobile - len(kept)}
    for metric in sorted({m for v in IN_VISIT_CALIBRATION_METRICS.values() for m in v}):
        counts[f"n_without_scored_spike_{metric}"] = int(kept[metric].isna().sum())
    return kept, counts


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-dir",
        type=Path,
        help=f"a run_state_evidence.py output (default: Processed-Data/"
        f"{STATE_EVIDENCE_RUN}/{STATE_EVIDENCE_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=f"default: Processed-Data/{STATE_CALIBRATION_IN_VISITS_DIR}",
    )
    args = parser.parse_args(argv)
    run_dir = (
        args.run_dir or processed_data_dir() / STATE_EVIDENCE_RUN / STATE_EVIDENCE_DIR
    )
    epochs = ok_epochs(read_manifest(run_dir))
    if not epochs:
        raise SystemExit(f"{run_dir}: no epoch the manifest marks ok")
    tables, counts = zip(*(epoch_windows(run_dir, e) for e in epochs), strict=True)
    windows = pd.concat(tables, ignore_index=True)
    bins, trends = calibration_statistics(
        windows, metrics=IN_VISIT_CALIBRATION_METRICS, regimes=("immobile",)
    )
    out_dir = args.output_dir or processed_data_dir() / STATE_CALIBRATION_IN_VISITS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    bins.to_csv(out_dir / "calibration_bins.csv", index=False)
    trends.to_csv(out_dir / "calibration_trends.csv", index=False)
    write_json(
        out_dir / "manifest.json",
        {
            "produced_by": produced_by(__file__),
            "commit": described_commit(),
            "state_evidence_run": {
                "path": str(run_dir),
                "manifest_sha256": sha256_file(run_dir / "manifest.json"),
            },
            "subset": ANALYZED_SUBSET,
            "session_cache": session_cache_record(epochs),
            "n_epochs": len(epochs),
            "windows": "immobile calibration windows whose first sample is inside "
            "a well visit",
            "metrics": {k: list(v) for k, v in IN_VISIT_CALIBRATION_METRICS.items()},
            "epochs": dict(zip(epochs, counts, strict=True)),
        },
        indent=1,
        final_newline=True,
    )
    totals = pd.DataFrame(counts).sum()
    print(f"{len(epochs)} epochs -> {out_dir}")
    print(totals.to_string())


if __name__ == "__main__":
    main()
