#!/usr/bin/env python
"""Compare the replay-detection inventories, one epoch at a time.

Nine inventories of each analysed epoch are read through their adapters
(``detector_comparison.adapters``) and compared on one substrate, with
per-detector denominators. Design:
``docs/design/replay_detection_comparison/03-inventory-comparison.md``.

The roster (:data:`ROSTER`):

- the SWR detectors ``kay``, ``karlsson``, ``yu`` and ``zugaro`` (the SWR
  source run, ``swr_source.SWR_INVENTORY_RUN_ID``) and ``carey``
  (``--carey-run-id``), the five SWR definitions;
- ``mua`` and ``state_space``, from the canonical decode frame;
- ``memoryless`` (cohort ``published``) and ``memoryless_criteria`` (cohort
  ``replay_criteria``), from the memoryless run (``--memoryless-run-id``).

Per epoch:

1. **Decode.** The canonical decode's ``data_over_time`` pickle is read once:
   its drops (``analysed_time.artifact_event_drops``) from the frame as saved,
   and the frame through the SWR source (``analysed_time.with_swr_source``). Its
   sha256 is recorded, and its first 12 characters name the decode in the
   ``mua`` and ``state_space`` tables.
2. **Inventories.** Each adapter returns its inventory, its observation table
   and the events the analysed-time rule set aside. A run with no usable
   record for the epoch reads unavailable, never empty.
3. **Speed.** One trace, the frame's ``head_speed`` on its grid, with one
   threshold (``DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC``) and one largest
   step (:data:`MAX_GAP_S`), gives the still intervals, where the speed is
   known (``exclusion_by_reason``'s ``speed_known``) and the samples that were
   not still (the recount's ``not_still_s``). The grid must step by
   :data:`DECODE_SAMPLE_S`. Every detector's epoch is the grid's first to
   last time, which must be the position span the SWR source run cut its band
   to (its record's ``position_span_s``); otherwise the epoch fails, with that
   reason, and writes nothing.
4. **Coverage.** ``coverage_from_observations``; each set-aside event's span
   leaves its own detector's coverage (``set_aside_events``); the
   speed-eligible events (``combine_inventories``) that bridge their own
   detector's unobserved time are set aside too (``remove_bridging_events``,
   reason ``bridging``).
5. **Comparison.** Every overlap pair, flagged; directional agreement and time
   coverage over the nine; the state-space fragment recount at every
   ``FRAGMENT_GAPS_S``, each followed by its pairs and directional agreement;
   the three cells (:data:`CELLS`); the overlap components, the exactly-one
   cell and no-detector time over the eight that leave ``memoryless_criteria``
   out (every published event is also a criteria event); the SWR label
   matrix of the three sequence cohorts under the five SWR definitions; and
   excluded time by reason for every pair, for ``all`` nine and for each matrix
   population (:data:`COMPARISONS`). Any second outside a comparison's shared exposure
   with no reason fails the epoch.

Each epoch writes, under
``Processed-Data/replay_detection_comparison/inventory/<run_id>/``, one
``{epoch_id}_<table>.parquet`` per table of :data:`TABLE_DTYPES`, as one set:
every table is made before any is written, and a failed epoch keeps none.
``manifest.json`` beside them holds the run's settings (the input runs, the
SWR source, the roster and its cohorts, the comparisons, the recount and the
speed rule, and why MUA and state-space native counts equal their eligible
counts) and one record per epoch: its status, the decode's sha256, the counts,
the unavailable detectors, each output's path and sha256, its wall clock and
the process's peak RSS so far (``process_peak_rss_gib``, cumulative over the
epochs before it). Only this process writes it, atomically, after
each epoch. A rerun under the same run id skips the epochs already ``ok``
unless ``--overwrite`` is given, and refuses settings other than the ones the
manifest was written with. The epochs are the epoch register's analysed ones,
narrowed by ``--animal``, ``--date`` and ``--epoch``, or the one
``--epoch-id``, run one at a time in this process, on the CPU.

The roster, the comparisons, the grid and the tables' layout are
``detector_comparison.inventory_run``'s, which every reader of a run shares;
the comparison of one epoch is ``detector_comparison.epoch_comparison``'s.

Usage:
    python scripts/detector_comparison/run_inventory_comparison.py --carey-run-id <run id> \\
        --memoryless-run-id e2331f976fd1-seed20260922 \\
        --epoch-id chimi20200225_02_r1_HPC
"""

import argparse
import resource
import sys
import time as clock
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.detector_comparison.adapters import (
    NATIVE_IS_ELIGIBLE,
)
from continuum_swr_replay.analysis.detector_comparison.epoch_comparison import (
    compare_epoch,
)
from continuum_swr_replay.analysis.detector_comparison.inventory_run import (
    CELL_DETECTORS,
    CELLS,
    COMPARISONS,
    DETECTORS,
    MAX_GAP_S,
    ROSTER,
    SEQUENCE_COHORTS,
    SWR_DEFINITIONS,
    decode_path,
    output_paths,
    speed_trace,
    stamped_tables,
)
from continuum_swr_replay.analysis.detector_comparison.recount import (
    FRAGMENT_GAPS_S,
)
from continuum_swr_replay.analysis.detector_comparison.schema import (
    STATE_SPACE,
)
from continuum_swr_replay.analysis.event_definition import (
    analysed_time,
    swr_source,
)
from continuum_swr_replay.parameters import (
    DECODE_SAMPLE_S,
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    ROOT_DIR,
)
from continuum_swr_replay.paths import (
    carey_root,
    inventory_comparison_root,
    memoryless_root,
)
from continuum_swr_replay.runs.batch import RunManifest, run_lock
from continuum_swr_replay.runs.cli import (
    add_epoch_filters,
    parse_run_id,
    select_analysed_epochs,
)
from continuum_swr_replay.runs.device import default_jax_platforms
from continuum_swr_replay.runs.io import (
    json_ready,
    replace_outputs,
    sha256_file,
)
from continuum_swr_replay.runs.provenance import produced_by, provenance_record
from continuum_swr_replay.runs.provenance import run_id as run_id_of

PRODUCED_BY = produced_by(__file__)
DESIGN_SPEC = "docs/design/replay_detection_comparison/03-inventory-comparison.md"

SPEED_TRACE = "head_speed of the canonical decode frame, on its grid"
BOUNDS_TOLERANCE_S = DECODE_SAMPLE_S / 2
"""How far the decode grid's first or last time may lie from the SWR run's
position span: less than half a sample, so both name the same samples."""


class _EpochRefused(Exception):
    """The epoch cannot be compared; the message says why."""


def bounds_mismatch(identifier: str, bounds: tuple[float, float]) -> str | None:
    """Why the decode grid's first and last time are not the epoch's, or None.

    They must be the position span the SWR source run cut its band to (its
    record's ``position_span_s``), within :data:`BOUNDS_TOLERANCE_S`: every
    detector is compared over that one span.
    """
    span = swr_source.ripple_power_inputs(identifier)["position_span_s"]
    if np.allclose(bounds, span, rtol=0.0, atol=BOUNDS_TOLERANCE_S):
        return None
    return (
        f"The decode grid spans {bounds[0]!r} to {bounds[1]!r} s, not the position "
        f"span {span[0]!r} to {span[1]!r} s SWR run "
        f"{swr_source.SWR_INVENTORY_RUN_ID} cut its band to; the detectors would "
        "not share one epoch."
    )


def run_epoch(
    identifier: str,
    *,
    output_dir: Path,
    carey_dir: Path,
    memoryless_dir: Path,
    provenance: Mapping[str, Any],
) -> dict:
    """Compare one epoch's inventories and write its tables.

    Returns
    -------
    dict
        The epoch's manifest record, in strict JSON types: ``status`` ``"ok"``,
        or ``"failed"`` with a ``reason`` (the decode grid does not span the
        position span the SWR run cut to) or the ``error`` and its
        ``traceback``, its outputs removed.
    """
    started = clock.perf_counter()
    stamp = {
        "produced_by": PRODUCED_BY,
        "run_id": output_dir.name,
        "epoch_id": identifier,
        "commit": str(provenance["commit"]),
    }
    record: dict[str, Any]
    try:
        path = decode_path(identifier)
        decode_sha256 = sha256_file(path)
        saved = pd.read_pickle(path)
        drops = analysed_time.artifact_event_drops(saved)
        dot = analysed_time.with_swr_source(saved)
        del saved
        time, _ = speed_trace(dot)
        bounds = (float(time[0]), float(time[-1]))
        reason = bounds_mismatch(identifier, bounds)
        if reason is not None:
            replace_outputs(output_paths(identifier, output_dir), {})
            raise _EpochRefused(reason)
        tables = compare_epoch(
            dot,
            drops,
            decode_run_id=decode_sha256[:12],
            carey_dir=carey_dir,
            memoryless_dir=memoryless_dir,
        )
        observations = tables["observations"]
        observed = observations.groupby("detector").available.any()
        counts = tables["counts"].set_index("detector")
        record = {
            "status": "ok",
            "decode": {"path": str(path), "sha256": decode_sha256},
            "epoch_bounds_s": list(bounds),
            "counts": {
                detector: counts.loc[
                    detector, ["n_native", "n_eligible", "n_moving", "n_set_aside"]
                ].to_dict()
                for detector in DETECTORS
            },
            "unavailable": {
                detector: observations.loc[
                    observations.detector == detector, "reason"
                ].iloc[0]
                for detector in DETECTORS
                if not observed[detector]
            },
            "outputs": replace_outputs(
                output_paths(identifier, output_dir), stamped_tables(tables, stamp)
            ),
        }
    except _EpochRefused as refused:
        record = {
            "status": "failed",
            "reason": str(refused),
            "decode": {"path": str(path), "sha256": decode_sha256},
            "epoch_bounds_s": list(bounds),
        }
    except Exception as error:  # noqa: BLE001 - recorded; the run goes on
        replace_outputs(output_paths(identifier, output_dir), {})
        record = {
            "status": "failed",
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }
    record["provenance"] = dict(provenance)
    record["wall_clock_s"] = clock.perf_counter() - started
    # the process's peak resident set over its life so far, every earlier
    # epoch included: not this epoch's own (KiB on Linux)
    record["process_peak_rss_gib"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    )
    ready: dict = json_ready(record)
    return ready


def summary(record: dict) -> str:
    """One epoch's outcome, as the console reports it."""
    if record["status"] != "ok":
        return f"{record['status']}: {record.get('reason') or record['error']}"
    eligible = ", ".join(
        f"{detector} {counts['n_eligible']}"
        for detector, counts in record["counts"].items()
    )
    unavailable = ", ".join(record["unavailable"]) or "none"
    return (
        f"ok: eligible {eligible}; unavailable: {unavailable}; "
        f"{record['wall_clock_s']:.0f} s, process peak RSS "
        f"{record['process_peak_rss_gib']:.1f} GiB"
    )


def run_settings(carey_run_id: str, memoryless_run_id: str) -> dict:
    """What every epoch of a run shares; a rerun under its run id must match."""
    settings: dict = json_ready(
        {
            "input_runs": {
                "swr": swr_source.SWR_INVENTORY_RUN_ID,
                "carey": carey_run_id,
                "memoryless": memoryless_run_id,
                "artifacts": swr_source.ARTIFACT_INTERVALS_RUN_ID,
            },
            "swr_source": swr_source.swr_source_name(),
            "roster": ROSTER,
            "swr_definitions": SWR_DEFINITIONS,
            "swr_cohorts": SEQUENCE_COHORTS,
            "cells": {
                name: {"sources": sources, "against": against}
                for name, (sources, against) in CELLS.items()
            },
            "no_detector_time": CELL_DETECTORS,
            "comparisons": COMPARISONS,
            "recount": {
                "detector": STATE_SPACE,
                "gaps_s": FRAGMENT_GAPS_S,
                "sample_interval_s": DECODE_SAMPLE_S,
            },
            "speed": {
                "trace": SPEED_TRACE,
                "threshold_cm_per_s": DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
                "max_gap_s": MAX_GAP_S,
            },
            "epoch_bounds": {
                "rule": "the decode grid's first and last time, which must be "
                "the SWR run's position_span_s",
                "tolerance_s": BOUNDS_TOLERANCE_S,
            },
            "peak_rss": "process_peak_rss_gib: the process's peak so far, "
            "cumulative over the epochs run before",
            "decode": decode_path("{epoch_id}").name + ", read through the SWR source",
            "native_is_eligible": NATIVE_IS_ELIGIBLE,
        }
    )
    return settings


def main(argv: list[str] | None = None) -> None:
    # CPU only: the package imports JAX, which this comparison never needs
    default_jax_platforms("cpu")
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--carey-run-id",
        required=True,
        type=parse_run_id,
        help="The Carey run folder's name.",
    )
    parser.add_argument(
        "--memoryless-run-id",
        required=True,
        type=parse_run_id,
        help="The memoryless run folder's name.",
    )
    add_epoch_filters(parser, epoch_id=True)
    parser.add_argument(
        "--overwrite", action="store_true", help="Redo epochs already written."
    )
    parser.add_argument(
        "--run-id",
        type=parse_run_id,
        help="The run's folder: letters, digits, '.', '_' and '-'. Defaults to the "
        "first 12 characters of git rev-parse HEAD, with -dirty when tracked "
        "files have uncommitted changes.",
    )
    args = parser.parse_args(argv)
    if args.epoch_id is not None and any(
        value is not None for value in (args.animal, args.date, args.epoch)
    ):
        parser.error("--epoch-id names one epoch; drop --animal, --date and --epoch.")

    provenance = provenance_record(ROOT_DIR)
    run_id = run_id_of(provenance, args.run_id)
    output_dir = inventory_comparison_root() / run_id
    with run_lock(output_dir):
        run(args, provenance, output_dir)


def run(args: argparse.Namespace, provenance: dict, output_dir: Path) -> None:
    """``main``'s run, under the run folder's lock."""
    carey_dir = carey_root() / args.carey_run_id
    memoryless_dir = memoryless_root() / args.memoryless_run_id
    for option, run_dir in (
        ("--carey-run-id", carey_dir),
        ("--memoryless-run-id", memoryless_dir),
    ):
        # a misspelt run would read unavailable in every epoch
        if not (run_dir / "manifest.json").exists():
            sys.exit(f"{option}: no run manifest at {run_dir}.")
    settings = run_settings(args.carey_run_id, args.memoryless_run_id)
    epochs = select_analysed_epochs(args)
    if not epochs:
        sys.exit("No analysed epochs match --animal, --date and --epoch.")

    manifest_path = output_dir / "manifest.json"
    manifest = RunManifest.resume(
        manifest_path,
        {
            "produced_by": PRODUCED_BY,
            "run_id": output_dir.name,
            "design_spec": DESIGN_SPEC,
            "settings": settings,
        },
    )
    to_run = manifest.to_redo(epochs, args.overwrite)
    manifest.write()
    print(f"{len(to_run)} epochs to run, one at a time", flush=True)

    failed = []
    for number, identifier in enumerate(to_run, start=1):
        record = run_epoch(
            identifier,
            output_dir=output_dir,
            carey_dir=carey_dir,
            memoryless_dir=memoryless_dir,
            provenance=provenance,
        )
        manifest.records[identifier] = record
        print(f"[{number}/{len(to_run)}] {identifier}: {summary(record)}", flush=True)
        if record["status"] != "ok":
            failed.append(identifier)
        manifest.write()
    print(f"manifest -> {manifest_path}", flush=True)
    if failed:
        sys.exit(
            f"{len(failed)} epochs failed: {', '.join(failed)}; see {manifest_path}"
        )


if __name__ == "__main__":
    main()
