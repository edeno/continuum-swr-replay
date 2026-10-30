#!/usr/bin/env python
"""Ridge cross-validated comparison of self vs self+cross history, per animal.

The per-session cross rungs of ``fit_duration_aware_history_rate.py`` mostly
fail to converge (about one expected cross-stream onset per session in the
150 ms window), and one animal's pooled cross fit does too. This runner asks
the same question on held-out data:

- per animal and non-local stream, the sessions' duration-aware designs are
  stacked with a session effect, on ``MODEL_CROSS_HISTORY_LONG_RATE``'s cohort
  (the cross block reaches 1 s after the other stream's onset and end);
- ``Mself_long_dur`` and ``Mselfcross_long_dur`` are fit with a ridge prior on
  the history coefficients, its strength chosen by inner cross-validation;
- their outer held-out log-likelihoods are compared over five folds of whole
  stops (``analysis/replay/ridge_history_cv.py``).

A session with fewer than ``MIN_SESSION_EVENTS`` response onsets is left out
(its unpenalised session effect has no estimate in a training set without
events); the count is recorded. Writes one pickle per run with the
per-(animal, stream) records and a ``.provenance.json`` sidecar.

Usage::

    python scripts/rate/fit_cross_history_ridge_cv.py --out-dir DIR
        [--animals wilbur j16] [--outcomes non_local_swr non_local_no_swr]
        [--workers N] [--stop-cohort at_well]
"""

from __future__ import annotations

import argparse
import logging
import resource
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.events.analysed_time import require_swr_source
from continuum_swr_replay.analysis.events.cohort_selection import (
    STOP_COHORTS,
    select_stop_cohort,
)
from continuum_swr_replay.analysis.events.swr_source import swr_source_name
from continuum_swr_replay.analysis.replay import attach_is_swr_to_onsets
from continuum_swr_replay.analysis.replay.covariate_timing import (
    assert_predictor_timing,
)
from continuum_swr_replay.analysis.replay.duration_aware_history import (
    attach_event_ends,
    build_design,
    lag_support,
)
from continuum_swr_replay.analysis.replay.fit_provenance import (
    record_inputs,
    write_fit_provenance,
)
from continuum_swr_replay.analysis.replay.model_specs import (
    MODEL_CROSS_HISTORY_LONG_RATE,
)
from continuum_swr_replay.analysis.replay.ridge_history_cv import (
    FOLD_SEED,
    MIN_SESSION_EVENTS,
    N_FOLDS,
    SIGMA_GRID,
    compare_cross_history,
)
from continuum_swr_replay.paths import table_path
from continuum_swr_replay.tables.covariate_registry import require_covariate_schema
from continuum_swr_replay.tables.io import read_table
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

LADDER = MODEL_CROSS_HISTORY_LONG_RATE
OUTCOMES = ("non_local_swr", "non_local_no_swr")
_LAG_PREFIXES = ("onset_lag", "offset_lag", "cross_lag", "cross_offset_lag")


def _fit_one(animal: str, outcome: str, sessions: list) -> tuple[tuple, dict, float]:
    """Stack one animal's sessions for one stream and compare. For workers."""
    t0 = time.perf_counter()
    frames, dropped = [], []
    for keys, stops_g, onsets_g in sessions:
        design, _ = build_design(keys, stops_g, onsets_g, outcome, ladder=LADDER)
        if design["count"].sum() < MIN_SESSION_EVENTS:
            dropped.append("_".join(map(str, keys)))
            continue
        design["session"] = "_".join(map(str, keys))
        frames.append(design)
    design = pd.concat(frames, ignore_index=True)
    support = {k: v.tolist() for k, v in lag_support(design).items()}
    design = design.drop(
        columns=[c for c in design.columns if c.startswith(_LAG_PREFIXES)]
    )
    rec = {
        "animal": animal,
        "outcome": outcome,
        "n_sessions_dropped": len(dropped),
        "sessions_dropped": dropped,
        "support": support,
        **compare_cross_history(design, LADDER),
        "max_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
    }
    return (animal, outcome), rec, time.perf_counter() - t0


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--animals", nargs="*", default=None)
    p.add_argument("--outcomes", nargs="*", choices=OUTCOMES, default=list(OUTCOMES))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--stop-cohort", choices=STOP_COHORTS, default="at_well")
    argv = sys.argv[1:] if argv is None else list(argv)
    args = p.parse_args(argv)

    for spec in LADDER.values():
        assert_predictor_timing(c for c in spec.design_columns() if c != "count")

    inputs = {
        name: table_path(name)
        for name in ("stop_table", "event_onset_table", "event_table")
    }
    stop = read_table(inputs["stop_table"])
    onset_table = read_table(inputs["event_onset_table"])
    event_table = read_table(inputs["event_table"])
    require_covariate_schema(stop, table="stop")
    require_covariate_schema(onset_table, table="event_onset")
    require_covariate_schema(event_table, table="event")
    require_swr_source(stop, what=str(inputs["stop_table"]))
    require_swr_source(onset_table, what=str(inputs["event_onset_table"]))
    require_swr_source(event_table, what=str(inputs["event_table"]))
    fit_inputs = record_inputs(inputs)
    onsets = attach_event_ends(
        attach_is_swr_to_onsets(onset_table, event_table), event_table
    )
    stop, cohort_report = select_stop_cohort(stop, cohort=args.stop_cohort)
    logger.info(
        "stop cohort = %s: %d -> %d stops",
        cohort_report["cohort"],
        cohort_report["n_stops_before"],
        cohort_report["n_stops_after"],
    )
    if args.animals:
        stop = stop[stop["animal"].isin(args.animals)]
    onset_groups = dict(tuple(onsets.groupby(list(SESSION_COLUMNS), sort=False)))
    by_animal: dict[str, list] = {}
    for keys, stops_g in stop.groupby(list(SESSION_COLUMNS), sort=True):
        if keys in onset_groups:
            by_animal.setdefault(keys[0], []).append(
                (keys, stops_g, onset_groups[keys])
            )
    jobs = [
        (animal, outcome, by_animal[animal])
        for animal in sorted(by_animal)
        for outcome in args.outcomes
    ]
    logger.info("%d (animal, stream) jobs, %d worker(s)", len(jobs), args.workers)

    t0 = time.perf_counter()
    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_fit_one, *job) for job in jobs]
            for fut in as_completed(futures):
                keys, rec, elapsed = fut.result()
                results.append(rec)
                logger.info("  %s in %.0fs", keys, elapsed)
    else:
        for job in jobs:
            keys, rec, elapsed = _fit_one(*job)
            results.append(rec)
            logger.info("  %s in %.0fs", keys, elapsed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "cross_history_ridge_cv.pkl"
    pd.to_pickle({"stop_cohort": args.stop_cohort, "results": results}, out)
    write_fit_provenance(
        out,
        fit_inputs,
        "ridge_poisson_glm",
        [sys.argv[0], *argv],
        extra={
            "stop_cohort": args.stop_cohort,
            "swr_source": swr_source_name(),
            "design": "at-risk time + onset/offset history, cross block to 1 s",
            "sigma_grid": list(SIGMA_GRID),
            "n_folds": N_FOLDS,
            "fold_seed": FOLD_SEED,
            "min_session_events": MIN_SESSION_EVENTS,
        },
    )
    logger.info("Saved %s in %.1fs", out, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
