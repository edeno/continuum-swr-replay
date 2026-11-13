#!/usr/bin/env python
"""Duration-aware self- and cross-history of the per-bin event rate.

The onset-only per-bin design treats each event as an instant onset: every 15 ms
bin of a stop is at risk, including the bins inside an ongoing event, and
self-/cross-history count past onsets. On simulated data that alone produces a
short-lag "refractory" dip, a self-history burst and two-way cross coupling
(:mod:`continuum_swr_replay.analysis.replay_models.rate.duration_history_simulation`).
This runner refits the history on the at-risk design
(``analysis/replay_models/rate/duration_aware_history.py:build_design``):

- a bin's exposure is the time no event of the response's detector covers (bins
  inside an ongoing event drop out): Kay SWR intervals for the detected-SWR
  population, every non-local event for the three non-local populations;
- history is measured from past events' onsets, their ends, or both.

Two modes:

- ``--mode session`` (default): per session and population, the
  ``MODEL_CROSS_HISTORY_DURATION_RATE`` rungs on ONE frozen cohort:
  ``Mnohist_dur``, ``Mself_onset_dur``, ``Mself_offset_dur``, ``Mself_dur`` for
  every population (``Mself_dur`` is ``M7r_time_histcos_rew`` on the at-risk
  design plus the offset block), and the three cross rungs for the two
  non-local streams (cross = the other stream). Each row carries the lag
  support (expected onsets per lag) and the onset/offset canonical
  correlation. Writes ``duration_aware_history_models{suffix}.pkl``.
- ``--mode pooled``: a session carries about one expected onset per history
  lag, too few to read a kernel. Per animal and population, the sessions'
  designs are stacked and every rung is fit with a session fixed effect
  (``fit_pooled_history``), on the at-risk design and, for comparison, the
  shipped rungs (``Mself``, ``Mselfcross``) on the shipped design. Each fit
  keeps its AIC and each history block's kernel with Wald SEs; each
  (animal, population) keeps its pooled lag support. Writes
  ``duration_aware_history_pooled{suffix}.pkl``.

The design otherwise matches ``fit_per_bin_rate_models.py``: same stop cohort
(``--stop-cohort``, default at-well), bins, TMAX, reward basis with the outcome
shift, excluded time and history window. No CV and no GOF. Each output gets a
``.provenance.json`` sidecar; ``cross_history_models.pkl`` is never touched.

Usage::

    python scripts/replay_models/rate/fit_duration_aware_history_rate.py [--mode session|pooled]
        [--workers N] [--stop-cohort at_well] [--max-per-animal N]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from continuum_swr_replay.analysis.events.analysed_time import require_swr_source
from continuum_swr_replay.analysis.events.cohort_selection import (
    STOP_COHORTS,
    select_stop_cohort,
    stop_cohort_suffix,
)
from continuum_swr_replay.analysis.events.swr_source import swr_source_name
from continuum_swr_replay.analysis.replay_models.covariate_timing import (
    assert_predictor_timing,
)
from continuum_swr_replay.analysis.replay_models.model_specs import (
    MODEL_CROSS_HISTORY_DURATION_RATE,
    MODEL_CROSS_HISTORY_RATE,
)
from continuum_swr_replay.analysis.replay_models.per_bin_design import (
    attach_is_swr_to_onsets,
)
from continuum_swr_replay.analysis.replay_models.rate.duration_aware_history import (
    POPULATION_PLAN,
    attach_event_ends,
    build_design,
    fit_pooled_history,
    lag_support,
    onset_offset_collinearity,
    stack_sessions,
)
from continuum_swr_replay.analysis.replay_models.rate.per_bin_rate_model import (
    fit_per_bin_rate_model,
)
from continuum_swr_replay.paths import model_dir, table_path
from continuum_swr_replay.runs.fit_provenance import (
    record_inputs,
    write_fit_provenance,
)
from continuum_swr_replay.tables.covariate_registry import require_covariate_schema
from continuum_swr_replay.tables.io import read_table
from continuum_swr_replay.tables.session_keys import (
    SESSION_COLUMNS,
    subset_max_per_animal,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

LADDER = MODEL_CROSS_HISTORY_DURATION_RATE
OUTPUT_STEM = {
    "session": "duration_aware_history_models",
    "pooled": "duration_aware_history_pooled",
}


def _fit_one_session(
    keys: tuple, stops_g: pd.DataFrame, onsets_g: pd.DataFrame
) -> tuple[tuple, list[dict], float]:
    """Fit every population's rungs for one session. Top-level for workers."""
    t0 = time.perf_counter()
    rows: list[dict] = []
    for outcome, (event_type, cross_pop) in POPULATION_PLAN.items():
        design, rungs = build_design(keys, stops_g, onsets_g, outcome)
        shared = {
            "outcome": outcome,
            **dict(zip(SESSION_COLUMNS, keys, strict=True)),
            "at_risk_event_type": event_type,
            "cross_population": cross_pop,
            "n_bins_not_at_risk": design.attrs.get("n_bins_not_at_risk"),
            "n_onsets_not_at_risk": design.attrs.get("n_onsets_not_at_risk"),
            "n_bins_excluded": design.attrs.get("n_bins_excluded", 0),
            **{
                f"support_{block}": support.round(3).tolist()
                for block, support in lag_support(design).items()
            },
            **onset_offset_collinearity(design),
        }
        for name in rungs:
            fit = fit_per_bin_rate_model(design, LADDER[name])
            row = {**shared, "model": name, "fit_ok": fit is not None}
            if fit is not None:
                row.update({k: v for k, v in fit.items() if k != "predicted"})
            rows.append(row)
    return keys, rows, time.perf_counter() - t0


def _fit_one_animal(
    animal: str, outcome: str, sessions: list
) -> tuple[tuple, dict, float]:
    """Fit one animal's stacked sessions for one population. For workers."""
    t0 = time.perf_counter()
    rec: dict = {"animal": animal, "outcome": outcome, "n_sessions": len(sessions)}
    design, rungs, _ = stack_sessions(sessions, outcome, duration_aware=True)
    rec |= {
        "n_bins": int(len(design)),
        "n_events": int(design["count"].sum()),
        "support": {k: v.tolist() for k, v in lag_support(design).items()},
        "duration_aware": {
            name: fit_pooled_history(design, LADDER[name]) for name in rungs
        },
    }
    del design
    shipped, _, _ = stack_sessions(sessions, outcome, duration_aware=False)
    names = ["Mself"] + (["Mselfcross"] if POPULATION_PLAN[outcome][1] else [])
    rec["shipped_support"] = {k: v.tolist() for k, v in lag_support(shipped).items()}
    rec["shipped"] = {
        name: fit_pooled_history(shipped, MODEL_CROSS_HISTORY_RATE[name])
        for name in names
    }
    return (animal, outcome), rec, time.perf_counter() - t0


def _run(fn, items: list[tuple], workers: int) -> list:
    """``fn`` over ``items``, its results in the order of ``items`` whatever
    order the workers finish in."""
    results: list = [None] * len(items)
    done = 0
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(fn, *item): i for i, item in enumerate(items)}
            for fut in as_completed(futures):
                keys, out, elapsed = fut.result()
                done += 1
                results[futures[fut]] = out
                logger.info("  [%d/%d] %s in %.1fs", done, len(items), keys, elapsed)
    else:
        for i, item in enumerate(items):
            keys, out, elapsed = fn(*item)
            done += 1
            results[i] = out
            logger.info("  [%d/%d] %s in %.1fs", done, len(items), keys, elapsed)
    return results


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=sorted(OUTPUT_STEM), default="session")
    p.add_argument("--max-per-animal", type=int, default=None, help="smoke subset")
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
    if args.max_per_animal is not None:
        stop = subset_max_per_animal(stop, args.max_per_animal)
    onset_groups = dict(tuple(onsets.groupby(list(SESSION_COLUMNS), sort=False)))
    sessions = [
        (keys, stops_g, onset_groups[keys])
        for keys, stops_g in stop.groupby(list(SESSION_COLUMNS), sort=False)
        if keys in onset_groups
    ]
    logger.info(
        "mode=%s: %d sessions, %d worker(s)", args.mode, len(sessions), args.workers
    )
    t0 = time.perf_counter()
    if args.mode == "session":
        rows = _run(_fit_one_session, sessions, args.workers)
        result = pd.DataFrame([r for session_rows in rows for r in session_rows])
        result["stop_cohort"] = args.stop_cohort
    else:
        by_animal: dict[str, list] = {}
        for item in sessions:
            by_animal.setdefault(item[0][0], []).append(item)
        jobs = [
            (animal, outcome, by_animal[animal])
            for animal in sorted(by_animal)
            for outcome in POPULATION_PLAN
        ]
        result = {
            "stop_cohort": args.stop_cohort,
            "results": _run(_fit_one_animal, jobs, args.workers),
        }

    out_dir = model_dir("per_bin_rate")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = stop_cohort_suffix(args.stop_cohort, smoke=args.max_per_animal is not None)
    out = out_dir / f"{OUTPUT_STEM[args.mode]}{suffix}.pkl"
    pd.to_pickle(result, out)
    write_fit_provenance(
        out,
        fit_inputs,
        "poisson_glm",
        [sys.argv[0], *argv],
        extra={
            "stop_cohort": args.stop_cohort,
            "swr_source": swr_source_name(),
            "design": "at-risk time + onset/offset history (duration-aware)",
            "mode": args.mode,
        },
    )
    logger.info("Saved %s in %.1fs", out, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
