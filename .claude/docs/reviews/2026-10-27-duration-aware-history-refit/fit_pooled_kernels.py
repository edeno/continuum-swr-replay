"""History kernels pooled within animal: duration-aware vs shipped design (read-only).

A real session carries about 1 expected onset per history lag (``lag_support``),
so a per-session kernel is not measurable. Here each animal's sessions are
pooled: the per-session designs of
``scripts/rate/fit_duration_aware_history_rate.py:build_design`` are
concatenated and fit with a session fixed effect (``C(session)``; the time
spline, reward basis and history are shared within the animal), for

- the duration-aware rungs (at-risk design): no history, self onset-only,
  offset-only, both; for the two non-local streams also + cross onset-only,
  offset-only, both;
- the shipped rungs on the shipped design (every bin at risk, onset history):
  ``Mself`` and, for the non-local streams, ``Mselfcross``.

Per fit: AIC, parameter count, convergence, and each history block's kernel
(log rate ratio per lag, Wald SE from the coefficient covariance) with the
pooled support (expected onsets per lag). Writes
``pooled_kernels_{cohort}.json``.

Usage::

    CSR_PROCESSED_DATA_DIR=/cumulus/edeno/continuum-swr-replay/Processed-Data \\
    PYTHONPATH=. uv run python \\
        .claude/docs/reviews/2026-10-27-duration-aware-history-refit/fit_pooled_kernels.py \\
        --cohort at_well --workers 20
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from continuum_swr_replay.analysis.events.cohort_selection import (
    STOP_COHORTS,
    select_stop_cohort,
)
from continuum_swr_replay.analysis.replay import attach_is_swr_to_onsets
from continuum_swr_replay.analysis.replay.design_constants import (
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
)
from continuum_swr_replay.analysis.replay.duration_history_simulation import (
    HISTORY_BLOCKS,
    lag_support,
)
from continuum_swr_replay.analysis.replay.model_specs import (
    MODEL_CROSS_HISTORY_DURATION_RATE,
    MODEL_CROSS_HISTORY_RATE,
)
from continuum_swr_replay.analysis.replay.per_bin_design import (
    raised_cosine_history_basis,
)
from continuum_swr_replay.paths import table_path
from continuum_swr_replay.tables.io import read_table
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS
from scripts.rate.fit_duration_aware_history_rate import (
    POPULATION_PLAN,
    attach_event_ends,
    build_design,
)

OUT = Path(__file__).resolve().parent
BASIS = raised_cosine_history_basis(N_HISTORY_LAGS, N_HISTORY_BASIS)


def _pooled_design(sessions: list, outcome: str, duration_aware: bool):
    frames = []
    rungs = None
    for keys, stops_g, onsets_g in sessions:
        design, rungs = build_design(
            keys, stops_g, onsets_g, outcome, duration_aware=duration_aware
        )
        design["session"] = "_".join(map(str, keys))
        frames.append(design)
    return pd.concat(frames, ignore_index=True), rungs


def _fit(design: pd.DataFrame, spec) -> dict:
    cols = [c for c in spec.design_columns() if c != "count"]
    formula = spec.formula() + " + C(session)"
    df = design[["count", "exposure", "session", *cols]]
    t0 = time.perf_counter()
    try:
        res = smf.glm(
            formula,
            data=df,
            family=sm.families.Poisson(),
            offset=np.log(df["exposure"].to_numpy(dtype=float)),
        ).fit()
    except Exception as exc:  # noqa: BLE001 -- recorded, the job goes on
        return {"ok": False, "error": repr(exc)}
    out = {
        "ok": True,
        "aic": float(res.aic),
        "llf": float(res.llf),
        "n_params": int(len(res.params)),
        "converged": bool(res.converged),
        "seconds": round(time.perf_counter() - t0, 1),
        "kernels": {},
    }
    cov = res.cov_params()
    for block, (prefix, _, _) in HISTORY_BLOCKS.items():
        names = [f"{prefix}{k}" for k in range(N_HISTORY_BASIS)]
        if not all(n in res.params.index for n in names):
            continue
        beta = res.params[names].to_numpy()
        sigma = cov.loc[names, names].to_numpy()
        out["kernels"][block] = {
            "log_rr": (beta @ BASIS).tolist(),
            "se": np.sqrt(np.einsum("kj,kl,lj->j", BASIS, sigma, BASIS)).tolist(),
            "coef": beta.tolist(),
        }
    return out


def _job(animal: str, outcome: str, sessions: list) -> dict:
    warnings.filterwarnings("ignore")
    rec: dict = {"animal": animal, "outcome": outcome, "n_sessions": len(sessions)}
    design, rungs = _pooled_design(sessions, outcome, duration_aware=True)
    rec["n_bins"] = int(len(design))
    rec["n_events"] = int(design["count"].sum())
    rec["support"] = {k: v.tolist() for k, v in lag_support(design).items()}
    rec["duration_aware"] = {
        name: _fit(design, MODEL_CROSS_HISTORY_DURATION_RATE[name]) for name in rungs
    }
    del design
    shipped, _ = _pooled_design(sessions, outcome, duration_aware=False)
    rec["shipped_support"] = {k: v.tolist() for k, v in lag_support(shipped).items()}
    names = ["Mself"] + (["Mselfcross"] if POPULATION_PLAN[outcome][1] else [])
    rec["shipped"] = {
        name: _fit(shipped, MODEL_CROSS_HISTORY_RATE[name]) for name in names
    }
    return rec


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", choices=STOP_COHORTS, default="at_well")
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--animals", nargs="*", default=None, help="smoke subset")
    p.add_argument("--outcomes", nargs="*", default=None, help="smoke subset")
    args = p.parse_args()
    smoke = args.animals is not None or args.outcomes is not None
    stop = read_table(table_path("stop_table"))
    onset_table = read_table(table_path("event_onset_table"))
    event_table = read_table(table_path("event_table"))
    onsets = attach_event_ends(
        attach_is_swr_to_onsets(onset_table, event_table), event_table
    )
    stop, _ = select_stop_cohort(stop, cohort=args.cohort)
    onset_groups = dict(tuple(onsets.groupby(list(SESSION_COLUMNS), sort=False)))
    by_animal: dict[str, list] = {}
    for keys, stops_g in stop.groupby(list(SESSION_COLUMNS), sort=False):
        if keys in onset_groups:
            by_animal.setdefault(keys[0], []).append(
                (keys, stops_g, onset_groups[keys])
            )
    jobs = [
        (a, o)
        for a in sorted(by_animal)
        for o in POPULATION_PLAN
        if (args.animals is None or a in args.animals)
        and (args.outcomes is None or o in args.outcomes)
    ]
    results = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_job, a, o, by_animal[a]): (a, o) for a, o in jobs}
        for fut in as_completed(futures):
            rec = fut.result()
            results.append(rec)
            fails = [
                n
                for n, f in {**rec["duration_aware"], **rec["shipped"]}.items()
                if not f["ok"]
            ]
            print(
                f"{time.perf_counter() - t0:7.0f}s {rec['animal']:7s} {rec['outcome']:17s} "
                f"bins={rec['n_bins']} events={rec['n_events']} failed={fails}",
                flush=True,
            )
    out = {"cohort": args.cohort, "results": results}
    path = OUT / f"pooled_kernels_{args.cohort}{'_smoke' if smoke else ''}.json"
    path.write_text(json.dumps(out))
    print("wrote", path)


if __name__ == "__main__":
    main()
