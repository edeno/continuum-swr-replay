#!/usr/bin/env python
"""Fit the per-bin conditional-intensity ladder (Family B) and persist it.

Family B is the point-process counterpart to the per-stop rate hierarchy
(``fit_per_stop_rate_models.py``). Where Family A fits one Poisson GLM per stop
with a ``log(immobility_time)`` dwell offset, here every model is fit on the
canonical (stop x bin) design from ``aggregate_to_timebins`` with a
``log(exposure)`` offset, modelling the within-stop time course explicitly.

The WHOLE ladder runs at one bin width (``BIN_S = 0.015``): for each
(session, outcome) the design is built ONCE — carrying the reward lags, the
self-history lags, and the M2r/M8 stop covariates — and all 11
``MODEL_PER_BIN_RATE`` models (M0_const..M9_rew_x_time) are fit against that
same design (each ``formula()`` selects its own columns). Sharing identical
bins is what makes the nested AIC / CV / GOF comparison valid. Designs are
built per session so all sessions' bins are never resident at once.

Each (session, outcome, model) contributes one tidy row with the fit
coefficients, the 10-fold stop-based cross-validation (``cv_bits_per_event``),
and the point-process goodness-of-fit (``ks_statistic`` etc.), mirroring the
arm/patch/per-stop ``all_models.pkl`` so the families concatenate on
``[outcome, model, animal, day, epoch]``.

The four outcomes (``POPULATIONS``) are detected SWR, all non-local, and the
non-local SWR / no-SWR split — the pooled ``non_local`` is the one-population
null tested against its split.

Usage:
    uv run python scripts/rate/fit_per_bin_rate_models.py               # full, serial
    uv run python scripts/rate/fit_per_bin_rate_models.py --workers 16  # full, parallel
    uv run python scripts/rate/fit_per_bin_rate_models.py --max-per-animal 1  # smoke
"""

from __future__ import annotations

import os

# Pin BLAS to one thread per process BEFORE numpy is imported: with the
# per-session process fan-out (--workers), multi-threaded BLAS would
# oversubscribe the cores and run far slower than single-threaded workers.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import pandas as pd  # noqa: E402

from continuum_swr_replay.analysis.replay import (  # noqa: E402
    BIN_S,
    MODEL_PER_BIN_RATE,
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    N_REWARD_LAGS,
    POPULATIONS,
    STOP_COVARIATES,
    TMAX,
    aggregate_to_timebins,
    attach_is_swr_to_onsets,
    cross_validate_per_bin_rate_model,
    fit_per_bin_rate_model,
    per_bin_rate_gof,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR  # noqa: E402

# Separated self-history coefficients legitimately overflow exp() (-> 0 / inf
# RR); these are median-robust at read time and are not fit failures.
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

SESSION_COLS = ["animal", "day", "epoch"]
RC = PROCESSED_DATA_DIR / "replay_content"


def _fit_one_session(
    keys: tuple,
    stops_g: pd.DataFrame,
    onsets_g: pd.DataFrame,
    run_cv: bool,
    run_gof: bool,
) -> tuple[tuple, list[dict], dict, dict, float]:
    """Fit all (outcome, model) cells for one session.

    Returns ``(keys, rows, none_counts, cv_none_counts, elapsed)``.
    ``none_counts`` maps ``(model, outcome) -> count`` of cells whose GLM fit
    returned ``None`` (too little data / unidentified / numerical failure);
    ``cv_none_counts`` maps the same key to cells that fit but whose
    cross-validation returned ``None`` (a fold failed or the design was not
    CV-able), which would otherwise drop the ``cv_*`` columns silently.
    Top-level (picklable) so it can run in a worker process.
    """
    t0 = time.perf_counter()
    rows: list[dict] = []
    none_counts: dict = {}
    cv_none_counts: dict = {}
    for outcome, selector in POPULATIONS.items():
        design = aggregate_to_timebins(
            stops_g,
            selector(onsets_g),
            bin_s=BIN_S,
            tmax=TMAX,
            reward_lags=N_REWARD_LAGS,
            self_history_lags=N_HISTORY_LAGS,
            self_history_basis=N_HISTORY_BASIS,
            stop_covariates=STOP_COVARIATES,
        )
        for name, spec in MODEL_PER_BIN_RATE.items():
            fit = fit_per_bin_rate_model(design, spec)
            if fit is None:
                none_counts[(name, outcome)] = none_counts.get((name, outcome), 0) + 1
                continue
            row = {
                "outcome": outcome,
                "model": name,
                **dict(zip(SESSION_COLS, keys, strict=True)),
                # 'predicted' is a per-bin Series (GOF input), not tidy data.
                **{k: v for k, v in fit.items() if k != "predicted"},
            }
            if run_gof:
                gof = per_bin_rate_gof(design, fit)
                # n_events / dispersion duplicate the fit dict (same rows);
                # keep the fit's to avoid a silent, fragile overwrite.
                gof.pop("n_events", None)
                gof.pop("dispersion", None)
                row.update(gof)
            if run_cv:
                cv = cross_validate_per_bin_rate_model(design, spec)
                if cv is not None:
                    row.update(cv)
                else:
                    cv_none_counts[(name, outcome)] = (
                        cv_none_counts.get((name, outcome), 0) + 1
                    )
            rows.append(row)
    return keys, rows, none_counts, cv_none_counts, time.perf_counter() - t0


def fit_per_bin_rate_models(
    stop_table: pd.DataFrame,
    onsets: pd.DataFrame,
    *,
    run_cv: bool = True,
    run_gof: bool = True,
    workers: int = 1,
) -> pd.DataFrame:
    """One row per (outcome, model, session): coefficients + CV + GOF.

    ``onsets`` must already carry the canonical ``is_swr`` column (run
    :func:`attach_is_swr_to_onsets` once before calling). ``workers > 1``
    fans the per-session work out over processes (sessions are independent).
    """
    onset_groups = dict(tuple(onsets.groupby(SESSION_COLS, sort=False)))
    items = [
        (keys, stops_g, onset_groups[keys])
        for keys, stops_g in stop_table.groupby(SESSION_COLS, sort=False)
        if keys in onset_groups
    ]

    rows: list[dict] = []
    none_counts: Counter = Counter()
    cv_none_counts: Counter = Counter()

    def _collect(keys, r, nc, cvnc, elapsed):
        rows.extend(r)
        none_counts.update(nc)
        cv_none_counts.update(cvnc)
        logger.info("  %s: %d fits in %.1fs", "_".join(map(str, keys)), len(r), elapsed)

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(_fit_one_session, k, s, o, run_cv, run_gof)
                for k, s, o in items
            ]
            for fut in as_completed(futures):
                _collect(*fut.result())
    else:
        for k, s, o in items:
            _collect(*_fit_one_session(k, s, o, run_cv, run_gof))

    if none_counts:
        logger.info(
            "None (unfit) counts per (model, outcome): %s",
            {f"{m}/{o}": c for (m, o), c in sorted(none_counts.items())},
        )
    if cv_none_counts:
        logger.info(
            "CV-None (fit ok, CV dropped) counts per (model, outcome): %s",
            {f"{m}/{o}": c for (m, o), c in sorted(cv_none_counts.items())},
        )
    return pd.DataFrame(rows)


def _subset_max_per_animal(stop_table: pd.DataFrame, n: int) -> pd.DataFrame:
    keep = (
        stop_table[SESSION_COLS]
        .drop_duplicates()
        .groupby("animal", sort=False)
        .head(n)
    )
    return stop_table.merge(keep, on=SESSION_COLS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-per-animal",
        type=int,
        default=None,
        help="Smoke mode: keep only the first N sessions per animal.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Process workers for the per-session fan-out (default serial).",
    )
    parser.add_argument("--no-cv", action="store_true", help="Skip cross-validation.")
    parser.add_argument("--no-gof", action="store_true", help="Skip goodness-of-fit.")
    args = parser.parse_args()

    stop = pd.read_pickle(RC / "stop_table.pkl")
    onsets = attach_is_swr_to_onsets(
        pd.read_pickle(RC / "event_onset_table.pkl"),
        pd.read_pickle(RC / "event_table.pkl"),
    )
    if args.max_per_animal is not None:
        stop = _subset_max_per_animal(stop, args.max_per_animal)

    logger.info(
        "Fitting per-bin ladder %s (+CV=%s, +GOF=%s) x %s over %d sessions "
        "at BIN_S=%.3f with %d worker(s)...",
        list(MODEL_PER_BIN_RATE),
        not args.no_cv,
        not args.no_gof,
        list(POPULATIONS),
        stop.groupby(SESSION_COLS).ngroups,
        BIN_S,
        args.workers,
    )
    t0 = time.perf_counter()
    models = fit_per_bin_rate_models(
        stop, onsets, run_cv=not args.no_cv, run_gof=not args.no_gof,
        workers=args.workers,
    )

    out_dir = RC / "models_per_bin_rate"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = "all_models_smoke.pkl" if args.max_per_animal else "all_models.pkl"
    out = out_dir / fname
    models.to_pickle(out)
    logger.info(
        "Saved %d rows (%d sessions, outcomes=%s, models=%s) to %s in %.1fs",
        len(models),
        models.groupby(SESSION_COLS).ngroups if len(models) else 0,
        sorted(models["outcome"].unique()) if len(models) else [],
        sorted(models["model"].unique()) if len(models) else [],
        out,
        time.perf_counter() - t0,
    )


if __name__ == "__main__":
    main()
