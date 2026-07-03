#!/usr/bin/env python
"""Fit the per-bin non-local rate models, `is_swr` version (see docs/models_table.md).

Two ladders on the canonical (stop x bin) design from `aggregate_to_timebins`
(15 ms bins, `log(exposure)` offset, Poisson), fit per session:

  * STRATIFIED (G0-G8): each spec fit to `non_local`, `non_local_swr`,
    `non_local_no_swr` separately. Comparing a coefficient across the three
    populations = the descriptive SWR-vs-non-SWR dissociation.
  * is_swr CONTRAST (S0-S10, design B): the `non_local_swr` and `non_local_no_swr`
    per-bin designs are STACKED (identical (stop,bin) rows + covariates, differing
    only in `count`), tagged by `is_swr`, and `is_swr:X` interactions test whether a
    task effect differs by SWR status. Self-history (`hist_rcos_*`) is stream-specific
    only in its raw per-stream values; it is still wrapped inside `is_swr * BASE`
    (S2-S10), so `is_swr:hist_rcos_k` refractoriness-by-SWR interactions ARE fit.
    Time/reward/task covariates are shared across the two rows of a bin. The within-bin
    pairing is carried by the per-session fit + across-session aggregation (no GLMM),
    matching the rest of Family B.

Covariate blocks BASE/PERISWITCH/BLOCK/SWITCHRATE/VALUE/RPE/UNCERT/UPDATE/REWXTIME
are defined below. The task
dimensions are the paper's (Comrie et al.), all from ONE model (`default_hmm`):
relative value (`hmm_relative_value`; the paper's single value+decision axis, cross-model
corroborated by q_chosen/ΔV r=-0.67), RPE (unscaled `reward - Q_leaf`), HMM reward-state
entropy (native `reward_entropy`, not the port-specific variant), global value update;
+ CONTEXT (peri-switch/
block) from `qvals_table` and `recent_switch_rate` from `add_uncertainty_covariates`.
One tidy row per (population, model, session); aggregate across sessions (signed-rank).

Usage:
    uv run python scripts/rate/fit_swr_contrast_rate.py --max-per-animal 1   # smoke (~4 min)
    uv run python scripts/rate/fit_swr_contrast_rate.py --workers 16         # full
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from patsy import PatsyError  # noqa: E402
from statsmodels.tools.sm_exceptions import PerfectSeparationError  # noqa: E402

from continuum_swr_replay.analysis.replay import (  # noqa: E402
    BIN_S,
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    N_REWARD_LAGS,
    POPULATIONS,
    TMAX,
    add_uncertainty_covariates,
    aggregate_to_timebins,
    attach_is_swr_to_onsets,
)
from continuum_swr_replay.analysis.replay.model_specs import (  # noqa: E402
    SWR_CONTRAST_RATE_G_SPECS as G_SPECS,
)
from continuum_swr_replay.analysis.replay.model_specs import (  # noqa: E402
    SWR_CONTRAST_RATE_S_SPECS as S_SPECS,
)
from continuum_swr_replay.analysis.replay.model_specs import (  # noqa: E402
    SWR_CONTRAST_RATE_STOP_COVARS as STOP_COVARS,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR  # noqa: E402

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

SESSION_COLS = ["animal", "day", "epoch"]
RC = PROCESSED_DATA_DIR / "replay_content"
STRATA = ("non_local", "non_local_swr", "non_local_no_swr")
MIN_BINS, MIN_EVENTS = 200, 20
_EXPECTED = (
    np.linalg.LinAlgError,
    ValueError,
    RuntimeError,
    PerfectSeparationError,
    PatsyError,
)


def _fit(
    design: pd.DataFrame, formula: str, need_cols: list[str]
) -> tuple[dict | None, str | None]:
    """Poisson GLM `count ~ formula` with a log(exposure) offset. NaN-drop then
    offset (mirrors fit_per_bin_rate_model to keep the offset aligned).

    Returns ``(row, reason)``: ``reason`` is ``None`` on success, else a short tag
    for the skip -- ``missing:<cols>`` (a needed column is absent -> a config/upstream
    bug, distinct from a data skip), ``too_few``, ``no_reward_contrast``, or the
    caught exception's class name. The caller tallies these so drop-outs are visible
    (a silently-shrunk per-model session count would bias the downstream aggregate).
    """
    missing = [c for c in need_cols if c not in design.columns]
    if missing:
        return None, f"missing:{','.join(missing)}"
    df = design.dropna(subset=[*need_cols, "exposure"])
    df = df[df["exposure"] > 0]
    n_events = int((df["count"] > 0).sum())
    if len(df) < MIN_BINS or n_events < MIN_EVENTS:
        return None, "too_few"
    if "reward_lag1" in need_cols and df["reward_lag1"].nunique() < 2:
        return None, "no_reward_contrast"  # reward terms unidentified without contrast
    offset = np.log(df["exposure"].to_numpy(dtype=float))
    try:
        res = smf.glm(
            f"count ~ {formula}", data=df, family=sm.families.Poisson(), offset=offset
        ).fit()
    except _EXPECTED as e:
        return None, type(e).__name__
    out = {f"coef_{k}": v for k, v in res.params.items()}
    out.update(
        n_bins=int(res.nobs), n_events=n_events, llf=float(res.llf), aic=float(res.aic)
    )
    return out, None


def _fit_one_session(keys, stops_g, onsets_g) -> tuple[tuple, list[dict], dict]:
    rows: list[dict] = []
    skips: dict = {}  # (model, population, reason) -> count
    designs = {
        pop: aggregate_to_timebins(
            stops_g,
            POPULATIONS[pop](onsets_g),
            bin_s=BIN_S,
            tmax=TMAX,
            reward_lags=N_REWARD_LAGS,
            self_history_lags=N_HISTORY_LAGS,
            self_history_basis=N_HISTORY_BASIS,
            stop_covariates=STOP_COVARS,
        )
        for pop in STRATA
    }
    base = dict(zip(SESSION_COLS, keys, strict=True))
    # A. stratified
    for pop in STRATA:
        for name, (formula, need) in G_SPECS.items():
            fit, reason = _fit(designs[pop], formula, need)
            if fit is not None:
                rows.append({"population": pop, "model": name, **base, **fit})
            else:
                skips[(name, pop, reason)] = skips.get((name, pop, reason), 0) + 1
    # B. is_swr contrast: stack the two non-local streams
    stacked = pd.concat(
        [
            designs["non_local_swr"].assign(is_swr=1.0),
            designs["non_local_no_swr"].assign(is_swr=0.0),
        ],
        ignore_index=True,
    )
    for name, (formula, need) in S_SPECS.items():
        fit, reason = _fit(stacked, formula, [*need, "is_swr"])
        if fit is not None:
            rows.append({"population": "contrast", "model": name, **base, **fit})
        else:
            skips[(name, "contrast", reason)] = (
                skips.get((name, "contrast", reason), 0) + 1
            )
    return keys, rows, skips


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    stop = add_uncertainty_covariates(pd.read_pickle(RC / "stop_table.pkl"))
    q = pd.read_pickle(RC / "models_per_bin_rate" / "qvals_table.pkl")
    qcols = [
        "stop_id",
        "trials_since_switch",
        "is_post_switch",
        "trials_since_block_change",
        "hmm_relative_value",
        "hmm_rpe",
        "reward_entropy",
        "hmm_value_update",
    ]
    missing = [c for c in qcols if c not in q.columns]
    if missing:
        raise ValueError(
            f"qvals_table missing {missing}; regenerate it "
            "(python -m continuum_swr_replay.analysis.behavior.qvals_table --write)"
        )
    n0 = len(stop)
    stop = stop.merge(q[qcols], on="stop_id", how="inner", validate="many_to_one")
    # fail loudly if a task covariate the specs need never made it onto the stop
    # table (rename / add_uncertainty_covariates / qvals regression) -- otherwise
    # every model using it would silently drop out of the output.
    missing_cov = [c for c in STOP_COVARS if c not in stop.columns]
    if missing_cov:
        raise ValueError(
            f"stop table missing task covariates {missing_cov} "
            "(check add_uncertainty_covariates + the qvals merge)"
        )
    logger.info(
        "stop x qvals join: %d -> %d stops (%.0f%% kept)",
        n0,
        len(stop),
        100 * len(stop) / n0,
    )
    onsets = attach_is_swr_to_onsets(
        pd.read_pickle(RC / "event_onset_table.pkl"),
        pd.read_pickle(RC / "event_table.pkl"),
    )
    return stop, onsets


def _subset_max_per_animal(stop: pd.DataFrame, n: int) -> pd.DataFrame:
    keep = stop[SESSION_COLS].drop_duplicates().groupby("animal", sort=False).head(n)
    return stop.merge(keep, on=SESSION_COLS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--max-per-animal", type=int, default=None, help="smoke: N sessions/animal"
    )
    ap.add_argument(
        "--workers", type=int, default=1, help="per-session process fan-out"
    )
    args = ap.parse_args()

    stop, onsets = _load()
    if args.max_per_animal is not None:
        stop = _subset_max_per_animal(stop, args.max_per_animal)

    onset_groups = dict(tuple(onsets.groupby(SESSION_COLS, sort=False)))
    items = [
        (k, s, onset_groups[k])
        for k, s in stop.groupby(SESSION_COLS, sort=False)
        if k in onset_groups
    ]
    logger.info(
        "Fitting SWR-contrast per-bin models (G0-G8 x %s + S0-S10) over %d sessions "
        "at BIN_S=%.3f with %d worker(s)...",
        list(STRATA),
        len(items),
        BIN_S,
        args.workers,
    )

    t0 = time.perf_counter()
    rows: list[dict] = []
    skips: Counter = Counter()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_fit_one_session, k, s, o) for k, s, o in items]
            for fut in as_completed(futs):
                _, r, sk = fut.result()
                rows.extend(r)
                skips.update(sk)
    else:
        for k, s, o in items:
            _, r, sk = _fit_one_session(k, s, o)
            rows.extend(r)
            skips.update(sk)
            logger.info("  %s: %d fits", "_".join(map(str, k)), len(r))

    # log skipped (model, population, reason) cells so a silently-shrunk per-model
    # session count is visible before the downstream signed-rank aggregate.
    if skips:
        cfg = {
            f"{m}/{p}": r
            for (m, p, r), _ in skips.items()
            if str(r).startswith("missing")
        }
        if cfg:
            logger.warning("MISSING-COLUMN skips (config/upstream bug): %s", cfg)
        logger.info(
            "skipped (model/pop -> reason: count): %s",
            {f"{m}/{p}:{r}": c for (m, p, r), c in sorted(skips.items())},
        )

    models = pd.DataFrame(rows)
    out_dir = RC / "models_per_bin_rate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (
        "swr_contrast_rate_smoke.pkl"
        if args.max_per_animal
        else "swr_contrast_rate.pkl"
    )
    models.to_pickle(out)
    logger.info(
        "Saved %d rows (%d sessions, populations=%s, models=%s) to %s in %.1fs",
        len(models),
        models.groupby(SESSION_COLS).ngroups if len(models) else 0,
        sorted(models["population"].unique()) if len(models) else [],
        sorted(models["model"].unique()) if len(models) else [],
        out,
        time.perf_counter() - t0,
    )


if __name__ == "__main__":
    main()
