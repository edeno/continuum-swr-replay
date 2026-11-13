#!/usr/bin/env python
"""Predictive ranking of the curated uncertainty/value covariates.

TEMPORARY / exploratory (underscore-prefixed). Which curated covariate adds the
most predictive power over the M7r base, and what is the best combination?

Base model = M7r (time spline + reward kernel + raised-cosine self-history), with
NO recent_trial_reward_rate (== mean of known_reward_outcome_lag_1..10, perfectly collinear).
Candidates = the curated, near-orthogonal set (all VIF <= 1.07).

Two importance measures per covariate, both parallel over sessions:
  * univariate : AIC(base + cov) - AIC(base)            -- marginal value alone
  * drop-one   : AIC(full - cov) - AIC(full)            -- unique value given the rest
For single-covariate moves ΔAIC ranks identically to in-sample ΔLL (the penalty
is constant), so AIC is the cheap, correct ranking metric. Because the set is
near-orthogonal, univariate ≈ drop-one, and the best combination is simply every
covariate whose addition lowers AIC -- no forward-selection search needed. To
confirm the block helps OUT of sample, 10-fold cv_bits is computed for base and
full only (the costly part), reported as the per-session held-out gain.

Usage:
    python scripts/exploratory/_covariate_ranking.py [--max-per-animal N] [--workers W]
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from continuum_swr_replay.analysis.behavior.covariates import (  # noqa: E402
    add_uncertainty_covariates,
)
from continuum_swr_replay.analysis.replay_models.model_specs import (  # noqa: E402
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    N_REWARD_LAGS,
    RATE_BIN_S,
    STOP_COVARIATES,
    TMAX,
    PerBinRateModelSpec,
)
from continuum_swr_replay.analysis.replay_models.per_bin_design import (  # noqa: E402  # noqa: E402
    POPULATIONS,
    aggregate_to_timebins,
    attach_is_swr_to_onsets,
)
from continuum_swr_replay.analysis.replay_models.rate.per_bin_rate_model import (  # noqa: E402
    cross_validate_per_bin_rate_model,
    fit_per_bin_rate_model,
)
from continuum_swr_replay.paths import (  # noqa: E402
    REPLAY_CONTENT_DIR,
    model_dir,
    table_path,
)
from continuum_swr_replay.tables.covariate_registry import (  # noqa: E402
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table  # noqa: E402
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS  # noqa: E402

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RC = REPLAY_CONTENT_DIR
POPULATIONS_TO_FIT = ("swr", "non_local_swr", "non_local_no_swr")
# Value columns merged from qvals_table at exact stop-start alignment. Preserve
# their canonical scientific names through fitting and persisted diagnostics.
VALUE_COLS = (
    "task_reward_probability_chosen_arm",
    "rpe_postoutcome",
    "task_reward_probability_patch_mean_gap_best_vs_second",
    "trials_since_block_change",
    "reward_belief_mean_chosen_arm_preoutcome",
)
# Focused head-to-head: nominal (true current) vs subjective (model belief) value,
# plus the robust exploration axis. leaky_q / trials_since_block dropped (null);
# q_rpe retired (leaky-Q artifact; model rpe already == R - q_chosen).
CANDIDATES = (
    "task_reward_probability_chosen_arm",
    "rpe_postoutcome",
    "reward_belief_mean_chosen_arm_preoutcome",
    "patch_switch_fraction_last5_completed_trips",
    "task_reward_probability_patch_mean_gap_best_vs_second",
)


def _spec(extra: tuple[str, ...]) -> PerBinRateModelSpec:
    name = "base" if not extra else "+".join(extra)
    return PerBinRateModelSpec(
        name,
        True,
        N_REWARD_LAGS,
        False,
        N_HISTORY_LAGS,
        False,
        False,
        "covariate-ranking spec",
        "M7r_time_histcos_rew",
        self_history_basis=N_HISTORY_BASIS,
        extra_covariates=extra,
    )


def _aic(design: pd.DataFrame, extra: tuple[str, ...]) -> float:
    fit = fit_per_bin_rate_model(design, _spec(extra))
    return fit["aic"] if fit else np.nan


def _cv(design: pd.DataFrame, extra: tuple[str, ...]) -> float:
    cv = cross_validate_per_bin_rate_model(design, _spec(extra))
    return cv["cv_bits_per_event"] if cv else np.nan


def _run_session(keys, stops_g, onsets_g) -> list[dict]:
    stop_covariates = (*STOP_COVARIATES, *CANDIDATES)
    rows: list[dict] = []
    for pop in POPULATIONS_TO_FIT:
        d = aggregate_to_timebins(
            stops_g,
            POPULATIONS[pop](onsets_g),
            bin_s=RATE_BIN_S,
            tmax=TMAX,
            reward_lags=N_REWARD_LAGS,
            self_history_lags=N_HISTORY_LAGS,
            self_history_basis=N_HISTORY_BASIS,
            stop_covariates=stop_covariates,
        )
        aic_base = _aic(d, ())
        aic_full = _aic(d, CANDIDATES)
        base = dict(zip(SESSION_COLUMNS, keys, strict=True))
        for c in CANDIDATES:
            drop = tuple(x for x in CANDIDATES if x != c)
            rows.append(
                {
                    **base,
                    "population": pop,
                    "covariate": c,
                    "d_aic_uni": _aic(d, (c,)) - aic_base,  # <0 = improves
                    "d_aic_drop": _aic(d, drop) - aic_full,  # >0 = removing hurts
                }
            )
        rows.append(
            {
                **base,
                "population": pop,
                "covariate": "(block)",
                "cv_base": _cv(d, ()),
                "cv_full": _cv(d, CANDIDATES),
                "d_aic_block": aic_full - aic_base,
            }
        )
    return rows


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    stop = read_table(table_path("stop_table", RC))
    require_covariate_schema(stop, table="stop")
    stop = add_uncertainty_covariates(stop)
    cont = read_table(table_path("qvals_table", RC))
    require_covariate_schema(cont, table="qvals")
    stop = stop.merge(
        cont[["stop_id", *VALUE_COLS]],
        on="stop_id",
        how="inner",
        validate="many_to_one",
    )
    first = (
        stop[list(SESSION_COLUMNS)]
        .drop_duplicates()
        .sort_values(list(SESSION_COLUMNS))
        .groupby("animal", as_index=False)
        .first()
    )
    merged = stop.merge(first, on=list(SESSION_COLUMNS), how="left", indicator=True)
    stop = stop[merged["_merge"].to_numpy() == "left_only"]
    onset_table = read_table(table_path("event_onset_table", RC))
    event_table = read_table(table_path("event_table", RC))
    require_covariate_schema(onset_table, table="event_onset")
    require_covariate_schema(event_table, table="event")
    onsets = attach_is_swr_to_onsets(onset_table, event_table)
    return stop, onsets


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-per-animal", type=int, default=None)
    p.add_argument("--workers", type=int, default=1)
    args = p.parse_args()

    t0 = time.perf_counter()
    stop, onsets = _load()
    if args.max_per_animal is not None:
        keep = (
            stop[list(SESSION_COLUMNS)]
            .drop_duplicates()
            .groupby("animal", as_index=False)
            .head(args.max_per_animal)
        )
        stop = stop.merge(keep, on=list(SESSION_COLUMNS))
    onset_groups = dict(tuple(onsets.groupby(list(SESSION_COLUMNS), sort=False)))
    tasks = [
        (keys, g, onset_groups[keys])
        for keys, g in stop.groupby(list(SESSION_COLUMNS), sort=False)
        if keys in onset_groups
    ]
    logger.info(
        "Ranking %d covariates over %d sessions (workers=%d)...",
        len(CANDIDATES),
        len(tasks),
        args.workers,
    )

    rows: list[dict] = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for i, r in enumerate(ex.map(_run_session, *zip(*tasks, strict=True)), 1):
                rows.extend(r)
                if i % 25 == 0:
                    logger.info("  %d/%d sessions", i, len(tasks))
    else:
        for keys, g, og in tasks:
            rows.extend(_run_session(keys, g, og))
    df = pd.DataFrame(rows)
    out = model_dir("per_bin_rate", RC) / (
        "covariate_ranking_smoke.pkl"
        if args.max_per_animal
        else "covariate_ranking.pkl"
    )
    df.to_pickle(out)

    cov = df[df.covariate != "(block)"]
    for pop in POPULATIONS_TO_FIT:
        s = cov[cov.population == pop]
        agg = (
            s.groupby("covariate")
            .agg(
                uni_dAIC=("d_aic_uni", "median"),
                uni_pct_improve=("d_aic_uni", lambda x: (x < 0).mean() * 100),
                drop_dAIC=("d_aic_drop", "median"),
                n=("d_aic_uni", "size"),
            )
            .sort_values("uni_dAIC")  # most negative (best) first
        )
        logger.info(
            "=== %s: univariate & drop-one ΔAIC (negative uni = improves) ===\n%s",
            pop,
            agg.round(2).to_string(),
        )

    blk = df[df.covariate == "(block)"]
    logger.info("=== whole curated block: held-out cv_bits (base -> full) ===")
    for pop in POPULATIONS_TO_FIT:
        b = blk[blk.population == pop]
        d = (b["cv_full"] - b["cv_base"]).dropna()
        logger.info(
            "  %-18s base=%.4f full=%.4f  Δcv_bits median=%+.4f  %%pos=%.0f%% n=%d",
            pop,
            b["cv_base"].median(),
            b["cv_full"].median(),
            d.median(),
            (d > 0).mean() * 100,
            len(d),
        )
    logger.info("Saved %d rows to %s in %.1fs", len(df), out, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
