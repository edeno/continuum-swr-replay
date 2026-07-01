#!/usr/bin/env python
"""Fit the per-bin replay rate vs behavioral-uncertainty / value, by population.

Tests whether non-local replay *rate* is differentially modulated by the
animal's behavioral uncertainty and the task's nominal value structure across
SWR-coupled vs non-SWR non-local events, controlling for the within-stop time
course, raised-cosine self-history, the reward kernel, and the recent reward
rate. Rate-only -- replay direction is out of scope.

Per session: load ``stop_table``; add ``recent_switch_rate``
(:func:`add_uncertainty_covariates`); inner-merge the nominal value + real
behavioral-model covariates from ``qvals_table.pkl`` on ``stop_id`` (one exactly
poke_out-aligned source -- no difflib CSV); drop the first (exposure) session per
animal; attach the canonical ``is_swr`` to the onsets.
Per population in ``{swr, non_local_swr, non_local_no_swr}`` build the per-bin
design (carrying the uncertainty + value covariates as ``stop_covariates``) and
fit a single spec -- ``MU_uncertainty`` -- instantiated HERE (not added to
``MODEL_PER_BIN_RATE`` / ``all_models.pkl``). The spec is M7r (time spline +
raised-cosine self-history + reward kernel + reward rate) plus the
uncertainty/value block, so the block is tested *beyond* those confounds.

Collect one tidy row per (session, population) with every covariate's
``coef_`` / ``bse_`` / ``pval_`` / ``rr_`` ->
``Processed-Data/replay_content/models_per_bin_rate/uncertainty_rate.pkl``.

Usage::

    uv run python scripts/fit_uncertainty_rate.py                    # full
    uv run python scripts/fit_uncertainty_rate.py --max-per-animal 1 # smoke
"""

from __future__ import annotations

import os

# Pin BLAS to one thread BEFORE numpy import (matches the Family B orchestrator).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402

import pandas as pd  # noqa: E402

from continuum_swr_replay.analysis.replay import (  # noqa: E402
    BIN_S,
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    N_REWARD_LAGS,
    STOP_COVARIATES,
    TMAX,
    PerBinRateModelSpec,
    add_uncertainty_covariates,
    aggregate_to_timebins,
    attach_is_swr_to_onsets,
    fit_per_bin_rate_model,
)
from continuum_swr_replay.analysis.replay._helpers import POPULATIONS  # noqa: E402
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR  # noqa: E402

warnings.filterwarnings("ignore")  # separated history coefs legitimately overflow exp()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

SESSION_COLS = ["animal", "day", "epoch"]
RC = PROCESSED_DATA_DIR / "replay_content"

# Nominal schedule covariates, now sourced from qvals_table (derived from Alison's
# p_rew_leaf/contingency_count, exactly poke_out-aligned) rather than the difflib
# contingency_table_csv -- so every covariate the model fits shares one alignment.
VALUE_COVARIATES = (
    "p_reward_current",
    "value_diff_patch",
    "trials_since_block_change",
)

# Nominal / model-free block (per-unit; these covariates are O(1) and are the
# task schedule + raw behavior, NOT the behavioral model). We prefer the real
# model over stand-ins, so the leaky-Q q_current is replaced by the model DeltaV
# (switch_advantage) and the nominal rpe by rpe_model below. value_diff_patch is
# KEPT: it is the nominal best-minus-second patch value from the schedule and is
# orthogonal to the model DeltaV (r=-0.04), so it carries independent information
# (an earlier drop assuming redundancy was wrong). Excluded for collinearity:
# wsls_deviation (r=-0.70 with rpe), omission_streak (r=-0.66 with reward_lag1).
CURATED_COVARIATES = (
    "p_reward_current",
    "value_diff_patch",
    "recent_switch_rate",
    "trials_since_block_change",
)

# Real behavioral-model reward-prediction error, used INSTEAD of the nominal rpe
# stand-in (rpe_model vs old nominal rpe: r=0.71 -- the model version matters).
# q_chosen (model chosen-port value) is deliberately NOT fit: it is r=0.90
# collinear with the DeltaV decision variable (switch_advantage), which carries
# the model's value signal here. O(1) scale, so per-unit like the nominal block.
MODEL_VALUE_COVARIATES = (
    "rpe_model",  # reward - q_chosen (model reward-prediction error)
)

# Real SpatialBanditTask decision variables (qvals_table.pkl, beta_stable_withleaf
# + HMM entropy; see docs/behavior_model_covariates.md). The paper's covariates:
# DeltaV, reward-state entropy, absolute value-update magnitudes. z-scored below
# (disparate/tiny scale) -> per-SD rate ratios.
RL_COVARIATES = (
    "switch_advantage_mu_preoutcome_sess",  # DeltaV = Switch - Stay value
    "reward_entropy",  # outcome uncertainty (HMM, upcoming/chosen port)
    "global_mu_update_abs_postoutcome_sess",  # |value update| summed over all ports
    "unchosen_patches_mu_update_abs_postoutcome_sess",  # |update| over alternative patches
)

# What the model actually fits + reports (nominal + model value + RL DVs).
FITTED_COVARIATES = (*CURATED_COVARIATES, *MODEL_VALUE_COVARIATES, *RL_COVARIATES)

# The three non-local-vs-SWR populations to dissociate (pooled non_local is the
# null and is not fit here).
POPULATIONS_TO_FIT = ("swr", "non_local_swr", "non_local_no_swr")

# M7r + the curated uncertainty/value block, instantiated in the runner so the
# shipped MODEL_PER_BIN_RATE ladder / all_models.pkl schema is untouched.
# reward_rate is OFF: reward_rate_trial_k10 (the reward_rate covariate) is exactly the mean of
# reward_lag1..10 (corr 1.000, VIF inf), so including it with the reward kernel
# makes the reward block rank-deficient; the kernel alone carries reward history.
MU_UNCERTAINTY = PerBinRateModelSpec(
    "MU_uncertainty",
    True,  # time spline
    N_REWARD_LAGS,  # reward kernel
    False,  # recent reward rate OFF (perfectly collinear with the kernel)
    N_HISTORY_LAGS,
    False,  # no switch term (recent_switch_rate carries the switch signal)
    False,  # no reward x time interaction
    "curated uncertainty/value beyond time/reward/history",
    "M7r_time_histcos_rew",
    self_history_basis=N_HISTORY_BASIS,
    extra_covariates=FITTED_COVARIATES,
)


def _drop_first_session_per_animal(stop_table: pd.DataFrame) -> pd.DataFrame:
    """Drop each animal's first (exposure) session by (day, epoch) order."""
    first = (
        stop_table[SESSION_COLS]
        .drop_duplicates()
        .sort_values(SESSION_COLS)
        .groupby("animal", sort=False)
        .head(1)
    )
    merged = stop_table.merge(first, on=SESSION_COLS, how="left", indicator=True)
    return stop_table[merged["_merge"].to_numpy() == "left_only"]


def _subset_max_per_animal(stop_table: pd.DataFrame, n: int) -> pd.DataFrame:
    keep = (
        stop_table[SESSION_COLS]
        .drop_duplicates()
        .sort_values(SESSION_COLS)
        .groupby("animal", sort=False)
        .head(n)
    )
    return stop_table.merge(keep, on=SESSION_COLS)


def fit_uncertainty_rate(
    stop_table: pd.DataFrame,
    onsets: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (session, population): MU_uncertainty fit coefficients.

    ``stop_table`` must already carry the uncertainty + value covariates and the
    reward lags; ``onsets`` must already carry ``is_swr``.
    """
    onset_groups = dict(tuple(onsets.groupby(SESSION_COLS, sort=False)))
    stop_covariates = (*STOP_COVARIATES, *FITTED_COVARIATES)

    rows: list[dict] = []
    attempted = dict.fromkeys(POPULATIONS_TO_FIT, 0)
    n_unfit = dict.fromkeys(POPULATIONS_TO_FIT, 0)
    for keys, stops_g in stop_table.groupby(SESSION_COLS, sort=False):
        if keys not in onset_groups:
            continue
        onsets_g = onset_groups[keys]
        for outcome in POPULATIONS_TO_FIT:
            attempted[outcome] += 1
            design = aggregate_to_timebins(
                stops_g,
                POPULATIONS[outcome](onsets_g),
                bin_s=BIN_S,
                tmax=TMAX,
                reward_lags=N_REWARD_LAGS,
                self_history_lags=N_HISTORY_LAGS,
                self_history_basis=N_HISTORY_BASIS,
                stop_covariates=stop_covariates,
            )
            fit = fit_per_bin_rate_model(design, MU_UNCERTAINTY)
            if fit is None:
                n_unfit[outcome] += 1
                logger.info("  %s / %s: unfit", "_".join(map(str, keys)), outcome)
                continue
            rows.append(
                {
                    "outcome": outcome,
                    "model": MU_UNCERTAINTY.name,
                    **dict(zip(SESSION_COLS, keys, strict=True)),
                    **{k: v for k, v in fit.items() if k != "predicted"},
                }
            )
    # Surface survivor bias: if many (session x population) cells failed to fit,
    # the median RRs for that population come from a biased survivor subset.
    for outcome in POPULATIONS_TO_FIT:
        att, bad = attempted[outcome], n_unfit[outcome]
        if att and bad:
            frac = bad / att
            (logger.warning if frac > 0.2 else logger.info)(
                "population %s: %d/%d session cells unfit (%.0f%%)", outcome, bad, att, 100 * frac
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-per-animal",
        type=int,
        default=None,
        help="Smoke mode: keep only the first N (post-exposure) sessions per animal.",
    )
    parser.add_argument(
        "--switch-window", type=int, default=5, help="recent_switch_rate window."
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    stop = pd.read_pickle(RC / "stop_table.pkl")
    onsets = attach_is_swr_to_onsets(
        pd.read_pickle(RC / "event_onset_table.pkl"),
        pd.read_pickle(RC / "event_table.pkl"),
    )

    # add_uncertainty_covariates supplies recent_switch_rate (the only model-free
    # covariate still fit); its leaky_q_value_diff / omission_streak / wsls_deviation
    # stand-ins are no longer used here.
    stop = add_uncertainty_covariates(stop, switch_window=args.switch_window)
    # Nominal schedule + real model covariates, ALL from qvals_table -- one
    # exactly poke_out-aligned source (retires the difflib contingency_table_csv
    # for this model). qvals' `rpe` (R - q_chosen) -> rpe_model. Inner join: keep
    # only stops mapped to a trial (post-final-trial tails ~1-3% drop out); model
    # DVs are NaN on the first trial of a session -> listwise-dropped at fit.
    qvals = pd.read_pickle(RC / "models_per_bin_rate" / "qvals_table.pkl")
    qcols = qvals[["stop_id", *VALUE_COVARIATES, "rpe", *RL_COVARIATES]].rename(
        columns={"rpe": "rpe_model"}
    )
    n_before = len(stop)
    stop = stop.merge(qcols, on="stop_id", how="inner", validate="many_to_one")
    frac = len(stop) / n_before
    if frac < 0.5:
        raise ValueError(
            f"qvals join kept only {frac:.1%} of stops -- check qvals_table stop_id convention"
        )
    logger.info("qvals join: %d -> %d stops (%.0f%% kept)", n_before, len(stop), 100 * frac)
    stop = _drop_first_session_per_animal(stop)
    if args.max_per_animal is not None:
        stop = _subset_max_per_animal(stop, args.max_per_animal)
    # z-score the RL decision variables AFTER the drop/subset so the per-SD scale
    # is the SD of exactly the rows that get fit (disparate/tiny scale: DeltaV
    # ~0.2, updates ~0.05). The nominal block + rpe_model stay per-unit.
    for c in RL_COVARIATES:
        frac_valid = stop[c].notna().mean()
        if frac_valid < 0.7:
            logger.warning(
                "RL covariate %r is non-NaN on only %.0f%% of fitted rows; its SD "
                "(hence the per-SD rate ratio) is set by that minority and rows with "
                "NaN drop from every model listwise",
                c, 100 * frac_valid,
            )
        sd = stop[c].std()
        if pd.isna(sd) or sd == 0:
            raise ValueError(f"RL covariate {c!r} has degenerate SD ({sd}); cannot z-score")
        stop[c] = (stop[c] - stop[c].mean()) / sd

    n_sessions = stop.groupby(SESSION_COLS).ngroups
    logger.info(
        "Fitting %s over %d sessions x %s populations (BIN_S=%.3f, "
        "switch_window=%d)...",
        MU_UNCERTAINTY.name,
        n_sessions,
        list(POPULATIONS_TO_FIT),
        BIN_S,
        args.switch_window,
    )
    models = fit_uncertainty_rate(stop, onsets)

    out_dir = RC / "models_per_bin_rate"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = (
        "uncertainty_rate_smoke.pkl" if args.max_per_animal else "uncertainty_rate.pkl"
    )
    out = out_dir / fname
    models.to_pickle(out)
    logger.info(
        "Saved %d rows (%d sessions, populations=%s) to %s in %.1fs",
        len(models),
        models.groupby(SESSION_COLS).ngroups if len(models) else 0,
        sorted(models["outcome"].unique()) if len(models) else [],
        out,
        time.perf_counter() - t0,
    )
    if len(models):
        for cov in FITTED_COVARIATES:
            col = f"rr_{cov}"
            if col not in models:
                continue
            summary = models.groupby("outcome")[col].median().to_dict()
            logger.info("median RR  %-26s %s", cov, summary)


if __name__ == "__main__":
    main()
