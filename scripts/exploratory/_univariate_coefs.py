#!/usr/bin/env python
"""SCRATCH: univariate (one-predictor) coefficients vs the joint full-model ones.

For each predictor, fit a model with ONLY that predictor (plus the family's
required baseline — arm fixed effects for arm, nothing for patch, intercept +
offset for rate) per session, and pool the per-session coefficients (median
exp(β) + Wilcoxon-FDR). Print alongside the joint estimate from the full model
so the marginal vs adjusted effects can be compared.
"""

from __future__ import annotations

import logging
import warnings

import joblib
import numpy as np
from continuum_swr_replay.analysis.replay.rate_model import fit_rate_model
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from continuum_swr_replay.analysis.replay._helpers import (
    ARM_FE_PREDICTORS,
    add_arm_fixed_effects,
)
from continuum_swr_replay.analysis.replay.fitting import fit_conditional_poisson
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.table_io import read_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
warnings.filterwarnings("ignore")
RC = PROCESSED_DATA_DIR / "replay_content"
SESSION = ["animal", "day", "epoch"]

ARM_P = ["is_current", "is_prev", "reward_rate_arm_k10", "visit_recency_demeaned",
         "is_last_rewarded_elsewhere"]
PATCH_P = ["is_current_patch", "is_last_rewarded_patch_elsewhere",
           "reward_rate_arm_k10_patch"]
RATE_P = ["reward_lag1", "is_after_switch", "reward_rate_trial_k10"]


def _univariate(table, predictors, fit_one):
    """Per-session single-predictor fits -> {pred: list of per-session coefs}."""
    out = {p: [] for p in predictors}
    for _, g in table.groupby(SESSION, sort=False):
        for p in predictors:
            res = fit_one(g, p)
            if res is not None and f"coef_{p}" in res and np.isfinite(res[f"coef_{p}"]):
                out[p].append(res[f"coef_{p}"])
    return out


def _pool(coefs_by_pred):
    rows = {}
    raw = []
    for p, cs in coefs_by_pred.items():
        cs = np.asarray(cs)
        if len(cs) < 3:
            rows[p] = (np.nan, np.nan, len(cs))
            continue
        try:
            pv = wilcoxon(cs)[1]
        except ValueError:
            pv = np.nan
        rows[p] = (float(np.exp(np.median(cs))), pv, len(cs))
        raw.append((p, pv))
    if raw:
        corr = multipletests([p for _, p in raw], method="fdr_bh")[1]
        for (p, _), pc in zip(raw, corr, strict=True):
            rows[p] = (rows[p][0], pc, rows[p][2])
    return rows


def main():
    # Arm: arm FE + one predictor (conditional Poisson on arm counts).
    arm_tbl = add_arm_fixed_effects(read_table(RC / "event_arm_table.pkl"))
    logging.info("arm univariate fits...")
    arm_uni = _pool(_univariate(
        arm_tbl, ARM_P,
        lambda g, p: fit_conditional_poisson(g, [*ARM_FE_PREDICTORS, p]),
    ))

    # Patch: one predictor alone (3-category conditional Poisson).
    patch_tbl = read_table(RC / "event_patch_table.pkl")
    logging.info("patch univariate fits...")
    patch_uni = _pool(_univariate(
        patch_tbl, PATCH_P,
        lambda g, p: fit_conditional_poisson(g, [p], response_col="patch_count_hard"),
    ))

    # Rate: intercept + one predictor (Poisson w/ log-immobility offset).
    stop = read_table(RC / "stop_table.pkl")
    logging.info("rate univariate fits...")
    rate_uni = _pool(_univariate(stop, RATE_P, lambda g, p: fit_rate_model(g, [p])))

    # Joint estimates from the saved full models.
    arm = joblib.load(RC / "models" / "all_models.pkl")
    patch = joblib.load(RC / "models_patch" / "all_models.pkl")
    rate = joblib.load(RC / "models_rate" / "all_models.pkl")

    def joint(df, model, p):
        c = df[df.model == model].get(f"coef_{p}")
        return np.exp(c.dropna().median()) if c is not None and c.notna().any() else np.nan

    def show(title, uni, df, joint_model):
        print(f"\n{title}   (univariate  vs  joint @ {joint_model})")
        print(f"  {'predictor':34s} {'uni ×':>8s} {'fdr p':>9s} {'n':>4s}   {'joint ×':>8s}")
        for p, (rr, pv, n) in uni.items():
            jr = joint(df, joint_model, p)
            print(f"  {p:34s} {rr:8.2f} {pv:9.1e} {n:4d}   {jr:8.2f}")

    show("ARM content (relative-rate)", arm_uni, arm, "M5")
    show("PATCH content (relative-rate)", patch_uni, patch, "P3")
    show("RATE (event-rate ratio)", rate_uni, rate, "R2")


if __name__ == "__main__":
    main()
