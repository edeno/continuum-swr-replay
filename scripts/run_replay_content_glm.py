#!/usr/bin/env python
"""Per-session and mixed-effects GLM analysis of replay spatial content.

Three analysis tiers:

1. Per-session GLM (base): Fit per session × event type, test across sessions.
   - Model A (patch_reduced): replay ~ is_current + is_last_rewarded + is_prev
   - Model B (arm): replay ~ is_current + is_last_rewarded + is_prev + is_next

2. Per-session GLM (interactions): Same as base, plus task-context interactions.
   - Model C (arm_context): Model B + is_after_switch × (is_current + is_prev)
                                     + is_after_omission × is_last_rewarded

3. Mixed-effects model: Single model across all sessions.
   - replay ~ is_current + is_last_rewarded + is_prev + is_next
              + is_swr × (is_current + is_last_rewarded + is_prev + is_next)
              + is_after_switch × (is_current + is_prev)
              + is_after_omission × is_last_rewarded
              + (1 + is_current | session)

Usage:
    uv run python scripts/run_replay_content_glm.py           # full run
    uv run python scripts/run_replay_content_glm.py --test    # 2 per animal
    uv run python scripts/run_replay_content_glm.py --max-per-animal 5
"""

import argparse
import logging
import pickle
import re
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import linregress, mannwhitneyu, wilcoxon
from statsmodels.stats.outliers_influence import variance_inflation_factor

import continuum_swr_replay.figure_utilities as figure_utilities
from continuum_swr_replay.analysis.constants import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    DEFAULT_NON_LOCAL_THRESHOLD,
)
from continuum_swr_replay.data_loaders import load_data
from continuum_swr_replay.data_loaders.constants import (
    NWB_FILES,
    TRACK_SEGMENT_TO_PATCH,
)
from continuum_swr_replay.parameters import COLORS, PROCESSED_DATA_DIR
from continuum_swr_replay.types import SessionDataKeys

warnings.filterwarnings("ignore")
figure_utilities.set_figure_defaults(context="paper")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

ALPHA = 0.01
OUTPUT_DIR = Path("paper/figures/replay_content_glm")

# Model specifications.
# split_by_swr=True: fit separately for SWR and non-SWR events.
# split_by_swr=False: fit on pooled data (for model comparison).
MODEL_SPECS = {
    "patch_reduced": {
        "predictors": ["is_current", "is_last_rewarded", "is_prev"],
        "level": "patch",
        "split_by_swr": True,
        "description": "Patch-level, 3 predictors (is_next dropped: collinear)",
    },
    "arm": {
        "predictors": ["is_current", "is_last_rewarded", "is_prev", "is_next"],
        "level": "arm",
        "split_by_swr": True,
        "description": "Arm-level (9 segments), 4 predictors",
    },
    "arm_context": {
        "predictors": [
            "is_current",
            "is_last_rewarded",
            "is_prev",
            "is_next",
            "is_after_switch",
            "switch_x_current",
            "switch_x_prev",
            "is_after_omission",
            "omission_x_rewarded",
        ],
        "level": "arm",
        "split_by_swr": True,
        "description": "Arm-level + switch/omission interactions",
    },
    # ── Model comparison pair: does SWR status improve prediction? ──
    "arm_pooled": {
        "predictors": ["is_current", "is_last_rewarded", "is_prev", "is_next"],
        "level": "arm",
        "split_by_swr": False,
        "description": "Arm-level, pooled SWR+non-SWR (null for comparison)",
    },
    "arm_swr_interaction": {
        "predictors": [
            "is_current",
            "is_last_rewarded",
            "is_prev",
            "is_next",
            "is_swr",
            "swr_x_current",
            "swr_x_rewarded",
            "swr_x_prev",
            "swr_x_next",
        ],
        "level": "arm",
        "split_by_swr": False,
        "description": "Arm-level + SWR interactions (tests if SWR matters)",
    },
}


# ═══════════════════════════════════════════════════════════════════════
# Spatial mapping
# ═══════════════════════════════════════════════════════════════════════


def map_estimate_to_patch(map_estimate, env):
    """Map decoded linear position to patch ID (1, 2, or 3)."""
    map_est = np.asarray(map_estimate, dtype=float)
    bin_df = env.place_bin_centers_nodes_df_
    lp = np.asarray(bin_df["linear_position"])
    ei = np.asarray(bin_df["edge_id"])
    result = np.full(len(map_est), np.nan)
    valid = ~np.isnan(map_est)
    if valid.any():
        idx = np.abs(lp[None, :] - map_est[valid, None]).argmin(axis=1)
        result[valid] = [TRACK_SEGMENT_TO_PATCH.get(int(ei[j]), np.nan) for j in idx]
    return result


def map_estimate_to_arm(map_estimate, env):
    """Map decoded linear position to arm/track segment ID (0-8)."""
    map_est = np.asarray(map_estimate, dtype=float)
    bin_df = env.place_bin_centers_nodes_df_
    lp = np.asarray(bin_df["linear_position"])
    ei = np.asarray(bin_df["edge_id"])
    result = np.full(len(map_est), np.nan)
    valid = ~np.isnan(map_est)
    if valid.any():
        idx = np.abs(lp[None, :] - map_est[valid, None]).argmin(axis=1)
        edges = ei[idx]
        result[valid] = [float(e) if e >= 0 else np.nan for e in edges]
    return result


def well_to_arm_set(well_id, track_graph):
    """Get the set of arm (edge) IDs adjacent to a well node.

    Each well connects to 2 arms. This means is_prev/is_next (which use
    wells) mark 2 of 9 arms as 1, while is_current (which uses
    track_segment_id) marks only 1 arm. Coefficients for is_prev/is_next
    reflect a 2-arm "well neighborhood" effect, not a single-arm effect.
    This asymmetry is intentional: the animal's previous/next location is
    known at the well level, not the arm level.
    """
    if np.isnan(well_id):
        return set()
    well_id = int(well_id)
    arms = set()
    for u, v, data in track_graph.edges(data=True):
        if u == well_id or v == well_id:
            arms.add(data["edge_id"])
    if len(arms) == 0:
        logging.warning(f"No arms found for well {well_id}")
    return arms


# ═══════════════════════════════════════════════════════════════════════
# Trial context
# ═══════════════════════════════════════════════════════════════════════


def add_trial_context(dot, trials):
    """Add trial-derived context at both patch and arm level."""
    dot_times = dot.index.values
    trial_end_times = trials["end_time"].values
    trial_idx = np.searchsorted(trial_end_times, dot_times, side="right")
    n_trials = len(trials)
    to_patches = trials["to_patch"].values
    from_patches = trials["from_patch"].values
    to_wells = trials["to_well"].values
    from_wells = trials["from_well"].values
    is_reward = trials["is_reward"].values
    is_patch_change = trials["is_patch_change"].values

    prev_idx = np.clip(trial_idx - 1, 0, n_trials - 1)
    next_idx = np.clip(trial_idx, 0, n_trials - 1)

    # Patch-level
    dot["prev_trial_from_patch"] = np.where(
        trial_idx > 0, from_patches[prev_idx], np.nan
    )
    dot["next_trial_to_patch"] = np.where(
        trial_idx < n_trials, to_patches[next_idx], np.nan
    )

    # Well-level
    dot["prev_trial_from_well"] = np.where(trial_idx > 0, from_wells[prev_idx], np.nan)
    dot["next_trial_to_well"] = np.where(
        trial_idx < n_trials, to_wells[next_idx], np.nan
    )

    # Most recently rewarded patch and well
    rewarded_end_times = trial_end_times[is_reward]
    rewarded_patches = to_patches[is_reward]
    rewarded_wells = to_wells[is_reward]
    if len(rewarded_end_times) > 0:
        rew_idx = np.searchsorted(rewarded_end_times, dot_times, side="right") - 1
        rew_idx_clipped = np.clip(rew_idx, 0, None)
        dot["last_rewarded_patch"] = np.where(
            rew_idx >= 0, rewarded_patches[rew_idx_clipped], np.nan
        )
        dot["last_rewarded_well"] = np.where(
            rew_idx >= 0, rewarded_wells[rew_idx_clipped], np.nan
        )
    else:
        dot["last_rewarded_patch"] = np.nan
        dot["last_rewarded_well"] = np.nan

    # Task context flags (scalar per timepoint).
    # These use prev_idx (the most recently completed trial):
    # - is_after_switch = 1 during the inter-trial period after a switch trial
    #   completed (not during the switch trial itself)
    # - is_after_omission = 1 after any unrewarded trial (not just the first).
    #   These can overlap: a switch trial that was unrewarded sets both to 1.
    dot["is_after_switch"] = np.where(
        trial_idx > 0, is_patch_change[prev_idx].astype(float), np.nan
    )
    dot["is_after_omission"] = np.where(
        trial_idx > 0, (~is_reward[prev_idx]).astype(float), np.nan
    )

    return dot


# ═══════════════════════════════════════════════════════════════════════
# Event collapsing and long-format construction
# ═══════════════════════════════════════════════════════════════════════


def _modal(series):
    """Mode of a series, or NaN if empty."""
    valid = series.dropna()
    return valid.mode().iloc[0] if len(valid) > 0 else np.nan


def collapse_to_events(nl, level="patch"):
    """Collapse non-local timebins to one row per event."""
    ev = nl[nl["non_local_event"] > 0]
    if len(ev) == 0:
        return pd.DataFrame()

    gcols = ["animal", "day", "epoch", "non_local_event"]

    # Columns to aggregate (mode per event)
    common_cols = ["is_swr_flag", "is_after_switch", "is_after_omission"]
    if level == "patch":
        spatial_cols = [
            "replay_patch",
            "patch_id",
            "prev_trial_from_patch",
            "next_trial_to_patch",
            "last_rewarded_patch",
        ]
    else:
        spatial_cols = [
            "replay_arm",
            "track_segment_id",
            "prev_trial_from_well",
            "next_trial_to_well",
            "last_rewarded_well",
        ]

    # Prepare is_swr as a scalar flag for the event
    ev = ev.copy()
    ev["is_swr_flag"] = (ev["SWR_event"] > 0).astype(float)

    all_cols = spatial_cols + common_cols
    available = [c for c in all_cols if c in ev.columns]

    # Context flags that should use event-onset value, not mode
    onset_cols = {"is_after_switch", "is_after_omission"}

    def _summary(g):
        result = {}
        for col in available:
            if col == "is_swr_flag":
                result[col] = float((g["SWR_event"] > 0).any())
            elif col in onset_cols:
                # Use first timepoint (event onset) for trial-level context
                result[col] = g[col].iloc[0] if g[col].notna().any() else np.nan
            else:
                result[col] = _modal(g[col])
        return pd.Series(result)

    return (
        ev.groupby(gcols)
        .apply(_summary, include_groups=False)
        .dropna(how="all")
        .reset_index()
    )


def make_long_format_patch(event_df):
    """Long format for patch-level model (3 rows per event)."""
    rows = []
    for _, e in event_df.iterrows():
        for p in [1.0, 2.0, 3.0]:
            row = {
                "replay_here": int(e["replay_patch"] == p),
                "is_current": int(e["patch_id"] == p),
                "is_last_rewarded": int(e.get("last_rewarded_patch") == p),
                "is_prev": int(e.get("prev_trial_from_patch") == p),
                "is_next": int(e.get("next_trial_to_patch") == p),
                "is_swr": int(e.get("is_swr_flag", 0)),
                "is_after_switch": float(e.get("is_after_switch", np.nan)),
                "is_after_omission": float(e.get("is_after_omission", np.nan)),
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    _add_interaction_columns(df)
    return df


def make_long_format_arm(event_df, track_graph):
    """Long format for arm-level model (9 rows per event)."""
    all_arms = sorted(set(TRACK_SEGMENT_TO_PATCH.keys()))
    rows = []
    for _, e in event_df.iterrows():
        current_arm = e.get("track_segment_id", np.nan)
        rewarded_arms = well_to_arm_set(
            e.get("last_rewarded_well", np.nan), track_graph
        )
        prev_arms = well_to_arm_set(e.get("prev_trial_from_well", np.nan), track_graph)
        next_arms = well_to_arm_set(e.get("next_trial_to_well", np.nan), track_graph)

        for arm in all_arms:
            row = {
                "replay_here": int(e.get("replay_arm") == arm),
                "is_current": int(current_arm == arm),
                "is_last_rewarded": int(arm in rewarded_arms),
                "is_prev": int(arm in prev_arms),
                "is_next": int(arm in next_arms),
                "is_swr": int(e.get("is_swr_flag", 0)),
                "is_after_switch": float(e.get("is_after_switch", np.nan)),
                "is_after_omission": float(e.get("is_after_omission", np.nan)),
            }
            rows.append(row)
    df = pd.DataFrame(rows)
    _add_interaction_columns(df)
    return df


def _add_interaction_columns(df):
    """Add interaction terms to long-format DataFrame (in place)."""
    df["switch_x_current"] = df["is_after_switch"] * df["is_current"]
    df["switch_x_prev"] = df["is_after_switch"] * df["is_prev"]
    df["omission_x_rewarded"] = df["is_after_omission"] * df["is_last_rewarded"]
    # SWR interactions (for mixed-effects model)
    df["swr_x_current"] = df["is_swr"] * df["is_current"]
    df["swr_x_rewarded"] = df["is_swr"] * df["is_last_rewarded"]
    df["swr_x_prev"] = df["is_swr"] * df["is_prev"]
    if "is_next" in df.columns:
        df["swr_x_next"] = df["is_swr"] * df["is_next"]
    else:
        df["swr_x_next"] = np.zeros(len(df))


# ═══════════════════════════════════════════════════════════════════════
# GLM fitting (per-session)
# ═══════════════════════════════════════════════════════════════════════


def fit_glm(long_df, predictors, alpha=ALPHA):
    """Fit L2-regularized logistic regression with diagnostics."""
    y = long_df["replay_here"].values
    X_raw = long_df[predictors].values.astype(float)
    X = sm.add_constant(X_raw)

    if y.sum() < 3 or (1 - y).sum() < 3 or X.shape[0] < 2 * len(predictors):
        return None

    try:
        model = sm.GLM(y, X, family=sm.families.Binomial())
        result = model.fit_regularized(alpha=alpha, L1_wt=0, refit=False)
        coefs = dict(zip(["intercept"] + predictors, result.params, strict=True))

        y_pred = result.predict(X)
        ll_model = np.sum(
            y * np.log(y_pred + 1e-10) + (1 - y) * np.log(1 - y_pred + 1e-10)
        )
        p_null = y.mean()
        ll_null = np.sum(y * np.log(p_null) + (1 - y) * np.log(1 - p_null))

        # VIF
        vifs = {}
        for i, pred in enumerate(predictors):
            try:
                vifs[f"vif_{pred}"] = variance_inflation_factor(X, i + 1)
            except Exception:
                vifs[f"vif_{pred}"] = np.nan

        # Condition number
        cond_number = float(np.linalg.cond(X.T @ X))

        # Predictor overlap (Jaccard) — only for base predictors
        overlap = {}
        for i, p1 in enumerate(predictors):
            for j, p2 in enumerate(predictors):
                if j <= i:
                    continue
                v1, v2 = X_raw[:, i], X_raw[:, j]
                both = ((v1 == 1) & (v2 == 1)).sum()
                either = ((v1 == 1) | (v2 == 1)).sum()
                overlap[f"jaccard_{p1}__{p2}"] = both / either if either > 0 else 0.0

        # Coefficient correlations from unregularized fit
        coef_corrs = {}
        try:
            unreg = model.fit(disp=0)
            vcov = unreg.cov_params()
            d = np.sqrt(np.diag(vcov))
            corr_mat = vcov / (d[:, None] * d[None, :])
            for i, p1 in enumerate(predictors):
                for j, p2 in enumerate(predictors):
                    if j <= i:
                        continue
                    coef_corrs[f"coef_corr_{p1}__{p2}"] = float(corr_mat[i + 1, j + 1])
        except Exception:
            pass

        # Deviance residuals
        deviance_resid = np.sign(y - y_pred) * np.sqrt(
            -2 * (y * np.log(y_pred + 1e-10) + (1 - y) * np.log(1 - y_pred + 1e-10))
        )

        # Calibration
        pred_bins = np.linspace(0, 1, 6)
        bin_idx = np.digitize(y_pred, pred_bins) - 1
        calib_slope = np.nan
        if len(np.unique(bin_idx)) >= 3:
            obs_rate = [
                y[bin_idx == b].mean()
                for b in range(len(pred_bins) - 1)
                if (bin_idx == b).sum() > 0
            ]
            pred_rate = [
                y_pred[bin_idx == b].mean()
                for b in range(len(pred_bins) - 1)
                if (bin_idx == b).sum() > 0
            ]
            if len(obs_rate) >= 2:
                slope, _, _, _, _ = linregress(pred_rate, obs_rate)
                calib_slope = float(slope)

        # AIC for model comparison (2*k - 2*ll)
        n_params = len(predictors) + 1  # +1 for intercept
        aic = 2 * n_params - 2 * ll_model
        bic = n_params * np.log(len(y)) - 2 * ll_model

        return {
            **{f"coef_{k}": v for k, v in coefs.items()},
            "n_rows": len(long_df),
            "n_params": n_params,
            "ll": float(ll_model),
            "ll_null": float(ll_null),
            "aic": float(aic),
            "bic": float(bic),
            "pseudo_r2": 1 - ll_model / ll_null
            if np.isfinite(ll_null) and ll_null != 0
            else np.nan,
            "brier": float(np.mean((y - y_pred) ** 2)),
            "max_abs_coef": float(np.max(np.abs(list(coefs.values())))),
            "cond_number": cond_number,
            **vifs,
            **overlap,
            **coef_corrs,
            "resid_mean": float(np.mean(deviance_resid)),
            "resid_std": float(np.std(deviance_resid)),
            "resid_skew": float(pd.Series(deviance_resid).skew()),
            "resid_abs_max": float(np.max(np.abs(deviance_resid))),
            "calib_slope": calib_slope,
        }
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════
# Mixed-effects model
# ═══════════════════════════════════════════════════════════════════════


def fit_mixed_effects(long_all, output_dir):
    """Fit mixed-effects logistic regression across all sessions.

    Uses statsmodels BinomialBayesMixedGLM with:
    - Fixed effects: base predictors + is_swr interactions + context interactions
    - Random effects: intercept + is_current slope per session

    Parameters
    ----------
    long_all : pd.DataFrame
        Concatenated long-format data from all sessions (arm level).
    output_dir : Path
        Directory to save results.
    """
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    logging.info("Fitting mixed-effects model...")

    # Need a session ID column
    long_all = long_all.copy()
    long_all["session"] = (
        long_all["animal"].astype(str)
        + "_"
        + long_all["day"].astype(str)
        + "_"
        + long_all["epoch"].astype(str)
    )

    # Drop rows with NaN in key columns
    required = [
        "replay_here",
        "is_current",
        "is_last_rewarded",
        "is_prev",
        "is_next",
        "is_swr",
        "is_after_switch",
        "is_after_omission",
    ]
    long_clean = long_all.dropna(subset=required).copy()

    # Ensure interaction columns exist
    if "switch_x_current" not in long_clean.columns:
        _add_interaction_columns(long_clean)

    n_sessions = long_clean["session"].nunique()
    n_rows = len(long_clean)
    logging.info(f"  {n_rows:,} rows, {n_sessions} sessions")

    if n_rows < 1000 or n_sessions < 5:
        logging.warning("  Insufficient data for mixed-effects model")
        return None

    # Random effects (shared across both models)
    vc_formulas = {
        "session_intercept": "0 + C(session)",
        "session_current_slope": "0 + C(session):is_current",
    }

    # ── Model 1: Base (pooled, no SWR terms) ──
    formula_base = (
        "replay_here ~ is_current + is_last_rewarded + is_prev + is_next"
        " + is_after_switch + switch_x_current + switch_x_prev"
        " + is_after_omission + omission_x_rewarded"
    )

    # ── Model 2: Full (with SWR interaction terms) ──
    formula_full = (
        "replay_here ~ is_current + is_last_rewarded + is_prev + is_next"
        " + swr_x_current + swr_x_rewarded + swr_x_prev + swr_x_next"
        " + is_after_switch + switch_x_current + switch_x_prev"
        " + is_after_omission + omission_x_rewarded"
    )

    try:
        # Fit base model
        logging.info("  Fitting base model (no SWR terms)...")
        model_base = BinomialBayesMixedGLM.from_formula(
            formula_base,
            vc_formulas,
            data=long_clean,
        )
        result_base = model_base.fit_map()
        logging.info("  Base model converged")

        # Fit full model
        logging.info("  Fitting full model (with SWR interactions)...")
        model_full = BinomialBayesMixedGLM.from_formula(
            formula_full,
            vc_formulas,
            data=long_clean,
        )
        result_full = model_full.fit_map()
        logging.info("  Full model converged")

        # Use full model for reporting
        model = model_full
        result = result_full

        # Extract results — BayesMixedGLMResults uses fe_mean/fe_sd
        fe_params = result.fe_mean
        fe_sd = result.fe_sd
        fe_names = model.fep_names

        logging.info("  Mixed-effects model converged")
        logging.info("  Fixed effects (mean ± SD):")
        for name, param, sd in zip(fe_names, fe_params, fe_sd, strict=True):
            logging.info(f"    {name:<30}: {param:.4f} ± {sd:.4f}")

        # Random effects variance
        vc_names = list(vc_formulas.keys())
        vcp_mean = result.vcp_mean
        vcp_sd = result.vcp_sd
        logging.info("  Random effects (log-SD mean ± SD):")
        for name, mean, sd in zip(vc_names, vcp_mean, vcp_sd, strict=True):
            logging.info(
                f"    {name:<30}: log-SD={mean:.4f} ± {sd:.4f}, SD={np.exp(mean):.4f}"
            )

        # ── Model comparison: base vs full ──
        logging.info("\n  MODEL COMPARISON: Base (no SWR) vs Full (with SWR)")
        base_fe_names = model_base.fep_names
        n_base = len(base_fe_names)
        n_full = len(fe_names)
        logging.info(f"    Base model: {n_base} fixed effects")
        logging.info(f"    Full model: {n_full} fixed effects")

        # Compare using log-likelihood at MAP (from the objective function)
        # BayesMixedGLM optimizes the log-posterior; we use the params to
        # compute the log-likelihood on the data for comparison
        def _compute_ll(result_obj, model_obj, data):
            """Compute log-likelihood from predicted probabilities."""
            y = data["replay_here"].values
            y_pred = result_obj.predict(model_obj.exog)
            return float(
                np.sum(
                    y * np.log(y_pred + 1e-10) + (1 - y) * np.log(1 - y_pred + 1e-10)
                )
            )

        ll_base = _compute_ll(result_base, model_base, long_clean)
        ll_full = _compute_ll(result_full, model_full, long_clean)
        lr_stat = -2 * (ll_base - ll_full)
        df_diff = n_full - n_base

        from scipy.stats import chi2

        lr_pvalue = chi2.sf(lr_stat, df=df_diff) if lr_stat > 0 else 1.0

        aic_base = 2 * n_base - 2 * ll_base
        aic_full = 2 * n_full - 2 * ll_full
        bic_base = n_base * np.log(n_rows) - 2 * ll_base
        bic_full = n_full * np.log(n_rows) - 2 * ll_full

        logging.info(f"    LL base: {ll_base:.1f}, LL full: {ll_full:.1f}")
        logging.info(f"    LR stat: {lr_stat:.1f}, df={df_diff}, p={lr_pvalue:.2e}")
        logging.info(
            f"    AIC base: {aic_base:.1f}, AIC full: {aic_full:.1f}, ΔAIC: {aic_base - aic_full:.1f}"
        )
        logging.info(
            f"    BIC base: {bic_base:.1f}, BIC full: {bic_full:.1f}, ΔBIC: {bic_base - bic_full:.1f}"
        )

        # Switch/omission co-occurrence diagnostic
        if (
            "is_after_switch" in long_clean.columns
            and "is_after_omission" in long_clean.columns
        ):
            both = (
                (long_clean["is_after_switch"] == 1)
                & (long_clean["is_after_omission"] == 1)
            ).mean()
            logging.info(f"  Switch+omission co-occurrence: {both * 100:.1f}% of rows")

        # Save results
        me_results = {
            "fe_names": list(fe_names),
            "fe_mean": list(fe_params),
            "fe_sd": list(fe_sd),
            "vc_names": vc_names,
            "vcp_mean": list(vcp_mean),
            "vcp_sd": list(vcp_sd),
            "vc_sd_exp": [float(np.exp(m)) for m in vcp_mean],
            "n_rows": n_rows,
            "n_sessions": n_sessions,
            "formula_base": formula_base,
            "formula_full": formula_full,
            # Model comparison
            "ll_base": ll_base,
            "ll_full": ll_full,
            "lr_stat": lr_stat,
            "lr_pvalue": lr_pvalue,
            "aic_base": aic_base,
            "aic_full": aic_full,
            "bic_base": bic_base,
            "bic_full": bic_full,
            "delta_aic": aic_base - aic_full,
            "delta_bic": bic_base - bic_full,
        }

        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "mixed_effects_results.pkl", "wb") as f:
            pickle.dump(me_results, f)
        logging.info(f"  Saved to {output_dir / 'mixed_effects_results.pkl'}")

        return me_results

    except Exception as e:
        logging.error(f"  Mixed-effects model failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════
# Session processing
# ═══════════════════════════════════════════════════════════════════════


def find_sessions(max_per_animal=None):
    """Find all sessions with data_over_time and environment files."""
    dot_files = sorted(PROCESSED_DATA_DIR.glob("*_HPC_80_data_over_time.pkl"))
    env_files = {
        f.stem.replace("_environment", ""): f
        for f in PROCESSED_DATA_DIR.glob("*_HPC_environment.pkl")
    }

    by_animal = {}
    for f in dot_files:
        match = re.match(r"(\w+\d{8})_(\d+_r\d+)_HPC_80_data_over_time\.pkl", f.name)
        if not match:
            continue
        animal_date, epoch = match.group(1), match.group(2)
        animal = re.match(r"([a-z]+)", animal_date).group(1)
        session_id = f"{animal_date}_{epoch}_HPC"
        if session_id not in env_files:
            continue

        nwb_match = [nwb for nwb in NWB_FILES if animal_date in nwb]
        if not nwb_match:
            continue

        by_animal.setdefault(animal, []).append(
            {
                "nwb_file_name": nwb_match[0],
                "epoch_name": epoch,
                "dot_file": f,
                "env_file": env_files[session_id],
            }
        )

    selected = []
    for animal in sorted(by_animal):
        sessions = by_animal[animal]
        if max_per_animal is not None:
            sessions = sessions[:max_per_animal]
        selected.extend(sessions)

    return selected


def process_session(session):
    """Load data, build events, fit all per-session GLM models."""
    nwb = session["nwb_file_name"]
    epoch = session["epoch_name"]

    dot = pd.read_pickle(session["dot_file"])
    with open(session["env_file"], "rb") as f:
        env = pickle.load(f)

    data = load_data(nwb, epoch, include=frozenset({SessionDataKeys.TRIALS}))
    trials = data.get("trials")
    if trials is None or len(trials) < 3:
        return [], None

    track_graph = data.get("track_graph")

    # Map decoded position at both levels
    dot["replay_patch"] = map_estimate_to_patch(dot["map_estimate"], env)
    dot["replay_arm"] = map_estimate_to_arm(dot["map_estimate"], env)
    dot["Non-Local_prob"] = (
        dot["Non-Local_Continuous_prob"] + dot["Non-Local_Fragmented_prob"]
    )
    dot["abs_distance"] = dot["ahead_behind_distance"].abs()

    dot = add_trial_context(dot, trials)

    # Filter
    nl = dot[
        (dot["head_speed"] < DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC)
        & (dot["Non-Local_prob"] > DEFAULT_NON_LOCAL_THRESHOLD)
    ].copy()

    if len(nl) < 100:
        return [], None

    nl["is_swr"] = nl["SWR_event"] > 0

    animal = nl["animal"].iloc[0]
    day = nl["day"].iloc[0]
    epoch_val = nl["epoch"].iloc[0]
    meta = {"animal": animal, "day": day, "epoch": epoch_val}

    per_session_results = []
    arm_long_for_mixed = None  # will collect for mixed-effects

    for model_name, spec in MODEL_SPECS.items():
        level = spec["level"]
        predictors = spec["predictors"]

        events = collapse_to_events(nl, level=level)
        if len(events) < 20:
            continue

        if level == "patch":
            long = make_long_format_patch(events)
        else:
            if track_graph is None:
                continue
            long = make_long_format_arm(events, track_graph)

        long = long.dropna(subset=predictors)

        # For models with interaction terms, also drop NaN in those columns
        interaction_cols = [p for p in predictors if "_x_" in p or "after" in p]
        if interaction_cols:
            long = long.dropna(subset=interaction_cols)

        n_spatial = 3 if level == "patch" else 9
        split_by_swr = spec.get("split_by_swr", True)

        if split_by_swr:
            # Fit separately for SWR and non-SWR
            for etype_label, etype_val in [("SWR", 1), ("Non-SWR", 0)]:
                sub = long[long["is_swr"] == etype_val]
                if len(sub) < 2 * len(predictors):
                    continue

                r = fit_glm(sub, predictors=predictors)
                if r is not None:
                    r.update(
                        {
                            **meta,
                            "event_type": etype_label,
                            "model": model_name,
                            "n_events": len(sub) // n_spatial,
                        }
                    )
                    per_session_results.append(r)
        else:
            # Fit on pooled data (both SWR and non-SWR together)
            if len(long) < 2 * len(predictors):
                continue

            r = fit_glm(long, predictors=predictors)
            if r is not None:
                r.update(
                    {
                        **meta,
                        "event_type": "pooled",
                        "model": model_name,
                        "n_events": len(long) // n_spatial,
                    }
                )
                per_session_results.append(r)

        # Collect arm-level long data for mixed-effects (base model, both SWR types)
        if model_name == "arm" and len(long) > 0:
            long_with_meta = long.copy()
            long_with_meta["animal"] = animal
            long_with_meta["day"] = day
            long_with_meta["epoch"] = epoch_val
            arm_long_for_mixed = long_with_meta

    del dot, nl
    return per_session_results, arm_long_for_mixed


# ═══════════════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════════════


def print_results(gdf):
    """Print results for each per-session model."""
    for model_name, spec in MODEL_SPECS.items():
        mdf = gdf[gdf["model"] == model_name]
        if len(mdf) == 0:
            continue

        predictors = spec["predictors"]
        print(f"\n{'#' * 70}")
        print(f"MODEL: {model_name} — {spec['description']}")
        print(f"{'#' * 70}")

        # Determine event types present in this model
        event_types = sorted(mdf["event_type"].unique())

        # Fit quality
        print(f"\nFIT QUALITY (alpha={ALPHA})")
        for et in event_types:
            s = mdf[mdf["event_type"] == et]
            if len(s) == 0:
                continue
            aic_str = f", AIC={s['aic'].median():.0f}" if "aic" in s.columns else ""
            print(
                f"  {et} (n={len(s)}): "
                f"R²={s['pseudo_r2'].median():.4f}, "
                f"Brier={s['brier'].median():.4f}, "
                f"Max|c|={s['max_abs_coef'].median():.2f}"
                f"{aic_str}"
            )

        # Coefficients with BH-FDR correction
        print("\nCOEFFICIENTS (p-values: Benjamini-Hochberg FDR corrected)")

        # Collect all raw p-values first
        raw_pvals = []  # (pred, test_name, p_value)
        coef_data = {}

        is_split = spec.get("split_by_swr", True)

        if is_split:
            for pred in predictors:
                col = f"coef_{pred}"
                if col not in mdf.columns:
                    continue
                sv = mdf.loc[mdf["event_type"] == "SWR", col].dropna()
                nv = mdf.loc[mdf["event_type"] == "Non-SWR", col].dropna()
                coef_data[pred] = {"sv": sv, "nv": nv}
                if len(sv) >= 5:
                    try:
                        _, p = wilcoxon(sv)
                        raw_pvals.append((pred, "swr", p))
                    except ValueError:
                        pass
                if len(nv) >= 5:
                    try:
                        _, p = wilcoxon(nv)
                        raw_pvals.append((pred, "noswr", p))
                    except ValueError:
                        pass
                if len(sv) >= 5 and len(nv) >= 5:
                    _, p = mannwhitneyu(sv, nv)
                    raw_pvals.append((pred, "compare", p))
        else:
            # Pooled model — single column
            for pred in predictors:
                col = f"coef_{pred}"
                if col not in mdf.columns:
                    continue
                vals = mdf.loc[mdf["event_type"] == "pooled", col].dropna()
                coef_data[pred] = {"pooled": vals}
                if len(vals) >= 5:
                    try:
                        _, p = wilcoxon(vals)
                        raw_pvals.append((pred, "pooled", p))
                    except ValueError:
                        pass

        # BH-FDR correction
        from statsmodels.stats.multitest import multipletests

        if raw_pvals:
            pvals_array = [x[2] for x in raw_pvals]
            _, corrected, _, _ = multipletests(pvals_array, method="fdr_bh")
            corrected_map = {
                (x[0], x[1]): cp for x, cp in zip(raw_pvals, corrected, strict=True)
            }
        else:
            corrected_map = {}

        if is_split:
            print(
                f"  {'Pred':<25} {'SWR':>8} {'noSWR':>8} "
                f"{'SWR!=0':>10} {'noSWR!=0':>10} {'compare':>10}"
            )
            print("  " + "-" * 75)
            for pred in predictors:
                if pred not in coef_data:
                    continue
                sv = coef_data[pred]["sv"]
                nv = coef_data[pred]["nv"]
                sp = corrected_map.get((pred, "swr"), None)
                np_p = corrected_map.get((pred, "noswr"), None)
                cp = corrected_map.get((pred, "compare"), None)
                sp_str = f"{sp:.2e}" if sp is not None else "—"
                np_str = f"{np_p:.2e}" if np_p is not None else "—"
                cp_str = f"{cp:.2e}" if cp is not None else "—"
                smed = f"{sv.median():.3f}" if len(sv) > 0 else "—"
                nmed = f"{nv.median():.3f}" if len(nv) > 0 else "—"
                print(
                    f"  {pred:<25} {smed:>8} {nmed:>8} "
                    f"{sp_str:>10} {np_str:>10} {cp_str:>10}"
                )
        else:
            print(f"  {'Pred':<25} {'median':>8} {'!=0 p':>10}")
            print("  " + "-" * 45)
            for pred in predictors:
                if pred not in coef_data:
                    continue
                vals = coef_data[pred]["pooled"]
                p_corr = corrected_map.get((pred, "pooled"), None)
                p_str = f"{p_corr:.2e}" if p_corr is not None else "—"
                med = f"{vals.median():.3f}" if len(vals) > 0 else "—"
                print(f"  {pred:<25} {med:>8} {p_str:>10}")

        # Key diagnostics (compact)
        print("\n  DIAGNOSTICS")
        for et in event_types:
            s = mdf[mdf["event_type"] == et]
            if len(s) == 0:
                continue
            # VIF
            vif_strs = []
            for pred in predictors[:4]:  # only base predictors
                col = f"vif_{pred}"
                if col in s.columns:
                    vals = s[col].dropna()
                    vals = vals[np.isfinite(vals)]
                    if len(vals) > 0:
                        vif_strs.append(
                            f"{pred.replace('is_', '')}={vals.median():.1f}"
                        )
            if vif_strs:
                print(f"    {et} VIF: {', '.join(vif_strs)}")
            # Condition number
            if "cond_number" in s.columns:
                cn = s["cond_number"].dropna()
                if len(cn) > 0:
                    print(f"    {et} Cond#: median={cn.median():.0f}")
            # Residuals
            if "resid_skew" in s.columns:
                sk = s["resid_skew"].dropna()
                cs = s.get("calib_slope", pd.Series(dtype=float)).dropna()
                if len(sk) > 0:
                    cs_str = f", calib={cs.median():.2f}" if len(cs) > 0 else ""
                    print(f"    {et} Resid: skew={sk.median():.3f}{cs_str}")

    # ── Model comparison: arm_pooled vs arm_swr_interaction ──
    pooled = gdf[gdf["model"] == "arm_pooled"]
    swr_int = gdf[gdf["model"] == "arm_swr_interaction"]
    if len(pooled) > 0 and len(swr_int) > 0:
        print(f"\n{'#' * 70}")
        print("MODEL COMPARISON: Does SWR status improve prediction?")
        print("arm_pooled (null) vs arm_swr_interaction (alternative)")
        print(f"{'#' * 70}")

        session_cols_mc = ["animal", "day", "epoch"]
        merged = pooled.merge(
            swr_int,
            on=session_cols_mc,
            suffixes=("_pooled", "_swr"),
        )

        if len(merged) > 0:
            # AIC comparison
            merged["delta_aic"] = merged["aic_pooled"] - merged["aic_swr"]
            merged["delta_bic"] = merged["bic_pooled"] - merged["bic_swr"]

            # Likelihood ratio test (LRT):
            # LR = -2 * (ll_null - ll_alt), df = n_params_alt - n_params_null
            merged["lr_stat"] = -2 * (merged["ll_pooled"] - merged["ll_swr"])
            df_diff = merged["n_params_swr"].iloc[0] - merged["n_params_pooled"].iloc[0]
            from scipy.stats import chi2

            merged["lr_pvalue"] = chi2.sf(merged["lr_stat"], df=df_diff)

            print(f"\n  Sessions compared: {len(merged)}")
            print(f"  Extra parameters in SWR model: {df_diff}")

            print("\n  ΔAIC (pooled - SWR interaction, positive = SWR model better):")
            daic = merged["delta_aic"]
            print(
                f"    median={daic.median():.1f}, "
                f"IQR=[{daic.quantile(0.25):.1f}, {daic.quantile(0.75):.1f}]"
            )
            print(
                f"    Sessions favoring SWR model (ΔAIC > 2): "
                f"{(daic > 2).sum()}/{len(daic)}"
            )
            print(
                f"    Sessions favoring pooled (ΔAIC < -2): "
                f"{(daic < -2).sum()}/{len(daic)}"
            )

            print("\n  ΔBIC (pooled - SWR interaction, positive = SWR model better):")
            dbic = merged["delta_bic"]
            print(
                f"    median={dbic.median():.1f}, "
                f"IQR=[{dbic.quantile(0.25):.1f}, {dbic.quantile(0.75):.1f}]"
            )

            print("\n  Likelihood ratio test:")
            print(
                f"    LR stat: median={merged['lr_stat'].median():.1f}, "
                f"IQR=[{merged['lr_stat'].quantile(0.25):.1f}, {merged['lr_stat'].quantile(0.75):.1f}]"
            )
            sig = (merged["lr_pvalue"] < 0.05).sum()
            print(f"    Sessions significant (p < 0.05): {sig}/{len(merged)}")

            # Wilcoxon on delta_aic across sessions
            if len(daic) >= 5:
                _, p = wilcoxon(daic)
                print(
                    f"    Wilcoxon on ΔAIC: p={p:.2e} "
                    f"({'SWR model wins' if daic.median() > 0 else 'pooled wins'})"
                )


def make_figure(gdf, output_dir):
    """Create figure comparing all models."""
    models_present = [m for m in MODEL_SPECS if (gdf["model"] == m).any()]
    n_models = len(models_present)
    if n_models == 0:
        return

    fig, axes = plt.subplots(
        n_models,
        3,
        figsize=(7, 3 * n_models),
        constrained_layout=True,
        squeeze=False,
    )

    for row, model_name in enumerate(models_present):
        mdf = gdf[gdf["model"] == model_name]
        predictors = MODEL_SPECS[model_name]["predictors"]

        # Panel A: Coefficients
        ax = axes[row, 0]
        plot_data = []
        for pred in predictors:
            col = f"coef_{pred}"
            if col not in mdf.columns:
                continue
            for et in ["SWR", "Non-SWR"]:
                for v in mdf.loc[mdf["event_type"] == et, col].dropna():
                    plot_data.append(
                        {
                            "predictor": pred.replace("is_", "").replace("_x_", "×"),
                            "event_type": et,
                            "coef": v,
                        }
                    )
        pdf = pd.DataFrame(plot_data)
        if len(pdf) > 0:
            sns.boxplot(
                data=pdf,
                x="predictor",
                y="coef",
                hue="event_type",
                palette={"SWR": COLORS["SWR"], "Non-SWR": COLORS["Non-Local"]},
                ax=ax,
                width=0.6,
                fliersize=1,
            )
        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_ylabel("Log-odds")
        ax.set_xlabel("")
        ax.set_title(f"{model_name}", fontsize=7)
        ax.legend(fontsize=4, frameon=False, title="")
        ax.tick_params(axis="x", rotation=45, labelsize=4)
        sns.despine(ax=ax)

        # Panel B: VIF (base predictors only)
        ax = axes[row, 1]
        base_preds = [p for p in predictors if "_x_" not in p and "after" not in p]
        vif_data = []
        for et in ["SWR", "Non-SWR"]:
            s = mdf[mdf["event_type"] == et]
            for pred in base_preds:
                col = f"vif_{pred}"
                if col in s.columns:
                    for v in s[col].dropna():
                        if np.isfinite(v):
                            vif_data.append(
                                {
                                    "pred": pred.replace("is_", ""),
                                    "et": et,
                                    "vif": v,
                                }
                            )
        vdf = pd.DataFrame(vif_data)
        if len(vdf) > 0:
            sns.boxplot(
                data=vdf,
                x="pred",
                y="vif",
                hue="et",
                palette={"SWR": COLORS["SWR"], "Non-SWR": COLORS["Non-Local"]},
                ax=ax,
                width=0.6,
                fliersize=1,
            )
        ax.axhline(5, color="red", linestyle="--", linewidth=0.5)
        ax.set_ylabel("VIF")
        ax.set_xlabel("")
        ax.set_title(f"{model_name}: VIF", fontsize=7)
        ax.legend(fontsize=4, frameon=False, title="")
        ax.tick_params(axis="x", rotation=30, labelsize=5)
        sns.despine(ax=ax)

        # Panel C: Pseudo R²
        ax = axes[row, 2]
        for et, c in [("SWR", COLORS["SWR"]), ("Non-SWR", COLORS["Non-Local"])]:
            vals = mdf.loc[mdf["event_type"] == et, "pseudo_r2"].dropna()
            if len(vals) > 0:
                ax.hist(vals, bins=20, alpha=0.6, color=c, label=et, edgecolor="none")
        ax.set_xlabel("Pseudo R²")
        ax.set_ylabel("Count")
        ax.set_title(f"{model_name}: Fit Quality", fontsize=7)
        ax.legend(fontsize=5, frameon=False)
        sns.despine(ax=ax)

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "glm_results.png", dpi=300)
    plt.close()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--test", action="store_true", help="2 sessions per animal")
    parser.add_argument("--max-per-animal", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--skip-mixed", action="store_true", help="Skip mixed-effects model (faster)"
    )
    args = parser.parse_args()

    max_per_animal = 2 if args.test else args.max_per_animal
    sessions = find_sessions(max_per_animal=max_per_animal)
    print(f"Found {len(sessions)} sessions")

    t0 = time.time()
    all_results = []
    all_arm_long = []  # for mixed-effects

    for i, session in enumerate(sessions):
        try:
            per_session, arm_long = process_session(session)
            all_results.extend(per_session)
            if arm_long is not None:
                all_arm_long.append(arm_long)
        except Exception as e:
            print(f"  Skip {session['nwb_file_name']} {session['epoch_name']}: {e}")
            continue

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            remaining = elapsed / (i + 1) * (len(sessions) - i - 1)
            print(
                f"  [{i + 1}/{len(sessions)}] {elapsed:.0f}s elapsed, "
                f"~{remaining:.0f}s remaining"
            )

    total_time = time.time() - t0
    print(f"\nProcessed {len(sessions)} sessions in {total_time:.0f}s")

    # Per-session results
    gdf = pd.DataFrame(all_results)
    print(f"Per-session fits: {len(gdf)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gdf.to_pickle(args.output_dir / "glm_coefficients.pkl")
    print(f"Saved to {args.output_dir / 'glm_coefficients.pkl'}")

    print_results(gdf)
    make_figure(gdf, args.output_dir)
    print(f"Saved figure to {args.output_dir / 'glm_results.png'}")

    # Mixed-effects model
    if not args.skip_mixed and len(all_arm_long) >= 5:
        print(f"\n{'=' * 70}")
        print("MIXED-EFFECTS MODEL")
        print(f"{'=' * 70}")
        long_all = pd.concat(all_arm_long, ignore_index=True)
        print(f"Total rows for mixed model: {len(long_all):,}")
        fit_mixed_effects(long_all, args.output_dir)
    elif args.skip_mixed:
        print("\nSkipped mixed-effects model (--skip-mixed)")
    else:
        print("\nInsufficient data for mixed-effects model")


if __name__ == "__main__":
    main()
