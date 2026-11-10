#!/usr/bin/env python
"""SCRATCH: summarize the replay-content model fits across all epochs.

Generalizes the per-epoch view in ``notebooks/single_epoch_walkthrough.py``
(AIC across the hierarchy + per-predictor coefficients) to the whole dataset,
for the arm hierarchy (M0–M7) and the rate hierarchy (R0–R3).

Per block, three panels:
  A. selection metric across the hierarchy — one faint line per epoch + bold
     median. Arm: cross-validated LL per event (higher = better, honest about
     overfitting); rate: AIC (lower = better; CV not computed for the rate
     Poisson).
  B. per-step improvement (Δ of the metric) as a box across epochs, annotated
     with % of epochs improving and a Wilcoxon p-value.
  C. coefficient forest at the full model — per-epoch coefficient distribution
     (median ± IQR, points colored by sign), with % of epochs significant.

Arm fits are loaded from the saved per-session table
(``replay_content/models/all_models.pkl``); rate fits are computed per epoch
from the saved ``stop_table.pkl`` and cached to ``scratch/``.

Output: ``scratch/modelfit_summary.png`` (gitignored).
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from continuum_swr_replay.figure_utilities import save_figure
from continuum_swr_replay.paths import REPLAY_CONTENT_DIR, model_dir
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
warnings.filterwarnings("ignore")

RC = REPLAY_CONTENT_DIR
SCRATCH = Path("scratch")
SESSION = list(SESSION_COLUMNS)

ARM_ORDER = ["M0", "M0a", "M1", "M2", "M3", "M4", "M5", "M6", "M7"]
# Predictor introduced at each arm model (for the coefficient forest at M7).
ARM_PREDICTORS = [
    "is_current_arm",
    "is_prev_arm",
    "reward_fraction_candidate_arm_last10_visits",
    "visit_recency_trials_event_leaf_centered",
    "is_last_rewarded_noncurrent_arm",
    "swr_x_is_current_arm",
    "swr_x_is_prev_arm",
    "swr_x_reward_fraction_candidate_arm_last10_visits",
    "swr_x_visit_recency_trials_event_leaf_centered",
    "swr_x_is_last_rewarded_noncurrent_arm",
]
PATCH_ORDER = ["P0", "P1", "P2", "P3", "P4"]
PATCH_PREDICTORS = [
    "is_current_patch",
    "is_last_rewarded_noncurrent_patch",
    "reward_fraction_candidate_arm_last10_visits_patch",
    "swr_x_is_current_patch",
    "swr_x_is_last_rewarded_noncurrent_patch",
]
RATE_ORDER = ["R0", "R1", "R2"]
RATE_PREDICTORS = [
    "known_reward_outcome_lag_1",
    "reward_fraction_last10_known_trials",
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_arm_fits() -> pd.DataFrame:
    df = joblib.load(model_dir("content_arm", RC) / "all_models.pkl")
    return df[df["model"].isin(ARM_ORDER)].copy()


def load_patch_fits() -> pd.DataFrame:
    """Per-session P0–P4 fits with CV-LL (persisted by fit_replay_content_models --cv)."""
    return joblib.load(model_dir("content_patch", RC) / "all_models.pkl")


def load_rate_fits() -> pd.DataFrame:
    """Per-session R0–R3 fits with CV-LL (persisted by fit_rate_models.py)."""
    return joblib.load(RC / "models_rate" / "all_models.pkl")


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def _wide(df: pd.DataFrame, value: str, order: list[str]) -> pd.DataFrame:
    """(epoch × model) wide frame of ``value``, columns ordered by ``order``."""
    w = df.pivot_table(index=SESSION, columns="model", values=value)
    return w[[m for m in order if m in w.columns]]


def panel_trajectory(ax, df, value, order, ylabel, higher_better):
    """Per-epoch metric improvement across the hierarchy, relative to baseline.

    Absolute CV-LL / AIC vary by an order of magnitude across epochs (event
    counts, decoder quality), which masks the within-epoch improvement. Plot
    each epoch relative to its own baseline model (first in ``order``) and
    orient so higher = better, so the common improvement shape is visible.
    """
    w = _wide(df, value, order).dropna(how="all")
    base = w.iloc[:, 0]
    rel = w.sub(base, axis=0)
    if not higher_better:
        rel = -rel  # AIC: reduction is improvement → positive
    x = np.arange(rel.shape[1])
    ax.plot(x, rel.to_numpy().T, color="0.8", lw=0.4, alpha=0.5)
    ax.plot(x, rel.median(axis=0).to_numpy(), "o-", color="C3", lw=2, label="median")
    ax.axhline(0, color="0.6", lw=0.8)
    # Robust y-limits so a few extreme epochs don't flatten the rest.
    flat = rel.to_numpy().ravel()
    flat = flat[np.isfinite(flat)]
    lo, hi = np.percentile(flat, [1, 99])
    ax.set_ylim(min(lo, -1), hi * 1.1 + 1)
    ax.set_xticks(x)
    ax.set_xticklabels(w.columns, fontsize=8)
    ax.set_ylabel(f"{ylabel}\nimprovement vs {order[0]}", fontsize=8)
    ax.set_title("cumulative improvement across hierarchy (↑ better)", fontsize=9)
    ax.legend(fontsize=7)


def panel_step_delta(ax, df, value, order, higher_better):
    """Per-step Δ(metric) box across epochs + %improving and Wilcoxon p."""
    w = _wide(df, value, order)
    steps = list(w.columns)
    deltas, labels = [], []
    for i in range(1, len(steps)):
        d = (w[steps[i]] - w[steps[i - 1]]).dropna()
        if not higher_better:
            d = -d  # so positive always means "improved"
        deltas.append(d.to_numpy())
        labels.append(f"{steps[i - 1]}→{steps[i]}")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.boxplot(deltas, showfliers=False, widths=0.6)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Δ metric (improvement +)", fontsize=8)
    ax.set_title("per-step improvement", fontsize=9)
    ymax = max((np.percentile(d, 90) if len(d) else 0) for d in deltas)
    for i, d in enumerate(deltas, start=1):
        if len(d) < 3:
            continue
        frac = float(np.mean(d > 0))
        try:
            p = wilcoxon(d)[1]
        except ValueError:
            p = np.nan
        ax.annotate(
            f"{100 * frac:.0f}%\np={p:.1g}",
            (i, ymax),
            fontsize=5,
            ha="center",
            va="bottom",
        )


def _star(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""


def panel_coef_heatmap(ax, df, models, predictors, title):
    """Pooled coefficient across the hierarchy as a predictor × model heatmap.

    Each cell is the pooled (median-across-epochs) multiplicative effect
    exp(β); blank where the predictor isn't in that model. The cell color is
    β = ln(exp(β)) on a diverging scale centered at 0 (× = 1, no effect), and
    the text is ``×<factor>`` plus a Wilcoxon signed-rank significance star
    (median ≠ 0 across epochs, Benjamini–Hochberg FDR-corrected over all cells)
    — the population pooling the pipeline uses.
    """
    n_p, n_m = len(predictors), len(models)
    beta = np.full((n_p, n_m), np.nan)
    raw_p, cells = [], []
    for i, pred in enumerate(predictors):
        for j, model in enumerate(models):
            coefs = df[df["model"] == model].get(f"coef_{pred}")
            if coefs is None:
                continue
            coefs = coefs.dropna().to_numpy()
            if len(coefs) < 3:
                continue
            beta[i, j] = np.median(coefs)
            try:
                raw_p.append(wilcoxon(coefs)[1])
            except ValueError:
                raw_p.append(np.nan)
            cells.append((i, j))
    # FDR across all populated cells.
    fdr = {}
    valid = [(c, p) for c, p in zip(cells, raw_p, strict=True) if np.isfinite(p)]
    if valid:
        corr = multipletests([p for _, p in valid], method="fdr_bh")[1]
        fdr = {c: pc for (c, _), pc in zip(valid, corr, strict=True)}

    vmax = np.nanmax(np.abs(beta)) if np.isfinite(beta).any() else 1.0
    im = ax.imshow(beta, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for i, j in cells:
        star = _star(fdr.get((i, j), 1.0))
        ax.text(
            j,
            i,
            f"×{np.exp(beta[i, j]):.2f}\n{star}",
            ha="center",
            va="center",
            fontsize=5,
            color="white" if abs(beta[i, j]) > 0.6 * vmax else "black",
        )
    ax.set_xticks(range(n_m))
    ax.set_xticklabels(models, fontsize=7)
    ax.set_yticks(range(n_p))
    ax.set_yticklabels(predictors, fontsize=6)
    ax.set_title(title, fontsize=9)
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("β = ln(×)", fontsize=6)
    cb.ax.tick_params(labelsize=5)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    arm = load_arm_fits()
    patch = load_patch_fits()
    rate = load_rate_fits()
    n = {
        k: v.groupby(SESSION).ngroups
        for k, v in (("candidate_arm", arm), ("patch", patch), ("rate", rate))
    }
    logging.info("epochs — arm %(arm)d, patch %(patch)d, rate %(rate)d", n)

    fig, axs = plt.subplots(
        3, 3, figsize=(16, 13), dpi=150, gridspec_kw={"width_ratios": [1.1, 1.1, 1.3]}
    )

    # All three blocks now use cross-validated LL (higher = better) as the
    # selection metric. Arm/patch: held-out per-event conditional LL; rate:
    # held-out per-stop Poisson LL.
    blocks = [
        (
            0,
            arm,
            "cv_ll_per_event_mean",
            "CV LL / event",
            ARM_ORDER,
            ARM_ORDER[2:],
            ARM_PREDICTORS,
            "ARM pooled exp(β): content relative-rate (M1–M7)",
            "ARM  M0–M7",
        ),
        (
            1,
            patch,
            "cv_ll_per_event_mean",
            "CV LL / event",
            PATCH_ORDER,
            PATCH_ORDER[1:],
            PATCH_PREDICTORS,
            "PATCH pooled exp(β): content relative-rate (P1–P4)",
            "PATCH  P0–P4",
        ),
        (
            2,
            rate,
            "cv_ll_per_stop_mean",
            "CV LL / stop",
            RATE_ORDER,
            RATE_ORDER[1:],
            RATE_PREDICTORS,
            "RATE pooled exp(β): event-rate ratio (R1–R3)",
            "RATE  R0–R3",
        ),
    ]
    for r, df, metric, ylabel, order, coef_models, preds, heat_title, tag in blocks:
        panel_trajectory(axs[r, 0], df, metric, order, ylabel, higher_better=True)
        panel_step_delta(axs[r, 1], df, metric, order, higher_better=True)
        panel_coef_heatmap(axs[r, 2], df, coef_models, preds, heat_title)
        which = {"candidate_arm": "candidate_arm", "patch": "patch", "rate": "rate"}[
            tag.split()[0].lower()
        ]
        axs[r, 0].annotate(
            f"{tag}  (n={n[which]} epochs)",
            (-0.15, 1.13),
            xycoords="axes fraction",
            fontsize=12,
            fontweight="bold",
        )

    fig.tight_layout()
    SCRATCH.mkdir(exist_ok=True)
    out = SCRATCH / "modelfit_summary.png"
    save_figure(
        fig, out.stem, base_dir=out.parent, dpi=150, formats=["png"], transparent=False
    )
    logging.info("Saved %s", out)


if __name__ == "__main__":
    np.random.seed(0)
    main()
