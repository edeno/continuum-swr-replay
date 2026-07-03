"""Coefficient figures for the is_swr rate + content contrast ladders.

For each effect: every session's coefficient is a faint jittered dot (colored by
animal); the 5 larger dots are per-animal medians; the black bar is the
across-animal median; the horizontal whisker is its 95% CI (bootstrap over the 5
per-animal medians -- the animal is the inferential unit, n=5). Continuous-predictor
coefficients are shown STANDARDIZED (per 1 SD of the predictor: raw_coef * SD, which
is exact for these Poisson GLMs) so effect sizes are comparable across predictors,
value constructs, and the two ladders. Binary-predictor terms (marked with a dagger)
are per-category and are not standardized.

Covariate SDs are computed here directly from the source tables (global/pooled),
so the figure is self-contained. Content state-modulation rungs (rpe/relval/entropy/
update/tss/block/switchrate x current) are already z-scored at fit time, so their
standardization factor is 1.0.

    uv run python scripts/figures/plot_swr_contrast_coefs.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

RC = PROCESSED_DATA_DIR / "replay_content"
OUT = Path("paper/figures/swr_contrast")
OUT.mkdir(parents=True, exist_ok=True)


def _global_sd(covariates):
    """Global (pooled) SD of each continuous covariate from its source table."""
    from continuum_swr_replay.analysis.replay._helpers import add_uncertainty_covariates

    arm = pd.read_pickle(RC / "event_arm_table.pkl")
    qv = pd.read_pickle(RC / "models_per_bin_rate" / "qvals_table.pkl")
    stop = add_uncertainty_covariates(pd.read_pickle(RC / "stop_table.pkl"))
    sd = {}
    for c in covariates:
        for src in (arm, qv, stop):
            if c in src.columns:
                sd[c] = float(np.nanstd(src[c].to_numpy(dtype=float)))
                break
    return sd


SD = _global_sd([
    "q_arm", "q_arm_hmm", "reward_rate_arm_k10", "p_reward_arm",
    "hmm_relative_value", "hmm_rpe", "reward_entropy", "hmm_value_update",
    "trials_since_switch", "trials_since_block_change", "recent_switch_rate",
])
ANIM = ["chimi", "j16", "peanut", "senor", "wilbur"]
CMAP = dict(zip(ANIM, plt.cm.tab10(np.linspace(0, 0.5, 5)), strict=True))
RNG = np.random.default_rng(0)


def std_factor(term):
    """SD to convert a raw coef to per-1-SD units; (factor, is_binary)."""
    BIN = {"is_current", "swr_x_current", "swr_x_prev",
           "swr_x_is_last_rewarded_elsewhere", "reward_lag1"}
    if term in BIN:
        return 1.0, True
    # per-arm / rate continuous covariates scaled by their global SD
    for cov, s in SD.items():
        if term == cov or term == f"swr_x_{cov}":
            return s, False
    # content state-modulation interactions are already z-scored at fit time
    return 1.0, False


def sessions(df, mod, pop, term):
    g = df[(df.model == mod) & (df.population == pop)]
    c = f"coef_{term}"
    if c not in g:
        return {}
    g = g.dropna(subset=[c])
    g = g[g[c].abs() < 10]
    f, _ = std_factor(term)
    return {a: g.loc[g.animal == a, c].to_numpy() * f for a in ANIM if (g.animal == a).any()}


def animal_ci(medians, n_boot=4000):
    v = np.asarray([m for m in medians if np.isfinite(m)])
    if len(v) < 2:
        return np.nan, np.nan
    boot = [np.median(RNG.choice(v, len(v), replace=True)) for _ in range(n_boot)]
    return np.percentile(boot, [2.5, 97.5])


def forest(ax, df, rows, title):
    y = np.arange(len(rows))[::-1]
    allv = []
    labels = []
    for yi, (label, mod, pop, term) in zip(y, rows, strict=True):
        _, is_bin = std_factor(term)
        labels.append(label + (" †" if is_bin else ""))
        s = sessions(df, mod, pop, term)
        amed = []
        for a in ANIM:
            v = s.get(a, np.empty(0))
            if not len(v):
                amed.append(np.nan)
                continue
            allv.append(v)
            ax.scatter(v, yi + RNG.uniform(-0.28, 0.28, len(v)), s=6, color=CMAP[a],
                       alpha=0.20, edgecolors="none", zorder=2)
            m = np.median(v)
            amed.append(m)
            ax.plot(m, yi, "o", ms=7, color=CMAP[a], mec="k", mew=0.6, zorder=5)
        gm = np.nanmedian(amed)
        lo, hi = animal_ci(amed)
        ax.plot([lo, hi], [yi, yi], color="k", lw=1.4, zorder=4)
        ax.plot([gm, gm], [yi - 0.34, yi + 0.34], color="k", lw=2.6, zorder=6)
    ax.axvline(0, color="0.4", lw=1, ls="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    if allv:  # robust x-limits so per-animal medians stay legible
        cat = np.concatenate(allv)
        lo, hi = np.percentile(cat, [1, 99])
        pad = 0.1 * (hi - lo + 1e-9)
        ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel("coefficient  (continuous: per 1 SD;  † binary: per category)")
    ax.set_title(title, fontsize=11, weight="bold")
    ax.grid(axis="x", alpha=0.25)


handles = [plt.Line2D([], [], marker="o", ls="", color=CMAP[a], label=a) for a in ANIM]

# ---- RATE ----
R = pd.read_pickle(RC / "models_per_bin_rate" / "swr_contrast_rate.pkl")
SWR, NON = "non_local_swr", "non_local_no_swr"
rate_rows = [
    ("reward lag-1 · SWR", "G0_base", SWR, "reward_lag1"),
    ("reward lag-1 · nonSWR", "G0_base", NON, "reward_lag1"),
    ("RPE · SWR", "G5_rpe", SWR, "hmm_rpe"),
    ("RPE · nonSWR", "G5_rpe", NON, "hmm_rpe"),
    ("rel. value · SWR", "G4_value", SWR, "hmm_relative_value"),
    ("rel. value · nonSWR", "G4_value", NON, "hmm_relative_value"),
    ("reward-entropy · SWR", "G6_uncert", SWR, "reward_entropy"),
    ("reward-entropy · nonSWR", "G6_uncert", NON, "reward_entropy"),
    ("value-update · SWR", "G7_update", SWR, "hmm_value_update"),
    ("value-update · nonSWR", "G7_update", NON, "hmm_value_update"),
    ("switch-rate · SWR", "G3_switchrate", SWR, "recent_switch_rate"),
    ("switch-rate · nonSWR", "G3_switchrate", NON, "recent_switch_rate"),
]
fig, ax = plt.subplots(figsize=(8.5, 6.6))
forest(ax, R, rate_rows, "Rate ladder — is_swr contrast (standardized; all sessions + animal 95% CI)")
ax.legend(handles=handles, fontsize=8, loc="lower right", title="animal", framealpha=0.95)
fig.tight_layout()
fig.savefig(OUT / "rate_swr_contrast_coefs.png", dpi=150)

# ---- CONTENT ----
C = pd.read_pickle(RC / "models" / "swr_contrast_content_arm.pkl")
mod_rows = [
    ("current-arm (C1)", "C1_current", "all", "is_current"),
    ("reward_rate_arm (C3)", "C3_armreward", "all", "reward_rate_arm_k10"),
    ("value q_arm_hmm (C4)", "C4_armvalue", "all", "q_arm_hmm"),
    ("RPE×current (C6)", "C6_rpe", "all", "rpe_x_current"),
    ("rel.value×current (C7)", "C7_value", "all", "relval_x_current"),
    ("entropy×current (C8)", "C8_uncert", "all", "entropy_x_current"),
    ("update×current (C9)", "C9_update", "all", "update_x_current"),
    ("periswitch×current (C10)", "C10_periswitch", "all", "tss_x_current"),
    ("block×current (C11)", "C11_block", "all", "block_x_current"),
    ("switchrate×current (C12)", "C12_switchrate", "all", "switchrate_x_current"),
]
con_rows = [
    ("swr × current (CS1)", "CS1_current", "all", "swr_x_current"),
    ("swr × prev (CS2)", "CS2_prev", "all", "swr_x_prev"),
    ("swr × last-rewarded (CS3)", "CS3_lastrew", "all", "swr_x_is_last_rewarded_elsewhere"),
    ("swr × reward_rate (CS4)", "CS4_armreward", "all", "swr_x_reward_rate_arm_k10"),
    ("swr × value hmm (CS5)", "CS5_armvalue", "all", "swr_x_q_arm_hmm"),
]
fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
forest(axes[0], C, mod_rows, "Content — dimension effects (all events, standardized)")
forest(axes[1], C, con_rows, "Content — SWR contrasts (swr_x_*, standardized)")
axes[1].legend(handles=handles, fontsize=8, loc="lower left", title="animal", framealpha=0.95)
fig.tight_layout()
fig.savefig(OUT / "content_swr_contrast_coefs.png", dpi=150)

print("wrote:")
for p in sorted(OUT.glob("*.png")):
    print("  ", p)
