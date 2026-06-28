#!/usr/bin/env python
"""Six-panel summary figure for the per-bin rate ladder (Family B).

The visual companion to ``docs/per_bin_rate_results.md`` — reads
``models_per_bin_rate/all_models.pkl`` and renders, per event-type population:
baseline rate (M0), the lag-1 reward dissociation (M7), the reward kernel and
self-history kernel (M7), the model-comparison ladder (ΔAIC + CV bits/event),
and the reward-rate / switch covariate dissociations (M2r / M8).

Usage:
    uv run python scripts/plot_per_bin_rate_summary.py
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
from patsy import dmatrix

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from continuum_swr_replay.analysis.replay import (  # noqa: E402
    POPULATIONS,
    attach_is_swr_to_onsets,
)
from continuum_swr_replay.analysis.replay.model_specs import (  # noqa: E402
    SPLINE_KNOTS,
    TMAX,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR as P  # noqa: E402

RC = P / "replay_content"
FIGS = P.parent / "paper" / "figures"
STYLE = {  # outcome -> (label, color)
    "swr": ("detected SWR", "k"),
    "non_local": ("non-local (all)", "C2"),
    "non_local_swr": ("NL w/ SWR", "C3"),
    "non_local_no_swr": ("NL no-SWR", "C0"),
}
LADDER = ["M0_const", "M1_rew1", "M2_rewhist", "M2r_rewrate", "M3_hist",
          "M3r_histcos", "M4_hist_rew", "M5_time", "M6_time_hist",
          "M7_time_hist_rew", "M7r_time_histcos_rew", "M8_full_switch",
          "M9_rew_x_time"]
POPS4 = ["swr", "non_local", "non_local_swr", "non_local_no_swr"]
POPS3 = ["swr", "non_local_swr", "non_local_no_swr"]
SHORT = ["SWR", "NL\nw/SWR", "NL\nno-SWR"]


def _coef(df, model, oc, coef):
    return df[(df.model == model) & (df.outcome == oc)][coef].dropna()


def _timecourse(stop, onsets, oc, dt=0.1, tmax=4.0):
    """Descriptive onset rate (events / s of immobility) vs time since stop onset.

    Numerator = onset-count histogram over t_since_stop_onset; denominator =
    immobility-seconds per t-bin (sum over stops of the bin's overlap with each
    stop's immobility duration). The denominator already makes this a proper
    rate despite varying stop durations; what changes with tc is the *support*
    (fewer, longer, more-rewarded stops survive) — see the survival overlay.
    """
    t = POPULATIONS[oc](onsets)["t_since_stop_onset"].to_numpy()
    t = t[(t >= 0) & (t < tmax)]
    edges = np.arange(0, tmax + dt, dt)
    num, _ = np.histogram(t, edges)
    immob = stop["immobility_time"].to_numpy()
    immob = immob[immob > 0]
    denom = np.array([np.clip(immob - e, 0, dt).sum() for e in edges[:-1]])
    rate = num / np.where(denom > 0, denom, np.nan)
    return edges[:-1] + dt / 2, rate


def _survival(stop, tmax=4.0, dt=0.1):
    """Fraction of stops still immobile at each tc (the support of the curve)."""
    immob = stop["immobility_time"].to_numpy()
    immob = immob[immob > 0]
    tc = np.arange(dt / 2, tmax, dt)
    return tc, np.array([(immob > t).mean() for t in tc])


def _shade_low_support(ax, stop, thresh=0.2, tmax=4.0):
    """Grey the tc region where <thresh of stops survive (curve is a thin subset)."""
    tc, surv = _survival(stop, tmax)
    below = tc[surv < thresh]
    if below.size:
        ax.axvspan(below.min(), tmax, color="0.85", zorder=0)
        ax.text(below.min(), ax.get_ylim()[1] * 0.97,
                f" <{thresh:.0%} stops survive", fontsize=6, va="top", color="0.4")


def _panel_timecourse(ax, stop, onsets):
    for oc in POPS4:
        tc, rate = _timecourse(stop, onsets, oc)
        ax.plot(tc, rate, color=STYLE[oc][1], label=STYLE[oc][0], lw=1.8)
    ax.set_xlabel("time since stop onset (s)")
    ax.set_ylabel("onset rate (events / s)")
    ax.set_title("A0. Within-stop time course (descriptive)")
    ax.set_xlim(0, 4)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    _shade_low_support(ax, stop)
    ax2 = ax.twinx()
    stc, surv = _survival(stop)
    ax2.plot(stc, surv, color="0.5", ls="--", lw=1.0)
    ax2.set_ylabel("fraction of stops still immobile", color="0.5", fontsize=8)
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis="y", colors="0.5")


def _spline_basis(tc_grid):
    """Fixed-knot spline basis at tc_grid (same basis used in the fits)."""
    f = f"bs(tc, knots={SPLINE_KNOTS}, lower_bound=0.0, upper_bound={TMAX}) - 1"
    return np.asarray(dmatrix(f, {"tc": tc_grid}, return_type="dataframe"))


def _panel_fitted_timecourse(ax, df, stop):
    """Reconstructed M7 spline time course (reward/history at reference), per pop.

    Now reconstructable because the spline knots are fixed (data-independent):
    rate(tc) = exp(intercept + B(tc) @ spline_coef), averaged across sessions.
    """
    grid = np.linspace(0, TMAX, 81)
    B = _spline_basis(grid)
    scols = sorted(
        c for c in df.columns
        if c.startswith("coef_bs(tc, knots=") and c.endswith("]")
        and "reward_lag1:" not in c
    )
    for oc in POPS4:
        s = df[(df.model == "M7_time_hist_rew") & (df.outcome == oc)]
        beta = s[scols].to_numpy(float)
        b0 = s["coef_Intercept"].to_numpy(float)
        ok = ~np.isnan(beta).any(axis=1) & ~np.isnan(b0)
        if ok.sum() == 0:
            continue
        curves = np.exp(b0[ok, None] + beta[ok] @ B.T)  # (n_sess, n_grid)
        ax.plot(grid, np.median(curves, axis=0), color=STYLE[oc][1],
                label=STYLE[oc][0], lw=1.8)
    ax.set_xlabel("time since stop onset (s)")
    ax.set_ylabel("fitted rate (events / s)")
    ax.set_title("A1. Fitted time course (M7 spline, reward/history at ref)")
    ax.set_xlim(0, 4)
    ax.legend(frameon=False, fontsize=7, loc="lower right")
    _shade_low_support(ax, stop)


def _panel_baseline(ax, df):
    for i, oc in enumerate(POPS4):
        v = np.exp(_coef(df, "M0_const", oc, "coef_Intercept"))
        err = [[v.median() - v.quantile(.25)], [v.quantile(.75) - v.median()]]
        ax.bar(i, v.median(), color=STYLE[oc][1], alpha=0.7, yerr=err, capsize=4)
    ax.set_xticks(range(4))
    ax.set_xticklabels([STYLE[o][0] for o in POPS4], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("baseline rate (events / s)")
    ax.set_title("A. Baseline rate (M0_const)")


def _panel_reward1(ax, df):
    data = [np.exp(_coef(df, "M7_time_hist_rew", oc, "coef_reward_lag1")) for oc in POPS3]
    bp = ax.boxplot(data, labels=SHORT, showfliers=False, patch_artist=True)
    for patch, oc in zip(bp["boxes"], POPS3, strict=True):
        patch.set_facecolor(STYLE[oc][1])
        patch.set_alpha(0.4)
    ax.axhline(1, color="0.5", ls=":")
    ax.set_ylim(0, 6)
    ax.set_ylabel("lag-1 reward RR (per session)")
    ax.set_title("B. Previous-trial reward (M7, controlled)")


def _panel_reward_kernel(ax, df):
    lags = np.arange(1, 11)
    for oc in POPS4:
        rr = [np.exp(_coef(df, "M7_time_hist_rew", oc, f"coef_reward_lag{j}").median())
              for j in lags]
        ax.plot(lags, rr, marker="o", color=STYLE[oc][1], label=STYLE[oc][0], lw=1.8, ms=4)
    ax.axhline(1, color="0.5", ls=":")
    ax.set_xticks(lags)
    ax.set_ylim(0, 4.3)
    ax.set_xlabel("reward lag (trials back)")
    ax.set_ylabel("rate ratio")
    ax.set_title("C. Reward kernel (M7): boost then suppression")
    ax.legend(frameon=False, fontsize=7)


def _panel_selfhistory(ax, df):
    lags = np.arange(1, 11)
    for oc in POPS4:
        rr = [np.exp(_coef(df, "M7_time_hist_rew", oc, f"coef_onset_lag{j}").median())
              for j in lags]
        ax.plot(lags * 15, rr, marker="o", color=STYLE[oc][1], label=STYLE[oc][0], lw=1.8, ms=4)
    ax.axhline(1, color="0.5", ls=":")
    ax.set_xlabel("self-history lag (ms)")
    ax.set_ylabel("rate ratio")
    ax.set_title("D. Self-history (M7): refractory then bursty ~100 ms")
    ax.legend(frameon=False, fontsize=7)


def _panel_ladder(ax, df):
    g = df[df.outcome == "non_local"]
    aic0 = g[g.model == "M0_const"]["aic"].median()
    daic = [g[g.model == m]["aic"].median() - aic0 for m in LADDER]
    x = np.arange(len(LADDER))
    ax.bar(x, daic, color="0.6")
    ax.set_xticks(x)
    ax.set_xticklabels(LADDER, rotation=70, ha="right", fontsize=6)
    ax.set_ylabel("median ΔAIC vs M0 (lower=better)")
    ax.set_title("E. Model ladder (non-local): ΔAIC + CV bits/event")
    ax2 = ax.twinx()
    cvb = [g[g.model == m]["cv_bits_per_event"].median() for m in LADDER]
    ax2.plot(x, cvb, marker="D", color="C1", lw=1.5, ms=4)
    ax2.axhline(0, color="C1", ls=":", alpha=0.5)
    ax2.set_ylabel("CV bits/event (orange, higher=better)", color="C1")
    ax2.tick_params(axis="y", colors="C1")


def _panel_covar(ax, df):
    w = 0.35
    blocks = [("M2r_rewrate", "coef_recent_trial_reward_rate"),
              ("M8_full_switch", "coef_is_after_switch")]
    for j, (model, coef) in enumerate(blocks):
        for i, oc in enumerate(POPS3):
            rr = np.exp(_coef(df, model, oc, coef).median())
            ax.bar(i + j * (w + 0.05) - 0.2, rr, width=w, color=STYLE[oc][1],
                   alpha=0.5 + 0.5 * (1 - j), edgecolor="k", linewidth=0.5)
    ax.axhline(1, color="0.5", ls=":")
    ax.set_xticks(range(3))
    ax.set_xticklabels(SHORT)
    ax.set_ylabel("rate ratio")
    ax.set_title("F. Reward-rate (dark) & switch (light) by type")


def main() -> None:
    df = pd.read_pickle(RC / "models_per_bin_rate" / "all_models.pkl")
    stop = pd.read_pickle(RC / "stop_table.pkl")
    onsets = attach_is_swr_to_onsets(
        pd.read_pickle(RC / "event_onset_table.pkl"),
        pd.read_pickle(RC / "event_table.pkl"),
    )
    FIGS.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplot_mosaic(
        [["time", "rate", "rew1", "ladder"],
         ["timefit", "rewk", "hist", "covar"]], figsize=(21, 9)
    )
    _panel_timecourse(ax["time"], stop, onsets)
    _panel_fitted_timecourse(ax["timefit"], df, stop)
    _panel_baseline(ax["rate"], df)
    _panel_reward1(ax["rew1"], df)
    _panel_reward_kernel(ax["rewk"], df)
    _panel_selfhistory(ax["hist"], df)
    _panel_ladder(ax["ladder"], df)
    _panel_covar(ax["covar"], df)
    fig.suptitle(
        "Per-bin replay-rate ladder (Family B): fitted effects by event type, 218 sessions",
        y=1.0,
    )
    fig.tight_layout()
    out = FIGS / "per_bin_rate_ladder_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
