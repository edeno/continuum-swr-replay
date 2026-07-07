#!/usr/bin/env python
"""Exploratory: per-stop rate-model predicted-vs-observed view for one session.

Companion to ``_nonlocal_model_fit.py`` (content). The rate model is per-*stop*
(it predicts the number of non-local events in each immobility period, including
the 0-event stops), so this view is a per-stop strip — one column per stop, no
9-arm grid. It fits a chosen per-stop Poisson model (MODEL_PER_STOP_RATE) on one
session and shows, sharing the stop x-axis:

  * the exposure (immobility_time) and the model's own PREDICTORS, as per-stop
    tracks;
  * OBSERVED events/stop (bars) with the model's PREDICTED expected count (line);
  * the RESIDUAL, observed - predicted, per stop.

Predicted count is reconstructed from the fitted coefficients:
``mu = exp(intercept + X beta) * immobility_time`` (the log-immobility offset).
Model and outcome are free parameters (``--rate-model``, ``--outcome``).

Usage
-----
    python scripts/exploratory/_nonlocal_rate_fit.py \
        --animal chimi --day 20200227 --epoch 04_r2 --rate-model R2
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

from continuum_swr_replay.analysis.replay.model_specs import (
    MODEL_PER_STOP_RATE,
    N_REWARD_LAGS,
)
from continuum_swr_replay.analysis.replay.per_stop_rate_model import (
    fit_per_stop_rate_model,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR, ROOT_DIR

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,
                     "font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"]})

RC_DIR = PROCESSED_DATA_DIR / "replay_content"
DEFAULT_OUT_DIR = ROOT_DIR / ".claude/research/nonlocal_rate_fit_outputs"
OUTCOMES = {"non_local": "n_non_local", "swr": "n_SWR",
            "non_local_swr": "n_non_local_SWR", "non_local_no_swr": "n_non_local_no_SWR"}

# Non-reward-lag predictors -> track spec.
RATE_PANEL = {
    "is_after_switch": ("binary", "is_after_switch", "after switch"),
    "is_after_omission": ("binary", "is_after_omission", "after omission"),
    "reward_rate_trial_k10": ("heat", "reward_rate_trial_k10", "reward rate trial",
                              "viridis"),
}


def _vlims(x):
    x = x[np.isfinite(x)]
    return (0.0, 1.0) if x.size == 0 else (float(x.min()), float(x.max()))


def build_figure(sub: pd.DataFrame, model_name: str, response_col: str,
                 title_stub: str):
    """Fit ``model_name`` on one session's stops; draw predictors + obs/pred/resid."""
    spec = MODEL_PER_STOP_RATE[model_name]
    predictors = list(spec.predictors)
    sub = sub.sort_values("stop_start").reset_index(drop=True)
    n_stops = len(sub)
    x = np.arange(n_stops)

    fit = fit_per_stop_rate_model(sub, predictors, response_col=response_col,
                                  offset_col="immobility_time")
    if fit is None:
        raise SystemExit(
            f"Rate model {model_name} did not fit for this session "
            "(too few stops / rank-deficient)."
        )
    # Predicted expected count: exp(intercept + X beta) * immobility_time.
    expo = sub["immobility_time"].to_numpy(float)
    eta = np.full(n_stops, fit["coef_intercept"], dtype=float)
    for p in predictors:
        eta = eta + fit[f"coef_{p}"] * sub[p].to_numpy(float)
    pred = np.exp(eta) * expo
    obs = sub[response_col].to_numpy(float)
    resid = obs - pred

    # Panels: exposure, then model predictors (reward-lag kernel folded to one),
    # then observed-vs-predicted counts, then residual.
    lag_cols = [f"reward_lag{j}" for j in range(1, N_REWARD_LAGS + 1)]
    panels = [("heat", "immobility_time", "immobility (s)", "cividis")]
    seen = set()
    for p in predictors:
        if p in lag_cols:
            if "kernel" not in seen:
                seen.add("kernel")
                panels.append(("kernel", None, "reward kernel (lag)"))
        elif p in RATE_PANEL and p not in seen:
            seen.add(p)
            panels.append(RATE_PANEL[p])
    panels += [("counts", None, "events / stop"), ("resid", None, "residual\nobs − pred")]

    H = {"heat": 0.5, "binary": 0.5, "kernel": 2.4, "counts": 3.2, "resid": 2.2}
    heights = [H[p[0]] for p in panels]
    fig = plt.figure(figsize=(max(11, n_stops * 0.035), 0.42 * sum(heights) + 1.4),
                     constrained_layout=True)
    gs = GridSpec(len(panels), 2, figure=fig, height_ratios=heights,
                  width_ratios=[60, 1])
    axes, ax0 = [], None
    for i in range(len(panels)):
        ax = fig.add_subplot(gs[i, 0], sharex=ax0)
        ax0 = ax0 or ax
        axes.append(ax)

    def track_style(ax, label):
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([0])
        ax.set_yticklabels([label], fontsize=8)
        ax.tick_params(axis="both", length=0)
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)

    def cbar(i, mappable, ticks=None):
        cb = fig.colorbar(mappable, cax=fig.add_subplot(gs[i, 1]), ticks=ticks)
        cb.ax.tick_params(labelsize=6, length=2)

    for i, p in enumerate(panels):
        ax, kind = axes[i], p[0]
        if kind == "binary":
            v = sub[p[1]].to_numpy(float)
            no, yes, miss = v == 0, v == 1, ~np.isfinite(v)
            ax.scatter(x[no], np.zeros(no.sum()), s=8, marker="s", c="0.85",
                       edgecolors="none", rasterized=True)
            ax.scatter(x[yes], np.zeros(yes.sum()), s=8, marker="s", c="#222",
                       edgecolors="none", rasterized=True)
            if miss.any():
                ax.scatter(x[miss], np.zeros(miss.sum()), s=6, marker="x", c="0.6",
                           linewidths=0.5, rasterized=True)
            track_style(ax, p[1] if len(p) < 3 else p[2])
        elif kind == "heat":
            _, col, label, cmap_name = p
            M = np.ma.masked_invalid(sub[col].to_numpy(float)[None, :])
            cmap = plt.get_cmap(cmap_name).copy()
            cmap.set_bad("0.9")
            vmin, vmax = _vlims(sub[col].to_numpy(float))
            im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                           extent=(-0.5, n_stops - 0.5, 0.5, -0.5),
                           interpolation="nearest", rasterized=True)
            track_style(ax, label)
            cbar(i, im, ticks=[round(vmin, 1), round(vmax, 1)])
        elif kind == "kernel":
            K = np.ma.masked_invalid(
                np.vstack([sub[c].to_numpy(float) for c in lag_cols]))
            cmap = plt.get_cmap("Greys").copy()
            cmap.set_bad("0.9")
            im = ax.imshow(K, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                           extent=(-0.5, n_stops - 0.5, N_REWARD_LAGS - 0.5, -0.5),
                           interpolation="nearest", rasterized=True)
            ax.set_yticks(range(N_REWARD_LAGS))
            ax.set_yticklabels([str(j + 1) for j in range(N_REWARD_LAGS)], fontsize=5)
            ax.set_ylabel(p[2], fontsize=8, rotation=0, ha="right", va="center",
                          labelpad=20)
            ax.tick_params(axis="both", length=0)
            for s in ("top", "right", "left", "bottom"):
                ax.spines[s].set_visible(False)
            cbar(i, im, ticks=[0, 1])
        elif kind == "counts":
            ax.bar(x, obs, width=1.0, color="0.75", label="observed", zorder=1)
            ax.plot(x, pred, color="crimson", lw=1.0, label="predicted E[count]",
                    zorder=2)
            ax.set_ylabel("events / stop", fontsize=8)
            ax.set_ylim(0, max(np.nanmax(obs), np.nanmax(pred)) * 1.1)
            ax.legend(loc="upper right", fontsize=7, frameon=False)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(labelsize=7)
        elif kind == "resid":
            colors = np.where(resid >= 0, "#c0392b", "#2874a6")
            ax.bar(x, resid, width=1.0, color=colors, zorder=1)
            ax.axhline(0, color="0.5", lw=0.5)
            ax.set_ylabel("residual\nobs − pred", fontsize=8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.tick_params(labelsize=7)

    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    good = np.isfinite(obs) & np.isfinite(pred)
    corr = np.corrcoef(obs[good], pred[good])[0, 1] if good.sum() > 2 else np.nan
    axes[0].set_title(
        f"{title_stub}  —  per-stop rate model {model_name} fit  "
        f"(outcome={response_col}, n_stops={fit['n_stops']}, "
        f"AIC={fit['aic']:.0f}, corr(obs,pred)={corr:.2f}, "
        f"converged={fit['converged']})", fontsize=11)
    axes[-1].set_xlim(-0.5, n_stops - 0.5)
    axes[-1].set_xlabel(
        "immobility period (stop, time-ordered)  —  exposure + predictors of "
        f"{model_name}, then observed vs predicted event count", fontsize=9)
    return fig, n_stops


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--animal", default="chimi")
    p.add_argument("--day", type=int, default=20200227)
    p.add_argument("--epoch", default="04_r2")
    p.add_argument("--rate-model", default="R2",
                   help=f"MODEL_PER_STOP_RATE rung ({', '.join(MODEL_PER_STOP_RATE)})")
    p.add_argument("--outcome", default="non_local", choices=list(OUTCOMES),
                   help="which count to model")
    p.add_argument("--out", default=None)
    p.add_argument("--pdf", action="store_true")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    st = pd.read_pickle(RC_DIR / "stop_table.pkl")
    sub = st[(st.animal == args.animal) & (st.day == args.day)
             & (st.epoch == args.epoch)]
    if sub.empty:
        raise SystemExit(f"No stops for {args.animal} {args.day} {args.epoch}.")

    fig, n = build_figure(sub, args.rate_model, OUTCOMES[args.outcome],
                          f"{args.animal} {args.day} {args.epoch}")
    out = args.out
    if out is None:
        os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
        out = str(DEFAULT_OUT_DIR / f"{args.animal}{args.day}_{args.epoch}_"
                  f"{args.rate_model}_{args.outcome}_fit.png")
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    written = [out]
    if args.pdf:
        pdf = out[:-4] + ".pdf" if out.endswith(".png") else out + ".pdf"
        fig.savefig(pdf, dpi=args.dpi, bbox_inches="tight")
        written.append(pdf)
    print(f"wrote {', '.join(written)} | stops: {n}")


if __name__ == "__main__":
    main()
