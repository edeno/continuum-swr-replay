#!/usr/bin/env python
"""Exploratory: content-model predicted-vs-observed view for one session.

Companion to ``_nonlocal_bout_arm_dots.py``. Fits a chosen conditional-Poisson
content model (M-ladder) on one session, then shows, sharing the event x-axis
and 9-arm grid:

  * the model's own PREDICTORS, drawn in the browser style (per-stop cells /
    per-event squares / arm strips);
  * the OBSERVED content response (the dot view from the browser, alpha =
    arm_fraction);
  * the model's PREDICTED per-arm allocation (softmax(Xbeta) within event);
  * the per-(event,arm) RESIDUAL, observed - predicted (diverging);
  * the per-arm CUMULATIVE (martingale) residual: since arm counts are
    Multinomial(N, p) per event, each arm's running sum(obs - N*p) standardized
    by its cumulative sd (N*p*(1-p)) shares a single +/-2 band; a line past +/-2
    marks an arm the model systematically over/under-allocates content to.

The model is a free parameter (``--content-model``); any MODEL_HIERARCHY rung
works. Predicted/residual use ``arm_fraction`` (MAP-based), which is what the
model fits, so this panel's observed dots use arm_fraction too (the browser uses
arm_fraction_posterior; the difference is the MAP vs full-posterior content
summary).

Usage
-----
    python scripts/exploratory/_nonlocal_model_fit.py \
        --animal chimi --day 20200227 --epoch 04_r2 --content-model M6
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from continuum_swr_replay.analysis.behavior._helpers import add_arm_fixed_effects
from continuum_swr_replay.analysis.behavior.choice.choice_prediction import (
    predict_arm_fractions,
)
from continuum_swr_replay.analysis.behavior.fitting import fit_conditional_poisson
from continuum_swr_replay.analysis.behavior.model_specs import MODEL_HIERARCHY
from continuum_swr_replay.parameters import ROOT_DIR
from continuum_swr_replay.paths import REPLAY_CONTENT_DIR, table_path
from continuum_swr_replay.tables.covariate_registry import (
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table
from continuum_swr_replay.visualization.palettes import patch_grouped_arm_palette

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    }
)

ARM_ROW_ORDER = [0, 1, 6, 2, 3, 7, 4, 5, 8]
ARM_LABEL = {a: (f"arm {a} (trunk)" if a >= 6 else f"arm {a}") for a in ARM_ROW_ORDER}
PALETTE = patch_grouped_arm_palette()
RC_DIR = REPLAY_CONTENT_DIR
DEFAULT_OUT_DIR = ROOT_DIR / ".claude/research/nonlocal_model_fit_outputs"

# Each predictor maps to how it is displayed. "ring" is shown as the current-arm
# ring in the observed panel; "swr" folds into a single per-event SWR track;
# arm_fe_* are the row identity (no panel).
COV_PANEL = {
    "is_prev_arm": ("armstrip", "prev_arm", "prev arm"),
    "is_last_rewarded_noncurrent_arm": (
        "armstrip",
        "last_rewarded_arm",
        "last rew arm",
    ),
    "reward_fraction_candidate_arm_last10_visits": (
        "cov",
        "reward_fraction_candidate_arm_last10_visits",
        "reward rate k10",
        ("plasma", False),
    ),
    "visit_recency_trials_event_leaf_centered": (
        "cov",
        "visit_recency_trials_event_leaf_centered",
        "visit recency",
        ("PuOr", True),
    ),
    "task_reward_probability_candidate_arm": (
        "cov",
        "task_reward_probability_candidate_arm",
        "P(reward)",
        ("cividis", False),
    ),
    "reward_belief_mean_candidate_arm_preoutcome": (
        "cov",
        "reward_belief_mean_candidate_arm_preoutcome",
        "beta value",
        ("RdBu_r", True),
    ),
}


def _vlims(M, diverging):
    if not np.isfinite(M).any():
        return 0.0, 1.0
    if diverging:
        v = float(np.nanmax(np.abs(M)))
        return -v, v
    return float(np.nanmin(M)), float(np.nanmax(M))


def build_figure(sub: pd.DataFrame, model_name: str, title_stub: str):
    """Fit ``model_name`` on one session and draw predictors + obs/pred/residual."""
    palette, n_arms = PALETTE, len(ARM_ROW_ORDER)
    spec = MODEL_HIERARCHY[model_name]
    predictors = list(spec.predictors)

    sub = sub.copy()
    add_arm_fixed_effects(sub)
    fit = fit_conditional_poisson(
        sub,
        predictors,
        response_col="arm_count_hard",
        event_col="event_id",
        n_categories=9,
    )
    if fit is None:
        raise SystemExit(
            f"Model {model_name} did not fit for this session (underpowered / "
            "rank-deficient). Try a lower rung (e.g. --content-model M2)."
        )
    pf = predict_arm_fractions(
        sub, fit, predictors, event_col="event_id", fraction_col="arm_fraction"
    )

    # Event ordering + per-arm y positions.
    order = pf.groupby("non_local_event")["event_start"].min().sort_values()
    col_of = {ev: i for i, ev in enumerate(order.index)}
    pf["col"] = pf["non_local_event"].map(col_of)
    n_bouts = len(order)
    y_of_arm = {a: (n_arms - 1 - i) for i, a in enumerate(ARM_ROW_ORDER)}
    pf["y"] = pf["candidate_arm"].map(y_of_arm)
    evo = pf.groupby("non_local_event").first().reindex(order.index)

    # Stop structure for per-stop cells + boundary separators.
    stop_col = evo["stop_id"].to_numpy()
    grp = np.zeros(n_bouts, dtype=int)
    grp[1:] = np.cumsum(stop_col[1:] != stop_col[:-1])
    n_stops = int(grp[-1]) + 1
    starts = np.concatenate([[0], np.where(np.diff(grp) != 0)[0] + 1])
    ends = np.concatenate([starts[1:] - 1, [n_bouts - 1]])
    stop_edges = np.concatenate([[starts[0] - 0.5], ends + 0.5])

    def to_stop(a):
        return a[..., starts]

    def arm_matrix(col, per_event):
        piv = pf.pivot_table(
            index="candidate_arm",
            columns="non_local_event",
            values=col,
            aggfunc="first",
        )
        m = piv.reindex(index=ARM_ROW_ORDER, columns=order.index).to_numpy()
        return m if per_event else to_stop(m)

    # Assemble the panel list: SWR track (if any swr_x_* predictor), the arm/cov
    # predictor panels (in model order, deduped), then obs / predicted / residual.
    uses_swr = any(p.startswith("swr_x") for p in predictors)
    show_ring = "is_current_arm" in predictors
    panels, seen = [], set()
    if uses_swr:
        panels.append(("swr",))
    for p in predictors:
        entry = COV_PANEL.get(p)
        if entry and entry[:2] not in seen:
            seen.add(entry[:2])
            panels.append(entry)
    panels += [("content",), ("pred",), ("resid",), ("cumresid",)]

    H = {
        "swr": 0.55,
        "armstrip": 0.55,
        "cov": 4.2,
        "content": 6.5,
        "pred": 4.2,
        "resid": 4.2,
        "cumresid": 3.2,
    }
    heights = [H[p[0]] for p in panels]
    fig = plt.figure(
        figsize=(max(11, n_bouts * 0.1), 0.16 * sum(heights) + 1.5),
        constrained_layout=True,
    )
    gs = GridSpec(
        len(panels), 2, figure=fig, height_ratios=heights, width_ratios=[60, 1]
    )
    axes = []
    ax0 = None
    for i in range(len(panels)):
        ax = fig.add_subplot(gs[i, 0], sharex=ax0)
        ax0 = ax0 or ax
        axes.append(ax)
    cols = np.arange(n_bouts)

    def track_style(ax, label):
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([0])
        ax.set_yticklabels([label], fontsize=8)
        ax.tick_params(axis="both", length=0)
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)

    def arm_ticks(ax, full):
        ax.set_yticks(range(n_arms))
        ax.set_yticklabels(
            [ARM_LABEL[a] if full else str(a) for a in ARM_ROW_ORDER],
            fontsize=(8 if full else 6.5),
        )
        for tick, a in zip(ax.get_yticklabels(), ARM_ROW_ORDER, strict=True):
            tick.set_color(palette[a][:3])
            if full:
                tick.set_fontweight("bold")
        ax.tick_params(axis="both", length=0)

    def cbar(i, mappable, ticks=None):
        cb = fig.colorbar(mappable, cax=fig.add_subplot(gs[i, 1]), ticks=ticks)
        cb.ax.tick_params(labelsize=6, length=2)

    def row_label(ax, text):
        ax.set_ylabel(
            text, fontsize=8, rotation=0, ha="right", va="center", labelpad=22
        )
        for s in ("top", "right", "left", "bottom"):
            ax.spines[s].set_visible(False)

    def event_heatmap(ax, M, cmap_name, diverging, i, label):
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("0.9")
        vmin, vmax = _vlims(np.ma.masked_invalid(M), diverging)
        im = ax.imshow(
            np.ma.masked_invalid(M),
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            extent=(-0.5, n_bouts - 0.5, n_arms - 0.5, -0.5),
            zorder=1,
            rasterized=True,
        )
        arm_ticks(ax, full=False)
        row_label(ax, label)
        cbar(i, im, ticks=[round(vmin, 2), round(vmax, 2)])

    for i, p in enumerate(panels):
        ax, kind = axes[i], p[0]
        if kind == "swr":
            v = evo["is_swr"].to_numpy(float)
            no, yes = v == 0, v == 1
            ax.scatter(
                cols[no],
                np.zeros(no.sum()),
                s=26,
                marker="s",
                c="0.85",
                edgecolors="0.6",
                linewidths=0.35,
                rasterized=True,
            )
            ax.scatter(
                cols[yes],
                np.zeros(yes.sum()),
                s=26,
                marker="s",
                c="#222",
                edgecolors="none",
                rasterized=True,
            )
            track_style(ax, "SWR")
        elif kind == "armstrip":
            vs = to_stop(evo[p[1]].to_numpy(float))
            cmap = ListedColormap([palette[a][:3] for a in range(9)])
            cmap.set_bad("white")
            ax.pcolormesh(
                stop_edges,
                [-0.5, 0.5],
                np.ma.masked_invalid(vs)[None, :],
                cmap=cmap,
                norm=BoundaryNorm(np.arange(-0.5, 9.5, 1.0), 9),
                rasterized=True,
            )
            track_style(ax, p[2])
        elif kind == "cov":
            _, col, label, (cmap_name, div) = p
            Ms = np.ma.masked_invalid(arm_matrix(col, per_event=False))
            cmap = plt.get_cmap(cmap_name).copy()
            cmap.set_bad("0.9")
            vmin, vmax = _vlims(Ms, div)
            mesh = ax.pcolormesh(
                stop_edges,
                np.arange(n_arms + 1) - 0.5,
                Ms,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                rasterized=True,
            )
            ax.set_ylim(n_arms - 0.5, -0.5)
            arm_ticks(ax, full=False)
            row_label(ax, label)
            cbar(i, mesh)
        elif kind == "content":
            for g in range(n_stops):
                if g % 2 == 0:
                    ax.axvspan(
                        starts[g] - 0.5,
                        ends[g] + 0.5,
                        color="0.5",
                        alpha=0.10,
                        lw=0,
                        zorder=0,
                    )
            frac = pf["arm_fraction"].to_numpy().clip(0, 1)
            cc = np.array([palette[a] for a in pf["candidate_arm"]], dtype=float)
            cc[:, 3] = frac
            ax.scatter(
                pf["col"],
                pf["y"],
                s=32,
                c=cc,
                edgecolors="none",
                zorder=2,
                rasterized=True,
            )
            if show_ring:
                cur = pf[pf["is_current_arm"] == 1]
                ax.scatter(
                    cur["col"],
                    cur["y"],
                    s=46,
                    facecolors="none",
                    edgecolors="black",
                    linewidths=0.9,
                    zorder=3,
                    rasterized=True,
                )
            for yb in (2.5, 5.5):
                ax.axhline(yb, color="0.85", lw=0.7, zorder=1)
            ax.set_ylim(-0.5, n_arms - 0.5)
            ax.set_yticks([y_of_arm[a] for a in ARM_ROW_ORDER])
            ax.set_yticklabels([ARM_LABEL[a] for a in ARM_ROW_ORDER], fontsize=8)
            for tick, a in zip(ax.get_yticklabels(), ARM_ROW_ORDER, strict=True):
                tick.set_color(palette[a][:3])
                tick.set_fontweight("bold")
            ax.tick_params(axis="both", length=0)
            row_label(ax, "OBSERVED\n(arm_fraction)")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        elif kind == "pred":
            event_heatmap(
                ax,
                arm_matrix("predicted_frac", per_event=True),
                "plasma",
                False,
                i,
                "PREDICTED\nP(arm)",
            )
        elif kind == "resid":
            event_heatmap(
                ax,
                arm_matrix("residual_frac", per_event=True),
                "RdBu_r",
                True,
                i,
                "RESIDUAL\nobs − pred",
            )
        elif kind == "cumresid":
            # Per-arm cumulative (martingale) residual. Per event the arm counts
            # are Multinomial(N, p): expected count = N*predicted_frac, marginal
            # variance = N*p*(1-p). Standardize each arm's running sum by its
            # cumulative sd so all 9 share a single ±2 band; a line past ±2 means
            # the model systematically over/under-allocates content to that arm.
            obs = arm_matrix("arm_count_hard", per_event=True)  # (9, n_bouts)
            p = arm_matrix("predicted_frac", per_event=True)
            n_event = np.nansum(obs, axis=0)  # bins per event
            res = obs - p * n_event[None, :]
            var = p * (1.0 - p) * n_event[None, :]
            bad = ~np.isfinite(res) | ~np.isfinite(var)
            cumR = np.cumsum(np.where(bad, 0.0, res), axis=1)
            cumV = np.cumsum(np.where(bad, 0.0, var), axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                z = np.where(cumV > 0, cumR / np.sqrt(cumV), np.nan)
            for r, a in enumerate(ARM_ROW_ORDER):
                ax.plot(cols, z[r], color=palette[a][:3], lw=1.0, label=f"arm {a}")
            for yb in (2, -2):
                ax.axhline(yb, color="0.6", lw=0.6, ls="--")
            ax.axhline(0, color="0.5", lw=0.5)
            ax.set_ylabel("cumulative residual\n(z, per arm)", fontsize=8)
            ax.tick_params(labelsize=7)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)

    for ax in axes:
        ax.vlines(
            stop_edges[1:-1],
            0,
            1,
            transform=ax.get_xaxis_transform(),
            colors="0.3",
            linewidths=0.45,
            alpha=0.5,
            zorder=4,
            rasterized=True,
        )
    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)

    axes[0].set_title(
        f"{title_stub}  —  content model {model_name} fit  "
        f"(n_events={fit['n_events']}, pseudo-R²={fit['pseudo_r2']:.3f}, "
        f"AIC={fit['aic']:.0f}, converged={fit['converged']})",
        fontsize=11,
    )
    axes[-1].set_xlim(-0.5, n_bouts - 0.5)
    axes[-1].set_xlabel(
        "non-local event (time-ordered; immobility, speed < 4 cm/s)  —  "
        f"predictors of {model_name}, then observed / predicted / residual "
        "arm content",
        fontsize=9,
    )
    legend = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="0.35",
            markersize=8,
            label="observed: opacity = arm_fraction",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=8,
            label="animal's current arm",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor="#222",
            markersize=7,
            label="SWR event",
        ),
    ]
    axes[-1].legend(
        handles=legend,
        loc="upper left",
        bbox_to_anchor=(0, -0.9),
        frameon=False,
        fontsize=7.5,
        ncol=3,
    )
    return fig, n_bouts


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--animal", default="chimi")
    p.add_argument("--day", type=int, default=20200227)
    p.add_argument("--epoch", default="04_r2")
    p.add_argument(
        "--content-model",
        default="M6",
        help=f"MODEL_HIERARCHY rung ({', '.join(MODEL_HIERARCHY)})",
    )
    p.add_argument("--out", default=None)
    p.add_argument("--pdf", action="store_true")
    p.add_argument("--dpi", type=int, default=150)
    args = p.parse_args()

    ev = read_table(table_path("event_arm_table", RC_DIR))
    require_covariate_schema(ev, table="event_arm")
    sub = ev[
        (ev.animal == args.animal)
        & (ev.session_date == args.day)
        & (ev.epoch == args.epoch)
    ]
    if sub.empty:
        raise SystemExit(f"No rows for {args.animal} {args.day} {args.epoch}.")

    fig, n = build_figure(
        sub, args.content_model, f"{args.animal} {args.day} {args.epoch}"
    )
    out = args.out
    if out is None:
        os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
        out = str(
            DEFAULT_OUT_DIR / f"{args.animal}{args.day}_{args.epoch}_"
            f"{args.content_model}_fit.png"
        )
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    written = [out]
    if args.pdf:
        pdf = out[:-4] + ".pdf" if out.endswith(".png") else out + ".pdf"
        fig.savefig(pdf, dpi=args.dpi, bbox_inches="tight")
        written.append(pdf)
    print(f"wrote {', '.join(written)} | events: {n}")


if __name__ == "__main__":
    main()
