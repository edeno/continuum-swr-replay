#!/usr/bin/env python
"""Exploratory: non-local bout covariate browser for one session.

Columns = non-local events (time-ordered; immobility-associated, the analyzed
window is immobile, speed < 4 cm/s). One column per non-local event — a single
immobility period (stop) may contain several events, grouped by the shaded
bands in the content panel. Rows = 9 arms (patch-grouped palette). The panels
mirror the covariates in the fitted model ladders so decoded content can be
eyeballed against the predictors:

  Rate models (per-bout tracks)
    reward = reward_outcome (this stop's own outcome), reward-history kernel
    (row k = reward outcome k stops back, built from the displayed stop
    sequence so it is column-aligned; approximates the rate models' trials-back
    reward_lag, which matches ~91% at lag 1),
    after switch / after omission (R1), reward rate trial k10 (R2), and the
    SWR-contrast value-state block (is_post_switch, trials_since_switch,
    trials_since_block_change, recent_switch_rate, hmm_relative_value, hmm_rpe,
    reward_entropy, hmm_value_update).
  Content models (per-arm, cell-aligned to the content panel)
    is_current (ring), is_prev / is_last_rewarded_elsewhere (arm strips),
    reward_rate_arm_k10, visit_recency_demeaned, visit_recency (raw),
    cumulative_reward, p_reward_arm, arm_dist_rank, q_arm_hmm (HMM value),
    q_arm (belief value; robustness construct).

Content magnitude is encoded as circle *alpha* (posterior-mass fraction), never
size. arm_dist_rank is the identifiable geometric baseline used by the
SWR-contrast (BASEc) and choice ladders; raw dist_to_arm and is_next are absent
because the models drop them (quasi-separation / alternation collinearity), and
the primary content hierarchy (M0-Mv) carries no distance term at all.

Granularity: the x-axis is one column per non-local *event*, but only the
content panel (arm_fraction_posterior) and the SWR track (is_swr) actually vary
event-to-event (verified: they differ within a stop for 100% / ~62% of
multi-event stops). Every other covariate is a per-*stop* quantity, so it is
drawn once per immobility period as a single cell spanning that stop's event
columns (variable width) rather than repeated per event. Per-event panels use
markers/dots at event columns; per-stop panels use pcolormesh over the stop
band edges. Both share the event x-axis, so a stop's cell sits directly above
the content dots for its events.

Usage
-----
    python scripts/exploratory/_nonlocal_bout_arm_dots.py \
        --animal chimi --day 20200227 --epoch 04_r2
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

from continuum_swr_replay.analysis.replay.covariates import add_uncertainty_covariates
from continuum_swr_replay.analysis.replay.model_specs import N_REWARD_LAGS
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR, ROOT_DIR
from continuum_swr_replay.visualization.palettes import patch_grouped_arm_palette

# Publication-friendly text: sans-serif with editable (Type-42) fonts so a PDF
# export opens with selectable/editable text. Arial if available, else fallback.
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
})

# Row order top->bottom: patch 1 (0,1,6), patch 2 (2,3,7), patch 3 (4,5,8).
ARM_ROW_ORDER = [0, 1, 6, 2, 3, 7, 4, 5, 8]
ARM_LABEL = {a: (f"arm {a} (trunk)" if a >= 6 else f"arm {a}") for a in ARM_ROW_ORDER}
PALETTE = patch_grouped_arm_palette()
RC_DIR = PROCESSED_DATA_DIR / "replay_content"
DEFAULT_OUT_DIR = ROOT_DIR / ".claude/research/nonlocal_bout_arm_outputs"


def _vlims(M: np.ndarray, diverging: bool) -> tuple[float, float]:
    """Color limits for a heatmap; falls back to (0, 1) if all values are NaN.

    A covariate can be entirely NaN for a session (e.g. a stop with no qvals
    coverage under the ``how="left"`` merge); guarding avoids a ``nanmax`` over
    a fully-masked array returning ``nan`` limits and a blank colorbar.
    """
    if not np.isfinite(M).any():
        return 0.0, 1.0
    if diverging:
        v = float(np.nanmax(np.abs(M)))
        return -v, v
    return float(np.nanmin(M)), float(np.nanmax(M))


def build_figure(
    sub: pd.DataFrame, sst: pd.DataFrame, title_stub: str
) -> tuple[plt.Figure, int]:
    """Build the covariate-browser figure for one session.

    Parameters
    ----------
    sub : pd.DataFrame
        ``event_arm_table`` rows for one session (9 arms x n_bouts rows),
        carrying ``non_local_event``, ``arm``, ``event_start``, ``stop_id``,
        ``arm_fraction_posterior``, ``is_current`` and the per-arm covariate
        columns.
    sst : pd.DataFrame
        ``stop_table`` rows for the same session with ``recent_switch_rate``
        (from ``add_uncertainty_covariates``) and the qvals value-state block
        already merged on ``stop_id``.
    title_stub : str
        Session label for the figure title, e.g. ``"chimi 20200227 04_r2"``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The assembled multi-panel figure.
    n_bouts : int
        Number of non-local bouts (columns) drawn.
    """
    palette = PALETTE
    n_arms = len(ARM_ROW_ORDER)

    # Time-order the bouts; map each event to a column index.
    order = sub.groupby("non_local_event")["event_start"].min().sort_values()
    col_of_event = {ev: i for i, ev in enumerate(order.index)}
    sub = sub.copy()
    sub["col"] = sub["non_local_event"].map(col_of_event)
    n_bouts = len(order)
    cols = np.arange(n_bouts)

    # y position per arm in the content panel (top row = first in ARM_ROW_ORDER).
    y_of_arm = {arm: (n_arms - 1 - i) for i, arm in enumerate(ARM_ROW_ORDER)}
    sub["y"] = sub["arm"].map(y_of_arm)

    # Per-bout covariates (constant across arms), in column order. Binary tracks
    # keep raw 0/1/NaN (not a bool mask) so missing values render distinctly
    # rather than collapsing to "no".
    evo = sub.groupby("non_local_event").first().reindex(order.index)
    rew_col = evo["reward_outcome"].to_numpy(dtype=float)
    swr_col = evo["is_swr"].to_numpy(dtype=float)
    switch_col = evo["is_after_switch"].to_numpy(dtype=float)
    omit_col = evo["is_after_omission"].to_numpy(dtype=float)
    prev_arm_col = evo["prev_arm"].to_numpy(dtype=float)
    lastrew_arm_col = evo["last_rewarded_arm"].to_numpy(dtype=float)

    # Trial-level recent reward rate (rate model R2) lives at the stop level.
    stop_rrt = sst.set_index("stop_id")["reward_rate_trial_k10"]
    rrt_col = evo["stop_id"].map(stop_rrt).to_numpy(dtype=float)

    # Reward-history kernel, built from the displayed stop sequence so it is
    # column-aligned: row k = the reward outcome k stops back. The rate models'
    # reward_lag is trials-back; on this stop/event axis trials and stops drift
    # apart past lag 1 (inter-trial stops), so a trials-back kernel cannot be
    # read positionally. Stops-back is aligned by construction and matches the
    # trials-back reward_lag1 ~91% of the time.
    stop_seq = evo["stop_id"].to_numpy()          # per-event stop id, time order
    uniq_stops = list(dict.fromkeys(stop_seq))    # ordered unique stops
    pos_of = {s: i for i, s in enumerate(uniq_stops)}
    ro_of = {}
    for s, r in zip(stop_seq, rew_col, strict=True):
        ro_of.setdefault(s, r)                    # reward_outcome is stop-constant
    ro_by_pos = np.array([ro_of[s] for s in uniq_stops])
    col_pos = np.array([pos_of[s] for s in stop_seq])
    kernel = np.full((N_REWARD_LAGS, n_bouts), np.nan)
    for k in range(1, N_REWARD_LAGS + 1):
        src = col_pos - k
        ok = src >= 0
        kernel[k - 1, ok] = ro_by_pos[src[ok]]

    # RL value-state block (SWR-contrast rate G*/S* + content C6-C12). These
    # columns are merged onto `sst` in main() from qvals_table + the runtime
    # recent_switch_rate; map to per-bout arrays via stop_id.
    def stop_scalar(col):
        return evo["stop_id"].map(sst.set_index("stop_id")[col]).to_numpy(dtype=float)

    postsw_col = stop_scalar("is_post_switch")  # raw 0/1/NaN
    tss_col = stop_scalar("trials_since_switch")
    tsbc_col = stop_scalar("trials_since_block_change")
    rsr_col = stop_scalar("recent_switch_rate")
    relval_col = stop_scalar("hmm_relative_value")
    rpe_col = stop_scalar("hmm_rpe")
    entropy_col = stop_scalar("reward_entropy")
    vupdate_col = stop_scalar("hmm_value_update")

    # Immobility-period (stop) membership. Each column is one non-local event;
    # several may share a stop. Per-stop covariates are drawn once per stop as a
    # cell spanning that stop's contiguous event columns (variable width), so
    # they do not visually repeat across a stop's events.
    stop_col = evo["stop_id"].to_numpy()
    stop_group = np.zeros(n_bouts, dtype=int)
    stop_group[1:] = np.cumsum(stop_col[1:] != stop_col[:-1])
    n_stops = int(stop_group[-1]) + 1
    starts = np.concatenate([[0], np.where(np.diff(stop_group) != 0)[0] + 1])
    ends = np.concatenate([starts[1:] - 1, [n_bouts - 1]])
    stop_edges = np.concatenate([[starts[0] - 0.5], ends + 0.5])  # len n_stops+1
    stop_centers = (starts + ends) / 2.0

    def to_stop(arr):
        """Collapse a per-event array (..., n_bouts) to per-stop (..., n_stops)."""
        return arr[..., starts]

    def cov_matrix(colname):
        """Pivot a per-arm covariate to (n_arms, n_bouts), rows in ARM_ROW_ORDER."""
        piv = sub.pivot_table(index="arm", columns="non_local_event",
                              values=colname, aggfunc="first")
        return piv.reindex(index=ARM_ROW_ORDER, columns=order.index).to_numpy()

    # --- panel plan (top -> bottom) -----------------------------------------
    # (kind, data, label, extra, gran); gran "event" draws per event column,
    # "stop" draws one cell per immobility period. Only content and SWR are
    # per-event; every other covariate is per-stop.
    BINARY_H, STRIP_H, HEAT_H, KERNEL_H, CONTENT_H, COV_H = \
        0.55, 0.55, 0.8, 2.6, 7.0, 4.2
    panels = [
        ("binary", rew_col, "reward", None, "stop"),
        ("binary", swr_col, "SWR", None, "event"),
        ("kernel", kernel, "reward (stops back)", ("Greys", False), "stop"),
        ("heatbout", rrt_col, "reward rate trial", ("viridis", False), "stop"),
        ("binary", switch_col, "after switch", None, "stop"),
        ("binary", omit_col, "after omission", None, "stop"),
        ("binary", postsw_col, "post switch", None, "stop"),
        ("heatbout", tss_col, "trials since switch", ("RdBu_r", True), "stop"),
        ("heatbout", tsbc_col, "trials since block", ("viridis", False), "stop"),
        ("heatbout", rsr_col, "recent switch rate (frac)", ("Oranges", False), "stop"),
        ("heatbout", relval_col, "rel value (ΔV)", ("RdBu_r", True), "stop"),
        ("heatbout", rpe_col, "RPE (reward−Q)", ("RdBu_r", True), "stop"),
        ("heatbout", entropy_col, "reward entropy", ("cividis", False), "stop"),
        ("heatbout", vupdate_col, "value update (|ΔQ|)", ("magma", False), "stop"),
        ("armstrip", prev_arm_col, "prev arm", None, "stop"),
        ("armstrip", lastrew_arm_col, "last rew arm", None, "stop"),
        ("content", None, "non-local content", None, "event"),
        ("cov", "q_arm_hmm", "HMM value (Q)", ("RdBu_r", True), "stop"),
        ("cov", "q_arm", "belief value (P)", ("viridis", False), "stop"),
        ("cov", "reward_rate_arm_k10", "reward rate k10 (frac)", ("plasma", False), "stop"),
        ("cov", "p_reward_arm", "P(reward)", ("cividis", False), "stop"),
        ("cov", "cumulative_reward", "cumul reward (n)", ("YlGn", False), "stop"),
        ("cov", "visit_recency_demeaned", "visit recency (dem)", ("PuOr", True), "stop"),
        ("cov", "visit_recency", "visit recency (raw)", ("Purples", False), "stop"),
        ("cov", "arm_dist_rank", "arm dist rank (1=near)", ("magma_r", False), "stop"),
    ]
    hmap = {"binary": BINARY_H, "heatbout": HEAT_H, "kernel": KERNEL_H,
            "armstrip": STRIP_H, "content": CONTENT_H, "cov": COV_H}
    heights = [hmap[k] for k, *_ in panels]

    fig = plt.figure(figsize=(max(11, n_bouts * 0.1), 0.155 * sum(heights) + 1.6),
                     constrained_layout=True)
    gs = GridSpec(len(panels), 2, figure=fig, height_ratios=heights,
                  width_ratios=[60, 1])
    axes = []
    ax0 = None
    for i, _panel in enumerate(panels):
        ax = fig.add_subplot(gs[i, 0], sharex=ax0)
        ax0 = ax0 or ax
        axes.append(ax)

    def style_track(axis, label):
        axis.set_ylim(-0.5, 0.5)
        axis.set_yticks([0])
        axis.set_yticklabels([label], fontsize=8)
        axis.tick_params(axis="both", length=0)
        for s in ("top", "right", "left", "bottom"):
            axis.spines[s].set_visible(False)

    def arm_yticks(axis, full):
        axis.set_yticks(range(n_arms))
        labels = [ARM_LABEL[a] if full else str(a) for a in ARM_ROW_ORDER]
        axis.set_yticklabels(labels, fontsize=(8 if full else 6.5))
        for tick, arm in zip(axis.get_yticklabels(), ARM_ROW_ORDER, strict=True):
            tick.set_color(palette[arm][:3])
            if full:
                tick.set_fontweight("bold")
        axis.tick_params(axis="both", length=0)

    def row_ylabel(axis, text):
        axis.set_ylabel(text, fontsize=8, rotation=0, ha="right", va="center",
                        labelpad=22)
        for s in ("top", "right", "left", "bottom"):
            axis.spines[s].set_visible(False)

    def cbar(axis_idx, mappable, ticks=None):
        cb = fig.colorbar(mappable, cax=fig.add_subplot(gs[axis_idx, 1]),
                          ticks=ticks)
        cb.ax.tick_params(labelsize=6, length=2)

    for i, (kind, data, label, extra, gran) in enumerate(panels):
        ax = axes[i]
        if kind == "content":
            # Per-event content: shaded bands mark immobility periods (stops).
            for g in range(n_stops):
                if g % 2 == 0:
                    ax.axvspan(starts[g] - 0.5, ends[g] + 0.5, color="0.5",
                               alpha=0.10, lw=0, zorder=0)
            frac = sub["arm_fraction_posterior"].to_numpy().clip(0, 1)
            cc = np.array([palette[a] for a in sub["arm"]], dtype=float)
            cc[:, 3] = frac
            ax.scatter(sub["col"], sub["y"], s=32, c=cc, edgecolors="none",
                       zorder=2, rasterized=True)
            cur = sub[sub["is_current"] == 1]
            ax.scatter(cur["col"], cur["y"], s=46, facecolors="none",
                       edgecolors="black", linewidths=0.9, zorder=3, rasterized=True)
            for yb in (2.5, 5.5):
                ax.axhline(yb, color="0.85", lw=0.7, zorder=1)
            ax.set_ylim(-0.5, n_arms - 0.5)
            # Content uses y_of_arm coords (arm 0 at top); tick each arm at its y.
            ax.set_yticks([y_of_arm[a] for a in ARM_ROW_ORDER])
            ax.set_yticklabels([ARM_LABEL[a] for a in ARM_ROW_ORDER], fontsize=8)
            for tick, arm in zip(ax.get_yticklabels(), ARM_ROW_ORDER, strict=True):
                tick.set_color(palette[arm][:3])
                tick.set_fontweight("bold")
            ax.tick_params(axis="both", length=0)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        elif kind == "binary" and gran == "event":
            # Per-event binary (SWR): one square per event; missing -> grey x.
            v = np.asarray(data, dtype=float)
            no, yes, miss = v == 0, v == 1, ~np.isfinite(v)
            ax.scatter(cols[no], np.zeros(no.sum()), s=26, marker="s", c="0.85",
                       edgecolors="0.6", linewidths=0.35, zorder=2, rasterized=True)
            ax.scatter(cols[yes], np.zeros(yes.sum()), s=26, marker="s",
                       c="#222222", edgecolors="none", zorder=2, rasterized=True)
            if miss.any():
                ax.scatter(cols[miss], np.zeros(miss.sum()), s=16, marker="x",
                           c="0.6", linewidths=0.6, zorder=3, rasterized=True)
            style_track(ax, label)
        elif kind == "binary":
            # Per-stop binary: one cell per immobility period spanning its band.
            vs = to_stop(np.asarray(data, dtype=float))
            cmap = ListedColormap(["0.85", "#222222"])
            cmap.set_bad("white")
            ax.pcolormesh(stop_edges, [-0.5, 0.5],
                          np.ma.masked_invalid(vs)[None, :], cmap=cmap,
                          norm=BoundaryNorm([-0.5, 0.5, 1.5], 2), rasterized=True)
            miss = ~np.isfinite(vs)
            if miss.any():
                ax.scatter(stop_centers[miss], np.zeros(miss.sum()), s=16,
                           marker="x", c="0.6", linewidths=0.6, zorder=3,
                           rasterized=True)
            style_track(ax, label)
        elif kind == "armstrip":
            # Per-stop arm identity: one arm-colored cell per immobility period.
            vs = to_stop(np.asarray(data, dtype=float))
            cmap = ListedColormap([palette[a][:3] for a in range(9)])
            cmap.set_bad("white")
            ax.pcolormesh(stop_edges, [-0.5, 0.5],
                          np.ma.masked_invalid(vs)[None, :], cmap=cmap,
                          norm=BoundaryNorm(np.arange(-0.5, 9.5, 1.0), 9),
                          rasterized=True)
            style_track(ax, label)
        elif kind == "heatbout":
            # Per-stop scalar: one cell per immobility period.
            cmap_name, diverging = extra
            vs = np.ma.masked_invalid(to_stop(np.asarray(data, dtype=float)))
            cmap = plt.get_cmap(cmap_name).copy()
            cmap.set_bad("0.9")
            vmin, vmax = _vlims(vs, diverging)
            mesh = ax.pcolormesh(stop_edges, [-0.5, 0.5], vs[None, :], cmap=cmap,
                                 vmin=vmin, vmax=vmax, rasterized=True)
            style_track(ax, label)
            cbar(i, mesh, ticks=[vmin, vmax])
        elif kind == "kernel":
            # Per-stop reward history (already stop-derived); one column per stop.
            cmap_name, _ = extra
            n_lag = data.shape[0]
            Ks = np.ma.masked_invalid(to_stop(data))  # (n_lag, n_stops)
            cmap = plt.get_cmap(cmap_name).copy()
            cmap.set_bad("0.9")
            mesh = ax.pcolormesh(stop_edges, np.arange(n_lag + 1) - 0.5, Ks,
                                 cmap=cmap, vmin=0, vmax=1, rasterized=True)
            ax.set_ylim(n_lag - 0.5, -0.5)  # lag 1 at top
            ax.set_yticks(range(n_lag))
            ax.set_yticklabels([str(j + 1) for j in range(n_lag)], fontsize=5.5)
            ax.tick_params(axis="both", length=0)
            row_ylabel(ax, label)
            cbar(i, mesh, ticks=[0, 1])
        elif kind == "cov":
            # Per-stop, per-arm: one column per immobility period.
            cmap_name, diverging = extra
            Ms = np.ma.masked_invalid(to_stop(cov_matrix(data)))  # (n_arms, n_stops)
            cmap = plt.get_cmap(cmap_name).copy()
            cmap.set_bad("0.9")
            vmin, vmax = _vlims(Ms, diverging)
            mesh = ax.pcolormesh(stop_edges, np.arange(n_arms + 1) - 0.5, Ms,
                                 cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
            ax.set_ylim(n_arms - 0.5, -0.5)  # arm 0 at top
            arm_yticks(ax, full=False)
            row_ylabel(ax, label)
            cbar(i, mesh)

    for ax in axes[:-1]:
        ax.tick_params(labelbottom=False)
    axes[0].set_title(
        f"{title_stub}  —  non-local bout covariate browser  "
        f"({n_bouts} non-local events in {n_stops} immobility periods, "
        f"{int((rew_col == 1).sum())} rewarded, {int((swr_col == 1).sum())} SWR)",
        fontsize=11,
    )
    axes[-1].set_xlim(-0.5, n_bouts - 0.5)
    axes[-1].set_xlabel(
        "non-local event (time-ordered; immobility, speed < 4 cm/s)  —  content "
        "and SWR are per event; every other row is per-stop, drawn as one cell "
        "spanning that immobility period's events",
        fontsize=9,
    )

    legend = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="0.35",
               markersize=8, label="content: opacity = posterior fraction"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor="black", markersize=8, label="animal's current arm"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#222222",
               markersize=7, label="binary track filled = yes"),
        Line2D([0], [0], marker="x", color="0.6", linestyle="none",
               markersize=7, label="missing covariate"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=palette[2],
               markersize=7, label="arm strip: color = arm identity"),
    ]
    axes[-1].legend(handles=legend, loc="upper left", bbox_to_anchor=(0, -0.55),
                    frameon=False, fontsize=7.5, ncol=5)
    return fig, n_bouts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--animal", default="chimi")
    p.add_argument("--day", type=int, default=20200227)
    p.add_argument("--epoch", default="04_r2")
    p.add_argument("--out", default=None,
                   help="output PNG path (default: .claude/research/...)")
    p.add_argument("--pdf", action="store_true",
                   help="also write a vector PDF alongside the PNG")
    p.add_argument("--dpi", type=int, default=150,
                   help="raster DPI (heatmaps/scatter are rasterized)")
    args = p.parse_args()

    event_arm = pd.read_pickle(RC_DIR / "event_arm_table.pkl")
    # Stop table + RL value-state block, assembled as the SWR-contrast runners
    # do: recent_switch_rate via add_uncertainty_covariates (default
    # switch_window=5), the rest merged from qvals_table on stop_id. Two
    # deliberate differences from the *rate* runner, both fine for eyeballing:
    # this uses how="left" (like the content runner) so bouts whose stop is
    # absent from qvals are kept and shown masked, rather than dropped by the
    # rate runner's inner join; and covariates are shown raw, not z-scored as
    # the fitters standardize them.
    stops = add_uncertainty_covariates(pd.read_pickle(RC_DIR / "stop_table.pkl"))
    qvals = pd.read_pickle(RC_DIR / "models_per_bin_rate" / "qvals_table.pkl")
    value_state = ["trials_since_switch", "is_post_switch",
                   "trials_since_block_change", "hmm_relative_value", "hmm_rpe",
                   "reward_entropy", "hmm_value_update"]
    stops = stops.merge(qvals[["stop_id", *value_state]], on="stop_id", how="left")

    def slc(df):
        return df[(df.animal == args.animal) & (df.day == args.day)
                  & (df.epoch == args.epoch)]

    sub, sst = slc(event_arm), slc(stops)
    if sub.empty:
        raise SystemExit(
            f"No event-arm rows for {args.animal} {args.day} {args.epoch}. "
            "Check --animal/--day/--epoch against event_arm_table.pkl."
        )

    fig, n_bouts = build_figure(sub, sst, f"{args.animal} {args.day} {args.epoch}")

    out = args.out
    if out is None:
        os.makedirs(DEFAULT_OUT_DIR, exist_ok=True)
        out = str(DEFAULT_OUT_DIR
                  / f"{args.animal}{args.day}_{args.epoch}_nonlocal_bout_arms.png")
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    written = [out]
    if args.pdf:
        pdf = out[:-4] + ".pdf" if out.endswith(".png") else out + ".pdf"
        fig.savefig(pdf, dpi=args.dpi, bbox_inches="tight")
        written.append(pdf)
    print(f"wrote {', '.join(written)} | events: {n_bouts}")


if __name__ == "__main__":
    main()
