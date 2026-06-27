#!/usr/bin/env python
"""Single-session event rasters, stops in CHRONOLOGICAL order (the Czanner-style
between-trial view). Produces two figures for the same session:
  - session_raster_swr.png      : detected SWR (black) + non-SWR non-local (blue)
  - session_raster_nonlocal.png : non-local w/ SWR (purple) + non-local no-SWR (blue)

Rows = stops in the order they occurred; x = time-since-stop-onset; gray = the stop
(immobility) window; left strip = reward_outcome (green=rewarded, gray=omitted);
orange lines = patch-switch boundaries (blocks labeled). Event ticks are vlines one
data-row tall, aligned 1:1 with the shaded window. Auto-picks a representative
session unless --session animal,day,epoch is given.
"""

from __future__ import annotations

import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR as P  # noqa: E402

RC = P / "replay_content"
SC = ["animal", "day", "epoch"]
XMAX = 8.0


def draw(ss: pd.DataFrame, row: pd.Series, populations: list, out_name: str,
         title: str, rate: bool = False) -> None:
    """Draw one chronological raster with marginals. populations = list of (label, df, color).

    Marginals are onset COUNT by default, or exposure-corrected RATE (events/s) when
    rate=True. Top marginal: vs time-since-stop-onset (within-stop time course).
    Right marginal: vs stop number (across-session/between-trial axis).
    """
    def sel(df):
        s = df[df.stop_id.isin(row.index) & (df.t_since_stop_onset < XMAX)]
        return s.t_since_stop_onset.to_numpy(), row.loc[s.stop_id].to_numpy()

    fig = plt.figure(figsize=(10, 15))
    gs = fig.add_gridspec(2, 2, width_ratios=[5, 1.1], height_ratios=[1.1, 8],
                          hspace=0.03, wspace=0.03)
    ax = fig.add_subplot(gs[1, 0])
    axtop = fig.add_subplot(gs[0, 0], sharex=ax)
    axright = fig.add_subplot(gs[1, 1], sharey=ax)
    axleg = fig.add_subplot(gs[0, 1])
    axleg.axis("off")

    # patch-switch boundaries segment the session (on the raster + right marginal)
    cp = ss.current_patch.to_numpy()
    bnds = [0] + [i for i in range(1, len(cp)) if cp[i] != cp[i - 1]] + [len(cp)]
    for b in bnds[1:-1]:
        ax.axhline(b - 0.5, color="C1", lw=1.3, alpha=0.9, zorder=1)
        axright.axhline(b - 0.5, color="C1", lw=1.0, alpha=0.5, zorder=1)
    ax.plot([], [], color="C1", lw=1.3, label="patch switch")
    for lo, hi in zip(bnds[:-1], bnds[1:], strict=False):
        ax.text(-0.42, (lo + hi) / 2, f"P{int(cp[lo])}", va="center", ha="center",
                fontsize=9, color="C1", weight="bold")
    # reward strip + shaded immobility window
    rcolors = np.where(ss.reward_outcome.to_numpy() == 1, "C2", "0.7")
    ax.barh(ss.row.to_numpy(), 0.12, left=-0.30, height=1.0, color=rcolors, zorder=2)
    ax.plot([], [], marker="s", ls="", color="C2", label="rewarded stop")
    ax.plot([], [], marker="s", ls="", color="0.7", label="omitted stop")
    ax.barh(ss.row.to_numpy(), ss.immobility_time.clip(upper=XMAX).to_numpy(),
            left=0, height=1.0, color="0.88", zorder=0, label="stop (immobility) window")
    # event ticks (raster) + marginals: raw count, or exposure-corrected rate.
    # Rate divides each bin by its available immobile-seconds (top: the stop-window
    # survival; right: total immobile time per stop-bin). The top rate is masked
    # where <5 stops survive — the right-censored tail is noisy and a selected
    # long/rewarded subpopulation (see the rate-vs-count caveat).
    tbins = np.arange(0, XMAX + 0.2, 0.2)
    rbins = np.arange(0, len(ss) + 15, 15)
    durv = ss.immobility_time.to_numpy()
    rowv = ss.row.to_numpy()
    if rate:
        top_expo = np.array([np.clip(durv - lo, 0, hi - lo).sum()
                             for lo, hi in zip(tbins[:-1], tbins[1:], strict=False)])
        surv = np.array([(durv > lo).sum() for lo in tbins[:-1]])
        top_expo = np.where(surv >= 5, top_expo, np.nan)
        right_expo = np.array([durv[(rowv >= lo) & (rowv < hi)].sum()
                               for lo, hi in zip(rbins[:-1], rbins[1:], strict=False)])
        right_expo = np.where(right_expo > 0, right_expo, np.nan)
    for label, df, color in populations:
        x, y = sel(df)
        ax.vlines(x, y - 0.5, y + 0.5, color=color, lw=0.7, label=f"{label} ({len(x)})")
        ct = np.histogram(x, bins=tbins)[0].astype(float)
        cr = np.histogram(y, bins=rbins)[0].astype(float)
        if rate:
            ct = ct / top_expo
            cr = cr / right_expo
        axtop.stairs(ct, tbins, color=color, lw=1.3)
        axright.stairs(cr, rbins, color=color, lw=1.3, orientation="horizontal")
    ax.set_xlim(-0.55, XMAX)
    ax.set_ylim(len(ss), -1)
    ax.set_xlabel("time since stop onset (s)")
    ax.set_ylabel("stop number (chronological →)")
    mlab = "onset rate (events/s)" if rate else "onset count"
    axtop.set_ylabel(mlab)
    axtop.tick_params(labelbottom=False)
    axright.set_xlabel(mlab)
    axright.tick_params(labelleft=False)
    h, lab = ax.get_legend_handles_labels()
    axleg.legend(h, lab, loc="center", fontsize=8, frameon=False)
    fig.suptitle(title, x=0.5, y=0.995, fontsize=11)
    out = P.parent / "paper" / "figures" / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


def main() -> None:
    onset = pd.read_pickle(RC / "event_onset_table.pkl")
    ev = pd.read_pickle(RC / "event_table.pkl")[["stop_id", "event_start", "is_swr"]].copy()
    ev["rank"] = ev.sort_values("event_start").groupby("stop_id").cumcount()
    nl = onset[onset.event_type == "non_local"].copy()
    nl["rank"] = nl.sort_values("onset_time").groupby("stop_id").cumcount()
    nl = nl.merge(ev[["stop_id", "rank", "is_swr"]], on=["stop_id", "rank"], how="left")
    swr = onset[onset.event_type == "SWR"]
    nl_swr = nl[nl.is_swr.eq(True)]
    nl_no = nl[nl.is_swr.eq(False)]
    st = pd.read_pickle(RC / "stop_table.pkl")

    # pick session: most non-SWR onsets among sessions with 100-700 stops.
    rate = "--rate" in sys.argv
    posargs = [x for x in sys.argv[1:] if not x.startswith("--")]
    if posargs:
        a, d, e = posargs[0].split(",")
        key = (a, int(d), e)
    else:
        nstops = st[st.immobility_time > 0].groupby(SC).size().rename("nstops")
        score = (
            nl_no.groupby(SC).size().rename("nnl").to_frame()
            .join(swr.groupby(SC).size().rename("nswr"))
            .join(nstops)
            .dropna()
        )
        score = score[(score.nstops >= 100) & (score.nstops <= 700)]
        key = score.assign(m=score[["nnl", "nswr"]].min(1)).m.idxmax()
    a, d, e = key
    print(f"session: {a} {d} {e}")

    ss = st[(st.animal == a) & (st.day == d) & (st.epoch == e) & (st.immobility_time > 0)]
    ss = ss.sort_values("stop_start").reset_index(drop=True)
    ss["row"] = np.arange(len(ss))
    row = ss.set_index("stop_id")["row"]

    suffix = "_rate" if rate else ""
    base = f"{a} {d} {e} · {len(ss)} stops · marginals: {'rate' if rate else 'count'}"
    draw(ss, row, [("SWR", swr, "k"), ("non-SWR non-local", nl_no, "C0")],
         f"session_raster_swr{suffix}.png",
         f"Single-session SWR raster (chronological): {base}", rate)
    draw(ss, row, [("non-local w/ SWR", nl_swr, "C3"), ("non-local no-SWR", nl_no, "C0")],
         f"session_raster_nonlocal{suffix}.png",
         f"Single-session non-local raster (chronological): {base}", rate)


if __name__ == "__main__":
    main()
