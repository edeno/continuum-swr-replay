#!/usr/bin/env python
"""Single-session event raster, stops in CHRONOLOGICAL order (the Czanner-style
between-trial view). Rows = stops in the order they occurred; x = time-since-stop-
onset; ticks = SWR (black) and non-SWR non-local (blue) onsets; left strip = reward
outcome; light marker = each stop's end (duration). A gradient down the rows would
be across-session drift (the SS-GLM between-trial state). Auto-picks a representative
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


def main() -> None:
    onset = pd.read_pickle(RC / "event_onset_table.pkl")
    ev = pd.read_pickle(RC / "event_table.pkl")[["stop_id", "event_start", "is_swr"]].copy()
    ev["rank"] = ev.sort_values("event_start").groupby("stop_id").cumcount()
    nl = onset[onset.event_type == "non_local"].copy()
    nl["rank"] = nl.sort_values("onset_time").groupby("stop_id").cumcount()
    nl = nl.merge(ev[["stop_id", "rank", "is_swr"]], on=["stop_id", "rank"], how="left")
    swr = onset[onset.event_type == "SWR"]
    nlno = nl[nl.is_swr.eq(False)]
    st = pd.read_pickle(RC / "stop_table.pkl")

    # pick session: most non-SWR onsets among sessions with 100-700 stops.
    if len(sys.argv) > 1:
        a, d, e = sys.argv[1].split(",")
        key = (a, int(d), e)
    else:
        nstops = st[st.immobility_time > 0].groupby(SC).size().rename("nstops")
        score = (
            nlno.groupby(SC).size().rename("nnl").to_frame()
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

    def sel(df):
        s = df[df.stop_id.isin(row.index) & (df.t_since_stop_onset < XMAX)]
        return s.t_since_stop_onset.to_numpy(), row.loc[s.stop_id].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 14))
    # patch-switch boundaries (where current_patch changes) segment the session
    # into patch-blocks; label each block with its patch. (is_after_switch marks
    # the multi-stop post-switch window, not the boundary — use the change point.)
    cp = ss.current_patch.to_numpy()
    bnds = [0] + [i for i in range(1, len(cp)) if cp[i] != cp[i - 1]] + [len(cp)]
    for b in bnds[1:-1]:
        ax.axhline(b - 0.5, color="C1", lw=1.3, alpha=0.9, zorder=1)
    ax.plot([], [], color="C1", lw=1.3, label="patch switch")
    for lo, hi in zip(bnds[:-1], bnds[1:], strict=False):
        ax.text(-0.42, (lo + hi) / 2, f"P{int(cp[lo])}", va="center", ha="center",
                fontsize=9, color="C1", weight="bold")
    # reward strip (left margin): one row-tall bar per stop, reward at THIS stop
    # (reward_outcome = the direct per-stop reward, not the reward-kernel lag).
    rcolors = np.where(ss.reward_outcome.to_numpy() == 1, "C2", "0.7")
    ax.barh(ss.row.to_numpy(), 0.12, left=-0.30, height=1.0, color=rcolors, zorder=2)
    ax.plot([], [], marker="s", ls="", color="C2", label="rewarded stop")
    ax.plot([], [], marker="s", ls="", color="0.7", label="omitted stop")
    # shade each stop's immobility window (0 -> stop end) = the available time
    ax.barh(ss.row.to_numpy(), ss.immobility_time.clip(upper=XMAX).to_numpy(),
            left=0, height=1.0, color="0.88", zorder=0, label="stop (immobility) window")
    # event ticks — exactly one row tall (vlines, data units) so each event aligns
    # 1:1 with its stop's shaded window; a scatter "|" marker would be a fixed
    # point size (~4x the row height at this density).
    x, y = sel(swr)
    ax.vlines(x, y - 0.5, y + 0.5, color="k", lw=0.7, label=f"SWR ({len(x)})")
    x, y = sel(nlno)
    ax.vlines(x, y - 0.5, y + 0.5, color="C0", lw=0.7, label=f"non-SWR non-local ({len(x)})")
    ax.set_xlim(-0.55, XMAX)
    ax.set_ylim(len(ss), -1)
    ax.set_xlabel("time since stop onset (s)")
    ax.set_ylabel("stop number (chronological →)")
    ax.set_title(f"Single-session event raster (chronological): {a} {d} {e} · {len(ss)} stops")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    fig.tight_layout()
    out = P.parent / "paper" / "figures" / "session_raster_chronological.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)


if __name__ == "__main__":
    main()
