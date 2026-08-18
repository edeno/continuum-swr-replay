#!/usr/bin/env python
"""Time-resolved event-onset rate vs time-since-reward-pump (Part B core).

The confound-free version of the lag-1 consummatory boost: per-bin onset rate
as a function of t_since_pump over ALL immobility bins (small t = post-reward
consummatory; large t = far from reward). Numerator = onset counts per t-bin
(event_onset_table); denominator = immobility-bin exposure per t-bin
(stop_timebin_table). Non-local onsets are split by canonical event-level
is_swr (matched within stop by time-rank).

rate(t) = onsets(t) / (exposure_bins(t) * bin_s)   [events / s of immobility]

NB: does NOT use is_reward_stop (that builder flag is broken — pumps fire at
the poke, just before the bout, so ~none fall strictly inside a bout).
"""

from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
from continuum_swr_replay.analysis.replay import attach_is_swr_to_onsets  # noqa: E402
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR as P  # noqa: E402
from continuum_swr_replay.table_io import read_table  # noqa: E402

RC = P / "replay_content"
BIN_S = 0.004
TBINS = np.arange(0, 10.01, 0.25)
CENT = 0.5 * (TBINS[:-1] + TBINS[1:])
COLORS = {"detected SWR": "k", "non-local w/ SWR": "C3", "non-local NO SWR": "C0"}


def main() -> None:
    dense = pd.read_parquet(RC / "stop_timebin_table.parquet", columns=["t_since_pump"])
    exp_h, _ = np.histogram(dense.t_since_pump.dropna().to_numpy(), TBINS)
    sec = exp_h * BIN_S  # seconds of immobility exposure per t-bin

    onset = attach_is_swr_to_onsets(
        read_table(RC / "event_onset_table.pkl"),
        read_table(RC / "event_table.pkl"),
    )
    nl = onset[onset.event_type == "non_local"]

    pops = {
        "detected SWR": onset.loc[onset.event_type == "SWR", "t_since_pump"],
        "non-local w/ SWR": nl.loc[nl.is_swr.eq(True), "t_since_pump"],
        "non-local NO SWR": nl.loc[nl.is_swr.eq(False), "t_since_pump"],
    }

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    print("population        | n      | peak rate | t_peak(s) | rate 7-10s | peak/tail")
    for name, t in pops.items():
        h, _ = np.histogram(t.dropna().to_numpy(), TBINS)
        rate = h / np.where(sec > 0, sec, np.nan)
        ax.plot(CENT, rate, color=COLORS[name], lw=2, label=f"{name} (n={int(t.notna().sum()):,})")
        pk = np.nanmax(rate)
        tpk = CENT[np.nanargmax(rate)]
        tail = h[(CENT >= 7) & (CENT < 10)].sum() / max(sec[(CENT >= 7) & (CENT < 10)].sum(), 1e-9)
        print(f"{name:17s} | {int(t.notna().sum()):6,} | {pk:9.3f} | {tpk:9.2f} | {tail:10.3f} | {pk / max(tail, 1e-9):.2f}")
    ax.axvspan(0, 1.0, color="0.9", zorder=0)
    ax.text(0.5, ax.get_ylim()[1] * 0.04, "consumption", ha="center", fontsize=8, color="0.4")
    ax.set_xlabel("time since reward pump (s)")
    ax.set_ylabel("onset rate (events / s of immobility)")
    ax.set_title("Time-resolved onset rate vs time since reward\n(equal-exposure 4 ms bins, all immobility)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(0, 10)
    fig.tight_layout()
    out = P.parent / ".claude" / "research" / "rate_psth_outputs" / "rate_timecourse_vs_pump.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
