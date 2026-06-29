#!/usr/bin/env python
"""Reality check on the reward-modulated non-SWR content.

TEMPORARY / exploratory. Joins the spike-evidence engine's per-event reality
metric (dll_per_spike: do the event's spikes support the decoded/MAP position over
the animal's own position, vs a matched non-event baseline ~ 0) to the stop's
reward outcome (reward_lag1), to ask whether the *post-reward off-current* non-SWR
content is spike-supported (real displacement) rather than a consummation/decode
artifact.

Engine output must carry start_time (added 2026-06-28). Events are matched to the
stop whose [stop_start, stop_start+immobility_time] interval contains start_time,
to attach reward_lag1. The engine `group` already splits non-SWR into remote
(off-current, |ahead-behind| > REMOTE) vs close (current-ish).

Usage: python scripts/_reality_reward_test.py [parquet_path]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

RC = PROCESSED_DATA_DIR / "replay_content"
PARQUET = sys.argv[1] if len(sys.argv) > 1 else (
    "/tmp/claude-39494/reality_reward/event_spike_evidence_metrics.parquet"
)


def main() -> None:
    ev = pd.read_parquet(PARQUET)
    if "start_time" not in ev.columns:
        raise SystemExit("parquet lacks start_time — re-run the engine with the patch")
    stop = pd.read_pickle(RC / "stop_table.pkl")
    stop = stop.assign(
        epoch_id=stop["animal"].astype(str) + stop["day"].astype(str)
        + "_" + stop["epoch"].astype(str) + "_HPC",
        stop_end=stop["stop_start"] + stop["immobility_time"],
    )

    # match each engine event to the stop interval containing its start_time
    rew = np.full(len(ev), np.nan)
    for eid, g in ev.groupby("epoch_id"):
        s = stop[stop.epoch_id == eid].sort_values("stop_start")
        if s.empty:
            continue
        st = s["stop_start"].to_numpy()
        en = s["stop_end"].to_numpy()
        rl = s["reward_lag1"].to_numpy(dtype=float)
        t = g["start_time"].to_numpy()
        idx = np.searchsorted(st, t, side="right") - 1
        ok = (idx >= 0) & (idx < len(st))
        within = np.zeros(len(t), dtype=bool)
        within[ok] = t[ok] <= en[idx[ok]]
        vals = np.full(len(t), np.nan)
        vals[ok & within] = rl[idx[ok & within]]
        rew[g.index.to_numpy()] = vals
    ev["reward_lag1"] = rew
    matched = ev["reward_lag1"].notna()
    print(f"events={len(ev)}, matched to a stop={matched.sum()} "
          f"({100*matched.mean():.0f}%); sessions={ev.epoch_id.nunique()}\n")

    d = ev[matched].copy()
    print(f"{'group':<16}{'reward':<9}{'n':>5}{'dll/spike med':>15}{'%>0':>7}")
    for grp in ["baseline", "nonSWR-close", "nonSWR-remote", "SWR-remote"]:
        sub = d[d.group == grp]
        for rlab, rmask in [("omit", sub.reward_lag1 == 0), ("rewarded", sub.reward_lag1 == 1)]:
            s = sub[rmask]["dll_per_spike"]
            if len(s) == 0:
                continue
            print(f"{grp:<16}{rlab:<9}{len(s):>5}{s.median():>15.3f}{(s>0).mean()*100:>6.0f}%")
        print()

    # key test: is non-SWR (esp. remote/off-current) content spike-supported AFTER reward?
    nr = d[(d.group == "nonSWR-remote") & (d.reward_lag1 == 1)]["dll_per_spike"]
    base = d[d.group == "baseline"]["dll_per_spike"]
    print(f"post-reward nonSWR-remote dll/spike: median {nr.median():.3f}, "
          f"{(nr>0).mean()*100:.0f}% > 0, n={len(nr)}  (baseline median {base.median():.3f})")

    # content shift: is the off-current (remote) fraction of non-SWR events higher after reward?
    ns = d[d.group.isin(["nonSWR-close", "nonSWR-remote"])]
    for rlab, rmask in [("omit", ns.reward_lag1 == 0), ("rewarded", ns.reward_lag1 == 1)]:
        sub = ns[rmask]
        frac = (sub.group == "nonSWR-remote").mean() if len(sub) else np.nan
        print(f"non-SWR off-current(remote) fraction [{rlab}]: {frac:.2f} (n={len(sub)})")


if __name__ == "__main__":
    main()
