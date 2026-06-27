#!/usr/bin/env python
"""Plot the time-resolved reward-effect GLM: fitted rewarded-vs-omitted rate
curves (the reward effect = the gap) over the quiescence time course, plus the
per-session reward rate-ratio distribution. Companion to _reward_glm.py."""

from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR as P  # noqa: E402

RC = P / "replay_content"
SC = ["animal", "day", "epoch"]
TMAX, DT, BIN_S = 4.0, 0.25, 0.004


def cells(df, rmap, value):
    df = df.copy()
    df["reward"] = df["stop_id"].map(rmap)
    df = df[(df["t_since_stop_onset"] < TMAX) & df["reward"].notna()]
    df["tbin"] = (df["t_since_stop_onset"] / DT).astype(int)
    return df.groupby([*SC, "reward", "tbin"]).size().rename(value)


def main():
    st = pd.read_pickle(RC / "stop_table.pkl")
    rmap = st.set_index("stop_id")["reward_lag1"]
    dense = pd.read_parquet(RC / "stop_timebin_table.parquet", columns=[*SC, "stop_id", "t_since_stop_onset"])
    exposure = cells(dense, rmap, "exposure")
    del dense
    onset = pd.read_pickle(RC / "event_onset_table.pkl")
    ev = pd.read_pickle(RC / "event_table.pkl")[["stop_id", "event_start", "is_swr"]].copy()
    ev["rank"] = ev.sort_values("event_start").groupby("stop_id").cumcount()
    nl = onset[onset.event_type == "non_local"].copy()
    nl["rank"] = nl.sort_values("onset_time").groupby("stop_id").cumcount()
    nl = nl.merge(ev[["stop_id", "rank", "is_swr"]], on=["stop_id", "rank"], how="left")
    pops = {
        "detected SWR": onset[onset.event_type == "SWR"],
        "non-local w/ SWR": nl[nl.is_swr.eq(True)],
        "non-local NO SWR": nl[nl.is_swr.eq(False)],
    }
    tc_grid = np.linspace(DT / 2, TMAX - DT / 2, 60)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    rr_session = {}
    for ax, (name, src) in zip(axes.flat[:3], pops.items(), strict=False):
        cell = exposure.to_frame().join(cells(src, rmap, "count")).reset_index().fillna({"count": 0})
        cell["tc"] = (cell["tbin"] + 0.5) * DT
        cell = cell[cell["exposure"] > 0]
        # pooled fit for the fitted curves
        pooled = cell.groupby(["reward", "tbin", "tc"], as_index=False).agg(count=("count", "sum"), exposure=("exposure", "sum"))
        m = smf.glm("count ~ bs(tc, df=4) + reward", data=pooled, family=sm.families.Poisson(),
                    offset=np.log(pooled["exposure"].to_numpy())).fit()
        for rew, ls, lab in [(1.0, "-", "rewarded"), (0.0, "--", "omitted")]:
            pred = pd.DataFrame({"tc": tc_grid, "reward": rew})
            rate = m.predict(pred, offset=np.zeros(len(pred))) / BIN_S  # onsets / s
            ax.plot(tc_grid, rate, ls, color="C1" if rew else "C0", lw=2, label=f"fit: {lab}")
        for rew, mk in [(1.0, "o"), (0.0, "x")]:
            d = pooled[pooled.reward == rew]
            ax.plot(d["tc"], d["count"] / d["exposure"] / BIN_S, mk, color="C1" if rew else "C0", ms=4, alpha=0.6)
        # per-session RR
        rows = []
        for keys, g in cell.groupby(SC, sort=False):
            if g["reward"].nunique() < 2 or len(g) < 8:
                continue
            try:
                r = smf.glm("count ~ bs(tc, df=4) + reward", data=g, family=sm.families.Poisson(),
                            offset=np.log(g["exposure"].to_numpy())).fit()
                rows.append(np.exp(r.params["reward"]))
            except Exception:
                continue
        rr_session[name] = np.array(rows)
        ax.set_title(f"{name}  (RR = {np.median(rr_session[name]):.2f})")
        ax.set_xlabel("time into stop (s)")
        ax.set_ylabel("onset rate (events / s)")
        ax.legend(frameon=False, fontsize=8)

    axd = axes.flat[3]
    data = [rr_session[n] for n in pops]
    axd.boxplot(data, labels=["SWR", "NL\nw/SWR", "NL\nNO SWR"], showfliers=False)
    axd.axhline(1.0, color="0.5", ls=":")
    axd.set_ylabel("reward rate-ratio (per session)")
    axd.set_title("Reward effect by population (1.0 = none)")
    axd.set_ylim(0, 5)
    fig.suptitle("Time-resolved reward GLM: rewarded vs omitted, controlling for time-into-stop", y=1.0)
    fig.tight_layout()
    out = P.parent / "paper" / "figures" / "reward_glm_effects.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("saved", out)
    for n in pops:
        print(f"  {n}: median RR {np.median(rr_session[n]):.2f}, n={len(rr_session[n])}")


if __name__ == "__main__":
    main()
