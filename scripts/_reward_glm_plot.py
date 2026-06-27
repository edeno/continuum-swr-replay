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
from continuum_swr_replay.analysis.replay import (  # noqa: E402
    aggregate_to_timebins,
    attach_is_swr_to_onsets,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR as P  # noqa: E402

RC = P / "replay_content"
SC = ["animal", "day", "epoch"]
TMAX, DT = 4.0, 0.25


def main() -> None:
    st = pd.read_pickle(RC / "stop_table.pkl")
    onset = attach_is_swr_to_onsets(
        pd.read_pickle(RC / "event_onset_table.pkl"),
        pd.read_pickle(RC / "event_table.pkl"),
    )
    nl = onset[onset.event_type == "non_local"]
    pops = {
        "detected SWR": onset[onset.event_type == "SWR"],
        "non-local w/ SWR": nl[nl.is_swr.eq(True)],
        "non-local NO SWR": nl[nl.is_swr.eq(False)],
    }
    tc_grid = np.linspace(DT / 2, TMAX - DT / 2, 60)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    rr_session = {}
    for ax, (name, src) in zip(axes.flat[:3], pops.items(), strict=False):
        # (session x reward x time-bin) cells from the canonical (stop x bin)
        # design; exposure is in seconds, so count / exposure is events / s.
        df = aggregate_to_timebins(st, src, bin_s=DT, tmax=TMAX, reward_lags=1)
        cell = (
            df.groupby([*SC, "reward_lag1", "binj"], as_index=False)
            .agg(count=("count", "sum"), exposure=("exposure", "sum"))
            .rename(columns={"reward_lag1": "reward"})
        )
        cell["tc"] = (cell["binj"] + 0.5) * DT
        cell = cell[cell["exposure"] > 0]
        # pooled fit for the fitted curves
        pooled = cell.groupby(["reward", "binj", "tc"], as_index=False).agg(count=("count", "sum"), exposure=("exposure", "sum"))
        m = smf.glm("count ~ bs(tc, df=4) + reward", data=pooled, family=sm.families.Poisson(),
                    offset=np.log(pooled["exposure"].to_numpy())).fit()
        for rew, ls, lab in [(1.0, "-", "rewarded"), (0.0, "--", "omitted")]:
            pred = pd.DataFrame({"tc": tc_grid, "reward": rew})
            rate = m.predict(pred, offset=np.zeros(len(pred)))  # onsets / s
            ax.plot(tc_grid, rate, ls, color="C1" if rew else "C0", lw=2, label=f"fit: {lab}")
        for rew, mk in [(1.0, "o"), (0.0, "x")]:
            d = pooled[pooled.reward == rew]
            ax.plot(d["tc"], d["count"] / d["exposure"], mk, color="C1" if rew else "C0", ms=4, alpha=0.6)
        # per-session RR
        rows = []
        for _keys, g in cell.groupby(SC, sort=False):
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
