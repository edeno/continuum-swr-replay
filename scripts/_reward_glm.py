#!/usr/bin/env python
"""Time-resolved reward-effect GLM (Part B).

Per session, per population: Poisson on per-bin onset counts aggregated into
(reward x time-into-stop) cells, with a natural-spline time course and a reward
main effect; offset = log(exposure bins). The `reward` coefficient is the
per-second rate ratio of rewarded vs omitted stops *controlling for time-into-
immobility* — the dwell-free reward effect. Tests whether non-SWR events are
reward-independent (RR ~ 1) while SWR retains a boost (RR > 1).

reward = reward_lag1 (immediate trial rewarded). Time-into-stop is capped where
both rewarded and omitted stops have exposure.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore")
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR as P  # noqa: E402

RC = P / "replay_content"
SC = ["animal", "day", "epoch"]
TMAX, DT = 4.0, 0.25


def cells(df: pd.DataFrame, rmap: pd.Series, value: str) -> pd.DataFrame:
    df = df.copy()
    df["reward"] = df["stop_id"].map(rmap)
    df = df[(df["t_since_stop_onset"] < TMAX) & df["reward"].notna()]
    df["tbin"] = (df["t_since_stop_onset"] / DT).astype(int)
    g = df.groupby([*SC, "reward", "tbin"]).size().rename(value)
    return g


def main() -> None:
    st = pd.read_pickle(RC / "stop_table.pkl")
    rmap = st.set_index("stop_id")["reward_lag1"]

    dense = pd.read_parquet(
        RC / "stop_timebin_table.parquet",
        columns=[*SC, "stop_id", "t_since_stop_onset"],
    )
    exposure = cells(dense, rmap, "exposure")
    del dense

    onset = pd.read_pickle(RC / "event_onset_table.pkl")
    ev = pd.read_pickle(RC / "event_table.pkl")[["stop_id", "event_start", "is_swr"]].copy()
    ev["rank"] = ev.sort_values("event_start").groupby("stop_id").cumcount()
    nl = onset[onset.event_type == "non_local"].copy()
    nl["rank"] = nl.sort_values("onset_time").groupby("stop_id").cumcount()
    nl = nl.merge(ev[["stop_id", "rank", "is_swr"]], on=["stop_id", "rank"], how="left")

    pops = {
        "SWR": onset[onset.event_type == "SWR"],
        "NL w/ SWR": nl[nl.is_swr.eq(True)],
        "NL NO SWR": nl[nl.is_swr.eq(False)],
    }

    coefs: dict[str, pd.DataFrame] = {}
    for name, src in pops.items():
        cnt = cells(src, rmap, "count")
        cell = (
            exposure.to_frame().join(cnt).reset_index().fillna({"count": 0})
        )
        cell["tc"] = (cell["tbin"] + 0.5) * DT
        rows = []
        for keys, g in cell.groupby(SC, sort=False):
            g = g[g["exposure"] > 0]
            if g["reward"].nunique() < 2 or len(g) < 8:
                continue
            try:
                res = smf.glm(
                    "count ~ bs(tc, df=4) + reward",
                    data=g,
                    family=sm.families.Poisson(),
                    offset=np.log(g["exposure"].to_numpy()),
                ).fit()
                rows.append({**dict(zip(SC, keys, strict=True)), "reward": res.params["reward"]})
            except Exception:
                continue
        coefs[name] = pd.DataFrame(rows)

    print(f"time-controlled reward effect (RR = exp(reward coef)), t<{TMAX}s, {DT}s cells")
    print("RR>1 = rewarded stops have a higher per-second rate at matched time-into-stop\n")
    for name, d in coefs.items():
        v = d["reward"].dropna()
        rr = np.exp(v.median())
        print(f"  {name:11s}: RR={rr:.2f} (median coef {v.median():+.3f}), {(v > 0).sum()}/{len(v)} sessions+ ")

    # dissociation: SWR vs NL NO SWR (paired)
    m = coefs["SWR"].merge(coefs["NL NO SWR"], on=SC, suffixes=("_swr", "_nl"))
    diff = m["reward_swr"] - m["reward_nl"]
    pa = m.assign(d=diff).groupby("animal")["d"].median()
    print(
        f"\nSWR reward effect > non-SWR (paired): {(diff > 0).mean():.0%} of {len(m)} sessions, "
        f"per-animal {(pa > 0).sum()}/{m.animal.nunique()}, p={wilcoxon(m.reward_swr, m.reward_nl).pvalue:.1e}"
    )


if __name__ == "__main__":
    main()
