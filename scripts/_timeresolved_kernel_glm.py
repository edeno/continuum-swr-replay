#!/usr/bin/env python
"""Unified time-resolved reward-kernel GLM (Part B).

Per session, per population: Poisson on onset counts aggregated into
(stop, time-into-stop bin) cells, with a natural-spline time course and the
lagged reward kernel reward_lag1..K; offset = log(exposure bins). reward_lag1
is the immediate/consummatory term (controlled for the quiescence time course
AND the recent reward history reward_lag2..K); reward_lag>=2 is the dwell-clean
RPE baseline. Optionally a reward_lag1 x spline(t) interaction tests whether the
consummatory effect concentrates early. Fit per session, aggregate within
animal -> sidesteps the pooled-weighting (Simpson) issue.
"""

from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR as P  # noqa: E402

RC = P / "replay_content"
SC = ["animal", "day", "epoch"]
TMAX, DT = 4.0, 0.25
K = 5
LAGS = [f"reward_lag{j}" for j in range(1, K + 1)]


def onset_cells(df: pd.DataFrame) -> pd.Series:
    df = df[df["t_since_stop_onset"] < TMAX].copy()
    df["tbin"] = (df["t_since_stop_onset"] / DT).astype(int)
    return df.groupby(["stop_id", "tbin"]).size().rename("count")


def fit_pop(base: pd.DataFrame, cnt: pd.Series, interaction: bool) -> pd.DataFrame:
    cell = base.join(cnt).reset_index().fillna({"count": 0})
    formula = "count ~ bs(tc, df=4) + " + " + ".join(LAGS)
    if interaction:
        formula += " + reward_lag1:bs(tc, df=4)"
    rows = []
    for keys, g in cell.groupby(SC, sort=False):
        g = g.dropna(subset=LAGS)
        if len(g) < 40 or g["reward_lag1"].nunique() < 2:
            continue
        try:
            res = smf.glm(
                formula, data=g, family=sm.families.Poisson(),
                offset=np.log(g["exposure"].to_numpy()),
            ).fit()
            rows.append({**dict(zip(SC, keys, strict=True)),
                         **{c: res.params.get(c, np.nan) for c in LAGS}})
        except Exception:
            continue
    return pd.DataFrame(rows)


def main() -> None:
    st = pd.read_pickle(RC / "stop_table.pkl")
    keep = ["stop_id", *SC, *LAGS]
    stops = st[keep].drop_duplicates("stop_id").set_index("stop_id")

    dense = pd.read_parquet(RC / "stop_timebin_table.parquet",
                            columns=["stop_id", "t_since_stop_onset"])
    dense = dense[dense["t_since_stop_onset"] < TMAX].copy()
    dense["tbin"] = (dense["t_since_stop_onset"] / DT).astype(int)
    exposure = dense.groupby(["stop_id", "tbin"]).size().rename("exposure")
    del dense
    base = exposure.to_frame().join(stops, on="stop_id")
    base["tc"] = (base.index.get_level_values("tbin") + 0.5) * DT

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

    coefs = {n: fit_pop(base, onset_cells(s), interaction=False) for n, s in pops.items()}
    print(f"=== reward kernel (RR=exp coef), per-session median, t<{TMAX}s, K={K} ===")
    print("(reward_lag1 = immediate/consummatory, controlled for time course + lags 2-5)\n")
    for n, d in coefs.items():
        line = f"  {n:11s}:"
        for c in LAGS:
            v = d[c].dropna()
            line += f"  {c[-4:]}={np.exp(v.median()):.2f}({(v > 0).sum()}/{len(v)})"
        print(line)
    m = coefs["SWR"].merge(coefs["NL NO SWR"], on=SC, suffixes=("_swr", "_nl"))
    diff = m["reward_lag1_swr"] - m["reward_lag1_nl"]
    pa = m.assign(d=diff).groupby("animal")["d"].median()
    print(f"\nlag1 dissociation (SWR vs NL-NO-SWR, time+history controlled): "
          f"{(diff > 0).mean():.0%} of {len(m)}, per-animal {(pa > 0).sum()}/{m.animal.nunique()}, "
          f"p={wilcoxon(m.reward_lag1_swr, m.reward_lag1_nl).pvalue:.1e}")
    print(f"\nNL-NO-SWR lag1 RR per-session median: {np.exp(coefs['NL NO SWR'].reward_lag1.median()):.3f} "
          f"(resolves the pooled-vs-per-session question)")

    # ── figure: the model's reward effects ──
    cols = {"SWR": "k", "NL w/ SWR": "C3", "NL NO SWR": "C0"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax = axes[0]
    xl = np.arange(1, K + 1)
    for n, d in coefs.items():
        med = np.array([np.exp(d[c].median()) for c in LAGS])
        lo = np.array([np.exp(d[c].quantile(0.25)) for c in LAGS])
        hi = np.array([np.exp(d[c].quantile(0.75)) for c in LAGS])
        ax.errorbar(xl + 0.05 * (list(coefs).index(n) - 1), med, yerr=[med - lo, hi - med],
                    marker="o", color=cols[n], label=n, capsize=3, lw=2)
    ax.axhline(1, color="0.5", ls=":")
    ax.set_xlabel("reward lag (trials back)")
    ax.set_ylabel("rate ratio (per-session median, IQR)")
    ax.set_title("Reward kernel (time-resolved)\nlag1 dissociates · lags 2-5 shared (RPE baseline)")
    ax.set_xticks(xl)
    ax.legend(frameon=False)

    ax = axes[1]
    data = [np.exp(coefs[n]["reward_lag1"].dropna()) for n in coefs]
    bp = ax.boxplot(data, labels=["SWR", "NL\nw/SWR", "NL\nNO SWR"], showfliers=False, patch_artist=True)
    for patch, n in zip(bp["boxes"], coefs, strict=False):
        patch.set_facecolor(cols[n])
        patch.set_alpha(0.4)
    ax.axhline(1, color="0.5", ls=":")
    ax.set_ylabel("immediate reward effect — lag1 RR (per session)")
    ax.set_title("Immediate/consummatory reward effect by population\n(1.0 = none; controlled for time course + lags 2-5)")
    ax.set_ylim(0, 5)
    fig.tight_layout()
    out = P.parent / "paper" / "figures" / "reward_kernel_timeresolved.png"
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
