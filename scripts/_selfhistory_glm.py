#!/usr/bin/env python
"""Self-history + reward-kernel time-resolved GLM (Part B), 15 ms bins.

Per population, per session: Poisson on onset counts in 15 ms within-stop bins,
with a quiescence time-course spline, the across-trial reward kernel, and the
same-event self-history as evenly-spaced 15 ms lagged onset counts (lag1 = 15 ms
back = refractoriness; later lags = burstiness/clustering decay). offset =
log(exposure). Self-history resets at stop boundaries (never crosses movement).

  onset ~ bs(t_since_stop, df=4) + reward_lag1..5 + onset_lag1..L
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
BINW, TMAX, K, L = 0.015, 4.0, 5, 10
RLAGS = [f"reward_lag{j}" for j in range(1, K + 1)]
HLAGS = [f"onset_lag{j}" for j in range(1, L + 1)]


def build_bins(stops: pd.DataFrame) -> pd.DataFrame:
    """One row per (stop, 15 ms bin) up to TMAX, with exposure + reward lags."""
    s = stops[stops["immobility_time"] > 0].copy()
    s["nbin"] = np.minimum(
        np.ceil(s["immobility_time"] / BINW), int(TMAX / BINW)
    ).astype(int)
    idx = np.repeat(np.arange(len(s)), s["nbin"].to_numpy())
    binj = np.concatenate([np.arange(n) for n in s["nbin"].to_numpy()])
    out = s.iloc[idx][[*SC, "stop_id", "immobility_time", *RLAGS]].reset_index(drop=True)
    out["binj"] = binj
    out["tc"] = (binj + 0.5) * BINW
    out["exposure"] = np.minimum(BINW, out["immobility_time"] - binj * BINW)
    return out[out["exposure"] > 1e-4]


def main() -> None:
    st = pd.read_pickle(RC / "stop_table.pkl")
    base = build_bins(st)

    onset = pd.read_pickle(RC / "event_onset_table.pkl")
    ev = pd.read_pickle(RC / "event_table.pkl")[["stop_id", "event_start", "is_swr"]].copy()
    ev["rank"] = ev.sort_values("event_start").groupby("stop_id").cumcount()
    nl = onset[onset.event_type == "non_local"].copy()
    nl["rank"] = nl.sort_values("onset_time").groupby("stop_id").cumcount()
    nl = nl.merge(ev[["stop_id", "rank", "is_swr"]], on=["stop_id", "rank"], how="left")
    pops = {
        "SWR": onset[onset.event_type == "SWR"],
        "non-local (all)": onset[onset.event_type == "non_local"],
        "NL NO SWR": nl[nl.is_swr.eq(False)],
        "NL w/ SWR": nl[nl.is_swr.eq(True)],
    }

    formula = "count ~ bs(tc, df=4) + " + " + ".join(RLAGS + HLAGS)
    coefs = {}
    for name, src in pops.items():
        src = src[src["t_since_stop_onset"] < TMAX].copy()
        src["binj"] = (src["t_since_stop_onset"] / BINW).astype(int)
        cnt = src.groupby(["stop_id", "binj"]).size().rename("count")
        df = base.merge(cnt, on=["stop_id", "binj"], how="left").fillna({"count": 0})
        df = df.sort_values(["stop_id", "binj"])
        for j in range(1, L + 1):
            df[f"onset_lag{j}"] = df.groupby("stop_id")["count"].shift(j).fillna(0)
        rows = []
        for keys, g in df.groupby(SC, sort=False):
            g = g.dropna(subset=RLAGS)
            if len(g) < 200 or g["count"].sum() < 20 or g["reward_lag1"].nunique() < 2:
                continue
            try:
                res = smf.glm(formula, data=g, family=sm.families.Poisson(),
                              offset=np.log(g["exposure"].to_numpy())).fit()
                rows.append({**dict(zip(SC, keys, strict=True)),
                             **{c: res.params.get(c, np.nan) for c in RLAGS + HLAGS}})
            except Exception:
                continue
        coefs[name] = pd.DataFrame(rows)
        print(f"fit {name}: {len(coefs[name])} sessions")

    print("\n=== self-history kernel (RR per 15ms lag; <1 refractory, >1 bursty) ===")
    for n, d in coefs.items():
        print(f"  {n:11s}: " + " ".join(
            f"{int(j * 15)}ms={np.exp(d[f'onset_lag{j}'].median()):.2f}" for j in range(1, L + 1)))
    print("\n=== reward lag1 RR (now controlled for clustering) ===")
    for n, d in coefs.items():
        v = d["reward_lag1"].dropna()
        print(f"  {n:11s}: RR={np.exp(v.median()):.2f} ({(v > 0).sum()}/{len(v)})")
    m = coefs["SWR"].merge(coefs["NL NO SWR"], on=SC, suffixes=("_s", "_n"))
    print(f"\nlag1 reward dissociation (clustering-controlled): "
          f"{(m.reward_lag1_s > m.reward_lag1_n).mean():.0%} of {len(m)}, "
          f"p={wilcoxon(m.reward_lag1_s, m.reward_lag1_n).pvalue:.1e}")

    # figure: self-history kernels
    fig, ax = plt.subplots(figsize=(8, 5))
    lagms = np.arange(1, L + 1) * 15
    for n, col in [("SWR", "k"), ("non-local (all)", "C2"), ("NL w/ SWR", "C3"), ("NL NO SWR", "C0")]:
        d = coefs[n]
        med = np.array([np.exp(d[f"onset_lag{j}"].median()) for j in range(1, L + 1)])
        lo = np.array([np.exp(d[f"onset_lag{j}"].quantile(0.25)) for j in range(1, L + 1)])
        hi = np.array([np.exp(d[f"onset_lag{j}"].quantile(0.75)) for j in range(1, L + 1)])
        ax.errorbar(lagms, med, yerr=[med - lo, hi - med], marker="o", color=col, label=n, capsize=3)
    ax.axhline(1, color="0.5", ls=":")
    ax.set_xlabel("time lag (ms)")
    ax.set_ylabel("self-history rate ratio (per-session median, IQR)")
    ax.set_title("Within-stop event autocorrelation\n(<1 = refractory, >1 = bursty)")
    ax.legend(frameon=False)
    fig.tight_layout()
    out = P.parent / "paper" / "figures" / "selfhistory_autocorr.png"
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
