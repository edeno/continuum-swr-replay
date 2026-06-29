#!/usr/bin/env python
"""Summarize the per-bin conditional-intensity ladder (Family B) effects.

Reads ``models_per_bin_rate/all_models.pkl`` and reports, per event-type
population (detected SWR, all non-local, non-local-with-SWR,
non-local-without-SWR), the fitted effects across the model ladder with
across-session significance and the cross-population comparisons:

1. Baseline rate (M0_const): events / s of immobility, and which type is higher.
2. Reward lag-1 (M1 raw, M7 controlled): does each type respond, and how the
   effect compares across types.
3. Reward kernel lags 1-10 (M7): boost-then-suppression shape, per type.
4. Self-history kernel lags 1-10 (M3_hist raw, M7 controlled): refractory window
   and burstiness peak, compared across types.
5. Other covariates (M2r recent reward rate, M8 switch).
6. Ladder fit summary (AIC / CV bits/event / KS-in-band) per model x type.

All "RR" are rate ratios = exp(coef); summaries are across-session medians with
per-animal sign consistency (n/5) and a signed-rank p where applicable.

Usage:
    uv run python scripts/summarize_per_bin_rate_models.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from continuum_swr_replay.analysis.replay.model_specs import BIN_S
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR as P

SC = ["animal", "day", "epoch"]
OUT = ["swr", "non_local", "non_local_swr", "non_local_no_swr"]
LABEL = {"swr": "detected SWR", "non_local": "non-local (all)",
         "non_local_swr": "NL w/ SWR", "non_local_no_swr": "NL no-SWR"}


def _one_sample(df: pd.DataFrame, model: str, coef: str, oc: str):
    """median RR, %pos, per-animal n/5 (sign of median), signed-rank p vs 0."""
    s = df[(df.model == model) & (df.outcome == oc)]
    v = s[coef].dropna()
    if len(v) == 0:
        return None
    med = v.median()
    pa = s.assign(c=s[coef]).groupby("animal")["c"].median().dropna()
    n5 = int((np.sign(pa) == np.sign(med)).sum())
    p = wilcoxon(v).pvalue if len(v) > 10 and v.abs().sum() > 0 else np.nan
    return np.exp(med), float((v > 0).mean()), n5, int(pa.size), p, len(v)


def _paired(df: pd.DataFrame, model: str, coef: str, a: str, b: str):
    """paired a-vs-b on coef: %(a>b), per-animal n/5, signed-rank p, n."""
    A = df[(df.model == model) & (df.outcome == a)][SC + [coef]]
    B = df[(df.model == model) & (df.outcome == b)][SC + [coef]]
    m = A.merge(B, on=SC, suffixes=("_a", "_b")).dropna()
    if len(m) < 10:
        return None
    d = m[f"{coef}_a"] - m[f"{coef}_b"]
    pa = m.assign(x=d).groupby("animal")["x"].median().dropna()
    return float((d > 0).mean()), int((pa > 0).sum()), int(pa.size), \
        wilcoxon(m[f"{coef}_a"], m[f"{coef}_b"]).pvalue, len(m)


def section_baseline(df):
    print("\n" + "=" * 78)
    print("1. BASELINE RATE  (M0_const) — events / s of immobility")
    print("=" * 78)
    for oc in OUT:
        s = df[(df.model == "M0_const") & (df.outcome == oc)]
        rate = np.exp(s["coef_Intercept"].dropna()) / 1.0  # exp(intercept)=events/s
        print(f"  {LABEL[oc]:16s}: {rate.median():.3f} events/s "
              f"(IQR {rate.quantile(.25):.3f}-{rate.quantile(.75):.3f}, n={len(rate)})")
    print("  (exp(intercept); offset is log(exposure_seconds))")
    print("  pairwise (median rate ratio between types, paired p):")
    for a, b in [("non_local_swr", "non_local_no_swr"), ("swr", "non_local"),
                 ("non_local_swr", "swr"), ("non_local_no_swr", "swr")]:
        A = df[(df.model == "M0_const") & (df.outcome == a)][SC + ["coef_Intercept"]]
        B = df[(df.model == "M0_const") & (df.outcome == b)][SC + ["coef_Intercept"]]
        m = A.merge(B, on=SC, suffixes=("_a", "_b")).dropna()
        ratio = np.exp((m["coef_Intercept_a"] - m["coef_Intercept_b"]).median())
        p = wilcoxon(m["coef_Intercept_a"], m["coef_Intercept_b"]).pvalue
        print(f"    {LABEL[a]:16s} / {LABEL[b]:16s} = {ratio:.2f}x   p={p:.1e}")


def section_coef(title, model, coef, note=""):
    print("\n" + "=" * 78)
    print(f"{title}   ({model}: {coef})")
    if note:
        print(note)
    print("=" * 78)
    print(f"  {'population':16s} {'RR':>6} {'%pos':>6} {'animals':>8} {'p(vs0)':>9} {'n':>5}")
    for oc in OUT:
        r = _one_sample(df, model, coef, oc)
        if r:
            rr, fp, n5, na, p, n = r
            print(f"  {LABEL[oc]:16s} {rr:6.2f} {fp:6.0%} {n5:5d}/{na} {p:9.1e} {n:5d}")


def section_kernel(title, model, prefix, n):
    print("\n" + "=" * 78)
    print(f"{title}   ({model}: {prefix}1..{n}, RR per lag)")
    print("=" * 78)
    lags = range(1, n + 1)
    for oc in OUT:
        s = df[(df.model == model) & (df.outcome == oc)]
        rr = [np.exp(s[f"coef_{prefix}{j}"].median()) for j in lags]
        print(f"  {LABEL[oc]:16s} " + " ".join(f"{v:5.2f}" for v in rr))
    return


def section_selfhistory(model):
    print("\n" + "=" * 78)
    print(f"   self-history kernel detail   ({model}: onset_lag, 15 ms lags)")
    print("=" * 78)
    print(f"  {'population':16s} {'refractory(<1 lags)':>20} {'burst peak RR':>14} {'@lag(ms)':>9}")
    for oc in OUT:
        s = df[(df.model == model) & (df.outcome == oc)]
        rr = np.array([np.exp(s[f"coef_onset_lag{j}"].median()) for j in range(1, 11)])
        refr = [j for j in range(1, 11) if rr[j - 1] < 1.0]
        refr_lags = f"{len(refr)} (≤{max(refr)*15 if refr else 0}ms)"
        pk = int(np.nanargmax(rr)) + 1
        print(f"  {LABEL[oc]:16s} {refr_lags:>20} {rr[pk-1]:14.2f} {pk*15:9d}")


def section_ladder():
    print("\n" + "=" * 78)
    print("6. LADDER FIT (median AIC, CV bits/event, KS-in-band frac), outcome=non_local")
    print("=" * 78)
    ladder = ["M0_const", "M1_rew1", "M2_rewhist", "M2r_rewrate", "M3_hist",
              "M3r_histcos", "M4_hist_rew", "M5_time", "M6_time_hist",
              "M7_time_hist_rew", "M7r_time_histcos_rew", "M8_full_switch",
              "M9_rew_x_time"]
    g = df[df.outcome == "non_local"]
    print(f"  {'model':22s} {'AIC':>8} {'cv_bits':>9} {'ks_band':>8} {'n':>4}")
    for m in ladder:
        s = g[g.model == m]
        if len(s):
            print(f"  {m:22s} {s.aic.median():8.0f} {s.cv_bits_per_event.median():9.3f} "
                  f"{s.ks_within_band.mean():8.2f} {len(s):4d}")


def main():
    global df
    df = pd.read_pickle(P / "replay_content" / "models_per_bin_rate" / "all_models.pkl")
    print(f"all_models.pkl: {len(df)} rows, {df.groupby(SC).ngroups} sessions, "
          f"{df.model.nunique()} models, BIN_S={BIN_S}")

    section_baseline(df)

    section_coef("2. REWARD LAG-1 (raw, uncontrolled)", "M1_rew1", "coef_reward_lag1")
    section_coef("2. REWARD LAG-1 (controlled: time+history+reward-kernel)",
                 "M7_time_hist_rew", "coef_reward_lag1")
    print("\n  paired NL-w/SWR vs NL-no-SWR (M7 reward_lag1):  "
          + str(_paired(df, "M7_time_hist_rew", "coef_reward_lag1",
                        "non_local_swr", "non_local_no_swr")))

    section_kernel("3. REWARD KERNEL (M7, controlled) — boost(lag1) then suppression",
                   "M7_time_hist_rew", "reward_lag", 10)

    section_kernel("4. SELF-HISTORY KERNEL (M3_hist, raw)", "M3_hist", "onset_lag", 10)
    section_kernel("4. SELF-HISTORY KERNEL (M7, controlled)", "M7_time_hist_rew",
                   "onset_lag", 10)
    section_selfhistory("M7_time_hist_rew")

    section_coef("5. RECENT REWARD RATE", "M2r_rewrate",
                 "coef_reward_rate_trial_k10")
    section_coef("5. PATCH SWITCH", "M8_full_switch", "coef_is_after_switch")

    section_ladder()


if __name__ == "__main__":
    main()
