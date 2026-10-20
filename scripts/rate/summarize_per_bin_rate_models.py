#!/usr/bin/env python
"""Summarize the per-bin conditional-intensity ladder (Family B) effects.

Reads ``models_per_bin_rate/all_models.pkl`` and reports, per event-type
population (detected SWR, all non-local, non-local-with-SWR,
non-local-without-SWR), the fitted effects across the model ladder with
across-session significance and the cross-population comparisons:

1. Baseline rate (M0_const): events / s of immobility, and which type is higher.
2. Reward lag-1 (M1 raw, M7 controlled): does each type respond, and how the
   effect compares across types.
3. Reward kernel lags 1-10 (M7r): boost-then-suppression shape, per type.
4. Self-history kernel lags 1-10 (M3r_histcos raw, M7r controlled): refractory
   window and burstiness peak, compared across types. Both are reconstructed
   from the four raised-cosine coefficients, not read per lag.
5. Reward-history basis components 0-2 (M7r), one section per component.
6. Ladder fit summary (AIC / CV bits/event / KS-in-band) per model x type.

All "RR" are rate ratios = exp(coef); summaries are across-session medians with
per-animal sign consistency (n/5) and a signed-rank p where applicable.

Usage:
    uv run python scripts/rate/summarize_per_bin_rate_models.py
    uv run python scripts/rate/summarize_per_bin_rate_models.py --allow-stale
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from continuum_swr_replay.analysis.replay.fit_provenance import require_fresh_inputs
from continuum_swr_replay.analysis.replay.model_specs import (
    BIN_S,
    MODEL_PER_BIN_RATE,
    N_HISTORY_LAGS,
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_N_LAGS,
)
from continuum_swr_replay.analysis.replay.per_bin_design import (
    reward_kernel_rate_ratios,
    self_history_rate_ratios,
)
from continuum_swr_replay.paths import REPLAY_CONTENT_DIR, model_dir, table_path
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

RC = REPLAY_CONTENT_DIR
ALL_MODELS = model_dir("per_bin_rate", RC) / "all_models.pkl"
# The tables fit_per_bin_rate_models.py fit all_models.pkl on. Checked against
# the artifact's provenance sidecar before anything is summarized, so a table
# regenerated after the fit cannot be reported through the old coefficients.
ALL_MODELS_INPUTS = {
    "stop_table": table_path("stop_table", RC, fmt="pkl"),
    "event_onset_table": table_path("event_onset_table", RC, fmt="pkl"),
    "event_table": table_path("event_table", RC, fmt="pkl"),
}

SC = list(SESSION_COLUMNS)
OUT = ["swr", "non_local", "non_local_swr", "non_local_no_swr"]
LABEL = {
    "swr": "detected SWR",
    "non_local": "non-local (all)",
    "non_local_swr": "NL w/ SWR",
    "non_local_no_swr": "NL no-SWR",
}
FULL = "M7r_time_histcos_rew"
IMMEDIATE_COEF = f"coef_{REWARD_HISTORY_COLUMNS[0]}"


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
    return (
        float((d > 0).mean()),
        int((pa > 0).sum()),
        int(pa.size),
        wilcoxon(m[f"{coef}_a"], m[f"{coef}_b"]).pvalue,
        len(m),
    )


def section_baseline(df):
    print("\n" + "=" * 78)
    print("1. BASELINE RATE  (M0_const) — events / s of immobility")
    print("=" * 78)
    for oc in OUT:
        s = df[(df.model == "M0_const") & (df.outcome == oc)]
        rate = np.exp(s["coef_Intercept"].dropna()) / 1.0  # exp(intercept)=events/s
        print(
            f"  {LABEL[oc]:16s}: {rate.median():.3f} events/s "
            f"(IQR {rate.quantile(0.25):.3f}-{rate.quantile(0.75):.3f}, n={len(rate)})"
        )
    print("  (exp(intercept); offset is log(exposure_seconds))")
    print("  pairwise (median rate ratio between types, paired p):")
    for a, b in [
        ("non_local_swr", "non_local_no_swr"),
        ("swr", "non_local"),
        ("non_local_swr", "swr"),
        ("non_local_no_swr", "swr"),
    ]:
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
    print(
        f"  {'population':16s} {'RR':>6} {'%pos':>6} {'animals':>8} {'p(vs0)':>9} {'n':>5}"
    )
    for oc in OUT:
        r = _one_sample(df, model, coef, oc)
        if r:
            rr, fp, n5, na, p, n = r
            print(f"  {LABEL[oc]:16s} {rr:6.2f} {fp:6.0%} {n5:5d}/{na} {p:9.1e} {n:5d}")


def section_kernel(title, model):
    print("\n" + "=" * 78)
    print(f"{title}   ({model}: reconstructed RR per lag 1..{N_HISTORY_LAGS})")
    print("=" * 78)
    for oc in OUT:
        s = df[(df.model == model) & (df.outcome == oc)]
        print(
            f"  {LABEL[oc]:16s} "
            + " ".join(f"{v:5.2f}" for v in self_history_rate_ratios(s))
        )


def section_reward_kernel(title, model):
    """Reward kernel reconstructed from the raised-cosine basis coefficients.

    The features are bump-weighted reward fractions, so an outcome at lag ``j``
    contributes ``sum_k coef_k * B[k, j] / sum_j' B[k, j']`` to the log rate.
    """
    print("\n" + "=" * 78)
    print(f"{title}   ({model}: reconstructed RR per lag 1..{REWARD_HISTORY_N_LAGS})")
    print("=" * 78)
    for oc in OUT:
        s = df[(df.model == model) & (df.outcome == oc)]
        rr = reward_kernel_rate_ratios(s)
        print(f"  {LABEL[oc]:16s} " + " ".join(f"{v:5.2f}" for v in rr))


def section_selfhistory(model):
    print("\n" + "=" * 78)
    print(f"   self-history kernel detail   ({model}: reconstructed, 15 ms lags)")
    print("=" * 78)
    print(
        f"  {'population':16s} {'refractory(<1 lags)':>20} {'burst peak RR':>14} {'@lag(ms)':>9}"
    )
    for oc in OUT:
        s = df[(df.model == model) & (df.outcome == oc)]
        rr = self_history_rate_ratios(s)
        refr = [j for j in range(1, 11) if rr[j - 1] < 1.0]
        refr_lags = f"{len(refr)} (≤{max(refr) * 15 if refr else 0}ms)"
        pk = int(np.nanargmax(rr)) + 1
        print(f"  {LABEL[oc]:16s} {refr_lags:>20} {rr[pk - 1]:14.2f} {pk * 15:9d}")


def section_ladder():
    print("\n" + "=" * 78)
    print(
        "6. LADDER FIT (median AIC, CV bits/event, KS-in-band frac), outcome=non_local"
    )
    print("=" * 78)
    # Stop-clock rungs only, taken from the executable spec so a new rung cannot
    # be silently missing here: the arrival rungs are fit on a subset of the
    # frozen cohort's rows, so their AIC is not on this ladder's scale.
    ladder = [n for n, s in MODEL_PER_BIN_RATE.items() if s.time_clock == "stop"]
    g = df[df.outcome == "non_local"]
    print(f"  {'model':22s} {'AIC':>8} {'cv_bits':>9} {'ks_band':>8} {'n':>4}")
    for m in ladder:
        s = g[g.model == m]
        if len(s):
            print(
                f"  {m:22s} {s.aic.median():8.0f} {s.cv_bits_per_event.median():9.3f} "
                f"{s.ks_within_band.mean():8.2f} {len(s):4d}"
            )


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser (also the provenance gate's opt-out)."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help=(
            "Summarize the fit even if its input tables have changed since it "
            "was fit (or carry no provenance sidecar at all)."
        ),
    )
    return parser


def main():
    args = build_parser().parse_args()
    require_fresh_inputs(ALL_MODELS, ALL_MODELS_INPUTS, allow_stale=args.allow_stale)
    global df
    df = pd.read_pickle(ALL_MODELS)
    print(
        f"all_models.pkl: {len(df)} rows, {df.groupby(SC).ngroups} sessions, "
        f"{df.model.nunique()} models, BIN_S={BIN_S}"
    )

    section_baseline(df)

    section_coef(
        "2. IMMEDIATE REWARD COMPONENT (raw, uncontrolled)", "M1_rew", IMMEDIATE_COEF
    )
    section_coef(
        "2. IMMEDIATE REWARD COMPONENT (controlled: time+history)",
        FULL,
        IMMEDIATE_COEF,
    )
    print(
        f"\n  paired NL-w/SWR vs NL-no-SWR (M7r {IMMEDIATE_COEF}):  "
        + str(_paired(df, FULL, IMMEDIATE_COEF, "non_local_swr", "non_local_no_swr"))
    )

    section_reward_kernel(
        "3. REWARD KERNEL (M7r, controlled) — reconstructed from the basis",
        FULL,
    )

    section_kernel("4. SELF-HISTORY KERNEL (M3r_histcos, raw)", "M3r_histcos")
    section_kernel("4. SELF-HISTORY KERNEL (M7r, controlled)", FULL)
    section_selfhistory(FULL)

    for k, column in enumerate(REWARD_HISTORY_COLUMNS):
        section_coef(
            f"5. REWARD-HISTORY BASIS COMPONENT {k}",
            FULL,
            f"coef_{column}",
        )
    section_ladder()


if __name__ == "__main__":
    main()
