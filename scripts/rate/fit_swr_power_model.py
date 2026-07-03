#!/usr/bin/env python
"""Per-event SWR-coupling strength (ripple power) vs behavioral covariates.

For each non-local event, the SWR-coupling strength is summarized by
``ripple_power_z`` (the event-mean of the z-scored consensus ripple power -- the
same signal the Kay detector thresholds, so ``is_swr`` is a thresholded version of
this continuous quantity). This script asks, across non-local events:

  1. Distribution shape of the SWR power -- bimodal (categorical SWR vs non-SWR)
     or unimodal (the SWR/non-SWR split is a cut on a continuum)?
  2. Duration confound -- is the (event-mean) power correlated with event duration?
  3. Power model -- does reward/value/exploration predict the SWR power of an
     event (power ~ covariates), i.e. acting on SWR-COUPLING strength rather than
     on whether an event occurs (occurrence is the separate per-bin rate model)?

Scope / caveat: this fits only the *conditional power-vs-covariates* relationship
by per-event OLS pooled across sessions. It is NOT a full marked point process --
it does not jointly model the event intensity, does not test whether the power
depends on event history (the conditional-independence assumption), and has no
shared-latent coupling between occurrence and power. Those are deferred to a joint
treatment. ``ripple_power_z`` here is the event-mean; a peri-onset peak (the
detector's actual criterion) is the better summary and is left as a follow-up.

Output: ``models_per_bin_rate/swr_power_model.pkl`` (per-session coefficients).

Usage: python scripts/rate/fit_swr_power_model.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skew, wilcoxon

from continuum_swr_replay.analysis.replay._helpers import add_uncertainty_covariates
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

RC = PROCESSED_DATA_DIR / "replay_content"
SC = ["animal", "day", "epoch"]
POWER_COL = "ripple_power_z"  # event-mean z-scored consensus ripple power
PREDICTORS = ["reward_lag1", "p_reward_current", "rpe",
              "recent_switch_rate", "value_diff_patch", "q_current",
              "time_since_stop"]


def _bimodality_coef(x: np.ndarray) -> float:
    """Sarle's bimodality coefficient; > 0.555 suggests bimodal (uniform ref)."""
    n = len(x)
    g = skew(x)
    k = (np.mean((x - x.mean()) ** 4) / x.var() ** 2) - 3.0  # excess kurtosis
    denom = k + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return (g**2 + 1) / denom


def _assemble() -> pd.DataFrame:
    ev = pd.read_pickle(RC / "event_table.pkl")
    stop = add_uncertainty_covariates(pd.read_pickle(RC / "stop_table.pkl"))
    # qvals_table (exact poke_out alignment). rpe/q_current are now the
    # beta-model experienced value (rpe = R - q_chosen, q_current = q_chosen),
    # replacing the retired difflib nominal-rpe / leaky-Q.
    cont = pd.read_pickle(RC / "models_per_bin_rate" / "qvals_table.pkl")
    cont = cont.rename(columns={"q_chosen": "q_current"})
    df = ev.merge(stop[["stop_id", "reward_lag1", "recent_switch_rate"]],
                  on="stop_id", how="left")
    df = df.merge(
        cont[["stop_id", "p_reward_current", "rpe", "value_diff_patch", "q_current"]],
        on="stop_id", how="left")
    return df


def fit_power_model(df: pd.DataFrame) -> pd.DataFrame:
    """Per-session OLS: ripple_power_z ~ standardized covariates. One row/session."""
    rows = []
    for keys, g in df.groupby(SC, sort=False):
        g = g[[*PREDICTORS, POWER_COL]].dropna()
        if len(g) < 30:
            continue
        X = g[PREDICTORS].to_numpy(dtype=float)
        X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
        y = g[POWER_COL].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(g)), X], y, rcond=None)
        rows.append({**dict(zip(SC, keys, strict=True)), "n_events": len(g),
                     **dict(zip(PREDICTORS, beta[1:], strict=True))})
    return pd.DataFrame(rows)


def main() -> None:
    df = _assemble()
    m = df[POWER_COL].to_numpy()
    print(f"=== SWR power distribution ({POWER_COL}, n={len(df)} non-local events) ===")
    print("  pctiles 5/25/50/75/95:", np.round(np.percentile(m, [5, 25, 50, 75, 95]), 2))
    print(f"  Sarle bimodality coef = {_bimodality_coef(m):.3f}  (>0.555 => bimodal)")
    for lab, sub in [("non-SWR events", df[~df.is_swr]), ("SWR events", df[df.is_swr])]:
        s = sub[POWER_COL]
        print(f"  {lab:16s} n={len(s):6d}  mean={s.mean():+.2f}  sd={s.std():.2f}")
    lo, hi = np.percentile(m, [1, 99])
    h, edges = np.histogram(m, bins=np.linspace(lo, hi, 21))
    print("  histogram (1-99 pctile):")
    for c, cnt in zip((edges[:-1] + edges[1:]) / 2, h, strict=True):
        print(f"    {c:+5.1f} | {'#' * int(40 * cnt / h.max())}")

    d = df[["event_duration", POWER_COL]].dropna()
    print(f"\n=== duration confound: corr(power, event_duration) ="
          f" {np.corrcoef(d.event_duration, d[POWER_COL])[0,1]:+.3f}"
          f"  (log-dur {np.corrcoef(np.log(d.event_duration), d[POWER_COL])[0,1]:+.3f}) ===")

    coefs = fit_power_model(df)
    out = RC / "models_per_bin_rate" / "swr_power_model.pkl"
    coefs.to_pickle(out)
    print(f"\n=== power model: {POWER_COL} ~ standardized covariates (per session) ===")
    print(f"  {len(coefs)} sessions, {coefs.animal.nunique()} animals; saved -> {out}\n")
    print(f"  {'covariate':<22}{'med coef':>10}{'%pos':>7}{'animals':>9}{'p':>10}")
    for c in PREDICTORS:
        s = coefs[c].dropna()
        n_an = (coefs.groupby("animal")[c].median() > 0).sum()
        print(f"  {c:<22}{s.median():>+10.3f}{(s > 0).mean() * 100:>6.0f}%"
              f"{f'{n_an}/5':>9}{wilcoxon(s)[1]:>10.1e}")


if __name__ == "__main__":
    main()
