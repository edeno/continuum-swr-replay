#!/usr/bin/env python
"""SCRATCH: is the replay-event rate elevated BEFORE or AFTER a patch switch,
controlling for the reward-outcome confound?

Switches follow omissions (lose-shift), so a naive peri-switch rate profile
just traces the reward profile. Here we:

  1. define per-stop patch context from current/prev/next_patch:
       - pre_switch  : next_patch != current_patch  (about to leave the patch)
       - post_switch : current_patch != prev_patch   (just arrived via a switch)
       - stay        : neither (baseline)
  2. fit a pooled Poisson GEE (cluster-robust on session, log-immobility
     offset) for the per-stop event rate, with is_after_omission included so
     the switch effect is net of reward outcome; report rate ratios exp(β).
  3. add the is_after_omission × {pre,post}_switch interactions.
  4. matched contrast: WITHIN after-omission stops, pre_switch vs stay — both
     follow an omission, differing only in the upcoming decision, so this
     isolates the switch decision from the omission effect.

Output: scratch/periswitch_rate.png + printed rate-ratio tables.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.cov_struct import Independence
from statsmodels.genmod.generalized_estimating_equations import GEE

from continuum_swr_replay.figure_utilities import save_figure
from continuum_swr_replay.paths import REPLAY_CONTENT_DIR, table_path
from continuum_swr_replay.tables.covariate_registry import (
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
warnings.filterwarnings("ignore")
RC = REPLAY_CONTENT_DIR
SCRATCH = Path("scratch")


def prep(stop: pd.DataFrame) -> pd.DataFrame:
    df = stop.copy()
    df = df[df["immobility_duration_s"] > 0].copy()
    for c in ("current_patch", "prev_patch", "next_patch"):
        df = df[df[c].notna()]
    df["pre_switch"] = (df["next_patch"] != df["current_patch"]).astype(int)
    df["post_switch"] = (df["current_patch"] != df["prev_patch"]).astype(int)
    df["after_omission"] = (
        (1.0 - df["is_rewarded_outcome"].astype(float)).fillna(0).astype(int)
    )
    df["session"] = (
        df["animal"].astype(str)
        + df["session_date"].astype(str)
        + "_"
        + df["epoch"].astype(str)
    )
    df["log_off"] = np.log(df["immobility_duration_s"].to_numpy(dtype=float))
    return df


def fit_gee(df, formula):
    res = GEE.from_formula(
        formula,
        groups="session",
        data=df,
        family=sm.families.Poisson(),
        offset=df["log_off"].to_numpy(),
        cov_struct=Independence(),
    ).fit()
    return res


def rr_table(res, terms):
    out = {}
    for t in terms:
        if t in res.params.index:
            b = res.params[t]
            lo, hi = res.conf_int().loc[t]
            out[t] = (np.exp(b), np.exp(lo), np.exp(hi), res.pvalues[t])
    return out


def main():
    stop = read_table(table_path("stop_table", RC))
    require_covariate_schema(stop, table="stop")
    df = prep(stop)
    n_sess = df["session"].nunique()
    logging.info(
        "%d stops, %d sessions | pre_switch=%d post_switch=%d after_omission=%d",
        len(df),
        n_sess,
        df.pre_switch.sum(),
        df.post_switch.sum(),
        df.after_omission.sum(),
    )

    # 1. Adjusted main-effects model.
    m1 = fit_gee(df, "n_events ~ pre_switch + post_switch + after_omission")
    rr1 = rr_table(m1, ["pre_switch", "post_switch", "after_omission"])

    # 2. Interaction model (does switch effect differ by reward outcome?).
    m2 = fit_gee(
        df,
        "n_events ~ pre_switch * after_omission + post_switch * after_omission",
    )
    rr2 = rr_table(
        m2,
        [
            "pre_switch",
            "post_switch",
            "after_omission",
            "pre_switch:after_omission",
            "post_switch:after_omission",
        ],
    )

    # 3. Matched contrast: within after-omission stops, pre_switch vs stay.
    om = df[df.after_omission == 1]
    m3 = fit_gee(om, "n_events ~ pre_switch + post_switch")
    rr3 = rr_table(m3, ["pre_switch", "post_switch"])

    def show(name, rr):
        print(f"\n{name}  (rate ratio exp(β), 95% CI, p):")
        for t, (r, lo, hi, p) in rr.items():
            print(f"  {t:34s} ×{r:.3f}  [{lo:.3f}, {hi:.3f}]  p={p:.1e}")

    show("Adjusted main effects (all stops)", rr1)
    show("Interaction model", rr2)
    show("Matched contrast (after-omission stops only)", rr3)

    # Plot: rate ratios for pre/post switch (adjusted) + matched contrast.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
    for ax, rr, title in (
        (axes[0], rr1, "adjusted for reward outcome (all stops)"),
        (axes[1], rr3, "after-omission stops only (matched)"),
    ):
        terms = [t for t in ("pre_switch", "post_switch", "after_omission") if t in rr]
        y = np.arange(len(terms))[::-1]
        ax.axvline(1.0, color="0.6", lw=0.8)
        ax.set_xscale("log")
        for yi, t in zip(y, terms, strict=True):
            r, lo, hi, p = rr[t]
            ax.plot([lo, hi], [yi, yi], color="black", lw=2)
            ax.plot(r, yi, "o", color="C0" if r >= 1 else "C1", ms=7)
            ax.annotate(
                f"×{r:.2f}{'*' if p < 0.05 else ''}",
                (r, yi + 0.15),
                fontsize=8,
                ha="center",
            )
        ax.set_yticks(y)
        ax.set_yticklabels(terms, fontsize=9)
        ax.set_xlabel("event-rate ratio exp(β)", fontsize=9)
        ax.set_title(title, fontsize=9)
    fig.suptitle("Replay event rate around patch switches (pooled GEE)", fontsize=11)
    fig.tight_layout()
    SCRATCH.mkdir(exist_ok=True)
    out = SCRATCH / "periswitch_rate.png"
    save_figure(
        fig, out.stem, base_dir=out.parent, dpi=140, formats=["png"], transparent=False
    )
    logging.info("Saved %s", out)


if __name__ == "__main__":
    main()
