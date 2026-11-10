#!/usr/bin/env python
"""SCRATCH: finer peri-switch replay-rate profile (rate vs trials-from-switch).

Extends ``_periswitch_rate.py`` from a pre/post binary to a temporal profile.
Stops are immobility bouts (many per trial), so per session we reconstruct a
trial index (maximal runs of constant ``occupancy_well_arm`` = well visits), mark
switch trials (``current_patch`` changes between consecutive trials), compute
each trial's SIGNED distance to the nearest switch trial (− = before, + =
after), map it back to the stops, and fit a pooled Poisson GEE of the per-stop
event rate across distance bins −3…+3 (baseline = ``far``, |d|≥4), adjusted
for ``is_omitted_outcome`` and clustered on session.

Output: scratch/periswitch_profile.png + printed rate ratios per distance bin.
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
MAXD = 3  # bins -MAXD..+MAXD, |d|>MAXD -> "far" baseline


def _signed_dist_to_switch(is_switch: np.ndarray) -> np.ndarray:
    """For each trial, signed distance to the nearest switch trial."""
    sw = np.flatnonzero(is_switch)
    n = len(is_switch)
    if len(sw) == 0:
        return np.full(n, np.iinfo(np.int64).max)
    idx = np.arange(n)
    pos = np.searchsorted(sw, idx)
    out = np.empty(n, dtype=np.int64)
    for i in idx:
        cands = []
        if pos[i] < len(sw):
            cands.append(sw[pos[i]])
        if pos[i] > 0:
            cands.append(sw[pos[i] - 1])
        nearest = min(cands, key=lambda s: abs(i - s))
        out[i] = i - nearest
    return out


def prep(stop: pd.DataFrame) -> pd.DataFrame:
    df = stop.copy()
    df = df[
        (df["immobility_duration_s"] > 0)
        & df["current_patch"].notna()
        & df["occupancy_well_arm"].notna()
    ].copy()
    df["session"] = (
        df["animal"].astype(str)
        + df["session_date"].astype(str)
        + "_"
        + df["epoch"].astype(str)
    )
    df["after_omission"] = (
        (1.0 - df["is_rewarded_outcome"].astype(float)).fillna(0).astype(int)
    )
    df["log_off"] = np.log(df["immobility_duration_s"].to_numpy(dtype=float))
    # Global order by (session, time) so positional indices line up per group.
    df = df.sort_values(["session", "stop_start"]).reset_index(drop=True)

    dist = np.empty(len(df), dtype=np.int64)
    for _, g in df.groupby("session", sort=False):
        arm = g["occupancy_well_arm"].to_numpy()
        patch = g["current_patch"].to_numpy()
        new_trial = np.r_[True, arm[1:] != arm[:-1]]
        trial = np.cumsum(new_trial) - 1  # per-stop trial index
        per_patch = pd.Series(patch).groupby(trial).first().to_numpy()  # per trial
        is_switch = np.r_[False, per_patch[1:] != per_patch[:-1]]
        td = _signed_dist_to_switch(is_switch)  # per trial
        dist[g.index.to_numpy()] = td[trial]  # per stop
    df["dist"] = dist
    df["bin"] = df["dist"].apply(
        lambda d: "far" if abs(d) > MAXD else f"{d:+d}".replace("+0", "0")
    )
    return df


def main():
    stop = read_table(table_path("stop_table", RC))
    require_covariate_schema(stop, table="stop")
    df = prep(stop)
    logging.info("%d stops, %d sessions", len(df), df["session"].nunique())
    print("stops per distance bin:")
    print(df["bin"].value_counts().to_string())

    res = GEE.from_formula(
        "n_events ~ C(bin, Treatment('far')) + after_omission",
        groups="session",
        data=df,
        family=sm.families.Poisson(),
        offset=df["log_off"].to_numpy(),
        cov_struct=Independence(),
    ).fit()

    order = [f"{d:+d}".replace("+0", "0") for d in range(-MAXD, MAXD + 1)]
    rr = {}
    for b in order:
        term = f"C(bin, Treatment('far'))[T.{b}]"
        if term in res.params.index:
            lo, hi = res.conf_int().loc[term]
            rr[b] = (
                np.exp(res.params[term]),
                np.exp(lo),
                np.exp(hi),
                res.pvalues[term],
            )
        else:
            rr[b] = (1.0, np.nan, np.nan, np.nan)  # the baseline itself
    print("\nrate ratio vs 'far' baseline (adjusted for after_omission):")
    for b in order:
        r, lo, hi, p = rr[b]
        ci = f"[{lo:.2f}, {hi:.2f}]" if np.isfinite(lo) else "(baseline)"
        print(
            f"  trials_from_switch {b:>3s}: ×{r:.3f} {ci}  p={p:.1e}"
            if np.isfinite(p)
            else f"  trials_from_switch {b:>3s}: ×{r:.3f} {ci}"
        )
    print(f"  after_omission: ×{np.exp(res.params['after_omission']):.3f}")

    # Plot the profile.
    xs = list(range(-MAXD, MAXD + 1))
    ys = [rr[b][0] for b in order]
    los = [rr[b][1] for b in order]
    his = [rr[b][2] for b in order]
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    ax.axhline(1.0, color="0.6", lw=0.8, ls="--", label="far baseline")
    ax.axvline(0, color="crimson", lw=1.0, alpha=0.6, label="switch")
    ax.errorbar(
        xs,
        ys,
        yerr=[
            np.array(ys) - np.nan_to_num(los, nan=ys),
            np.nan_to_num(his, nan=ys) - np.array(ys),
        ],
        fmt="o-",
        color="C0",
        capsize=3,
        lw=2,
    )
    ax.set_xticks(xs)
    ax.set_xlabel("trials from patch switch  (− before · + after)")
    ax.set_ylabel("event-rate ratio vs far  (exp β)")
    ax.set_title("Peri-switch replay-rate profile (pooled GEE, adj. for omission)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    SCRATCH.mkdir(exist_ok=True)
    out = SCRATCH / "periswitch_profile.png"
    save_figure(
        fig, out.stem, base_dir=out.parent, dpi=140, formats=["png"], transparent=False
    )
    logging.info("Saved %s", out)


if __name__ == "__main__":
    main()
