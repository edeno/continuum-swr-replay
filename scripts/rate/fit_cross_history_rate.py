#!/usr/bin/env python
"""Bivariate per-bin conditional intensity: does one non-local stream's history
drive the other's rate?

Family B (per-bin point process) extended to a bivariate coupling. For each
session we fit the non-local-**no-SWR** onset rate per (stop x bin) with a
within-stop time spline + reward kernel + its own self-history, and ask whether
adding the **with-SWR** non-local stream's recent history (a cross-history block)
improves the fit -- i.e. is a non-SWR non-local event made more/less likely by a
recent with-SWR non-local event? The nested comparison (self vs self+cross)
isolates the cross contribution; the cross-history coefficients give its sign and
timescale. The reverse direction (with-SWR rate ~ ... + non-SWR cross history) is
fit too, so the coupling is reported both ways.

Per session, per direction:
  M_self      = bs(tc) + reward kernel + self-history (raised cosine)
  M_selfcross = M_self + cross-history (raised cosine)
We report the cross-history coefficients, the self->cross AIC drop, and the
cross cv-bits/event gain; then aggregate across sessions (median, %+, per-animal
sign consistency, FDR Wilcoxon).

Usage: python scripts/rate/fit_cross_history_rate.py [--max-per-animal N] [--write]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from continuum_swr_replay.analysis.replay import (  # noqa: E402
    BIN_S,
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    N_REWARD_LAGS,
    POPULATIONS,
    TMAX,
    PerBinRateModelSpec,
    aggregate_to_timebins,
    attach_is_swr_to_onsets,
    cross_validate_per_bin_rate_model,
    fit_per_bin_rate_model,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RC = PROCESSED_DATA_DIR / "replay_content"
SESSION_COLS = ["animal", "day", "epoch"]

# Both directions: (response population, cross/predictor population).
DIRECTIONS = {
    "nonSWR_from_SWR": ("non_local_no_swr", "non_local_swr"),
    "SWR_from_nonSWR": ("non_local_swr", "non_local_no_swr"),
}


def _spec(name: str, *, cross: bool) -> PerBinRateModelSpec:
    """M_self (cross=False) or M_self+cross (cross=True). Reward via the kernel
    only (no reward_rate stop covariate), so the design needs no regenerated
    stop covariate and builds on the current tables."""
    return PerBinRateModelSpec(
        name,
        True,  # time spline
        N_REWARD_LAGS,  # reward kernel
        False,  # reward_rate off
        N_HISTORY_LAGS,  # self-history window
        False,  # no switch term
        False,  # no reward x time interaction
        "non-local cross-history coupling",
        None,
        self_history_basis=N_HISTORY_BASIS,
        cross_history_lags=N_HISTORY_LAGS if cross else 0,
        cross_history_basis=N_HISTORY_BASIS if cross else 0,
    )


M_SELF = _spec("Mself", cross=False)
M_CROSS = _spec("Mselfcross", cross=True)
CROSS_COEFS = [f"coef_cross_hist_rcos_{k}" for k in range(N_HISTORY_BASIS)]


def _fit_direction(stops_g, onsets_g, resp_pop, cross_pop) -> dict | None:
    """Fit M_self and M_self+cross for one session/direction; return a row."""
    resp = POPULATIONS[resp_pop](onsets_g)
    cross = POPULATIONS[cross_pop](onsets_g)
    common = {
        "bin_s": BIN_S,
        "tmax": TMAX,
        "reward_lags": N_REWARD_LAGS,
        "self_history_lags": N_HISTORY_LAGS,
        "self_history_basis": N_HISTORY_BASIS,
    }
    design_self = aggregate_to_timebins(stops_g, resp, **common)
    design_cross = aggregate_to_timebins(
        stops_g, resp, cross_onset_times=cross,
        cross_history_lags=N_HISTORY_LAGS, cross_history_basis=N_HISTORY_BASIS,
        **common,
    )
    fit_self = fit_per_bin_rate_model(design_self, M_SELF)
    fit_cross = fit_per_bin_rate_model(design_cross, M_CROSS)
    if fit_self is None or fit_cross is None:
        return None
    cv_self = cross_validate_per_bin_rate_model(design_self, M_SELF)
    cv_cross = cross_validate_per_bin_rate_model(design_cross, M_CROSS)
    row = {
        "n_bins": fit_cross["n_bins"],
        "n_events": fit_cross["n_events"],
        # self -> +cross improvement (positive = cross history helps)
        "delta_aic": fit_self["aic"] - fit_cross["aic"],
        **{c: fit_cross.get(c, np.nan) for c in CROSS_COEFS},
    }
    if cv_self is not None and cv_cross is not None:
        row["delta_cv_bits"] = (
            cv_cross["cv_bits_per_event"] - cv_self["cv_bits_per_event"]
        )
    return row


def fit_cross_history(stop_table: pd.DataFrame, onsets: pd.DataFrame) -> pd.DataFrame:
    onset_groups = dict(tuple(onsets.groupby(SESSION_COLS, sort=False)))
    rows = []
    for keys, stops_g in stop_table.groupby(SESSION_COLS, sort=False):
        if keys not in onset_groups:
            continue
        onsets_g = onset_groups[keys]
        for direction, (resp_pop, cross_pop) in DIRECTIONS.items():
            r = _fit_direction(stops_g, onsets_g, resp_pop, cross_pop)
            if r is not None:
                rows.append({**dict(zip(SESSION_COLS, keys, strict=True)),
                             "direction": direction, **r})
    return pd.DataFrame(rows)


def _summary(models: pd.DataFrame) -> None:
    for direction, g in models.groupby("direction"):
        print(f"\n=== {direction}  (n={len(g)} sessions) ===")
        da = g["delta_aic"].dropna()
        if len(da) >= 5:
            print(f"  self->+cross AIC drop: median {da.median():+.2f}  "
                  f"%improved {100 * (da > 0).mean():.0f}%  "
                  f"p={wilcoxon(da)[1]:.1e}")
        else:
            print("  (n<5)")
        if "delta_cv_bits" in g:
            dc = g["delta_cv_bits"].dropna()
            if len(dc) >= 5:
                print(f"  cross cv-bits/event gain: median {dc.median():+.4f}  "
                      f"%+ {100 * (dc > 0).mean():.0f}%")
        # cross-history coefficients (FDR across the basis)
        present = [c for c in CROSS_COEFS if c in g and g[c].notna().sum() >= 5]
        if present:
            pvals = [wilcoxon(g[c].dropna())[1] for c in present]
            fdr = multipletests(pvals, method="fdr_bh")[1]
            print("  cross-history basis coefficients (early->late lag):")
            for c, q in zip(present, fdr, strict=True):
                s = g[c].dropna()
                n_an = (g.groupby("animal")[c].median() > 0).sum()
                print(f"    {c.removeprefix('coef_'):<18}{s.median():>+8.3f}"
                      f"{(s > 0).mean() * 100:>6.0f}%{f'{n_an}/5':>7}  q={q:.1e}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-per-animal", type=int, default=None,
                   help="smoke: cap sessions per animal")
    p.add_argument("--write", action="store_true")
    args = p.parse_args()

    stop = pd.read_pickle(RC / "stop_table.pkl")
    onsets = attach_is_swr_to_onsets(
        pd.read_pickle(RC / "event_onset_table.pkl"),
        pd.read_pickle(RC / "event_table.pkl"),
    )
    if args.max_per_animal is not None:
        keep = (
            stop[SESSION_COLS].drop_duplicates()
            .groupby("animal").head(args.max_per_animal)
        )
        stop = stop.merge(keep, on=SESSION_COLS)
        logger.info("smoke: %d sessions", len(stop[SESSION_COLS].drop_duplicates()))

    models = fit_cross_history(stop, onsets)
    _summary(models)
    if args.write:
        out = RC / "models_per_bin_rate" / "cross_history_models.pkl"
        models.to_pickle(out)
        print(f"\nwrote {len(models)} rows -> {out}")


if __name__ == "__main__":
    main()
