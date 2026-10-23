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

Per session, per direction (specs: ``model_specs.MODEL_CROSS_HISTORY_RATE``):
  Mself      = bs(tc) + reward kernel + self-history (raised cosine)
  Mselfcross = Mself + cross-history (raised cosine)
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

from continuum_swr_replay.analysis.events.analysed_time import (  # noqa: E402
    require_swr_source,
)
from continuum_swr_replay.analysis.events.swr_source import (  # noqa: E402
    swr_source_name,
)
from continuum_swr_replay.analysis.replay import (  # noqa: E402
    BIN_S,
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    POPULATIONS,
    TMAX,
    aggregate_to_timebins,
    attach_is_swr_to_onsets,
    cross_validate_per_bin_rate_model,
    fit_per_bin_rate_model,
    freeze_per_bin_cohort,
)
from continuum_swr_replay.analysis.replay.model_specs import (  # noqa: E402
    MODEL_CROSS_HISTORY_RATE,
    longest_history_window_s,
)
from continuum_swr_replay.paths import REPLAY_CONTENT_DIR, model_dir, table_path
from continuum_swr_replay.tables.covariate_registry import (  # noqa: E402
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RC = REPLAY_CONTENT_DIR

# Both directions: (response population, cross/predictor population).
DIRECTIONS = {
    "nonSWR_from_SWR": ("non_local_no_swr", "non_local_swr"),
    "SWR_from_nonSWR": ("non_local_swr", "non_local_no_swr"),
}

M_SELF = MODEL_CROSS_HISTORY_RATE["Mself"]
M_CROSS = MODEL_CROSS_HISTORY_RATE["Mselfcross"]
# Both designs drop the bins whose longest history window (self or cross)
# touches excluded time, so the two are fit to the same bins.
HISTORY_WINDOW_S = longest_history_window_s([M_SELF, M_CROSS])
CROSS_COEFS = [f"coef_cross_hist_rcos_{k}" for k in range(N_HISTORY_BASIS)]


def _fit_direction(stops_g, onsets_g, resp_pop, cross_pop) -> dict | None:
    """Fit M_self and M_self+cross for one session/direction; return a row."""
    resp = POPULATIONS[resp_pop](onsets_g)
    cross = POPULATIONS[cross_pop](onsets_g)
    common = {
        "bin_s": BIN_S,
        "tmax": TMAX,
        "reward_basis": True,
        "self_history_lags": N_HISTORY_LAGS,
        "self_history_basis": N_HISTORY_BASIS,
        "history_window_s": HISTORY_WINDOW_S,
    }
    design_self = aggregate_to_timebins(stops_g, resp, **common)
    design_cross = aggregate_to_timebins(
        stops_g,
        resp,
        cross_onset_times=cross,
        cross_history_lags=N_HISTORY_LAGS,
        cross_history_basis=N_HISTORY_BASIS,
        **common,
    )
    # One cohort per design before the first fit: the fitters refuse an
    # incomplete design rather than subsetting it per rung.
    design_self, _ = freeze_per_bin_cohort(design_self, [M_SELF])
    design_cross, _ = freeze_per_bin_cohort(design_cross, [M_CROSS])
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
    onset_groups = dict(tuple(onsets.groupby(list(SESSION_COLUMNS), sort=False)))
    rows = []
    for keys, stops_g in stop_table.groupby(list(SESSION_COLUMNS), sort=False):
        if keys not in onset_groups:
            continue
        onsets_g = onset_groups[keys]
        for direction, (resp_pop, cross_pop) in DIRECTIONS.items():
            r = _fit_direction(stops_g, onsets_g, resp_pop, cross_pop)
            if r is not None:
                rows.append(
                    {
                        **dict(zip(SESSION_COLUMNS, keys, strict=True)),
                        "direction": direction,
                        **r,
                    }
                )
    return pd.DataFrame(rows)


def _summary(models: pd.DataFrame) -> None:
    for direction, g in models.groupby("direction"):
        print(f"\n=== {direction}  (n={len(g)} sessions) ===")
        da = g["delta_aic"].dropna()
        if len(da) >= 5:
            print(
                f"  self->+cross AIC drop: median {da.median():+.2f}  "
                f"%improved {100 * (da > 0).mean():.0f}%  "
                f"p={wilcoxon(da)[1]:.1e}"
            )
        else:
            print("  (n<5)")
        if "delta_cv_bits" in g:
            dc = g["delta_cv_bits"].dropna()
            if len(dc) >= 5:
                print(
                    f"  cross cv-bits/event gain: median {dc.median():+.4f}  "
                    f"%+ {100 * (dc > 0).mean():.0f}%"
                )
        # cross-history coefficients (FDR across the basis)
        present = [c for c in CROSS_COEFS if c in g and g[c].notna().sum() >= 5]
        if present:
            pvals = [wilcoxon(g[c].dropna())[1] for c in present]
            fdr = multipletests(pvals, method="fdr_bh")[1]
            print("  cross-history basis coefficients (early->late lag):")
            for c, q in zip(present, fdr, strict=True):
                s = g[c].dropna()
                n_an = (g.groupby("animal")[c].median() > 0).sum()
                print(
                    f"    {c.removeprefix('coef_'):<18}{s.median():>+8.3f}"
                    f"{(s > 0).mean() * 100:>6.0f}%{f'{n_an}/5':>7}  q={q:.1e}"
                )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--max-per-animal",
        type=int,
        default=None,
        help="smoke: cap sessions per animal",
    )
    p.add_argument("--write", action="store_true")
    args = p.parse_args()

    stop_path = table_path("stop_table", RC, fmt="pkl")
    onset_path = table_path("event_onset_table", RC, fmt="pkl")
    event_path = table_path("event_table", RC, fmt="pkl")
    stop = read_table(stop_path)
    onset_table = read_table(onset_path)
    event_table = read_table(event_path)
    require_covariate_schema(stop, table="stop")
    require_covariate_schema(onset_table, table="event_onset")
    require_covariate_schema(event_table, table="event")
    # The bins are dropped by the configured source's excluded time, so the
    # tables must come from that source (its artifact run included).
    require_swr_source(stop, what=str(stop_path))
    require_swr_source(onset_table, what=str(onset_path))
    require_swr_source(event_table, what=str(event_path))
    onsets = attach_is_swr_to_onsets(onset_table, event_table)
    if args.max_per_animal is not None:
        keep = (
            stop[list(SESSION_COLUMNS)]
            .drop_duplicates()
            .groupby("animal")
            .head(args.max_per_animal)
        )
        stop = stop.merge(keep, on=list(SESSION_COLUMNS))
        logger.info(
            "smoke: %d sessions", len(stop[list(SESSION_COLUMNS)].drop_duplicates())
        )

    models = fit_cross_history(stop, onsets)
    _summary(models)
    if args.write:
        out = model_dir("per_bin_rate", RC) / "cross_history_models.pkl"
        models.attrs["swr_source"] = swr_source_name()
        models.to_pickle(out)
        print(f"\nwrote {len(models)} rows -> {out}")


if __name__ == "__main__":
    main()
