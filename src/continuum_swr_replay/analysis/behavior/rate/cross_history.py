"""Bivariate per-bin conditional intensity: does one non-local stream's history
drive the other's rate?

Per session and direction (:data:`DIRECTIONS`), ``MODEL_CROSS_HISTORY_RATE``'s
``Mself`` (time spline + reward kernel + self-history) and ``Mselfcross``
(+ the other stream's history) are fit each on its own frozen cohort; a
session-direction whose two cohorts differ in bins, event bins or stops
(``fit_comparability.why_not_comparable``) is not fit. Each row keeps the
self->cross AIC difference (child minus parent), the cv-bits/event gain and the
cross-history coefficients (:func:`fit_cross_history`).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

from ..fit_comparability import why_not_comparable
from ..model_specs import (
    BIN_S,
    MODEL_CROSS_HISTORY_RATE,
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    TMAX,
    longest_history_window_s,
)
from ..per_bin_design import POPULATIONS, aggregate_to_timebins
from .per_bin_rate_model import (
    cross_validate_per_bin_rate_model,
    fit_per_bin_rate_model,
    freeze_per_bin_cohort,
)

logger = logging.getLogger(__name__)

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
# A fit's cohort-size column -> the cohort freeze's count of it.
COHORT_SIZE_SOURCES = {
    "n_bins": "n_bins",
    "n_events": "n_event_bins",
    "n_stops": "n_stops",
}


def fit_direction(stops_g, onsets_g, resp_pop, cross_pop) -> dict | None:
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
    design_self, freeze_self = freeze_per_bin_cohort(design_self, [M_SELF])
    design_cross, freeze_cross = freeze_per_bin_cohort(design_cross, [M_CROSS])
    # The two rungs are frozen on different columns, so their AIC difference
    # compares only when the two cohorts are the same size.
    paired = pd.DataFrame(
        [
            {
                f"{column}_{rung}": freeze[f"{source}_after"]
                for rung, freeze in (("self", freeze_self), ("cross", freeze_cross))
                for column, source in COHORT_SIZE_SOURCES.items()
            }
        ]
    )
    differs = why_not_comparable(paired, ("_self", "_cross")).iloc[0]
    if differs.any():
        logger.warning(
            "%s -> %s: not fit, the Mself and Mselfcross cohorts differ in %s (%s)",
            cross_pop,
            resp_pop,
            list(differs.index[differs]),
            paired.iloc[0].to_dict(),
        )
        return None
    fit_self = fit_per_bin_rate_model(design_self, M_SELF)
    fit_cross = fit_per_bin_rate_model(design_cross, M_CROSS)
    if fit_self is None or fit_cross is None:
        return None
    cv_self = cross_validate_per_bin_rate_model(design_self, M_SELF)
    cv_cross = cross_validate_per_bin_rate_model(design_cross, M_CROSS)
    row = {
        "n_bins": fit_cross["n_bins"],
        "n_events": fit_cross["n_events"],
        # child minus parent: negative = cross history helps
        "delta_aic": fit_cross["aic"] - fit_self["aic"],
        **{c: fit_cross.get(c, np.nan) for c in CROSS_COEFS},
    }
    if cv_self is not None and cv_cross is not None:
        row["delta_cv_bits"] = (
            cv_cross["cv_bits_per_event"] - cv_self["cv_bits_per_event"]
        )
    return row


def fit_cross_history(stop_table: pd.DataFrame, onsets: pd.DataFrame) -> pd.DataFrame:
    """One row per (session, direction) whose two rungs fit (:func:`fit_direction`).

    ``onsets`` must carry ``is_swr``; a stop-session without onsets is skipped.
    """
    onset_groups = dict(tuple(onsets.groupby(list(SESSION_COLUMNS), sort=False)))
    rows = []
    for keys, stops_g in stop_table.groupby(list(SESSION_COLUMNS), sort=False):
        if keys not in onset_groups:
            continue
        onsets_g = onset_groups[keys]
        for direction, (resp_pop, cross_pop) in DIRECTIONS.items():
            r = fit_direction(stops_g, onsets_g, resp_pop, cross_pop)
            if r is not None:
                rows.append(
                    {
                        **dict(zip(SESSION_COLUMNS, keys, strict=True)),
                        "direction": direction,
                        **r,
                    }
                )
    return pd.DataFrame(rows)
