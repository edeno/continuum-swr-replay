#!/usr/bin/env python
"""Fit the per-stop replay-rate hierarchy (R0-R2) and persist it.

The rate models are ordinary Poisson GLMs on per-stop event counts with a
``log(immobility_time)`` offset (see ``rate_model.fit_rate_model``) — a
different model family from the conditional arm/patch hierarchies, so they
live outside ``fit_replay_content_models.py``. This script fits R0-R2
per session from the saved ``stop_table.pkl``, adds K-fold cross-validated
per-stop log-likelihood (``cross_validate_rate_model``), and saves a tidy
one-row-per-(outcome, model, session) table mirroring the arm/patch
``all_models.pkl``.

The response is fit three ways, tagged by the ``outcome`` column: the total
non-local count (``n_events``, outcome=``all``) and — split by SWR overlap —
the SWR non-local count (``n_events_swr``, outcome=``swr``) and the non-SWR
non-local count (``n_events_nonswr``, outcome=``non_swr``). The split counts
are derived deterministically from the canonical ``event_table.pkl``
(per-stop counts by ``is_swr``); fitting R0-R2 on each is an *outcome variant*
of the same ``MODEL_RATE`` specs, not a new model (cf. ``rate_model`` docstring).
This is what answers "is the *occurrence rate* of non-SWR vs SWR events
modulated by reward history" — distinct from ``is_swr_stop`` as a predictor,
which was dropped as tautological with the pooled response.

Usage:
    uv run python scripts/fit_rate_models.py
"""

from __future__ import annotations

import logging

import pandas as pd

from continuum_swr_replay.analysis.replay.model_specs import MODEL_RATE
from continuum_swr_replay.analysis.replay.rate_model import (
    cross_validate_rate_model,
    fit_rate_model,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

SESSION_COLS = ["animal", "day", "epoch"]
RC = PROCESSED_DATA_DIR / "replay_content"
# Outcome variants: total, SWR-only, and non-SWR-only per-stop event counts.
RESPONSE_COLS = {"all": "n_events", "swr": "n_events_swr", "non_swr": "n_events_nonswr"}


def add_swr_split_counts(stop_table: pd.DataFrame, event_table: pd.DataFrame) -> pd.DataFrame:
    """Add per-stop SWR / non-SWR non-local event counts from the event table.

    Deterministic: ``groupby(stop_id, is_swr)`` counts merged onto the stop
    table; stops with no events of a type get 0. The existing ``n_events``
    total is left untouched (and ``n_events_swr + n_events_nonswr`` should equal
    ``n_events`` up to the event/stop assignment convention — checked in main).
    """
    ev = event_table.assign(_swr=event_table["is_swr"].astype(bool))
    counts = (
        ev.groupby(["stop_id", "_swr"])
        .size()
        .unstack("_swr", fill_value=0)
        .rename(columns={True: "n_events_swr", False: "n_events_nonswr"})
    )
    for col in ("n_events_swr", "n_events_nonswr"):
        if col not in counts.columns:
            counts[col] = 0
    out = stop_table.merge(
        counts[["n_events_swr", "n_events_nonswr"]],
        left_on="stop_id",
        right_index=True,
        how="left",
    )
    out[["n_events_swr", "n_events_nonswr"]] = (
        out[["n_events_swr", "n_events_nonswr"]].fillna(0).astype(int)
    )
    return out


def fit_rate_hierarchy(stop_table: pd.DataFrame, run_cv: bool = True) -> pd.DataFrame:
    """One row per (outcome, model, session): R0-R2 fit stats, coefficients, CV-LL.

    ``outcome`` ∈ {all, swr, non_swr} selects the per-stop count response column.
    """
    rows = []
    for keys, group in stop_table.groupby(SESSION_COLS, sort=False):
        for outcome, response_col in RESPONSE_COLS.items():
            if response_col not in group.columns:
                continue
            for name, spec in MODEL_RATE.items():
                res = fit_rate_model(group, list(spec.predictors), response_col=response_col)
                if res is None:
                    continue
                row = {
                    "outcome": outcome,
                    "model": name,
                    **dict(zip(SESSION_COLS, keys, strict=True)),
                    **res,
                }
                if run_cv:
                    cv = cross_validate_rate_model(
                        group, list(spec.predictors), response_col=response_col
                    )
                    if cv is not None:
                        row.update(cv)
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    stop = pd.read_pickle(RC / "stop_table.pkl")
    events = pd.read_pickle(RC / "event_table.pkl")
    stop = add_swr_split_counts(stop, events)

    # Sanity: the SWR + non-SWR split should reconstruct the total event count.
    mismatch = int((stop["n_events_swr"] + stop["n_events_nonswr"] != stop["n_events"]).sum())
    if mismatch:
        logger.warning(
            "%d/%d stops where n_events_swr + n_events_nonswr != n_events "
            "(event/stop assignment convention differs)",
            mismatch,
            len(stop),
        )

    logger.info(
        "Fitting rate hierarchy R0-R2 (+CV) x {all, swr, non_swr} over %d sessions...",
        stop.groupby(SESSION_COLS).ngroups,
    )
    rate = fit_rate_hierarchy(stop, run_cv=True)

    out_dir = RC / "models_rate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "all_models.pkl"
    rate.to_pickle(out)
    logger.info(
        "Saved %d rows (%d sessions, outcomes=%s) to %s",
        len(rate),
        rate.groupby(SESSION_COLS).ngroups,
        sorted(rate["outcome"].unique()),
        out,
    )


if __name__ == "__main__":
    main()
