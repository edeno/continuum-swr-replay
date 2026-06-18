#!/usr/bin/env python
"""Fit the per-stop replay-rate hierarchy (R0-R3) and persist it.

The rate models are ordinary Poisson GLMs on per-stop event counts with a
``log(immobility_time)`` offset (see ``rate_model.fit_rate_model``) — a
different model family from the conditional arm/patch hierarchies, so they
live outside ``fit_replay_content_models.py``. This script fits R0-R3
per session from the saved ``stop_table.pkl``, adds K-fold cross-validated
per-stop log-likelihood (``cross_validate_rate_model``), and saves a tidy
one-row-per-(model, session) table mirroring the arm/patch ``all_models.pkl``.

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


def fit_rate_hierarchy(stop_table: pd.DataFrame, run_cv: bool = True) -> pd.DataFrame:
    """One row per (model, session): R0-R3 fit stats, coefficients, and CV-LL."""
    rows = []
    for keys, group in stop_table.groupby(SESSION_COLS, sort=False):
        for name, spec in MODEL_RATE.items():
            res = fit_rate_model(group, list(spec.predictors))
            if res is None:
                continue
            row = {"model": name, **dict(zip(SESSION_COLS, keys, strict=True)), **res}
            if run_cv:
                cv = cross_validate_rate_model(group, list(spec.predictors))
                if cv is not None:
                    row.update(cv)
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    stop = pd.read_pickle(RC / "stop_table.pkl")
    logger.info(
        "Fitting rate hierarchy R0-R3 (+CV) over %d sessions...",
        stop.groupby(SESSION_COLS).ngroups,
    )
    rate = fit_rate_hierarchy(stop, run_cv=True)

    out_dir = RC / "models_rate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "all_models.pkl"
    rate.to_pickle(out)
    logger.info(
        "Saved %d rows (%d sessions) to %s",
        len(rate),
        rate.groupby(SESSION_COLS).ngroups,
        out,
    )


if __name__ == "__main__":
    main()
