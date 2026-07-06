#!/usr/bin/env python
"""Fit the per-stop replay-rate hierarchy (R0-R2 + Rlag) and persist it.

The rate models are ordinary Poisson GLMs on per-stop event counts with a
``log(immobility_time)`` offset (see ``per_stop_rate_model.fit_per_stop_rate_model``) — a
different model family from the conditional arm/patch hierarchies, so they
live outside ``fit_replay_content_models.py``. This script fits R0-R2 + Rlag
per session from the saved ``stop_table.pkl``, adds K-fold cross-validated
per-stop log-likelihood (``cross_validate_per_stop_rate_model``), and saves a tidy
one-row-per-(outcome, model, session) table mirroring the arm/patch
``all_models.pkl``.

The response is fit on four per-stop count outcomes (tagged by ``outcome``),
all read directly from the canonical ``stop_table`` columns:

- ``swr``              -> ``n_SWR``              : detected ripples (LFP) — the
                                                  Ambrose-2016 positive control.
- ``non_local``        -> ``n_non_local``        : all non-local decoding events.
- ``non_local_swr``    -> ``n_non_local_SWR``    : non-local events overlapping a SWR.
- ``non_local_no_swr`` -> ``n_non_local_no_SWR`` : non-local events without a SWR.

Fitting each MODEL_PER_STOP_RATE model on each outcome is an outcome *variant* of the specs,
not a new model. Replicates the canonical SWR-rate-vs-reward result (reward /
omission, recent reward history) on detected ripples, then asks whether the
non-SWR non-local events share those rate effects.

Usage:
    uv run python scripts/rate/fit_per_stop_rate_models.py
"""

from __future__ import annotations

import logging

import pandas as pd

from continuum_swr_replay.analysis.replay.cohort_audit import (
    cohort_record,
    write_cohort_audit,
)
from continuum_swr_replay.analysis.replay.model_specs import (
    MODEL_PER_STOP_RATE,
    N_REWARD_LAGS,
)
from continuum_swr_replay.analysis.replay.per_stop_rate_model import (
    cross_validate_per_stop_rate_model,
    fit_per_stop_rate_model,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

SESSION_COLS = ["animal", "day", "epoch"]
RC = PROCESSED_DATA_DIR / "replay_content"
# Per-stop count outcomes (canonical stop_table columns), tagged by `outcome`.
RESPONSE_COLS = {
    "swr": "n_SWR",
    "non_local": "n_non_local",
    "non_local_swr": "n_non_local_SWR",
    "non_local_no_swr": "n_non_local_no_SWR",
}


def fit_per_stop_rate_hierarchy(
    stop_table: pd.DataFrame, run_cv: bool = True
) -> tuple[pd.DataFrame, list[dict]]:
    """One row per (outcome, model, session): R0-R2 + Rlag fit stats, coefs, CV-LL.

    Also returns one stop-grain cohort-audit record per (model, outcome): dropna /
    immobility<=0 stops go to ``n_drop_own_predictor_missing``; a skipped session's
    remaining stops (min_stops / rank gate) go to ``n_excluded_session``; the fitted
    ``n_stops`` sum is ``n_modeled``. Reconciles by construction.
    """
    rows = []
    acc: dict = {}  # (model, outcome) -> counts
    for keys, group in stop_table.groupby(SESSION_COLS, sort=False):
        for outcome, response_col in RESPONSE_COLS.items():
            if response_col not in group.columns:
                continue
            for name, spec in MODEL_PER_STOP_RATE.items():
                preds = list(spec.predictors)
                a = acc.setdefault((name, outcome), {
                    "n_in": 0, "n_own": 0, "n_modeled": 0,
                    "sessions_in": 0, "sessions_modeled": 0,
                    "animals": set(), "n_nonconv": 0, "n_sep": 0,
                })
                a["sessions_in"] += 1
                a["n_in"] += len(group)
                avail = [
                    c for c in (response_col, "immobility_time", *preds)
                    if c in group.columns
                ]
                bad = group[avail].isna().any(axis=1) if avail else pd.Series(False, index=group.index)
                if "immobility_time" in group.columns:
                    bad = bad | (group["immobility_time"] <= 0)
                a["n_own"] += int(bad.sum())

                res = fit_per_stop_rate_model(group, preds, response_col=response_col)
                if res is None:
                    continue
                a["n_modeled"] += int(res["n_stops"])
                a["sessions_modeled"] += 1
                a["animals"].add(keys[0])
                a["n_nonconv"] += res.get("converged") is False
                a["n_sep"] += bool(res.get("separated"))
                row = {
                    "outcome": outcome,
                    "model": name,
                    **dict(zip(SESSION_COLS, keys, strict=True)),
                    **res,
                }
                if run_cv:
                    cv = cross_validate_per_stop_rate_model(
                        group, preds, response_col=response_col
                    )
                    if cv is not None:
                        row.update(cv)
                rows.append(row)

    records = [
        cohort_record(
            name, outcome, "stop",
            n_in=a["n_in"], n_modeled=a["n_modeled"],
            sessions_in=a["sessions_in"], sessions_modeled=a["sessions_modeled"],
            animals_modeled=len(a["animals"]),
            drop_notes="dropna/immobility<=0 -> own; session gates -> excluded",
            n_drop_own_predictor_missing=a["n_own"],
            n_excluded_session=a["n_in"] - a["n_own"] - a["n_modeled"],
            n_sessions_flagged_nonconverged=a["n_nonconv"],
            n_sessions_flagged_separated=a["n_sep"],
        )
        for (name, outcome), a in acc.items()
    ]
    return pd.DataFrame(rows), records


def main() -> None:
    stop = pd.read_pickle(RC / "stop_table.pkl")
    missing = [c for c in RESPONSE_COLS.values() if c not in stop.columns]
    if missing:
        raise SystemExit(
            f"stop_table missing outcome columns {missing}; "
            "regenerate it first (scripts/decoding/regenerate_stop_table.py)."
        )
    lag_cols = [f"reward_lag{j}" for j in range(1, N_REWARD_LAGS + 1)]
    if any(c not in stop.columns for c in lag_cols):
        logger.warning(
            "stop_table lacks reward_lag columns — the Rlag model will be SKIPPED "
            "(every session returns None). Regenerate the stop_table first."
        )
    logger.info(
        "Fitting rate hierarchy %s (+CV) x %s over %d sessions...",
        list(MODEL_PER_STOP_RATE),
        list(RESPONSE_COLS),
        stop.groupby(SESSION_COLS).ngroups,
    )
    rate, cohort_audits = fit_per_stop_rate_hierarchy(stop, run_cv=True)

    out_dir = RC / "models_per_stop_rate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "all_models.pkl"
    rate.to_pickle(out)
    write_cohort_audit(cohort_audits, out_dir / "all_models_cohort_audit.csv")
    logger.info(
        "Saved %d rows (%d sessions, outcomes=%s) to %s",
        len(rate),
        rate.groupby(SESSION_COLS).ngroups,
        sorted(rate["outcome"].unique()),
        out,
    )


if __name__ == "__main__":
    main()
