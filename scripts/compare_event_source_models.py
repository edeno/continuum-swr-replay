#!/usr/bin/env python
"""Run the replay-content analysis suite on one event source and save fits.

Mirrors the non-local pipeline so non-local / ContFrag-SWR / ContFrag-MUA
can be compared head-to-head. For a given event source it fits, per
session:

- Patch hierarchy P0-P3 (conditional Poisson on patch counts)
- Rate hierarchy R0/R1 (Poisson on n_events per stop, log-immobility offset)
- MODEL_PATCH_SWITCH PS0-PS2 (patch conditional Poisson, switch events only)
- Choice-prediction conditional logit (replay_frac vs behavioral baseline)
- Switch-aligned replay content (frac on/off current patch by lag)

The arm hierarchy (M0-M6 / M0-M5) is fit by ``fit_replay_content_models.py``
and is not repeated here.

Usage::

    uv run python scripts/compare_event_source_models.py --event-source non_local
    uv run python scripts/compare_event_source_models.py --event-source SWR
    uv run python scripts/compare_event_source_models.py --event-source MUA
"""

from __future__ import annotations

import argparse
import logging
import pickle

import pandas as pd

from continuum_swr_replay.analysis.replay import (
    MODEL_CHOICE,
    MODEL_PATCH,
    MODEL_PATCH_SWITCH,
    MODEL_RATE,
    build_choice_dataset,
    build_event_patch_table,
    build_switch_aligned_table,
    fit_choice_model,
    fit_conditional_poisson,
    fit_rate_model,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("compare_event_source")
# Silence the per-session rank-deficiency / convergence chatter.
for noisy in (
    "continuum_swr_replay.analysis.replay.rate_model",
    "continuum_swr_replay.analysis.replay.fitting",
    "continuum_swr_replay.analysis.replay.choice_prediction",
):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)

SESS = ["animal", "day", "epoch"]

# event source -> (input directory, table-name suffix)
_SOURCES = {
    "non_local": ("replay_content", ""),
    "SWR": ("contfrag_replay_content", "_swr_HPC"),
    "MUA": ("contfrag_replay_content", "_mua_HPC"),
}


def _load_tables(event_source: str) -> dict[str, pd.DataFrame]:
    """Load event / event_arm / stop tables for an event source."""
    sub, suffix = _SOURCES[event_source]
    d = PROCESSED_DATA_DIR / sub
    tables = {}
    for short, base in [
        ("event_table", "event_table"),
        ("event_arm_table", "event_arm_table"),
        ("stop_table", "stop_table"),
    ]:
        path = d / f"{base}{suffix}.pkl"
        tables[short] = pickle.load(open(path, "rb"))
        logger.info("loaded %s: %d rows", path.name, len(tables[short]))
    return tables


def fit_patch_hierarchy(
    event_table: pd.DataFrame, event_arm_table: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Fit P0-P3 conditional Poisson per session on patch counts."""
    patch_table = build_event_patch_table(event_table, event_arm_table)
    out = {}
    for name, spec in MODEL_PATCH.items():
        preds = list(spec.predictors)
        rows = []
        for keys, g in patch_table.groupby(SESS):
            if g["event_id"].nunique() < 10:
                continue
            r = fit_conditional_poisson(
                g, preds, response_col="patch_count_hard",
                event_col="event_id", n_categories=3,
            )
            if r is not None:
                r2 = dict(zip(SESS, keys, strict=True))
                r2.update(r)
                rows.append(r2)
        out[name] = pd.DataFrame(rows)
    return out


def fit_patch_switch(
    event_table: pd.DataFrame, event_arm_table: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Fit PS0-PS2 on switch events only (is_after_switch == 1)."""
    patch_table = build_event_patch_table(event_table, event_arm_table)
    sw = event_arm_table.groupby("event_id")["is_after_switch"].first()
    patch_table = patch_table.merge(
        sw.rename("is_after_switch"), left_on="event_id", right_index=True, how="left"
    )
    switch_patch = patch_table[patch_table["is_after_switch"] == 1]
    out = {}
    for name, spec in MODEL_PATCH_SWITCH.items():
        preds = list(spec.predictors)
        rows = []
        for keys, g in switch_patch.groupby(SESS):
            if g["event_id"].nunique() < 10:
                continue
            r = fit_conditional_poisson(
                g, preds, response_col="patch_count_hard",
                event_col="event_id", n_categories=3,
            )
            if r is not None:
                r2 = dict(zip(SESS, keys, strict=True))
                r2.update(r)
                rows.append(r2)
        out[name] = pd.DataFrame(rows)
    return out


def fit_rate_hierarchy(stop_table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Fit the MODEL_RATE Poisson rate models (R0/R1) per session."""
    out = {}
    for name, spec in MODEL_RATE.items():
        preds = list(spec.predictors)
        rows = []
        for keys, g in stop_table.groupby(SESS):
            r = fit_rate_model(g, preds, min_stops=20)
            if r is not None:
                r2 = dict(zip(SESS, keys, strict=True))
                r2.update(r)
                rows.append(r2)
        out[name] = pd.DataFrame(rows)
    return out


def fit_choice(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Fit choice-prediction conditional logit per session."""
    choice = build_choice_dataset(
        tables["event_table"], tables["event_arm_table"], tables["stop_table"]
    )
    out = {}
    for name, spec in MODEL_CHOICE.items():
        # Drop any spec predictor absent from this choice dataset
        # (the behavioural block is defined optimistically).
        preds = [p for p in spec.predictors if p in choice.columns]
        rows = []
        for keys, g in choice.groupby(SESS):
            if g["stop_id"].nunique() < 10:
                continue
            r = fit_choice_model(g, preds)
            if r is not None:
                r2 = dict(zip(SESS, keys, strict=True))
                r2.update(r)
                rows.append(r2)
        out[name] = pd.DataFrame(rows)
    return out


def run_switch_aligned(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Switch-aligned replay content by lag."""
    return build_switch_aligned_table(
        tables["stop_table"], tables["event_table"],
        tables["event_arm_table"], max_lag=10,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--event-source", choices=list(_SOURCES), default="non_local"
    )
    args = parser.parse_args()
    src = args.event_source

    tables = _load_tables(src)
    out_dir = PROCESSED_DATA_DIR / "replay_content" / "event_source_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "patch_hierarchy": fit_patch_hierarchy(
            tables["event_table"], tables["event_arm_table"]
        ),
        "patch_switch": fit_patch_switch(
            tables["event_table"], tables["event_arm_table"]
        ),
        "rate_hierarchy": fit_rate_hierarchy(tables["stop_table"]),
        "choice": fit_choice(tables),
        "switch_aligned": run_switch_aligned(tables),
    }
    for analysis, res in results.items():
        n = (
            len(res)
            if isinstance(res, pd.DataFrame)
            else {k: len(v) for k, v in res.items()}
        )
        logger.info("%s [%s]: %s", analysis, src, n)

    out_path = out_dir / f"{src}_comparison_fits.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    logger.info("saved %s", out_path)


if __name__ == "__main__":
    main()
