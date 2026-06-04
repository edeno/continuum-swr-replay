#!/usr/bin/env python
"""Run the replay-content analysis suite on one event source and save fits.

Mirrors the non-local pipeline so non-local / ContFrag-SWR / ContFrag-MUA
can be compared head-to-head. For a given event source it fits, per
session:

- Patch hierarchy P0-P4 (conditional Poisson on patch counts)
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

# event source -> (input directory, table-name stem). ``None`` stem means
# the non-local replay-content tables (HPC clusterless only, no suffix).
# The ContFrag sources are per-brain-area; the brain area is appended at
# load time so mPFC/OFC outputs aren't silently read as HPC.
_SOURCES = {
    "non_local": ("replay_content", None),
    "SWR": ("contfrag_replay_content", "swr"),
    "MUA": ("contfrag_replay_content", "mua"),
}


def _load_tables(event_source: str, brain_area: str) -> dict[str, pd.DataFrame]:
    """Load event / event_arm / stop tables for an event source."""
    sub, stem = _SOURCES[event_source]
    if stem is None:
        suffix = ""  # non-local replay content is HPC clusterless only
        if brain_area != "HPC":
            logger.warning(
                "event-source 'non_local' is HPC-only; ignoring --brain-area %s",
                brain_area,
            )
    else:
        suffix = f"_{stem}_{brain_area}"
    d = PROCESSED_DATA_DIR / sub
    tables = {}
    for short, base in [
        ("event_table", "event_table"),
        ("event_arm_table", "event_arm_table"),
        ("stop_table", "stop_table"),
    ]:
        path = d / f"{base}{suffix}.pkl"
        tables[short] = pd.read_pickle(path)
        logger.info("loaded %s: %d rows", path.name, len(tables[short]))
    return tables


def _fit_per_session(table, models, fit_fn) -> dict[str, pd.DataFrame]:
    """Fit each model in ``models`` per session and stack the results.

    ``fit_fn(group, spec)`` returns a per-session fit dict, or ``None``
    to skip the session (insufficient data / non-convergence).
    """
    out = {}
    for name, spec in models.items():
        rows = []
        for keys, g in table.groupby(SESS):
            r = fit_fn(g, spec)
            if r is not None:
                rows.append({**dict(zip(SESS, keys, strict=True)), **r})
        out[name] = pd.DataFrame(rows)
    return out


def _fit_patch_conditional(g: pd.DataFrame, spec) -> dict | None:
    """Patch conditional Poisson on one session; skip if < 10 events."""
    if g["event_id"].nunique() < 10:
        return None
    return fit_conditional_poisson(
        g, list(spec.predictors), response_col="patch_count_hard",
        event_col="event_id", n_categories=3,
    )


def fit_patch_hierarchy(patch_table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Fit the MODEL_PATCH conditional Poisson hierarchy (P0-P4) per session."""
    return _fit_per_session(patch_table, MODEL_PATCH, _fit_patch_conditional)


def fit_patch_switch(
    patch_table: pd.DataFrame, event_arm_table: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Fit PS0-PS2 on switch events only (is_after_switch == 1)."""
    sw = event_arm_table.groupby("event_id")["is_after_switch"].first()
    switch_patch = patch_table.merge(
        sw.rename("is_after_switch"), left_on="event_id", right_index=True, how="left"
    )
    switch_patch = switch_patch[switch_patch["is_after_switch"] == 1]
    return _fit_per_session(switch_patch, MODEL_PATCH_SWITCH, _fit_patch_conditional)


def fit_rate_hierarchy(stop_table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Fit the MODEL_RATE Poisson rate models (R0/R1) per session."""
    return _fit_per_session(
        stop_table,
        MODEL_RATE,
        lambda g, spec: fit_rate_model(g, list(spec.predictors), min_stops=20),
    )


def fit_choice(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Fit the MODEL_CHOICE conditional logit per session."""
    choice = build_choice_dataset(
        tables["event_table"], tables["event_arm_table"], tables["stop_table"]
    )

    def _fit(g: pd.DataFrame, spec) -> dict | None:
        if g["stop_id"].nunique() < 10:
            return None
        # Drop any spec predictor absent from this choice dataset
        # (the behavioural block is defined optimistically).
        preds = [p for p in spec.predictors if p in choice.columns]
        return fit_choice_model(g, preds)

    return _fit_per_session(choice, MODEL_CHOICE, _fit)


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
    parser.add_argument(
        "--brain-area",
        default="HPC",
        help="Brain area for ContFrag sources (SWR/MUA); ignored for "
        "non_local (HPC-only). Default: HPC.",
    )
    args = parser.parse_args()
    src = args.event_source

    tables = _load_tables(src, args.brain_area)
    out_dir = PROCESSED_DATA_DIR / "replay_content" / "event_source_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Built once — both patch fits consume it.
    patch_table = build_event_patch_table(
        tables["event_table"], tables["event_arm_table"]
    )

    results = {
        "patch_hierarchy": fit_patch_hierarchy(patch_table),
        "patch_switch": fit_patch_switch(patch_table, tables["event_arm_table"]),
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

    # Include the brain area: ContFrag sources are per-brain-area, so
    # SWR/HPC and SWR/mPFC must not overwrite each other. (non_local is
    # HPC-only; its name is then non_local_HPC.)
    out_path = out_dir / f"{src}_{args.brain_area}_comparison_fits.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    logger.info("saved %s", out_path)


if __name__ == "__main__":
    main()
