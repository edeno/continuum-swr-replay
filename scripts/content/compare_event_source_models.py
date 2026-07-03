#!/usr/bin/env python
"""Run the replay-content analysis suite on one event source and save fits.

Mirrors the non-local pipeline so non-local / ContFrag-SWR / ContFrag-MUA
can be compared head-to-head. For a given event source it fits, per
session:

- Patch hierarchy P0-P4 for non-local events; P0-P3 for source-restricted
  SWR/MUA events where SWR interaction terms are not identifiable
- Rate hierarchy R0-R2 (the ``is_swr_stop`` term R3 was dropped: it is
  tautological with the per-stop non-local/SWR event count)
- MODEL_PATCH_SWITCH PS0-PS3 for non-local events; PS0-PS2 for
  source-restricted SWR/MUA events
- Choice-prediction conditional logit (replay_frac vs behavioral baseline)
- Switch-aligned replay content (frac on/off current patch by lag)

The arm hierarchy (M0-M6 / M0-M5) is fit by ``fit_replay_content_models.py``
and is not repeated here.

Usage::

    uv run python scripts/content/compare_event_source_models.py --event-source non_local
    uv run python scripts/content/compare_event_source_models.py --event-source SWR
    uv run python scripts/content/compare_event_source_models.py --event-source MUA
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import pickle
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.replay import (
    MODEL_CHOICE,
    MODEL_PATCH,
    MODEL_PATCH_MUA,
    MODEL_PATCH_SWITCH,
    MODEL_PATCH_SWR,
    MODEL_PER_STOP_RATE,
    build_choice_dataset,
    build_event_patch_table,
    build_switch_aligned_table,
    fit_choice_model,
    fit_conditional_poisson,
    fit_per_stop_rate_model,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("compare_event_source")
# Silence the per-session rank-deficiency / convergence chatter.
for noisy in (
    "continuum_swr_replay.analysis.replay.per_stop_rate_model",
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


def _effective_brain_area(event_source: str, brain_area: str) -> str:
    """Return the data brain area actually used for this event source."""
    if event_source == "non_local":
        return "HPC"
    return brain_area


def _patch_models_for_source(event_source: str):
    """Patch hierarchy valid for the event-source definition."""
    if event_source == "SWR":
        return MODEL_PATCH_SWR
    if event_source == "MUA":
        return MODEL_PATCH_MUA
    return MODEL_PATCH


def _patch_switch_models_for_source(event_source: str):
    """Switch hierarchy valid for the event-source definition."""
    if event_source in {"SWR", "MUA"}:
        return {name: MODEL_PATCH_SWITCH[name] for name in ("PS0", "PS1", "PS2")}
    return MODEL_PATCH_SWITCH


def _rate_models_for_source(event_source: str):
    """Rate hierarchy for the event-source definition (R0-R2 for all sources).

    The ``is_swr_stop`` term (formerly R3) was removed from ``MODEL_PER_STOP_RATE``: it
    is tautological with a per-stop event count that overlaps SWRs by
    construction, so it is dropped for every source, not just SWR.
    """
    return MODEL_PER_STOP_RATE


def _choice_models_for_source(event_source: str):  # noqa: ARG001
    """Primary choice models fit on one common stop cohort."""
    return {
        name: MODEL_CHOICE[name]
        for name in ("C_replay_only", "C_behavior", "C_full")
    }


def _choice_swr_split_models_for_source(event_source: str):
    """SWR-split choice variant, skipped when SWR defines every event."""
    if event_source == "SWR":
        return {}
    return {
        name: MODEL_CHOICE[name]
        for name in ("C_behavior", "C_full_swr_split")
    }


def _comparison_metadata(event_source: str, brain_area: str) -> dict[str, object]:
    """Provenance and interpretation metadata saved with comparison fits."""
    effective_area = _effective_brain_area(event_source, brain_area)
    return {
        "event_source": event_source,
        "brain_area": effective_area,
        "requested_brain_area": brain_area,
        "same_identified_events": False,
        "comparison_kind": "event-definition head-to-head",
        "event_definition_note": (
            "Different event sources intentionally define different event sets; "
            "head-to-head means the same analysis suite on each field-standard "
            "event definition, not matched event identities across sources."
        ),
        "patch_models": tuple(_patch_models_for_source(event_source)),
        "patch_switch_models": tuple(_patch_switch_models_for_source(event_source)),
        "rate_models": tuple(_rate_models_for_source(event_source)),
        "choice_models": tuple(_choice_models_for_source(event_source)),
        "choice_swr_split_models": tuple(
            _choice_swr_split_models_for_source(event_source)
        ),
    }


def _comparison_output_path(
    out_dir: Path, event_source: str, brain_area: str
) -> Path:
    """Comparison pickle path keyed by the effective data source."""
    effective_area = _effective_brain_area(event_source, brain_area)
    return out_dir / f"{event_source}_{effective_area}_comparison_fits.pkl"


def _load_tables(event_source: str, brain_area: str) -> dict[str, pd.DataFrame]:
    """Load event / event_arm / stop tables for an event source."""
    sub, stem = _SOURCES[event_source]
    effective_area = _effective_brain_area(event_source, brain_area)
    if stem is None:
        suffix = ""  # non-local replay content is HPC clusterless only
        if brain_area != effective_area:
            logger.warning(
                "event-source 'non_local' is HPC-only; ignoring --brain-area %s",
                brain_area,
            )
    else:
        suffix = f"_{stem}_{effective_area}"
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


def _predictor_union(models) -> tuple[str, ...]:
    """Sorted union of predictors used by a model block."""
    return tuple(
        sorted(
            {predictor for spec in models.values() for predictor in spec.predictors}
        )
    )


def _as_python_scalar(value):
    """Convert numpy scalar IDs to stable Python scalars for provenance."""
    if hasattr(value, "item"):
        return value.item()
    return value


def _sorted_unit_ids(values) -> tuple:
    """Stable sorted tuple of non-null unit IDs."""
    ids = [_as_python_scalar(v) for v in pd.Series(values).dropna().unique()]
    return tuple(sorted(ids, key=repr))


def _hash_unit_ids(ids: tuple) -> str:
    """Hash sorted included IDs so cohorts can be compared cheaply."""
    digest = hashlib.sha256()
    for unit_id in ids:
        digest.update(str(unit_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _freeze_common_cohort(
    group: pd.DataFrame,
    *,
    keys,
    analysis: str,
    models,
    unit_col: str,
    required_base: list[str],
    min_units: int,
    response_col: str | None = None,
    drop_zero_response: bool = False,
    offset_col: str | None = None,
    choice_response_col: str | None = None,
    selection_filter: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Return one complete-case cohort shared by every model in a block."""
    session = dict(zip(SESS, keys, strict=True))
    predictors = _predictor_union(models)
    required_cols = list(dict.fromkeys([unit_col, *required_base, *predictors]))
    missing_cols = [col for col in required_cols if col not in group.columns]
    ids_before = (
        _sorted_unit_ids(group[unit_col]) if unit_col in group.columns else ()
    )

    provenance = {
        **session,
        "analysis": analysis,
        "models": tuple(models),
        "unit_col": unit_col,
        "predictors": predictors,
        "required_columns": tuple(required_cols),
        "selection_filter": selection_filter,
        "n_rows_before": len(group),
        "n_units_before": len(ids_before),
        "n_rows_after": 0,
        "n_units_after": 0,
        "n_missing_units": 0,
        "n_nonpositive_offset_units": 0,
        "n_zero_response_units": 0,
        "n_invalid_choice_units": 0,
        "included_ids": (),
        "included_ids_hash": _hash_unit_ids(()),
        "id_hash_algorithm": "sha256_sorted_unit_ids_v1",
        "status": "ok",
        "failed_models": (),
        "missing_columns": tuple(missing_cols),
    }
    if missing_cols:
        provenance["status"] = "missing_columns"
        return group.iloc[0:0].copy(), provenance

    cohort = group.copy()

    has_nan = cohort[required_cols].isna().any(axis=1)
    missing_ids = _sorted_unit_ids(cohort.loc[has_nan, unit_col])
    if missing_ids:
        cohort = cohort[~cohort[unit_col].isin(missing_ids)].copy()
    provenance["n_missing_units"] = len(missing_ids)

    if offset_col is not None:
        nonpositive = cohort[offset_col] <= 0
        nonpositive_ids = _sorted_unit_ids(cohort.loc[nonpositive, unit_col])
        if nonpositive_ids:
            cohort = cohort[~cohort[unit_col].isin(nonpositive_ids)].copy()
        provenance["n_nonpositive_offset_units"] = len(nonpositive_ids)

    if drop_zero_response:
        if response_col is None:
            raise ValueError("response_col is required when drop_zero_response=True")
        response_sum = cohort.groupby(unit_col)[response_col].sum()
        zero_ids = _sorted_unit_ids(response_sum[response_sum <= 0].index)
        if zero_ids:
            cohort = cohort[~cohort[unit_col].isin(zero_ids)].copy()
        provenance["n_zero_response_units"] = len(zero_ids)

    if choice_response_col is not None and len(cohort):
        chosen_per_unit = cohort.groupby(unit_col)[choice_response_col].sum()
        invalid_ids = _sorted_unit_ids(chosen_per_unit[chosen_per_unit != 1].index)
        if invalid_ids:
            cohort = cohort[~cohort[unit_col].isin(invalid_ids)].copy()
        provenance["n_invalid_choice_units"] = len(invalid_ids)

    cohort = cohort.reset_index(drop=True)
    ids_after = _sorted_unit_ids(cohort[unit_col])
    provenance["n_rows_after"] = len(cohort)
    provenance["n_units_after"] = len(ids_after)
    provenance["included_ids"] = ids_after
    provenance["included_ids_hash"] = _hash_unit_ids(ids_after)
    if len(ids_after) < min_units:
        provenance["status"] = "too_few_units"

    return cohort, provenance


def _fit_per_session_common_cohort(
    table: pd.DataFrame,
    models,
    fit_fn,
    *,
    analysis: str,
    unit_col: str,
    required_base: list[str],
    min_units: int,
    response_col: str | None = None,
    drop_zero_response: bool = False,
    offset_col: str | None = None,
    choice_response_col: str | None = None,
    selection_filter: str | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Fit all models on one frozen per-session cohort and record provenance."""
    out = {name: [] for name in models}
    provenance_rows = []
    if len(table) == 0:
        return {name: pd.DataFrame() for name in models}, pd.DataFrame()

    for keys, group in table.groupby(SESS):
        cohort, provenance = _freeze_common_cohort(
            group,
            keys=keys,
            analysis=analysis,
            models=models,
            unit_col=unit_col,
            required_base=required_base,
            min_units=min_units,
            response_col=response_col,
            drop_zero_response=drop_zero_response,
            offset_col=offset_col,
            choice_response_col=choice_response_col,
            selection_filter=selection_filter,
        )
        if provenance["status"] != "ok":
            provenance_rows.append(provenance)
            continue

        session_results = {}
        failed_models = []
        for name, spec in models.items():
            fit = fit_fn(cohort, spec)
            if fit is None:
                failed_models.append(name)
            else:
                session_results[name] = fit

        if failed_models:
            provenance["status"] = "fit_failed"
            provenance["failed_models"] = tuple(failed_models)
            provenance_rows.append(provenance)
            continue

        provenance["status"] = "fit"
        provenance_rows.append(provenance)
        session = dict(zip(SESS, keys, strict=True))
        cohort_fields = {
            "cohort_n_units": provenance["n_units_after"],
            "cohort_n_rows": provenance["n_rows_after"],
            "cohort_included_ids_hash": provenance["included_ids_hash"],
        }
        for name, fit in session_results.items():
            out[name].append({**session, **cohort_fields, **fit})

    return (
        {name: pd.DataFrame(rows) for name, rows in out.items()},
        pd.DataFrame(provenance_rows),
    )


def _fit_patch_conditional(g: pd.DataFrame, spec) -> dict | None:
    """Patch conditional Poisson on one session; skip if < 10 events."""
    if g["event_id"].nunique() < 10:
        return None
    return fit_conditional_poisson(
        g, list(spec.predictors), response_col="patch_count_hard",
        event_col="event_id", n_categories=3,
    )


def fit_patch_hierarchy(
    patch_table: pd.DataFrame, event_source: str
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Fit the source-appropriate patch hierarchy per session."""
    return _fit_per_session_common_cohort(
        patch_table,
        _patch_models_for_source(event_source),
        _fit_patch_conditional,
        analysis="patch_hierarchy",
        unit_col="event_id",
        required_base=["patch_count_hard"],
        response_col="patch_count_hard",
        drop_zero_response=True,
        min_units=10,
    )


def fit_patch_switch(
    patch_table: pd.DataFrame, event_arm_table: pd.DataFrame, event_source: str
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Fit source-appropriate switch patch models on switch events only."""
    sw = event_arm_table.groupby("event_id")["is_after_switch"].first()
    switch_patch = patch_table.merge(
        sw.rename("is_after_switch"), left_on="event_id", right_index=True, how="left"
    )
    switch_patch = switch_patch[switch_patch["is_after_switch"] == 1]
    return _fit_per_session_common_cohort(
        switch_patch,
        _patch_switch_models_for_source(event_source),
        _fit_patch_conditional,
        analysis="patch_switch",
        unit_col="event_id",
        required_base=["patch_count_hard"],
        response_col="patch_count_hard",
        drop_zero_response=True,
        min_units=10,
        selection_filter="is_after_switch == 1",
    )


def fit_per_stop_rate_hierarchy(
    stop_table: pd.DataFrame, event_source: str
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Fit the source-appropriate Poisson rate hierarchy per session."""
    return _fit_per_session_common_cohort(
        stop_table,
        _rate_models_for_source(event_source),
        lambda g, spec: fit_per_stop_rate_model(g, list(spec.predictors), min_stops=20),
        analysis="rate_hierarchy",
        unit_col="stop_id",
        required_base=["n_events", "immobility_time"],
        response_col="n_events",
        offset_col="immobility_time",
        min_units=20,
    )


def _fit_choice_block(
    choice: pd.DataFrame, models, analysis: str
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Fit a choice model block on one common stop cohort."""
    return _fit_per_session_common_cohort(
        choice,
        models,
        lambda g, spec: fit_choice_model(g, list(spec.predictors)),
        analysis=analysis,
        unit_col="stop_id",
        required_base=["chosen"],
        choice_response_col="chosen",
        min_units=10,
    )


def fit_choice_blocks(
    tables: dict[str, pd.DataFrame], event_source: str
) -> tuple[dict[str, dict[str, pd.DataFrame]], dict[str, pd.DataFrame]]:
    """Fit primary and optional SWR-split choice blocks."""
    choice = build_choice_dataset(
        tables["event_table"], tables["event_arm_table"], tables["stop_table"]
    )
    blocks: dict[str, dict[str, pd.DataFrame]] = {}
    provenance: dict[str, pd.DataFrame] = {}

    primary, primary_prov = _fit_choice_block(
        choice, _choice_models_for_source(event_source), "choice"
    )
    blocks["choice"] = primary
    provenance["choice"] = primary_prov

    split_models = _choice_swr_split_models_for_source(event_source)
    if split_models:
        split, split_prov = _fit_choice_block(
            choice, split_models, "choice_swr_split"
        )
        blocks["choice_swr_split"] = split
        provenance["choice_swr_split"] = split_prov

    return blocks, provenance


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

    patch_hierarchy, patch_hierarchy_prov = fit_patch_hierarchy(patch_table, src)
    patch_switch, patch_switch_prov = fit_patch_switch(
        patch_table, tables["event_arm_table"], src
    )
    rate_hierarchy, rate_hierarchy_prov = fit_per_stop_rate_hierarchy(
        tables["stop_table"], src
    )
    choice_blocks, choice_provenance = fit_choice_blocks(tables, src)

    cohort_provenance = {
        "patch_hierarchy": patch_hierarchy_prov,
        "patch_switch": patch_switch_prov,
        "rate_hierarchy": rate_hierarchy_prov,
        **choice_provenance,
    }
    results = {
        "metadata": _comparison_metadata(src, args.brain_area),
        "cohort_provenance": cohort_provenance,
        "patch_hierarchy": patch_hierarchy,
        "patch_switch": patch_switch,
        "rate_hierarchy": rate_hierarchy,
        **choice_blocks,
        "switch_aligned": run_switch_aligned(tables),
    }
    for analysis, res in results.items():
        if analysis in {"metadata", "cohort_provenance"}:
            continue
        n = (
            len(res)
            if isinstance(res, pd.DataFrame)
            else {k: len(v) for k, v in res.items()}
        )
        logger.info("%s [%s]: %s", analysis, src, n)

    # Include the brain area: ContFrag sources are per-brain-area, so
    # SWR/HPC and SWR/mPFC must not overwrite each other. (non_local is
    # HPC-only; its name is then non_local_HPC.)
    out_path = _comparison_output_path(out_dir, src, args.brain_area)
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    logger.info("saved %s", out_path)


if __name__ == "__main__":
    main()
