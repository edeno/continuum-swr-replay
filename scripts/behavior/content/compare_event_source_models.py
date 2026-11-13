#!/usr/bin/env python
"""Run the replay-content analysis suite on one event source and save fits.

Mirrors the non-local pipeline so non-local / ContFrag-SWR / ContFrag-MUA
can be compared head-to-head. For a given event source it fits, per
session:

- Patch hierarchy P0-P3 and Pv for non-local events; P0-P2 for source-restricted
  SWR/MUA events where SWR interaction terms are not identifiable
- Rate hierarchy R0-R1 (the ``is_swr_stop`` term was dropped: it is
  tautological with the per-stop non-local/SWR event count)
- MODEL_PATCH_SWITCH PS0-PS3 for non-local events; PS0-PS2 for
  source-restricted SWR/MUA events
- Choice-prediction conditional logit (replay_fraction vs behavioral baseline)
- Switch-aligned replay content (frac on/off current patch by lag)

The arm hierarchy (M0-M6 / M0-M5) is fit by ``fit_replay_content_models.py``
and is not repeated here.

Usage::

    uv run python scripts/behavior/content/compare_event_source_models.py --event-source non_local
    uv run python scripts/behavior/content/compare_event_source_models.py --event-source SWR
    uv run python scripts/behavior/content/compare_event_source_models.py --event-source MUA
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.behavior import (
    MODEL_PER_STOP_RATE,
    build_event_patch_table,
    build_switch_aligned_table,
)
from continuum_swr_replay.analysis.behavior.choice.choice_prediction import (
    CHOICE_MODELS,
    choice_swr_split_models_for_source,
    fit_choice_blocks,
)
from continuum_swr_replay.analysis.behavior.cohort_audit import (
    records_from_common_cohort_provenance,
    write_cohort_audit,
)
from continuum_swr_replay.analysis.behavior.content.session_fits import (
    fit_patch_hierarchy,
    fit_patch_switch,
    patch_models_for_source,
    patch_switch_models_for_source,
)
from continuum_swr_replay.analysis.behavior.rate.per_stop_rate_model import (
    fit_event_source_rate_hierarchy,
)
from continuum_swr_replay.analysis.events.analysed_time import require_swr_source
from continuum_swr_replay.analysis.events.swr_source import swr_source_name
from continuum_swr_replay.paths import (
    EVENT_SOURCE_COMPARISON_DIR,
    contfrag_replay_content_dir,
    event_source_comparison_fits_name,
    replay_content_dir,
)
from continuum_swr_replay.runs.fit_provenance import (
    record_inputs,
    write_fit_provenance,
)
from continuum_swr_replay.tables.covariate_registry import (
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger("compare_event_source")
# Silence the per-session rank-deficiency / convergence chatter.
for noisy in (
    "continuum_swr_replay.analysis.behavior.rate.per_stop_rate_model",
    "continuum_swr_replay.analysis.behavior.fitting",
    "continuum_swr_replay.analysis.behavior.choice.choice_prediction",
):
    logging.getLogger(noisy).setLevel(logging.CRITICAL)


# event source -> (input directory's accessor, table-name stem). ``None`` stem means
# the non-local replay-content tables (HPC clusterless only, no suffix).
# The ContFrag sources are per-brain-area; the brain area is appended at
# load time so mPFC/OFC outputs aren't silently read as HPC.
_SOURCES = {
    "non_local": (replay_content_dir, None),
    "SWR": (contfrag_replay_content_dir, "swr"),
    "MUA": (contfrag_replay_content_dir, "mua"),
}


def _effective_brain_area(event_source: str, brain_area: str) -> str:
    """Return the data brain area actually used for this event source."""
    if event_source == "non_local":
        return "HPC"
    return brain_area


# Rate hierarchy, the same for every event source. The ``is_swr_stop`` term
# (formerly R3) was removed from ``MODEL_PER_STOP_RATE``: it is tautological
# with a per-stop event count that overlaps SWRs by construction, so it is
# dropped for every source, not just SWR.
_RATE_MODELS = MODEL_PER_STOP_RATE


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
        "patch_models": tuple(patch_models_for_source(event_source)),
        "patch_switch_models": tuple(patch_switch_models_for_source(event_source)),
        "rate_models": tuple(_RATE_MODELS),
        "choice_models": tuple(CHOICE_MODELS),
        "choice_swr_split_models": tuple(
            choice_swr_split_models_for_source(event_source)
        ),
    }


def _comparison_output_path(out_dir: Path, event_source: str, brain_area: str) -> Path:
    """Comparison pickle path keyed by the effective data source."""
    effective_area = _effective_brain_area(event_source, brain_area)
    return out_dir / event_source_comparison_fits_name(event_source, effective_area)


def _table_paths(event_source: str, brain_area: str) -> dict[str, Path]:
    """The event / event_arm / stop table files of an event source."""
    input_dir, stem = _SOURCES[event_source]
    d = input_dir()
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
    return {
        short: d / f"{short}{suffix}.parquet"
        for short in ("event_table", "event_arm_table", "stop_table")
    }


def _load_tables(event_source: str, brain_area: str) -> dict[str, pd.DataFrame]:
    """Load event / event_arm / stop tables for an event source."""
    tables = {}
    for short, path in _table_paths(event_source, brain_area).items():
        tables[short] = read_table(path)
        require_covariate_schema(
            tables[short],
            table={
                "event_table": "event",
                "event_arm_table": "event_arm",
                "stop_table": "stop",
            }[short],
        )
        require_swr_source(tables[short], what=str(path))
        logger.info("loaded %s: %d rows", path.name, len(tables[short]))
    return tables


def run_switch_aligned(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Switch-aligned replay content by lag."""
    return build_switch_aligned_table(
        tables["stop_table"],
        tables["event_table"],
        tables["event_arm_table"],
        max_lag=10,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-source", choices=list(_SOURCES), default="non_local")
    parser.add_argument(
        "--brain-area",
        default="HPC",
        help="Brain area for ContFrag sources (SWR/MUA); ignored for "
        "non_local (HPC-only). Default: HPC.",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="Smoke: fit only the first N sessions; writes *_smoke outputs.",
    )
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(argv)
    src = args.event_source

    tables = _load_tables(src, args.brain_area)
    # Hash the inputs now, while the fit reads them; the sidecar is written
    # beside the output so Part 3 can check it against the current tables.
    fit_inputs = record_inputs(_table_paths(src, args.brain_area))
    if args.max_sessions is not None:
        keep = (
            tables["stop_table"][list(SESSION_COLUMNS)]
            .drop_duplicates()
            .head(args.max_sessions)
        )
        tables = {
            name: frame.merge(keep, on=list(SESSION_COLUMNS))
            for name, frame in tables.items()
        }
    out_dir = replay_content_dir() / EVENT_SOURCE_COMPARISON_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Built once — both patch fits consume it.
    patch_table = build_event_patch_table(
        tables["event_table"], tables["event_arm_table"]
    )

    patch_hierarchy, patch_hierarchy_prov = fit_patch_hierarchy(patch_table, src)
    patch_switch, patch_switch_prov = fit_patch_switch(
        patch_table, tables["event_arm_table"], src
    )
    rate_hierarchy, rate_hierarchy_prov = fit_event_source_rate_hierarchy(
        tables["stop_table"], src
    )
    choice_blocks, choice_provenance = fit_choice_blocks(
        tables["event_table"], tables["event_arm_table"], tables["stop_table"], src
    )

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
    if args.max_sessions is not None:
        out_path = out_path.with_name(out_path.stem + "_smoke.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    write_fit_provenance(
        out_path,
        fit_inputs,
        "conditional_logit_and_poisson",
        [__file__, *argv],
        extra={
            "event_source": src,
            "brain_area": _effective_brain_area(src, args.brain_area),
            "swr_source": swr_source_name(),
        },
    )
    logger.info("saved %s", out_path)

    # Standardized cohort audit for every common-cohort block (patch / rate / choice),
    # mapped from the native common_cohort.freeze_session_cohort provenance. This puts the choice
    # cohort on the same reconciling footing as the rate/content ladders.
    audit_records = [
        rec
        for label, prov in cohort_provenance.items()
        if (rec := records_from_common_cohort_provenance(prov, label, "all"))
        is not None
    ]
    if audit_records:
        audit_path = out_path.with_name(out_path.stem + "_cohort_audit.csv")
        write_cohort_audit(audit_records, audit_path)
        logger.info("saved cohort audit %s", audit_path)


if __name__ == "__main__":
    main()
