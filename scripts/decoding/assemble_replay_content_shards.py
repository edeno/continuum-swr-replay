#!/usr/bin/env python
"""Assemble disjoint Replay table shards into one canonical candidate bundle.

Per-animal shards are a computational optimization only. This reducer restores
the exact monolithic scientific contract: it verifies disjoint session sets,
recomputes assembly-level stop counts and the patch table, discards shard-level
standardized columns, and fits one pooled leaf-arm standardization reference
over the complete corpus before writing the candidate tables.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.events.swr_source import (
    require_swr_source,
    stamp_swr_source,
)
from continuum_swr_replay.analysis.replay._tables_common import (
    EVENT_ARM_STANDARDIZED_COLS,
)
from continuum_swr_replay.analysis.replay.covariates import load_contingency
from continuum_swr_replay.analysis.replay.prepare_data import (
    _assert_value_block_matches_source,
)
from continuum_swr_replay.analysis.replay.table_contracts import (
    validate_event_arm_table,
    validate_event_patch_table,
)
from continuum_swr_replay.analysis.replay.tables_arm_patch import (
    build_event_patch_table,
)
from continuum_swr_replay.analysis.replay.tables_stop import add_nonlocal_swr_split
from continuum_swr_replay.data_loaders.constants import WELL_ARMS
from continuum_swr_replay.data_loaders.epoch_register import analysed_epoch_ids
from continuum_swr_replay.tables.covariate_registry import (
    stamp_covariate_schema,
)
from continuum_swr_replay.tables.io import publish_canonical, read_table
from continuum_swr_replay.tables.parity import BASENAME_TO_TABLE
from continuum_swr_replay.tables.session_keys import (
    session_key_set as _session_keys,
)
from continuum_swr_replay.tables.standardization import (
    assert_standardized_present,
    attach_standardized,
    save_reference,
)

LOGGER = logging.getLogger(__name__)
INPUT_TABLES = (
    "event_table",
    "event_arm_table",
    "stop_table",
    "event_sequence_table",
)


def assemble_replay_content_shards(
    shard_dirs: list[Path],
    *,
    output_dir: Path,
    contingency_dir: Path,
    expected_sessions: int,
) -> dict[str, pd.DataFrame]:
    """Validate, combine, and publish one globally standardized Replay bundle."""
    if not shard_dirs:
        raise ValueError("at least one shard directory is required")

    pieces: dict[str, list[pd.DataFrame]] = {name: [] for name in INPUT_TABLES}
    observed_sessions: set[tuple[object, ...]] = set()
    for shard_dir in shard_dirs:
        frames = {name: read_table(shard_dir / name) for name in INPUT_TABLES}
        # Every shard must come from the SWR source the bundle is stamped with.
        for name, frame in frames.items():
            require_swr_source(frame, what=f"{shard_dir / name}")
        shard_sessions = _session_keys(frames["stop_table"])
        overlap = observed_sessions & shard_sessions
        if overlap:
            raise ValueError(
                f"Replay shards overlap on {len(overlap)} session(s), e.g. "
                f"{next(iter(overlap))}"
            )
        observed_sessions |= shard_sessions
        validate_event_arm_table(
            frames["event_arm_table"], event_table=frames["event_table"]
        )
        if int(frames["stop_table"]["n_selected_events"].sum()) != len(
            frames["event_table"]
        ):
            raise ValueError(f"{shard_dir} violates stop/event count conservation")
        for name, frame in frames.items():
            pieces[name].append(frame)

    if len(observed_sessions) != expected_sessions:
        raise ValueError(
            f"assembled stop-table scope has {len(observed_sessions)} sessions; "
            f"expected {expected_sessions}"
        )

    event_table = pd.concat(pieces["event_table"], ignore_index=True)
    event_arm_table = pd.concat(pieces["event_arm_table"], ignore_index=True)
    stop_table = pd.concat(pieces["stop_table"], ignore_index=True)
    event_sequence_table = pd.concat(pieces["event_sequence_table"], ignore_index=True)
    if event_table["event_id"].duplicated().any():
        raise ValueError("assembled event_table contains duplicate event_id values")
    if stop_table["stop_id"].duplicated().any():
        raise ValueError("assembled stop_table contains duplicate stop_id values")
    # event_sequence_table has many rows per event_id, so it needs a referential
    # check rather than a uniqueness one: every sequence event_id must resolve to
    # an assembled event (no orphan from a shard mismatch -- the silent
    # event_id mis-join CLAUDE.md warns about), and no whole row may be
    # duplicated (a shard double-count). This table is otherwise absent from
    # every integrity gate.
    sequence_orphans = set(event_sequence_table["event_id"].unique()) - set(
        event_table["event_id"]
    )
    if sequence_orphans:
        raise ValueError(
            f"event_sequence_table references {len(sequence_orphans)} event_id(s) "
            "absent from the assembled event_table"
        )
    if event_sequence_table.duplicated().any():
        raise ValueError(
            "assembled event_sequence_table contains fully duplicated rows "
            "(shard double-count)"
        )

    # Per-shard z-scores use the wrong reference support. Remove them and fit
    # exactly one reference on the complete six-leaf-arm corpus.
    standardized_columns = [f"{column}_z" for column in EVENT_ARM_STANDARDIZED_COLS]
    event_arm_table = event_arm_table.drop(columns=standardized_columns, errors="raise")
    validate_event_arm_table(event_arm_table, event_table=event_table)
    stop_table = add_nonlocal_swr_split(stop_table, event_table)
    event_patch_table = build_event_patch_table(event_table, event_arm_table)
    validate_event_patch_table(event_patch_table, event_table=event_table)

    if int(stop_table["n_selected_events"].sum()) != len(event_table):
        raise ValueError("assembled stop/event count conservation failed")
    leaf_support = event_arm_table["candidate_arm"].isin(WELL_ARMS)
    event_arm_table, reference = attach_standardized(
        event_arm_table,
        EVENT_ARM_STANDARDIZED_COLS,
        support_mask=leaf_support,
        support_label="leaf_arms",
    )
    assert_standardized_present(
        event_arm_table,
        EVENT_ARM_STANDARDIZED_COLS,
        support_mask=leaf_support,
    )

    tables = {
        "event_table": stamp_covariate_schema(event_table, table="event"),
        "event_arm_table": stamp_covariate_schema(event_arm_table, table="event_arm"),
        "event_patch_table": stamp_covariate_schema(
            event_patch_table, table="event_patch"
        ),
        "stop_table": stamp_covariate_schema(stop_table, table="stop"),
        "event_sequence_table": stamp_covariate_schema(
            event_sequence_table, table="event_sequence"
        ),
    }
    for frame in tables.values():
        stamp_swr_source(frame)
    contingency = load_contingency(
        contingency_dir / "models_per_bin_rate" / "qvals_table.pkl"
    )
    _assert_value_block_matches_source(tables["event_arm_table"], contingency)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_reference(reference, output_dir / "standardization_reference.json")
    for name, frame in tables.items():
        # publish_canonical is the one write door; it re-stamps (idempotent) and
        # writes atomically so an unstamped canonical table cannot be persisted.
        path = publish_canonical(
            frame,
            output_dir,
            table=BASENAME_TO_TABLE[name],
            basename=name,
            require_swr_source=require_swr_source,
        )
        LOGGER.info("saved %s -> %s (%d rows)", name, path, len(frame))
    return tables


def assemble_replay_stop_shards(
    shard_dirs: list[Path], *, output_dir: Path, expected_sessions: int
) -> pd.DataFrame:
    """Combine disjoint stop-only shards for the qvals bootstrap."""
    pieces: list[pd.DataFrame] = []
    observed_sessions: set[tuple[object, ...]] = set()
    for shard_dir in shard_dirs:
        frame = read_table(shard_dir / "stop_table")
        require_swr_source(frame, what=f"{shard_dir / 'stop_table'}")
        shard_sessions = _session_keys(frame)
        overlap = observed_sessions & shard_sessions
        if overlap:
            raise ValueError(
                f"stop shards overlap on {len(overlap)} session(s), e.g. "
                f"{next(iter(overlap))}"
            )
        observed_sessions |= shard_sessions
        pieces.append(frame)
    if len(observed_sessions) != expected_sessions:
        raise ValueError(
            f"assembled stop-table scope has {len(observed_sessions)} sessions; "
            f"expected {expected_sessions}"
        )
    stop_table = pd.concat(pieces, ignore_index=True)
    if stop_table["stop_id"].isna().any() or stop_table["stop_id"].duplicated().any():
        raise ValueError("assembled stop table requires unique nonmissing stop_id")
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp_swr_source(stop_table)
    path = publish_canonical(
        stop_table,
        output_dir,
        table="stop",
        basename="stop_table",
        require_swr_source=require_swr_source,
    )
    LOGGER.info("saved stop_table -> %s (%d rows)", path, len(stop_table))
    return stop_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--contingency-dir", required=True, type=Path)
    parser.add_argument(
        "--expected-sessions",
        type=int,
        default=len(analysed_epoch_ids()),
        help="Epochs (sessions) the tables must hold; default: the epoch register's "
        "analysed epochs (HPC cohort).",
    )
    parser.add_argument(
        "--stop-only",
        action="store_true",
        help="Assemble only stop_table shards for the qvals bootstrap.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    shard_dirs = [path.resolve() for path in args.shard_dirs]
    if args.stop_only:
        assemble_replay_stop_shards(
            shard_dirs,
            output_dir=args.output_dir.resolve(),
            expected_sessions=args.expected_sessions,
        )
    else:
        assemble_replay_content_shards(
            shard_dirs,
            output_dir=args.output_dir.resolve(),
            contingency_dir=args.contingency_dir.resolve(),
            expected_sessions=args.expected_sessions,
        )


if __name__ == "__main__":
    main()
