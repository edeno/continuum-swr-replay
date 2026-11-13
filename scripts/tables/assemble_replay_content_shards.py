#!/usr/bin/env python
"""Assemble disjoint Replay table shards into one canonical candidate bundle.

Per-animal shards are a computational optimization only. The assembly
(:func:`assemble_replay_content_shards`) restores the monolithic
build: it verifies disjoint session sets, recomputes the assembly-level stop
counts and the patch table, discards the shards' standardized columns, and
fits one pooled leaf-arm standardization reference over the complete corpus
before writing the candidate tables. ``--stop-only`` joins the shards of a
stop-only build for the qvals bootstrap
(:func:`assemble_replay_stop_shards`).
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.behavior.covariates import load_contingency
from continuum_swr_replay.analysis.event_tables._common import (
    EVENT_ARM_STANDARDIZED_COLS,
)
from continuum_swr_replay.analysis.event_tables.prepare_data import (
    assemble_canonical_tables,
    standardize_event_arm,
)
from continuum_swr_replay.analysis.event_tables.table_contracts import (
    validate_event_arm_table,
    validate_event_patch_table,
)
from continuum_swr_replay.analysis.events.analysed_time import (
    publish_swr_table,
    stamp_swr_source,
)
from continuum_swr_replay.data_loaders.epoch_register import analysed_epoch_ids
from continuum_swr_replay.paths import (
    BASENAME_TO_TABLE,
    STANDARDIZATION_REFERENCE,
    table_path,
)
from continuum_swr_replay.tables.covariate_registry import stamp_covariate_schema

# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.tables._table_runner import (  # noqa: E402
    publish_replay_tables,
    read_disjoint_shards,
)

_SHARD_TABLES = (
    "event_table",
    "event_arm_table",
    "stop_table",
    "event_sequence_table",
)


def assemble_replay_content_shards(
    shard_dirs: Sequence[Path],
    *,
    output_dir: Path,
    contingency_dir: Path,
    expected_sessions: int,
) -> dict[str, pd.DataFrame]:
    """Join per-animal shards of the replay tables into one published bundle.

    The bundle is the one ``prepare_replay_content_data.py`` builds over every
    session at once: the shards' tables are concatenated in shard order, and the
    assembly-level columns (the non-local SWR split, the event-patch table) and
    the leaf-arm standardization of the event-arm table are computed over the
    whole corpus. Each shard's own ``_z`` columns, standardized over its animal
    alone, are dropped first.

    Parameters
    ----------
    shard_dirs : sequence of Path
        The shard folders, each written by ``prepare_replay_content_data.py``
        for one animal.
    output_dir : Path
        Where the five tables and the standardization reference are published.
    contingency_dir : Path
        The folder of the ``qvals_table`` the event-arm value block must match.
    expected_sessions : int
        The number of sessions the shards must hold together.

    Returns
    -------
    dict of str to pd.DataFrame
        The published tables, keyed by basename.

    Raises
    ------
    ValueError
        If the shards overlap or miss sessions, a shard or the bundle breaks
        stop/event count conservation, an ``event_id`` or ``stop_id`` repeats, an
        ``event_sequence_table`` row repeats or names an event the bundle lacks,
        or the value block disagrees with the ``qvals_table``.
    """
    shards = read_disjoint_shards(
        shard_dirs,
        {name: name for name in _SHARD_TABLES},
        what="Replay",
        expected_sessions=expected_sessions,
    )
    for shard_dir, frames in zip(shard_dirs, shards, strict=True):
        validate_event_arm_table(
            frames["event_arm_table"], event_table=frames["event_table"]
        )
        if int(frames["stop_table"]["n_selected_events"].sum()) != len(
            frames["event_table"]
        ):
            raise ValueError(f"{shard_dir} violates stop/event count conservation")

    standardized_columns = [f"{column}_z" for column in EVENT_ARM_STANDARDIZED_COLS]
    (
        event_table,
        event_arm_table,
        event_patch_table,
        stop_table,
        event_sequence_table,
    ) = assemble_canonical_tables(
        [frames["event_table"] for frames in shards],
        [
            frames["event_arm_table"].drop(columns=standardized_columns, errors="raise")
            for frames in shards
        ],
        [frames["stop_table"] for frames in shards],
        [frames["event_sequence_table"] for frames in shards],
    )
    if event_table["event_id"].duplicated().any():
        raise ValueError("assembled event_table contains duplicate event_id values")
    if stop_table["stop_id"].duplicated().any():
        raise ValueError("assembled stop_table contains duplicate stop_id values")
    # event_sequence_table has many rows per event_id, so it needs a referential
    # check rather than a uniqueness one: every sequence event_id must resolve to
    # an assembled event (no orphan from a shard mismatch, the silent event_id
    # mis-join), and no whole row may be duplicated (a shard double-count).
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
    validate_event_arm_table(event_arm_table, event_table=event_table)
    validate_event_patch_table(event_patch_table, event_table=event_table)
    if int(stop_table["n_selected_events"].sum()) != len(event_table):
        raise ValueError("assembled stop/event count conservation failed")

    contingency = load_contingency(table_path("qvals_table", contingency_dir))
    output_dir.mkdir(parents=True, exist_ok=True)
    event_arm_table = standardize_event_arm(
        event_arm_table, contingency, output_dir / STANDARDIZATION_REFERENCE
    )
    tables = {
        "event_table": event_table,
        "event_arm_table": event_arm_table,
        "event_patch_table": event_patch_table,
        "stop_table": stop_table,
        "event_sequence_table": event_sequence_table,
    }
    # The concatenated shards carry the shards' stamps, which would let the
    # publish skip its checks: stamp (and so check) the bundle itself.
    for name, frame in tables.items():
        stamp_covariate_schema(frame, table=BASENAME_TO_TABLE[name])
    publish_replay_tables(
        tuple((name, BASENAME_TO_TABLE[name], frame) for name, frame in tables.items()),
        output_dir,
    )
    return tables


def assemble_replay_stop_shards(
    shard_dirs: Sequence[Path], *, output_dir: Path, expected_sessions: int
) -> pd.DataFrame:
    """Join per-animal shards of the stop-only table and publish it.

    The stop-only table (``prepare_replay_content_data.py --stop-only``) is the
    input of the qvals bootstrap; this joins the shards of a per-animal stop-only
    build.

    Parameters
    ----------
    shard_dirs : sequence of Path
        The shard folders, each holding one ``stop_table``.
    output_dir : Path
        Where the joined ``stop_table`` is published.
    expected_sessions : int
        The number of sessions the shards must hold together.

    Returns
    -------
    pd.DataFrame
        The published stop table.

    Raises
    ------
    ValueError
        If the shards overlap or miss sessions, or a ``stop_id`` is missing or
        repeats.
    """
    shards = read_disjoint_shards(
        shard_dirs,
        {"stop_table": "stop_table"},
        what="stop",
        expected_sessions=expected_sessions,
    )
    stop_table = pd.concat(
        [frames["stop_table"] for frames in shards], ignore_index=True
    )
    if stop_table["stop_id"].isna().any() or stop_table["stop_id"].duplicated().any():
        raise ValueError("assembled stop table requires unique nonmissing stop_id")
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp_swr_source(stop_table)
    path = publish_swr_table(
        stop_table,
        output_dir,
        table="stop",
        basename="stop_table",
    )
    logging.info("saved stop_table -> %s (%d rows)", path, len(stop_table))
    return stop_table


def main(argv: Sequence[str] | None = None) -> None:
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
    args = parser.parse_args(argv)
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
