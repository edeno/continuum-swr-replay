#!/usr/bin/env python
"""Assemble disjoint per-animal ContFrag shards into one canonical bundle."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from prepare_contfrag_replay_content_data import contfrag_table_basenames

from continuum_swr_replay.analysis.hippocampus.swr_source import (
    require_swr_source,
    stamp_swr_source,
)
from continuum_swr_replay.analysis.replay import build_event_patch_table
from continuum_swr_replay.analysis.replay.covariate_registry import (
    stamp_covariate_schema,
)
from continuum_swr_replay.analysis.replay.table_contracts import (
    validate_event_arm_table,
    validate_event_patch_table,
)
from continuum_swr_replay.session_keys import (
    session_key_set as _session_keys,
)
from continuum_swr_replay.table_io import publish_canonical, read_table
from continuum_swr_replay.table_parity import BASENAME_TO_TABLE

LOGGER = logging.getLogger(__name__)




def assemble_contfrag_shards(
    shard_dirs: list[Path],
    *,
    output_dir: Path,
    event_source: str,
    brain_area: str,
    expected_sessions: int,
) -> dict[str, pd.DataFrame]:
    """Validate and concatenate disjoint shards, then rebuild patch rows."""
    if not shard_dirs:
        raise ValueError("at least one shard directory is required")
    basenames = contfrag_table_basenames(event_source, brain_area)
    input_names = ("event_table", "event_arm_table", "stop_table")
    pieces: dict[str, list[pd.DataFrame]] = {name: [] for name in input_names}
    observed_sessions: set[tuple[object, ...]] = set()

    for shard_dir in shard_dirs:
        frames = {
            name: read_table(shard_dir / basenames[name]) for name in input_names
        }
        # Every shard must come from the SWR source the bundle is stamped with.
        for name, frame in frames.items():
            require_swr_source(frame, what=f"{shard_dir / basenames[name]}")
        shard_sessions = _session_keys(frames["stop_table"])
        overlap = observed_sessions & shard_sessions
        if overlap:
            raise ValueError(
                f"ContFrag shards overlap on {len(overlap)} session(s), e.g. "
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
            f"assembled ContFrag scope has {len(observed_sessions)} sessions; "
            f"expected {expected_sessions}"
        )

    event_table = pd.concat(pieces["event_table"], ignore_index=True)
    event_arm_table = pd.concat(pieces["event_arm_table"], ignore_index=True)
    stop_table = pd.concat(pieces["stop_table"], ignore_index=True)
    if (
        event_table["event_id"].isna().any()
        or event_table["event_id"].duplicated().any()
    ):
        raise ValueError("assembled event_table requires unique nonmissing event_id")
    if stop_table["stop_id"].isna().any() or stop_table["stop_id"].duplicated().any():
        raise ValueError("assembled stop_table requires unique nonmissing stop_id")
    validate_event_arm_table(event_arm_table, event_table=event_table)
    if int(stop_table["n_selected_events"].sum()) != len(event_table):
        raise ValueError("assembled stop/event count conservation failed")

    event_patch_table = build_event_patch_table(event_table, event_arm_table)
    event_patch_table["event_source"] = event_source
    event_patch_table["brain_area"] = brain_area
    validate_event_patch_table(event_patch_table, event_table=event_table)

    tables = {
        "event_table": stamp_covariate_schema(event_table, table="event"),
        "event_arm_table": stamp_covariate_schema(
            event_arm_table, table="event_arm"
        ),
        "event_patch_table": stamp_covariate_schema(
            event_patch_table, table="event_patch"
        ),
        "stop_table": stamp_covariate_schema(stop_table, table="stop"),
    }
    for frame in tables.values():
        stamp_swr_source(frame)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        # publish_canonical is the one write door (re-stamp idempotent, atomic).
        path = publish_canonical(
            frame, output_dir, table=BASENAME_TO_TABLE[name], basename=basenames[name]
        )
        LOGGER.info("saved %s -> %s (%d rows)", name, path, len(frame))
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--event-source", required=True, choices=("SWR", "MUA"))
    parser.add_argument(
        "--brain-area", required=True, choices=("HPC", "mPFC", "OFC")
    )
    parser.add_argument("--expected-sessions", type=int, default=218)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    assemble_contfrag_shards(
        [path.resolve() for path in args.shard_dirs],
        output_dir=args.output_dir.resolve(),
        event_source=args.event_source,
        brain_area=args.brain_area,
        expected_sessions=args.expected_sessions,
    )


if __name__ == "__main__":
    main()
