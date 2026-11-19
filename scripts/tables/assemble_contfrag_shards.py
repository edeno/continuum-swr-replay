#!/usr/bin/env python
"""Assemble disjoint per-animal ContFrag shards into one canonical bundle.

:func:`assemble_contfrag_shards` joins the shards' tables and publishes them.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.contfrag.prepare_data import (
    contfrag_event_patch_table,
    contfrag_table_basenames,
)
from continuum_swr_replay.analysis.event_tables.table_contracts import (
    validate_event_arm_table,
    validate_event_patch_table,
)
from continuum_swr_replay.data_loaders.epoch_register import analyzed_epoch_ids

# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.tables._table_runner import (  # noqa: E402
    publish_contfrag_tables,
    read_disjoint_shards,
)


def assemble_contfrag_shards(
    shard_dirs: list[Path],
    *,
    output_dir: Path,
    event_source: str,
    brain_area: str,
    expected_sessions: int,
) -> dict[str, pd.DataFrame]:
    """Join per-animal shards of the ContFrag tables into one published bundle.

    The shards' event, event-arm and stop tables are concatenated in shard
    order and the event-patch table is projected from the whole event-arm
    table, as ``prepare_contfrag_replay_content_data.py`` does over every session at
    once.

    Parameters
    ----------
    shard_dirs : list of Path
        The shard folders, each written by
        ``prepare_contfrag_replay_content_data.py`` for one animal.
    output_dir : Path
        Where the four tables are published.
    event_source : str
        ``"SWR"`` or ``"MUA"``.
    brain_area : str
        ``"HPC"``, ``"mPFC"`` or ``"OFC"``.
    expected_sessions : int
        The number of sessions the shards must hold together.

    Returns
    -------
    dict of str to pd.DataFrame
        The published tables (``_table_runner.publish_contfrag_tables``).

    Raises
    ------
    ValueError
        If the shards overlap or miss sessions, a shard or the bundle breaks
        stop/event count conservation, or an ``event_id`` or ``stop_id`` is
        missing or repeats.
    """
    basenames = contfrag_table_basenames(event_source, brain_area)
    shards = read_disjoint_shards(
        shard_dirs,
        {
            name: basenames[name]
            for name in ("event_table", "event_arm_table", "stop_table")
        },
        what="ContFrag",
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

    event_table = pd.concat(
        [frames["event_table"] for frames in shards], ignore_index=True
    )
    event_arm_table = pd.concat(
        [frames["event_arm_table"] for frames in shards], ignore_index=True
    )
    stop_table = pd.concat(
        [frames["stop_table"] for frames in shards], ignore_index=True
    )
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

    event_patch_table = contfrag_event_patch_table(
        event_table, event_arm_table, event_source=event_source, brain_area=brain_area
    )
    validate_event_patch_table(event_patch_table, event_table=event_table)

    return publish_contfrag_tables(
        event_table,
        event_arm_table,
        event_patch_table,
        stop_table,
        output_dir=output_dir,
        basenames=basenames,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--event-source", required=True, choices=("SWR", "MUA"))
    parser.add_argument("--brain-area", required=True, choices=("HPC", "mPFC", "OFC"))
    parser.add_argument(
        "--expected-sessions",
        type=int,
        default=len(analyzed_epoch_ids()),
        help="Epochs (sessions) the tables must hold; default: the epoch register's "
        "analyzed epochs (HPC cohort).",
    )
    args = parser.parse_args(argv)
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
