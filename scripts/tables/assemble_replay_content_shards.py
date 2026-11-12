#!/usr/bin/env python
"""Assemble disjoint Replay table shards into one canonical candidate bundle.

Per-animal shards are a computational optimization only. The assembly
(``prepare_data.assemble_replay_content_shards``) restores the monolithic
build: it verifies disjoint session sets, recomputes the assembly-level stop
counts and the patch table, discards the shards' standardized columns, and
fits one pooled leaf-arm standardization reference over the complete corpus
before writing the candidate tables. ``--stop-only`` joins the shards of a
stop-only build for the qvals bootstrap
(``prepare_data.assemble_replay_stop_shards``).
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from continuum_swr_replay.analysis.behavior.tables.prepare_data import (
    assemble_replay_content_shards,
    assemble_replay_stop_shards,
)
from continuum_swr_replay.data_loaders.epoch_register import analysed_epoch_ids


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
