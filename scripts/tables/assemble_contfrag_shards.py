#!/usr/bin/env python
"""Assemble disjoint per-animal ContFrag shards into one canonical bundle.

Command-line front end of
:func:`continuum_swr_replay.analysis.contfrag.prepare_data.assemble_contfrag_shards`.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from continuum_swr_replay.analysis.contfrag.prepare_data import (
    assemble_contfrag_shards,
)
from continuum_swr_replay.data_loaders.epoch_register import analysed_epoch_ids


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shard_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--event-source", required=True, choices=("SWR", "MUA"))
    parser.add_argument("--brain-area", required=True, choices=("HPC", "mPFC", "OFC"))
    parser.add_argument(
        "--expected-sessions",
        type=int,
        default=len(analysed_epoch_ids()),
        help="Epochs (sessions) the tables must hold; default: the epoch register's "
        "analysed epochs (HPC cohort).",
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
