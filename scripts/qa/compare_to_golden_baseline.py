#!/usr/bin/env python
"""Diff a candidate replay_content root against a frozen golden baseline.

The refactor safety net: a change that is supposed to preserve results must
regenerate canonical tables that are byte-identical to the frozen baseline. This
raises on any structural or value drift in any canonical table, using the shared
keyed-parity core (``table_parity.compare_table_roots``) so it speaks the same
language as the regeneration manifest's baseline gate.

Exact by default (``atol=0``); pass ``--expected-drift-columns`` only when a
change is deliberately allowed to move specific columns, and
``--expected-added-columns`` when a table has legitimately *gained* a column
that the frozen baseline predates (the baseline stays frozen; the new columns
are excluded from the comparison and every other structural difference still
raises).

Usage:
    uv run python scripts/qa/compare_to_golden_baseline.py \
        --candidate Processed-Data/replay_content \
        --baseline  Processed-Data/replay_content/_golden_baseline_v5

    # Columns gained since ``_golden_baseline_v5_v4cache`` was frozen: the
    # stop table's reward-history basis, the event-patch table's allocation
    # fraction, and the stop table's two bout-start clocks. Declaring a column
    # a table does not carry is harmless, so this one declaration covers the
    # whole multi-table gate. Every declaration must be repeated on every run
    # -- omitting one makes the gate fail, by design.
    uv run python scripts/qa/compare_to_golden_baseline.py \
        --baseline Processed-Data/replay_content/_golden_baseline_v5_v4cache \
        --expected-added-columns reward_history_rcos_0 reward_history_rcos_1 \
                                 reward_history_rcos_2 reward_history_incomplete \
                                 patch_fraction \
                                 last_well_arrival_time_s outcome_known_time_s
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.table_io import read_table
from continuum_swr_replay.table_parity import (
    CANONICAL_TABLE_BASENAMES,
    compare_table_roots,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)
RC = PROCESSED_DATA_DIR / "replay_content"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=RC)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument(
        "--tables",
        nargs="+",
        default=list(CANONICAL_TABLE_BASENAMES),
        help="subset of canonical table basenames to compare",
    )
    parser.add_argument(
        "--expected-drift-columns",
        nargs="*",
        default=[],
        help="columns allowed to differ (default none -> exact parity)",
    )
    parser.add_argument(
        "--expected-added-columns",
        nargs="*",
        default=[],
        help=(
            "columns the candidate may carry that the baseline predates "
            "(comma- or space-separated); they are excluded from the "
            "comparison, and any other candidate-only column still raises"
        ),
    )
    parser.add_argument("--atol", type=float, default=0.0)
    args = parser.parse_args()
    # Accept both `--expected-added-columns a b` and `--expected-added-columns a,b`
    # so a four-name declaration can be pasted as one token.
    expected_added_columns = tuple(
        name
        for token in args.expected_added_columns
        for name in token.split(",")
        if name
    )

    # Two basename-keyed readers, one per root's layout: the candidate nests
    # qvals under models_per_bin_rate/; the frozen baseline is flat.
    def read_candidate(basename: str):
        if basename == "qvals_table":
            return read_table(args.candidate / "models_per_bin_rate" / "qvals_table")
        return read_table(args.candidate / basename)

    def read_baseline(basename: str):
        return read_table(args.baseline / basename)

    report = compare_table_roots(
        args.candidate,
        args.baseline,
        tuple(args.tables),
        expected_drift_columns=frozenset(args.expected_drift_columns),
        expected_added_columns=expected_added_columns,
        atol=args.atol,
        read_candidate=read_candidate,
        read_baseline=read_baseline,
    )
    # Undeclared drift already raised inside diff_table (uncaught -> nonzero
    # exit). Anything left in the report is drift the caller explicitly allowed
    # via --expected-drift-columns, so it passes with an informational log.
    declared = {name: cols for name, cols in report.items() if cols}
    if declared:
        logger.info("matches baseline except in declared drift column(s): %s", declared)
    elif expected_added_columns:
        logger.info(
            "matches baseline across %d table(s), ignoring declared added "
            "column(s): %s",
            len(report),
            list(expected_added_columns),
        )
    else:
        logger.info("byte-identical to baseline across %d table(s)", len(report))


if __name__ == "__main__":
    main()
