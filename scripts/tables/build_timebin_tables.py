#!/usr/bin/env python
"""Build the within-stop event-onset table over all sessions.

Per session: load ``dot`` + pump-event times, build the sparse event-onset
table (see ``analysis.event_tables.timebin_tables``), concatenate, stamp with
the canonical covariate schema, and save ``event_onset_table.parquet`` (one row
per onset).

Usage:
    uv run python scripts/tables/build_timebin_tables.py --max-per-animal 1   # smoke
    uv run python scripts/tables/build_timebin_tables.py                      # full
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    publish_swr_table,
    read_dot_with_swr_source,
    stamp_swr_source,
)
from continuum_swr_replay.analysis.event_tables.timebin_tables import (  # noqa: E402
    build_event_onset_table,
)
from continuum_swr_replay.data_loaders import load_data  # noqa: E402
from continuum_swr_replay.data_types import SessionDataKeys  # noqa: E402
from continuum_swr_replay.paths import replay_content_dir
from continuum_swr_replay.tables.session_keys import n_sessions  # noqa: E402

# Sibling-script import: run as a script, only this file's directory is on
# sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.tables._table_runner import find_sessions  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


def build_one(session: dict) -> pd.DataFrame:
    # SWR labels from the one SWR source, not the decode's stored ripples
    dot = read_dot_with_swr_source(session["dot_file"])
    data = load_data(
        session["nwb_file_name"],
        session["epoch_name"],
        include=frozenset({SessionDataKeys.PUMP_EVENTS}),
    )
    pumps = data.get("pump_events")
    pump_times = (
        pumps["start_time"].to_numpy()
        if pumps is not None and len(pumps)
        else np.array([])
    )
    return build_event_onset_table(dot, pump_times)


def validate(onset: pd.DataFrame) -> None:
    assert (onset.t_since_stop_onset >= 0).all(), "negative onset t_since_stop_onset"
    logger.info(
        "validation OK | onsets: SWR=%d non_local=%d MUA=%d | t_since_pump "
        "non-null %.0f%%",
        int((onset.event_type == "SWR").sum()),
        int((onset.event_type == "non_local").sum()),
        int((onset.event_type == "MUA").sum()),
        100 * onset.t_since_pump.notna().mean(),
    )


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-animal", type=int, default=None)
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=replay_content_dir(),
        help=(
            "Directory the table is written to. Defaults to the canonical "
            "replay_content root; pass a staging directory to build without "
            "overwriting it."
        ),
    )
    args = ap.parse_args(argv)

    sessions = find_sessions(max_per_animal=args.max_per_animal)
    logger.info("building the event-onset table for %d sessions...", len(sessions))
    session_tables, failures, t0 = [], [], time.time()
    for i, s in enumerate(sessions):
        try:
            session_tables.append(build_one(s))
        except Exception as e:  # noqa: BLE001
            failures.append((s["nwb_file_name"], s["epoch_name"], str(e)))
            continue
        if (i + 1) % 20 == 0:
            logger.info("  %d/%d (%.0fs)", i + 1, len(sessions), time.time() - t0)

    # Fail closed. The table is joined to the event and stop tables on stop
    # keys and feeds every Family-B rate fit; a silently dropped session would
    # ship a table covering fewer sessions than its siblings, which no
    # downstream reader (they only check the schema stamp) would catch.
    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(sessions)} session(s) failed to build an "
            f"event-onset table; sample={failures[:10]}. Nothing was written."
        )
    onset = pd.concat(session_tables, ignore_index=True)
    n = n_sessions(onset)
    if n != len(sessions):
        raise RuntimeError(
            f"event_onset table has {n} sessions; expected {len(sessions)}"
        )
    validate(onset)

    if args.no_save:
        logger.info("--no-save: onsets=%d, not written", len(onset))
        return
    # publish_swr_table checks the source and writes atomically; the table was
    # built from decode frames read through the SWR source.
    publish_swr_table(
        stamp_swr_source(onset),
        args.output_dir,
        table="event_onset",
        basename="event_onset_table",
    )
    logger.info("saved %d onsets -> event_onset_table.parquet", len(onset))


if __name__ == "__main__":
    main()
