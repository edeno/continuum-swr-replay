#!/usr/bin/env python
"""Build the Part B within-stop time-binned tables over all sessions.

Per session: load ``dot`` + pump-event times, build the sparse event-onset
table and the dense 4 ms stop-timebin table (see
``analysis.replay.timebin_tables``), concatenate, and save:

- ``event_onset_table.pkl``      (sparse, ~one row per onset)
- ``stop_timebin_table.parquet`` (dense, ~one row per (stop, 4 ms bin))

Usage:
    uv run python scripts/decoding/build_timebin_tables.py --max-per-animal 1   # smoke
    uv run python scripts/decoding/build_timebin_tables.py                      # full
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay.prepare_data import find_sessions
from continuum_swr_replay.analysis.replay.timebin_tables import (  # noqa: E402
    build_event_onset_table,
    build_stop_timebin_table,
)
from continuum_swr_replay.data_loaders import load_data  # noqa: E402
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.table_io import write_table  # noqa: E402
from continuum_swr_replay.types import SessionDataKeys  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
RC = PROCESSED_DATA_DIR / "replay_content"


def build_one(session: dict) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    dot = pd.read_pickle(session["dot_file"])
    data = load_data(
        session["nwb_file_name"], session["epoch_name"],
        include=frozenset({SessionDataKeys.PUMP_EVENTS}),
    )
    pumps = data.get("pump_events")
    pump_times = (
        pumps["start_time"].to_numpy() if pumps is not None and len(pumps) else np.array([])
    )
    return build_event_onset_table(dot, pump_times), build_stop_timebin_table(dot, pump_times)


def validate(onset: pd.DataFrame, dense: pd.DataFrame) -> None:
    # Dense onset totals reconstruct the sparse onset counts. Both derive from
    # the same _event_onset_times + assign_bout, so they must agree exactly.
    for etype, col in [("SWR", "n_SWR"), ("non_local", "n_non_local"), ("MUA", "n_MUA")]:
        a = int((onset.event_type == etype).sum())
        b = int(dense[col].sum())
        assert a == b, f"{etype}: onset rows {a} != dense {col} sum {b}"
    assert (dense.t_since_stop_onset >= 0).all(), "negative t_since_stop_onset"
    assert (onset.t_since_stop_onset >= 0).all(), "negative onset t_since_stop_onset"
    # Reward stops: at least some bins land soon after the pump.
    rw = dense[dense.is_reward_stop & dense.t_since_pump.notna()]
    near = float((rw.t_since_pump < 2.0).mean()) if len(rw) else float("nan")
    logger.info(
        "validation OK | onsets: SWR=%d non_local=%d MUA=%d | dense bins=%d "
        "(reward-stop bins=%d, %.0f%% within 2s of pump) | t_since_pump non-null %.0f%%",
        int((onset.event_type == "SWR").sum()),
        int((onset.event_type == "non_local").sum()),
        int((onset.event_type == "MUA").sum()),
        len(dense),
        int(dense.is_reward_stop.sum()),
        100 * near,
        100 * dense.t_since_pump.notna().mean(),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-animal", type=int, default=None)
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    sessions = find_sessions(max_per_animal=args.max_per_animal)
    logger.info("building time-binned tables for %d sessions...", len(sessions))
    onsets, denses, t0 = [], [], time.time()
    for i, s in enumerate(sessions):
        try:
            o, d = build_one(s)
        except Exception as e:  # noqa: BLE001
            logger.warning("skip %s %s: %s", s["nwb_file_name"], s["epoch_name"], e)
            continue
        if len(d):
            onsets.append(o)
            denses.append(d)
        if (i + 1) % 20 == 0:
            logger.info("  %d/%d (%.0fs)", i + 1, len(sessions), time.time() - t0)
    onset = pd.concat(onsets, ignore_index=True)
    dense = pd.concat(denses, ignore_index=True)
    validate(onset, dense)

    if args.no_save:
        logger.info("--no-save: onsets=%d, dense bins=%d, not written", len(onset), len(dense))
        return
    write_table(onset, RC / "event_onset_table")
    dense.to_parquet(RC / "stop_timebin_table.parquet", compression="zstd", index=False)
    logger.info(
        "saved %d onsets -> event_onset_table.pkl, %d bins -> stop_timebin_table.parquet",
        len(onset), len(dense),
    )


if __name__ == "__main__":
    main()
