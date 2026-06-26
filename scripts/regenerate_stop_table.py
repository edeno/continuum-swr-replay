#!/usr/bin/env python
"""Regenerate ``stop_table.pkl`` with the new count columns.

Re-runs the canonical :func:`build_stop_table` per session (which now emits
``n_SWR`` / ``n_MUA`` / ``n_non_local`` alongside the existing columns), then
adds the canonical ``n_non_local_SWR`` / ``n_non_local_no_SWR`` split by joining
the existing ``event_table.pkl``'s ``is_swr`` on ``stop_id``. This rebuilds only
the stop table — it does NOT rebuild the event/arm/patch tables — so it is much
cheaper than a full ``prepare_replay_content_data.py`` run.

The stop-bout definition is unchanged, so ``stop_id`` matches the existing
``event_table`` row-for-row and the merge is exact.

Usage:
    uv run python scripts/regenerate_stop_table.py --max-per-animal 1   # smoke
    uv run python scripts/regenerate_stop_table.py                      # full
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_replay_content_data import find_sessions  # noqa: E402

from continuum_swr_replay.analysis.replay import build_stop_table  # noqa: E402
from continuum_swr_replay.analysis.replay.tables import (  # noqa: E402
    add_nonlocal_swr_split,
)
from continuum_swr_replay.data_loaders import load_data  # noqa: E402
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR  # noqa: E402
from continuum_swr_replay.types import SessionDataKeys  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)
RC = PROCESSED_DATA_DIR / "replay_content"


def build_one(session: dict) -> pd.DataFrame | None:
    """Load dot + trials for one session and build its stop table."""
    dot = pd.read_pickle(session["dot_file"])
    data = load_data(
        session["nwb_file_name"],
        session["epoch_name"],
        include=frozenset({SessionDataKeys.TRIALS}),
    )
    trials, track_graph = data.get("trials"), data.get("track_graph")
    if trials is None or track_graph is None:
        return None
    return build_stop_table(dot, trials, track_graph)


def validate(stop: pd.DataFrame) -> None:
    """Hard checks on the regenerated table before it is trusted."""
    assert (stop["n_non_local"] == stop["n_events"]).all(), "n_non_local != n_events alias"
    split_ok = (
        stop["n_non_local_SWR"] + stop["n_non_local_no_SWR"] == stop["n_non_local"]
    )
    assert split_ok.all(), f"{(~split_ok).sum()} stops where SWR split != n_non_local"
    assert (stop["is_swr_stop"] == (stop["n_SWR"] > 0)).all(), "is_swr_stop != n_SWR>0"
    assert (stop["is_mua_stop"] == (stop["n_MUA"] > 0)).all(), "is_mua_stop != n_MUA>0"
    assert (stop[["n_SWR", "n_MUA", "n_non_local_SWR"]] >= 0).all().all(), "negative count"
    logger.info(
        "validation OK | n_SWR>0 in %d/%d stops, total SWRs=%d, total non-local=%d "
        "(SWR-overlap=%d, no-SWR=%d)",
        int((stop["n_SWR"] > 0).sum()),
        len(stop),
        int(stop["n_SWR"].sum()),
        int(stop["n_non_local"].sum()),
        int(stop["n_non_local_SWR"].sum()),
        int(stop["n_non_local_no_SWR"].sum()),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-animal", type=int, default=None, help="smoke subset")
    ap.add_argument("--no-save", action="store_true", help="build + validate only")
    args = ap.parse_args()

    sessions = find_sessions(max_per_animal=args.max_per_animal)
    logger.info("building stop tables for %d sessions...", len(sessions))
    tables, t0 = [], time.time()
    for i, s in enumerate(sessions):
        try:
            st = build_one(s)
        except Exception as e:  # noqa: BLE001
            logger.warning("skip %s %s: %s", s["nwb_file_name"], s["epoch_name"], e)
            continue
        if st is not None and len(st):
            tables.append(st)
        if (i + 1) % 20 == 0:
            logger.info("  %d/%d (%.0fs)", i + 1, len(sessions), time.time() - t0)
    stop = pd.concat(tables, ignore_index=True)

    event_table = pd.read_pickle(RC / "event_table.pkl")
    stop = add_nonlocal_swr_split(stop, event_table)
    validate(stop)

    if args.no_save:
        logger.info("--no-save: built %d stops, not written", len(stop))
        return
    out = RC / "stop_table.pkl"
    if out.exists():
        backup = out.with_suffix(".pkl.bak")
        with open(backup, "wb") as f:
            pickle.dump(pd.read_pickle(out), f)
        logger.info("backed up existing -> %s", backup.name)
    stop.to_pickle(out)
    logger.info("saved %d stops -> %s", len(stop), out)


if __name__ == "__main__":
    main()
