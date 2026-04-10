#!/usr/bin/env python
"""Prepare master event-level and long-format datasets for replay content analysis.

Processes all sessions once, computing all predictors needed for M0-M7.
Saves to Processed-Data/replay_content/ for fast iteration on model fitting.

Usage:
    uv run python scripts/prepare_replay_content_data.py           # full run
    uv run python scripts/prepare_replay_content_data.py --test    # 2 per animal
"""

import argparse
import logging
import pickle
import re
import time

import pandas as pd

from continuum_swr_replay.analysis.hippocampus.replay_content.data_preparation import (
    prepare_session,
)
from continuum_swr_replay.analysis.hippocampus.replay_content.predictors import (
    add_per_arm_continuous_predictors,
    add_temporal_interactions,
)
from continuum_swr_replay.data_loaders import load_data
from continuum_swr_replay.data_loaders.constants import NWB_FILES
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.types import SessionDataKeys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)

OUTPUT_DIR = PROCESSED_DATA_DIR / "replay_content"


def find_sessions(max_per_animal: int | None = None):
    """Find all sessions with data_over_time and environment files."""
    dot_files = sorted(PROCESSED_DATA_DIR.glob("*_HPC_80_data_over_time.pkl"))
    env_files = {
        f.stem.replace("_environment", ""): f
        for f in PROCESSED_DATA_DIR.glob("*_HPC_environment.pkl")
    }

    by_animal: dict[str, list] = {}
    for f in dot_files:
        match = re.match(r"(\w+\d{8})_(\d+_r\d+)_HPC_80_data_over_time\.pkl", f.name)
        if not match:
            continue
        animal_date, epoch = match.group(1), match.group(2)
        animal = re.match(r"([a-z]+)", animal_date).group(1)
        session_id = f"{animal_date}_{epoch}_HPC"
        if session_id not in env_files:
            continue

        nwb_match = [nwb for nwb in NWB_FILES if animal_date in nwb]
        if not nwb_match:
            continue

        by_animal.setdefault(animal, []).append(
            {
                "nwb_file_name": nwb_match[0],
                "epoch_name": epoch,
                "dot_file": f,
                "env_file": env_files[session_id],
            }
        )

    selected = []
    for animal in sorted(by_animal):
        sessions = by_animal[animal]
        if max_per_animal is not None:
            sessions = sessions[:max_per_animal]
        selected.extend(sessions)

    return selected


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--test", action="store_true", help="2 sessions per animal")
    parser.add_argument("--max-per-animal", type=int, default=None)
    args = parser.parse_args()

    max_per_animal = 2 if args.test else args.max_per_animal
    sessions = find_sessions(max_per_animal=max_per_animal)
    logging.info("Found %d sessions", len(sessions))

    t0 = time.time()
    all_events = []
    all_long = []
    trials_per_session: dict[tuple, pd.DataFrame] = {}
    track_graphs: dict[tuple, object] = {}
    session_cols = ["animal", "day", "epoch"]

    for i, session in enumerate(sessions):
        nwb = session["nwb_file_name"]
        epoch = session["epoch_name"]

        try:
            # Load pre-saved data (fast)
            dot = pd.read_pickle(session["dot_file"])
            with open(session["env_file"], "rb") as f:
                env = pickle.load(f)

            # Load trials from Spyglass (slow, ~30s)
            data = load_data(nwb, epoch, include=frozenset({SessionDataKeys.TRIALS}))
            trials = data.get("trials")
            track_graph = data.get("track_graph")

        except Exception as e:
            logging.warning("Skip %s %s: %s", nwb, epoch, e)
            continue

        result = prepare_session(dot, env, trials, track_graph)
        if result is None:
            continue

        events, long = result

        # Store trials and track_graph for visit_recency computation
        animal = events["animal"].iloc[0]
        day = events["day"].iloc[0]
        epoch_val = events["epoch"].iloc[0]
        session_key = (animal, day, epoch_val)
        trials_per_session[session_key] = trials
        track_graphs[session_key] = track_graph

        all_events.append(events)
        all_long.append(long)

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            remaining = elapsed / (i + 1) * (len(sessions) - i - 1)
            logging.info(
                "[%d/%d] %ds elapsed, ~%ds remaining",
                i + 1,
                len(sessions),
                int(elapsed),
                int(remaining),
            )

        del dot  # free memory

    if not all_events:
        logging.error("No sessions produced data")
        return

    # Concatenate
    events_all = pd.concat(all_events, ignore_index=True)
    long_all = pd.concat(all_long, ignore_index=True)

    logging.info(
        "Events: %d, Long rows: %d, Sessions: %d",
        len(events_all),
        len(long_all),
        events_all.groupby(session_cols).ngroups,
    )

    # Compute per-arm continuous predictors (visit_recency, cumulative_reward)
    logging.info("Computing visit_recency and cumulative_reward...")
    t1 = time.time()
    long_all = add_per_arm_continuous_predictors(
        long_all, events_all, trials_per_session, track_graphs
    )
    logging.info("Per-arm predictors computed in %ds", int(time.time() - t1))

    # Add temporal interaction columns
    add_temporal_interactions(long_all)

    # Add interaction columns from event_collapsing (switch, omission, SWR)
    # These are already added by make_long_format_arm, but temporal ones are new

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    events_path = OUTPUT_DIR / "events.pkl"
    events_all.to_pickle(events_path)
    logging.info("Saved events to %s (%d rows)", events_path, len(events_all))

    long_path = OUTPUT_DIR / "long.pkl"
    long_all.to_pickle(long_path)
    logging.info("Saved long-format to %s (%d rows)", long_path, len(long_all))

    # Summary statistics
    total_time = time.time() - t0
    logging.info("Total time: %ds", int(total_time))
    logging.info("Columns in events: %s", sorted(events_all.columns.tolist()))
    logging.info("Columns in long: %s", sorted(long_all.columns.tolist()))

    # Validate
    n_events = len(events_all)
    n_long = len(long_all)
    expected_long = n_events * 9
    if n_long != expected_long:
        logging.warning(
            "Long row count %d != expected %d (events * 9)", n_long, expected_long
        )

    for col in [
        "replay_here",
        "is_current",
        "is_prev",
        "is_next",
        "is_swr",
        "time_since_stop",
        "log_distance",
        "visit_recency",
        "cumulative_reward",
    ]:
        if col in long_all.columns:
            nan_frac = long_all[col].isna().mean()
            logging.info("  %s: %.1f%% NaN", col, nan_frac * 100)
        else:
            logging.warning("  %s: MISSING", col)


if __name__ == "__main__":
    main()
