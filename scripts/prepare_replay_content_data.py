#!/usr/bin/env python
"""Prepare master event-level and long-format datasets for replay content analysis.

Processes all sessions once, computing all predictors needed for M0-M7.
Saves to Processed-Data/replay_content/ for fast iteration on model fitting.

All per-arm predictors (visit_recency, cumulative_reward, dist_to_arm),
temporal interactions, and event UIDs are computed inside prepare_session()
so that each session is self-contained.

Produces two long-format outputs:
- long_count.pkl: count-based (primary, for ConditionalPoisson)
- long_binary.pkl: binary modal-arm (robustness, for ConditionalLogit)

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
    all_long_count = []
    all_long_binary = []
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

        events, long_count, long_binary = result

        all_events.append(events)
        all_long_count.append(long_count)
        if len(long_binary) > 0:
            all_long_binary.append(long_binary)

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
    long_count_all = pd.concat(all_long_count, ignore_index=True)
    long_binary_all = (
        pd.concat(all_long_binary, ignore_index=True) if all_long_binary else None
    )

    logging.info(
        "Events: %d, Count rows: %d, Sessions: %d",
        len(events_all),
        len(long_count_all),
        events_all.groupby(session_cols).ngroups,
    )

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    events_path = OUTPUT_DIR / "events.pkl"
    events_all.to_pickle(events_path)
    logging.info("Saved events to %s (%d rows)", events_path, len(events_all))

    count_path = OUTPUT_DIR / "long_count.pkl"
    long_count_all.to_pickle(count_path)
    logging.info(
        "Saved count long-format to %s (%d rows)", count_path, len(long_count_all)
    )

    if long_binary_all is not None:
        binary_path = OUTPUT_DIR / "long_binary.pkl"
        long_binary_all.to_pickle(binary_path)
        logging.info(
            "Saved binary long-format to %s (%d rows)",
            binary_path,
            len(long_binary_all),
        )

    # Summary statistics
    total_time = time.time() - t0
    logging.info("Total time: %ds", int(total_time))
    logging.info("Columns in count long: %s", sorted(long_count_all.columns.tolist()))

    # Validate
    for col in [
        "arm_count",
        "event_uid",
        "is_current",
        "is_prev",
        "is_next",
        "is_swr",
        "time_since_stop",
        "visit_recency",
        "cumulative_reward",
        "dist_to_arm",
        "time_x_current",
        "time_x_prev",
        "time_x_next",
    ]:
        if col in long_count_all.columns:
            nan_frac = long_count_all[col].isna().mean()
            logging.info("  %s: %.1f%% NaN", col, nan_frac * 100)
        else:
            logging.warning("  %s: MISSING", col)


if __name__ == "__main__":
    main()
