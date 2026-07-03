#!/usr/bin/env python
"""Stage 5B: per-session retrospective replay-content alignment.

Reads the canonical tables under ``PROCESSED_DATA_DIR/replay_content/``
and computes, for every session, three event-level alignment metrics
between replay content during a stop and the arm the animal visited
next:

- ``chosen_arm_fraction`` — replay mass on ``next_arm``
- ``ends_at_next``        — last in-event MAP arm equals ``next_arm``
- ``contains_path_to_next`` — MAP-arm trace covers the geometric path
  from ``current_arm`` to ``next_arm``

Each metric is compared against a within-session
``next_arm_circular_shift`` null over ``--n-shuffles`` draws. Output
is one row per session.

Usage::

    uv run python scripts/pfc/run_stage5b_retrospective.py
    uv run python scripts/pfc/run_stage5b_retrospective.py --n-shuffles 200
    uv run python scripts/pfc/run_stage5b_retrospective.py --swr-only
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay import (
    retrospective_alignment_with_shuffle,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)

INPUT_DIR = PROCESSED_DATA_DIR / "replay_content"
OUTPUT_PATH = INPUT_DIR / "stage5b_retrospective.pkl"

SESSION_COLS = ["animal", "day", "epoch"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-shuffles",
        type=int,
        default=100,
        help="Number of next_arm circular-shift shuffles per session.",
    )
    parser.add_argument(
        "--swr-only",
        action="store_true",
        help="Restrict to SWR-coincident events (is_swr == True).",
    )
    parser.add_argument(
        "--non-swr-only",
        action="store_true",
        help="Restrict to non-SWR events (is_swr == False).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Master seed for the shuffle RNG.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_PATH),
        help="Output pickle path. Defaults next to the input tables.",
    )
    args = parser.parse_args()

    if args.swr_only and args.non_swr_only:
        raise SystemExit("Pick at most one of --swr-only / --non-swr-only.")

    label = "all"
    if args.swr_only:
        label = "swr_only"
    elif args.non_swr_only:
        label = "non_swr_only"

    output_path = args.output
    if label != "all" and output_path == str(OUTPUT_PATH):
        output_path = str(INPUT_DIR / f"stage5b_retrospective_{label}.pkl")

    logging.info("Loading canonical tables from %s", INPUT_DIR)
    event_table = pd.read_pickle(INPUT_DIR / "event_table.pkl")
    event_arm_table = pd.read_pickle(INPUT_DIR / "event_arm_table.pkl")
    event_sequence_table = pd.read_pickle(INPUT_DIR / "event_sequence_table.pkl")
    logging.info(
        "events=%d  event_arm rows=%d  event_seq rows=%d  sessions=%d",
        len(event_table),
        len(event_arm_table),
        len(event_sequence_table),
        event_table.groupby(SESSION_COLS).ngroups,
    )

    if args.swr_only:
        keep = event_table["is_swr"].astype(bool)
        event_table = event_table[keep].copy()
        event_arm_table = event_arm_table[event_arm_table["is_swr"].astype(bool)].copy()
        event_sequence_table = event_sequence_table.merge(
            event_table[[*SESSION_COLS, "event_id"]],
            on=[*SESSION_COLS, "event_id"],
            how="inner",
        )
        logging.info("Filtered to SWR events: %d", len(event_table))
    elif args.non_swr_only:
        keep = ~event_table["is_swr"].astype(bool)
        event_table = event_table[keep].copy()
        event_arm_table = event_arm_table[
            ~event_arm_table["is_swr"].astype(bool)
        ].copy()
        event_sequence_table = event_sequence_table.merge(
            event_table[[*SESSION_COLS, "event_id"]],
            on=[*SESSION_COLS, "event_id"],
            how="inner",
        )
        logging.info("Filtered to non-SWR events: %d", len(event_table))

    rng = np.random.default_rng(args.seed)
    rows = []
    sessions = sorted(event_table.groupby(SESSION_COLS).groups.keys())
    for i, sess_key in enumerate(sessions):
        animal, day, epoch = sess_key
        ev_s = event_table[
            (event_table["animal"] == animal)
            & (event_table["day"] == day)
            & (event_table["epoch"] == epoch)
        ]
        ea_s = event_arm_table[
            (event_arm_table["animal"] == animal)
            & (event_arm_table["day"] == day)
            & (event_arm_table["epoch"] == epoch)
        ]
        es_s = event_sequence_table[
            (event_sequence_table["animal"] == animal)
            & (event_sequence_table["day"] == day)
            & (event_sequence_table["epoch"] == epoch)
        ]
        if len(ev_s) < 2:
            logging.info(
                "Skipping %s/%s/%s (only %d events)", animal, day, epoch, len(ev_s)
            )
            continue

        t0 = time.time()
        try:
            result = retrospective_alignment_with_shuffle(
                ev_s,
                ea_s,
                es_s,
                n_shuffles=args.n_shuffles,
                rng=np.random.default_rng(rng.integers(0, 2**32)),
            )
        except Exception as exc:  # noqa: BLE001 — surface error per session, keep going
            logging.error("FAILED %s/%s/%s: %s", animal, day, epoch, exc)
            continue

        result.update(
            {
                "animal": animal,
                "day": day,
                "epoch": epoch,
                "subset": label,
                "elapsed_s": time.time() - t0,
            }
        )
        rows.append(result)
        if (i + 1) % 25 == 0 or i + 1 == len(sessions):
            logging.info("  done %d/%d", i + 1, len(sessions))

    out = pd.DataFrame(rows)
    out.to_pickle(output_path)
    logging.info("Saved %d sessions → %s", len(out), output_path)

    # Summary across sessions for quick stdout interpretation.
    logging.info("===== Summary across %d sessions (subset=%s) =====", len(out), label)
    for metric in (
        "chosen_arm_fraction",
        "ends_at_next",
        "contains_path_to_next",
        "chosen_patch_fraction",
        "ends_at_next_patch",
    ):
        real_col = f"real_mean_{metric}"
        diff_col = f"real_minus_shuffled_{metric}"
        if real_col not in out.columns:
            continue
        med_real = float(out[real_col].median())
        med_diff = float(out[diff_col].median())
        n_pos = int((out[diff_col] > 0).sum())
        logging.info(
            "  %-25s  real=%.3f  Δ vs shuffle=%+.3f  pos=%d/%d",
            metric,
            med_real,
            med_diff,
            n_pos,
            len(out),
        )


if __name__ == "__main__":
    main()
