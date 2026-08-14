#!/usr/bin/env python
"""Stage 5B: per-session retrospective replay-content alignment.

Reads the canonical tables under ``PROCESSED_DATA_DIR/replay_content/``
and computes, for every session, five event-level alignment metrics
between replay content during a stop and the arm the animal visited
next:

- ``chosen_arm_fraction`` — replay mass on ``next_arm``
- ``ends_at_next``        — last in-event MAP arm equals ``next_arm``
- ``contains_path_to_next`` — MAP-arm trace covers the geometric path
  from ``current_arm`` to ``next_arm``
- ``chosen_patch_fraction`` — replay mass on ``next_patch``
- ``ends_at_next_patch`` — final MAP arm belongs to ``next_patch``

Each metric is compared against a within-session, visit-context future-label
shuffle over ``--n-shuffles`` draws. A visit context is a contiguous
``current_arm`` run within ``stop_id`` because one long immobility bout can span
a trial boundary or return to an earlier arm. Arm metrics condition on current
arm, patch metrics on current patch, and path coverage on current arm, route
length, and coarse duration. Each family has its own cohort and denominator.
Output is one row per session with explicit retained/excluded unit and stratum
counts.

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
    at_well_event_cohort,
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


def _family_unit_counts(
    result: dict[str, float | int | bool | str],
) -> dict[str, int]:
    """Movable visit-context counts for each calibrated metric family.

    Only families the run actually requested appear in ``result``, so read
    whichever suffixed keys are present rather than assuming all three.
    """
    suffix = "_family"
    prefix = "n_shuffle_units_"
    counts: dict[str, int] = {}
    for key, value in result.items():
        if not (key.startswith(prefix) and key.endswith(suffix)):
            continue
        family = key[len(prefix) : -len(suffix)]
        # The retained count is ``n_shuffle_units_<family>_family``. Sibling
        # keys qualify the same suffix -- ``..._input_<family>_family``,
        # ``..._excluded_singleton_<family>_family`` -- and their zeros would
        # read as a non-movable family. A family name is a single token.
        if "_" in family:
            continue
        counts[family] = int(value)
    return counts


def _retrospective_status(
    result: dict[str, float | int | bool | str],
) -> str:
    """Explicit persisted status for a session's retrospective analysis.

    A family needs two movable visit contexts before its null can move a
    label, and the families are calibrated independently, so a session can
    estimate some metrics and not others. Distinguish that from the case
    where nothing is estimable: ``ok`` means every requested family is
    movable, ``partial_movable_families`` that at least one is not, and
    ``no_movable_family`` that none are.
    """
    counts = _family_unit_counts(result)
    if not counts:
        return "no_movable_family"
    movable = [count >= 2 for count in counts.values()]
    if all(movable):
        return "ok"
    return "partial_movable_families" if any(movable) else "no_movable_family"


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
    n_before_location = len(event_table)
    event_table, event_arm_table, event_sequence_table = at_well_event_cohort(
        event_table, event_arm_table, event_sequence_table
    )
    logging.info(
        "Explicit at-well cohort: retained %d/%d events",
        len(event_table),
        n_before_location,
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
    n_no_movable = 0
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
        t0 = time.time()
        # Fail the batch before publication when any session violates the
        # location/observability contract.  Continuing here would turn a
        # scientific integrity error into an unreported session exclusion and
        # then persist a plausible-looking partial cohort.
        result = retrospective_alignment_with_shuffle(
            ev_s,
            ea_s,
            es_s,
            n_shuffles=args.n_shuffles,
            rng=np.random.default_rng(rng.integers(0, 2**32)),
        )
        family_unit_counts = _family_unit_counts(result)
        retrospective_status = _retrospective_status(result)
        if retrospective_status != "ok":
            logging.info(
                "Retaining %s/%s/%s with status=%s; families without two "
                "movable visit contexts report NaN metrics (%s)",
                animal,
                day,
                epoch,
                retrospective_status,
                family_unit_counts,
            )
            n_no_movable += 1

        result.update(
            {
                "animal": animal,
                "day": day,
                "epoch": epoch,
                "subset": label,
                "retrospective_status": retrospective_status,
                "elapsed_s": time.time() - t0,
            }
        )
        rows.append(result)
        if (i + 1) % 25 == 0 or i + 1 == len(sessions):
            logging.info("  done %d/%d", i + 1, len(sessions))

    out = pd.DataFrame(rows)
    out.to_pickle(output_path)
    logging.info("Saved %d sessions → %s", len(out), output_path)
    logging.info(
        "Persisted %d/%d sessions with at least one non-movable metric family",
        n_no_movable,
        len(sessions),
    )

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
        finite_metric = np.isfinite(pd.to_numeric(out[diff_col], errors="coerce"))
        n_pos = int((out.loc[finite_metric, diff_col] > 0).sum())
        n_finite = int(finite_metric.sum())
        logging.info(
            "  %-25s  real=%.3f  Δ vs shuffle=%+.3f  pos=%d/%d",
            metric,
            med_real,
            med_diff,
            n_pos,
            n_finite,
        )


if __name__ == "__main__":
    main()
