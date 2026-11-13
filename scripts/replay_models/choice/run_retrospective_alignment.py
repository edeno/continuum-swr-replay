#!/usr/bin/env python
"""Stage 5B: per-session retrospective replay-content alignment.

Reads the canonical tables under ``paths.replay_content_dir()``
and computes, for every session, five event-level alignment metrics
between replay content during a stop and the arm the animal visited
next:

- ``chosen_arm_fraction`` — replay mass on ``next_arm``
- ``ends_at_next``        — last in-event MAP arm equals ``next_arm``
- ``contains_path_to_next`` — MAP-arm trace covers the geometric path
  from ``occupancy_well_arm`` to ``next_arm``
- ``chosen_patch_fraction`` — replay mass on ``next_patch``
- ``ends_at_next_patch`` — final MAP arm belongs to ``next_patch``

Each metric is compared against a within-session, visit-context future-label
shuffle over ``--n-shuffles`` draws. A visit context is a contiguous
``occupancy_well_arm`` run within ``stop_id`` because one long immobility bout can span
a trial boundary or return to an earlier arm. Arm metrics condition on current
arm, patch metrics on current patch, and path coverage on current arm, route
length, and coarse duration. Each family has its own cohort and denominator.
Output is one row per session with explicit retained/excluded unit and stratum
counts.

Usage::

    uv run python scripts/replay_models/choice/run_retrospective_alignment.py
    uv run python scripts/replay_models/choice/run_retrospective_alignment.py --n-shuffles 200
    uv run python scripts/replay_models/choice/run_retrospective_alignment.py --swr-only
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.events.analysed_time import require_swr_source
from continuum_swr_replay.analysis.events.cohort_selection import STOP_COHORT_RULES
from continuum_swr_replay.analysis.replay_models.choice.retrospective import (
    at_well_event_cohort,
    retrospective_alignment_by_session,
    select_event_subset,
)
from continuum_swr_replay.analysis.shared_statistics import session_median_statistics
from continuum_swr_replay.paths import replay_content_dir, table_path
from continuum_swr_replay.runs.fit_provenance import (
    record_inputs,
    write_fit_provenance,
)
from continuum_swr_replay.tables.covariate_registry import (
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)


def fit_input_paths() -> dict[str, Path]:
    """The tables this analysis is cut from, hashed into the provenance sidecar;
    under ``paths.replay_content_dir()``, read when called."""
    rc = replay_content_dir()
    return {
        "event_table": table_path("event_table", rc),
        "event_arm_table": table_path("event_arm_table", rc),
        "event_sequence_table": table_path("event_sequence_table", rc),
    }


#: The only population these metrics are defined on. A retrospective metric
#: asks what the animal did after a stop at a well, and needs the event's
#: current arm as an endpoint, so an in-transit event has nothing to compare.
#: The flag exists to make the cohort readable from the command line and the
#: sidecar, not to offer a choice.
STOP_COHORTS = ("at_well",)


def main(argv: Sequence[str] | None = None) -> None:
    input_dir = replay_content_dir()
    default_output = input_dir / "stage5b_retrospective.pkl"
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
        default=str(default_output),
        help="Output pickle path. Defaults next to the input tables.",
    )
    parser.add_argument(
        "--stop-cohort",
        type=str,
        default="at_well",
        choices=list(STOP_COHORTS),
        help=(
            "Population analysed (default, and only value: at_well). The "
            "event-grain half of the cohort the hippocampal rate and content "
            "ladders are read on; an in-transit event has no current arm to "
            "measure alignment from."
        ),
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="Analyse only the first N sessions (smoke runs; writes a partial table).",
    )
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(argv)

    if args.swr_only and args.non_swr_only:
        raise SystemExit("Pick at most one of --swr-only / --non-swr-only.")

    label = "all"
    if args.swr_only:
        label = "swr_only"
    elif args.non_swr_only:
        label = "non_swr_only"

    output_path = args.output
    if label != "all" and output_path == str(default_output):
        output_path = str(input_dir / f"stage5b_retrospective_{label}.pkl")

    logging.info("Loading canonical tables from %s", input_dir)
    inputs = fit_input_paths()
    event_table = read_table(inputs["event_table"])
    event_arm_table = read_table(inputs["event_arm_table"])
    event_sequence_table = read_table(inputs["event_sequence_table"])
    # Hash the tables while this run is reading them: a regeneration hours
    # later must not be certified as this analysis's input.
    fit_inputs = record_inputs(inputs)
    require_covariate_schema(event_table, table="event")
    require_covariate_schema(event_arm_table, table="event_arm")
    require_covariate_schema(event_sequence_table, table="event_sequence")
    # is_swr splits the events, so the tables must come from the configured
    # SWR source (an atomic set: all three carry its stamp).
    require_swr_source(event_table, what=str(inputs["event_table"]))
    require_swr_source(event_arm_table, what=str(inputs["event_arm_table"]))
    require_swr_source(event_sequence_table, what=str(inputs["event_sequence_table"]))
    logging.info(
        "events=%d  event_arm rows=%d  event_seq rows=%d  sessions=%d",
        len(event_table),
        len(event_arm_table),
        len(event_sequence_table),
        event_table.groupby(list(SESSION_COLUMNS)).ngroups,
    )
    n_before_location = len(event_table)
    event_table, event_arm_table, event_sequence_table = at_well_event_cohort(
        event_table, event_arm_table, event_sequence_table
    )
    n_after_location = len(event_table)
    logging.info(
        "Cohort %s (%s): retained %d/%d events",
        args.stop_cohort,
        STOP_COHORT_RULES[(args.stop_cohort, "event")],
        n_after_location,
        n_before_location,
    )

    event_table, event_arm_table, event_sequence_table = select_event_subset(
        event_table, event_arm_table, event_sequence_table, label
    )
    if label == "swr_only":
        logging.info("Filtered to SWR events: %d", len(event_table))
    elif label == "non_swr_only":
        logging.info("Filtered to non-SWR events: %d", len(event_table))

    out = retrospective_alignment_by_session(
        event_table,
        event_arm_table,
        event_sequence_table,
        n_shuffles=args.n_shuffles,
        seed=args.seed,
        subset=label,
        stop_cohort=args.stop_cohort,
        max_sessions=args.max_sessions,
    )
    n_sessions = len(out)
    n_no_movable = int(out["retrospective_status"].ne("ok").sum()) if n_sessions else 0
    out.to_pickle(output_path)
    sidecar = write_fit_provenance(
        output_path,
        fit_inputs,
        "retrospective_label_shuffle",
        [sys.argv[0], *argv],
        extra={
            "stop_cohort": args.stop_cohort,
            "cohort_grain": "event",
            "cohort_rule": STOP_COHORT_RULES[(args.stop_cohort, "event")],
            "n_events_before_cohort": n_before_location,
            "n_events_after_cohort": n_after_location,
            # Events left after the cohort AND the subset filter, over every
            # session in the table -- not restricted to the sessions a partial
            # run analysed, which `n_sessions_analysed` names.
            "n_events_in_subset": int(len(event_table)),
            "subset": label,
            "n_shuffles": args.n_shuffles,
            "seed": args.seed,
            "n_sessions_analysed": n_sessions,
            "is_partial_run": args.max_sessions is not None,
        },
    )
    logging.info("Saved %d sessions → %s", len(out), output_path)
    logging.info("Wrote provenance to %s", sidecar)
    logging.info(
        "Persisted %d/%d sessions with at least one non-movable metric family",
        n_no_movable,
        n_sessions,
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
        real_summary, diff_summary = (
            session_median_statistics(
                pd.to_numeric(out[column], errors="coerce"), out["animal"]
            )
            for column in (real_col, diff_col)
        )
        logging.info(
            "  %-25s  real=%.3f  Δ vs shuffle=%+.3f  pos=%d/%d",
            metric,
            real_summary["median"],
            diff_summary["median"],
            diff_summary["n_positive"],
            diff_summary["n_sessions"],
        )


if __name__ == "__main__":
    main()
