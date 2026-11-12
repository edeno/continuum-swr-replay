#!/usr/bin/env python
"""Audit a candidate stop table against its pre-change reference.

The reward-history basis columns are *added* to the stop table; nothing about
the stop definition changes. This script proves that: the shared keyed-parity
core (``tables.parity.diff_table``) checks every pre-existing column with the
four new ones declared as expected additions, so any drift in a pre-existing
column, any undeclared new column, and any row-count or key-set change raise.
On top of that this script pins what is specific to the stop table and to the
reward-history block: the on-disk row order and canonical column order, the
schema attrs, the feature range, the 0/1 indicator, the share of stops and
events the indicator flags, and the fact that the flagged stops form an
unbroken prefix of every (animal, session_date, epoch) ordered by ``stop_idx``
-- the flag marks the start-of-session stops that have fewer than two known
outcomes, so anything else would mean the outcome clock is not monotone.

Usage::

    uv run python scripts/checks/audit_stop_table_reward_history.py \
        --candidate Processed-Data/replay_content/stop_table.parquet \
        --reference Processed-Data/replay_content/_pre_fix_backup_20260904/stop_table.parquet
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.behavior.model_specs import (
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_INCOMPLETE,
)
from continuum_swr_replay.runs.io import sha256_file
from continuum_swr_replay.tables.covariate_registry import row_keys
from continuum_swr_replay.tables.io import read_table
from continuum_swr_replay.tables.parity import diff_table
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

DEFAULT_ADDED_COLUMNS = (*REWARD_HISTORY_COLUMNS, REWARD_HISTORY_INCOMPLETE)


def assert_row_and_column_order(
    candidate: pd.DataFrame, reference: pd.DataFrame, added: tuple[str, ...]
) -> None:
    """Raise unless the candidate preserves the reference's row/column order.

    ``diff_table`` sorts both sides on the row key, so it is blind to the order
    rows and columns are written in. That order is part of the canonical
    contract here -- ``stop_idx`` is positional and the column tuple is declared
    in ``_tables_common`` -- so it is pinned separately.
    """
    if not candidate["stop_id"].equals(reference["stop_id"]):
        raise ValueError("stop_id sequence differs (set or order)")
    kept = [column for column in candidate.columns if column not in set(added)]
    if kept != list(reference.columns):
        raise ValueError(
            "with the added columns removed, the candidate column order does not "
            f"match the reference: {kept} != {list(reference.columns)}"
        )


def incomplete_share(stop: pd.DataFrame) -> pd.DataFrame:
    """Return the stop- and event-weighted share of flagged stops, per animal."""
    flag = stop[REWARD_HISTORY_INCOMPLETE].to_numpy(dtype=float) == 1.0
    events = stop["n_selected_events"].to_numpy(dtype=float)
    frame = pd.DataFrame(
        {"animal": stop["animal"].to_numpy(), "flag": flag, "events": events}
    )

    def summarize(group: pd.DataFrame) -> pd.Series:
        flagged_events = group.loc[group["flag"], "events"].sum()
        return pd.Series(
            {
                "n_stops": len(group),
                "n_stops_incomplete": int(group["flag"].sum()),
                "pct_stops_incomplete": 100.0 * group["flag"].mean(),
                "n_events": group["events"].sum(),
                "n_events_incomplete": flagged_events,
                "pct_events_incomplete": 100.0 * flagged_events / group["events"].sum(),
            }
        )

    per_animal = frame.groupby("animal").apply(summarize, include_groups=False)
    overall = summarize(frame).to_frame("ALL").T
    return pd.concat([per_animal, overall])


def assert_prefix_per_session(stop: pd.DataFrame) -> int:
    """Raise unless flagged stops are an unbroken prefix of each session.

    Returns the number of sessions with at least one flagged stop.
    """
    violations = []
    n_sessions_with_flag = 0
    for keys, group in stop.groupby(list(SESSION_COLUMNS), sort=True):
        ordered = group.sort_values("stop_idx")
        flag = ordered[REWARD_HISTORY_INCOMPLETE].to_numpy(dtype=float) == 1.0
        n_flagged = int(flag.sum())
        if n_flagged:
            n_sessions_with_flag += 1
        if not flag[:n_flagged].all():
            violations.append((keys, n_flagged))
    if violations:
        raise ValueError(
            f"{len(violations)} session(s) where flagged stops are not a prefix; "
            f"sample={violations[:5]}"
        )
    return n_sessions_with_flag


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--added-columns",
        nargs="*",
        default=list(DEFAULT_ADDED_COLUMNS),
        help=(
            "columns the candidate carries that the reference predates; "
            "defaults to the reward-history basis and its indicator"
        ),
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args(argv)

    added = tuple(args.added_columns)
    candidate = read_table(args.candidate)
    reference = read_table(args.reference)

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit(f"candidate : {args.candidate}  sha256={sha256_file(args.candidate)}")
    emit(f"reference : {args.reference}  sha256={sha256_file(args.reference)}")
    emit(f"rows      : candidate {len(candidate)} | reference {len(reference)}")
    emit(
        f"columns   : candidate {len(candidate.columns)} | "
        f"reference {len(reference.columns)} | added {list(added)}"
    )
    emit(f"attrs candidate: {candidate.attrs}")
    emit(f"attrs reference: {reference.attrs}")
    if candidate.attrs != reference.attrs:
        raise ValueError("schema attrs differ between candidate and reference")

    assert_row_and_column_order(candidate, reference, added)
    emit("order     : stop_id sequence and pre-existing column order preserved")

    # Raises on any drift in a pre-existing column, on any candidate-only column
    # outside `added`, and on any row-count or key-set change.
    diff_table(
        candidate,
        reference,
        keys=list(row_keys(table="stop")),
        label="stop table vs reference",
        expected_added_columns=added,
        atol=0.0,
    )
    emit(
        f"parity    : all {len(reference.columns)} pre-existing columns identical "
        "(exact, NaN positions included)"
    )

    emit()
    emit("added columns:")
    emit(candidate[list(added)].describe().T.to_string())
    features = candidate[list(REWARD_HISTORY_COLUMNS)]
    if features.isna().any().any():
        raise ValueError("reward-history features contain NaN")
    if not ((features >= 0.0) & (features <= 1.0)).all().all():
        raise ValueError("reward-history features outside [0, 1]")
    if not candidate[REWARD_HISTORY_INCOMPLETE].isin([0.0, 1.0]).all():
        raise ValueError("reward_history_incomplete is not 0/1")
    emit("features in [0, 1], no NaN; indicator is 0/1")

    emit()
    emit(incomplete_share(candidate).to_string())

    n_sessions_with_flag = assert_prefix_per_session(candidate)
    n_sessions = candidate[list(SESSION_COLUMNS)].drop_duplicates().shape[0]
    emit(
        f"prefix    : flagged stops are an unbroken prefix in all {n_sessions} "
        f"session(s); {n_sessions_with_flag} session(s) carry at least one flag"
    )

    if args.report is not None:
        args.report.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
