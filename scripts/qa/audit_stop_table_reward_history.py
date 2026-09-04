#!/usr/bin/env python
"""Audit a candidate stop table against its pre-change reference.

The reward-history basis columns are *added* to the stop table; nothing about
the stop definition changes. This script proves that: every pre-existing column
must be identical to the reference (same rows, same ``stop_id`` sequence, same
dtypes, same values including NaN positions, same schema attrs), and the only
difference is the declared set of added columns.

It also characterizes the added structural-missingness flag: the share of stops
and of events (weighted by ``n_selected_events``) with
``reward_history_incomplete == 1``, overall and per animal, and asserts that the
flagged stops form an unbroken prefix of every (animal, session_date, epoch)
ordered by ``stop_idx`` -- the flag marks the start-of-session stops that have
fewer than two known outcomes, so anything else would mean the outcome clock is
not monotone.

Usage::

    uv run python scripts/qa/audit_stop_table_reward_history.py \
        --candidate Processed-Data/replay_content/_staging/stop_table.parquet \
        --reference Processed-Data/replay_content/_pre_fix_backup_20260904/stop_table.parquet
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay.model_specs import (
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_INCOMPLETE,
)
from continuum_swr_replay.table_io import read_table

SESSION_KEYS = ["animal", "session_date", "epoch"]


def sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of ``path``."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_preexisting_columns(
    candidate: pd.DataFrame, reference: pd.DataFrame, added: tuple[str, ...]
) -> pd.DataFrame:
    """Diff every pre-existing column positionally; return the per-column report.

    Rows are compared in file order (not re-sorted): a reordered table is a
    different artifact even if it holds the same values, because ``stop_idx``
    and the downstream joins are written in this order.
    """
    extra = tuple(c for c in candidate.columns if c not in reference.columns)
    missing = tuple(c for c in reference.columns if c not in candidate.columns)
    if missing:
        raise ValueError(f"candidate is missing reference column(s): {missing}")
    if set(extra) != set(added):
        raise ValueError(
            f"candidate-only columns {sorted(extra)} != declared added columns "
            f"{sorted(added)}"
        )
    if len(candidate) != len(reference):
        raise ValueError(f"{len(candidate)} rows vs reference {len(reference)}")
    kept = [column for column in candidate.columns if column not in set(added)]
    if kept != list(reference.columns):
        # The canonical column order is part of the contract: dropping the added
        # columns must recover the reference order exactly (the new columns are
        # inserted in the middle of the tuple, not appended).
        raise ValueError(
            "with the added columns removed, the candidate column order does not "
            f"match the reference: {kept} != {list(reference.columns)}"
        )

    rows = []
    for column in reference.columns:
        cand = candidate[column]
        ref = reference[column]
        dtype_ok = cand.dtype == ref.dtype
        # Only float columns go through the float path. Integer dtypes -- and
        # ``stop_id`` is uint64 -- would lose precision above 2**53 in the cast,
        # so they are compared exactly alongside the object/boolean columns.
        if (
            pd.api.types.is_numeric_dtype(ref)
            and not pd.api.types.is_bool_dtype(ref)
            and not pd.api.types.is_integer_dtype(ref)
        ):
            cvals = cand.to_numpy(dtype=float, na_value=np.nan)
            rvals = ref.to_numpy(dtype=float, na_value=np.nan)
            nan_mismatch = int((np.isnan(cvals) != np.isnan(rvals)).sum())
            both = ~np.isnan(cvals) & ~np.isnan(rvals)
            max_abs = (
                float(np.abs(cvals[both] - rvals[both]).max()) if both.any() else 0.0
            )
            n_mismatch = int((cvals[both] != rvals[both]).sum())
        else:
            c_missing = cand.isna().to_numpy()
            r_missing = ref.isna().to_numpy()
            # Normalize every missing marker (NaN, None, pd.NA) to None first:
            # pd.NA propagates through ``!=`` and makes the comparison raise.
            c_obj = cand.astype(object).to_numpy()
            r_obj = ref.astype(object).to_numpy()
            c_obj[c_missing] = None
            r_obj[r_missing] = None
            nan_mismatch = int((c_missing != r_missing).sum())
            n_mismatch = int(((c_obj != r_obj) & ~(c_missing & r_missing)).sum())
            max_abs = float("nan")
        rows.append(
            {
                "column": column,
                "dtype": str(ref.dtype),
                "dtype_match": dtype_ok,
                "n_value_mismatch": n_mismatch,
                "n_nan_position_mismatch": nan_mismatch,
                "max_abs_diff": max_abs,
                "identical": dtype_ok and n_mismatch == 0 and nan_mismatch == 0,
            }
        )
    return pd.DataFrame(rows)


def incomplete_share(stop: pd.DataFrame) -> pd.DataFrame:
    """Return the stop- and event-weighted share of flagged stops, per animal."""
    flag = stop[REWARD_HISTORY_INCOMPLETE].to_numpy(dtype=float) == 1.0
    events = stop["n_selected_events"].to_numpy(dtype=float)
    frame = pd.DataFrame(
        {"animal": stop["animal"].to_numpy(), "flag": flag, "events": events}
    )
    per_animal = frame.groupby("animal").apply(
        lambda g: pd.Series(
            {
                "n_stops": len(g),
                "n_stops_incomplete": int(g["flag"].sum()),
                "pct_stops_incomplete": 100.0 * g["flag"].mean(),
                "n_events": g["events"].sum(),
                "n_events_incomplete": g.loc[g["flag"], "events"].sum(),
                "pct_events_incomplete": 100.0
                * g.loc[g["flag"], "events"].sum()
                / g["events"].sum(),
            }
        ),
        include_groups=False,
    )
    overall = pd.DataFrame(
        {
            "n_stops": [len(frame)],
            "n_stops_incomplete": [int(frame["flag"].sum())],
            "pct_stops_incomplete": [100.0 * frame["flag"].mean()],
            "n_events": [frame["events"].sum()],
            "n_events_incomplete": [frame.loc[frame["flag"], "events"].sum()],
            "pct_events_incomplete": [
                100.0 * frame.loc[frame["flag"], "events"].sum() / frame["events"].sum()
            ],
        },
        index=["ALL"],
    )
    return pd.concat([per_animal, overall])


def assert_prefix_per_session(stop: pd.DataFrame) -> int:
    """Raise unless flagged stops are an unbroken prefix of each session.

    Returns the number of sessions with at least one flagged stop.
    """
    violations = []
    n_sessions_with_flag = 0
    for keys, group in stop.groupby(SESSION_KEYS, sort=True):
        ordered = group.sort_values("stop_idx")
        flag = ordered[REWARD_HISTORY_INCOMPLETE].to_numpy(dtype=float) == 1.0
        n_flagged = int(flag.sum())
        if n_flagged:
            n_sessions_with_flag += 1
        if not flag[:n_flagged].all():
            violations.append((keys, n_flagged, int(np.flatnonzero(~flag).min())))
    if violations:
        raise ValueError(
            f"{len(violations)} session(s) where flagged stops are not a prefix; "
            f"sample={violations[:5]}"
        )
    return n_sessions_with_flag


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    added = (*REWARD_HISTORY_COLUMNS, REWARD_HISTORY_INCOMPLETE)
    candidate = read_table(args.candidate)
    reference = read_table(args.reference)

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit(f"candidate : {args.candidate}  sha256={sha256(args.candidate)}")
    emit(f"reference : {args.reference}  sha256={sha256(args.reference)}")
    emit(f"rows      : candidate {len(candidate)} | reference {len(reference)}")
    emit(
        f"columns   : candidate {len(candidate.columns)} | "
        f"reference {len(reference.columns)} | added {list(added)}"
    )
    emit(f"attrs candidate: {candidate.attrs}")
    emit(f"attrs reference: {reference.attrs}")
    if candidate.attrs != reference.attrs:
        raise ValueError("schema attrs differ between candidate and reference")

    if not candidate["stop_id"].equals(reference["stop_id"]):
        raise ValueError("stop_id sequence differs (set or order)")
    emit("stop_id   : identical set AND order")

    report = compare_preexisting_columns(candidate, reference, added)
    n_bad = int((~report["identical"]).sum())
    emit(
        f"parity    : {int(report['identical'].sum())}/{len(report)} pre-existing "
        f"columns identical"
    )
    if n_bad:
        emit(report.loc[~report["identical"]].to_string(index=False))
        raise ValueError(f"{n_bad} pre-existing column(s) drifted")

    emit()
    emit("added columns:")
    describe = candidate[list(added)].describe().T
    emit(describe.to_string())
    features = candidate[list(REWARD_HISTORY_COLUMNS)]
    if features.isna().any().any():
        raise ValueError("reward-history features contain NaN")
    if not ((features >= 0.0) & (features <= 1.0)).all().all():
        raise ValueError("reward-history features outside [0, 1]")
    if not candidate[REWARD_HISTORY_INCOMPLETE].isin([0.0, 1.0]).all():
        raise ValueError("reward_history_incomplete is not 0/1")
    emit("features in [0, 1], no NaN; indicator is 0/1")

    emit()
    shares = incomplete_share(candidate)
    emit(shares.to_string())

    n_sessions_with_flag = assert_prefix_per_session(candidate)
    n_sessions = candidate[SESSION_KEYS].drop_duplicates().shape[0]
    emit(
        f"prefix    : flagged stops are an unbroken prefix in all {n_sessions} "
        f"session(s); {n_sessions_with_flag} session(s) carry at least one flag"
    )

    if args.report is not None:
        args.report.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
