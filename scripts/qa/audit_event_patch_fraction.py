#!/usr/bin/env python
"""Audit a candidate event-patch table against its pre-change reference.

``patch_fraction`` is *added* to the event-patch table; nothing about the patch
definition or any pre-existing aggregate changes. This script proves that: the
shared keyed-parity core (``tables.parity.diff_table``) checks every pre-existing
column with the new one declared as an expected addition, so any drift in a
pre-existing column, any undeclared new column, and any row-count or key-set
change raise. On top of that this script pins what is specific to the patch
table and to the fraction contract: the on-disk row order and column order, the
schema attrs, and the three scientific properties the fraction must satisfy --
it equals ``patch_count_hard`` over the event's real-arm bin total
(``n_nonlocal_bins - n_undecoded_bins``, read from the event table), it sums to
one over the three patches of every event that has at least one real-arm bin and
is NaN for all three patches otherwise, and it reproduces the event-arm table's
``arm_fraction`` summed over each patch's two leaf arms plus its trunk.

Usage::

    uv run python scripts/qa/audit_event_patch_fraction.py \
        --candidate Processed-Data/replay_content/_staging/event_patch_table.parquet \
        --reference Processed-Data/replay_content/_pre_fix_backup_20260905/event_patch_table.parquet \
        --event-table Processed-Data/replay_content/event_table.parquet \
        --event-arm-table Processed-Data/replay_content/event_arm_table.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH
from continuum_swr_replay.tables.covariate_registry import row_keys
from continuum_swr_replay.tables.io import read_table, sha256_file
from continuum_swr_replay.tables.parity import diff_table

DEFAULT_ADDED_COLUMNS = ("patch_fraction",)
# patch_fraction divides an exact integer count by an exact integer total, so
# the only admissible error is one float64 rounding step per summand.
FRACTION_ATOL = 1e-12


def assert_row_and_column_order(
    candidate: pd.DataFrame, reference: pd.DataFrame, added: tuple[str, ...]
) -> None:
    """Raise unless the candidate preserves the reference's row/column order.

    ``diff_table`` sorts both sides on the row key, so it is blind to the order
    rows and columns are written in. The patch table's row order follows the
    event order of the canonical event table, and its column order is the
    builder's construction order, so both are pinned separately here.
    """
    key = ["event_id", "patch"]
    if (
        not candidate[key]
        .reset_index(drop=True)
        .equals(reference[key].reset_index(drop=True))
    ):
        raise ValueError("(event_id, patch) row sequence differs (set or order)")
    kept = [column for column in candidate.columns if column not in set(added)]
    if kept != list(reference.columns):
        raise ValueError(
            "with the added columns removed, the candidate column order does not "
            f"match the reference: {kept} != {list(reference.columns)}"
        )


def real_arm_bins(event_table: pd.DataFrame) -> pd.Series:
    """Return the per-event count of in-event bins whose MAP hit a real arm.

    Parameters
    ----------
    event_table : pd.DataFrame
        Canonical event table, one row per ``event_id``.

    Returns
    -------
    pd.Series
        ``n_nonlocal_bins - n_undecoded_bins`` indexed by ``event_id``.
    """
    return (
        pd.to_numeric(event_table["n_nonlocal_bins"], errors="coerce")
        - pd.to_numeric(event_table["n_undecoded_bins"], errors="coerce")
    ).set_axis(event_table["event_id"])


def assert_fraction_definition(
    patch: pd.DataFrame, event_table: pd.DataFrame
) -> dict[str, int]:
    """Raise unless ``patch_fraction`` is the count over the real-arm bin total.

    Returns the count of events with and without real-arm bins.
    """
    denominator = (
        real_arm_bins(event_table).reindex(patch["event_id"]).to_numpy(dtype=float)
    )
    counts = patch["patch_count_hard"].to_numpy(dtype=float)
    observed = patch["patch_fraction"].to_numpy(dtype=float)
    defined = denominator > 0
    expected = np.full(observed.shape, np.nan)
    expected[defined] = counts[defined] / denominator[defined]

    if not np.array_equal(np.isnan(observed), np.isnan(expected)):
        raise ValueError("patch_fraction NaN positions do not match the definition")
    max_abs = float(np.nanmax(np.abs(observed[defined] - expected[defined])))
    if max_abs > FRACTION_ATOL:
        raise ValueError(
            f"patch_fraction differs from patch_count_hard / real-arm bins by "
            f"{max_abs:.3e} (> {FRACTION_ATOL:.0e})"
        )
    if ((observed[defined] < 0.0) | (observed[defined] > 1.0)).any():
        raise ValueError("patch_fraction outside [0, 1]")

    totals = patch.groupby("event_id", sort=False)["patch_fraction"].sum(min_count=1)
    support = patch.groupby("event_id", sort=False)["patch_fraction"].apply(
        lambda values: int(np.isfinite(values.to_numpy(dtype=float)).sum())
    )
    partial = support[(support != 0) & (support != 3)]
    if not partial.empty:
        raise ValueError(
            "patch_fraction has partial patch support for event_id(s) "
            f"{partial.index.tolist()[:5]}"
        )
    finite_totals = totals.to_numpy(dtype=float)
    unit = np.isfinite(finite_totals)
    if not np.allclose(finite_totals[unit], 1.0, rtol=0.0, atol=FRACTION_ATOL):
        worst = float(np.max(np.abs(finite_totals[unit] - 1.0)))
        raise ValueError(f"patch_fraction sums deviate from one by up to {worst:.3e}")
    return {
        "n_events": int(len(totals)),
        "n_events_with_real_arm_bins": int(unit.sum()),
        "n_events_without_real_arm_bins": int((~unit).sum()),
        "max_abs_sum_error": float(
            np.max(np.abs(finite_totals[unit] - 1.0)) if unit.any() else 0.0
        ),
    }


def assert_matches_arm_fractions(
    patch: pd.DataFrame, event_arm_table: pd.DataFrame
) -> float:
    """Raise unless each patch fraction equals the sum over its three arms.

    Returns the maximum absolute discrepancy.
    """
    arm_patch = event_arm_table["candidate_arm"].map(TRACK_SEGMENT_TO_PATCH)
    per_patch = (
        event_arm_table.groupby(
            [event_arm_table["event_id"], arm_patch.rename("patch")], sort=False
        )["arm_fraction"]
        .sum(min_count=1)
        .rename("arm_fraction_patch_sum")
        .reset_index()
    )
    merged = patch[["event_id", "patch", "patch_fraction"]].merge(
        per_patch, on=["event_id", "patch"], how="left", validate="one_to_one"
    )
    if len(merged) != len(patch):
        raise ValueError("patch rows do not map one-to-one onto arm-table patches")
    observed = merged["patch_fraction"].to_numpy(dtype=float)
    expected = merged["arm_fraction_patch_sum"].to_numpy(dtype=float)
    if not np.array_equal(np.isnan(observed), np.isnan(expected)):
        raise ValueError(
            "patch_fraction and the summed arm_fraction disagree on NaN positions"
        )
    max_abs = float(np.nanmax(np.abs(observed - expected)))
    if max_abs > FRACTION_ATOL:
        raise ValueError(
            f"patch_fraction differs from its arms' summed arm_fraction by {max_abs:.3e}"
        )
    return max_abs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument(
        "--event-table",
        type=Path,
        required=True,
        help="canonical event table supplying the fraction denominator",
    )
    parser.add_argument(
        "--event-arm-table",
        type=Path,
        required=True,
        help="canonical event-arm table supplying arm_fraction for the cross-check",
    )
    parser.add_argument(
        "--added-columns",
        nargs="*",
        default=list(DEFAULT_ADDED_COLUMNS),
        help=(
            "columns the candidate carries that the reference predates; "
            "defaults to the patch allocation fraction"
        ),
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    added = tuple(args.added_columns)
    candidate = read_table(args.candidate)
    reference = read_table(args.reference)
    event_table = read_table(args.event_table)
    event_arm_table = read_table(args.event_arm_table)

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
    emit("order     : (event_id, patch) sequence and pre-existing column order kept")

    # Raises on any drift in a pre-existing column, on any candidate-only column
    # outside `added`, and on any row-count or key-set change.
    diff_table(
        candidate,
        reference,
        keys=list(row_keys(table="event_patch")),
        label="event_patch table vs reference",
        expected_added_columns=added,
        atol=0.0,
    )
    emit(
        f"parity    : all {len(reference.columns)} pre-existing columns identical "
        "(exact, NaN positions included)"
    )

    emit()
    emit("added column:")
    emit(candidate[list(added)].describe().T.to_string())

    definition = assert_fraction_definition(candidate, event_table)
    emit(
        "definition: patch_fraction == patch_count_hard / "
        "(n_nonlocal_bins - n_undecoded_bins) exactly; "
        f"{definition['n_events_with_real_arm_bins']} of {definition['n_events']} "
        "event(s) have real-arm bins and sum to one "
        f"(max |sum - 1| = {definition['max_abs_sum_error']:.3e}); "
        f"{definition['n_events_without_real_arm_bins']} event(s) are all-NaN"
    )

    max_arm_gap = assert_matches_arm_fractions(candidate, event_arm_table)
    emit(
        "hierarchy : patch_fraction equals the sum of its two leaf arms' and its "
        f"trunk's arm_fraction (max |difference| = {max_arm_gap:.3e})"
    )

    if args.report is not None:
        args.report.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
