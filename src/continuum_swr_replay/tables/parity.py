"""Keyed table-to-table parity checks shared across the pipeline.

One implementation of "does this table match that table, row for row" so the
regeneration manifest's baseline gate and the golden-baseline refactor check
speak the same language instead of forking parity engines. A table is compared
against a reference on its registered row key: structural drift (differing key
set, row count, or schema) always raises; per-column value drift raises unless
the column is in an explicitly declared allowlist.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.paths import BASENAME_TO_TABLE, CANONICAL_TABLE_BASENAMES
from continuum_swr_replay.tables.covariate_registry import row_keys
from continuum_swr_replay.tables.io import read_table

__all__ = [
    "diff_table",
    "compare_table_roots",
]


def diff_table(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    keys: list[str],
    label: str,
    expected_drift_columns: frozenset[str] = frozenset(),
    expected_added_columns: Sequence[str] = (),
    atol: float = 1e-9,
) -> dict[str, object]:
    """Diff `candidate` against `baseline` on `keys`; raise on undeclared drift.

    Structural drift (column set, row count, key set) always raises. Float
    columns are compared with absolute tolerance `atol` (0.0 for exact) and
    must agree on NaN positions; integer, boolean, and object columns are
    compared exactly, because a float cast silently loses integer precision
    above 2**53 (`stop_id` is a uint64 hash, so almost every value is above it).
    A column that drifts and is not in `expected_drift_columns` raises. Returns
    the per-column drift report.

    `expected_added_columns` names columns a *new* table version may carry that
    the baseline does not -- the case where a pipeline gains a covariate and the
    frozen baseline must stay frozen. Those columns are excluded from the
    comparison; any other candidate-only column, and any baseline-only column,
    still raises. Declaring a column that the candidate does not actually carry
    is not an error, so one declaration can cover a multi-table gate where only
    one table gained the columns.
    """
    added = set(expected_added_columns)
    candidate_only = set(candidate.columns) - set(baseline.columns)
    baseline_only = set(baseline.columns) - set(candidate.columns)
    if baseline_only or (candidate_only - added):
        raise ValueError(
            f"{label}: column set differs from baseline "
            f"(candidate-only={sorted(candidate_only - added)}, "
            f"baseline-only={sorted(baseline_only)}); "
            f"declared additions={sorted(added)}"
        )
    candidate = candidate.drop(columns=sorted(candidate_only))
    if len(candidate) != len(baseline):
        raise ValueError(f"{label}: {len(candidate)} rows vs baseline {len(baseline)}")
    cand = candidate.set_index(keys).sort_index()
    base = baseline.set_index(keys).sort_index()
    if not cand.index.equals(base.index):
        raise ValueError(f"{label}: {keys} key set differs from baseline")
    base = base[cand.columns]
    drifted: dict[str, object] = {}
    for column in cand.columns:
        cseries = cand[column]
        bseries = base[column]
        if pd.api.types.is_float_dtype(cseries) and pd.api.types.is_float_dtype(
            bseries
        ):
            cvals = cseries.to_numpy(dtype=float, na_value=np.nan)
            bvals = bseries.to_numpy(dtype=float, na_value=np.nan)
            nan_mismatch = int((np.isnan(cvals) != np.isnan(bvals)).sum())
            diff = np.abs(cvals - bvals)
            max_abs = float(np.nanmax(diff)) if diff.size else 0.0
            if max_abs > atol or nan_mismatch:
                drifted[column] = {
                    "max_abs_diff": max_abs,
                    "nan_position_mismatch": nan_mismatch,
                }
        else:
            # Integer/boolean/object: compared exactly. Integers must not go
            # through the float path -- above 2**53 the cast is lossy and a
            # one-unit change compares equal. NaN, None, and pd.NA all read as
            # missing and must compare equal to each other, so missing entries
            # are normalized to None first (pd.NA also propagates through `!=`
            # and would make the comparison raise).
            c_missing = cseries.isna().to_numpy()
            b_missing = bseries.isna().to_numpy()
            c_obj = cseries.astype(object).to_numpy()
            b_obj = bseries.astype(object).to_numpy()
            c_obj[c_missing] = None
            b_obj[b_missing] = None
            mismatch = int((c_obj != b_obj).sum())
            missing_mismatch = int((c_missing != b_missing).sum())
            if mismatch:
                drifted[column] = {
                    "n_mismatch": mismatch,
                    "missing_position_mismatch": missing_mismatch,
                }
    undeclared = sorted(set(drifted) - expected_drift_columns)
    if undeclared:
        raise ValueError(
            f"{label}: drifted from baseline in undeclared columns {undeclared}; "
            "declare them if the change is intended, otherwise this is a regression"
        )
    return drifted


def compare_table_roots(
    candidate_root: Path,
    baseline_root: Path,
    table_basenames: tuple[str, ...] = CANONICAL_TABLE_BASENAMES,
    *,
    expected_drift_columns: frozenset[str] = frozenset(),
    expected_added_columns: Sequence[str] = (),
    atol: float = 1e-9,
    read_candidate: Callable[[str], pd.DataFrame] | None = None,
    read_baseline: Callable[[str], pd.DataFrame] | None = None,
) -> dict[str, dict[str, object]]:
    """Diff every named canonical table under two roots, keyed by its row key.

    Each side is read by its own basename-keyed reader, so two roots with
    different on-disk layouts (e.g. a candidate that nests ``qvals_table`` under
    ``models_per_bin_rate/`` and a flat baseline) are expressed where the
    asymmetry lives rather than inferred from a path prefix. Both default to
    ``read_table(root / basename)``, the flat layout of a frozen baseline. A
    pipeline root nests ``qvals_table`` (``paths.table_path``), so read one
    with ``lambda name: read_table(table_path(name, root, fmt=""))``.

    A table absent under either root raises ``FileNotFoundError`` (via the
    reader) rather than being skipped -- a partial baseline is a hard error, not
    a pass. Raises ``ValueError`` on the first table that drifts in an undeclared
    column.

    ``expected_drift_columns`` applies to *every* table in ``table_basenames``;
    if you need to allow a column to move in one table but not another, call this
    per table rather than declaring it globally. ``expected_added_columns`` is
    global in the same way, but harmless to over-declare: a name the table does
    not carry is simply not matched, so declaring one table's new columns does
    not weaken the check on the others.
    """
    if read_candidate is None:
        read_candidate = lambda name: read_table(candidate_root / name)  # noqa: E731
    if read_baseline is None:
        read_baseline = lambda name: read_table(baseline_root / name)  # noqa: E731
    report: dict[str, dict[str, object]] = {}
    for basename in table_basenames:
        keys = list(row_keys(table=BASENAME_TO_TABLE[basename]))
        report[basename] = diff_table(
            read_candidate(basename),
            read_baseline(basename),
            keys=keys,
            label=f"baseline parity: {basename}",
            expected_drift_columns=expected_drift_columns,
            expected_added_columns=expected_added_columns,
            atol=atol,
        )
    return report
