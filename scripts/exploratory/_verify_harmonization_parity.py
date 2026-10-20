#!/usr/bin/env python
"""Fail-closed parity gate for the atomic covariate-name migration.

The baseline must be the immediate post-harmonization, pre-rename artifact set,
not an older scientific-definition backup.  Each baseline frame is normalized
exactly once through the table-scoped registry; the regenerated frame must
already carry the current covariate contract stamp. The checker then requires identical row
and session keys, columns, dtypes, missingness, and values.

No scientific change is allowlisted here.  Alias collapse and qvals numerical-
path consolidation are themselves validated by ``finalize_covariate_schema``.
Any remaining difference is a rename-migration failure.

``finalize_covariate_schema`` translates the ``qvals`` grain only -- the rest of
the rename ledger is the appendix of
``docs/design/2026-08-21-covariate-naming-scheme.md``, not executable code -- so
a baseline at any other grain must already carry canonical names.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.tables.covariate_registry import (
    finalize_covariate_schema,
    require_covariate_schema,
    row_keys,
)
from continuum_swr_replay.tables.io import read_table, table_exists

BASELINE_ROOT = Path(
    os.environ.get(
        "COVARIATE_RENAME_BASELINE",
        "/tmp/continuum-covariate-rename-baseline-20260822",
    )
)
RC = PROCESSED_DATA_DIR / "replay_content"
CF = PROCESSED_DATA_DIR / "contfrag_replay_content"

# label, registry table/grain, immediate pre-rename baseline, regenerated table
TABLES: tuple[tuple[str, str, Path, Path], ...] = (
    (
        "replay event",
        "event",
        BASELINE_ROOT / "replay/event_table.parquet",
        RC / "event_table.pkl",
    ),
    (
        "replay event_arm",
        "event_arm",
        BASELINE_ROOT / "replay/event_arm_table.parquet",
        RC / "event_arm_table.pkl",
    ),
    (
        "replay event_patch",
        "event_patch",
        BASELINE_ROOT / "replay/event_patch_table.parquet",
        RC / "event_patch_table.pkl",
    ),
    (
        "replay stop",
        "stop",
        BASELINE_ROOT / "replay/stop_table.parquet",
        RC / "stop_table.pkl",
    ),
    (
        "replay qvals",
        "qvals",
        BASELINE_ROOT / "replay/qvals_table.parquet",
        RC / "models_per_bin_rate/qvals_table.pkl",
    ),
    (
        "contfrag SWR event_arm",
        "event_arm",
        BASELINE_ROOT / "contfrag/event_arm_table_swr_HPC.parquet",
        CF / "event_arm_table_swr_HPC.pkl",
    ),
    (
        "contfrag SWR event_patch",
        "event_patch",
        BASELINE_ROOT / "contfrag/event_patch_table_swr_HPC.parquet",
        CF / "event_patch_table_swr_HPC.pkl",
    ),
    (
        "contfrag SWR stop",
        "stop",
        BASELINE_ROOT / "contfrag/stop_table_swr_HPC.parquet",
        CF / "stop_table_swr_HPC.pkl",
    ),
    (
        "contfrag MUA event_arm",
        "event_arm",
        BASELINE_ROOT / "contfrag/event_arm_table_mua_HPC.parquet",
        CF / "event_arm_table_mua_HPC.pkl",
    ),
    (
        "contfrag MUA event_patch",
        "event_patch",
        BASELINE_ROOT / "contfrag/event_patch_table_mua_HPC.parquet",
        CF / "event_patch_table_mua_HPC.pkl",
    ),
    (
        "contfrag MUA stop",
        "stop",
        BASELINE_ROOT / "contfrag/stop_table_mua_HPC.parquet",
        CF / "stop_table_mua_HPC.pkl",
    ),
)


def _infer_table(label: str) -> str:
    for table in ("event_arm", "event_patch", "stop", "event", "qvals"):
        if table in label:
            return table
    raise ValueError(f"cannot infer registry table/grain from label {label!r}")


def _exact_series_equal(old: pd.Series, new: pd.Series) -> tuple[bool, str]:
    if old.dtype != new.dtype:
        return False, f"dtype {old.dtype} -> {new.dtype}"
    if not np.array_equal(old.isna().to_numpy(), new.isna().to_numpy()):
        return False, "missingness mask differs"
    if old.equals(new):
        return True, ""
    mismatch = ~(old.eq(new) | (old.isna() & new.isna()))
    return False, f"{int(mismatch.sum())} value(s) differ"


def verify_table(
    label: str,
    baseline_path: Path,
    regenerated_path: Path,
    *,
    table: str | None = None,
) -> bool:
    """Compare one baseline/regenerated pair on exact canonical row keys."""
    table = table or _infer_table(label)
    baseline = finalize_covariate_schema(read_table(baseline_path), table=table)
    regenerated = read_table(regenerated_path)
    try:
        require_covariate_schema(regenerated, table=table)
    except (TypeError, ValueError) as exc:
        print(f"\n=== {label} ===\n  [FAIL] regenerated schema: {exc}")
        return False

    keys = list(row_keys(table=table))
    print(
        f"\n=== {label} === baseline={baseline.shape} regenerated={regenerated.shape}"
    )
    ok = True
    if baseline.duplicated(keys).any() or regenerated.duplicated(keys).any():
        print("  [FAIL] duplicate scientific row keys")
        return False

    old = baseline.set_index(keys, verify_integrity=True).sort_index()
    new = regenerated.set_index(keys, verify_integrity=True).sort_index()
    dropped = old.index.difference(new.index)
    added = new.index.difference(old.index)
    if len(dropped) or len(added):
        print(f"  [FAIL] row keys: dropped={len(dropped)}, added={len(added)}")
        ok = False
    else:
        print(f"  row keys: {len(old)} exact [ok]")

    session_cols = [c for c in ("animal", "session_date", "epoch") if c in baseline]
    if session_cols:
        old_sessions = (
            baseline[session_cols].drop_duplicates().set_index(session_cols).index
        )
        new_sessions = (
            regenerated[session_cols].drop_duplicates().set_index(session_cols).index
        )
        lost_sessions = old_sessions.difference(new_sessions)
        gained_sessions = new_sessions.difference(old_sessions)
        if len(lost_sessions) or len(gained_sessions):
            print(
                f"  [FAIL] sessions: dropped={len(lost_sessions)}, "
                f"added={len(gained_sessions)}"
            )
            ok = False
        else:
            print(f"  sessions: {len(old_sessions)} exact [ok]")

    old_columns = list(old.columns)
    new_columns = list(new.columns)
    if old_columns != new_columns:
        print(
            "  [FAIL] canonical columns/order differ: "
            f"removed={sorted(set(old_columns) - set(new_columns))}, "
            f"added={sorted(set(new_columns) - set(old_columns))}"
        )
        return False

    common = old.index.intersection(new.index)
    failures: list[str] = []
    for column in old_columns:
        equal, detail = _exact_series_equal(
            old.loc[common, column], new.loc[common, column]
        )
        if not equal:
            failures.append(f"{column} ({detail})")
    if failures:
        print(f"  [FAIL] {len(failures)} column(s) differ: {failures}")
        ok = False
    else:
        print(f"  values/dtypes/missingness: {len(old_columns)} columns exact [ok]")
    return ok


def main() -> int:
    all_ok = True
    n_verified = 0
    for label, table, baseline_path, regenerated_path in TABLES:
        if not table_exists(baseline_path) or not table_exists(regenerated_path):
            print(
                f"\n=== {label} === [FAIL] missing baseline or regenerated table: "
                f"{baseline_path}, {regenerated_path}"
            )
            all_ok = False
            continue
        n_verified += 1
        all_ok &= verify_table(
            label,
            baseline_path,
            regenerated_path,
            table=table,
        )
    if n_verified == 0:
        print("\n[FAIL] no baseline/regenerated table pair was verified")
        all_ok = False
    print("\n" + "=" * 60)
    print(
        f"COVARIATE RENAME PARITY ({n_verified}/{len(TABLES)} tables):",
        "PASS" if all_ok else "FAIL",
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
