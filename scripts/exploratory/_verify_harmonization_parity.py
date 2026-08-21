#!/usr/bin/env python
"""Key-aligned covariate-definition parity / verification gate.

Verifies that the covariate-definition migration (two-clock reward timing,
per-arm-rate window 5->10, new value block, and table regeneration) changed
*only* what it was meant to. Compares each regenerated canonical table against
its pre-rebuild backup
(``_prephase1_backup``) and checks four things:

1. rebuild integrity   - every shared, unchanged-definition column reproduces
   the backup (exact on int/bool/key columns, ``np.allclose`` on floats). Because
   every model design matrix is a deterministic function of these table columns
   (the fitters select/multiply columns; arm-FE / spline / raised-cosine bases are
   deterministic from unchanged inputs), column identity here *is* the
   design-matrix identity for the unchanged-definition covariates.
2. rename-only identity - reserved for actual naming-only changes. The trial
   reward rate is deliberately *not* in this category because its availability
   now follows the outcome clock rather than destination arrival.
3. changed covariates   - trial/per-arm reward rates, nominal/learned values,
   their interactions, and patch analogues: old->new summary printed, no
   equality assertion (these are *meant* to change).
4. coverage             - added/dropped keys and sessions are reported explicitly;
   unequal cohorts are aligned on scientific row keys rather than compared by
   position.

Exit non-zero if any asserted check fails. Run from the repo root:
    uv run python scripts/exploratory/_verify_harmonization_parity.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.table_io import read_table, table_exists

RC = PROCESSED_DATA_DIR / "replay_content"
CF = PROCESSED_DATA_DIR / "contfrag_replay_content"
ATOL, RTOL = 1e-9, 1e-6

# old name -> new name for genuine naming-only changes. This is intentionally
# empty: reward_rate_trial_k10 also adopted the outcome-availability clock.
RENAME_ONLY: dict[str, str] = {}
CHANGED = {
    "recent_trial_reward_rate": "reward_rate_trial_k10",  # timing + rename
    "recent_reward_rate": "reward_rate_arm_k10",  # 5- -> 10-visit window
    "nominal_reward_prob": "p_reward_arm",  # deferred-NaN -> populated
    "swr_x_recent_reward_rate": "swr_x_reward_rate_arm_k10",  # derived from changed
}
# New columns with no backup counterpart (value block).
NEW_COLS = {
    "q_arm", "q_arm_patch", "p_reward_arm_patch", "reward_rate_arm_k10_patch",
}
SESSION_COLS = ["animal", "day", "epoch"]

# Same-named columns whose definitions intentionally changed in the unified
# location, timing, reward-history, and whole-event classification migration.
# Every other shared column is equality-checked on exact scientific row keys.
# Keeping this allowlist explicit makes a newly drifting column fail the gate
# rather than disappear from a hand-picked eight-column "core".
INTENTIONALLY_CHANGED_SAME_NAME = frozenset(
    {
        # Whole-window physical location and all derived context contrasts.
        "current_arm",
        "current_patch",
        "prev_arm",
        "prev_patch",
        "next_arm",
        "next_patch",
        "last_rewarded_arm",
        "is_current",
        "is_prev",
        "is_next",
        "is_same_patch",
        "is_last_rewarded",
        "is_last_rewarded_elsewhere",
        "current_x_last_rewarded",
        "switch_x_current",
        "switch_x_prev",
        "omission_x_last_rewarded",
        "time_x_current",
        "time_x_prev",
        "time_x_next",
        # Arrival/outcome clocks and future-invariant history definitions.
        "reward_outcome",
        "is_after_reward",
        "is_after_omission",
        "is_after_switch",
        "is_after_stay",
        "is_stay",
        "cumulative_reward",
        "cumulative_reward_patch",
        "visit_recency",
        "visit_recency_demeaned",
        "visit_recency_patch",
        "swr_x_visit_recency_demeaned",
        "time_since_port_entry",
        "time_since_reward",
        *{f"reward_lag{j}" for j in range(1, 11)},
        # Physical-location distance/rank now uses the canonical mapping.
        "dist_to_arm",
        "arm_dist_rank",
        # Whole-event SWR/MUA overlap and dependent indicators/counts.
        "is_swr",
        "is_mua",
        "is_swr_stop",
        "is_mua_stop",
        "n_SWR",
        "n_MUA",
        "n_non_local_SWR",
        "n_non_local_no_SWR",
        "swr_x_current",
        "swr_x_prev",
        "swr_x_next",
        "swr_x_last_rewarded",
        "swr_x_is_last_rewarded_elsewhere",
        "mua_x_current",
        "mua_x_prev",
        "mua_x_next",
        "mua_x_last_rewarded",
        "continuous_x_current",
        "continuous_x_prev",
        "continuous_x_next",
        "continuous_x_last_rewarded",
        # Patch projections of the changed arm/context definitions.
        "is_current_patch",
        "is_prev_patch",
        "is_last_rewarded_patch",
        "is_last_rewarded_patch_elsewhere",
        "switch_x_current_patch",
        "swr_x_current_patch",
        "swr_x_prev_patch",
        "swr_x_last_rewarded_patch_elsewhere",
    }
)

# (label, backup_path, new_path)
TABLES = [
    ("replay event_arm",
     RC / "_prephase1_backup/event_arm_table.pkl", RC / "event_arm_table.pkl"),
    ("replay event_patch",
     RC / "_prephase1_backup/event_patch_table.pkl", RC / "event_patch_table.pkl"),
    ("replay stop",
     RC / "_prephase1_backup/stop_table.pkl", RC / "stop_table.pkl"),
    ("contfrag SWR event_arm",
     CF / "_prephase1_backup/event_arm_table_swr_HPC.pkl",
     CF / "event_arm_table_swr_HPC.pkl"),
    ("contfrag SWR event_patch",
     CF / "_prephase1_backup/event_patch_table_swr_HPC.pkl",
     CF / "event_patch_table_swr_HPC.pkl"),
    ("contfrag SWR stop",
     CF / "_prephase1_backup/stop_table_swr_HPC.pkl",
     CF / "stop_table_swr_HPC.pkl"),
    ("contfrag MUA event_arm",
     CF / "_prephase1_backup/event_arm_table_mua_HPC.pkl",
     CF / "event_arm_table_mua_HPC.pkl"),
    ("contfrag MUA event_patch",
     CF / "_prephase1_backup/event_patch_table_mua_HPC.pkl",
     CF / "event_patch_table_mua_HPC.pkl"),
    ("contfrag MUA stop",
     CF / "_prephase1_backup/stop_table_mua_HPC.pkl",
     CF / "stop_table_mua_HPC.pkl"),
]


def _col_reproduces(bs: np.ndarray, ns: np.ndarray, is_float: bool) -> bool:
    if is_float:
        return bool(np.allclose(
            bs.astype(float), ns.astype(float), atol=ATOL, rtol=RTOL, equal_nan=True
        ))
    return bool(
        (pd.Series(bs).fillna("<NA>").values == pd.Series(ns).fillna("<NA>").values).all()
    )


def _summarize(col: pd.Series) -> str:
    a = col.to_numpy(dtype=float)
    finite = np.isfinite(a)
    if not finite.any():
        return "all-NaN"
    return (f"NaN={np.mean(~finite) * 100:4.1f}%  "
            f"range=[{np.nanmin(a):.3f},{np.nanmax(a):.3f}]  mean={np.nanmean(a):.3f}")


def _row_keys(label: str, frame: pd.DataFrame) -> list[str]:
    if "event_arm" in label:
        candidates = [*SESSION_COLS, "event_id", "arm"]
    elif "event_patch" in label:
        candidates = [*SESSION_COLS, "event_id", "patch"]
    elif "stop" in label:
        candidates = [*SESSION_COLS, "stop_idx"]
    else:  # pragma: no cover - TABLES is a closed registry
        raise ValueError(f"no row-key contract declared for {label!r}")
    missing = [column for column in candidates if column not in frame]
    if missing:
        raise ValueError(f"{label} is missing row-key columns {missing}")
    return candidates


def verify_table(label: str, bpath, npath) -> bool:
    b = read_table(bpath)
    n = read_table(npath)
    ok = True
    print(f"\n=== {label} ===  backup {b.shape}  new {n.shape}")

    keys = _row_keys(label, b)
    if keys != _row_keys(label, n):  # pragma: no cover - deterministic
        raise AssertionError("backup/new row-key contracts diverged")
    if b.duplicated(keys).any() or n.duplicated(keys).any():
        print("  [FAIL] duplicate scientific row keys")
        return False

    b_keyed = b.set_index(keys, verify_integrity=True).sort_index()
    n_keyed = n.set_index(keys, verify_integrity=True).sort_index()
    common = b_keyed.index.intersection(n_keyed.index)
    dropped = b_keyed.index.difference(n_keyed.index)
    added = n_keyed.index.difference(b_keyed.index)
    shared_fraction = len(common) / max(max(len(b_keyed), len(n_keyed)), 1)
    print(
        f"  keys: shared={len(common)} ({shared_fraction:.1%} of larger table), "
        f"dropped={len(dropped)}, added={len(added)}"
    )
    if len(dropped) or len(added):
        print("  [FAIL] scientific row-key sets differ")
        ok = False

    # 4. coverage
    if len(b) != len(n):
        print(f"  [FAIL] row count {len(b)} -> {len(n)}")
        ok = False
    if set(SESSION_COLS).issubset(b.columns) and set(SESSION_COLS).issubset(n.columns):
        nb = b[SESSION_COLS].drop_duplicates().shape[0]
        nn = n[SESSION_COLS].drop_duplicates().shape[0]
        tag = "ok" if nb == nn else "FAIL"
        print(f"  coverage: sessions {nb} -> {nn} [{tag}]; rows {len(b)} -> {len(n)}")
        if nb != nn:
            ok = False

    # 1. rebuild integrity on key-aligned, unchanged-definition columns
    changed_targets = {
        new for old, new in CHANGED.items() if old != new
    } | set(RENAME_ONLY.values()) | NEW_COLS | set(INTENTIONALLY_CHANGED_SAME_NAME)
    shared_columns = sorted(
        (set(b.columns) & set(n.columns)) - set(keys) - changed_targets
    )
    expected_removed = {
        old for old, new in CHANGED.items() if old != new
    } | set(RENAME_ONLY)
    unexpected_removed = sorted(set(b.columns) - set(n.columns) - expected_removed)
    if unexpected_removed:
        ok = False
        print(f"  [FAIL] shared-definition columns disappeared: {unexpected_removed}")
    mism = []
    for c in shared_columns:
        is_float = (pd.api.types.is_float_dtype(b[c])
                    or pd.api.types.is_float_dtype(n[c]))
        if not _col_reproduces(
            b_keyed.loc[common, c].to_numpy(),
            n_keyed.loc[common, c].to_numpy(),
            is_float,
        ):
            mism.append(c)
    if mism:
        ok = False
        print(f"  [FAIL] rebuild-integrity: {len(mism)} shared col(s) differ: {mism}")
    else:
        print(
            f"  rebuild-integrity: {len(shared_columns)} key-aligned unchanged cols "
            f"reproduce [ok]: {shared_columns}"
        )

    # 2. rename-only identity (values must be byte-identical)
    for old, new in RENAME_ONLY.items():
        if old in b.columns and new in n.columns:
            is_float = pd.api.types.is_float_dtype(n[new])
            if _col_reproduces(
                b_keyed.loc[common, old].to_numpy(),
                n_keyed.loc[common, new].to_numpy(),
                is_float,
            ):
                print(f"  rename-only: {old} == {new} [ok]")
            else:
                ok = False
                print(f"  [FAIL] rename-only {old} -> {new} changed values")

    # 3. changed covariates: old -> new summary (no assert)
    for old, new in CHANGED.items():
        if old in b.columns and new in n.columns:
            print(f"  changed: {old} [{_summarize(b_keyed.loc[common, old])}]")
            print(f"        -> {new} [{_summarize(n_keyed.loc[common, new])}]")
    changed_same = sorted(
        set(b.columns) & set(n.columns) & set(INTENTIONALLY_CHANGED_SAME_NAME)
    )
    if changed_same:
        print(f"  changed-in-place ({len(changed_same)}): {changed_same}")

    # new value-block columns: present + finite-where-defined
    for c in sorted(NEW_COLS):
        if c in n.columns:
            print(f"  new: {c} [{_summarize(n[c])}]")
    return ok


def main() -> int:
    all_ok = True
    n_verified = 0
    for label, bpath, npath in TABLES:
        if not (table_exists(bpath) and table_exists(npath)):
            print(f"\n=== {label} ===  [skip] missing backup or new table")
            continue
        n_verified += 1
        all_ok &= verify_table(label, bpath, npath)
    if n_verified == 0:
        print("\n[FAIL] no backup/new table pair was verified")
        all_ok = False
    print("\n" + "=" * 60)
    print(
        f"HARMONIZATION PARITY ({n_verified}/{len(TABLES)} tables):",
        "PASS" if all_ok else "FAIL",
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
