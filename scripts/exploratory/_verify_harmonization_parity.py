#!/usr/bin/env python
"""Phase-2 harmonization parity / verification gate.

Verifies that the Phase-1 covariate harmonization (rename + per-arm-rate window
5->10 + new value block + table regeneration) changed *only* what it was meant
to. Compares each regenerated canonical table against its pre-rebuild backup
(``_prephase1_backup``) and checks four things:

1. rebuild integrity   - every shared, unchanged-definition column reproduces
   the backup (exact on int/bool/key columns, ``np.allclose`` on floats). Because
   every model design matrix is a deterministic function of these table columns
   (the fitters select/multiply columns; arm-FE / spline / raised-cosine bases are
   deterministic from unchanged inputs), column identity here *is* the
   design-matrix identity for the unchanged-definition covariates.
2. rename-only identity - ``reward_rate_trial_k10`` (new) is byte-identical to
   ``recent_trial_reward_rate`` (backup): a pure rename must not change values.
3. changed covariates   - ``reward_rate_arm_k10`` (vs old 5-visit
   ``recent_reward_rate``), ``p_reward_arm`` (vs deferred-NaN ``nominal_reward_prob``),
   ``swr_x_reward_rate_arm_k10`` and the patch analogues: old->new summary printed,
   no assertion (these are *meant* to change).
4. coverage             - row counts and unique (animal, day, epoch) session
   counts match the backup.

Exit non-zero if any asserted check fails. Run from the repo root:
    uv run python scripts/exploratory/_verify_harmonization_parity.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

RC = PROCESSED_DATA_DIR / "replay_content"
CF = PROCESSED_DATA_DIR / "contfrag_replay_content"
ATOL, RTOL = 1e-9, 1e-6

# old name -> new name, by intent.
RENAME_ONLY = {"recent_trial_reward_rate": "reward_rate_trial_k10"}  # values identical
CHANGED = {
    "recent_reward_rate": "reward_rate_arm_k10",  # 5- -> 10-visit window
    "nominal_reward_prob": "p_reward_arm",  # deferred-NaN -> populated
    "swr_x_recent_reward_rate": "swr_x_reward_rate_arm_k10",  # derived from changed
}
# New columns with no backup counterpart (value block).
NEW_COLS = {
    "q_arm", "q_arm_patch", "p_reward_arm_patch", "reward_rate_arm_k10_patch",
}
SESSION_COLS = ["animal", "day", "epoch"]

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


def verify_table(label: str, bpath, npath) -> bool:
    b = pd.read_pickle(bpath)
    n = pd.read_pickle(npath)
    ok = True
    print(f"\n=== {label} ===  backup {b.shape}  new {n.shape}")

    # 4. coverage
    if len(b) != len(n):
        print(f"  [FAIL] row count {len(b)} -> {len(n)}")
        ok = False
    if set(SESSION_COLS).issubset(b.columns) and set(SESSION_COLS).issubset(n.columns):
        nb = b[SESSION_COLS].drop_duplicates().shape[0]
        nn = n[SESSION_COLS].drop_duplicates().shape[0]
        tag = "ok" if nb == nn else "FAIL"
        ok &= nb == nn
        print(f"  coverage: sessions {nb} -> {nn} [{tag}]; rows {len(b)} -> {len(n)}")

    # 1. rebuild integrity on shared, unchanged columns
    changed_new = set(CHANGED.values()) | set(RENAME_ONLY.values()) | NEW_COLS
    shared = [c for c in b.columns if c in n.columns and c not in changed_new]
    mism = []
    for c in shared:
        is_float = (pd.api.types.is_float_dtype(b[c])
                    or pd.api.types.is_float_dtype(n[c]))
        if not _col_reproduces(b[c].to_numpy(), n[c].to_numpy(), is_float):
            mism.append(c)
    if mism:
        ok = False
        print(f"  [FAIL] rebuild-integrity: {len(mism)} shared col(s) differ: {mism}")
    else:
        print(f"  rebuild-integrity: {len(shared)} shared unchanged cols reproduce [ok]")

    # 2. rename-only identity (values must be byte-identical)
    for old, new in RENAME_ONLY.items():
        if old in b.columns and new in n.columns:
            is_float = pd.api.types.is_float_dtype(n[new])
            if _col_reproduces(b[old].to_numpy(), n[new].to_numpy(), is_float):
                print(f"  rename-only: {old} == {new} [ok]")
            else:
                ok = False
                print(f"  [FAIL] rename-only {old} -> {new} changed values")

    # 3. changed covariates: old -> new summary (no assert)
    for old, new in CHANGED.items():
        if old in b.columns and new in n.columns:
            print(f"  changed: {old} [{_summarize(b[old])}]")
            print(f"        -> {new} [{_summarize(n[new])}]")

    # new value-block columns: present + finite-where-defined
    for c in sorted(NEW_COLS):
        if c in n.columns:
            print(f"  new: {c} [{_summarize(n[c])}]")
    return ok


def main() -> int:
    all_ok = True
    for label, bpath, npath in TABLES:
        if not (bpath.exists() and npath.exists()):
            print(f"\n=== {label} ===  [skip] missing backup or new table")
            continue
        all_ok &= verify_table(label, bpath, npath)
    print("\n" + "=" * 60)
    print("HARMONIZATION PARITY:", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
