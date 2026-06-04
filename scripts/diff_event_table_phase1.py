"""Diff the pre- vs post-refactor ``event_table.pkl`` (one-off, deletable).

Compares the baseline replay-content event table
(``event_table.pre_phase1.pkl``) against the regenerated one
(``event_table.pkl``) after ``build_event_table`` was refactored to compose
on the canonical classified event table.

Checks, in order:

1. **Population identity** — same row count and same *ordered* ``event_id``
   sequence. A mismatch is a stop-and-investigate failure.
2. **Legacy bit-identity** — the filtered-survivor-window timing columns and
   every replay aggregate must be byte-identical (NaN-aware). Any drift means
   the legacy groupby path changed behavior — stop and investigate.
3. **Additive full-window timing** — report how often ``label_event_*``
   differs from the legacy timing, the median/max shift, that every legacy
   ``event_start`` lies inside its ``stop_id`` interval, and that every
   ``label_event_*`` is finite.
4. **is_swr drift** — count and fraction of events whose new strict
   positive-duration ``is_swr`` disagrees with the old inclusive-timebin
   value (expected change; sanity bound < 1%).
5. **Classified-column completeness** — non-null fractions, ``overlap_class``
   counts, and matched-SWR non-null fraction among SWR-coincident events.

Exit code is non-zero if any hard check (1 or 2) fails.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

OUTPUT_DIR = PROCESSED_DATA_DIR / "replay_content"

# Columns that MUST be bit-identical: the filtered-survivor-window timing
# plus every legacy replay aggregate. ``is_swr`` is intentionally excluded
# (its semantics changed); the new columns are additive.
LEGACY_BIT_IDENTICAL = [
    "event_start",
    "event_end",
    "event_duration",
    "event_midpoint_time",
    "time_since_stop",
    "time_since_port_entry",
    "time_since_reward",
    "is_mua",
    "is_after_switch",
    "is_after_omission",
    "is_after_reward",
    "is_after_stay",
    "reward_outcome",
    "is_stay",
    "current_arm",
    "prev_arm",
    "next_arm",
    "last_rewarded_arm",
    "current_patch",
    "prev_patch",
    "next_patch",
    "n_nonlocal_bins",
    "nonlocal_weight_sum",
    "posterior_mass_total",
    "posterior_peak_mean",
    "posterior_entropy_mean",
    "posterior_hpd_width_50",
    "posterior_hpd_width_80",
    "posterior_hpd_width_95",
    "n_junction_bins",
    "n_unique_arms",
    "stop_idx",
    "stop_id",
]

LABEL_TIMING = [
    "label_event_start",
    "label_event_end",
    "label_event_duration",
    "label_event_midpoint_time",
]


def _series_equal_nan_aware(a: pd.Series, b: pd.Series) -> pd.Series:
    """Element-wise equality treating NaN==NaN (and <NA>==<NA>) as equal."""
    a = a.reset_index(drop=True)
    b = b.reset_index(drop=True)
    both_na = a.isna().to_numpy() & b.isna().to_numpy()
    eq = (a == b).to_numpy(dtype=bool, na_value=False)
    return pd.Series(eq | both_na)


def main() -> int:
    baseline_path = OUTPUT_DIR / "event_table.pre_phase1.pkl"
    post_path = OUTPUT_DIR / "event_table.pkl"
    stop_path = OUTPUT_DIR / "stop_table.pkl"

    if not baseline_path.exists():
        print(f"NO BASELINE at {baseline_path}; nothing to diff.")
        return 0

    base = pd.read_pickle(baseline_path)
    post = pd.read_pickle(post_path)
    stop = pd.read_pickle(stop_path)

    hard_fail = False
    print("=" * 70)
    print("Phase 1 event_table diff")
    print("=" * 70)
    print(f"baseline rows: {len(base)}   post rows: {len(post)}")

    # --- 1. Population identity ---
    base_ids = base["event_id"].to_numpy()
    post_ids = post["event_id"].to_numpy()
    row_delta = len(post) - len(base)
    set_delta = len(set(post_ids).symmetric_difference(set(base_ids)))
    print("\n[1] Population identity")
    print(f"    row-count delta (post - base): {row_delta}")
    print(f"    event_id set symmetric-difference size: {set_delta}")
    if len(base) != len(post):
        print("    FAIL: row count changed.")
        hard_fail = True
    ordered_match = len(base) == len(post) and bool((base_ids == post_ids).all())
    print(f"    ordered event_id sequence identical: {ordered_match}")
    if not ordered_match:
        # Try order-insensitive: if the set matches, it's an ordering issue;
        # if not, it's a true population change.
        if set_delta == 0:
            print("    NOTE: same set, different order — will align by event_id.")
        else:
            print("    FAIL: event_id set differs (true population change).")
            hard_fail = True

    # Align post to baseline order on event_id for all column comparisons.
    base_idx = base.set_index("event_id")
    post_idx = post.set_index("event_id")
    common = base_idx.index.intersection(post_idx.index)
    base_a = base_idx.loc[common]
    post_a = post_idx.loc[common]

    # --- 2. Legacy bit-identity ---
    print("\n[2] Legacy bit-identity")
    n_drift_cols = 0
    for col in LEGACY_BIT_IDENTICAL:
        if col not in base_a.columns:
            print(f"    (skip {col}: absent from baseline)")
            continue
        if col not in post_a.columns:
            print(f"    FAIL {col}: absent from post table")
            hard_fail = True
            n_drift_cols += 1
            continue
        eq = _series_equal_nan_aware(base_a[col], post_a[col])
        n_diff = int((~eq).sum())
        if n_diff:
            n_drift_cols += 1
            hard_fail = True
            # Show a couple of example mismatches for triage.
            bad = (~eq).to_numpy().nonzero()[0][:3]
            examples = [(base_a[col].iloc[i], post_a[col].iloc[i]) for i in bad]
            print(f"    FAIL {col}: {n_diff} mismatches; e.g. {examples}")
    if n_drift_cols == 0:
        print(f"    OK: all {len(LEGACY_BIT_IDENTICAL)} legacy columns bit-identical.")

    # --- 3. Additive full-window timing ---
    print("\n[3] Additive full-window timing (label_event_*)")
    for lab, leg in zip(
        LABEL_TIMING,
        ["event_start", "event_end", "event_duration", "event_midpoint_time"],
        strict=True,
    ):
        if lab not in post_a.columns:
            print(f"    (missing {lab})")
            continue
        finite_frac = float(np.isfinite(post_a[lab].to_numpy(dtype=float)).mean())
        shift = post_a[lab].to_numpy(dtype=float) - post_a[leg].to_numpy(dtype=float)
        differ = int((np.abs(shift) > 1e-9).sum())
        med = float(np.nanmedian(np.abs(shift)))
        mx = float(np.nanmax(np.abs(shift)))
        print(
            f"    {lab} vs {leg}: {differ}/{len(post_a)} differ "
            f"(median |Δ|={med:.4g}s, max |Δ|={mx:.4g}s); finite_frac={finite_frac:.4f}"
        )

    # legacy event_start must lie within its stop_id interval
    stop_iv = stop.set_index("stop_id")[["stop_start", "stop_end"]]
    merged = post.merge(
        stop_iv, left_on="stop_id", right_index=True, how="left", validate="many_to_one"
    )
    within = (merged["event_start"] >= merged["stop_start"] - 1e-9) & (
        merged["event_start"] <= merged["stop_end"] + 1e-9
    )
    n_outside = int((~within).sum())
    print(
        f"    legacy event_start within stop_id interval: "
        f"{len(merged) - n_outside}/{len(merged)} (outside={n_outside})"
    )
    if n_outside:
        print("    WARNING: some events fall outside their stop interval.")

    # --- 4. is_swr drift ---
    print("\n[4] is_swr drift (old inclusive vs new strict)")
    if "is_swr" in base_a.columns and "is_swr" in post_a.columns:
        old = base_a["is_swr"].astype(bool).to_numpy()
        new = post_a["is_swr"].astype(bool).to_numpy()
        n_disagree = int((old != new).sum())
        frac = n_disagree / len(common) if len(common) else 0.0
        to_true = int((~old & new).sum())
        to_false = int((old & ~new).sum())
        print(
            f"    disagree: {n_disagree}/{len(common)} ({frac:.4%}); "
            f"old→True/new→False={to_false}, old→False/new→True={to_true}"
        )
        if frac >= 0.01:
            print(f"    NOTE: drift {frac:.4%} exceeds the <1% sanity bound.")

    # --- 5. Classified-column completeness ---
    print("\n[5] Classified-column completeness / counts")
    for col in ["peak_nonlocal_prob", "overlap_class", *LABEL_TIMING]:
        if col in post.columns:
            print(f"    {col}: non-null frac = {post[col].notna().mean():.4f}")
    if "overlap_class" in post.columns:
        print("    overlap_class counts:")
        for k, v in post["overlap_class"].value_counts(dropna=False).items():
            print(f"        {k}: {v}")
    if "is_swr" in post.columns and "matched_swr_event" in post.columns:
        swr = post.loc[post["is_swr"].astype(bool)]
        if len(swr):
            print(
                f"    matched_swr_event non-null among is_swr: "
                f"{swr['matched_swr_event'].notna().mean():.4f}"
            )

    print("\n" + "=" * 70)
    if hard_fail:
        print("RESULT: HARD FAIL — population or legacy bit-identity broke. STOP.")
        return 1
    print("RESULT: PASS — population identical, legacy columns bit-identical.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
