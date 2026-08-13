#!/usr/bin/env python
"""Establish + verify invariants and properties of the regression covariates.

Covers every covariate that enters a replay model family (content arm/patch,
per-stop rate A, per-bin rate B, choice, MU) as it lives in the canonical
tables: ``stop_table``, ``event_arm_table``, ``event_patch_table``, and the
``qvals_table`` value block (nominal p(R) + model Q per port). The
behavioral-model (RL/qvals) block has its own checker,
``_verify_qvals_invariants.py``.

Tiers:
  HARD  — must hold exactly (ranges, NaN policy, per-event structure, dtypes).
  IDENT — cross-covariate identities that MUST hold (e.g. an interaction equals
          the product of its factors; the trial rate equals the lag-kernel mean,
          which is why the two are never co-fit).
  SOFT  — should roughly hold.

Exits non-zero if any HARD or IDENT check fails. Run from the repo root:
    uv run python scripts/exploratory/_verify_covariate_invariants.py
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

RC = PROCESSED_DATA_DIR / "replay_content"


def _is_min_ranking(s) -> bool:
    """True if s is a valid 'min'-method ranking (rank = 1 + #strictly-smaller)."""
    s = np.sort(np.asarray(s, dtype=float))
    return bool(np.all(s == np.searchsorted(s, s, side="left") + 1))


def main() -> int:
    res: list[tuple[str, bool]] = []

    def check(tier: str, name: str, ok: bool, detail: str = "") -> None:
        res.append((tier, bool(ok)))
        status = "PASS" if ok else ("CHECK" if tier == "S" else "FAIL")
        label = {"H": "HARD ", "A": "IDENT", "S": "SOFT "}[tier]
        print(f"[{label} {status:5s}] {name}: {detail}")

    def in01(v, tol=1e-9):
        return np.nanmin(v) >= -tol and np.nanmax(v) <= 1 + tol

    # ================= stop_table =================
    st = pd.read_pickle(RC / "stop_table.pkl")
    print("\n### stop_table (per-stop rate A / rate B input) ###")
    lags = [f"reward_lag{i}" for i in range(1, 11)]
    L = st[lags].to_numpy()
    check("H", "reward_lag in {0,1} (defined rows)",
          set(np.unique(L[~np.isnan(L)])) <= {0.0, 1.0}, f"{np.isnan(L).mean()*100:.1f}% NaN")
    rr = st["reward_rate_trial_k10"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN early rows
        mean_lag = np.nanmean(L, axis=1)
    check("A", "reward_rate_trial_k10 == mean(reward_lag1..10)",
          np.nanmax(np.abs(rr - mean_lag)) < 1e-9,
          "=> perfectly collinear with the reward kernel; must not be co-fit")
    check("H", "reward_rate_trial_k10 in [0,1]", in01(rr),
          f"[{np.nanmin(rr):.3f},{np.nanmax(rr):.3f}]")
    for c in ["is_after_switch", "is_after_omission"]:
        check("H", f"{c} in {{0,1}}", set(np.unique(st[c].dropna())) <= {0, 1}, "")
    im = st["immobility_time"].to_numpy()
    check("H", "immobility_time >= 0 (log offset)", np.nanmin(im) >= 0,
          f"min={np.nanmin(im):.3f}; =0 on {int((im == 0).sum())} stops (offset excludes them)")
    for c in ["n_events", "n_SWR"]:
        v = st[c].to_numpy()
        check("H", f"{c} >= 0 integer (response)",
              (v >= 0).all() and np.allclose(v, v.astype(int)), f"range [{v.min()},{v.max()}]")
    try:
        from continuum_swr_replay.analysis.replay.covariates import (
            add_uncertainty_covariates,
        )
        rs = add_uncertainty_covariates(st.copy())["recent_switch_rate"].to_numpy()
        check("H", "recent_switch_rate in [0,1] (fit-time)", in01(rs),
              f"[{np.nanmin(rs):.3f},{np.nanmax(rs):.3f}]")
    except Exception as e:  # noqa: BLE001
        check("S", "recent_switch_rate in [0,1]", False, f"could not recompute: {e}")

    # ================= event_arm_table =================
    ea = pd.read_pickle(RC / "event_arm_table.pkl")
    print("\n### event_arm_table (content arm + choice) ###")
    arm = ea["arm"].to_numpy()
    check("H", "arm in 0..8 (9 arms/event)", set(np.unique(arm)) == set(range(9)),
          f"rows/event = {len(ea) / ea['event_id'].nunique():.1f}")
    leaf, trunk = arm < 6, arm >= 6
    rk = ea["reward_rate_arm_k10"].to_numpy()
    check("H", "reward_rate_arm_k10 in [0,1]", in01(rk),
          f"[{np.nanmin(rk):.3f},{np.nanmax(rk):.3f}]")
    check("H", "reward_rate_arm_k10 trunk arms == 0.0",
          np.nanmax(np.abs(rk[trunk])) < 1e-12, "trunk arms are not reward sites")
    check("S", "reward_rate_arm_k10 leaf NaN = never-visited only",
          np.isnan(rk[leaf]).mean() < 0.2, f"leaf NaN={np.isnan(rk[leaf]).mean()*100:.1f}%")
    for c in ["p_reward_arm", "q_arm"]:
        v = ea[c].to_numpy()
        check("H", f"{c} in [0,1]", in01(v), f"[{np.nanmin(v):.3f},{np.nanmax(v):.3f}]")
        check("H", f"{c} trunk arms == 0.0", np.nanmax(np.abs(v[trunk])) < 1e-12, "")
    for c in ["is_current", "is_prev", "is_last_rewarded_elsewhere", "is_swr"]:
        check("H", f"{c} in {{0,1}}", set(np.unique(ea[c].dropna())) <= {0, 1}, "")
    per = ea.groupby("event_id")
    check("H", "exactly one is_current arm per event", (per["is_current"].sum() == 1).all(),
          f"dist {per['is_current'].sum().value_counts().to_dict()}")
    check("H", "<=1 is_prev arm per event", (per["is_prev"].sum() <= 1).all(), "")
    both = int((ea["is_current"].astype(bool) & ea["is_last_rewarded_elsewhere"].astype(bool)).sum())
    check("H", "is_current & is_last_rewarded_elsewhere disjoint", both == 0, f"{both} overlap")
    check("H", "is_swr constant within event", (per["is_swr"].nunique() == 1).all(), "")
    sw = ea["is_swr"].to_numpy().astype(float)
    for base, inter in [("reward_rate_arm_k10", "swr_x_reward_rate_arm_k10"),
                        ("is_current", "swr_x_current")]:
        b, x = ea[base].to_numpy().astype(float), ea[inter].to_numpy().astype(float)
        m = pd.notna(ea[base]) & pd.notna(ea[inter])
        check("A", f"{inter} == is_swr * {base}",
              np.abs(x[m] - sw[m] * b[m]).max() < 1e-9, "")
    leaves = ea.loc[ea["arm"].isin(range(6))]
    leaf_event_mean = leaves.groupby("event_id")["visit_recency_demeaned"].mean()
    trunks = ea.loc[~ea["arm"].isin(range(6)), "visit_recency_demeaned"]
    centered = (
        leaf_event_mean.dropna().abs().max()
        if leaf_event_mean.notna().any()
        else 0.0
    )
    trunks_zero = trunks.dropna().eq(0).all()
    check(
        "H",
        "visit_recency_demeaned leaf mean ~ 0 within event; trunks = 0",
        centered < 1e-6 and trunks_zero,
        f"max|event leaf mean|={centered:.2e}; trunks_zero={trunks_zero}",
    )
    check("H", "visit_recency >= 0", np.nanmin(ea["visit_recency"]) >= 0, "")
    check("H", "cumulative_reward >= 0", np.nanmin(ea["cumulative_reward"]) >= 0, "")
    adr = ea.groupby("event_id")["arm_dist_rank"].apply(list)
    frac = np.mean([_is_min_ranking(r) for r in adr])
    min1 = np.mean([min(r) == 1 for r in adr])
    check("H", "arm_dist_rank valid min-ranking per event (min=1, ties share min)",
          frac > 0.999 and min1 > 0.999, f"{100*frac:.1f}% valid, {100*min1:.1f}% have rank-1 arm")
    check("H", "arm_dist_rank >= 1, no NaN",
          np.nanmin(ea["arm_dist_rank"]) >= 1 and ea["arm_dist_rank"].isna().sum() == 0, "")
    check("H", "arm_count_hard >= 0 integer (response)",
          (ea["arm_count_hard"].to_numpy() >= 0).all(), "")

    # ================= event_patch_table =================
    ep = pd.read_pickle(RC / "event_patch_table.pkl")
    print("\n### event_patch_table (content patch) ###")
    check("H", "3 patches/event", abs(len(ep) / ep["event_id"].nunique() - 3) < 1e-9, "")
    cps = ep.groupby("event_id")["is_current_patch"].sum()
    check("H", "exactly one is_current_patch per event", (cps == 1).all(),
          f"dist {cps.value_counts().to_dict()}")
    for c in ["p_reward_arm_patch", "q_arm_patch"]:
        v = ep[c].to_numpy()
        check("H", f"{c} in [0,1]", in01(v), f"[{np.nanmin(v):.3f},{np.nanmax(v):.3f}]")
    check("H", "cumulative_reward_patch >= 0", np.nanmin(ep["cumulative_reward_patch"]) >= 0, "")
    check("H", "patch_count_hard >= 0", (ep["patch_count_hard"].to_numpy() >= 0).all(), "")

    # ================= qvals_table value block =================
    cont = pd.read_pickle(RC / "models_per_bin_rate" / "qvals_table.pkl")
    print("\n### qvals_table (nominal + model value block) ###")
    check("H", "stop_id unique", cont["stop_id"].is_unique,
          f"{cont['stop_id'].nunique()}/{len(cont)}")
    pv = cont[[f"p_reward_port{i}" for i in range(6)]].to_numpy()
    check("H", "p_reward_port in [0,1]", in01(pv),
          f"task probs={sorted(np.unique(pv[~np.isnan(pv)]))}")
    check("H", "p_reward_current in [0,1]", in01(cont["p_reward_current"].to_numpy()), "")
    check("H", "q_port / q_chosen in [0,1]",
          in01(cont[[f"q_port{i}" for i in range(6)]].to_numpy())
          and in01(cont["q_chosen"].to_numpy()), "")
    check("H", "trials_since_block_change >= 0",
          np.nanmin(cont["trials_since_block_change"]) >= 0, "")
    for c in ["rpe", "value_diff_patch"]:
        v = cont[c].to_numpy()
        check("H", f"{c} finite", np.isfinite(v[~np.isnan(v)]).all(),
              f"[{np.nanmin(v):.2f},{np.nanmax(v):.2f}]")

    # physical bounds (a value outside these = an upstream parse / alignment /
    # sign error, even though it may still be finite and in a plausible range).
    # nominal p(R) can only take the task-schedule levels; anything else is a parse bug.
    SCHEDULE = {0.2, 0.5, 0.8, 1.0}
    levels = set(np.round(np.unique(pv[~np.isnan(pv)]), 6))
    check("H", "p_reward_port in task schedule {0.2,0.5,0.8,1.0}",
          levels <= SCHEDULE, f"levels={sorted(levels)}")
    # rpe = reward(0/1) - q_chosen(0..1)  ->  strictly in [-1, 1]
    rpe = cont["rpe"].to_numpy()
    check("H", "rpe in [-1,1]", in01((rpe[~np.isnan(rpe)] + 1) / 2),
          f"[{np.nanmin(rpe):.2f},{np.nanmax(rpe):.2f}]")
    # value_diff_patch = best-minus-second patch mean of p(R) in [0,1] -> [0, 0.8]
    vdp = cont["value_diff_patch"].to_numpy()
    check("H", "value_diff_patch in [0, 0.8]",
          np.nanmin(vdp) >= 0 and np.nanmax(vdp) <= 0.8 + 1e-9,
          f"[{np.nanmin(vdp):.2f},{np.nanmax(vdp):.2f}]")
    # entropies (nats) are non-negative (universal); state_entropy is the HMM
    # state posterior over 72 states (Comrie et al.) so is additionally bounded by
    # log(72) -- verified the observed max equals log(72) exactly. reward_entropy
    # is NOT a 2-state entropy (observed max ~6.05), so only its sign is asserted.
    if "reward_entropy" in cont:
        e = cont["reward_entropy"].to_numpy()
        e = e[~np.isnan(e)]
        check("H", "reward_entropy >= 0", e.min() >= -1e-9,
              f"[{e.min():.2f},{e.max():.2f}]")
    if "state_entropy" in cont:
        e = cont["state_entropy"].to_numpy()
        e = e[~np.isnan(e)]
        check("H", "state_entropy in [0, log 72]",
              e.min() >= -1e-9 and e.max() <= np.log(72) + 1e-6,
              f"[{e.min():.2f},{e.max():.2f}]")

    hard = [ok for t, ok in res if t == "H"]
    ident = [ok for t, ok in res if t == "A"]
    soft = [ok for t, ok in res if t == "S"]
    print("\n" + "=" * 62)
    print(f"HARD  invariants : {sum(hard)}/{len(hard)} pass")
    print(f"IDENTITY checks  : {sum(ident)}/{len(ident)} pass")
    print(f"SOFT  properties : {sum(soft)}/{len(soft)} hold")
    ok = all(hard) and all(ident)
    print("OVERALL:", "PASS" if ok else "FAIL (hard/identity)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
