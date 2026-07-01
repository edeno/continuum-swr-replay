#!/usr/bin/env python
"""Establish + verify invariants and properties of the behavioral-model covariates.

The RL-model latents in ``qvals_table.pkl`` (value, decision variables, value
updates, entropy, RPE — see ``docs/behavior_model_covariates.md``) feed the
MU-uncertainty replay-rate model. The model betas are fit upstream (Alison
Comrie's DataJoint pipeline / Comrie et al. bioRxiv 2024.09.23.613567); this
script audits everything *downstream* of the fetched betas — the derived
decision variables, their alignment onto stops, and the pre/post-outcome timing.

Three tiers:
  HARD  — must hold exactly (identities, partitions, bounds, dtypes). A failure
          means a covariate is mis-computed or mis-aligned.
  ALIGN — trial->stop mapping and session-boundary handling (thresholded).
  SOFT  — should roughly hold (value<->reward, entropy<->uncertainty). A failure
          is a "look here", not necessarily a bug.

Exits non-zero if any HARD or ALIGN check fails. Run from the repo root:
    uv run python scripts/_verify_qvals_invariants.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

RC = PROCESSED_DATA_DIR / "replay_content"
Q = [f"Q{i}" for i in range(1, 7)]
MU = [f"betadist_mu{i}" for i in range(1, 7)]
VAR = [f"betadist_var{i}" for i in range(1, 7)]


def main() -> int:
    q = pd.read_pickle(RC / "models_per_bin_rate" / "qvals_table.pkl")
    st = pd.read_pickle(RC / "stop_table.pkl")[
        ["stop_id", "animal", "epoch", "current_arm", "reward_outcome"]
    ]
    m = q.merge(st, on="stop_id", how="left", suffixes=("", "_st"))

    results: list[tuple[str, bool]] = []

    def check(tier: str, name: str, ok: bool, detail: str) -> None:
        results.append((tier, ok))
        status = "PASS" if ok else ("CHECK" if tier == "S" else "FAIL")
        label = {"H": "HARD", "A": "ALIGN", "S": "SOFT"}[tier]
        print(f"[{label:5s} {status:5s}] {name}: {detail}")

    def corr(a, b):
        d = pd.DataFrame({"a": a, "b": b}).dropna()
        return d["a"].corr(d["b"]) if len(d) > 2 else np.nan

    arm = q["arm"].to_numpy()
    qmat, mu, var = q[Q].to_numpy(), q[MU].to_numpy(), q[VAR].to_numpy()

    # ---- HARD: structure + identities + bounds ----
    check("H", "stop_id unique", q["stop_id"].is_unique,
          f"{q['stop_id'].nunique()}/{len(q)} unique")
    check("H", "arm in 0..5", bool(((arm >= 0) & (arm <= 5)).all()),
          f"range [{arm.min()},{arm.max()}]")
    d = np.abs(q["q_chosen"].to_numpy() - qmat[np.arange(len(q)), arm])
    check("H", "q_chosen == Q[chosen port]", np.nanmax(d) < 1e-9,
          f"max|diff|={np.nanmax(d):.2e}")
    for cols, name in [(Q, "Q"), (MU, "betadist_mu")]:
        v = q[cols].to_numpy()
        check("H", f"{name} in [0,1]",
              np.nanmin(v) >= -1e-9 and np.nanmax(v) <= 1 + 1e-9,
              f"range [{np.nanmin(v):.3f},{np.nanmax(v):.3f}]")
    check("H", "betadist_var >= 0", np.nanmin(var) >= -1e-12, f"min={np.nanmin(var):.2e}")
    check("H", "betadist_var <= mu(1-mu) (Beta bound)",
          int(np.nansum(var > mu * (1 - mu) + 1e-9)) == 0, "0 violations expected")

    d = (q["switch_advantage_mu_preoutcome_sess"]
         - (q["best_other_patch_mean_mu_preoutcome_sess"]
            - q["stay_sibling_mu_preoutcome_sess"])).abs()
    check("H", "switch_advantage == best_other - stay (DeltaV)", np.nanmax(d) < 1e-9,
          f"max|diff|={np.nanmax(d):.2e}")
    d = (q["rpe"] - (q["trial_reward"] - q["q_chosen"])).abs()
    check("H", "rpe == trial_reward - q_chosen", np.nanmax(d) < 1e-9,
          f"max|diff|={np.nanmax(d):.2e}")
    check("H", "rpe_abs == |rpe|", (q["rpe_abs"] - q["rpe"].abs()).abs().max() < 1e-9, "")

    g = q["global_mu_update_abs_postoutcome_sess"]
    cp = q["chosen_patch_mu_update_abs_postoutcome_sess"]
    up = q["unchosen_patches_mu_update_abs_postoutcome_sess"]
    cl = q["chosen_leaf_mu_update_abs_postoutcome_sess"]
    gs = q["global_mu_update_signed_postoutcome_sess"]
    check("H", "global_upd_abs == chosen_patch + unchosen_patches (partition)",
          (g - (cp + up)).abs().max() < 1e-9, f"max|diff|={(g - (cp + up)).abs().max():.2e}")
    mk = cl.notna() & cp.notna()
    check("H", "chosen_leaf_upd_abs <= chosen_patch_upd_abs",
          int((cl[mk] > cp[mk] + 1e-9).sum()) == 0, f"over {mk.sum()} defined rows")
    mk = g.notna() & gs.notna()
    check("H", "global_upd_abs >= |global_upd_signed| (triangle)",
          int((g[mk] + 1e-9 < gs[mk].abs()).sum()) == 0, f"over {mk.sum()} defined rows")
    for c in ["reward_entropy", "state_entropy"]:
        check("H", f"{c} >= 0", np.nanmin(q[c]) >= -1e-12, f"min={np.nanmin(q[c]):.3f}")
    for c in ["trial_reward", "in_poke", "is_rewarding"]:
        check("H", f"{c} in {{0,1}}",
              set(pd.unique(q[c].dropna())) <= {0, 1}, "")

    # ---- ALIGNMENT ----
    cov = len(q) / len(st)
    check("A", "coverage (mapped/total stops)", 0.90 <= cov <= 1.0,
          f"{100*cov:.1f}% ({len(q)}/{len(st)})")
    poke = m["in_poke"] == 1
    armok = (m.loc[poke, "arm"] == m.loc[poke, "current_arm"]).mean()
    check("A", "at-well arm == current_arm", armok > 0.98,
          f"{100*armok:.2f}% (n={int(poke.sum())})")
    rok = (m.loc[poke, "trial_reward"] == m.loc[poke, "reward_outcome"]).mean()
    check("A", "at-well trial_reward == reward_outcome", rok > 0.97, f"{100*rok:.2f}%")
    first = m.groupby(["animal", "epoch"])["trial_number_by_epoch"].transform("min")
    is_first = m["trial_number_by_epoch"] == first
    na_first = m.loc[is_first, "switch_advantage_mu_preoutcome_sess"].isna().mean()
    na_rest = m.loc[~is_first, "switch_advantage_mu_preoutcome_sess"].isna().mean()
    check("A", "session reset: 1st trial DVs NaN, rest defined",
          na_first > 0.95 and na_rest < 0.15,
          f"NaN@first={100*na_first:.0f}% NaN@rest={100*na_rest:.0f}%")

    # ---- SOFT ----
    check("S", "q_chosen positively predicts reward",
          corr(q["q_chosen"], q["trial_reward"]) > 0.05,
          f"corr={corr(q['q_chosen'], q['trial_reward']):+.3f}")
    check("S", "q_chosen ~ betadist_mu[chosen]",
          corr(q["q_chosen"], mu[np.arange(len(q)), arm]) > 0.5,
          f"corr={corr(q['q_chosen'], mu[np.arange(len(q)), arm]):+.3f}")
    sa = q["switch_advantage_mu_preoutcome_sess"].dropna()
    check("S", "switch_advantage spans both signs", sa.min() < 0 < sa.max(),
          f"range [{sa.min():+.3f},{sa.max():+.3f}], mean {sa.mean():+.3f}")
    ce = corr(q["reward_entropy"], (q["q_chosen"] - 0.5).abs())
    check("S", "reward_entropy peaks at p~0.5 (neg corr w/ |q-.5|)", ce < 0, f"corr={ce:+.3f}")
    check("S", "value updates mostly positive", np.nanmean(g > 0) > 0.8,
          f"{100*np.nanmean(g > 0):.0f}% >0, mean={np.nanmean(g):.4f}")
    rpe = q["rpe"].dropna()
    check("S", "rpe both signs, ~calibrated (|mean| small)",
          rpe.min() < 0 < rpe.max() and abs(rpe.mean()) < 0.2,
          f"mean={rpe.mean():+.3f} range [{rpe.min():+.2f},{rpe.max():+.2f}]")

    hard = [ok for t, ok in results if t == "H"]
    align = [ok for t, ok in results if t == "A"]
    soft = [ok for t, ok in results if t == "S"]
    print("\n" + "=" * 60)
    print(f"HARD invariants : {sum(hard)}/{len(hard)} pass")
    print(f"ALIGNMENT       : {sum(align)}/{len(align)} pass")
    print(f"SOFT properties : {sum(soft)}/{len(soft)} hold")
    ok = all(hard) and all(align)
    print("OVERALL:", "PASS" if ok else "FAIL (hard/alignment)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
