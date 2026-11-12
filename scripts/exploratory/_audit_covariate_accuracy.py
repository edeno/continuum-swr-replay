#!/usr/bin/env python
"""Behavioral-validity + coverage audit of the model covariates.

Complements ``_verify_covariate_invariants.py`` (which checks ranges/identities):
this validates the covariates against **independent** behavioral signals and
characterizes what the fit drops. Run from the repo root:

    uv run python scripts/exploratory/_audit_covariate_accuracy.py

Checks:
  #1a Calibration -- nominal ``task_reward_probability_chosen_arm`` vs the
      *independent* DIO ``is_rewarded_outcome`` (at-well stops). Aligned ->
      empirical tracks nominal.
  #1b Behavioral validity -- model
      ``relative_value_switch_minus_stay_preoutcome`` (ΔV) vs the animal's
      *actual* switch/stay. Correct + aligned -> ΔV higher on switch trials.
  #2  Leakage exposure -- fraction of events preceding reward delivery
      (``t_since_pump`` < 0), which would make post-outcome covariates
      anticipatory.
  #6a Coverage -- stops unmapped to a trial: rate by animal + within-epoch
      position (expected: the post-final-trial tail).
  #6b Complete-case drop -- listwise NaN over the fitted covariates, by
      covariate / animal / within-epoch position.

Non-RL covariates (the models fit far more than the qvals block):
  - spatial ``occupancy_well_arm`` cross-source agreement (event-arm vs stop);
  - decode content (``arm_fraction_posterior`` / ``arm_fraction``) normalization
    + junction-bin / NaN integrity.
Reward-kernel covariates are trial-referenced and covered by the invariant
verifier + the #1a calibration (see _nonrl_checks docstring).
  - prior-visit candidate-arm reward-fraction reconstruction vs the stored column, gated
    behind ``--reconstruct-arm-rate N`` (needs Spyglass trials; slow).

Diagnostic (prints a report); does not exit non-zero.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.behavior import add_uncertainty_covariates
from continuum_swr_replay.paths import REPLAY_CONTENT_DIR, table_path
from continuum_swr_replay.tables.covariate_registry import (
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table

RC = REPLAY_CONTENT_DIR


def _fur():
    """The uncertainty rate fit's module, for its covariate sets."""
    from continuum_swr_replay.analysis.behavior.rate import uncertainty_rate

    return uncertainty_rate


def _within_epoch_pos(df: pd.DataFrame) -> pd.Series:
    key = df["animal"] + "_" + df["session_date"].astype(str) + "_" + df["epoch"]
    return df.assign(_k=key).groupby("_k")["stop_idx"].rank(pct=True)


def main(reconstruct_n: int = 0) -> None:
    q = read_table(table_path("qvals_table", RC))
    st = read_table(table_path("stop_table", RC))
    require_covariate_schema(q, table="qvals")
    require_covariate_schema(st, table="stop")
    fur = _fur()

    # ---- #1a calibration: nominal p vs independent DIO reward ----
    aw = q[["stop_id", "task_reward_probability_chosen_arm", "in_poke"]].merge(
        st[["stop_id", "is_rewarded_outcome"]], on="stop_id", how="left"
    )
    aw = aw[aw["in_poke"]].copy()
    aw["is_rewarded_outcome"] = aw["is_rewarded_outcome"].astype(float)
    cal = aw.groupby("task_reward_probability_chosen_arm")["is_rewarded_outcome"].agg(
        ["size", "mean"]
    )
    x, y = cal.index.to_numpy(), cal["mean"].to_numpy()
    print(
        "=== #1a task_reward_probability_chosen_arm calibration vs DIO reward (at-well) ==="
    )
    print(cal.round(3).to_string())
    print(
        f"  slope(emp~nominal)={np.polyfit(x, y, 1)[0]:.2f} (expect ~1.0), "
        f"max|emp-nominal|={np.max(np.abs(y - x)):.3f}"
    )

    # ---- #1b ΔV predicts actual switch ----
    dv = (
        q[["stop_id", "relative_value_switch_minus_stay_preoutcome"]]
        .merge(st[["stop_id", "is_stay", "animal"]], on="stop_id", how="left")
        .dropna(subset=["relative_value_switch_minus_stay_preoutcome", "is_stay"])
    )
    dv["is_switch"] = (~dv["is_stay"].astype(bool)).astype(float)
    print("\n=== #1b ΔV (switch minus stay value) predicts actual switch ===")
    print(f"  {'animal':8s} {'n':>6s} {'dV|switch':>10s} {'dV|stay':>9s} {'r_pb':>7s}")
    for a, g in dv.groupby("animal"):
        v, s = (
            g["relative_value_switch_minus_stay_preoutcome"].to_numpy(),
            g["is_switch"].to_numpy(),
        )
        r = np.corrcoef(v, s)[0, 1] if g["is_switch"].nunique() > 1 else np.nan
        print(
            f"  {a:8s} {len(g):6d} {v[s == 1].mean():10.3f} {v[s == 0].mean():9.3f} {r:7.3f}"
        )
    print("  expect dV|switch > dV|stay and r_pb > 0")

    # ---- #2 leakage exposure ----
    ons = read_table(table_path("event_onset_table", RC))
    require_covariate_schema(ons, table="event_onset")
    rew = ons[ons["is_reward_stop"]]
    tsp = rew["t_since_pump"].dropna()
    print("\n=== #2 leakage exposure (events vs reward pump) ===")
    print(
        f"  events on rewarded stops: {len(tsp)}; before pump (leak): "
        f"{100 * (tsp < 0).mean():.1f}%; median t_since_pump={tsp.median():.2f}s"
    )

    # ---- #6a coverage: unmapped stops ----
    st2 = st.copy()
    st2["mapped"] = st2["stop_id"].isin(set(q["stop_id"]))
    st2["pos"] = _within_epoch_pos(st2)
    print("\n=== #6a coverage: stops unmapped to a trial ===")
    print(
        f"  unmapped {(~st2['mapped']).sum()}/{len(st2)} "
        f"({100 * (~st2['mapped']).mean():.1f}%); by animal "
        f"{(100 * (1 - st2.groupby('animal')['mapped'].mean())).round(1).to_dict()}"
    )
    print(
        f"  unmapped within-epoch pos {st2.loc[~st2['mapped'], 'pos'].mean():.2f} "
        f"(mapped {st2.loc[st2['mapped'], 'pos'].mean():.2f}); 1.0=end-of-epoch"
    )

    # ---- #6b complete-case drop over fitted covariates ----
    stc = add_uncertainty_covariates(st)  # keeps animal/day/epoch/stop_idx from st
    qcols = ["stop_id", *fur.VALUE_COVARIATES, "rpe_postoutcome", *fur.RL_COVARIATES]
    d = stc.merge(q[qcols], on="stop_id", how="inner")
    fitted = [c for c in fur.FITTED_COVARIATES if c in d]
    anynan = d[fitted].isna().any(axis=1)
    d["pos"] = _within_epoch_pos(d)
    print("\n=== #6b complete-case drop over fitted covariates ===")
    for c in fitted:
        r = 100 * d[c].isna().mean()
        if r > 0:
            print(f"  {c:52s} {r:5.1f}% NaN")
    print(
        f"  listwise drop {anynan.sum()} ({100 * anynan.mean():.1f}%); by animal "
        f"{(100 * d.assign(x=anynan).groupby('animal')['x'].mean()).round(1).to_dict()}"
    )
    print(
        f"  dropped within-epoch pos {d.loc[anynan.values, 'pos'].mean():.2f} "
        f"(kept {d.loc[~anynan.values, 'pos'].mean():.2f})"
    )

    _nonrl_checks(st)
    if reconstruct_n:
        _reconstruct_reward_rate_arm(reconstruct_n)


def _reconstruct_reward_rate_arm(n_sessions: int) -> None:
    """Independently reconstruct candidate-arm prior-visit reward fraction.

    The per-arm rolling reward rate is trial-referenced (rewarded fraction of the
    last 10 visits to that arm), so a faithful check needs the trials table -- this
    loads trials for a sample of sessions and reimplements the rolling-mean loop
    (reusing only the topology/timing helpers, not the rate logic under test). Heavy
    (Spyglass), so gated behind --reconstruct-arm-rate. Validated 100% on 2 sessions.
    """
    from continuum_swr_replay.analysis.behavior import trial_outcome_times
    from continuum_swr_replay.analysis.behavior._helpers import build_well_to_arm_map
    from continuum_swr_replay.data_loaders import load_data
    from continuum_swr_replay.data_types import SessionDataKeys

    ea = read_table(table_path("event_arm_table", RC))
    et = read_table(table_path("event_table", RC))
    require_covariate_schema(ea, table="event_arm")
    require_covariate_schema(et, table="event")
    tcol = "event_midpoint_s"
    ea = ea.merge(
        et[["event_id", tcol]].rename(columns={tcol: "t"}), on="event_id", how="left"
    )
    print(
        f"\n=== non-RL: candidate-arm reward-fraction reconstruction "
        f"({n_sessions} session(s), Spyglass) ==="
    )
    for _, s in (
        ea[["animal", "session_date", "epoch"]]
        .drop_duplicates()
        .head(n_sessions)
        .iterrows()
    ):
        sub = ea[
            (ea.animal == s.animal)
            & (ea.session_date == s.session_date)
            & (ea.epoch == s.epoch)
        ]
        d = load_data(
            f"{s.animal}{int(s.session_date)}_.nwb",
            s.epoch,
            include=frozenset({SessionDataKeys.TRIALS}),
        )
        w2a = build_well_to_arm_map(d["track_graph"])
        # This reconstructs a reward-bearing covariate, so the history clock is
        # feedback availability rather than goal arrival.
        tr = d["trials"].assign(_c=trial_outcome_times(d["trials"])).sort_values("_c")
        cut = tr["_c"].to_numpy()
        rew = tr["is_reward"].to_numpy().astype(float)
        arm_of = np.array(
            [
                w2a.get(int(w)) if not np.isnan(w) else -1
                for w in tr["to_well"].to_numpy()
            ]
        )
        reach = set(w2a.values())
        vis = {a: (cut[arm_of == a], rew[arm_of == a]) for a in range(9)}
        got = sub["reward_fraction_candidate_arm_last10_visits"].to_numpy()
        arms, ts = sub["candidate_arm"].to_numpy(), sub["t"].to_numpy()
        exp = np.full(len(sub), np.nan)
        for i in range(len(sub)):
            a, t = int(arms[i]), ts[i]
            if np.isnan(t):
                continue
            if a not in reach:
                exp[i] = 0.0 if (cut < t).any() else np.nan
                continue
            vc, vr = vis[a]
            m = vc < t
            if m.any():
                exp[i] = vr[m][-10:].mean()
        both = ~np.isnan(got) & ~np.isnan(exp)
        agree = (
            np.isclose(got[both], exp[both], atol=1e-9).mean()
            if both.any()
            else float("nan")
        )
        nanm = (np.isnan(got) == np.isnan(exp)).mean()
        print(
            f"  {s.animal} {int(s.session_date)} {s.epoch}: value-agree {100 * agree:.2f}% "
            f"(n={both.sum()}), NaN-pattern {100 * nanm:.2f}%"
        )


def _nonrl_checks(st: pd.DataFrame) -> None:
    """Non-RL covariate ground-truth: the models fit far more than the qvals block.

    Note: reward-kernel covariates (known_reward_outcome_lag_{k},
    last_rewarded_arm, and outcome indicators)
    are TRIAL-referenced (built from the trials table, not the stop sequence), and
    are already covered by _verify_covariate_invariants (reward_lag in {0,1},
    reward fraction == mean(known_reward_outcome_lag_1..10)) plus the #1a
    DIO-reward calibration of their atom (is_rewarded_outcome). Checking them
    here against a stop-sequence oracle
    would mismatch on grain, not correctness -- so they are intentionally omitted.
    """
    ea = read_table(table_path("event_arm_table", RC))
    require_covariate_schema(ea, table="event_arm")
    print("\n=== non-RL: occupancy_well_arm cross-source (event_arm vs stop_table) ===")
    m = (
        ea[["stop_id", "occupancy_well_arm"]]
        .drop_duplicates()
        .merge(
            st[["stop_id", "occupancy_well_arm"]],
            on="stop_id",
            suffixes=("_ea", "_st"),
        )
        .dropna()
    )
    print(
        f"  agree {100 * (m['occupancy_well_arm_ea'] == m['occupancy_well_arm_st']).mean():.2f}% "
        f"(n={len(m)}); position-derived arm consistent across builders"
    )

    print("\n=== non-RL: decode content integrity (per-event over 9 arms) ===")
    for col in ["arm_fraction_posterior", "arm_fraction"]:
        if col not in ea:
            continue
        s = ea.groupby("event_id")[col].sum()
        s = s[s.notna() & (s > 0)]
        print(
            f"  {col}: sum-to-1 for {100 * ((s > 0.99) & (s < 1.01)).mean():.1f}% "
            f"of events (median {s.median():.4f}); junction bins would break this"
        )
    uni = ea.groupby("event_id")["arm_fraction_posterior"].apply(
        lambda x: x.isna().all() or x.notna().all()
    )
    print(
        f"  arm_fraction_posterior NaN uniform per event (all-or-none): "
        f"{100 * uni.mean():.1f}%; arm range {int(ea['candidate_arm'].min())}..{int(ea['candidate_arm'].max())} (must be 0..8)"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--reconstruct-arm-rate",
        type=int,
        default=0,
        metavar="N",
        help="also reconstruct candidate-arm reward fraction on N sessions (Spyglass; slow)",
    )
    args = ap.parse_args()
    main(reconstruct_n=args.reconstruct_arm_rate)
