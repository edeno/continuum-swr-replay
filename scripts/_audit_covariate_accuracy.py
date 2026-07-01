#!/usr/bin/env python
"""Behavioral-validity + coverage audit of the model covariates.

Complements ``_verify_covariate_invariants.py`` (which checks ranges/identities):
this validates the covariates against **independent** behavioral signals and
characterizes what the fit drops. Run from the repo root:

    uv run python scripts/_audit_covariate_accuracy.py

Checks:
  #1a Calibration -- nominal ``p_reward_current`` vs the *independent* DIO
      ``reward_outcome`` (at-well stops). Aligned -> empirical tracks nominal.
  #1b Behavioral validity -- model ``switch_advantage`` (ΔV) vs the animal's
      *actual* switch/stay. Correct + aligned -> ΔV higher on switch trials.
  #2  Leakage exposure -- fraction of events preceding reward delivery
      (``t_since_pump`` < 0), which would make post-outcome covariates
      anticipatory.
  #6a Coverage -- stops unmapped to a trial: rate by animal + within-epoch
      position (expected: the post-final-trial tail).
  #6b Complete-case drop -- listwise NaN over the fitted covariates, by
      covariate / animal / within-epoch position.

Diagnostic (prints a report); does not exit non-zero.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay import add_uncertainty_covariates
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

RC = PROCESSED_DATA_DIR / "replay_content"


def _fur():
    """Load fit_uncertainty_rate for its covariate sets (scripts/ isn't a package)."""
    path = Path(__file__).with_name("fit_uncertainty_rate.py")
    spec = importlib.util.spec_from_file_location("fur", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _within_epoch_pos(df: pd.DataFrame) -> pd.Series:
    key = df["animal"] + "_" + df["day"].astype(str) + "_" + df["epoch"]
    return df.assign(_k=key).groupby("_k")["stop_idx"].rank(pct=True)


def main() -> None:
    q = pd.read_pickle(RC / "models_per_bin_rate" / "qvals_table.pkl")
    st = pd.read_pickle(RC / "stop_table.pkl")
    fur = _fur()

    # ---- #1a calibration: nominal p vs independent DIO reward ----
    aw = q[["stop_id", "p_reward_current", "in_poke"]].merge(
        st[["stop_id", "reward_outcome"]], on="stop_id", how="left"
    )
    aw = aw[aw["in_poke"]].copy()
    aw["reward_outcome"] = aw["reward_outcome"].astype(float)
    cal = aw.groupby("p_reward_current")["reward_outcome"].agg(["size", "mean"])
    x, y = cal.index.to_numpy(), cal["mean"].to_numpy()
    print("=== #1a p_reward_current calibration vs DIO reward (at-well) ===")
    print(cal.round(3).to_string())
    print(f"  slope(emp~nominal)={np.polyfit(x, y, 1)[0]:.2f} (expect ~1.0), "
          f"max|emp-nominal|={np.max(np.abs(y - x)):.3f}")

    # ---- #1b ΔV predicts actual switch ----
    dv = q[["stop_id", "switch_advantage_mu_preoutcome_sess"]].merge(
        st[["stop_id", "is_stay", "animal"]], on="stop_id", how="left"
    ).dropna(subset=["switch_advantage_mu_preoutcome_sess", "is_stay"])
    dv["is_switch"] = (~dv["is_stay"].astype(bool)).astype(float)
    print("\n=== #1b ΔV (switch_advantage) predicts actual switch ===")
    print(f"  {'animal':8s} {'n':>6s} {'dV|switch':>10s} {'dV|stay':>9s} {'r_pb':>7s}")
    for a, g in dv.groupby("animal"):
        v, s = g["switch_advantage_mu_preoutcome_sess"].to_numpy(), g["is_switch"].to_numpy()
        r = np.corrcoef(v, s)[0, 1] if g["is_switch"].nunique() > 1 else np.nan
        print(f"  {a:8s} {len(g):6d} {v[s == 1].mean():10.3f} {v[s == 0].mean():9.3f} {r:7.3f}")
    print("  expect dV|switch > dV|stay and r_pb > 0")

    # ---- #2 leakage exposure ----
    ons = pd.read_pickle(RC / "event_onset_table.pkl")
    rew = ons[ons["is_reward_stop"]]
    tsp = rew["t_since_pump"].dropna()
    print("\n=== #2 leakage exposure (events vs reward pump) ===")
    print(f"  events on rewarded stops: {len(tsp)}; before pump (leak): "
          f"{100 * (tsp < 0).mean():.1f}%; median t_since_pump={tsp.median():.2f}s")

    # ---- #6a coverage: unmapped stops ----
    st2 = st.copy()
    st2["mapped"] = st2["stop_id"].isin(set(q["stop_id"]))
    st2["pos"] = _within_epoch_pos(st2)
    print("\n=== #6a coverage: stops unmapped to a trial ===")
    print(f"  unmapped {(~st2['mapped']).sum()}/{len(st2)} "
          f"({100 * (~st2['mapped']).mean():.1f}%); by animal "
          f"{(100 * (1 - st2.groupby('animal')['mapped'].mean())).round(1).to_dict()}")
    print(f"  unmapped within-epoch pos {st2.loc[~st2['mapped'], 'pos'].mean():.2f} "
          f"(mapped {st2.loc[st2['mapped'], 'pos'].mean():.2f}); 1.0=end-of-epoch")

    # ---- #6b complete-case drop over fitted covariates ----
    stc = add_uncertainty_covariates(st)  # keeps animal/day/epoch/stop_idx from st
    qcols = ["stop_id", *fur.VALUE_COVARIATES, "rpe", *fur.RL_COVARIATES]
    d = stc.merge(q[qcols].rename(columns={"rpe": "rpe_model"}), on="stop_id", how="inner")
    fitted = [c for c in fur.FITTED_COVARIATES if c in d]
    anynan = d[fitted].isna().any(axis=1)
    d["pos"] = _within_epoch_pos(d)
    print("\n=== #6b complete-case drop over fitted covariates ===")
    for c in fitted:
        r = 100 * d[c].isna().mean()
        if r > 0:
            print(f"  {c:52s} {r:5.1f}% NaN")
    print(f"  listwise drop {anynan.sum()} ({100 * anynan.mean():.1f}%); by animal "
          f"{(100 * d.assign(x=anynan).groupby('animal')['x'].mean()).round(1).to_dict()}")
    print(f"  dropped within-epoch pos {d.loc[anynan.values, 'pos'].mean():.2f} "
          f"(kept {d.loc[~anynan.values, 'pos'].mean():.2f})")


if __name__ == "__main__":
    main()
