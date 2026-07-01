#!/usr/bin/env python
"""Per-stop RL-model values via DataJoint — canonical, complete, timestamp-aligned.

Builds the per-stop behavioral-model covariate table (`qvals_table.pkl`),
sourcing everything from Alison's shared DataJoint schemas (read-only):

  * VALUES: `alison_rlmodel.BehaviorModelResults` (the `beta_stable_withleaf`
    fit), read from its AnalysisNwbfile — the q_vals model output re-keyed to
    (nwb_file_name, epoch, trial_number_by_epoch).
  * TIMES: `alison_behav.TrialsInfoByEpoch.ByTrial` — poke_in_ts / poke_out_ts /
    reward_on_ts / reward_off_ts per (nwb_file_name, epoch, trial_number_by_epoch),
    stored as MySQL blobs (complete for all 5 animals, incl. peanut).
  * JOIN: values x times on (nwb_file_name, epoch, trial_number_by_epoch).
  * STOP -> TRIAL: Alison's rule (alison_decoding.add_trial_info_to_clusterless_results):
    within an epoch, trial t spans (poke_out_ts[t-1], poke_out_ts[t]];
    i.e. trial = searchsorted(poke_out_ts, stop_time, side='right'); stops at or
    after the last poke_out are unassigned (her trailing-NaN drop).

Beta values are enriched with the vendored decision-variable pipeline
(`analysis.behavior._spatial_bandit_rl.make_full_beta_bernoulli_dv_df`: DeltaV,
Stay/Switch value, value-update magnitudes, betadist_mu), HMM reward/state entropy
(`default_hmm`) is merged on KEYS, and RPE is derived; the curated COVARIATES are
broadcast to stops. See docs/behavior_model_covariates.md for the data dictionary.

Virtual modules reflect the schema from the DB, so none of Alison's module code
(julia / spyglass / os.chdir) is imported. Independent validation: mapped arm
(leaf-1) vs position `current_arm`; mapped reward vs DIO `reward_outcome`.

Usage: python -m continuum_swr_replay.analysis.behavior.qvals_table [--animal A] [--write]
Output: Processed-Data/replay_content/models_per_bin_rate/qvals_table.pkl
"""

from __future__ import annotations

import argparse
import warnings
from typing import Any

import datajoint as dj
import numpy as np
import pandas as pd
from spyglass.common import AnalysisNwbfile
from spyglass.utils.dj_helper_fn import fetch_nwb

from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

from ._spatial_bandit_rl import make_full_beta_bernoulli_dv_df

# the vendored DV pipeline inserts many columns one-by-one (Alison's style)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

RC = PROCESSED_DATA_DIR / "replay_content"
ANIMALS = ("chimi", "j16", "peanut", "senor", "wilbur")
# stop_table uses lowercase animal labels; BehaviorModel capitalizes Senor.
ANIMAL_SUBJECT = {"chimi": "chimi", "j16": "j16", "peanut": "peanut",
                  "senor": "Senor", "wilbur": "wilbur"}
MODEL_KEY = {"behavior_model_params_name": "beta_stable_withleaf",
             "trials_info_by_rat_params_name": "stable_default",
             "trials_info_csv_params_name": "original_csv_format"}
# default_hmm fit on the same trials -> the paper's reward-state entropy covariate.
HMM_KEY = {"behavior_model_params_name": "default_hmm",
           "trials_info_by_rat_params_name": "stable_default",
           "trials_info_csv_params_name": "original_csv_format"}
HMM_COLS = ["state_entropy", "reward_entropy"]
KEYS = ["nwb_file_name", "epoch", "trial_number_by_epoch"]
TIME_COLS = ["poke_in_ts", "poke_out_ts", "reward_on_ts", "reward_off_ts"]
# Beta value timing: row t holds PRE-outcome belief for trial t (vendored DV docs).

# Curated per-stop covariate set (paper-aligned): the Comrie et al. theta-sequences decision
# variables (computed by the vendored beta-bernoulli pipeline), HMM reward
# entropy, RPE, and the raw value/uncertainty/choice columns.
COVARIATES = [
    "arm", "q_chosen", "rpe", "rpe_abs",
    # expected value (per port / patch / belief mean)
    *[f"Q{i}" for i in range(1, 7)], *[f"Qstem{i}" for i in range(1, 4)],
    *[f"betadist_mu{i}" for i in range(1, 7)],
    # decision variables: Stay / Switch / Relative (DeltaV)
    "stay_sibling_mu_preoutcome_sess", "best_other_patch_mean_mu_preoutcome_sess",
    "switch_advantage_mu_preoutcome_sess", "switch_advantage_preoutcome_sess_Qstem",
    # absolute value-update magnitudes, 4 scopes (paper) + global signed
    "global_mu_update_abs_postoutcome_sess", "chosen_leaf_mu_update_abs_postoutcome_sess",
    "chosen_patch_mu_update_abs_postoutcome_sess", "unchosen_patches_mu_update_abs_postoutcome_sess",
    "global_mu_update_signed_postoutcome_sess",
    # uncertainty
    *[f"betadist_var{i}" for i in range(1, 7)],
    "global_leaf_var_sum_preoutcome_sess", "chosen_leaf_var_preoutcome_sess",
    "prev_local_variance", "prev_global_variance", "upcoming_leaf_variance", "next_choice_variance",
    # reward-state entropy (HMM)
    "state_entropy", "reward_entropy",
    # depletion / choice probabilities
    *[f"depletion_leaf_{i}" for i in range(1, 7)],
    *[f"depletion_mean_stem_{i}" for i in range(1, 4)],
    "stem_choice_p", "leaf_choice_p", "stem_stay_p", *[f"stem_{i}_p" for i in range(1, 4)],
]


def _fetch_model_df(rl: Any, subject: str, key: dict) -> pd.DataFrame:
    """Read a BehaviorModelResults dataframe (per-trial, keyed by KEYS)."""
    sel = rl.BehaviorModelResults & {"subject_id": subject, **key}
    results = fetch_nwb(sel, (AnalysisNwbfile, "analysis_file_abs_path"))
    if not results:
        raise ValueError(f"no BehaviorModelResults for {subject!r} with {key}")
    return results[0]["behavior_model_results"].reset_index(drop=True)


def _per_trial(animal: str, beh: Any, rl: Any) -> tuple[pd.DataFrame, list[str]]:
    """Join enriched model covariates to poke timestamps, one row per trial.

    Beta values are enriched with the vendored decision-variable pipeline
    (DeltaV, value updates, etc.), HMM reward entropy is merged on KEYS, and
    RPE/q_chosen/arm are derived; the curated COVARIATES are broadcast to stops.
    """
    subject = ANIMAL_SUBJECT[animal]
    vals = make_full_beta_bernoulli_dv_df(_fetch_model_df(rl, subject, MODEL_KEY))
    hmm = _fetch_model_df(rl, subject, HMM_KEY)[[*KEYS, *HMM_COLS]]
    vals = vals.merge(hmm, on=KEYS, how="left", validate="one_to_one")
    # leaf must be 1..6: the fancy-index below would else wrap (leaf=0 -> Q6) or
    # raise, silently mis-mapping q_chosen / arm.
    if not vals["leaf"].between(1, 6).all():
        raise ValueError(f"leaf outside 1..6 for {subject!r}: {sorted(vals['leaf'].unique())}")
    qmat = vals[[f"Q{i}" for i in range(1, 7)]].to_numpy()
    vals["q_chosen"] = qmat[np.arange(len(vals)), vals["leaf"].to_numpy() - 1]
    vals["arm"] = vals["leaf"].to_numpy() - 1
    vals["rpe"] = vals["reward"].to_numpy() - vals["q_chosen"].to_numpy()
    vals["rpe_abs"] = vals["rpe"].abs()
    model_cols = [c for c in COVARIATES if c in vals.columns]
    pt_attrs_entropy = float(vals["reward_entropy"].notna().mean())

    nwbs = vals["nwb_file_name"].unique().tolist()
    times = pd.DataFrame((beh.TrialsInfoByEpoch.ByTrial
                          & [{"nwb_file_name": n} for n in nwbs]).fetch())
    times = times[[*KEYS, *TIME_COLS, "leaf", "reward"]]
    # epoch arrives as an int epoch-number from DataJoint; if it were a string the
    # epoch_number join below would silently drop every stop.
    if not pd.api.types.is_integer_dtype(times["epoch"].dtype):
        raise TypeError(f"expected integer epoch from DataJoint, got {times['epoch'].dtype}")

    pt = times.merge(vals[[*KEYS, "leaf", "reward", *model_cols]], on=KEYS,
                     suffixes=("_time", "_val"), validate="one_to_one")
    pt.attrs["leaf_agree"] = float((pt["leaf_time"] == pt["leaf_val"]).mean())
    pt.attrs["entropy_cov"] = pt_attrs_entropy
    return pt, model_cols


def build_animal(animal: str, beh: Any, rl: Any, stop_table: pd.DataFrame) -> pd.DataFrame:
    pt, model_cols = _per_trial(animal, beh, rl)
    st = stop_table[stop_table.animal == animal].copy()
    st["nwb_file_name"] = animal + st["day"].astype(str) + "_.nwb"
    st["epoch_number"] = st["epoch"].str.split("_").str[0].astype("int64")
    mid = ((st["stop_start"] + st["stop_end"]) / 2).to_numpy()

    rows = []
    for (nwb, epnum), g in pt.groupby(["nwb_file_name", "epoch"]):
        g = g.sort_values("trial_number_by_epoch").reset_index(drop=True)
        po, pi = g["poke_out_ts"].to_numpy(), g["poke_in_ts"].to_numpy()
        ron, roff = g["reward_on_ts"].to_numpy(), g["reward_off_ts"].to_numpy()
        # searchsorted needs sorted poke_out; trial order is temporal, so this holds.
        assert (np.diff(po) >= 0).all(), f"poke_out_ts not sorted: {nwb} epoch {epnum}"
        s_idx = np.where(((st["nwb_file_name"] == nwb) & (st["epoch_number"] == epnum)).to_numpy())[0]
        if len(s_idx) == 0:
            continue
        # Alison's rule: trial t = [poke_out[t-1], poke_out[t]) (her between inclusive='left');
        # searchsorted side='right' reproduces it. Stops past the last poke_out drop out.
        ti = np.searchsorted(po, mid[s_idx], side="right")
        valid = ti < len(g)
        for k, se in zip(s_idx[valid], ti[valid], strict=True):
            trow = g.iloc[int(se)]
            rows.append({
                "stop_id": st.iloc[k]["stop_id"],
                "epoch": st.iloc[k]["epoch"],
                "trial_number_by_epoch": int(trow["trial_number_by_epoch"]),
                "in_poke": bool(pi[se] <= mid[k] <= po[se]),
                "is_rewarding": bool((not np.isnan(ron[se])) and ron[se] <= mid[k] <= roff[se]),
                "trial_reward": int(trow["reward_time"]),  # poke-record reward (times' 'reward', '_time' suffix)
                **{c: trow[c] for c in model_cols},
            })
    out = pd.DataFrame(rows)
    out.attrs["leaf_agree"] = pt.attrs["leaf_agree"]
    out.attrs["entropy_cov"] = pt.attrs["entropy_cov"]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--animal", choices=ANIMALS, default=None)
    p.add_argument("--write", action="store_true")
    args = p.parse_args()
    animals = [args.animal] if args.animal else list(ANIMALS)

    beh = dj.create_virtual_module("alison_behav", "alison_behav")
    rl = dj.create_virtual_module("alison_rlmodel", "alison_rlmodel")
    st_all = pd.read_pickle(RC / "stop_table.pkl")

    parts = []
    for a in animals:
        t = build_animal(a, beh, rl, st_all)
        st = st_all[st_all.animal == a]
        obs = st.set_index("stop_id")
        arm_ok = np.mean(t["arm"].to_numpy() == obs["current_arm"].reindex(t["stop_id"]).to_numpy())
        rew_ok = np.mean(t["trial_reward"].to_numpy() == obs["reward_outcome"].reindex(t["stop_id"]).to_numpy())
        print(f"{a:7s} mapped {len(t):6d}/{len(st):6d} stops ({100*len(t)/len(st):.0f}%)  "
              f"in_poke={100*t['in_poke'].mean():.0f}%  "
              f"arm==current_arm {100*arm_ok:.1f}%  reward==reward_outcome {100*rew_ok:.1f}%  "
              f"[leaf-join {100*t.attrs['leaf_agree']:.1f}%  entropy {100*t.attrs['entropy_cov']:.0f}%]")
        if len(t) / len(st) < 0.90 or arm_ok < 0.95:
            print(f"  WARNING [{a}]: coverage {100*len(t)/len(st):.0f}% (expect ~97-99%) or "
                  f"arm agreement {100*arm_ok:.1f}% (expect ~100%) -- possible mapping regression")
        parts.append(t)
    tab = pd.concat(parts, ignore_index=True)
    print(f"\ntotal mapped stops: {len(tab)}  unique: {tab.stop_id.nunique()}  cols: {tab.shape[1]}")

    if args.write:
        out = RC / "models_per_bin_rate" / "qvals_table.pkl"
        tab.to_pickle(out)
        print(f"\nwrote {len(tab)} rows -> {out}")


if __name__ == "__main__":
    main()
