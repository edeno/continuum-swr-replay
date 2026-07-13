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
    within an epoch, trial t spans [poke_out_ts[t-1], poke_out_ts[t]);
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
from scipy.special import betaln, digamma

from continuum_swr_replay.analysis.replay.standardization import (
    assert_standardized_present,
    attach_standardized,
    save_reference,
)
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
# Continuous stop-level covariates standardized to pooled-GLOBAL unit SD as ``<col>_z``
# by this builder (after the all-epoch concat). This is the qvals side of the persisted
# _z that used to be stamped post-hoc by _add_standardized_cols; prepare_data only READS
# this table and must not re-standardize it.
QVALS_STANDARDIZED_COLS = [
    "trials_since_switch", "trials_since_block_change",
    # beta-bernoulli value / decision / uncertainty block:
    "rpe", "switch_advantage_mu_preoutcome_sess",
    "global_mu_update_abs_postoutcome_sess", "beta_reward_entropy",
]
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
    # per-port model value (alias of Q1..6, 0-indexed) for the content/contfrag
    # builders: q_port{arm} -> q_arm (attach_contingency_value_block).
    *[f"q_port{i}" for i in range(6)],
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
    # beta belief-entropy (mean per-leaf Beta differential entropy)
    "beta_reward_entropy",
    # depletion / choice probabilities
    *[f"depletion_leaf_{i}" for i in range(1, 7)],
    *[f"depletion_mean_stem_{i}" for i in range(1, 4)],
    "stem_choice_p", "leaf_choice_p", "stem_stay_p", *[f"stem_{i}_p" for i in range(1, 4)],
]

# Nominal schedule covariates derived from TrialsInfoByEpoch (the same table read
# for poke times, so aligned by the same poke_out rule -- no difflib/CSV). Source
# is Alison's `p_rew_leaf1..6` (nominal p(R), percent) + `contingency_count`;
# value_diff_patch / trials_since_block_change use this project's definitions
# (Alison computes no nominal value-diff -- her decision variables are model-based).
NOMINAL_COVARIATES = [
    "p_reward_current", "p_reward_best", "value_diff_patch", "trials_since_block_change",
    # peri-switch trial index (signed distance to the nearest patch switch) +
    # post-switch indicator; the paper's "trials since switch" axis (Comrie et al Fig 2).
    "trials_since_switch", "is_post_switch",
    # per-port nominal p(R); the content/contfrag builders map p_reward_port{arm}
    # -> p_reward_arm (attach_contingency_value_block).
    *[f"p_reward_port{i}" for i in range(6)],
]
# leaf 1-6 (0-indexed arm) -> patch (WELL_PATCH_MAPPING); for the nominal patch value.
WELL_PATCH = np.array([0, 0, 1, 1, 2, 2])


def peri_switch_index(patch: np.ndarray, clip: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Signed trials-to-nearest-patch-switch + post-switch flag for one epoch.

    A *switch* is a trial whose patch differs from the previous trial's. For each
    trial, returns its signed distance to the **nearest** switch: ``+k`` if the
    nearest switch is ``k`` trials behind (post-switch, departing), ``-k`` if it is
    ``k`` trials ahead (pre-switch, approaching), ``0`` at a switch trial. This is
    the paper's "trials since Switch trial" axis (Comrie et al. Fig 2), signed so a
    piecewise-linear fit recovers the pre- and post-switch slopes.

    Parameters
    ----------
    patch : np.ndarray, shape (n_trials,)
        Patch id per trial, in temporal (trial) order.
    clip : int
        Clip magnitude to ``[-clip, clip]``.

    Returns
    -------
    trials_since_switch, is_post_switch : np.ndarray, each shape (n_trials,)
        Signed peri-switch index and the post/pre indicator (1 post, 0 pre). Both
        ``NaN`` when the epoch has no switch (index undefined).
    """
    n = len(patch)
    if n == 0:
        return np.array([]), np.array([])
    switch = np.zeros(n, dtype=bool)
    switch[1:] = patch[1:] != patch[:-1]
    pos = np.where(switch)[0]
    if pos.size == 0:
        return np.full(n, np.nan), np.full(n, np.nan)
    idx = np.arange(n)
    li = np.searchsorted(pos, idx, side="right") - 1        # last switch at/before idx
    since = np.where(li >= 0, idx - pos[np.clip(li, 0, None)], np.inf)
    ni = np.searchsorted(pos, idx, side="right")            # first switch after idx
    until = np.where(ni < pos.size, pos[np.clip(ni, 0, pos.size - 1)] - idx, np.inf)
    post = since <= until                                    # nearest switch is behind (ties -> post)
    signed = np.clip(np.where(post, since, -until).astype(float), -clip, clip)
    return signed, post.astype(float)


def _add_nominal_contingency(times: pd.DataFrame) -> pd.DataFrame:
    """Add nominal schedule covariates from p_rew_leaf1..6 (percent) + contingency_count.

    p_reward_current / p_reward_best come straight from Alison's nominal p_rew_leaf.
    value_diff_patch = best-minus-second patch mean (patch = the 2 leaves sharing a
    stem); trials_since_block_change = position within the current contingency block;
    trials_since_switch / is_post_switch = signed peri-switch index (see
    :func:`peri_switch_index`). All on the exactly-aligned per-trial rows.
    """
    times = times.sort_values(
        ["nwb_file_name", "epoch", "trial_number_by_epoch"]
    ).reset_index(drop=True)
    # guard the fancy-index below (ports[arange, leaf-1]); leaf must be 1..6.
    if not times["leaf"].between(1, 6).all():
        raise ValueError(f"times leaf outside 1..6: {sorted(times['leaf'].unique())}")
    ports = times[[f"p_rew_leaf{i}" for i in range(1, 7)]].to_numpy(dtype=float) / 100.0
    arm = times["leaf"].to_numpy() - 1
    times["p_reward_current"] = ports[np.arange(len(times)), arm]
    times["p_reward_best"] = ports.max(axis=1)
    for i in range(6):
        times[f"p_reward_port{i}"] = ports[:, i]  # per-port nominal p(R)
    patch_val = np.stack([ports[:, WELL_PATCH == p].mean(axis=1) for p in range(3)], axis=1)
    srt = np.sort(patch_val, axis=1)
    times["value_diff_patch"] = srt[:, -1] - srt[:, -2]
    times["trials_since_block_change"] = times.groupby(
        ["nwb_file_name", "epoch", "contingency_count"]
    ).cumcount()
    # peri-switch index: patch = the stem the leaf sits on (WELL_PATCH); a switch is
    # a between-trial patch change. Computed per epoch on the trial-ordered rows.
    patch_all = WELL_PATCH[times["leaf"].to_numpy() - 1]
    tss = np.full(len(times), np.nan)
    isp = np.full(len(times), np.nan)
    for _, g in times.groupby(["nwb_file_name", "epoch"], sort=False):
        rows = g.index.to_numpy()
        tss[rows], isp[rows] = peri_switch_index(patch_all[rows])
    times["trials_since_switch"] = tss
    times["is_post_switch"] = isp
    return times


def add_experienced_value_columns(vals: pd.DataFrame) -> pd.DataFrame:
    """Derive the experienced-value columns from a per-trial beta-model frame.

    Adds ``q_chosen`` (model value of the chosen arm = ``Q{leaf}``), ``arm``
    (``leaf - 1``, 0-indexed), ``q_port{0..5}`` (per-arm model value ``Q{i+1}``),
    ``rpe`` (``reward - q_chosen``) and ``rpe_abs``. Mutates and returns ``vals``.

    Requires columns ``Q1..Q6``, ``leaf`` (1..6) and ``reward``. Raises
    ``ValueError`` if ``leaf`` is outside 1..6 -- the fancy-index below would
    otherwise wrap (``leaf=0 -> Q6``) and silently mis-map every chosen value.
    """
    if not vals["leaf"].between(1, 6).all():
        raise ValueError(f"leaf outside 1..6: {sorted(vals['leaf'].unique())}")
    qmat = vals[[f"Q{i}" for i in range(1, 7)]].to_numpy()
    arm = vals["leaf"].to_numpy() - 1
    vals["q_chosen"] = qmat[np.arange(len(vals)), arm]
    vals["arm"] = arm
    for i in range(6):
        vals[f"q_port{i}"] = qmat[:, i]  # per-port model value (Q{i+1}), for q_arm
    vals["rpe"] = vals["reward"].to_numpy() - vals["q_chosen"].to_numpy()
    vals["rpe_abs"] = vals["rpe"].abs()
    return vals


def add_beta_belief_entropy(vals: pd.DataFrame) -> pd.DataFrame:
    """Add ``beta_reward_entropy``: mean per-leaf Beta differential entropy.

    The beta-bernoulli analog of the retired HMM reward-state entropy -- a per-trial scalar
    summarizing how uncertain the reward-probability beliefs are across the six
    leaves. For leaf ``i`` with posterior ``Beta(a_i, b_i)`` (the fit's native
    ``betadist_α{i}`` / ``betadist_β{i}``, both >= 1), the differential entropy is

        H_i = betaln(a, b) - (a-1)ψ(a) - (b-1)ψ(b) + (a+b-2)ψ(a+b),

    averaged over the six leaves. Differential entropy can be negative (unlike the
    HMM discrete Shannon entropy) -- expected; the covariate is used standardized.
    Mutates and returns ``vals``. Raises if the posterior params are absent.
    """
    missing = [f"betadist_{g}{i}" for i in range(1, 7) for g in ("α", "β")
               if f"betadist_{g}{i}" not in vals.columns]
    if missing:
        raise ValueError(
            f"beta posterior params missing {missing}; cannot build beta_reward_entropy"
        )
    ent = np.zeros(len(vals))
    for i in range(1, 7):
        a = vals[f"betadist_α{i}"].to_numpy(dtype=float)
        b = vals[f"betadist_β{i}"].to_numpy(dtype=float)
        ent += (betaln(a, b) - (a - 1) * digamma(a) - (b - 1) * digamma(b)
                + (a + b - 2) * digamma(a + b))
    vals["beta_reward_entropy"] = ent / 6.0
    return vals


def assign_stops_to_trials(
    poke_out_ts: np.ndarray, stop_mid: np.ndarray
) -> np.ndarray:
    """Assign each stop midpoint to a trial by poke-out interval.

    Alison's rule: trial ``t`` spans ``[poke_out[t-1], poke_out[t])`` (with
    ``poke_out[-1] = -inf``), reproduced by ``searchsorted(..., side="right")``.

    Parameters
    ----------
    poke_out_ts : np.ndarray, shape (n_trials,)
        Per-trial poke-out times, sorted ascending (temporal trial order).
    stop_mid : np.ndarray, shape (n_stops,)
        Stop midpoint times.

    Returns
    -------
    np.ndarray, shape (n_stops,)
        Trial index per stop, or ``-1`` for stops at or after the last poke-out
        (no enclosing trial -> dropped by the caller).
    """
    ti = np.searchsorted(poke_out_ts, stop_mid, side="right")
    return np.where(ti < len(poke_out_ts), ti, -1)


def _fetch_model_df(rl: Any, subject: str, key: dict) -> pd.DataFrame:
    """Read a BehaviorModelResults dataframe (per-trial, keyed by KEYS)."""
    # Deferred import: pulling spyglass at module top opens a DataJoint
    # connection on import, which would make DB-free unit tests of the pure
    # helpers (e.g. _add_nominal_contingency) require live DB access.
    from spyglass.common import AnalysisNwbfile
    from spyglass.utils.dj_helper_fn import fetch_nwb

    sel = rl.BehaviorModelResults & {"subject_id": subject, **key}
    results = fetch_nwb(sel, (AnalysisNwbfile, "analysis_file_abs_path"))
    if not results:
        raise ValueError(f"no BehaviorModelResults for {subject!r} with {key}")
    return results[0]["behavior_model_results"].reset_index(drop=True)


def _per_trial(animal: str, beh: Any, rl: Any) -> tuple[pd.DataFrame, list[str]]:
    """Join enriched model covariates to poke timestamps, one row per trial.

    Beta values are enriched with the vendored decision-variable pipeline
    (DeltaV, value updates, belief entropy, etc.) and RPE/q_chosen/arm are
    derived; the curated COVARIATES are broadcast to stops.
    """
    subject = ANIMAL_SUBJECT[animal]
    vals = make_full_beta_bernoulli_dv_df(_fetch_model_df(rl, subject, MODEL_KEY))
    vals = add_experienced_value_columns(vals)  # q_chosen / arm / q_port / rpe
    vals = add_beta_belief_entropy(vals)  # beta_reward_entropy (from betadist_α/β)
    model_cols = [c for c in COVARIATES if c in vals.columns]
    pt_attrs_entropy = float(vals["beta_reward_entropy"].notna().mean())

    nwbs = vals["nwb_file_name"].unique().tolist()
    times = pd.DataFrame((beh.TrialsInfoByEpoch.ByTrial
                          & [{"nwb_file_name": n} for n in nwbs]).fetch())
    # epoch arrives as an int epoch-number from DataJoint; if it were a string the
    # epoch_number join below would silently drop every stop.
    if not pd.api.types.is_integer_dtype(times["epoch"].dtype):
        raise TypeError(f"expected integer epoch from DataJoint, got {times['epoch'].dtype}")
    times = _add_nominal_contingency(times)
    times = times[[*KEYS, *TIME_COLS, "leaf", "reward", *NOMINAL_COVARIATES]]

    pt = times.merge(vals[[*KEYS, "leaf", "reward", *model_cols]], on=KEYS,
                     suffixes=("_time", "_val"), validate="one_to_one")
    pt.attrs["leaf_agree"] = float((pt["leaf_time"] == pt["leaf_val"]).mean())
    pt.attrs["entropy_cov"] = pt_attrs_entropy
    # nominal block (from times) + model block (from vals) are both broadcast to stops
    return pt, [*model_cols, *NOMINAL_COVARIATES]


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
        # trial t = [poke_out[t-1], poke_out[t]); stops past the last poke_out
        # get -1 and drop out.
        ti = assign_stops_to_trials(po, mid[s_idx])
        valid = ti >= 0
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
        # arm / reward agreement is only meaningful at at-well (in_poke) stops:
        # on transit stops the mapped trial's destination leaf (arm 0-5) need not
        # equal the physical current_arm (which can be a trunk arm 6-8), so a
        # whole-set figure understates agreement and would trip a spurious
        # mapping-regression warning. Restrict the checks to in_poke stops.
        poke = t["in_poke"].to_numpy()
        cur = obs["current_arm"].reindex(t["stop_id"]).to_numpy()
        rout = obs["reward_outcome"].reindex(t["stop_id"]).to_numpy()
        arm_ok = np.mean(t["arm"].to_numpy()[poke] == cur[poke])
        rew_ok = np.mean(t["trial_reward"].to_numpy()[poke] == rout[poke])
        entropy_cov = t.attrs["entropy_cov"]
        print(f"{a:7s} mapped {len(t):6d}/{len(st):6d} stops ({100*len(t)/len(st):.0f}%)  "
              f"in_poke={100*t['in_poke'].mean():.0f}%  "
              f"at-well arm=={100*arm_ok:.1f}%  at-well reward=={100*rew_ok:.1f}%  "
              f"[leaf-join {100*t.attrs['leaf_agree']:.1f}%  entropy {100*entropy_cov:.0f}%]")
        if len(t) / len(st) < 0.90 or arm_ok < 0.95 or entropy_cov < 0.5:
            print(f"  WARNING [{a}]: coverage {100*len(t)/len(st):.0f}% (expect ~97-99%), "
                  f"at-well arm agreement {100*arm_ok:.1f}% (expect ~100%), or "
                  f"beta_reward_entropy coverage {100*entropy_cov:.0f}% (expect ~100%; a low "
                  f"figure voids the belief-entropy covariate) -- possible mapping/model regression")
        parts.append(t)
    tab = pd.concat(parts, ignore_index=True)
    print(f"\ntotal mapped stops: {len(tab)}  unique: {tab.stop_id.nunique()}  cols: {tab.shape[1]}")

    # Fail loudly if the per-arm value block is incomplete: model_cols filters
    # COVARIATES by presence, so an upstream BehaviorModelResults/schema change
    # dropping a Q{i}/p_rew_leaf{i} would silently omit a per-port column and
    # make attach_contingency_value_block blank the whole value block downstream.
    required = {"stop_id", "p_reward_current", "q_chosen", *[f"p_reward_port{i}" for i in range(6)],
                *[f"q_port{i}" for i in range(6)], "beta_reward_entropy"}
    missing = required - set(tab.columns)
    if missing:
        raise ValueError(f"qvals_table is missing required columns {sorted(missing)}")

    # Pooled-GLOBAL standardization of the continuous stop-level covariates (<col>_z),
    # owned by THIS builder (prepare_data reads qvals but must not re-standardize it).
    # Computed over the full concatenated table; reference persisted for reproducibility.
    tab, qv_reference = attach_standardized(tab, QVALS_STANDARDIZED_COLS)
    assert_standardized_present(tab, QVALS_STANDARDIZED_COLS)
    z_cols = [f"{c}_z" for c in QVALS_STANDARDIZED_COLS if c in tab.columns]

    if args.write:
        out = RC / "models_per_bin_rate" / "qvals_table.pkl"
        tab.to_pickle(out)
        save_reference(
            qv_reference,
            RC / "models_per_bin_rate" / "qvals_standardization_reference.json",
        )
        print(f"\nwrote {len(tab)} rows ({len(z_cols)} _z cols) -> {out}")


if __name__ == "__main__":
    main()
