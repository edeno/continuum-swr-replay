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
# default_hmm = the paper's "Global Model" (Comrie et al.). We derive the
# paper-faithful decision variables from it (single-model, unlike the beta block):
# see add_hmm_decision_variables. Fetch the native columns those formulas need.
HMM_KEY = {"behavior_model_params_name": "default_hmm",
           "trials_info_by_rat_params_name": "stable_default",
           "trials_info_csv_params_name": "original_csv_format"}
HMM_COLS = ["state_entropy", "reward_entropy"]
# paper-faithful HMM decision variables (canonical dimensions for the is_swr rate
# analysis): relative value (val_difference), RPE, global value update -- all from
# default_hmm (single model, matches Comrie et al. Fig 1/5). The entropy dimension
# uses the native HMM `reward_entropy` (in HMM_COLS); the paper's port-specific
# `reward_entropy_current_leaf` is absent from this (HMM_KEY-aligned) fit.
HMM_DV_COLS = ["hmm_relative_value", "hmm_rpe", "hmm_value_update"]
# Per-leaf HMM value (native Q1..6 of the Global Model), exposed 0-indexed as the
# HMM analog of the beta ``q_port{0..5}`` block so the content/choice builders can
# form a *single-model* per-arm value (``q_arm_hmm``) harmonized with the HMM value
# covariates the is_swr rate ladder uses -- see attach_contingency_value_block.
HMM_Q_PORT_COLS = [f"hmm_q_port{i}" for i in range(6)]
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
    # reward-state entropy (HMM)
    "state_entropy", "reward_entropy",
    # paper-faithful HMM decision variables (single-model canonical set; Comrie et al.)
    *HMM_DV_COLS,
    # per-leaf HMM value (Q1..6, 0-indexed) -> q_arm_hmm, the HMM per-arm value
    *HMM_Q_PORT_COLS,
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


def add_hmm_decision_variables(hmm: pd.DataFrame) -> pd.DataFrame:
    """Paper-faithful decision variables from the HMM ("Global Model", Comrie et al.).

    Ports the value-relevant parts of Alison's ``rl_helpers`` HMM pipeline
    (``add_rpe_deltaq_to_hmm_plus`` + ``create_hmm_plus``'s ``val_difference``)
    directly onto ``default_hmm``'s native columns, so the value / RPE / update /
    entropy covariates all come from **one** model (vs the beta block's mix). Adds:

    - ``hmm_relative_value`` -- ΔV: mean of the other stems minus the initial
      (starting) stem value, ``(ΣQstem - Qstem_initial)/2 - Qstem_initial`` (Fig 1F).
    - ``hmm_rpe`` -- the canonical RPE ``reward - Q_current_leaf`` (Fig 5).
      Alison additionally rescales Q *before* the subtraction
      (``(Q_leaf + 0.6) / 1.25``, her own "forced rescaling to get in zone of
      0 to 1"), giving ``reward - 0.8*Q_leaf - 0.48``. That is a cosmetic
      display transform, NOT an affine transform of ``reward - Q_leaf`` (it
      reweights Q relative to reward), so we deliberately use the unscaled
      textbook RPE here rather than reproduce the plotting rescale.
    - ``hmm_value_update`` -- global update ``Σ_leaf |Q_leaf(t+1) - Q_leaf(t)|``
      (``delta_Qglobal_t_abs``, Fig 5D-right); NaN on the last trial of an epoch.

    (The entropy dimension uses the native ``reward_entropy`` carried via HMM_COLS,
    not computed here.) Requires ``default_hmm`` native cols (``Q1..6``, ``Qstem1..3``,
    ``leaf``, ``stemchoice``, ``reward``) + KEYS. Per-epoch (no cross-epoch bleed).
    """
    required = [
        *[f"Q{i}" for i in range(1, 7)],
        *[f"Qstem{i}" for i in range(1, 4)],
        "leaf",
        "stemchoice",
        "reward",
    ]
    missing = [c for c in required if c not in hmm.columns]
    if missing:
        raise ValueError(
            f"HMM fit (default_hmm) missing native cols {missing}; wrong "
            "BehaviorModelResults fit restriction?"
        )
    d = hmm.sort_values(KEYS).reset_index(drop=True)
    # leaf/stemchoice index into length-6/length-3 Q arrays as `x - 1`; an
    # out-of-range value would silently wrap (0 -> -1 -> last column) or
    # truncate a float, corrupting the value covariate with no error. Guard
    # both, allowing the first-trial NaN that stemchoice.shift() introduces.
    if not d["leaf"].between(1, 6).all():
        raise ValueError(f"HMM leaf outside 1..6: {sorted(d['leaf'].unique())}")
    sc = d["stemchoice"].dropna()
    if not sc.between(1, 3).all():
        raise ValueError(f"HMM stemchoice outside 1..3: {sorted(sc.unique())}")
    g = d.groupby(["nwb_file_name", "epoch"], sort=False)
    qmat = d[[f"Q{i}" for i in range(1, 7)]].to_numpy()
    # Expose the per-leaf HMM value 0-indexed (hmm_q_port{i} = Q{i+1}), the HMM
    # counterpart of the beta q_port block, for the per-arm value harmonization.
    for i in range(6):
        d[f"hmm_q_port{i}"] = qmat[:, i]
    leaf = d["leaf"].to_numpy().astype(int)
    q_leaf_t = qmat[np.arange(len(d)), leaf - 1]
    d["hmm_rpe"] = d["reward"].to_numpy() - q_leaf_t
    nxt = np.column_stack([g[f"Q{i}"].shift(-1).to_numpy() for i in range(1, 7)])
    d["hmm_value_update"] = np.abs(nxt - qmat).sum(axis=1)  # NaN at epoch's last trial
    qstem = d[[f"Qstem{i}" for i in range(1, 4)]].to_numpy()
    si = g["stemchoice"].shift().to_numpy()  # the *initial* (starting) stem this trial
    q_init = np.full(len(d), np.nan)
    ok = ~np.isnan(si)
    q_init[ok] = qstem[np.where(ok)[0], si[ok].astype(int) - 1]
    d["hmm_relative_value"] = (qstem.sum(axis=1) - q_init) / 2 - q_init
    return d


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
    (DeltaV, value updates, etc.), HMM reward entropy is merged on KEYS, and
    RPE/q_chosen/arm are derived; the curated COVARIATES are broadcast to stops.
    """
    subject = ANIMAL_SUBJECT[animal]
    vals = make_full_beta_bernoulli_dv_df(_fetch_model_df(rl, subject, MODEL_KEY))
    hmm = add_hmm_decision_variables(_fetch_model_df(rl, subject, HMM_KEY))
    hmm = hmm[[*KEYS, *HMM_COLS, *HMM_DV_COLS, *HMM_Q_PORT_COLS]]
    vals = vals.merge(hmm, on=KEYS, how="left", validate="one_to_one")
    vals = add_experienced_value_columns(vals)  # q_chosen / arm / q_port / rpe
    model_cols = [c for c in COVARIATES if c in vals.columns]
    pt_attrs_entropy = float(vals["reward_entropy"].notna().mean())

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
                  f"reward_entropy coverage {100*entropy_cov:.0f}% (expect >=50%; a low "
                  f"figure voids the RL entropy covariate) -- possible mapping/HMM regression")
        parts.append(t)
    tab = pd.concat(parts, ignore_index=True)
    print(f"\ntotal mapped stops: {len(tab)}  unique: {tab.stop_id.nunique()}  cols: {tab.shape[1]}")

    # Fail loudly if the per-arm value block is incomplete: model_cols filters
    # COVARIATES by presence, so an upstream BehaviorModelResults/schema change
    # dropping a Q{i}/p_rew_leaf{i} would silently omit a per-port column and
    # make attach_contingency_value_block blank the whole value block downstream.
    required = {"stop_id", "p_reward_current", "q_chosen", *[f"p_reward_port{i}" for i in range(6)],
                *[f"q_port{i}" for i in range(6)], *HMM_Q_PORT_COLS}
    missing = required - set(tab.columns)
    if missing:
        raise ValueError(f"qvals_table is missing required columns {sorted(missing)}")

    if args.write:
        out = RC / "models_per_bin_rate" / "qvals_table.pkl"
        tab.to_pickle(out)
        print(f"\nwrote {len(tab)} rows -> {out}")


if __name__ == "__main__":
    main()
