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
Stay/Switch value, value-update magnitudes, betadist_mu), the belief entropy
`beta_reward_entropy` is derived from the Beta posterior (`betadist_α/β`), and RPE
is derived; the curated COVARIATES are broadcast to stops. See docs/behavior_model_covariates.md for the data dictionary.

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

from continuum_swr_replay.analysis.constants import OMISSION_KNOWN_LATENCY_SEC
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
ANIMAL_SUBJECT = {
    "chimi": "chimi",
    "j16": "j16",
    "peanut": "peanut",
    "senor": "Senor",
    "wilbur": "wilbur",
}
MODEL_KEY = {
    "behavior_model_params_name": "beta_stable_withleaf",
    "trials_info_by_rat_params_name": "stable_default",
    "trials_info_csv_params_name": "original_csv_format",
}
# Continuous stop-level covariates standardized to pooled-GLOBAL unit SD as ``<col>_z``
# by this builder (after the all-epoch concat). This is the qvals side of the persisted
# _z that used to be stamped post-hoc by _add_standardized_cols; prepare_data only READS
# this table and must not re-standardize it.
QVALS_STANDARDIZED_COLS = [
    "trials_since_switch",
    "trials_since_block_change",
    # beta-bernoulli value / decision / uncertainty block:
    "rpe",
    "switch_advantage_mu_preoutcome_sess",
    "global_mu_update_abs_postoutcome_sess",
    "beta_reward_entropy",
]
KEYS = ["nwb_file_name", "epoch", "trial_number_by_epoch"]
TIME_COLS = ["poke_in_ts", "poke_out_ts", "reward_on_ts", "reward_off_ts"]
# Beta value timing: row t holds PRE-outcome belief for trial t (vendored DV docs).

# Curated per-stop covariate set (paper-aligned): the Comrie et al. theta-sequences decision
# variables (computed by the vendored beta-bernoulli pipeline), the derived beta
# belief entropy, RPE, and the raw value/uncertainty/choice columns.
COVARIATES = [
    "arm",
    "q_chosen",
    "rpe",
    "rpe_abs",
    # expected value (per port / patch / belief mean)
    *[f"Q{i}" for i in range(1, 7)],
    *[f"Qstem{i}" for i in range(1, 4)],
    *[f"betadist_mu{i}" for i in range(1, 7)],
    # per-port model value (alias of Q1..6, 0-indexed) for the content/contfrag
    # builders: q_port{arm} -> q_arm (attach_contingency_value_block).
    *[f"q_port{i}" for i in range(6)],
    # decision variables: Stay / Switch / Relative (DeltaV)
    "stay_sibling_mu_preoutcome_sess",
    "best_other_patch_mean_mu_preoutcome_sess",
    "switch_advantage_mu_preoutcome_sess",
    "switch_advantage_preoutcome_sess_Qstem",
    # absolute value-update magnitudes, 4 scopes (paper) + global signed
    "global_mu_update_abs_postoutcome_sess",
    "chosen_leaf_mu_update_abs_postoutcome_sess",
    "chosen_patch_mu_update_abs_postoutcome_sess",
    "unchosen_patches_mu_update_abs_postoutcome_sess",
    "global_mu_update_signed_postoutcome_sess",
    # uncertainty
    *[f"betadist_var{i}" for i in range(1, 7)],
    "global_leaf_var_sum_preoutcome_sess",
    "chosen_leaf_var_preoutcome_sess",
    "prev_local_variance",
    "prev_global_variance",
    "upcoming_leaf_variance",
    "next_choice_variance",
    # beta belief-entropy (mean per-leaf Beta differential entropy)
    "beta_reward_entropy",
    # depletion / choice probabilities
    *[f"depletion_leaf_{i}" for i in range(1, 7)],
    *[f"depletion_mean_stem_{i}" for i in range(1, 4)],
    "stem_choice_p",
    "leaf_choice_p",
    "stem_stay_p",
    *[f"stem_{i}_p" for i in range(1, 4)],
]

# Nominal schedule covariates derived from TrialsInfoByEpoch (the same table read
# for poke times, so aligned by the same poke_out rule -- no difflib/CSV). Source
# is Alison's `p_rew_leaf1..6` (nominal p(R), percent) + `contingency_count`;
# value_diff_patch / trials_since_block_change use this project's definitions
# (Alison computes no nominal value-diff -- her decision variables are model-based).
NOMINAL_COVARIATES = [
    "p_reward_current",
    "p_reward_best",
    "value_diff_patch",
    "trials_since_block_change",
    # peri-switch trial index (signed distance to the nearest patch switch) +
    # post-switch indicator; the paper's "trials since switch" axis (Comrie et al Fig 2).
    "trials_since_switch",
    "is_post_switch",
    # per-port nominal p(R); the content/contfrag builders map p_reward_port{arm}
    # -> p_reward_arm (attach_contingency_value_block).
    *[f"p_reward_port{i}" for i in range(6)],
]
# leaf 1-6 (0-indexed arm) -> patch (WELL_PATCH_MAPPING); for the nominal patch value.
WELL_PATCH = np.array([0, 0, 1, 1, 2, 2])


def peri_switch_index(
    patch: np.ndarray, clip: int = 20
) -> tuple[np.ndarray, np.ndarray]:
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
    li = np.searchsorted(pos, idx, side="right") - 1  # last switch at/before idx
    since = np.where(li >= 0, idx - pos[np.clip(li, 0, None)], np.inf)
    ni = np.searchsorted(pos, idx, side="right")  # first switch after idx
    until = np.where(ni < pos.size, pos[np.clip(ni, 0, pos.size - 1)] - idx, np.inf)
    post = since <= until  # nearest switch is behind (ties -> post)
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
    patch_val = np.stack(
        [ports[:, WELL_PATCH == p].mean(axis=1) for p in range(3)], axis=1
    )
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
    missing = [
        f"betadist_{g}{i}"
        for i in range(1, 7)
        for g in ("α", "β")
        if f"betadist_{g}{i}" not in vals.columns
    ]
    if missing:
        raise ValueError(
            f"beta posterior params missing {missing}; cannot build beta_reward_entropy"
        )
    ent = np.zeros(len(vals))
    for i in range(1, 7):
        a = vals[f"betadist_α{i}"].to_numpy(dtype=float)
        b = vals[f"betadist_β{i}"].to_numpy(dtype=float)
        ent += (
            betaln(a, b)
            - (a - 1) * digamma(a)
            - (b - 1) * digamma(b)
            + (a + b - 2) * digamma(a + b)
        )
    result = ent / 6.0
    # Guard the "α,β ≥ 1 → finite" assumption: a NaN/±inf here would otherwise flow
    # into the pooled-SD standardization and silently blank EVERY beta_reward_entropy_z
    # (and its downstream rate/content coefficients). Fail loud instead.
    if not np.isfinite(result).all():
        n_bad = int((~np.isfinite(result)).sum())
        raise ValueError(
            f"beta_reward_entropy has {n_bad} non-finite values; check betadist_α/β "
            "for NaN or non-positive (≤0) posterior params"
        )
    vals["beta_reward_entropy"] = result
    return vals


def assign_stops_to_trials(
    poke_out_ts: np.ndarray, stop_context_time: np.ndarray
) -> np.ndarray:
    """Assign each stop's frozen context time to a trial by poke-out interval.

    Alison's rule: trial ``t`` spans ``[poke_out[t-1], poke_out[t])`` (with
    ``poke_out[-1] = -inf``), reproduced by ``searchsorted(..., side="right")``.

    Parameters
    ----------
    poke_out_ts : np.ndarray, shape (n_trials,)
        Per-trial poke-out times, sorted ascending (temporal trial order).
    stop_context_time : np.ndarray, shape (n_stops,)
        Time at which stop-level covariates are frozen. Canonical callers use
        ``stop_start``; a speed bout can cross a trial boundary, so using its
        midpoint would attach a later trial than the rest of the stop table.

    Returns
    -------
    np.ndarray, shape (n_stops,)
        Trial index per stop, or ``-1`` for stops at or after the last poke-out
        (no enclosing trial -> dropped by the caller).
    """
    ti = np.searchsorted(poke_out_ts, stop_context_time, side="right")
    return np.where(ti < len(poke_out_ts), ti, -1)


def _fetch_model_df(rl: Any, subject: str, key: dict) -> pd.DataFrame:
    """Read a BehaviorModelResults dataframe (per-trial, keyed by KEYS)."""
    # Deferred import: pulling spyglass at module top opens a DataJoint
    # connection on import, which would make DB-free unit tests of the pure
    # helpers (e.g. _add_nominal_contingency) require live DB access.
    from spyglass.common import AnalysisNwbfile
    from spyglass.utils.nwb_helper_fn import get_nwb_file

    sel = rl.BehaviorModelResults & {"subject_id": subject, **key}
    # Resolve the AnalysisNwbfile by hand rather than via ``fetch_nwb``. The
    # helper only exists on Spyglass mixin tables, and ``rl`` is a virtual
    # module reflected from Alison's schema, whose tables are plain DataJoint.
    recs = sel.fetch(
        "analysis_file_name", "behavior_model_results_object_id", as_dict=True
    )
    if not recs:
        raise ValueError(f"no BehaviorModelResults for {subject!r} with {key}")
    if len(recs) > 1:
        raise ValueError(
            f"{len(recs)} BehaviorModelResults rows for {subject!r} with {key}; "
            "expected exactly one -- the model key no longer identifies a "
            "single fit, so which values get used would be arbitrary"
        )
    rec = recs[0]
    path = AnalysisNwbfile.get_abs_path(rec["analysis_file_name"])
    obj = get_nwb_file(path).objects[rec["behavior_model_results_object_id"]]
    return obj.to_dataframe().reset_index(drop=True)


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
    times = pd.DataFrame(
        (beh.TrialsInfoByEpoch.ByTrial & [{"nwb_file_name": n} for n in nwbs]).fetch()
    )
    # epoch arrives as an int epoch-number from DataJoint; if it were a string the
    # epoch_number join below would silently drop every stop.
    if not pd.api.types.is_integer_dtype(times["epoch"].dtype):
        raise TypeError(
            f"expected integer epoch from DataJoint, got {times['epoch'].dtype}"
        )
    times = _add_nominal_contingency(times)
    times = times[[*KEYS, *TIME_COLS, "leaf", "reward", *NOMINAL_COVARIATES]]

    pt = times.merge(
        vals[[*KEYS, "leaf", "reward", *model_cols]],
        on=KEYS,
        suffixes=("_time", "_val"),
        validate="one_to_one",
    )
    pt.attrs["leaf_agree"] = float((pt["leaf_time"] == pt["leaf_val"]).mean())
    pt.attrs["entropy_cov"] = pt_attrs_entropy
    # nominal block (from times) + model block (from vals) are both broadcast to stops
    return pt, [*model_cols, *NOMINAL_COVARIATES]


def _outcome_known_time(
    *,
    rewarded: bool,
    poke_in_time: float,
    poke_out_time: float,
    reward_on_time: float,
) -> float:
    """Time at which the current trip's outcome is observable.

    Reward onset is an explicit positive-feedback cue. An omission has none, so
    its boundary is ``poke_in + OMISSION_KNOWN_LATENCY_SEC`` -- the point by
    which a pump would have fired had the trial been rewarded, capped at
    poke-out because leaving is itself conclusive. The latency exceeds every
    reward latency in the corpus, so this cannot label an omission while a pump
    could still arrive.

    The previous rule used poke-out directly. That is a *behavioural* boundary
    against the rewarded case's *stimulus* one, which is both circular (dwell
    length plausibly covaries with the replay being measured) and empirically
    ruinous: it placed the boundary a median 880 ms after the information was
    available and excluded 99% of omission events from every gated fit.
    """
    if not np.isfinite(poke_in_time):
        raise ValueError("current trial lacks finite poke_in_ts")
    if not np.isfinite(poke_out_time):
        raise ValueError("current trial lacks finite poke_out_ts")
    if poke_out_time < poke_in_time:
        raise ValueError("current trial has poke_out_ts before poke_in_ts")
    if rewarded and not np.isfinite(reward_on_time):
        raise ValueError("rewarded trial lacks finite reward_on_ts")
    if rewarded and not (poke_in_time <= reward_on_time <= poke_out_time):
        raise ValueError(
            "rewarded trial reward_on_ts must lie within poke_in_ts/poke_out_ts"
        )
    if not rewarded and np.isfinite(reward_on_time):
        raise ValueError("omission trial unexpectedly carries finite reward_on_ts")
    if rewarded:
        return float(reward_on_time)
    # Capped at poke-out: if the animal left sooner than the latency, its
    # departure already settles the outcome and the boundary cannot exceed
    # the poke it belongs to.
    return float(min(poke_in_time + OMISSION_KNOWN_LATENCY_SEC, poke_out_time))


def build_animal(
    animal: str, beh: Any, rl: Any, stop_table: pd.DataFrame
) -> pd.DataFrame:
    pt, model_cols = _per_trial(animal, beh, rl)
    st = stop_table[stop_table.animal == animal].copy()
    st["nwb_file_name"] = animal + st["day"].astype(str) + "_.nwb"
    st["epoch_number"] = st["epoch"].str.split("_").str[0].astype("int64")
    context_time = st["stop_start"].to_numpy(dtype=float)
    mid = ((st["stop_start"] + st["stop_end"]) / 2).to_numpy()
    if not np.isfinite(context_time).all():
        raise ValueError("stop_start must be finite for qvals trial assignment")

    rows = []
    for (nwb, epnum), g in pt.groupby(["nwb_file_name", "epoch"]):
        g = g.sort_values("trial_number_by_epoch").reset_index(drop=True)
        po, pi = g["poke_out_ts"].to_numpy(), g["poke_in_ts"].to_numpy()
        ron, roff = g["reward_on_ts"].to_numpy(), g["reward_off_ts"].to_numpy()
        # searchsorted needs sorted poke_out; trial order is temporal, so this holds.
        if not (np.diff(po) >= 0).all():
            raise ValueError(f"poke_out_ts not sorted: {nwb} epoch {epnum}")
        s_idx = np.where(
            ((st["nwb_file_name"] == nwb) & (st["epoch_number"] == epnum)).to_numpy()
        )[0]
        if len(s_idx) == 0:
            continue
        # trial t = [poke_out[t-1], poke_out[t]); stops past the last poke_out
        # get -1 and drop out.
        # All stop-level predictors use the bout-start context. A bout may span
        # a trial boundary; midpoint assignment would silently attach a later
        # trial than the canonical stop/event covariates. Physical post-outcome
        # consumers may still use bins after the stored outcome boundary, but
        # they retain the trial that was active when the bout began.
        ti = assign_stops_to_trials(po, context_time[s_idx])
        valid = ti >= 0
        for k, se in zip(s_idx[valid], ti[valid], strict=True):
            trow = g.iloc[int(se)]
            rewarded = bool(trow["reward_time"])
            try:
                outcome_known_time = _outcome_known_time(
                    rewarded=rewarded,
                    poke_in_time=pi[se],
                    poke_out_time=po[se],
                    reward_on_time=ron[se],
                )
            except ValueError as error:
                raise ValueError(
                    f"{error} for {nwb} epoch {epnum} trial {se}"
                ) from error
            rows.append(
                {
                    "stop_id": st.iloc[k]["stop_id"],
                    "epoch": st.iloc[k]["epoch"],
                    "trial_number_by_epoch": int(trow["trial_number_by_epoch"]),
                    "qvals_context_time": float(context_time[k]),
                    # NOTE: the two flags below are read at the bout MIDPOINT,
                    # while the trial they are attached to is chosen by the
                    # bout START. They describe whether the animal happened to
                    # be poking / receiving reward mid-bout, not the instant
                    # the covariates were frozen; they are descriptive only and
                    # must not be used to gate a cohort.
                    "in_poke": bool(pi[se] <= mid[k] <= po[se]),
                    "is_rewarding": bool(
                        (not np.isnan(ron[se])) and ron[se] <= mid[k] <= roff[se]
                    ),
                    "trial_reward": int(
                        trow["reward_time"]
                    ),  # poke-record reward (times' 'reward', '_time' suffix)
                    # Physical availability boundary for post-outcome covariates:
                    # reward onset on rewarded trials; conservatively, poke-out on
                    # omissions because this table has no explicit negative-
                    # feedback timestamp.
                    "outcome_known_time": outcome_known_time,
                    **{c: trow[c] for c in model_cols},
                }
            )
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
        # Arm / reward agreement is only meaningful where the stop table has a
        # location to agree with. Restrict on the canonical whole-window
        # ``at_well`` label, NOT ``in_poke``: the latter is read at the bout
        # midpoint, so it marks bouts whose window is not wholly at one well,
        # and those carry ``current_arm = NaN`` by construction. Comparing
        # against that NaN scores every such stop as a disagreement and reads
        # as a mapping regression that is not there.
        at_well = (
            obs["at_well"].reindex(t["stop_id"]).eq(True).fillna(False).to_numpy()
        )
        cur = obs["current_arm"].reindex(t["stop_id"]).to_numpy(dtype=float)
        rout = obs["reward_outcome"].reindex(t["stop_id"]).to_numpy(dtype=float)
        arm_ok = np.mean(t["arm"].to_numpy(dtype=float)[at_well] == cur[at_well])
        # ``reward_outcome`` is frozen at the bout start, which precedes the
        # trip's own outcome, so it is missing on essentially every at-well
        # stop. Score agreement only where the reference exists and report the
        # comparable share, rather than counting "not yet known" as "wrong".
        rew_known = at_well & np.isfinite(rout)
        rew_ok = (
            np.mean(t["trial_reward"].to_numpy(dtype=float)[rew_known] == rout[rew_known])
            if rew_known.any()
            else float("nan")
        )
        rew_share = float(rew_known.sum()) / max(int(at_well.sum()), 1)
        entropy_cov = t.attrs["entropy_cov"]
        print(
            f"{a:7s} mapped {len(t):6d}/{len(st):6d} stops ({100 * len(t) / len(st):.0f}%)  "
            f"in_poke={100 * t['in_poke'].mean():.0f}%  "
            f"at-well arm=={100 * arm_ok:.1f}%  "
            f"at-well reward=={100 * rew_ok:.1f}% (on {100 * rew_share:.0f}% w/ known outcome)  "
            f"[leaf-join {100 * t.attrs['leaf_agree']:.1f}%  entropy {100 * entropy_cov:.0f}%]"
        )
        # beta_reward_entropy is computed in-process from betadist_α/β (same MODEL_KEY
        # pull as q_port), so coverage should be ~100% — anything materially below flags
        # a partial-fit/join regression (the old 0.5 HMM-merge floor is too loose now).
        if len(t) / len(st) < 0.90 or arm_ok < 0.95 or entropy_cov < 0.95:
            print(
                f"  WARNING [{a}]: coverage {100 * len(t) / len(st):.0f}% (expect ~97-99%), "
                f"at-well arm agreement {100 * arm_ok:.1f}% (expect ~100%), or "
                f"beta_reward_entropy coverage {100 * entropy_cov:.0f}% (expect ~100%; a low "
                f"figure voids the belief-entropy covariate) -- possible mapping/model regression"
            )
        parts.append(t)
    tab = pd.concat(parts, ignore_index=True)
    print(
        f"\ntotal mapped stops: {len(tab)}  unique: {tab.stop_id.nunique()}  cols: {tab.shape[1]}"
    )

    # Fail loudly if the per-arm value block is incomplete: model_cols filters
    # COVARIATES by presence, so an upstream BehaviorModelResults/schema change
    # dropping a Q{i}/p_rew_leaf{i} would silently omit a per-port column and
    # make attach_contingency_value_block blank the whole value block downstream.
    required = {
        "stop_id",
        "p_reward_current",
        "q_chosen",
        *[f"p_reward_port{i}" for i in range(6)],
        *[f"q_port{i}" for i in range(6)],
        "beta_reward_entropy",
    }
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
