"""Per-stop RL-model values via DataJoint — canonical, complete, timestamp-aligned.

Builds the per-stop behavioral-model covariate table, sourcing everything from
Alison's shared DataJoint schemas (read-only). :func:`run_build_qvals_table` is
the whole build; the command-line front end that resolves its arguments is
``scripts/behavior/build_qvals_table.py``, which is also where the sources, the
join rule, and the validation contract are documented.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import datajoint as dj
import numpy as np
import pandas as pd
from scipy.special import betaln, digamma

from continuum_swr_replay.data_loaders.constants import port_columns
from continuum_swr_replay.data_loaders.epoch_register import analysed_animals
from continuum_swr_replay.data_loaders.trial_clocks import (
    outcome_known_time_from_task_visit,
)
from continuum_swr_replay.paths import (
    QVALS_STANDARDIZATION_REFERENCE,
    REPLAY_CONTENT_DIR,
    table_path,
)
from continuum_swr_replay.tables.covariate_registry import (
    REGISTRY,
    require_covariate_schema,
    stamp_covariate_schema,
)
from continuum_swr_replay.tables.io import publish_canonical, read_table
from continuum_swr_replay.tables.standardization import (
    Reference,
    assert_standardized_present,
    attach_standardized,
    save_reference,
)

from ._spatial_bandit_rl import make_full_beta_bernoulli_dv_df

# the vendored DV pipeline inserts many columns one-by-one (Alison's style)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

RC = REPLAY_CONTENT_DIR
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
    "periswitch_index_trials",
    "trials_since_block_change",
    # beta-bernoulli value / decision / uncertainty block:
    "rpe_postoutcome",
    "relative_value_switch_minus_stay_preoutcome",
    "reward_belief_mean_update_l1_all_ports_postoutcome",
    "reward_belief_differential_entropy_leaf_mean_preoutcome",
]
KEYS = ["nwb_file_name", "epoch", "trial_number_by_epoch"]
TIME_COLS = ["poke_in_ts", "poke_out_ts", "reward_on_ts", "reward_off_ts"]
# Beta value timing: row t holds PRE-outcome belief for trial t (vendored DV docs).

# Curated per-stop covariate set (paper-aligned): the Comrie et al. theta-sequences decision
# variables (computed by the vendored beta-bernoulli pipeline), the derived beta
# belief entropy, RPE, and the raw value/uncertainty/choice columns.
COVARIATES = [
    "chosen_arm",
    "reward_belief_mean_chosen_arm_preoutcome",
    "rpe_postoutcome",
    "rpe_postoutcome_abs",
    # expected value (per port / patch / belief mean)
    *[f"decision_value_patch_{i}" for i in range(1, 4)],
    *port_columns("reward_belief_mean"),
    *port_columns("reward_belief_variance"),
    # decision variables: Stay / Switch / Relative (DeltaV)
    "stay_value_preoutcome",
    "switch_value_preoutcome",
    "relative_value_switch_minus_stay_preoutcome",
    "relative_decision_value_switch_minus_stay_preoutcome",
    # absolute value-update magnitudes, 4 scopes (paper) + global signed
    "reward_belief_mean_update_l1_all_ports_postoutcome",
    "reward_belief_mean_update_chosen_arm_postoutcome_abs",
    "reward_belief_mean_update_l1_chosen_patch_postoutcome",
    "reward_belief_mean_update_l1_alternative_patches_postoutcome",
    "reward_belief_mean_update_sum_all_ports_postoutcome",
    # additional uncertainty summaries
    "reward_belief_variance_leaf_sum_preoutcome",
    "reward_belief_variance_chosen_arm_preoutcome",
    "prev_local_variance",
    "upcoming_leaf_variance",
    "next_choice_variance",
    # beta belief-entropy (mean per-leaf Beta differential entropy)
    "reward_belief_differential_entropy_leaf_mean_preoutcome",
    # choice probabilities
    "stem_choice_p",
    "leaf_choice_logp",
    "stem_stay_p",
    *[f"stem_{i}_p" for i in range(1, 4)],
]

# The four behavioral-model quantities that enter the paper's fitted model
# ladders.  Keep this contract separate from ``COVARIATES``: that larger list
# also contains diagnostic/export-only quantities which may legitimately be
# absent from a particular upstream model version.  Fitted terms must never
# disappear merely because the adapter filtered columns by presence.
REQUIRED_FITTED_MODEL_COVARIATES = (
    "relative_value_switch_minus_stay_preoutcome",
    "rpe_postoutcome",
    "reward_belief_differential_entropy_leaf_mean_preoutcome",
    "reward_belief_mean_update_l1_all_ports_postoutcome",
)

# Model fields needed to verify identities and populate the downstream
# per-candidate-arm value/uncertainty block, whether or not they are themselves
# direct fitted terms.
_REQUIRED_MODEL_CONTRACT_COVARIATES = (
    "chosen_arm",
    "reward_belief_mean_chosen_arm_preoutcome",
    "rpe_postoutcome_abs",
    *port_columns("reward_belief_mean"),
    *port_columns("reward_belief_variance"),
)

# The vendored decision-variable code uses adjacent model rows for these
# quantities. If an internal model trial is absent, pandas ``shift``/``diff``
# would otherwise bridge the gap and produce a plausible value for the wrong
# transition. The authoritative timing vector below identifies and masks only
# those affected neighbors; direct current-trial quantities (beliefs, entropy,
# RPE) remain available.
_PREVIOUS_TRIAL_DEPENDENT_MODEL_COVARIATES = (
    "stay_value_preoutcome",
    "switch_value_preoutcome",
    "relative_value_switch_minus_stay_preoutcome",
    "relative_decision_value_switch_minus_stay_preoutcome",
    "prev_local_variance",
)
_NEXT_TRIAL_DEPENDENT_MODEL_COVARIATES = (
    "reward_belief_mean_update_l1_all_ports_postoutcome",
    "reward_belief_mean_update_chosen_arm_postoutcome_abs",
    "reward_belief_mean_update_l1_chosen_patch_postoutcome",
    "reward_belief_mean_update_l1_alternative_patches_postoutcome",
    "reward_belief_mean_update_sum_all_ports_postoutcome",
    "upcoming_leaf_variance",
    "next_choice_variance",
)

# Nominal schedule covariates derived from TrialsInfoByEpoch (the same table read
# for poke times, so aligned by the same poke_out rule -- no difflib/CSV). Source
# is Alison's `p_rew_leaf1..6` (nominal p(R), percent) + `contingency_count`;
# the patch-mean probability gap / trials_since_block_change use this project's definitions
# (Alison computes no nominal value-diff -- her decision variables are model-based).
NOMINAL_COVARIATES = [
    "task_reward_probability_chosen_arm",
    "task_reward_probability_best_arm",
    "task_reward_probability_patch_mean_gap_best_vs_second",
    "trials_since_block_change",
    # peri-switch trial index (signed distance to the nearest patch switch) +
    # post-switch indicator; the paper's "trials since switch" axis (Comrie et al Fig 2).
    "periswitch_index_trials",
    "is_periswitch_post",
    # per-port nominal p(R); the content/contfrag builders map task_reward_probability_port_{arm}
    # -> p_reward_arm (attach_contingency_value_block).
    *port_columns("task_reward_probability"),
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

    The chosen-arm and best-arm task probabilities come directly from Alison's
    nominal ``p_rew_leaf`` fields. The patch-mean gap is best minus second-best
    patch mean (two leaves per patch); ``trials_since_block_change`` is position
    within the current contingency block. The peri-switch fields are the signed
    index and post-switch indicator from :func:`peri_switch_index`. All are
    computed on the exactly aligned per-trial rows.
    """
    times = times.sort_values(
        ["nwb_file_name", "epoch", "trial_number_by_epoch"]
    ).reset_index(drop=True)
    # guard the fancy-index below (ports[arange, leaf-1]); leaf must be 1..6.
    if not times["leaf"].between(1, 6).all():
        raise ValueError(f"times leaf outside 1..6: {sorted(times['leaf'].unique())}")
    ports = times[[f"p_rew_leaf{i}" for i in range(1, 7)]].to_numpy(dtype=float) / 100.0
    arm = times["leaf"].to_numpy() - 1
    times["task_reward_probability_chosen_arm"] = ports[np.arange(len(times)), arm]
    times["task_reward_probability_best_arm"] = ports.max(axis=1)
    for i in range(6):
        times[f"task_reward_probability_port_{i}"] = ports[
            :, i
        ]  # per-port nominal p(R)
    patch_val = np.stack(
        [ports[:, WELL_PATCH == p].mean(axis=1) for p in range(3)], axis=1
    )
    srt = np.sort(patch_val, axis=1)
    times["task_reward_probability_patch_mean_gap_best_vs_second"] = (
        srt[:, -1] - srt[:, -2]
    )
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
    times["periswitch_index_trials"] = tss
    times["is_periswitch_post"] = isp
    return times


def add_experienced_value_columns(vals: pd.DataFrame) -> pd.DataFrame:
    """Derive native adapter columns from a per-trial Beta-model frame.

    Before the canonical ingestion boundary, adds native ``q_chosen`` (belief
    mean of the chosen arm = ``Q{leaf}``), ``arm`` (``leaf - 1``, 0-indexed),
    ``rpe`` (``reward - q_chosen``), and ``rpe_abs``. These become the explicit
    ``reward_belief_*`` / ``rpe_postoutcome*`` columns in
    :func:`canonicalize_behavior_model_columns`.

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


# The qvals rows of the covariate registry, as the plain lists the ingestion
# function applies. A numerical-path family is one quantity computed twice by
# the behavioral model: its first column survives and the others are checked
# against it, then dropped.
_QVALS_ENTRIES = tuple(e for e in REGISTRY if e.table == "qvals")
_NUMERICAL_PATH_FAMILIES = tuple(
    e.current for e in _QVALS_ENTRIES if e.action == "canonicalize"
)
_DEAD_MODEL_COLUMNS = tuple(
    c for e in _QVALS_ENTRIES if e.action == "drop" for c in e.current
)
_NATIVE_TO_CANONICAL = {
    e.current[0]: e.canonical for e in _QVALS_ENTRIES if e.canonical is not None
}
# Two numerical paths to one quantity agree to floating point, not bit for bit.
_NUMERICAL_PATH_ATOL = 1e-15
_NUMERICAL_PATH_RTOL = 1e-12


def _check_numerical_paths_agree(vals: pd.DataFrame, survivor: str, twin: str) -> None:
    """Raise unless two model columns hold the same quantity to floating point.

    They must share their missing rows, hold no infinity, have at least one
    finite row to compare (an all-missing pair is unverifiable, not equal), and
    agree within ``atol=1e-15, rtol=1e-12`` on the finite rows.
    """
    a = vals[survivor].to_numpy(dtype=float)
    b = vals[twin].to_numpy(dtype=float)
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        raise ValueError(
            f"numerical-path twins differ in missingness: {survivor!r} vs {twin!r}"
        )
    if np.isinf(a).any() or np.isinf(b).any():
        raise ValueError(
            f"numerical-path twins contain infinity: {survivor!r} vs {twin!r}"
        )
    finite = np.isfinite(a)
    if len(vals) and not finite.any():
        raise ValueError(
            f"numerical-path agreement unverifiable: {survivor!r} vs {twin!r} "
            "are all-NaN in this frame"
        )
    if not np.allclose(
        a[finite],
        b[finite],
        atol=_NUMERICAL_PATH_ATOL,
        rtol=_NUMERICAL_PATH_RTOL,
    ):
        worst = float(np.max(np.abs(a[finite] - b[finite])))
        raise ValueError(
            f"numerical-path twins exceed tolerance: {survivor!r} vs {twin!r} "
            f"(max abs diff {worst:.2e})"
        )


def canonicalize_behavior_model_columns(vals: pd.DataFrame) -> pd.DataFrame:
    """Translate native Alison-model outputs into the canonical qvals schema.

    The vendored computation deliberately retains its upstream column names.
    This is the single ingestion boundary where those native names become
    repository-owned names, using the ``qvals`` rows of the covariate registry.

    The model exposes each port's belief mean twice: as its fitted value
    ``Q{i+1}`` and as the Beta posterior mean ``betadist_mu{i+1}`` (computed
    from ``betadist_α``/``betadist_β`` by the vendored pipeline). It likewise
    exposes the leaf-summed belief variance twice (``prev_global_variance`` and
    ``global_leaf_var_sum_preoutcome_sess``). Each such pair is checked to agree
    to floating point (see :func:`_check_numerical_paths_agree`) and only its
    first column is kept, so a divergence between the value fit and the Beta
    posterior is caught here. The constant depletion outputs of the withleaf fit
    are dropped, and every surviving native column is renamed to its canonical
    name (``Q{i+1}`` -> ``reward_belief_mean_port_{i}``, ``Qstem*`` ->
    ``decision_value_patch_*``, ``rpe`` -> ``rpe_postoutcome`` ...).

    Raises ``ValueError`` if a family is only partly present, a pair disagrees,
    or two columns would get one name.
    """
    present = set(vals.columns)
    drop = [c for c in _DEAD_MODEL_COLUMNS if c in present]
    for family in _NUMERICAL_PATH_FAMILIES:
        here = [c for c in family if c in present]
        if not here:
            continue
        if len(here) != len(family):
            raise ValueError(
                f"numerical-path family {family} is incomplete; missing "
                f"{sorted(set(family) - present)}"
            )
        survivor, *twins = family
        for twin in twins:
            _check_numerical_paths_agree(vals, survivor, twin)
        drop.extend(twins)
    out = vals.drop(columns=drop).rename(columns=_NATIVE_TO_CANONICAL)
    if out.columns.duplicated().any():
        dupes = sorted(set(out.columns[out.columns.duplicated()]))
        raise ValueError(f"qvals ingestion collides columns onto: {dupes}")
    return out


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


def _model_covariate_columns(vals: pd.DataFrame) -> list[str]:
    """Return curated model columns after enforcing the scientific contract.

    ``COVARIATES`` intentionally includes optional diagnostic exports.  In
    contrast, paper-fitted predictors and the columns required to verify or
    attach them are mandatory and produce a targeted error at ingestion.
    """
    missing_fitted = sorted(set(REQUIRED_FITTED_MODEL_COVARIATES) - set(vals.columns))
    if missing_fitted:
        raise ValueError(
            "behavior model is missing required fitted model covariates "
            f"{missing_fitted}"
        )
    missing_contract = sorted(
        set(_REQUIRED_MODEL_CONTRACT_COVARIATES) - set(vals.columns)
    )
    if missing_contract:
        raise ValueError(
            "behavior model is missing required qvals contract covariates "
            f"{missing_contract}"
        )
    return [column for column in COVARIATES if column in vals.columns]


def _validate_trial_keys(
    frame: pd.DataFrame, *, source: str, require_consecutive: bool
) -> None:
    """Validate one-row-per-trial keys before any timing/model alignment."""
    missing_columns = set(KEYS) - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"{source} trial table is missing key columns {sorted(missing_columns)}"
        )
    if frame[KEYS].isna().any().any():
        raise ValueError(f"{source} trial keys contain missing values")
    duplicate = frame.duplicated(KEYS, keep=False)
    if duplicate.any():
        sample = frame.loc[duplicate, KEYS].head(5).to_dict("records")
        raise ValueError(
            f"{source} trial keys are duplicated ({int(duplicate.sum())} rows); "
            f"sample={sample}"
        )

    trial_number = pd.to_numeric(frame["trial_number_by_epoch"], errors="coerce")
    valid_integer = (
        np.isfinite(trial_number.to_numpy(dtype=float))
        & (trial_number.to_numpy(dtype=float) >= 0)
        & (
            trial_number.to_numpy(dtype=float)
            == np.floor(trial_number.to_numpy(dtype=float))
        )
    )
    if not valid_integer.all():
        bad = frame.loc[~valid_integer, KEYS].head(5).to_dict("records")
        raise ValueError(
            f"{source} trial numbers must be finite nonnegative integers; sample={bad}"
        )

    if not require_consecutive:
        return
    for session_key, group in frame.groupby(
        ["nwb_file_name", "epoch"], sort=False, dropna=False
    ):
        observed = np.sort(group["trial_number_by_epoch"].to_numpy(dtype=int))
        expected = np.arange(len(observed), dtype=int)
        if not np.array_equal(observed, expected):
            raise ValueError(
                "timing trial numbers must be consecutive from zero within each "
                f"session/epoch; {session_key} has {observed.tolist()}"
            )


def _raise_identity_error(
    frame: pd.DataFrame, mask: np.ndarray, *, message: str
) -> None:
    """Raise a key-localized error for an exact trial-level identity failure."""
    if not mask.any():
        return
    sample = frame.loc[mask, KEYS].head(5).to_dict("records")
    raise ValueError(
        f"{message} on {int(mask.sum())} matched trial(s); sample={sample}"
    )


def _mask_cross_trial_covariates_at_model_gaps(
    joined: pd.DataFrame,
) -> tuple[int, int]:
    """Blank neighbor-derived values where the model row sequence has a gap.

    The vendored calculations correctly use adjacent rows when the model table
    is complete, but cannot know about a timing-only trial that was absent from
    their input. A missing previous row invalidates trial-start/relative-value
    quantities on the following row; a missing next row invalidates
    post-outcome updates on the preceding row.
    """
    group_keys = [joined["nwb_file_name"], joined["epoch"]]
    available = joined["_model_row_available"].astype(bool)
    position = joined.groupby(["nwb_file_name", "epoch"], sort=False).cumcount()
    reverse_position = (
        joined.iloc[::-1]
        .groupby(["nwb_file_name", "epoch"], sort=False)
        .cumcount()
        .iloc[::-1]
    )
    previous_available = available.groupby(group_keys, sort=False).shift(1)
    next_available = available.groupby(group_keys, sort=False).shift(-1)
    invalid_previous = (
        available & position.gt(0) & previous_available.eq(False)
    ).to_numpy(dtype=bool)
    invalid_next = (
        available & reverse_position.gt(0) & next_available.eq(False)
    ).to_numpy(dtype=bool)

    previous_columns = [
        column
        for column in _PREVIOUS_TRIAL_DEPENDENT_MODEL_COVARIATES
        if column in joined.columns
    ]
    next_columns = [
        column
        for column in _NEXT_TRIAL_DEPENDENT_MODEL_COVARIATES
        if column in joined.columns
    ]
    if previous_columns:
        joined.loc[invalid_previous, previous_columns] = np.nan
    if next_columns:
        joined.loc[invalid_next, next_columns] = np.nan
    return int(invalid_previous.sum()), int(invalid_next.sum())


def _join_model_to_trial_timebase(
    times: pd.DataFrame, vals: pd.DataFrame, model_cols: list[str]
) -> pd.DataFrame:
    """Attach model rows to the complete, authoritative trial timing table.

    The outer audit makes asymmetric coverage explicit.  A model-only key is a
    provenance/alignment error and is fatal.  A timing-only key is retained as
    its true interval with unavailable model fields, so it cannot shift every
    later stop onto the wrong trial.  Stop broadcasting omits only stops in
    those explicitly unavailable intervals.
    """
    _validate_trial_keys(times, source="timing", require_consecutive=True)
    _validate_trial_keys(vals, source="model", require_consecutive=False)

    required_model_columns = {"leaf", "reward", *model_cols}
    missing_model_columns = required_model_columns - set(vals.columns)
    if missing_model_columns:
        raise ValueError(
            f"model trial table is missing join columns {sorted(missing_model_columns)}"
        )
    missing_timing_columns = {"leaf", "reward", *TIME_COLS} - set(times.columns)
    if missing_timing_columns:
        raise ValueError(
            "timing trial table is missing authoritative columns "
            f"{sorted(missing_timing_columns)}"
        )

    timing = times.copy()
    timing["_timing_row_order"] = np.arange(len(timing), dtype=np.int64)
    joined = timing.merge(
        vals[[*KEYS, "leaf", "reward", *model_cols]],
        on=KEYS,
        how="outer",
        suffixes=("_time", "_val"),
        indicator="_trial_source",
        validate="one_to_one",
        sort=False,
    )
    model_only = joined["_trial_source"].eq("right_only")
    if model_only.any():
        sample = joined.loc[model_only, KEYS].head(5).to_dict("records")
        raise ValueError(
            f"model-only trial keys are not present in the authoritative timing "
            f"table ({int(model_only.sum())} row(s)); sample={sample}"
        )

    joined = (
        joined.loc[~model_only]
        .sort_values("_timing_row_order", kind="stable")
        .reset_index(drop=True)
    )
    joined["_model_row_available"] = joined["_trial_source"].eq("both")
    matched = joined["_model_row_available"].to_numpy(dtype=bool)

    leaf_bad = matched & ~(
        joined["leaf_time"].to_numpy() == joined["leaf_val"].to_numpy()
    )
    _raise_identity_error(joined, leaf_bad, message="timing/model leaf labels disagree")
    reward_bad = matched & ~(
        joined["reward_time"].to_numpy() == joined["reward_val"].to_numpy()
    )
    _raise_identity_error(
        joined, reward_bad, message="timing/model reward labels disagree"
    )

    expected_rpe = joined["reward_time"].to_numpy(dtype=float) - joined[
        "reward_belief_mean_chosen_arm_preoutcome"
    ].to_numpy(dtype=float)
    actual_rpe = joined["rpe_postoutcome"].to_numpy(dtype=float)
    rpe_bad = matched & ~(actual_rpe == expected_rpe)
    _raise_identity_error(
        joined,
        rpe_bad,
        message=(
            "rpe_postoutcome != reward - reward_belief_mean_chosen_arm_preoutcome"
        ),
    )
    actual_rpe_abs = joined["rpe_postoutcome_abs"].to_numpy(dtype=float)
    rpe_abs_bad = matched & ~(actual_rpe_abs == np.abs(actual_rpe))
    _raise_identity_error(
        joined,
        rpe_abs_bad,
        message="rpe_postoutcome_abs != abs(rpe_postoutcome)",
    )

    n_after_gap, n_before_gap = _mask_cross_trial_covariates_at_model_gaps(joined)

    n_timing_only = int((~matched).sum())
    gap_counts = (
        joined.loc[~matched]
        .groupby(["nwb_file_name", "epoch"], sort=False, dropna=False)
        .size()
        .rename("n_timing_trials_without_model")
        .reset_index()
        .to_dict("records")
    )
    joined = joined.drop(columns=["_trial_source", "_timing_row_order"])
    joined.attrs["leaf_agree"] = 1.0
    joined.attrs["reward_agree"] = 1.0
    joined.attrs["n_timing_trials"] = len(times)
    joined.attrs["n_model_trials"] = len(vals)
    joined.attrs["n_timing_trials_without_model"] = n_timing_only
    joined.attrs["n_model_trials_after_timing_gap"] = n_after_gap
    joined.attrs["n_model_trials_before_timing_gap"] = n_before_gap
    joined.attrs["timing_trials_without_model_by_session"] = gap_counts
    return joined


def _per_trial(animal: str, beh: Any, rl: Any) -> tuple[pd.DataFrame, list[str]]:
    """Join enriched model covariates to poke timestamps, one row per trial.

    Beta values are enriched with the vendored decision-variable pipeline
    (DeltaV, value updates, belief entropy, etc.) and RPE/q_chosen/arm are
    derived; the curated COVARIATES are broadcast to stops.
    """
    subject = ANIMAL_SUBJECT[animal]
    vals = _fetch_model_df(rl, subject, MODEL_KEY)
    _validate_trial_keys(vals, source="model", require_consecutive=False)
    vals = make_full_beta_bernoulli_dv_df(vals)
    vals = add_experienced_value_columns(vals)  # q_chosen / arm / rpe
    vals = add_beta_belief_entropy(vals)  # beta_reward_entropy (from betadist_α/β)
    vals = canonicalize_behavior_model_columns(vals)
    model_cols = _model_covariate_columns(vals)
    pt_attrs_entropy = float(
        vals["reward_belief_differential_entropy_leaf_mean_preoutcome"].notna().mean()
    )

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
    _validate_trial_keys(times, source="timing", require_consecutive=True)
    times = _add_nominal_contingency(times)
    times = times[[*KEYS, *TIME_COLS, "leaf", "reward", *NOMINAL_COVARIATES]]

    pt = _join_model_to_trial_timebase(times, vals, model_cols)
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
    which a pump would have fired had the trial been rewarded, capped at the
    task visit's final poke-out because leaving is itself conclusive. The
    timing input follows Comrie's collapsed visit rows, so a same-port re-poke
    remains part of this one visit. The latency exceeds every
    reward latency in the corpus, so this cannot label an omission while a pump
    could still arrive.

    The previous rule used poke-out directly. That is a *behavioural* boundary
    against the rewarded case's *stimulus* one, which is both circular (dwell
    length plausibly covaries with the replay being measured) and empirically
    ruinous: it placed the boundary a median 880 ms after the information was
    available and excluded 99% of omission events from every gated fit.
    """
    return outcome_known_time_from_task_visit(
        rewarded=rewarded,
        first_poke_in_time=poke_in_time,
        final_poke_out_time=poke_out_time,
        reward_on_time=reward_on_time,
    )


def _broadcast_trial_covariates_to_stops(
    animal: str,
    pt: pd.DataFrame,
    broadcast_cols: list[str],
    stop_table: pd.DataFrame,
) -> pd.DataFrame:
    """Broadcast authoritative trial rows to stop-start intervals.

    Stops assigned to timing-only trials are intentionally omitted: the trial
    interval is known but its fitted model state is not.  Crucially, retaining
    that interval in ``pt`` means all later stops keep their true trial index.
    """
    st = stop_table[stop_table.animal == animal].copy()
    st["nwb_file_name"] = animal + st["session_date"].astype(str) + "_.nwb"
    st["epoch_number"] = st["epoch"].str.split("_").str[0].astype("int64")
    context_time = st["stop_start"].to_numpy(dtype=float)
    mid = ((st["stop_start"] + st["stop_end"]) / 2).to_numpy()
    if not np.isfinite(context_time).all():
        raise ValueError("stop_start must be finite for qvals trial assignment")

    rows = []
    n_stops_timing_mapped = 0
    n_stops_missing_model_trial = 0
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
        timing_mapped = ti >= 0
        n_stops_timing_mapped += int(timing_mapped.sum())
        model_available = np.zeros(len(ti), dtype=bool)
        model_available[timing_mapped] = g["_model_row_available"].to_numpy(dtype=bool)[
            ti[timing_mapped]
        ]
        missing_model = timing_mapped & ~model_available
        n_stops_missing_model_trial += int(missing_model.sum())
        valid = timing_mapped & model_available
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
                    f"{error} for {nwb} epoch {epnum} trial "
                    f"{int(trow['trial_number_by_epoch'])}"
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
                    "pump_on_at_bout_mid": bool(
                        (not np.isnan(ron[se])) and ron[se] <= mid[k] <= roff[se]
                    ),
                    "trial_reward": int(
                        trow["reward_time"]
                    ),  # poke-record reward (times' 'reward', '_time' suffix)
                    # Physical availability boundary for post-outcome covariates:
                    # reward onset on rewarded trials; on omissions, the latest
                    # possible pump latency after arrival, capped at poke-out.
                    "outcome_known_time": outcome_known_time,
                    **{c: trow[c] for c in broadcast_cols},
                }
            )
    out = pd.DataFrame(rows)
    out.attrs.update(pt.attrs)
    out.attrs["n_stops_timing_mapped"] = n_stops_timing_mapped
    out.attrs["n_stops_missing_model_trial"] = n_stops_missing_model_trial
    return out


def build_animal(
    animal: str, beh: Any, rl: Any, stop_table: pd.DataFrame
) -> pd.DataFrame:
    pt, broadcast_cols = _per_trial(animal, beh, rl)
    return _broadcast_trial_covariates_to_stops(animal, pt, broadcast_cols, stop_table)


def _validate_qvals_output(tab: pd.DataFrame) -> None:
    """Validate the persisted stop-value contract before standardization/write."""
    required = {
        "stop_id",
        "task_reward_probability_chosen_arm",
        "chosen_arm",
        "reward_belief_mean_chosen_arm_preoutcome",
        "rpe_postoutcome_abs",
        *REQUIRED_FITTED_MODEL_COVARIATES,
        *port_columns("task_reward_probability"),
        *port_columns("reward_belief_mean"),
        *port_columns("reward_belief_variance"),
        "reward_belief_differential_entropy_leaf_mean_preoutcome",
        "outcome_known_time",
        "qvals_context_time",
        "trial_reward",
    }
    missing = required - set(tab.columns)
    if missing:
        raise ValueError(f"qvals_table is missing required columns {sorted(missing)}")
    if tab["stop_id"].isna().any() or tab["stop_id"].duplicated().any():
        duplicates = int(tab["stop_id"].duplicated(keep=False).sum())
        raise ValueError(
            "qvals_table requires one nonmissing row per stop_id; "
            f"found {duplicates} row(s) with duplicated IDs"
        )

    probability_cols = [
        "task_reward_probability_chosen_arm",
        "reward_belief_mean_chosen_arm_preoutcome",
        *port_columns("task_reward_probability"),
        *port_columns("reward_belief_mean"),
    ]
    probability_values = tab[probability_cols].to_numpy(dtype=float)
    if not np.isfinite(probability_values).all():
        raise ValueError("qvals probability/value columns must be finite on every row")
    if ((probability_values < 0.0) | (probability_values > 1.0)).any():
        raise ValueError("qvals probability/value columns must lie in [0, 1]")
    entropy = tab["reward_belief_differential_entropy_leaf_mean_preoutcome"].to_numpy(
        dtype=float
    )
    if not np.isfinite(entropy).all() or (entropy > 1e-12).any():
        # Differential entropy is legitimately negative for concentrated Beta
        # distributions and has a maximum of zero for the uniform Beta(1, 1).
        raise ValueError("beta_reward_entropy must be finite and no greater than zero")
    variance_cols = port_columns("reward_belief_variance")
    variances = tab[variance_cols].to_numpy(dtype=float)
    if (
        not np.isfinite(variances).all()
        or ((variances < 0.0) | (variances > 0.25)).any()
    ):
        raise ValueError("Beta posterior variances must be finite and lie in [0, 0.25]")
    timing = tab[["qvals_context_time", "outcome_known_time"]].to_numpy(dtype=float)
    if not np.isfinite(timing).all():
        raise ValueError("qvals context and outcome-known times must be finite")
    reward = tab["trial_reward"].to_numpy(dtype=float)
    if not np.isfinite(reward).all() or not np.isin(reward, (0.0, 1.0)).all():
        raise ValueError("trial_reward must be finite binary")
    chosen_belief = tab["reward_belief_mean_chosen_arm_preoutcome"].to_numpy(
        dtype=float
    )
    rpe = tab["rpe_postoutcome"].to_numpy(dtype=float)
    expected_rpe = reward - chosen_belief
    if not np.array_equal(rpe, expected_rpe):
        raise ValueError(
            "rpe_postoutcome != trial_reward - reward_belief_mean_chosen_arm_preoutcome"
        )
    rpe_abs = tab["rpe_postoutcome_abs"].to_numpy(dtype=float)
    if not np.array_equal(rpe_abs, np.abs(rpe)):
        raise ValueError("rpe_postoutcome_abs != abs(rpe_postoutcome)")


def write_qvals_outputs(
    tab: pd.DataFrame, reference: Reference, replay_content_dir: Path
) -> Path:
    """Publish ``qvals_table`` and its standardization reference.

    Both go where :func:`~continuum_swr_replay.paths.table_path` puts the table:
    the replay-content root, beside the other canonical tables. Returns the
    table's path.
    """
    out = publish_canonical(
        tab, replay_content_dir, table="qvals", basename="qvals_table"
    )
    save_reference(reference, replay_content_dir / QVALS_STANDARDIZATION_REFERENCE)
    return out


def run_build_qvals_table(args: argparse.Namespace) -> None:
    """Build every requested animal's qvals rows, validate, and optionally publish."""
    replay_content_dir = args.replay_content_dir.resolve()
    animals = [args.animal] if args.animal else list(analysed_animals())

    beh = dj.create_virtual_module("alison_behav", "alison_behav")
    rl = dj.create_virtual_module("alison_rlmodel", "alison_rlmodel")
    st_all = read_table(table_path("stop_table", replay_content_dir, fmt="pkl"))
    require_covariate_schema(st_all, table="stop")

    parts = []
    for a in animals:
        t = build_animal(a, beh, rl, st_all)
        st = st_all[st_all.animal == a]
        obs = st.set_index("stop_id")
        # Arm / reward agreement is only meaningful where the stop table has a
        # location to agree with. Restrict on the canonical whole-window
        # ``at_well`` label, NOT ``in_poke``: the latter is read at the bout
        # midpoint, so it marks bouts whose window is not wholly at one well,
        # and those carry ``occupancy_well_arm = NaN`` by construction. Comparing
        # against that NaN scores every such stop as a disagreement and reads
        # as a mapping regression that is not there.
        at_well = obs["at_well"].reindex(t["stop_id"]).eq(True).fillna(False).to_numpy()
        cur = obs["occupancy_well_arm"].reindex(t["stop_id"]).to_numpy(dtype=float)
        rout = obs["is_rewarded_outcome"].reindex(t["stop_id"]).to_numpy(dtype=float)
        arm_ok = np.mean(t["chosen_arm"].to_numpy(dtype=float)[at_well] == cur[at_well])
        # ``is_rewarded_outcome`` is frozen at the bout start, which is inside
        # the arrival -> outcome window for about a fifth of at-well stops
        # (observable at 80.06 % of at-well bout starts; .claude/docs/reviews/
        # 2026-09-06-stop-table-clock-columns-regen.md), so it is missing
        # there. Score agreement only where the reference exists and report the
        # comparable share, rather than counting "not yet known" as "wrong".
        rew_known = at_well & np.isfinite(rout)
        rew_ok = (
            np.mean(
                t["trial_reward"].to_numpy(dtype=float)[rew_known] == rout[rew_known]
            )
            if rew_known.any()
            else float("nan")
        )
        rew_share = float(rew_known.sum()) / max(int(at_well.sum()), 1)
        entropy_cov = t.attrs["entropy_cov"]
        n_trial_gaps = t.attrs["n_timing_trials_without_model"]
        n_gap_stops = t.attrs["n_stops_missing_model_trial"]
        n_before_gap = t.attrs["n_model_trials_before_timing_gap"]
        n_after_gap = t.attrs["n_model_trials_after_timing_gap"]
        print(
            f"{a:7s} mapped {len(t):6d}/{len(st):6d} stops ({100 * len(t) / len(st):.0f}%)  "
            f"in_poke={100 * t['in_poke'].mean():.0f}%  "
            f"at-well arm=={100 * arm_ok:.1f}%  "
            f"at-well reward=={100 * rew_ok:.1f}% (on {100 * rew_share:.0f}% w/ known outcome)  "
            f"[leaf/reward join exact  entropy {100 * entropy_cov:.0f}%  "
            f"model gaps={n_trial_gaps} trials/{n_gap_stops} stops; "
            f"masked neighbors={n_before_gap} before/{n_after_gap} after]"
        )
        if n_trial_gaps:
            print(
                "  timing-only model trials by session: "
                f"{t.attrs['timing_trials_without_model_by_session']}"
            )
        # beta_reward_entropy is computed in-process from betadist_α/β (same MODEL_KEY
        # pull as Q1..Q6), so coverage should be ~100% — anything materially below flags
        # a partial-fit/join regression (the old 0.5 HMM-merge floor is too loose now).
        if (
            len(t) / len(st) < 0.90
            or not np.isfinite(arm_ok)
            or arm_ok < 0.95
            or not np.isfinite(entropy_cov)
            or entropy_cov < 0.95
        ):
            raise ValueError(
                f"qvals mapping failed for {a}: coverage "
                f"{100 * len(t) / len(st):.1f}%, at-well arm agreement "
                f"{100 * arm_ok:.1f}%, beta entropy coverage "
                f"{100 * entropy_cov:.1f}%; refusing to publish a partial or "
                "misaligned scientific covariate table"
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
    _validate_qvals_output(tab)

    # Pooled-GLOBAL standardization of the continuous stop-level covariates (<col>_z),
    # owned by THIS builder (prepare_data reads qvals but must not re-standardize it).
    # Computed over the full concatenated table; reference persisted for reproducibility.
    tab, qv_reference = attach_standardized(tab, QVALS_STANDARDIZED_COLS)
    assert_standardized_present(tab, QVALS_STANDARDIZED_COLS)
    tab = stamp_covariate_schema(tab, table="qvals")
    z_cols = [f"{c}_z" for c in QVALS_STANDARDIZED_COLS if c in tab.columns]

    if args.write:
        out = write_qvals_outputs(tab, qv_reference, replay_content_dir)
        print(f"\nwrote {len(tab)} rows ({len(z_cols)} _z cols) -> {out}")
