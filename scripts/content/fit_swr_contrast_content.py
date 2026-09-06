#!/usr/bin/env python
"""Fit the beta-bernoulli replay-CONTENT ladder (arm + patch), is_swr version.

The content analog of ``scripts/rate/fit_swr_contrast_rate.py``. Where the rate ladder
models HOW OFTEN non-local events occur (Poisson count per time bin), this models
WHICH ARM / PATCH an event represents: a conditional Poisson over the 9 arms (or 3
patches) within each event (``strata(event_id)`` ≡ clogit). Design spec:

Structural rule: in a conditional-over-arms likelihood a covariate CONSTANT within
an event cancels, so all event-level scalars (reward, rpe, value, entropy, context,
SWR status) enter ONLY as interactions with arm features. Per-arm covariates
(``reward_fraction_candidate_arm_last10_visits``,
``reward_belief_mean_candidate_arm_preoutcome``,
``is_last_rewarded_noncurrent_arm``)
enter as main effects.

Ladders (arm; fit per session, per is_swr split {all, SWR, non-SWR}):
  C0..C2  spatial baseline (ARMFE -> is_current -> +autocorrelation controls)
  C3..C5  per-arm value/reward CONTENT (main effects)
  C6..C9  state modulation of current-bias (rpe/value/uncert/update : is_current)
  C10..C12 CONTEXT split (periswitch/block/switchrate : is_current)
  CS1..CS5 is_swr:feature contrasts (pooled, 2-way; the dissociation)
Event-scalar dissociations (C6-C12) come from comparing the split, not a pooled 3-way.

Every rung built on the base (``BASEc`` / ``BASEc_PATCH``) carries a within-stop-time
control AND an event-length control -- z(log ``n_nonlocal_bins``) crossed with the
current and previous arm/patch indicators. SWR-coupled events are the longer ones and
a longer trajectory visits more arms, so without it ``swr_x_is_current_arm`` partly
reads "longer events leave the current arm". ``CS1nt_current`` / ``CPS1nt_current``
are the deliberately uncontrolled sensitivity twins. Because rewarded and omitted
stops differ in their SWR share, ``--outcome-stratum {all,rewarded,omitted}`` refits
the ladder within one current-outcome stratum (an event whose outcome is not yet
known is in neither) into ``_rewarded`` / ``_omitted`` artifacts.

Aggregation is AUDITED. Per-session conditional-Poisson coefficients carry SE and
p-values. **Primary population evidence is at the animal level (n=5)**: the animal-
balanced median coef and per-animal sign consistency (``n_animals_pos`` / ``n_animals``).
The Wilcoxon signed-rank across sessions is reported as a **descriptive** consistency
statistic ONLY — sessions within an animal are not independent, so it is not a valid
population p-value — with a Benjamini-Hochberg FDR across the rung family per population.
``typical_session_se`` is the median within-session SE, not the aggregate's uncertainty.
Per-arm value uses the beta-bernoulli construct
(``reward_belief_mean_candidate_arm_preoutcome``). The response is the per-event
allocation fraction (``--response hard_count`` refits the timebin counts into
separate ``_hard_count`` artifacts). A provenance sidecar (git hash, params) is
written, plus a ``.provenance.json`` beside each fitted pickle recording the
tables it was fit on and a ``.z_reference.json`` recording the fit-cohort
standardization moments. DEFERRED follow-ups (in the provenance): the **hierarchical Poisson
with animal random effects** that would replace the descriptive session signed-rank with
rigorous population inference; a common complete-case cohort; and within-group-centered
VIF (`fitting.py` uses the raw design).

Usage:
    uv run python scripts/content/fit_swr_contrast_content.py               # full
    uv run python scripts/content/fit_swr_contrast_content.py --max-per-animal 2   # smoke
    uv run python scripts/content/fit_swr_contrast_content.py --estimator dirichlet_multinomial
    uv run python scripts/content/fit_swr_contrast_content.py --outcome-stratum rewarded
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from continuum_swr_replay.analysis.replay._helpers import (
    add_arm_fixed_effects,
)
from continuum_swr_replay.analysis.replay.cohort_audit import (
    QVALS_GAP_COLS,
    cohort_record,
    drop_profile,
    term_diagnostics,
    write_cohort_audit,
)
from continuum_swr_replay.analysis.replay.content_response import (
    ESTIMATOR_CHOICES,
    RESPONSE_CHOICES,
    default_response_for_estimator,
    renormalize_fraction_over_candidates,
    resolve_response_choice,
    resolve_response_column,
    validate_fraction_response,
)
from continuum_swr_replay.analysis.replay.contrast_aggregation import (
    aggregate_contrast,
    specs_from_ladder,
)
from continuum_swr_replay.analysis.replay.covariate_registry import (
    require_covariate_schema,
)
from continuum_swr_replay.analysis.replay.covariate_timing import (
    assert_predictor_timing,
)
from continuum_swr_replay.analysis.replay.covariates import add_uncertainty_covariates
from continuum_swr_replay.analysis.replay.dirichlet_multinomial import (
    DEFAULT_MIN_EVENTS_SESSION_PHI,
    fit_sessions_dirichlet_multinomial,
)
from continuum_swr_replay.analysis.replay.dirichlet_multinomial import (
    ESTIMATOR as DIRICHLET_MULTINOMIAL,
)
from continuum_swr_replay.analysis.replay.fit_provenance import (
    record_inputs,
    write_fit_provenance,
)
from continuum_swr_replay.analysis.replay.fitting import (
    fit_conditional_poisson,
    poisson_estimator_name,
)
from continuum_swr_replay.analysis.replay.model_specs import (
    DESCRIPTIVE_CONTENT_RUNGS,
    LEAF_ONLY_CONTRAST_RUNGS,
    LEAF_ONLY_N_CATEGORIES,
    leaf_only_predictors,
)
from continuum_swr_replay.analysis.replay.model_specs import (
    SWR_CONTRAST_CONTENT_ARM_KEY_TERM as ARM_KEY_TERM,
)
from continuum_swr_replay.analysis.replay.model_specs import (
    SWR_CONTRAST_CONTENT_ARM_LADDER as ARM_LADDER,
)
from continuum_swr_replay.analysis.replay.model_specs import (
    SWR_CONTRAST_CONTENT_PATCH_KEY_TERM as PATCH_KEY_TERM,
)
from continuum_swr_replay.analysis.replay.model_specs import (
    SWR_CONTRAST_CONTENT_PATCH_LADDER as PATCH_LADDER,
)
from continuum_swr_replay.analysis.replay.standardization import (
    assert_standardized_matches_reference,
    attach_standardized,
    load_reference,
    save_reference,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.session_keys import SESSION_COLUMNS
from continuum_swr_replay.table_io import read_table

# Quasi-separated coefficients legitimately overflow exp() when rate ratios are formed
# outside the fit (RuntimeWarning). Convergence/perfect-separation warnings are recorded
# per fit as converged/separated/n_convergence_warnings flags inside the fitters (see
# analysis.replay._fit_diagnostics), so only the benign RuntimeWarning noise is silenced
# here -- NOT a blanket ignore that would hide fit-quality signals.
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RC = PROCESSED_DATA_DIR / "replay_content"
SC = list(SESSION_COLUMNS)
MIN_EVENTS = 18

#: Worker processes the Dirichlet-multinomial per-session fits are spread over.
DEFAULT_N_JOBS = 8

#: Concentration contrast the Dirichlet-multinomial adds to every rung.
LOG_PHI_IS_SWR = "log_phi_is_swr"

# The tables every fit in this family is built from. Their hashes go into the
# provenance sidecar of each artifact written here, and the readers of those
# artifacts check the same mapping, so a regenerated table cannot be read
# through coefficients that predate it.
FIT_INPUTS = {
    "event_arm_table": RC / "event_arm_table.pkl",
    "event_patch_table": RC / "event_patch_table.pkl",
    "stop_table": RC / "stop_table.pkl",
    "qvals_table": RC / "models_per_bin_rate" / "qvals_table.pkl",
}


# event-level scalars z-scored (so :is_current coefs are per-SD & comparable)
_ZSCALARS = [
    "rpe_postoutcome",
    "relative_value_switch_minus_stay_preoutcome",
    "reward_belief_differential_entropy_leaf_mean_preoutcome",
    "reward_belief_mean_update_l1_all_ports_postoutcome",
    "periswitch_index_trials",
    "trials_since_block_change",
    "patch_switch_fraction_last5_completed_trips",
]


# Fit-time standardization: one support label for every ``_z`` this runner makes.
# The moments are taken over the FULL assembled cohort (before the smoke subset and
# before the outcome stratum), so the strata share one per-SD unit and their
# coefficients stay comparable. The reference is persisted beside each artifact.
_FIT_SUPPORT_LABEL = "fit_cohort_all_rows"

# Event-length control. SWR-coupled events are the longer ones, and a longer
# trajectory visits more arms, so ``swr_x_is_current_arm`` partly reads "longer
# events leave the current arm" unless event length is in the design. The control
# is z(log n_nonlocal_bins) crossed with the indicator it modulates.
_LOG_EVENT_LENGTH = "log_n_nonlocal_bins"
_LOG1P_STOP_TIME = "log1p_time_since_stop_start_s"

_ARM_Z_SOURCES = (*_ZSCALARS, _LOG1P_STOP_TIME, _LOG_EVENT_LENGTH)
_PATCH_Z_SOURCES = (
    "periswitch_index_trials",
    "patch_switch_fraction_last5_completed_trips",
    _LOG1P_STOP_TIME,
    _LOG_EVENT_LENGTH,
)

# Current outcome strata. Rewarded and omitted events differ in their SWR share, so
# the contrast is refit within each; an event whose outcome is not yet known belongs
# to neither.
OUTCOME_STRATA = ("all", "rewarded", "omitted")


def _standardize(frame: pd.DataFrame, cols) -> tuple[pd.DataFrame, dict]:
    """Attach ``<col>_z`` over the fit cohort and return the reference used."""
    return attach_standardized(frame, list(cols), support_label=_FIT_SUPPORT_LABEL)


def _attach_log_event_length(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach ``log_n_nonlocal_bins``; refuse an event shorter than one bin.

    Every materialized non-local event spans at least one decoder bin by
    construction, so a zero/negative/missing count is a broken table, not a
    zero-length event -- and ``log`` would silently emit -inf/NaN for it.
    """
    n_bins = pd.to_numeric(frame["n_nonlocal_bins"], errors="coerce").to_numpy(
        dtype=float
    )
    bad = ~np.isfinite(n_bins) | (n_bins < 1)
    if bad.any():
        raise ValueError(
            f"n_nonlocal_bins must be finite and >= 1 on every event; "
            f"{int(bad.sum())} of {len(n_bins)} rows violate it "
            f"(min={np.nanmin(n_bins)})"
        )
    frame[_LOG_EVENT_LENGTH] = np.log(n_bins)
    return frame


def select_outcome_stratum(ev: pd.DataFrame, stratum: str) -> tuple[pd.DataFrame, dict]:
    """Restrict a cohort to one current-outcome stratum.

    Parameters
    ----------
    ev : pandas.DataFrame
        Event-grain cohort carrying ``event_id``, ``is_rewarded_outcome`` and
        ``is_swr`` (one row per event x candidate arm/patch).
    stratum : {"all", "rewarded", "omitted"}
        ``"all"`` keeps every event. ``"rewarded"`` keeps
        ``is_rewarded_outcome == 1``, ``"omitted"`` keeps ``== 0``; an event whose
        outcome is not yet known is in neither.

    Returns
    -------
    (frame, audit)
        The restricted frame and the event-grain counts, including the
        unknown-outcome events split by ``is_swr`` so the exclusion is reported.
    """
    if stratum not in OUTCOME_STRATA:
        raise ValueError(
            f"unknown outcome stratum {stratum!r}; expected one of {OUTCOME_STRATA}"
        )
    events = ev.drop_duplicates("event_id")
    outcome = pd.to_numeric(
        events["is_rewarded_outcome"].astype("object"), errors="coerce"
    )
    unknown = outcome.isna()
    is_swr = events["is_swr"].astype("object").astype(bool)
    if stratum == "all":
        kept = ev
    else:
        wanted = 1.0 if stratum == "rewarded" else 0.0
        keep_ids = set(events.loc[~unknown & (outcome == wanted), "event_id"])
        kept = ev[ev["event_id"].isin(keep_ids)]
    audit = {
        "stratum": stratum,
        "n_events_in": int(len(events)),
        "n_events_kept": int(kept["event_id"].nunique()),
        "n_events_unknown_outcome": int(unknown.sum()),
        "n_events_unknown_outcome_swr": int((unknown & is_swr).sum()),
        "n_events_unknown_outcome_non_swr": int((unknown & ~is_swr).sum()),
    }
    return kept, audit


def _z_reference_path(artifact_path):
    """``<artifact-stem>.z_reference.json`` beside a fitted artifact."""
    return artifact_path.with_name(f"{artifact_path.stem}.z_reference.json")


def _artifact_suffix(args: argparse.Namespace, response: str) -> str:
    """Artifact-name suffix: smoke subset, response, estimator, and stratum.

    The response is named only where it is not the estimator's own default, so
    a Dirichlet-multinomial artifact says ``_dm`` rather than repeating in
    ``_hard_count`` the one response that estimator has. ``_dm`` comes before
    the stratum suffix, so the stratified artifacts of one estimator sort
    together.
    """
    estimator = getattr(args, "estimator", "conditional_poisson")
    smoke = "_smoke" if args.max_per_animal is not None else ""
    default_response = default_response_for_estimator(estimator)
    response_suffix = "" if response == default_response else f"_{response}"
    dm = "_dm" if estimator == "dirichlet_multinomial" else ""
    stratum = "" if args.outcome_stratum == "all" else f"_{args.outcome_stratum}"
    return smoke + response_suffix + dm + stratum


# Raw-scale per-arm covariates whose coefficients the aggregation restates per SD
# (``aggregate_contrast``'s ``std_factor``). Their SDs are frozen over the FULL
# assembled cohort, like the ``_z`` reference, so a stratified refit reports effects
# in the same unit as the pooled one. The patch ladder's key terms are all binary or
# already standardized, so it has no raw-scale term to rescale.
_ARM_SD_TERMS = (
    "reward_belief_mean_candidate_arm_preoutcome",
    "reward_fraction_candidate_arm_last10_visits",
    "task_reward_probability_candidate_arm",
)
_PATCH_SD_TERMS: tuple[str, ...] = ()

# Row support for the per-arm SD unit. Every one of ``_ARM_SD_TERMS`` is a
# leaf-derived belief/reward quantity that is a STRUCTURAL ZERO on the three trunk
# arms, and the rungs that carry them are fitted on the leaves (see
# ``LEAF_ONLY_CONTRAST_RUNGS``). Taking the SD over all nine candidates therefore
# states the effect per SD of a spread the fitted rows never see -- measured on the
# canonical cohort, the trunk zeros inflate the arm-value SD by 2.1x, which would
# shrink every per-SD value effect by the same factor.
_ARM_SD_SUPPORT_LABEL = "leaf_arms"
_PATCH_SD_SUPPORT_LABEL = _FIT_SUPPORT_LABEL


def _leaf_arm_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """The six leaf-arm rows of an arm cohort; the trunks carry structural zeros."""
    if "candidate_arm" not in frame.columns:
        raise KeyError(
            "arm cohort has no 'candidate_arm' column; the per-SD unit cannot be "
            "restricted to the leaf arms it is read on"
        )
    return frame[
        pd.to_numeric(frame["candidate_arm"], errors="coerce") < LEAF_ONLY_N_CATEGORIES
    ]


def _sd_reference(frame: pd.DataFrame, cols) -> dict:
    """Pooled SD of each raw covariate present in ``frame`` (the per-SD unit)."""
    return {
        c: float(np.nanstd(frame[c].to_numpy(dtype=float)))
        for c in cols
        if c in frame.columns
    }


def prepare_cohorts(
    ev: pd.DataFrame,
    pt: pd.DataFrame,
    *,
    outcome_stratum: str,
    max_per_animal: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Freeze the cohort-wide per-SD scales, then apply the stratum and smoke subset.

    Order is load-bearing: the SDs (like the ``_z`` reference the assemblers
    return) are estimated on the FULL cohorts, so the ``rewarded`` and ``omitted``
    refits state their per-SD effects in the same unit as the pooled fit instead of
    each rescaling to its own stratum.

    Parameters
    ----------
    ev, pt : pandas.DataFrame
        The fully assembled arm and patch cohorts.
    outcome_stratum : {"all", "rewarded", "omitted"}
        Current-outcome stratum to fit.
    max_per_animal : int, optional
        Smoke subset: keep the first N sessions per animal.

    Returns
    -------
    (ev, pt, scale, stratum_audit)
        The selected cohorts, the frozen per-SD scales (each with the row support it
        was taken over -- ``leaf_arms`` for the arm cohort), and the outcome-stratum
        event counts.
    """
    scale = {
        # Support is recorded per cohort because they differ: the arm SDs are taken
        # over the leaf arms only (the trunks are structural zeros for every term the
        # aggregation rescales), the patch SDs over all rows.
        "arm_cohort_support": _ARM_SD_SUPPORT_LABEL,
        "arm_cohort": _sd_reference(_leaf_arm_rows(ev), _ARM_SD_TERMS),
        "patch_cohort_support": _PATCH_SD_SUPPORT_LABEL,
        "patch_cohort": _sd_reference(pt, _PATCH_SD_TERMS),
    }
    ev, stratum_audit = select_outcome_stratum(ev, outcome_stratum)
    if outcome_stratum != "all":
        # One stratum, one event set: the patch cohort follows the arm cohort's
        # events rather than re-deriving the outcome on patch rows.
        pt = pt[pt["event_id"].isin(set(ev["event_id"]))]
    if max_per_animal is not None:
        keep = (
            ev[SC].drop_duplicates().groupby("animal", sort=False).head(max_per_animal)
        )
        ev = ev.merge(keep, on=SC)
        pt = pt.merge(keep, on=SC)
    return ev, pt, scale, stratum_audit


def _assemble_arm() -> tuple[pd.DataFrame, dict]:
    """Load the arm cohort and attach its design columns and ``_z`` reference."""
    ev = read_table(FIT_INPUTS["event_arm_table"])
    require_covariate_schema(ev, table="event_arm")
    reference_path = RC / "standardization_reference.json"
    if not reference_path.exists():
        raise FileNotFoundError(
            f"{reference_path} is required to validate arm_uncertainty_z; "
            "regenerate the canonical replay-content tables"
        )
    try:
        assert_standardized_matches_reference(
            ev,
            ("reward_belief_variance_candidate_arm_preoutcome",),
            load_reference(reference_path),
            support_label="leaf_arms",
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "event_arm_table arm_uncertainty_z has a stale or invalid "
            "standardization reference; regenerate the canonical table and "
            f"reference before fitting: {exc}"
        ) from exc
    add_arm_fixed_effects(ev)
    stop = read_table(FIT_INPUTS["stop_table"])
    require_covariate_schema(stop, table="stop")
    stop = add_uncertainty_covariates(stop)
    q = read_table(FIT_INPUTS["qvals_table"])
    require_covariate_schema(q, table="qvals")
    ev = ev.merge(
        stop[["stop_id", "known_reward_outcome_lag_1", "patch_switch_fraction_last5_completed_trips"]], on="stop_id", how="left"
    )
    ev = ev.merge(
        q[
            [
                "stop_id",
                "rpe_postoutcome",
                "relative_value_switch_minus_stay_preoutcome",
                "reward_belief_differential_entropy_leaf_mean_preoutcome",
                "reward_belief_mean_update_l1_all_ports_postoutcome",
                "periswitch_index_trials",
                "is_periswitch_post",
                "trials_since_block_change",
            ]
        ],
        on="stop_id",
        how="left",
    )
    return _attach_arm_design_columns(ev)


def _attach_arm_design_columns(ev: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Standardize the arm cohort's scalars and build every interaction product.

    Parameters
    ----------
    ev : pandas.DataFrame
        Arm cohort (one row per event x candidate arm) carrying the raw event
        scalars, ``n_nonlocal_bins``, ``time_since_stop_start_s``, and the arm
        indicators.

    Returns
    -------
    (ev, reference)
        ``ev`` with the ``<col>_z`` columns and interaction products appended, and
        the standardization reference actually used (persisted beside the fit).
    """
    ev[_LOG1P_STOP_TIME] = np.log1p(ev["time_since_stop_start_s"].to_numpy(dtype=float))
    _attach_log_event_length(ev)
    ev, z_reference = _standardize(ev, _ARM_Z_SOURCES)
    cur = ev["is_current_arm"].to_numpy(dtype=float)
    prev = ev["is_prev_arm"].to_numpy(dtype=float)
    swr = ev["is_swr"].to_numpy(dtype=float)
    # State-modulation interactions spell out both canonical factors.
    ev["rpe_postoutcome_z_x_is_current_arm"] = cur * ev["rpe_postoutcome_z"]
    ev["relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm"] = cur * ev[
        "relative_value_switch_minus_stay_preoutcome_z"
    ]
    ev[
        "reward_belief_differential_entropy_leaf_mean_preoutcome_z_x_is_current_arm"
    ] = cur * ev[
        "reward_belief_differential_entropy_leaf_mean_preoutcome_z"
    ]
    ev[
        "reward_belief_mean_update_l1_all_ports_postoutcome_z_x_is_current_arm"
    ] = cur * ev[
        "reward_belief_mean_update_l1_all_ports_postoutcome_z"
    ]
    # context interactions
    ev["periswitch_index_trials_z_x_is_current_arm"] = (
        cur * ev["periswitch_index_trials_z"]
    )
    ev["is_periswitch_post_x_is_current_arm"] = (
        cur * ev["is_periswitch_post"].to_numpy(dtype=float)
    )
    ev["periswitch_index_trials_z_x_is_periswitch_post_x_is_current_arm"] = (
        cur
        * ev["periswitch_index_trials_z"]
        * ev["is_periswitch_post"].to_numpy(dtype=float)
    )
    ev["trials_since_block_change_z_x_is_current_arm"] = (
        cur * ev["trials_since_block_change_z"]
    )
    ev["patch_switch_fraction_last5_completed_trips_z_x_is_current_arm"] = cur * ev[
        "patch_switch_fraction_last5_completed_trips_z"
    ]
    # within-stop-time duration control (z(log1p t) : is_current)
    ev["log1p_time_since_stop_start_s_z_x_is_current_arm"] = (
        cur * ev[f"{_LOG1P_STOP_TIME}_z"]
    )
    # event-length control (z(log n bins) : is_current / : is_prev)
    ev["log_n_nonlocal_bins_z_x_is_current_arm"] = cur * ev[f"{_LOG_EVENT_LENGTH}_z"]
    ev["log_n_nonlocal_bins_z_x_is_prev_arm"] = prev * ev[f"{_LOG_EVENT_LENGTH}_z"]
    # SWR interactions inherit the canonical base-covariate spelling.
    ev["swr_x_reward_belief_mean_candidate_arm_preoutcome"] = swr * ev[
        "reward_belief_mean_candidate_arm_preoutcome"
    ].to_numpy(dtype=float)
    ev["swr_x_reward_belief_variance_candidate_arm_preoutcome_z"] = swr * ev[
        "reward_belief_variance_candidate_arm_preoutcome_z"
    ].to_numpy(dtype=float)
    return ev, z_reference


def _assemble_patch(arm_ev: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Load the patch cohort and attach its design columns and ``_z`` reference."""
    pt = read_table(FIT_INPUTS["event_patch_table"])
    require_covariate_schema(pt, table="event_patch")
    # patch table lacks stop_id: map event_id -> stop_id from the arm table
    e2s = arm_ev[["event_id", "stop_id"]].drop_duplicates("event_id")
    pt = pt.merge(e2s, on="event_id", how="left")
    # Within-stop time and event length are event-level (constant across
    # arms/patches within an event); carry them for the duration and
    # event-length controls.
    e2t = arm_ev[
        ["event_id", "time_since_stop_start_s", "n_nonlocal_bins"]
    ].drop_duplicates("event_id")
    pt = pt.merge(e2t, on="event_id", how="left")
    stop = read_table(FIT_INPUTS["stop_table"])
    require_covariate_schema(stop, table="stop")
    stop = add_uncertainty_covariates(stop)
    q = read_table(FIT_INPUTS["qvals_table"])
    require_covariate_schema(q, table="qvals")
    pt = pt.merge(stop[["stop_id", "patch_switch_fraction_last5_completed_trips"]], on="stop_id", how="left")
    pt = pt.merge(q[["stop_id", "periswitch_index_trials"]], on="stop_id", how="left")
    return _attach_patch_design_columns(pt)


def _attach_patch_design_columns(pt: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Standardize the patch cohort's scalars and build every interaction product.

    The patch cohort is its own standardization scope: the same-named arm-cohort
    entries are computed over a different set of rows and must not be reused here.

    Parameters
    ----------
    pt : pandas.DataFrame
        Patch cohort (one row per event x candidate patch) carrying the raw patch
        scalars, ``n_nonlocal_bins``, ``time_since_stop_start_s``, and the patch
        indicators.

    Returns
    -------
    (pt, reference)
        ``pt`` with the ``<col>_z`` columns and interaction products appended, and
        the standardization reference actually used.
    """
    pt[_LOG1P_STOP_TIME] = np.log1p(pt["time_since_stop_start_s"].to_numpy(dtype=float))
    _attach_log_event_length(pt)
    pt, z_reference = _standardize(pt, _PATCH_Z_SOURCES)
    cp = pt["is_current_patch"].to_numpy(float)
    pp = pt["is_prev_patch"].to_numpy(float)
    swr = pt["is_swr"].to_numpy(float)
    pt["periswitch_index_trials_z_x_is_current_patch"] = (
        cp * pt["periswitch_index_trials_z"]
    )
    # Patch analogues of the arm within-stop-time and event-length controls.
    pt["log1p_time_since_stop_start_s_z_x_is_current_patch"] = (
        cp * pt[f"{_LOG1P_STOP_TIME}_z"]
    )
    pt["log_n_nonlocal_bins_z_x_is_current_patch"] = cp * pt[f"{_LOG_EVENT_LENGTH}_z"]
    pt["log_n_nonlocal_bins_z_x_is_prev_patch"] = pp * pt[f"{_LOG_EVENT_LENGTH}_z"]
    pt["patch_switch_fraction_last5_completed_trips_z_x_is_current_patch"] = cp * pt[
        "patch_switch_fraction_last5_completed_trips_z"
    ]
    pt["swr_x_is_current_patch"] = swr * cp
    return pt, z_reference


# Event-level columns carried for the cohort-audit drop profile (mechanism A vs B).
_PROFILE_COLS = (
    "time_since_stop_start_s",
    "is_rewarded_outcome",
    "reward_belief_mean_candidate_arm_preoutcome",
    "relative_value_switch_minus_stay_preoutcome",
)


def _prepare_session(
    g: pd.DataFrame, predictors: list[str], *, response_col: str
) -> tuple[pd.DataFrame | None, dict]:
    """Apply the event-level cohort rules to one session, before any fit.

    Returns ``(fittable_rows_or_None, prep)``. ``prep`` carries the event-level
    cohort counts (``n_in`` and the mutually exclusive ``n_drop_qvals_gap`` /
    ``n_drop_own_predictor_missing`` / ``n_drop_zero_response``), the surviving
    event count, a skip ``reason``, and a one-row-per-dropped-event ``dropped``
    frame for the profile.

    Separated from the fit itself because the Dirichlet-multinomial fits a whole
    ladder's worth of sessions in one batch, and the cohort rules that decide
    which rows it may see must be the same ones the conditional Poisson applies.

    A session missing a required predictor column is a clean, auditable skip
    (``reason="missing:<cols>"`` → ``n_excluded_session``), never a fitter call
    with absent columns.
    """
    n_in = int(g["event_id"].nunique()) if "event_id" in g.columns else 0
    empty = g.iloc[0:0]
    required = [response_col, "event_id", *predictors]
    missing = [c for c in required if c not in g.columns]
    if missing:
        return None, {
            "n_in": n_in,
            "n_qvals": 0,
            "n_own": 0,
            "n_zero": 0,
            "n_surv": n_in,
            "dropped": empty,
            "reason": f"missing:{','.join(missing)}",
        }

    # Split NaN-dropped events into the qvals gap (mechanism B) vs own-predictor
    # missingness (mechanism A), qvals taking priority so the buckets stay disjoint.
    qvals_preds = [c for c in predictors if c in QVALS_GAP_COLS]
    # The response is not a predictor. An event whose response is undefined --
    # every candidate NaN, which is what a leaf-only event with no leaf mass
    # renormalizes to -- has no allocation to fit, and that is a response drop,
    # not predictor missingness. Keeping it out of this split books it under
    # n_drop_zero_response below, as the ladder's cohort audit does.
    other_cols = [c for c in predictors if c not in QVALS_GAP_COLS]
    qvals_nan = (
        set(g.loc[g[qvals_preds].isna().any(axis=1), "event_id"]) if qvals_preds else set()
    )
    other_nan = (
        set(g.loc[g[other_cols].isna().any(axis=1), "event_id"]) if other_cols else set()
    )
    other_nan -= qvals_nan
    n_qvals, n_own = len(qvals_nan), len(other_nan)
    dropped_events = qvals_nan | other_nan

    # Preserve the full candidate set within every retained event (row-wise NaN drops
    # would shrink an event's arm/patch set and desync it from the null's n_categories).
    surv = g[~g["event_id"].isin(dropped_events)]
    # skipna=False: an event missing the response on any candidate has no
    # defined allocation, and fitting the candidates it does have would shrink
    # its candidate set out of step with the null's n_categories. Undefined and
    # empty allocations are the same drop -- there is nothing to fit.
    totals = surv.groupby("event_id")[response_col].agg(
        lambda values: values.sum(skipna=False)
    )
    zero_events = set(totals[totals.isna() | totals.le(0)].index)
    n_zero = len(zero_events)
    surv = surv[~surv["event_id"].isin(zero_events)]
    n_surv = int(surv["event_id"].nunique())

    dropped = _dropped_event_rows(g, dropped_events | zero_events)
    prep = {
        "n_in": n_in,
        "n_qvals": n_qvals,
        "n_own": n_own,
        "n_zero": n_zero,
        "n_surv": n_surv,
        "dropped": dropped,
        "reason": "",
    }
    if n_surv < MIN_EVENTS:
        return None, {**prep, "reason": "too_few"}
    return surv, prep


def _session_audit(prep: dict, r: dict | None, *, reason: str = "") -> dict:
    """Close one session's audit record around whatever the fitter returned.

    Any survivor not modeled -- the fitter's ``n < 2 * n_params`` gate, a failed
    fit, or a design the estimator refused -- is a session/fit-level exclusion,
    which is what makes ``n_in - Σdrops == n_modeled`` reconcile exactly.
    """
    n_modeled = int(r["n_events"]) if r is not None else 0
    if r is None:
        reason = reason or prep["reason"] or "fit_none"
    else:
        reason = ""
    return _audit(
        prep["n_in"],
        prep["n_qvals"],
        prep["n_own"],
        prep["n_zero"],
        prep["n_surv"] - n_modeled,
        n_modeled,
        reason,
        prep["dropped"],
    )


def _fit_session(
    g: pd.DataFrame, predictors: list[str], *, response_col: str, n_categories: int
) -> tuple[dict | None, dict]:
    """Fit one session's conditional Poisson and return ``(row_or_None, audit)``."""
    surv, prep = _prepare_session(g, predictors, response_col=response_col)
    if surv is None:
        return None, _session_audit(prep, None)
    r = fit_conditional_poisson(
        surv,
        predictors=predictors,
        response_col=response_col,
        event_col="event_id",
        n_categories=n_categories,
    )
    return r, _session_audit(prep, r)


def _audit(n_in, n_qvals, n_own, n_zero, n_excluded, n_modeled, reason, dropped):
    return {
        "n_in": n_in,
        "n_drop_qvals_gap": n_qvals,
        "n_drop_own_predictor_missing": n_own,
        "n_drop_zero_response": n_zero,
        "n_excluded_session": n_excluded,
        "n_modeled": n_modeled,
        "reason": reason,
        "dropped": dropped,
    }


def _dropped_event_rows(g: pd.DataFrame, event_ids: set) -> pd.DataFrame:
    """One row per dropped event, profile columns only (for drop_profile)."""
    cols = [c for c in _PROFILE_COLS if c in g.columns]
    sub = g[g["event_id"].isin(event_ids)]
    return sub.drop_duplicates("event_id")[cols] if cols else g.iloc[0:0]


def _concentration_cols(frame: pd.DataFrame) -> tuple[str, ...]:
    """Event-level columns entering ``log phi`` for one population's rows.

    ``is_swr`` is the concentration modifier the Dirichlet-multinomial exists to
    measure -- whether an SWR event's mass is spread over more arms than a
    non-SWR event's -- but only in the pooled population. Within the ``SWR`` or
    ``nonSWR`` split the flag is constant, its coefficient is not identified,
    and asking for it would return an infinite standard error rather than a
    concentration difference that is simply not in question there.
    """
    if "is_swr" not in frame.columns:
        return ()
    values = set(frame["is_swr"].dropna().astype(bool).unique())
    return ("is_swr",) if values == {False, True} else ()


def _fit_population(
    pop_frame: pd.DataFrame,
    predictors: list[str],
    *,
    estimator: str,
    response_col: str,
    n_categories: int,
    n_jobs: int,
    min_events_session_phi: int,
) -> list[tuple[tuple, dict | None, dict]]:
    """Fit every session of one (rung, population), in groupby order.

    Returns ``(session_key, row_or_None, audit)`` per session. Both estimators
    see exactly the same prepared rows; only the fit differs. The
    Dirichlet-multinomial fits are batched so they can share a process pool and
    a pooled concentration intercept for the short sessions, and a session whose
    within-event design it refuses becomes an audited skip
    (``reason="dm_design:..."``) rather than an exception out of the ladder.
    """
    if estimator != "dirichlet_multinomial":
        return [
            (
                tuple(keys),
                *_fit_session(
                    g,
                    predictors,
                    response_col=response_col,
                    n_categories=n_categories,
                ),
            )
            for keys, g in pop_frame.groupby(SC, sort=False)
        ]

    prepared: list[tuple[tuple, pd.DataFrame | None, dict]] = [
        (tuple(keys), *_prepare_session(g, predictors, response_col=response_col))
        for keys, g in pop_frame.groupby(SC, sort=False)
    ]
    animal_position = SC.index("animal")
    fits, skips = fit_sessions_dirichlet_multinomial(
        [
            (key, key[animal_position], surv)
            for key, surv, _prep in prepared
            if surv is not None
        ],
        predictors,
        min_events=min_events_session_phi,
        n_jobs=n_jobs,
        response_col=response_col,
        event_col="event_id",
        concentration_cols=_concentration_cols(pop_frame),
        n_categories=n_categories,
    )
    rows = dict(fits)
    reasons = dict(skips)
    return [
        (
            key,
            rows.get(key),
            _session_audit(prep, rows.get(key), reason=reasons.get(key, "")),
        )
        for key, _surv, prep in prepared
    ]


def _run(
    ev: pd.DataFrame,
    ladder: dict,
    *,
    response_col: str,
    n_categories: int,
    estimator: str = "conditional_poisson",
    n_jobs: int = 1,
    min_events_session_phi: int = DEFAULT_MIN_EVENTS_SESSION_PHI,
) -> tuple[pd.DataFrame, list[dict]]:
    """Fit the ladder per session and build one cohort-audit record per (model, pop).

    Returns ``(results, cohort_records)``; the audit records reconcile
    ``n_in - Σdrops == n_modeled`` (grain = event) via :func:`cohort_record`.
    """
    rows = []
    audits = []
    splits = {"all": ev, "SWR": ev[ev.is_swr], "nonSWR": ev[~ev.is_swr]}
    for name, (preds, scope) in ladder.items():
        pops = ["all"] if scope == "pooled" else ["all", "SWR", "nonSWR"]
        for pop in pops:
            acc = dict.fromkeys((
                "n_in", "n_drop_qvals_gap", "n_drop_own_predictor_missing",
                "n_drop_zero_response", "n_excluded_session", "n_modeled",
            ), 0)
            sessions_in = sessions_modeled = n_nonconv = n_sep = 0
            animals_modeled: set = set()
            skip_reasons: set = set()
            dropped_frames = []
            # Belief rungs are fitted over the leaf arms only: the trunks carry a
            # structural zero for every leaf-derived belief covariate, so they add
            # no information about those constructs. Persisted z units are also
            # defined on leaf support. See LEAF_ONLY_CONTRAST_RUNGS.
            leaf_only = name in LEAF_ONLY_CONTRAST_RUNGS
            pop_frame = splits[pop]
            fit_categories = n_categories
            fit_preds = preds
            renormalized = False
            if leaf_only:
                pop_frame = pop_frame[pop_frame["candidate_arm"] < LEAF_ONLY_N_CATEGORIES]
                fit_categories = LEAF_ONLY_N_CATEGORIES
                fit_preds = leaf_only_predictors(preds)
                # A fraction is a share of the event's whole nine-arm mass, and
                # the conditional likelihood weights an event by the total it
                # carries, so dropping the trunks unrescaled would weight an
                # event by its leaf share -- and SWR events put far more of
                # their bins on the trunks than non-SWR events do, which is the
                # very contrast being estimated. Rescale to unit mass over the
                # six leaves. Counts keep their timebin weighting by design.
                renormalized = response_col.endswith("_fraction")
                if renormalized:
                    pop_frame = renormalize_fraction_over_candidates(
                        pop_frame, response_col=response_col
                    )
            for keys, r, a in _fit_population(
                pop_frame,
                list(fit_preds),
                estimator=estimator,
                response_col=response_col,
                n_categories=fit_categories,
                n_jobs=n_jobs,
                min_events_session_phi=min_events_session_phi,
            ):
                sessions_in += 1
                for k in acc:
                    acc[k] += a[k]
                if len(a["dropped"]):
                    dropped_frames.append(a["dropped"])
                if r is None:
                    if a["reason"]:
                        skip_reasons.add(a["reason"].split(":")[0])
                    continue
                sessions_modeled += 1
                animals_modeled.add(keys[SC.index("animal")])
                n_nonconv += r.get("converged") is False
                n_sep += bool(r.get("separated"))
                rows.append(
                    {
                        "model": name,
                        "population": pop,
                        **dict(zip(SC, keys, strict=True)),
                        # ``sd_<term>`` travels with the coefficients because the
                        # aggregation's identifiability backstop is read on the
                        # effect scale (|coef| x sd). Without it a runaway fit is
                        # judged by its raw coefficient, which drops identified
                        # sessions whose predictor has a small SD and keeps
                        # unidentified ones whose predictor has a large one.
                        **{
                            k: v
                            for k, v in r.items()
                            if k.startswith(("coef_", "bse_", "pval_", "sd_"))
                        },
                        # Per-term separation flags, so a fit whose self-history
                        # coefficient ran away is not dropped for every term.
                        "separated_terms": r.get("separated_terms"),
                        "n_events": r["n_events"],
                        "n_params": r["n_params"],
                        # Which allocation the row was fit on, and therefore at
                        # which grain its llf/AIC/SEs are stated.
                        "estimator": r.get("estimator"),
                        "response_col": response_col,
                        # True where the trunk rows were dropped and the
                        # remaining fractions rescaled to unit mass per event.
                        "response_renormalized": renormalized,
                        # Whether the row's llf carries the response's
                        # normalizing constants, so a later comparison cannot
                        # difference a full likelihood against a conditional one.
                        "llf_includes_constants": r.get("llf_includes_constants"),
                        # Where this fit's concentration intercept came from:
                        # its own likelihood, or the animal-pooled median that a
                        # session too short to estimate it is pinned to.
                        "phi_source": r.get("phi_source"),
                        # fit-diagnostics flags (additive; drive the aggregation's
                        # session-level nonconverged/separated counts).
                        "converged": r.get("converged"),
                        "separated": r.get("separated"),
                        "n_convergence_warnings": r.get("n_convergence_warnings"),
                    }
                )
            dropped = (
                pd.concat(dropped_frames, ignore_index=True) if dropped_frames
                else pd.DataFrame()
            )
            notes = drop_profile(dropped)
            if skip_reasons:
                # Sessions the estimator produced no fit for are already counted
                # under n_excluded_session; naming why keeps a refused design
                # distinguishable from a session that was merely too short.
                notes = "; ".join(
                    filter(None, [notes, "no fit: " + ", ".join(sorted(skip_reasons))])
                )
            audits.append(cohort_record(
                name, pop, "event",
                n_in=acc["n_in"], n_modeled=acc["n_modeled"],
                sessions_in=sessions_in, sessions_modeled=sessions_modeled,
                animals_modeled=len(animals_modeled), drop_notes=notes,
                n_drop_qvals_gap=acc["n_drop_qvals_gap"],
                n_drop_own_predictor_missing=acc["n_drop_own_predictor_missing"],
                n_drop_zero_response=acc["n_drop_zero_response"],
                n_excluded_session=acc["n_excluded_session"],
                n_sessions_flagged_nonconverged=n_nonconv,
                n_sessions_flagged_separated=n_sep,
            ))
    return pd.DataFrame(rows), audits


def _summ(res: pd.DataFrame, ladder: dict, key_term: dict) -> None:
    print(
        f"\n{'model':18s}{'pop':8s}{'term':30s}{'n':>4}{'medCoef':>9}"
        f"{'%pos':>6}{'anim':>6}{'signed-rank p':>14}"
    )
    for name in ladder:
        term = key_term.get(name)
        if term is None:
            continue
        col = f"coef_{term}"
        scope = ladder[name][1]
        for pop in ["all"] if scope == "pooled" else ["all", "SWR", "nonSWR"]:
            s = res[(res.model == name) & (res.population == pop)]
            if col not in s.columns:
                continue
            x = s[col].dropna()
            x = x[x.abs() < 10]  # drop quasi-separated sessions (coefs are O(1))
            if len(x) < 5:
                continue
            nan = (s.groupby("animal")[col].median() > 0).sum()
            p = wilcoxon(x)[1] if len(x) > 5 else np.nan
            star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""
            print(
                f"{name:18s}{pop:8s}{term:30s}{len(x):4d}{x.median():+9.3f}"
                f"{100 * (x > 0).mean():5.0f}%{f'{nan}/5':>6}{p:14.2e} {star}"
            )


def _aggregate(
    res: pd.DataFrame, ladder: dict, key_term: dict, level: str, sd: dict | None = None
) -> pd.DataFrame:
    """Audited across-session aggregation (shared engine); one row per rung x pop.

    See :mod:`continuum_swr_replay.analysis.replay.contrast_aggregation` for the
    column semantics (animal-level primary, session signed-rank descriptive, BH-FDR,
    standardized per-SD columns). This wrapper just expands the content ladder into
    (model, population, term) specs so the content and rate ladders mirror.
    """
    return aggregate_contrast(res, specs_from_ladder(ladder, key_term), level, sd)


def _concentration_specs(ladder: dict) -> list[tuple[str, str, str]]:
    """Aggregation specs for the concentration contrast, one per rung x population.

    ``log_phi_is_swr`` is not any rung's key term -- it is the same quantity on
    every rung, the difference in how tightly an SWR event's mass is
    concentrated -- so it is aggregated as its own family rather than folded
    into the beta contrast's BH-FDR. Specs are emitted for every rung and
    population; the ones whose fits carry no such coefficient (the conditional
    Poisson everywhere, and the within-SWR splits, where the flag is constant)
    produce no row.
    """
    return [
        (name, pop, LOG_PHI_IS_SWR)
        for name in ladder
        for pop in (
            ["all"] if ladder[name][1] == "pooled" else ["all", "SWR", "nonSWR"]
        )
    ]


def _renormalized_rungs(*results: pd.DataFrame) -> list[str]:
    """Rung names whose fitted rows were produced on a rescaled response.

    Read off the rows that were actually fitted rather than off the leaf-only
    rung set, so a run that skipped or never requested a rung cannot claim it.

    Parameters
    ----------
    *results : pd.DataFrame
        Per-session result frames carrying ``model`` and, where the fit rescaled
        its response to the fitted candidate set, ``response_renormalized``.

    Returns
    -------
    list of str
        Sorted rung names, empty when no fitted row was rescaled.
    """
    names: set[str] = set()
    for res in results:
        if res.empty or "response_renormalized" not in res.columns:
            continue
        rescaled = res["response_renormalized"].fillna(False).astype(bool)
        names |= set(res.loc[rescaled, "model"])
    return sorted(names)


def _provenance(
    args: argparse.Namespace,
    n_sessions: int,
    z_reference: dict | None = None,
    outcome_stratum_audit: dict | None = None,
    per_sd_scale: dict | None = None,
    renormalized_rungs: list[str] | None = None,
) -> dict:
    """Reproducibility record saved alongside the audited results."""
    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        git_hash = "unknown"
    return {
        "git_hash": git_hash,
        # Fit-time _z reference (event-cohort mean/SD for the event scalars). This is
        # the runner's OWN standardization -- NOT the builder's pooled-global table _z.
        "fit_time_z_reference": z_reference or {},
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_sessions": int(n_sessions),
        "smoke_max_per_animal": args.max_per_animal,
        # Which per-session estimator produced the coefficients, and therefore
        # whether the table carries a concentration contrast at all.
        "estimator": getattr(args, "estimator", "conditional_poisson"),
        "min_events_session_phi": getattr(args, "min_events_session_phi", None),
        # Cohort selection: which current-outcome stratum was fit, and how many
        # events had no known outcome and were therefore in neither stratum.
        "outcome_stratum": args.outcome_stratum,
        "outcome_stratum_audit": outcome_stratum_audit or {},
        # Per-SD units the audited table's standardized effects are stated in.
        # Frozen over the full cohort, so they do not move between strata.
        "per_sd_scale": per_sd_scale or {},
        "inference": "per-session conditional Poisson; across-session Wilcoxon "
        "signed-rank across sessions is DESCRIPTIVE only (sessions within an animal "
        "are not independent); primary population evidence is per-animal sign "
        "consistency (n_animals_pos / n_animals, n=5). BH-FDR across the rung family "
        "per population is applied to the descriptive session signed-rank p.",
        # Rungs whose fitted rows were produced on a response rescaled to the
        # candidate set they conditioned on (the leaf-only rungs, on the
        # fraction response). Read off the rows that were fitted, so a rung that
        # produced none is not named here.
        "response_renormalized_rungs": list(renormalized_rungs or []),
        "q_arm_source": (
            "beta-bernoulli per-arm value "
            "(reward_belief_mean_candidate_arm_preoutcome)"
        ),
        "deferred": [
            "mixed conditional logit with animal random effects (the rigorous "
            "population inference; supersedes the descriptive session signed-rank p). "
            "NOT available in statsmodels (ConditionalPoisson takes no random effects "
            "or cluster-robust cov); realistic path is NumPyro (JAX already a dep). "
            "Caveat: the animal random-effect variance is estimated from n=5 groups, "
            "so this does honest bookkeeping, not a power upgrade.",
            "common complete-case cohort across rungs",
            "within-group-centered VIF (fitting.py:420 uses the raw design)",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--max-per-animal",
        type=int,
        default=None,
        help="smoke: keep first N sessions per animal",
    )
    ap.add_argument(
        "--estimator",
        type=str,
        default="conditional_poisson",
        choices=list(ESTIMATOR_CHOICES),
        help=(
            "Per-session estimator. 'conditional_poisson' (default, shipped) "
            "fits the softmax allocation. 'dirichlet_multinomial' adds an "
            "explicit within-event concentration, so an event's autocorrelated "
            "timebins are not counted as independent draws and every rung also "
            f"reports coef_{LOG_PHI_IS_SWR} -- a direct test of whether SWR "
            "events spread their mass over more arms. It is defined on the "
            "integer counts, so it forces --response hard_count, and it writes "
            "'_dm' artifacts of its own."
        ),
    )
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help=(
            "Worker processes for the Dirichlet-multinomial per-session fits "
            f"(default: {DEFAULT_N_JOBS}). Ignored by the conditional Poisson."
        ),
    )
    ap.add_argument(
        "--min-events-session-phi",
        type=int,
        default=DEFAULT_MIN_EVENTS_SESSION_PHI,
        help=(
            "Modelled events a session needs to estimate its own "
            "Dirichlet-multinomial concentration intercept (default: "
            f"{DEFAULT_MIN_EVENTS_SESSION_PHI}); shorter sessions are pinned to "
            "the median intercept of that animal's longer sessions."
        ),
    )
    ap.add_argument(
        "--response",
        type=str,
        default=None,
        choices=list(RESPONSE_CHOICES),
        help=(
            "Per-event response. Defaults to the estimator's own: 'fraction' "
            "for the conditional Poisson, 'hard_count' for the "
            "Dirichlet-multinomial (which has no fraction form). 'fraction' fits "
            "arm_fraction / patch_fraction, so each event contributes one "
            "observation of unit mass. 'hard_count' fits the timebin counts, "
            "weighting an event by its length and treating its autocorrelated "
            "bins as independent draws; it writes '_hard_count' artifacts so "
            "the two never overwrite each other."
        ),
    )
    ap.add_argument(
        "--outcome-stratum",
        type=str,
        default="all",
        choices=list(OUTCOME_STRATA),
        help=(
            "Current-outcome stratum to fit. Rewarded and omitted stops differ in "
            "their SWR share, so 'rewarded' / 'omitted' refit the ladder within "
            "one outcome (events whose outcome is not yet known are in neither) "
            "and write '_rewarded' / '_omitted' artifacts."
        ),
    )
    args = ap.parse_args()
    # One resolution point for the response: the estimator's own default, and a
    # refusal where the two cannot be combined.
    response = resolve_response_choice(args.estimator, args.response)
    arm_response = resolve_response_column("arm", response)
    patch_response = resolve_response_column("patch", response)
    estimator = (
        DIRICHLET_MULTINOMIAL
        if args.estimator == "dirichlet_multinomial"
        else poisson_estimator_name(arm_response)
    )
    fit_kwargs = {
        "estimator": args.estimator,
        "n_jobs": args.n_jobs,
        "min_events_session_phi": args.min_events_session_phi,
    }

    # Causal-hygiene gate: every rung's predictors must be pre-outcome, OR the
    # rung must be an explicitly-declared descriptive rung (RPE/value association or
    # peri-switch alignment). A NEW leaky covariate on an undeclared rung fails here.
    for ladder in (ARM_LADDER, PATCH_LADDER):
        for name, (preds, _scope) in ladder.items():
            assert_predictor_timing(
                preds, descriptive=(name in DESCRIPTIVE_CONTENT_RUNGS)
            )

    sfx = _artifact_suffix(args, response)
    (RC / "models").mkdir(parents=True, exist_ok=True)
    # The standardization references are taken over the FULL assembled cohorts,
    # before the outcome stratum and the smoke subset, so every stratum's
    # coefficients are stated in the same per-SD unit.
    ev, arm_z_reference = _assemble_arm()
    pt, patch_z_reference = _assemble_patch(ev)
    # Hash the tables now, while they are the ones being fit: the ladder runs
    # for long enough that a table could be regenerated before the artifacts are
    # written, and a sidecar hashed at write time would certify bytes no fit saw.
    fit_inputs = record_inputs(FIT_INPUTS)
    if response == "fraction":
        validate_fraction_response(
            ev, arm_response, table_label=FIT_INPUTS["event_arm_table"]
        )
        validate_fraction_response(
            pt, patch_response, table_label=FIT_INPUTS["event_patch_table"]
        )
    logger.info("response = %s / %s (%s)", arm_response, patch_response, estimator)
    fit_z_reference = {"arm_cohort": arm_z_reference, "patch_cohort": patch_z_reference}
    ev, pt, per_sd_scale, stratum_audit = prepare_cohorts(
        ev,
        pt,
        outcome_stratum=args.outcome_stratum,
        max_per_animal=args.max_per_animal,
    )
    logger.info(
        "outcome stratum %s: %d of %d events kept; %d excluded for an unknown "
        "outcome (%d SWR / %d non-SWR)",
        stratum_audit["stratum"],
        stratum_audit["n_events_kept"],
        stratum_audit["n_events_in"],
        stratum_audit["n_events_unknown_outcome"],
        stratum_audit["n_events_unknown_outcome_swr"],
        stratum_audit["n_events_unknown_outcome_non_swr"],
    )

    logger.info(
        "ARM ladder: %d rows, %d events, %d sessions",
        len(ev),
        ev["event_id"].nunique(),
        ev.groupby(SC).ngroups,
    )
    arm, arm_audit = _run(
        ev, ARM_LADDER, response_col=arm_response, n_categories=9, **fit_kwargs
    )
    arm_path = RC / "models" / f"swr_contrast_content_arm{sfx}.pkl"
    arm.to_pickle(arm_path)
    write_fit_provenance(arm_path, fit_inputs, estimator, sys.argv)
    save_reference(arm_z_reference, _z_reference_path(arm_path))

    logger.info(
        "PATCH ladder: %d rows, %d events, %d sessions",
        len(pt),
        pt["event_id"].nunique(),
        pt.groupby(SC).ngroups,
    )
    patch, patch_audit = _run(
        pt, PATCH_LADDER, response_col=patch_response, n_categories=3, **fit_kwargs
    )
    patch_path = RC / "models" / f"swr_contrast_content_patch{sfx}.pkl"
    patch.to_pickle(patch_path)
    write_fit_provenance(patch_path, fit_inputs, estimator, sys.argv)
    save_reference(patch_z_reference, _z_reference_path(patch_path))

    # Cohort-audit table (one row per model x population; reconciles n_in - Σdrops =
    # n_modeled at the event grain) + term-level quasi-separation diagnostics.
    write_cohort_audit(
        arm_audit + patch_audit,
        RC / "models" / f"swr_contrast_content_cohort_audit{sfx}.csv",
    )
    term_diag = pd.concat(
        [
            term_diagnostics(arm, specs_from_ladder(ARM_LADDER, ARM_KEY_TERM)),
            term_diagnostics(patch, specs_from_ladder(PATCH_LADDER, PATCH_KEY_TERM)),
        ],
        ignore_index=True,
    )
    term_diag.to_csv(
        RC / "models" / f"swr_contrast_content_term_diagnostics{sfx}.csv", index=False
    )

    logger.info("saved arm=%d patch=%d rows -> %s", len(arm), len(patch), RC / "models")
    print("\n########## ARM ##########")
    _summ(arm, ARM_LADDER, ARM_KEY_TERM)
    print("\n########## PATCH ##########")
    _summ(patch, PATCH_LADDER, PATCH_KEY_TERM)

    # Audited aggregation: SE, per-session significance, per-animal sign, signed-rank
    # p, BH-FDR across the rung family, and standardized (per-SD) effect sizes — plus a
    # provenance sidecar. The per-SD units are the pooled SDs frozen over the full
    # cohort by prepare_cohorts, so a stratified refit is on the pooled unit.
    agg = pd.concat(
        [
            _aggregate(
                arm,
                ARM_LADDER,
                ARM_KEY_TERM,
                "candidate_arm",
                per_sd_scale["arm_cohort"],
            ),
            _aggregate(
                patch,
                PATCH_LADDER,
                PATCH_KEY_TERM,
                "patch",
                per_sd_scale["patch_cohort"],
            ),
            # The concentration contrast, next to the beta contrast it modifies:
            # under the Dirichlet-multinomial every rung also estimates how much
            # more (or less) an SWR event's mass is spread out, which is the
            # "SWR broadens" claim tested directly rather than read off a
            # coefficient on the mean structure.
            aggregate_contrast(
                arm, _concentration_specs(ARM_LADDER), "candidate_arm"
            ),
            aggregate_contrast(patch, _concentration_specs(PATCH_LADDER), "patch"),
        ],
        ignore_index=True,
    )
    audited_path = RC / "models" / f"swr_contrast_content_audited{sfx}.pkl"
    agg.to_pickle(audited_path)
    write_fit_provenance(audited_path, fit_inputs, estimator, sys.argv)
    prov = _provenance(
        args,
        ev.groupby(SC).ngroups,
        fit_z_reference,
        stratum_audit,
        per_sd_scale,
        _renormalized_rungs(arm, patch),
    )
    (RC / "models" / f"swr_contrast_content_provenance{sfx}.json").write_text(
        json.dumps(prov, indent=2)
    )
    logger.info("saved audited table (%d rows) + provenance -> %s", len(agg), RC / "models")
    print(
        "\n########## AUDITED  (animal-level = primary; session signed-rank = descriptive) ##########"
    )
    cols = [
        "level", "model", "population", "term", "n_sessions", "n_animals",
        "coef_animal_median", "n_animals_pos", "coef_session_median",
        "typical_session_se", "pct_sig_session", "session_signrank_p",
        "session_signrank_fdr_q",
    ]
    with pd.option_context("display.max_rows", None, "display.width", 220):
        print(agg[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
