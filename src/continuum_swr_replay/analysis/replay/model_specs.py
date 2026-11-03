"""Model definitions for replay content analysis.

Single source of truth for every model family, each a dict of
:class:`ModelSpec` (or :class:`PerBinRateModelSpec`):

- ``MODEL_HIERARCHY`` (+ ``_SWR`` / ``_MUA``) — arm-level nested
  hierarchy, conditional Poisson on per-arm timebin counts.
- ``MODEL_PATCH`` (+ ``_SWR`` / ``_MUA`` / ``_SWITCH``) — patch-level
  conditional Poisson (9 arms collapsed to 3 patches).
- ``MODEL_PER_STOP_RATE`` — ordinary Poisson on per-stop event counts
  (different estimator; see its section below).
- ``MODEL_PER_BIN_RATE`` — the per-bin conditional-intensity ladder, with
  the auxiliary per-bin models ``MODEL_UNCERTAINTY_RATE``,
  ``MODEL_CROSS_HISTORY_RATE`` and its duration-aware twin
  ``MODEL_CROSS_HISTORY_DURATION_RATE`` fit by their own runners.
- ``MODEL_CHOICE`` — conditional logit on the animal's next-arm
  choice (different estimator again).

The ``is_swr``-contrast ladders (``SWR_CONTRAST_*``) are defined here too.

Defining constraint for the conditional-Poisson **arm and patch**
hierarchies: only arm-varying predictors are identifiable — event-level
constants (time since stop start, is_swr, etc.) cancel in the within-event
normalization and appear only as interactions with arm-varying predictors.
That rule does NOT apply to ``MODEL_PER_STOP_RATE`` (ordinary Poisson, carries
an intercept) or ``MODEL_CHOICE``; see their own sections.

The design-decision narrative — why each rung carries the predictors it does,
which predictors are excluded and why, and the reversed reasoning and retracted
numbers that are part of the audit trail — lives in
``docs/design/model_history.md``. This module is the machine-readable catalog;
that document is the record of how it got this way. Per-rung reasoning that
sits next to a specific ``ModelSpec`` stays inline below.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from typing import NamedTuple

import pandas as pd

from continuum_swr_replay.data_loaders.constants import N_LEAF_ARMS

from .content_response import renormalize_fraction_over_candidates

# Shared with the covariate builders, which must not import this module: see
# ``design_constants`` for why the numbers live there and not here. Re-exported
# below so ``from ...model_specs import BIN_S`` keeps working.
from .design_constants import (
    ARM_FE_PREDICTORS,
    ARRIVAL_CLOCK_COLUMN,
    ARRIVAL_SPLINE_KNOTS,
    BIN_S,
    N_CROSS_LONG_BASIS,
    N_CROSS_LONG_LAGS,
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    N_REWARD_LAGS,
    RECENCY_SPLINE_BOUNDS,
    RECENCY_SPLINE_COLUMNS,
    RECENCY_SPLINE_KNOTS,
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_INCOMPLETE,
    REWARD_HISTORY_N_BASIS,
    REWARD_HISTORY_N_LAGS,
    SPLINE_KNOTS,
    TMAX,
    TMAX_ARRIVAL,
)

__all__ = [
    "ARM_FE_PREDICTORS",
    "ARRIVAL_CLOCK_COLUMN",
    "ARRIVAL_SPLINE_KNOTS",
    "BIN_S",
    "BELIEF_DEPENDENT_CONTENT_MODELS",
    "MODEL_CHOICE",
    "MODEL_CROSS_HISTORY_DURATION_RATE",
    "MODEL_CROSS_HISTORY_LONG_RATE",
    "MODEL_CROSS_HISTORY_RATE",
    "MODEL_HIERARCHY",
    "MODEL_HIERARCHY_MUA",
    "MODEL_HIERARCHY_SWR",
    "MODEL_PATCH",
    "MODEL_PATCH_MUA",
    "MODEL_PATCH_SWITCH",
    "MODEL_PATCH_SWR",
    "MODEL_PER_BIN_RATE",
    "MODEL_PER_STOP_RATE",
    "MODEL_UNCERTAINTY_RATE",
    "MU_UNCERTAINTY_COVARIATES",
    "MU_UNCERTAINTY_CURATED_COVARIATES",
    "MU_UNCERTAINTY_MODEL_VALUE_COVARIATES",
    "MU_UNCERTAINTY_RL_COVARIATES",
    "ModelSpec",
    "N_HISTORY_BASIS",
    "N_HISTORY_LAGS",
    "N_REWARD_LAGS",
    "PER_BIN_DESIGN",
    "PER_BIN_HISTORY_ANCHORS",
    "PER_BIN_HISTORY_COLUMNS",
    "POST_OUTCOME_COHORT",
    "POST_OUTCOME_COVARIATES",
    "POST_OUTCOME_RATE_RUNGS",
    "PRIMARY_COHORT",
    "PerBinRateModelSpec",
    "PerBinTerm",
    "RECENCY_SPLINE_BOUNDS",
    "RECENCY_SPLINE_COLUMNS",
    "RECENCY_SPLINE_KNOTS",
    "REWARD_HISTORY_COLUMNS",
    "REWARD_HISTORY_INCOMPLETE",
    "REWARD_HISTORY_N_BASIS",
    "REWARD_HISTORY_N_LAGS",
    "SPLINE_KNOTS",
    "STOP_COVARIATES",
    "SWR_CONTRAST_CONTENT_ARM_KEY_TERM",
    "SWR_CONTRAST_CONTENT_ARM_LADDER",
    "SWR_CONTRAST_CONTENT_BASEC",
    "SWR_CONTRAST_CONTENT_BASEC_NT",
    "SWR_CONTRAST_CONTENT_BASEC_PATCH",
    "SWR_CONTRAST_CONTENT_BASEC_PATCH_NT",
    "SWR_CONTRAST_CONTENT_PATCH_KEY_TERM",
    "SWR_CONTRAST_CONTENT_PATCH_LADDER",
    "SWR_CONTRAST_RATE_G_SPECS",
    "SWR_CONTRAST_RATE_S_SPECS",
    "SWR_CONTRAST_RATE_STOP_COVARS",
    "TMAX",
    "TMAX_ARRIVAL",
    "default_content_model_names",
    "longest_history_window_s",
]


@dataclass(frozen=True)
class ModelSpec:
    """Specification for one model in the hierarchy.

    Attributes
    ----------
    name : str
        Short identifier (e.g., "M0", "M1").
    predictors : tuple[str, ...]
        Column names to use as fixed-effect predictors. Immutable tuple
        for true frozen dataclass safety.
    description : str
        Human-readable description of what the model tests.
    parent : str | None
        Name of the parent model for nested comparison. None for M0.
    """

    name: str
    predictors: tuple[str, ...]
    description: str
    parent: str | None


# These rungs depend on the external contingency/qvals artifact. They must not
# enter the default base-ladder union because unmapped stops would then shrink
# otherwise qvals-independent M0-M6/P0-P4 fits. Explicit belief comparisons use
# their own matched cohort instead.
BELIEF_DEPENDENT_CONTENT_MODELS = frozenset({"Mv", "Mu", "Mvu", "Pv"})


def default_content_model_names(
    hierarchy: Mapping[str, ModelSpec],
) -> tuple[str, ...]:
    """Canonical qvals-independent model names for a default content fit."""
    return tuple(
        name for name in hierarchy if name not in BELIEF_DEPENDENT_CONTENT_MODELS
    )


# ── Primary arm-level hierarchy ──
#
# M0a (arm fixed effects) is the baseline below M1, and every rung from M1 on
# carries the 8 arm-FE dummies so the spatial / reward predictors are estimated
# above each arm's static occupancy. M5 adds the spatial × SWR interactions, M6
# the value × SWR ones. There is no reward-memory rung, and MUA event-type
# effects are assessed by splitting the cohort rather than by interaction terms.
# The last-rewarded family is REFUSED as unidentified — the pooled contrast
# rungs (C5_lastrew, CS3_lastrew, CS5b_valuevsprevgoal) did not rescue it
# either, and no coefficient from them is citable. Why (incl. the arm-FE
# convergence trade-off, the retired M5 rung and the refusal):
# ``docs/design/model_history.md`` § "Primary arm-level hierarchy".


_M1 = (*ARM_FE_PREDICTORS, "is_current_arm")
_M2 = (*_M1, "is_prev_arm")
# An unvisited leaf has ``reward_fraction_candidate_arm_last10_visits == 0``
# (encoded, not NaN) so its event is retained. M4 adds
# ``is_candidate_arm_never_visited`` alongside time-local recency,
# separating novelty from an observed zero reward rate while preserving nesting.
_M3 = (*_M2, "reward_fraction_candidate_arm_last10_visits")
_M4 = (*_M3, *RECENCY_SPLINE_COLUMNS, "is_candidate_arm_never_visited")
_M5 = (*_M4, "swr_x_is_current_arm", "swr_x_is_prev_arm")
_M6 = (
    *_M5,
    "swr_x_reward_fraction_candidate_arm_last10_visits",
    *(f"swr_x_{_c}" for _c in RECENCY_SPLINE_COLUMNS),
)
# Value block: candidate-arm task reward probability + candidate-arm
# beta-model reward-belief mean as conditional main effects on the M4
# spatial/recency base (covariate_dictionary.md D2/D3). The beta belief mean is the
# single value construct shared with the is_swr rate ladder's beta value covariates.
# Additive in *spec* — but because both value columns are NaN for stops absent
# from qvals, co-fitting Mv in the same
# run restricts the shared complete-case event set, so it is excluded from the default
# model list along with the other belief-dependent rungs (see
# BELIEF_DEPENDENT_CONTENT_MODELS). Request them explicitly on their own matched
# cohort: --models M4,Mv,Mu,Mvu (arm) or --models P3,Pv (patch).
_Mv = (
    *_M4,
    "task_reward_probability_candidate_arm",
    "reward_belief_mean_candidate_arm_preoutcome",
)

# Beta-belief exploration block: candidate-arm reward-belief variance added on
# the M4 spatial/recency base. Mvu pairs it with the candidate-arm belief mean,
# so value+uncertainty come from the same Beta posterior — a construct-coherent
# exploit-vs-explore contrast (one behavioral model). Variance is added after
# centered candidate-arm visit recency (M4) so its coefficient is the
# recency-independent effect (docs: Beta variance is +0.54 correlated with
# visit recency by construction, but the effect
# survives orthogonalization). See docs/behavior_models.md.
# Standardized as ``reward_belief_variance_candidate_arm_preoutcome_z``: raw
# Beta variance has SD ~0.02, ~5-6x smaller than the belief mean, which makes
# the per-session covariance singular; the _z form uses one
# pooled SD over the six reward-site arms (not structural-zero trunks).
_Mu = (*_M4, "reward_belief_variance_candidate_arm_preoutcome_z")
_Mvu = (
    *_M4,
    "reward_belief_mean_candidate_arm_preoutcome",
    "reward_belief_variance_candidate_arm_preoutcome_z",
)

MODEL_HIERARCHY: dict[str, ModelSpec] = {
    "M0": ModelSpec(
        name="M0",
        predictors=(),
        description="Null (uniform allocation across arms)",
        parent=None,
    ),
    "M0a": ModelSpec(
        name="M0a",
        predictors=ARM_FE_PREDICTORS,
        description="Per-arm geometry baseline (arm fixed effects)",
        parent="M0",
    ),
    "M1": ModelSpec(
        name="M1",
        predictors=_M1,
        description="+ current-arm bias",
        parent="M0a",
    ),
    "M2": ModelSpec(
        name="M2",
        predictors=_M2,
        description="+ previous-arm bias (spatial context)",
        parent="M1",
    ),
    "M3": ModelSpec(
        name="M3",
        predictors=_M3,
        description="+ recent reward rate",
        parent="M2",
    ),
    "M4": ModelSpec(
        name="M4",
        predictors=_M4,
        description=(
            "+ visit recency (centered across visited leaf arms within event) "
            "and never-visited novelty"
        ),
        parent="M3",
    ),
    "M5": ModelSpec(
        name="M5",
        predictors=_M5,
        description="+ SWR × spatial interactions",
        parent="M4",
    ),
    "M6": ModelSpec(
        name="M6",
        predictors=_M6,
        description="+ SWR × value/memory interactions",
        parent="M5",
    ),
    "Mv": ModelSpec(
        name="Mv",
        predictors=_Mv,
        description=(
            "+ candidate-arm task reward probability + beta reward-belief mean"
        ),
        parent="M4",
    ),
    "Mu": ModelSpec(
        name="Mu",
        predictors=_Mu,
        description="+ candidate-arm beta reward-belief variance (exploration)",
        parent="M4",
    ),
    "Mvu": ModelSpec(
        name="Mvu",
        predictors=_Mvu,
        description=(
            "+ candidate-arm beta reward-belief mean AND variance "
            "(construct-coherent exploit-vs-explore)"
        ),
        parent="Mu",
    ),
}


# ── Patch-level models ──
#
# Collapse 9 arms to 3 patches (sum arm_count_hard across arms within
# each patch). Eliminates trunk/leaf distinction and provides a cleaner
# test of the value/exploration question. The response is patch-level
# count; the conditional Poisson conditions on the event total as before
# but with 3 candidate patches instead of 9 arms.
#
# Predictors use ``_patch`` suffix to distinguish from the arm-level
# versions — e.g., ``is_current_patch`` (= is the animal in this
# patch?) and ``reward_fraction_candidate_arm_last10_visits_patch``
# (= mean recent reward fraction
# across the two reward-site arms in the patch). This mirrors the arm
# ladder's rate definition and avoids reintroducing a visit-frequency
# confound through an accumulated reward count.
#
# There is no patch reward-memory rung, for the reason the arm ladder has none
# (removed 2026-08-18, P3/P4 renumbered down into P2/P3). ``MODEL_PATCH_SWITCH``
# remains the dedicated previous-vs-current contrast. Numbers:
# ``docs/design/model_history.md`` § "Patch ladder: the retired patch
# reward-memory rung".

# ── Event-source-restricted hierarchies (ContFrag pipeline) ──
#
# When events are externally restricted to SWR or MUA windows, the
# corresponding flag is constant by construction (every SWR-event has
# is_swr == 1; every MUA-event has is_mua == 1) so that flag's
# interactions with arm-varying predictors collapse onto the main
# effects and are unidentifiable. The hierarchy therefore mirrors
# MODEL_HIERARCHY's M0–M4 and stops there.
#
# A cross-modality rung is deferred; see ``docs/design/model_history.md``
# § "Event-source-restricted hierarchies".

_M0_TO_M4 = {
    name: MODEL_HIERARCHY[name] for name in ("M0", "M0a", "M1", "M2", "M3", "M4")
}

MODEL_HIERARCHY_SWR: dict[str, ModelSpec] = dict(_M0_TO_M4)
MODEL_HIERARCHY_MUA: dict[str, ModelSpec] = dict(_M0_TO_M4)


MODEL_PATCH: dict[str, ModelSpec] = {
    "P0": ModelSpec(
        name="P0",
        predictors=(),
        description="Null (uniform across patches)",
        parent=None,
    ),
    "P1": ModelSpec(
        name="P1",
        predictors=("is_current_patch",),
        description="Current-patch bias",
        parent="P0",
    ),
    "P2": ModelSpec(
        name="P2",
        predictors=(
            "is_current_patch",
            "reward_fraction_candidate_arm_last10_visits_patch",
        ),
        description="+ patch aggregate of candidate-arm recent reward fraction",
        parent="P1",
    ),
    "P3": ModelSpec(
        name="P3",
        predictors=(
            "is_current_patch",
            "reward_fraction_candidate_arm_last10_visits_patch",
            "swr_x_is_current_patch",
        ),
        description="+ SWR × current-patch interaction",
        parent="P2",
    ),
    "Pv": ModelSpec(
        name="Pv",
        predictors=(
            "is_current_patch",
            "reward_fraction_candidate_arm_last10_visits_patch",
            "task_reward_probability_candidate_arm_patch",
            "reward_belief_mean_candidate_arm_preoutcome_patch",
        ),
        description="+ patch aggregates of task reward probability and "
        "pre-outcome reward-belief mean",
        parent="P2",
    ),
}


# ── Event-source-restricted patch hierarchies (ContFrag pipeline) ──
#
# Mirror MODEL_PATCH's P0–P2 but drop P3: under ContFrag event-source
# restriction is_swr is constant, so the swr_x_* interactions collapse
# onto the main effects and are unidentifiable — same reason the arm
# hierarchy SWR/MUA versions stop at M4.

_P0_TO_P2 = {name: MODEL_PATCH[name] for name in ("P0", "P1", "P2")}
MODEL_PATCH_SWR: dict[str, ModelSpec] = dict(_P0_TO_P2)
MODEL_PATCH_MUA: dict[str, ModelSpec] = dict(_P0_TO_P2)

# ── Post-switch patch model ──
#
# Fit ONLY on events where ``did_last_completed_trip_switch_patch == 1``. On switch trials
# current_patch ≠ prev_patch, so they are identifiable (correlation
# ≈ -0.5, VIF < 2). On stay trials these are collinear (r = 0.95,
# VIF > 20) and must not be combined.
#
# Pilot result on 7 j16 sessions: ``docs/design/model_history.md``
# § "Post-switch patch model".

MODEL_PATCH_SWITCH: dict[str, ModelSpec] = {
    "PS0": ModelSpec(
        name="PS0",
        predictors=(),
        description="Null (switch events only)",
        parent=None,
    ),
    "PS1": ModelSpec(
        name="PS1",
        predictors=("is_current_patch",),
        description="Current-patch bias (switch events only)",
        parent="PS0",
    ),
    "PS2": ModelSpec(
        name="PS2",
        predictors=("is_current_patch", "is_prev_patch"),
        description="+ previous-patch suppression",
        parent="PS1",
    ),
    "PS3": ModelSpec(
        name="PS3",
        predictors=(
            "is_current_patch",
            "is_prev_patch",
            "swr_x_is_current_patch",
            "swr_x_is_prev_patch",
        ),
        description="+ SWR × patch interactions (switch events only)",
        parent="PS2",
    ),
}


# ── Rate models ──
#
# A DIFFERENT estimator from the arm/patch hierarchies: an ordinary
# Poisson on the selected per-stop event count (``n_selected_events``) with a
# ``log(immobility_duration_s)`` offset, fit by
# ``per_stop_rate_model.fit_per_stop_rate_model``.
# It carries an intercept, so — unlike the conditional Poisson —
# stop-level constants do NOT cancel and ``is_after_*`` are valid
# main effects.
#
# ``predictors=()`` for R0 means intercept-only (the offset is always
# included by the fitter). Current-outcome indicators are intentionally absent
# from R1: they are post-outcome quantities and do not have a stable
# trial-relative relationship to the bout-start reward-history covariates.
#
# R1 carries the whole reward history in three numbers: the known-outcome lag
# vector projected onto the log-spaced raised-cosine basis
# (``REWARD_HISTORY_COLUMNS``) plus the structural-missingness indicator
# ``REWARD_HISTORY_INCOMPLETE``. The ladder stops there. Why — the content
# ladder's M3 analogy, the dropped ``is_swr_stop`` term, and the deferred
# block-level rung: ``docs/design/model_history.md`` § "Per-stop rate ladder".

MODEL_PER_STOP_RATE: dict[str, ModelSpec] = {
    "R0": ModelSpec(
        name="R0",
        predictors=(),
        description="Null (intercept-only baseline event rate)",
        parent=None,
    ),
    "R1": ModelSpec(
        name="R1",
        # The reward-history basis is closed at the **outcome** clock: the
        # lag vector holds outcomes already observable at bout start, so lag 1
        # can be the arrived trip's outcome after its physical feedback
        # boundary or the preceding known outcome before it. Past-only, but not
        # a fixed trial-relative index.
        #
        # ``reward_history_incomplete`` is the structural-missingness nuisance
        # main effect for stops whose slowest bump has no known outcome in its
        # support (fewer than two known outcomes, features zero-filled). It is
        # fit, never reported.
        #
        # The immediate-switch flag was removed from this general-purpose
        # reward-history ladder: it is often constant within a session's
        # complete-case cohort, which makes the design rank-deficient and
        # rejects the whole session. See the naming spec's "Immediate switch
        # is not a nuisance adjustment" section; a dedicated switch analysis
        # uses ``patch_switch_fraction_last5_completed_trips`` instead.
        predictors=(*REWARD_HISTORY_COLUMNS, REWARD_HISTORY_INCOMPLETE),
        description="+ reward history (raised-cosine basis over the last 10 known outcomes)",
        parent="R0",
    ),
}


# ── Per-bin conditional-intensity ladder (Family B) ──
#
# A point-process GLM conditional-intensity ladder (Truccolo-Eden-Brown 2005):
#
#     log lambda = baseline + f(time-since-stop) + g(self-history)
#                  + sum(beta * covariates)
#
# fit on the canonical (stop x bin) design from
# :func:`per_bin_design.aggregate_to_timebins`. Unlike ``MODEL_PER_STOP_RATE`` (one
# row per stop, ``log(immobility_duration_s)`` dwell offset), every model here is one
# row per (stop, ``BIN_S``-wide bin) with an equal-exposure ``log(exposure)``
# offset, so the within-stop time course is modelled explicitly by a spline
# rather than averaged into a per-stop dwell correction.
#
# The three blocks (time-since-stop spline, self-history/autoregressive,
# reward-history basis) toggle independently so each block's contribution is
# readable with and without the time course, plus the reward x time
# interaction. ALL models share one bin width ``BIN_S`` so the
# ladder is nested and AIC / CV / deviance / GOF are directly comparable (fine
# bins are required anyway for the self-history term and time-rescaling GOF).
# The bin width, windows, knots and clock column are in ``design_constants``,
# imported above; the two spline formula strings built from them are here
# because only the ladder uses them.
_SPLINE = f"bs(tc, knots={SPLINE_KNOTS}, lower_bound=0.0, upper_bound={TMAX})"
_ARRIVAL_SPLINE = (
    f"bs({ARRIVAL_CLOCK_COLUMN}, knots={ARRIVAL_SPLINE_KNOTS}, "
    f"lower_bound=0.0, upper_bound={TMAX_ARRIVAL})"
)


class PerBinTerm(NamedTuple):
    """One additive term of a per-bin formula and the design columns it reads.

    A per-bin rung is a list of these: its patsy formula joins the ``term``
    strings and its design columns are the ``columns`` in order, so the two
    cannot disagree.

    Attributes
    ----------
    term : str
        The term as written in the patsy formula.
    columns : tuple of str
        The design columns the term reads (a spline reads its clock column).
    history_lags : int
        Bins of event history (the stop's own events, or a cross stream's) the
        term looks back over; 0 for a term that reads no event history. Sets
        the ladder's history window.
    """

    term: str
    columns: tuple[str, ...]
    history_lags: int = 0


def _spline_term(time_clock: str) -> PerBinTerm:
    """The time-course spline on the stop clock (``tc``) or the arrival clock."""
    if time_clock == "arrival":
        return PerBinTerm(_ARRIVAL_SPLINE, (ARRIVAL_CLOCK_COLUMN,))
    return PerBinTerm(_SPLINE, ("tc",))


def _column_terms(columns: Iterable[str], history_lags: int = 0) -> list[PerBinTerm]:
    """One plain linear term per design column."""
    return [PerBinTerm(column, (column,), history_lags) for column in columns]


def _interaction(*factors: PerBinTerm) -> PerBinTerm:
    """The patsy ``a:b`` interaction of ``factors``; reads all their columns."""
    columns = tuple(dict.fromkeys(c for factor in factors for c in factor.columns))
    return PerBinTerm(":".join(factor.term for factor in factors), columns)


def _reward_basis_terms(n_basis: int) -> list[PerBinTerm]:
    """The first ``n_basis`` reward-history features plus the missingness flag."""
    return _column_terms((*REWARD_HISTORY_COLUMNS[:n_basis], REWARD_HISTORY_INCOMPLETE))


def _history_terms(prefix: str, n_lags: int, n_basis: int) -> list[PerBinTerm]:
    """An event-history block over ``n_lags`` bins.

    ``{prefix}hist_rcos_{k}`` raised-cosine features when ``n_basis > 0``, else
    the per-lag ``onset_lag{j}`` (self) / ``cross_lag{j}`` (cross) dummies, or
    their event-end twins ``offset_lag{j}`` / ``cross_offset_lag{j}``.
    """
    if n_basis:
        columns = [f"{prefix}hist_rcos_{k}" for k in range(n_basis)]
    else:
        lag = _HISTORY_LAG_PREFIX[prefix]
        columns = [f"{lag}{j}" for j in range(1, n_lags + 1)]
    return _column_terms(columns, n_lags)


# Per-lag dummy names of each history block, keyed by its basis prefix: onsets
# of the stop's own stream (""), of the cross stream ("cross_"), and the
# duration-aware blocks on event ENDS ("off", "cross_off").
_HISTORY_LAG_PREFIX = {
    "": "onset_lag",
    "cross_": "cross_lag",
    "off": "offset_lag",
    "cross_off": "cross_offset_lag",
}
# History-block prefix of each anchor (``PerBinRateModelSpec.*_history_anchors``).
_ANCHOR_PREFIX = {"onset": "", "offset": "off"}


# Per-stop covariates the shipped ladder needs beyond the reward-history basis
# and the self-history lags — carried into the per-bin design by
# aggregate_to_timebins. Shared by the fitter orchestrator and the
# model-checking plotter so the inline refit matches the persisted fits. Empty:
# the reward-history basis is requested by ``reward_basis``, not as a covariate,
# and the immediate-switch term was retired with the M8 rung (see R1).
STOP_COVARIATES: tuple[str, ...] = ()

# The per-bin rate ladders (``MODEL_PER_BIN_RATE``, ``MODEL_UNCERTAINTY_RATE``
# and the is_swr-contrast ``BASE`` block) are fit on the at-risk design: a bin
# inside an ongoing event of the outcome's detector is not at risk of an onset
# and is dropped (``aggregate_to_timebins(at_risk_events=...)``, built by
# ``duration_aware_history.at_risk_events``). Their self-history control is
# counted from event ENDS. Counted from onsets it is not estimable there: the
# bins a few lags after an onset are mostly inside that event, so the
# short-lag terms have almost no at-risk bins (28 % of the per-session
# duration-aware fits with onset history did not converge, against 5 % with
# end history). On the old all-bins design the onset block ran to -inf
# instead, because the bin after an onset never holds one. The onset-locked
# rhythm is read from the pooled duration-aware history fits, not here.
PER_BIN_HISTORY_ANCHORS: tuple[str, ...] = ("offset",)
#: Written on every per-bin rate fit row; readers refuse rows without it (fits
#: from the old all-bins, onset-history design).
PER_BIN_DESIGN = "at_risk_end_history"
#: The self-history basis columns those ladders fit.
PER_BIN_HISTORY_COLUMNS: tuple[str, ...] = tuple(
    f"{_ANCHOR_PREFIX[anchor]}hist_rcos_{k}"
    for anchor in PER_BIN_HISTORY_ANCHORS
    for k in range(N_HISTORY_BASIS)
)


@dataclass(frozen=True)
class PerBinRateModelSpec:
    """One model in the per-bin conditional-intensity ladder.

    Parallel to :class:`ModelSpec` (kept separate so the simpler arm/patch/
    per-stop families are unaffected). The toggle fields drive both
    :func:`per_bin_design.aggregate_to_timebins` (which columns to build) and
    :meth:`terms`, the one list of (term, columns) that :meth:`formula` (the
    patsy formula), :meth:`design_columns` and :meth:`history_lags` are read
    from, so the spec is the single source of truth for each model.

    Attributes
    ----------
    name : str
        Short identifier (e.g. ``"M0_const"``).
    time_spline : bool
        Include the fixed-knot ``bs(tc, ...)`` within-stop time-course spline.
    reward_basis : int
        Number of reward-history raised-cosine features (0 or
        ``REWARD_HISTORY_N_BASIS``). When > 0 the rung carries
        ``REWARD_HISTORY_COLUMNS`` plus the structural-missingness nuisance
        main effect ``REWARD_HISTORY_INCOMPLETE``. Features are weighted
        fractions of past known outcomes; omission == ``1 - reward``.
    reward_rate : bool
        Include ``reward_fraction_last10_known_trials`` (continuous). No shipped
        rung sets it; the column must then be supplied via ``stop_covariates``.
    self_history_lags : int
        Number of ``onset_lag{j}`` terms (0 or ``N_HISTORY_LAGS``); the
        within-stop autoregressive / refractoriness block.
    switch : bool
        Retired and must stay ``False``. The per-bin switch main-effect
        extension (M8) was removed because
        ``did_last_completed_trip_switch_patch`` is often
        constant within a session's complete-case cohort, which makes the
        design rank-deficient. No shipped spec sets this, and ``formula`` no
        longer reads it.
    reward_time_interaction : bool
        Include ``reward_history_rcos_0 : bs(tc, ...)`` — does the immediate
        reward-history component modulate the time course?
    description : str
        Human-readable description.
    parent : str | None
        Name of the parent model for nested comparison.
    """

    name: str
    time_spline: bool
    reward_basis: int
    reward_rate: bool
    self_history_lags: int
    switch: bool
    reward_time_interaction: bool
    description: str
    parent: str | None
    # If > 0, the self-history block uses a raised-cosine basis
    # (``hist_rcos_0..k-1``) over the ``self_history_lags`` window instead of the
    # per-lag ``onset_lag`` dummies -- the smooth basis cannot separate on the
    # refractory bins, so it generalizes where the dummies overfit. Default 0
    # keeps the dummy encoding (and the per-lag kernel-shape readout).
    self_history_basis: int = 0
    # Cross-population history: lagged onset counts of a *different* event stream
    # (the ``cross_onset_times`` passed to ``aggregate_to_timebins``), so one
    # population's conditional intensity can depend on another's recent past --
    # e.g. does the non-SWR non-local rate depend on recent with-SWR non-local
    # events (a bivariate/Hawkes coupling)? ``cross_history_lags`` adds
    # ``cross_lag{j}`` dummies; ``cross_history_basis > 0`` instead uses the
    # raised-cosine ``cross_hist_rcos_{k}`` projection over that window. Default 0
    # leaves the shipped ladder unchanged.
    cross_history_lags: int = 0
    cross_history_basis: int = 0
    # Duration-aware history: which past-event times each history block is
    # measured from. ``"onset"`` is the shipped block (``hist_rcos_k`` /
    # ``cross_hist_rcos_k``); ``"offset"`` adds the same lags/basis counted
    # from event ENDS (``offhist_rcos_k`` / ``cross_offhist_rcos_k``, or the
    # ``offset_lag{j}`` / ``cross_offset_lag{j}`` dummies), so a dependence
    # locked to the previous event's end is not read as one locked to its
    # onset. An offset block's design is built with
    # ``aggregate_to_timebins(offset_history=True, at_risk_events=...)``. The
    # default ``("onset",)`` leaves every shipped rung unchanged.
    self_history_anchors: tuple[str, ...] = ("onset",)
    cross_history_anchors: tuple[str, ...] = ("onset",)
    # Extra per-stop covariates appended to the formula as plain linear terms
    # (e.g. the ``MU_uncertainty`` uncertainty/value block). Default
    # ``()`` so the shipped ``MODEL_PER_BIN_RATE`` ladder is byte-identical and
    # the persisted ``all_models.pkl`` schema is unchanged. These names must be
    # carried into the per-bin design via ``aggregate_to_timebins``'
    # ``stop_covariates``.
    extra_covariates: tuple[str, ...] = ()
    # Which clock the time-course spline is anchored to. ``"stop"`` uses ``tc``
    # (time since the bout began) over ``[0, TMAX]``; ``"arrival"`` uses
    # ``ARRIVAL_CLOCK_COLUMN`` over ``[0, TMAX_ARRIVAL]``, and the runner
    # additionally restricts an arrival rung to that window AFTER the ladder's
    # common-cohort freeze. Rungs on different clocks are fit to different rows
    # and are therefore never compared numerically with each other (the clock
    # travels with every fitted row as ``time_clock``, and
    # ``model_comparison.compare_nested_models`` invalidates a cross-clock
    # pair); an arrival rung's parent must itself be an arrival rung.
    time_clock: str = "stop"

    def __post_init__(self) -> None:
        if self.time_clock not in ("stop", "arrival"):
            raise ValueError(
                "PerBinRateModelSpec.time_clock must be 'stop' or 'arrival'; "
                f"got {self.time_clock!r}."
            )
        for field_name in ("self_history_anchors", "cross_history_anchors"):
            anchors = getattr(self, field_name)
            if (
                not anchors
                or set(anchors) - set(_ANCHOR_PREFIX)
                or len(set(anchors)) != len(anchors)
            ):
                raise ValueError(
                    f"PerBinRateModelSpec.{field_name} must be distinct values "
                    f"of {tuple(_ANCHOR_PREFIX)}; got {anchors!r}."
                )
        if self.reward_basis not in (0, REWARD_HISTORY_N_BASIS):
            raise ValueError(
                "PerBinRateModelSpec.reward_basis must be 0 or "
                f"{REWARD_HISTORY_N_BASIS} (the shipped reward-history basis "
                f"width); got {self.reward_basis}."
            )
        if self.switch:
            raise ValueError(
                "PerBinRateModelSpec.switch is retired: the per-bin switch "
                "main effect (M8) was removed because "
                "did_last_completed_trip_switch_patch is often "
                "constant within a session's complete-case cohort. Do not "
                "re-enable it; use a dedicated switch analysis instead."
            )

    def terms(self) -> list[PerBinTerm]:
        """The rung's formula terms, in formula order, each with its columns.

        The one statement of what a rung fits: :meth:`formula`,
        :meth:`design_columns` and :meth:`history_lags` all read it.
        """
        spline = _spline_term(self.time_clock)
        terms: list[PerBinTerm] = []
        if self.time_spline:
            terms.append(spline)
        if self.reward_basis:
            terms += _reward_basis_terms(self.reward_basis)
        if self.reward_rate:
            terms += _column_terms(("reward_fraction_last10_known_trials",))
        for anchor in self.self_history_anchors:
            terms += _history_terms(
                _ANCHOR_PREFIX[anchor],
                self.self_history_lags,
                self.self_history_basis,
            )
        for anchor in self.cross_history_anchors:
            terms += _history_terms(
                "cross_" + _ANCHOR_PREFIX[anchor],
                self.cross_history_lags,
                self.cross_history_basis,
            )
        # ``switch`` is retired (see the field docstring); the completed-trip
        # switch flag is never added to a per-bin model.
        if self.reward_time_interaction:
            (immediate,) = _column_terms(REWARD_HISTORY_COLUMNS[:1])
            terms.append(_interaction(immediate, spline))
        terms += _column_terms(self.extra_covariates)
        return terms

    def formula(self) -> str:
        """patsy formula (response ``count``; the offset is applied at fit time)."""
        terms = [term.term for term in self.terms()]
        return "count ~ " + (" + ".join(terms) if terms else "1")

    def design_columns(self) -> list[str]:
        """Columns the rung's formula references (besides the offset's ``exposure``).

        ``count``, then (for an arrival rung) the arrival clock, then each
        term's columns in formula order, without repeats.
        """
        cols = ["count"]
        if self.time_clock == "arrival":
            # An arrival rung lives on the arrival clock whether or not it
            # splines it: the runner windows it on that column, so a bin without
            # one is not in the rung's support and must leave the ladder's
            # common cohort.
            cols.append(ARRIVAL_CLOCK_COLUMN)
        cols += [column for term in self.terms() for column in term.columns]
        return list(dict.fromkeys(cols))

    def history_lags(self) -> int:
        """Bins of event history the rung's longest-reaching term looks back."""
        return max((term.history_lags for term in self.terms()), default=0)


_NB = REWARD_HISTORY_N_BASIS
# Each spec is built with keyword arguments so the toggle each model turns on is
# readable in place (which term the rung adds vs. its parent), rather than being
# decoded from argument position. ``switch`` is retired and stays False.
MODEL_PER_BIN_RATE: dict[str, PerBinRateModelSpec] = {
    "M0_const": PerBinRateModelSpec(
        name="M0_const",
        time_spline=False,
        reward_basis=0,
        reward_rate=False,
        self_history_lags=0,
        switch=False,
        reward_time_interaction=False,
        description="homogeneous Poisson baseline",
        parent=None,
    ),
    "M1_rew": PerBinRateModelSpec(
        name="M1_rew",
        time_spline=False,
        reward_basis=_NB,
        reward_rate=False,
        self_history_lags=0,
        switch=False,
        reward_time_interaction=False,
        description="+ reward history (raised-cosine basis over the last 10 known outcomes)",
        parent="M0_const",
    ),
    # The five ``onset_lag1..10`` dummy rungs (M3_hist, M4_hist_rew,
    # M6_time_hist, M7_time_hist_rew, M9_rew_x_time) are RETIRED — they could
    # not be cross-validated at all. The raised-cosine variants below span the
    # same window and ARE scored; the reward x time interaction survives as
    # ``M9r_rew_x_time``. Numbers: ``docs/design/model_history.md``
    # § "Retirements in the per-bin rate ladder (Family B)".
    "M5_time": PerBinRateModelSpec(
        name="M5_time",
        time_spline=True,
        reward_basis=0,
        reward_rate=False,
        self_history_lags=0,
        switch=False,
        reward_time_interaction=False,
        description="+ time-since-stop (spline)",
        parent="M0_const",
    ),
    # M8_full_switch (the per-bin switch main-effect extension) was retired;
    # nothing parents off it. Why: ``docs/design/model_history.md``
    # § "Per-bin rate ladder: the retired M8 switch rung (moved 2026-09-06)".
    # Raised-cosine self-history variants: a smooth basis over the same window
    # (it lifted held-out cv_bits_per_event from negative to positive vs the
    # onset_lag dummies), counted from event ends on the at-risk design
    # (``PER_BIN_HISTORY_ANCHORS``).
    "M3r_histcos": PerBinRateModelSpec(
        name="M3r_histcos",
        time_spline=False,
        reward_basis=0,
        reward_rate=False,
        self_history_lags=N_HISTORY_LAGS,
        switch=False,
        reward_time_interaction=False,
        description="+ self-history from event ends (raised-cosine basis)",
        parent="M0_const",
        self_history_basis=N_HISTORY_BASIS,
        self_history_anchors=PER_BIN_HISTORY_ANCHORS,
    ),
    # Time + self-history without the reward block, so the reward block has a
    # nested null that already holds time AND self-history: the "reward given
    # time + self-history" step is M7r - M6r, not M7r - M5_time (which mixes
    # the reward block in with the self-history block).
    "M6r_time_histcos": PerBinRateModelSpec(
        name="M6r_time_histcos",
        time_spline=True,
        reward_basis=0,
        reward_rate=False,
        self_history_lags=N_HISTORY_LAGS,
        switch=False,
        reward_time_interaction=False,
        description="+ time + raised-cosine self-history from event ends",
        parent="M5_time",
        self_history_basis=N_HISTORY_BASIS,
        self_history_anchors=PER_BIN_HISTORY_ANCHORS,
    ),
    "M7r_time_histcos_rew": PerBinRateModelSpec(
        name="M7r_time_histcos_rew",
        time_spline=True,
        reward_basis=_NB,
        reward_rate=False,
        self_history_lags=N_HISTORY_LAGS,
        switch=False,
        reward_time_interaction=False,
        description="+ time + raised-cosine self-history from event ends + reward history",
        parent="M6r_time_histcos",
        self_history_basis=N_HISTORY_BASIS,
        self_history_anchors=PER_BIN_HISTORY_ANCHORS,
    ),
    "M9r_rew_x_time": PerBinRateModelSpec(
        name="M9r_rew_x_time",
        time_spline=True,
        reward_basis=_NB,
        reward_rate=False,
        self_history_lags=N_HISTORY_LAGS,
        switch=False,
        reward_time_interaction=True,
        description="M7r + reward x time interaction",
        parent="M7r_time_histcos_rew",
        self_history_basis=N_HISTORY_BASIS,
        self_history_anchors=PER_BIN_HISTORY_ANCHORS,
    ),
    # Arrival-anchored sub-ladder. These rungs re-anchor the time-course block
    # on ``ARRIVAL_CLOCK_COLUMN`` and are fit on the arrival window
    # ``[0, TMAX_ARRIVAL]``, a strict subset of the frozen cohort's rows. Their
    # own null therefore cannot be ``M0_const`` (fit on the whole frozen
    # cohort): AIC / deviance / held-out likelihood all scale with the number of
    # rows, so differencing across the two row sets would measure the window,
    # not the clock. ``M0a_const_arrival`` is that null -- the same homogeneous
    # Poisson formula as ``M0_const``, refit on the arrival window -- persisted
    # as a real rung so every rung reports an ``n_bins`` equal to every other
    # rung's WITHIN its clock, and so the sub-ladder's comparisons are the
    # ordinary parent/child ones. The stop-clock and arrival-clock ladders are
    # reported side by side and never differenced against each other.
    "M0a_const_arrival": PerBinRateModelSpec(
        name="M0a_const_arrival",
        time_spline=False,
        reward_basis=0,
        reward_rate=False,
        self_history_lags=0,
        switch=False,
        reward_time_interaction=False,
        description="homogeneous Poisson baseline on the arrival window",
        parent=None,
        time_clock="arrival",
    ),
    "M5a_time_arrival": PerBinRateModelSpec(
        name="M5a_time_arrival",
        time_spline=True,
        reward_basis=0,
        reward_rate=False,
        self_history_lags=0,
        switch=False,
        reward_time_interaction=False,
        description="+ time-since-arrival (spline)",
        parent="M0a_const_arrival",
        time_clock="arrival",
    ),
    "M7ra_time_arrival_histcos_rew": PerBinRateModelSpec(
        name="M7ra_time_arrival_histcos_rew",
        time_spline=True,
        reward_basis=_NB,
        reward_rate=False,
        self_history_lags=N_HISTORY_LAGS,
        switch=False,
        reward_time_interaction=False,
        description=(
            "+ time-since-arrival + raised-cosine self-history from event ends"
            " + reward history"
        ),
        parent="M5a_time_arrival",
        self_history_basis=N_HISTORY_BASIS,
        self_history_anchors=PER_BIN_HISTORY_ANCHORS,
        time_clock="arrival",
    ),
}


# ── Auxiliary per-bin rate models ──
#
# Fit by their own runners and persisted in their own bundles, never in
# ``MODEL_PER_BIN_RATE`` / ``all_models.pkl`` (whose schema they would change).

# MU uncertainty (``scripts/rate/fit_uncertainty_rate.py`` ->
# ``uncertainty_rate.pkl``): M7r plus the curated uncertainty/value block, so
# the block is tested beyond the time course, self-history and reward kernel.
#
# Nominal / model-free block (per-unit; these covariates are O(1) and are the
# task schedule + raw behavior, NOT the behavioral model). We prefer the real
# model over stand-ins, so the leaky-Q comparison is replaced by the model DeltaV
# and nominal-RPE stand-ins by the model's canonical RPE below.
# task_reward_probability_patch_mean_gap_best_vs_second is
# KEPT: it is the nominal best-minus-second patch value from the schedule and is
# orthogonal to the model DeltaV (r=-0.04), so it carries independent information
# (an earlier drop assuming redundancy was wrong). Excluded for collinearity:
# wsls_deviation (r=-0.70 with rpe), omission_streak (r=-0.66 with known_reward_outcome_lag_1).
MU_UNCERTAINTY_CURATED_COVARIATES = (
    "task_reward_probability_chosen_arm",
    "task_reward_probability_patch_mean_gap_best_vs_second",
    "patch_switch_fraction_last5_completed_trips",
    "trials_since_block_change",
)

# Real behavioral-model reward-prediction error, used instead of the retired
# nominal-RPE stand-in (the two had r=0.71, so the model version matters).
# q_chosen (model chosen-port value) is deliberately NOT fit: it is r=0.90
# collinear with the DeltaV decision variable (switch_advantage), which carries
# the model's value signal here. O(1) scale, so per-unit like the nominal block.
MU_UNCERTAINTY_MODEL_VALUE_COVARIATES = (
    "rpe_postoutcome",  # reward - chosen-arm pre-outcome belief
)

# Real SpatialBanditTask decision variables (qvals_table, beta_stable_withleaf;
# see docs/behavior_model_covariates.md). The paper's covariates: DeltaV, belief
# entropy, absolute value-update magnitudes. The runner z-scores them
# (disparate/tiny scale) -> per-SD rate ratios.
MU_UNCERTAINTY_RL_COVARIATES = (
    "relative_value_switch_minus_stay_preoutcome",  # DeltaV = Switch - Stay value
    "reward_belief_differential_entropy_leaf_mean_preoutcome",  # belief uncertainty (mean per-leaf Beta differential entropy)
    "reward_belief_mean_update_l1_all_ports_postoutcome",  # |value update| summed over all ports
    "reward_belief_mean_update_l1_alternative_patches_postoutcome",  # |update| over alternative patches
)

# What the model actually fits + reports (nominal + model value + RL DVs).
MU_UNCERTAINTY_COVARIATES = (
    *MU_UNCERTAINTY_CURATED_COVARIATES,
    *MU_UNCERTAINTY_MODEL_VALUE_COVARIATES,
    *MU_UNCERTAINTY_RL_COVARIATES,
)

MODEL_UNCERTAINTY_RATE: dict[str, PerBinRateModelSpec] = {
    "MU_uncertainty": PerBinRateModelSpec(
        name="MU_uncertainty",
        time_spline=True,
        reward_basis=REWARD_HISTORY_N_BASIS,
        reward_rate=False,
        self_history_lags=N_HISTORY_LAGS,
        switch=False,  # recent_switch_rate carries the switch signal
        reward_time_interaction=False,
        description="curated uncertainty/value beyond time/reward/history",
        parent="M7r_time_histcos_rew",
        self_history_basis=N_HISTORY_BASIS,
        self_history_anchors=PER_BIN_HISTORY_ANCHORS,
        extra_covariates=MU_UNCERTAINTY_COVARIATES,
    ),
}


# Cross-history coupling (``scripts/rate/fit_cross_history_rate.py``): does one
# non-local stream's recent history drive the other's rate? ``Mself`` is the
# time spline + reward kernel + self-history; ``Mselfcross`` adds the other
# stream's raised-cosine cross-history. Reward enters via the kernel only (no
# reward_rate stop covariate), so the design needs no regenerated stop
# covariate and builds on the current tables. The pair is compared by its
# runner on two designs (self vs self+cross), not by ``parent``.
def _cross_history_spec(
    name: str, *, cross: bool, duration_aware: bool = False
) -> PerBinRateModelSpec:
    anchors = ("onset", "offset") if duration_aware else ("onset",)
    return PerBinRateModelSpec(
        name=name,
        time_spline=True,
        reward_basis=REWARD_HISTORY_N_BASIS,
        reward_rate=False,
        self_history_lags=N_HISTORY_LAGS,
        switch=False,
        reward_time_interaction=False,
        description=(
            "non-local cross-history coupling, duration-aware"
            if duration_aware
            else "non-local cross-history coupling"
        ),
        parent=None,
        self_history_basis=N_HISTORY_BASIS,
        cross_history_lags=N_HISTORY_LAGS if cross else 0,
        cross_history_basis=N_HISTORY_BASIS if cross else 0,
        self_history_anchors=anchors,
        cross_history_anchors=anchors,
    )


MODEL_CROSS_HISTORY_RATE: dict[str, PerBinRateModelSpec] = {
    "Mself": _cross_history_spec("Mself", cross=False),
    "Mselfcross": _cross_history_spec("Mselfcross", cross=True),
}


# The duration-aware twin. Events have duration and one detector's events never
# overlap, so the onset-only design above counts the bins inside an ongoing
# event as at risk and can read a dependence on the previous event's END as one
# on its onset. These rungs measure history from onsets, ends or both; their
# design is built with the at-risk time (``aggregate_to_timebins(at_risk_events=
# <every event of the detector>, offset_history=True)``), which drops the bins
# inside an ongoing event. The fixed blocks are ``Mnohist_dur`` -> ``Mself_dur``
# (self history, both anchors) -> ``Mselfcross_dur`` (+ cross, both anchors);
# the one-anchor rungs give the onset-vs-offset contrasts. Runner:
# ``scripts/rate/fit_duration_aware_history_rate.py``; the simulation behind
# them is ``analysis/replay/duration_history_simulation.py``.
def _duration_aware_spec(
    name: str,
    *,
    self_anchors: tuple[str, ...] | None,
    cross_anchors: tuple[str, ...] | None = None,
) -> PerBinRateModelSpec:
    spec = _cross_history_spec(
        name, cross=cross_anchors is not None, duration_aware=True
    )
    return replace(
        spec,
        self_history_lags=N_HISTORY_LAGS if self_anchors else 0,
        self_history_basis=N_HISTORY_BASIS if self_anchors else 0,
        self_history_anchors=self_anchors or ("onset",),
        cross_history_anchors=cross_anchors or ("onset",),
    )


_BOTH = ("onset", "offset")
MODEL_CROSS_HISTORY_DURATION_RATE: dict[str, PerBinRateModelSpec] = {
    "Mnohist_dur": _duration_aware_spec("Mnohist_dur", self_anchors=None),
    "Mself_onset_dur": _duration_aware_spec("Mself_onset_dur", self_anchors=("onset",)),
    "Mself_offset_dur": _duration_aware_spec(
        "Mself_offset_dur", self_anchors=("offset",)
    ),
    "Mself_dur": _duration_aware_spec("Mself_dur", self_anchors=_BOTH),
    "Mselfcross_onset_dur": _duration_aware_spec(
        "Mselfcross_onset_dur", self_anchors=_BOTH, cross_anchors=("onset",)
    ),
    "Mselfcross_offset_dur": _duration_aware_spec(
        "Mselfcross_offset_dur", self_anchors=_BOTH, cross_anchors=("offset",)
    ),
    "Mselfcross_dur": _duration_aware_spec(
        "Mselfcross_dur", self_anchors=_BOTH, cross_anchors=_BOTH
    ),
}


# The long-window pair of the ridge cross-validated comparison
# (``analysis/replay/ridge_history_cv.py``). A single session holds about one
# expected cross-stream onset in the 150 ms cross window, so the unpenalised
# per-session cross rungs above mostly fail to converge. These two rungs are
# ``Mself_dur`` and ``Mselfcross_dur`` with the cross block (onset- and
# end-locked) reaching 1 s (``N_CROSS_LONG_LAGS``, ``N_CROSS_LONG_BASIS``
# bumps); both are frozen on the long window's cohort, so they share rows. They
# are fit pooled within animal with a ridge penalty on the history coefficients
# and compared by held-out log-likelihood, not by AIC.
MODEL_CROSS_HISTORY_LONG_RATE: dict[str, PerBinRateModelSpec] = {
    "Mself_long_dur": replace(
        MODEL_CROSS_HISTORY_DURATION_RATE["Mself_dur"], name="Mself_long_dur"
    ),
    "Mselfcross_long_dur": replace(
        MODEL_CROSS_HISTORY_DURATION_RATE["Mselfcross_dur"],
        name="Mselfcross_long_dur",
        cross_history_lags=N_CROSS_LONG_LAGS,
        cross_history_basis=N_CROSS_LONG_BASIS,
    ),
}


# ── Choice-prediction models ──
#
# A DIFFERENT estimator again: a conditional logit on the animal's
# next-arm choice (``choice_prediction.fit_choice_model``), used to
# ask whether replay content predicts behaviour beyond a behavioural
# baseline.
#
# These three are not a single linear nested chain: ``C_full`` nests
# both ``C_behavior`` (+ ``replay_fraction``) and ``C_replay_only``
# (+ the behavioural block). ``parent`` encodes the primary test —
# ``C_full`` vs ``C_behavior``, i.e. does replay add predictive power
# beyond behaviour — while ``C_replay_only`` is a standalone
# replay-only reference (``parent=None``).
#
# Two predictors are deliberately excluded from the behavioural block:
# ``is_same_patch`` (near-collinear with candidate-arm distance) and
# ``distance_to_candidate_arm_cm`` itself (quasi-separation). The trade-off —
# the replay coefficient is then not cleanly distance-controlled — and the
# principled alternative are in ``docs/design/model_history.md``
# § "Choice models: the two excluded behavioural predictors".
# Callers should still drop any column absent from the choice dataset
# before fitting.

_CHOICE_BEHAVIOR = (
    "visit_recency_trials",
    "is_candidate_arm_never_visited",
    "reward_fraction_candidate_arm_last10_visits",
    "reward_belief_mean_candidate_arm_preoutcome",
    "candidate_arm_distance_rank",  # ranked choice-set distance (identifiable; replaces the
    # quasi-separating raw candidate-arm distance as the geometric baseline,
    # dictionary §4)
)

MODEL_CHOICE: dict[str, ModelSpec] = {
    "C_replay_only": ModelSpec(
        name="C_replay_only",
        predictors=("replay_fraction",),
        description="Replay content only",
        parent=None,
    ),
    "C_behavior": ModelSpec(
        name="C_behavior",
        predictors=_CHOICE_BEHAVIOR,
        description="Behavioural baseline (recency, recent reward rate, learned Q)",
        parent=None,
    ),
    "C_full": ModelSpec(
        name="C_full",
        predictors=(*_CHOICE_BEHAVIOR, "replay_fraction"),
        description="Behavioural baseline + replay content",
        parent="C_behavior",
    ),
    "C_full_swr_split": ModelSpec(
        name="C_full_swr_split",
        predictors=(
            *_CHOICE_BEHAVIOR,
            "replay_fraction_swr",
            "replay_fraction_non_swr",
        ),
        description=(
            "Behavioural baseline + SWR-overlap and non-SWR replay separately"
        ),
        parent="C_behavior",
    ),
}


# ── SWR-contrast ladders ────────────────────────────────────────────────────
# Relocated from scripts/rate/fit_swr_contrast_rate.py and
# scripts/content/fit_swr_contrast_content.py
# so this module is the single source of truth for every model family. The runners
# import the SWR_CONTRAST_* exports below and keep only their data-assembly/fit code.
# Built from the same term helpers as PerBinRateModelSpec.terms.

# --- Rate is_swr-contrast ladder (per-bin Poisson: G0-G8 stratified, S0-S10 contrast) ---
# Each block is a tuple of formula terms with their columns. BASE is the time
# spline + the full reward-history basis + the raised-cosine self-history,
# counted from event ends on the at-risk design (``PER_BIN_HISTORY_ANCHORS``).
_BLOCKS: dict[str, tuple[PerBinTerm, ...]] = {
    "BASE": (
        _spline_term("stop"),
        *_reward_basis_terms(REWARD_HISTORY_N_BASIS),
        *(
            term
            for anchor in PER_BIN_HISTORY_ANCHORS
            for term in _history_terms(
                _ANCHOR_PREFIX[anchor], N_HISTORY_LAGS, N_HISTORY_BASIS
            )
        ),
    ),
    "PERISWITCH": (
        *_column_terms(("periswitch_index_trials", "is_periswitch_post")),
        _interaction(*_column_terms(("periswitch_index_trials", "is_periswitch_post"))),
    ),
    "BLOCK": tuple(_column_terms(("trials_since_block_change",))),
    "SWITCHRATE": tuple(
        _column_terms(("patch_switch_fraction_last5_completed_trips",))
    ),
    # beta relative value DeltaV. Sign convention: Switch - Stay; ~= -(current stay value), r = -0.952 with stay_sibling_mu_preoutcome_sess and -0.890 with q_chosen, so a POSITIVE coefficient means the current option's value is LOW.
    # An earlier "Stay-vs-Switch" label here had it backwards.
    "VALUE": tuple(_column_terms(("relative_value_switch_minus_stay_preoutcome",))),
    # beta reward prediction error (reward - q_chosen)
    "RPE": tuple(_column_terms(("rpe_postoutcome",))),
    # beta belief entropy (per-leaf Beta differential entropy)
    "UNCERT": tuple(
        _column_terms(("reward_belief_differential_entropy_leaf_mean_preoutcome",))
    ),
    # beta value-update magnitude
    "UPDATE": tuple(
        _column_terms(("reward_belief_mean_update_l1_all_ports_postoutcome",))
    ),
    # immediate reward-history component x within-stop time course (dwell-shape)
    "REWXTIME": (
        _interaction(*_column_terms(REWARD_HISTORY_COLUMNS[:1]), _spline_term("stop")),
    ),
}


def _block_formula(block: str) -> str:
    """A block's formula fragment.

    ``REWARD_HISTORY_INCOMPLETE`` is left out: it is a structural-missingness
    nuisance main effect that every rung appends once, outside any ``is_swr *
    (...)`` expansion, while the block's columns still carry it.
    """
    return " + ".join(
        term.term for term in _BLOCKS[block] if term.term != REWARD_HISTORY_INCOMPLETE
    )


def _block_cols(block: str) -> list[str]:
    """A block's design columns, in term order, without repeats."""
    return list(dict.fromkeys(c for term in _BLOCKS[block] for c in term.columns))


SWR_CONTRAST_RATE_STOP_COVARS = tuple(
    c
    for b in ("PERISWITCH", "BLOCK", "SWITCHRATE", "VALUE", "RPE", "UNCERT", "UPDATE")
    for c in _block_cols(b)
)
# Bins of event history each column's term looks back over, from the blocks.
_CONTRAST_HISTORY_LAGS = {
    c: term.history_lags
    for terms in _BLOCKS.values()
    for term in terms
    if term.history_lags
    for c in term.columns
}


def _blocks_formula(blocks):
    return " + ".join(_block_formula(b) for b in blocks)


def _blocks_cols(blocks):
    return [c for b in blocks for c in _block_cols(b)]


def _g_specs():
    """Stratified rungs. ``REWARD_HISTORY_INCOMPLETE`` is appended as a plain
    main effect (it is a structural-missingness nuisance, never a dimension the
    ladder reports)."""
    ladders = {
        "G0_base": ["BASE"],
        "G1_periswitch": ["BASE", "PERISWITCH"],
        "G2_block": ["BASE", "BLOCK"],
        "G3_switchrate": ["BASE", "SWITCHRATE"],
        "G4_value": ["BASE", "VALUE"],
        "G5_rpe": ["BASE", "RPE"],
        "G6_uncert": ["BASE", "UNCERT"],
        "G7_update": ["BASE", "UPDATE"],
        "G8_rewxtime": ["BASE", "REWXTIME"],
    }
    return {
        n: (f"{_blocks_formula(b)} + {REWARD_HISTORY_INCOMPLETE}", _blocks_cols(b))
        for n, b in ladders.items()
    }


def _s_specs():
    """is_swr-contrast rungs; see :func:`_g_specs` for the indicator."""
    base, peri, blk, swr = (
        _block_formula(b) for b in ("BASE", "PERISWITCH", "BLOCK", "SWITCHRATE")
    )
    val, rpe, unc, upd, rxt = (
        _block_formula(b) for b in ("VALUE", "RPE", "UNCERT", "UPDATE", "REWXTIME")
    )
    raw = {
        "S0_rate": (f"is_swr + {base}", ["BASE"]),
        "S1_rew": (
            f"{base} + is_swr + is_swr:{REWARD_HISTORY_COLUMNS[0]}",
            ["BASE"],
        ),
        "S2_base": (f"is_swr * ({base})", ["BASE"]),
        "S3_periswitch": (f"is_swr * ({base} + {peri})", ["BASE", "PERISWITCH"]),
        "S4_block": (f"is_swr * ({base} + {blk})", ["BASE", "BLOCK"]),
        "S5_switchrate": (f"is_swr * ({base} + {swr})", ["BASE", "SWITCHRATE"]),
        "S6_value": (f"is_swr * ({base} + {val})", ["BASE", "VALUE"]),
        "S7_rpe": (f"is_swr * ({base} + {rpe})", ["BASE", "RPE"]),
        "S8_uncert": (f"is_swr * ({base} + {unc})", ["BASE", "UNCERT"]),
        "S9_update": (f"is_swr * ({base} + {upd})", ["BASE", "UPDATE"]),
        "S10_rewxtime": (f"is_swr * ({base} + {rxt})", ["BASE", "REWXTIME"]),
    }
    # The missingness indicator is appended OUTSIDE the ``is_swr * (...)``
    # expansion: it is a nuisance main effect, and an is_swr:incomplete
    # interaction would report a contrast of structural missingness.
    return {
        n: (f"{f} + {REWARD_HISTORY_INCOMPLETE}", _blocks_cols(bl))
        for n, (f, bl) in raw.items()
    }


SWR_CONTRAST_RATE_G_SPECS = _g_specs()
SWR_CONTRAST_RATE_S_SPECS = _s_specs()


def longest_history_window_s(
    rungs: Iterable[PerBinRateModelSpec | tuple[str, list[str]]],
) -> float:
    """Seconds the longest history covariate of a per-bin ladder looks back.

    A ladder is fit on one frozen cohort, so its bins are dropped by the
    window of its longest-reaching rung: a bin whose window touches the
    artifact mask reads unobserved time (``per_bin_design.aggregate_to_timebins``,
    ``history_window_s``). A rung's window is the longest ``history_lags`` of
    its terms (:class:`PerBinTerm`) times ``BIN_S``; an is_swr-contrast rung
    (``(formula, columns)``) reaches as far as the block terms its columns come
    from. The reward-history block reads trial outcomes, not neural time, and
    adds none.

    Parameters
    ----------
    rungs : iterable
        ``PerBinRateModelSpec`` rungs, or the ``(formula, columns)`` pairs of
        ``SWR_CONTRAST_RATE_G_SPECS`` / ``SWR_CONTRAST_RATE_S_SPECS``.

    Returns
    -------
    float
        Seconds; 0.0 for a ladder with no history covariate.
    """
    n_lags = 0
    for rung in rungs:
        if isinstance(rung, PerBinRateModelSpec):
            n_lags = max(n_lags, rung.history_lags())
        else:
            _, columns = rung
            n_lags = max([n_lags, *(_CONTRAST_HISTORY_LAGS.get(c, 0) for c in columns)])
    return n_lags * BIN_S


# Covariates that are only defined once the trial's outcome is known. A stop that
# begins before ``outcome_known_time`` carries no value for them, so the rate
# runner NaNs them there; the rungs that need them are therefore fit on a smaller
# cohort than the rest of the ladder.
POST_OUTCOME_COVARIATES = (
    "rpe_postoutcome",
    "reward_belief_mean_update_l1_all_ports_postoutcome",
)

# The is_swr-contrast rate rungs that carry a post-outcome covariate
# (``G5_rpe``, ``G7_update``, ``S7_rpe``, ``S9_update``). They are fit on the
# POST_OUTCOME_COHORT together with their ladder's null, so their AIC comparison
# shares rows; they are not members of the nested ladder frozen on the
# PRIMARY_COHORT.
POST_OUTCOME_RATE_RUNGS = frozenset(
    name
    for specs in (SWR_CONTRAST_RATE_G_SPECS, SWR_CONTRAST_RATE_S_SPECS)
    for name, (_formula, need) in specs.items()
    if any(covariate in POST_OUTCOME_COVARIATES for covariate in need)
)

# Cohort labels carried on every fitted is_swr-contrast rate row.
PRIMARY_COHORT = "primary"
POST_OUTCOME_COHORT = "post_outcome"

# --- Content is_swr-contrast ladder (conditional Poisson over 9 arms / 3 patches) ---
# name -> (predictors, scope)  scope: "split" (per is_swr split) | "pooled" (contrast)
ARMFE = [f"is_candidate_arm_{a}" for a in range(8)]
# ``candidate_arm_distance_rank`` is deliberately absent: its rank-1 level is
# identical to ``is_current_arm``, so carrying it makes every C*/CS*
# ``is_current_arm`` non-comparable with ``M1``/``C1_current``. Numbers:
# ``docs/design/model_history.md`` § "Content is_swr-contrast base: why
# candidate_arm_distance_rank is absent".
BASEc_NT = [
    *ARMFE,
    "is_current_arm",
    "is_prev_arm",
    *RECENCY_SPLINE_COLUMNS,
    "is_candidate_arm_never_visited",
]
_TSC = "log1p_time_since_stop_start_s_z_x_is_current_arm"
# Event-length control: z(log n_nonlocal_bins) crossed with the current and
# previous arm indicators, on every rung built on ``BASEc``; the ``nt`` twins
# keep ``BASEc_NT`` as the uncontrolled sensitivity fit. Why:
# ``docs/design/model_history.md`` § "Content is_swr-contrast base: the
# event-length control".
_LENC = "log_n_nonlocal_bins_z_x_is_current_arm"
_LENP = "log_n_nonlocal_bins_z_x_is_prev_arm"
BASEc = [*BASEc_NT, _TSC, _LENC, _LENP]
# SWR-specific arm baselines, on every pooled rung: SWR and non-SWR events each
# get their own arm fixed effects, so the swr_x term of a rung compares the two
# kinds of event at the same arm rather than absorbing SWR's shift of content
# onto the trunk arms. Why: ``docs/design/model_history.md`` § "Content
# is_swr-contrast: SWR-specific arm baselines".
SWR_ARMFE = [f"swr_x_{c}" for c in ARMFE]
BASEc_SWR = [*BASEc, *SWR_ARMFE]
BASEc_NT_SWR = [*BASEc_NT, *SWR_ARMFE]
SWR_CONTRAST_CONTENT_ARM_LADDER: dict[str, tuple[list[str], str]] = {
    "C0_armfe": (ARMFE, "split"),
    "C1_current": ([*ARMFE, "is_current_arm"], "split"),
    "C2_base": (BASEc, "split"),
    "C3_armreward": ([*BASEc, "reward_fraction_candidate_arm_last10_visits"], "split"),
    "C4_armvalue": ([*BASEc, "reward_belief_mean_candidate_arm_preoutcome"], "split"),
    "C5_lastrew": ([*BASEc, "is_last_rewarded_noncurrent_arm"], "split"),
    "C6_rpe": ([*BASEc, "rpe_postoutcome_z_x_is_current_arm"], "split"),
    "C7_value": (
        [
            *BASEc,
            "relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm",
        ],
        "split",
    ),
    "C8_uncert": (
        [
            *BASEc,
            "reward_belief_differential_entropy_leaf_mean_preoutcome_z_x_is_current_arm",
        ],
        "split",
    ),
    "C9_update": (
        [
            *BASEc,
            "reward_belief_mean_update_l1_all_ports_postoutcome_z_x_is_current_arm",
        ],
        "split",
    ),
    "C10_periswitch": (
        [
            *BASEc,
            "periswitch_index_trials_z_x_is_current_arm",
            "is_periswitch_post_x_is_current_arm",
            "periswitch_index_trials_z_x_is_periswitch_post_x_is_current_arm",
        ],
        "split",
    ),
    "C11_block": (
        [*BASEc, "trials_since_block_change_z_x_is_current_arm"],
        "split",
    ),
    "C12_switchrate": (
        [
            *BASEc,
            "patch_switch_fraction_last5_completed_trips_z_x_is_current_arm",
        ],
        "split",
    ),
    # Every ``swr_x_`` coefficient on a ``BASEc`` rung is the LENGTH-ADJUSTED
    # (DIRECT) SWR effect; the ``nt`` twin on ``BASEc_NT`` holds the TOTAL SWR
    # effect. Both are shipped and reported side by side; neither supersedes the
    # other. Estimand argument: ``docs/design/model_history.md``
    # § "Content is_swr-contrast: the direct/total estimand of every swr_x term".
    "CS1_current": ([*BASEc_SWR, "swr_x_is_current_arm"], "pooled"),
    "CS1nt_current": ([*BASEc_NT_SWR, "swr_x_is_current_arm"], "pooled"),
    "CS2_prev": ([*BASEc_SWR, "swr_x_is_prev_arm"], "pooled"),
    "CS3_lastrew": (
        [
            *BASEc_SWR,
            "is_last_rewarded_noncurrent_arm",
            "swr_x_is_last_rewarded_noncurrent_arm",
        ],
        "pooled",
    ),
    "CS4_armreward": (
        [
            *BASEc_SWR,
            "reward_fraction_candidate_arm_last10_visits",
            "swr_x_reward_fraction_candidate_arm_last10_visits",
        ],
        "pooled",
    ),
    "CS4nt_armreward": (
        [
            *BASEc_NT_SWR,
            "reward_fraction_candidate_arm_last10_visits",
            "swr_x_reward_fraction_candidate_arm_last10_visits",
        ],
        "pooled",
    ),
    "CS5_armvalue": (
        [
            *BASEc_SWR,
            "reward_belief_mean_candidate_arm_preoutcome",
            "swr_x_reward_belief_mean_candidate_arm_preoutcome",
        ],
        "pooled",
    ),
    "CS5nt_armvalue": (
        [
            *BASEc_NT_SWR,
            "reward_belief_mean_candidate_arm_preoutcome",
            "swr_x_reward_belief_mean_candidate_arm_preoutcome",
        ],
        "pooled",
    ),
    # Discriminant vs Gillespie et al. 2021 (Neuron): their previous-goal
    # enrichment (Fig 8) is the reduction the graded-value avoidance must survive.
    # CS5b adds the previous-goal analog (is_last_rewarded_elsewhere + its SWR
    # interaction) on top of CS5, so the
    # ``swr_x_reward_belief_mean_candidate_arm_preoutcome`` term is read as the
    # value effect net of previous-goal. The rung is REFUSED as unidentified
    # (most fitted sessions are flagged separated); it is kept so the refusal is
    # reproducible, not because its coefficient is citable. Fitted values are
    # reported in paper/claims_table.md (row 19f) — do not duplicate them here.
    "CS5b_valuevsprevgoal": (
        [
            *BASEc_SWR,
            "reward_belief_mean_candidate_arm_preoutcome",
            "swr_x_reward_belief_mean_candidate_arm_preoutcome",
            "is_last_rewarded_noncurrent_arm",
            "swr_x_is_last_rewarded_noncurrent_arm",
        ],
        "pooled",
    ),
    # Per-arm beta belief UNCERTAINTY (exploration). BASEc carries no value term, so
    # these rungs are construct-clean (beta uncertainty on the spatial base).
    # C13 (split): independent uncertainty coefficient on the SWR-only
    # and non-SWR-only subsets. CS6 (pooled): main = non-SWR effect, swr_x_ = the
    # formal SWR-vs-non-SWR difference. Mirrors the C4/CS5 value pair.
    "C13_armuncert": (
        [*BASEc, "reward_belief_variance_candidate_arm_preoutcome_z"],
        "split",
    ),
    "CS6_armuncert": (
        [
            *BASEc_SWR,
            "reward_belief_variance_candidate_arm_preoutcome_z",
            "swr_x_reward_belief_variance_candidate_arm_preoutcome_z",
        ],
        "pooled",
    ),
    "CS6nt_armuncert": (
        [
            *BASEc_NT_SWR,
            "reward_belief_variance_candidate_arm_preoutcome_z",
            "swr_x_reward_belief_variance_candidate_arm_preoutcome_z",
        ],
        "pooled",
    ),
}
SWR_CONTRAST_CONTENT_ARM_KEY_TERM = {
    "C1_current": "is_current_arm",
    "C2_base": "is_current_arm",
    "C3_armreward": "reward_fraction_candidate_arm_last10_visits",
    "C4_armvalue": "reward_belief_mean_candidate_arm_preoutcome",
    "C5_lastrew": "is_last_rewarded_noncurrent_arm",
    "C6_rpe": "rpe_postoutcome_z_x_is_current_arm",
    "C7_value": "relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm",
    "C8_uncert": "reward_belief_differential_entropy_leaf_mean_preoutcome_z_x_is_current_arm",
    "C9_update": "reward_belief_mean_update_l1_all_ports_postoutcome_z_x_is_current_arm",
    "C10_periswitch": "periswitch_index_trials_z_x_is_current_arm",
    "C11_block": "trials_since_block_change_z_x_is_current_arm",
    "C12_switchrate": "patch_switch_fraction_last5_completed_trips_z_x_is_current_arm",
    "CS1_current": "swr_x_is_current_arm",
    "CS1nt_current": "swr_x_is_current_arm",
    "CS2_prev": "swr_x_is_prev_arm",
    "CS3_lastrew": "swr_x_is_last_rewarded_noncurrent_arm",
    "CS4_armreward": "swr_x_reward_fraction_candidate_arm_last10_visits",
    "CS4nt_armreward": "swr_x_reward_fraction_candidate_arm_last10_visits",
    "CS5_armvalue": "swr_x_reward_belief_mean_candidate_arm_preoutcome",
    "CS5nt_armvalue": "swr_x_reward_belief_mean_candidate_arm_preoutcome",
    "CS5b_valuevsprevgoal": "swr_x_reward_belief_mean_candidate_arm_preoutcome",
    "C13_armuncert": "reward_belief_variance_candidate_arm_preoutcome_z",
    "CS6_armuncert": "swr_x_reward_belief_variance_candidate_arm_preoutcome_z",
    "CS6nt_armuncert": "swr_x_reward_belief_variance_candidate_arm_preoutcome_z",
}
BASEc_PATCH_NT = [
    "is_current_patch",
    "is_prev_patch",
    "visit_recency_trials_patch",
    "is_patch_never_visited",
]
# Patch analogues of the arm within-stop-time and event-length controls (see BASEc).
_TSC_PATCH = "log1p_time_since_stop_start_s_z_x_is_current_patch"
_LENC_PATCH = "log_n_nonlocal_bins_z_x_is_current_patch"
_LENP_PATCH = "log_n_nonlocal_bins_z_x_is_prev_patch"
BASEc_PATCH = [*BASEc_PATCH_NT, _TSC_PATCH, _LENC_PATCH, _LENP_PATCH]
# ``CP2_olddecay`` and ``CPS3_olddecay`` were retired 2026-08-18; see
# ``docs/design/model_history.md`` § "Content patch is_swr-contrast: the retired
# CP2/CPS3 rungs".
_LR_PATCH = "is_last_rewarded_noncurrent_patch"
SWR_CONTRAST_CONTENT_PATCH_LADDER: dict[str, tuple[list[str], str]] = {
    "CP0_base": (BASEc_PATCH, "split"),
    "CP1_lastrew": ([*BASEc_PATCH, _LR_PATCH], "split"),
    "CP3_newgrowth": (
        [*BASEc_PATCH, "periswitch_index_trials_z_x_is_current_patch"],
        "split",
    ),
    "CP4_switchrate": (
        [
            *BASEc_PATCH,
            "patch_switch_fraction_last5_completed_trips_z_x_is_current_patch",
        ],
        "split",
    ),
    # Time- and length-controlled current-patch contrast (both inherited from
    # BASEc_PATCH) + uncontrolled twin, mirroring the arm CS1/CS1nt pair.
    "CPS1_current": ([*BASEc_PATCH, "swr_x_is_current_patch"], "pooled"),
    "CPS1nt_current": ([*BASEc_PATCH_NT, "swr_x_is_current_patch"], "pooled"),
    "CPS2_lastrew": (
        [*BASEc_PATCH, _LR_PATCH, "swr_x_is_last_rewarded_noncurrent_patch"],
        "pooled",
    ),
}
SWR_CONTRAST_CONTENT_PATCH_KEY_TERM = {
    "CP0_base": "is_current_patch",
    "CP1_lastrew": _LR_PATCH,
    "CP3_newgrowth": "periswitch_index_trials_z_x_is_current_patch",
    "CP4_switchrate": "patch_switch_fraction_last5_completed_trips_z_x_is_current_patch",
    "CPS1_current": "swr_x_is_current_patch",
    "CPS1nt_current": "swr_x_is_current_patch",
    "CPS2_lastrew": "swr_x_is_last_rewarded_noncurrent_patch",
}
SWR_CONTRAST_CONTENT_BASEC = BASEc
SWR_CONTRAST_CONTENT_BASEC_NT = BASEc_NT
SWR_CONTRAST_CONTENT_BASEC_PATCH = BASEc_PATCH
SWR_CONTRAST_CONTENT_BASEC_PATCH_NT = BASEc_PATCH_NT

# ── Phase-4 causal-hygiene: descriptive-rung allowlists ──
#
# These rungs carry a POST_OUTCOME (RPE / value-update — embeds the current reward) or
# FUTURE_REFERENCING (``periswitch_index_trials`` / ``is_periswitch_post`` —
# signs toward the NEXT switch)
# covariate that CANNOT be a forward predictor. They are RETAINED
# under an explicit *descriptive* claim (a value/RPE *association*, or a *peri-switch
# alignment*), NOT gated. The runners call ``assert_predictor_timing(descriptive=True)``
# for exactly these names; a NEW rung that introduces a leaky covariate but is NOT listed
# here fails the timing assertion (it is not silently promoted to a predictive claim).
# See docs/models_table.md and analysis.replay.covariate_timing.
# Contrast rungs fitted over the six leaf arms only, not all nine.
#
# The trunk arms carry a structural zero for every leaf-derived belief covariate
# (candidate-arm reward-belief mean, task reward probability, and
# reward-belief variance), so including them adds rows that
# say nothing about those covariates while manufacturing collinearity between
# them. Their persisted z units are now defined over the six reward-site arms;
# the nine-arm design still retains trunk response rows, but structural zeros no
# longer determine the scale of value/uncertainty coefficients.
#
# Restricted to the standalone contrast rungs, which are fitted one rung at a
# time and so can switch candidate sets individually. The arm-ladder belief
# rungs (Mv/Mu/Mvu) are nested -- their parents are M4 and Mu -- and changing
# the candidate set changes the conditional likelihood, so a leaf-only Mv
# cannot be compared to a nine-arm M4 by ΔAIC. They are instead compared to
# leaf-only M0-M4 fitted in the same run: ``fit_replay_content_models.py
# --leaf-only --models M0,M0a,M1,M2,M3,M4,Mv,Mu,Mvu`` restricts the whole chain
# to the six leaves, so every rung in it shares one six-category likelihood.
LEAF_ONLY_CONTRAST_RUNGS = frozenset(
    {
        "C4_armvalue",
        "C13_armuncert",
        "CS5_armvalue",
        "CS5nt_armvalue",
        "CS5b_valuevsprevgoal",
        "CS6_armuncert",
        "CS6nt_armuncert",
    }
)
LEAF_ONLY_N_CATEGORIES = N_LEAF_ARMS
# Arm fixed effects for a leaf-only fit: five dummies for six leaves, arm 5 the
# reference. The nine-arm set (is_candidate_arm_0..7) is rank-deficient once the trunks are
# gone -- is_candidate_arm_6 and is_candidate_arm_7 are then identically zero -- so a leaf-only rung
# carrying it does not fit at all.
LEAF_ONLY_ARM_FE_PREDICTORS: tuple[str, ...] = tuple(
    f"is_candidate_arm_{i}" for i in range(5)
)


def leaf_only_predictors(predictors) -> list[str]:
    """Drop the arm fixed effects (and their SWR-specific twins) that are
    structurally zero on a leaf-only fit."""
    dropped = set(ARM_FE_PREDICTORS) - set(LEAF_ONLY_ARM_FE_PREDICTORS)
    dropped |= {f"swr_x_{p}" for p in dropped}
    return [p for p in predictors if p not in dropped]


def rung_predictors(spec: ModelSpec, leaf_only: bool) -> list[str]:
    """Predictors one rung is fitted with, under the run's candidate set.

    On a leaf-only fit the arm dummies for arms 5-7 are identically zero (arm 5
    becomes the reference and the two trunk dummies have no rows left), so the
    design is rank-deficient unless they are dropped.

    Parameters
    ----------
    spec : ModelSpec
        The rung's specification.
    leaf_only : bool
        Whether the fit conditions on the six leaf arms.

    Returns
    -------
    list[str]
        Column names for this rung's design matrix.
    """
    predictors = list(spec.predictors)
    return leaf_only_predictors(predictors) if leaf_only else predictors


def restrict_to_leaf_arms(long_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the six leaf-arm candidate rows of every event.

    The trunk rows are dropped, not folded into the leaves: each retained row
    keeps the allocation it had over the nine candidates, so an event whose
    replay ran along a trunk contributes correspondingly less mass to the
    conditional likelihood.

    Parameters
    ----------
    long_df : pd.DataFrame
        Long-format event × candidate-arm frame carrying ``candidate_arm``.

    Returns
    -------
    pd.DataFrame
        The leaf-arm rows, with a fresh index.

    Raises
    ------
    ValueError
        If ``candidate_arm`` is absent.
    """
    if "candidate_arm" not in long_df.columns:
        raise ValueError(
            "A leaf-only fit needs the candidate_arm column to identify the leaf "
            f"rows; columns are {list(long_df.columns)}."
        )
    leaves = long_df["candidate_arm"] < LEAF_ONLY_N_CATEGORIES
    return long_df.loc[leaves].reset_index(drop=True)


def leaf_only_cohort(
    long_df: pd.DataFrame,
    *,
    response_col: str,
    response: str,
    event_col: str = "event_id",
) -> tuple[pd.DataFrame, bool]:
    """Restrict a cohort to the leaf arms and put its response on that support.

    A fraction is a share of the event's whole nine-arm mass, and the
    conditional likelihood weights an event by the total it carries, so keeping
    the persisted fractions after dropping the trunk rows would weight an event
    by its leaf share. That share is not balanced across the comparisons these
    ladders exist for -- SWR events put a substantially larger part of their
    bins on the trunks than non-SWR events do -- so it is rescaled to unit mass
    over the six leaves. A count response is left alone: weighting an event by
    its number of decoded timebins is what the count response means.

    Parameters
    ----------
    long_df : pd.DataFrame
        Long-format event x candidate-arm cohort, all nine arms.
    response_col : str
        The resolved response column.
    response : str
        The ``--response`` value the column came from, one of
        :data:`~continuum_swr_replay.analysis.replay.content_response.RESPONSE_CHOICES`.
    event_col : str, optional
        Event grouping key.

    Returns
    -------
    pd.DataFrame
        The leaf-arm rows, with the response rescaled when it is a fraction.
    bool
        Whether the response was rescaled, for the fit rows and the provenance.
    """
    leaves = restrict_to_leaf_arms(long_df)
    if response != "fraction":
        return leaves, False
    return (
        renormalize_fraction_over_candidates(
            leaves, response_col=response_col, event_col=event_col
        ),
        True,
    )


DESCRIPTIVE_CONTENT_RUNGS = frozenset(
    {
        "C6_rpe",
        "C9_update",  # POST_OUTCOME: RPE / value-update association
        "C10_periswitch",  # FUTURE_REFERENCING: peri-switch alignment
        "CP3_newgrowth",
    }
)
DESCRIPTIVE_RATE_RUNGS = frozenset(
    {
        "G5_rpe",
        "G7_update",
        "S7_rpe",
        "S9_update",  # POST_OUTCOME association
        "G1_periswitch",
        "S3_periswitch",  # FUTURE_REFERENCING peri-switch
    }
)
