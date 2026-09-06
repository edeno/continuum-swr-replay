"""Model definitions for replay content analysis.

Single source of truth for the four model families, each a dict of
:class:`ModelSpec`:

- ``MODEL_HIERARCHY`` (+ ``_SWR`` / ``_MUA``) — arm-level nested
  hierarchy, conditional Poisson on per-arm timebin counts.
- ``MODEL_PATCH`` (+ ``_SWR`` / ``_MUA`` / ``_SWITCH``) — patch-level
  conditional Poisson (9 arms collapsed to 3 patches).
- ``MODEL_PER_STOP_RATE`` — ordinary Poisson on per-stop event counts
  (different estimator; see its section below).
- ``MODEL_CHOICE`` — conditional logit on the animal's next-arm
  choice (different estimator again).

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

from collections.abc import Mapping
from dataclasses import dataclass

# Shared with the covariate builders, which must not import this module: see
# ``design_constants`` for why the numbers live there and not here. Re-exported
# below so ``from ...model_specs import BIN_S`` keeps working.
from .design_constants import (
    ARM_FE_PREDICTORS,
    ARRIVAL_CLOCK_COLUMN,
    ARRIVAL_SPLINE_KNOTS,
    BIN_S,
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
    "MODEL_HIERARCHY",
    "MODEL_HIERARCHY_MUA",
    "MODEL_HIERARCHY_SWR",
    "MODEL_PATCH",
    "MODEL_PATCH_MUA",
    "MODEL_PATCH_SWITCH",
    "MODEL_PATCH_SWR",
    "MODEL_PER_BIN_RATE",
    "MODEL_PER_STOP_RATE",
    "ModelSpec",
    "N_HISTORY_BASIS",
    "N_HISTORY_LAGS",
    "N_REWARD_LAGS",
    "POST_OUTCOME_COHORT",
    "POST_OUTCOME_COVARIATES",
    "POST_OUTCOME_RATE_RUNGS",
    "PRIMARY_COHORT",
    "PerBinRateModelSpec",
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
# :func:`_helpers.aggregate_to_timebins`. Unlike ``MODEL_PER_STOP_RATE`` (one
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
# Per-stop covariates the shipped ladder needs beyond the reward-history basis
# and the self-history lags — carried into the per-bin design by
# aggregate_to_timebins. Shared by the fitter orchestrator and the
# model-checking plotter so the inline refit matches the persisted fits. Empty:
# the reward-history basis is requested by ``reward_basis``, not as a covariate,
# and the immediate-switch term was retired with the M8 rung (see R1).
STOP_COVARIATES: tuple[str, ...] = ()


@dataclass(frozen=True)
class PerBinRateModelSpec:
    """One model in the per-bin conditional-intensity ladder.

    Parallel to :class:`ModelSpec` (kept separate so the simpler arm/patch/
    per-stop families are unaffected). The toggle fields drive both
    :func:`_helpers.aggregate_to_timebins` (which columns to build) and
    :meth:`formula` (the patsy formula), so the spec is the single source of
    truth for each model.

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
    # Extra per-stop covariates appended to the formula as plain linear terms
    # (e.g. the uncertainty/value block instantiated in the runner). Default
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

    def formula(self) -> str:
        """patsy formula (response ``count``; the offset is applied at fit time)."""
        spline = _ARRIVAL_SPLINE if self.time_clock == "arrival" else _SPLINE
        terms: list[str] = []
        if self.time_spline:
            terms.append(spline)
        if self.reward_basis:
            terms += [
                *REWARD_HISTORY_COLUMNS[: self.reward_basis],
                REWARD_HISTORY_INCOMPLETE,
            ]
        if self.reward_rate:
            terms.append("reward_fraction_last10_known_trials")
        if self.self_history_basis:
            terms += [f"hist_rcos_{k}" for k in range(self.self_history_basis)]
        else:
            terms += [f"onset_lag{j}" for j in range(1, self.self_history_lags + 1)]
        if self.cross_history_basis:
            terms += [f"cross_hist_rcos_{k}" for k in range(self.cross_history_basis)]
        else:
            terms += [f"cross_lag{j}" for j in range(1, self.cross_history_lags + 1)]
        # ``switch`` is retired (see the field docstring); the completed-trip
        # switch flag is never added to a per-bin model.
        if self.reward_time_interaction:
            terms.append(f"{REWARD_HISTORY_COLUMNS[0]}:{spline}")
        terms += list(self.extra_covariates)
        return "count ~ " + (" + ".join(terms) if terms else "1")


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
    # Raised-cosine self-history variants: smooth basis over the same window so
    # the history block cannot separate on the refractory bins (lifts held-out
    # cv_bits_per_event from negative to positive vs the onset_lag dummies).
    "M3r_histcos": PerBinRateModelSpec(
        name="M3r_histcos",
        time_spline=False,
        reward_basis=0,
        reward_rate=False,
        self_history_lags=N_HISTORY_LAGS,
        switch=False,
        reward_time_interaction=False,
        description="+ self-history (raised-cosine basis)",
        parent="M0_const",
        self_history_basis=N_HISTORY_BASIS,
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
        description="+ time + raised-cosine self-history",
        parent="M5_time",
        self_history_basis=N_HISTORY_BASIS,
    ),
    "M7r_time_histcos_rew": PerBinRateModelSpec(
        name="M7r_time_histcos_rew",
        time_spline=True,
        reward_basis=_NB,
        reward_rate=False,
        self_history_lags=N_HISTORY_LAGS,
        switch=False,
        reward_time_interaction=False,
        description="+ time + raised-cosine self-history + reward history",
        parent="M6r_time_histcos",
        self_history_basis=N_HISTORY_BASIS,
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
            "+ time-since-arrival + raised-cosine self-history + reward history"
        ),
        parent="M5a_time_arrival",
        self_history_basis=N_HISTORY_BASIS,
        time_clock="arrival",
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
# Relocated from scripts/fit_swr_contrast_rate.py and fit_swr_contrast_content.py
# so this module is the single source of truth for every model family. The runners
# import the SWR_CONTRAST_* exports below and keep only their data-assembly/fit code.
# Reuses the module-level _SPLINE / N_REWARD_LAGS / N_HISTORY_BASIS defined above.

# --- Rate is_swr-contrast ladder (per-bin Poisson: G0-G8 stratified, S0-S10 contrast) ---
_REWBASIS = " + ".join(REWARD_HISTORY_COLUMNS)
_HISTCOS = " + ".join(f"hist_rcos_{k}" for k in range(N_HISTORY_BASIS))
_BLOCK_F = {
    "BASE": f"{_SPLINE} + {_REWBASIS} + {_HISTCOS}",
    "PERISWITCH": (
        "periswitch_index_trials + is_periswitch_post + "
        "periswitch_index_trials:is_periswitch_post"
    ),
    "BLOCK": "trials_since_block_change",
    "SWITCHRATE": "patch_switch_fraction_last5_completed_trips",
    # beta relative value DeltaV. Sign convention: Switch - Stay; ~= -(current stay value), r = -0.952 with stay_sibling_mu_preoutcome_sess and -0.890 with q_chosen, so a POSITIVE coefficient means the current option's value is LOW.
    # An earlier "Stay-vs-Switch" label here had it backwards.
    "VALUE": "relative_value_switch_minus_stay_preoutcome",
    "RPE": "rpe_postoutcome",  # beta reward prediction error (reward - q_chosen)
    "UNCERT": "reward_belief_differential_entropy_leaf_mean_preoutcome",  # beta belief entropy (per-leaf Beta differential entropy)
    "UPDATE": "reward_belief_mean_update_l1_all_ports_postoutcome",  # beta value-update magnitude
    # immediate reward-history component x within-stop time course (dwell-shape)
    "REWXTIME": f"{REWARD_HISTORY_COLUMNS[0]}:{_SPLINE}",
}
_BLOCK_COLS = {
    "BASE": [
        "tc",
        *REWARD_HISTORY_COLUMNS,
        REWARD_HISTORY_INCOMPLETE,
        *[f"hist_rcos_{k}" for k in range(N_HISTORY_BASIS)],
    ],
    "PERISWITCH": ["periswitch_index_trials", "is_periswitch_post"],
    "BLOCK": ["trials_since_block_change"],
    "SWITCHRATE": ["patch_switch_fraction_last5_completed_trips"],
    "VALUE": ["relative_value_switch_minus_stay_preoutcome"],
    "RPE": ["rpe_postoutcome"],
    "UNCERT": ["reward_belief_differential_entropy_leaf_mean_preoutcome"],
    "UPDATE": ["reward_belief_mean_update_l1_all_ports_postoutcome"],
    "REWXTIME": [REWARD_HISTORY_COLUMNS[0], "tc"],
}
SWR_CONTRAST_RATE_STOP_COVARS = tuple(
    c
    for b in ("PERISWITCH", "BLOCK", "SWITCHRATE", "VALUE", "RPE", "UNCERT", "UPDATE")
    for c in _BLOCK_COLS[b]
)


def _blocks_formula(blocks):
    return " + ".join(_BLOCK_F[b] for b in blocks)


def _blocks_cols(blocks):
    return [c for b in blocks for c in _BLOCK_COLS[b]]


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
    b = _BLOCK_F
    base, peri, blk, swr = b["BASE"], b["PERISWITCH"], b["BLOCK"], b["SWITCHRATE"]
    val, rpe, unc, upd, rxt = (
        b["VALUE"],
        b["RPE"],
        b["UNCERT"],
        b["UPDATE"],
        b["REWXTIME"],
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
    "CS1_current": ([*BASEc, "swr_x_is_current_arm"], "pooled"),
    "CS1nt_current": ([*BASEc_NT, "swr_x_is_current_arm"], "pooled"),
    "CS2_prev": ([*BASEc, "swr_x_is_prev_arm"], "pooled"),
    "CS3_lastrew": (
        [
            *BASEc,
            "is_last_rewarded_noncurrent_arm",
            "swr_x_is_last_rewarded_noncurrent_arm",
        ],
        "pooled",
    ),
    "CS4_armreward": (
        [
            *BASEc,
            "reward_fraction_candidate_arm_last10_visits",
            "swr_x_reward_fraction_candidate_arm_last10_visits",
        ],
        "pooled",
    ),
    "CS4nt_armreward": (
        [
            *BASEc_NT,
            "reward_fraction_candidate_arm_last10_visits",
            "swr_x_reward_fraction_candidate_arm_last10_visits",
        ],
        "pooled",
    ),
    "CS5_armvalue": (
        [
            *BASEc,
            "reward_belief_mean_candidate_arm_preoutcome",
            "swr_x_reward_belief_mean_candidate_arm_preoutcome",
        ],
        "pooled",
    ),
    "CS5nt_armvalue": (
        [
            *BASEc_NT,
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
            *BASEc,
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
            *BASEc,
            "reward_belief_variance_candidate_arm_preoutcome_z",
            "swr_x_reward_belief_variance_candidate_arm_preoutcome_z",
        ],
        "pooled",
    ),
    "CS6nt_armuncert": (
        [
            *BASEc_NT,
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
LEAF_ONLY_N_CATEGORIES = 6
# Arm fixed effects for a leaf-only fit: five dummies for six leaves, arm 5 the
# reference. The nine-arm set (is_candidate_arm_0..7) is rank-deficient once the trunks are
# gone -- is_candidate_arm_6 and is_candidate_arm_7 are then identically zero -- so a leaf-only rung
# carrying it does not fit at all.
LEAF_ONLY_ARM_FE_PREDICTORS: tuple[str, ...] = tuple(
    f"is_candidate_arm_{i}" for i in range(5)
)


def leaf_only_predictors(predictors) -> list[str]:
    """Drop the arm fixed effects that are structurally zero on a leaf-only fit."""
    dropped = set(ARM_FE_PREDICTORS) - set(LEAF_ONLY_ARM_FE_PREDICTORS)
    return [p for p in predictors if p not in dropped]


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
