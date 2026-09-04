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

__all__ = [
    "ARM_FE_PREDICTORS",
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
    "PerBinRateModelSpec",
    "SPLINE_KNOTS",
    "STOP_COVARIATES",
    "SWR_CONTRAST_CONTENT_ARM_KEY_TERM",
    "SWR_CONTRAST_CONTENT_ARM_LADDER",
    "SWR_CONTRAST_CONTENT_BASEC",
    "SWR_CONTRAST_CONTENT_BASEC_PATCH",
    "SWR_CONTRAST_CONTENT_PATCH_KEY_TERM",
    "SWR_CONTRAST_CONTENT_PATCH_LADDER",
    "SWR_CONTRAST_RATE_G_SPECS",
    "SWR_CONTRAST_RATE_S_SPECS",
    "SWR_CONTRAST_RATE_STOP_COVARS",
    "TMAX",
    "default_content_model_names",
]


# ── Arm fixed effects ──
#
# Eight indicator columns ``is_candidate_arm_0`` … ``is_candidate_arm_7`` (``arm_fe_k`` =
# 1 iff ``arm == k``). Arm 8 is the dropped reference category — with
# all nine dummies the set is collinear with the conditional
# likelihood's per-event normalization. These are generated at fit
# time from the canonical ``arm`` column by
# ``_helpers.add_arm_fixed_effects`` (they are not stored in the
# canonical event-arm table).
ARM_FE_PREDICTORS: tuple[str, ...] = tuple(f"is_candidate_arm_{a}" for a in range(8))


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
# M0a (arm fixed effects) enters as a baseline below M1, and every
# model from M1 on carries the 8 arm-FE dummies so the spatial /
# reward predictors are estimated above each arm's static occupancy.
#
# CONVERGENCE / VIF NOTE: the pre-arm-FE hierarchy was tuned for
# max VIF < 3 through M4 and converged 7/7 on the j16 pilot. Adding
# arm fixed effects deliberately introduces competition between
# ``is_current`` and the dummy of whichever arm the animal occupies
# — VIF will rise (especially in camp-heavy sessions) and a session
# where the animal never leaves one well can become rank-deficient
# (is_current ≡ that arm's dummy) and fail to converge. The
# ``n_events < 2 * n_predictors`` gate in ``fit_conditional_poisson``
# also drops more small sessions now (M6 needs ≥26 events).
#
# There is no reward-memory rung. There was one -- M5 carried
# ``is_last_rewarded_elsewhere`` and the SWR rungs sat above it at M6
# and M7 -- and it was removed on 2026-08-18, with the SWR rungs
# renumbered down into M5 and M6.
#
# The predictor is not identifiable per-session, and it does not measure
# what its name says. It is an **omission-only** indicator: it cannot fire
# on a rewarded stop, because if the animal was just rewarded here then
# the last-rewarded arm is here and the residual is zero. It fires on 0.0%
# of rewarded at-well events and 67.4% of omission ones, and every event
# where it fires is an omission. So its support is capped by the ~10%
# omission rate, and any coefficient answers "within omission stops, is
# the previously-rewarded arm over- or under-represented" rather than
# anything about reward memory in general.
#
# On top of that it is not estimable per-session. It fires on 6.1% of
# events, and in 99 of 208 sessions no replay mass ever lands on the arm
# it marks, which is perfect separation: the coefficient runs to -inf
# and the standard error explodes (median -3.503, SE > 10 in 49% of
# sessions). Which sessions separate tracks their omission count (median
# 14 omission events, against 22 for the identified ones). Its SWR interaction was worse -- constant within every
# event in 58 sessions -- and because the ladder freezes one
# complete-case cohort over the union of all rungs, the refusals
# propagated to every rung including M0, fitting 145 of 215 sessions
# where 215 of 218 had fitted before.
#
# The raw ``is_last_rewarded`` cannot take its place: it correlates
# 0.919 with ``is_current``, which is the near-collinearity the
# "elsewhere" residual existed to remove. Reward memory therefore lives
# only in the pooled contrast rungs C5_lastrew / CS3_lastrew, where
# pooling across sessions supplies the within-event variation that a
# per-session conditional fit does not have.
#
# SWR interactions enter at M5 / M6. M5 carries the *spatial* × SWR
# interactions (``swr_x_is_current_arm``, ``swr_x_is_prev_arm``) — does the local/previous
# bias differ for SWR-overlapping events? M6 adds the *value* × SWR
# interactions (``swr_x_reward_fraction_candidate_arm_last10_visits`` and the
# centered-recency spline interactions) — does the reward structure of replay
# differ on SWR events? Splitting M5 from M6 lets the nested LR cleanly
# separate "spatial is SWR-modulated" from "reward is SWR-modulated".
#
# An SWR split (fit M0–M4 separately on is_swr=True and is_swr=False
# event subsets) is the unrestricted complement to these interaction
# terms — same pattern documented for MUA below.
#
# MUA event-type effects are assessed via split analysis (fit M0–M4
# separately on MUA-only and non-MUA events) rather than interaction
# terms, because mua_x_current ≈ mua_x_last_rewarded (r = 0.98,
# VIF > 70).

# Natural-cubic-spline basis for visit recency, on the raw trial-count scale.
#
# Recency is deterministic on the two arms that already have their own
# indicators: exactly 1 on the current arm and exactly 2 on the previous arm, on
# 100.00% of location-resolved events. A basis evaluated there is a fixed
# vector, hence exactly collinear with ``is_current`` + ``is_prev`` -- fitting it
# unmasked is rank-deficient. The basis is therefore masked to the arms where
# recency is not already determined (visited leaves that are neither current nor
# previous, where recency is >= 3 by construction), and one basis column is
# dropped as reference because a natural spline basis is a partition of unity
# and would otherwise sum to the mask.
#
# Knots are fixed globally rather than fitted per session: a per-session basis
# would give each session its own units and make coefficients incomparable
# across the medians the ladders aggregate over. 12 and 40 trials sit near the
# tertiles of the identified support (13 and 36), and are the best-conditioned
# of the placements tried.
#
# Raw rather than log1p, by preference for knots that read directly in trials.
# Measured trade-off: the two are indistinguishable on fit (median paired Δllf
# 5.6 on a ~22,700 session log-likelihood, log1p ahead in 13 of 30 sessions),
# but raw is less well conditioned -- within-event demeaned condition number
# 4.12 against 2.74, and the third basis column has SD 0.032 against the first's
# 0.386 because recency is right-skewed (median 21 on the support, max 198). If
# per-session instability in these columns becomes a problem, the log1p scale is
# the first thing to try: only these two constants change.
# Interior knot positions (in trial counts) and clamp bounds for the natural
# cubic spline on visit recency. The two knots (12, 40 trials) place flexibility
# across the bulk of the recency distribution; the bounds (3, 200) clamp
# extreme/right-censored recencies before the basis is evaluated. Values are
# empirical (chosen for the observed recency distribution), not derived — retune
# them together if the recency scale changes.
RECENCY_SPLINE_KNOTS: tuple[float, float] = (12.0, 40.0)
RECENCY_SPLINE_BOUNDS: tuple[float, float] = (3.0, 200.0)
RECENCY_SPLINE_COLUMNS: tuple[str, ...] = (
    "visit_recency_trials_event_leaf_centered_spline_1",
    "visit_recency_trials_event_leaf_centered_spline_2",
    "visit_recency_trials_event_leaf_centered_spline_3",
)


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
# There is no patch reward-memory rung, for the reason the arm ladder
# has none (see the M5 note above). ``is_last_rewarded_patch`` overlaps
# ``is_current_patch`` at r = 0.945, and the "elsewhere" residual that
# was meant to break that overlap fires on 2.9% of events and is
# perfectly separated in 77 of 184 sessions. At patch level it also
# carries no signal to recover: mean ``patch_count_hard`` is 14.76 on
# the marked patch against 14.81 elsewhere, where ``is_current_patch``
# shows 17.16 against 13.63.
#
# It cost more here than in the arm ladder. Carrying it in P2/P3 lost 31
# sessions and its SWR interaction in P4 lost 120, leaving 95 of 215 --
# the frozen-union cohort propagates each refusal to every rung. Removed
# 2026-08-18, P3 and P4 renumbered down into P2 and P3.
# ``MODEL_PATCH_SWITCH`` remains the dedicated test of the
# previous-vs-current contrast.

# ── Event-source-restricted hierarchies (ContFrag pipeline) ──
#
# When events are externally restricted to SWR or MUA windows, the
# corresponding flag is constant by construction (every SWR-event has
# is_swr == 1; every MUA-event has is_mua == 1) so that flag's
# interactions with arm-varying predictors collapse onto the main
# effects and are unidentifiable. The hierarchy therefore mirrors
# MODEL_HIERARCHY's M0–M4 and stops there.
#
# A cross-modality rung (mua_x_* inside SWR-events, swr_x_* inside
# MUA-events) is potentially identifiable when the cross-modality
# flag varies, but should only be added once the ContFrag event_arm
# tables are built and the cross-modality variability is verified.

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
# Results on 7 j16 sessions:
# - is_current_patch is STRONGER after switches (median +1.66 larger
#   coefficient, 7/7, p=0.016) — replay immediately represents the
#   new patch
# - is_prev_patch is reliably NEGATIVE (7/7, median −1.8) — the
#   patch the animal just left is actively suppressed, below even
#   the unrelated third patch

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
# R1 uses the newest outcome observable by bout start. R2 adds
# ``reward_fraction_last10_known_trials``
# (rewarded fraction of the last 10 trials) — the stop-level analog
# of the content hierarchy's reward-history axis
# (``reward_fraction_candidate_arm_last10_visits``
# at M3). R2 vs R1 tests whether replay *occurrence* tracks a deeper
# reward history, not just the newest known outcome; it is
# also the confound check for whether an M3 content effect is partly
# a rate effect. Candidate-arm visit recency has no stop-level analog (every
# trial visits some arm), so the rate hierarchy stops at R2.
#
# The hierarchy intentionally stops at R2. An ``is_swr_stop`` term
# (1 if any timebin in the stop had an SWR) was dropped: ``n_selected_events``
# counts non-local events, which overlap SWRs heavily by construction,
# so ``is_swr_stop`` is largely tautological with the response and says
# little about whether SWRs *drive* replay occurrence. The SWR-vs-non-SWR
# question is answered at the event level (the ``swr_x_*`` content
# interactions / SWR-split fits), not via the per-stop rate. A future
# R-level term (+ ``trials_since_block_change``, ``block_number``) is
# deferred until the contingency-block loader is wired.

MODEL_PER_STOP_RATE: dict[str, ModelSpec] = {
    "R0": ModelSpec(
        name="R0",
        predictors=(),
        description="Null (intercept-only baseline event rate)",
        parent=None,
    ),
    "R1": ModelSpec(
        name="R1",
        # ``known_reward_outcome_lag_1`` is the most recent outcome already observable at
        # bout start. It can be the arrived trip's outcome after its physical
        # feedback boundary or the preceding known outcome before that boundary;
        # it is past-only, but not a fixed trial-relative index.
        #
        # The immediate-switch flag was removed from this general-purpose
        # reward-history ladder: it is often constant within a session's
        # complete-case cohort, which makes the design rank-deficient and
        # rejects the whole session. See the naming spec's "Immediate switch
        # is not a nuisance adjustment" section; a dedicated switch analysis
        # uses ``patch_switch_fraction_last5_completed_trips`` instead.
        predictors=("known_reward_outcome_lag_1",),
        description="+ most-recent-known reward at bout start",
        parent="R0",
    ),
    "R2": ModelSpec(
        name="R2",
        predictors=(
            "known_reward_outcome_lag_1",
            "reward_fraction_last10_known_trials",
        ),
        description="+ recent reward history (rewarded fraction, last 10 trials)",
        parent="R1",
    ),
}

# ── Lagged reward-kernel rate model ──
#
# A non-nested reparameterization of R2: instead of one most-recent-known
# indicator (``known_reward_outcome_lag_1``) plus a smoothed window
# (``reward_fraction_last10_known_trials``) — which overlap (lag-1 sits inside the window)
# and conflate two timescales — model the reward outcome at each of the last
# ``N_REWARD_LAGS`` trials as its own predictor. Reading the per-lag rate-ratio
# kernel describes successive outcomes known by bout start. Lag 1 is the newest
# known outcome, not uniformly the current or previous trial: the mapping changes
# when the arrived trip crosses its physical outcome boundary. All lags are
# past-only, but coefficient labels must not imply a fixed trial index.
N_REWARD_LAGS = 10

MODEL_PER_STOP_RATE["Rlag"] = ModelSpec(
    name="Rlag",
    # Immediate-switch flag removed (see R1) -- a constant switch column within
    # a session's complete-case cohort makes this design rank-deficient.
    predictors=tuple(
        f"known_reward_outcome_lag_{j}" for j in range(1, N_REWARD_LAGS + 1)
    ),
    description="Lagged reward kernel: 10 most-recent-known outcomes",
    parent="R0",
)


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
# The four blocks (time-since-stop spline, self-history/autoregressive, reward
# [lag-1 / kernel / continuous rate], switch) toggle independently so each
# block's contribution is readable with and without the time course, plus the
# reward x time interaction. ALL models share one bin width ``BIN_S`` so the
# ladder is nested and AIC / CV / deviance / GOF are directly comparable (fine
# bins are required anyway for the self-history term and time-rescaling GOF).
BIN_S = 0.015  # 15 ms — ladder-wide (refractory <60 ms, bursty ~100 ms)
TMAX = 4.0  # model only the first 4 s of each stop (see overview Non-Goals)
# Fixed interior knots (seconds) for the within-stop time-course spline, with
# boundary knots pinned to [0, TMAX]. Fixed (not data-driven quantile) knots make
# the spline basis a deterministic function of tc: the fitted curve is
# reconstructable from the saved coefficients (rebuild bs with these knots), the
# spline coefficients are comparable across sessions (same basis everywhere), and
# held-out CV never extrapolates beyond the training knots. Cubic bs over [0, 4]
# with 3 interior knots -> 6 spline columns.
SPLINE_KNOTS = (1.0, 2.0, 3.0)
_SPLINE = f"bs(tc, knots={SPLINE_KNOTS}, lower_bound=0.0, upper_bound={TMAX})"
N_HISTORY_LAGS = 10  # onset_lag1..10 = 150 ms self-history window
N_HISTORY_BASIS = 4  # raised-cosine bumps over the history window (*_histcos models)
# Per-stop covariates the ladder needs beyond the reward/self-history lags
# (M2r's continuous reward rate) — carried into the per-bin design by
# aggregate_to_timebins. Shared by the fitter orchestrator and the
# model-checking plotter so the inline refit matches the persisted fits.
# The immediate-switch term was retired with the M8 rung (see R1).
STOP_COVARIATES = ("reward_fraction_last10_known_trials",)


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
    reward_lags : int
        Number of ``known_reward_outcome_lag_{j}`` terms (0, 1, or ``N_REWARD_LAGS``). Past
        trial outcomes; omission == ``1 - reward`` (same regressor).
    reward_rate : bool
        Include ``reward_fraction_last10_known_trials`` (continuous; an alternative
        encoding of reward history to the lag kernel).
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
        Include ``known_reward_outcome_lag_1 : bs(tc, ...)`` — does reward modulate
        the time course?
    description : str
        Human-readable description.
    parent : str | None
        Name of the parent model for nested comparison.
    """

    name: str
    time_spline: bool
    reward_lags: int
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

    def __post_init__(self) -> None:
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
        terms: list[str] = []
        if self.time_spline:
            terms.append(_SPLINE)
        terms += [
            f"known_reward_outcome_lag_{j}" for j in range(1, self.reward_lags + 1)
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
            terms.append(f"known_reward_outcome_lag_1:{_SPLINE}")
        terms += list(self.extra_covariates)
        return "count ~ " + (" + ".join(terms) if terms else "1")


_N = N_REWARD_LAGS
# Each spec is built with keyword arguments so the toggle each model turns on is
# readable in place (which term the rung adds vs. its parent), rather than being
# decoded from argument position. ``switch`` is retired and stays False.
MODEL_PER_BIN_RATE: dict[str, PerBinRateModelSpec] = {
    "M0_const": PerBinRateModelSpec(
        name="M0_const",
        time_spline=False,
        reward_lags=0,
        reward_rate=False,
        self_history_lags=0,
        switch=False,
        reward_time_interaction=False,
        description="homogeneous Poisson baseline",
        parent=None,
    ),
    "M1_rew1": PerBinRateModelSpec(
        name="M1_rew1",
        time_spline=False,
        reward_lags=1,
        reward_rate=False,
        self_history_lags=0,
        switch=False,
        reward_time_interaction=False,
        description="+ most-recent-known reward at bout start (known_reward_outcome_lag_1)",
        parent="M0_const",
    ),
    "M2_rewhist": PerBinRateModelSpec(
        name="M2_rewhist",
        time_spline=False,
        reward_lags=_N,
        reward_rate=False,
        self_history_lags=0,
        switch=False,
        reward_time_interaction=False,
        description="+ reward history kernel (known_reward_outcome_lag_1..10)",
        parent="M1_rew1",
    ),
    "M2r_rewrate": PerBinRateModelSpec(
        name="M2r_rewrate",
        time_spline=False,
        reward_lags=0,
        reward_rate=True,
        self_history_lags=0,
        switch=False,
        reward_time_interaction=False,
        description="+ recent reward rate (continuous; alt to kernel)",
        parent="M0_const",
    ),
    "M3_hist": PerBinRateModelSpec(
        name="M3_hist",
        time_spline=False,
        reward_lags=0,
        reward_rate=False,
        self_history_lags=N_HISTORY_LAGS,
        switch=False,
        reward_time_interaction=False,
        description="+ self-history (autoregressive)",
        parent="M0_const",
    ),
    "M4_hist_rew": PerBinRateModelSpec(
        name="M4_hist_rew",
        time_spline=False,
        reward_lags=_N,
        reward_rate=False,
        self_history_lags=N_HISTORY_LAGS,
        switch=False,
        reward_time_interaction=False,
        description="+ self-history + reward history",
        parent="M3_hist",
    ),
    "M5_time": PerBinRateModelSpec(
        name="M5_time",
        time_spline=True,
        reward_lags=0,
        reward_rate=False,
        self_history_lags=0,
        switch=False,
        reward_time_interaction=False,
        description="+ time-since-stop (spline)",
        parent="M0_const",
    ),
    "M6_time_hist": PerBinRateModelSpec(
        name="M6_time_hist",
        time_spline=True,
        reward_lags=0,
        reward_rate=False,
        self_history_lags=N_HISTORY_LAGS,
        switch=False,
        reward_time_interaction=False,
        description="+ time + self-history",
        parent="M5_time",
    ),
    "M7_time_hist_rew": PerBinRateModelSpec(
        name="M7_time_hist_rew",
        time_spline=True,
        reward_lags=_N,
        reward_rate=False,
        self_history_lags=N_HISTORY_LAGS,
        switch=False,
        reward_time_interaction=False,
        description="+ time + self-history + reward history",
        parent="M6_time_hist",
    ),
    # M8_full_switch (the per-bin switch main-effect extension) was retired:
    # The completed-trip switch flag is often constant within a session's complete-case
    # cohort, making the design rank-deficient. See the naming spec's
    # "Immediate switch is not a nuisance adjustment" section. Nothing parents
    # off it (M9 parents off M7), so removal leaves the ladder intact.
    "M9_rew_x_time": PerBinRateModelSpec(
        name="M9_rew_x_time",
        time_spline=True,
        reward_lags=_N,
        reward_rate=False,
        self_history_lags=N_HISTORY_LAGS,
        switch=False,
        reward_time_interaction=True,
        description="M7 + reward x time interaction",
        parent="M7_time_hist_rew",
    ),
    # Raised-cosine self-history variants: smooth basis over the same window so
    # the history block cannot separate on the refractory bins (lifts held-out
    # cv_bits_per_event from negative to positive vs the onset_lag dummies).
    "M3r_histcos": PerBinRateModelSpec(
        name="M3r_histcos",
        time_spline=False,
        reward_lags=0,
        reward_rate=False,
        self_history_lags=N_HISTORY_LAGS,
        switch=False,
        reward_time_interaction=False,
        description="+ self-history (raised-cosine basis)",
        parent="M0_const",
        self_history_basis=N_HISTORY_BASIS,
    ),
    "M7r_time_histcos_rew": PerBinRateModelSpec(
        name="M7r_time_histcos_rew",
        time_spline=True,
        reward_lags=_N,
        reward_rate=False,
        self_history_lags=N_HISTORY_LAGS,
        switch=False,
        reward_time_interaction=False,
        description="+ time + raised-cosine self-history + reward history",
        parent="M5_time",
        self_history_basis=N_HISTORY_BASIS,
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
# - ``is_same_patch``: a coarse binary of ``distance_to_candidate_arm_cm`` (same-patch
#   arms are close); the two were near-collinear (VIF > 700).
# - ``distance_to_candidate_arm_cm`` itself: the animal chose the closest arm in ~95-99%
#   of stops (median 0.99 in sessions that previously failed to fit),
#   producing quasi-separation that crashed ``ConditionalLogit.fit`` in
#   ~25% of sessions. Dropping it restores identifiability across all
#   sessions and de-saturates the baseline so ``C_full`` vs
#   ``C_behavior`` has room to detect a replay effect. Trade-off: the
#   replay coefficient is no longer cleanly controlled for distance,
#   and ``replay_fraction`` is correlated with low candidate-arm distance (replay
#   biases toward the current/closest arm), so a positive
#   ``replay_fraction`` coefficient could partly launder distance. A
#   switch-style restriction (fit only on stops where the animal did
#   not pick the closest arm) is the principled alternative; defer.
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
_REWK_PREV = " + ".join(
    f"known_reward_outcome_lag_{j}" for j in range(2, N_REWARD_LAGS + 1)
)
_HISTCOS = " + ".join(f"hist_rcos_{k}" for k in range(N_HISTORY_BASIS))
_BLOCK_F = {
    "BASE": f"{_SPLINE} + known_reward_outcome_lag_1 + {_REWK_PREV} + {_HISTCOS}",
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
    "REWXTIME": f"known_reward_outcome_lag_1:{_SPLINE}",  # reward x within-stop time course (dwell-shape)
}
_BLOCK_COLS = {
    "BASE": [
        "tc",
        *[f"known_reward_outcome_lag_{j}" for j in range(1, N_REWARD_LAGS + 1)],
        *[f"hist_rcos_{k}" for k in range(N_HISTORY_BASIS)],
    ],
    "PERISWITCH": ["periswitch_index_trials", "is_periswitch_post"],
    "BLOCK": ["trials_since_block_change"],
    "SWITCHRATE": ["patch_switch_fraction_last5_completed_trips"],
    "VALUE": ["relative_value_switch_minus_stay_preoutcome"],
    "RPE": ["rpe_postoutcome"],
    "UNCERT": ["reward_belief_differential_entropy_leaf_mean_preoutcome"],
    "UPDATE": ["reward_belief_mean_update_l1_all_ports_postoutcome"],
    "REWXTIME": ["known_reward_outcome_lag_1", "tc"],
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
    return {n: (_blocks_formula(b), _blocks_cols(b)) for n, b in ladders.items()}


def _s_specs():
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
        "S1_rew1": (f"{base} + is_swr + is_swr:known_reward_outcome_lag_1", ["BASE"]),
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
    return {n: (f, _blocks_cols(bl)) for n, (f, bl) in raw.items()}


SWR_CONTRAST_RATE_G_SPECS = _g_specs()
SWR_CONTRAST_RATE_S_SPECS = _s_specs()

# --- Content is_swr-contrast ladder (conditional Poisson over 9 arms / 3 patches) ---
# name -> (predictors, scope)  scope: "split" (per is_swr split) | "pooled" (contrast)
ARMFE = [f"is_candidate_arm_{a}" for a in range(8)]
# ``candidate_arm_distance_rank`` is deliberately absent. It takes five tied
# values over a fixed track topology, its rank-1 level is *identical* to
# ``is_current_arm``
# (agreement 1.000000 across 59,374 location-resolved events), and it ranks last
# in unique likelihood of any covariate here. Carrying it collapsed the
# current-arm coefficient from +0.705 (5/5 animals) to +0.102 (3/5) and doubled
# its standard error, making every C*/CS* ``is_current_arm`` non-comparable with
# ``M1``/``C1_current``. Its removal leaves the SWR interactions untouched
# (the SWR × candidate-arm belief-mean coefficient changed -2.390 -> -2.407).
# This also restores consistency with the
# module docstring above, which excludes ``distance_to_candidate_arm_cm`` and
# ``is_same_patch``
# on the same grounds -- candidate-arm distance rank re-imported most of their
# content.
BASEc = [
    *ARMFE,
    "is_current_arm",
    "is_prev_arm",
    *RECENCY_SPLINE_COLUMNS,
    "is_candidate_arm_never_visited",
]
_TSC = "log1p_time_since_stop_start_s_z_x_is_current_arm"
SWR_CONTRAST_CONTENT_ARM_LADDER: dict[str, tuple[list[str], str]] = {
    "C0_armfe": (ARMFE, "split"),
    "C1_current": ([*ARMFE, "is_current_arm"], "split"),
    "C2_base": (BASEc, "split"),
    "C3_armreward": ([*BASEc, "reward_fraction_candidate_arm_last10_visits"], "split"),
    "C4_armvalue": ([*BASEc, "reward_belief_mean_candidate_arm_preoutcome"], "split"),
    "C5_lastrew": ([*BASEc, "is_last_rewarded_noncurrent_arm"], "split"),
    "C6_rpe": ([*BASEc, _TSC, "rpe_postoutcome_z_x_is_current_arm"], "split"),
    "C7_value": (
        [
            *BASEc,
            _TSC,
            "relative_value_switch_minus_stay_preoutcome_z_x_is_current_arm",
        ],
        "split",
    ),
    "C8_uncert": (
        [
            *BASEc,
            _TSC,
            "reward_belief_differential_entropy_leaf_mean_preoutcome_z_x_is_current_arm",
        ],
        "split",
    ),
    "C9_update": (
        [
            *BASEc,
            _TSC,
            "reward_belief_mean_update_l1_all_ports_postoutcome_z_x_is_current_arm",
        ],
        "split",
    ),
    "C10_periswitch": (
        [
            *BASEc,
            _TSC,
            "periswitch_index_trials_z_x_is_current_arm",
            "is_periswitch_post_x_is_current_arm",
            "periswitch_index_trials_z_x_is_periswitch_post_x_is_current_arm",
        ],
        "split",
    ),
    "C11_block": (
        [*BASEc, _TSC, "trials_since_block_change_z_x_is_current_arm"],
        "split",
    ),
    "C12_switchrate": (
        [
            *BASEc,
            _TSC,
            "patch_switch_fraction_last5_completed_trips_z_x_is_current_arm",
        ],
        "split",
    ),
    # CS1 includes the within-stop-time current-arm interaction
    # UNCONDITIONALLY so ``swr_x_is_current_arm`` is the SWR-specific
    # current-arm effect BEYOND the general time course (SWR and non-SWR events
    # differ in stop-time). CS1nt is the uncontrolled sensitivity twin; both
    # report the same canonical interaction term.
    "CS1_current": ([*BASEc, _TSC, "swr_x_is_current_arm"], "pooled"),
    "CS1nt_current": ([*BASEc, "swr_x_is_current_arm"], "pooled"),
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
    "CS5_armvalue": (
        [
            *BASEc,
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
    # value effect net of previous-goal. Fitted values are reported in
    # paper/claims_table.md (row 19f) — do not duplicate numbers here.
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
    "CS5_armvalue": "swr_x_reward_belief_mean_candidate_arm_preoutcome",
    "CS5b_valuevsprevgoal": "swr_x_reward_belief_mean_candidate_arm_preoutcome",
    "C13_armuncert": "reward_belief_variance_candidate_arm_preoutcome_z",
    "CS6_armuncert": "swr_x_reward_belief_variance_candidate_arm_preoutcome_z",
}
BASEc_PATCH = [
    "is_current_patch",
    "is_prev_patch",
    "visit_recency_trials_patch",
    "is_patch_never_visited",
]
# ``CP2_olddecay`` and ``CPS3_olddecay`` were retired 2026-08-18. Both crossed
# ``is_last_rewarded_noncurrent_patch`` with a peri-switch term, and
# the indicator correlates -0.96/-0.97 with its own interaction, so the design
# was rank-deficient in 83 and 87 of 215 sessions. What survived the |coef|<10
# filter was selection on the estimate: CP2 reported +4.83 with a typical
# session SE of 7,313, against +0.36 on the sessions that were actually
# identified. A 3%-support indicator crossed with a time term is not estimable
# per session in a 3-alternative model.
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
    # Time-controlled current-patch contrast + uncontrolled twin,
    # mirroring the arm CS1/CS1nt pair.
    "CPS1_current": (
        [
            *BASEc_PATCH,
            "log1p_time_since_stop_start_s_z_x_is_current_patch",
            "swr_x_is_current_patch",
        ],
        "pooled",
    ),
    "CPS1nt_current": ([*BASEc_PATCH, "swr_x_is_current_patch"], "pooled"),
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
SWR_CONTRAST_CONTENT_BASEC_PATCH = BASEc_PATCH

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
# Restricted to the standalone contrast rungs. The arm-ladder belief rungs
# (Mv/Mu/Mvu) are nested -- their parents are M4 and Mu, fitted over nine arms --
# and changing the candidate set changes the conditional likelihood, so a
# leaf-only Mv could not be compared to a nine-arm M4 by ΔAIC. Making those
# leaf-only would require a leaf-only parent rung as well.
LEAF_ONLY_CONTRAST_RUNGS = frozenset(
    {
        "C4_armvalue",
        "C13_armuncert",
        "CS5_armvalue",
        "CS5b_valuevsprevgoal",
        "CS6_armuncert",
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
