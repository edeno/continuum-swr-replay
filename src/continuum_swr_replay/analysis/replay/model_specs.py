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

The design notes below concern the conditional-Poisson **arm and
patch** hierarchies. Their defining constraint: only arm-varying
predictors are identifiable — event-level constants (time_since_stop,
is_swr, etc.) cancel in the within-event normalization and appear
only as interactions with arm-varying predictors. That rule does NOT
apply to ``MODEL_PER_STOP_RATE`` (ordinary Poisson, carries an intercept) or
``MODEL_CHOICE``; see their own sections.

Design decisions informed by exploratory analysis on 7 j16 sessions:

- Arm fixed effects (``arm_fe_0`` … ``arm_fe_7``, 8 dummies with arm
  8 as reference) enter as a baseline model ``M0a`` and are carried
  by every richer model. They absorb each arm's static replay
  occupancy so ``is_current`` / ``is_prev`` / ``is_last_rewarded``
  measure bias *above* an arm's own baseline rather than being
  inflated by it — important in camp-heavy sessions where the
  animal sits at one well and ``is_current`` then concentrates on a
  single arm. See :data:`ARM_FE_PREDICTORS`.
- ``reward_rate_arm_k10`` (rewarded fraction of the last 10 visits to
  an arm) is used as the reward-history predictor at M3/M4 rather
  than ``cumulative_reward`` (a raw reward *count*). The count
  conflates value with visit frequency — a frequently-visited arm
  scores high even at a mediocre reward rate — so its M3 coefficient
  partly re-encodes spatial recency. ``reward_rate_arm_k10`` is a
  true rate. Cost: it is undefined (NaN) for never-visited arms, so
  the M3+ complete-case event set is smaller than M0–M2's.
- M4 uses ``visit_recency_demeaned`` — ``visit_recency`` with each
  arm's session-mean subtracted — not the raw column. Raw
  ``visit_recency`` has a large static per-arm component (some arms
  are visited far more than others) that the ``arm_fe`` block
  already absorbs; with arm FE in the model the two collide (raw
  ``visit_recency`` VIF ≈ 27). Demeaning within arm removes the part
  ``arm_fe`` carries, leaving the identifiable signal: whether an arm
  was visited longer ago than is typical for that arm.
- ``is_last_rewarded`` is NOT used as a predictor: it is ~98%
  identical to ``is_current`` (the animal is usually rewarded where
  it currently sits), so a model carrying both is near-collinear
  (VIF tail >100, occasional rank-deficient non-convergence). M5/M6
  instead use ``is_last_rewarded_elsewhere`` = the last-rewarded arm
  restricted to events where it differs from the current arm.
  ``is_current`` (M1) already carries the rewarded-at-current-arm
  case, so the "elsewhere" indicator is exactly the identifiable
  residual — reward memory at a non-current arm — with disjoint
  support from ``is_current``. The trade is honest: it is sparse
  (~2% of events), so its coefficient has low power rather than a
  falsely-confident inflated estimate.
- ``is_next`` excluded: the alternation structure makes
  ``prev_arm == next_arm`` on >90% of events.
- ``current_x_last_rewarded`` excluded: VIF 97-100 with both
  ``is_current`` and ``is_last_rewarded`` (98% overlap in typical
  sessions). Causes convergence failures.
- ``dist_to_arm`` and ``is_same_patch`` excluded: proven
  uninformative (null when estimated independently; VIF 28 and 12
  with ``is_current``). The spatial effect is binary "here vs
  not here," not a distance gradient.
- SWR interactions limited to spatial predictors only
  (``swr_x_current``, ``swr_x_prev``). ``swr_x_last_rewarded`` was
  dropped because it was r=1.0 with ``swr_x_current`` under the old
  ``is_last_rewarded`` predictor; with ``is_last_rewarded_elsewhere``
  a ``swr_x_last_rewarded_elsewhere`` term would now be identifiable
  but is left out — at ~2% event support its SWR interaction is too
  sparse to power.
- MUA interactions are a separate analysis (split, not interaction)
  to avoid the same collinearity.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ARM_FE_PREDICTORS",
    "BIN_S",
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
    "TMAX",
]


# ── Arm fixed effects ──
#
# Eight indicator columns ``arm_fe_0`` … ``arm_fe_7`` (``arm_fe_k`` =
# 1 iff ``arm == k``). Arm 8 is the dropped reference category — with
# all nine dummies the set is collinear with the conditional
# likelihood's per-event normalization. These are generated at fit
# time from the canonical ``arm`` column by
# ``_helpers.add_arm_fixed_effects`` (they are not stored in the
# canonical event-arm table).
ARM_FE_PREDICTORS: tuple[str, ...] = tuple(f"arm_fe_{a}" for a in range(8))


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
# also drops more small sessions now (M6 needs ≥30 events).
#
# M5/M6 use ``is_last_rewarded_elsewhere`` rather than the raw
# ``is_last_rewarded``. The raw indicator overlaps ``is_current``
# ~98%; on the full dataset that drove M5/M6 max VIF into the 100s
# with 3/218 non-convergences. The "elsewhere" indicator has
# disjoint support from ``is_current`` so the collinearity is gone,
# at the cost of being sparse (a session with no current≠last-
# rewarded events yields an all-zero column and M5/M6 are then
# unidentifiable for that session — fit returns None, as before,
# but for a clear reason). Convergence / VIF must be re-validated
# on the regenerated tables.
#
# SWR interactions enter at M6 / M7. M6 carries the *spatial* × SWR
# interactions (swr_x_current, swr_x_prev) — does the local/previous
# bias differ for SWR-overlapping events? M7 adds the *value/memory*
# × SWR interactions (swr_x_reward_rate_arm_k10,
# swr_x_visit_recency_demeaned, swr_x_is_last_rewarded_elsewhere) —
# does the reward/memory structure of replay differ on SWR events?
# Splitting M6 from M7 lets the nested LR cleanly separate "spatial
# is SWR-modulated" from "reward memory is SWR-modulated". The M7
# interactions are sparser than M6's (the reward-memory one is on
# ~2% of events × is_swr subset); honest non-identifiability is
# expected in sessions with no SWR-overlap memory events.
#
# An SWR split (fit M0–M7 separately on is_swr=True and is_swr=False
# event subsets) is the unrestricted complement to these interaction
# terms — same pattern documented for MUA below.
#
# MUA event-type effects are assessed via split analysis (fit M0–M7
# separately on MUA-only and non-MUA events) rather than interaction
# terms, because mua_x_current ≈ mua_x_last_rewarded (r = 0.98,
# VIF > 70).

_M1 = (*ARM_FE_PREDICTORS, "is_current")
_M2 = (*_M1, "is_prev")
_M3 = (*_M2, "reward_rate_arm_k10")
_M4 = (*_M3, "visit_recency_demeaned")
_M5 = (*_M4, "is_last_rewarded_elsewhere")
_M6 = (*_M5, "swr_x_current", "swr_x_prev")
_M7 = (
    *_M6,
    "swr_x_reward_rate_arm_k10",
    "swr_x_visit_recency_demeaned",
    "swr_x_is_last_rewarded_elsewhere",
)
# Value block: per-arm nominal value (p_reward_arm) + experienced leaky-Q level
# (q_arm) as conditional main effects on the M5 spatial/reward-memory base
# (covariate_dictionary.md D2/D3). Additive to M5 -- M0-M7 are unchanged.
_Mv = (*_M5, "p_reward_arm", "q_arm")

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
        description="+ visit recency (within-arm demeaned)",
        parent="M3",
    ),
    "M5": ModelSpec(
        name="M5",
        predictors=_M5,
        description="+ last-rewarded arm (when not the current arm)",
        parent="M4",
    ),
    "M6": ModelSpec(
        name="M6",
        predictors=_M6,
        description="+ SWR × spatial interactions",
        parent="M5",
    ),
    "M7": ModelSpec(
        name="M7",
        predictors=_M7,
        description="+ SWR × value/memory interactions",
        parent="M6",
    ),
    "Mv": ModelSpec(
        name="Mv",
        predictors=_Mv,
        description="+ per-arm nominal value (p_reward_arm) + experienced "
        "leaky-Q level (q_arm)",
        parent="M5",
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
# patch?) and ``cumulative_reward_patch`` (= sum of rewards across
# both wells in this patch over the last 10 trials).
#
# P2/P3 use ``is_last_rewarded_patch_elsewhere`` rather than the raw
# ``is_last_rewarded_patch``. The raw indicator overlaps
# ``is_current_patch`` heavily (the animal is usually rewarded in
# the patch it currently occupies) — the same near-collinearity the
# arm hierarchy avoids with ``is_last_rewarded_elsewhere``. The
# "elsewhere" form is the last-rewarded patch restricted to events
# where it is not the current patch; ``is_current_patch`` (P1)
# already carries the rewarded-in-current-patch case, so this is the
# identifiable residual with disjoint support. ``MODEL_PATCH_SWITCH``
# remains the dedicated test of the previous-vs-current contrast.

# ── Event-source-restricted hierarchies (ContFrag pipeline) ──
#
# When events are externally restricted to SWR or MUA windows, the
# corresponding flag is constant by construction (every SWR-event has
# is_swr == 1; every MUA-event has is_mua == 1) so that flag's
# interactions with arm-varying predictors collapse onto the main
# effects and are unidentifiable. The hierarchy therefore mirrors
# MODEL_HIERARCHY's M0–M5 and stops there.
#
# A cross-modality M6 (mua_x_* inside SWR-events, swr_x_* inside
# MUA-events) is potentially identifiable when the cross-modality
# flag varies, but should only be added once the ContFrag event_arm
# tables are built and the cross-modality variability is verified.

_M0_TO_M5 = {
    name: MODEL_HIERARCHY[name]
    for name in ("M0", "M0a", "M1", "M2", "M3", "M4", "M5")
}

MODEL_HIERARCHY_SWR: dict[str, ModelSpec] = dict(_M0_TO_M5)
MODEL_HIERARCHY_MUA: dict[str, ModelSpec] = dict(_M0_TO_M5)


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
        predictors=("is_current_patch", "is_last_rewarded_patch_elsewhere"),
        description="+ last-rewarded patch (when not the current patch)",
        parent="P1",
    ),
    "P3": ModelSpec(
        name="P3",
        predictors=(
            "is_current_patch",
            "is_last_rewarded_patch_elsewhere",
            "cumulative_reward_patch",
        ),
        description="+ patch-level reward history",
        parent="P2",
    ),
    "P4": ModelSpec(
        name="P4",
        predictors=(
            "is_current_patch",
            "is_last_rewarded_patch_elsewhere",
            "cumulative_reward_patch",
            "swr_x_current_patch",
            "swr_x_last_rewarded_patch_elsewhere",
        ),
        description="+ SWR × patch interactions (spatial + memory)",
        parent="P3",
    ),
    "Pv": ModelSpec(
        name="Pv",
        predictors=(
            "is_current_patch",
            "is_last_rewarded_patch_elsewhere",
            "cumulative_reward_patch",
            "p_reward_arm_patch",
            "q_arm_patch",
        ),
        description="+ per-patch nominal value (p_reward_arm_patch) + "
        "experienced leaky-Q level (q_arm_patch)",
        parent="P3",
    ),
}


# ── Event-source-restricted patch hierarchies (ContFrag pipeline) ──
#
# Mirror MODEL_PATCH's P0–P3 but drop P4: under ContFrag event-source
# restriction is_swr is constant, so the swr_x_* interactions collapse
# onto the main effects and are unidentifiable — same reason the arm
# hierarchy SWR/MUA versions stop at M5.

_P0_TO_P3 = {name: MODEL_PATCH[name] for name in ("P0", "P1", "P2", "P3")}
MODEL_PATCH_SWR: dict[str, ModelSpec] = dict(_P0_TO_P3)
MODEL_PATCH_MUA: dict[str, ModelSpec] = dict(_P0_TO_P3)

# ── Post-switch patch model ──
#
# Fit ONLY on events where is_after_switch == 1. On switch trials
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
            "swr_x_current_patch",
            "swr_x_prev_patch",
        ),
        description="+ SWR × patch interactions (switch events only)",
        parent="PS2",
    ),
}


# ── Rate models ──
#
# A DIFFERENT estimator from the arm/patch hierarchies: an ordinary
# Poisson on the per-stop event count (``n_events``) with a
# ``log(immobility_time)`` offset, fit by ``per_stop_rate_model.fit_per_stop_rate_model``.
# It carries an intercept, so — unlike the conditional Poisson —
# stop-level constants do NOT cancel and ``is_after_*`` are valid
# main effects.
#
# ``predictors=()`` for R0 means intercept-only (the offset is always
# included by the fitter). ``is_after_reward`` is intentionally absent
# from R1: it equals ``1 - is_after_omission`` wherever both are
# defined, so including it with the intercept is rank-deficient.
#
# R1 looks only one trial back. R2 adds ``reward_rate_trial_k10``
# (rewarded fraction of the last 10 trials) — the stop-level analog
# of the content hierarchy's reward-history axis (``reward_rate_arm_k10``
# at M3). R2 vs R1 tests whether replay *occurrence* tracks a deeper
# reward history, not just the immediately preceding trial; it is
# also the confound check for whether an M3 content effect is partly
# a rate effect. ``visit_recency`` has no stop-level analog (every
# trial visits some arm), so the rate hierarchy stops at R2.
#
# The hierarchy intentionally stops at R2. An ``is_swr_stop`` term
# (1 if any timebin in the stop had an SWR) was dropped: ``n_events``
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
        predictors=("is_after_omission", "is_after_switch"),
        description="+ previous-trial reward / switch context",
        parent="R0",
    ),
    "R2": ModelSpec(
        name="R2",
        predictors=(
            "is_after_omission",
            "is_after_switch",
            "reward_rate_trial_k10",
        ),
        description="+ recent reward history (rewarded fraction, last 10 trials)",
        parent="R1",
    ),
}

# ── Lagged reward-kernel rate model ──
#
# A non-nested reparameterization of R2: instead of one immediate-trial
# indicator (``is_after_omission``) plus a smoothed window
# (``reward_rate_trial_k10``) — which overlap (lag-1 sits inside the window)
# and conflate two timescales — model the reward outcome at each of the last
# ``N_REWARD_LAGS`` trials as its own predictor. Reading the per-lag rate-ratio
# kernel separates the dwell-confounded immediate effect (``reward_lag1`` sets
# the current stop's duration) from the dwell-clean reward *memory* at
# ``reward_lag>=2`` (earlier trials don't set the current stop's duration).
# ``reward_lag1`` also uses the strict goal-arrival cutoff, so it is always
# past-only — unlike R1's ``is_after_omission`` (built from trial start time),
# which can encode the *future* reward outcome for a stop during the approach
# phase. This is why the two disagree on ~15% of stops (inter-trial stops).
N_REWARD_LAGS = 10

MODEL_PER_STOP_RATE["Rlag"] = ModelSpec(
    name="Rlag",
    predictors=tuple(f"reward_lag{j}" for j in range(1, N_REWARD_LAGS + 1))
    + ("is_after_switch",),
    description="Lagged reward kernel: per-trial reward at lags 1-10 + switch",
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
# row per stop, ``log(immobility_time)`` dwell offset), every model here is one
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
# (M2r's continuous reward rate, M8's switch term) — carried into the per-bin
# design by aggregate_to_timebins. Shared by the fitter orchestrator and the
# model-checking plotter so the inline refit matches the persisted fits.
STOP_COVARIATES = ("reward_rate_trial_k10", "is_after_switch")


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
        Number of ``reward_lag{j}`` terms (0, 1, or ``N_REWARD_LAGS``). Past
        trial outcomes; omission == ``1 - reward`` (same regressor).
    reward_rate : bool
        Include ``reward_rate_trial_k10`` (continuous; an alternative
        encoding of reward history to the lag kernel).
    self_history_lags : int
        Number of ``onset_lag{j}`` terms (0 or ``N_HISTORY_LAGS``); the
        within-stop autoregressive / refractoriness block.
    switch : bool
        Include ``is_after_switch``.
    reward_time_interaction : bool
        Include ``reward_lag1 : bs(tc, ...)`` — does reward modulate
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

    def formula(self) -> str:
        """patsy formula (response ``count``; the offset is applied at fit time)."""
        terms: list[str] = []
        if self.time_spline:
            terms.append(_SPLINE)
        terms += [f"reward_lag{j}" for j in range(1, self.reward_lags + 1)]
        if self.reward_rate:
            terms.append("reward_rate_trial_k10")
        if self.self_history_basis:
            terms += [f"hist_rcos_{k}" for k in range(self.self_history_basis)]
        else:
            terms += [f"onset_lag{j}" for j in range(1, self.self_history_lags + 1)]
        if self.cross_history_basis:
            terms += [f"cross_hist_rcos_{k}" for k in range(self.cross_history_basis)]
        else:
            terms += [f"cross_lag{j}" for j in range(1, self.cross_history_lags + 1)]
        if self.switch:
            terms.append("is_after_switch")
        if self.reward_time_interaction:
            terms.append(f"reward_lag1:{_SPLINE}")
        terms += list(self.extra_covariates)
        return "count ~ " + (" + ".join(terms) if terms else "1")


# (time, rewL, rate, hist, switch, interact)
_N = N_REWARD_LAGS
MODEL_PER_BIN_RATE: dict[str, PerBinRateModelSpec] = {
    "M0_const": PerBinRateModelSpec(
        "M0_const", False, 0, False, 0, False, False,
        "homogeneous Poisson baseline", None),
    "M1_rew1": PerBinRateModelSpec(
        "M1_rew1", False, 1, False, 0, False, False,
        "+ previous reward (reward_lag1)", "M0_const"),
    "M2_rewhist": PerBinRateModelSpec(
        "M2_rewhist", False, _N, False, 0, False, False,
        "+ reward history kernel (reward_lag1..10)", "M1_rew1"),
    "M2r_rewrate": PerBinRateModelSpec(
        "M2r_rewrate", False, 0, True, 0, False, False,
        "+ recent reward rate (continuous; alt to kernel)", "M0_const"),
    "M3_hist": PerBinRateModelSpec(
        "M3_hist", False, 0, False, N_HISTORY_LAGS, False, False,
        "+ self-history (autoregressive)", "M0_const"),
    "M4_hist_rew": PerBinRateModelSpec(
        "M4_hist_rew", False, _N, False, N_HISTORY_LAGS, False, False,
        "+ self-history + reward history", "M3_hist"),
    "M5_time": PerBinRateModelSpec(
        "M5_time", True, 0, False, 0, False, False,
        "+ time-since-stop (spline)", "M0_const"),
    "M6_time_hist": PerBinRateModelSpec(
        "M6_time_hist", True, 0, False, N_HISTORY_LAGS, False, False,
        "+ time + self-history", "M5_time"),
    "M7_time_hist_rew": PerBinRateModelSpec(
        "M7_time_hist_rew", True, _N, False, N_HISTORY_LAGS, False, False,
        "+ time + self-history + reward history", "M6_time_hist"),
    "M8_full_switch": PerBinRateModelSpec(
        "M8_full_switch", True, _N, False, N_HISTORY_LAGS, True, False,
        "+ switch (full additive)", "M7_time_hist_rew"),
    "M9_rew_x_time": PerBinRateModelSpec(
        "M9_rew_x_time", True, _N, False, N_HISTORY_LAGS, False, True,
        "M7 + reward x time interaction", "M7_time_hist_rew"),
    # Raised-cosine self-history variants: smooth basis over the same window so
    # the history block cannot separate on the refractory bins (lifts held-out
    # cv_bits_per_event from negative to positive vs the onset_lag dummies).
    "M3r_histcos": PerBinRateModelSpec(
        "M3r_histcos", False, 0, False, N_HISTORY_LAGS, False, False,
        "+ self-history (raised-cosine basis)", "M0_const",
        self_history_basis=N_HISTORY_BASIS),
    "M7r_time_histcos_rew": PerBinRateModelSpec(
        "M7r_time_histcos_rew", True, _N, False, N_HISTORY_LAGS, False, False,
        "+ time + raised-cosine self-history + reward history", "M5_time",
        self_history_basis=N_HISTORY_BASIS),
}


# ── Choice-prediction models ──
#
# A DIFFERENT estimator again: a conditional logit on the animal's
# next-arm choice (``choice_prediction.fit_choice_model``), used to
# ask whether replay content predicts behaviour beyond a behavioural
# baseline.
#
# These three are not a single linear nested chain: ``C_full`` nests
# both ``C_behavior`` (+ ``replay_frac``) and ``C_replay_only``
# (+ the behavioural block). ``parent`` encodes the primary test —
# ``C_full`` vs ``C_behavior``, i.e. does replay add predictive power
# beyond behaviour — while ``C_replay_only`` is a standalone
# replay-only reference (``parent=None``).
#
# Two predictors are deliberately excluded from the behavioural block:
# - ``is_same_patch``: a coarse binary of ``dist_to_arm`` (same-patch
#   arms are close); the two were near-collinear (VIF > 700).
# - ``dist_to_arm`` itself: the animal chose the closest arm in ~95-99%
#   of stops (median 0.99 in sessions that previously failed to fit),
#   producing quasi-separation that crashed ``ConditionalLogit.fit`` in
#   ~25% of sessions. Dropping it restores identifiability across all
#   sessions and de-saturates the baseline so ``C_full`` vs
#   ``C_behavior`` has room to detect a replay effect. Trade-off: the
#   replay coefficient is no longer cleanly controlled for distance,
#   and ``replay_frac`` is correlated with low ``dist_to_arm`` (replay
#   biases toward the current/closest arm), so a positive
#   ``replay_frac`` coefficient could partly launder distance. A
#   switch-style restriction (fit only on stops where the animal did
#   not pick the closest arm) is the principled alternative; defer.
# Callers should still drop any column absent from the choice dataset
# before fitting.

_CHOICE_BEHAVIOR = (
    "visit_recency",
    "cumulative_reward",
    "reward_rate_arm_k10",
    "arm_dist_rank",  # ranked choice-set distance (identifiable; replaces the
    # quasi-separating raw dist_to_arm as the geometric baseline, dictionary §4)
)

MODEL_CHOICE: dict[str, ModelSpec] = {
    "C_replay_only": ModelSpec(
        name="C_replay_only",
        predictors=("replay_frac",),
        description="Replay content only",
        parent=None,
    ),
    "C_behavior": ModelSpec(
        name="C_behavior",
        predictors=_CHOICE_BEHAVIOR,
        description="Behavioural baseline (recency, reward history)",
        parent=None,
    ),
    "C_full": ModelSpec(
        name="C_full",
        predictors=(*_CHOICE_BEHAVIOR, "replay_frac"),
        description="Behavioural baseline + replay content",
        parent="C_behavior",
    ),
    "C_full_swr_split": ModelSpec(
        name="C_full_swr_split",
        predictors=(*_CHOICE_BEHAVIOR, "replay_frac_swr", "replay_frac_non_swr"),
        description=(
            "Behavioural baseline + SWR-overlap and non-SWR replay separately"
        ),
        parent="C_behavior",
    ),
}
