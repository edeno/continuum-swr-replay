"""Model hierarchy definitions for replay content analysis.

Defines a sequence of nested models for the conditional Poisson
(multinomial-count) framework. Only arm-varying predictors appear as
main effects; event-level constants (time_since_stop, is_swr, etc.)
appear only in interactions with arm-varying predictors, because they
cancel in the conditional likelihood's within-event normalization.

Design decisions informed by exploratory analysis on 7 j16 sessions:

- Arm fixed effects (``arm_fe_0`` … ``arm_fe_7``, 8 dummies with arm
  8 as reference) enter as a baseline model ``M0a`` and are carried
  by every richer model. They absorb each arm's static replay
  occupancy so ``is_current`` / ``is_prev`` / ``is_last_rewarded``
  measure bias *above* an arm's own baseline rather than being
  inflated by it — important in camp-heavy sessions where the
  animal sits at one well and ``is_current`` then concentrates on a
  single arm. See :data:`ARM_FE_PREDICTORS`.
- ``recent_reward_rate`` (rewarded fraction of the last 5 visits to
  an arm) is used as the reward-history predictor at M3/M4 rather
  than ``cumulative_reward`` (a raw reward *count*). The count
  conflates value with visit frequency — a frequently-visited arm
  scores high even at a mediocre reward rate — so its M3 coefficient
  partly re-encodes spatial recency. ``recent_reward_rate`` is a
  true rate. Cost: it is undefined (NaN) for never-visited arms, so
  the M3+ complete-case event set is smaller than M0–M2's.
- ``is_next`` excluded: the alternation structure makes
  ``prev_arm == next_arm`` on >90% of events.
- ``current_x_last_rewarded`` excluded: VIF 97-100 with both
  ``is_current`` and ``is_last_rewarded`` (98% overlap in typical
  sessions). Causes convergence failures.
- ``dist_to_arm`` and ``is_same_patch`` excluded: proven
  uninformative (null when estimated independently; VIF 28 and 12
  with ``is_current``). The spatial effect is binary "here vs
  not here," not a distance gradient.
- SWR interactions limited to spatial predictors only:
  ``swr_x_last_rewarded`` is r=1.0 with ``swr_x_current`` (because
  ``is_current ≈ is_last_rewarded``), so it's dropped.
- MUA interactions are a separate analysis (split, not interaction)
  to avoid the same collinearity.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ARM_FE_PREDICTORS",
    "MODEL_HIERARCHY",
    "MODEL_HIERARCHY_MUA",
    "MODEL_HIERARCHY_SWR",
    "MODEL_PATCH",
    "MODEL_PATCH_MUA",
    "MODEL_PATCH_SWITCH",
    "MODEL_PATCH_SWR",
    "ModelSpec",
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
# also drops more small sessions now (M6 needs ≥30 events). These
# claims must be re-validated on the full dataset; do not assume the
# old 7/7 convergence still holds.
#
# SWR interactions enter at M6. These test whether the current-arm
# and previous-arm biases differ during SWR events. Only spatial
# interactions are included (swr_x_current, swr_x_prev); reward
# interactions (swr_x_last_rewarded) are excluded due to perfect
# collinearity with swr_x_current.
#
# MUA event-type effects are assessed via split analysis (fit M0–M6
# separately on MUA-only and non-MUA events) rather than interaction
# terms, because mua_x_current ≈ mua_x_last_rewarded (r = 0.98,
# VIF > 70).

_M1 = (*ARM_FE_PREDICTORS, "is_current")
_M2 = (*_M1, "is_prev")
_M3 = (*_M2, "recent_reward_rate")
_M4 = (*_M3, "visit_recency")
_M5 = (*_M4, "is_last_rewarded")
_M6 = (*_M5, "swr_x_current", "swr_x_prev")

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
        description="+ visit recency",
        parent="M3",
    ),
    "M5": ModelSpec(
        name="M5",
        predictors=_M5,
        description="+ last-rewarded arm",
        parent="M4",
    ),
    "M6": ModelSpec(
        name="M6",
        predictors=_M6,
        description="+ SWR × spatial interactions",
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
# COLLINEARITY CAVEAT (P2): ``is_last_rewarded_patch`` enters at P2
# alongside ``is_current_patch``. These suffer the same overlap as
# the arm-level ``is_current`` / ``is_last_rewarded`` pair (the
# animal is usually rewarded in the patch it currently occupies),
# and the overlap is *worse* at 3-patch granularity than at 9-arm
# granularity. The P2→P1 likelihood-ratio test is therefore only
# powered by stay-vs-switch and omission events, where the two
# patches diverge; on pure exploit stretches the P2 coefficient is
# weakly identified. ``MODEL_PATCH_SWITCH`` (below) is the clean
# test of the previous-vs-current contrast — it restricts to switch
# events where ``current_patch != prev_patch`` by construction.

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
        predictors=("is_current_patch", "is_last_rewarded_patch"),
        description="+ last-rewarded patch",
        parent="P1",
    ),
    "P3": ModelSpec(
        name="P3",
        predictors=(
            "is_current_patch",
            "is_last_rewarded_patch",
            "cumulative_reward_patch",
        ),
        description="+ patch-level reward history",
        parent="P2",
    ),
}


# ── Event-source-restricted patch hierarchies (ContFrag pipeline) ──
#
# MODEL_PATCH carries no event-type interaction, so the SWR / MUA
# patch hierarchies are identical in structure — the only difference
# is the event table they are fit on (ContFrag SWR-event vs MUA-event
# patch counts). They mirror MODEL_PATCH's full P0–P3, unlike the arm
# hierarchies which drop M6 (there is no patch-level M6 to drop).

MODEL_PATCH_SWR: dict[str, ModelSpec] = dict(MODEL_PATCH)
MODEL_PATCH_MUA: dict[str, ModelSpec] = dict(MODEL_PATCH)

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
}
