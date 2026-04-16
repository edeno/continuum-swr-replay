"""Model hierarchy definitions for replay content analysis.

Defines a sequence of nested models (M0-M8) for the conditional Poisson
(multinomial-count) framework. Only arm-varying predictors appear as main
effects; event-level constants (time_since_stop, is_swr, etc.) appear only
in interactions with arm-varying predictors, because they cancel in the
conditional likelihood's within-event normalization.

``is_next`` is excluded from the hierarchy because the Spatial Bandit
task's alternation structure (consecutive pokes at the same port are
never rewarded) makes ``prev_arm == next_arm`` on >90% of events.
``is_next`` remains in the canonical tables for post-hoc retrospective
analyses (Stage 5B) but does not enter the primary content model.

``current_x_last_rewarded`` enters at M3 to absorb the 58% overlap
between ``is_current`` and ``is_last_rewarded`` (the animal is often
AT the just-rewarded port during immobility). Without the interaction
their coefficients double-count the overlap.
"""

from __future__ import annotations

from dataclasses import dataclass


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


# ── Arm-varying main effects ──

_SPATIAL: tuple[str, ...] = ("is_current", "is_prev")

_REWARD_BINARY: tuple[str, ...] = (
    "is_last_rewarded",
    "current_x_last_rewarded",
)

_REWARD_HISTORY: tuple[str, ...] = (
    "visit_recency",
    "cumulative_reward",
)

_DISTANCE: tuple[str, ...] = ("dist_to_arm", "is_same_patch")

# ── Event-level × arm-varying interactions ──

_TEMPORAL_INTERACTIONS: tuple[str, ...] = (
    "time_x_current",
    "time_x_prev",
)

_SWR_INTERACTIONS: tuple[str, ...] = (
    "swr_x_current",
    "swr_x_prev",
    "swr_x_last_rewarded",
)

_MUA_INTERACTIONS: tuple[str, ...] = (
    "mua_x_current",
    "mua_x_prev",
    "mua_x_last_rewarded",
)

_CONTEXT_INTERACTIONS: tuple[str, ...] = (
    "switch_x_current",
    "switch_x_prev",
    "omission_x_last_rewarded",
)


MODEL_HIERARCHY: dict[str, ModelSpec] = {
    "M0": ModelSpec(
        name="M0",
        predictors=(),
        description="Null (uniform allocation across arms)",
        parent=None,
    ),
    "M1": ModelSpec(
        name="M1",
        predictors=("is_current",),
        description="Current-arm bias",
        parent="M0",
    ),
    "M2": ModelSpec(
        name="M2",
        predictors=_SPATIAL,
        description="Spatial context (current + previous)",
        parent="M1",
    ),
    "M3": ModelSpec(
        name="M3",
        predictors=_SPATIAL + _REWARD_BINARY,
        description="Most-recent reward (binary)",
        parent="M2",
    ),
    "M4": ModelSpec(
        name="M4",
        predictors=_SPATIAL + _REWARD_BINARY + _REWARD_HISTORY,
        description="Reward history (continuous)",
        parent="M3",
    ),
    "M5": ModelSpec(
        name="M5",
        predictors=_SPATIAL + _REWARD_BINARY + _REWARD_HISTORY + _DISTANCE,
        description="Distance control",
        parent="M4",
    ),
    "M6": ModelSpec(
        name="M6",
        predictors=(
            _SPATIAL
            + _REWARD_BINARY
            + _REWARD_HISTORY
            + _DISTANCE
            + _TEMPORAL_INTERACTIONS
        ),
        description="Temporal dynamics",
        parent="M5",
    ),
    "M7": ModelSpec(
        name="M7",
        predictors=(
            _SPATIAL
            + _REWARD_BINARY
            + _REWARD_HISTORY
            + _DISTANCE
            + _TEMPORAL_INTERACTIONS
            + _SWR_INTERACTIONS
            + _MUA_INTERACTIONS
        ),
        description="SWR + MUA event-type interactions",
        parent="M6",
    ),
    "M8": ModelSpec(
        name="M8",
        predictors=(
            _SPATIAL
            + _REWARD_BINARY
            + _REWARD_HISTORY
            + _DISTANCE
            + _TEMPORAL_INTERACTIONS
            + _SWR_INTERACTIONS
            + _MUA_INTERACTIONS
            + _CONTEXT_INTERACTIONS
        ),
        description="Full model with task context interactions",
        parent="M7",
    ),
}

# ── Alternative spatial specifications ──
#
# Models A and B answer the same question ("does replay depend on
# spatial proximity?") with different parameterizations to check
# robustness against the severe collinearity (VIF 10–55) among
# is_current / dist_to_arm / is_same_patch on the bandit graph.
#
# Model A: distance gradient only (1 continuous predictor, no collinearity)
# Model B: binary indicators only (low collinearity, categorical)
#
# Each is layered with the same reward predictors so reward effects
# can be compared across specifications.

MODEL_A: dict[str, ModelSpec] = {
    "A0": ModelSpec(
        name="A0",
        predictors=(),
        description="Null",
        parent=None,
    ),
    "A1": ModelSpec(
        name="A1",
        predictors=("dist_to_arm",),
        description="Distance gradient",
        parent="A0",
    ),
    "A2": ModelSpec(
        name="A2",
        predictors=("dist_to_arm", "is_last_rewarded"),
        description="+ most-recent reward",
        parent="A1",
    ),
    "A3": ModelSpec(
        name="A3",
        predictors=(
            "dist_to_arm",
            "is_last_rewarded",
            "visit_recency",
            "cumulative_reward",
        ),
        description="+ reward history",
        parent="A2",
    ),
}

MODEL_B: dict[str, ModelSpec] = {
    "B0": ModelSpec(
        name="B0",
        predictors=(),
        description="Null",
        parent=None,
    ),
    "B1": ModelSpec(
        name="B1",
        predictors=("is_current",),
        description="Current-arm indicator",
        parent="B0",
    ),
    "B2": ModelSpec(
        name="B2",
        predictors=("is_current", "is_same_patch"),
        description="+ same-patch indicator",
        parent="B1",
    ),
    "B3": ModelSpec(
        name="B3",
        predictors=(
            "is_current",
            "is_same_patch",
            "is_last_rewarded",
            "current_x_last_rewarded",
        ),
        description="+ most-recent reward (with interaction)",
        parent="B2",
    ),
    "B4": ModelSpec(
        name="B4",
        predictors=(
            "is_current",
            "is_same_patch",
            "is_last_rewarded",
            "current_x_last_rewarded",
            "visit_recency",
            "cumulative_reward",
        ),
        description="+ reward history",
        parent="B3",
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
