"""Model hierarchy definitions for replay content analysis.

Defines a sequence of nested models for the conditional Poisson
(multinomial-count) framework. Only arm-varying predictors appear as
main effects; event-level constants (time_since_stop, is_swr, etc.)
appear only in interactions with arm-varying predictors, because they
cancel in the conditional likelihood's within-event normalization.

Design decisions informed by exploratory analysis on 7 j16 sessions:

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
    "MODEL_HIERARCHY",
    "MODEL_PATCH",
    "MODEL_PATCH_SWITCH",
    "ModelSpec",
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


# ── Primary arm-level hierarchy ──
#
# Designed for reliable per-session convergence (max VIF < 3 through
# M5; < 25 at M6 where is_last_rewarded enters). Tested on 7 j16
# sessions: M0–M5 converge 7/7; M6–M7 converge 7/7.
#
# SWR interactions enter at M7. These test whether the current-arm
# and previous-arm biases differ during SWR events. Only spatial
# interactions are included (swr_x_current, swr_x_prev); reward
# interactions (swr_x_last_rewarded) are excluded due to perfect
# collinearity with swr_x_current.
#
# MUA event-type effects are assessed via split analysis (fit M0–M6
# separately on MUA-only and non-MUA events) rather than interaction
# terms, because mua_x_current ≈ mua_x_last_rewarded (r = 0.98,
# VIF > 70).

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
        predictors=("is_current", "is_prev"),
        description="Spatial context (current + previous)",
        parent="M1",
    ),
    "M3": ModelSpec(
        name="M3",
        predictors=("is_current", "is_prev", "cumulative_reward"),
        description="+ cumulative reward history",
        parent="M2",
    ),
    "M4": ModelSpec(
        name="M4",
        predictors=("is_current", "is_prev", "cumulative_reward", "visit_recency"),
        description="+ visit recency",
        parent="M3",
    ),
    "M5": ModelSpec(
        name="M5",
        predictors=(
            "is_current",
            "is_prev",
            "cumulative_reward",
            "visit_recency",
            "is_last_rewarded",
        ),
        description="+ last-rewarded arm",
        parent="M4",
    ),
    "M6": ModelSpec(
        name="M6",
        predictors=(
            "is_current",
            "is_prev",
            "cumulative_reward",
            "visit_recency",
            "is_last_rewarded",
            "swr_x_current",
            "swr_x_prev",
        ),
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
