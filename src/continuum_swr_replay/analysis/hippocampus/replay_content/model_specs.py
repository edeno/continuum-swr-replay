"""Model hierarchy definitions for replay content analysis.

Defines a sequence of nested models (M0-M7) for the conditional Poisson
(multinomial-count) framework. Only arm-varying predictors appear as main
effects; event-level constants (time_since_stop, is_swr, etc.) appear only
in interactions with arm-varying predictors, because they cancel in the
conditional likelihood's within-event normalization.
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
# These vary across the 9 arms within an event and are identifiable.

_BASE: tuple[str, ...] = ("is_current", "is_prev", "is_next")

# Reward and recency (arm-varying: binary + continuous)
_REWARD_RECENCY: tuple[str, ...] = (
    "is_last_rewarded",
    "visit_recency",
    "cumulative_reward",
)

# Arm-specific distance (arm-varying: continuous)
_DISTANCE: tuple[str, ...] = ("dist_to_arm",)

# ── Event-level × arm-varying interactions ──
# Event-level variables (time_since_stop, is_swr, etc.) are constant across
# all 9 arms within an event and cancel in the conditional likelihood.
# They can only appear as interactions with arm-varying predictors.

_TEMPORAL_INTERACTIONS: tuple[str, ...] = (
    "time_x_current",
    "time_x_next",
    "time_x_prev",
)

_SWR_INTERACTIONS: tuple[str, ...] = (
    "swr_x_current",
    "swr_x_next",
    "swr_x_prev",
    "swr_x_rewarded",
)

_CONTEXT_INTERACTIONS: tuple[str, ...] = (
    "switch_x_current",
    "switch_x_prev",
    "omission_x_rewarded",
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
        description="Proximity bias",
        parent="M0",
    ),
    "M2": ModelSpec(
        name="M2",
        predictors=_BASE,
        description="Spatial context (current + previous + next)",
        parent="M1",
    ),
    "M3": ModelSpec(
        name="M3",
        predictors=_BASE + _REWARD_RECENCY,
        description="Reward history + visit recency (Gillespie hypothesis)",
        parent="M2",
    ),
    "M4": ModelSpec(
        name="M4",
        predictors=_BASE + _REWARD_RECENCY + _DISTANCE,
        description="Arm-specific distance control",
        parent="M3",
    ),
    "M5": ModelSpec(
        name="M5",
        predictors=_BASE + _REWARD_RECENCY + _DISTANCE + _TEMPORAL_INTERACTIONS,
        description="Temporal dynamics (Mallory hypothesis)",
        parent="M4",
    ),
    "M6": ModelSpec(
        name="M6",
        predictors=(
            _BASE
            + _REWARD_RECENCY
            + _DISTANCE
            + _TEMPORAL_INTERACTIONS
            + _SWR_INTERACTIONS
        ),
        description="SWR distinction (survives after timing control?)",
        parent="M5",
    ),
    "M7": ModelSpec(
        name="M7",
        predictors=(
            _BASE
            + _REWARD_RECENCY
            + _DISTANCE
            + _TEMPORAL_INTERACTIONS
            + _SWR_INTERACTIONS
            + _CONTEXT_INTERACTIONS
        ),
        description="Full model with task context interactions",
        parent="M6",
    ),
}
