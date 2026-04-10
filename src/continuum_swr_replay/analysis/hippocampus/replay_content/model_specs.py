"""Model hierarchy definitions for replay content analysis.

Defines a sequence of nested models (M0-M7) that build up complexity,
allowing systematic model comparison via AIC/BIC/LRT at each step.
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
    predictors : list[str]
        Column names to use as fixed-effect predictors.
    description : str
        Human-readable description of what the model tests.
    parent : str | None
        Name of the parent model for nested comparison. None for M0.
    """

    name: str
    predictors: list[str]
    description: str
    parent: str | None


# Base predictors (arm-level binary indicators)
_BASE = ["is_current", "is_prev", "is_next"]

# Reward and recency (binary main effect + continuous, per-arm)
# is_last_rewarded must be included as main effect before its interactions
# (swr_x_rewarded, omission_x_rewarded) appear in later models.
_REWARD_RECENCY = ["is_last_rewarded", "visit_recency", "cumulative_reward"]

# Distance control (continuous, event-level)
_DISTANCE = ["log_distance"]

# Temporal dynamics (continuous + interactions)
_TEMPORAL = ["time_since_stop", "time_x_current", "time_x_prev"]

# SWR distinction (binary + interactions)
_SWR = [
    "is_swr",
    "swr_x_current",
    "swr_x_rewarded",
    "swr_x_prev",
    "swr_x_next",
]

# Task context (binary + interactions)
_CONTEXT = [
    "is_after_switch",
    "switch_x_current",
    "switch_x_prev",
    "is_after_omission",
    "omission_x_rewarded",
]


MODEL_HIERARCHY: dict[str, ModelSpec] = {
    "M0": ModelSpec(
        name="M0",
        predictors=[],
        description="Null (intercept only)",
        parent=None,
    ),
    "M1": ModelSpec(
        name="M1",
        predictors=["is_current"],
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
        description="Distance control (proximity confound)",
        parent="M3",
    ),
    "M5": ModelSpec(
        name="M5",
        predictors=_BASE + _REWARD_RECENCY + _DISTANCE + _TEMPORAL,
        description="Temporal dynamics (Mallory hypothesis)",
        parent="M4",
    ),
    "M6": ModelSpec(
        name="M6",
        predictors=_BASE + _REWARD_RECENCY + _DISTANCE + _TEMPORAL + _SWR,
        description="SWR distinction (survives after timing control?)",
        parent="M5",
    ),
    "M7": ModelSpec(
        name="M7",
        predictors=(_BASE + _REWARD_RECENCY + _DISTANCE + _TEMPORAL + _SWR + _CONTEXT),
        description="Full model with task context interactions",
        parent="M6",
    ),
}
