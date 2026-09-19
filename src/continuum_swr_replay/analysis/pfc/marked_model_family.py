"""Recommended descriptive marked-kernel models for mPFC replay responses.

The single-mark model in :mod:`continuum_swr_replay.analysis.pfc.marked_glm`
establishes the temporal estimand.  This module extends that estimator to a
small family of question-specific, jointly adjusted designs.  A family shares
one full fit across all of its target marks and compares each target with the
otherwise identical model lacking only that target's temporal block.

There are deliberately no p-values, label shuffles, or neuron-selection rules
here.  The outputs are descriptive effect kernels and blocked held-out
prediction changes.  The models use only event/content tables already shipped
by the replay pipeline; they do not decode mPFC population activity.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.pfc.batched_glm import (
    DEFAULT_NULL_DESIGN_BATCH_SIZE,
    PreparedHeldoutResponse,
    fit_poisson_glms,
    make_ridge_penalty,
    prepare_heldout_response,
    score_poisson_designs_heldout,
)
from continuum_swr_replay.analysis.pfc.marked_glm import (
    EVENT_DURATION_MARK,
    MARKED_EVENT_CLASSES,
    EventKernelBlock,
    KernelDesign,
    MarkScale,
    build_kernel_design,
    standardize_event_marks,
)
from continuum_swr_replay.data_loaders.utils import parse_nwb_filename
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.session_keys import SESSION_COLUMNS
from continuum_swr_replay.table_io import read_table, resolved_table_path, sha256_file

DEFAULT_EVENT_ARM_TABLE_PATH = (
    PROCESSED_DATA_DIR / "replay_content" / "event_arm_table.parquet"
)
DEFAULT_EVENT_TABLE_PATH = PROCESSED_DATA_DIR / "replay_content" / "event_table.parquet"

CURRENT_ARM_FRACTION = "current_arm_fraction"
PREVIOUS_ARM_FRACTION = "previous_arm_fraction"
CURRENT_PATCH_FRACTION = "current_patch_fraction"
LEAF_CONTENT_FRACTION = "leaf_content_fraction"
CONTENT_UNCERTAINTY = "content_weighted_belief_uncertainty"
CONTENT_VALUE = "content_weighted_belief_value"
CONTENT_RECENT_REWARD = "content_weighted_recent_reward"
CONTENT_RECENCY = "content_weighted_relative_recency"
CONTENT_NOVELTY = "content_weighted_novelty_fraction"
LOG_STOP_TIME = "log1p_time_since_stop_start_s"
SWITCH_CONTEXT = "switch_patch_context"
REWARDED_CONTEXT = "rewarded_context"
LOG_SPATIAL_EXTENT = "log1p_spatial_extent_cm"
LOG_REVERSALS = "log1p_n_reversals"
PAUSE_FRACTION = "pause_fraction"
RETROSPECTIVE_BIAS = "retrospective_bias_cm"
RIPPLE_POWER = "ripple_power_z"
MUA_RATE = "mua_rate_z"
THETA_POWER = "theta_z"
DECODE_ENTROPY = "posterior_entropy_mean"
CURRENT_X_SWITCH = "current_arm_fraction_x_switch"
UNCERTAINTY_X_SWITCH = "content_uncertainty_x_switch"
CURRENT_X_REWARD = "current_arm_fraction_x_rewarded"
UNCERTAINTY_X_REWARD = "content_uncertainty_x_rewarded"
CURRENT_X_RIPPLE = "current_arm_fraction_x_ripple_power"
UNCERTAINTY_X_RIPPLE = "content_uncertainty_x_ripple_power"

# The shipped single-mark profile used four designs of width 67 (268 batched
# feature columns, about 34 GB on the largest profiled session). Rounding that
# measured workload up by 7.5% permits two 144-column core reduced designs per
# batch while retaining ample headroom on an 80 GB A100. Interaction reductions
# remain one design per batch.
PROFILED_BATCH_FEATURE_BUDGET = 2 * 144

MARK_DESCRIPTIONS: Mapping[str, str] = MappingProxyType(
    {
        EVENT_DURATION_MARK: "natural log of inclusive event duration in seconds",
        CURRENT_ARM_FRACTION: "replay fraction on the animal's current arm",
        PREVIOUS_ARM_FRACTION: "replay fraction on the previous arm",
        CURRENT_PATCH_FRACTION: "replay fraction in the animal's current patch",
        LEAF_CONTENT_FRACTION: "replay fraction on the six reward-bearing leaf arms",
        CONTENT_UNCERTAINTY: (
            "leaf-content-weighted Beta belief variance, normalized by represented "
            "leaf mass; undefined for events with no leaf content"
        ),
        CONTENT_VALUE: (
            "leaf-content-weighted Beta belief mean, normalized by represented "
            "leaf mass"
        ),
        CONTENT_RECENT_REWARD: (
            "leaf-content-weighted reward fraction over each arm's last ten visits"
        ),
        CONTENT_RECENCY: ("leaf-content-weighted within-event relative visit recency"),
        CONTENT_NOVELTY: "fraction of represented leaf content on never-visited arms",
        LOG_STOP_TIME: "log(1 + seconds from stop onset to replay onset)",
        SWITCH_CONTEXT: "whether the last completed trip switched patch",
        REWARDED_CONTEXT: "whether the latest physically known outcome was rewarded",
        LOG_SPATIAL_EXTENT: "log(1 + replay spatial extent in cm)",
        LOG_REVERSALS: "log(1 + number of decoded trajectory reversals)",
        PAUSE_FRACTION: "fraction of decoded trajectory steps classified as pauses",
        RETROSPECTIVE_BIAS: "maximum behind minus maximum ahead distance in cm",
        RIPPLE_POWER: "event-mean z-scored consensus ripple power",
        MUA_RATE: "event-mean z-scored hippocampal multiunit rate",
        THETA_POWER: "event-mean z-scored theta power",
        DECODE_ENTROPY: "mean decoded posterior entropy (quality control)",
        CURRENT_X_SWITCH: "current-arm replay fraction times patch-switch context",
        UNCERTAINTY_X_SWITCH: "content-weighted uncertainty times switch context",
        CURRENT_X_REWARD: "current-arm replay fraction times rewarded context",
        UNCERTAINTY_X_REWARD: "content-weighted uncertainty times rewarded context",
        CURRENT_X_RIPPLE: "current-arm replay fraction times ripple power",
        UNCERTAINTY_X_RIPPLE: "content-weighted uncertainty times ripple power",
    }
)


@dataclass(frozen=True)
class MarkedFamilySpec:
    """One jointly adjusted marked-kernel question.

    ``scientific_marks`` receive leave-one-block-out held-out comparisons.
    ``nuisance_marks`` remain in every comparison.  Every mark still has a
    fitted temporal kernel, so a nuisance adjustment is not forced to be
    instantaneous or constant over the peri-event window.
    """

    name: str
    scientific_marks: tuple[str, ...]
    nuisance_marks: tuple[str, ...]
    question: str
    tier: str = "core"
    required_varying_nuisance_marks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        all_marks = (*self.nuisance_marks, *self.scientific_marks)
        if not self.name or not self.scientific_marks:
            raise ValueError("a model spec needs a name and at least one target mark")
        if len(all_marks) != len(set(all_marks)):
            raise ValueError(f"model {self.name!r} contains duplicate marks")
        unknown = sorted(set(all_marks) - set(MARK_DESCRIPTIONS))
        if unknown:
            raise ValueError(f"model {self.name!r} uses unknown marks {unknown}")
        if self.tier not in {"core", "interaction"}:
            raise ValueError("tier must be 'core' or 'interaction'")
        if not set(self.required_varying_nuisance_marks).issubset(self.nuisance_marks):
            raise ValueError("required varying marks must be nuisance marks")

    @property
    def all_marks(self) -> tuple[str, ...]:
        """Marks in deterministic design order: adjustments, then targets."""
        return (*self.nuisance_marks, *self.scientific_marks)


def _spec(
    name: str,
    scientific: tuple[str, ...],
    nuisance: tuple[str, ...],
    question: str,
    *,
    tier: str = "core",
    required_varying_nuisance_marks: tuple[str, ...] = (),
) -> MarkedFamilySpec:
    return MarkedFamilySpec(
        name,
        scientific,
        nuisance,
        question,
        tier,
        required_varying_nuisance_marks,
    )


# Six compact core designs carry six marked predictors per class. The three
# content horse-races and all three interaction sensitivities carry nine. Thus
# the full family uses only two compiled widths (155 and 221 columns with the
# shipped 11-function basis) while retaining the controls each question needs.
RECOMMENDED_MODEL_SPECS: Mapping[str, MarkedFamilySpec] = MappingProxyType(
    {
        "spatial_role": _spec(
            "spatial_role",
            (CURRENT_ARM_FRACTION, PREVIOUS_ARM_FRACTION, CURRENT_PATCH_FRACTION),
            (EVENT_DURATION_MARK, LEAF_CONTENT_FRACTION, DECODE_ENTROPY),
            "Does response timing vary with current, previous, or same-patch content?",
        ),
        "belief_content": _spec(
            "belief_content",
            (CONTENT_UNCERTAINTY, CONTENT_VALUE),
            (
                EVENT_DURATION_MARK,
                LEAF_CONTENT_FRACTION,
                CURRENT_ARM_FRACTION,
                CONTENT_RECENT_REWARD,
                CONTENT_RECENCY,
                CONTENT_NOVELTY,
                DECODE_ENTROPY,
            ),
            "Does mPFC track represented-arm uncertainty beyond value and reward history?",
        ),
        "reward_memory": _spec(
            "reward_memory",
            (CONTENT_RECENT_REWARD, CONTENT_RECENCY),
            (
                EVENT_DURATION_MARK,
                LEAF_CONTENT_FRACTION,
                CURRENT_ARM_FRACTION,
                CONTENT_UNCERTAINTY,
                CONTENT_VALUE,
                DECODE_ENTROPY,
                CONTENT_NOVELTY,
            ),
            "Do recent reward and recency explain responses beyond belief state?",
        ),
        "novelty_content": _spec(
            "novelty_content",
            (CONTENT_NOVELTY,),
            (
                EVENT_DURATION_MARK,
                LEAF_CONTENT_FRACTION,
                CURRENT_ARM_FRACTION,
                CONTENT_UNCERTAINTY,
                CONTENT_VALUE,
                CONTENT_RECENT_REWARD,
                CONTENT_RECENCY,
                DECODE_ENTROPY,
            ),
            "Does represented never-visited content matter beyond graded recency?",
        ),
        "trajectory_geometry": _spec(
            "trajectory_geometry",
            (
                LOG_SPATIAL_EXTENT,
                LOG_REVERSALS,
                PAUSE_FRACTION,
                RETROSPECTIVE_BIAS,
            ),
            (EVENT_DURATION_MARK, DECODE_ENTROPY),
            "Does response timing vary with extent, reversals, pauses, or direction?",
        ),
        "within_stop_timing": _spec(
            "within_stop_timing",
            (LOG_STOP_TIME,),
            (
                EVENT_DURATION_MARK,
                CURRENT_ARM_FRACTION,
                LEAF_CONTENT_FRACTION,
                CONTENT_VALUE,
                DECODE_ENTROPY,
            ),
            "Does the replay response change over the course of an immobility stop?",
        ),
        "switch_context": _spec(
            "switch_context",
            (SWITCH_CONTEXT,),
            (
                EVENT_DURATION_MARK,
                CURRENT_ARM_FRACTION,
                CONTENT_VALUE,
                DECODE_ENTROPY,
                REWARDED_CONTEXT,
            ),
            "Does patch-switch context change the response beyond known reward?",
        ),
        "reward_context": _spec(
            "reward_context",
            (REWARDED_CONTEXT,),
            (
                EVENT_DURATION_MARK,
                CURRENT_ARM_FRACTION,
                CONTENT_VALUE,
                DECODE_ENTROPY,
                SWITCH_CONTEXT,
            ),
            "Does known reward change the response beyond patch-switch context?",
        ),
        "graded_coupling": _spec(
            "graded_coupling",
            (RIPPLE_POWER, MUA_RATE, THETA_POWER),
            (EVENT_DURATION_MARK, LEAF_CONTENT_FRACTION, DECODE_ENTROPY),
            "Is binary SWR membership hiding a graded hippocampal coupling continuum?",
        ),
        "exploration_content_interaction": _spec(
            "exploration_content_interaction",
            (CURRENT_X_SWITCH, UNCERTAINTY_X_SWITCH),
            (
                EVENT_DURATION_MARK,
                CURRENT_ARM_FRACTION,
                CONTENT_UNCERTAINTY,
                SWITCH_CONTEXT,
                LEAF_CONTENT_FRACTION,
                CONTENT_VALUE,
                DECODE_ENTROPY,
            ),
            "Does exploration change current-arm or uncertainty modulation?",
            tier="interaction",
            required_varying_nuisance_marks=(SWITCH_CONTEXT,),
        ),
        "reward_content_interaction": _spec(
            "reward_content_interaction",
            (CURRENT_X_REWARD, UNCERTAINTY_X_REWARD),
            (
                EVENT_DURATION_MARK,
                CURRENT_ARM_FRACTION,
                CONTENT_UNCERTAINTY,
                REWARDED_CONTEXT,
                LEAF_CONTENT_FRACTION,
                CONTENT_VALUE,
                DECODE_ENTROPY,
            ),
            "Does known reward change current-arm or uncertainty modulation?",
            tier="interaction",
            required_varying_nuisance_marks=(REWARDED_CONTEXT,),
        ),
        "coupling_content_interaction": _spec(
            "coupling_content_interaction",
            (CURRENT_X_RIPPLE, UNCERTAINTY_X_RIPPLE),
            (
                EVENT_DURATION_MARK,
                CURRENT_ARM_FRACTION,
                CONTENT_UNCERTAINTY,
                RIPPLE_POWER,
                LEAF_CONTENT_FRACTION,
                CONTENT_VALUE,
                DECODE_ENTROPY,
            ),
            "Does graded ripple power change current-arm or uncertainty modulation?",
            tier="interaction",
            required_varying_nuisance_marks=(RIPPLE_POWER,),
        ),
    }
)
CORE_MODEL_NAMES = tuple(
    name for name, spec in RECOMMENDED_MODEL_SPECS.items() if spec.tier == "core"
)
INTERACTION_MODEL_NAMES = tuple(
    name for name, spec in RECOMMENDED_MODEL_SPECS.items() if spec.tier == "interaction"
)


@dataclass(frozen=True)
class MarkedFamilyLadder:
    """One joint full design and every leave-one-target-block-out design."""

    spec: MarkedFamilySpec
    reduced: dict[tuple[str, str], KernelDesign]
    full: KernelDesign
    events: dict[str, pd.DataFrame]
    scales: dict[str, dict[str, MarkScale]]
    classes: tuple[str, ...]
    fit_mask: pd.Series
    spline_basis: NDArray
    lags: NDArray


@dataclass(frozen=True)
class MarkedFamilyResults:
    """Per-unit comparisons and full-data temporal kernels for one spec."""

    statistics: pd.DataFrame
    typical_curves: dict[str, NDArray]
    mark_curves: dict[str, dict[str, NDArray]]
    diagnostics: dict[str, Any]


class ModelIneligibleError(ValueError):
    """A session/class lacks enough finite variation for a requested model."""


def _numeric(frame: pd.DataFrame, column: str) -> NDArray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _validate_event_arm_geometry(event_arm_table: pd.DataFrame) -> None:
    counts = event_arm_table.groupby("event_id", sort=False).size()
    if not (counts == 9).all():
        event_id = counts.index[counts.ne(9)][0]
        raise ValueError(f"event_id {event_id!r} does not have exactly nine arm rows")
    candidates = event_arm_table.groupby("event_id", sort=False)["candidate_arm"].agg(
        lambda values: tuple(sorted(pd.to_numeric(values, errors="coerce")))
    )
    expected = tuple(range(9))
    invalid = candidates.map(lambda values: values != expected)
    if invalid.any():
        raise ValueError(
            f"event_id {candidates.index[invalid][0]!r} does not contain arms 0..8"
        )
    weights = _numeric(event_arm_table, "arm_fraction")
    if not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
        raise ValueError("arm_fraction must be finite and in [0, 1]")
    totals = (
        pd.Series(weights, index=event_arm_table.index)
        .groupby(event_arm_table["event_id"], sort=False)
        .sum()
    )
    if not np.allclose(totals, 1.0, atol=1e-8, rtol=0.0):
        raise ValueError("arm_fraction must sum to one within every event")


def _role_fraction(
    event_arm_table: pd.DataFrame,
    role: str,
    *,
    allow_undefined: bool,
) -> pd.Series:
    raw = event_arm_table[role].astype("boolean")
    groups = event_arm_table["event_id"]
    n_defined = raw.notna().groupby(groups, sort=False).sum()
    if allow_undefined:
        partial = (n_defined != 0) & (n_defined != 9)
        if partial.any():
            raise ValueError(
                f"event_id {partial.index[partial][0]!r} has partially missing {role}"
            )
    elif (n_defined != 9).any():
        raise ValueError(f"{role} must be defined on every arm row")
    weights = _numeric(event_arm_table, "arm_fraction")
    selected = raw.fillna(False).to_numpy(dtype=bool)
    result = (
        pd.Series(np.where(selected, weights, 0.0))
        .groupby(groups.to_numpy(), sort=False)
        .sum()
    )
    if allow_undefined:
        result.loc[n_defined.eq(0)] = np.nan
    return result


def _leaf_weighted_mark(event_arm_table: pd.DataFrame, source: str) -> pd.Series:
    values = _numeric(event_arm_table, source)
    candidate = pd.to_numeric(
        event_arm_table["candidate_arm"], errors="coerce"
    ).to_numpy(dtype=float)
    leaf = candidate < 6
    finite_leaf = leaf & np.isfinite(values)
    groups = event_arm_table["event_id"]
    finite_counts = (
        pd.Series(finite_leaf.astype(int)).groupby(groups.to_numpy(), sort=False).sum()
    )
    if not finite_counts.isin([0, 6]).all():
        event_id = finite_counts.index[~finite_counts.isin([0, 6])][0]
        raise ValueError(
            f"event_id {event_id!r} has partially available leaf values for {source}"
        )
    weights = _numeric(event_arm_table, "arm_fraction")
    numerator = (
        pd.Series(np.where(finite_leaf, weights * values, 0.0))
        .groupby(groups.to_numpy(), sort=False)
        .sum()
    )
    denominator = (
        pd.Series(np.where(finite_leaf, weights, 0.0))
        .groupby(groups.to_numpy(), sort=False)
        .sum()
    )
    return numerator / denominator.where(denominator > 1e-12)


def build_recommended_event_mark_table(
    event_arm_table: pd.DataFrame,
    event_table: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse canonical arm/event tables to one row of recommended marks/event.

    Belief, reward and recency quantities are averaged only over represented
    leaf-arm content and renormalized by represented leaf mass.  Trunk arms'
    structural zeros therefore never masquerade as zero uncertainty or value.
    ``leaf_content_fraction`` remains in the model as a separate adjustment.
    """
    arm_required = {
        "event_id",
        "candidate_arm",
        "arm_fraction",
        "event_start_s",
        "event_end_s",
        "event_duration_s",
        "is_swr",
        "is_current_arm",
        "is_prev_arm",
        "is_same_patch_as_current",
        "reward_belief_variance_candidate_arm_preoutcome",
        "reward_belief_mean_candidate_arm_preoutcome",
        "reward_fraction_candidate_arm_last10_visits",
        "visit_recency_trials_event_leaf_centered",
        "is_candidate_arm_never_visited",
    }
    event_required = {
        "event_id",
        "event_start_s",
        "event_end_s",
        "event_duration_s",
        "is_swr",
        "time_since_stop_start_s",
        "did_last_completed_trip_switch_patch",
        "is_rewarded_outcome",
        "spatial_extent_cm",
        "n_reversals",
        PAUSE_FRACTION,
        "max_ahead_cm",
        "max_behind_cm",
        RIPPLE_POWER,
        MUA_RATE,
        THETA_POWER,
        DECODE_ENTROPY,
    }
    missing_arm = sorted(arm_required - set(event_arm_table))
    missing_event = sorted(event_required - set(event_table))
    if missing_arm or missing_event:
        raise KeyError(
            f"missing model-family columns: arm={missing_arm}, event={missing_event}"
        )
    if event_arm_table.empty or event_table.empty:
        raise ValueError("event-arm and event tables must both be non-empty")
    if event_table["event_id"].duplicated().any():
        duplicate = event_table.loc[event_table.event_id.duplicated(), "event_id"].iloc[
            0
        ]
        raise ValueError(
            f"event_id {duplicate!r} appears more than once in event_table"
        )
    arm_ids = pd.Index(event_arm_table.event_id.unique())
    event_ids = pd.Index(event_table.event_id)
    if set(arm_ids) != set(event_ids):
        raise ValueError(
            "event and event-arm tables do not contain the same event_id set"
        )
    _validate_event_arm_geometry(event_arm_table)

    arm_metadata = event_arm_table[
        ["event_id", "event_start_s", "event_end_s", "event_duration_s", "is_swr"]
    ].drop_duplicates()
    if arm_metadata.event_id.duplicated().any():
        event_id = arm_metadata.loc[
            arm_metadata.event_id.duplicated(), "event_id"
        ].iloc[0]
        raise ValueError(f"event_id {event_id!r} has inconsistent event-arm metadata")
    event_metadata = event_table.set_index("event_id").loc[arm_metadata.event_id]
    for column in ("event_start_s", "event_end_s", "event_duration_s"):
        if not np.allclose(
            _numeric(arm_metadata, column),
            _numeric(event_metadata, column),
            atol=1e-9,
            rtol=0.0,
        ):
            raise ValueError(f"event and event-arm {column} values disagree")
    if not np.array_equal(
        arm_metadata.is_swr.astype("boolean").to_numpy(dtype=bool),
        event_metadata.is_swr.astype("boolean").to_numpy(dtype=bool),
    ):
        raise ValueError("event and event-arm is_swr values disagree")

    groups = event_arm_table["event_id"]
    weights = _numeric(event_arm_table, "arm_fraction")
    leaf = pd.to_numeric(event_arm_table.candidate_arm, errors="coerce").to_numpy() < 6
    marks = pd.DataFrame(index=arm_ids)
    marks.index.name = "event_id"
    marks[CURRENT_ARM_FRACTION] = _role_fraction(
        event_arm_table, "is_current_arm", allow_undefined=True
    ).reindex(arm_ids)
    marks[PREVIOUS_ARM_FRACTION] = _role_fraction(
        event_arm_table, "is_prev_arm", allow_undefined=False
    ).reindex(arm_ids)
    marks[CURRENT_PATCH_FRACTION] = _role_fraction(
        event_arm_table, "is_same_patch_as_current", allow_undefined=True
    ).reindex(arm_ids)
    marks[LEAF_CONTENT_FRACTION] = (
        pd.Series(np.where(leaf, weights, 0.0))
        .groupby(groups.to_numpy(), sort=False)
        .sum()
        .reindex(arm_ids)
    )
    for source, output in (
        ("reward_belief_variance_candidate_arm_preoutcome", CONTENT_UNCERTAINTY),
        ("reward_belief_mean_candidate_arm_preoutcome", CONTENT_VALUE),
        ("reward_fraction_candidate_arm_last10_visits", CONTENT_RECENT_REWARD),
        ("visit_recency_trials_event_leaf_centered", CONTENT_RECENCY),
        ("is_candidate_arm_never_visited", CONTENT_NOVELTY),
    ):
        marks[output] = _leaf_weighted_mark(event_arm_table, source).reindex(arm_ids)

    scalar_columns = sorted(event_required - {"event_id"})
    event_scalars = event_table.set_index("event_id").loc[arm_ids, scalar_columns]
    marks = marks.join(event_scalars, validate="one_to_one")
    duration = _numeric(marks, "event_duration_s")
    if not np.isfinite(duration).all() or np.any(duration <= 0):
        raise ValueError("event_duration_s must be finite and positive")
    marks[EVENT_DURATION_MARK] = np.log(duration)
    stop_time = _numeric(marks, "time_since_stop_start_s")
    extent = _numeric(marks, "spatial_extent_cm")
    reversals = _numeric(marks, "n_reversals")
    if np.any(stop_time < 0) or np.any(extent < 0) or np.any(reversals < 0):
        raise ValueError("time, spatial extent and reversal count must be non-negative")
    marks[LOG_STOP_TIME] = np.log1p(stop_time)
    marks[LOG_SPATIAL_EXTENT] = np.log1p(extent)
    marks[LOG_REVERSALS] = np.log1p(reversals)
    marks[RETROSPECTIVE_BIAS] = _numeric(marks, "max_behind_cm") - _numeric(
        marks, "max_ahead_cm"
    )
    marks[SWITCH_CONTEXT] = _numeric(marks, "did_last_completed_trip_switch_patch")
    marks[REWARDED_CONTEXT] = _numeric(marks, "is_rewarded_outcome")
    marks[CURRENT_X_SWITCH] = marks[CURRENT_ARM_FRACTION] * marks[SWITCH_CONTEXT]
    marks[UNCERTAINTY_X_SWITCH] = marks[CONTENT_UNCERTAINTY] * marks[SWITCH_CONTEXT]
    marks[CURRENT_X_REWARD] = marks[CURRENT_ARM_FRACTION] * marks[REWARDED_CONTEXT]
    marks[UNCERTAINTY_X_REWARD] = marks[CONTENT_UNCERTAINTY] * marks[REWARDED_CONTEXT]
    marks[CURRENT_X_RIPPLE] = marks[CURRENT_ARM_FRACTION] * marks[RIPPLE_POWER]
    marks[UNCERTAINTY_X_RIPPLE] = marks[CONTENT_UNCERTAINTY] * marks[RIPPLE_POWER]
    return marks.reset_index()


def attach_recommended_event_marks(
    events: dict[str, pd.DataFrame],
    mark_table: pd.DataFrame,
    *,
    classes: tuple[str, ...] = MARKED_EVENT_CLASSES,
) -> dict[str, pd.DataFrame]:
    """Join one-row-per-event recommended marks to canonical class frames."""
    required = {"event_id", "event_start_s", "event_end_s", "is_swr"}
    missing = sorted(required - set(mark_table))
    if missing:
        raise KeyError(f"mark table is missing columns {missing}")
    if mark_table.event_id.duplicated().any():
        raise ValueError("mark table must contain one row per event_id")
    requested: list[object] = []
    for event_class in classes:
        if event_class not in events:
            raise KeyError(f"event class {event_class!r} is missing")
        requested.extend(events[event_class].event_id.tolist())
    requested_index = pd.Index(requested)
    if requested_index.has_duplicates:
        raise ValueError("event_id appears in more than one canonical class row")
    available = mark_table.set_index("event_id", drop=False)
    missing_ids = requested_index.difference(available.index)
    if len(missing_ids):
        raise ValueError(
            f"recommended marks are missing event_id(s) {missing_ids[:5].tolist()}"
        )
    selected = available.loc[requested_index].reset_index(drop=True)
    mark_columns = [column for column in MARK_DESCRIPTIONS if column in selected]
    metadata = selected[
        ["event_id", "event_start_s", "event_end_s", "is_swr", *mark_columns]
    ]

    attached: dict[str, pd.DataFrame] = {}
    for event_class in classes:
        frame = events[event_class].copy()
        expected_swr = event_class == "non_local_SWR"
        rows = metadata.loc[metadata.event_id.isin(frame.event_id)].copy()
        rows = rows.set_index("event_id").loc[frame.event_id].reset_index()
        if not np.all(
            rows.is_swr.astype("boolean").to_numpy(dtype=bool) == expected_swr
        ):
            raise ValueError(f"mark is_swr labels disagree with {event_class!r}")
        for canonical, mark_time in (
            ("start_time", "event_start_s"),
            ("end_time", "event_end_s"),
        ):
            if not np.allclose(
                _numeric(frame, canonical),
                _numeric(rows, mark_time),
                atol=1e-9,
                rtol=0.0,
            ):
                raise ValueError(
                    f"recommended mark timing disagrees for {event_class!r}"
                )
        for column in mark_columns:
            if column in frame:
                left = _numeric(frame, column)
                right = _numeric(rows, column)
                if not np.allclose(left, right, atol=1e-12, rtol=0.0, equal_nan=True):
                    raise ValueError(
                        f"existing mark {column!r} disagrees with family table"
                    )
                frame = frame.drop(columns=column)
        joined = frame.merge(
            rows[["event_id", *mark_columns]],
            on="event_id",
            how="left",
            sort=False,
            validate="one_to_one",
        )
        if not np.array_equal(joined.event_id.to_numpy(), frame.event_id.to_numpy()):
            raise ValueError(f"recommended mark join changed {event_class!r} order")
        attached[event_class] = joined
    return attached


def load_session_recommended_event_marks(
    nwb_file_name: str,
    epoch_name: str,
    events: dict[str, pd.DataFrame],
    *,
    event_arm_table_path: str | Path = DEFAULT_EVENT_ARM_TABLE_PATH,
    event_table_path: str | Path = DEFAULT_EVENT_TABLE_PATH,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Read, derive and attach all recommended marks for one session."""
    arm_path = Path(event_arm_table_path)
    event_path = Path(event_table_path)
    arm_table = read_table(arm_path)
    event_table = read_table(event_path)
    for name, table in (("event-arm", arm_table), ("event", event_table)):
        missing = sorted(set(SESSION_COLUMNS) - set(table))
        if missing:
            raise KeyError(f"{name} table is missing session columns {missing}")
    animal, session_date = parse_nwb_filename(nwb_file_name)

    def session_rows(table: pd.DataFrame) -> pd.DataFrame:
        keep = (
            table.animal.eq(animal)
            & table.session_date.eq(session_date)
            & table.epoch.eq(epoch_name)
        )
        return table.loc[keep].copy()

    session_arm = session_rows(arm_table)
    session_event = session_rows(event_table)
    if session_arm.empty or session_event.empty:
        raise ValueError(
            f"missing event-family rows for ({animal}, {session_date}, {epoch_name})"
        )
    marks = build_recommended_event_mark_table(session_arm, session_event)
    return (
        attach_recommended_event_marks(events, marks),
        {
            "event_arm_table_sha256": sha256_file(resolved_table_path(arm_path)),
            "event_table_sha256": sha256_file(resolved_table_path(event_path)),
        },
    )


def _block_name(event_class: str, mark: str | None = None) -> str:
    return f"{event_class}:event" if mark is None else f"{event_class}:{mark}"


def _without_block(design: KernelDesign, block_name: str) -> KernelDesign:
    if block_name not in design.block_slices:
        raise KeyError(f"design has no block {block_name!r}")
    span = design.block_slices[block_name]
    keep = np.ones(design.matrix.shape[1], dtype=bool)
    keep[span] = False
    blocks = tuple(block for block in design.blocks if block.name != block_name)
    width = span.stop - span.start
    slices: dict[str, slice] = {}
    for block in blocks:
        old = design.block_slices[block.name]
        shift = width if old.start > span.start else 0
        slices[block.name] = slice(old.start - shift, old.stop - shift)
    return KernelDesign(design.matrix[:, keep], slices, blocks)


def build_marked_family_ladder(
    events: dict[str, pd.DataFrame],
    fit_mask: pd.Series,
    spline_basis: NDArray,
    lags: NDArray,
    spec: MarkedFamilySpec,
    *,
    classes: tuple[str, ...] = MARKED_EVENT_CLASSES,
    min_marked_events: int = 10,
) -> MarkedFamilyLadder:
    """Build one joint family design and every target-specific reduced design."""
    standardized: dict[str, pd.DataFrame] = {}
    scales: dict[str, dict[str, MarkScale]] = {}
    for event_class in classes:
        if event_class not in events:
            raise KeyError(f"event class {event_class!r} is missing")
        try:
            standardized[event_class], scales[event_class] = standardize_event_marks(
                events[event_class],
                spec.all_marks,
                allow_missing=spec.all_marks,
                allow_constant=tuple(
                    mark
                    for mark in spec.nuisance_marks
                    if mark not in spec.required_varying_nuisance_marks
                ),
                min_marked_events=min_marked_events,
            )
        except ValueError as error:
            raise ModelIneligibleError(
                f"model {spec.name!r}, event class {event_class!r}: {error}"
            ) from error
    base_blocks = tuple(
        EventKernelBlock(_block_name(event_class), event_class)
        for event_class in classes
    )
    nuisance_blocks = tuple(
        EventKernelBlock(_block_name(event_class, mark), event_class, f"{mark}_z")
        for event_class in classes
        for mark in spec.nuisance_marks
    )
    scientific_blocks = tuple(
        EventKernelBlock(_block_name(event_class, mark), event_class, f"{mark}_z")
        for event_class in classes
        for mark in spec.scientific_marks
    )
    full = build_kernel_design(
        standardized,
        fit_mask,
        spline_basis,
        lags,
        (*base_blocks, *nuisance_blocks, *scientific_blocks),
    )
    reduced = {
        (event_class, mark): _without_block(full, _block_name(event_class, mark))
        for event_class in classes
        for mark in spec.scientific_marks
    }
    return MarkedFamilyLadder(
        spec=spec,
        reduced=reduced,
        full=full,
        events=standardized,
        scales=scales,
        classes=classes,
        fit_mask=fit_mask,
        spline_basis=np.asarray(spline_basis, dtype=float),
        lags=np.asarray(lags),
    )


def bounded_design_batch_size(
    requested: int,
    n_features: int,
    *,
    feature_budget: int = PROFILED_BATCH_FEATURE_BUDGET,
) -> int:
    """Bound scorer microbatches by the profiled batch×feature workload."""
    if requested < 1 or n_features < 1 or feature_budget < 1:
        raise ValueError("requested, n_features and feature_budget must be positive")
    return min(requested, max(1, feature_budget // n_features))


def _score(
    designs: NDArray,
    response: PreparedHeldoutResponse,
    penalty_sigma: float,
    *,
    requested_batch_size: int,
):
    width = np.asarray(designs).shape[-1]
    batch_size = bounded_design_batch_size(requested_batch_size, width)
    return score_poisson_designs_heldout(
        designs,
        response,
        make_ridge_penalty(width, penalty_sigma),
        design_batch_size=batch_size,
    )


def _event_mark_geometry(ladder: MarkedFamilyLadder) -> dict[str, dict[str, object]]:
    """Return event-level correlation/rank diagnostics for each replay class."""
    marks = ladder.spec.all_marks
    output: dict[str, dict[str, object]] = {}
    for event_class in ladder.classes:
        values = ladder.events[event_class][[f"{mark}_z" for mark in marks]].to_numpy(
            dtype=float
        )
        standard_deviation = values.std(axis=0, ddof=0)
        varying = standard_deviation > 0
        correlation = np.full((len(marks), len(marks)), np.nan)
        if varying.any():
            varying_correlation = np.atleast_2d(
                np.corrcoef(values[:, varying], rowvar=False)
            )
            correlation[np.ix_(varying, varying)] = varying_correlation
        off_diagonal = correlation.copy()
        np.fill_diagonal(off_diagonal, np.nan)
        finite_off_diagonal = np.abs(off_diagonal[np.isfinite(off_diagonal)])
        rank = int(np.linalg.matrix_rank(values))
        output[event_class] = {
            "marks": list(marks),
            "correlation": correlation.tolist(),
            "max_abs_off_diagonal_correlation": (
                float(finite_off_diagonal.max()) if finite_off_diagonal.size else None
            ),
            "rank": rank,
            "n_columns": int(values.shape[1]),
            "condition_number_if_full_rank": (
                float(np.linalg.cond(values)) if rank == values.shape[1] else None
            ),
            "constant_nuisance_marks": [
                mark
                for mark, is_varying in zip(marks, varying, strict=True)
                if not is_varying
            ],
        }
    return output


def fit_marked_family_ladder(
    ladder: MarkedFamilyLadder,
    counts: NDArray,
    folds: NDArray,
    *,
    penalty_sigma: float,
    design_batch_size: int = DEFAULT_NULL_DESIGN_BATCH_SIZE,
    prepared_response: PreparedHeldoutResponse | None = None,
) -> MarkedFamilyResults:
    """Fit one model family without p-values, shuffles, or response filtering."""
    if penalty_sigma <= 0 or not np.isfinite(penalty_sigma):
        raise ValueError("penalty_sigma must be finite and positive")
    fit_counts = np.asarray(counts, dtype=float)
    if fit_counts.ndim == 1:
        fit_counts = fit_counts[:, None]
    if fit_counts.ndim != 2 or fit_counts.shape[0] != ladder.full.matrix.shape[0]:
        raise ValueError("counts must have one row per fitted design row")
    if not np.isfinite(fit_counts).all() or np.any(fit_counts < 0):
        raise ValueError("counts must be finite and non-negative")
    fold_array = np.asarray(folds)
    if fold_array.shape != (fit_counts.shape[0],):
        raise ValueError("folds must contain one label per fitted row")

    if prepared_response is None:
        response = prepare_heldout_response(fit_counts, fold_array)
    else:
        response = prepared_response
        if (
            response.n_time != fit_counts.shape[0]
            or response.n_units != fit_counts.shape[1]
        ):
            raise ValueError(
                "prepared_response shape does not match counts: "
                f"{(response.n_time, response.n_units)} versus {fit_counts.shape}"
            )
        if response.n_folds != np.unique(fold_array).size:
            raise ValueError("prepared_response fold count does not match folds")
    comparison_order = tuple(ladder.reduced)
    reduced_designs = np.stack([ladder.reduced[key].matrix for key in comparison_order])
    started = time.perf_counter()
    reduced_score = _score(
        reduced_designs,
        response,
        penalty_sigma,
        requested_batch_size=design_batch_size,
    )
    full_score = _score(
        ladder.full.matrix[None, :, :],
        response,
        penalty_sigma,
        requested_batch_size=design_batch_size,
    )
    heldout_seconds = time.perf_counter() - started

    fit_started = time.perf_counter()
    fitted = fit_poisson_glms(
        ladder.full.matrix,
        fit_counts,
        penalty=make_ridge_penalty(ladder.full.matrix.shape[1], penalty_sigma),
        compute_covariance=False,
    )
    full_fit_seconds = time.perf_counter() - fit_started
    usable = np.asarray(fitted.converged, dtype=bool)
    statistics: dict[str, object] = {
        "family_full: converged": usable,
        "family_full: n_iter": np.asarray(fitted.n_iter, dtype=int),
        "family_full_heldout_deviance": full_score.total[0],
    }
    nan_deltas: dict[str, int] = {}
    for comparison_index, (event_class, mark) in enumerate(comparison_order):
        prefix = f"{event_class}:{mark}"
        reduced_total = reduced_score.total[comparison_index]
        delta = reduced_total - full_score.total[0]
        statistics[f"{prefix}: reduced_heldout_deviance"] = reduced_total
        statistics[f"{prefix}: heldout_delta_deviance"] = delta
        statistics[f"{prefix}: heldout_delta_deviance_folds"] = list(
            (reduced_score.per_fold[comparison_index] - full_score.per_fold[0]).T
        )
        statistics[f"{prefix}: full_all_folds_converged"] = np.all(
            full_score.converged[0], axis=0
        )
        statistics[f"{prefix}: reduced_all_folds_converged"] = np.all(
            reduced_score.converged[comparison_index], axis=0
        )
        nan_deltas[prefix] = int((~np.isfinite(delta)).sum())

    typical_curves: dict[str, NDArray] = {}
    mark_curves: dict[str, dict[str, NDArray]] = {}
    for event_class in ladder.classes:
        base = ladder.full.block_slices[_block_name(event_class)]
        parameters = np.asarray(fitted.params[:, base], dtype=float).copy()
        parameters[~usable] = np.nan
        typical_curves[event_class] = ladder.spline_basis @ parameters.T
        mark_curves[event_class] = {}
        for mark in ladder.spec.scientific_marks:
            span = ladder.full.block_slices[_block_name(event_class, mark)]
            parameters = np.asarray(fitted.params[:, span], dtype=float).copy()
            parameters[~usable] = np.nan
            mark_curves[event_class][mark] = ladder.spline_basis @ parameters.T

    diagnostics: dict[str, Any] = {
        "model_name": ladder.spec.name,
        "tier": ladder.spec.tier,
        "scientific_marks": list(ladder.spec.scientific_marks),
        "nuisance_marks": list(ladder.spec.nuisance_marks),
        "comparison_order": [list(key) for key in comparison_order],
        "scales": {
            event_class: {mark: scale._asdict() for mark, scale in class_scales.items()}
            for event_class, class_scales in ladder.scales.items()
        },
        "design_widths": {
            "full": int(ladder.full.matrix.shape[1]),
            "reduced": int(reduced_designs.shape[2]),
        },
        "effective_design_batch_sizes": {
            "full": bounded_design_batch_size(
                design_batch_size, ladder.full.matrix.shape[1]
            ),
            "reduced": bounded_design_batch_size(
                design_batch_size, reduced_designs.shape[2]
            ),
        },
        "full_design_rank": int(fitted.rank),
        "full_design_rank_deficiency": int(ladder.full.matrix.shape[1] - fitted.rank),
        "event_mark_geometry_by_class": _event_mark_geometry(ladder),
        "heldout_design_rank_by_fold": {
            "full": np.asarray(full_score.rank[0]).tolist(),
            **{
                f"{event_class}:{mark}:reduced": np.asarray(
                    reduced_score.rank[index]
                ).tolist()
                for index, (event_class, mark) in enumerate(comparison_order)
            },
        },
        "n_units_full_fit_not_converged": int((~usable).sum()),
        "n_units_nan_heldout_delta_by_comparison": nan_deltas,
        "timing_seconds": {
            "observed_heldout_scoring": heldout_seconds,
            "full_data_fit": full_fit_seconds,
        },
    }
    return MarkedFamilyResults(
        statistics=pd.DataFrame(statistics),
        typical_curves=typical_curves,
        mark_curves=mark_curves,
        diagnostics=diagnostics,
    )
