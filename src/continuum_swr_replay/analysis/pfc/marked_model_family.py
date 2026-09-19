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
    BatchedHeldoutResults,
    PreparedHeldoutResponse,
    fit_poisson_glms,
    make_ridge_penalty,
    prepare_heldout_response,
    score_poisson_designs_heldout,
)
from continuum_swr_replay.analysis.pfc.canonical_events import DEFAULT_EVENT_TABLE_PATH
from continuum_swr_replay.analysis.pfc.marked_glm import (
    DEFAULT_EVENT_ARM_TABLE_PATH,
    EVENT_DURATION_MARK,
    MARKED_EVENT_CLASSES,
    MIN_MARKED_EVENTS_PER_CLASS,
    EventKernelBlock,
    KernelDesign,
    MarkScale,
    _block_name,
    _without_block,
    build_kernel_design,
    standardize_event_marks,
)
from continuum_swr_replay.data_loaders.utils import parse_nwb_filename
from continuum_swr_replay.session_keys import SESSION_COLUMNS
from continuum_swr_replay.table_io import read_table, resolved_table_path, sha256_file

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
        CONTENT_RECENCY: (
            "leaf-content-weighted within-event relative visit recency, averaged "
            "over visited leaf arms only; undefined when the represented leaf "
            "content is entirely on never-visited arms"
        ),
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

    Attributes
    ----------
    name : str
        Model name; keys the shipped family and every output artifact.
    scientific_marks : tuple[str, ...]
        Target marks, each compared with the design lacking only its own
        per-class temporal block.
    nuisance_marks : tuple[str, ...]
        Marks kept in the full design and in every reduced design.
    question : str
        One-sentence scientific question the comparison answers.
    tier : str
        ``"core"`` or ``"interaction"``; interaction models carry product
        marks and their two constituent main effects.
    required_varying_nuisance_marks : tuple[str, ...]
        Nuisance marks that must vary within every class. They are the
        modifiers of an interaction target, whose product mark is otherwise
        identically zero; a session that lacks their variation is refused
        rather than fit.
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


# Six compact core designs carry six marked predictors per class. The three
# content horse-races and all three interaction sensitivities carry nine. Thus
# the full family uses only two compiled widths (155 and 221 columns with the
# shipped 11-function basis) while retaining the controls each question needs.
RECOMMENDED_MODEL_SPECS: Mapping[str, MarkedFamilySpec] = MappingProxyType(
    {
        "spatial_role": MarkedFamilySpec(
            name="spatial_role",
            scientific_marks=(
                CURRENT_ARM_FRACTION,
                PREVIOUS_ARM_FRACTION,
                CURRENT_PATCH_FRACTION,
            ),
            nuisance_marks=(
                EVENT_DURATION_MARK,
                LEAF_CONTENT_FRACTION,
                DECODE_ENTROPY,
            ),
            question=(
                "Does response timing vary with current, previous, or same-patch "
                "content?"
            ),
        ),
        "belief_content": MarkedFamilySpec(
            name="belief_content",
            scientific_marks=(CONTENT_UNCERTAINTY, CONTENT_VALUE),
            nuisance_marks=(
                EVENT_DURATION_MARK,
                LEAF_CONTENT_FRACTION,
                CURRENT_ARM_FRACTION,
                CONTENT_RECENT_REWARD,
                CONTENT_RECENCY,
                CONTENT_NOVELTY,
                DECODE_ENTROPY,
            ),
            question=(
                "Does mPFC track represented-arm uncertainty beyond value and "
                "reward history?"
            ),
        ),
        "reward_memory": MarkedFamilySpec(
            name="reward_memory",
            scientific_marks=(CONTENT_RECENT_REWARD, CONTENT_RECENCY),
            nuisance_marks=(
                EVENT_DURATION_MARK,
                LEAF_CONTENT_FRACTION,
                CURRENT_ARM_FRACTION,
                CONTENT_UNCERTAINTY,
                CONTENT_VALUE,
                DECODE_ENTROPY,
                CONTENT_NOVELTY,
            ),
            question=(
                "Do recent reward and recency explain responses beyond belief state?"
            ),
        ),
        "novelty_content": MarkedFamilySpec(
            name="novelty_content",
            scientific_marks=(CONTENT_NOVELTY,),
            nuisance_marks=(
                EVENT_DURATION_MARK,
                LEAF_CONTENT_FRACTION,
                CURRENT_ARM_FRACTION,
                CONTENT_UNCERTAINTY,
                CONTENT_VALUE,
                CONTENT_RECENT_REWARD,
                CONTENT_RECENCY,
                DECODE_ENTROPY,
            ),
            question=(
                "Does represented never-visited content matter beyond graded recency?"
            ),
        ),
        "trajectory_geometry": MarkedFamilySpec(
            name="trajectory_geometry",
            scientific_marks=(
                LOG_SPATIAL_EXTENT,
                LOG_REVERSALS,
                PAUSE_FRACTION,
                RETROSPECTIVE_BIAS,
            ),
            nuisance_marks=(EVENT_DURATION_MARK, DECODE_ENTROPY),
            question=(
                "Does response timing vary with extent, reversals, pauses, or "
                "direction?"
            ),
        ),
        "within_stop_timing": MarkedFamilySpec(
            name="within_stop_timing",
            scientific_marks=(LOG_STOP_TIME,),
            nuisance_marks=(
                EVENT_DURATION_MARK,
                CURRENT_ARM_FRACTION,
                LEAF_CONTENT_FRACTION,
                CONTENT_VALUE,
                DECODE_ENTROPY,
            ),
            question=(
                "Does the replay response change over the course of an immobility stop?"
            ),
        ),
        "switch_context": MarkedFamilySpec(
            name="switch_context",
            scientific_marks=(SWITCH_CONTEXT,),
            nuisance_marks=(
                EVENT_DURATION_MARK,
                CURRENT_ARM_FRACTION,
                CONTENT_VALUE,
                DECODE_ENTROPY,
                REWARDED_CONTEXT,
            ),
            question=(
                "Does patch-switch context change the response beyond known reward?"
            ),
        ),
        "reward_context": MarkedFamilySpec(
            name="reward_context",
            scientific_marks=(REWARDED_CONTEXT,),
            nuisance_marks=(
                EVENT_DURATION_MARK,
                CURRENT_ARM_FRACTION,
                CONTENT_VALUE,
                DECODE_ENTROPY,
                SWITCH_CONTEXT,
            ),
            question=(
                "Does known reward change the response beyond patch-switch context?"
            ),
        ),
        "graded_coupling": MarkedFamilySpec(
            name="graded_coupling",
            scientific_marks=(RIPPLE_POWER, MUA_RATE, THETA_POWER),
            nuisance_marks=(
                EVENT_DURATION_MARK,
                LEAF_CONTENT_FRACTION,
                DECODE_ENTROPY,
            ),
            question=(
                "Is binary SWR membership hiding a graded hippocampal coupling "
                "continuum?"
            ),
        ),
        "exploration_content_interaction": MarkedFamilySpec(
            name="exploration_content_interaction",
            scientific_marks=(CURRENT_X_SWITCH, UNCERTAINTY_X_SWITCH),
            nuisance_marks=(
                EVENT_DURATION_MARK,
                CURRENT_ARM_FRACTION,
                CONTENT_UNCERTAINTY,
                SWITCH_CONTEXT,
                LEAF_CONTENT_FRACTION,
                CONTENT_VALUE,
                DECODE_ENTROPY,
            ),
            question="Does exploration change current-arm or uncertainty modulation?",
            tier="interaction",
            required_varying_nuisance_marks=(SWITCH_CONTEXT,),
        ),
        "reward_content_interaction": MarkedFamilySpec(
            name="reward_content_interaction",
            scientific_marks=(CURRENT_X_REWARD, UNCERTAINTY_X_REWARD),
            nuisance_marks=(
                EVENT_DURATION_MARK,
                CURRENT_ARM_FRACTION,
                CONTENT_UNCERTAINTY,
                REWARDED_CONTEXT,
                LEAF_CONTENT_FRACTION,
                CONTENT_VALUE,
                DECODE_ENTROPY,
            ),
            question="Does known reward change current-arm or uncertainty modulation?",
            tier="interaction",
            required_varying_nuisance_marks=(REWARDED_CONTEXT,),
        ),
        "coupling_content_interaction": MarkedFamilySpec(
            name="coupling_content_interaction",
            scientific_marks=(CURRENT_X_RIPPLE, UNCERTAINTY_X_RIPPLE),
            nuisance_marks=(
                EVENT_DURATION_MARK,
                CURRENT_ARM_FRACTION,
                CONTENT_UNCERTAINTY,
                RIPPLE_POWER,
                LEAF_CONTENT_FRACTION,
                CONTENT_VALUE,
                DECODE_ENTROPY,
            ),
            question=(
                "Does graded ripple power change current-arm or uncertainty modulation?"
            ),
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
    """One joint full design and every leave-one-target-block-out design.

    Attributes
    ----------
    spec : MarkedFamilySpec
        The question this ladder was built for.
    reduced : dict[tuple[str, str], KernelDesign]
        Keyed by ``(event_class, scientific_mark)``: the full design minus
        only that class's block for that target mark.
    full : KernelDesign
        The joint design holding every class's ordinary, nuisance and target
        block, of shape ``(n_fit_bins, 1 + n_blocks * n_basis)``.
    events : dict[str, pd.DataFrame]
        Per event class, the standardized event frame the design was built
        from, carrying ``<mark>_z`` and ``<mark>_available`` columns.
    scales : dict[str, dict[str, MarkScale]]
        Per event class and mark, the centering and scaling reference.
    classes : tuple[str, ...]
        Event classes, in design-block and comparison order.
    fit_mask : pd.Series of bool
        Fitted-bin mask indexed by session time, shape ``(n_time,)``.
    spline_basis : NDArray, shape (n_lags, n_basis)
        Basis functions evaluated at ``lags``.
    lags : NDArray of int, shape (n_lags,)
        Lag indices in samples, ascending by one sample.
    """

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
    """Per-unit comparisons and full-data temporal kernels for one spec.

    Attributes
    ----------
    statistics : pd.DataFrame
        One row per unit, with the shared ``family_full_heldout_deviance``
        and ``full_all_folds_converged`` columns and, per
        ``"<event_class>:<mark>"`` comparison, its reduced deviance, delta,
        per-fold deltas and reduced convergence.
    typical_curves : dict[str, NDArray]
        ``typical_curves[c]``, shape ``(n_lags, n_units)``: class ``c``'s
        ordinary log-rate kernel at typical mark values. Non-converged units
        are ``nan``.
    mark_curves : dict[str, dict[str, NDArray]]
        ``mark_curves[c][mark]``, shape ``(n_lags, n_units)``: the change in
        class ``c``'s log-rate kernel per within-session, within-class SD of
        that target mark. Non-converged units are ``nan``.
    diagnostics : dict[str, Any]
        JSON-serializable audit record: mark scales, design widths, ranks,
        mark geometry, NaN-channel counts and timings.
    """

    statistics: pd.DataFrame
    typical_curves: dict[str, NDArray]
    mark_curves: dict[str, dict[str, NDArray]]
    diagnostics: dict[str, Any]


class ModelIneligibleError(ValueError):
    """A session/class lacks enough finite variation for a requested model."""


def _numeric(frame: pd.DataFrame, column: str) -> NDArray:
    """Read one column as float, coercing non-numeric entries to ``nan``."""
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _validate_event_arm_geometry(event_arm_table: pd.DataFrame) -> None:
    """Refuse an arm table that is not nine arms of unit total weight per event."""
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
    """Sum each event's arm weight over the arms flagged by one role column."""
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


def _leaf_weighted_mark(
    event_arm_table: pd.DataFrame,
    source: str,
    *,
    exclude: NDArray | None = None,
) -> pd.Series:
    """Average ``source`` over leaf arms, weighted by represented content.

    The average runs over the six reward-bearing leaf arms whose ``source``
    value is finite, weighted by ``arm_fraction`` and renormalized by the
    represented leaf mass, so trunk arms' structural entries never enter.
    ``exclude`` additionally removes rows from *both* the numerator and the
    denominator; it is for sources whose per-arm entry can be a structural fill
    rather than a measurement, where including the row would displace the
    average by an arbitrary amount rather than merely reweight it. The
    "0 or 6 finite leaf rows per event" availability invariant is enforced on
    the finite mask *before* ``exclude`` is applied, so excluding rows never
    masquerades as partially missing data.

    Parameters
    ----------
    event_arm_table : pd.DataFrame
        Canonical nine-row-per-event arm table, shape ``(9 * n_events, ...)``.
        Needs ``event_id``, ``candidate_arm``, ``arm_fraction`` and ``source``.
    source : str
        Per-arm column to average.
    exclude : NDArray or None, shape (9 * n_events,), optional
        Boolean row mask of entries to drop from numerator and denominator.
        Default ``None`` keeps every finite leaf row.

    Returns
    -------
    pd.Series
        One value per ``event_id`` in first-appearance order. NaN where the
        retained leaf mass is zero, i.e. where the quantity is undefined.

    Raises
    ------
    ValueError
        If an event has some but not all six leaf values finite, or if
        ``exclude`` does not have one entry per table row.
    """
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
    retained = finite_leaf
    if exclude is not None:
        excluded = np.asarray(exclude, dtype=bool)
        if excluded.shape != finite_leaf.shape:
            raise ValueError(
                f"exclude must have one entry per arm row; got {excluded.shape} "
                f"for {finite_leaf.shape}"
            )
        retained = finite_leaf & ~excluded
    weights = _numeric(event_arm_table, "arm_fraction")
    numerator = (
        pd.Series(np.where(retained, weights * values, 0.0))
        .groupby(groups.to_numpy(), sort=False)
        .sum()
    )
    denominator = (
        pd.Series(np.where(retained, weights, 0.0))
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

    ``content_weighted_relative_recency`` additionally drops never-visited leaf
    arms from both numerator and denominator.  Their recency entry is the
    right-censoring fill ``n_completed + 1`` (up to ~165 trials on the shipped
    table), a property of the censoring rule rather than a measured recency, so
    admitting it would set the mark off the censoring rule; the upstream
    producer ``center_visit_recency_within_event`` excludes them for the same
    reason.  An event whose represented leaf content sits entirely on
    never-visited arms therefore has no recency estimand and is NaN, which the
    ``allow_missing`` regressor path zero-fills after standardization.  The
    never-visited mass itself is still carried by
    ``content_weighted_novelty_fraction``, and ``content_weighted_recent_reward``
    keeps every finite leaf row because its never-visited entry is a true 0.0.

    Parameters
    ----------
    event_arm_table : pd.DataFrame, shape (9 * n_events, ...)
        One session's canonical nine-row-per-event arm table.
    event_table : pd.DataFrame, shape (n_events, ...)
        The same session's canonical one-row-per-event table.

    Returns
    -------
    pd.DataFrame, shape (n_events, ...)
        One row per event, keyed by ``event_id`` in the arm table's
        first-appearance order, carrying the event metadata columns and every
        mark named in :data:`MARK_DESCRIPTIONS`. Undefined marks are ``nan``.

    Raises
    ------
    KeyError
        If either table lacks a required column.
    ValueError
        If the tables are empty, disagree on the event set, timing or
        ``is_swr``, if the arm geometry is not nine unit-summing arms per
        event, if a leaf quantity is only partially available within an
        event, or if a duration, time, extent or reversal count is not a
        finite non-negative value.
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
    never_visited = _numeric(event_arm_table, "is_candidate_arm_never_visited") == 1.0
    for source, output, exclude in (
        ("reward_belief_variance_candidate_arm_preoutcome", CONTENT_UNCERTAINTY, None),
        ("reward_belief_mean_candidate_arm_preoutcome", CONTENT_VALUE, None),
        ("reward_fraction_candidate_arm_last10_visits", CONTENT_RECENT_REWARD, None),
        ("visit_recency_trials_event_leaf_centered", CONTENT_RECENCY, never_visited),
        ("is_candidate_arm_never_visited", CONTENT_NOVELTY, None),
    ):
        marks[output] = _leaf_weighted_mark(
            event_arm_table, source, exclude=exclude
        ).reindex(arm_ids)

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
    """Join one-row-per-event recommended marks to canonical class frames.

    Parameters
    ----------
    events : dict[str, pd.DataFrame]
        Per event class, the cohort-restricted canonical events carrying
        ``event_id``, ``start_time`` and ``end_time``.
    mark_table : pd.DataFrame, shape (n_events, ...)
        Output of :func:`build_recommended_event_mark_table`.
    classes : tuple[str, ...]
        Event classes to mark; each must be present in ``events``.

    Returns
    -------
    dict[str, pd.DataFrame]
        Copies of the input frames, in their original row order, with every
        available mark column appended.

    Raises
    ------
    KeyError
        If ``mark_table`` lacks a required column or a class is missing.
    ValueError
        If an ``event_id`` is duplicated or unmarked, if the marks' ``is_swr``
        or timing disagree with a class frame, if an already-present mark
        column disagrees with the family table, or if the join changes a
        frame's row order.
    """
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
    """Read, derive and attach all recommended marks for one session.

    Parameters
    ----------
    nwb_file_name : str
        Session NWB file, e.g. ``"chimi20200212_.nwb"``.
    epoch_name : str
        Epoch within that file, e.g. ``"02_r1"``.
    events : dict[str, pd.DataFrame]
        Per event class, the canonical events to mark.
    event_arm_table_path : str or Path, optional
        Canonical event-arm table to read. Defaults to
        :data:`DEFAULT_EVENT_ARM_TABLE_PATH`.
    event_table_path : str or Path, optional
        Canonical event table to read. Defaults to
        :data:`DEFAULT_EVENT_TABLE_PATH`.

    Returns
    -------
    marked : dict[str, pd.DataFrame]
        ``events`` with every recommended mark attached.
    provenance : dict[str, str]
        SHA-256 digests of the two tables the marks were derived from.

    Raises
    ------
    KeyError
        If either table lacks the canonical session columns.
    ValueError
        If the session has no rows in either table, or if mark derivation or
        attachment refuses the session's tables.
    """
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


def build_marked_family_ladder(
    events: dict[str, pd.DataFrame],
    fit_mask: pd.Series,
    spline_basis: NDArray,
    lags: NDArray,
    spec: MarkedFamilySpec,
    *,
    classes: tuple[str, ...] = MARKED_EVENT_CLASSES,
    min_marked_events: int = MIN_MARKED_EVENTS_PER_CLASS,
) -> MarkedFamilyLadder:
    """Build one joint family design and every target-specific reduced design.

    Every mark is standardized within session and event class. Nuisance marks
    other than ``spec.required_varying_nuisance_marks`` may be constant, in
    which case their standardized block is identically zero; a constant target
    or required-varying modifier makes the session ineligible for the model.

    Parameters
    ----------
    events : dict[str, pd.DataFrame]
        Per event class, the events carrying every raw mark column.
    fit_mask : pd.Series of bool, shape (n_time,)
        Fitted-bin mask indexed by session time.
    spline_basis : NDArray, shape (n_lags, n_basis)
        Basis functions evaluated at ``lags``.
    lags : NDArray of int, shape (n_lags,)
        Lag indices in samples, ascending by one sample.
    spec : MarkedFamilySpec
        The question whose full and reduced designs are built.
    classes : tuple[str, ...]
        Event classes, in design-block order; each must be in ``events``.
    min_marked_events : int
        Fewest finite events a class may contribute to a marked kernel.

    Returns
    -------
    MarkedFamilyLadder

    Raises
    ------
    KeyError
        If an event class is missing from ``events``.
    ModelIneligibleError
        If a class cannot support the spec: a mark is non-finite, has fewer
        than ``min_marked_events`` finite events, or is constant when the spec
        does not permit it. The message names the model, the class and the
        mark.
    """
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
    """Bound scorer microbatches by the profiled batch×feature workload.

    Parameters
    ----------
    requested : int
        Designs per device batch the caller asked for.
    n_features : int
        Columns of one design in the batch.
    feature_budget : int
        Batched feature columns the profiled device tolerates. Defaults to
        :data:`PROFILED_BATCH_FEATURE_BUDGET`.

    Returns
    -------
    int
        ``min(requested, feature_budget // n_features)``, floored at one so a
        design wider than the whole budget is still scored, one at a time.

    Raises
    ------
    ValueError
        If any argument is below one.
    """
    if requested < 1 or n_features < 1 or feature_budget < 1:
        raise ValueError("requested, n_features and feature_budget must be positive")
    return min(requested, max(1, feature_budget // n_features))


def _score(
    designs: NDArray,
    response: PreparedHeldoutResponse,
    penalty_sigma: float,
    *,
    requested_batch_size: int,
) -> BatchedHeldoutResults:
    """Score equal-width designs under one ridge prior and the batch bound."""
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
    """Fit one model family without p-values, shuffles, or response filtering.

    The full design is fit and scored once and shared by every comparison, so
    each target's held-out delta is the deviance the model loses when only
    that class's block for that mark is removed from the common design.

    Parameters
    ----------
    ladder : MarkedFamilyLadder
        The joint full design and every leave-one-target-block-out design.
    counts : NDArray, shape (n_fit_bins, n_units) or (n_fit_bins,)
        Spike counts on the ladder's fitted bins.
    folds : NDArray of int, shape (n_fit_bins,)
        Held-out fold label of each fitted bin.
    penalty_sigma : float
        Ridge prior SD applied to every non-intercept column.
    design_batch_size : int
        Designs scored per device batch, before the width bound of
        :func:`bounded_design_batch_size`.
    prepared_response : PreparedHeldoutResponse, optional
        A response already transferred to the device, to be reused across the
        models of one session. It must have been prepared from exactly these
        counts and this fold assignment.

    Returns
    -------
    MarkedFamilyResults

    Raises
    ------
    ValueError
        If ``penalty_sigma`` is not finite and positive, if ``counts`` do not
        match the design rows or are not finite and non-negative, if ``folds``
        do not label every fitted row, or if ``prepared_response`` was built
        from different counts or a different fold assignment.
    """
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
        # A response prepared from a different fold assignment would score
        # every design on the wrong held-out rows, silently and without a
        # shape error, so the stored labels themselves are compared.
        if not np.array_equal(
            np.asarray(response.folds), np.asarray(fold_array, dtype=np.int64)
        ):
            raise ValueError(
                "prepared_response was built from a different fold assignment"
            )
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
        # The full model is fit once and shared, so its per-fold convergence
        # is one column, not one copy per comparison.
        "full_all_folds_converged": np.all(full_score.converged[0], axis=0),
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
