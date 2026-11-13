"""Recommended descriptive marked-kernel models for mPFC replay responses.

The single-mark model in :mod:`continuum_swr_replay.analysis.pfc.marked_glm`
establishes the temporal estimand.  This module extends that estimator to a
small family of question-specific, jointly adjusted designs.  A family shares
one full fit across all of its target marks and compares each target with the
otherwise identical model lacking only that target's temporal block.

There are no p-values here, and no neuron selection beyond leaving out a unit
with no spike when the baseline carries a prior (:func:`fitted_unit_indices`).  The outputs are
descriptive effect kernels and blocked held-out prediction changes, plus an
optional shuffled-mark reference for each slope kernel (``slope_permutations``
draws; 0, i.e. none, in the shipped follow-up run).  The models use only event/content tables already shipped
by the replay pipeline; they do not decode mPFC population activity.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
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
    fit_poisson_column_variants,
    fit_poisson_glms,
    make_ridge_penalty,
    prepare_heldout_response,
    score_poisson_designs_heldout,
    unusable_as_nan,
)
from continuum_swr_replay.analysis.pfc.event_sets import SWR_NO_NON_LOCAL
from continuum_swr_replay.analysis.pfc.marked_glm import (
    EVENT_DURATION_MARK,
    EXTENDED_MARKED_EVENT_CLASSES,
    MARKED_EVENT_CLASSES,
    MIN_LEVEL_COUNT,
    MIN_MARKED_EVENTS_PER_CLASS,
    RIPPLE_POWER_MARK,
    EventKernelBlock,
    KernelDesign,
    MarkScale,
    _block_name,
    _without_block,
    build_kernel_design,
    mark_block_geometry,
    mark_blocks_from_weights,
    standardize_event_marks,
)
from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH
from continuum_swr_replay.decoding.identifiers import parse_nwb_filename
from continuum_swr_replay.paths import table_path
from continuum_swr_replay.runs.io import sha256_file
from continuum_swr_replay.tables.columns import require_columns
from continuum_swr_replay.tables.io import read_table, resolved_table_path
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

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
RIPPLE_POWER = RIPPLE_POWER_MARK
MUA_RATE = "mua_rate_z"
THETA_POWER = "theta_z"
DECODE_ENTROPY = "posterior_entropy_mean"
CURRENT_X_SWITCH = "current_arm_fraction_x_switch"
UNCERTAINTY_X_SWITCH = "content_uncertainty_x_switch"
CURRENT_X_REWARD = "current_arm_fraction_x_rewarded"
UNCERTAINTY_X_REWARD = "content_uncertainty_x_rewarded"
CURRENT_X_RIPPLE = "current_arm_fraction_x_ripple_power"
UNCERTAINTY_X_RIPPLE = "content_uncertainty_x_ripple_power"
TRIPS_TO_PATCH_SWITCH = "trips_to_patch_switch"
NEXT_TRIP_SWITCHES_PATCH = "next_trip_switches_patch"

#: ``trips_to_patch_switch`` counts at most this many trips; a value of 3
#: means "three or more trips stay in the patch first".
TRIPS_TO_PATCH_SWITCH_CAP = 3

#: Marks that are properties of the stop, shared by every event in it. Their
#: shuffled-mark reference permutes whole stops' values among stops.
STOP_LEVEL_MARKS = (REWARDED_CONTEXT, TRIPS_TO_PATCH_SWITCH, NEXT_TRIP_SWITCHES_PATCH)

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
        TRIPS_TO_PATCH_SWITCH: (
            "completed trips from this stop until the first later trip that "
            "leaves the patch (0 when the next trip leaves), capped at 3; "
            "undefined after the session's last patch switch"
        ),
        NEXT_TRIP_SWITCHES_PATCH: (
            "whether the trip that starts at this stop leaves the patch "
            "(trips_to_patch_switch == 0)"
        ),
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
        marks and their two constituent main effects. ``"marginal"`` is a
        generated single-target spec (:func:`marginal_specs`).
    required_varying_nuisance_marks : tuple[str, ...]
        Nuisance marks that must vary within every class. They are the
        modifiers of an interaction target, whose product mark is otherwise
        identically zero; a session that lacks their variation is refused
        rather than fit.
    event_classes : tuple[str, ...]
        The marked event classes, in design-block order. A spec that lists
        ``"SWR_no_non_local"`` may carry only :data:`SWR_ONLY_MARKS`.
    pooled : bool
        When True, each target gets one marked block shared by every class
        (its events standardized within their own class, then convolved
        together), so its slope kernel is one curve for all classes; every
        class keeps its own base kernel and adjustment blocks.
    baseline_context_marks : tuple[str, ...]
        Stop-level context marks that also enter the per-stop baseline
        (:func:`~continuum_swr_replay.analysis.pfc.marked_glm.build_baseline_columns`
        ``context_columns``), so a stationary rate difference between stops is
        not read as an event-locked effect of that mark.
    """

    name: str
    scientific_marks: tuple[str, ...]
    nuisance_marks: tuple[str, ...]
    question: str
    tier: str = "core"
    required_varying_nuisance_marks: tuple[str, ...] = ()
    event_classes: tuple[str, ...] = MARKED_EVENT_CLASSES
    pooled: bool = False
    baseline_context_marks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        all_marks = (*self.nuisance_marks, *self.scientific_marks)
        if not self.name or not self.scientific_marks:
            raise ValueError("a model spec needs a name and at least one target mark")
        if len(all_marks) != len(set(all_marks)):
            raise ValueError(f"model {self.name!r} contains duplicate marks")
        unknown = sorted(set(all_marks) - set(MARK_DESCRIPTIONS))
        if unknown:
            raise ValueError(f"model {self.name!r} uses unknown marks {unknown}")
        if self.tier not in SPEC_TIERS:
            raise ValueError(f"tier must be one of {SPEC_TIERS}")
        if not set(self.required_varying_nuisance_marks).issubset(self.nuisance_marks):
            raise ValueError("required varying marks must be nuisance marks")
        if not self.event_classes or len(set(self.event_classes)) != len(
            self.event_classes
        ):
            raise ValueError("event_classes must be non-empty and unique")
        unknown_classes = set(self.event_classes) - set(EXTENDED_MARKED_EVENT_CLASSES)
        if unknown_classes:
            raise ValueError(f"unknown event classes {sorted(unknown_classes)}")
        if SWR_NO_NON_LOCAL in self.event_classes and not set(all_marks) <= set(
            SWR_ONLY_MARKS
        ):
            raise ValueError(
                f"model {self.name!r}: {SWR_NO_NON_LOCAL} events carry only "
                f"{list(SWR_ONLY_MARKS)}"
            )
        if not set(self.baseline_context_marks) <= set(STOP_LEVEL_MARKS):
            raise ValueError("baseline context marks must be stop-level marks")

    @property
    def all_marks(self) -> tuple[str, ...]:
        """Marks in deterministic design order: adjustments, then targets."""
        return (*self.nuisance_marks, *self.scientific_marks)

    @property
    def target_classes(self) -> tuple[str, ...]:
        """Classes keying the target blocks: ``("pooled",)`` when pooled."""
        return (POOLED_CLASS,) if self.pooled else self.event_classes

    def for_classes(self, available: tuple[str, ...]) -> MarkedFamilySpec:
        """The spec restricted to the event classes a session was cut with.

        Parameters
        ----------
        available : tuple of str
            Marked classes the session's event set provides.

        Returns
        -------
        MarkedFamilySpec
            A copy keeping only ``event_classes`` present in ``available``.

        Raises
        ------
        ValueError
            If none is left.
        """
        kept = tuple(c for c in self.event_classes if c in available)
        if not kept:
            raise ValueError(f"model {self.name!r} has no class among {available}")
        return replace(self, event_classes=kept)


#: The marks an SWR without replay carries: it has no decoded content, no
#: decode-quality summary and no stop-context join of its own.
SWR_ONLY_MARKS = (EVENT_DURATION_MARK, RIPPLE_POWER_MARK)

#: Pseudo-class of a pooled marked block: every class's events, each mark
#: standardized within its own class, convolved into one block.
POOLED_CLASS = "pooled"

#: Spec tiers: the v1 core and interaction models, and the generated
#: single-target marginal tier.
SPEC_TIERS = ("core", "interaction", "marginal")

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

#: Content targets: their marginal specs also adjust for leaf content.
CONTENT_TARGETS = (CURRENT_ARM_FRACTION, CONTENT_RECENT_REWARD, CONTENT_UNCERTAINTY)

# The follow-up questions: what is replayed, how strong the event is, and the
# stop it happens in. The ripple-power models also mark the SWRs without
# replay when a session is cut with the extended event set.
FOLLOWUP_MODEL_SPECS: Mapping[str, MarkedFamilySpec] = MappingProxyType(
    {
        "replay_content": MarkedFamilySpec(
            name="replay_content",
            scientific_marks=CONTENT_TARGETS,
            nuisance_marks=(EVENT_DURATION_MARK, LEAF_CONTENT_FRACTION, DECODE_ENTROPY),
            question=(
                "Does the response change with replay of the current arm, of "
                "recently rewarded arms, or of uncertain arms?"
            ),
        ),
        "event_strength": MarkedFamilySpec(
            name="event_strength",
            scientific_marks=(EVENT_DURATION_MARK, RIPPLE_POWER, MUA_RATE),
            nuisance_marks=(DECODE_ENTROPY,),
            question=(
                "Does the response scale with event duration, ripple power or "
                "hippocampal multiunit rate?"
            ),
        ),
        "stop_context": MarkedFamilySpec(
            name="stop_context",
            scientific_marks=(REWARDED_CONTEXT, TRIPS_TO_PATCH_SWITCH),
            nuisance_marks=(EVENT_DURATION_MARK, CURRENT_ARM_FRACTION, DECODE_ENTROPY),
            question=(
                "Does the response change with the stop's reward outcome or with "
                "how close the next patch switch is?"
            ),
            baseline_context_marks=(TRIPS_TO_PATCH_SWITCH,),
        ),
        "ripple_power": MarkedFamilySpec(
            name="ripple_power",
            scientific_marks=(RIPPLE_POWER,),
            nuisance_marks=(EVENT_DURATION_MARK,),
            question=(
                "How does the response scale with ripple power in each class, "
                "SWRs without replay included?"
            ),
            event_classes=EXTENDED_MARKED_EVENT_CLASSES,
        ),
        "ripple_power_pooled": MarkedFamilySpec(
            name="ripple_power_pooled",
            scientific_marks=(RIPPLE_POWER,),
            nuisance_marks=(EVENT_DURATION_MARK,),
            question=(
                "How does the response scale with ripple power over all classes "
                "together?"
            ),
            event_classes=EXTENDED_MARKED_EVENT_CLASSES,
            pooled=True,
        ),
    }
)
FOLLOWUP_MODEL_NAMES = tuple(FOLLOWUP_MODEL_SPECS)


def marginal_specs(
    joint_specs: Mapping[str, MarkedFamilySpec] = FOLLOWUP_MODEL_SPECS,
    *,
    extra_targets: tuple[str, ...] = (NEXT_TRIP_SWITCHES_PATCH,),
) -> Mapping[str, MarkedFamilySpec]:
    """One single-target spec per distinct target of the joint models.

    The marginal tier is the family's total effect: each target fit alone,
    adjusted only for event duration and decode entropy (duration itself only
    for entropy), content targets also for leaf content. Shared variance the
    joint models credit to no target goes to each target here. Every marginal
    spec is fit on the two replay classes; a stop-level context target other
    than the reward outcome (already a baseline column) is also a baseline
    context column.

    Parameters
    ----------
    joint_specs : Mapping of str to MarkedFamilySpec
        The joint models whose targets are generated, in order.
    extra_targets : tuple of str
        Further targets with a marginal spec but no joint model.

    Returns
    -------
    Mapping of str to MarkedFamilySpec
        Keyed ``marginal_<mark>``, in first-appearance order.
    """
    targets = list(
        dict.fromkeys(
            [
                *(m for spec in joint_specs.values() for m in spec.scientific_marks),
                *extra_targets,
            ]
        )
    )
    specs: dict[str, MarkedFamilySpec] = {}
    for mark in targets:
        if mark == EVENT_DURATION_MARK:
            adjustments: tuple[str, ...] = (DECODE_ENTROPY,)
        elif mark in CONTENT_TARGETS:
            adjustments = (EVENT_DURATION_MARK, LEAF_CONTENT_FRACTION, DECODE_ENTROPY)
        else:
            adjustments = (EVENT_DURATION_MARK, DECODE_ENTROPY)
        name = f"marginal_{mark}"
        specs[name] = MarkedFamilySpec(
            name=name,
            scientific_marks=(mark,),
            nuisance_marks=adjustments,
            question=f"How does the response change with {mark} alone?",
            tier="marginal",
            baseline_context_marks=(
                (mark,) if mark in STOP_LEVEL_MARKS and mark != REWARDED_CONTEXT else ()
            ),
        )
    return MappingProxyType(specs)


MARGINAL_MODEL_SPECS = marginal_specs()
MARGINAL_MODEL_NAMES = tuple(MARGINAL_MODEL_SPECS)


def model_set_of(model_name: str) -> str:
    """``"recommended"`` for a v1 model, ``"followup"`` for the follow-up set.

    The follow-up set holds :data:`FOLLOWUP_MODEL_SPECS` and their marginal
    tier.

    Raises
    ------
    KeyError
        For a name in neither set.
    """
    if model_name in RECOMMENDED_MODEL_SPECS:
        return "recommended"
    if model_name in FOLLOWUP_MODEL_SPECS or model_name in MARGINAL_MODEL_SPECS:
        return "followup"
    raise KeyError(f"unknown model {model_name!r}")


#: Every spec a runner can fit, by name.
ALL_MODEL_SPECS: Mapping[str, MarkedFamilySpec] = MappingProxyType(
    {**RECOMMENDED_MODEL_SPECS, **FOLLOWUP_MODEL_SPECS, **MARGINAL_MODEL_SPECS}
)


@dataclass(frozen=True)
class MarkedFamilyLadder:
    """One joint full design and every leave-one-target-block-out design.

    Attributes
    ----------
    spec : MarkedFamilySpec
        The question this ladder was built for.
    reduced : dict[tuple[str, str], KernelDesign]
        Keyed by ``(target_class, scientific_mark)``: the full design minus
        only that target block.
    full : KernelDesign
        The joint design holding every class's ordinary, nuisance and target
        block, of shape ``(n_fit_bins, 1 + n_blocks * n_basis)``.
    events : dict[str, pd.DataFrame]
        Per event class, the standardized event frame the design was built
        from, carrying ``<mark>_z`` and ``<mark>_available`` columns; for a
        pooled spec also ``"pooled"``, every class's rows stacked in class
        order with their class in ``event_class``.
    scales : dict[str, dict[str, MarkScale]]
        Per event class and mark, the centering and scaling reference.
    classes : tuple[str, ...]
        Event classes, in design-block order; each has a base kernel.
    target_classes : tuple[str, ...]
        Keys of the target blocks, in comparison order: ``classes``, or
        ``("pooled",)`` for a pooled spec.
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
    target_classes: tuple[str, ...]
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
        that target mark (``c`` is ``"pooled"`` for a pooled spec).
        Non-converged units are ``nan``.
    diagnostics : dict[str, Any]
        JSON-serializable audit record: mark scales, design widths, ranks,
        mark geometry, NaN-channel counts and timings.
    slope_null : dict[str, dict[str, NDArray]]
        ``slope_null[c][mark]``, shape ``(n_draws, n_report_lags, n_units)``,
        float32: the same slope kernel refit with the target's values
        shuffled (:func:`fit_marked_family_slope_null`). Empty without draws.
    full_params : NDArray, shape (n_units, n_features), optional
        The observed full-data fit's coefficients (NaN for a unit that did not
        converge); they start the shuffled refits.
    """

    statistics: pd.DataFrame
    typical_curves: dict[str, NDArray]
    mark_curves: dict[str, dict[str, NDArray]]
    diagnostics: dict[str, Any]
    slope_null: dict[str, dict[str, NDArray]] = field(default_factory=dict)
    full_params: NDArray | None = None


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


def _patch_of_arm(arms: NDArray) -> NDArray:
    """Patch of each leaf arm (well); NaN where the arm is unknown."""
    return np.array(
        [
            float(TRACK_SEGMENT_TO_PATCH[int(arm)]) if np.isfinite(arm) else np.nan
            for arm in np.asarray(arms, dtype=float)
        ]
    )


def stop_patch_switch_marks(stops: pd.DataFrame) -> pd.DataFrame:
    """Forward patch-switch context of every stop of one session.

    Trips are the session's completed trials. Trial ``k`` ends at the arm that
    stops with ``completed_trial_id == k`` record as ``completed_trial_arm``,
    and stops with ``completed_trial_id == k - 1`` record the same arm as their
    ``next_arm``; the two are checked against each other. The trip that starts
    at a stop whose last completed trial is ``k`` is trial ``k + 1``, from
    trial ``k``'s arm to trial ``k + 1``'s; it switches patch when the two
    arms' patches differ. ``trips_to_patch_switch`` is the number of trips,
    from that one on, that stay in the patch before the first one that leaves
    it: 0 when the next trip leaves. It is capped at
    :data:`TRIPS_TO_PATCH_SWITCH_CAP`, which it also takes when no switch
    follows but at least that many staying trips do; otherwise it is undefined
    (NaN) for a stop with no completed trial, when a trip on the way has an
    unknown arm, and after the session's last switch.

    Parameters
    ----------
    stops : pd.DataFrame, shape (n_stops, ...)
        One session's stop-table rows, every stop (not only a cohort):
        ``stop_idx``, ``completed_trial_id`` (nullable integer),
        ``completed_trial_arm`` and ``next_arm`` (0--5 or NaN).

    Returns
    -------
    pd.DataFrame, shape (n_stops, 2)
        Indexed by ``stop_idx``: ``trips_to_patch_switch`` (0, 1, 2, 3 or NaN)
        and ``next_trip_switches_patch`` (1.0, 0.0 or NaN).

    Raises
    ------
    ValueError
        If a required column is missing, ``stop_idx`` repeats, or two stops
        disagree on a trial's arm.
    """
    require_columns(
        stops,
        ["stop_idx", "completed_trial_id", "completed_trial_arm", "next_arm"],
        "the stops",
    )
    if stops["stop_idx"].duplicated().any():
        raise ValueError("stop_idx must be unique within a session")
    trial = pd.to_numeric(stops["completed_trial_id"], errors="coerce").to_numpy(
        dtype=float
    )
    arm_of_trial: dict[int, float] = {}
    for trial_ids, arms in (
        (trial, _numeric(stops, "completed_trial_arm")),
        (trial + 1, _numeric(stops, "next_arm")),
    ):
        for trial_id, arm in zip(trial_ids, arms, strict=True):
            if not (np.isfinite(trial_id) and np.isfinite(arm)):
                continue
            known = arm_of_trial.setdefault(int(trial_id), float(arm))
            if known != arm:
                raise ValueError(
                    f"stops disagree on the arm of trial {int(trial_id)}: "
                    f"{known} and {arm}"
                )
    n_trials = max(arm_of_trial, default=-1) + 1
    patch = _patch_of_arm(
        np.array([arm_of_trial.get(index, np.nan) for index in range(n_trials)])
    )
    # switches[k]: whether trial k leaves the patch trial k - 1 ended in.
    switches = np.full(n_trials, np.nan)
    known = np.isfinite(patch[1:]) & np.isfinite(patch[:-1])
    switches[1:][known] = (patch[1:] != patch[:-1])[known].astype(float)
    # trips_from[k]: trips from trial k on that stay before the first that
    # leaves, filled backward; NaN after the last switch or past an unknown.
    # With no switch ahead, the capped value is still known once at least
    # TRIPS_TO_PATCH_SWITCH_CAP staying trips are: it is the cap.
    trips_from = np.full(n_trials + 1, np.nan)
    known_stays = np.zeros(n_trials + 1)
    for index in range(n_trials - 1, -1, -1):
        if switches[index] == 1.0:
            trips_from[index] = 0.0
        elif switches[index] == 0.0:
            trips_from[index] = trips_from[index + 1] + 1.0
            known_stays[index] = known_stays[index + 1] + 1.0
    trips_from = np.where(
        np.isnan(trips_from) & (known_stays >= TRIPS_TO_PATCH_SWITCH_CAP),
        TRIPS_TO_PATCH_SWITCH_CAP,
        np.minimum(trips_from, TRIPS_TO_PATCH_SWITCH_CAP),
    )
    next_trial = trial + 1
    has_next = np.isfinite(next_trial) & (next_trial < n_trials)
    trips = np.full(len(stops), np.nan)
    trips[has_next] = trips_from[next_trial[has_next].astype(int)]
    out = pd.DataFrame(
        {
            TRIPS_TO_PATCH_SWITCH: trips,
            NEXT_TRIP_SWITCHES_PATCH: np.where(
                np.isfinite(trips), (trips == 0.0).astype(float), np.nan
            ),
        },
        index=pd.Index(stops["stop_idx"].to_numpy(), name="stop_idx"),
    )
    return out


def load_session_stop_patch_switch_marks(
    nwb_file_name: str,
    epoch_name: str,
    *,
    stop_table_path: str | Path | None = None,
) -> tuple[pd.DataFrame, str]:
    """Read one session's stops and derive their forward patch-switch context.

    Parameters
    ----------
    nwb_file_name : str
        Session NWB file, e.g. ``"chimi20200212_.nwb"``.
    epoch_name : str
        Epoch within that file, e.g. ``"02_r1"``.
    stop_table_path : str or Path, optional
        Stop table to read. Defaults to ``table_path("stop_table")``,
        resolved at the call.

    Returns
    -------
    marks : pd.DataFrame
        :func:`stop_patch_switch_marks` over every stop of the session.
    stop_table_sha256 : str
        Digest of the table read.

    Raises
    ------
    ValueError
        If the session has no stops.
    """
    path = (
        table_path("stop_table") if stop_table_path is None else Path(stop_table_path)
    )
    table = read_table(path)
    animal, session_date = parse_nwb_filename(nwb_file_name)
    session = table.loc[
        table.animal.eq(animal)
        & table.session_date.eq(session_date)
        & table.epoch.eq(epoch_name)
    ]
    if session.empty:
        raise ValueError(f"no stops for ({animal}, {session_date}, {epoch_name})")
    return (
        stop_patch_switch_marks(session),
        sha256_file(resolved_table_path(path)),
    )


def build_recommended_event_mark_table(
    event_arm_table: pd.DataFrame,
    event_table: pd.DataFrame,
    stop_marks: pd.DataFrame | None = None,
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
    stop_marks : pd.DataFrame, optional
        :func:`stop_patch_switch_marks` for the session, indexed by
        ``stop_idx``; joined to each event through the event table's
        ``stop_idx``. ``None`` leaves the forward patch-switch marks out.

    Returns
    -------
    pd.DataFrame, shape (n_events, ...)
        One row per event, keyed by ``event_id`` in the arm table's
        first-appearance order, carrying the event metadata columns (and
        ``stop_idx`` when the event table has it) and every mark named in
        :data:`MARK_DESCRIPTIONS` that the inputs define. Undefined marks are
        ``nan``.

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
    if "stop_idx" in event_table:
        scalar_columns.append("stop_idx")
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
    if stop_marks is not None:
        if "stop_idx" not in marks:
            raise KeyError("event_table needs stop_idx to join stop-level marks")
        stop_idx = pd.to_numeric(marks["stop_idx"], errors="coerce")
        unknown = ~stop_idx.isin(stop_marks.index)
        if unknown.any():
            raise ValueError(
                f"{int(unknown.sum())} event(s) sit in a stop_idx the stop marks "
                f"lack, first {marks.index[unknown.to_numpy()][0]!r}"
            )
        for column in (TRIPS_TO_PATCH_SWITCH, NEXT_TRIP_SWITCHES_PATCH):
            marks[column] = stop_marks[column].reindex(stop_idx).to_numpy(dtype=float)
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
        If a class is missing.
    ValueError
        If ``mark_table`` lacks a required column, if an ``event_id`` is
        duplicated or unmarked, if the marks' ``is_swr``
        or timing disagree with a class frame, if an already-present mark
        column disagrees with the family table, or if the join changes a
        frame's row order.
    """
    require_columns(
        mark_table,
        ["event_id", "event_start_s", "event_end_s", "is_swr"],
        "the mark table",
    )
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
    # The stop each event sits in, for the stop-level mark permutation.
    if "stop_idx" in selected:
        mark_columns.append("stop_idx")
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
    event_arm_table_path: str | Path | None = None,
    event_table_path: str | Path | None = None,
    stop_table_path: str | Path | None = None,
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
        ``table_path("event_arm_table")``, resolved at the call.
    event_table_path : str or Path, optional
        Canonical event table to read. Defaults to
        ``table_path("event_table")``, resolved at the call.
    stop_table_path : str or Path, optional
        Canonical stop table the forward patch-switch marks are derived from
        (:func:`stop_patch_switch_marks`). Defaults to
        ``table_path("stop_table")``, resolved at the call.

    Returns
    -------
    marked : dict[str, pd.DataFrame]
        ``events`` with every recommended mark attached.
    provenance : dict[str, str]
        SHA-256 digests of the three tables the marks were derived from.

    Raises
    ------
    ValueError
        If either table lacks the canonical session columns, if the session has
        no rows in either table, or if mark derivation or attachment refuses the
        session's tables.
    """
    arm_path = (
        table_path("event_arm_table")
        if event_arm_table_path is None
        else Path(event_arm_table_path)
    )
    event_path = (
        table_path("event_table")
        if event_table_path is None
        else Path(event_table_path)
    )
    arm_table = read_table(arm_path)
    event_table = read_table(event_path)
    for name, table in (("event_arm_table", arm_table), ("event_table", event_table)):
        require_columns(table, SESSION_COLUMNS, name)
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
    stop_marks, stop_sha256 = load_session_stop_patch_switch_marks(
        nwb_file_name, epoch_name, stop_table_path=stop_table_path
    )
    marks = build_recommended_event_mark_table(session_arm, session_event, stop_marks)
    return (
        attach_recommended_event_marks(events, marks),
        {
            "event_arm_table_sha256": sha256_file(resolved_table_path(arm_path)),
            "event_table_sha256": sha256_file(resolved_table_path(event_path)),
            "stop_table_sha256": stop_sha256,
        },
    )


def _standardize_for_spec(
    frame: pd.DataFrame, spec: MarkedFamilySpec, min_marked_events: int
) -> tuple[pd.DataFrame, dict[str, MarkScale]]:
    """Standardize one class's marks as the family's designs need them."""
    return standardize_event_marks(
        frame,
        spec.all_marks,
        allow_missing=spec.all_marks,
        allow_constant=tuple(
            mark
            for mark in spec.nuisance_marks
            if mark not in spec.required_varying_nuisance_marks
        ),
        min_marked_events=min_marked_events,
    )


def _require_binary_levels(frame: pd.DataFrame, marks: tuple[str, ...]) -> None:
    """Refuse a two-level mark with fewer than :data:`MIN_LEVEL_COUNT` at a level.

    A mark with exactly two distinct finite values in the class is binary; a
    slope through a level a handful of events carry is not readable, so each
    level needs :data:`MIN_LEVEL_COUNT` events. A one-level mark is left to
    the constant-mark rule of
    :func:`~continuum_swr_replay.analysis.pfc.marked_glm.standardize_event_marks`.

    Raises
    ------
    ValueError
        Naming the mark and its level counts.
    """
    for mark in marks:
        if mark not in frame:
            continue
        values = pd.to_numeric(frame[mark], errors="coerce").to_numpy(dtype=float)
        levels, counts = np.unique(values[np.isfinite(values)], return_counts=True)
        if len(levels) == 2 and counts.min() < MIN_LEVEL_COUNT:
            described = ", ".join(
                f"{level:g}: {count}"
                for level, count in zip(levels, counts, strict=True)
            )
            raise ValueError(
                f"binary mark {mark!r} has levels {{{described}}}; at least "
                f"{MIN_LEVEL_COUNT} events at each level are required"
            )


def class_eligibility(
    events: dict[str, pd.DataFrame],
    spec: MarkedFamilySpec,
    *,
    min_marked_events: int = MIN_MARKED_EVENTS_PER_CLASS,
) -> dict[str, str | None]:
    """Which of a spec's classes can carry its marks, and why not.

    A class can when every mark standardizes (enough finite events, varying
    where the spec needs it) and every binary mark has
    :data:`MIN_LEVEL_COUNT` events at each level -- the checks
    :func:`build_marked_family_ladder` makes. A runner drops the classes that
    cannot (:meth:`MarkedFamilySpec.for_classes`) instead of refusing the
    whole model.

    Parameters
    ----------
    events : dict[str, pd.DataFrame]
        Per event class, the events carrying every raw mark column; a class
        absent from ``events`` is ineligible.
    spec : MarkedFamilySpec
    min_marked_events : int

    Returns
    -------
    dict
        ``class -> None`` when eligible, else the reason.
    """
    reasons: dict[str, str | None] = {}
    for event_class in spec.event_classes:
        if event_class not in events:
            reasons[event_class] = "no events of this class in the session's event set"
            continue
        try:
            _require_binary_levels(events[event_class], spec.all_marks)
            _standardize_for_spec(events[event_class], spec, min_marked_events)
        except (ValueError, KeyError) as error:
            reasons[event_class] = str(error)
        else:
            reasons[event_class] = None
    return reasons


def build_marked_family_ladder(
    events: dict[str, pd.DataFrame],
    fit_mask: pd.Series,
    spline_basis: NDArray,
    lags: NDArray,
    spec: MarkedFamilySpec,
    *,
    classes: tuple[str, ...] | None = None,
    min_marked_events: int = MIN_MARKED_EVENTS_PER_CLASS,
    baseline: NDArray | None = None,
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
    classes : tuple[str, ...], optional
        Event classes, in design-block order; each must be in ``events``.
        Defaults to ``spec.event_classes``.
    min_marked_events : int
        Fewest finite events a class may contribute to a marked kernel.
    baseline : NDArray, shape (n_fit_bins, n_baseline), optional
        Per-stop baseline columns placed after the intercept of the full and
        every reduced design; free in marked_glm; in the family, free unless
        ``baseline_penalty_sigma`` is given
        (:func:`~continuum_swr_replay.analysis.pfc.marked_glm.build_baseline_columns`).

    Returns
    -------
    MarkedFamilyLadder

    Raises
    ------
    KeyError
        If an event class is missing from ``events``.
    ModelIneligibleError
        If a class cannot support the spec: a mark is non-finite, has fewer
        than ``min_marked_events`` finite events, is constant when the spec
        does not permit it, or is binary with fewer than
        :data:`MIN_LEVEL_COUNT` events at a level. The message names the
        model, the class and the mark.
    """
    if classes is None:
        classes = spec.event_classes
    standardized: dict[str, pd.DataFrame] = {}
    scales: dict[str, dict[str, MarkScale]] = {}
    for event_class in classes:
        if event_class not in events:
            raise KeyError(f"event class {event_class!r} is missing")
        try:
            _require_binary_levels(events[event_class], spec.all_marks)
            standardized[event_class], scales[event_class] = _standardize_for_spec(
                events[event_class], spec, min_marked_events
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
    target_classes = (POOLED_CLASS,) if spec.pooled else tuple(classes)
    design_events = dict(standardized)
    if spec.pooled:
        # One block for every class: the classes' events, each mark already
        # standardized within its own class, convolve into the same columns.
        # It equals the sum of the per-class blocks (the convolution is
        # linear), which is a shared slope across the classes.
        columns = [
            "start_time",
            *(
                f"{mark}_{suffix}"
                for mark in spec.scientific_marks
                for suffix in ("z", "available")
            ),
            *(
                ["stop_idx"]
                if all("stop_idx" in standardized[c] for c in classes)
                else []
            ),
        ]
        design_events[POOLED_CLASS] = pd.concat(
            [
                standardized[event_class][columns].assign(event_class=event_class)
                for event_class in classes
            ],
            ignore_index=True,
        )
    scientific_blocks = tuple(
        EventKernelBlock(_block_name(target, mark), target, f"{mark}_z")
        for target in target_classes
        for mark in spec.scientific_marks
    )
    full = build_kernel_design(
        design_events,
        fit_mask,
        spline_basis,
        lags,
        (*base_blocks, *nuisance_blocks, *scientific_blocks),
        baseline=baseline,
    )
    reduced = {
        (target, mark): _without_block(full, _block_name(target, mark))
        for target in target_classes
        for mark in spec.scientific_marks
    }
    return MarkedFamilyLadder(
        spec=spec,
        reduced=reduced,
        full=full,
        events=design_events,
        scales=scales,
        classes=tuple(classes),
        target_classes=target_classes,
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


#: Default prior SD (log-rate units per coefficient) for the marked blocks;
#: the base kernels keep the canonical σ = 3. The three-session smoke of
#: 2026-09-18 showed that one loose prior over a 155- or 221-column design
#: inflates the base kernels by about a quarter whether or not the marks carry
#: content and lets held-out predictions blow up on many units. Total held-out
#: deviance improves monotonically as this prior tightens and is never better
#: than the base-only model, so there is no interior optimum: 0.5 is the
#: operating point that restores the canonical base kernels within 2 %, ends
#: the blow-ups, and still admits a per-SD modulation of roughly ×0.7 to ×1.4
#: (``docs/design/pfc_marked_model_family.md``, "What the first smoke showed").
DEFAULT_MARK_PENALTY_SIGMA: float | None = 0.5


def family_ridge_penalty(
    design: KernelDesign,
    penalty_sigma: float,
    mark_penalty_sigma: float | None = None,
    baseline_penalty_sigma: float | None = None,
) -> NDArray:
    """Per-column ridge penalty with a separate prior for the marked blocks.

    Parameters
    ----------
    design : KernelDesign
        The design whose blocks set which columns are marked.
    penalty_sigma : float
        Prior SD on the ordinary (base) kernel coefficients, log-rate units.
    mark_penalty_sigma : float, optional
        Prior SD on every marked block's coefficients. ``None`` uses
        ``penalty_sigma`` for them too.
    baseline_penalty_sigma : float, optional
        Prior SD on the per-stop baseline columns, log-rate units. ``None``
        leaves them free. A unit with no spikes in the stops at one well has
        no finite estimate of that well's free baseline coefficient
        (separation), so its fit cannot converge; a weak prior keeps it finite
        and barely moves the estimate of a unit with spikes there.

    Returns
    -------
    NDArray, shape (n_columns,)
        ``0`` for the intercept, ``0`` (or ``1 / baseline_penalty_sigma**2``)
        for the ``design.n_baseline`` baseline columns, ``1 / penalty_sigma**2``
        for base columns and ``1 / mark_penalty_sigma**2`` for marked columns.
    """
    penalty = make_ridge_penalty(
        design.matrix.shape[1], penalty_sigma, n_unpenalized=1 + design.n_baseline
    )
    if baseline_penalty_sigma is not None:
        if baseline_penalty_sigma <= 0 or not np.isfinite(baseline_penalty_sigma):
            raise ValueError("baseline_penalty_sigma must be finite and positive")
        penalty[1 : 1 + design.n_baseline] = 1.0 / baseline_penalty_sigma**2
    if mark_penalty_sigma is None:
        return penalty
    if mark_penalty_sigma <= 0 or not np.isfinite(mark_penalty_sigma):
        raise ValueError("mark_penalty_sigma must be finite and positive")
    for block in design.blocks:
        if block.mark_column is not None:
            penalty[design.block_slices[block.name]] = 1.0 / mark_penalty_sigma**2
    return penalty


def _score(
    designs: NDArray,
    response: PreparedHeldoutResponse,
    penalty: NDArray,
    *,
    requested_batch_size: int,
) -> BatchedHeldoutResults:
    """Score equal-width designs under one penalty vector and the batch bound."""
    width = np.asarray(designs).shape[-1]
    if np.asarray(penalty).shape != (width,):
        raise ValueError("penalty must have one value per design column")
    batch_size = bounded_design_batch_size(requested_batch_size, width)
    return score_poisson_designs_heldout(
        designs,
        response,
        np.asarray(penalty, dtype=float),
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


#: How a slope reference shuffles a target. ``residualized_within_class``
#: regresses the target on the model's other marks of its class and permutes
#: only the residual, so a draw keeps the target's correlation with them;
#: ``raw_within_class`` permutes the target itself (the residual on an
#: intercept alone), which decorrelates it from every other mark and makes the
#: reference too narrow when the marks are correlated.
SLOPE_PERMUTATION_METHODS = ("residualized_within_class", "raw_within_class")
DEFAULT_SLOPE_PERMUTATION_METHOD = "residualized_within_class"

#: Shuffled refits fit together per device call. The held-out scorer's
#: profiled budget is for ``n_folds = 5`` fold fits per design; a full-data
#: refit is one fit, so five times as many columns fit the same workload.
FULL_FIT_BATCH_FEATURE_BUDGET = 5 * PROFILED_BATCH_FEATURE_BUDGET
DEFAULT_SLOPE_NULL_BATCH_SIZE = 10


def _permutation_units(stop_idx: NDArray, values: NDArray) -> tuple[NDArray, NDArray]:
    """Group events into stop-level permutation units.

    A unit is a stop, split by value where the stop's events disagree.

    Returns
    -------
    unit_of_event : NDArray of int, shape (n_events,)
    first_event_of_unit : NDArray of int, shape (n_units,)

    Raises
    ------
    ValueError
        If an event has no stop.
    """
    stops = pd.to_numeric(pd.Series(stop_idx), errors="coerce").to_numpy(dtype=float)
    if np.isnan(stops).any():
        raise ValueError(
            f"{int(np.isnan(stops).sum())} event(s) with a defined stop-level mark "
            "have no stop_idx; a stop-level permutation needs every event's stop"
        )
    units = pd.DataFrame({"stop": stops, "value": np.asarray(values, dtype=float)})
    unit_of_event = units.groupby(["stop", "value"], sort=False).ngroup().to_numpy()
    first = pd.Series(np.arange(len(units))).groupby(unit_of_event).first().to_numpy()
    return unit_of_event, first


def stop_level_permutation_weights(
    rng: np.random.Generator,
    values: NDArray,
    available: NDArray,
    stop_idx: NDArray,
    n_draws: int,
) -> NDArray:
    """Shuffle a stop-level mark among stops, moving each stop's events together.

    The permuted unit is the stop: the defined values of the stops' events are
    permuted among the stops, and every event of a stop takes its stop's new
    value, so the reference keeps the within-stop dependence and the multiset
    of stop values. A stop whose events disagree on the value (the latest
    known outcome can change inside a stop) is split into one unit per value.
    Events where the mark is undefined keep their value and never move.

    Parameters
    ----------
    rng : np.random.Generator
        Source of the permutations; one draw at a time.
    values : NDArray, shape (n_events,)
        Each event's (standardized) mark value.
    available : NDArray of bool, shape (n_events,)
        Whether each event's mark is defined.
    stop_idx : NDArray, shape (n_events,)
        The stop each event sits in; must be finite where ``available``.
    n_draws : int
        Draws to return.

    Returns
    -------
    NDArray, shape (n_draws, n_events)
        Each draw's per-event values.

    Raises
    ------
    ValueError
        If an available event has no stop.
    """
    values = np.asarray(values, dtype=float)
    available = np.asarray(available, dtype=bool)
    out = np.repeat(values[None, :], n_draws, axis=0)
    marked = np.flatnonzero(available)
    if marked.size == 0:
        return out
    unit_of_event, first = _permutation_units(
        np.asarray(stop_idx)[marked], values[marked]
    )
    unit_values = values[marked][first]
    for draw in range(n_draws):
        out[draw, marked] = unit_values[rng.permutation(len(unit_values))][
            unit_of_event
        ]
    return out


def residualized_permutation_weights(
    rng: np.random.Generator,
    frame: pd.DataFrame,
    mark: str,
    other_marks: tuple[str, ...],
    n_draws: int,
    *,
    stop_level: bool,
) -> tuple[NDArray, float]:
    """Shuffled target weights that keep the target's relation to other marks.

    On the events where the target is defined, its standardized value is
    regressed (least squares, with an intercept) on the other marks'
    standardized values as the design carries them (zero where a mark is
    undefined). Each draw is the fitted value plus a permutation of the
    residuals, re-standardized over those events as the observed mark is.
    Event-level marks permute residuals among events; a stop-level mark is
    regressed and permuted at the stop level (a stop's other marks averaged
    over its events, :func:`_permutation_units`), so every event of a stop
    keeps one value. With no other marks this is the plain permutation of the
    target.

    Parameters
    ----------
    rng : np.random.Generator
    frame : pd.DataFrame, shape (n_events, ...)
        One class's standardized events: ``<m>_z`` and ``<m>_available`` for
        the target and every other mark, and ``stop_idx`` for a stop-level
        target.
    mark : str
        The target.
    other_marks : tuple of str
        The model's other marks in this class.
    n_draws : int
    stop_level : bool
        Permute whole stops.

    Returns
    -------
    weights : NDArray, shape (n_draws, n_events)
        Zero where the target is undefined, as in the design.
    r_squared : float
        Share of the target's variance (over its defined events, or stops) the
        other marks explain; ``nan`` with fewer than two units.
    """
    values = frame[f"{mark}_z"].to_numpy(dtype=float)
    available = frame[f"{mark}_available"].to_numpy(dtype=bool)
    weights = np.repeat(values[None, :], n_draws, axis=0)
    marked = np.flatnonzero(available)
    if marked.size < 2:
        return weights, float("nan")
    predictors = np.column_stack(
        [
            np.ones(marked.size),
            *(
                frame[f"{other}_z"].to_numpy(dtype=float)[marked]
                for other in other_marks
            ),
        ]
    )
    if stop_level:
        if "stop_idx" not in frame:
            raise KeyError(f"stop-level mark {mark!r} needs each event's stop_idx")
        unit_of_event, first = _permutation_units(
            frame["stop_idx"].to_numpy()[marked], values[marked]
        )
        n_units = len(first)
        response = values[marked][first]
        # A stop's other marks are averaged over its events.
        counts = np.bincount(unit_of_event, minlength=n_units).astype(float)
        unit_predictors = np.stack(
            [
                np.bincount(unit_of_event, weights=column, minlength=n_units) / counts
                for column in predictors.T
            ],
            axis=1,
        )
    else:
        unit_of_event = np.arange(marked.size)
        response = values[marked]
        unit_predictors = predictors
    coefficients, *_ = np.linalg.lstsq(unit_predictors, response, rcond=None)
    fitted = unit_predictors @ coefficients
    residual = response - fitted
    total = np.sum((response - response.mean()) ** 2)
    r_squared = float(1.0 - np.sum(residual**2) / total) if total > 0 else float("nan")
    for draw in range(n_draws):
        unit_values = fitted + residual[rng.permutation(len(residual))]
        event_values = unit_values[unit_of_event]
        spread = event_values.std(ddof=0)
        weights[draw, marked] = (
            (event_values - event_values.mean()) / spread if spread > 0 else 0.0
        )
    return weights, r_squared


def _slope_null_weights(
    rng: np.random.Generator,
    ladder: MarkedFamilyLadder,
    target_class: str,
    mark: str,
    n_draws: int,
    method: str,
) -> tuple[NDArray, dict[str, float]]:
    """One target block's shuffled weights, shuffled within each class.

    A pooled block's frame stacks the classes' events in class order, so its
    weights are each class's own draws, stacked the same way; no value crosses
    a class.

    Returns
    -------
    weights : NDArray, shape (n_draws, n_block_events)
    r_squared : dict
        Per class, the target's R² on the other marks (``nan`` for the raw
        method, which uses none).
    """
    if method not in SLOPE_PERMUTATION_METHODS:
        raise ValueError(f"method must be one of {SLOPE_PERMUTATION_METHODS}")
    classes = ladder.classes if target_class == POOLED_CLASS else (target_class,)
    others = (
        tuple(other for other in ladder.spec.all_marks if other != mark)
        if method == "residualized_within_class"
        else ()
    )
    parts = []
    r_squared: dict[str, float] = {}
    for event_class in classes:
        draws, r2 = residualized_permutation_weights(
            rng,
            ladder.events[event_class],
            mark,
            others,
            n_draws,
            stop_level=mark in STOP_LEVEL_MARKS,
        )
        parts.append(draws)
        r_squared[event_class] = r2 if others else float("nan")
    weights = np.concatenate(parts, axis=1)
    if weights.shape[1] != len(ladder.events[target_class]):
        raise ValueError(f"weights do not match the {target_class!r} block's events")
    return weights, r_squared


def slope_null_batch_size(requested: int, n_features: int) -> int:
    """Refits per device call, bounded by :data:`FULL_FIT_BATCH_FEATURE_BUDGET`."""
    return bounded_design_batch_size(
        requested, n_features, feature_budget=FULL_FIT_BATCH_FEATURE_BUDGET
    )


def fit_marked_family_slope_null(
    ladder: MarkedFamilyLadder,
    counts: NDArray,
    observed: MarkedFamilyResults,
    *,
    penalty_sigma: float,
    mark_penalty_sigma: float | None = DEFAULT_MARK_PENALTY_SIGMA,
    baseline_penalty_sigma: float | None = None,
    n_draws: int,
    seed: int,
    targets: tuple[str, ...] | None = None,
    report_lags: NDArray | None = None,
    method: str = DEFAULT_SLOPE_PERMUTATION_METHOD,
    batch_size: int = DEFAULT_SLOPE_NULL_BATCH_SIZE,
) -> tuple[dict[str, dict[str, NDArray]], dict[str, Any]]:
    """Shuffled-mark reference for every target block's slope kernel.

    Each draw refits the full-data design (no held-out scoring, the observed
    fit's prior) with only one target block rebuilt from the target shuffled
    within its class (``method``; see :data:`SLOPE_PERMUTATION_METHODS`);
    every other block, the other classes' included, keeps its observed
    values. The draws of a block are fit together in device batches, each
    warm-started from the observed fit's coefficients
    (:func:`~continuum_swr_replay.analysis.pfc.batched_glm.fit_poisson_column_variants`),
    which moves the iterations, not the optimum.

    Parameters
    ----------
    ladder : MarkedFamilyLadder
    counts : NDArray, shape (n_fit_bins, n_units)
    observed : MarkedFamilyResults
        The observed fit of this ladder (its ``full_params`` start the draws).
    penalty_sigma, mark_penalty_sigma, baseline_penalty_sigma : float
        The observed fit's priors (see :func:`family_ridge_penalty`).
    n_draws : int
        Refits per target block.
    seed : int
        Target block ``k`` (in comparison order) draws from child ``k`` of
        ``np.random.SeedSequence(seed)``.
    targets : tuple of str, optional
        Target marks to draw for; ``None`` draws for every one.
    report_lags : NDArray of bool, shape (n_lags,), optional
        Lags the null curves are kept on; ``None`` keeps every lag.
    method : str
        One of :data:`SLOPE_PERMUTATION_METHODS`.
    batch_size : int
        Requested refits per device call, before
        :func:`slope_null_batch_size`.

    Returns
    -------
    null : dict
        ``null[c][mark]``, shape ``(n_draws, n_report_lags, n_units)``,
        float32; NaN for a unit whose refit did not converge.
    diagnostics : dict
        ``slope_permutations``, ``slope_permutation_seed``,
        ``slope_permutation_method``, ``slope_null_target_r_squared`` (per
        block, per class), ``slope_null_converged_draws_by_unit``,
        ``slope_null_batch_size`` and ``slope_null_seconds``.
    """
    started = time.perf_counter()
    report = (
        np.ones(len(ladder.lags), dtype=bool)
        if report_lags is None
        else np.asarray(report_lags, dtype=bool)
    )
    if report.shape != (len(ladder.lags),):
        raise ValueError("report_lags must have one entry per lag")
    if n_draws < 0:
        raise ValueError("n_draws must be non-negative")
    unknown = set(targets or ()) - set(ladder.spec.scientific_marks)
    if unknown:
        raise ValueError(f"slope_permutation_targets {sorted(unknown)} are not targets")
    if method not in SLOPE_PERMUTATION_METHODS:
        raise ValueError(f"method must be one of {SLOPE_PERMUTATION_METHODS}")
    fit_counts = np.asarray(counts, dtype=float)
    if fit_counts.ndim == 1:
        fit_counts = fit_counts[:, None]
    penalty = family_ridge_penalty(
        ladder.full, penalty_sigma, mark_penalty_sigma, baseline_penalty_sigma
    )
    width = ladder.full.matrix.shape[1]
    batch = slope_null_batch_size(batch_size, width)
    comparison_order = tuple(ladder.reduced)
    children = np.random.SeedSequence(seed).spawn(len(comparison_order))
    null: dict[str, dict[str, NDArray]] = {}
    converged_draws: dict[str, list[int]] = {}
    r_squared: dict[str, dict[str, float]] = {}
    for index, (target_class, mark) in enumerate(comparison_order):
        if n_draws == 0 or (targets is not None and mark not in targets):
            continue
        rng = np.random.default_rng(children[index])
        span = ladder.full.block_slices[_block_name(target_class, mark)]
        geometry = mark_block_geometry(
            ladder.events[target_class], ladder.fit_mask, ladder.lags, target_class
        )
        weights, r_squared[f"{target_class}:{mark}"] = _slope_null_weights(
            rng, ladder, target_class, mark, n_draws, method
        )
        fits = fit_poisson_column_variants(
            ladder.full.matrix,
            span,
            lambda draws, w=weights, g=geometry: mark_blocks_from_weights(
                w[draws], g, ladder.spline_basis
            ),
            n_draws,
            fit_counts,
            penalty,
            initial_params=observed.full_params,
            design_batch_size=batch,
        )
        parameters = fits.params[:, :, span]  # (n_draws, n_units, n_basis)
        curves = np.einsum("lb,dub->dlu", ladder.spline_basis[report], parameters)
        null.setdefault(target_class, {})[mark] = curves.astype(np.float32)
        converged_draws[f"{target_class}:{mark}"] = (
            np.isfinite(curves).all(axis=1).sum(axis=0).tolist()
        )
    return null, {
        "slope_permutations": int(n_draws),
        "slope_permutation_seed": int(seed),
        "slope_permutation_method": method,
        "slope_null_target_r_squared": r_squared,
        "slope_null_converged_draws_by_unit": converged_draws,
        "slope_null_batch_size": int(batch),
        "slope_null_seconds": time.perf_counter() - started,
    }


def fitted_unit_indices(
    counts: NDArray, baseline_penalty_sigma: float | None
) -> NDArray:
    """The units a session's family is fit on, as indices of the recorded units.

    With a prior on the per-stop baseline columns every unit with a spike has a
    finite fit; a unit with no spike in the fitted bins carries no information
    about any coefficient, so it is left out. Without that prior every unit is
    kept.

    Parameters
    ----------
    counts : NDArray, shape (n_fit_bins, n_units)
        Spike counts on the fitted bins.
    baseline_penalty_sigma : float or None
        Ridge prior SD on the per-stop baseline columns (``None``: free).

    Returns
    -------
    NDArray of int, shape (n_kept,)
        Kept unit indices, ascending.
    """
    if baseline_penalty_sigma is None:
        return np.arange(counts.shape[1])
    return np.flatnonzero(counts.sum(axis=0) > 0)


def fit_marked_family_ladder(
    ladder: MarkedFamilyLadder,
    counts: NDArray,
    folds: NDArray,
    *,
    penalty_sigma: float,
    mark_penalty_sigma: float | None = DEFAULT_MARK_PENALTY_SIGMA,
    baseline_penalty_sigma: float | None = None,
    design_batch_size: int = DEFAULT_NULL_DESIGN_BATCH_SIZE,
    prepared_response: PreparedHeldoutResponse | None = None,
    slope_permutations: int = 0,
    slope_permutation_seed: int = 0,
    slope_permutation_targets: tuple[str, ...] | None = None,
    report_lags: NDArray | None = None,
    slope_permutation_method: str = DEFAULT_SLOPE_PERMUTATION_METHOD,
) -> MarkedFamilyResults:
    """Fit one model family without p-values or response filtering.

    The full design is fit and scored once and shared by every comparison, so
    each target's held-out delta is the deviance the model loses when only
    that class's block for that mark is removed from the common design.

    With ``slope_permutations`` R > 0 every target block also gets a
    shuffled-mark reference for its slope kernel
    (:func:`fit_marked_family_slope_null`, run after the observed fit; a
    caller that gates on the observed fit calls that function itself).

    Parameters
    ----------
    ladder : MarkedFamilyLadder
        The joint full design and every leave-one-target-block-out design.
    counts : NDArray, shape (n_fit_bins, n_units) or (n_fit_bins,)
        Spike counts on the ladder's fitted bins.
    folds : NDArray of int, shape (n_fit_bins,)
        Held-out fold label of each fitted bin.
    penalty_sigma : float
        Ridge prior SD on the base kernels' coefficients.
    mark_penalty_sigma : float, optional
        Ridge prior SD on every marked block's coefficients; ``None`` shares
        ``penalty_sigma``. See :data:`DEFAULT_MARK_PENALTY_SIGMA`.
    baseline_penalty_sigma : float, optional
        Ridge prior SD on the per-stop baseline columns; ``None`` leaves them
        free (see :func:`family_ridge_penalty`).
    design_batch_size : int
        Designs scored per device batch, before the width bound of
        :func:`bounded_design_batch_size`.
    prepared_response : PreparedHeldoutResponse, optional
        A response already transferred to the device, to be reused across the
        models of one session. It must have been prepared from exactly these
        counts and this fold assignment.
    slope_permutations : int
        Shuffled-mark refits per target block; 0 skips the reference.
    slope_permutation_seed : int
        Seed; target block ``k`` (in comparison order) draws from child ``k``
        of ``np.random.SeedSequence(slope_permutation_seed)``.
    slope_permutation_targets : tuple of str, optional
        Target marks to draw a reference for; ``None`` draws for every one.
    report_lags : NDArray of bool, shape (n_lags,), optional
        Lags the null curves are kept on; ``None`` keeps every lag.
    slope_permutation_method : str
        One of :data:`SLOPE_PERMUTATION_METHODS`.

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
    if mark_penalty_sigma is not None and (
        mark_penalty_sigma <= 0 or not np.isfinite(mark_penalty_sigma)
    ):
        raise ValueError("mark_penalty_sigma must be finite and positive")
    if slope_permutations < 0:
        raise ValueError("slope_permutations must be non-negative")
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
    # Every reduced design of one spec lays out intercept, base blocks, then
    # marked blocks in the same order, so one penalty vector serves the batch.
    reduced_penalty = family_ridge_penalty(
        ladder.reduced[comparison_order[0]],
        penalty_sigma,
        mark_penalty_sigma,
        baseline_penalty_sigma,
    )
    full_penalty = family_ridge_penalty(
        ladder.full, penalty_sigma, mark_penalty_sigma, baseline_penalty_sigma
    )
    reduced_score = _score(
        reduced_designs,
        response,
        reduced_penalty,
        requested_batch_size=design_batch_size,
    )
    full_score = _score(
        ladder.full.matrix[None, :, :],
        response,
        full_penalty,
        requested_batch_size=design_batch_size,
    )
    heldout_seconds = time.perf_counter() - started

    fit_started = time.perf_counter()
    fitted = fit_poisson_glms(
        ladder.full.matrix,
        fit_counts,
        penalty=full_penalty,
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
        parameters = unusable_as_nan(fitted.params[:, base], usable)
        typical_curves[event_class] = ladder.spline_basis @ parameters.T
    for target_class in ladder.target_classes:
        mark_curves[target_class] = {}
        for mark in ladder.spec.scientific_marks:
            span = ladder.full.block_slices[_block_name(target_class, mark)]
            parameters = unusable_as_nan(fitted.params[:, span], usable)
            mark_curves[target_class][mark] = ladder.spline_basis @ parameters.T

    diagnostics: dict[str, Any] = {
        "model_name": ladder.spec.name,
        "penalty_sigma_log_rate": float(penalty_sigma),
        "mark_penalty_sigma_log_rate": (
            None if mark_penalty_sigma is None else float(mark_penalty_sigma)
        ),
        "baseline_penalty_sigma_log_rate": (
            None if baseline_penalty_sigma is None else float(baseline_penalty_sigma)
        ),
        "tier": ladder.spec.tier,
        "scientific_marks": list(ladder.spec.scientific_marks),
        "nuisance_marks": list(ladder.spec.nuisance_marks),
        "event_classes": list(ladder.classes),
        "target_classes": list(ladder.target_classes),
        "n_baseline_columns": int(ladder.full.n_baseline),
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
    results = MarkedFamilyResults(
        statistics=pd.DataFrame(statistics),
        typical_curves=typical_curves,
        mark_curves=mark_curves,
        diagnostics=diagnostics,
        full_params=np.asarray(fitted.params, dtype=float),
    )
    if slope_permutations == 0:
        return results
    slope_null, null_diagnostics = fit_marked_family_slope_null(
        ladder,
        fit_counts,
        results,
        penalty_sigma=penalty_sigma,
        mark_penalty_sigma=mark_penalty_sigma,
        baseline_penalty_sigma=baseline_penalty_sigma,
        n_draws=slope_permutations,
        seed=slope_permutation_seed,
        targets=slope_permutation_targets,
        report_lags=report_lags,
        method=slope_permutation_method,
    )
    return replace(
        results,
        slope_null=slope_null,
        diagnostics={**diagnostics, **null_diagnostics},
    )
