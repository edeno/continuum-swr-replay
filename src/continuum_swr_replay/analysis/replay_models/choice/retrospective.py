"""Retrospective alignment of replay content with the next choice.

Where the prospective choice models ask whether replay content predicts
the next choice (adding `arm_fraction` to a behavior-only baseline), these
metrics ask the weaker, post-hoc question: does the replay content during
a stop align with whichever arm the animal actually visited next?

Five event-level metrics are computed across three comparison families:

- ``chosen_arm_fraction``: per-event ``arm_fraction`` at ``arm == next_arm``.
- ``ends_at_next``: does the last in-event MAP arm equal ``next_arm``?
- ``contains_path_to_next``: does the event's MAP-arm trace cover the
  geometric path from ``occupancy_well_arm`` to ``next_arm``?
- ``chosen_patch_fraction``: replay mass assigned to ``next_patch``.
- ``ends_at_next_patch``: does the final MAP arm belong to ``next_patch``?

The null is :func:`next_arm_circular_shift` (in :mod:`shuffles`), with one
future label per contiguous current-arm run within ``stop_id``. Arm metrics
condition on current arm, patch metrics on current patch, and path coverage on
current arm, exact route length, and a coarse event-duration bin. Each family
has its own observable cohort and denominator. Non-identifiable strata are
excluded and reported rather than treated as evidence of no effect.

The per-event metrics themselves, and the well-to-well path they use, are
built with the event table
(:mod:`continuum_swr_replay.analysis.event_tables.retrospective_metrics`); this
module adds the shuffle null and the per-session alignment.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    select_event_cohort,
)
from continuum_swr_replay.data_loaders.constants import (
    TRACK_SEGMENT_TO_PATCH,
    WELL_PATCH_MAPPING,
)
from continuum_swr_replay.tables.columns import require_columns
from continuum_swr_replay.tables.session_keys import (
    SESSION_COLUMNS,
    present_session_columns,
)

from ...event_tables.retrospective_metrics import (
    _VALID_WELL_ARMS,
    arm_path_well_to_well,
    chosen_arm_fraction,
    chosen_patch_fraction,
    contains_path_to_next,
    ends_at_next,
    ends_at_next_patch,
)
from .shuffles import next_arm_circular_shift

__all__ = [
    "at_well_event_cohort",
    "family_unit_counts",
    "freeze_retrospective_cohort",
    "retrospective_alignment_by_session",
    "retrospective_alignment_with_shuffle",
    "retrospective_status",
    "select_event_subset",
]

logger = logging.getLogger(__name__)


#: Movable visit contexts a metric family needs before its null can move a
#: label; below it the family's statistics are NaN.
MIN_MOVABLE_VISIT_CONTEXTS = 2


def _visit_context_ids(event_table: pd.DataFrame) -> pd.Series:
    """One identifier per behavioral visit: ``(stop, completed trip, arm)``.

    ``stop_id`` identifies an immobility bout, not a visit. A bout can cross a
    trial boundary, and the animal can leave and return before it ends, so a
    bout may contain several visits with different future choices.

    ``completed_trial_id`` is what separates them. Keying on contiguous
    ``occupancy_well_arm`` runs alone would merge two visits to the *same* well within
    one bout -- the intervening trip is invisible once transit events are
    filtered out -- and the only symptom would be a hard failure on the
    one-label-per-unit check. The completion-order ID advances across that
    trip, so the two visits stay distinct.

    Raises
    ------
    ValueError
        If ``completed_trial_id`` is absent, which marks a table built before
        the event schema carried per-event trial context.
    """
    require_columns(
        event_table,
        ["stop_id", "completed_trial_id", "occupancy_well_arm"],
        "event_table",
    )
    if event_table.empty:
        return pd.Series(index=event_table.index, dtype=object)
    return pd.Series(
        list(
            zip(
                event_table["stop_id"],
                event_table["completed_trial_id"],
                event_table["occupancy_well_arm"],
                strict=True,
            )
        ),
        index=event_table.index,
        dtype=object,
    )


def at_well_event_cohort(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Filter all three event tables to one explicit at-well event cohort.

    Membership is the shared event-grain rule and nothing else:
    :func:`~continuum_swr_replay.analysis.event_definition.cohort_selection.select_event_cohort`
    with ``cohort="at_well"``, the same call the content ladders make. What this
    function adds on top is retrospective-specific: the finite
    ``occupancy_well_arm`` conjunct (an endpoint the path metrics can name) and
    the three-table key alignment that keeps arm and sequence rows from
    acquiring metric-specific denominators.

    Canonical tables use the nullable ``at_well`` label. Legacy inputs without
    that column are rejected: their occupancy field may contain the retired
    destination-well fallback for in-transit events. ``at_well`` and
    ``occupancy_state == "at_well"`` are equal by contract on a canonical table
    (``table_contracts`` refuses one where they disagree), so reading the
    membership from ``occupancy_state`` selects the same events while leaving
    one rule to change.

    All three returned frames carry a fresh ``RangeIndex``: the shared selector
    resets the event frame's index, and the arm/sequence frames come out of a
    merge. Row ORDER is preserved; only the labels change. Nothing downstream
    joins on those labels -- the metrics key on ``event_id`` -- but a caller
    holding an index from before the call cannot reuse it after.
    """
    if event_table.empty:
        return (
            event_table.copy(),
            event_arm_table.iloc[0:0].copy(),
            event_sequence_table.iloc[0:0].copy(),
        )
    require_columns(
        event_table, ["at_well", "occupancy_state", "occupancy_well_arm"], "event_table"
    )
    at_well, _ = select_event_cohort(event_table, cohort="at_well")
    current = pd.to_numeric(at_well["occupancy_well_arm"], errors="coerce")
    events = at_well.loc[np.isfinite(current)].reset_index(drop=True)
    key_cols = [*present_session_columns(event_table), "event_id"]
    keys = events[key_cols].drop_duplicates()

    def _restrict(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.iloc[0:0].copy()
        require_columns(frame, key_cols, "event-family table")
        return frame.merge(keys, on=key_cols, how="inner", validate="many_to_one")

    return events, _restrict(event_arm_table), _restrict(event_sequence_table)


# ─────────────────────────────────────────────────────────────────────
# Shuffle null
# ─────────────────────────────────────────────────────────────────────


_DEFAULT_METRICS = (
    "chosen_arm_fraction",
    "ends_at_next",
    "contains_path_to_next",
    "chosen_patch_fraction",
    "ends_at_next_patch",
)

_ARM_TARGET_METRICS = frozenset({"chosen_arm_fraction", "ends_at_next"})
_PATCH_TARGET_METRICS = frozenset({"chosen_patch_fraction", "ends_at_next_patch"})
_PATH_METRICS = frozenset({"contains_path_to_next"})
_METRIC_FAMILIES = {
    "arm": _ARM_TARGET_METRICS,
    "patch": _PATCH_TARGET_METRICS,
    "path": _PATH_METRICS,
}
_SHUFFLE_VISIT_UNIT = "_retrospective_visit_unit"
_SHUFFLE_CURRENT_ARM = "_retrospective_current_arm"
_SHUFFLE_CURRENT_PATCH = "_retrospective_current_patch"
_SHUFFLE_ROUTE_LENGTH = "_retrospective_route_length"
_SHUFFLE_EVENT_LENGTH_BIN = "_retrospective_event_length_bin"
_SHUFFLE_HELPER_COLUMNS = (
    _SHUFFLE_VISIT_UNIT,
    _SHUFFLE_CURRENT_ARM,
    _SHUFFLE_CURRENT_PATCH,
    _SHUFFLE_ROUTE_LENGTH,
    _SHUFFLE_EVENT_LENGTH_BIN,
)


@dataclass(frozen=True)
class _FrozenRetrospectiveCohort:
    """Common observable cohort plus its calibrated-shuffle definition."""

    event_table: pd.DataFrame
    event_arm_table: pd.DataFrame
    event_sequence_table: pd.DataFrame
    audit: dict[str, int]
    family: str
    strata_cols: tuple[str, ...]
    variation_cols: tuple[str, ...]
    label_cols: tuple[str, ...]


def _validate_requested_metrics(metrics: tuple[str, ...]) -> tuple[str, ...]:
    """Validate and de-duplicate the public metric selection."""
    requested = tuple(dict.fromkeys(metrics))
    if not requested:
        raise ValueError("At least one retrospective metric must be requested")
    unknown = sorted(set(requested).difference(_DEFAULT_METRICS))
    if unknown:
        raise ValueError(f"Unknown retrospective metric(s): {unknown}")
    return requested


def _requested_metrics_by_family(
    metrics: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    """Partition requested metrics without imposing a cross-family cohort."""
    requested = _validate_requested_metrics(metrics)
    return {
        family: tuple(metric for metric in requested if metric in family_metrics)
        for family, family_metrics in _METRIC_FAMILIES.items()
        if any(metric in family_metrics for metric in requested)
    }


def _compute_requested_metrics(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    """Compute only selected metrics on one already-frozen event cohort."""
    metrics = _validate_requested_metrics(metrics)
    key_cols = [*present_session_columns(event_table), "event_id"]
    out = event_table[key_cols].copy()
    calculators = {
        "chosen_arm_fraction": lambda: chosen_arm_fraction(
            event_table, event_arm_table
        ),
        "ends_at_next": lambda: ends_at_next(event_table, event_sequence_table),
        "contains_path_to_next": lambda: contains_path_to_next(
            event_table, event_sequence_table
        ),
        "chosen_patch_fraction": lambda: chosen_patch_fraction(
            event_table, event_arm_table
        ),
        "ends_at_next_patch": lambda: ends_at_next_patch(
            event_table, event_sequence_table
        ),
    }
    for metric in metrics:
        calculated = calculators[metric]()
        out = out.merge(
            calculated,
            on=key_cols,
            how="left",
            validate="one_to_one",
        )
    return out


def _per_session_metric_means(
    metrics_df: pd.DataFrame,
    metrics: tuple[str, ...] | None = None,
) -> dict[str, float]:
    """Reduce per-event metrics to the session-level statistics
    these metrics report."""
    out: dict[str, float] = {}
    selected = (
        tuple(col for col in _DEFAULT_METRICS if col in metrics_df)
        if metrics is None
        else metrics
    )
    for col in selected:
        out[col] = float(np.nanmean(metrics_df[col].astype(float)))
    return out


def _observable_event_keys(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
    metrics: tuple[str, ...],
) -> pd.DataFrame:
    """Keys observable for every requested metric and every label shuffle.

    Endpoint metrics deliberately do *not* require a finite final MAP arm.
    Their scientific question is whether the event ends at the target, so an
    undecoded final bin is a negative endpoint rather than missing data.  Path
    coverage still requires at least one finite decoded arm; otherwise there
    is no decoded sequence on which to evaluate set coverage.
    """
    key_cols = [*present_session_columns(event_table), "event_id"]
    keys = event_table[key_cols].drop_duplicates()
    if any(
        metric in {"chosen_arm_fraction", "chosen_patch_fraction"} for metric in metrics
    ):
        require_columns(
            event_arm_table, ["candidate_arm", "arm_fraction"], "event_arm_table"
        )
        arm = event_arm_table.copy()
        arm["_finite_fraction"] = np.isfinite(
            pd.to_numeric(arm["arm_fraction"], errors="coerce")
        )
        arm_observed = arm.groupby(key_cols, as_index=False).agg(
            _n_arm_rows=("candidate_arm", "size"),
            _arm_set=("candidate_arm", lambda values: frozenset(values)),
            _all_finite=("_finite_fraction", "all"),
        )
        arm_observed = arm_observed[
            arm_observed["_n_arm_rows"].eq(len(TRACK_SEGMENT_TO_PATCH))
            & arm_observed["_arm_set"].eq(frozenset(TRACK_SEGMENT_TO_PATCH))
            & arm_observed["_all_finite"]
        ][key_cols]
        keys = keys.merge(arm_observed, on=key_cols, how="inner")

    endpoint_metrics = {"ends_at_next", "ends_at_next_patch"}
    path_metrics = {"contains_path_to_next"}
    requested_sequence_metrics = set(metrics) & (endpoint_metrics | path_metrics)
    if requested_sequence_metrics:
        require_columns(
            event_sequence_table, ["decoded_arm_map"], "event_sequence_table"
        )

    if set(metrics) & path_metrics:
        sequence = event_sequence_table.assign(
            _valid_decoded=np.isfinite(
                pd.to_numeric(event_sequence_table["decoded_arm_map"], errors="coerce")
            )
        )
        grouped_sequence = sequence.groupby(key_cols, sort=False)
        sequence_observed = grouped_sequence.agg(
            _any_valid_decoded=("_valid_decoded", "any")
        ).reset_index()
        sequence_observed = sequence_observed[sequence_observed["_any_valid_decoded"]]
        sequence_observed = sequence_observed[key_cols]
        keys = keys.merge(sequence_observed, on=key_cols, how="inner")
    return keys


def _freeze_retrospective_cohort_with_audit(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = _DEFAULT_METRICS,
) -> _FrozenRetrospectiveCohort:
    """Freeze one metric family's cohort and calibrated shuffle strata."""
    metrics = _validate_requested_metrics(metrics)
    matching_families = [
        family
        for family, family_metrics in _METRIC_FAMILIES.items()
        if set(metrics).issubset(family_metrics)
    ]
    if len(matching_families) != 1:
        raise ValueError(
            "A frozen retrospective cohort can contain only one metric family; "
            f"got {metrics}"
        )
    family = matching_families[0]
    n_events_input = int(len(event_table))
    required_event_columns = [
        "occupancy_well_arm",
        "next_arm",
        "stop_id",
        "event_midpoint_s",
    ]
    if family == "patch":
        required_event_columns += ["current_patch", "next_patch"]
    if family == "path":
        required_event_columns.append("n_nonlocal_bins")
    require_columns(event_table, required_event_columns, "event_table")
    session_cols = present_session_columns(event_table)
    if session_cols and event_table[session_cols].drop_duplicates().shape[0] > 1:
        raise ValueError(
            "retrospective alignment requires one session at a time; "
            "cross-session label exchange is invalid"
        )

    events, event_arm, event_sequence = at_well_event_cohort(
        event_table, event_arm_table, event_sequence_table
    )
    n_events_after_at_well = int(len(events))
    numeric_current = pd.to_numeric(events["occupancy_well_arm"], errors="coerce")
    invalid_current = ~numeric_current.isin(_VALID_WELL_ARMS)
    if invalid_current.any():
        raise ValueError(
            "Retrospective endpoints must be integer well arms 0..5; found "
            f"{int(invalid_current.sum())} invalid current_arm values"
        )
    events["occupancy_well_arm"] = numeric_current.astype(int)
    if events["stop_id"].isna().any():
        raise ValueError("Retrospective shuffle unit stop_id contains NaN")
    midpoint = pd.to_numeric(events["event_midpoint_s"], errors="coerce")
    if not np.isfinite(midpoint).all():
        raise ValueError("Retrospective event_midpoint_time must be finite")
    events["event_midpoint_s"] = midpoint.astype(float)
    # Derive visits before removing undefined future labels. Otherwise an
    # intervening visit with no next choice could disappear and collapse a
    # later return to the same arm into the earlier visit.
    events[_SHUFFLE_VISIT_UNIT] = _visit_context_ids(events)
    visit_groups = events.groupby(_SHUFFLE_VISIT_UNIT, sort=False, dropna=False)
    available_next_labels = [
        column for column in ("next_arm", "next_patch") if column in events
    ]
    for column in available_next_labels:
        n_unique = visit_groups[column].nunique(dropna=False)
        inconsistent = n_unique[n_unique > 1]
        if not inconsistent.empty:
            raise ValueError(
                "Retrospective shuffle requires one future label per "
                "contiguous visit context; unit "
                f"{inconsistent.index[0]!r} has multiple {column!r} values"
            )

    numeric_next = pd.to_numeric(events["next_arm"], errors="coerce")
    valid_next = np.isfinite(numeric_next)
    events = events.loc[valid_next].copy()
    numeric_next = numeric_next.loc[valid_next]
    invalid_next = ~numeric_next.isin(_VALID_WELL_ARMS)
    if invalid_next.any():
        raise ValueError(
            "Retrospective endpoints must be integer well arms 0..5; found "
            f"{int(invalid_next.sum())} invalid next_arm values"
        )
    events["next_arm"] = numeric_next.astype(int)
    n_events_after_valid_next = int(len(events))

    key_cols = [*present_session_columns(events), "event_id"]

    def _restrict_family(
        frame: pd.DataFrame,
        keys: pd.DataFrame,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame.iloc[0:0].copy()
        require_columns(frame, key_cols, "event-family table")
        return frame.merge(keys, on=key_cols, how="inner", validate="many_to_one")

    event_keys = events[key_cols].drop_duplicates()
    event_arm = _restrict_family(event_arm, event_keys)
    event_sequence = _restrict_family(event_sequence, event_keys)

    if family == "patch":
        numeric_current_patch = pd.to_numeric(events["current_patch"], errors="coerce")
        numeric_patch = pd.to_numeric(events["next_patch"], errors="coerce")
        expected_current_patch = (
            events["occupancy_well_arm"].map(WELL_PATCH_MAPPING).astype(float)
        )
        expected_patch = events["next_arm"].map(WELL_PATCH_MAPPING).astype(float)
        invalid_current_patch = ~np.isfinite(
            numeric_current_patch
        ) | numeric_current_patch.ne(expected_current_patch)
        invalid_next_patch = ~np.isfinite(numeric_patch) | numeric_patch.ne(
            expected_patch
        )
        if invalid_current_patch.any() or invalid_next_patch.any():
            raise ValueError(
                "Patch retrospective metrics require finite patch labels "
                "consistent with their arms; found "
                f"{int(invalid_current_patch.sum())} invalid current_patch and "
                f"{int(invalid_next_patch.sum())} invalid next_patch event(s)"
            )
        events["current_patch"] = numeric_current_patch.astype(int)
        events["next_patch"] = numeric_patch.astype(int)

    observable_keys = _observable_event_keys(
        events,
        event_arm,
        event_sequence,
        metrics,
    )
    events = events.merge(
        observable_keys, on=key_cols, how="inner", validate="one_to_one"
    )
    event_arm = _restrict_family(event_arm, observable_keys)
    event_sequence = _restrict_family(event_sequence, observable_keys)
    n_events_after_observability = int(len(events))

    events[_SHUFFLE_CURRENT_ARM] = events["occupancy_well_arm"].astype(int)

    if family == "arm":
        strata_cols = [_SHUFFLE_CURRENT_ARM]
        variation_cols = ["next_arm"]
        label_cols = ["next_arm"]
    elif family == "patch":
        events[_SHUFFLE_CURRENT_PATCH] = events["current_patch"].astype(int)
        strata_cols = [_SHUFFLE_CURRENT_PATCH]
        variation_cols = ["next_patch"]
        # Shift the paired future-arm label with its patch label even though
        # patch metrics use only next_patch.
        label_cols = ["next_arm", "next_patch"]
    else:
        events[_SHUFFLE_ROUTE_LENGTH] = np.fromiter(
            (
                len(arm_path_well_to_well(current, target))
                for current, target in zip(
                    events["occupancy_well_arm"], events["next_arm"], strict=True
                )
            ),
            dtype=np.int64,
            count=len(events),
        )
        lengths = pd.to_numeric(events["n_nonlocal_bins"], errors="coerce")
        invalid_length = (
            ~np.isfinite(lengths) | lengths.le(0) | lengths.ne(np.floor(lengths))
        )
        if invalid_length.any():
            raise ValueError(
                "Path retrospective metrics require positive integer "
                f"n_nonlocal_bins; found {int(invalid_length.sum())} invalid event(s)"
            )
        # Labels move at visit-context grain, so event duration also needs a
        # visit-grain summary. Octave bins of the within-visit median preserve
        # broad duration opportunity without making almost every visit unique.
        visit_median_length = lengths.groupby(events[_SHUFFLE_VISIT_UNIT]).transform(
            "median"
        )
        events[_SHUFFLE_EVENT_LENGTH_BIN] = np.floor(
            np.log2(visit_median_length)
        ).astype(int)
        strata_cols = [
            _SHUFFLE_CURRENT_ARM,
            _SHUFFLE_ROUTE_LENGTH,
            _SHUFFLE_EVENT_LENGTH_BIN,
        ]
        variation_cols = ["next_arm"]
        label_cols = ["next_arm"]

    grouped_units = events.groupby(_SHUFFLE_VISIT_UNIT, sort=False, dropna=False)
    unit_constant_cols = list(strata_cols)
    for column in dict.fromkeys(unit_constant_cols):
        n_unique = grouped_units[column].nunique(dropna=False)
        inconsistent = n_unique[n_unique > 1]
        if not inconsistent.empty:
            raise ValueError(
                "Retrospective shuffle requires constant strata within each "
                "contiguous visit context; unit "
                f"{inconsistent.index[0]!r} has multiple {column!r} values"
            )

    unit_rows = events.drop_duplicates(_SHUFFLE_VISIT_UNIT, keep="first")[
        [_SHUFFLE_VISIT_UNIT, *strata_cols, *label_cols]
    ].copy()
    stratum_group = unit_rows.groupby(strata_cols, sort=False, dropna=False)
    n_units_in_stratum = stratum_group[_SHUFFLE_VISIT_UNIT].transform("size")
    singleton = n_units_in_stratum.lt(2)
    # Every relevant future label must vary. A stratum that is constant in
    # even one requested target cannot identify that target's null contrast.
    any_constant = pd.Series(False, index=unit_rows.index)
    for column in variation_cols:
        any_constant |= stratum_group[column].transform("nunique").lt(2)
    deterministic = n_units_in_stratum.ge(2) & any_constant
    movable = ~(singleton | deterministic)
    retained_units = unit_rows.loc[movable, _SHUFFLE_VISIT_UNIT]
    retained_unit_set = set(retained_units)

    n_strata_input = int(unit_rows[strata_cols].drop_duplicates().shape[0])
    n_strata_singleton = int(
        unit_rows.loc[singleton, strata_cols].drop_duplicates().shape[0]
    )
    n_strata_deterministic = int(
        unit_rows.loc[deterministic, strata_cols].drop_duplicates().shape[0]
    )
    n_units_input = int(len(unit_rows))
    n_units_singleton = int(singleton.sum())
    n_units_deterministic = int(deterministic.sum())

    events = events.loc[events[_SHUFFLE_VISIT_UNIT].isin(retained_unit_set)].copy()
    retained_keys = events[key_cols].drop_duplicates()
    event_arm = _restrict_family(event_arm, retained_keys)
    event_sequence = _restrict_family(event_sequence, retained_keys)
    audit = {
        "n_events_input": n_events_input,
        "n_events_after_at_well": n_events_after_at_well,
        "n_events_after_valid_next": n_events_after_valid_next,
        "n_events_after_common_observability": n_events_after_observability,
        "n_events_excluded_immovable": (
            n_events_after_observability - int(len(events))
        ),
        "n_shuffle_units_input": n_units_input,
        "n_shuffle_units": int(len(retained_units)),
        "n_shuffle_units_excluded_singleton": n_units_singleton,
        "n_shuffle_units_excluded_deterministic": n_units_deterministic,
        "n_shuffle_strata_input": n_strata_input,
        "n_shuffle_strata": n_strata_input
        - n_strata_singleton
        - n_strata_deterministic,
        "n_shuffle_strata_excluded_singleton": n_strata_singleton,
        "n_shuffle_strata_excluded_deterministic": n_strata_deterministic,
    }
    return _FrozenRetrospectiveCohort(
        event_table=events,
        event_arm_table=event_arm,
        event_sequence_table=event_sequence,
        audit=audit,
        family=family,
        strata_cols=tuple(strata_cols),
        variation_cols=tuple(variation_cols),
        label_cols=tuple(label_cols),
    )


def freeze_retrospective_cohort(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
    *,
    metrics: tuple[str, ...] = _DEFAULT_METRICS,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the union of requested families' identifiable cohorts.

    Each metric family is frozen independently. The returned event keys are
    their union, not a denominator for inference; use the per-metric counts from
    :func:`retrospective_alignment_with_shuffle` for scientific reporting.

    Returns
    -------
    tuple of pd.DataFrame
        The three tables restricted to the union of frozen family cohorts.
    """
    family_metrics = _requested_metrics_by_family(metrics)
    frozen_families = [
        _freeze_retrospective_cohort_with_audit(
            event_table,
            event_arm_table,
            event_sequence_table,
            metrics=selected,
        )
        for selected in family_metrics.values()
    ]
    key_cols = [*present_session_columns(event_table), "event_id"]
    union_events = (
        pd.concat(
            [frozen.event_table for frozen in frozen_families],
            ignore_index=True,
        )
        .drop_duplicates(key_cols, keep="first")
        .drop(columns=_SHUFFLE_HELPER_COLUMNS, errors="ignore")
    )
    union_keys = union_events[key_cols].drop_duplicates()

    def _restrict(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame.iloc[0:0].copy()
        return frame.merge(
            union_keys,
            on=key_cols,
            how="inner",
            validate="many_to_one",
        )

    return union_events, _restrict(event_arm_table), _restrict(event_sequence_table)


def retrospective_alignment_with_shuffle(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
    *,
    n_shuffles: int = 100,
    metrics: tuple[str, ...] = _DEFAULT_METRICS,
    rng: np.random.Generator,
) -> dict[str, float | int | bool | str]:
    """Real vs. shuffled alignment for one session.

    Each metric family gets its own visit-context cohort and calibrated null:
    arm metrics condition on ``occupancy_well_arm``; patch metrics condition on
    ``current_patch``; path coverage conditions on ``occupancy_well_arm``,
    exact route length, and coarse duration. The statistic is real minus
    shuffled.

    Metrics share a cohort only within their family. A family with no movable
    stratum reports zero events and NaN statistics without removing observable
    events from another family.

    Parameters
    ----------
    event_table, event_arm_table, event_sequence_table : pd.DataFrame
        Must come from a SINGLE session. Cross-session shuffling
        would cross-contaminate behavioral statistics.
    n_shuffles : int, default 100
        Number of circular-shift draws.
    metrics : tuple of str
        Subset of :data:`_DEFAULT_METRICS`. Useful in tests for
        skipping the sequence-table-dependent metrics.
    rng : numpy Generator
        The caller's generator for the shifts; required, so a run is
        reproducible.

    Returns
    -------
    dict
        Keys: ``real_mean_<metric>``, ``shuffled_mean_<metric>``,
        ``real_minus_shuffled_<metric>``, ``shuffled_std_<metric>``,
        ``n_events_<metric>`` and family-specific retained/excluded visit and
        stratum audits. A legacy ``n_events`` key is included only when one
        metric family was requested; multi-family output has no misleading
        global denominator. Provenance fields record the visit-unit and cohort
        policy; path output explicitly records its movable-strata scope.

    Raises
    ------
    ValueError
        If a required column is absent.
    RuntimeError
        If a metric's observed count disagrees with the frozen cohort
        size, which would mean the common-cohort guarantee was broken.
    """
    metrics = _validate_requested_metrics(metrics)
    if n_shuffles < 0:
        raise ValueError("n_shuffles must be nonnegative")
    family_metrics = _requested_metrics_by_family(metrics)
    out: dict[str, float | int | bool | str] = {
        "n_shuffles": int(n_shuffles),
        "cohort_policy": "metric_family_specific",
        "shuffle_unit_definition": "contiguous_current_arm_run_within_stop_id",
    }
    if "path" in family_metrics:
        out["path_null_scope"] = "movable_current_arm_route_length_duration_strata"
        out["path_null_effectively_cross_patch"] = True

    for family, selected_metrics in family_metrics.items():
        frozen = _freeze_retrospective_cohort_with_audit(
            event_table,
            event_arm_table,
            event_sequence_table,
            metrics=selected_metrics,
        )
        family_suffix = f"{family}_family"
        n_events = int(len(frozen.event_table))
        for audit_name, value in frozen.audit.items():
            out[f"{audit_name}_{family_suffix}"] = value
        out[f"n_events_{family_suffix}"] = n_events

        if len(family_metrics) == 1:
            out.update(frozen.audit)
            out["n_events"] = n_events

        if frozen.audit["n_shuffle_units"] < MIN_MOVABLE_VISIT_CONTEXTS:
            for metric in selected_metrics:
                out[f"n_events_{metric}"] = 0
                out[f"real_mean_{metric}"] = float("nan")
                out[f"shuffled_mean_{metric}"] = float("nan")
                out[f"shuffled_std_{metric}"] = float("nan")
                out[f"real_minus_shuffled_{metric}"] = float("nan")
            continue

        real = _compute_requested_metrics(
            frozen.event_table,
            frozen.event_arm_table,
            frozen.event_sequence_table,
            selected_metrics,
        )
        real_counts = {
            metric: int(real[metric].notna().sum()) for metric in selected_metrics
        }
        for metric, count in real_counts.items():
            if count != n_events:
                raise RuntimeError(
                    f"Retrospective metric {metric} observed {count} events but "
                    f"its frozen {family} cohort contains {n_events}"
                )
        real_means = _per_session_metric_means(real, selected_metrics)
        shuffled_acc: dict[str, list[float]] = {
            metric: [] for metric in selected_metrics
        }
        for _ in range(n_shuffles):
            shuffled_ev = next_arm_circular_shift(
                frozen.event_table,
                columns=frozen.label_cols,
                unit_col=_SHUFFLE_VISIT_UNIT,
                strata_cols=frozen.strata_cols,
                variation_cols=frozen.variation_cols,
                require_movement=True,
                rng=rng,
            )
            shuffled_metrics = _compute_requested_metrics(
                shuffled_ev,
                frozen.event_arm_table,
                frozen.event_sequence_table,
                selected_metrics,
            )
            shuffled_means = _per_session_metric_means(
                shuffled_metrics, selected_metrics
            )
            for metric in selected_metrics:
                shuffled_count = int(shuffled_metrics[metric].notna().sum())
                if shuffled_count != n_events:
                    raise RuntimeError(
                        f"Shuffled retrospective metric {metric} observed "
                        f"{shuffled_count} events but its frozen {family} "
                        f"cohort contains {n_events}"
                    )
                shuffled_acc[metric].append(shuffled_means[metric])

        for metric in selected_metrics:
            out[f"real_mean_{metric}"] = real_means[metric]
            out[f"n_events_{metric}"] = real_counts[metric]
            if shuffled_acc[metric]:
                shuffled_mean = float(np.mean(shuffled_acc[metric]))
                out[f"shuffled_mean_{metric}"] = shuffled_mean
                out[f"shuffled_std_{metric}"] = float(np.std(shuffled_acc[metric]))
                out[f"real_minus_shuffled_{metric}"] = (
                    real_means[metric] - shuffled_mean
                )
            else:
                out[f"shuffled_mean_{metric}"] = float("nan")
                out[f"shuffled_std_{metric}"] = float("nan")
                out[f"real_minus_shuffled_{metric}"] = float("nan")
    return out


def family_unit_counts(
    result: dict[str, float | int | bool | str],
) -> dict[str, int]:
    """Movable visit-context counts for each calibrated metric family.

    Only families the run actually requested appear in ``result`` (a
    :func:`retrospective_alignment_with_shuffle` output), so read whichever
    suffixed keys are present rather than assuming all three.
    """
    suffix = "_family"
    prefix = "n_shuffle_units_"
    counts: dict[str, int] = {}
    for key, value in result.items():
        if not (key.startswith(prefix) and key.endswith(suffix)):
            continue
        family = key[len(prefix) : -len(suffix)]
        # The retained count is ``n_shuffle_units_<family>_family``. Sibling
        # keys qualify the same suffix -- ``..._input_<family>_family``,
        # ``..._excluded_singleton_<family>_family`` -- and their zeros would
        # read as a non-movable family. A family name is a single token.
        if "_" in family:
            continue
        counts[family] = int(value)
    return counts


def retrospective_status(
    result: dict[str, float | int | bool | str],
) -> str:
    """Explicit persisted status for a session's retrospective analysis.

    A family needs :data:`MIN_MOVABLE_VISIT_CONTEXTS` movable visit contexts
    before its null can move a label, and the families are calibrated
    independently, so a session can estimate some metrics and not others.
    Distinguish that from the case where nothing is estimable: ``ok`` means
    every requested family is movable, ``partial_movable_families`` that at
    least one is not, and ``no_movable_family`` that none are.
    """
    counts = family_unit_counts(result)
    if not counts:
        return "no_movable_family"
    movable = [count >= MIN_MOVABLE_VISIT_CONTEXTS for count in counts.values()]
    if all(movable):
        return "ok"
    return "partial_movable_families" if any(movable) else "no_movable_family"


#: The event subsets the alignment may be read on, by ``is_swr``.
EVENT_SUBSETS = ("all", "swr_only", "non_swr_only")


def select_event_subset(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
    subset: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Keep the SWR or the non-SWR events of the three event tables.

    ``swr_only`` keeps ``is_swr`` events, ``non_swr_only`` the others, ``all``
    every event. The event and event-arm tables are filtered on their own
    ``is_swr``; the sequence table keeps the rows of the kept events (an inner
    join on the session keys and ``event_id``).

    Raises
    ------
    ValueError
        If ``subset`` is not one of :data:`EVENT_SUBSETS`.
    """
    if subset not in EVENT_SUBSETS:
        raise ValueError(f"subset must be one of {EVENT_SUBSETS}, not {subset!r}")
    if subset == "all":
        return event_table, event_arm_table, event_sequence_table
    want_swr = subset == "swr_only"
    keep = event_table["is_swr"].astype(bool) == want_swr
    event_table = event_table[keep].copy()
    event_arm_table = event_arm_table[
        event_arm_table["is_swr"].astype(bool) == want_swr
    ].copy()
    event_sequence_table = event_sequence_table.merge(
        event_table[[*SESSION_COLUMNS, "event_id"]],
        on=[*SESSION_COLUMNS, "event_id"],
        how="inner",
    )
    return event_table, event_arm_table, event_sequence_table


def retrospective_alignment_by_session(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    event_sequence_table: pd.DataFrame,
    *,
    n_shuffles: int,
    seed: int,
    subset: str,
    stop_cohort: str,
    max_sessions: int | None = None,
) -> pd.DataFrame:
    """:func:`retrospective_alignment_with_shuffle` on every session, in order.

    The sessions are taken in sorted key order (the first ``max_sessions``
    only, when given). Each draws its shuffles from its own generator, seeded
    by one draw of ``numpy.random.default_rng(seed)`` in that order, so a
    session's result does not depend on which sessions ran before it beyond
    its position. Every session is kept, with its :func:`retrospective_status`;
    a family without :data:`MIN_MOVABLE_VISIT_CONTEXTS` movable visit contexts
    reports NaN metrics. A session that violates the location/observability
    contract raises, so a partial cohort is never returned.

    Parameters
    ----------
    event_table, event_arm_table, event_sequence_table : pd.DataFrame
        The three event tables of the analyzed cohort, all sessions.
    n_shuffles : int
        Circular-shift draws per session.
    seed : int
        Master seed.
    subset, stop_cohort : str
        Labels written on every row.
    max_sessions : int, optional
        Analyze only the first sessions.

    Returns
    -------
    pd.DataFrame
        One row per session: the alignment result, the session keys,
        ``subset``, ``stop_cohort``, ``retrospective_status`` and
        ``elapsed_s``.
    """
    rng = np.random.default_rng(seed)
    rows = []
    sessions = sorted(event_table.groupby(list(SESSION_COLUMNS)).groups.keys())
    if max_sessions is not None:
        sessions = sessions[:max_sessions]
        logger.info("Smoke run: first %d session(s) only", len(sessions))
    # Each table split by session once; a session a table lacks gets its empty
    # frame (same columns and dtypes).
    tables = (event_table, event_arm_table, event_sequence_table)
    by_session = [
        dict(tuple(table.groupby(list(SESSION_COLUMNS), sort=False)))
        for table in tables
    ]
    for i, sess_key in enumerate(sessions):
        animal, session_date, epoch = sess_key
        ev_s, ea_s, es_s = (
            groups.get(sess_key, table.iloc[:0])
            for groups, table in zip(by_session, tables, strict=True)
        )
        t0 = time.time()
        # Fail the batch before publication when any session violates the
        # location/observability contract.  Continuing here would turn a
        # scientific integrity error into an unreported session exclusion and
        # then persist a plausible-looking partial cohort.
        result = retrospective_alignment_with_shuffle(
            ev_s,
            ea_s,
            es_s,
            n_shuffles=n_shuffles,
            rng=np.random.default_rng(rng.integers(0, 2**32)),
        )
        status = retrospective_status(result)
        if status != "ok":
            logger.info(
                "Retaining %s/%s/%s with status=%s; families without two "
                "movable visit contexts report NaN metrics (%s)",
                animal,
                session_date,
                epoch,
                status,
                family_unit_counts(result),
            )

        result.update(
            {
                "animal": animal,
                "session_date": session_date,
                "epoch": epoch,
                "subset": subset,
                "stop_cohort": stop_cohort,
                "retrospective_status": status,
                "elapsed_s": time.time() - t0,
            }
        )
        rows.append(result)
        if (i + 1) % 25 == 0 or i + 1 == len(sessions):
            logger.info("  done %d/%d", i + 1, len(sessions))
    return pd.DataFrame(rows)
