"""Shared whole-unit complete-case cohort selection.

Model ladders are only directly comparable when every rung is fit to the same
events or stops.  :func:`freeze_common_cohort` applies that selection once at
the unit grain: if any row for a unit fails a required check, every row for the
unit is removed.  The helper deliberately owns selection only; callers remain
responsible for model-specific provenance, minimum-sample gates, and fitting.

:func:`select_stop_cohort` (and its event-grain sibling
:func:`select_event_cohort`) owns the other half of the question: not which
units are complete, but which population the ladder is about at all.  Both the
rate and the content families route their occupancy restriction through it, so
one rule -- stated once, here -- decides the stops both families describe.

:func:`select_visit_context` selects rows by their well-visit label
(``visit_context``): by default the events and stops that start inside a well
visit (``analysis.constants.STARTS_IN_WELL_VISIT``: ``in_visit`` and
``spans_visit_end``), the one subset the analyses of Figures 2 to 5 read.
:func:`count_visit_context`
counts the rows per label, kept and dropped, for a summary to record.
:func:`subset_staleness` says why an output's manifest does not record the
subset (:data:`ANALYZED_SUBSET`), and :func:`check_visit_context_starts`
refuses a table whose labels disagree with the visits a time mask reads.
:func:`select_events_in` keeps the rows of a stored per-event table (one that
carries ``event_id``) whose event is among those selected, so a table made over
every event is read over the same events.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from continuum_swr_replay.analysis.constants import (
    STARTS_IN_WELL_VISIT,
    VISIT_CONTEXTS,
)
from continuum_swr_replay.analysis.event_definition.swr_source import event_epoch_ids
from continuum_swr_replay.analysis.event_definition.well_visits import in_well_visit
from continuum_swr_replay.tables.columns import require_columns


def _as_python_scalar(value: Any) -> Any:
    """Convert NumPy-style scalar IDs to stable Python scalars."""
    if hasattr(value, "item"):
        return value.item()
    return value


def sorted_unit_ids(values: Any) -> tuple[Any, ...]:
    """Return a deterministic tuple of unique, non-missing unit IDs."""
    ids = [_as_python_scalar(value) for value in pd.Series(values).dropna().unique()]
    return tuple(sorted(ids, key=repr))


def freeze_common_cohort(
    frame: pd.DataFrame,
    *,
    unit_col: str,
    required_cols: Sequence[str],
    response_col: str | None = None,
    drop_zero_response: bool = False,
    offset_col: str | None = None,
    choice_response_col: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Freeze one whole-unit complete-case cohort.

    Filtering occurs in a fixed, mutually exclusive order:

    1. remove a unit if any of its rows is missing a required value;
    2. remove a unit if any numeric required value is infinite;
    3. remove a unit if any row has a nonpositive exposure/offset;
    4. optionally remove a unit whose response sums to zero or less;
    5. optionally remove a choice set containing a non-binary response or whose
       response does not sum to exactly one.

    Columns named by ``response_col``, ``offset_col``, and
    ``choice_response_col`` are automatically included among the required
    complete-case columns.  The returned report records unit IDs and counts at
    each filter, making it possible for runners to construct their own
    provenance without duplicating the scientific selection logic.

    Parameters
    ----------
    frame : pandas.DataFrame
        Long- or stop-format rows containing one or more rows per unit.
    unit_col : str
        Event/stop identifier.  A failed row removes every row with this ID.
    required_cols : sequence of str
        Predictors and base columns that must be present and non-missing.
    response_col : str, optional
        Response summed per unit when ``drop_zero_response`` is true.
    drop_zero_response : bool
        Remove units whose response sum is nonpositive.
    offset_col : str, optional
        Exposure column; any nonpositive row removes its whole unit.
    choice_response_col : str, optional
        Binary choice indicator; values must be 0/1 and their sum must equal
        exactly one within every unit.

    Returns
    -------
    cohort : pandas.DataFrame
        A copy containing all rows for retained units, with a fresh integer
        index.  ``frame`` is never mutated.
    report : dict
        Required columns, row/unit counts, included IDs, and mutually exclusive
        dropped-ID/count fields for each filtering stage.

    Raises
    ------
    ValueError
        If any required column is absent, or if zero-response filtering is
        requested without ``response_col``.
    """
    if isinstance(required_cols, (str, bytes)):
        raise TypeError("required_cols must be a sequence of column names")
    if drop_zero_response and response_col is None:
        raise ValueError("response_col is required when drop_zero_response=True")

    required = list(
        dict.fromkeys(
            [
                unit_col,
                *required_cols,
                *([response_col] if response_col is not None else []),
                *([offset_col] if offset_col is not None else []),
                *([choice_response_col] if choice_response_col is not None else []),
            ]
        )
    )
    require_columns(frame, required, "the table to freeze")
    if frame[unit_col].isna().any():
        raise ValueError(f"{unit_col} must not contain missing unit identifiers")

    cohort = frame.copy()
    ids_before = sorted_unit_ids(cohort[unit_col])

    has_missing = cohort[required].isna().any(axis=1)
    missing_ids = sorted_unit_ids(cohort.loc[has_missing, unit_col])
    if missing_ids:
        cohort = cohort.loc[~cohort[unit_col].isin(missing_ids)].copy()

    numeric_required = [
        column for column in required if is_numeric_dtype(cohort[column])
    ]
    if numeric_required:
        numeric_values = cohort[numeric_required].to_numpy(dtype=float)
        has_nonfinite = ~np.isfinite(numeric_values).all(axis=1)
        nonfinite_ids = sorted_unit_ids(cohort.loc[has_nonfinite, unit_col])
    else:
        nonfinite_ids = ()
    if nonfinite_ids:
        cohort = cohort.loc[~cohort[unit_col].isin(nonfinite_ids)].copy()

    nonpositive_offset_ids: tuple[Any, ...] = ()
    if offset_col is not None:
        nonpositive = cohort[offset_col] <= 0
        nonpositive_offset_ids = sorted_unit_ids(cohort.loc[nonpositive, unit_col])
        if nonpositive_offset_ids:
            cohort = cohort.loc[~cohort[unit_col].isin(nonpositive_offset_ids)].copy()

    zero_response_ids: tuple[Any, ...] = ()
    if drop_zero_response:
        response_sum = cohort.groupby(unit_col, observed=True)[response_col].sum()
        zero_response_ids = sorted_unit_ids(response_sum[response_sum <= 0].index)
        if zero_response_ids:
            cohort = cohort.loc[~cohort[unit_col].isin(zero_response_ids)].copy()

    invalid_choice_ids: tuple[Any, ...] = ()
    if choice_response_col is not None and len(cohort):
        is_binary = cohort[choice_response_col].isin((0, 1))
        nonbinary_ids = sorted_unit_ids(cohort.loc[~is_binary, unit_col])
        binary_rows = cohort.loc[is_binary]
        chosen_per_unit = binary_rows.groupby(unit_col, observed=True)[
            choice_response_col
        ].sum()
        wrong_sum_ids = sorted_unit_ids(chosen_per_unit[chosen_per_unit != 1].index)
        invalid_choice_ids = sorted_unit_ids((*nonbinary_ids, *wrong_sum_ids))
        if invalid_choice_ids:
            cohort = cohort.loc[~cohort[unit_col].isin(invalid_choice_ids)].copy()

    cohort = cohort.reset_index(drop=True)
    included_ids = sorted_unit_ids(cohort[unit_col])
    report = {
        "required_columns": tuple(required),
        "n_rows_before": len(frame),
        "n_units_before": len(ids_before),
        "missing_unit_ids": missing_ids,
        "n_missing_units": len(missing_ids),
        "nonfinite_unit_ids": nonfinite_ids,
        "n_nonfinite_units": len(nonfinite_ids),
        "nonpositive_offset_unit_ids": nonpositive_offset_ids,
        "n_nonpositive_offset_units": len(nonpositive_offset_ids),
        "zero_response_unit_ids": zero_response_ids,
        "n_zero_response_units": len(zero_response_ids),
        "invalid_choice_unit_ids": invalid_choice_ids,
        "n_invalid_choice_units": len(invalid_choice_ids),
        "n_rows_after": len(cohort),
        "n_units_after": len(included_ids),
        "included_ids": included_ids,
    }
    return cohort, report


#: The stop/event populations a rate or content ladder may be fit on, and the
#: rule each one states, keyed by ``(cohort, grain)``. ``at_well`` is the
#: canonical shared cohort: both the rate and the content families are read on
#: the animal's time at a well, so a rate claim and a content claim describe the
#: same population. The other two exist as separately-published sensitivity
#: bundles -- an in-transit population is never pooled with an at-well one.
#:
#: The two grains need DIFFERENT predicates to mean the same thing, because
#: ``occupancy_state`` is not the same statement about a stop as it is about an
#: event. A stop is labeled ``mixed`` whenever its sampled states differ, and
#: 14,845 of the 107,400 canonical stops are -- but they are at-well bouts:
#: every one that has a last-well arrival begins after it (median 0.73 s), their
#: median nearest-well distance is 0 cm at both ends, and 96.8% of their events
#: are themselves event-grain ``at_well``. Nearly all of them start inside a
#: well visit and run past its final poke-out (``visit_context ==
#: "spans_visit_end"``), so it is their last samples that read as in-transit.
#: Excluding them from the stop cohort would drop 18,274 of the
#: 59,699 content events' stops (30.6%) and make the two cohorts CROSS rather
#: than nest, so the stop predicate is "well-anchored": ``at_well`` or
#: ``mixed``. An event carries no such ambiguity -- an event window
#: that is not wholly at one well has no defined current arm -- so the event
#: predicate stays the strict one.
STOP_COHORT_RULES = {
    ("at_well", "stop"): (
        "occupancy_state in {'at_well', 'mixed'} (well-anchored: a bout that "
        "begins after arriving at a well and sits at it)"
    ),
    ("at_well", "event"): "occupancy_state == 'at_well'",
    ("in_transit", "stop"): "occupancy_state == 'in_transit'",
    ("in_transit", "event"): "occupancy_state == 'in_transit'",
    ("all_stops", "stop"): "no occupancy restriction (every immobility stop)",
    ("all_stops", "event"): "no occupancy restriction (every event)",
}

#: Occupancy states each ``(cohort, grain)`` predicate keeps. ``all_stops``
#: keeps everything and so states no set.
_COHORT_STATES = {
    ("at_well", "stop"): frozenset({"at_well", "mixed"}),
    ("at_well", "event"): frozenset({"at_well"}),
    ("in_transit", "stop"): frozenset({"in_transit"}),
    ("in_transit", "event"): frozenset({"in_transit"}),
}

#: CLI-facing tuple of the accepted cohort names, in the order runners list them.
STOP_COHORTS = ("at_well", "in_transit", "all_stops")

#: Artifact-name suffix each cohort publishes under. The canonical at-well
#: bundle keeps the unsuffixed names every reader already hard-codes, so a
#: sensitivity bundle can never be mistaken for the canonical one.
STOP_COHORT_ARTIFACT_SUFFIX = {
    "at_well": "",
    "in_transit": "_intransit",
    "all_stops": "_allstops",
}


def stop_cohort_suffix(cohort: str, *, smoke: bool = False) -> str:
    """Artifact-name suffix for ``cohort`` (empty for the canonical at-well one).

    The cohort comes first and the smoke marker last, so the canonical bundle
    keeps the names every reader already hard-codes and a sensitivity smoke
    reads as ``all_models_allstops_smoke.pkl``.

    Parameters
    ----------
    cohort : {"at_well", "all_stops", "in_transit"}
        Cohort the run was fit on.
    smoke : bool
        Whether this is a ``--max-per-animal`` smoke run.

    Returns
    -------
    str
        ``""``, ``"_allstops"``, ``"_intransit"``, each optionally followed by
        ``"_smoke"``.

    Raises
    ------
    ValueError
        If ``cohort`` is not one of the three accepted names.
    """
    try:
        suffix = STOP_COHORT_ARTIFACT_SUFFIX[cohort]
    except KeyError:
        raise ValueError(
            f"unknown cohort {cohort!r}; expected one of {list(STOP_COHORTS)}"
        ) from None
    return suffix + ("_smoke" if smoke else "")


_OCCUPANCY_COLUMN = "occupancy_state"


def _occupancy_counts(states: pd.Series) -> dict[str, int]:
    """Occupancy-state tallies, with missing states named rather than dropped."""
    counts = states.value_counts(dropna=False)
    return {
        ("missing" if pd.isna(state) else str(state)): int(count)
        for state, count in counts.items()
    }


def _occupancy_event_counts(frame: pd.DataFrame, event_col: str) -> dict[str, int]:
    """Occupancy-state tallies at the event grain, not the candidate-row grain.

    A long content frame carries one row per event x candidate, so row counts
    would overstate the cohort by the candidate-set size; the event counts are
    what the ladder is read on.
    """
    states = frame[_OCCUPANCY_COLUMN].fillna("missing")
    counts = frame.groupby(states, observed=True)[event_col].nunique()
    return {str(state): int(count) for state, count in counts.items()}


def _cohort_mask(frame: pd.DataFrame, cohort: str, grain: str) -> pd.Series:
    """Row mask selecting ``cohort`` at ``grain``; raises if it cannot be evaluated."""
    if cohort not in STOP_COHORTS:
        raise ValueError(
            f"unknown cohort {cohort!r}; expected one of {list(STOP_COHORTS)}"
        )
    require_columns(frame, [_OCCUPANCY_COLUMN], f"the {grain} table")
    if cohort == "all_stops":
        return pd.Series(True, index=frame.index)
    # A KeyError here means an unrecognized grain, not a data problem.
    return frame[_OCCUPANCY_COLUMN].isin(_COHORT_STATES[(cohort, grain)])


def select_stop_cohort(
    stop_table: pd.DataFrame, *, cohort: str = "at_well"
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Restrict a stop table to one occupancy cohort.

    The rate and content families are only describing the same population when
    they are fit on the same stops.  This is the single place that rule lives:
    every runner in both families routes its stop/event selection through it so
    the cohort is a stated, auditable choice rather than a filter re-derived per
    script.

    Parameters
    ----------
    stop_table : pandas.DataFrame
        One row per immobility stop, carrying ``occupancy_state``.
    cohort : {"at_well", "all_stops", "in_transit"}
        Population to keep.  ``at_well`` is canonical and, at the stop grain,
        means *well-anchored*: ``occupancy_state`` of ``at_well`` **or**
        ``mixed``.  A ``mixed`` stop is an at-well bout that runs past the
        visit's final poke-out, so its last samples read as in-transit (see
        :data:`STOP_COHORT_RULES`); excluding
        those stops would leave 30.6% of the content cohort's events with no
        stop in the rate cohort.

    Returns
    -------
    selected : pandas.DataFrame
        A copy holding the selected rows with a fresh integer index.
        ``stop_table`` is never mutated.
    report : dict
        ``cohort``, the ``rule`` string, ``n_stops_before``/``n_stops_after``,
        ``n_stops_dropped``, and the ``occupancy_state`` tallies before and
        after (``counts_before`` / ``counts_after``), for the runner log and the
        provenance sidecar.

    Raises
    ------
    ValueError
        If ``occupancy_state`` is absent, or ``cohort`` is not one of the three
        accepted names.
    """
    keep = _cohort_mask(stop_table, cohort, "stop")
    selected = stop_table.loc[keep].reset_index(drop=True)
    report = {
        "cohort": cohort,
        "rule": STOP_COHORT_RULES[(cohort, "stop")],
        "n_stops_before": int(len(stop_table)),
        "n_stops_after": int(len(selected)),
        "n_stops_dropped": int(len(stop_table) - len(selected)),
        "counts_before": _occupancy_counts(stop_table[_OCCUPANCY_COLUMN]),
        "counts_after": (
            _occupancy_counts(selected[_OCCUPANCY_COLUMN]) if len(selected) else {}
        ),
    }
    return selected, report


def select_event_cohort(
    long_df: pd.DataFrame, *, cohort: str = "at_well", event_col: str = "event_id"
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply :data:`STOP_COHORT_RULES` at the event grain.

    The content ladders are fit on a long (event x candidate) frame rather than
    on the stop table, but the population they mean is the same one.  The
    predicate is NOT identical, because ``occupancy_state`` says a different
    thing about an event than about a stop: an event is ``at_well`` only when
    its whole window is at one well (off that, it has no defined current arm),
    while a stop is ``mixed`` as soon as any sampled state differs.  So the
    event grain keeps strictly ``at_well`` and the stop grain keeps the
    well-anchored stops those events live in.  Keeping both grains in one module
    is what makes the two families' cohorts nest: every event this function
    keeps lies in a stop :func:`select_stop_cohort` keeps, up to a 1.0% residual
    of off-well events inside well-anchored stops (609 of 60,308), which are in
    the rate cohort and not the content one.

    Parameters
    ----------
    long_df : pandas.DataFrame
        One row per event x candidate (arm or patch), carrying
        ``occupancy_state`` and ``event_col``.
    cohort : {"at_well", "all_stops", "in_transit"}
        Population to keep.  ``at_well`` is canonical.
    event_col : str
        Event identifier, counted in the report.

    Returns
    -------
    selected : pandas.DataFrame
        A copy holding the selected rows with a fresh integer index.
    report : dict
        As :func:`select_stop_cohort`, at the event grain
        (``n_events_before``/``n_events_after``/``n_events_dropped`` alongside
        the row counts; ``counts_before``/``counts_after`` tally distinct
        events, not candidate rows).

    Raises
    ------
    ValueError
        If ``occupancy_state`` or ``event_col`` is absent, or ``cohort`` is not
        one of the three accepted names.
    """
    keep = _cohort_mask(long_df, cohort, "event")
    require_columns(long_df, [event_col], "the event table")
    selected = long_df.loc[keep].reset_index(drop=True)
    report = {
        "cohort": cohort,
        "rule": STOP_COHORT_RULES[(cohort, "event")],
        "n_rows_before": int(len(long_df)),
        "n_rows_after": int(len(selected)),
        "n_events_before": int(long_df[event_col].nunique()),
        "n_events_after": int(selected[event_col].nunique()),
        "counts_before": _occupancy_event_counts(long_df, event_col),
        "counts_after": (
            _occupancy_event_counts(selected, event_col) if len(selected) else {}
        ),
    }
    report["n_events_dropped"] = report["n_events_before"] - report["n_events_after"]
    return selected, report


#: Column names a stop table carries for the absolute-time bounds of one
#: immobility bout. Both are inclusive edges in seconds on the same clock as
#: ``position_info.index``.
_STOP_BOUND_COLUMNS = ("stop_start", "stop_end")


def _stop_bounds(stops: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Start/end arrays of ``stops``, sorted by start and checked for overlap.

    Overlap is a correctness precondition, not a style preference: the
    ``searchsorted`` membership test below only inspects the latest stop that
    begins at or before a time, so an earlier, longer stop containing that time
    would be missed silently.

    Raises
    ------
    ValueError
        If either bound column is absent, a bound is missing/non-finite, a stop
        ends before it starts, or two stops overlap.
    """
    require_columns(stops, _STOP_BOUND_COLUMNS, "the stop table")
    order = np.argsort(stops["stop_start"].to_numpy(dtype=float), kind="stable")
    starts = stops["stop_start"].to_numpy(dtype=float)[order]
    ends = stops["stop_end"].to_numpy(dtype=float)[order]
    if not (np.isfinite(starts).all() and np.isfinite(ends).all()):
        raise ValueError("stop bounds must all be finite")
    if (ends < starts).any():
        raise ValueError("every stop must satisfy stop_end >= stop_start")
    if starts.size > 1 and (starts[1:] < ends[:-1]).any():
        overlapping = int((starts[1:] < ends[:-1]).sum())
        raise ValueError(
            f"{overlapping} stop interval(s) overlap their predecessor; the "
            "interval mask assumes disjoint stops. Restrict to one session "
            "before calling."
        )
    return starts, ends


def stop_interval_mask(times: np.ndarray, stops: pd.DataFrame) -> np.ndarray:
    """Mark the times that fall inside one of ``stops``.

    The stop table states the cohort as intervals; a time-bin analysis needs it
    as a per-sample mask. This is that translation, done once so every
    consumer -- the PFC unit GLMs today, any later per-bin PFC model -- restricts
    to the same bins.

    Parameters
    ----------
    times : numpy.ndarray, shape (n_time,)
        Sample times in seconds, on the same clock as the stop bounds.
    stops : pandas.DataFrame
        Stops to test against, carrying ``stop_start`` and ``stop_end``. Must be
        disjoint, which the canonical stop table is within a session. An empty
        frame marks nothing.

    Returns
    -------
    numpy.ndarray of bool, shape (n_time,)
        True where ``stop_start <= t <= stop_end`` for some stop. Both edges are
        inclusive, matching the closed intervals the stop table records.

    Raises
    ------
    ValueError
        If ``stop_start`` or ``stop_end`` is absent, or the stops are
        non-finite, inverted, or overlapping.
    """
    times = np.asarray(times, dtype=float)
    starts, ends = _stop_bounds(stops)
    if starts.size == 0:
        return np.zeros(times.shape, dtype=bool)
    # The latest stop beginning at or before each time is the only one that can
    # contain it, because the stops are disjoint.
    index = np.searchsorted(starts, times, side="right") - 1
    has_candidate = index >= 0
    candidate = np.where(has_candidate, index, 0)
    return np.asarray(has_candidate & (times <= ends[candidate]), dtype=bool)


def events_in_stops(event_starts: np.ndarray, stops: pd.DataFrame) -> np.ndarray:
    """Mark the events whose onset falls inside one of ``stops``.

    An event is assigned to a stop by its onset alone, so an event that begins
    inside a kept stop stays whole even if its tail runs past the stop's end.
    That keeps the event count here equal to the count any onset-keyed table
    reports for the same cohort.

    Parameters
    ----------
    event_starts : numpy.ndarray, shape (n_events,)
        Event onset times in seconds, on the stop table's clock.
    stops : pandas.DataFrame
        Stops to test against; see :func:`stop_interval_mask`.

    Returns
    -------
    numpy.ndarray of bool, shape (n_events,)
        True where the onset lies inside a kept stop.
    """
    return stop_interval_mask(np.asarray(event_starts, dtype=float), stops)


_VISIT_CONTEXT_COLUMN = "visit_context"

ANALYZED_SUBSET = (
    "the events that start inside a well visit (visit_context in_visit or "
    "spans_visit_end); seconds, time bins and SWRs inside a well visit (from the "
    "first poke-in at a well to the final poke-out; an SWR counts when its first "
    "sample is inside)"
)
"""What an output made over the analyses' one subset records as its ``subset``
(in its manifest), so a reader can refuse or mark pending one made otherwise."""


def subset_staleness(manifest: Mapping[str, Any], what: str) -> str | None:
    """Why an output is not over the analyses' subset, or None.

    An output made over the subset records :data:`ANALYZED_SUBSET` as its
    ``subset``; one recording another or none was made over other events or
    other time. The events summary marks the values read from such an output
    pending; a figure refuses it.

    Parameters
    ----------
    manifest : Mapping
        The output's manifest.
    what : str
        The output, for the note.
    """
    if manifest.get("subset") != ANALYZED_SUBSET:
        return (
            f"{what} was not made over the events that start inside a well visit "
            "and the time inside one; rerun it before citing"
        )
    return None


def _visit_contexts(contexts: str | Sequence[str]) -> tuple[str, ...]:
    """The requested labels as a tuple, refusing one the vocabulary lacks."""
    requested = (contexts,) if isinstance(contexts, str) else tuple(contexts)
    unknown = sorted(set(requested) - set(VISIT_CONTEXTS))
    if not requested or unknown:
        raise ValueError(
            f"unknown visit_context value(s) {unknown or requested!r}; expected "
            f"some of {list(VISIT_CONTEXTS)}"
        )
    return requested


def select_visit_context(
    table: pd.DataFrame, contexts: str | Sequence[str] = STARTS_IN_WELL_VISIT
) -> pd.DataFrame:
    """Keep the rows of a table whose ``visit_context`` is one of ``contexts``.

    Every event and stop carries a well-visit label (``visit_context``, made by
    ``event_tables.visit_context``); the tables keep the rows outside a visit,
    and this is where an analysis drops them. The default keeps the rows that
    start inside a well visit (from the first nose-poke in at a well to the
    final poke-out there): ``analysis.constants.STARTS_IN_WELL_VISIT``, the
    rows wholly inside one (``in_visit``) and those that start inside one and
    end after its final poke-out (``spans_visit_end``), because an event
    belongs to the time when it starts. Pass ``"in_visit"`` for the rows wholly
    inside a visit only.

    Parameters
    ----------
    table : pandas.DataFrame
        Any canonical table carrying ``visit_context`` (event, stop,
        event-arm or event-patch grain).
    contexts : str or sequence of str
        The labels to keep, from ``analysis.constants.VISIT_CONTEXTS``;
        default ``STARTS_IN_WELL_VISIT``.

    Returns
    -------
    pandas.DataFrame
        A copy of the kept rows with a fresh integer index; ``table`` is not
        changed.

    Raises
    ------
    ValueError
        If ``table`` has no ``visit_context``, or a requested label is not in
        the vocabulary.
    """
    keep = _visit_contexts(contexts)
    require_columns(table, [_VISIT_CONTEXT_COLUMN], "the table to select from")
    return table.loc[table[_VISIT_CONTEXT_COLUMN].isin(keep)].reset_index(drop=True)


def count_visit_context(
    table: pd.DataFrame,
    contexts: str | Sequence[str] = STARTS_IN_WELL_VISIT,
    *,
    unit_col: str | None = None,
) -> pd.DataFrame:
    """Count a table's rows (or units) per ``visit_context``, kept or dropped.

    The record a summary keeps of what :func:`select_visit_context` with the
    same ``contexts`` keeps and drops.

    Parameters
    ----------
    table : pandas.DataFrame
        A table carrying ``visit_context``.
    contexts : str or sequence of str
        The labels :func:`select_visit_context` would keep; default
        ``STARTS_IN_WELL_VISIT``.
    unit_col : str, optional
        Count distinct values of this column (``event_id`` on a candidate-row
        table) instead of rows.

    Returns
    -------
    pandas.DataFrame
        One row per label of ``analysis.constants.VISIT_CONTEXTS``, in that
        order, indexed by ``visit_context``: ``n`` (rows, or distinct units)
        and ``kept`` (bool). Labels with no rows have ``n`` 0.

    Raises
    ------
    ValueError
        As :func:`select_visit_context`, or if ``unit_col`` is absent.
    """
    keep = _visit_contexts(contexts)
    require_columns(
        table,
        [_VISIT_CONTEXT_COLUMN, *([unit_col] if unit_col else [])],
        "the table to count",
    )
    labels = table[_VISIT_CONTEXT_COLUMN]
    if unit_col is None:
        counts = labels.value_counts()
    else:
        counts = table.groupby(labels, observed=True)[unit_col].nunique()
    out = pd.DataFrame(
        {
            "n": counts.reindex(list(VISIT_CONTEXTS), fill_value=0).astype(int),
            "kept": [label in keep for label in VISIT_CONTEXTS],
        }
    )
    out.index.name = _VISIT_CONTEXT_COLUMN
    return out


def select_events_in(table: pd.DataFrame, event_ids: Sequence[Any]) -> pd.DataFrame:
    """Keep the rows of a per-event table whose ``event_id`` is among ``event_ids``.

    A stored table made over every event (a per-event measure, its time bins,
    an event with its matched control row under the same id) is read over the
    events :func:`select_visit_context` keeps by passing their ``event_id``.
    Every row of a kept id is kept, however many it has.

    Parameters
    ----------
    table : pandas.DataFrame
        A table carrying ``event_id``.
    event_ids : sequence
        The ids to keep, for example ``select_visit_context(events)["event_id"]``.

    Returns
    -------
    pandas.DataFrame
        A copy of the kept rows, in their order, with a fresh integer index;
        ``table`` is not changed.

    Raises
    ------
    ValueError
        If ``table`` has no ``event_id``.
    """
    require_columns(table, ["event_id"], "the per-event table to select from")
    keep = table["event_id"].isin(pd.unique(np.asarray(event_ids)))
    return table.loc[keep].reset_index(drop=True)


def check_visit_context_starts(
    events: pd.DataFrame, visits: Mapping[str, tuple[np.ndarray, np.ndarray]]
) -> None:
    """Raise unless each event's label says what its start says about the visits.

    The selection reads ``visit_context``; the time-masked outputs read the
    visits themselves (``well_visits.in_well_visit``). They describe one subset
    only if, for every event, ``visit_context`` is one of
    ``STARTS_IN_WELL_VISIT`` exactly when ``event_start_s`` is inside one of
    its epoch's visits. A table labeled from other visits than ``visits`` (a
    refreshed session cache, another table) fails here.

    Parameters
    ----------
    events : pandas.DataFrame
        Every event of the table: ``animal``, ``session_date``, ``epoch``,
        ``event_start_s`` and ``visit_context``.
    visits : Mapping
        Each epoch id (``swr_source.swr_epoch_id_of_session``) to its visits'
        ``(arrival_times, departure_times)``.

    Raises
    ------
    ValueError
        If an epoch of ``events`` has no visits in ``visits``, or some events'
        label and start disagree (their count, and an example).
    """
    require_columns(
        events,
        ["event_start_s", _VISIT_CONTEXT_COLUMN],
        "the events to check against the visits",
    )
    epoch_ids = event_epoch_ids(events).to_numpy()
    if missing := sorted(set(epoch_ids) - set(visits)):
        raise ValueError(
            f"no well visits for {len(missing)} epochs of the events, e.g. "
            f"{missing[:3]}"
        )
    starts = events["event_start_s"].to_numpy(dtype=np.float64)
    inside = np.zeros(len(events), dtype=bool)
    for identifier in pd.unique(epoch_ids):
        rows = epoch_ids == identifier
        inside[rows] = in_well_visit(starts[rows], *visits[identifier])
    labeled = events[_VISIT_CONTEXT_COLUMN].isin(STARTS_IN_WELL_VISIT).to_numpy()
    disagree = np.flatnonzero(inside != labeled)
    if disagree.size:
        example = events.iloc[disagree[0]]
        raise ValueError(
            f"{disagree.size} of {len(events)} events' visit_context disagrees with "
            "their start against the well visits, e.g. the event of "
            f"{epoch_ids[disagree[0]]} starting at {example['event_start_s']} s, "
            f"labeled {example[_VISIT_CONTEXT_COLUMN]!r} but "
            f"{'inside' if inside[disagree[0]] else 'outside'} a visit"
        )
