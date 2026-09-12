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
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype


def _as_python_scalar(value: Any) -> Any:
    """Convert NumPy-style scalar IDs to stable Python scalars."""
    if hasattr(value, "item"):
        return value.item()
    return value


def _sorted_unit_ids(values: Any) -> tuple[Any, ...]:
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
    KeyError
        If any required column is absent.
    ValueError
        If zero-response filtering is requested without ``response_col``.
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
    missing_cols = [column for column in required if column not in frame.columns]
    if missing_cols:
        raise KeyError(
            f"freeze_common_cohort missing required column(s): {missing_cols}"
        )
    if frame[unit_col].isna().any():
        raise ValueError(f"{unit_col} must not contain missing unit identifiers")

    cohort = frame.copy()
    ids_before = _sorted_unit_ids(cohort[unit_col])

    has_missing = cohort[required].isna().any(axis=1)
    missing_ids = _sorted_unit_ids(cohort.loc[has_missing, unit_col])
    if missing_ids:
        cohort = cohort.loc[~cohort[unit_col].isin(missing_ids)].copy()

    numeric_required = [
        column for column in required if is_numeric_dtype(cohort[column])
    ]
    if numeric_required:
        numeric_values = cohort[numeric_required].to_numpy(dtype=float)
        has_nonfinite = ~np.isfinite(numeric_values).all(axis=1)
        nonfinite_ids = _sorted_unit_ids(cohort.loc[has_nonfinite, unit_col])
    else:
        nonfinite_ids = ()
    if nonfinite_ids:
        cohort = cohort.loc[~cohort[unit_col].isin(nonfinite_ids)].copy()

    nonpositive_offset_ids: tuple[Any, ...] = ()
    if offset_col is not None:
        nonpositive = cohort[offset_col] <= 0
        nonpositive_offset_ids = _sorted_unit_ids(cohort.loc[nonpositive, unit_col])
        if nonpositive_offset_ids:
            cohort = cohort.loc[~cohort[unit_col].isin(nonpositive_offset_ids)].copy()

    zero_response_ids: tuple[Any, ...] = ()
    if drop_zero_response:
        response_sum = cohort.groupby(unit_col, observed=True)[response_col].sum()
        zero_response_ids = _sorted_unit_ids(response_sum[response_sum <= 0].index)
        if zero_response_ids:
            cohort = cohort.loc[~cohort[unit_col].isin(zero_response_ids)].copy()

    invalid_choice_ids: tuple[Any, ...] = ()
    if choice_response_col is not None and len(cohort):
        is_binary = cohort[choice_response_col].isin((0, 1))
        nonbinary_ids = _sorted_unit_ids(cohort.loc[~is_binary, unit_col])
        binary_rows = cohort.loc[is_binary]
        chosen_per_unit = binary_rows.groupby(unit_col, observed=True)[
            choice_response_col
        ].sum()
        wrong_sum_ids = _sorted_unit_ids(chosen_per_unit[chosen_per_unit != 1].index)
        invalid_choice_ids = _sorted_unit_ids((*nonbinary_ids, *wrong_sum_ids))
        if invalid_choice_ids:
            cohort = cohort.loc[~cohort[unit_col].isin(invalid_choice_ids)].copy()

    cohort = cohort.reset_index(drop=True)
    included_ids = _sorted_unit_ids(cohort[unit_col])
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
#: rule each one states. ``at_well`` is the canonical shared cohort: both the
#: rate and the content families are read on stops the animal spent at a well,
#: so a rate claim and a content claim describe the same population. The other
#: two exist as separately-published sensitivity bundles -- an in-transit
#: population is never pooled with an at-well one.
STOP_COHORT_RULES = {
    "at_well": "occupancy_state == 'at_well'",
    "in_transit": "occupancy_state == 'in_transit'",
    "all_stops": "no occupancy restriction (every immobility stop)",
}

#: CLI-facing tuple of the accepted cohort names, in the order runners list them.
STOP_COHORTS = tuple(STOP_COHORT_RULES)

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
            f"unknown cohort {cohort!r}; expected one of {list(STOP_COHORT_RULES)}"
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


def _cohort_mask(frame: pd.DataFrame, cohort: str) -> pd.Series:
    """Row mask selecting ``cohort``; raises if the rule cannot be evaluated."""
    if cohort not in STOP_COHORT_RULES:
        raise ValueError(
            f"unknown cohort {cohort!r}; expected one of {list(STOP_COHORT_RULES)}"
        )
    if _OCCUPANCY_COLUMN not in frame.columns:
        raise KeyError(
            f"cohort selection needs the {_OCCUPANCY_COLUMN!r} column; the frame "
            f"carries {list(frame.columns)[:12]}... . Regenerate the canonical "
            "stop/event tables, which carry it by construction."
        )
    if cohort == "all_stops":
        return pd.Series(True, index=frame.index)
    return frame[_OCCUPANCY_COLUMN].eq(cohort)


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
        Population to keep.  ``at_well`` is canonical.

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
    KeyError
        If ``occupancy_state`` is absent.
    ValueError
        If ``cohort`` is not one of the three accepted names.
    """
    keep = _cohort_mask(stop_table, cohort)
    selected = stop_table.loc[keep].reset_index(drop=True)
    report = {
        "cohort": cohort,
        "rule": STOP_COHORT_RULES[cohort],
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
    on the stop table, but the population they mean is the same one: each event
    carries the ``occupancy_state`` of the stop it occurred in, so the same rule
    reads the same column.  Keeping both grains in one module is what makes the
    two families' cohorts comparable -- :func:`select_stop_cohort` and this
    function select the same stops.

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
    KeyError
        If ``occupancy_state`` or ``event_col`` is absent.
    ValueError
        If ``cohort`` is not one of the three accepted names.
    """
    keep = _cohort_mask(long_df, cohort)
    if event_col not in long_df.columns:
        raise KeyError(f"event cohort selection needs the {event_col!r} column")
    selected = long_df.loc[keep].reset_index(drop=True)
    report = {
        "cohort": cohort,
        "rule": STOP_COHORT_RULES[cohort],
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
