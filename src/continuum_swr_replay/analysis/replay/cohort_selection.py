"""Shared whole-unit complete-case cohort selection.

Model ladders are only directly comparable when every rung is fit to the same
events or stops.  :func:`freeze_common_cohort` applies that selection once at
the unit grain: if any row for a unit fails a required check, every row for the
unit is removed.  The helper deliberately owns selection only; callers remain
responsible for model-specific provenance, minimum-sample gates, and fitting.
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
