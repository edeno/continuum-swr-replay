"""Which per-event response the content ladders are fit on.

A content fit asks how an event's non-local mass is distributed over the nine
candidate arms (or three patches). Two columns can express that distribution.
``arm_count_hard`` / ``patch_count_hard`` count the event's decoded timebins per
candidate; ``arm_fraction`` / ``patch_fraction`` divide those counts by the
event's real-arm bin total, so every event contributes exactly 1.0 no matter how
long it ran.

The fraction is the shipped response. The counts make a long event carry ~32
times the weight of a short one and treat its heavily autocorrelated bins as
independent multinomial draws, so a session's likelihood, AIC and standard
errors are all stated at a grain the data do not have. On the fraction the
likelihood is at the event grain, which is the grain the science is about --
"which arm did this replay represent" -- and the analytic null collapses to
``-n_events * log(n_categories)``. The counts stay available behind
``--response hard_count`` for the retired-count comparison and for the
Dirichlet-multinomial fitter, which needs the integer counts.

This module owns the response name resolution and the fraction contract, so the
ladder runner and the is_swr contrast runner cannot drift apart on either.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

__all__ = [
    "RESPONSE_CHOICES",
    "FRACTION_SUM_TOLERANCE",
    "resolve_response_column",
    "validate_fraction_response",
]

#: CLI values both content runners accept for ``--response``.
RESPONSE_CHOICES = ("fraction", "hard_count")

#: How far an event's candidate fractions may sum from 1.0. The builder divides
#: exact integer counts by one event-level integer total, so the only admissible
#: departure is float rounding over at most nine terms.
FRACTION_SUM_TOLERANCE = 1e-9

_RESPONSE_COLUMNS = {
    ("arm", "fraction"): "arm_fraction",
    ("arm", "hard_count"): "arm_count_hard",
    ("patch", "fraction"): "patch_fraction",
    ("patch", "hard_count"): "patch_count_hard",
}


def resolve_response_column(hierarchy_kind: str, response: str) -> str:
    """Return the response column for one hierarchy and ``--response`` value.

    Parameters
    ----------
    hierarchy_kind : str
        ``"arm"`` or ``"patch"``.
    response : str
        One of :data:`RESPONSE_CHOICES`.

    Returns
    -------
    str
        ``arm_fraction`` / ``patch_fraction`` for ``"fraction"``,
        ``arm_count_hard`` / ``patch_count_hard`` for ``"hard_count"``.

    Raises
    ------
    ValueError
        If either argument is not one of the accepted values.
    """
    try:
        return _RESPONSE_COLUMNS[(hierarchy_kind, response)]
    except KeyError:
        raise ValueError(
            f"no content response column for hierarchy {hierarchy_kind!r} and "
            f"--response {response!r}; hierarchy must be 'arm' or 'patch' and "
            f"response one of {list(RESPONSE_CHOICES)}"
        ) from None


def validate_fraction_response(
    frame: pd.DataFrame,
    response_col: str,
    *,
    event_col: str = "event_id",
    table_label: object = "the content table",
    tolerance: float = FRACTION_SUM_TOLERANCE,
) -> None:
    """Enforce the fraction contract on the loaded table, before any fit.

    Every event either has the response defined on all of its candidates, where
    the values lie in [0, 1] and sum to 1.0, or has it undefined on all of them
    (the event decoded to junction bins only, and is dropped by the common
    cohort). An event with the response defined on some candidates and not
    others would be fit as a partial allocation whose total is neither 1 nor
    zero, so it raises rather than being silently rescaled.

    Parameters
    ----------
    frame : pd.DataFrame
        Long-format candidate rows, one row per event x candidate.
    response_col : str
        The fraction column to check.
    event_col : str, optional
        Event grouping key.
    table_label : object, optional
        Table path or name, quoted in the error messages.
    tolerance : float, optional
        Absolute tolerance on the per-event sum and on the [0, 1] bounds.

    Raises
    ------
    ValueError
        If the column is absent, non-numeric, out of bounds, partially defined
        within an event, or does not sum to 1.0 within ``tolerance``.
    """
    if response_col not in frame.columns:
        raise ValueError(
            f"Canonical content table {table_label} has no {response_col!r} "
            "column, so it cannot be fit on the fraction response. Regenerate "
            "it with the current prepare script, or pass --response hard_count."
        )
    series = frame[response_col]
    if not is_numeric_dtype(series) or is_bool_dtype(series):
        raise ValueError(
            f"Canonical content table {table_label} requires {response_col} to "
            f"be numeric, got dtype {series.dtype}"
        )

    values = series.to_numpy(dtype=float)
    defined = np.isfinite(values)
    if (~defined & ~np.isnan(values)).any():
        raise ValueError(
            f"Canonical content table {table_label} has non-finite "
            f"{response_col} values that are not missing"
        )
    out_of_range = defined & ((values < -tolerance) | (values > 1.0 + tolerance))
    if out_of_range.any():
        raise ValueError(
            f"Canonical content table {table_label} has {int(out_of_range.sum())} "
            f"{response_col} value(s) outside [0, 1]; worst is "
            f"{values[out_of_range][np.argmax(np.abs(values[out_of_range] - 0.5))]:.6g}"
        )

    per_event = (
        pd.DataFrame({event_col: frame[event_col].to_numpy(), "_value": values})
        .groupby(event_col, observed=True, sort=False)["_value"]
        .agg(["size", "count", "sum"])
    )
    partial = per_event["count"].gt(0) & per_event["count"].lt(per_event["size"])
    if partial.any():
        raise ValueError(
            f"Canonical content table {table_label} defines {response_col} on "
            f"only some candidates of {int(partial.sum())} event(s) "
            f"(e.g. {list(per_event.index[partial][:3])}); the fraction must be "
            "defined for every candidate of an event or for none of it"
        )
    complete = per_event["count"].eq(per_event["size"])
    deviation = (per_event.loc[complete, "sum"] - 1.0).abs()
    off = deviation > tolerance
    if off.any():
        worst = deviation[off].idxmax()
        raise ValueError(
            f"Canonical content table {table_label} has {int(off.sum())} event(s) "
            f"whose {response_col} does not sum to 1.0 within {tolerance:g}; "
            f"worst is event {worst!r} at {per_event.loc[worst, 'sum']:.12g}"
        )
