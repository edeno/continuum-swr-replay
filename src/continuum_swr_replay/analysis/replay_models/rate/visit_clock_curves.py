"""Model-free event-start rates on the three visit clocks, and how stops end.

The descriptive counterpart of the visit rate models: the event starts per
at-risk second in bins of each visit clock (time since the first poke-in, time
until the final poke-out, time until the end of the current immobile stretch),
read from the visit time-bin design (``replay_models.visit_design``), and the
stops that start inside a visit, per animal, by how they end. Two of the clocks
count down to a future event (``covariate_timing.FUTURE_REFERENCING``), so these
are descriptions, not predictions.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    select_visit_context,
)
from continuum_swr_replay.tables.columns import require_columns

__all__ = ["VISIT_CLOCKS", "stop_end_kind_counts", "visit_clock_rate_curves"]

#: The visit clocks of the time-bin design, in the order they are described.
VISIT_CLOCKS: tuple[str, ...] = (
    "time_since_visit_start_s",
    "time_until_visit_end_s",
    "time_until_stretch_end_s",
)


def visit_clock_rate_curves(
    bins: pd.DataFrame,
    *,
    clock: str,
    edges: np.ndarray,
    by: Sequence[str] = ("kind", "animal"),
) -> pd.DataFrame:
    """Event starts per at-risk second in bins of one clock, per group.

    Each design bin is placed by its clock value (at the bin's center), so its
    exposure and count go wholly to one clock bin. Design bins whose clock is
    missing or outside ``edges`` are left out.

    Parameters
    ----------
    bins : pd.DataFrame
        Design bins (``visit_timebins``, already selected): ``clock``,
        ``exposure``, ``count`` and the ``by`` columns.
    clock : str
        The clock column to bin on.
    edges : np.ndarray, shape (n_clock_bins + 1,)
        Ascending clock bin edges (s); a clock bin is ``[edges[i], edges[i + 1])``.
    by : sequence of str
        The grouping columns, default ``("kind", "animal")``.

    Returns
    -------
    pd.DataFrame
        One row per group and clock bin (every clock bin of every group, empty
        ones included): the ``by`` columns, ``clock``, ``clock_bin_start_s``,
        ``clock_bin_end_s``, ``n_starts`` (int), ``at_risk_s`` and ``rate``
        (``n_starts / at_risk_s``; NaN where ``at_risk_s`` is 0).
    """
    by = list(by)
    require_columns(bins, [clock, "exposure", "count", *by], "visit clock curves")
    edges = np.asarray(edges, dtype=float)
    if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0):
        raise ValueError("edges must be one-dimensional, ascending, of length >= 2")
    value = bins[clock].to_numpy(dtype=float)
    index = np.searchsorted(edges, value, side="right") - 1
    keep = np.isfinite(value) & (index >= 0) & (index < edges.size - 1)
    placed = bins.loc[keep, by].assign(
        clock_bin=index[keep],
        n_starts=bins.loc[keep, "count"].to_numpy(dtype=float),
        at_risk_s=bins.loc[keep, "exposure"].to_numpy(dtype=float),
    )
    sums = placed.groupby([*by, "clock_bin"], observed=True)[
        ["n_starts", "at_risk_s"]
    ].sum()
    groups = bins[by].drop_duplicates().sort_values(by)
    full = pd.MultiIndex.from_tuples(
        [
            (*group, clock_bin)
            for group in groups.itertuples(index=False)
            for clock_bin in range(edges.size - 1)
        ],
        names=[*by, "clock_bin"],
    )
    out = sums.reindex(full, fill_value=0.0).reset_index()
    out["n_starts"] = out["n_starts"].round().astype(np.int64)
    at_risk = out["at_risk_s"].to_numpy()
    out["rate"] = np.divide(
        out["n_starts"].to_numpy(dtype=float),
        at_risk,
        out=np.full(len(out), np.nan),
        where=at_risk > 0,
    )
    clock_bin = out.pop("clock_bin").to_numpy()
    out.insert(len(by), "clock", clock)
    out.insert(len(by) + 1, "clock_bin_start_s", edges[clock_bin])
    out.insert(len(by) + 2, "clock_bin_end_s", edges[clock_bin + 1])
    return out


def stop_end_kind_counts(stop_table: pd.DataFrame) -> pd.DataFrame:
    """The analyzed stops of each animal by how they end (``stop_end_kind``).

    The analyzed stops are those that start inside a well visit
    (``select_visit_context``'s default, ``STARTS_IN_WELL_VISIT``), the stops
    the visit time bins read.

    Parameters
    ----------
    stop_table : pd.DataFrame
        ``animal``, ``visit_context`` and ``stop_end_kind``.

    Returns
    -------
    pd.DataFrame
        ``animal``, ``stop_end_kind``, ``n_stops`` (int) and ``share`` (of the
        animal's analyzed stops), one row per kind an animal has.
    """
    require_columns(
        stop_table,
        ["animal", "visit_context", "stop_end_kind"],
        "stop-end kind counts",
    )
    counts = (
        select_visit_context(stop_table)
        .groupby(["animal", "stop_end_kind"], observed=True)
        .size()
        .rename("n_stops")
        .reset_index()
    )
    counts["share"] = counts["n_stops"] / counts.groupby("animal")["n_stops"].transform(
        "sum"
    )
    return counts
