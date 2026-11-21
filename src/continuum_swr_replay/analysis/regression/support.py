"""The support rule: whether a session's data can estimate an unpenalized term.

Decided from the rows before the fit, per column and session. A *unit* is what
the data hold independent replicates of (a visit for a rate model, an event
for a content model); a row *holds an event* when it carries one (a bin with a
count above zero; every candidate row of a content event).

- A level of a :class:`.terms.Category`: ``n_high`` counts the units with an
  event on a row of the level (for a role category, the events with a
  candidate in the role), and the level passes on that count alone
  (:func:`level_support`); ``n_low``, the units with an event on a row of
  another level, is recorded only.
- A 0/1 slope: ``n_high`` counts the units with an event on a row where the
  column is 1, ``n_low`` those with an event on a row where it is 0, and both
  must pass (:func:`binary_support`).
- A continuous slope (over the rows of its level, for a slope by a category):
  among the rows that hold an event, ``n_high`` counts the units with a row
  above the median of those rows and ``n_low`` those with a row below it (ties
  to neither), so a column constant there fails (:func:`slope_support`).

A slope passes when both counts are at least ``MIN_UNITS``, a level when its
own count is. Smooth terms carry a penalty and have no support rule.

A column that fails is made inactive in that session. When it is a level of a
category, the session's rows of that level are dropped from its fit and
counted, not merged into the reference level (merging inflated the belief
variance's false positives in simulation, to 20 % in one case), and every term
by that level leaves the session with them (:func:`level_rows_dropped`). The
same row rule holds for a level made inactive because the session has no
finite maximum along it.

A term that fails in more than ``MAX_FAILING_SHARE`` (the fitter's, 20 %) of
the sessions with a fit leaves the per-session model
(:func:`terms_failing_share`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import NamedTuple, Protocol

import numpy as np
import pandas as pd

from .constants import MAX_FAILING_SHARE
from .newton import CONVERGED
from .terms import Design

__all__ = [
    "MIN_UNITS",
    "Support",
    "binary_support",
    "design_support",
    "level_rows_dropped",
    "level_support",
    "slope_support",
    "terms_failing_share",
]

#: The fewest units with an event on each side for a column to pass.
MIN_UNITS = 5


class Support(NamedTuple):
    """One column's support in one session.

    Attributes
    ----------
    n_low, n_high : int
        Units with an event on each side (0 and 1; below and above the median).
    passes : bool
        Both at least ``MIN_UNITS``.
    """

    n_low: int
    n_high: int
    passes: bool


def _support(n_low: int, n_high: int) -> Support:
    return Support(int(n_low), int(n_high), min(n_low, n_high) >= MIN_UNITS)


def level_support(
    values: np.ndarray, unit: np.ndarray, has_event: np.ndarray
) -> Support:
    """The support of a level of a category: it passes when at least
    ``MIN_UNITS`` units have an event on its rows.

    Parameters
    ----------
    values : np.ndarray, shape (n_rows,)
        The level's column, 0 or 1.
    unit : np.ndarray, shape (n_rows,)
    has_event : np.ndarray of bool, shape (n_rows,)
    """
    both = binary_support(values, unit, has_event)
    return both._replace(passes=both.n_high >= MIN_UNITS)


def binary_support(
    values: np.ndarray, unit: np.ndarray, has_event: np.ndarray
) -> Support:
    """The support of a 0/1 column.

    Parameters
    ----------
    values : np.ndarray, shape (n_rows,)
        The column, 0 or 1.
    unit : np.ndarray, shape (n_rows,)
        Each row's unit.
    has_event : np.ndarray of bool, shape (n_rows,)
        True for a row that holds an event.
    """
    values = np.asarray(values, dtype=float)
    if not np.isin(values, (0.0, 1.0)).all():
        raise ValueError("a binary column takes only 0 and 1")
    high = np.unique(unit[has_event & (values == 1.0)]).size
    low = np.unique(unit[has_event & (values == 0.0)]).size
    return _support(low, high)


def slope_support(
    values: np.ndarray, unit: np.ndarray, has_event: np.ndarray
) -> Support:
    """The support of a continuous slope, split at the median of the rows that
    hold an event.

    Parameters
    ----------
    values : np.ndarray, shape (n_rows,)
        The covariate on the rows the slope reads (its level's rows).
    unit : np.ndarray, shape (n_rows,)
    has_event : np.ndarray of bool, shape (n_rows,)
    """
    x, units = np.asarray(values, dtype=float)[has_event], unit[has_event]
    if x.size == 0:
        return _support(0, 0)
    median = np.median(x)
    return _support(
        np.unique(units[x < median]).size, np.unique(units[x > median]).size
    )


def design_support(
    design: Design,
    rows: pd.DataFrame,
    X: np.ndarray,
    unit: np.ndarray,
    has_event: np.ndarray,
) -> dict[str, Support]:
    """Every unpenalized column's support in one session.

    Parameters
    ----------
    design : Design
    rows : pd.DataFrame
        The session's rows, as :meth:`.terms.Design.session` was given them.
    X : np.ndarray, shape (n_rows, p)
        Their design.
    unit : np.ndarray, shape (n_rows,)
    has_event : np.ndarray of bool, shape (n_rows,)

    Returns
    -------
    dict of str to Support
        Coefficient name -> its support, for the columns with a support rule
        (``DesignColumn.support``), in column order.
    """
    out = {}
    for j, column in enumerate(design.columns):
        if column.support is None:
            continue
        reads = np.ones(len(rows), dtype=bool)
        if column.level is not None and not column.drops_rows:
            reads = design.level_rows(rows, column.level)
        rule = {
            "level": level_support,
            "binary": binary_support,
            "slope": slope_support,
        }[column.support]
        out[column.name] = rule(X[reads, j], unit[reads], has_event[reads])
    return out


def level_rows_dropped(
    design: Design, rows: pd.DataFrame, inactive: Iterable[str]
) -> tuple[np.ndarray, dict[str, int], dict[str, str]]:
    """The row rule for the levels of a category made inactive.

    Parameters
    ----------
    design : Design
    rows : pd.DataFrame
        The session's rows.
    inactive : iterable of str
        Coefficient names made inactive; those that are levels of a category
        (``DesignColumn.drops_rows``) drop their rows.

    Returns
    -------
    keep : np.ndarray of bool, shape (n_rows,)
        The rows kept.
    dropped : dict of str to int
        Level coefficient name -> its rows dropped.
    with_them : dict of str to str
        Coefficient name -> the level it leaves with, for every column of a
        term by a dropped level (zero on the rows kept).
    """
    inactive = set(inactive)
    keep = np.ones(len(rows), dtype=bool)
    dropped, with_them = {}, {}
    for column in design.columns:
        if column.name in inactive and column.drops_rows:
            level = design.level_rows(rows, column.level)
            dropped[column.name] = int((level & keep).sum())
            keep &= ~level
            for other in design.columns:
                if other.level == column.level and not other.drops_rows:
                    with_them[other.name] = column.name
    return keep, dropped, with_them


class _Record(Protocol):
    status: str
    support: Mapping[str, Mapping[str, object]]


def terms_failing_share(records: Sequence[_Record]) -> dict[str, float]:
    """The columns that fail the support rule in too many sessions.

    Parameters
    ----------
    records : sequence of fit records
        One per session (:class:`.record.FitRecord`): ``status`` and
        ``support`` are read. Only the sessions with a fit (status
        "converged") count.

    Returns
    -------
    dict of str to float
        Coefficient name -> the share of the sessions with a fit in which it
        fails, for those whose share is above ``MAX_FAILING_SHARE``.
    """
    fitted = [r for r in records if r.status == CONVERGED]
    names = {name for r in fitted for name in r.support}
    out = {}
    for name in sorted(names):
        failing = sum(
            not r.support[name]["passes"] for r in fitted if name in r.support
        )
        share = failing / len(fitted)
        if share > MAX_FAILING_SHARE:
            out[name] = share
    return out
