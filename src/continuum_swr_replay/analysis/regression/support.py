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
by that level leaves the session with them (the level's
``DesignTerm.dependents``; :mod:`.sessions` applies the rule). The
same row rule holds for a level made inactive because the session has no
finite maximum along it.

A term that fails in more than ``MAX_FAILING_SHARE`` (the fitter's, 20 %) of
the sessions with a fit leaves the per-session model
(:func:`terms_failing_share`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import NamedTuple, Protocol

import numpy as np

from .constants import MAX_FAILING_SHARE
from .newton import CONVERGED
from .terms import SMALLEST_COLUMN_SIZE, Design

__all__ = [
    "MIN_UNITS",
    "Support",
    "binary_support",
    "design_support",
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
    n_low, n_high : int or None
        Units with an event on each side (0 and 1; below and above the median);
        None for a smooth term that failed the design check.
    passes : bool
        The counts reach ``MIN_UNITS`` (both for a slope, ``n_high`` for a
        level) and the column passes the design check.
    reason : str
        Why it fails, in words; empty when it passes.
    """

    n_low: int | None
    n_high: int | None
    passes: bool
    reason: str = ""


def _support(n_low: int, n_high: int) -> Support:
    passes = min(n_low, n_high) >= MIN_UNITS
    return Support(int(n_low), int(n_high), passes, "" if passes else _FEW)


_FEW = f"fewer than MIN_UNITS = {MIN_UNITS} units with an event"


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
    passes = both.n_high >= MIN_UNITS
    return both._replace(passes=passes, reason="" if passes else _FEW)


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


_RULES = {"level": level_support, "binary": binary_support, "slope": slope_support}


def design_support(
    design: Design,
    X: np.ndarray,
    level_rows: Mapping[tuple, np.ndarray],
    unit: np.ndarray,
    has_event: np.ndarray,
) -> dict[str, Support]:
    """Every unpenalized term's support in one session.

    Parameters
    ----------
    design : Design
    X : np.ndarray, shape (n_rows, p)
        The session's design.
    level_rows : mapping of tuple to np.ndarray of bool, shape (n_rows,)
        Each level's rows (:attr:`.terms.SessionDesign.level_rows`).
    unit : np.ndarray, shape (n_rows,)
    has_event : np.ndarray of bool, shape (n_rows,)

    Returns
    -------
    dict of str to Support
        Term name -> its support, for the terms with a support rule (a level,
        a 0/1 slope, a continuous slope), in column order; then, for a smooth
        or a lag functional that fails the design check in the session
        (:meth:`.terms.Design.small_terms`), its name -> a failing entry
        without counts. A slope that fails the design check fails, with that
        reason.
    """
    small = {
        name: (
            f"largest absolute value {size:.2g} below SMALLEST_COLUMN_SIZE = "
            f"{SMALLEST_COLUMN_SIZE:g} in the session"
        )
        for name, size in design.small_terms(X).items()
    }
    out = {}
    for term in design.terms:
        if term.kind not in _RULES:
            continue
        reads = (
            np.ones(X.shape[0], dtype=bool)
            if term.reads is None
            else level_rows[term.reads]
        )
        (j,) = term.columns
        counted = _RULES[term.kind](X[reads, j], unit[reads], has_event[reads])
        if term.name in small:
            counted = counted._replace(passes=False, reason=small.pop(term.name))
        out[term.name] = counted
    for name, reason in small.items():  # the smooths and lag functionals
        out[name] = Support(None, None, False, reason)
    return out


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
