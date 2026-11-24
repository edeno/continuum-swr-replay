"""The support rule: whether a session's design can estimate a simple term.

Decided from the design before the fit, per column and session. No rule
reads the response. This is the rule since 2026-11-23, with the weak penalty
on simple terms (``docs/design/regression_fitter.md``, "The support rule"). A
*unit* is what the data hold independent replicates of (a visit for a rate
model, an event for a content model).

- A level of a :class:`.terms.Category`: ``n_high`` counts the units with a
  row of the level, and the level passes on that count alone
  (:func:`level_support`); ``n_low``, the units with a row of another level,
  is recorded only.
- A 0/1 slope: ``n_high`` counts the units with a row where the column is 1,
  ``n_low`` those with a row where it is 0, and both must pass
  (:func:`binary_support`).
- A continuous slope (over the rows of its level, for a slope by a category):
  ``n_high`` counts the units with a row above the median of those rows and
  ``n_low`` those with a row below it (ties to neither), so a column constant
  there fails (:func:`slope_support`).

A slope passes when both counts are at least ``MIN_UNITS``, a level when its
own count is. The rule holds for every simple term, penalized or not. A
smooth and a lag functional have no counts; like every term they are judged
by the design check (:meth:`.terms.Design.small_terms`).

Under the grouped softmax every candidate row belongs to an event, so the
counts are those of the event-based rule this one replaces. Under the Poisson
and the Bernoulli a side may have rows and few events or none. A penalized
term always has a finite maximum and is fitted. An unpenalized term has no
finite maximum only when it is a 0/1 slope or a level with no event on a side
(under the Bernoulli, also with an event on every row of a side); the refit
then leaves it out (:mod:`.sessions`). In every other case it has a finite
maximum and is fitted: a continuous slope, or a level whose events are in one
to four units. The event-based rule left these out; this rule fits them, with
a large standard error. For example, a level with 10 visits and one event,
unpenalized, is fitted at -3.67 with a standard error of 1.0
(``test_sessions.py::test_an_unpenalized_level_with_one_event_is_fitted``).
So an unpenalized simple term is a baseline or a nuisance term, whose value
is not reported or tested. A tested coefficient must carry the penalty;
:func:`require_tested_penalized` checks it.

A column that fails is made inactive in that session. When it is a level of a
category, the session's rows of that level are dropped from its fit and
counted, not merged into the reference level (merging inflated the belief
variance's false positives in simulation, to 20 % in one case), and every term
by that level leaves the session with them (the level's
``DesignTerm.dependents``; :mod:`.sessions` applies the rule). The
same row rule holds for a level made inactive because the session has no
finite maximum along it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum
from typing import NamedTuple

import numpy as np

from .terms import Design

__all__ = [
    "MIN_UNITS",
    "Removed",
    "Support",
    "binary_support",
    "design_support",
    "level_support",
    "require_tested_penalized",
    "slope_support",
]

#: The fewest units with a row on each side for a column to pass.
MIN_UNITS = 5


class Removed(StrEnum):
    """Why a term is inactive in a session's fit (a record's ``left_out``)."""

    #: Fewer than ``MIN_UNITS`` units with a row on a side (the support rule).
    SUPPORT = "support"
    #: Below ``SMALLEST_COLUMN_SIZE`` on the session's rows (the design check).
    TOO_SMALL = "too_small"
    #: Fitted for a level whose rows were dropped.
    WITH_LEVEL = "with_level"
    #: Left out by the refit of a fit with no finite maximum, in a round.
    NO_FINITE_MAXIMUM = "no_finite_maximum"


class Support(NamedTuple):
    """One term's support in one session.

    Attributes
    ----------
    n_low, n_high : int or None
        Units with a row on each side (0 and 1; below and above the median;
        another level and the level); None for a smooth term that failed the
        design check.
    reason : str
        Why it fails: ``Removed.SUPPORT`` or ``Removed.TOO_SMALL``; empty when
        it passes.
    size : float or None
        For a term that fails the design check, its largest absolute value in
        the session; None otherwise.
    """

    n_low: int | None
    n_high: int | None
    reason: str = ""
    size: float | None = None

    @property
    def passes(self) -> bool:
        """The counts reach ``MIN_UNITS`` (both for a slope, ``n_high`` for a
        level) and the term passes the design check: no reason to fail."""
        return self.reason == ""

    def record(self) -> dict:
        """The record's entry: ``{"n_low", "n_high", "passes", "reason",
        "size"}``."""
        return {
            "n_low": self.n_low,
            "n_high": self.n_high,
            "passes": self.passes,
            "reason": self.reason,
            "size": self.size,
        }


def _n_units(unit: np.ndarray) -> int:
    """The distinct units among ``unit``, integer codes from 0."""
    return int(np.count_nonzero(np.bincount(unit))) if unit.size else 0


def _two_sided(unit: np.ndarray, low: np.ndarray, high: np.ndarray) -> Support:
    """The units with a row on each side; both must reach ``MIN_UNITS``."""
    n_low, n_high = _n_units(unit[low]), _n_units(unit[high])
    return Support(
        n_low, n_high, "" if min(n_low, n_high) >= MIN_UNITS else Removed.SUPPORT
    )


def level_support(values: np.ndarray, unit: np.ndarray) -> Support:
    """The support of a level of a category: it passes when at least
    ``MIN_UNITS`` units have a row of the level.

    Parameters
    ----------
    values : np.ndarray, shape (n_rows,)
        The level's column, 0 or 1.
    unit : np.ndarray of int, shape (n_rows,)
        Each row's unit, as integer codes from 0 (``pd.factorize``).
    """
    both = binary_support(values, unit)
    return both._replace(reason="" if both.n_high >= MIN_UNITS else Removed.SUPPORT)


def binary_support(values: np.ndarray, unit: np.ndarray) -> Support:
    """The support of a 0/1 column: the units with a row on each side.

    Parameters
    ----------
    values : np.ndarray, shape (n_rows,)
        The column, 0 or 1.
    unit : np.ndarray of int, shape (n_rows,)
        Each row's unit, as integer codes from 0 (``pd.factorize``).
    """
    values = np.asarray(values, dtype=float)
    if not np.isin(values, (0.0, 1.0)).all():
        raise ValueError("a binary column takes only 0 and 1")
    return _two_sided(unit, values == 0.0, values == 1.0)


def slope_support(values: np.ndarray, unit: np.ndarray) -> Support:
    """The support of a continuous slope, split at the median of its rows.

    Parameters
    ----------
    values : np.ndarray, shape (n_rows,)
        The covariate on the rows the slope reads (its level's rows).
    unit : np.ndarray of int, shape (n_rows,)
        Each row's unit, as integer codes from 0 (``pd.factorize``).
    """
    x = np.asarray(values, dtype=float)
    if not x.size:
        return _two_sided(unit, np.zeros(0, bool), np.zeros(0, bool))
    median = np.median(x)
    return _two_sided(unit, x < median, x > median)


_RULES = {"level": level_support, "binary": binary_support, "slope": slope_support}


def design_support(
    design: Design,
    X: np.ndarray,
    level_rows: Mapping[tuple, np.ndarray],
    unit: np.ndarray,
) -> dict[str, Support]:
    """Every simple term's support in one session.

    Parameters
    ----------
    design : Design
    X : np.ndarray, shape (n_rows, p)
        The session's design.
    level_rows : mapping of tuple to np.ndarray of bool, shape (n_rows,)
        Each level's rows (:attr:`.terms.SessionDesign.level_rows`).
    unit : np.ndarray of int, shape (n_rows,)
        Each row's unit, as integer codes from 0 (``pd.factorize``).

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
    small = design.small_terms(X)
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
        counted = _RULES[term.kind](X[reads, j], unit[reads])
        if term.name in small:
            counted = counted._replace(
                reason=Removed.TOO_SMALL, size=small.pop(term.name)
            )
        out[term.name] = counted
    for name, size in small.items():  # the smooths and lag functionals
        out[name] = Support(None, None, Removed.TOO_SMALL, size)
    return out


def require_tested_penalized(design: Design, tested: Iterable[str]) -> None:
    """Refuse a tested coefficient without a fixed penalty.

    The convention: every coefficient that a model tests carries the weak
    penalty; an unpenalized simple term is a baseline or a nuisance term
    (module docstring). Each model module calls this with the coefficients
    of its declared tests.

    Parameters
    ----------
    design : Design
    tested : iterable of str
        The names that the model tests: a coefficient (``x``,
        ``role[previous]``, ``s(clock_s).3``) or a term (``s(clock_s)``; a
        category, ``role``, checked on each of its levels).

    Raises
    ------
    ValueError
        If a name is not a coefficient or a term of the design, or if a
        name has no fixed penalty (on any of its coefficients, for a
        term). The error names every such name.
    """
    unknown, unpenalized = [], []
    for name in tested:
        if name in design.by_name:
            parts = [design.by_name[name]]
        elif name in design.coef_names:
            parts = [design.column_terms[design.coef_names.index(name)]]
        else:
            parts = [t for t in design.terms if t.term == name]
        if not parts:
            unknown.append(name)
        elif any(t.block is None for t in parts):
            unpenalized.append(name)
    problems = []
    if unknown:
        problems.append(
            f"not a coefficient or term of the design: {', '.join(map(repr, unknown))}"
        )
    if unpenalized:
        problems.append(
            "tested without a fixed penalty (a tested coefficient carries the "
            f"weak penalty): {', '.join(map(repr, unpenalized))}"
        )
    if problems:
        raise ValueError("; ".join(problems))
