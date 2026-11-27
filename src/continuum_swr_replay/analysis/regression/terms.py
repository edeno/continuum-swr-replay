"""Model terms and the design they build, the same for every session.

A model is a list of terms and, for the diagonal families, an intercept:

- :class:`Smooth`: a cubic P-spline of one column over a fixed range
  (:class:`.pspline.PSplineBasis`, clamped beyond it) with one sum-to-zero
  constraint (on the pooled rows, or on declared points, ``center_on``);
  with ``by``, one smooth for each level of a category, zero on the
  rows of the other levels. Its ``penalty`` is required: ``"none"`` (the same
  basis, unpenalized) or a fixed log penalty.
- :class:`LinearFunctional`: a smooth curve over lags whose column ``b`` is
  ``sum_l B_b(log l) r_l`` (``r_l`` the value at lag ``l``), with no constraint
  and the same required ``penalty``.
- :class:`Category`: one 0/1 column for each level but the reference and
  the levels declared ``uncoded``, whose rows are at the reference in the
  category's term and which stay levels for a ``by``. A
  missing value (None or NaN) makes its row missing; any other value outside
  the levels is an error. The reference level must have rows in every fit:
  without them the intercept (or, under the grouped softmax, the other
  levels) and the levels' columns are collinear, and the loop refuses the
  fit, naming it, the category and its reference level, whether or not the
  category is penalized (a penalty would hide the collinearity and split the
  levels by the prior alone). For a category that is a baseline (below), each
  session of a fit keeps the reference level, so the reference level must
  have rows in every session of the fit.
- :class:`Slope`: one column, or one for each level of a category.

A :class:`Category` and a :class:`Slope` take the same required ``penalty``
as a smooth. A fixed log penalty puts the identity on each of the term's
columns (each coded level; each level's slope), as its own penalty block: a
Normal prior of mean 0 on each coefficient, of SD
``exp(-rho / 2)`` (the inverse of :func:`log_penalty_from_prior_sd`). The coefficient is in the unit the
convention reports (a 0/1 step; one pooled SD of a continuous covariate, see
Standardization), so the prior SD is in that unit too. A category used only
as the ``by`` of another term is not a term: its ``penalty`` is not read.
- :class:`Offset`: ``log`` of a column (an exposure), the Poisson offset.

:func:`build_design` computes, once, from the pooled rows of the model (every
session's rows of the model and kind) what every session shares: the knots, the
sum-to-zero constraints, the penalties and the standardization of each slope.
Nothing pooled is computed per session. :meth:`Design.session` maps one
session's rows to its design matrix and offset.

Names. Every term and coefficient is named by one rule, which estimands and
records use to find them:

- a :class:`Smooth` is the term ``s(<column>)``, its columns
  ``s(<column>).<b>`` (``b`` from 0);
- a :class:`Slope` is the term and the coefficient ``<column>``;
- a :class:`LinearFunctional` is ``lf(<first lag column>..<last lag
  column>)``, its columns ``lf(...).<b>``;
- a :class:`Category` is the term ``<column>``, each level's coefficient
  ``<column>[<level>]``;
- a smooth or a slope fitted for one level of a category ``by`` appends
  ``:<by column>=<level>`` to its term name (``s(clock_s):role=current``,
  ``reward_fraction:role=current``), and its columns follow from it;
- the intercept is ``(Intercept)``.

Standardization (:func:`standardization`): each covariate of a design has one
scale and one centering reference, computed once over the pooled rows of every
level on which any :class:`Slope` of the design reads it (all rows for a slope
without ``by``), so that "per SD" means the same for every term of the
covariate. A covariate that is 0/1 on those rows enters as it is. Any other is
divided by its pooled standard deviation (population form, ``ddof = 0``), and a
slope fitted for each level of a category (an interaction) is first centered
at the pooled mean, the fixed reference; a slope alone is not centered.

The design check: the fitter limits each Newton step to ``MAX_STEP_LENGTH`` in
coefficient units, so a column whose values are all very small needs a
coefficient too large to reach in the steps allowed, and its fit ends at the
step limit. A term's size is its largest absolute value over the rows; a
smooth's columns are measured together, as one term. Only the terms whose
columns take values other than 0 and 1 are measured: a smooth, a lag
functional and a continuous slope (standardized, a slope can still be small
on the pooled rows that are not missing: when it shares its scale with a
slope of the covariate on another level, or when its spread lies on rows
missing for another term). On the pooled rows that are not missing, a term
below ``SMALLEST_COLUMN_SIZE`` is the caller's error (a units problem):
:func:`build_design` raises, naming it. In one session, a term below it on the
session's rows (:meth:`Design.small_terms`) fails the support rule in that
session and is made inactive. A term zero on every row of a session (a level
the session does not have, a term by such a level) is not small: the support
rule makes it inactive, a level or a slope by its counts, a smooth or a lag
functional because its columns are zero there.

Each term of a built design is described once, by a :class:`DesignTerm`: its
kind, columns, level, covariate, penalty block and the terms that leave a
session with it. The fitting loop and the support rule read these.

Baselines (design C, ``docs/design/2026-11-24-unit-of-fit.md``, section 4.1).
A model declares its baseline terms to :func:`build_design` (``baseline``):
the intercept (``(Intercept)``) or a :class:`Category` by its column. In a fit
of several sessions (:meth:`Design.for_sessions`) each baseline term is
replaced by one copy for each session, zero on the rows of the other sessions:
the intercept becomes one 0/1 column for each session, and each coded level of
a category one column for each session, so each session keeps the category's
reference level. A copy is named ``<name>@<animal>/<session_date>/<epoch>``
(``(Intercept)@a/20260101/02_r1``, ``arm[3]@a/20260101/02_r1``) and is a
``level`` term whose level is the base level and the session key
(``("(Intercept)", key)`` for the intercept, ``(column, value, key)`` for a
level), with no dependents and its ``session`` set to the key. So the loop
judges and drops it as a level, on its session's rows: an intercept copy that
fails the support rule, or that the refit leaves out, drops its session's rows
from the fit; a level copy that fails drops that level's rows in its session
only. This is the one statement of the rule; the loop (:mod:`.sessions`)
applies it. A baseline carries no penalty, and no term may be fitted ``by`` a
baseline category (such a term could not leave one session's rows when that
session's copy of the level fails): :func:`build_design` refuses both. A
baseline is never tested and never read by a contrast. The collinearity
record projects the copies out, as it does the intercept, and neither the edf
nor :attr:`Design.term_columns` holds them. In a fit of one session nothing is
copied.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd

from continuum_swr_replay.tables.session_keys import SessionKey

from .constants import LOG_PENALTY_BOUNDS
from .pspline import PSplineBasis, sum_to_zero_constraint
from .reml import PenaltyBlock

__all__ = [
    "INTERCEPT",
    "SMALLEST_COLUMN_SIZE",
    "SessionKey",
    "Category",
    "Design",
    "DesignTerm",
    "LinearFunctional",
    "Offset",
    "SessionDesign",
    "Slope",
    "Smooth",
    "Standardization",
    "build_design",
    "copy_name",
    "log_penalty_from_prior_sd",
    "standardization",
]

#: The design check's threshold on a column's largest absolute value over a
#: session's rows. Measured (simulated designs of 600 rows: an intercept, three
#: N(0,1) columns and a P-spline of 7 columns, Poisson, Bernoulli and grouped
#: softmax, five seeds, each N(0,1) column multiplied in turn by 1e-4 to 1e-2,
#: fitted by ``reml.fit_at_penalties`` at log penalty 2): every fit whose column's
#: largest absolute value was 3.3e-3 or less ended at the step limit in at least
#: one case, and every fit at 6.0e-3 or more converged. 1e-2 is three times the
#: largest that failed.
SMALLEST_COLUMN_SIZE = 1e-2

#: The intercept's coefficient and term name.
INTERCEPT = "(Intercept)"


def copy_name(name: str, key: SessionKey) -> str:
    """The name of a session's copy of a baseline term or coefficient:
    ``<name>@<animal>/<session_date>/<epoch>``."""
    return f"{name}@{'/'.join(key)}"


Penalty = Literal["none"] | float


def _check_penalty(penalty: Penalty) -> None:
    """Raise unless ``penalty`` is ``"none"`` or a log penalty in the bounds."""
    if penalty == "none":
        return
    if isinstance(penalty, bool) or not isinstance(penalty, int | float):
        raise ValueError(f'penalty must be "none" or a log penalty, not {penalty!r}')
    lo, hi = LOG_PENALTY_BOUNDS
    if not lo <= penalty <= hi:
        raise ValueError(f"log penalty {penalty} outside the bounds [{lo}, {hi}]")


def log_penalty_from_prior_sd(sd: float) -> float:
    """The log penalty of a Normal prior of SD ``sd`` on one coefficient.

    The fit minimizes ``-l + exp(rho) b**2 / 2``, so a prior ``N(0, sd**2)``
    is ``rho = log(1 / sd**2)``. Under the Gaussian the penalty is on the
    scale of the residual sum of squares (:mod:`.reml`), so there ``sd`` is in
    units of the residual SD.

    Parameters
    ----------
    sd : float
        The prior SD, positive and finite, in the coefficient's unit.

    Returns
    -------
    float
    """
    if not (np.isfinite(sd) and sd > 0):
        raise ValueError(f"a prior SD must be positive and finite, not {sd}")
    return float(-2.0 * np.log(sd))


@dataclass(frozen=True)
class Category:
    """A category: one 0/1 column for each level but ``reference``.

    Every row's value must be one of ``levels``. A category is also the ``by``
    of a :class:`Smooth` or :class:`Slope`.

    Attributes
    ----------
    column : str
    levels : tuple
        Every value the column takes, in the order of its columns.
    reference : object
        The level without a column (absorbed by the intercept or, under the
        grouped softmax, by the other levels).
    penalty : "none" or float
        ``"none"`` for no penalty, else the fixed log penalty of the identity
        on each coded level's column (module docstring); not read when the
        category is only a ``by``.
    uncoded : tuple
        Levels, beside the reference, with no column of their own in the
        category's term: there their rows are at the reference. They stay
        levels for a ``by`` (a smooth or a slope can be fitted for one), and
        their rows are not the reference's rows when the loop asks for those.
        Use it only for a level that is an exact function of other terms
        (its coefficient is not a parameter of the model). It is not a way to
        merge a level into the reference, which the support rule refuses
        (:mod:`.support`). The fitter does not check that the level is such a
        function: the caller must check it on its rows. Empty by default:
        every level but the reference has a column.
    """

    column: str
    levels: tuple
    reference: object
    penalty: Penalty
    uncoded: tuple = ()

    def __post_init__(self) -> None:
        _check_penalty(self.penalty)
        object.__setattr__(self, "levels", tuple(self.levels))
        object.__setattr__(self, "uncoded", tuple(self.uncoded))
        if len(set(self.levels)) != len(self.levels):
            raise ValueError(f"{self.column}: levels repeat: {self.levels}")
        if self.reference not in self.levels:
            raise ValueError(
                f"{self.column}: reference {self.reference!r} not among the levels "
                f"{self.levels}"
            )
        if len(set(self.uncoded)) != len(self.uncoded):
            raise ValueError(f"{self.column}: uncoded levels repeat: {self.uncoded}")
        unknown = [level for level in self.uncoded if level not in self.levels]
        if unknown:
            raise ValueError(
                f"{self.column}: uncoded levels {unknown} not among the levels "
                f"{self.levels}"
            )
        if self.reference in self.uncoded:
            raise ValueError(
                f"{self.column}: the reference {self.reference!r} has no column "
                "already; it cannot be uncoded"
            )
        if not _coded(self):
            raise ValueError(
                f"{self.column}: no coded level: every level is the reference or "
                f"uncoded ({self.uncoded})"
            )


def _by_levels(by: Category | None, levels: tuple | None, name: str) -> tuple:
    """The levels of ``by`` a term is fitted for: ``levels``, or all."""
    if by is None:
        if levels is not None:
            raise ValueError(f"{name}: levels given without a by category")
        return ()
    if levels is None:
        return by.levels
    levels = tuple(levels)
    unknown = [level for level in levels if level not in by.levels]
    if unknown or not levels:
        raise ValueError(
            f"{name}: levels {levels} must be a non-empty subset of {by.column}'s "
            f"levels {by.levels}"
        )
    return levels


@dataclass(frozen=True)
class Smooth:
    """A cubic P-spline of ``column`` over ``[lo, hi]``, clamped beyond it.

    Attributes
    ----------
    column : str
    lo, hi : float
        The range of the knots (the same for every session).
    n_basis : int
        Basis functions before the sum-to-zero constraint (one fewer after).
    penalty : "none" or float
        ``"none"`` for no penalty, else the fixed log strength of the
        second-order difference penalty.
    by : Category or None
        One smooth for each of ``levels`` of this category, each zero on the
        rows of the other levels, with its own constraint and penalty.
    levels : tuple or None
        The levels of ``by`` that get a smooth; all when None.
    center_on : tuple of float or None
        The points of ``column`` the sum-to-zero constraint is computed on:
        the curve sums to zero over these points (the same points for every
        level of ``by``), so its mean over them is zero. None (the default):
        over the pooled rows (of the level). The points must be finite and
        inside ``[lo, hi]``.
    """

    column: str
    lo: float
    hi: float
    n_basis: int
    penalty: Penalty
    by: Category | None = None
    levels: tuple | None = None
    center_on: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        _check_penalty(self.penalty)
        PSplineBasis(self.lo, self.hi, self.n_basis)
        object.__setattr__(
            self, "levels", _by_levels(self.by, self.levels, f"s({self.column})")
        )
        if self.center_on is not None:
            points = tuple(float(x) for x in self.center_on)
            if not points or not all(
                np.isfinite(x) and self.lo <= x <= self.hi for x in points
            ):
                raise ValueError(
                    f"s({self.column}): center_on must be finite points inside "
                    f"[{self.lo}, {self.hi}], not {self.center_on}"
                )
            object.__setattr__(self, "center_on", points)


@dataclass(frozen=True)
class LinearFunctional:
    """A smooth curve ``f`` over log lag, entering as ``sum_l f(log l) r_l``.

    Column ``b`` of the design is ``sum_l B_b(log l) r_l``, with ``r_l`` the
    value of ``lag_columns[l]`` (for reward history, 1 rewarded, 0 omitted).
    There is no sum-to-zero constraint.

    Attributes
    ----------
    lag_columns : tuple of str
        One column per lag, in the order of ``lags``.
    lags : tuple of int
        The lags (at least 1); the basis is over ``log(lags)``.
    lo, hi : float
        The range of the knots, on the log-lag scale.
    n_basis : int
    penalty : "none" or float
        As :class:`Smooth`.
    """

    lag_columns: tuple[str, ...]
    lags: tuple[int, ...]
    lo: float
    hi: float
    n_basis: int
    penalty: Penalty

    def __post_init__(self) -> None:
        object.__setattr__(self, "lag_columns", tuple(self.lag_columns))
        object.__setattr__(self, "lags", tuple(int(lag) for lag in self.lags))
        if len(self.lag_columns) != len(self.lags) or not self.lags:
            raise ValueError("one lag column per lag, at least one")
        if min(self.lags) < 1:
            raise ValueError(f"lags must be at least 1, not {self.lags}")
        _check_penalty(self.penalty)
        PSplineBasis(self.lo, self.hi, self.n_basis)


@dataclass(frozen=True)
class Slope:
    """A linear term in ``column``; with ``by``, one for each of ``levels``.

    Attributes
    ----------
    column : str
    penalty : "none" or float
        ``"none"`` for no penalty, else the fixed log penalty of the identity
        on the slope's column (on each level's, with ``by``).
    by : Category or None
    levels : tuple or None
        The levels of ``by`` that get a slope (the rows of the others are zero
        and their values are not read); all when None.
    """

    column: str
    penalty: Penalty
    by: Category | None = None
    levels: tuple | None = None

    def __post_init__(self) -> None:
        _check_penalty(self.penalty)
        object.__setattr__(
            self, "levels", _by_levels(self.by, self.levels, self.column)
        )


@dataclass(frozen=True)
class Offset:
    """The Poisson offset ``log(column)``, the column an exposure (> 0)."""

    column: str


Term = Smooth | LinearFunctional | Category | Slope | Offset


class Standardization(NamedTuple):
    """A covariate's scale: ``x / sd``, or ``(x - reference) / sd`` for a
    slope in an interaction.

    Attributes
    ----------
    sd : float
        The pooled population standard deviation (``ddof = 0``).
    reference : float
        The pooled mean.
    """

    sd: float
    reference: float


def standardization(values: np.ndarray) -> Standardization | None:
    """The pooled standardization of a covariate.

    Parameters
    ----------
    values : np.ndarray, shape (n_rows,)
        The pooled rows' values (finite).

    Returns
    -------
    Standardization or None
        None for a 0/1 column, which enters as it is (never centered, never
        scaled).

    Raises
    ------
    ValueError
        If the values are constant.
    """
    values = np.asarray(values, dtype=float)
    if np.isin(values, (0.0, 1.0)).all():
        return None
    sd = float(np.std(values, ddof=0))
    if not sd > 0:
        raise ValueError("a constant column cannot be standardized")
    return Standardization(sd, float(np.mean(values)))


Kind = Literal["intercept", "level", "binary", "slope", "smooth", "lag"]

#: The kinds whose columns take values other than 0 and 1, so that they can be
#: below ``SMALLEST_COLUMN_SIZE`` (the design check measures only these).
_MEASURED = ("slope", "smooth", "lag")


@dataclass(frozen=True, eq=False)
class DesignTerm:
    """One term of a design as a session's fit judges it and removes it.

    Attributes
    ----------
    name : str
        A level's or a slope's coefficient name; a smooth's or a lag
        functional's term name; the intercept's. The support rule and the
        records name the term by it.
    term : str
        The model term it belongs to: a level's category (``role`` for
        ``role[previous]``); else ``name``. The edf and the collinearity are
        per model term.
    kind : {"intercept", "level", "binary", "slope", "smooth", "lag"}
        ``level`` a level of a :class:`Category`; ``binary`` a 0/1
        :class:`Slope`; ``slope`` a continuous :class:`Slope`; ``lag`` a
        :class:`LinearFunctional`.
    columns : np.ndarray of int, shape (k,)
        Its design columns.
    level : tuple or None
        ``(column, value)``: the level a ``level`` codes, or the level a term
        ``by`` a category is fitted for; None otherwise.
    covariate : str or None
        The column a slope or a smooth reads; None otherwise.
    block : int or None
        Its penalty block's place in :attr:`Design.blocks`; None without a
        penalty. Each level of a penalized category has its own block.
    dependents : tuple of str
        For a level, the names of the terms fitted for it: when the level is
        removed from a session, its rows are dropped (not merged into the
        reference level) and these terms leave with it.
    session : tuple or None
        For a session's copy of a baseline term (module docstring), the
        session's key; None otherwise.
    """

    name: str
    term: str
    kind: Kind
    columns: np.ndarray
    level: tuple | None = None
    covariate: str | None = None
    block: int | None = None
    dependents: tuple[str, ...] = ()
    session: tuple | None = None

    @property
    def is_copy(self) -> bool:
        """A session's copy of a baseline term (module docstring)."""
        return self.session is not None

    @property
    def reads(self) -> tuple | None:
        """The level whose rows the term reads; None for every row (a level
        is judged on every row)."""
        return None if self.kind == "level" else self.level


@dataclass(frozen=True, eq=False)
class _Built:
    """A term as built on the pooled rows: its name, its design terms and how
    to compute its columns at any rows."""

    name: str
    term: Term
    design_terms: tuple[DesignTerm, ...]
    level: tuple | None = None  # (by column, level) for a term by a category
    basis: PSplineBasis | None = None
    constraint: np.ndarray | None = None
    penalty: np.ndarray | None = None  # a simple term's: on each column
    scale: Standardization | None = None

    @property
    def category(self) -> Category | None:
        """The category the term codes or is fitted by."""
        if isinstance(self.term, Category):
            return self.term
        return getattr(self.term, "by", None)

    @property
    def centered(self) -> bool:
        """A standardized slope by a category is centered at its reference."""
        return self.scale is not None and self.term.by is not None

    @property
    def log_penalty(self) -> float | None:
        """The fixed log penalty; None without one."""
        return None if self.penalty is None else float(self.term.penalty)

    def values(self, rows: pd.DataFrame, levels: _Levels) -> np.ndarray:
        """The term's columns at ``rows``, shape ``(n_rows, k)``; NaN where a
        value it reads is missing (a missing level of the category it codes
        or is fitted by, among them)."""
        out = self._values(rows, levels)
        if self.category is not None:
            out[levels.index(self.category) < 0] = np.nan
        return out

    def _values(self, rows: pd.DataFrame, levels: _Levels) -> np.ndarray:
        """:meth:`values` before the rows of a missing level are set to NaN."""
        term = self.term
        n = len(rows)
        if isinstance(term, Category):
            index = levels.index(term)
            return np.column_stack(
                [index == term.levels.index(level) for level in _coded(term)]
            ).astype(float)
        if isinstance(term, LinearFunctional):
            lags = rows[list(term.lag_columns)].to_numpy(dtype=float)
            return lags @ self.basis.design(np.log(np.asarray(term.lags, float)))
        x = rows[term.column].to_numpy(dtype=float)
        if isinstance(term, Smooth):
            if self.level is None:
                return self._spline(x)
            inside = levels.rows(term.by, self.level[1])
            out = np.empty((n, self.constraint.shape[1]))
            # A row of another level is zero: the basis at 0 times 0, whose
            # signed zeros the rows of the level do not change.
            out[~inside] = 0.0 * self._spline(np.zeros(1))
            out[inside] = self._spline(x[inside])
            return out
        mask = (
            np.ones(n, dtype=bool)
            if self.level is None
            else levels.rows(term.by, self.level[1])
        )
        x = np.where(mask, x, 0.0)
        if self.scale is not None:
            reference = self.scale.reference if self.centered else 0.0
            x[mask] = (x[mask] - reference) / self.scale.sd
        return x[:, None]

    def _spline(self, x: np.ndarray) -> np.ndarray:
        """The constrained basis at ``x``, shape ``(n, k)``; NaN where ``x`` is
        not finite."""
        out = self.basis.design(np.nan_to_num(x)) @ self.constraint
        out[~np.isfinite(x)] = np.nan
        return out

    @property
    def columns(self) -> slice:
        """The term's design columns."""
        return slice(
            int(self.design_terms[0].columns[0]),
            int(self.design_terms[-1].columns[-1]) + 1,
        )

    def record(self) -> dict | None:
        """The term's basis record (C7 ``basis``); None for a term without one."""
        if self.basis is None:
            return None
        kind = "pspline" if isinstance(self.term, Smooth) else "linear_functional"
        return {
            "kind": kind,
            "knots": self.basis.knots.tolist(),
            "degree": self.basis.degree,
            "range": [self.basis.lo, self.basis.hi],
            "constraint": None if self.constraint is None else self.constraint.tolist(),
            "clamp": True,
        }


def _coded(category: Category) -> list:
    """The levels with a column: every level but the reference and the
    uncoded levels."""
    return [
        level
        for level in category.levels
        if level != category.reference and level not in category.uncoded
    ]


class _Levels:
    """Each category's level at a set of rows, read and checked once.

    ``index(category)`` is each row's place in ``category.levels``, -1 where
    the value is missing (None or NaN: the row is missing); any other value
    outside the levels is an error. ``rows(category, value)`` is the rows of
    one level.
    """

    def __init__(self, rows: pd.DataFrame) -> None:
        self._frame = rows
        self._index: dict[Category, np.ndarray] = {}

    def index(self, category: Category) -> np.ndarray:
        """Each row's place in ``category.levels``; -1 for a missing value."""
        if category not in self._index:
            self._index[category] = _level_index(self._frame, category)
        return self._index[category]

    def rows(self, category: Category, value: object) -> np.ndarray:
        """True for a row whose ``category`` is ``value``."""
        return self.index(category) == category.levels.index(value)


def _level_index(rows: pd.DataFrame, category: Category) -> np.ndarray:
    """Each row's place in ``category.levels``, shape ``(n_rows,)``: -1 for a
    missing value (None or NaN); a value outside the levels is an error."""
    values = rows[category.column]
    missing = values.isna().to_numpy()
    index = pd.Index(list(category.levels)).get_indexer(values)
    index[missing] = -1
    unknown = (index == -1) & ~missing
    if unknown.any():
        raise ValueError(
            f"{category.column}: values outside the levels {category.levels}: "
            f"{sorted(map(str, set(values.to_numpy()[unknown])))}"
        )
    return index


def _level_name(term: Smooth | Slope, base: str, level: object) -> str:
    return f"{base}:{term.by.column}={level}"


def _build_term(
    term: Term,
    pooled: pd.DataFrame,
    scales: dict[str, Standardization | None],
    levels: _Levels,
) -> list[_Built]:
    """The built pieces of one term (one per level of a ``by``), their design
    terms' columns numbered from 0 within the piece."""
    if isinstance(term, Category):
        return [_build_category(term, levels)]
    if isinstance(term, LinearFunctional):
        return [_build_lag(term)]
    by_levels = term.levels if term.by is not None else (None,)
    if isinstance(term, Smooth):
        return [_build_smooth(term, level, pooled, levels) for level in by_levels]
    return [_build_slope(term, level, scales) for level in by_levels]


def _build_category(term: Category, levels: _Levels) -> _Built:
    """A category: one level column for each coded level."""
    levels.index(term)
    kinds = tuple(
        DesignTerm(
            f"{term.column}[{level}]",
            term.column,
            "level",
            np.array([j]),
            level=(term.column, level),
        )
        for j, level in enumerate(_coded(term))
    )
    return _Built(term.column, term, kinds, penalty=_identity(term))


def _build_lag(term: LinearFunctional) -> _Built:
    """A lag functional: its basis over the log lags."""
    name = f"lf({term.lag_columns[0]}..{term.lag_columns[-1]})"
    basis = PSplineBasis(term.lo, term.hi, term.n_basis)
    kind = DesignTerm(name, name, "lag", np.arange(term.n_basis))
    penalty = None if term.penalty == "none" else basis.penalty()
    return _Built(name, term, (kind,), basis=basis, penalty=penalty)


def _build_smooth(
    term: Smooth, level: object, pooled: pd.DataFrame, levels: _Levels
) -> _Built:
    """A smooth, or its piece on one level of its ``by``: the sum-to-zero
    constraint on the pooled rows (of the level), or on ``center_on`` when
    given, and the penalty ``Z' P Z``."""
    base = f"s({term.column})"
    name = base if level is None else _level_name(term, base, level)
    x = pooled[term.column].to_numpy(dtype=float)
    if level is not None:
        x = x[levels.rows(term.by, level)]
    x = _finite(x, name)
    basis = PSplineBasis(term.lo, term.hi, term.n_basis)
    centered = x if term.center_on is None else np.asarray(term.center_on)
    Z = sum_to_zero_constraint(basis.design(centered))
    level_key = None if level is None else (term.by.column, level)
    kind = DesignTerm(
        name,
        name,
        "smooth",
        np.arange(term.n_basis - 1),
        level=level_key,
        covariate=term.column,
    )
    penalty = None if term.penalty == "none" else Z.T @ basis.penalty() @ Z
    return _Built(
        name, term, (kind,), level_key, basis=basis, constraint=Z, penalty=penalty
    )


def _build_slope(
    term: Slope, level: object, scales: dict[str, Standardization | None]
) -> _Built:
    """A slope, or its piece on one level of its ``by``: 0/1 unscaled, a
    continuous column standardized."""
    name = term.column if level is None else _level_name(term, term.column, level)
    scale = scales[term.column]
    level_key = None if level is None else (term.by.column, level)
    kind = DesignTerm(
        name,
        name,
        "binary" if scale is None else "slope",
        np.arange(1),
        level=level_key,
        covariate=term.column,
    )
    return _Built(name, term, (kind,), level_key, scale=scale, penalty=_identity(term))


def _identity(term: Category | Slope) -> np.ndarray | None:
    """A simple term's penalty on each of its columns: the identity; None
    without a penalty."""
    return None if term.penalty == "none" else np.eye(1)


def _covariate_scales(
    terms: Sequence[Term], pooled: pd.DataFrame, levels: _Levels
) -> dict[str, Standardization | None]:
    """Each slope covariate's one standardization: over the pooled rows of
    every level on which any slope of the covariate reads it."""
    reads: dict[str, np.ndarray] = {}
    for term in terms:
        if not isinstance(term, Slope):
            continue
        rows = np.ones(len(pooled), dtype=bool)
        if term.by is not None:
            index = levels.index(term.by)
            rows = np.isin(index, [term.by.levels.index(lv) for lv in term.levels])
        reads[term.column] = reads.get(term.column, np.zeros(len(pooled), bool)) | rows
    return {
        column: standardization(
            _finite(pooled[column].to_numpy(dtype=float)[rows], column)
        )
        for column, rows in reads.items()
    }


def _finite(x: np.ndarray, name: str) -> np.ndarray:
    """``x``, checked non-empty and finite."""
    if x.size == 0:
        raise ValueError(f"{name}: no pooled rows")
    if not np.isfinite(x).all():
        raise ValueError(
            f"{name}: {np.sum(~np.isfinite(x))} pooled rows not finite; drop "
            "the rows with missing values before building the design"
        )
    return x


def _small_terms(X: np.ndarray, terms: Sequence[DesignTerm]) -> dict[str, float]:
    """The design check on the rows of ``X``: each term of a measured kind
    whose largest absolute value is above zero and below
    ``SMALLEST_COLUMN_SIZE``, with that value (a smooth's columns measured
    together, as one term)."""
    size = np.max(np.abs(X), axis=0, initial=0.0)
    return _below_smallest(
        lambda term: float(np.max(size[term.columns], initial=0.0)), terms
    )


def _below_smallest(
    largest: Callable[[DesignTerm], float], terms: Sequence[DesignTerm]
) -> dict[str, float]:
    """Each measured term, in the check's order, whose largest absolute value
    (``largest(term)``) is above zero and below ``SMALLEST_COLUMN_SIZE``,
    with that value."""
    out = {}
    for term in _check_order(terms):
        size = largest(term)
        if 0.0 < size < SMALLEST_COLUMN_SIZE:
            out[term.name] = size
    return out


def _check_order(terms: Sequence[DesignTerm]) -> list[DesignTerm]:
    """The measured terms, the smooths and lag functionals first."""
    measured = [t for t in terms if t.kind in _MEASURED]
    return sorted(measured, key=lambda t: t.kind == "slope")


class SessionDesign(NamedTuple):
    """A design built at a set of rows (:meth:`Design.build`).

    Attributes
    ----------
    X : np.ndarray, shape (n_rows, p)
        NaN where a value read is missing.
    offset : np.ndarray or None, shape (n_rows,)
        ``log`` of the exposure (NaN where it is not positive); None without
        an :class:`Offset`.
    missing : np.ndarray of bool, shape (n_rows,)
        True for a row with a missing value the design reads.
    level_rows : dict of tuple to np.ndarray of bool
        ``(column, value)`` -> the rows of that level, for every level a
        design term codes or is fitted for, and every uncoded level of a
        category term.
    """

    X: np.ndarray
    offset: np.ndarray | None
    missing: np.ndarray
    level_rows: dict[tuple, np.ndarray]


@dataclass(frozen=True)
class Design:
    """The terms of a model as built on its pooled rows (:func:`build_design`).

    Attributes
    ----------
    terms : tuple of DesignTerm
        One per term as a session's fit judges it, in column order: the
        intercept, each level of a category, each slope, smooth and lag
        functional (one per level of a ``by``).
    coef_names : tuple of str
        One per design column, in order.
    blocks : tuple of PenaltyBlock
        One per term with a fixed penalty, in the order of the terms; a
        penalized category has one per coded level.
    log_penalties : np.ndarray, shape (n_blocks,)
        Each block's fixed log penalty.
    offset : Offset or None
    baseline : tuple of str
        The baseline terms (module docstring): ``(Intercept)`` or a category's
        column.
    sessions : tuple of SessionKey
        The sessions of a fit whose baselines are copied
        (:meth:`for_sessions`); empty for the design of the pooled rows.
    """

    terms: tuple[DesignTerm, ...]
    coef_names: tuple[str, ...]
    blocks: tuple[PenaltyBlock, ...]
    log_penalties: np.ndarray
    offset: Offset | None
    _built: tuple[_Built, ...] = field(repr=False)
    baseline: tuple[str, ...] = ()
    sessions: tuple[SessionKey, ...] = ()
    # A design with copies: the design of the pooled rows, and for each column
    # the base column it copies.
    _base: Design | None = field(default=None, repr=False)
    _source: np.ndarray | None = field(default=None, repr=False)

    @cached_property
    def by_name(self) -> dict[str, DesignTerm]:
        """Term name -> its :class:`DesignTerm`."""
        return {term.name: term for term in self.terms}

    @cached_property
    def column_terms(self) -> tuple[DesignTerm, ...]:
        """Each design column's :class:`DesignTerm`, in column order."""
        return tuple(term for term in self.terms for _ in term.columns)

    @property
    def categories(self) -> tuple[Category, ...]:
        """The :class:`Category` terms, in order (a category that is only the
        ``by`` of another term is not a term)."""
        return tuple(b.term for b in self._built if isinstance(b.term, Category))

    @property
    def intercept_column(self) -> int | None:
        """The intercept's column (0); None without one."""
        return 0 if self.terms and self.terms[0].kind == "intercept" else None

    @property
    def term_columns(self) -> dict[str, np.ndarray]:
        """Model term name -> its column indices, in column order (the
        intercept and the baseline copies are not terms)."""
        out: dict[str, list[np.ndarray]] = {}
        for term in self.terms:
            if term.kind != "intercept" and not term.is_copy:
                out.setdefault(term.term, []).append(term.columns)
        return {name: np.concatenate(parts) for name, parts in out.items()}

    @property
    def projected_columns(self) -> list[int]:
        """The columns the collinearity record projects out, in column order:
        the intercept's, or every baseline copy's (module docstring)."""
        return [
            int(j)
            for term in self.terms
            if term.kind == "intercept" or term.is_copy
            for j in term.columns
        ]

    def session_baselines(self) -> dict[tuple, list[str]]:
        """Session key -> the coefficient names of its baseline copies, in
        column order; empty without copies."""
        out: dict[tuple, list[str]] = {key: [] for key in self.sessions}
        for term in self.terms:
            if term.is_copy:
                out[term.session] += [self.coef_names[j] for j in term.columns]
        return out

    def baseline_read(self, c: np.ndarray) -> list[str]:
        """The coefficients of a baseline term on which ``c`` (shape
        ``(n_coef,)``, over :attr:`coef_names`) is not zero."""
        return [
            self.coef_names[j]
            for j in np.flatnonzero(c)
            if self.column_terms[j].term in self.baseline
        ]

    def contrast_on(self, c: np.ndarray, coef_names: Sequence[str]) -> np.ndarray:
        """``c`` over this design's coefficients as a contrast of a fit whose
        coefficients are ``coef_names`` (a fit of one unit, whose baseline
        copies replace the design's baselines), by name.

        Parameters
        ----------
        c : np.ndarray, shape (n_coef,)
        coef_names : sequence of str
            The fit's coefficient names (a record's ``coef_names``).

        Returns
        -------
        np.ndarray, shape (len(coef_names),)

        Raises
        ------
        ValueError
            If ``c`` reads a baseline: a baseline has one copy per session in
            a fit of several sessions, and no contrast reads it.
        """
        read = self.baseline_read(c)
        if read:
            raise ValueError(f"a contrast reads the baseline {read}")
        index = {name: j for j, name in enumerate(coef_names)}
        out = np.zeros(len(coef_names))
        for j in np.flatnonzero(c):
            out[index[self.coef_names[j]]] = c[j]
        return out

    def for_sessions(self, keys: Sequence[SessionKey]) -> Design:
        """The design of one fit of the sessions ``keys``: each baseline term
        copied for each session (module docstring); this design itself for
        one session or without a baseline.

        Parameters
        ----------
        keys : sequence of SessionKey
            The fit's sessions, in the order of their rows.

        Returns
        -------
        Design
            Its :meth:`build` takes each row's place in ``keys``.
        """
        if len(keys) < 2 or not self.baseline:
            return self
        terms, source, blocks, names = [], [], [], []
        for term in self.terms:
            copied = term.term in self.baseline
            for key in keys if copied else [None]:
                columns = np.arange(len(source), len(source) + term.columns.size)
                source.extend(term.columns.tolist())
                if not copied:  # a baseline has no penalty (build_design)
                    block = None
                    if term.block is not None:
                        block = len(blocks)
                        blocks.append(PenaltyBlock(columns, self.blocks[term.block].S))
                    names.extend(self.coef_names[j] for j in term.columns)
                    terms.append(replace(term, columns=columns, block=block))
                    continue
                level = (INTERCEPT, key) if term.level is None else (*term.level, key)
                names.append(copy_name(term.name, key))
                terms.append(
                    DesignTerm(
                        names[-1],
                        term.term,
                        "level",
                        columns,
                        level=level,
                        session=tuple(key),
                    )
                )
        return replace(
            self,
            terms=tuple(terms),
            coef_names=tuple(names),
            blocks=tuple(blocks),
            sessions=tuple(tuple(key) for key in keys),
            _base=self,
            _source=np.array(source, dtype=int),
        )

    @property
    def columns(self) -> tuple[str, ...]:
        """The columns of the rows that :meth:`build` reads, each once, in
        the order of the terms: a term's column (its lag columns for a
        :class:`LinearFunctional`), its ``by`` category's column, and the
        offset's exposure last."""
        out: list[str] = []
        for b in self._built:
            term = b.term
            if isinstance(term, LinearFunctional):
                out += term.lag_columns
            else:
                out.append(term.column)
            if getattr(term, "by", None) is not None:
                out.append(term.by.column)
        if self.offset is not None:
            out.append(self.offset.column)
        return tuple(dict.fromkeys(out))

    def basis_record(self) -> dict[str, dict]:
        """Term -> its basis (C7 ``basis``), for the terms with a basis."""
        return {b.name: b.record() for b in self._built if b.basis is not None}

    def log_penalty_record(self) -> dict[str, dict]:
        """Term -> how its penalty is set (C7 ``log_penalty``), for every
        term (a category by its column, a slope by a category for each
        level): ``{"set": "fixed", "log_penalty": value}`` or ``{"set":
        "none", "log_penalty": None}``."""
        return {
            b.name: {
                "set": "none" if b.log_penalty is None else "fixed",
                "log_penalty": b.log_penalty,
            }
            for b in self._built
        }

    def standardization_record(self) -> dict[str, dict]:
        """Covariate -> its standardization (C7 ``standardization``): its SD,
        ``ddof`` (0), reference and the terms centered at it, one entry per
        scaled covariate (every term of the covariate shares it)."""
        out: dict[str, dict] = {}
        for b in self._built:
            if b.scale is not None:
                entry = out.setdefault(
                    b.term.column,
                    {
                        "sd": b.scale.sd,
                        "ddof": 0,
                        "reference": b.scale.reference,
                        "centered": [],
                    },
                )
                if b.centered:
                    entry["centered"].append(b.name)
        return out

    def build(
        self, rows: pd.DataFrame, session: np.ndarray | None = None
    ) -> SessionDesign:
        """The design at ``rows``, missing values included (as NaN).

        Parameters
        ----------
        rows : pd.DataFrame
        session : np.ndarray of int, shape (n_rows,), or None
            Each row's place in :attr:`sessions`, each session's rows
            together and in that order; required for a design with copies
            (:meth:`for_sessions`), not read otherwise.

        Raises
        ------
        ValueError
            If a category holds a value outside its levels, or a design with
            copies is built without ``session``.
        """
        if self._base is not None:
            return self._build_copies(rows, session)
        levels = _Levels(rows)
        X = np.empty((len(rows), len(self.coef_names)))
        if self.intercept_column is not None:
            X[:, self.intercept_column] = 1.0
        for b in self._built:
            X[:, b.columns] = b.values(rows, levels)
        level_rows = {
            t.level: levels.rows(b.category, t.level[1])
            for b in self._built
            for t in b.design_terms
            if t.level is not None
        }
        for category in self.categories:
            for level in category.uncoded:
                level_rows[category.column, level] = levels.rows(category, level)
        missing = ~np.isfinite(X).all(axis=1)
        offset = self._offset(rows)
        if offset is not None:
            missing |= ~np.isfinite(offset)
        return SessionDesign(X, offset, missing, level_rows)

    def _build_copies(
        self, rows: pd.DataFrame, session: np.ndarray | None
    ) -> SessionDesign:
        """:meth:`build` of a design with copies: the base design's columns,
        each copy zero outside its session's rows (which lie together)."""
        if session is None or len(session) != len(rows):
            raise ValueError("a design with baseline copies needs each row's session")
        base = self._base.build(rows)
        X = np.zeros((len(rows), len(self.coef_names)))
        place = {key: s for s, key in enumerate(self.sessions)}
        bounds = np.searchsorted(session, np.arange(len(self.sessions) + 1))
        level_rows = dict(base.level_rows)
        for term in self.terms:
            source = self._source[term.columns]
            if not term.is_copy:
                X[:, term.columns] = base.X[:, source]
                continue
            s = place[term.session]
            own = slice(bounds[s], bounds[s + 1])
            X[own, term.columns] = base.X[own, source]
            rows_of = np.zeros(len(rows), dtype=bool)
            rows_of[own] = True
            level_rows[term.level] = (
                rows_of
                if term.level[0] == INTERCEPT
                else rows_of & level_rows[term.level[:2]]
            )
        return SessionDesign(X, base.offset, base.missing, level_rows)

    def _offset(self, rows: pd.DataFrame) -> np.ndarray | None:
        """``log`` of the exposure, NaN where it is not positive; None without
        an :class:`Offset`."""
        if self.offset is None:
            return None
        exposure = rows[self.offset.column].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(exposure > 0, np.log(exposure), np.nan)

    def _check_pooled(self, rows: pd.DataFrame, levels: _Levels) -> None:
        """The design check on the pooled rows that are not missing (one
        missing value would make a column's largest value NaN and hide it),
        one built term at a time: no pooled matrix is formed.

        Raises
        ------
        ValueError
            If a term is below ``SMALLEST_COLUMN_SIZE``, naming each.
        """
        offset = self._offset(rows)
        missing = np.zeros(len(rows), bool) if offset is None else ~np.isfinite(offset)
        largest: dict[str, np.ndarray] = {}  # term -> its largest value per row
        for b in self._built:
            values = b.values(rows, levels)
            missing |= ~np.isfinite(values).all(axis=1)
            start = b.columns.start
            for t in b.design_terms:
                if t.kind in _MEASURED:
                    largest[t.name] = np.max(
                        np.abs(values[:, t.columns - start]), axis=1
                    )
        _refuse_small(
            _below_smallest(
                lambda term: float(np.max(largest[term.name][~missing], initial=0.0)),
                self.terms,
            )
        )

    def missing_rows(self, rows: pd.DataFrame) -> np.ndarray:
        """True for a row with a missing value the design reads, shape
        ``(n_rows,)`` (an exposure that is not positive counts as missing)."""
        return self.build(rows).missing

    def session(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        """One session's design and offset, at rows without missing values.

        Parameters
        ----------
        rows : pd.DataFrame
            The session's rows, without missing values (:meth:`missing_rows`).

        Returns
        -------
        X : np.ndarray, shape (n_rows, p)
        offset : np.ndarray or None, shape (n_rows,)
            ``log`` of the exposure; None without an :class:`Offset`.

        Raises
        ------
        ValueError
            If a value read is missing or a category holds a value outside
            its levels. A column too small in the session is not refused here
            (:meth:`small_terms`; the support rule makes it inactive).
        """
        built = self.build(rows)
        if built.missing.any():
            raise ValueError(
                f"{int(built.missing.sum())} rows with missing values; "
                "drop them first (Design.missing_rows)"
            )
        return built.X, built.offset

    def small_terms(self, X: np.ndarray) -> dict[str, float]:
        """The design check in one session: each term (named as
        :class:`DesignTerm` names it; a smooth's columns measured together)
        whose largest absolute value over the rows of ``X`` is above zero and
        below ``SMALLEST_COLUMN_SIZE``, with that value."""
        return _small_terms(X, self.terms)


def _refuse_small(small: dict[str, float]) -> None:
    """Refuse the terms of the pooled design check, naming each."""
    if small:
        named = ", ".join(f"{name} ({size:.2g})" for name, size in small.items())
        raise ValueError(
            "columns whose largest absolute value is below SMALLEST_COLUMN_SIZE = "
            f"{SMALLEST_COLUMN_SIZE:g}: {named}; the fit would end at the "
            "step limit: rescale them"
        )


def build_design(
    terms: Sequence[Term],
    pooled_rows: pd.DataFrame,
    *,
    intercept: bool,
    baseline: Sequence[str] = (),
) -> Design:
    """Build a model's terms on its pooled rows.

    Parameters
    ----------
    terms : sequence of terms
        At most one :class:`Offset`.
    pooled_rows : pd.DataFrame
        Every session's rows of the model and kind: the knots' constraints
        and the standardizations are computed on them, and the rows a slope
        or a smooth reads must be finite there.
    intercept : bool
        Put an intercept in column 0 (not for the grouped softmax, which
        cannot see a column constant within a group).
    baseline : sequence of str
        The model's baseline terms, copied for each session in a fit of
        several sessions (module docstring): ``(Intercept)`` (with
        ``intercept``) or the column of a :class:`Category` term.

    Returns
    -------
    Design

    Raises
    ------
    ValueError
        If a term fails the design check on the pooled rows that are not
        missing (a units problem), naming it; if a baseline is not the
        intercept or a category term of the design.
    """
    offsets = [term for term in terms if isinstance(term, Offset)]
    if len(offsets) > 1:
        raise ValueError("at most one offset")
    levels = _Levels(pooled_rows)
    scales = _covariate_scales(terms, pooled_rows, levels)
    built = [
        piece
        for term in terms
        if not isinstance(term, Offset)
        for piece in _build_term(term, pooled_rows, scales, levels)
    ]
    names = [b.name for b in built]
    if len(set(names)) != len(names):
        raise ValueError(f"terms repeat: {names}")
    design_terms = []
    if intercept:
        design_terms.append(
            DesignTerm(INTERCEPT, INTERCEPT, "intercept", np.array([0]))
        )
    blocks, log_penalties = [], []
    start = int(intercept)
    pieces = []
    for b in built:
        # One block per design term: a smooth or a lag functional is one
        # design term on all its columns, a category one per coded level.
        placed = []
        for t in b.design_terms:
            block = None
            if b.penalty is not None:
                block = len(blocks)
                blocks.append(PenaltyBlock(t.columns + start, b.penalty))
                log_penalties.append(b.log_penalty)
            placed.append(replace(t, columns=t.columns + start, block=block))
        placed = tuple(placed)
        start += sum(t.columns.size for t in placed)
        pieces.append(replace(b, design_terms=placed))
        design_terms.extend(placed)
    design_terms = [
        replace(
            t,
            dependents=tuple(
                d.name for d in design_terms if d.kind != "level" and d.level == t.level
            ),
        )
        if t.kind == "level"
        else t
        for t in design_terms
    ]
    coef_names = [INTERCEPT] * intercept
    for t in design_terms[int(intercept) :]:
        if t.kind in ("smooth", "lag"):
            coef_names += [f"{t.name}.{b}" for b in range(t.columns.size)]
        else:
            coef_names.append(t.name)
    design = Design(
        terms=tuple(design_terms),
        coef_names=tuple(coef_names),
        blocks=tuple(blocks),
        log_penalties=np.array(log_penalties, dtype=float),
        offset=offsets[0] if offsets else None,
        _built=tuple(pieces),
    )
    categories = {t.column: t for t in terms if isinstance(t, Category)}
    allowed = [INTERCEPT] * intercept + list(categories)
    wrong = [name for name in baseline if name not in allowed]
    if wrong or len(set(baseline)) != len(baseline):
        raise ValueError(
            f"a baseline is the intercept or a category term, once: {list(baseline)} "
            f"(the design's: {allowed})"
        )
    penalized = [
        n for n in baseline if n in categories and categories[n].penalty != "none"
    ]
    if penalized:
        raise ValueError(f"a baseline has no penalty: {penalized}")
    by_baseline = [
        b.name
        for b in built
        if getattr(b.term, "by", None) is not None and b.term.by.column in baseline
    ]
    if by_baseline:
        raise ValueError(
            f"terms by the baseline category cannot leave one session: {by_baseline}"
        )
    design = replace(design, baseline=tuple(baseline))
    design._check_pooled(pooled_rows, levels)
    return design
