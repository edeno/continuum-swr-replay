"""Model terms and the design they build, the same for every session.

A model is a list of terms and, for the diagonal families, an intercept:

- :class:`Smooth`: a cubic P-spline of one column over a fixed range
  (:class:`.pspline.PSplineBasis`, clamped beyond it) with one sum-to-zero
  constraint; with ``by``, one smooth for each level of a category, zero on the
  rows of the other levels. Its ``penalty`` is required: ``"none"`` (the same
  basis, unpenalized) or a fixed log penalty.
- :class:`LinearFunctional`: a smooth curve over lags whose column ``b`` is
  ``sum_l B_b(log l) r_l`` (``r_l`` the value at lag ``l``), with no constraint
  and the same required ``penalty``.
- :class:`Category`: one 0/1 column for each level but the reference. A
  missing value (None or NaN) makes its row missing; any other value outside
  the levels is an error. The reference level must have rows in every
  session: without them the intercept (or, under the grouped softmax, the
  other levels) and the levels' columns are collinear, and the session's
  first fit is an error naming them.
- :class:`Slope`: one column, or one for each level of a category.
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
the session does not have) is not small: the support rule's counts make it
inactive.

Each term of a built design is described once, by a :class:`DesignTerm`: its
kind, columns, level, covariate, penalty block and the terms that leave a
session with it. The fitting loop and the support rule read these.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Literal, NamedTuple

import numpy as np
import pandas as pd

from .constants import LOG_PENALTY_BOUNDS
from .pspline import PSplineBasis
from .reml import PenaltyBlock

__all__ = [
    "INTERCEPT",
    "SMALLEST_COLUMN_SIZE",
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
    """

    column: str
    levels: tuple
    reference: object

    def __post_init__(self) -> None:
        object.__setattr__(self, "levels", tuple(self.levels))
        if len(set(self.levels)) != len(self.levels):
            raise ValueError(f"{self.column}: levels repeat: {self.levels}")
        if self.reference not in self.levels:
            raise ValueError(
                f"{self.column}: reference {self.reference!r} not among the levels "
                f"{self.levels}"
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
    """

    column: str
    lo: float
    hi: float
    n_basis: int
    penalty: Penalty
    by: Category | None = None
    levels: tuple | None = None

    def __post_init__(self) -> None:
        _check_penalty(self.penalty)
        PSplineBasis(self.lo, self.hi, self.n_basis)
        object.__setattr__(
            self, "levels", _by_levels(self.by, self.levels, f"s({self.column})")
        )


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
    by : Category or None
    levels : tuple or None
        The levels of ``by`` that get a slope (the rows of the others are zero
        and their values are not read); all when None.
    """

    column: str
    by: Category | None = None
    levels: tuple | None = None

    def __post_init__(self) -> None:
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
        penalty.
    dependents : tuple of str
        For a level, the names of the terms fitted for it: when the level is
        removed from a session, its rows are dropped (not merged into the
        reference level) and these terms leave with it.
    """

    name: str
    term: str
    kind: Kind
    columns: np.ndarray
    level: tuple | None = None
    covariate: str | None = None
    block: int | None = None
    dependents: tuple[str, ...] = ()

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
    penalty: np.ndarray | None = None
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
        value it reads is missing."""
        term = self.term
        n = len(rows)
        if isinstance(term, Category):
            index = levels.index(term)
            out = np.column_stack(
                [index == term.levels.index(level) for level in _coded(term)]
            ).astype(float)
            out[index < 0] = np.nan
            return out
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
            out[levels.index(term.by) < 0] = np.nan
            return out
        mask = (
            np.ones(n, dtype=bool)
            if self.level is None
            else levels.rows(term.by, self.level[1])
        )
        x = np.where(mask, x, 0.0)
        if self.scale is not None:
            reference = self.scale.reference if self.centered else 0.0
            x = np.where(mask, (x - reference) / self.scale.sd, 0.0)
        out = x[:, None]
        if self.level is not None:
            out[levels.index(term.by) < 0] = np.nan
        return out

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
    """The levels with a column: every level but the reference."""
    return [level for level in category.levels if level != category.reference]


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
    codes, uniques = pd.factorize(values)
    place = pd.Index(uniques).get_indexer(list(category.levels))
    lookup = np.full(len(uniques), -2)
    lookup[place[place >= 0]] = np.flatnonzero(place >= 0)
    index = np.full(len(codes), -1)
    index[codes >= 0] = lookup[codes[codes >= 0]]
    unknown = index == -2
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
        return [_Built(term.column, term, kinds)]
    if isinstance(term, LinearFunctional):
        name = f"lf({term.lag_columns[0]}..{term.lag_columns[-1]})"
        basis = PSplineBasis(term.lo, term.hi, term.n_basis)
        kind = DesignTerm(name, name, "lag", np.arange(term.n_basis))
        penalty = None if term.penalty == "none" else basis.penalty()
        return [_Built(name, term, (kind,), basis=basis, penalty=penalty)]
    by_levels = term.levels if term.by is not None else (None,)
    out = []
    for level in by_levels:
        level_key = None if level is None else (term.by.column, level)
        if isinstance(term, Smooth):
            base = f"s({term.column})"
            name = base if level is None else _level_name(term, base, level)
            x = pooled[term.column].to_numpy(dtype=float)
            if level is not None:
                x = x[levels.rows(term.by, level)]
            x = _finite(x, name)
            basis = PSplineBasis(term.lo, term.hi, term.n_basis)
            smooth = basis.constrained(x)
            kind = DesignTerm(
                name,
                name,
                "smooth",
                np.arange(term.n_basis - 1),
                level=level_key,
                covariate=term.column,
            )
            fixed = term.penalty != "none"
            out.append(
                _Built(
                    name,
                    term,
                    (kind,),
                    level_key,
                    basis=basis,
                    constraint=smooth.constraint,
                    penalty=smooth.penalty if fixed else None,
                )
            )
        else:
            name = (
                term.column if level is None else _level_name(term, term.column, level)
            )
            scale = scales[term.column]
            kind = DesignTerm(
                name,
                name,
                "binary" if scale is None else "slope",
                np.arange(1),
                level=level_key,
                covariate=term.column,
            )
            out.append(_Built(name, term, (kind,), level_key, scale=scale))
    return out


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
    out = {}
    for term in _check_order(terms):
        largest = float(np.max(size[term.columns], initial=0.0))
        if 0.0 < largest < SMALLEST_COLUMN_SIZE:
            out[term.name] = largest
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
        design term codes or is fitted for.
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
        One per term with a fixed penalty, in the order of the terms.
    log_penalties : np.ndarray, shape (n_blocks,)
        Each block's fixed log penalty.
    offset : Offset or None
    """

    terms: tuple[DesignTerm, ...]
    coef_names: tuple[str, ...]
    blocks: tuple[PenaltyBlock, ...]
    log_penalties: np.ndarray
    offset: Offset | None
    _built: tuple[_Built, ...] = field(repr=False)

    @cached_property
    def by_name(self) -> dict[str, DesignTerm]:
        """Term name -> its :class:`DesignTerm`."""
        return {term.name: term for term in self.terms}

    @cached_property
    def column_terms(self) -> tuple[DesignTerm, ...]:
        """Each design column's :class:`DesignTerm`, in column order."""
        return tuple(term for term in self.terms for _ in term.columns)

    @property
    def intercept_column(self) -> int | None:
        """The intercept's column (0); None without one."""
        return 0 if self.terms and self.terms[0].kind == "intercept" else None

    @property
    def term_columns(self) -> dict[str, np.ndarray]:
        """Model term name -> its column indices, in column order (the
        intercept is not a term)."""
        out: dict[str, list[np.ndarray]] = {}
        for term in self.terms:
            if term.kind != "intercept":
                out.setdefault(term.term, []).append(term.columns)
        return {name: np.concatenate(parts) for name, parts in out.items()}

    def basis_record(self) -> dict[str, dict]:
        """Term -> its basis (C7 ``basis``), for the terms with a basis."""
        return {b.name: b.record() for b in self._built if b.basis is not None}

    def log_penalty_record(self) -> dict[str, dict]:
        """Term -> how its penalty is set (C7 ``log_penalty``), for the terms
        with a basis: ``{"set": "fixed", "log_penalty": value}`` or
        ``{"set": "none", "log_penalty": None}``."""
        return {
            b.name: {
                "set": "none" if b.log_penalty is None else "fixed",
                "log_penalty": b.log_penalty,
            }
            for b in self._built
            if b.basis is not None
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

    def build(self, rows: pd.DataFrame) -> SessionDesign:
        """The design at ``rows``, missing values included (as NaN).

        Raises
        ------
        ValueError
            If a category holds a value outside its levels.
        """
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
        missing = ~np.isfinite(X).all(axis=1)
        offset = self._offset(rows)
        if offset is not None:
            missing |= ~np.isfinite(offset)
        return SessionDesign(X, offset, missing, level_rows)

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
        small = {}
        for term in _check_order(self.terms):
            size = float(np.max(largest[term.name][~missing], initial=0.0))
            if 0.0 < size < SMALLEST_COLUMN_SIZE:
                small[term.name] = size
        _refuse_small(small)

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
    terms: Sequence[Term], pooled_rows: pd.DataFrame, *, intercept: bool
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

    Returns
    -------
    Design

    Raises
    ------
    ValueError
        If a term fails the design check on the pooled rows that are not
        missing (a units problem), naming it.
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
        block = None
        if b.penalty is not None:
            block = len(blocks)
            width = sum(t.columns.size for t in b.design_terms)
            blocks.append(PenaltyBlock(np.arange(start, start + width), b.penalty))
            log_penalties.append(b.log_penalty)
        placed = tuple(
            replace(t, columns=t.columns + start, block=block) for t in b.design_terms
        )
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
    design._check_pooled(pooled_rows, levels)
    return design
