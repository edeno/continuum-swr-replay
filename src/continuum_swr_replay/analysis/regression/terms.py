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
  the levels is an error.
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

The design check (:func:`check_column_sizes`): the fitter limits each Newton
step to ``MAX_STEP_LENGTH`` in coefficient units, so a column whose values are
all very small needs a coefficient too large to reach in the steps allowed, and
its fit ends at the step limit. :meth:`Design.session` refuses, naming it, a
column whose largest absolute value over the session's rows is above zero and
below ``SMALLEST_COLUMN_SIZE``. A column zero on every row of a session (a level
the session does not have) is not refused: the support rule makes it inactive.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
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
    "DesignColumn",
    "LinearFunctional",
    "Offset",
    "Slope",
    "Smooth",
    "Standardization",
    "build_design",
    "check_column_sizes",
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
    ddof : int
        0.
    reference : float
        The pooled mean.
    """

    sd: float
    ddof: int
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
    return Standardization(sd, 0, float(np.mean(values)))


def check_column_sizes(X: np.ndarray, groups: Sequence[tuple[str, np.ndarray]]) -> None:
    """Refuse a column, or a smooth's set of columns, whose values are all very
    small.

    Parameters
    ----------
    X : np.ndarray, shape (n_rows, p)
    groups : sequence of (str, np.ndarray of int)
        A name and the columns measured together: one column, or every column
        of a smooth term. A smooth's single columns can be tiny in a session
        that covers a small part of its range while the term is not.

    Raises
    ------
    ValueError
        If a group's largest absolute value is above zero and below
        ``SMALLEST_COLUMN_SIZE``; the message names each such group.
    """
    size = np.max(np.abs(X), axis=0, initial=0.0)
    small = []
    for name, columns in groups:
        largest = float(np.max(size[columns], initial=0.0))
        if 0.0 < largest < SMALLEST_COLUMN_SIZE:
            small.append(f"{name} ({largest:.2g})")
    if small:
        raise ValueError(
            "columns whose largest absolute value is below SMALLEST_COLUMN_SIZE = "
            f"{SMALLEST_COLUMN_SIZE:g}: {', '.join(small)}; the fit would end at the "
            "step limit: rescale them"
        )


class DesignColumn(NamedTuple):
    """One column of a design.

    Attributes
    ----------
    name : str
        The coefficient's name.
    term : str
        The term's name.
    support : {"level", "binary", "slope"} or None
        The support rule that applies (:mod:`.support`): ``"level"`` for a
        level of a :class:`Category`, ``"binary"`` for a 0/1 slope,
        ``"slope"`` for a continuous slope; None for a smooth's column and the
        intercept.
    level : tuple or None
        ``(column, value)`` of a category: the level a :class:`Category`
        column codes, or the level a term ``by`` a category is fitted for; None
        otherwise.
    drops_rows : bool
        True for a level of a :class:`Category`: when the column is made
        inactive in a session, the session's rows of that level are dropped
        (not merged into the reference level), and every term ``by`` that
        level leaves the session with them.
    """

    name: str
    term: str
    support: Literal["level", "binary", "slope"] | None
    level: tuple | None
    drops_rows: bool = False


@dataclass(frozen=True)
class _Built:
    """A term as built on the pooled rows: its name, its columns and how to
    compute them at any rows."""

    name: str
    term: Term
    columns: tuple[DesignColumn, ...]
    level: tuple | None = None  # (by column, level) for a term by a category
    basis: PSplineBasis | None = None
    constraint: np.ndarray | None = None
    penalty: np.ndarray | None = None
    log_penalty: float | None = None
    scale: Standardization | None = None
    centered: bool = False

    def values(self, rows: pd.DataFrame) -> np.ndarray:
        """The term's columns at ``rows``, shape ``(n_rows, k)``; NaN where a
        value it reads is missing."""
        term = self.term
        n = len(rows)
        if isinstance(term, Category):
            value = _category_values(rows, term)
            levels = [level for level in term.levels if level != term.reference]
            out = np.column_stack([value == level for level in levels]).astype(float)
            out[pd.isna(value)] = np.nan
            return out
        if isinstance(term, LinearFunctional):
            lags = rows[list(term.lag_columns)].to_numpy(dtype=float)
            return lags @ self.basis.design(np.log(np.asarray(term.lags, float)))
        mask = np.ones(n, dtype=bool) if self.level is None else _in_level(rows, self)
        x = rows[term.column].to_numpy(dtype=float)
        x = np.where(mask, x, 0.0)
        if isinstance(term, Smooth):
            out = self.basis.design(np.nan_to_num(x)) @ self.constraint
            out[~np.isfinite(x)] = np.nan
            out *= mask[:, None]
        else:
            if self.scale is not None:
                reference = self.scale.reference if self.centered else 0.0
                x = np.where(mask, (x - reference) / self.scale.sd, 0.0)
            out = x[:, None]
        if self.level is not None:
            out[pd.isna(rows[term.by.column].to_numpy())] = np.nan
        return out

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


def _category_values(rows: pd.DataFrame, category: Category) -> np.ndarray:
    """The category's column, checked to hold only its levels or a missing
    value (None or NaN: the row is missing)."""
    value = rows[category.column].to_numpy()
    unknown = ~(pd.Series(value).isin(category.levels).to_numpy() | pd.isna(value))
    if unknown.any():
        raise ValueError(
            f"{category.column}: values outside the levels {category.levels}: "
            f"{sorted(map(str, set(value[unknown])))}"
        )
    return value


def _in_level(rows: pd.DataFrame, built: _Built) -> np.ndarray:
    """The rows in the level a term ``by`` a category is fitted for."""
    column, level = built.level
    return _category_values(rows, built.term.by) == level


def _level_name(term: Smooth | Slope, base: str, level: object) -> str:
    return f"{base}:{term.by.column}={level}"


def _build_term(
    term: Term, pooled: pd.DataFrame, scales: dict[str, Standardization | None]
) -> list[_Built]:
    """The built pieces of one term (one per level of a ``by``)."""
    if isinstance(term, Category):
        _category_values(pooled, term)
        columns = tuple(
            DesignColumn(
                f"{term.column}[{level}]",
                term.column,
                "level",
                (term.column, level),
                drops_rows=True,
            )
            for level in term.levels
            if level != term.reference
        )
        return [_Built(term.column, term, columns)]
    if isinstance(term, LinearFunctional):
        name = f"lf({term.lag_columns[0]}..{term.lag_columns[-1]})"
        basis = PSplineBasis(term.lo, term.hi, term.n_basis)
        columns = tuple(
            DesignColumn(f"{name}.{b}", name, None, None) for b in range(term.n_basis)
        )
        return [
            _Built(
                name,
                term,
                columns,
                basis=basis,
                penalty=None if term.penalty == "none" else basis.penalty(),
                log_penalty=None if term.penalty == "none" else float(term.penalty),
            )
        ]
    levels = term.levels if term.by is not None else (None,)
    out = []
    for level in levels:
        level_key = None if level is None else (term.by.column, level)
        if isinstance(term, Smooth):
            base = f"s({term.column})"
            name = base if level is None else _level_name(term, base, level)
            draft = _Built(name, term, (), level_key)
            x = _pooled_values(pooled, draft, term.column)
            basis = PSplineBasis(term.lo, term.hi, term.n_basis)
            smooth = basis.constrained(x)
            columns = tuple(
                DesignColumn(f"{name}.{b}", name, None, level_key)
                for b in range(term.n_basis - 1)
            )
            fixed = term.penalty != "none"
            out.append(
                _Built(
                    name,
                    term,
                    columns,
                    level_key,
                    basis=basis,
                    constraint=smooth.constraint,
                    penalty=smooth.penalty if fixed else None,
                    log_penalty=float(term.penalty) if fixed else None,
                )
            )
        else:
            name = (
                term.column if level is None else _level_name(term, term.column, level)
            )
            scale = scales[term.column]
            rule = "binary" if scale is None else "slope"
            columns = (DesignColumn(name, name, rule, level_key),)
            centered = scale is not None and term.by is not None
            out.append(
                _Built(name, term, columns, level_key, scale=scale, centered=centered)
            )
    return out


def _covariate_scales(
    terms: Sequence[Term], pooled: pd.DataFrame
) -> dict[str, Standardization | None]:
    """Each slope covariate's one standardization: over the pooled rows of
    every level on which any slope of the covariate reads it."""
    reads: dict[str, np.ndarray] = {}
    for term in terms:
        if not isinstance(term, Slope):
            continue
        rows = np.ones(len(pooled), dtype=bool)
        if term.by is not None:
            rows = np.isin(_category_values(pooled, term.by), term.levels)
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


def _pooled_values(pooled: pd.DataFrame, draft: _Built, column: str) -> np.ndarray:
    """The pooled values a term reads (its level's rows), checked finite."""
    x = pooled[column].to_numpy(dtype=float)
    if draft.level is not None:
        x = x[_in_level(pooled, draft)]
    return _finite(x, draft.name)


@dataclass(frozen=True)
class Design:
    """The terms of a model as built on its pooled rows (:func:`build_design`).

    Attributes
    ----------
    terms : tuple
        As given.
    intercept : bool
        Column 0 is the intercept.
    columns : tuple of DesignColumn
        One per design column, in order.
    blocks : tuple of PenaltyBlock
        One per term with a fixed penalty, in the order of the terms.
    log_penalties : np.ndarray, shape (n_blocks,)
        Each block's fixed log penalty.
    block_terms : tuple of str
        Each block's term name.
    offset : Offset or None
    """

    terms: tuple
    intercept: bool
    columns: tuple[DesignColumn, ...]
    blocks: tuple[PenaltyBlock, ...]
    log_penalties: np.ndarray
    block_terms: tuple[str, ...]
    offset: Offset | None
    _built: tuple[_Built, ...] = field(repr=False)

    @property
    def coef_names(self) -> list[str]:
        """The coefficient names, in column order."""
        return [column.name for column in self.columns]

    @property
    def term_columns(self) -> dict[str, np.ndarray]:
        """Term name -> its column indices, in term order (the intercept is
        not a term)."""
        out: dict[str, list[int]] = {}
        for j, column in enumerate(self.columns):
            if column.term != INTERCEPT:
                out.setdefault(column.term, []).append(j)
        return {name: np.array(index) for name, index in out.items()}

    @property
    def size_groups(self) -> list[tuple[str, np.ndarray]]:
        """What the design check measures together: every column of a smooth
        term (a :class:`Smooth` or :class:`LinearFunctional`) as one group,
        named by the term; every other column alone, named by its
        coefficient."""
        smooth = {b.name for b in self._built if b.basis is not None}
        groups = [
            (name, cols) for name, cols in self.term_columns.items() if name in smooth
        ]
        groups += [
            (column.name, np.array([j]))
            for j, column in enumerate(self.columns)
            if column.term not in smooth
        ]
        return groups

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
        ``ddof``, reference and the terms centered at it, one entry per scaled
        covariate (every term of the covariate shares it)."""
        out: dict[str, dict] = {}
        for b in self._built:
            if b.scale is not None:
                entry = out.setdefault(
                    b.term.column, {**b.scale._asdict(), "centered": []}
                )
                if b.centered:
                    entry["centered"].append(b.name)
        return out

    def _matrix(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        """``X`` and the offset, NaN where a value read is missing."""
        parts = [np.ones((len(rows), 1))] if self.intercept else []
        parts += [b.values(rows) for b in self._built]
        X = np.hstack(parts) if parts else np.empty((len(rows), 0))
        offset = None
        if self.offset is not None:
            exposure = rows[self.offset.column].to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                offset = np.where(exposure > 0, np.log(exposure), np.nan)
        return X, offset

    def missing_rows(self, rows: pd.DataFrame) -> np.ndarray:
        """True for a row with a missing value the design reads, shape
        ``(n_rows,)`` (an exposure that is not positive counts as missing)."""
        X, offset = self._matrix(rows)
        missing = ~np.isfinite(X).all(axis=1)
        if offset is not None:
            missing |= ~np.isfinite(offset)
        return missing

    def level_rows(self, rows: pd.DataFrame, level: tuple) -> np.ndarray:
        """True for a row of ``level`` (``(column, value)`` of a category)."""
        column, value = level
        return rows[column].to_numpy() == value

    def session(self, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        """One session's design and offset.

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
            If a value read is missing, a category holds a value outside its
            levels, or a column fails the design check
            (:func:`check_column_sizes`).
        """
        if self.missing_rows(rows).any():
            raise ValueError(
                f"{int(self.missing_rows(rows).sum())} rows with missing values; "
                "drop them first (Design.missing_rows)"
            )
        X, offset = self._matrix(rows)
        check_column_sizes(X, self.size_groups)
        return X, offset


def build_design(
    terms: Sequence[Term], pooled_rows: pd.DataFrame, *, intercept: bool
) -> Design:
    """Build a model's terms on its pooled rows.

    Parameters
    ----------
    terms : sequence of terms
        At most one :class:`Offset`.
    pooled_rows : pd.DataFrame
        Every session's rows of the model and kind, without missing values:
        the knots' constraints and the standardizations are computed on them.
    intercept : bool
        Put an intercept in column 0 (not for the grouped softmax, which
        cannot see a column constant within a group).

    Returns
    -------
    Design
    """
    offsets = [term for term in terms if isinstance(term, Offset)]
    if len(offsets) > 1:
        raise ValueError("at most one offset")
    scales = _covariate_scales(terms, pooled_rows)
    built = [
        piece
        for term in terms
        if not isinstance(term, Offset)
        for piece in _build_term(term, pooled_rows, scales)
    ]
    names = [b.name for b in built]
    if len(set(names)) != len(names):
        raise ValueError(f"terms repeat: {names}")
    columns = [DesignColumn(INTERCEPT, INTERCEPT, None, None)] * intercept
    blocks, log_penalties, block_terms = [], [], []
    for b in built:
        start = len(columns)
        columns.extend(b.columns)
        if b.penalty is not None:
            blocks.append(PenaltyBlock(np.arange(start, len(columns)), b.penalty))
            log_penalties.append(b.log_penalty)
            block_terms.append(b.name)
    return Design(
        terms=tuple(terms),
        intercept=intercept,
        columns=tuple(columns),
        blocks=tuple(blocks),
        log_penalties=np.array(log_penalties, dtype=float),
        block_terms=tuple(block_terms),
        offset=offsets[0] if offsets else None,
        _built=tuple(built),
    )
