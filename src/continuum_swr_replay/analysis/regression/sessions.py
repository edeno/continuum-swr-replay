"""The per-session fitting loop: one design, every session fitted alone.

:func:`fit_sessions` takes a :class:`.terms.Design`, each session's rows, a
likelihood and the names of the response and unit columns, and returns one
:class:`.record.FitRecord` per session. It knows nothing of the task.

A session's design is built once, on all its rows (sorted by unit under the
grouped softmax); from then on one mask over those rows says which are kept,
with one reason for each row removed, and the design of the kept rows is a
slice. For each session, in order:

1. The rows with a missing value the design reads are removed.
2. The levels of the categories, to a fixed point: the support rule
   (:mod:`.support`) is counted on the rows kept, and each level that fails is
   removed: its rows are dropped (not merged into the reference level) and
   the terms fitted for it leave with it (``DesignTerm.dependents``). Under
   the grouped softmax an event with no response on the rows kept is
   removed: the first time as given (``events_without_response``), after
   that as emptied by dropped rows (``events_emptied_by_dropped_rows``).
3. Every other term is judged once, on the final rows: one that fails the
   support rule (or the design check in the session) is inactive in the fit.
   The record holds those counts, and for a removed level the counts that
   made it fail.
4. The session is fitted at the design's fixed log penalties by
   :func:`.reml.refit_session`, which makes the columns running away inactive
   while the fit has no finite maximum along unpenalized columns only. At the
   session's first fit a design not identified, or a numerical failure, is an
   error that names the session. When a level is left out there, it is
   removed as in step 2 and the steps run again from step 2 on the rows then
   kept. A term leaves whole: when the refit leaves out part of the columns
   of a term of several unpenalized columns (a smooth without a penalty, a
   lag functional), the whole term is removed and the session is fitted
   again without it.

A session with no rows, or no events, left is not fitted (``NOT_FITTED``: no
fit was tried); its record says why and the loop goes on. A fit that was tried
and not kept, because the refit ends along the intercept (nothing is dropped or
left out for it), along a penalty direction, after too many rounds or with any
other status, keeps the last fit's status and the true counts, with NaN
coefficients and its cause. Every cause is a code of a closed set
(:class:`.record.NoFit` for a session, :class:`.support.Removed` for a term;
:func:`.record.describe` gives their words).

A session with columns left out keeps its other terms: those columns'
coefficients and covariance are NaN and ``left_out`` says why (with the
refit's round in ``left_out_round`` for no finite maximum). The units are
the ``unit`` column (a visit for a rate model; for the grouped softmax, each
unit is one event, a group of rows). A row holds an event when its response is
above zero (Poisson, Bernoulli), and every row does under the softmax and the
Gaussian. The collinearity of each term is measured at the fit
(:func:`.collinearity.term_collinearity`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .collinearity import term_collinearity
from .likelihoods import Bernoulli, GroupedSoftmax, Likelihood, Poisson
from .newton import CONVERGED, NO_FINITE_MAXIMUM, NOT_IDENTIFIED
from .record import FitRecord, NoFit
from .reml import (
    PenaltyBlock,
    RemlScore,
    SessionData,
    SessionRefit,
    refit_session,
)
from .support import Removed, Support, design_support
from .terms import Design, DesignTerm

__all__ = ["NOT_FITTED", "SessionKey", "fit_sessions"]

SessionKey = tuple[str, str, str]

#: The status of a session for which no fit was tried: no rows, or no events
#: of the kind, are left.
NOT_FITTED = "not fitted"

# Why a row of a session is not in its fit (0: it is).
_KEPT, _MISSING, _LEVEL, _NO_RESPONSE, _EMPTIED = 0, 1, 2, 3, 4


@dataclass
class _Removal:
    """A term removed from one session's fit: its cause, the columns removed
    (with the refit's round of each, for a fit with no finite maximum) and,
    for a level, the support counts when it was removed."""

    cause: Removed
    columns: list[int]
    rounds: list[int] = field(default_factory=list)
    counts: Support | None = None


@dataclass(frozen=True)
class _Identity:
    key: SessionKey
    model: str
    kind: str
    table_sha256: Mapping[str, str]


class _Session:
    """One session's state: its design, built once on all its rows; a reason
    for each row removed; the terms removed; the last support counts."""

    def __init__(
        self,
        design: Design,
        lik: Likelihood,
        rows: pd.DataFrame,
        response: str,
        unit: str,
        name: str,
    ) -> None:
        rows = rows.reset_index(drop=True)
        if isinstance(lik, GroupedSoftmax):
            rows = rows.sort_values(unit, kind="stable").reset_index(drop=True)
        built = design.build(rows)
        self.design, self.lik, self.name = design, lik, name
        self.X, self.offset, self.level_rows = built.X, built.offset, built.level_rows
        self.y = rows[response].to_numpy(dtype=float)
        self.unit = pd.factorize(rows[unit].to_numpy(), use_na_sentinel=False)[0]
        self.reason = np.where(built.missing, _MISSING, _KEPT).astype(np.int8)
        self.dropped_by = np.full(len(rows), -1)  # the place in levels_dropped
        self.levels_dropped: list[str] = []
        self.removed: dict[str, _Removal] = {}
        self.support: dict[str, Support] = {}
        self.events_counted = False
        self.rounds = 0  # the refit rounds before the current fit
        self.no_fit = ""
        self.fitted_with: tuple[SessionData, list[str]] | None = None

    @property
    def kept(self) -> np.ndarray:
        """True for a row in the fit, shape ``(n_rows,)``."""
        return self.reason == _KEPT

    def _not_fittable(self) -> str:
        """Why the rows kept cannot be fitted (a ``NoFit`` code); empty when
        they can."""
        kept = self.kept
        if not kept.any():
            return NoFit.NO_ROWS
        if not _holds_event(self.lik, self.y[kept]).any():
            return NoFit.NO_EVENTS
        return ""

    def _remove_empty_events(self) -> None:
        """Under the softmax, remove the events with no response on the rows
        kept: the first time as given, after that as emptied by dropped rows."""
        kept = self.kept
        response = np.where(np.isnan(self.y), 0.0, self.y)
        total = np.bincount(self.unit[kept], weights=response[kept])
        empty = np.zeros(kept.size, dtype=bool)
        empty[kept] = total[self.unit[kept]] <= 0
        self.reason[empty] = _EMPTIED if self.events_counted else _NO_RESPONSE
        self.events_counted = True

    def _drop_level(self, level: DesignTerm) -> None:
        """Drop a removed level's rows; its dependents leave with it."""
        rows = self.kept & self.level_rows[level.level]
        self.reason[rows] = _LEVEL
        self.dropped_by[rows] = len(self.levels_dropped)
        self.levels_dropped.append(level.name)
        for name in level.dependents:
            if name not in self.removed:
                columns = self.design.by_name[name].columns.tolist()
                self.removed[name] = _Removal(Removed.WITH_LEVEL, columns)

    def judge(self) -> str:
        """The levels to a fixed point, then every other term once on the
        final rows (steps 2 and 3 of the module docstring).

        Returns
        -------
        str
            Why the session cannot be fitted (a ``NoFit`` code); empty when
            it can.
        """
        while True:
            why = self._not_fittable()
            if not why:
                if isinstance(self.lik, GroupedSoftmax):
                    self._remove_empty_events()
                why = self._not_fittable()
            if why:
                return why
            kept = self.kept
            self.support = design_support(
                self.design,
                self.X[kept],
                {level: rows[kept] for level, rows in self.level_rows.items()},
                self.unit[kept],
                _holds_event(self.lik, self.y[kept]),
            )
            for name, counted in self.support.items():
                term = self.design.by_name[name]
                if term.kind == "level" and not counted.passes:
                    self.removed.setdefault(
                        name,
                        _Removal(counted.reason, term.columns.tolist(), counts=counted),
                    )
            pending = [
                term
                for term in self.design.terms
                if term.kind == "level"
                and term.name in self.removed
                and term.name not in self.levels_dropped
            ]
            if not pending:
                return ""
            for term in pending:
                self._drop_level(term)

    def left_out(self) -> dict[str, str]:
        """Coefficient name -> why it is inactive (a ``Removed`` code): the
        levels removed, the terms that left with them, the columns the refit
        left out, then the other terms that fail the support rule on the
        final rows."""
        names = self.design.coef_names
        out: dict[str, str] = {}
        for cause in ("level", Removed.WITH_LEVEL, Removed.NO_FINITE_MAXIMUM):
            for name, removal in self.removed.items():
                is_level = self.design.by_name[name].kind == "level"
                if (cause == "level") != is_level or (
                    not is_level and removal.cause != cause
                ):
                    continue
                for j in removal.columns:
                    out[names[j]] = removal.cause
        for name, counted in self.support.items():
            if not counted.passes:
                for j in self.design.by_name[name].columns:
                    out.setdefault(names[j], counted.reason)
        return out

    def left_out_round(self) -> dict[str, int]:
        """Coefficient name -> the refit's round that left it out, for the
        coefficients left out for no finite maximum."""
        names = self.design.coef_names
        return {
            names[j]: at
            for removal in self.removed.values()
            if removal.cause == Removed.NO_FINITE_MAXIMUM
            for j, at in zip(removal.columns, removal.rounds, strict=True)
        }

    def recorded_support(self) -> dict[str, Support]:
        """The support counts on the final rows; a removed level's when it was
        removed."""
        return {
            name: (
                self.removed[name].counts
                if name in self.removed and self.removed[name].counts is not None
                else counted
            )
            for name, counted in self.support.items()
        }

    def cohort_counts(self) -> dict:
        """The rows and events removed before the fit, by reason."""
        counts = {
            "rows_missing_values": int(np.sum(self.reason == _MISSING)),
            "rows_dropped": {
                name: int(np.sum((self.reason == _LEVEL) & (self.dropped_by == k)))
                for k, name in enumerate(self.levels_dropped)
            },
        }
        if isinstance(self.lik, GroupedSoftmax):
            counts["events_emptied_by_dropped_rows"] = self._units(_EMPTIED)
            if self.events_counted:
                counts["events_without_response"] = self._units(_NO_RESPONSE)
        return counts

    def _units(self, reason: int) -> int:
        """The units with a row removed for ``reason``."""
        return int(np.unique(self.unit[self.reason == reason]).size)

    def data(self) -> tuple[SessionData, list[PenaltyBlock], np.ndarray, list[str]]:
        """The fit's data on the rows kept, every term removed inactive, and
        the penalty blocks with an active column, their log penalties and
        term names (a block whose term left with a level is not fitted)."""
        kept = self.kept
        left_out = self.left_out()
        inactive = np.array([c in left_out for c in self.design.coef_names])
        X, y = self.X[kept], self.y[kept]
        if isinstance(self.lik, GroupedSoftmax):
            aux = np.unique(self.unit[kept], return_inverse=True)[1]
        elif isinstance(self.lik, Poisson):
            aux = (
                np.zeros(int(kept.sum())) if self.offset is None else self.offset[kept]
            )
        else:
            aux = None
        blocked = [t for t in self.design.terms if t.block is not None]
        fitted = [t for t in blocked if not inactive[t.columns].all()]
        blocks = [self.design.blocks[t.block] for t in fitted]
        rho = self.design.log_penalties[[t.block for t in fitted]]
        data = SessionData(X, y, aux, self.name, ~inactive)
        return data, blocks, rho, [t.name for t in fitted]

    def fit(self, first: bool) -> SessionRefit:
        """The session's refit at the design's log penalties.

        Raises
        ------
        ValueError
            At the session's first fit, if its design is not identified or
            the fit fails for a numerical reason, naming the session.
        """
        data, blocks, rho, names = self.data()
        refit = refit_session(self.lik, data, blocks, rho)
        start = refit.refits[0].fit.status
        if first and start not in (CONVERGED, NO_FINITE_MAXIMUM):
            what = (
                "its design is not identified, in the coefficients "
                + _named(self.design, names, refit.refits[0])
                + " (a category's reference level must have rows in every session)"
                if start == NOT_IDENTIFIED
                else f"its fit ended as {start}, a numerical failure"
            )
            raise ValueError(f"session {self.name}: at its first fit {what}")
        self.fitted_with = (data, names)
        return refit

    def apply(self, refit: SessionRefit) -> bool:
        """Remove what the refit left out: a level as in step 2; a term of
        several unpenalized columns (a smooth without a penalty, a lag
        functional) whole, when the refit left out part of its columns.

        Returns
        -------
        bool
            True when the session is fitted again: the refit converged with a
            level left out (the steps run again on the rows then kept), or a
            term left whole.
        """
        level_left = whole = False
        for column, round_ in zip(refit.left_out, refit.rounds, strict=True):
            term = self.design.column_terms[column]
            at = self.rounds + int(round_)
            if term.kind == "level":
                self.removed[term.name] = _Removal(
                    Removed.NO_FINITE_MAXIMUM,
                    [int(column)],
                    [at],
                    counts=self.support[term.name],
                )
                level_left = True
            else:
                removal = self.removed.setdefault(
                    term.name, _Removal(Removed.NO_FINITE_MAXIMUM, [])
                )
                removal.columns.append(int(column))
                removal.rounds.append(at)
        for name, removal in self.removed.items():
            columns = self.design.by_name[name].columns
            if len(removal.columns) < columns.size:  # the refit left out part
                rest = [int(j) for j in columns if j not in removal.columns]
                removal.columns += rest
                removal.rounds += [min(removal.rounds)] * len(rest)
                whole = True
        self.rounds += len(refit.refits) - 1
        self.no_fit = "" if refit.fitted else NoFit(refit.ended.value)
        return (refit.fitted and level_left) or whole


def _holds_event(lik: Likelihood, y: np.ndarray) -> np.ndarray:
    """True for a row that holds an event (module docstring)."""
    if isinstance(lik, Poisson | Bernoulli):
        return y > 0
    return np.ones(y.size, dtype=bool)


def _n_events(lik: Likelihood, y: np.ndarray, unit: np.ndarray) -> float:
    """The events in the rows: the count total (Poisson, Bernoulli), the
    units (grouped softmax), the rows (Gaussian)."""
    if isinstance(lik, Poisson | Bernoulli):
        return float(y.sum())
    if isinstance(lik, GroupedSoftmax):
        return float(np.unique(unit).size)
    return float(y.size)


def _edf(
    design: Design, fit: RemlScore, block_terms: list[str], active: np.ndarray
) -> dict[str, float]:
    """Term -> edf: a penalized term's from the fit, an unpenalized term's
    number of active columns."""
    penalized = dict(zip(block_terms, fit.edf.tolist(), strict=True))
    return {
        name: penalized.get(name, float(active[cols].sum()))
        for name, cols in design.term_columns.items()
    }


def _named(design: Design, block_terms: list[str], score: RemlScore) -> str:
    """The coefficients a design not identified involves, by name: a design
    column by its coefficient, a smooth's coordinate by its term."""
    names = []
    for direction in score.unidentified_directions:
        if direction.block is None:
            names.append(design.coef_names[direction.index])
        else:
            names.append(f"{block_terms[direction.block]} ({direction})")
    return ", ".join(dict.fromkeys(names))


def _fit_session(
    design: Design,
    rows: pd.DataFrame,
    lik: Likelihood,
    identity: _Identity,
    response: str,
    unit: str,
) -> FitRecord:
    """One session's fit and record (the steps of the module docstring)."""
    session = _Session(design, lik, rows, response, unit, "/".join(identity.key))
    first = True
    while True:
        why = session.judge()
        if why:
            return _record(session, identity, why)
        refit = session.fit(first)
        first = False
        if not session.apply(refit):
            return _record(session, identity, session.no_fit, refit)


def _record(
    session: _Session,
    identity: _Identity,
    no_fit: str,
    refit: SessionRefit | None = None,
) -> FitRecord:
    """The session's :class:`.record.FitRecord`: of its last fit, kept or not,
    or, with no ``refit``, of a session for which no fit was tried."""
    design = session.design
    kept = session.kept
    p = len(design.coef_names)
    beta, vcov = np.full(p, np.nan), np.full((p, p), np.nan)  # no fit tried
    collinearity, edf = {}, {}
    status, decrement, n_iter = NOT_FITTED, float("nan"), 0
    n_events = _n_events(session.lik, session.y[kept], session.unit[kept])
    if refit is not None:
        data, block_terms = session.fitted_with
        last = refit.refits[-1]
        if not refit.fitted:
            last = last.without_fit(len(block_terms))
        status = str(last.fit.status)
        decrement, n_iter = float(last.fit.newton_decrement), int(last.fit.n_iter)
        beta, vcov = last.fit.beta, last.covariance
        if refit.fitted:
            collinearity = term_collinearity(
                session.lik,
                data.X,
                data.y,
                data.aux,
                beta,
                design.term_columns,
                intercept=design.intercept_column,
            )
            edf = _edf(design, last, block_terms, np.isfinite(beta))
    animal, session_date, epoch = identity.key
    return FitRecord(
        model=identity.model,
        kind=identity.kind,
        animal=animal,
        session_date=session_date,
        epoch=epoch,
        likelihood=session.lik.name,
        n_rows=int(kept.sum()),
        n_units=int(np.unique(session.unit[kept]).size),
        n_events=n_events,
        cohort_counts=session.cohort_counts(),
        coef_names=list(design.coef_names),
        coef=beta.tolist(),
        vcov=vcov.tolist(),
        log_penalty=design.log_penalty_record(),
        edf=edf,
        status=status,
        newton_decrement=decrement,
        n_iter=n_iter,
        left_out=session.left_out(),
        left_out_round=session.left_out_round(),
        support={c: s.record() for c, s in session.recorded_support().items()},
        collinearity=collinearity,
        basis=design.basis_record(),
        standardization=design.standardization_record(),
        table_sha256=dict(identity.table_sha256),
        no_fit=no_fit,
    )


def fit_sessions(
    design: Design,
    sessions: Mapping[SessionKey, pd.DataFrame],
    lik: Likelihood,
    *,
    model: str,
    kind: str,
    response: str,
    unit: str,
    table_sha256: Mapping[str, str],
) -> list[FitRecord]:
    """Every session's fit and record, in the order given (module docstring).

    Parameters
    ----------
    design : Design
        Built on the pooled rows of every session given.
    sessions : mapping of (animal, session_date, epoch) to pd.DataFrame
        Each session's rows: the columns the design reads, ``response`` and
        ``unit``.
    lik : Likelihood
    model, kind : str
        Recorded.
    response, unit : str
        The response column and the unit column (under the grouped softmax,
        the group id).
    table_sha256 : mapping of str to str
        The tables the rows were read from -> their hashes; recorded.

    Returns
    -------
    list of FitRecord
        One per session.

    Raises
    ------
    ValueError
        If the design has an intercept under the grouped softmax, or an
        offset under another likelihood than the Poisson; if a session has a
        row whose unit id is missing (before any design is built, naming the
        session, the unit column and the rows); if a session's first
        fit finds its design not identified or fails for a numerical reason,
        naming the session.
    """
    if design.intercept_column is not None and isinstance(lik, GroupedSoftmax):
        raise ValueError("the grouped softmax cannot see an intercept")
    if design.offset is not None and not isinstance(lik, Poisson):
        raise ValueError("an Offset term needs the Poisson likelihood")
    for key, rows in sessions.items():
        n_missing = int(rows[unit].isna().sum())
        if n_missing:
            raise ValueError(
                f"session {'/'.join(key)}: {n_missing} rows with a missing {unit} "
                "(a unit id is never missing in a whole table)"
            )
    return [
        _fit_session(
            design, rows, lik, _Identity(key, model, kind, table_sha256), response, unit
        )
        for key, rows in sessions.items()
    ]
