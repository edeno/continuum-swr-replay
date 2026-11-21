"""The per-session fitting loop: one design, every session fitted alone.

:func:`fit_sessions` takes a :class:`.terms.Design`, each session's rows, a
likelihood and the names of the response and unit columns, and returns one
:class:`.record.FitRecord` per session. It knows nothing of the task. For each
session, in order:

1. The rows with a missing value the design reads are dropped and counted.
2. The design is built (:meth:`.terms.Design.session`) and the support rule is
   applied to every unpenalized column (:func:`.support.design_support`). A
   column that fails is made inactive; a failing level of a category drops
   its rows and takes the terms by that level with it
   (:func:`.support.level_rows_dropped`). Under the grouped softmax an event
   whose response was all on dropped rows is dropped too. The levels are
   decided first and only accumulate: their support is counted again on the
   rows kept, and each thin level's rows dropped, until no level fails. Every
   other term is then judged once, on the final rows, and that alone decides
   whether it is active. The record holds those counts, and for a dropped
   level the counts that made it fail. A session with no rows, or no events, left is not
   fitted: its record says why and the loop goes on.
3. The session is fitted at the design's fixed log penalties by
   :func:`.reml.refit_session`, which makes the columns running away inactive
   while the fit has no finite maximum along unpenalized columns only. A level
   of a category made inactive there drops its rows as the support rule does,
   and the session is fitted again with every column left out so far
   inactive. At the session's first fit a design not identified, or a
   numerical failure, is an error that names the session; in a refit, a
   direction through a penalty block or any other status leaves the session
   without a fit, with the reason in words. When the refit ends because its
   fit has no finite maximum along the intercept
   (:data:`.reml.ALONG_THE_INTERCEPT`), nothing is dropped or left out: the
   session gets the not-fitted record with that reason.
4. The collinearity of each term is measured at the fit
   (:func:`.collinearity.term_collinearity`), and the record is built.

A session with columns left out keeps its other terms: those columns'
coefficients and covariance are NaN and ``left_out`` says why. A term by a
level whose rows were dropped leaves the session with its penalty block. The
units are the ``unit`` column (a visit for a rate model; for the grouped
softmax, each unit is one event, a group of rows). A row holds an event when its
response is above zero (Poisson, Bernoulli), and every row does under the
softmax and the Gaussian.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .collinearity import term_collinearity
from .likelihoods import Bernoulli, Gaussian, GroupedSoftmax, Likelihood, Poisson
from .newton import CONVERGED, NO_FINITE_MAXIMUM, NOT_IDENTIFIED
from .record import FitRecord
from .reml import (
    ALONG_THE_INTERCEPT,
    PenaltyBlock,
    RemlScore,
    SessionData,
    refit_session,
)
from .support import design_support, level_rows_dropped
from .terms import Design

__all__ = ["NOT_FITTED", "SessionKey", "fit_session", "fit_sessions"]

SessionKey = tuple[str, str, str]

_NAMES = {
    Poisson: "poisson",
    Bernoulli: "bernoulli",
    Gaussian: "gaussian",
    GroupedSoftmax: "grouped_softmax",
}


@dataclass(frozen=True)
class _Prepared:
    """A session's rows, design and response after the row rules."""

    rows: pd.DataFrame
    X: np.ndarray
    y: np.ndarray
    aux: object
    unit: np.ndarray
    level_rows: dict


def _prepare(
    design: Design,
    lik: Likelihood,
    rows: pd.DataFrame,
    response: str,
    unit: str,
    keep: np.ndarray | None,
    counts: dict,
    counted_as: str,
) -> _Prepared:
    """The rows kept (all when ``keep`` is None), ordered for the likelihood,
    with their design; softmax events with no response left are dropped and
    counted under ``counted_as``."""
    rows = (rows if keep is None else rows[keep]).reset_index(drop=True)
    if isinstance(lik, GroupedSoftmax):
        rows = rows.sort_values(unit, kind="stable").reset_index(drop=True)
        total = rows.groupby(unit, sort=False)[response].transform("sum").to_numpy()
        empty = total <= 0
        counts[counted_as] = counts.get(counted_as, 0) + int(
            rows.loc[empty, unit].nunique()
        )
        rows = rows[~empty].reset_index(drop=True)
        if rows.empty:  # nothing left to fit: the caller records why
            return _Prepared(
                rows,
                np.empty((0, len(design.coef_names))),
                np.empty(0),
                None,
                np.empty(0),
                {},
            )
    built = design.build(rows)
    X, offset = built.X, built.offset
    y = rows[response].to_numpy(dtype=float)
    units = rows[unit].to_numpy()
    if isinstance(lik, GroupedSoftmax):
        aux = np.unique(units, return_inverse=True)[1]
    elif isinstance(lik, Poisson):
        aux = np.zeros(len(rows)) if offset is None else offset
    else:
        if offset is not None:
            raise ValueError("an Offset term needs the Poisson likelihood")
        aux = None
    return _Prepared(rows, X, y, aux, units, built.level_rows)


def _session_blocks(
    design: Design, inactive: np.ndarray
) -> tuple[list[PenaltyBlock], np.ndarray, list[str]]:
    """The design's blocks with at least one active column, their log
    penalties and term names (a block whose term left with a dropped level
    has every column inactive)."""
    names = {t.block: t.name for t in design.terms if t.block is not None}
    kept = [
        k for k, block in enumerate(design.blocks) if not inactive[block.columns].all()
    ]
    for k in kept:
        if inactive[design.blocks[k].columns].any():
            raise ValueError(f"{names[k]}: part of a smooth is inactive")
    return (
        [design.blocks[k] for k in kept],
        design.log_penalties[kept],
        [names[k] for k in kept],
    )


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


def fit_session(
    design: Design,
    rows: pd.DataFrame,
    lik: Likelihood,
    *,
    key: SessionKey,
    model: str,
    kind: str,
    response: str,
    unit: str,
    table_sha256: Mapping[str, str],
) -> FitRecord:
    """One session's fit and record (the steps of the module docstring).

    Parameters
    ----------
    design : Design
    rows : pd.DataFrame
        The session's rows: the columns the design reads, ``response`` and
        ``unit``.
    lik : Likelihood
    key : (str, str, str)
        ``(animal, session_date, epoch)``.
    model, kind : str
        Recorded.
    response, unit : str
        The response column and the unit column.
    table_sha256 : mapping of str to str
        The tables the rows were read from -> their hashes; recorded.

    Returns
    -------
    FitRecord

    Raises
    ------
    ValueError
        If the session's first fit finds its design not identified or fails
        for a numerical reason, naming the session; or a check of the design
        fails.
    """
    name = "/".join(key)
    counts: dict = {}
    rows = rows.reset_index(drop=True)
    missing = design.missing_rows(rows)
    counts["rows_missing_values"] = int(missing.sum())
    identity = {"key": key, "model": model, "kind": kind, "table_sha256": table_sha256}
    rows = rows[~missing].reset_index(drop=True)
    levels_out: dict[str, str] = {}  # levels left out, with the reason
    level_counts: dict = {}  # a level left out: its counts when it was
    with_them: dict[str, str] = {}  # columns of terms by a dropped level
    fit_out: dict[str, str] = {}  # other columns the refit left out
    support: dict = {}
    counts["rows_dropped"] = {}
    if isinstance(lik, GroupedSoftmax):
        counts["events_emptied_by_dropped_rows"] = 0
    round_offset, first, reason = 0, True, ""

    def current_left_out() -> dict[str, str]:
        out = {**levels_out, **with_them, **fit_out}
        for entry, counted in support.items():
            if not counted.passes:
                for column in _columns_of(design, entry):
                    out.setdefault(column, "support")
        return out

    def recorded_support() -> dict:
        return {**support, **level_counts}

    while True:
        # The levels first: count their support on the rows kept, drop the
        # rows of each thin level, until no level fails. The other terms are
        # then judged once, on the final rows (``support`` of the last round).
        while True:
            empty = _no_data(lik, rows, response)
            if not empty:
                # Events with no response as given; then those emptied by
                # dropped rows.
                counted_as = (
                    "events_emptied_by_dropped_rows"
                    if "events_without_response" in counts
                    else "events_without_response"
                )
                prepared = _prepare(
                    design, lik, rows, response, unit, None, counts, counted_as
                )
                rows = prepared.rows
                empty = _no_data(lik, rows, response)
            if empty:
                return _not_fitted(
                    design,
                    lik,
                    rows,
                    unit,
                    counts,
                    empty,
                    support=recorded_support(),
                    left_out=current_left_out(),
                    **identity,
                )
            has_event = _has_event(lik, prepared.y)
            support = design_support(
                design, prepared.X, prepared.level_rows, prepared.unit, has_event
            )
            for column, counted in support.items():
                if (
                    design.by_name[column].kind == "level"
                    and not counted.passes
                    and column not in levels_out
                ):
                    levels_out[column] = "support"
                    level_counts[column] = counted
            levels = [c for c in levels_out if c not in counts["rows_dropped"]]
            if not levels:
                break
            keep, dropped, by_level = level_rows_dropped(
                design, prepared.level_rows, levels
            )
            counts["rows_dropped"].update(dropped)
            for column, level in by_level.items():
                with_them.setdefault(column, f"rows of {level} dropped")
            rows = rows[keep].reset_index(drop=True)
        left_out = current_left_out()
        inactive = np.array([c in left_out for c in design.coef_names])
        blocks, rho, block_terms = _session_blocks(design, inactive)
        data = SessionData(prepared.X, prepared.y, prepared.aux, name, ~inactive)
        refit = refit_session(lik, data, blocks, rho)
        start = refit.refits[0].fit.status
        if first and start not in (CONVERGED, NO_FINITE_MAXIMUM):
            what = (
                "its design is not identified, in the coefficients "
                + _named(design, blocks, block_terms, refit.refits[0])
                + " (a category's reference level must have rows in every session)"
                if start == NOT_IDENTIFIED
                else f"its fit ended as {start}, a numerical failure"
            )
            raise ValueError(f"session {name}: at its first fit {what}")
        first = False
        if refit.reason == ALONG_THE_INTERCEPT:
            return _not_fitted(
                design,
                lik,
                rows,
                unit,
                counts,
                refit.reason,
                support=recorded_support(),
                left_out=left_out,
                **identity,
            )
        pending = False
        for column, round_ in zip(refit.left_out, refit.rounds, strict=True):
            coefficient = design.coef_names[column]
            why = f"no finite maximum (round {round_offset + round_})"
            if design.column_terms[column].kind == "level":
                levels_out[coefficient] = why
                level_counts[coefficient] = support[coefficient]
                pending = True
            else:
                fit_out[coefficient] = why
        round_offset += len(refit.refits) - 1
        if refit.left_out.size or not refit.fitted:
            reason = refit.reason
        if not (refit.fitted and pending):
            left_out = current_left_out()
            break
    return _record(
        design,
        lik,
        prepared,
        refit.refits[-1],
        refit.fitted,
        block_terms,
        key=key,
        model=model,
        kind=kind,
        counts=counts,
        left_out=left_out,
        support=recorded_support(),
        table_sha256=table_sha256,
        reason=reason,
    )


#: The status of a session left without a fit before any fit: no rows, or no
#: events of the kind, are left.
NOT_FITTED = "not fitted"


def _no_data(lik: Likelihood, rows: pd.DataFrame, response: str) -> str:
    """Why rows cannot be fitted, in words; empty when they can."""
    if rows.empty:
        return "no rows left to fit"
    if isinstance(lik, Poisson | Bernoulli) and not (rows[response] > 0).any():
        return "no events of the kind in the rows left to fit"
    return ""


def _not_fitted(
    design: Design,
    lik: Likelihood,
    rows: pd.DataFrame,
    unit: str,
    counts: dict,
    reason: str,
    *,
    key: SessionKey,
    model: str,
    kind: str,
    table_sha256: Mapping[str, str],
    support: Mapping[str, object] | None = None,
    left_out: Mapping[str, str] | None = None,
) -> FitRecord:
    """The record of a session left without a fit before any fit."""
    p = len(design.coef_names)
    animal, session_date, epoch = key
    return FitRecord(
        model=model,
        kind=kind,
        animal=animal,
        session_date=session_date,
        epoch=epoch,
        likelihood=_NAMES[type(lik)],
        n_rows=len(rows),
        n_units=int(rows[unit].nunique()) if len(rows) else 0,
        n_events=0.0,
        cohort_counts=counts,
        coef_names=list(design.coef_names),
        coef=[float("nan")] * p,
        vcov=np.full((p, p), np.nan).tolist(),
        log_penalty=design.log_penalty_record(),
        edf={},
        status=NOT_FITTED,
        newton_decrement=float("nan"),
        n_iter=0,
        left_out=dict(left_out or {}),
        support={c: s._asdict() for c, s in (support or {}).items()},
        collinearity={},
        basis=design.basis_record(),
        standardization=design.standardization_record(),
        table_sha256=dict(table_sha256),
        fit_reason=reason,
    )


def _named(
    design: Design,
    blocks: list[PenaltyBlock],
    block_terms: list[str],
    score: RemlScore,
) -> str:
    """The coefficients a design not identified involves, by name: a design
    column by its coefficient, a smooth's coordinate by its term."""
    names = []
    for direction in score.unidentified_directions:
        if direction.block is None:
            names.append(design.coef_names[direction.index])
        else:
            names.append(f"{block_terms[direction.block]} ({direction})")
    return ", ".join(dict.fromkeys(names))


def _columns_of(design: Design, name: str) -> list[str]:
    """The coefficients of a term."""
    return [design.coef_names[j] for j in design.by_name[name].columns]


def _has_event(lik: Likelihood, y: np.ndarray) -> np.ndarray:
    """True for a row that holds an event (module docstring)."""
    if isinstance(lik, Poisson | Bernoulli):
        return y > 0
    return np.ones(y.size, dtype=bool)


def _record(
    design: Design,
    lik: Likelihood,
    prepared: _Prepared,
    last: RemlScore,
    fitted: bool,
    block_terms: list[str],
    *,
    key: SessionKey,
    model: str,
    kind: str,
    counts: dict,
    left_out: dict[str, str],
    support: dict,
    table_sha256: Mapping[str, str],
    reason: str,
) -> FitRecord:
    """The session's :class:`.record.FitRecord`."""
    p = len(design.coef_names)
    beta = last.fit.beta if fitted else np.full(p, np.nan)
    vcov = last.covariance if fitted else np.full((p, p), np.nan)
    active = np.isfinite(beta)
    collinearity, edf = {}, {}
    if fitted:
        terms = design.term_columns
        collinearity = term_collinearity(
            lik,
            prepared.X,
            prepared.y,
            prepared.aux,
            beta,
            terms,
            intercept=design.intercept_column,
        )
        edf = _edf(design, last, block_terms, active)
    if isinstance(lik, Poisson | Bernoulli):
        n_events = float(prepared.y.sum())
    elif isinstance(lik, GroupedSoftmax):
        n_events = float(np.unique(prepared.unit).size)
    else:
        n_events = float(prepared.y.size)
    animal, session_date, epoch = key
    return FitRecord(
        model=model,
        kind=kind,
        animal=animal,
        session_date=session_date,
        epoch=epoch,
        likelihood=_NAMES[type(lik)],
        n_rows=int(prepared.y.size),
        n_units=int(np.unique(prepared.unit).size),
        n_events=n_events,
        cohort_counts=counts,
        coef_names=list(design.coef_names),
        coef=beta.tolist(),
        vcov=vcov.tolist(),
        log_penalty=design.log_penalty_record(),
        edf=edf,
        status=str(last.fit.status),
        newton_decrement=float(last.fit.newton_decrement),
        n_iter=int(last.fit.n_iter),
        left_out=left_out,
        support={c: s._asdict() for c, s in support.items()},
        collinearity=collinearity,
        basis=design.basis_record(),
        standardization=design.standardization_record(),
        table_sha256=dict(table_sha256),
        fit_reason=reason,
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
    """Every session's fit and record, in the order given (:func:`fit_session`).

    Parameters
    ----------
    design : Design
        Built on the pooled rows of every session given.
    sessions : mapping of (animal, session_date, epoch) to pd.DataFrame
    lik : Likelihood
    model, kind, response, unit, table_sha256
        As :func:`fit_session`.

    Returns
    -------
    list of FitRecord
        One per session.
    """
    if design.intercept_column is not None and isinstance(lik, GroupedSoftmax):
        raise ValueError("the grouped softmax cannot see an intercept")
    return [
        fit_session(
            design,
            rows,
            lik,
            key=key,
            model=model,
            kind=kind,
            response=response,
            unit=unit,
            table_sha256=table_sha256,
        )
        for key, rows in sessions.items()
    ]
