"""The per-session fitting loop: one design, every session fitted alone.

:func:`fit_sessions` takes a :class:`.terms.Design`, each session's rows, a
likelihood and the names of the response and unit columns, and returns one
:class:`.record.FitRecord` per session. It knows nothing of the task. For each
session, in order:

1. The rows with a missing value the design reads are dropped and counted.
2. The design is built (:meth:`.terms.Design.session`, with the design check),
   and the support rule is applied to every unpenalized column
   (:func:`.support.design_support`). A column that fails is made inactive; a
   failing level of a category drops its rows and takes the terms by that
   level with it (:func:`.support.level_rows_dropped`). Under the grouped
   softmax an event whose response was all on dropped rows is dropped too.
3. The session is fitted at the design's fixed log penalties by
   :func:`.reml.refit_session`, which makes the columns running away inactive
   while the fit has no finite maximum along unpenalized columns only. A level
   of a category made inactive there drops its rows as the support rule does,
   and the session is fitted again with every column left out so far
   inactive. At the session's first fit a design not identified, or a
   numerical failure, is an error that names the session; in a refit, a
   direction through a penalty block or any other status leaves the session
   without a fit, with the reason in words.
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
from .reml import PenaltyBlock, RemlScore, SessionData, refit_session
from .support import design_support, level_rows_dropped
from .terms import INTERCEPT, Design

__all__ = ["SessionKey", "fit_session", "fit_sessions"]

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


def _prepare(
    design: Design,
    lik: Likelihood,
    rows: pd.DataFrame,
    response: str,
    unit: str,
    keep: np.ndarray,
    counts: dict,
) -> _Prepared:
    """The rows kept, ordered for the likelihood, with their design; softmax
    events with no response left are dropped and counted."""
    rows = rows[keep].reset_index(drop=True)
    if isinstance(lik, GroupedSoftmax):
        rows = rows.sort_values(unit, kind="stable").reset_index(drop=True)
        total = rows.groupby(unit, sort=False)[response].transform("sum").to_numpy()
        empty = total <= 0
        counts["events_without_response"] = counts.get(
            "events_without_response", 0
        ) + int(rows.loc[empty, unit].nunique())
        rows = rows[~empty].reset_index(drop=True)
    X, offset = design.session(rows)
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
    return _Prepared(rows, X, y, aux, units)


def _session_blocks(
    design: Design, inactive: np.ndarray
) -> tuple[list[PenaltyBlock], np.ndarray, list[str]]:
    """The design's blocks with at least one active column, their log
    penalties and term names (a block whose term left with a dropped level
    has every column inactive)."""
    kept = [
        k for k, block in enumerate(design.blocks) if not inactive[block.columns].all()
    ]
    for k in kept:
        if inactive[design.blocks[k].columns].any():
            raise ValueError(f"{design.block_terms[k]}: part of a smooth is inactive")
    return (
        [design.blocks[k] for k in kept],
        design.log_penalties[kept],
        [design.block_terms[k] for k in kept],
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
    prepared = _prepare(design, lik, rows, response, unit, ~missing, counts)
    has_event = _has_event(lik, prepared.y)
    support = design_support(
        design, prepared.rows, prepared.X, prepared.unit, has_event
    )
    left_out = {c: "support" for c, s in support.items() if not s.passes}
    counts["rows_dropped"] = {}
    applied: set[str] = set()
    round_offset, first, reason = 0, True, ""
    while True:
        levels = [c for c in left_out if c not in applied and _drops_rows(design, c)]
        if levels:
            keep, dropped, with_them = level_rows_dropped(design, prepared.rows, levels)
            applied.update(levels)
            counts["rows_dropped"].update(dropped)
            for column, level in with_them.items():
                left_out.setdefault(column, f"rows of {level} dropped")
            prepared = _prepare(
                design, lik, prepared.rows, response, unit, keep, counts
            )
        inactive = np.array([c in left_out for c in design.coef_names])
        blocks, rho, block_terms = _session_blocks(design, inactive)
        data = SessionData(prepared.X, prepared.y, prepared.aux, name, ~inactive)
        refit = refit_session(lik, data, blocks, rho)
        start = refit.refits[0].fit.status
        if first and start not in (CONVERGED, NO_FINITE_MAXIMUM):
            what = (
                "its design is not identified (mark the columns it cannot estimate "
                "inactive)"
                if start == NOT_IDENTIFIED
                else f"its fit ended as {start}, a numerical failure"
            )
            raise ValueError(f"session {name}: at its first fit {what}")
        first = False
        for column, round_ in zip(refit.left_out, refit.rounds, strict=True):
            left_out[design.coef_names[column]] = (
                f"no finite maximum (round {round_offset + round_})"
            )
        round_offset += len(refit.refits) - 1
        if refit.left_out.size or not refit.fitted:
            reason = refit.reason
        pending = [c for c in left_out if c not in applied and _drops_rows(design, c)]
        if not (refit.fitted and pending):
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
        support=support,
        table_sha256=table_sha256,
        reason=reason,
    )


def _drops_rows(design: Design, coefficient: str) -> bool:
    """True for a level of a category (its rows leave with it)."""
    return design.columns[design.coef_names.index(coefficient)].drops_rows


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
            intercept=0 if design.intercept else None,
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
        coef_names=design.coef_names,
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
    if INTERCEPT in design.coef_names and isinstance(lik, GroupedSoftmax):
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
