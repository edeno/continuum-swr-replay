"""The fit-row record: one row per (model, kind, session) of every stored fit.

:class:`FitRecord` holds what a fit's readers need to use it and to judge it:
its identity, what the likelihood saw, the coefficients with their Bayesian
covariance ``(H + S)^-1``, how each penalty was set, the effective degrees of
freedom, the convergence measure, the columns left out and why, the support
rule's counts, the collinearity measures, the bases and the standardizations,
and the hashes of the tables it read. :func:`write_records` writes rows as a
parquet table, the structured fields as JSON strings (:data:`JSON_FIELDS`);
:func:`read_records` reads them back.

A record whose ``status`` is not "converged", whose support fails for a declared
term, or whose declared term is over the collinearity threshold is kept; its
readers flag it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from pathlib import Path

import pandas as pd

from continuum_swr_replay.tables.io import read_table, resolved_table_path, write_table

from .constants import REFIT_MAX_ROUNDS
from .reml import ALONG_THE_INTERCEPT
from .support import MIN_UNITS, Removed
from .terms import SMALLEST_COLUMN_SIZE

__all__ = [
    "JSON_FIELDS",
    "FitRecord",
    "NoFit",
    "describe",
    "read_records",
    "records_frame",
    "write_records",
]


class NoFit(StrEnum):
    """Why a session has no kept fit (a record's ``no_fit``)."""

    #: No rows left to fit: none given, all missing, or all dropped.
    NO_ROWS = "no_rows"
    #: No events of the kind in the rows left to fit.
    NO_EVENTS = "no_events"
    #: The refit's fit has no finite maximum along the intercept.
    ALONG_THE_INTERCEPT = "along_the_intercept"
    #: The refit's fit has no finite maximum along a penalty block.
    ALONG_A_PENALTY = "along_a_penalty"
    #: Still no finite maximum after ``REFIT_MAX_ROUNDS`` rounds.
    TOO_MANY_ROUNDS = "too_many_rounds"
    #: The refit's fit ended with another status than converged or no finite
    #: maximum.
    OTHER_STATUS = "other_status"


_WORDS = {
    Removed.SUPPORT: f"fewer than MIN_UNITS = {MIN_UNITS} units with an event",
    Removed.TOO_SMALL: (
        f"largest absolute value below SMALLEST_COLUMN_SIZE = "
        f"{SMALLEST_COLUMN_SIZE:g} in the session"
    ),
    Removed.WITH_LEVEL: "left with its level, whose rows were dropped",
    NoFit.NO_ROWS: "no rows left to fit",
    NoFit.NO_EVENTS: "no events of the kind in the rows left to fit",
    NoFit.ALONG_THE_INTERCEPT: ALONG_THE_INTERCEPT,
    NoFit.ALONG_A_PENALTY: (
        "no finite maximum along a penalty block, whose columns cannot be made inactive"
    ),
    NoFit.TOO_MANY_ROUNDS: (
        f"no finite maximum after REFIT_MAX_ROUNDS = {REFIT_MAX_ROUNDS} rounds of "
        "columns made inactive"
    ),
    NoFit.OTHER_STATUS: "the fit at the log penalties ended with another status",
}


def describe(code: str, round_: int | None = None) -> str:
    """A cause of a record (``Removed`` or ``NoFit``) in words, for a person.

    Parameters
    ----------
    code : str
        A ``left_out`` value, a support entry's ``reason`` or ``no_fit``.
    round_ : int or None
        The round of a term left out for no finite maximum
        (``left_out_round``).
    """
    if code == Removed.NO_FINITE_MAXIMUM:
        return f"no finite maximum (round {round_})"
    return _WORDS[code]


@dataclass(frozen=True)
class FitRecord:
    """One fit of one session.

    Attributes
    ----------
    model, kind : str
        The model's name and the event kind it is fitted for.
    animal, session_date, epoch : str
        The session.
    likelihood : str
        ``poisson``, ``bernoulli``, ``gaussian`` or ``grouped_softmax``.
    n_rows, n_units : int
        The rows the fit saw, and its units (visits, events or units).
    n_events : float
        The events in the rows it saw, a fit kept or not: the count total
        (Poisson, Bernoulli), the events (grouped softmax), the rows
        (Gaussian).
    cohort_counts : dict
        Rows left out before the fit, by reason: ``rows_missing_values``,
        ``rows_dropped`` (level coefficient -> its rows, the support rule's
        row rule; a row shared by levels of two categories that fail in the
        same pass is counted under the first in column order), and under the
        grouped softmax
        ``events_without_response`` (events given with no response) and
        ``events_emptied_by_dropped_rows`` (events whose response was all on
        dropped rows); a caller adds its own cohort counts.
    coef_names : list of str
    coef : list of float
        NaN at the columns left out, and everywhere for a session without a
        fit.
    vcov : list of list of float
        ``(H + S)^-1``; NaN rows and columns as ``coef``.
    log_penalty : dict
        Smooth term -> ``{"set": "fixed" or "none", "log_penalty": value or
        None}``.
    edf : dict
        Term -> effective degrees of freedom (an unpenalized term: its active
        columns; a term left out: 0); empty for a session without a fit.
    status : str
        The last fit's ``NewtonFit.status``; only "converged" is a kept fit.
        "not fitted" when no fit was tried: no rows, or no events, left to
        fit.
    newton_decrement : float
    n_iter : int
        The last fit's Newton steps.
    left_out : dict
        Coefficient name -> why it is inactive, a ``Removed`` code.
    left_out_round : dict
        Coefficient name -> the refit's round that left it out, for the
        coefficients left out for no finite maximum.
    support : dict
        Term name -> ``{"n_low", "n_high", "passes", "reason", "size"}``
        (:class:`.support.Support`; ``reason`` a ``Removed`` code or empty).
    collinearity : dict
        Term -> ``{"r2", "vif"}`` or ``{"observed", "worst", "estimate"}``;
        empty for a session without a fit.
    basis : dict
        Smooth term -> ``{"kind", "knots", "degree", "range", "constraint",
        "clamp"}``.
    standardization : dict
        Covariate -> ``{"sd", "ddof", "reference"}``.
    table_sha256 : dict
        The tables the fit read -> their hashes.
    no_fit : str
        Why the session has no kept fit, a ``NoFit`` code; empty for a kept
        fit. :func:`describe` gives the words of every code.
    """

    model: str
    kind: str
    animal: str
    session_date: str
    epoch: str
    likelihood: str
    n_rows: int
    n_units: int
    n_events: float
    cohort_counts: dict
    coef_names: list
    coef: list
    vcov: list
    log_penalty: dict
    edf: dict
    status: str
    newton_decrement: float
    n_iter: int
    left_out: dict
    left_out_round: dict
    support: dict
    collinearity: dict
    basis: dict
    standardization: dict
    table_sha256: dict
    no_fit: str


#: The fields stored as JSON strings.
JSON_FIELDS = tuple(f.name for f in fields(FitRecord) if f.type in ("dict", "list"))


def records_frame(records: Sequence[FitRecord]) -> pd.DataFrame:
    """One row per record, the :data:`JSON_FIELDS` as JSON strings."""
    rows = []
    for record in records:
        row = asdict(record)
        for name in JSON_FIELDS:
            row[name] = json.dumps(row[name])
        rows.append(row)
    return pd.DataFrame(rows, columns=[f.name for f in fields(FitRecord)])


def write_records(records: Sequence[FitRecord], path: str | Path) -> Path:
    """Write ``records`` to a parquet table at ``path``, atomically
    (:func:`continuum_swr_replay.tables.io.write_table`).

    Any suffix of ``path`` is replaced with ``.parquet``.

    Returns
    -------
    Path
        The file written.
    """
    return write_table(records_frame(records), path)


def read_records(path: str | Path) -> list[FitRecord]:
    """The records of a table written by :func:`write_records`.

    Raises
    ------
    FileNotFoundError
        If there is no file at ``path`` (with the suffix ``.parquet``),
        naming it.
    """
    parquet = resolved_table_path(path)
    if not parquet.exists():
        raise FileNotFoundError(f"no fit records at {parquet}")
    frame = read_table(parquet)
    names = [f.name for f in fields(FitRecord)]
    if list(frame.columns) != names:
        raise ValueError(f"{path}: columns {list(frame.columns)}, not {names}")
    out = []
    for row in frame.to_dict(orient="records"):
        for name in JSON_FIELDS:
            row[name] = json.loads(row[name])
        out.append(FitRecord(**row))
    return out
