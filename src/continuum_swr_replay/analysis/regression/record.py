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
from pathlib import Path

import pandas as pd

from continuum_swr_replay.tables.io import write_atomically

__all__ = ["JSON_FIELDS", "FitRecord", "read_records", "records_frame", "write_records"]


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
        The events it saw: the count total (Poisson, Bernoulli), the events
        (grouped softmax), the rows (Gaussian).
    cohort_counts : dict
        Rows left out before the fit, by reason: ``rows_missing_values``,
        ``rows_dropped`` (level coefficient -> its rows, the support rule's
        row rule), ``events_without_response`` (softmax groups whose response
        was all on dropped rows); a caller adds its own cohort counts.
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
        The last fit's ``NewtonFit.status``; only "converged" is a fit.
    newton_decrement : float
    n_iter : int
        The last fit's Newton steps.
    left_out : dict
        Coefficient name -> why it is inactive: ``support``, ``rows of <level>
        dropped`` or ``no finite maximum (round k)``.
    support : dict
        Coefficient name -> ``{"n_low", "n_high", "passes"}``.
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
    fit_reason : str
        How the fit ended, in words, when columns were left out for a fit
        with no finite maximum or the session has no fit; else empty.
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
    support: dict
    collinearity: dict
    basis: dict
    standardization: dict
    table_sha256: dict
    fit_reason: str


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
    """Write ``records`` to a parquet table at ``path``, atomically.

    Returns
    -------
    Path
        The file written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = records_frame(records)
    write_atomically(path, lambda temporary: frame.to_parquet(temporary, index=False))
    return path


def read_records(path: str | Path) -> list[FitRecord]:
    """The records of a table written by :func:`write_records`."""
    frame = pd.read_parquet(path)
    names = [f.name for f in fields(FitRecord)]
    if list(frame.columns) != names:
        raise ValueError(f"{path}: columns {list(frame.columns)}, not {names}")
    out = []
    for row in frame.to_dict(orient="records"):
        for name in JSON_FIELDS:
            row[name] = json.loads(row[name])
        out.append(FitRecord(**row))
    return out
