"""The fit-row record: one row per (model, kind, session) of every stored fit.

:class:`FitRecord` holds what a fit's readers need to use it and to judge it:
its identity, what the likelihood saw, the coefficients with their Bayesian
covariance ``(H + S)^-1``, how each penalty was set, the effective degrees of
freedom, the convergence measure, the columns left out and why, the support
rule's counts, the collinearity measures, the bases and the standardizations,
and the hashes of the tables it read.

A record whose ``status`` is not "converged", whose support fails for a declared
term, or whose declared term is over the collinearity threshold is kept; its
readers flag it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .reml import RefitEnd

__all__ = ["FitRecord", "NoFit"]


NoFit = StrEnum(
    "NoFit",
    {
        "NO_ROWS": "no_rows",
        "NO_EVENTS": "no_events",
        **{end.name: end.value for end in RefitEnd if end is not RefitEnd.CONVERGED},
    },
    module=__name__,
)
NoFit.__doc__ = """Why a session has no kept fit (a record's ``no_fit``).

``NO_ROWS``: no rows left to fit (none given, all missing, or all dropped).
``NO_EVENTS``: no events of the kind in the rows left to fit. The others are
the ways :func:`.reml.refit_session` ends without a fit, its
:class:`.reml.RefitEnd` but ``CONVERGED``, of the same names and values:
``ALONG_THE_INTERCEPT``, ``ALONG_A_PENALTY``, ``TOO_MANY_ROUNDS`` and
``OTHER_STATUS``.
"""


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
        Term -> ``{"set": "fixed" or "none", "log_penalty": value or None}``,
        for every term (:meth:`.terms.Design.log_penalty_record`).
    edf : dict
        Term -> effective degrees of freedom (an unpenalized term: its active
        columns; a term left out: 0); a level of a penalized category also by
        its coefficient name. A penalized simple coefficient's is between 0
        (the penalty decides) and 1 (the data decide). Empty for a session
        without a fit. For a penalized category, the category's entry is the
        sum of its levels' entries, so a total over all entries counts those
        levels twice: sum the model terms only (the entries that are not a
        level's coefficient name).
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
        fit.
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
