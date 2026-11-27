"""The fit-row record: one row per (model, kind, unit of the fit) of every fit.

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

from continuum_swr_replay.tables.session_keys import unit_key

from .reml import RefitEnd

__all__ = ["FitRecord", "NoFit"]


class NoFit(StrEnum):
    """Why a session has no kept fit (a record's ``no_fit``).

    ``NO_ROWS`` and ``NO_EVENTS`` are the loop's; the others are the ways
    :func:`.reml.refit_session` ends without a fit, its :class:`.reml.RefitEnd`
    but ``CONVERGED``, of the same names and values.
    """

    #: No rows left to fit: none given, all missing, or all dropped.
    NO_ROWS = "no_rows"
    #: No events of the kind in the rows left to fit.
    NO_EVENTS = "no_events"
    ALONG_THE_INTERCEPT = RefitEnd.ALONG_THE_INTERCEPT.value
    ALONG_A_PENALTY = RefitEnd.ALONG_A_PENALTY.value
    TOO_MANY_ROUNDS = RefitEnd.TOO_MANY_ROUNDS.value
    OTHER_STATUS = RefitEnd.OTHER_STATUS.value


@dataclass(frozen=True)
class FitRecord:
    """One fit of one unit: a session, a recording day or an animal.

    Attributes
    ----------
    model, kind : str
        The model's name and the event kind it is fitted for.
    animal, session_date, epoch : str
        The unit's key (``fit_by``).
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
        ``events_without_response`` (events given with no response: their
        shares over all the rows given sum to zero, a missing share counted
        as zero) and ``events_emptied_by_dropped_rows`` (events given with a
        response, all of it on rows removed before the fit: rows with a
        missing value or rows of a dropped level); a caller adds its own
        cohort counts.
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
        level's coefficient name). A day copy of a per-day level
        (:mod:`.terms`, "Day copies") is such a level: it has its own entry,
        by its copy's name, and the category's entry sums the copies.
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
    fit_by : str
        The unit of the fit: ``"session"``, ``"day"`` or ``"animal"``. At
        ``"day"`` the record's ``epoch`` is ``""``; at ``"animal"`` its
        ``session_date`` and ``epoch`` are ``""``.
    sessions : list of dict
        One entry per session in the fit, in the order of its rows:
        ``animal``, ``session_date``, ``epoch``; ``n_rows``, ``n_units`` and
        ``n_events`` the fit saw of it; ``rows_missing_values``,
        ``rows_dropped`` (a dropped level, baseline copy or day copy -> its
        rows in the session, where above zero) and, under the grouped
        softmax, ``events_emptied_by_dropped_rows`` and
        ``events_without_response``; and ``baseline``, the coefficient names
        of the session's baseline copies (empty in a fit of one session,
        where nothing is copied; a day copy is not a baseline and is not
        listed: its name carries its day). A copy left out, its support and
        its edf are in ``left_out``, ``support`` and ``edf`` by its name.
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
    fit_by: str
    sessions: list

    @property
    def unit_key(self) -> tuple:
        """The key of the fit's unit: ``(animal, session_date, epoch)``,
        ``(animal, session_date)`` or ``(animal,)`` by ``fit_by``."""
        return unit_key((self.animal, self.session_date, self.epoch), self.fit_by)
