"""The declared log penalty of a smooth term: the one that gives a target edf.

A per-session smooth term is fitted at one fixed log penalty, declared in the
model and the same in every session; no penalty is chosen from the data for a
per-session fit. :func:`log_penalty_for_edf` finds that value once, on the
model's sessions: the log penalty at which the median over sessions of the
term's effective degrees of freedom (edf) equals a target, every other
penalty held at its declared value. The caller writes the value into the
model's term as a constant.

Each session is fitted by :func:`.sessions.fit_sessions`, the loop the model
itself uses (its support rule, its rows and its refits), and the edf is the
one its records hold. The median is over the sessions with a kept fit in which
the term is fitted, so the set of sessions can change with the penalty: a
session whose fit is not kept at one penalty drops out of the median there.
It is searched by bisection on the log penalty inside ``LOG_PENALTY_BOUNDS``,
assuming that the median edf falls as the penalty rises; the search is
deterministic. A bound is fitted only when the bracket ends there, to tell
whether the target is out of reach. The median at the returned penalty must
be the target to within ``CALIBRATION_EDF_TOL``: a larger gap means the
median jumped past the target (a session entered or left it), and the
calibration raises rather than return that penalty.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import NamedTuple

import numpy as np
import pandas as pd

from .constants import (
    CALIBRATION_EDF_TOL,
    CALIBRATION_LOG_PENALTY_TOL,
    LOG_PENALTY_BOUNDS,
)
from .likelihoods import Likelihood
from .newton import CONVERGED
from .sessions import SessionKey, fit_sessions
from .terms import Design

__all__ = ["EdfCalibration", "log_penalty_for_edf"]


class EdfCalibration(NamedTuple):
    """The log penalty that gives the target median edf.

    Attributes
    ----------
    log_penalty : float
        The middle of the bisection's last bracket, narrower than
        ``CALIBRATION_LOG_PENALTY_TOL``.
    median_edf : float
        The median over sessions of the term's edf at ``log_penalty``, within
        ``CALIBRATION_EDF_TOL`` of the target.
    n_sessions_used : int
        The sessions the median is over: those with a kept fit in which the
        term is fitted at ``log_penalty``.
    edf : pd.Series, shape (n_sessions,)
        Each session's edf of the term at ``log_penalty``, indexed by the
        session keys in the order given; NaN for a session without a kept
        fit or in which the term is not fitted.
    """

    log_penalty: float
    median_edf: float
    n_sessions_used: int
    edf: pd.Series


def _term_edf(
    design: Design,
    sessions: Mapping[SessionKey, pd.DataFrame],
    lik: Likelihood,
    term: str,
    response: str,
    unit: str,
) -> pd.Series:
    """Each session's edf of ``term`` at the design's log penalties; NaN where
    the fit is not kept or the term is not fitted."""
    names = [design.coef_names[j] for j in design.by_name[term].columns]
    records = fit_sessions(
        design,
        sessions,
        lik,
        model="calibration",
        kind="calibration",
        response=response,
        unit=unit,
        table_sha256={},
    )
    edf = [
        r.edf[term]
        if r.status == CONVERGED and not any(n in r.left_out for n in names)
        else np.nan
        for r in records
    ]
    return pd.Series(edf, index=pd.Index(list(sessions)), dtype=float, name=term)


def log_penalty_for_edf(
    design: Design,
    sessions: Mapping[SessionKey, pd.DataFrame],
    lik: Likelihood,
    *,
    term: str,
    target_edf: float,
    response: str,
    unit: str,
) -> EdfCalibration:
    """The log penalty of ``term`` at which its median edf over sessions is
    ``target_edf`` (module docstring).

    Parameters
    ----------
    design : Design
        The model's design, built on the pooled rows of ``sessions``. Its
        other penalties are held at their declared values; the value
        declared for ``term`` is not used.
    sessions : mapping of (animal, session_date, epoch) to pd.DataFrame
        Each session's rows, as :func:`.sessions.fit_sessions` takes them.
    lik : Likelihood
    term : str
        The name of a term with a fixed penalty (a smooth, or a lag
        functional), as :class:`.terms.DesignTerm` names it.
    target_edf : float
        The term's edf counts its penalty's null space (for a constrained
        P-spline, the line).
    response, unit : str
        As :func:`.sessions.fit_sessions` takes them.

    Returns
    -------
    EdfCalibration

    Raises
    ------
    ValueError
        If ``term`` is not a term of the design or has no fixed penalty; if
        no session has a kept fit with the term at a log penalty tried; if
        the target is not between the median edf at the upper bound and at
        the lower bound of ``LOG_PENALTY_BOUNDS`` (it cannot be reached
        inside them); if the median edf at the returned log penalty differs
        from the target by more than ``CALIBRATION_EDF_TOL`` (the median
        jumped past it); if a session's first fit fails
        (:func:`.sessions.fit_sessions`).
    """
    if term not in design.by_name:
        raise ValueError(f"{term!r} is not a term of the design")
    block = design.by_name[term].block
    if block is None:
        raise ValueError(f"{term!r} has no fixed penalty to calibrate")

    def at(log_penalty: float) -> pd.Series:
        # The fit reads Design.log_penalties; this design's log_penalty_record
        # still names the declared value, but its records are not kept.
        log_penalties = design.log_penalties.copy()
        log_penalties[block] = log_penalty
        edf = _term_edf(
            replace(design, log_penalties=log_penalties),
            sessions,
            lik,
            term,
            response,
            unit,
        )
        if edf.isna().all():
            raise ValueError(
                f"no session has a kept fit with {term!r} at log penalty {log_penalty}"
            )
        return edf

    lo, hi = LOG_PENALTY_BOUNDS
    while hi - lo > CALIBRATION_LOG_PENALTY_TOL:
        middle = 0.5 * (lo + hi)
        if np.nanmedian(at(middle)) > target_edf:
            lo = middle
        else:
            hi = middle
    # A bracket left at a bound may not hold the target; the bound is fitted
    # only then, since at the lower bound a smooth of many columns can need
    # more than NEWTON_MAX_STEPS steps.
    bound_lo, bound_hi = LOG_PENALTY_BOUNDS
    for bound, sign in ((bound_lo, 1.0), (bound_hi, -1.0)):
        if bound in (lo, hi):
            edf = float(np.nanmedian(at(bound)))
            if sign * (edf - target_edf) < 0:
                raise ValueError(
                    f"a median edf of {target_edf} for {term!r} cannot be reached "
                    f"inside the log penalty bounds [{bound_lo}, {bound_hi}]: the "
                    f"median edf at {bound} is {edf:.4g}"
                )
    log_penalty = 0.5 * (lo + hi)
    edf = at(log_penalty)
    median_edf = float(np.nanmedian(edf))
    n_used = int(edf.notna().sum())
    if abs(median_edf - target_edf) > CALIBRATION_EDF_TOL:
        raise ValueError(
            f"the median edf of {term!r} at log penalty {log_penalty:.6g} is "
            f"{median_edf:.4g}, over {n_used} of {edf.size} sessions; it differs "
            f"from the target {target_edf} by more than {CALIBRATION_EDF_TOL}: the "
            "median jumps past the target, as when a session's fit is kept on "
            "one side of the penalty and not on the other"
        )
    return EdfCalibration(log_penalty, median_edf, n_used, edf)
