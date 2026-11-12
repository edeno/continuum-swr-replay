"""Fit one content rung on every session of a cohort.

A content rung is fit per session. Which sessions are fit, by which estimator
at which grain, and what a session's row carries are decided here once, so the
content runner and any check of its fits produce the same rows.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

from ..fitting import (
    fit_conditional_logit,
    fit_conditional_poisson,
    fit_null_model_conditional_logit,
)
from ..tables.table_contracts import CV_RESULT_COLUMNS
from .dirichlet_multinomial import (
    DEFAULT_MIN_EVENTS_SESSION_PHI,
    concentration_cols_for_cohort,
    fit_sessions_dirichlet_multinomial,
)
from .model_checking import cross_validate_conditional_poisson

logger = logging.getLogger(__name__)


def _fit_dirichlet_multinomial_per_session(
    long_df: pd.DataFrame,
    model_name: str,
    predictors: list[str],
    *,
    response_col: str,
    event_col: str,
    n_categories: int,
    min_rows: int,
    n_jobs: int,
    min_events_session_phi: int,
) -> pd.DataFrame:
    """Fit the Dirichlet-multinomial rung for every session of ``long_df``.

    The per-session fits run in a process pool with a pooled fallback for the
    short sessions; see
    :func:`~continuum_swr_replay.analysis.behavior.content.dirichlet_multinomial.fit_sessions_dirichlet_multinomial`.
    Sessions the fitter refuses -- a rank-deficient within-event design raises,
    a session with too few events for its parameters returns nothing -- are
    recorded on the returned frame's ``attrs["estimator_skips"]`` so the cohort
    audit can name them, rather than crashing the ladder.
    """
    animal_position = SESSION_COLUMNS.index("animal")
    sessions = [
        (tuple(keys), keys[animal_position], group)
        for keys, group in long_df.groupby(list(SESSION_COLUMNS))
        if len(group) >= min_rows
    ]
    fits, skips = fit_sessions_dirichlet_multinomial(
        sessions,
        list(predictors),
        min_events=min_events_session_phi,
        n_jobs=n_jobs,
        response_col=response_col,
        event_col=event_col,
        concentration_cols=concentration_cols_for_cohort(long_df),
        n_categories=n_categories,
    )
    results = []
    for keys, result in fits:
        result["model"] = model_name
        result["response_col"] = response_col
        for col, val in zip(SESSION_COLUMNS, keys, strict=True):
            result[col] = val
        results.append(result)
    for keys, reason in skips:
        logger.warning(
            "%s: no Dirichlet-multinomial fit for session %s (%s)",
            model_name,
            "/".join(map(str, keys)),
            reason,
        )
    frame = pd.DataFrame(results)
    frame.attrs["estimator_skips"] = [
        {"session": "/".join(map(str, keys)), "reason": reason}
        for keys, reason in skips
    ]
    return frame


def fit_model_per_session(
    long_df: pd.DataFrame,
    model_name: str,
    predictors: list[str],
    run_cv: bool = False,
    response_col: str = "arm_count_hard",
    event_col: str = "event_id",
    n_categories: int = 9,
    grain: str = "timebin",
    estimator: str = "conditional_poisson",
    n_jobs: int = 1,
    min_events_session_phi: int = DEFAULT_MIN_EVENTS_SESSION_PHI,
) -> pd.DataFrame:
    """Fit one content rung across all sessions.

    A session with fewer than ``2 * n_categories`` rows (two events) is not
    fit. A session the fitter returns nothing for is left out of the frame.

    Parameters
    ----------
    long_df : pd.DataFrame
        Canonical event-arm or event-patch table.
    model_name : str
        Model identifier (e.g., "M0").
    predictors : list[str]
        Predictor columns for this model.
    run_cv : bool
        If True, also run event-based cross-validation per session (timebin
        grain only). A session whose cross-validation fails keeps its fit, with
        the cross-validation columns NaN.
    response_col : str
        Column name for the response: the per-event allocation fraction
        (``arm_fraction`` / ``patch_fraction``) or the timebin counts
        (``arm_count_hard`` / ``patch_count_hard``).
    event_col : str
        Column name for the event grouping key.
    n_categories : int
        Candidates per event: 9 arms, 6 leaf arms, or 3 patches.
    grain : {"timebin", "event"}
        ``"timebin"`` fits the conditional Poisson on the response;
        ``"event"`` collapses each event to the candidate that won and fits
        the conditional logit (the uniform null when there are no predictors).
    estimator : str
        ``"conditional_poisson"`` (the shipped softmax fit, whose grain is set
        by ``grain``) or ``"dirichlet_multinomial"``, which adds an explicit
        within-event concentration and is fit on the counts only.
    n_jobs : int
        Worker processes for the Dirichlet-multinomial per-session fits.
    min_events_session_phi : int
        Sessions with fewer modelled events than this have their concentration
        intercept pinned to the animal-pooled value.

    Returns
    -------
    pd.DataFrame
        One row per session with coefficients and fit statistics, plus the
        ``estimator`` the fitter used and the ``response_col`` it was fit on --
        the two facts that say at which grain the row's ``llf`` / ``aic`` /
        ``bse_`` are stated, so a fraction and a count fit cannot be pooled by
        a later reader that sees only coefficients. Under the
        Dirichlet-multinomial, ``attrs["estimator_skips"]`` lists the sessions
        the estimator refused, each with its reason.
    """
    results = []

    # Minimum 2 events per session, scaled by candidate-set size:
    # n_categories=9 (arm) → 18 rows, n_categories=3 (patch) → 6 rows.
    # The previous hardcoded 18 silently skipped valid 2–5 event patch
    # sessions before fit_conditional_poisson could apply its own
    # n_events < 2 * len(predictors) gate.
    min_rows = n_categories * 2

    if estimator == "dirichlet_multinomial":
        return _fit_dirichlet_multinomial_per_session(
            long_df,
            model_name,
            predictors,
            response_col=response_col,
            event_col=event_col,
            n_categories=n_categories,
            min_rows=min_rows,
            n_jobs=n_jobs,
            min_events_session_phi=min_events_session_phi,
        )

    for keys, group in long_df.groupby(list(SESSION_COLUMNS)):
        if len(group) < min_rows:
            continue

        if grain == "event" and not predictors:
            r = fit_null_model_conditional_logit(
                group,
                event_col=event_col,
                response_col=response_col,
                n_arms=n_categories,
            )
        elif grain == "event":
            # One observation per event -- which arm won -- instead of one per
            # timebin. The timebin grain treats a median 32 bins per event as
            # independent multinomial draws when they are heavily
            # autocorrelated, so its standard errors are optimistic by roughly
            # 5x; collapsing makes n honest at the event level, at the cost of
            # discarding the within-event distribution (so "SWR spreads mass
            # over more arms" is not askable here).
            r = fit_conditional_logit(
                group,
                predictors=list(predictors),
                response_col=response_col,
                event_col=event_col,
            )
        else:
            r = fit_conditional_poisson(
                group,
                predictors=list(predictors),
                response_col=response_col,
                event_col=event_col,
                n_categories=n_categories,
            )
        if r is None:
            continue

        if run_cv and grain != "event":
            cv = cross_validate_conditional_poisson(
                group,
                predictors=list(predictors),
                event_col=event_col,
                response_col=response_col,
                n_categories=n_categories,
            )
            # A failed CV must not cost the session its primary coefficient. CV
            # is a diagnostic on the fit, not a precondition for it, and letting
            # it gate the cohort means the reported effects depend on whether
            # cross-validation was requested -- measured at 213 sessions without
            # --cv against 208 with it. The CV columns are left NaN instead, so
            # held-out comparisons run on the sessions that have scores (and can
            # report that count) while the coefficients keep the full cohort.
            if cv is not None:
                r.update(cv)
            else:
                # Keep the CV schema stable whether or not a session scored, so
                # downstream checks can distinguish "this session has no score"
                # from "CV never ran at all" instead of tripping over an absent
                # column.
                r.update(dict.fromkeys(CV_RESULT_COLUMNS, np.nan))

        r["model"] = model_name
        r["response_col"] = response_col
        for col, val in zip(SESSION_COLUMNS, keys, strict=True):
            r[col] = val

        results.append(r)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)
