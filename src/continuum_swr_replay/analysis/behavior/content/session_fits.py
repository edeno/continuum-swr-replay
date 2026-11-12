"""Fit the content rungs on every session of a cohort.

A content rung is fit per session. Which sessions are fit, by which estimator
at which grain, and what a session's row carries are decided here once, so a
runner and any check of its fits produce the same rows:

- :func:`fit_model_per_session` fits one rung of the content ladder;
- :func:`fit_contrast_ladder` fits the is_swr-contrast ladder, per rung and
  population, and books each one's cohort audit.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

from .._fit_diagnostics import fit_flag_counts
from ..cohort_audit import cohort_record, drop_profile, estimator_skip_note
from ..contrast_aggregation import LOG_EVENT_LENGTH, prepare_session
from ..fitting import (
    fit_conditional_logit,
    fit_conditional_poisson,
    fit_null_model_conditional_logit,
)
from ..tables.table_contracts import CV_RESULT_COLUMNS
from .cohort import CONTRAST_POPULATIONS, contrast_fit_cohort
from .dirichlet_multinomial import (
    DEFAULT_MIN_EVENTS_SESSION_PHI,
    ESTIMATOR_SKIP_CLASSES,
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


def resolve_concentration_cols(
    frame: pd.DataFrame, *, with_event_length: bool
) -> tuple[str, ...]:
    """Event-level columns entering ``log phi`` for one population's rows.

    The Dirichlet-multinomial likelihood uses an event's length ``L_e`` only as
    the multinomial index: ``log phi_e`` carries no length term, so the fitted
    concentration is assumed to be the same for a 25-bin event and a 90-bin one.
    SWR events run about 1.7x longer (median 45 vs 26 bins) than non-SWR events, so the shipped
    ``log_phi_is_swr`` is the TOTAL concentration contrast -- whatever an SWR
    event does to within-event redundancy, length included. Adding
    ``log_n_nonlocal_bins_z`` to the concentration design gives the
    length-adjusted twin, exactly parallel to the total/direct distinction the
    ``nt`` rungs already draw on the mean structure.

    The length term is added only where the cohort identifies ``is_swr`` at all:
    inside an SWR (or non-SWR) split there is no concentration contrast to
    decompose, and the fits there must keep the same design as before.

    Parameters
    ----------
    frame : pd.DataFrame
        The rows about to be fitted.
    with_event_length : bool
        Whether to append the z-scored log event length.

    Returns
    -------
    tuple of str

    Raises
    ------
    ValueError
        When the length term is requested and the cohort does not carry it. A
        silently dropped term would publish a ``_dm_len`` artifact identical to
        the unadjusted one under a name that claims otherwise.
    """
    base = concentration_cols_for_cohort(frame)
    if not with_event_length or base != ("is_swr",):
        return base
    column = f"{LOG_EVENT_LENGTH}_z"
    if column not in frame.columns:
        raise ValueError(
            f"--concentration-length needs {column!r} on the cohort; the "
            "assembled arm and patch frames standardize it in "
            "contrast_aggregation.assemble_arm / assemble_patch"
        )
    return (*base, column)


def _contrast_session_audit(prep: dict, r: dict | None, *, reason: str = "") -> dict:
    """Close one session's audit record around whatever the fitter returned.

    Any survivor not modeled -- the fitter's ``n < 2 * n_params`` gate, a failed
    fit, or a design the estimator refused -- is a session/fit-level exclusion,
    which is what makes ``n_in - Σdrops == n_modeled`` reconcile exactly.
    """
    n_modeled = int(r["n_events"]) if r is not None else 0
    if r is None:
        reason = reason or prep["reason"] or "fit_none"
    else:
        reason = ""
    return {
        "n_in": prep["n_in"],
        "n_drop_qvals_gap": prep["n_qvals"],
        "n_drop_own_predictor_missing": prep["n_own"],
        "n_drop_zero_response": prep["n_zero"],
        "n_excluded_session": prep["n_surv"] - n_modeled,
        "n_modeled": n_modeled,
        "reason": reason,
        "dropped": prep["dropped"],
    }


def fit_contrast_session(
    g: pd.DataFrame, predictors: list[str], *, response_col: str, n_categories: int
) -> tuple[dict | None, dict]:
    """Fit one session's conditional Poisson and return ``(row_or_None, audit)``.

    The session's rows go through the contrast ladder's event-level cohort rules
    (:func:`~continuum_swr_replay.analysis.behavior.contrast_aggregation.prepare_session`)
    first; ``audit`` books where its events went.
    """
    surv, prep = prepare_session(g, predictors, response_col=response_col)
    if surv is None:
        return None, _contrast_session_audit(prep, None)
    r = fit_conditional_poisson(
        surv,
        predictors=predictors,
        response_col=response_col,
        event_col="event_id",
        n_categories=n_categories,
    )
    return r, _contrast_session_audit(prep, r)


def _fit_contrast_population(
    pop_frame: pd.DataFrame,
    predictors: list[str],
    *,
    estimator: str,
    response_col: str,
    n_categories: int,
    n_jobs: int,
    min_events_session_phi: int,
    concentration_length: bool = False,
) -> list[tuple[tuple, dict | None, dict]]:
    """Fit every session of one (rung, population), in groupby order.

    Returns ``(session_key, row_or_None, audit)`` per session. Both estimators
    see exactly the same prepared rows; only the fit differs. The
    Dirichlet-multinomial fits are batched so they can share a process pool and
    a pooled concentration intercept for the short sessions, and a session whose
    within-event design it refuses becomes an audited skip
    (``reason="dm_design:..."``) rather than an exception out of the ladder.
    """
    session_columns = list(SESSION_COLUMNS)
    if estimator != "dirichlet_multinomial":
        return [
            (
                tuple(keys),
                *fit_contrast_session(
                    g,
                    predictors,
                    response_col=response_col,
                    n_categories=n_categories,
                ),
            )
            for keys, g in pop_frame.groupby(session_columns, sort=False)
        ]

    prepared: list[tuple[tuple, pd.DataFrame | None, dict]] = [
        (tuple(keys), *prepare_session(g, predictors, response_col=response_col))
        for keys, g in pop_frame.groupby(session_columns, sort=False)
    ]
    animal_position = session_columns.index("animal")
    fits, skips = fit_sessions_dirichlet_multinomial(
        [
            (key, key[animal_position], surv)
            for key, surv, _prep in prepared
            if surv is not None
        ],
        predictors,
        min_events=min_events_session_phi,
        n_jobs=n_jobs,
        response_col=response_col,
        event_col="event_id",
        concentration_cols=resolve_concentration_cols(
            pop_frame, with_event_length=concentration_length
        ),
        n_categories=n_categories,
    )
    rows = dict(fits)
    reasons = dict(skips)
    return [
        (
            key,
            rows.get(key),
            _contrast_session_audit(prep, rows.get(key), reason=reasons.get(key, "")),
        )
        for key, _surv, prep in prepared
    ]


def _contrast_fit_row(
    rung: str,
    population: str,
    keys: tuple,
    r: dict,
    *,
    response_col: str,
    renormalized: bool,
) -> dict:
    """The row one session's contrast fit contributes to the ladder's table."""
    return {
        "model": rung,
        "population": population,
        **dict(zip(SESSION_COLUMNS, keys, strict=True)),
        # ``sd_<term>`` travels with the coefficients because the
        # aggregation's identifiability backstop is read on the
        # effect scale (|coef| x sd). Without it a runaway fit is
        # judged by its raw coefficient, which drops identified
        # sessions whose predictor has a small SD and keeps
        # unidentified ones whose predictor has a large one.
        **{
            k: v
            for k, v in r.items()
            if k.startswith(("coef_", "bse_", "pval_", "sd_"))
        },
        # Per-term separation flags, so a fit whose self-history
        # coefficient ran away is not dropped for every term.
        "separated_terms": r.get("separated_terms"),
        "n_events": r["n_events"],
        # Events on each side of the is_swr concentration
        # modifier. A modifier resting on one or two SWR events
        # is finite, unflagged, and invisible to every
        # identifiability rule; only these counts expose it.
        # NaN under the conditional Poisson and in the within-SWR
        # splits, where there is no such modifier.
        "n_events_swr": r.get("n_events_swr", np.nan),
        "n_events_nonswr": r.get("n_events_nonswr", np.nan),
        "n_params": r["n_params"],
        # Which allocation the row was fit on, and therefore at
        # which grain its llf/AIC/SEs are stated.
        "estimator": r.get("estimator"),
        "response_col": response_col,
        # True where the trunk rows were dropped and the
        # remaining fractions rescaled to unit mass per event.
        "response_renormalized": renormalized,
        # Whether the row's llf carries the response's
        # normalizing constants, so a later comparison cannot
        # difference a full likelihood against a conditional one.
        "llf_includes_constants": r.get("llf_includes_constants"),
        # Where this fit's concentration intercept came from:
        # its own likelihood, or the animal-pooled median that a
        # session too short to estimate it is pinned to.
        "phi_source": r.get("phi_source"),
        # fit-diagnostics flags (additive; drive the aggregation's
        # session-level nonconverged/separated counts).
        "converged": r.get("converged"),
        "separated": r.get("separated"),
        "n_convergence_warnings": r.get("n_convergence_warnings"),
        # Whether the fit reached its optimum, for the bundle's
        # gates: the conditional-Poisson fitter's Newton finish
        # (NaN for an estimator without one). Terms the session
        # cannot inform are listed, "|"-joined, with NaN
        # coefficients.
        "llf": r.get("llf", np.nan),
        "newton_decrement": r.get("newton_decrement", np.nan),
        "max_abs_grad": r.get("max_abs_grad", np.nan),
        "n_newton_steps": r.get("n_newton_steps", np.nan),
        "unidentified_terms": r.get("unidentified_terms", ""),
    }


#: The event counts a contrast cohort audit sums over a population's sessions.
_CONTRAST_AUDIT_COUNTS = (
    "n_in",
    "n_drop_qvals_gap",
    "n_drop_own_predictor_missing",
    "n_drop_zero_response",
    "n_excluded_session",
    "n_modeled",
)


def fit_contrast_ladder(
    cohort: pd.DataFrame,
    ladder: dict,
    *,
    response_col: str,
    response: str,
    n_categories: int,
    estimator: str = "conditional_poisson",
    n_jobs: int = 1,
    min_events_session_phi: int = DEFAULT_MIN_EVENTS_SESSION_PHI,
    concentration_length: bool = False,
) -> tuple[pd.DataFrame, list[dict]]:
    """Fit the is_swr-contrast ladder per session, auditing each rung x population.

    A ``"pooled"`` rung is fit on every event; a ``"split"`` rung also on the
    SWR and the non-SWR events alone. Each (rung, population) is fit on the
    rows of
    :func:`~continuum_swr_replay.analysis.behavior.content.cohort.contrast_fit_cohort`,
    every session of it through the event-level cohort rules
    (:func:`~continuum_swr_replay.analysis.behavior.contrast_aggregation.prepare_session`).

    Parameters
    ----------
    cohort : pd.DataFrame
        The assembled arm or patch cohort, one row per event x candidate.
    ladder : dict
        Rung name -> ``(predictors, scope)``, scope ``"pooled"`` or ``"split"``.
    response_col : str
        The resolved response column.
    response : {"fraction", "hard_count"}
        The response the column came from.
    n_categories : int
        Candidates per event in ``cohort``: 9 arms or 3 patches.
    estimator : {"conditional_poisson", "dirichlet_multinomial"}, optional
        The per-session estimator.
    n_jobs : int, optional
        Worker processes for the Dirichlet-multinomial session fits.
    min_events_session_phi : int, optional
        Sessions below this many modelled events have their concentration
        intercept pinned to the animal-pooled median.
    concentration_length : bool, optional
        Dirichlet-multinomial only: add the z-scored log event length to the
        concentration design where the population identifies ``is_swr``
        (:func:`resolve_concentration_cols`).

    Returns
    -------
    results : pd.DataFrame
        One row per (rung, population, session) fitted.
    cohort_records : list of dict
        One :func:`~continuum_swr_replay.analysis.behavior.cohort_audit.cohort_record`
        per (rung, population), at the event grain:
        ``n_in - Σdrops == n_modeled``. A session that produced no fit is
        counted under ``n_excluded_session``; a refusal by the estimator itself
        names its session in ``drop_notes``, the routine skips are summarized
        by class. The flagged-fit counts are
        :func:`~continuum_swr_replay.analysis.behavior._fit_diagnostics.fit_flag_counts`
        of the population's fits.
    """
    rows = []
    audits = []
    animal_position = list(SESSION_COLUMNS).index("animal")
    for name, (preds, scope) in ladder.items():
        pops = ["all"] if scope == "pooled" else list(CONTRAST_POPULATIONS)
        for pop in pops:
            acc = dict.fromkeys(_CONTRAST_AUDIT_COUNTS, 0)
            sessions_in = 0
            pop_rows = []
            animals_modeled: set = set()
            session_skips: list[tuple[str, str]] = []
            dropped_frames = []
            fit_cohort = contrast_fit_cohort(
                cohort,
                name,
                preds,
                pop,
                response_col=response_col,
                response=response,
                n_categories=n_categories,
            )
            for keys, r, a in _fit_contrast_population(
                fit_cohort.frame,
                fit_cohort.predictors,
                estimator=estimator,
                response_col=response_col,
                n_categories=fit_cohort.n_categories,
                n_jobs=n_jobs,
                min_events_session_phi=min_events_session_phi,
                concentration_length=concentration_length,
            ):
                sessions_in += 1
                for k in acc:
                    acc[k] += a[k]
                if len(a["dropped"]):
                    dropped_frames.append(a["dropped"])
                if r is None:
                    if a["reason"]:
                        session_skips.append(("/".join(map(str, keys)), a["reason"]))
                    continue
                animals_modeled.add(keys[animal_position])
                pop_rows.append(
                    _contrast_fit_row(
                        name,
                        pop,
                        keys,
                        r,
                        response_col=response_col,
                        renormalized=fit_cohort.response_renormalized,
                    )
                )
            rows.extend(pop_rows)
            dropped = (
                pd.concat(dropped_frames, ignore_index=True)
                if dropped_frames
                else pd.DataFrame()
            )
            notes = drop_profile(dropped)
            # Sessions that produced no fit are already counted under
            # n_excluded_session; naming why keeps a design the estimator
            # refused distinguishable from a session that was merely too short.
            skip_note = estimator_skip_note(session_skips, ESTIMATOR_SKIP_CLASSES)
            if skip_note:
                notes = "; ".join(filter(None, [notes, skip_note]))
            audits.append(
                cohort_record(
                    name,
                    pop,
                    "event",
                    n_in=acc["n_in"],
                    n_modeled=acc["n_modeled"],
                    sessions_in=sessions_in,
                    sessions_modeled=len(pop_rows),
                    animals_modeled=len(animals_modeled),
                    drop_notes=notes,
                    n_drop_qvals_gap=acc["n_drop_qvals_gap"],
                    n_drop_own_predictor_missing=acc["n_drop_own_predictor_missing"],
                    n_drop_zero_response=acc["n_drop_zero_response"],
                    n_excluded_session=acc["n_excluded_session"],
                    **fit_flag_counts(pd.DataFrame(pop_rows)),
                )
            )
    return pd.DataFrame(rows), audits
