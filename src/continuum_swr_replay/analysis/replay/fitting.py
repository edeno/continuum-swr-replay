"""Per-session model fitting for the replay content model hierarchy.

Primary model: Conditional Poisson regression (multinomial-count softmax),
where the response is per-arm timebin counts and the model conditions on the
event. Its likelihood is the one statsmodels' ``ConditionalPoisson`` defines;
it is maximized here, by BFGS on columns scaled to unit within-event SD and a
Newton finish with the analytic Hessian (see :func:`fit_conditional_poisson`).

Robustness model: Conditional Logit on dominant arm (highest count).

The null model (M0) is computed analytically:
    ℓ₀ = -Σ_e L_e · log|A_e|
where L_e is the total timebin count and |A_e| is the number of arms
with nonzero availability (currently always 9).
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import minimize
from scipy.stats import norm
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.stats.outliers_influence import variance_inflation_factor

from ._fit_diagnostics import (
    NUISANCE_TERMS,
    capture_fit_warnings,
    count_fit_quality_warnings,
    fit_status,
    separated_terms,
)

logger = logging.getLogger(__name__)


# Iteration budget for the conditional fits.
#
# statsmodels defaults to BFGS with maxiter=100, which is not enough for the
# larger rungs: M6 carries 21 predictors whose column SDs span a 159x range (the
# recency spline basis is the small end), and badly-scaled columns make BFGS take
# many small steps. 43 of 213 M6 sessions stopped at the iteration cap rather
# than at an optimum -- 11.4 log-likelihood units short on the session examined
# -- and because a non-converged fold invalidates the whole CV metric, that also
# voided cross-validation for 72 sessions.
#
# It is not a design problem: that session's within-event design is full rank
# with condition number 12.3, and BFGS, LBFGS and Newton all reach the same
# optimum once allowed to run. Nor is it expensive -- the same fit takes 2.0s
# failing at 100 iterations and 2.1s converging at 1000.
CONDITIONAL_FIT_MAXITER = 1000

# The Newton finish of the conditional-Poisson fit.
#
# BFGS stops when the gradient of the per-row MEAN log-likelihood is below 1e-5.
# On the total log-likelihood that allows a gradient of 1e-5 x n_rows, and in a
# weakly informed direction a small gradient hides a large gap: the optimum
# lies g' H^-1 g / 2 higher (H the information). F3 measured gaps of up to 4.8
# log-likelihood units on the shipped content fits, every one flagged converged
# (``.claude/research/refresh_followup/f3_optimizer_report.md``). Newton steps
# use the exact curvature and close that gap in a few steps. The Newton
# decrement g' H^-1 g / 2 is the gain the next step predicts; the fit is at its
# optimum once it is below NEWTON_DECREMENT_TOL.
NEWTON_DECREMENT_TOL = 1e-8
NEWTON_MAX_STEPS = 20
# A Newton step is halved until the log-likelihood does not fall. 2**-30 of a
# step is below the log-likelihood's floating-point resolution, so a step that
# still lowers it there is abandoned and the fit stops where it is.
_MAX_STEP_HALVINGS = 30

# Smallest information, along any direction, a fitted design may have at its
# optimum. It is measured on the columns divided by their within-event SD, so
# one unit means the same for every column: below 1e-3 the SE along that
# direction exceeds 30 within-event SDs.
#
# Such a direction is a term the data cannot bound. F3 found 62 headline content
# fits with one. In 60 it is a nuisance term whose rows get all or none of each
# event's mass: ``is_candidate_arm_never_visited`` (1-28 rows per fit), or
# ``is_candidate_arm_0`` in a session where no event reached arm 0. Its
# coefficient runs to +/-inf and the information along it decays with it, so
# whether such a fit used to be kept depended on where BFGS stopped and on
# whether statsmodels' numerical Hessian then inverted (14 refused, 48 kept).
# With the Newton finish it no longer does. Along a term running off on an
# exponential tail, the decrement is lambda / (2 * Delta**2), with lambda the
# information and Delta the column's range in within-event SDs. A finish that
# reaches NEWTON_DECREMENT_TOL therefore leaves lambda below 2e-8 * Delta**2,
# which is under this threshold for any column whose range is under 220
# within-event SDs. On the real fits the two groups are four orders of
# magnitude apart: the 2,583 identified fits have at least 0.30, and Newton run
# on without this check takes all 62 to 3.1e-5 or below. The decision is the
# same whether Newton starts from zero, from BFGS or from the old fit.
#
# What happens next depends on how many columns the direction involves; see
# ``fit_conditional_poisson``. A direction carried by one column is a term the
# session cannot inform: the fit is kept at that term's limit and the term is
# reported as not estimated. A direction that mixes columns is refused.
MIN_SCALED_INFORMATION = 1e-3

# A column is named as carrying an uninformed direction when at least 1 % of its
# unit weight lies in that direction (a loading of 0.1 or more).
_MIN_NULL_DIRECTION_WEIGHT = 0.01

# A set of uninformed directions is carried by single columns when it is spanned
# by that many columns, each with at least this weight in it (a loading of 0.99
# or more). In 61 of F3's 62 fits the columns have weight 1.0000 to four places;
# in the other, the direction spreads 0.125 over each of eight arm dummies.
_SINGLE_COLUMN_WEIGHT = 0.98

# Response columns whose per-event values are an allocation fraction summing to
# 1.0 rather than a timebin count. The distinction is not cosmetic: with a
# fraction the event contributes a single draw, so the model information (and
# every AIC/BIC built on it) is at the event grain, and the analytic null
# collapses to -n_events * log(n_categories).
FRACTION_RESPONSE_COLUMNS = frozenset({"arm_fraction", "patch_fraction"})

CONDITIONAL_POISSON_COUNT = "conditional_poisson_count"
CONDITIONAL_POISSON_FRACTION = "conditional_poisson_fraction"
CONDITIONAL_LOGIT_DOMINANT = "conditional_logit_dominant"


def poisson_estimator_name(response_col: str) -> str:
    """Estimator label for a conditional-Poisson fit on ``response_col``.

    Fraction responses are named separately from counts because they change
    what one row of the results frame means -- a coefficient's SE, AIC and
    pseudo-R2 are per event rather than per timebin -- and downstream summaries
    must not pool the two scales. Any other response (soft counts, posterior
    mass) is on the count scale.
    """
    return (
        CONDITIONAL_POISSON_FRACTION
        if response_col in FRACTION_RESPONSE_COLUMNS
        else CONDITIONAL_POISSON_COUNT
    )


def event_clustered_covariance(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, params: np.ndarray
) -> np.ndarray:
    """Sandwich covariance of a conditional-Poisson fit, clustered by event.

    The conditional likelihood treats the ``L_e`` timebins of an event as
    independent multinomial draws. They are not: a replay is a smooth
    trajectory, so its bins are near-copies of each other and the model
    information overstates how much an event carries. The sandwich
    ``H^-1 (sum_e g_e g_e') H^-1`` replaces the assumed per-event information
    with the observed between-event variability of the score, which is what the
    coefficient's sampling variance actually reflects.

    Parameters
    ----------
    X : np.ndarray, shape (n_rows, n_params)
        Design matrix, one row per (event, candidate category).
    y : np.ndarray, shape (n_rows,)
        Response (timebin counts or per-event allocation fractions).
    groups : np.ndarray, shape (n_rows,)
        Event id per row; rows need not be sorted.
    params : np.ndarray, shape (n_params,)
        Fitted coefficients.

    Returns
    -------
    np.ndarray, shape (n_params, n_params)
        Event-clustered covariance of ``params``.
    """
    X, y, starts = _sorted_by_event(X, y, groups)
    _, scores, bread = _event_terms(X, y, starts, params)
    bread_inv = np.linalg.pinv(bread)
    return bread_inv @ (scores.T @ scores) @ bread_inv


def _sorted_by_event(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows sorted by event, keeping their order within an event.

    Returns
    -------
    X : np.ndarray, shape (n_rows, n_params)
    y : np.ndarray, shape (n_rows,)
    starts : np.ndarray, shape (n_events,)
        Index of each event's first row.
    """
    order = np.argsort(groups, kind="stable")
    _, starts = np.unique(groups[order], return_index=True)
    return X[order], y[order], starts


def _event_terms(
    X: np.ndarray,
    y: np.ndarray,
    starts: np.ndarray,
    params: np.ndarray,
    *,
    information: bool = True,
) -> tuple[float, np.ndarray, np.ndarray | None]:
    """Conditional-Poisson log-likelihood, per-event scores and information.

    For event ``e`` with response ``y_ea``, total ``n_e = sum_a y_ea`` and
    linear predictor ``eta_ea = x_ea' params``::

        l_e = sum_a y_ea eta_ea - n_e log sum_a exp(eta_ea)
        score_e = sum_a (y_ea - n_e p_ea) x_ea,  p_ea = softmax_a(eta_e)
        I_e = n_e (sum_a p_ea x_ea x_ea' - xbar_e xbar_e'),  xbar_e = sum_a p_ea x_ea

    ``I = sum_e I_e`` is the observed information (minus the Hessian; it does
    not depend on ``y`` beyond ``n_e``). It is the bread of
    :func:`event_clustered_covariance` and the curvature of the Newton finish.

    Parameters
    ----------
    X : np.ndarray, shape (n_rows, n_params)
        Design, rows sorted by event (see :func:`_sorted_by_event`).
    y : np.ndarray, shape (n_rows,)
        Response, same order.
    starts : np.ndarray, shape (n_events,)
        Index of each event's first row.
    params : np.ndarray, shape (n_params,)
        Coefficients.
    information : bool
        Compute the information too. BFGS needs only the log-likelihood and
        the gradient.

    Returns
    -------
    llf : float
    scores : np.ndarray, shape (n_events, n_params)
        Each event's score; their sum is the gradient.
    information : np.ndarray, shape (n_params, n_params), or None
    """
    sizes = np.diff(starts, append=X.shape[0])
    eta = X @ params
    top = np.maximum.reduceat(eta, starts)
    weight = np.exp(eta - np.repeat(top, sizes))
    weight_sum = np.add.reduceat(weight, starts)
    prob = weight / np.repeat(weight_sum, sizes)
    total = np.add.reduceat(y, starts)
    expected = np.repeat(total, sizes) * prob
    llf = float(y @ eta - total @ (top + np.log(weight_sum)))
    scores = np.add.reduceat((y - expected)[:, None] * X, starts)
    if not information:
        return llf, scores, None
    xbar = np.add.reduceat(prob[:, None] * X, starts)
    return llf, scores, (X.T * expected) @ X - (xbar.T * total) @ xbar


def _rows_of_events_whose_response_varies(
    y: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    """Row mask of the events whose response is not the same on all their rows.

    statsmodels' ``ConditionalPoisson``, which fit this model before, leaves an
    event out of the likelihood when ``np.std`` of its response is 0. The rule
    is kept, with the same test on the same values in the same order, so the
    likelihood and its optimum are unchanged.
    """
    order = np.argsort(groups, kind="stable")
    _, starts, inverse = np.unique(
        groups[order], return_index=True, return_inverse=True
    )
    varies = np.array([np.std(chunk) > 0 for chunk in np.split(y[order], starts[1:])])
    keep = np.empty(len(y), dtype=bool)
    keep[order] = varies[inverse]
    return keep


class _NewtonFinish(NamedTuple):
    """Where the Newton finish stopped, on the scaled columns."""

    params: np.ndarray
    llf: float
    gradient: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    decrement: float
    n_steps: int


def _bfgs_start(X: np.ndarray, y: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """BFGS from zero with statsmodels' settings: the old fit, on scaled columns.

    The objective is minus the mean log-likelihood per row and BFGS stops on
    ``gtol=1e-5`` in the sup norm, as ``ConditionalPoisson.fit`` ran it. Where
    it stops matters little: the Newton finish takes it from there.
    """
    n_rows = X.shape[0]

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        llf, scores, _ = _event_terms(X, y, starts, params, information=False)
        return -llf / n_rows, -scores.sum(axis=0) / n_rows

    return minimize(
        objective,
        np.zeros(X.shape[1]),
        jac=True,
        method="BFGS",
        options={"gtol": 1e-5, "norm": np.inf, "maxiter": CONDITIONAL_FIT_MAXITER},
    ).x


def _newton_finish(
    X: np.ndarray, y: np.ndarray, starts: np.ndarray, params: np.ndarray
) -> _NewtonFinish:
    """Newton steps with the analytic information until the decrement is small.

    Each step solves ``I d = g`` and is halved until the log-likelihood does not
    fall. It stops when the decrement ``g' I^-1 g / 2`` is below
    ``NEWTON_DECREMENT_TOL``, after ``NEWTON_MAX_STEPS`` steps, when no fraction
    of the step raises the log-likelihood, or at once when the information has a
    direction below ``MIN_SCALED_INFORMATION`` (the caller refuses that design;
    the decrement is then NaN).

    A Newton step and its decrement do not depend on how the columns are scaled:
    with ``beta_s = s * beta`` the step on the scaled columns maps onto the step
    on the original ones exactly. Taking it on the scaled columns is the same
    step, solved on a better-conditioned matrix.
    """
    llf, scores, information = _event_terms(X, y, starts, params)
    n_steps = 0
    while True:
        gradient = scores.sum(axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(information)
        if eigenvalues[0] < MIN_SCALED_INFORMATION:
            decrement = np.nan
            break
        step = eigenvectors @ ((eigenvectors.T @ gradient) / eigenvalues)
        decrement = float(gradient @ step) / 2
        if decrement < NEWTON_DECREMENT_TOL or n_steps >= NEWTON_MAX_STEPS:
            break
        for halving in range(_MAX_STEP_HALVINGS + 1):
            trial = params + step / 2**halving
            trial_llf, trial_scores, trial_information = _event_terms(
                X, y, starts, trial
            )
            if trial_llf >= llf:
                break
        else:
            break
        params, llf = trial, trial_llf
        scores, information = trial_scores, trial_information
        n_steps += 1
    return _NewtonFinish(
        params, llf, gradient, eigenvalues, eigenvectors, decrement, n_steps
    )


def _terms_on(directions: np.ndarray, names: list[str]) -> list[str]:
    """Names of the columns that carry weight on the given unit directions.

    ``directions`` has one orthonormal direction per column, shape
    ``(n_params, n_directions)``.
    """
    weight = (directions**2).sum(axis=1)
    return [
        name
        for name, w in zip(names, weight, strict=True)
        if w >= _MIN_NULL_DIRECTION_WEIGHT
    ]


def _within_group_centered(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """``X`` minus each group's mean of each column."""
    return X - pd.DataFrame(X).groupby(groups).transform("mean").to_numpy()


def _within_group_null_directions(X: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Directions :func:`within_group_full_rank` finds missing, shape (p, k).

    The right singular vectors of the within-group-centred design whose singular
    value is zero by ``numpy.linalg.matrix_rank``'s tolerance, so they are the
    directions that make the rank fall short.
    """
    centered = _within_group_centered(X, groups)
    _, singular, right = np.linalg.svd(centered, full_matrices=False)
    tol = singular.max() * max(centered.shape) * np.finfo(float).eps
    return right[singular <= tol].T


def _rank_deficient_terms(
    X: np.ndarray, groups: np.ndarray, names: list[str]
) -> list[str]:
    """Columns on the directions :func:`within_group_full_rank` found missing."""
    return _terms_on(_within_group_null_directions(X, groups), names)


def _single_columns(directions: np.ndarray) -> list[int] | None:
    """The columns that carry ``directions`` one each, or None if they mix columns.

    ``directions`` is an orthonormal basis, shape ``(n_params, k)``. It is
    carried by single columns when exactly ``k`` columns each have at least
    ``_SINGLE_COLUMN_WEIGHT`` of their unit weight in it. The test is on the
    whole set, not on each vector, because the basis of a repeated eigenvalue is
    any rotation of it.
    """
    weight = (directions**2).sum(axis=1)
    single = [int(k) for k in np.flatnonzero(weight >= _SINGLE_COLUMN_WEIGHT)]
    return single if len(single) == directions.shape[1] else None


def _rows_sent_to_zero_probability(
    x: np.ndarray, y: np.ndarray, groups: np.ndarray, in_likelihood: np.ndarray
) -> np.ndarray | None:
    """Rows that a column's coefficient running to -inf or +inf takes out.

    If no event in the likelihood has mass on a row where ``x`` is above its
    event minimum, the log-likelihood rises without bound as the coefficient
    goes to -inf, and those rows' probabilities go to 0. The limit of the fit is
    then the fit without those rows, on which ``x`` is constant within each
    event. Likewise for rows below the event maximum and +inf.

    Returns
    -------
    np.ndarray of bool, shape (n_rows,), or None
        The rows to remove, or None if the column is not separated this way.
    """
    by_event = pd.Series(x).groupby(groups)
    above_min = x > by_event.transform("min").to_numpy()
    below_max = x < by_event.transform("max").to_numpy()
    for vanishing in (above_min, below_max):
        if vanishing.any() and not np.any(y[vanishing & in_likelihood] > 0):
            return vanishing
    return None


def _fit_scaled(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, in_likelihood: np.ndarray
) -> tuple[np.ndarray, np.ndarray, _NewtonFinish]:
    """BFGS and the Newton finish on ``X`` divided by its within-event SD.

    Returns the scale, the scaled design (all rows) and the finish. The scale
    comes from all of ``X``'s rows, which the caller has checked are full rank
    within events; the likelihood uses the ``in_likelihood`` rows.
    """
    scale = _within_group_centered(X, groups).std(axis=0)
    assert np.all(scale > 0), "a column with no within-event SD passed the rank check"
    X_scaled = X / scale
    X_fit, y_fit, starts = _sorted_by_event(
        X_scaled[in_likelihood], y[in_likelihood], groups[in_likelihood]
    )
    finish = _newton_finish(X_fit, y_fit, starts, _bfgs_start(X_fit, y_fit, starts))
    return scale, X_scaled, finish


class _LimitFit(NamedTuple):
    """A fit with every term the session cannot inform taken to its limit."""

    rows: np.ndarray  # bool, the rows still in the fit
    columns: list[int]  # the columns estimated, in design order
    scale: np.ndarray  # their within-event SD on those rows
    X_scaled: np.ndarray  # those rows and columns, scaled
    finish: _NewtonFinish
    n_newton_steps: int  # over every pass


def _fit_to_the_limit(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    in_likelihood: np.ndarray,
    names: list[str],
) -> _LimitFit | None:
    """Fit, taking any term the session cannot inform to its limit.

    A direction without information that a single column carries is a term
    whose rows get all or none of each event's mass, so its coefficient runs to
    +/-inf. In that limit the rows it moves away from have probability 0, and
    they carry no mass, so the limit is the fit without those rows and without
    the column, which is then constant within each event. Its log-likelihood is
    the supremum of the full model's, and the other coefficients are their
    limits. Dropping only the column would be a different model: it gives those
    rows a probability that their zero mass contradicts, and on the real fits it
    moves the headline term by up to 0.76 SE.

    A column that the removed rows leave constant within every event is not
    informed either, and is dropped with it. A direction that mixes columns, or
    a single column whose rows do not split this way, is refused (None), as is a
    design with no column left.
    """
    rows = np.ones(len(y), dtype=bool)
    columns = list(range(X.shape[1]))
    n_newton_steps = 0
    while True:
        scale, X_scaled, finish = _fit_scaled(
            X[rows][:, columns], y[rows], groups[rows], in_likelihood[rows]
        )
        n_newton_steps += finish.n_steps
        uninformed = finish.eigenvalues < MIN_SCALED_INFORMATION
        if not uninformed.any():
            return _LimitFit(rows, columns, scale, X_scaled, finish, n_newton_steps)
        directions = finish.eigenvectors[:, uninformed]
        vanishing = None
        for k in _single_columns(directions) or []:
            vanishing = _rows_sent_to_zero_probability(
                X[rows, columns[k]], y[rows], groups[rows], in_likelihood[rows]
            )
            if vanishing is not None:
                break
        if vanishing is None:
            logger.warning(
                "fit_conditional_poisson: the data do not bound %s (information "
                "%.2g < %.0e on the within-event-SD scale, along a direction no "
                "single column carries); not fitting this design",
                ", ".join(_terms_on(directions, [names[c] for c in columns])),
                finish.eigenvalues[0],
                MIN_SCALED_INFORMATION,
            )
            return None
        unbounded = columns[k]
        rows[np.flatnonzero(rows)[vanishing]] = False
        columns = [c for c in columns if c != unbounded]
        left_constant: list[int] = []
        if columns:
            null = _within_group_null_directions(X[rows][:, columns], groups[rows])
            if null.shape[1]:
                single = _single_columns(null)
                if single is None:
                    logger.warning(
                        "fit_conditional_poisson: without the rows %s rules out, "
                        "%s are collinear within events; not fitting this design",
                        names[unbounded],
                        ", ".join(_terms_on(null, [names[c] for c in columns])),
                    )
                    return None
                left_constant = [columns[i] for i in single]
                columns = [c for c in columns if c not in left_constant]
        logger.warning(
            "fit_conditional_poisson: no event has mass on the rows %s rules out, "
            "so its coefficient runs to +/-inf; fitted at that limit, with %s "
            "reported as not estimated",
            names[unbounded],
            ", ".join(names[c] for c in [unbounded, *left_constant]),
        )
        if not columns:
            logger.warning(
                "fit_conditional_poisson: no term of %s is informed; not fitting",
                names,
            )
            return None


def within_group_full_rank(X: np.ndarray, groups: np.ndarray) -> bool:
    """True iff the within-group-demeaned design is full column rank.

    Conditional Poisson / Logit identify only **within-group** variation:
    a predictor that is constant within every group (event/stop) is
    absorbed by the conditioning and is unidentified — statsmodels still
    returns a finite estimate with an exploding standard error rather than
    failing. Demeaning each predictor within its group and checking the
    rank catches that case *and* globally collinear predictors (a
    duplicated predictor demeans to a collinear column too).

    Parameters
    ----------
    X : np.ndarray, shape (n_rows, n_predictors)
        Design matrix (no intercept; conditioned out).
    groups : np.ndarray, shape (n_rows,)
        Group label per row (the conditioning variable, e.g. event_id).

    Returns
    -------
    bool
        True when every predictor carries within-group variation and the
        demeaned design is full rank; False if rank-deficient.
    """
    if X.shape[1] == 0:
        return True
    centered = _within_group_centered(X, groups)
    return int(np.linalg.matrix_rank(centered)) == X.shape[1]


def fit_conditional_poisson(
    long_df: pd.DataFrame,
    predictors: list[str],
    response_col: str = "arm_count_hard",
    event_col: str = "event_id",
    n_categories: int = 9,
) -> dict | None:
    """Fit a conditional Poisson model on long-format count data.

    No intercept is included — it is conditioned out. Event-level
    constants are not identifiable and must not appear in predictors.

    Parameters
    ----------
    long_df : pd.DataFrame
        Long-format data (one row per (event, category)) with counts and
        predictors. Defaults to 9 rows per event for the arm-level fit;
        pass ``n_categories=3`` for the patch-level fit.
    predictors : list[str]
        Column names for category-varying predictors.
    response_col : str
        Column with non-negative per-category response values. Integer
        timebin counts (``arm_count_hard`` / ``patch_count_hard``) are
        the primary use; fractional responses (``arm_count_soft``,
        ``arm_posterior_mass``) are also supported for robustness
        analyses.
    event_col : str
        Column with globally unique event IDs (grouping variable).
    n_categories : int, optional
        Number of candidate categories per event under the null
        (uniform) model. ``9`` for arm-level fits, ``3`` for
        patch-level. Used in pseudo-R² null likelihood and forwarded to
        ``fit_null_model_conditional``.

    Returns
    -------
    dict | None
        Coefficients, standard errors, p-values, fit statistics, and the
        optimizer's end state: ``newton_decrement`` (the log-likelihood gain one
        more Newton step predicts), ``max_abs_grad`` (largest gradient
        component of the total log-likelihood) and ``n_newton_steps``.
        ``converged`` is True only when the decrement is below
        ``NEWTON_DECREMENT_TOL``. ``unidentified_terms`` ("|"-joined) names the
        terms this session cannot inform: a single column along which the
        information at the optimum is below ``MIN_SCALED_INFORMATION``, because
        its coefficient runs to +/-inf (see :func:`_fit_to_the_limit`). Their
        coefficient and p-value are NaN and their SEs infinite; the other terms
        are their values in that limit. None if there are too few events, if the
        design is rank-deficient within events, or if a direction without
        information mixes several columns. The log names the columns involved.

    Notes
    -----
    The fit is BFGS from zero followed by Newton steps, on the columns divided
    by their within-event SD. Dividing a column by a positive constant
    reparametrizes the model and changes nothing else: the log-likelihood is a
    function of ``X @ beta`` alone, so the optimum, the log-likelihood, the
    model covariance and the event-clustered covariance all map back exactly
    (``beta = beta_s / s``, ``Cov = D^-1 Cov_s D^-1`` with ``D = diag(s)``). It
    changes only the optimizer's path. The model SEs come from the inverse of
    the analytic information at the optimum.
    """
    if len(predictors) == 0:
        return fit_null_model_conditional(
            long_df,
            event_col,
            response_col,
            n_arms=n_categories,
            estimator=poisson_estimator_name(response_col),
        )

    required = [response_col, event_col] + predictors
    missing = [c for c in required if c not in long_df.columns]
    if missing:
        logger.error("Missing columns: %s", missing)
        return None

    df = long_df[required].copy()

    # Event-wise missing-data filtering: drop entire events if ANY row
    # within the event has NaN in any required column. This preserves the
    # 9-arm candidate set so that the conditional likelihood and the null
    # model use the same |A_e| for every event.
    has_nan = df[required].isna().any(axis=1)
    events_with_nan = df.loc[has_nan, event_col].unique()
    if len(events_with_nan) > 0:
        logger.info(
            "fit_conditional_poisson: dropping %d event(s) with NaN in a required "
            "column before fitting %s",
            len(events_with_nan),
            predictors,
        )
    df = df[~df[event_col].isin(events_with_nan)].reset_index(drop=True)

    df = _drop_zero_count_events(df, event_col, response_col)

    y = df[response_col].values.astype(float)
    groups = df[event_col].values

    n_events = len(np.unique(groups))
    if n_events < 2 * len(predictors):
        logger.info(
            "fit_conditional_poisson: %d events < 2*%d predictors; not fitting %s",
            n_events,
            len(predictors),
            predictors,
        )
        return None

    X = df[predictors].values.astype(float)

    # Reject designs the conditional model cannot identify: globally
    # collinear predictors AND predictors with no within-group variation
    # (constant within every event), which would otherwise be fit with
    # finite-but-meaningless coefficients and exploding SEs.
    if not within_group_full_rank(X, groups):
        logger.warning(
            "fit_conditional_poisson: no within-event information in %s (constant "
            "within every event, or collinear); not fitting this design",
            ", ".join(_rank_deficient_terms(X, groups, predictors)),
        )
        return None

    # Divide each column by its within-event SD before optimizing (_fit_scaled).
    # This is a reparametrization, not a change of model: the conditional
    # log-likelihood depends on the design only through X @ beta =
    # (X / s) @ (s * beta), so the optimum, the log-likelihood and both
    # covariances map back exactly, and only the optimizer's path and stopping
    # point change. The conditional likelihood sees only the variation within
    # each event, so the within-event SD is the scale its curvature is set by;
    # the global SD (the ``sd_*`` outputs) can be large for a column whose
    # within-event variation is tiny. The columns are not centred: a per-event
    # constant drops out of a conditional likelihood, so centring would change
    # nothing and would be one more step to undo.
    #
    # within_group_full_rank counts a singular value of the centred design as
    # zero below numpy.linalg.matrix_rank's default tolerance,
    # sigma_max * max(n_rows, n_params) * machine epsilon. A column with no
    # within-event variation is a zero column of the centred design, so the
    # check above has refused it and every scale is positive.
    in_likelihood = _rows_of_events_whose_response_varies(y, groups)
    limit = _fit_to_the_limit(X, y, groups, in_likelihood, predictors)
    if limit is None:
        return None
    finish, scale = limit.finish, limit.scale
    estimated = [predictors[k] for k in limit.columns]
    converged = bool(finish.decrement < NEWTON_DECREMENT_TOL)
    if not converged:
        logger.warning(
            "fit_conditional_poisson: not converged; the optimum is %.2g "
            "log-likelihood units away after %d Newton steps (fitting %s)",
            finish.decrement,
            finish.n_steps,
            predictors,
        )

    # Back to the original scale: beta = beta_s / s and Cov = D^-1 Cov_s D^-1,
    # with Cov_s the inverse of the analytic information at the optimum.
    params = finish.params / scale
    variance_scaled = (finish.eigenvectors**2 / finish.eigenvalues).sum(axis=1)
    bse_values = np.sqrt(variance_scaled) / scale
    pvalues = 2 * norm.sf(np.abs(params / bse_values))
    # Judge separation on the effect scale: a basis column with a small SD reaches a
    # large coefficient at an ordinary effect size, and would otherwise be mis-flagged.
    predictor_sd = X.std(axis=0)
    separated = separated_terms(
        params,
        bse_values,
        predictor_sd=predictor_sd[limit.columns],
        term_names=estimated,
        nuisance_terms=NUISANCE_TERMS,
    )
    # Event-clustered SEs alongside the model SEs: the model SE is the one the
    # conditional likelihood implies, the clustered SE is the one the events'
    # own variability supports. Aggregation prefers the clustered column.
    # Computed on the scaled columns and mapped back like the model covariance.
    cluster_cov = event_clustered_covariance(
        limit.X_scaled, y[limit.rows], groups[limit.rows], finish.params
    )
    bse_cluster_values = np.sqrt(np.diag(cluster_cov)) / scale

    # A term the session cannot inform is reported, not estimated: NaN
    # coefficient and p-value, infinite SEs. Readers skip it for that term only.
    coefs = dict.fromkeys(predictors, np.nan)
    bse = dict.fromkeys(predictors, np.inf)
    pvals = dict.fromkeys(predictors, np.nan)
    bse_cluster = dict.fromkeys(predictors, np.inf)
    coefs.update(zip(estimated, params, strict=True))
    bse.update(zip(estimated, bse_values, strict=True))
    pvals.update(zip(estimated, pvalues, strict=True))
    bse_cluster.update(zip(estimated, bse_cluster_values, strict=True))

    llf = finish.llf
    n_params = len(predictors)
    aic = 2 * n_params - 2 * llf
    # BIC with n = n_events (the number of conditioning groups).
    # Alternative choices (n_rows = n_event x n_arm pairs, or total
    # count) yield different penalty scales; we use n_events for
    # consistency with the conditional-Poisson likelihood structure.
    bic = n_params * np.log(n_events) - 2 * llf

    # Pseudo-R² against the analytical null (uniform allocation over
    # ``n_categories`` candidate slots per event). The arm-level fit
    # uses 9; the patch-level fit (3 patches per event) must override
    # this — otherwise log(9) is used as the null and pseudo-R² /
    # AIC / BIC for P0–P3 silently scale against the wrong baseline.
    # The inferred ``len(df) // n_events`` would be wrong when NaN
    # filtering drops categories from some events (reviewer flag #1).
    total_count = float(y.sum())
    llf_null = -total_count * np.log(n_categories)
    pseudo_r2 = (
        1.0 - llf / llf_null if np.isfinite(llf_null) and llf_null != 0 else np.nan
    )

    output = {
        **{f"coef_{k}": v for k, v in coefs.items()},
        **{f"bse_{k}": v for k, v in bse.items()},
        **{f"pval_{k}": v for k, v in pvals.items()},
        **{f"bse_cluster_{k}": float(v) for k, v in bse_cluster.items()},
        "estimator": poisson_estimator_name(response_col),
        # statsmodels' conditional likelihood omits the multinomial coefficient,
        # so this llf is comparable only with other fits that also omit it.
        "llf_includes_constants": False,
        "n_events": n_events,
        "n_rows": len(df),
        "n_params": n_params,
        "llf": float(llf),
        "llf_null": float(llf_null),
        "aic": float(aic),
        "bic": float(bic),
        "pseudo_r2": float(pseudo_r2),
        **{
            f"sd_{name}": float(s)
            for name, s in zip(predictors, predictor_sd, strict=True)
        },
        "converged": converged,
        "separated": len(separated) > 0,
        # "|"-joined so the tuple survives a round trip through pickle and CSV.
        "separated_terms": "|".join(separated),
        # The fit raises no warnings of its own; a fit short of its optimum
        # counts as one, as in the Dirichlet-multinomial fitter.
        "n_convergence_warnings": int(not converged),
        "newton_decrement": float(finish.decrement),
        # On the original scale: the scaled gradient is the original one / s.
        "max_abs_grad": float(np.max(np.abs(finish.gradient * scale))),
        "n_newton_steps": int(limit.n_newton_steps),
        # Terms this session cannot inform ("|"-joined, like separated_terms).
        "unidentified_terms": "|".join(
            p for k, p in enumerate(predictors) if k not in limit.columns
        ),
    }

    # Diagnostics
    try:
        diag = _compute_conditional_diagnostics(X, predictors)
        output.update(diag)
        _warn_on_collinear_design(diag, predictors)
    except Exception as exc:
        logger.debug("VIF/conditional diagnostics skipped: %s", exc)

    return output


def fit_null_model_conditional(
    long_df: pd.DataFrame,
    event_col: str = "event_id",
    response_col: str = "arm_count_hard",
    n_arms: int = 9,
    estimator: str = CONDITIONAL_POISSON_COUNT,
) -> dict:
    """Analytical null model for conditional Poisson.

    Under the null, each arm has equal probability 1/|A_e|.
    The conditional log-likelihood is:
        ℓ₀ = -Σ_e L_e · log(|A_e|)
    where L_e = Σ_a c_ea is the total count for event e, and |A_e| is the
    number of candidate arms (coded generically, currently always 9).

    Parameters
    ----------
    long_df : pd.DataFrame
        Long-format data with counts and event IDs.
    event_col : str
        Column with event UIDs.
    response_col : str
        Column with arm counts (or per-event allocation fractions, in which
        case L_e = 1 and the null is -n_events * log(n_arms)).
    n_arms : int
        Number of candidate arms per event.
    estimator : str
        Estimator label of the ladder this null belongs to; passed by the
        caller so the null rung is comparable with the rungs above it.

    Returns
    -------
    dict
        Null model fit statistics with 0 parameters.
    """
    df = long_df[[event_col, response_col]].copy()
    has_nan = df[[event_col, response_col]].isna().any(axis=1)
    events_with_nan = df.loc[has_nan, event_col].unique()
    df = df[~df[event_col].isin(events_with_nan)].reset_index(drop=True)
    df = _drop_zero_count_events(df, event_col, response_col)

    event_totals = df.groupby(event_col)[response_col].sum()
    n_events = len(event_totals)

    # ℓ₀ = -Σ_e L_e · log(|A_e|)
    # With constant |A_e| = n_arms for all events:
    llf = -float(event_totals.sum()) * np.log(n_arms)

    return {
        "estimator": estimator,
        "llf_includes_constants": False,
        "n_events": n_events,
        "n_rows": len(df),
        "n_params": 0,
        "llf": float(llf),
        "llf_null": float(llf),
        "aic": float(-2 * llf),  # 2*0 - 2*llf
        "bic": float(-2 * llf),  # 0*log(n) - 2*llf
        "pseudo_r2": 0.0,
        "predicted_frac": 1.0 / n_arms,
    }


def fit_null_model_conditional_logit(
    long_df: pd.DataFrame,
    event_col: str = "event_id",
    response_col: str = "arm_count_hard",
    n_arms: int = 9,
    estimator: str = CONDITIONAL_LOGIT_DOMINANT,
) -> dict:
    """Analytical null for the dominant-arm conditional logit.

    The event-grain counterpart of :func:`fit_null_model_conditional`. There the
    unit is a timebin, so the null log-likelihood scales with the event's total
    count; here the unit is the *event* -- one choice among ``n_arms`` -- so it
    scales with the number of events:

        l0 = -n_events * log(n_arms)

    Without this the null rung produces no fit at the event grain and the whole
    ladder is refused publication, since every rung's ΔAIC is measured against
    it.

    Parameters
    ----------
    long_df : pd.DataFrame
        Long-format data with counts and event IDs.
    event_col, response_col : str
        Event id and arm-count columns.
    n_arms : int
        Size of the candidate set per event.
    estimator : str
        Estimator label of the ladder this null belongs to.

    Returns
    -------
    dict
        Null fit statistics with 0 parameters, keyed to match
        :func:`fit_conditional_logit`.
    """
    df = long_df[[event_col, response_col]].copy()
    has_nan = df.isna().any(axis=1)
    df = df[~df[event_col].isin(df.loc[has_nan, event_col].unique())]
    df = _drop_zero_count_events(df.reset_index(drop=True), event_col, response_col)

    n_events = int(df[event_col].nunique())
    llf = -n_events * float(np.log(n_arms))
    return {
        "estimator": estimator,
        "llf_includes_constants": False,
        "n_events": n_events,
        "n_rows": len(df),
        "n_params": 0,
        "llf": llf,
        "llf_null": llf,
        "aic": float(-2 * llf),
        "bic": float(-2 * llf),
        "pseudo_r2": 0.0,
        "predicted_frac": 1.0 / n_arms,
        "converged": True,
        "separated": False,
        "separated_terms": "",
        "n_convergence_warnings": 0,
    }


def fit_conditional_logit(
    long_df: pd.DataFrame,
    predictors: list[str],
    response_col: str = "arm_count_hard",
    event_col: str = "event_id",
) -> dict | None:
    """Fit conditional logit on dominant arm (robustness check).

    The response is binary: 1 for the arm with the highest count within
    each event, 0 for all others. Ties broken by lowest arm_id.

    Parameters
    ----------
    long_df : pd.DataFrame
        Long-format data with arm counts and event identifiers.
    predictors : list[str]
        Arm-varying predictor columns.
    response_col : str
        Column with arm counts for determining dominant arm.
    event_col : str
        Column with globally unique event IDs.

    Returns
    -------
    dict | None
        Coefficients and fit statistics, or None if fitting fails.
    """
    if len(predictors) == 0:
        return None

    required = [event_col, response_col] + predictors
    missing = [c for c in required if c not in long_df.columns]
    if missing:
        logger.warning("fit_conditional_logit missing columns: %s", missing)
        return None

    arm_col = (
        "candidate_arm"
        if "candidate_arm" in long_df.columns
        else "arm_id"
        if "arm_id" in long_df.columns
        else None
    )
    # Guard against a duplicate column if the resolved arm column is also
    # one of the predictors (would otherwise create a same-named column).
    select_cols = required + (
        [arm_col] if arm_col is not None and arm_col not in required else []
    )
    df = long_df[select_cols].copy()

    # Event-wise NaN filtering: drop entire events if any row has NaN,
    # consistent with fit_conditional_poisson.
    has_nan = df[required].isna().any(axis=1)
    events_with_nan = df.loc[has_nan, event_col].unique()
    if len(events_with_nan) > 0:
        logger.info(
            "fit_conditional_logit: dropping %d event(s) with NaN in a required "
            "column before fitting %s",
            len(events_with_nan),
            predictors,
        )
    df = df[~df[event_col].isin(events_with_nan)].reset_index(drop=True)

    df = _drop_zero_count_events(df, event_col, response_col)

    # Determine dominant arm per event. Break ties by the lowest arm id
    # when an arm column is available; otherwise fall back to pandas'
    # first-row idxmax behavior for minimal legacy inputs.
    if arm_col is not None:
        idx = (
            df.sort_values(
                [event_col, response_col, arm_col], ascending=[True, False, True]
            )
            .groupby(event_col, sort=False)
            .head(1)
            .index
        )
    else:
        idx = df.groupby(event_col)[response_col].idxmax()
    df["dominant"] = 0
    df.loc[idx, "dominant"] = 1
    y = df["dominant"].values

    groups = df[event_col].values
    n_events = len(np.unique(groups))
    if n_events < 2 * len(predictors):
        logger.info(
            "fit_conditional_logit: %d events < 2*%d predictors; not fitting %s",
            n_events,
            len(predictors),
            predictors,
        )
        return None

    X = df[predictors].values.astype(float)

    # Reject designs the conditional model cannot identify (see
    # fit_conditional_poisson): globally collinear or within-group-constant
    # predictors yield finite-but-meaningless coefficients/SEs.
    if not within_group_full_rank(X, groups):
        logger.warning(
            "fit_conditional_logit: rank-deficient or within-group-constant "
            "design; not fitting %s",
            predictors,
        )
        return None

    try:
        with capture_fit_warnings() as caught:
            model = ConditionalLogit(y, X, groups=groups)
            result = model.fit(disp=0, maxiter=CONDITIONAL_FIT_MAXITER)
    except Exception:
        logger.warning(
            "ConditionalLogit failed for predictors %s", predictors, exc_info=True
        )
        return None
    predictor_sd = X.std(axis=0)
    status = fit_status(
        result,
        "conditional",
        caught=caught,
        predictor_sd=predictor_sd,
        term_names=predictors,
        nuisance_terms=NUISANCE_TERMS,
    )

    coefs = dict(zip(predictors, result.params, strict=True))
    bse = dict(zip(predictors, result.bse, strict=True))
    pvals = dict(zip(predictors, result.pvalues, strict=True))

    return {
        **{f"coef_{k}": v for k, v in coefs.items()},
        **{f"bse_{k}": v for k, v in bse.items()},
        **{f"pval_{k}": v for k, v in pvals.items()},
        "estimator": CONDITIONAL_LOGIT_DOMINANT,
        "llf_includes_constants": False,
        "n_events": n_events,
        "n_rows": len(df),
        "n_params": len(predictors),
        "llf": float(result.llf),
        "aic": float(2 * len(predictors) - 2 * result.llf),
        "bic": float(len(predictors) * np.log(n_events) - 2 * result.llf),
        **{
            f"sd_{name}": float(s)
            for name, s in zip(predictors, predictor_sd, strict=True)
        },
        "converged": status["converged"],
        "separated": status["separated"],
        # "|"-joined so the tuple survives a round trip through pickle and CSV.
        "separated_terms": "|".join(status["separated_terms"]),
        "n_convergence_warnings": count_fit_quality_warnings(caught),
    }


# ── Internal helpers ──


def _drop_zero_count_events(
    df: pd.DataFrame, event_col: str, response_col: str
) -> pd.DataFrame:
    """Drop events whose ``response_col`` sums to 0 across all rows.

    The conditional likelihood is undefined (0/0) for such events;
    fit_conditional_logit's idxmax would silently pick an arbitrary
    "dominant" arm otherwise.

    An all-zero response is what an event that decoded nowhere looks like --
    every in-event timebin failed to attribute to a real arm, so
    ``align_count_matrix`` filled the row with zeros. Callers that go through
    ``freeze_common_cohort(drop_zero_response=True)`` have already recorded
    those units by ID in ``zero_response_unit_ids``, so this is a backstop; a
    caller that skips the cohort freeze would otherwise lose the events here
    with no trace, unlike the NaN drop in the fitters, which logs.
    """
    event_totals = df.groupby(event_col)[response_col].transform("sum")
    kept = df[event_totals > 0].reset_index(drop=True)
    n_dropped = df[event_col].nunique() - kept[event_col].nunique()
    if n_dropped:
        logger.info(
            "dropping %d event(s) whose %s sums to 0 (decoded nowhere)",
            n_dropped,
            response_col,
        )
    return kept


# Condition number above which collinearity is strong enough that individual
# coefficients should not be read separately. Belsley's conventional band is
# 10-30 for moderate and >30 for strong; use the upper end so the warning marks
# designs whose coefficients are genuinely not separable.
COND_NUMBER_WARN = 30.0


def _warn_on_collinear_design(diagnostics: dict, predictors: list[str]) -> None:
    """Log when a design's coefficients should not be read separately.

    Warns rather than raises. Some collinearity here is structural and
    intended -- ``is_candidate_arm_never_visited`` is 1 exactly where the
    pre-centering candidate-arm visit recency is maximal, so M4 carries the
    two together on purpose --
    and refusing to fit would remove the rung rather than qualify it. The
    point is that the joint step stays interpretable while the individual
    coefficients do not, which is a reader's problem, not the fitter's.
    """
    cond_number = diagnostics.get("cond_number")
    if cond_number is None or not np.isfinite(cond_number):
        return
    if cond_number <= COND_NUMBER_WARN:
        return
    worst = sorted(
        (
            (value, name)
            for name, value in (
                (p, diagnostics.get(f"vif_{p}", np.nan)) for p in predictors
            )
            if np.isfinite(value)
        ),
        reverse=True,
    )[:3]
    logger.warning(
        "Collinear design (condition number %.1f > %.0f): read the joint "
        "model-comparison step, not the individual coefficients. Highest VIF: %s",
        cond_number,
        COND_NUMBER_WARN,
        ", ".join(f"{name}={value:.1f}" for value, name in worst) or "unavailable",
    )


def _compute_conditional_diagnostics(
    X: np.ndarray,
    predictors: list[str],
) -> dict:
    """Compute diagnostics for conditional model (VIF, condition number).

    ``cond_number`` is the condition number of the **column-scaled design**,
    which is what the conventional 10/30 bands refer to. Two earlier variants
    are deliberately not used: ``cond(X)`` unscaled reports the arbitrary
    relative units of the columns rather than their collinearity, and
    ``cond(X.T @ X)`` is that quantity *squared*, so a design at the
    conventional threshold reads as ~900. Scaling each column to unit norm
    isolates the part that matters -- how close the columns are to linearly
    dependent -- and makes the number comparable with the same column produced
    by ``uncertainty_content.within_arm_horserace``.
    """
    result = {}

    if X.shape[1] > 0:
        norms = np.linalg.norm(X, axis=0)
        # A structurally-zero column (e.g. an arm indicator absent from this
        # session) carries no scale; leave it as-is rather than dividing by 0.
        scale = np.where(norms > 0, norms, 1.0)
        result["cond_number"] = float(np.linalg.cond(X / scale))

    # VIF (add constant for VIF computation)
    X_with_const = sm.add_constant(X)
    for i, pred in enumerate(predictors):
        try:
            result[f"vif_{pred}"] = variance_inflation_factor(X_with_const, i + 1)
        except Exception:
            result[f"vif_{pred}"] = np.nan

    return result
