"""Model checking utilities for the replay content model hierarchy.

Provides cross-validation (held-out conditional log-likelihood),
Pearson residuals, and diagnostic summaries for the conditional
Poisson (multinomial-count) model.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalPoisson

from .._fit_diagnostics import (
    EXPECTED_FIT_ERRORS,
    NUISANCE_TERMS,
    capture_fit_warnings,
    fit_status,
)

logger = logging.getLogger(__name__)


def cross_validate_conditional_poisson(
    long_df: pd.DataFrame,
    predictors: list[str],
    event_col: str = "event_id",
    response_col: str = "arm_count_hard",
    n_folds: int = 5,
    n_categories: int = 9,
) -> dict | None:
    """Event-based K-fold CV for conditional Poisson model.

    Splits events into folds. For each fold, fits the model on training
    events and evaluates held-out conditional log-likelihood on test events.

    Parameters
    ----------
    long_df : pd.DataFrame
        Long-format data (one row per (event, category)) with counts and
        predictors. ``n_categories`` rows per event in the canonical
        case (9 for arm-level, 3 for patch-level fits).
    predictors : list[str]
        Category-varying predictor columns.
    event_col : str
        Column with globally unique event IDs.
    response_col : str
        Column with integer category counts.
    n_folds : int
        Number of CV folds.
    n_categories : int, optional
        Number of candidate categories per event under the null
        (uniform) model. Use 9 for arm-level fits, 3 for patch-level.
        Affects the null fold-LL only.

    Returns
    -------
    dict | None
        CV metrics: mean/std held-out conditional LL per event and per
        timebin. None if insufficient data.
    """
    from sklearn.model_selection import GroupKFold

    if event_col not in long_df.columns or response_col not in long_df.columns:
        return None

    required = [event_col, response_col] + predictors
    df = long_df[required].copy()

    # Event-wise NaN filtering: drop entire events if any row has NaN,
    # so the candidate set stays at 9 arms for all retained events.
    has_nan = df[required].isna().any(axis=1)
    events_with_nan = df.loc[has_nan, event_col].unique()
    df = df[~df[event_col].isin(events_with_nan)].reset_index(drop=True)
    if len(df) == 0:
        return None

    # Drop zero-count events to match ``fit_conditional_poisson`` (and
    # the fitter script's pre-filter). The conditional likelihood is
    # 0 / 0 for events whose response sums to zero; without this the
    # null-fold LL counts those events in ``n_test_events`` and the
    # per-event mean drifts (we verified a single zero-count event
    # shifts ``cv_ll_per_event_mean`` from -2.7465 to -3.2958 while
    # the per-timebin LL stays identical).
    from ..fitting import CONDITIONAL_FIT_MAXITER, _drop_zero_count_events

    df = _drop_zero_count_events(df, event_col, response_col)
    if len(df) == 0:
        return None

    groups = df[event_col].values
    unique_events = np.unique(groups)
    n_events = len(unique_events)

    if n_events < n_folds * 2:
        return None

    y = df[response_col].values.astype(float)
    # Arm-level fits use 9; patch-level fits override to 3. Without this
    # being threaded through, the null fold-LL would silently use log(9)
    # for patch CV and the resulting baseline would be wrong.
    n_arms = n_categories

    gkf = GroupKFold(n_splits=n_folds)

    ll_per_event = []
    ll_per_timebin = []
    n_nonconverged = 0
    n_separated = 0

    for train_idx, test_idx in gkf.split(df, y, groups):
        y_train = y[train_idx]
        y_test = y[test_idx]
        groups_test = groups[test_idx]

        if len(predictors) == 0:
            # Null model: uniform allocation
            # ℓ = -Σ_e L_e · log(n_arms)
            event_totals = pd.Series(y_test).groupby(groups_test).sum()
            fold_ll = -float(event_totals.sum()) * np.log(n_arms)
            n_test_events = len(event_totals)
            total_timebins = event_totals.sum()
        else:
            X_train = df.iloc[train_idx][predictors].values.astype(float)
            X_test = df.iloc[test_idx][predictors].values.astype(float)
            groups_train = groups[train_idx]

            # A fold fit that raises is a hard failure -> the whole CV metric is
            # undefined (matches per_bin/per_stop CV). Convergence/separation are
            # SOFT: the fold is kept and scored, but any bad fold invalidates the
            # metric below -- never drop a fold, which would change the held-out set.
            try:
                with capture_fit_warnings() as caught:
                    model = ConditionalPoisson(y_train, X_train, groups=groups_train)
                    result = model.fit(disp=0, maxiter=CONDITIONAL_FIT_MAXITER)
            except EXPECTED_FIT_ERRORS:
                logger.debug(
                    "CV fold fit failed for predictors %s", predictors, exc_info=True
                )
                return None
            status = fit_status(
                result,
                "conditional",
                caught=caught,
                predictor_sd=X_train.std(axis=0),
                term_names=predictors,
                nuisance_terms=NUISANCE_TERMS,
            )
            n_nonconverged += status["converged"] is False
            n_separated += status["separated"]

            # Compute held-out conditional LL
            fold_ll = _held_out_conditional_ll(
                y_test, X_test, groups_test, result.params
            )
            n_test_events = len(np.unique(groups_test))
            total_timebins = y_test.sum()

        if n_test_events > 0:
            ll_per_event.append(fold_ll / n_test_events)
        if total_timebins > 0:
            ll_per_timebin.append(fold_ll / total_timebins)

    if len(ll_per_event) < 2:
        return None

    # Any nonconverged/separated fold invalidates the metric (nan) rather than
    # dropping the fold; the counts are reported alongside.
    invalid = (n_nonconverged > 0) or (n_separated > 0)
    return {
        "cv_ll_per_event_mean": np.nan if invalid else float(np.mean(ll_per_event)),
        "cv_ll_per_event_std": np.nan if invalid else float(np.std(ll_per_event)),
        "cv_ll_per_timebin_mean": np.nan if invalid else float(np.mean(ll_per_timebin)),
        "cv_ll_per_timebin_std": np.nan if invalid else float(np.std(ll_per_timebin)),
        "cv_n_folds": len(ll_per_event),
        "cv_n_nonconverged_folds": int(n_nonconverged),
        "cv_n_separated_folds": int(n_separated),
    }


def _held_out_conditional_ll(
    y: np.ndarray,
    X: np.ndarray,
    groups: np.ndarray,
    params: np.ndarray,
) -> float:
    """Compute conditional log-likelihood on held-out data.

    For each event e with arm counts c_ea and linear predictor η_ea = Xβ:
        ℓ_e = Σ_a c_ea · η_ea - L_e · log(Σ_a exp(η_ea))

    Parameters
    ----------
    y : array of shape (n_rows,)
        Arm counts.
    X : array of shape (n_rows, n_predictors)
        Predictor matrix.
    groups : array of shape (n_rows,)
        Event group IDs.
    params : array of shape (n_predictors,)
        Fitted coefficients.

    Returns
    -------
    float
        Total conditional log-likelihood across all events.
    """
    eta = X @ params
    y = np.asarray(y, dtype=float)

    unique_groups, inverse = np.unique(groups, return_inverse=True)
    n_groups = len(unique_groups)

    # Per-group sums via bincount
    L = np.bincount(inverse, weights=y, minlength=n_groups)
    cy_eta = np.bincount(inverse, weights=y * eta, minlength=n_groups)

    # Per-group log-sum-exp (numerically stable)
    # Sort by group for reduceat
    order = np.argsort(inverse, kind="stable")
    eta_sorted = eta[order]
    inv_sorted = inverse[order]
    group_starts = np.searchsorted(inv_sorted, np.arange(n_groups))

    max_eta = np.maximum.reduceat(eta_sorted, group_starts)
    exp_shifted = np.exp(eta[order] - max_eta[inv_sorted])
    sum_exp = np.bincount(inv_sorted, weights=exp_shifted, minlength=n_groups)
    log_sum_exp = max_eta + np.log(sum_exp)

    # Total LL, skipping groups where L == 0
    event_ll = cy_eta - L * log_sum_exp
    return float(event_ll[L > 0].sum())


def compute_pearson_residuals(
    long_df: pd.DataFrame,
    fit_result: dict,
    predictors: list[str],
    event_col: str = "event_id",
    response_col: str = "arm_count_hard",
) -> pd.DataFrame:
    """Compute Pearson residuals at the event-arm level.

    Parameters
    ----------
    long_df : pd.DataFrame
        Long-format data.
    fit_result : dict
        Output of fit_conditional_poisson.
    predictors : list[str]
        Predictor columns.
    event_col : str
        Event ID column.
    response_col : str
        Count column.

    Returns
    -------
    pd.DataFrame
        Original data with added ``predicted_frac`` and ``pearson_resid``
        columns.
    """
    required = [event_col, response_col] + predictors
    df = long_df.copy()

    # Event-wise NaN filtering (same as fitter): drop entire events
    # if any row has NaN in required columns.
    has_nan = df[required].isna().any(axis=1)
    events_with_nan = df.loc[has_nan, event_col].unique()
    df = df[~df[event_col].isin(events_with_nan)].reset_index(drop=True)

    # Remove zero-count events (same as fitter).
    event_totals = df.groupby(event_col)[response_col].transform("sum")
    df = df[event_totals > 0].reset_index(drop=True)

    # Unidentified terms at their limits (see conditional_linear_predictor): a
    # row a separated term rules out has prediction 0, and so residual 0.
    from ..fitting import conditional_linear_predictor

    X = df[predictors].values.astype(float)
    eta = conditional_linear_predictor(
        X, df[event_col].to_numpy(), fit_result, list(predictors)
    )

    # Predicted fractions per event (softmax within event, stabilized)
    df["_eta"] = eta
    max_eta = df.groupby(event_col)["_eta"].transform("max")
    df["_shifted_exp"] = np.exp(df["_eta"] - max_eta)
    sum_shifted = df.groupby(event_col)["_shifted_exp"].transform("sum")
    df["predicted_frac"] = df["_shifted_exp"] / sum_shifted

    # Observed fraction
    total_count = df.groupby(event_col)[response_col].transform("sum")
    df["observed_frac"] = df[response_col] / total_count.clip(lower=1)

    # Pearson residual: (observed - expected) / sqrt(expected * (1 - expected/L_e))
    expected_count = df["predicted_frac"] * total_count
    variance = expected_count * (1 - df["predicted_frac"])
    df["pearson_resid"] = (df[response_col] - expected_count) / np.sqrt(
        variance.clip(lower=1e-10)
    )

    df = df.drop(columns=["_eta", "_shifted_exp"])
    return df
