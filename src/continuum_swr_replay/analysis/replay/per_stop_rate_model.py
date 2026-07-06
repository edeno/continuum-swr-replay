"""Poisson rate model for replay event occurrence (Stage 2).

Tests whether the NUMBER of replay events per stop depends on
behavioral context, independently of replay CONTENT. The response is
``n_events`` (count of non-local events per stop); the offset is
``log(immobility_time)`` so that coefficients are interpretable as
multiplicative changes in event rate per second of immobility.

Model hierarchy (``MODEL_PER_STOP_RATE`` in ``model_specs.py``):

- **R0**: intercept only — baseline event rate.
- **R1**: + ``is_after_omission``, ``is_after_switch`` — does the
  immediately preceding trial's context change event rate?
  ``is_after_reward`` is intentionally excluded: it equals
  ``1 - is_after_omission`` wherever both are defined, so including
  it alongside the intercept is rank-deficient. The reward effect is
  read off the sign of the ``is_after_omission`` coefficient.
- **R2**: + ``recent_trial_reward_rate`` (rewarded fraction of the
  last 10 trials) — does event rate track a deeper reward history,
  beyond the single previous trial?

A future R3 (+ ``trials_since_block_change``, ``block_number``) is
deferred until the contingency-block loader is wired. Fitting R0-R2
separately on SWR-only / MUA-only / neither stops is an analysis
variant, not a separate model.

``exp(γ)`` = multiplicative change in events per second of
immobility. ``exp(γ) = 1.5`` means 50% more events per second when
the predictor is active.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm

from ._fit_diagnostics import (
    capture_fit_warnings,
    count_fit_quality_warnings,
    fit_status,
)

logger = logging.getLogger(__name__)


def fit_per_stop_rate_model(
    stop_table: pd.DataFrame,
    predictors: list[str],
    *,
    response_col: str = "n_events",
    offset_col: str = "immobility_time",
    min_stops: int = 20,
) -> dict | None:
    """Fit a Poisson GLM on event counts per stop.

    Parameters
    ----------
    stop_table : pd.DataFrame
        One session's ``stop_table`` (includes zero-event stops).
    predictors : list[str]
        Columns to use as fixed effects. Empty list fits the
        intercept-only null (R0).
    response_col : str
        Count column. Default ``n_events``.
    offset_col : str
        Exposure column — ``log(offset_col)`` enters as an offset.
        Stops with ``offset_col <= 0`` are excluded (single-sample
        bouts where ``immobility_time == 0``).
    min_stops : int
        Minimum number of valid stops to attempt fitting.

    Returns
    -------
    dict | None
        Coefficients (``coef_*``), standard errors (``bse_*``),
        p-values (``pval_*``), rate ratios (``rr_*`` = exp(coef)),
        plus fit statistics. None if fitting fails.
    """
    required = [response_col, offset_col] + predictors
    missing = [c for c in required if c not in stop_table.columns]
    if missing:
        logger.error("Missing columns: %s", missing)
        return None

    df = stop_table[required].copy()
    df = df.dropna(subset=required)

    # Exclude zero-duration stops (log(0) = -inf).
    df = df[df[offset_col] > 0].reset_index(drop=True)

    if len(df) < min_stops:
        return None

    y = df[response_col].values.astype(float)
    offset = np.log(df[offset_col].values.astype(float))

    if predictors:
        # Force ``add_constant`` to always prepend an intercept column,
        # even if one of the predictors is itself constant within
        # ``df``. With the default ``has_constant="skip"`` the call is
        # a no-op for a constant predictor, the returned ``X`` has
        # ``len(predictors)`` columns instead of
        # ``len(predictors) + 1``, ``col_names`` claims an extra
        # ``"intercept"`` slot, and the post-fit
        # ``zip(col_names, result.params, strict=True)`` raises
        # ValueError after the GLM has already run.
        raw_X = df[predictors].to_numpy(dtype=float)
        X = sm.add_constant(raw_X, has_constant="add")
        col_names = ["intercept"] + predictors

        # Full-rank check rejects constant predictors, perfect
        # collinearity, duplicated columns, and any other zero
        # linear combination of design columns — statsmodels would
        # otherwise pseudo-invert and emit finite coefficients /
        # p-values for an unidentified model.
        rank = int(np.linalg.matrix_rank(X))
        if rank < X.shape[1]:
            logger.warning(
                "Poisson GLM not fit: design matrix [intercept, %s] is "
                "rank-deficient (rank %d < %d) within the %d-stop subset.",
                ", ".join(predictors),
                rank,
                X.shape[1],
                len(df),
            )
            return None
    else:
        X = np.ones((len(df), 1))
        col_names = ["intercept"]

    try:
        with capture_fit_warnings() as caught:
            model = sm.GLM(y, X, family=sm.families.Poisson(), offset=offset)
            result = model.fit(disp=0)
    except Exception:
        logger.warning("Poisson GLM failed for predictors %s", predictors)
        return None
    status = fit_status(result, "glm")

    coefs = dict(zip(col_names, result.params, strict=True))
    bse = dict(zip(col_names, result.bse, strict=True))
    pvals = dict(zip(col_names, result.pvalues, strict=True))
    n_params = len(col_names)
    bic_llf = n_params * np.log(len(df)) - 2 * float(result.llf)

    output = {
        **{f"coef_{k}": float(v) for k, v in coefs.items()},
        **{f"bse_{k}": float(v) for k, v in bse.items()},
        **{f"pval_{k}": float(v) for k, v in pvals.items()},
        **{f"rr_{k}": float(np.exp(v)) for k, v in coefs.items()},
        "n_stops": len(df),
        "n_events_total": int(y.sum()),
        "n_params": n_params,
        "llf": float(result.llf),
        "aic": float(result.aic),
        "bic": float(bic_llf),
        "deviance": float(result.deviance),
        "pearson_chi2": float(result.pearson_chi2),
        "mean_rate": float(y.sum() / np.exp(offset).sum()),
        "converged": status["converged"],
        "separated": status["separated"],
        "n_convergence_warnings": count_fit_quality_warnings(caught),
    }

    return output


def cross_validate_per_stop_rate_model(
    stop_table: pd.DataFrame,
    predictors: list[str],
    response_col: str = "n_events",
    offset_col: str = "immobility_time",
    n_folds: int = 5,
    min_stops: int = 20,
    seed: int = 0,
) -> dict | None:
    """K-fold held-out log-likelihood for the per-stop rate Poisson.

    Stop-based K-fold CV mirroring :func:`fit_per_stop_rate_model`'s design (intercept +
    predictors, ``log(offset_col)`` offset). For each fold the model is fit on
    the training stops and the mean held-out per-stop Poisson log-likelihood is
    evaluated on the test stops; the returned ``cv_ll_per_stop_mean`` is the
    average across folds (higher = better predictive fit). This is the rate
    analogue of :func:`model_checking.cross_validate_conditional_poisson`, which
    only applies to the conditional (arm/patch) models.

    Returns None if columns are missing, too few stops, or any fold's training
    design is rank-deficient (matching ``fit_per_stop_rate_model``'s identifiability
    guard).
    """
    from scipy.stats import poisson
    from sklearn.model_selection import KFold

    required = [response_col, offset_col] + predictors
    if any(c not in stop_table.columns for c in required):
        return None
    df = stop_table[required].dropna()
    df = df[df[offset_col] > 0].reset_index(drop=True)
    if len(df) < max(min_stops, n_folds * 2):
        return None

    y = df[response_col].to_numpy(dtype=float)
    offset = np.log(df[offset_col].to_numpy(dtype=float))
    if predictors:
        X = sm.add_constant(df[predictors].to_numpy(dtype=float), has_constant="add")
    else:
        X = np.ones((len(df), 1))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_ll = []
    n_nonconverged = 0
    n_separated = 0
    for train, test in kf.split(X):
        if np.linalg.matrix_rank(X[train]) < X.shape[1]:
            return None
        try:
            res = sm.GLM(
                y[train], X[train], family=sm.families.Poisson(), offset=offset[train]
            ).fit(disp=0)
        except Exception:
            return None
        status = fit_status(res, "glm")
        n_nonconverged += not status["converged"]
        n_separated += status["separated"]
        mu = np.exp(np.clip(X[test] @ res.params + offset[test], -30, 30))
        ll = poisson.logpmf(y[test], mu)
        if not np.all(np.isfinite(ll)):
            return None
        fold_ll.append(float(np.mean(ll)))

    # Deterministic rule: any nonconverged/separated fold invalidates the whole CV
    # metric (nan) rather than dropping the fold -- dropping a fold changes the
    # held-out set and makes CV metrics non-comparable across rungs. The fold counts
    # are reported alongside so the invalidation is visible.
    invalid = (n_nonconverged > 0) or (n_separated > 0)
    return {
        "cv_ll_per_stop_mean": np.nan if invalid else float(np.mean(fold_ll)),
        "cv_ll_per_stop_std": np.nan if invalid else float(np.std(fold_ll)),
        "cv_n_folds": n_folds,
        "cv_n_nonconverged_folds": int(n_nonconverged),
        "cv_n_separated_folds": int(n_separated),
    }
