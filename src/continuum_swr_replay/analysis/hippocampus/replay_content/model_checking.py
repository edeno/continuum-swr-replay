"""Model checking utilities for the replay content model hierarchy.

Provides cross-validation, aggregated diagnostic summaries, and
separation detection beyond the per-fit diagnostics in fitting.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .fitting import ALPHA, _log_likelihood


def cross_validate_session(
    long_df: pd.DataFrame,
    predictors: list[str],
    n_folds: int = 5,
    alpha: float = ALPHA,
) -> dict | None:
    """K-fold cross-validation for one session's long-format data.

    Splits events (not rows) into folds to avoid data leakage — all 9
    rows for an event go into the same fold.

    Parameters
    ----------
    long_df : pd.DataFrame
        Long-format data for a single session.
    predictors : list[str]
        Predictor columns.
    n_folds : int
        Number of CV folds.
    alpha : float
        L2 regularization strength.

    Returns
    -------
    dict | None
        CV metrics: mean/std Brier score, mean/std log-loss.
        None if insufficient data.
    """
    from sklearn.model_selection import GroupKFold

    # Use non_local_event as the group variable for event-based splitting.
    # GroupKFold ensures all rows for an event stay in the same fold.
    if "non_local_event" not in long_df.columns:
        return None

    groups = long_df["non_local_event"].values
    n_unique_events = len(np.unique(groups))

    if n_unique_events < n_folds * 2:
        return None

    y = long_df["replay_here"].values
    gkf = GroupKFold(n_splits=n_folds)

    brier_scores = []
    ll_scores = []

    for train_idx, test_idx in gkf.split(long_df, y, groups):
        y_train = y[train_idx]
        y_test = y[test_idx]

        if y_train.sum() < 3 or y_test.sum() < 1:
            continue

        if len(predictors) == 0:
            # Null model: predict test fold with training mean
            p_train = y_train.mean()
            y_pred = np.full_like(y_test, p_train, dtype=float)
        else:
            X_train = sm.add_constant(
                long_df.iloc[train_idx][predictors].values.astype(float)
            )
            X_test = sm.add_constant(
                long_df.iloc[test_idx][predictors].values.astype(float)
            )
            try:
                model = sm.GLM(y_train, X_train, family=sm.families.Binomial())
                result = model.fit_regularized(alpha=alpha, L1_wt=0, refit=False)
                y_pred = result.predict(X_test)
            except Exception:
                continue

        brier_scores.append(float(np.mean((y_test - y_pred) ** 2)))
        ll_scores.append(_log_likelihood(y_test, y_pred) / len(y_test))

    if len(brier_scores) < 2:
        return None

    return {
        "cv_brier_mean": float(np.mean(brier_scores)),
        "cv_brier_std": float(np.std(brier_scores)),
        "cv_ll_mean": float(np.mean(ll_scores)),
        "cv_ll_std": float(np.std(ll_scores)),
        "cv_n_folds": len(brier_scores),
    }


def check_separation(
    fit_result: dict,
    predictors: list[str],
    threshold: float = 5.0,
) -> dict:
    """Check for quasi-complete separation indicators.

    Parameters
    ----------
    fit_result : dict
        Output of fit_glm.
    predictors : list[str]
        Predictor names.
    threshold : float
        Coefficient magnitude threshold suggesting separation.

    Returns
    -------
    dict
        Separation flags and the largest coefficient.
    """
    max_coef = fit_result.get("max_abs_coef", 0.0)
    large_coefs = []
    for pred in predictors:
        col = f"coef_{pred}"
        if col in fit_result and abs(fit_result[col]) > threshold:
            large_coefs.append(pred)

    return {
        "separation_flag": len(large_coefs) > 0,
        "n_large_coefs": len(large_coefs),
        "large_coef_names": large_coefs,
        "max_abs_coef": max_coef,
    }


def aggregate_diagnostics(
    model_results: pd.DataFrame,
    predictors: list[str],
) -> dict:
    """Aggregate per-session diagnostics into a summary.

    Parameters
    ----------
    model_results : pd.DataFrame
        Per-session fit results for one model.
    predictors : list[str]
        Predictor names.

    Returns
    -------
    dict
        Summary statistics for each diagnostic metric.
    """
    summary = {
        "n_sessions": len(model_results),
    }

    # Fit quality
    for metric in ["pseudo_r2", "brier", "aic", "bic", "ll"]:
        if metric in model_results.columns:
            vals = model_results[metric].dropna()
            if len(vals) > 0:
                summary[f"{metric}_median"] = float(vals.median())
                summary[f"{metric}_q25"] = float(vals.quantile(0.25))
                summary[f"{metric}_q75"] = float(vals.quantile(0.75))

    # Collinearity
    if "cond_number" in model_results.columns:
        vals = model_results["cond_number"].dropna()
        if len(vals) > 0:
            summary["cond_number_median"] = float(vals.median())
            summary["cond_number_max"] = float(vals.max())

    # VIF per predictor
    for pred in predictors:
        col = f"vif_{pred}"
        if col in model_results.columns:
            vals = model_results[col].dropna()
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                summary[f"vif_{pred}_median"] = float(vals.median())
                summary[f"vif_{pred}_max"] = float(vals.max())

    # Residuals
    for metric in ["resid_mean", "resid_std", "resid_skew", "resid_abs_max"]:
        if metric in model_results.columns:
            vals = model_results[metric].dropna()
            if len(vals) > 0:
                summary[f"{metric}_median"] = float(vals.median())

    # Calibration
    if "calib_slope" in model_results.columns:
        vals = model_results["calib_slope"].dropna()
        if len(vals) > 0:
            summary["calib_slope_median"] = float(vals.median())

    # Separation
    if "max_abs_coef" in model_results.columns:
        vals = model_results["max_abs_coef"].dropna()
        summary["n_sessions_large_coef"] = int((vals > 5.0).sum())

    # CV metrics (if present)
    for metric in ["cv_brier_mean", "cv_ll_mean"]:
        if metric in model_results.columns:
            vals = model_results[metric].dropna()
            if len(vals) > 0:
                summary[f"{metric}_median"] = float(vals.median())

    return summary


def print_diagnostic_summary(
    summary: dict,
    model_name: str,
    predictors: list[str],
) -> None:
    """Print a formatted diagnostic summary."""
    print(f"\n  DIAGNOSTICS for {model_name}:")
    print(f"    Sessions: {summary.get('n_sessions', 0)}")

    # Fit quality
    for metric in ["pseudo_r2", "brier", "aic"]:
        key = f"{metric}_median"
        if key in summary:
            print(f"    {metric}: {summary[key]:.4f}")

    # Collinearity
    if "cond_number_median" in summary:
        print(
            f"    Condition #: median={summary['cond_number_median']:.0f}, "
            f"max={summary.get('cond_number_max', 0):.0f}"
        )

    # VIF
    vif_strs = []
    for pred in predictors[:6]:  # limit to first 6 for readability
        key = f"vif_{pred}_median"
        if key in summary:
            vif_strs.append(f"{pred}={summary[key]:.1f}")
    if vif_strs:
        print(f"    VIF: {', '.join(vif_strs)}")

    # Residuals
    if "resid_skew_median" in summary:
        print(f"    Resid skew: {summary['resid_skew_median']:.3f}")

    # Calibration
    if "calib_slope_median" in summary:
        print(f"    Calibration: {summary['calib_slope_median']:.2f} (1.0=perfect)")

    # Separation
    n_large = summary.get("n_sessions_large_coef", 0)
    if n_large > 0:
        print(f"    WARNING: {n_large} sessions with |coef| > 5")

    # CV
    if "cv_brier_mean_median" in summary:
        print(f"    CV Brier: {summary['cv_brier_mean_median']:.4f}")
