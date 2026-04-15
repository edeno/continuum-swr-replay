"""Per-session model fitting for the replay content model hierarchy.

Primary model: Conditional Poisson regression (multinomial-count softmax)
via statsmodels ConditionalPoisson, where the response is per-arm timebin
counts and the model conditions on the event.

Robustness model: Conditional Logit on dominant arm (highest count).

The null model (M0) is computed analytically:
    ℓ₀ = -Σ_e L_e · log|A_e|
where L_e is the total timebin count and |A_e| is the number of arms
with nonzero availability (currently always 9).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import linregress
from statsmodels.discrete.conditional_models import (
    ConditionalLogit,
    ConditionalPoisson,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)

ALPHA = 0.01  # L2 regularization strength (fallback ridge)


def fit_conditional_poisson(
    long_df: pd.DataFrame,
    predictors: list[str],
    response_col: str = "arm_count",
    event_col: str = "event_uid",
) -> dict | None:
    """Fit a conditional Poisson model on long-format count data.

    No intercept is included — it is conditioned out. Event-level
    constants are not identifiable and must not appear in predictors.

    Parameters
    ----------
    long_df : pd.DataFrame
        Long-format data (9 rows per event) with arm counts and predictors.
    predictors : list[str]
        Column names for arm-varying predictors.
    response_col : str
        Column with non-negative per-arm response values. Integer timebin
        counts (``arm_count_hard``) are the primary use; fractional
        responses (``arm_count_soft``, ``arm_posterior_mass``) are also
        supported for robustness analyses.
    event_col : str
        Column with globally unique event IDs (grouping variable).

    Returns
    -------
    dict | None
        Coefficients, standard errors, p-values, fit statistics.
        None if fitting fails or insufficient data.
    """
    if len(predictors) == 0:
        return fit_null_model_conditional(long_df, event_col, response_col)

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
    df = df[~df[event_col].isin(events_with_nan)].reset_index(drop=True)

    # Remove events where all arm counts are zero (degenerate for
    # conditional likelihood — undefined 0/0 allocation).
    event_totals = df.groupby(event_col)[response_col].transform("sum")
    df = df[event_totals > 0].reset_index(drop=True)

    y = df[response_col].values.astype(float)
    groups = df[event_col].values

    n_events = len(np.unique(groups))
    if n_events < 2 * len(predictors):
        return None

    X = df[predictors].values.astype(float)

    try:
        model = ConditionalPoisson(y, X, groups=groups)
        result = model.fit(disp=0)
    except Exception:
        logger.warning("ConditionalPoisson failed for predictors %s", predictors)
        return None

    coefs = dict(zip(predictors, result.params, strict=True))
    bse = dict(zip(predictors, result.bse, strict=True))
    pvals = dict(zip(predictors, result.pvalues, strict=True))

    llf = result.llf
    n_params = len(predictors)
    aic = 2 * n_params - 2 * llf
    bic = n_params * np.log(n_events) - 2 * llf

    output = {
        **{f"coef_{k}": v for k, v in coefs.items()},
        **{f"bse_{k}": v for k, v in bse.items()},
        **{f"pval_{k}": v for k, v in pvals.items()},
        "n_events": n_events,
        "n_rows": len(df),
        "n_params": n_params,
        "llf": float(llf),
        "aic": float(aic),
        "bic": float(bic),
    }

    # Diagnostics
    try:
        diag = _compute_conditional_diagnostics(X, predictors)
        output.update(diag)
    except Exception:
        pass

    return output


def fit_null_model_conditional(
    long_df: pd.DataFrame,
    event_col: str = "event_uid",
    response_col: str = "arm_count",
    n_arms: int = 9,
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
        Column with arm counts.
    n_arms : int
        Number of candidate arms per event.

    Returns
    -------
    dict
        Null model fit statistics with 0 parameters.
    """
    event_totals = long_df.groupby(event_col)[response_col].sum()
    n_events = len(event_totals)

    # ℓ₀ = -Σ_e L_e · log(|A_e|)
    # With constant |A_e| = n_arms for all events:
    llf = -float(event_totals.sum()) * np.log(n_arms)

    return {
        "n_events": n_events,
        "n_rows": len(long_df),
        "n_params": 0,
        "llf": float(llf),
        "aic": float(-2 * llf),  # 2*0 - 2*llf
        "bic": float(-2 * llf),  # 0*log(n) - 2*llf
    }


def fit_conditional_logit(
    long_df: pd.DataFrame,
    predictors: list[str],
    event_col: str = "event_uid",
) -> dict | None:
    """Fit conditional logit on dominant arm (robustness check).

    The response is binary: 1 for the arm with the highest count within
    each event, 0 for all others. Ties broken by lowest arm_id.

    Parameters
    ----------
    long_df : pd.DataFrame
        Long-format data with ``arm_count`` and event identifiers.
    predictors : list[str]
        Arm-varying predictor columns.
    event_col : str
        Column with globally unique event IDs.

    Returns
    -------
    dict | None
        Coefficients and fit statistics, or None if fitting fails.
    """
    if len(predictors) == 0:
        return None

    response_col = "arm_count" if "arm_count" in long_df.columns else "replay_here"
    required = [event_col, response_col] + predictors
    missing = [c for c in required if c not in long_df.columns]
    if missing:
        return None

    df = long_df[required].copy()

    # Event-wise NaN filtering: drop entire events if any row has NaN,
    # consistent with fit_conditional_poisson.
    has_nan = df[required].isna().any(axis=1)
    events_with_nan = df.loc[has_nan, event_col].unique()
    df = df[~df[event_col].isin(events_with_nan)].reset_index(drop=True)

    # Determine dominant arm per event
    if response_col == "arm_count":
        idx = df.groupby(event_col)["arm_count"].idxmax()
        df["dominant"] = 0
        df.loc[idx, "dominant"] = 1
        y = df["dominant"].values
    else:
        y = df["replay_here"].values

    groups = df[event_col].values
    n_events = len(np.unique(groups))
    if n_events < 2 * len(predictors):
        return None

    X = df[predictors].values.astype(float)

    try:
        model = ConditionalLogit(y, X, groups=groups)
        result = model.fit(disp=0)
    except Exception:
        return None

    coefs = dict(zip(predictors, result.params, strict=True))
    bse = dict(zip(predictors, result.bse, strict=True))
    pvals = dict(zip(predictors, result.pvalues, strict=True))

    return {
        **{f"coef_{k}": v for k, v in coefs.items()},
        **{f"bse_{k}": v for k, v in bse.items()},
        **{f"pval_{k}": v for k, v in pvals.items()},
        "n_events": n_events,
        "n_rows": len(df),
        "n_params": len(predictors),
        "llf": float(result.llf),
        "aic": float(2 * len(predictors) - 2 * result.llf),
        "bic": float(len(predictors) * np.log(n_events) - 2 * result.llf),
    }


# ── Legacy binomial GLM (kept for backward compatibility) ──


def fit_glm(
    long_df: pd.DataFrame,
    predictors: list[str],
    alpha: float = ALPHA,
) -> dict | None:
    """Fit L2-regularized logistic regression on long-format data.

    .. deprecated::
        Use ``fit_conditional_poisson`` (primary) or
        ``fit_conditional_logit`` (robustness) instead.

    Parameters
    ----------
    long_df : pd.DataFrame
        Long-format data with ``replay_here`` response.
    predictors : list[str]
        Column names to use as predictors.
    alpha : float
        L2 regularization strength.

    Returns
    -------
    dict | None
        Coefficients and fit statistics. None if insufficient data.
    """
    y = long_df["replay_here"].values

    if y.sum() < 3 or (1 - y).sum() < 3:
        return None

    if len(predictors) == 0:
        return _fit_null_model(y)

    X_raw = long_df[predictors].values.astype(float)
    X = sm.add_constant(X_raw)

    if X.shape[0] < 2 * len(predictors):
        return None

    try:
        model = sm.GLM(y, X, family=sm.families.Binomial())
        result = model.fit_regularized(alpha=alpha, L1_wt=0, refit=False)
        coefs = dict(zip(["intercept"] + predictors, result.params, strict=True))

        y_pred = result.predict(X)
        ll_model = _log_likelihood(y, y_pred)
        ll_null = _null_log_likelihood(y)

        n_params = len(predictors) + 1
        aic = 2 * n_params - 2 * ll_model
        bic = n_params * np.log(len(y)) - 2 * ll_model

        diagnostics = _compute_binomial_diagnostics(
            y, y_pred, X, X_raw, predictors, model
        )

        return {
            **{f"coef_{k}": v for k, v in coefs.items()},
            "n_rows": len(long_df),
            "n_params": n_params,
            "ll": float(ll_model),
            "ll_null": float(ll_null),
            "aic": float(aic),
            "bic": float(bic),
            "pseudo_r2": (
                1 - ll_model / ll_null
                if np.isfinite(ll_null) and ll_null != 0
                else np.nan
            ),
            "brier": float(np.mean((y - y_pred) ** 2)),
            "max_abs_coef": float(np.max(np.abs(list(coefs.values())))),
            **diagnostics,
        }
    except Exception:
        return None


# ── Internal helpers ──


def _fit_null_model(y: np.ndarray) -> dict:
    """Fit intercept-only (null) model for binomial GLM."""
    p = y.mean()
    ll = _null_log_likelihood(y)
    y_pred = np.full_like(y, p, dtype=float)
    brier = float(np.mean((y - y_pred) ** 2))

    return {
        "coef_intercept": float(np.log(p / (1 - p))) if 0 < p < 1 else 0.0,
        "n_rows": len(y),
        "n_params": 1,
        "ll": float(ll),
        "ll_null": float(ll),
        "aic": float(2 - 2 * ll),
        "bic": float(np.log(len(y)) - 2 * ll),
        "pseudo_r2": 0.0,
        "brier": brier,
        "max_abs_coef": 0.0,
        "cond_number": 1.0,
        "resid_mean": 0.0,
        "resid_std": float(np.std(y - y_pred)),
        "resid_skew": 0.0,
        "resid_abs_max": float(np.max(np.abs(y - y_pred))),
        "calib_slope": np.nan,
    }


def _log_likelihood(y: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute log-likelihood for binary outcomes."""
    eps = 1e-10
    return float(np.sum(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps)))


def _null_log_likelihood(y: np.ndarray) -> float:
    """Compute log-likelihood for intercept-only model."""
    p = y.mean()
    eps = 1e-10
    return float(np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))


def _compute_conditional_diagnostics(
    X: np.ndarray,
    predictors: list[str],
) -> dict:
    """Compute diagnostics for conditional model (VIF, condition number)."""
    result = {}

    # Condition number
    if X.shape[1] > 0:
        result["cond_number"] = float(np.linalg.cond(X.T @ X))

    # VIF (add constant for VIF computation)
    X_with_const = sm.add_constant(X)
    for i, pred in enumerate(predictors):
        try:
            result[f"vif_{pred}"] = variance_inflation_factor(X_with_const, i + 1)
        except Exception:
            result[f"vif_{pred}"] = np.nan

    return result


def _compute_binomial_diagnostics(
    y: np.ndarray,
    y_pred: np.ndarray,
    X: np.ndarray,
    X_raw: np.ndarray,
    predictors: list[str],
    model: sm.GLM,
) -> dict:
    """Compute VIF, condition number, residuals, calibration for binomial GLM."""
    result = {}

    for i, pred in enumerate(predictors):
        try:
            result[f"vif_{pred}"] = variance_inflation_factor(X, i + 1)
        except Exception:
            result[f"vif_{pred}"] = np.nan

    result["cond_number"] = float(np.linalg.cond(X.T @ X))

    for i, p1 in enumerate(predictors):
        for j, p2 in enumerate(predictors):
            if j <= i:
                continue
            v1, v2 = X_raw[:, i], X_raw[:, j]
            both = ((v1 == 1) & (v2 == 1)).sum()
            either = ((v1 == 1) | (v2 == 1)).sum()
            result[f"jaccard_{p1}__{p2}"] = both / either if either > 0 else 0.0

    try:
        unreg = model.fit(disp=0)
        vcov = unreg.cov_params()
        d = np.sqrt(np.diag(vcov))
        corr_mat = vcov / (d[:, None] * d[None, :])
        for i, p1 in enumerate(predictors):
            for j, p2 in enumerate(predictors):
                if j <= i:
                    continue
                result[f"coef_corr_{p1}__{p2}"] = float(corr_mat[i + 1, j + 1])
    except Exception:
        pass

    eps = 1e-10
    deviance_resid = np.sign(y - y_pred) * np.sqrt(
        -2 * (y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))
    )
    result["resid_mean"] = float(np.mean(deviance_resid))
    result["resid_std"] = float(np.std(deviance_resid))
    result["resid_skew"] = float(pd.Series(deviance_resid).skew())
    result["resid_abs_max"] = float(np.max(np.abs(deviance_resid)))

    pred_bins = np.linspace(0, 1, 6)
    bin_idx = np.digitize(y_pred, pred_bins) - 1
    calib_slope = np.nan
    if len(np.unique(bin_idx)) >= 3:
        obs_rate = [
            y[bin_idx == b].mean()
            for b in range(len(pred_bins) - 1)
            if (bin_idx == b).sum() > 0
        ]
        pred_rate = [
            y_pred[bin_idx == b].mean()
            for b in range(len(pred_bins) - 1)
            if (bin_idx == b).sum() > 0
        ]
        if len(obs_rate) >= 2:
            slope, _, _, _, _ = linregress(pred_rate, obs_rate)
            calib_slope = float(slope)
    result["calib_slope"] = calib_slope

    return result
