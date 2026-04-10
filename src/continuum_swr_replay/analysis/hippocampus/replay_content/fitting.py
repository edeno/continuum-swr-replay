"""Per-session GLM fitting for the replay content model hierarchy.

Fits L2-regularized logistic regression per session, returning coefficients
and comprehensive diagnostics for each model in the hierarchy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import linregress
from statsmodels.stats.outliers_influence import variance_inflation_factor

ALPHA = 0.01  # L2 regularization strength


def fit_glm(
    long_df: pd.DataFrame,
    predictors: list[str],
    alpha: float = ALPHA,
) -> dict | None:
    """Fit L2-regularized logistic regression on long-format data.

    Handles the null model (empty predictors list) as a special case
    with intercept only.

    Parameters
    ----------
    long_df : pd.DataFrame
        Long-format data with ``replay_here`` response and predictor columns.
    predictors : list[str]
        Column names to use as predictors. Empty list fits intercept-only.
    alpha : float
        L2 regularization strength.

    Returns
    -------
    dict | None
        Coefficients, fit quality, and diagnostics. None if insufficient data.
    """
    y = long_df["replay_here"].values

    if y.sum() < 3 or (1 - y).sum() < 3:
        return None

    # Null model: intercept only
    if len(predictors) == 0:
        return _fit_null_model(y)

    # Check data sufficiency
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

        # Diagnostics
        diagnostics = _compute_diagnostics(y, y_pred, X, X_raw, predictors, model)

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


def _fit_null_model(y: np.ndarray) -> dict:
    """Fit intercept-only (null) model."""
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


def _compute_diagnostics(
    y: np.ndarray,
    y_pred: np.ndarray,
    X: np.ndarray,
    X_raw: np.ndarray,
    predictors: list[str],
    model: sm.GLM,
) -> dict:
    """Compute VIF, condition number, residuals, calibration."""
    result = {}

    # VIF
    for i, pred in enumerate(predictors):
        try:
            result[f"vif_{pred}"] = variance_inflation_factor(X, i + 1)
        except Exception:
            result[f"vif_{pred}"] = np.nan

    # Condition number
    result["cond_number"] = float(np.linalg.cond(X.T @ X))

    # Predictor overlap (Jaccard) for binary predictors
    for i, p1 in enumerate(predictors):
        for j, p2 in enumerate(predictors):
            if j <= i:
                continue
            v1, v2 = X_raw[:, i], X_raw[:, j]
            both = ((v1 == 1) & (v2 == 1)).sum()
            either = ((v1 == 1) | (v2 == 1)).sum()
            result[f"jaccard_{p1}__{p2}"] = both / either if either > 0 else 0.0

    # Coefficient correlations from unregularized fit
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

    # Deviance residuals
    eps = 1e-10
    deviance_resid = np.sign(y - y_pred) * np.sqrt(
        -2 * (y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))
    )
    result["resid_mean"] = float(np.mean(deviance_resid))
    result["resid_std"] = float(np.std(deviance_resid))
    result["resid_skew"] = float(pd.Series(deviance_resid).skew())
    result["resid_abs_max"] = float(np.max(np.abs(deviance_resid)))

    # Calibration
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


def fit_mixed_effects(
    long_all: pd.DataFrame,
    predictors: list[str],
    session_cols: list[str] | None = None,
) -> dict | None:
    """Fit mixed-effects logistic regression across all sessions.

    Uses statsmodels BinomialBayesMixedGLM with random intercept and
    random is_current slope per session.

    Parameters
    ----------
    long_all : pd.DataFrame
        Concatenated long-format data from all sessions.
    predictors : list[str]
        Fixed-effect predictor columns.
    session_cols : list[str] | None
        Session grouping columns.

    Returns
    -------
    dict | None
        Fixed effects, random effects variance, log-likelihood.
        None if fitting fails or insufficient data.
    """
    import logging

    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    if session_cols is None:
        session_cols = ["animal", "day", "epoch"]

    if len(predictors) == 0:
        return None

    long_all = long_all.copy()
    long_all["session"] = long_all[session_cols].astype(str).agg("_".join, axis=1)

    # Drop NaN in required columns
    required = ["replay_here"] + predictors
    long_clean = long_all.dropna(subset=required)

    n_sessions = long_clean["session"].nunique()
    n_rows = len(long_clean)

    if n_rows < 1000 or n_sessions < 5:
        return None

    # Build formula from predictor list
    formula = "replay_here ~ " + " + ".join(predictors)

    # Random effects: intercept per session
    vc_formulas = {"session_intercept": "0 + C(session)"}
    if "is_current" in predictors:
        vc_formulas["session_current_slope"] = "0 + C(session):is_current"

    try:
        model = BinomialBayesMixedGLM.from_formula(
            formula, vc_formulas, data=long_clean
        )
        result = model.fit_map()

        fe_params = result.fe_mean
        fe_sd = result.fe_sd
        fe_names = list(model.fep_names)

        # Random effects
        vc_names = list(vc_formulas.keys())
        vcp_mean = result.vcp_mean
        vcp_sd = result.vcp_sd

        # Log-likelihood using full posterior mean prediction
        # (fixed effects + random effects), not just fixed effects.
        # result.predict() only uses fixed effects, so we compute manually.
        y = long_clean["replay_here"].values
        linear_pred = model.exog @ result.fe_mean + model.exog_vc @ result.vc_mean
        y_pred = 1.0 / (1.0 + np.exp(-linear_pred))
        ll = _log_likelihood(y, y_pred)

        # Parameter count includes fixed effects + variance components
        n_vc_params = len(vc_formulas)
        n_total_params = len(fe_names) + n_vc_params
        aic = 2 * n_total_params - 2 * ll
        bic = n_total_params * np.log(n_rows) - 2 * ll

        return {
            "fe_names": fe_names,
            "fe_mean": list(fe_params),
            "fe_sd": list(fe_sd),
            "vc_names": vc_names,
            "vcp_mean": list(vcp_mean),
            "vcp_sd": list(vcp_sd),
            "vc_sd_exp": {
                name: float(np.exp(m))
                for name, m in zip(vc_names, vcp_mean, strict=True)
            },
            "n_rows": n_rows,
            "n_sessions": n_sessions,
            "n_fe_params": len(fe_names),
            "n_total_params": n_total_params,
            "ll": float(ll),
            "aic": float(aic),
            "bic": float(bic),
            "formula": formula,
        }

    except Exception as e:
        logging.error("Mixed-effects model failed: %s", e)
        return None
