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
from statsmodels.discrete.conditional_models import (
    ConditionalLogit,
    ConditionalPoisson,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)


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
        Coefficients, standard errors, p-values, fit statistics.
        None if fitting fails or insufficient data.
    """
    if len(predictors) == 0:
        return fit_null_model_conditional(
            long_df, event_col, response_col, n_arms=n_categories
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

    # Reject rank-deficient designs (e.g. a duplicated or perfectly
    # collinear predictor). statsmodels would otherwise return a fit with
    # huge standard errors and finite-but-meaningless p-values / AIC.
    if X.shape[1] > 0 and np.linalg.matrix_rank(X) < X.shape[1]:
        logger.warning(
            "fit_conditional_poisson: rank-deficient design (rank %d < %d "
            "predictors); not fitting %s",
            int(np.linalg.matrix_rank(X)),
            X.shape[1],
            predictors,
        )
        return None

    try:
        model = ConditionalPoisson(y, X, groups=groups)
        result = model.fit(disp=0)
    except Exception:
        logger.warning(
            "ConditionalPoisson failed for predictors %s", predictors, exc_info=True
        )
        return None

    coefs = dict(zip(predictors, result.params, strict=True))
    bse = dict(zip(predictors, result.bse, strict=True))
    pvals = dict(zip(predictors, result.pvalues, strict=True))

    llf = result.llf
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
        "n_events": n_events,
        "n_rows": len(df),
        "n_params": n_params,
        "llf": float(llf),
        "llf_null": float(llf_null),
        "aic": float(aic),
        "bic": float(bic),
        "pseudo_r2": float(pseudo_r2),
    }

    # Diagnostics
    try:
        diag = _compute_conditional_diagnostics(X, predictors)
        output.update(diag)
    except Exception as exc:
        logger.debug("VIF/conditional diagnostics skipped: %s", exc)

    return output


def fit_null_model_conditional(
    long_df: pd.DataFrame,
    event_col: str = "event_id",
    response_col: str = "arm_count_hard",
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
        "arm"
        if "arm" in long_df.columns
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
            df.sort_values([event_col, response_col, arm_col], ascending=[True, False, True])
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

    # Reject rank-deficient designs (see fit_conditional_poisson) — a
    # collinear design yields finite-but-meaningless coefficients/SEs.
    if X.shape[1] > 0 and np.linalg.matrix_rank(X) < X.shape[1]:
        logger.warning(
            "fit_conditional_logit: rank-deficient design (rank %d < %d "
            "predictors); not fitting %s",
            int(np.linalg.matrix_rank(X)),
            X.shape[1],
            predictors,
        )
        return None

    try:
        model = ConditionalLogit(y, X, groups=groups)
        result = model.fit(disp=0)
    except Exception:
        logger.warning(
            "ConditionalLogit failed for predictors %s", predictors, exc_info=True
        )
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


# ── Internal helpers ──


def _drop_zero_count_events(
    df: pd.DataFrame, event_col: str, response_col: str
) -> pd.DataFrame:
    """Drop events whose ``response_col`` sums to 0 across all rows.

    The conditional likelihood is undefined (0/0) for such events;
    fit_conditional_logit's idxmax would silently pick an arbitrary
    "dominant" arm otherwise.
    """
    event_totals = df.groupby(event_col)[response_col].transform("sum")
    return df[event_totals > 0].reset_index(drop=True)


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
