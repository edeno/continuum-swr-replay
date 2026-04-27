"""Compare nested models via AIC, BIC, and likelihood ratio tests.

AIC is the primary comparison metric. BIC is secondary. LRT is valid
for unregularized fits only.

Provides functions for comparing consecutive models in the hierarchy
(M0 vs M1, M1 vs M2, etc.) both at the per-session level and aggregated
across sessions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2, wilcoxon

from .model_specs import MODEL_HIERARCHY


def compare_nested_models(
    parent_results: pd.DataFrame,
    child_results: pd.DataFrame,
    session_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compare two nested models across sessions.

    Merges per-session fit results from parent and child models, computes
    delta-AIC, delta-BIC, and per-session likelihood ratio statistics.

    Parameters
    ----------
    parent_results : pd.DataFrame
        Per-session results from the simpler (parent) model. Must have
        columns: ll, aic, bic, n_params, plus session identifiers.
    child_results : pd.DataFrame
        Per-session results from the more complex (child) model.
    session_cols : list[str] | None
        Session grouping columns.

    Returns
    -------
    pd.DataFrame
        One row per session with delta_aic, delta_bic, lr_stat, lr_pvalue.
    """
    if session_cols is None:
        session_cols = ["animal", "day", "epoch"]

    # Accept both "ll" (legacy binomial) and "llf" (conditional Poisson)
    ll_col = "llf" if "llf" in parent_results.columns else "ll"

    fit_cols = [ll_col, "aic", "bic", "n_params"]
    # Include n_events if available (for LRT validity check)
    if "n_events" in parent_results.columns and "n_events" in child_results.columns:
        fit_cols = fit_cols + ["n_events"]

    merged = parent_results[session_cols + fit_cols].merge(
        child_results[session_cols + fit_cols],
        on=session_cols,
        suffixes=("_parent", "_child"),
    )

    if len(merged) == 0:
        return pd.DataFrame()

    ll_parent = f"{ll_col}_parent"
    ll_child = f"{ll_col}_child"

    # Delta AIC/BIC and LRT require parent and child to be fit on
    # the same observations. Callers must ensure this by pre-filtering
    # to a common complete-case event set before fitting.
    merged["delta_aic"] = merged["aic_parent"] - merged["aic_child"]
    merged["delta_bic"] = merged["bic_parent"] - merged["bic_child"]

    merged["lr_stat"] = -2 * (merged[ll_parent] - merged[ll_child])
    merged["df_diff"] = merged["n_params_child"] - merged["n_params_parent"]
    merged["lr_pvalue"] = chi2.sf(
        merged["lr_stat"].clip(lower=0), df=merged["df_diff"].clip(lower=1)
    )

    # Invalidate all comparison metrics when parent and child used
    # different event counts (indicates the caller did not pre-filter
    # to a common complete-case event set).
    if "n_events_parent" in merged.columns and "n_events_child" in merged.columns:
        different_n = merged["n_events_parent"] != merged["n_events_child"]
        if different_n.any():
            import logging

            logging.warning(
                "%d/%d sessions have mismatched event counts between "
                "parent and child — invalidating comparisons. "
                "Pre-filter to a common event set before fitting.",
                different_n.sum(),
                len(merged),
            )
        for col in ["delta_aic", "delta_bic", "lr_stat", "lr_pvalue"]:
            merged.loc[different_n, col] = np.nan

    return merged


def summarize_comparison(
    comparison_df: pd.DataFrame,
    parent_name: str,
    child_name: str,
) -> dict:
    """Summarize a nested model comparison across sessions.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output of compare_nested_models.
    parent_name : str
        Name of the parent model (e.g., "M0").
    child_name : str
        Name of the child model (e.g., "M1").

    Returns
    -------
    dict
        Summary statistics including median delta-AIC, fraction of sessions
        preferring each model, and Wilcoxon test on delta-AIC.
    """
    if len(comparison_df) == 0:
        return {"parent": parent_name, "child": child_name, "n_sessions": 0}

    daic = comparison_df["delta_aic"]
    dbic = comparison_df["delta_bic"]
    lr_sig = (comparison_df["lr_pvalue"] < 0.05).sum()

    result = {
        "parent": parent_name,
        "child": child_name,
        "n_sessions": len(comparison_df),
        "df_diff": int(comparison_df["df_diff"].median()),
        "delta_aic_median": float(daic.median()),
        "delta_aic_q25": float(daic.quantile(0.25)),
        "delta_aic_q75": float(daic.quantile(0.75)),
        "delta_bic_median": float(dbic.median()),
        "child_better_aic": int((daic > 2).sum()),
        "parent_better_aic": int((daic < -2).sum()),
        "child_better_bic": int((dbic > 2).sum()),
        "lrt_significant": int(lr_sig),
    }

    # Wilcoxon signed-rank on delta-AIC. Drop NaN sessions first —
    # scipy's default nan_policy="propagate" returns NaN whenever any
    # input is NaN, which silently invalidates the omnibus p whenever
    # the mismatched-n_events guard marked some sessions NaN.
    daic_valid = daic.dropna()
    if len(daic_valid) >= 5:
        try:
            _, p = wilcoxon(daic_valid)
            result["wilcoxon_p"] = float(p)
        except ValueError:
            result["wilcoxon_p"] = np.nan
    else:
        result["wilcoxon_p"] = np.nan

    return result


def run_full_hierarchy_comparison(
    all_results: pd.DataFrame,
    session_cols: list[str] | None = None,
    hierarchy: dict | None = None,
) -> pd.DataFrame:
    """Compare all consecutive models in the hierarchy.

    Parameters
    ----------
    all_results : pd.DataFrame
        Combined per-session results with a ``model`` column indicating
        which model each row comes from.
    session_cols : list[str] | None
        Session grouping columns.
    hierarchy : dict | None
        Model hierarchy dict mapping model name → ModelSpec. Defaults
        to :data:`MODEL_HIERARCHY` (arm-level M0–M6); pass
        :data:`MODEL_PATCH` to compare the patch hierarchy.

    Returns
    -------
    pd.DataFrame
        One row per comparison step.
    """
    if hierarchy is None:
        hierarchy = MODEL_HIERARCHY
    comparisons = []

    for model_name, spec in hierarchy.items():
        if spec.parent is None:
            continue

        parent_df = all_results[all_results["model"] == spec.parent]
        child_df = all_results[all_results["model"] == model_name]

        if len(parent_df) == 0 or len(child_df) == 0:
            continue

        comp = compare_nested_models(parent_df, child_df, session_cols=session_cols)
        summary = summarize_comparison(comp, spec.parent, model_name)
        comparisons.append(summary)

    return pd.DataFrame(comparisons)
