"""Compare nested models via AIC, BIC, and likelihood ratio tests.

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

    merged = parent_results[session_cols + ["ll", "aic", "bic", "n_params"]].merge(
        child_results[session_cols + ["ll", "aic", "bic", "n_params"]],
        on=session_cols,
        suffixes=("_parent", "_child"),
    )

    if len(merged) == 0:
        return pd.DataFrame()

    # Delta AIC/BIC: positive = child model better
    merged["delta_aic"] = merged["aic_parent"] - merged["aic_child"]
    merged["delta_bic"] = merged["bic_parent"] - merged["bic_child"]

    # Likelihood ratio test
    merged["lr_stat"] = -2 * (merged["ll_parent"] - merged["ll_child"])
    df_diff = merged["n_params_child"].iloc[0] - merged["n_params_parent"].iloc[0]
    merged["df_diff"] = int(df_diff)
    merged["lr_pvalue"] = chi2.sf(merged["lr_stat"].clip(lower=0), df=max(df_diff, 1))

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
        "df_diff": int(comparison_df["df_diff"].iloc[0]),
        "delta_aic_median": float(daic.median()),
        "delta_aic_q25": float(daic.quantile(0.25)),
        "delta_aic_q75": float(daic.quantile(0.75)),
        "delta_bic_median": float(dbic.median()),
        "child_better_aic": int((daic > 2).sum()),
        "parent_better_aic": int((daic < -2).sum()),
        "child_better_bic": int((dbic > 2).sum()),
        "lrt_significant": int(lr_sig),
    }

    # Wilcoxon signed-rank on delta-AIC
    if len(daic) >= 5:
        try:
            _, p = wilcoxon(daic)
            result["wilcoxon_p"] = float(p)
        except ValueError:
            result["wilcoxon_p"] = np.nan
    else:
        result["wilcoxon_p"] = np.nan

    return result


def run_full_hierarchy_comparison(
    all_results: pd.DataFrame,
    session_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compare all consecutive models in the hierarchy.

    Parameters
    ----------
    all_results : pd.DataFrame
        Combined per-session results with a ``model`` column indicating
        which model each row comes from.
    session_cols : list[str] | None
        Session grouping columns.

    Returns
    -------
    pd.DataFrame
        One row per comparison step (M0→M1, M1→M2, ..., M6→M7).
    """
    comparisons = []

    for model_name, spec in MODEL_HIERARCHY.items():
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
