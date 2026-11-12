"""Visualization functions for replay content model hierarchy results.

All functions accept pre-computed results (DataFrames or dicts) and return
matplotlib figures. No model fitting or data loading happens here.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from continuum_swr_replay.analysis.replay.model_specs import (
    MODEL_HIERARCHY,
)
from continuum_swr_replay.analysis.statistics import benjamini_hochberg, signed_rank_p
from continuum_swr_replay.visualization.palettes import COLORS


def plot_model_hierarchy_comparison(
    comparison_df: pd.DataFrame,
    ax: plt.Axes | None = None,
    scale_label: str | None = None,
) -> plt.Figure | None:
    """Plot ΔAIC across the model hierarchy.

    Bar chart showing median ΔAIC at each step (M0→M1, M1→M2, ...),
    with error bars for IQR and annotations for session counts.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Output of run_full_hierarchy_comparison.
    ax : plt.Axes | None
        Axes to plot on. If None, creates a new figure.
    scale_label : str | None
        How the plotted ΔAIC must be read, from the fit's estimator (e.g.
        ``"event-scale (quasi)"`` for a fraction fit, ``"timebin-scale"`` for a
        count fit). Named on the axis when given: the two scales are not
        comparable, and a fraction fit's AIC is a quasi-likelihood quantity.

    Returns
    -------
    plt.Figure | None
        Figure if created, None if ax was provided.
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
    else:
        fig = None

    steps = [f"{row['parent']}→{row['child']}" for _, row in comparison_df.iterrows()]
    daic = comparison_df["delta_aic_median"].values
    q25 = comparison_df.get("delta_aic_q25", pd.Series(dtype=float)).values
    q75 = comparison_df.get("delta_aic_q75", pd.Series(dtype=float)).values

    colors = [COLORS["Non-Local"] if d > 0 else COLORS["SWR"] for d in daic]
    ax.bar(range(len(steps)), daic, color=colors, alpha=0.7, edgecolor="none")

    # Error bars if IQR available
    if len(q25) == len(daic) and not np.all(np.isnan(q25)):
        yerr_low = daic - q25
        yerr_high = q75 - daic
        ax.errorbar(
            range(len(steps)),
            daic,
            yerr=[yerr_low, yerr_high],
            fmt="none",
            color="black",
            capsize=3,
            linewidth=0.8,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(steps, rotation=45, ha="right")
    ax.set_ylabel(
        "ΔAIC (positive = child better)"
        if scale_label is None
        else f"ΔAIC, {scale_label} (positive = child better)"
    )
    ax.set_title("Model Hierarchy Comparison")

    # Annotate with child-wins / total
    for i, (_, row) in enumerate(comparison_df.iterrows()):
        n = row.get("n_sessions", 0)
        wins = row.get("child_better_aic", 0)
        ax.text(
            i,
            daic[i] + (1 if daic[i] >= 0 else -2),
            f"{wins}/{n}",
            ha="center",
            fontsize=5,
            va="bottom" if daic[i] >= 0 else "top",
        )

    sns.despine(ax=ax)
    return fig


def plot_coefficient_forest(
    model_results: pd.DataFrame,
    predictors: list[str],
    model_name: str = "",
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Forest plot of per-session coefficients for one model.

    Parameters
    ----------
    model_results : pd.DataFrame
        Per-session fit results for one model.
    predictors : list[str]
        Predictor names.
    model_name : str
        Model name for title.
    ax : plt.Axes | None
        Axes to plot on.

    Returns
    -------
    plt.Figure | None
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(
            figsize=(4, 0.4 * len(predictors) + 1), constrained_layout=True
        )
    else:
        fig = None

    # Collect coefficients and test against 0
    raw_pvals = []
    medians = []
    q25s = []
    q75s = []
    labels = []

    for pred in predictors:
        col = f"coef_{pred}"
        if col not in model_results.columns:
            continue
        vals = model_results[col].dropna()
        if len(vals) < 3:
            continue

        medians.append(vals.median())
        q25s.append(vals.quantile(0.25))
        q75s.append(vals.quantile(0.75))
        labels.append(pred.replace("is_", "").replace("_x_", "×"))

        raw_pvals.append(signed_rank_p(vals))

    if not labels:
        return fig

    # FDR correction across the plotted predictors that have a p
    corrected = benjamini_hochberg(raw_pvals)

    y_pos = range(len(labels))
    medians = np.array(medians)
    q25s = np.array(q25s)
    q75s = np.array(q75s)

    ax.barh(
        y_pos,
        medians,
        xerr=[medians - q25s, q75s - medians],
        color=COLORS["Non-Local"],
        alpha=0.7,
        edgecolor="none",
        capsize=3,
        height=0.6,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Log-odds coefficient (median ± IQR)")
    ax.set_title(f"{model_name} Coefficients" if model_name else "Coefficients")

    # Significance stars
    for i, p_corr in enumerate(corrected):
        if p_corr < 0.001:
            sig = "***"
        elif p_corr < 0.01:
            sig = "**"
        elif p_corr < 0.05:
            sig = "*"
        else:
            continue
        x_pos = q75s[i] + 0.02 if medians[i] >= 0 else q25s[i] - 0.02
        ax.text(
            x_pos,
            i,
            sig,
            fontsize=7,
            va="center",
            ha="left" if medians[i] >= 0 else "right",
        )

    ax.invert_yaxis()
    sns.despine(ax=ax)
    return fig


def plot_pseudo_r2_progression(
    all_results: pd.DataFrame,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Plot pseudo-R² progression across the model hierarchy.

    Parameters
    ----------
    all_results : pd.DataFrame
        Combined per-session results with ``model`` column.
    ax : plt.Axes | None
        Axes to plot on.

    Returns
    -------
    plt.Figure | None
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    else:
        fig = None

    model_names = [m for m in MODEL_HIERARCHY if (all_results["model"] == m).any()]
    medians = []
    q25s = []
    q75s = []

    for m in model_names:
        vals = all_results.loc[all_results["model"] == m, "pseudo_r2"].dropna()
        medians.append(vals.median())
        q25s.append(vals.quantile(0.25))
        q75s.append(vals.quantile(0.75))

    medians = np.array(medians)
    q25s = np.array(q25s)
    q75s = np.array(q75s)

    ax.fill_between(
        range(len(model_names)), q25s, q75s, alpha=0.3, color=COLORS["Non-Local"]
    )
    ax.plot(
        range(len(model_names)),
        medians,
        "o-",
        color=COLORS["Non-Local"],
        markersize=5,
        linewidth=1.5,
    )
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names)
    ax.set_ylabel("Pseudo R²")
    ax.set_title("Model Fit Quality Progression")
    sns.despine(ax=ax)

    return fig


def plot_behavior_prediction_results(
    cv_results: pd.DataFrame,
    chance_level: float = 1 / 6,
    ax: plt.Axes | None = None,
) -> plt.Figure | None:
    """Plot behavior prediction CV accuracy across models.

    Parameters
    ----------
    cv_results : pd.DataFrame
        Output of cross_validate_choice_prediction.
    chance_level : float
        Chance accuracy (1/n_wells).
    ax : plt.Axes | None
        Axes to plot on.

    Returns
    -------
    plt.Figure | None
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 3), constrained_layout=True)
    else:
        fig = None

    models = ["baseline", "replay_all", "replay_swr", "replay_noswr"]
    model_labels = [
        "Baseline\n(location)",
        "All\nreplay",
        "SWR\nreplay",
        "Non-SWR\nreplay",
    ]
    palette = [
        "#999999",
        COLORS["Non-Local"],
        COLORS["SWR"],
        COLORS["Non-Local"],
    ]

    plot_data = []
    for model_name, label, color in zip(models, model_labels, palette, strict=True):
        acc_col = f"{model_name}_accuracy"
        if acc_col in cv_results.columns:
            vals = cv_results[acc_col].dropna()
            for v in vals:
                plot_data.append({"model": label, "accuracy": v, "color": color})

    pdf = pd.DataFrame(plot_data)
    if len(pdf) == 0:
        return fig

    sns.boxplot(
        data=pdf,
        x="model",
        y="accuracy",
        palette=dict(zip(model_labels, palette, strict=True)),
        ax=ax,
        width=0.5,
        fliersize=1,
    )
    ax.axhline(
        chance_level,
        color="black",
        linestyle="--",
        linewidth=0.8,
        label=f"Chance ({chance_level:.3f})",
    )
    ax.set_ylabel("CV Accuracy")
    ax.set_xlabel("")
    ax.set_title("Does Replay Predict Next Choice?")
    ax.legend(fontsize=5, frameon=False)
    sns.despine(ax=ax)

    return fig
