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
from scipy.stats import chi2

from continuum_swr_replay.analysis.statistics import (
    session_median_statistics,
    signed_rank_p,
)
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

from .model_specs import MODEL_HIERARCHY

# Fit-scale metadata carried from the per-session frames onto the comparison and
# its summary: which estimator produced the numbers, which response column it was
# fit on, whether its log-likelihood carries the response's normalizing
# constants, which time clock a per-bin rung's rows are on, and which stop
# population it was fit over. All five must agree between parent and child for a
# comparison to mean anything, so ``summarize_comparison`` refuses a step where
# they do not.
METADATA_COLUMNS = (
    "estimator",
    "response_col",
    "llf_includes_constants",
    "time_clock",
    "stop_cohort",
)

#: Metadata columns whose disagreement invalidates a per-session comparison, and
#: why. ``llf_includes_constants``: the Dirichlet-multinomial likelihood carries
#: the response's normalizing constants and a statsmodels conditional likelihood
#: does not. ``time_clock``: a per-bin rung anchored on the arrival clock is fit
#: on the arrival window, a strict subset of the frozen cohort's rows, so a
#: cross-clock delta measures the window rather than the rungs. The row-count
#: guard below would catch that too, but the clock says WHY.
INVALIDATING_METADATA_COLUMNS = {
    "llf_includes_constants": (
        "compares a likelihood that includes the response's normalizing "
        "constants against one that does not — equalize the constants (see "
        "dirichlet_multinomial.multinomial_log_constant) before differencing "
        "information criteria across estimators"
    ),
    "time_clock": (
        "compares rungs anchored on different time clocks, which are fit to "
        "different rows (an arrival-clock rung is fit on the arrival window) — "
        "compare an arrival rung with its own clock's null instead"
    ),
    "stop_cohort": (
        "compares rungs fit on different stop populations (at-well stops "
        "against all stops or in-transit stops), which are different sets of "
        "rows and different science — compare a rung with a null fit on its "
        "own cohort instead"
    ),
}


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
        columns: llf, aic, bic, n_params, plus session identifiers.
    child_results : pd.DataFrame
        Per-session results from the more complex (child) model.
    session_cols : list[str] | None
        Session grouping columns.

    Returns
    -------
    pd.DataFrame
        One row per session with delta_aic, delta_bic, lr_stat, lr_pvalue.
        ``delta_aic`` / ``delta_bic`` / ``lr_stat`` / ``lr_pvalue`` are NaN for
        any session whose parent and child report different ``n_events``,
        ``n_bins``, or ``n_stops`` — rungs fit to different rows are not
        comparable — and likewise for any session whose two rungs disagree on
        one of :data:`INVALIDATING_METADATA_COLUMNS`: on
        ``llf_includes_constants`` because the Dirichlet-multinomial likelihood
        carries the multinomial coefficient and a statsmodels conditional
        likelihood does not, so differencing them measures that constant rather
        than the rungs, on ``time_clock`` because a per-bin rung anchored on
        the arrival clock is fit on the arrival window rather than the whole
        frozen cohort, and on ``stop_cohort`` because an at-well fit and an
        all-stops fit are about different populations.
    """
    if session_cols is None:
        session_cols = list(SESSION_COLUMNS)

    # Cohort-size columns present in BOTH frames are carried through so the
    # guard below can invalidate a comparison whose rungs were fit to different
    # rows. Which one is the unit depends on the family: events (conditional
    # arm/patch), bins (per-bin conditional intensity), stops (per-stop rate).
    count_cols = [
        col
        for col in ("n_events", "n_bins", "n_stops")
        if col in parent_results.columns and col in child_results.columns
    ]
    # The scale the two rungs were fit at travels with the comparison. A delta
    # AIC means one thing when each event contributed a single observation and
    # another when it contributed thirty autocorrelated timebins, and a summary
    # that does not carry the estimator cannot say which.
    meta_cols = [
        col
        for col in METADATA_COLUMNS
        if col in parent_results.columns and col in child_results.columns
    ]
    fit_cols = ["llf", "aic", "bic", "n_params", *count_cols, *meta_cols]

    merged = parent_results[session_cols + fit_cols].merge(
        child_results[session_cols + fit_cols],
        on=session_cols,
        suffixes=("_parent", "_child"),
    )

    if len(merged) == 0:
        return pd.DataFrame()

    ll_parent = "llf_parent"
    ll_child = "llf_child"

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

    # Invalidate all comparison metrics when parent and child used different
    # numbers of events, bins, or stops (the caller did not freeze one cohort
    # for the ladder before fitting). AIC / BIC / deviance all scale with the
    # number of rows, so a rung fit to fewer rows wins mechanically.
    if count_cols:
        different_n = pd.Series(False, index=merged.index)
        mismatched: list[str] = []
        for col in count_cols:
            differs = merged[f"{col}_parent"] != merged[f"{col}_child"]
            if differs.any():
                mismatched.append(f"{col} ({int(differs.sum())})")
            different_n = different_n | differs
        if different_n.any():
            import logging

            logging.warning(
                "%d/%d sessions have mismatched cohort sizes between parent and "
                "child [%s] — invalidating comparisons. Freeze one cohort for "
                "the ladder before fitting.",
                int(different_n.sum()),
                len(merged),
                ", ".join(mismatched),
            )
        for col in ["delta_aic", "delta_bic", "lr_stat", "lr_pvalue"]:
            merged.loc[different_n, col] = np.nan

    # Invalidate the same metrics when the two rungs disagree on a metadata
    # column that makes their information criteria incommensurable. Both sides
    # must actually say so: a legacy frame that records nothing is unknown, not
    # different.
    for column, reason in INVALIDATING_METADATA_COLUMNS.items():
        parent_col, child_col = f"{column}_parent", f"{column}_child"
        if parent_col not in merged.columns or child_col not in merged.columns:
            continue
        stated = merged[parent_col].notna() & merged[child_col].notna()
        mismatched = stated & (merged[parent_col] != merged[child_col])
        if mismatched.any():
            import logging

            logging.warning(
                "%d/%d sessions %s — invalidating those comparisons.",
                int(mismatched.sum()),
                len(merged),
                reason,
            )
        for col in ["delta_aic", "delta_bic", "lr_stat", "lr_pvalue"]:
            merged.loc[mismatched, col] = np.nan

    return merged


def _comparison_metadata(
    comparison_df: pd.DataFrame, parent_name: str, child_name: str
) -> dict:
    """Fit-scale metadata shared by a comparison's parent and child.

    Returns one value per column in :data:`METADATA_COLUMNS`, or ``None`` where
    the frames carried none (legacy artifacts fit before the estimator was
    recorded).
    """
    metadata: dict = {}
    for column in METADATA_COLUMNS:
        values: set = set()
        for side in ("parent", "child"):
            suffixed = f"{column}_{side}"
            if suffixed in comparison_df.columns:
                values |= set(comparison_df[suffixed].dropna().unique())
        if len(values) > 1:
            raise ValueError(
                f"{parent_name} -> {child_name}: rungs report different "
                f"{column} values {sorted(map(str, values))}; their information "
                "criteria are on different scales and cannot be differenced"
            )
        metadata[column] = values.pop() if values else None
    return metadata


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
        preferring each model, the Wilcoxon test on delta-AIC, and the
        ``estimator`` / ``response_col`` the two rungs were fit with. The
        estimator is what says at which grain ``delta_aic_*``, ``child_better_aic``
        and ``lrt_significant`` are stated -- per event on a fraction or
        dominant-choice fit, per decoded timebin on a count fit -- so a reader
        cannot pool the two scales by accident.

    Raises
    ------
    ValueError
        If the parent and child rungs report different estimators or response
        columns: their information criteria are then on different scales and
        their difference is not a model comparison.
    """
    metadata = _comparison_metadata(comparison_df, parent_name, child_name)
    if len(comparison_df) == 0:
        return {
            "parent": parent_name,
            "child": child_name,
            "n_sessions": 0,
            **metadata,
        }

    daic = comparison_df["delta_aic"]
    dbic = comparison_df["delta_bic"]
    lr_sig = (comparison_df["lr_pvalue"] < 0.05).sum()

    result = {
        "parent": parent_name,
        "child": child_name,
        **metadata,
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

    # Wilcoxon signed-rank on delta-AIC over the sessions with a finite one
    # (the mismatched-n_events guard marks a session NaN).
    result["wilcoxon_p"] = signed_rank_p(daic)

    # Animal-level view of the same step. Sessions within an animal are not
    # independent and the corpus is unbalanced, so the session Wilcoxon above is a
    # cross-session consistency statistic, not a population test on n=5. The
    # per-animal median delta-AIC is balanced across animals, and how many animals
    # agree on the sign is the claim a reader should weigh.
    if "animal" in comparison_df.columns:
        known = daic.notna()
        animals = session_median_statistics(
            daic[known], comparison_df.loc[known, "animal"]
        )
        result["delta_aic_animal_median"] = animals["animal_median"]
        result["n_animals"] = animals["n_animals"]
        result["n_animals_child_better"] = animals["n_animals_positive"]
    else:
        result["delta_aic_animal_median"] = np.nan
        result["n_animals"] = 0
        result["n_animals_child_better"] = 0

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
