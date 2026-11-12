"""Compare nested models via AIC, BIC, and likelihood ratio tests.

AIC is the primary comparison metric. BIC is secondary. LRT is valid
for unregularized fits only.

Provides functions for comparing consecutive models in the hierarchy
(M0 vs M1, M1 vs M2, etc.) both at the per-session level and aggregated
across sessions.

An AIC or BIC difference is the child rung's minus the parent rung's, within
session, as in
:func:`~continuum_swr_replay.analysis.behavior.coefficient_statistics.aic_difference_statistics`:
negative means the child rung is better. Which sessions of a rung pair can be
compared at all is :func:`why_not_comparable`, which both use.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import chi2

from continuum_swr_replay.analysis.shared_statistics import (
    session_median_statistics,
    signed_rank_p,
)
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

from ..model_specs import MODEL_HIERARCHY

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

#: Cohort-size columns: rungs fit to different numbers of events, bins or stops
#: were fit to different rows, and AIC, BIC and deviance all scale with the rows,
#: so the rung fit to fewer wins mechanically. Which one is the unit depends on
#: the family: events (conditional arm/patch), bins (per-bin conditional
#: intensity), stops (per-stop rate).
COHORT_SIZE_COLUMNS = ("n_events", "n_bins", "n_stops")

#: The columns a session's two fits must agree on to be compared
#: (:func:`why_not_comparable`).
COMPARABILITY_COLUMNS = (*COHORT_SIZE_COLUMNS, *INVALIDATING_METADATA_COLUMNS)

#: Why a session whose fits differ on a cohort-size column is not compared.
_COHORT_SIZE_REASON = (
    "have mismatched cohort sizes between parent and child — freeze one cohort "
    "for the ladder before fitting"
)


def why_not_comparable(paired: pd.DataFrame, suffixes: tuple[str, str]) -> pd.DataFrame:
    """Which sessions of a rung pair can be compared, and why the others cannot.

    Parameters
    ----------
    paired : pd.DataFrame, shape (n_sessions, n_columns)
        The two rungs' fits merged on session, each rung's columns carried with
        its suffix. A column of :data:`COMPARABILITY_COLUMNS` is checked when
        both rungs carry it.
    suffixes : tuple of str
        The two rungs' column suffixes, for example ``("_parent", "_child")``.

    Returns
    -------
    differs : pd.DataFrame of bool, shape (n_sessions, n_checked)
        On ``paired``'s index, one column per checked column of
        :data:`COMPARABILITY_COLUMNS`: True where the session's two fits differ
        on it. A session can be compared when no column is True. A
        cohort-size column (:data:`COHORT_SIZE_COLUMNS`) differs wherever its two
        values differ, a missing one included: rows of unknown number are not
        known to be the same rows. A column of
        :data:`INVALIDATING_METADATA_COLUMNS` differs only where both fits state
        a value: a fit that records nothing is unknown, not different.
    """
    first, second = suffixes
    differing = {}
    for column in COMPARABILITY_COLUMNS:
        a, b = f"{column}{first}", f"{column}{second}"
        if a not in paired or b not in paired:
            continue
        differs = paired[a] != paired[b]
        if column not in COHORT_SIZE_COLUMNS:
            differs &= paired[a].notna() & paired[b].notna()
        differing[column] = differs.astype(bool)
    return pd.DataFrame(differing, index=paired.index, dtype=bool)


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
        ``delta_aic`` is ``AIC(child) - AIC(parent)`` and ``delta_bic`` the
        same for BIC: negative means the child is better. ``lr_stat`` is
        ``2 * (llf_child - llf_parent)``.
        ``delta_aic`` / ``delta_bic`` / ``lr_stat`` / ``lr_pvalue`` are NaN for
        any session whose two fits cannot be compared
        (:func:`why_not_comparable`): they report different ``n_events``,
        ``n_bins``, or ``n_stops`` — rungs fit to different rows are not
        comparable — or disagree on one of
        :data:`INVALIDATING_METADATA_COLUMNS`: on
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
    # rows.
    count_cols = [
        col
        for col in COHORT_SIZE_COLUMNS
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
    merged["delta_aic"] = merged["aic_child"] - merged["aic_parent"]
    merged["delta_bic"] = merged["bic_child"] - merged["bic_parent"]

    merged["lr_stat"] = -2 * (merged[ll_parent] - merged[ll_child])
    merged["df_diff"] = merged["n_params_child"] - merged["n_params_parent"]
    merged["lr_pvalue"] = chi2.sf(
        merged["lr_stat"].clip(lower=0), df=merged["df_diff"].clip(lower=1)
    )

    # Invalidate all comparison metrics for a session whose two fits cannot be
    # compared: fit to different rows, or with incommensurable likelihoods.
    differs = why_not_comparable(merged, ("_parent", "_child"))
    for column in differs.columns[differs.any()]:
        logging.warning(
            "%d/%d sessions %s [%s] — invalidating those comparisons.",
            int(differs[column].sum()),
            len(merged),
            INVALIDATING_METADATA_COLUMNS.get(column, _COHORT_SIZE_REASON),
            column,
        )
    invalid = differs.any(axis=1)
    merged.loc[invalid, ["delta_aic", "delta_bic", "lr_stat", "lr_pvalue"]] = np.nan

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
        Summary statistics of the per-session ``delta_aic``
        (``AIC(child) - AIC(parent)``; negative means the child is better):
        its median and quartiles (``delta_aic_median``, ``delta_aic_q25``,
        ``delta_aic_q75``), the median ``delta_bic``, the sessions in which
        the child's AIC is lower than the parent's by more than 2
        (``child_better_aic``), higher by more than 2 (``parent_better_aic``),
        and the same for BIC (``child_better_bic``), the Wilcoxon test on
        ``delta_aic``, the median over animals of each animal's median
        ``delta_aic`` (``delta_aic_animal_median``), the animals whose median
        is negative (``n_animals_child_better``), and the
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
        "child_better_aic": int((daic < -2).sum()),
        "parent_better_aic": int((daic > 2).sum()),
        "child_better_bic": int((dbic < -2).sum()),
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
        animals = session_median_statistics(daic, comparison_df["animal"])
        result["delta_aic_animal_median"] = animals["animal_median"]
        result["n_animals"] = animals["n_animals"]
        result["n_animals_child_better"] = animals["n_animals_negative"]
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
