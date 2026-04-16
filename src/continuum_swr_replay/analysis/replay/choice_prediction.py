"""Stage 5A: does replay content predict the next choice?

Builds a stop × arm choice dataset from the canonical tables, then
fits conditional logistic regression (or conditional Poisson with
binary response) to test whether arm-level replay content improves
prediction of the animal's next destination above a behavior-only
baseline.

The key predictor is ``replay_frac`` — the mean ``arm_fraction``
across all events during a stop. This is the fraction of non-local
content on each arm, averaged over replay events within the stop.
It's known BEFORE the animal makes its next choice (replay happens
during immobility, before the next trip begins).

``is_next`` / ``next_arm`` do NOT enter as predictors — they're the
response variable. The model is strictly predictive, not
retrospective (Stage 5B handles the retrospective alignment).
"""

from __future__ import annotations

import logging

import pandas as pd

from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH

logger = logging.getLogger(__name__)

ALL_ARMS = sorted(TRACK_SEGMENT_TO_PATCH.keys())


def build_choice_dataset(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    stop_table: pd.DataFrame,
) -> pd.DataFrame:
    """Build a stop × arm choice dataset for Stage 5A.

    For each stop that has (a) at least one replay event and (b) a
    valid ``next_arm`` (known next destination), produces one row per
    candidate arm. The current arm is excluded from the choice set
    (consecutive pokes at the same port are never rewarded).

    Parameters
    ----------
    event_table, event_arm_table, stop_table
        Canonical tables from :mod:`tables`.

    Returns
    -------
    pd.DataFrame
        One row per (stop, candidate arm). Columns:

        - ``stop_id`` — grouping key for conditional logistic
        - ``arm`` — candidate arm ID
        - ``chosen`` — 1 if the animal chose this arm, 0 otherwise
        - ``replay_frac`` — mean ``arm_fraction`` across events in
          this stop (0 if no event decoded to this arm)
        - ``visit_recency``, ``cumulative_reward``,
          ``recent_reward_rate``, ``dist_to_arm``, ``is_same_patch``
          — behavioral baseline predictors (from the last event in
          the stop)
        - ``n_events`` — number of replay events in this stop
        - Session identifiers (``animal``, ``day``, ``epoch``)
    """
    # Stops with replay and a valid next choice.
    has_replay = stop_table["has_replay"] & stop_table["next_arm"].notna()
    valid_stops = stop_table.loc[has_replay].copy()

    if len(valid_stops) == 0:
        return pd.DataFrame()

    # Aggregate replay content to stop × arm: mean arm_fraction
    # across events within each stop.
    eat = event_arm_table[["event_id", "stop_id", "arm", "arm_fraction",
                           "visit_recency", "cumulative_reward",
                           "recent_reward_rate", "dist_to_arm",
                           "is_same_patch"]].copy()

    # Mean arm_fraction per (stop, arm) across events in that stop.
    stop_arm = (
        eat.groupby(["stop_id", "arm"])
        .agg(
            replay_frac=("arm_fraction", "mean"),
            visit_recency=("visit_recency", "last"),
            cumulative_reward=("cumulative_reward", "last"),
            recent_reward_rate=("recent_reward_rate", "last"),
            dist_to_arm=("dist_to_arm", "last"),
            is_same_patch=("is_same_patch", "last"),
        )
        .reset_index()
    )

    # Join with stop-level info.
    stop_info = valid_stops[
        ["stop_id", "animal", "day", "epoch",
         "current_arm", "next_arm", "n_events",
         "reward_outcome", "is_after_reward", "is_after_omission"]
    ].copy()

    choice_df = stop_arm.merge(stop_info, on="stop_id", how="inner")

    # Exclude current arm from choice set.
    choice_df = choice_df[
        choice_df["arm"] != choice_df["current_arm"]
    ].reset_index(drop=True)

    # Response: did the animal choose this arm?
    choice_df["chosen"] = (
        choice_df["arm"] == choice_df["next_arm"]
    ).astype(int)

    # Sanity: exactly one chosen arm per stop.
    chosen_per_stop = choice_df.groupby("stop_id")["chosen"].sum()
    valid_stop_ids = chosen_per_stop[chosen_per_stop == 1].index
    choice_df = choice_df[
        choice_df["stop_id"].isin(valid_stop_ids)
    ].reset_index(drop=True)

    return choice_df


def fit_choice_model(
    choice_df: pd.DataFrame,
    predictors: list[str],
    stop_col: str = "stop_id",
) -> dict | None:
    """Fit conditional logistic on the choice dataset.

    Uses ``statsmodels.ConditionalLogit`` — same conditional
    likelihood framework as the content model's conditional Poisson,
    but with binary response (chose / didn't choose).

    Parameters
    ----------
    choice_df : pd.DataFrame
        Output of :func:`build_choice_dataset`.
    predictors : list[str]
        Arm-varying predictors (e.g., ``["replay_frac"]``).
    stop_col : str
        Grouping variable.

    Returns
    -------
    dict | None
        Coefficients, SE, p-values, fit statistics. None on failure.
    """
    from statsmodels.discrete.conditional_models import (
        ConditionalLogit,
    )

    required = ["chosen", stop_col] + predictors
    missing = [c for c in required if c not in choice_df.columns]
    if missing:
        logger.error("Missing columns: %s", missing)
        return None

    df = choice_df[required].dropna().copy()

    # Drop stops where no arm is chosen (shouldn't happen after
    # build_choice_dataset, but defensive).
    chosen_per_stop = df.groupby(stop_col)["chosen"].sum()
    valid = chosen_per_stop[chosen_per_stop == 1].index
    df = df[df[stop_col].isin(valid)].reset_index(drop=True)

    n_stops = df[stop_col].nunique()
    if n_stops < 2 * len(predictors):
        return None

    y = df["chosen"].values.astype(float)
    X = df[predictors].values.astype(float)
    groups = df[stop_col].values

    try:
        model = ConditionalLogit(y, X, groups=groups)
        result = model.fit(disp=0)
    except Exception:
        logger.warning("ConditionalLogit failed for %s", predictors)
        return None

    coefs = dict(zip(predictors, result.params, strict=True))
    bse = dict(zip(predictors, result.bse, strict=True))
    pvals = dict(zip(predictors, result.pvalues, strict=True))

    return {
        **{f"coef_{k}": float(v) for k, v in coefs.items()},
        **{f"bse_{k}": float(v) for k, v in bse.items()},
        **{f"pval_{k}": float(v) for k, v in pvals.items()},
        "n_stops": n_stops,
        "n_rows": len(df),
        "n_params": len(predictors),
        "llf": float(result.llf),
    }
