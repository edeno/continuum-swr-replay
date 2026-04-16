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

import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders.constants import TRACK_SEGMENT_TO_PATCH

logger = logging.getLogger(__name__)

ALL_ARMS = sorted(TRACK_SEGMENT_TO_PATCH.keys())


# ── Residualized replay ──


def predict_arm_fractions(
    event_arm_table: pd.DataFrame,
    fit_result: dict,
    predictors: list[str],
    event_col: str = "event_id",
) -> pd.DataFrame:
    """Compute model-predicted arm fractions and residuals per event-arm.

    Given a fitted content model (e.g. M3), computes
    ``softmax(X @ beta)`` within each event to get predicted fractions,
    then ``residual_frac = arm_fraction - predicted_frac``.

    Parameters
    ----------
    event_arm_table : pd.DataFrame
        Canonical event-arm table (9 rows per event).
    fit_result : dict
        Output of :func:`fitting.fit_conditional_poisson`. Must contain
        ``coef_{predictor}`` keys.
    predictors : list[str]
        Predictor column names used in the content model.
    event_col : str
        Event grouping column.

    Returns
    -------
    pd.DataFrame
        Copy of *event_arm_table* with ``predicted_frac`` and
        ``residual_frac`` columns added.
    """
    eat = event_arm_table.copy()

    params = np.array([fit_result[f"coef_{p}"] for p in predictors])
    X = eat[predictors].values.astype(float)
    eta = X @ params

    # Numerically stable softmax within each event
    eat["_eta"] = eta
    max_eta = eat.groupby(event_col)["_eta"].transform("max")
    eat["_shifted_exp"] = np.exp(eat["_eta"] - max_eta)
    sum_shifted = eat.groupby(event_col)["_shifted_exp"].transform("sum")
    eat["predicted_frac"] = eat["_shifted_exp"] / sum_shifted

    eat["residual_frac"] = eat["arm_fraction"] - eat["predicted_frac"]

    return eat.drop(columns=["_eta", "_shifted_exp"])


def build_residualized_choice_dataset(
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    stop_table: pd.DataFrame,
    content_fit: dict,
    content_predictors: list[str],
) -> pd.DataFrame:
    """Build choice dataset with both raw and residualized replay.

    Fits ``predict_arm_fractions`` on the content model, aggregates
    residuals to the stop × arm level, then merges into the standard
    choice dataset from :func:`build_choice_dataset`.

    Parameters
    ----------
    event_table, event_arm_table, stop_table
        Canonical tables.
    content_fit : dict
        Fitted content model result (from ``fit_conditional_poisson``).
    content_predictors : list[str]
        Predictors used in the content model (e.g. M3 predictors).

    Returns
    -------
    pd.DataFrame
        Choice dataset with ``replay_frac`` (raw) and
        ``replay_frac_residual`` (content-model residualized) columns.
    """
    eat_resid = predict_arm_fractions(event_arm_table, content_fit, content_predictors)

    # Aggregate residual_frac to stop × arm level.
    stop_arm_resid = (
        eat_resid.groupby(["stop_id", "arm"])
        .agg(replay_frac_residual=("residual_frac", "mean"))
        .reset_index()
    )

    choice_df = build_choice_dataset(event_table, event_arm_table, stop_table)
    if len(choice_df) == 0:
        return choice_df

    choice_df = choice_df.merge(
        stop_arm_resid[["stop_id", "arm", "replay_frac_residual"]],
        on=["stop_id", "arm"],
        how="left",
    )
    choice_df["replay_frac_residual"] = choice_df["replay_frac_residual"].fillna(0.0)

    return choice_df


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
    eat = event_arm_table[
        [
            "event_id",
            "stop_id",
            "arm",
            "arm_fraction",
            "visit_recency",
            "cumulative_reward",
            "recent_reward_rate",
            "dist_to_arm",
            "is_same_patch",
        ]
    ].copy()

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
        [
            "stop_id",
            "animal",
            "day",
            "epoch",
            "current_arm",
            "next_arm",
            "n_events",
            "reward_outcome",
            "is_after_reward",
            "is_after_omission",
        ]
    ].copy()

    choice_df = stop_arm.merge(stop_info, on="stop_id", how="inner")

    # Exclude current arm from choice set.
    choice_df = choice_df[choice_df["arm"] != choice_df["current_arm"]].reset_index(
        drop=True
    )

    # Response: did the animal choose this arm?
    choice_df["chosen"] = (choice_df["arm"] == choice_df["next_arm"]).astype(int)

    # Sanity: exactly one chosen arm per stop.
    chosen_per_stop = choice_df.groupby("stop_id")["chosen"].sum()
    valid_stop_ids = chosen_per_stop[chosen_per_stop == 1].index
    choice_df = choice_df[choice_df["stop_id"].isin(valid_stop_ids)].reset_index(
        drop=True
    )

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


# ── Switch-aligned timing analysis ──


def build_switch_aligned_table(
    stop_table: pd.DataFrame,
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    *,
    content_fit: dict | None = None,
    content_predictors: list[str] | None = None,
    max_lag: int = 20,
) -> pd.DataFrame:
    """Align replay content to stops preceding patch switches.

    Identifies unique switching episodes (first switch in a run of
    consecutive switches), then collects the preceding ``max_lag``
    stops. For each stop with replay, computes patch-level replay
    content: fraction on the current patch vs non-current patches.

    Parameters
    ----------
    stop_table : pd.DataFrame
        Canonical stop table (includes zero-event stops).
    event_table : pd.DataFrame
        Canonical event table.
    event_arm_table : pd.DataFrame
        Canonical event-arm table (9 rows per event).
    content_fit : dict, optional
        Fitted content model for residualization. If provided along
        with *content_predictors*, adds ``replay_frac_residual_noncurrent``.
    content_predictors : list[str], optional
        Predictors used in *content_fit*.
    max_lag : int
        Maximum number of stops before the switch to include.

    Returns
    -------
    pd.DataFrame
        One row per (episode, lag) for stops with replay. Columns:

        - ``episode_id`` — unique switching episode identifier
        - ``session`` — session identifier string
        - ``switch_stop_idx`` — stop_idx of the switch
        - ``stop_idx``, ``stop_id`` — the replay stop
        - ``lag`` — stops before switch (negative; -1 = immediately before)
        - ``current_patch`` — patch the animal was in during the run
        - ``replay_frac_current_patch`` — fraction of replay on current patch
        - ``replay_frac_noncurrent`` — fraction on non-current patches
        - ``n_events`` — events in this stop
        - ``replay_frac_residual_noncurrent`` — (if content model provided)
    """
    st = stop_table.copy()

    # Need is_stay to identify switches. Filter to valid rows.
    valid = st[st["is_stay"].notna()].copy()
    if len(valid) == 0:
        return pd.DataFrame()

    valid["is_switch"] = (valid["is_stay"] == 0).astype(int)

    # First switch in each run: preceded by a stay (or first stop).
    valid["prev_is_switch"] = valid["is_switch"].shift(1, fill_value=0)
    first_switches = valid[(valid["is_switch"] == 1) & (valid["prev_is_switch"] == 0)]

    if len(first_switches) == 0:
        return pd.DataFrame()

    # Optionally residualize
    eat = event_arm_table
    if content_fit is not None and content_predictors is not None:
        eat = predict_arm_fractions(eat, content_fit, content_predictors)

    # Compute per-stop patch-level replay fractions.
    # Group events by stop, compute fraction on current patch.
    stop_replay = _stop_patch_replay(eat, event_table)

    rows = []
    for ep_i, (_idx, sw_row) in enumerate(first_switches.iterrows()):
        sw_stop_idx = sw_row["stop_idx"]
        current_patch = sw_row["current_patch"]
        animal = sw_row["animal"]
        day = sw_row["day"]
        epoch = sw_row["epoch"]

        # Preceding stops within max_lag
        preceding = st[
            (st["stop_idx"] < sw_stop_idx)
            & (st["stop_idx"] >= sw_stop_idx - max_lag)
            & st["has_replay"]
        ]

        for _, ps_row in preceding.iterrows():
            lag = int(ps_row["stop_idx"] - sw_stop_idx)  # negative
            sid = ps_row["stop_id"]

            if sid not in stop_replay.index:
                continue

            sr = stop_replay.loc[sid]
            row = {
                "episode_id": ep_i,
                "animal": animal,
                "day": day,
                "epoch": epoch,
                "switch_stop_idx": int(sw_stop_idx),
                "stop_idx": int(ps_row["stop_idx"]),
                "stop_id": sid,
                "lag": lag,
                "current_patch": current_patch,
                "replay_frac_current_patch": sr.get(
                    "replay_frac_current_patch", np.nan
                ),
                "replay_frac_noncurrent": sr.get("replay_frac_noncurrent", np.nan),
                "n_events": int(ps_row["n_events"]),
            }
            if "replay_frac_residual_noncurrent" in sr.index:
                row["replay_frac_residual_noncurrent"] = sr[
                    "replay_frac_residual_noncurrent"
                ]
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_control_aligned_table(
    stop_table: pd.DataFrame,
    event_table: pd.DataFrame,
    event_arm_table: pd.DataFrame,
    switch_episodes: pd.DataFrame,
    *,
    content_fit: dict | None = None,
    content_predictors: list[str] | None = None,
    max_lag: int = 20,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Build matched control table for switch-aligned comparison.

    For each switching episode, finds a random stretch of ``max_lag``
    consecutive stay stops in the same session and patch. Returns the
    same columns as :func:`build_switch_aligned_table`.

    Parameters
    ----------
    stop_table : pd.DataFrame
        Canonical stop table.
    event_table, event_arm_table
        Canonical tables.
    switch_episodes : pd.DataFrame
        Output of :func:`build_switch_aligned_table`. Used to extract
        episode metadata (session, current_patch, switch_stop_idx).
    content_fit, content_predictors
        For residualization (passed through to ``_stop_patch_replay``).
    max_lag : int
        Window size for control stretch.
    rng : numpy Generator, optional
        For reproducibility.

    Returns
    -------
    pd.DataFrame
        Same schema as ``build_switch_aligned_table`` but with
        ``episode_id`` matching the switch episode and ``lag`` computed
        relative to the control anchor point.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    st = stop_table.copy()

    eat = event_arm_table
    if content_fit is not None and content_predictors is not None:
        eat = predict_arm_fractions(eat, content_fit, content_predictors)

    stop_replay = _stop_patch_replay(eat, event_table)

    # Get unique episode metadata
    if len(switch_episodes) == 0:
        return pd.DataFrame()
    ep_meta = switch_episodes.groupby("episode_id").first()[
        ["animal", "day", "epoch", "current_patch", "switch_stop_idx"]
    ]

    rows = []
    for ep_id, meta in ep_meta.iterrows():
        animal = meta["animal"]
        day = meta["day"]
        epoch = meta["epoch"]
        patch = meta["current_patch"]
        sw_idx = meta["switch_stop_idx"]

        # Find stay stops in same session, excluding the max_lag
        # window around the actual switch
        session_stays = st[
            (st["animal"] == animal)
            & (st["day"] == day)
            & (st["epoch"] == epoch)
            & (st["is_stay"] == 1)
            & (
                (st["stop_idx"] < sw_idx - max_lag)
                | (st["stop_idx"] > sw_idx + max_lag)
            )
        ].sort_values("stop_idx")

        if len(session_stays) < 5:
            continue

        # Pick a random anchor: must have ≥5 preceding stays
        possible_anchors = session_stays.iloc[5:]
        if len(possible_anchors) == 0:
            continue
        anchor_idx = rng.integers(0, len(possible_anchors))
        anchor_stop_idx = possible_anchors.iloc[anchor_idx]["stop_idx"]

        # Collect preceding stops with replay
        preceding = st[
            (st["stop_idx"] < anchor_stop_idx)
            & (st["stop_idx"] >= anchor_stop_idx - max_lag)
            & st["has_replay"]
        ]

        for _, ps_row in preceding.iterrows():
            lag = int(ps_row["stop_idx"] - anchor_stop_idx)
            sid = ps_row["stop_id"]
            if sid not in stop_replay.index:
                continue
            sr = stop_replay.loc[sid]
            row = {
                "episode_id": ep_id,
                "animal": animal,
                "day": day,
                "epoch": epoch,
                "switch_stop_idx": int(anchor_stop_idx),
                "stop_idx": int(ps_row["stop_idx"]),
                "stop_id": sid,
                "lag": lag,
                "current_patch": patch,
                "replay_frac_current_patch": sr.get(
                    "replay_frac_current_patch", np.nan
                ),
                "replay_frac_noncurrent": sr.get("replay_frac_noncurrent", np.nan),
                "n_events": int(ps_row["n_events"]),
            }
            if "replay_frac_residual_noncurrent" in sr.index:
                row["replay_frac_residual_noncurrent"] = sr[
                    "replay_frac_residual_noncurrent"
                ]
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _stop_patch_replay(
    event_arm_table: pd.DataFrame,
    event_table: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-stop replay fraction on current vs non-current patches.

    Returns a DataFrame indexed by ``stop_id`` with columns:
    ``replay_frac_current_patch``, ``replay_frac_noncurrent``, and
    optionally ``replay_frac_residual_noncurrent`` if
    ``residual_frac`` is present in *event_arm_table*.
    """
    eat = event_arm_table.copy()

    # Map arms to patches
    eat["_patch"] = eat["arm"].map(TRACK_SEGMENT_TO_PATCH)

    # current_patch is already in the event_arm_table (carried from
    # event_table during build). Fall back to a merge if missing.
    if "current_patch" not in eat.columns:
        evt_patch = event_table[["event_id", "current_patch"]].drop_duplicates()
        eat = eat.merge(evt_patch, on="event_id", how="left")

    # Flag arms in the current patch
    eat["is_current_patch_arm"] = (eat["_patch"] == eat["current_patch"]).astype(int)

    # Per-event patch fractions, then average to stop level
    event_patch_frac = (
        eat.groupby(["stop_id", "event_id"])
        .apply(
            lambda g: pd.Series(
                {
                    "replay_frac_current_patch": g.loc[
                        g["is_current_patch_arm"] == 1, "arm_fraction"
                    ].sum(),
                    "replay_frac_noncurrent": g.loc[
                        g["is_current_patch_arm"] == 0, "arm_fraction"
                    ].sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    result = event_patch_frac.groupby("stop_id")[
        ["replay_frac_current_patch", "replay_frac_noncurrent"]
    ].mean()

    # Residualized version if available
    if "residual_frac" in eat.columns:
        event_resid_frac = (
            eat.groupby(["stop_id", "event_id"])
            .apply(
                lambda g: pd.Series(
                    {
                        "replay_frac_residual_noncurrent": g.loc[
                            g["is_current_patch_arm"] == 0, "residual_frac"
                        ].sum(),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )
        resid_agg = event_resid_frac.groupby("stop_id")[
            ["replay_frac_residual_noncurrent"]
        ].mean()
        result = result.join(resid_agg)

    return result
