"""Behavior prediction analysis: does replay content predict subsequent choice?

Following Gillespie et al. 2021 (their Figure 5D), we test whether knowing
what was replayed during immobility improves prediction of the animal's
next arm choice beyond knowing where the animal currently is.

The approach:
1. Group replay events by trial (events between consecutive choices)
2. Compute replay evidence: fraction of events at each arm per trial
3. Baseline model: predict next arm from current arm only
4. Replay model: predict next arm from current arm + replay evidence
5. Cross-validate with leave-one-session-out
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut

from continuum_swr_replay.analysis.replay.spatial_mapping import well_to_arm_set

logger = logging.getLogger(__name__)

ALL_ARMS = list(range(9))


def aggregate_replay_per_trial(
    events: pd.DataFrame,
    track_graph,
    session_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate replay events into per-trial replay evidence vectors.

    For each trial (defined by unique animal/day/epoch/next_trial_to_well
    groupings of consecutive events), computes the fraction of replay
    events representing each arm.

    Parameters
    ----------
    events : pd.DataFrame
        Event-level data with replay_arm, track_segment_id,
        next_trial_to_well, is_swr_flag, and session identifiers.
    track_graph : object
        Track graph for well_to_arm_set mapping.
    session_cols : list[str] | None
        Session grouping columns.

    Returns
    -------
    pd.DataFrame
        One row per trial with columns:
        - current_arm: animal's current arm (modal track_segment_id)
        - next_well: where the animal goes next
        - next_arm_set: set of arms adjacent to next_well
        - replay_frac_0 ... replay_frac_8: fraction of events at each arm
        - replay_frac_swr_0 ... replay_frac_swr_8: SWR-only fractions
        - replay_frac_noswr_0 ... replay_frac_noswr_8: non-SWR fractions
        - n_events, n_swr, n_noswr: event counts
        - session identifiers
    """
    if session_cols is None:
        session_cols = ["animal", "day", "epoch"]

    # We need to group events into "bouts" — sequences of events
    # that precede a particular choice. Use event_midpoint_time to
    # identify bout boundaries: events with similar timestamps and
    # the same next_trial_to_well belong to the same bout.
    # Simpler: group by session + next_trial_to_well + non_local_event range.
    # Actually, the cleanest grouping is by the inter-trial interval:
    # all events that share the same (session, next_trial_to_well, trial_block).
    # Since next_trial_to_well changes when the animal makes a choice,
    # consecutive events with the same next_trial_to_well are in the same bout.

    # Detect bout boundaries: next_trial_to_well changes
    events = events.sort_values(session_cols + ["event_midpoint_time"]).copy()

    # Create a bout ID: changes whenever session or next_trial changes
    session_key = events[session_cols].astype(str).agg("_".join, axis=1)
    bout_change = (session_key != session_key.shift()) | (
        events["next_trial_to_well"] != events["next_trial_to_well"].shift()
    )
    events["bout_id"] = bout_change.cumsum()

    rows = []
    for _bout_id, group in events.groupby("bout_id"):
        if len(group) < 1:
            continue

        next_well = group["next_trial_to_well"].iloc[0]
        if np.isnan(next_well):
            continue

        current_arm = group["track_segment_id"].mode().iloc[0]
        if np.isnan(current_arm):
            continue

        # Next arm: the arms adjacent to the next well
        next_arms = well_to_arm_set(next_well, track_graph)

        # Replay evidence: fraction of events at each arm
        n_events = len(group)
        replay_arms = group["replay_arm"].dropna()

        row = {}
        for col in session_cols:
            row[col] = group[col].iloc[0]
        row["current_arm"] = int(current_arm)
        row["next_well"] = int(next_well)
        row["n_events"] = n_events

        # All events
        for arm in ALL_ARMS:
            row[f"replay_frac_{arm}"] = (replay_arms == arm).sum() / max(
                len(replay_arms), 1
            )

        # SWR events
        swr_mask = group["is_swr_flag"] == 1
        swr_arms = group.loc[swr_mask, "replay_arm"].dropna()
        row["n_swr"] = int(swr_mask.sum())
        for arm in ALL_ARMS:
            row[f"replay_frac_swr_{arm}"] = (swr_arms == arm).sum() / max(
                len(swr_arms), 1
            )

        # Non-SWR events
        noswr_arms = group.loc[~swr_mask, "replay_arm"].dropna()
        row["n_noswr"] = int((~swr_mask).sum())
        for arm in ALL_ARMS:
            row[f"replay_frac_noswr_{arm}"] = (noswr_arms == arm).sum() / max(
                len(noswr_arms), 1
            )

        # Target: which arm does the animal go to?
        # Mark 1 for arms adjacent to the next well
        for arm in ALL_ARMS:
            row[f"next_is_{arm}"] = int(arm in next_arms)

        rows.append(row)

    return pd.DataFrame(rows)


def fit_choice_model(
    bout_data: pd.DataFrame,
    include_replay: bool = False,
    replay_type: str = "all",
) -> tuple[LogisticRegression, np.ndarray, np.ndarray]:
    """Fit multinomial logistic regression predicting next arm choice.

    Parameters
    ----------
    bout_data : pd.DataFrame
        Per-trial data from aggregate_replay_per_trial.
    include_replay : bool
        If True, include replay evidence as predictors.
    replay_type : {"all", "swr", "noswr"}
        Which replay events to use as predictors.

    Returns
    -------
    tuple
        (fitted_model, X, y) where y is the next arm choice (0-8).
    """
    # Target: next well (unambiguous — avoids 2-arm-per-well ambiguity).
    # Using well as target is cleaner than picking one of 2 adjacent arms.
    y = bout_data["next_well"].astype(int).values

    # Baseline: one-hot current arm
    current_dummies = pd.get_dummies(bout_data["current_arm"], prefix="curr").reindex(
        columns=[f"curr_{arm}" for arm in ALL_ARMS], fill_value=0
    )

    if include_replay:
        if replay_type == "swr":
            replay_cols = [f"replay_frac_swr_{arm}" for arm in ALL_ARMS]
        elif replay_type == "noswr":
            replay_cols = [f"replay_frac_noswr_{arm}" for arm in ALL_ARMS]
        else:
            replay_cols = [f"replay_frac_{arm}" for arm in ALL_ARMS]

        X = pd.concat([current_dummies, bout_data[replay_cols]], axis=1).values.astype(
            float
        )
    else:
        X = current_dummies.values.astype(float)

    # Fit multinomial logistic regression
    model = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=1.0,  # regularization
    )
    model.fit(X, y)

    return model, X, y


def cross_validate_choice_prediction(
    bout_data: pd.DataFrame,
    session_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Leave-one-session-out CV comparing baseline vs replay-informed models.

    Parameters
    ----------
    bout_data : pd.DataFrame
        Per-trial data from aggregate_replay_per_trial.
    session_cols : list[str] | None
        Session grouping columns.

    Returns
    -------
    pd.DataFrame
        One row per session with accuracy and log-loss for each model.
    """
    if session_cols is None:
        session_cols = ["animal", "day", "epoch"]

    # Session groups for LOGO CV
    session_key = bout_data[session_cols].astype(str).agg("_".join, axis=1)
    groups = session_key.values

    # Target: next well (unambiguous)
    y = bout_data["next_well"].astype(int).values

    # Features
    current_dummies = (
        pd.get_dummies(bout_data["current_arm"], prefix="curr")
        .reindex(columns=[f"curr_{arm}" for arm in ALL_ARMS], fill_value=0)
        .values.astype(float)
    )

    replay_all = bout_data[[f"replay_frac_{arm}" for arm in ALL_ARMS]].values.astype(
        float
    )
    replay_swr = bout_data[
        [f"replay_frac_swr_{arm}" for arm in ALL_ARMS]
    ].values.astype(float)
    replay_noswr = bout_data[
        [f"replay_frac_noswr_{arm}" for arm in ALL_ARMS]
    ].values.astype(float)

    X_baseline = current_dummies
    X_replay_all = np.hstack([current_dummies, replay_all])
    X_replay_swr = np.hstack([current_dummies, replay_swr])
    X_replay_noswr = np.hstack([current_dummies, replay_noswr])

    logo = LeaveOneGroupOut()
    results = []

    for train_idx, test_idx in logo.split(X_baseline, y, groups):
        session_id = groups[test_idx[0]]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_train)) < 2 or len(y_test) < 3:
            continue

        row = {"session": session_id, "n_test": len(y_test)}

        for model_name, X in [
            ("baseline", X_baseline),
            ("replay_all", X_replay_all),
            ("replay_swr", X_replay_swr),
            ("replay_noswr", X_replay_noswr),
        ]:
            X_tr, X_te = X[train_idx], X[test_idx]

            try:
                clf = LogisticRegression(
                    multi_class="multinomial",
                    solver="lbfgs",
                    max_iter=1000,
                    C=1.0,
                )
                clf.fit(X_tr, y_train)
                y_pred = clf.predict(X_te)
                y_prob = clf.predict_proba(X_te)

                accuracy = float(np.mean(y_pred == y_test))

                # Log-loss (negative log-likelihood per sample)
                eps = 1e-10
                # Map y_test to indices in clf.classes_
                class_map = {c: i for i, c in enumerate(clf.classes_)}
                y_idx = np.array([class_map.get(yi, 0) for yi in y_test])
                ll = np.mean(np.log(y_prob[np.arange(len(y_test)), y_idx] + eps))

                row[f"{model_name}_accuracy"] = accuracy
                row[f"{model_name}_logloss"] = float(-ll)

            except Exception:
                row[f"{model_name}_accuracy"] = np.nan
                row[f"{model_name}_logloss"] = np.nan

        results.append(row)

    return pd.DataFrame(results)
