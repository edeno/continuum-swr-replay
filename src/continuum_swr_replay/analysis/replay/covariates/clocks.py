"""Canonical arrival and outcome-availability clocks for behavioral history."""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders.trial_clocks import (
    outcome_known_times_from_task_visits,
)


def trial_arrival_times(trials: pd.DataFrame) -> np.ndarray:
    """Per-trial destination-arrival timestamps for visit/location history.

    Uses ``goal_arrival_time`` (the moment the animal *arrives at* the
    goal well -- poke-in, which precedes reward delivery) when available,
    falling back to ``end_time`` per row when ``goal_arrival_time`` is NaN,
    and per frame when the column is absent. This clock controls destination,
    visit-recency, and switch history only. Reward-bearing history must use
    :func:`trial_outcome_times` instead.
    """
    if "goal_arrival_time" in trials.columns:
        cutoff = trials["goal_arrival_time"].to_numpy(dtype=float)
        end = trials["end_time"].to_numpy(dtype=float)
        if np.isinf(cutoff).any():
            raise ValueError(
                "goal_arrival_time contains infinity; corrupted physical timing "
                "must not silently fall back to end_time"
            )
        missing = np.isnan(cutoff)
        return np.where(missing, end, cutoff)
    return trials["end_time"].to_numpy(dtype=float)


def trial_outcome_times(trials: pd.DataFrame) -> np.ndarray:
    """Per-trial timestamp at which reward versus omission is observable.

    Newly generated trial tables carry ``outcome_known_time``: first pump onset
    for a rewarded trial and ``min(arrival + OMISSION_KNOWN_LATENCY_SEC,
    destination final poke-out)`` for an omission. Older or
    manually constructed frames lack that field; for those frames ``end_time``
    is an explicit conservative fallback for both outcomes. This fallback can
    delay rewarded history, but it cannot expose reward before delivery.

    Every returned value must be finite and lie within
    ``[destination_arrival, end_time]``. Invalid explicit timestamps raise
    rather than silently falling back to another clock.
    """
    if len(trials) == 0:
        return np.array([], dtype=float)
    if "end_time" not in trials.columns:
        raise KeyError("trial outcome timing requires end_time")

    end = trials["end_time"].to_numpy(dtype=float)
    arrival = trial_arrival_times(trials)
    if "destination_first_poke_out_time" in trials:
        first_poke_out = trials["destination_first_poke_out_time"].to_numpy(dtype=float)
        if (
            not np.isfinite(first_poke_out).all()
            or (first_poke_out < arrival).any()
            or (first_poke_out > end).any()
        ):
            raise ValueError(
                "destination first-poke-out times must lie between arrival and "
                "trial end"
            )
    reward = None
    if "is_reward" in trials.columns:
        reward = trials["is_reward"].to_numpy(dtype=float)
        if not np.isfinite(reward).all() or not np.isin(reward, (0.0, 1.0)).all():
            raise ValueError("trial is_reward values must be finite binary values")

    reward_column = next(
        (
            column
            for column in ("reward_on_time", "reward_on_ts")
            if column in trials.columns
        ),
        None,
    )
    expected_from_physical_clock = None
    if reward_column is not None:
        if reward is None:
            raise KeyError(
                f"{reward_column} requires is_reward to identify rewarded trials"
            )
        reward_on = trials[reward_column].to_numpy(dtype=float)
        rewarded = reward.astype(bool)
        if (~np.isfinite(reward_on[rewarded])).any():
            raise ValueError("rewarded trials must have finite reward-on timestamps")
        if np.isfinite(reward_on[~rewarded]).any():
            raise ValueError("omission trials cannot carry reward-on timestamps")
        # Must mirror the canonical derivation in
        # ``data_loaders.trials.make_trials_df_from_well_visits`` and
        # ``qvals_table._outcome_known_time``: pump onset when rewarded, else
        # arrival plus the omission latency, capped at the destination visit's
        # final poke-out (``end_time``). Same-port re-pokes are one task visit
        # under Comrie's parser; a transient beam release cannot establish an
        # omission while a pump could still be credited to that visit.
        expected_from_physical_clock = outcome_known_times_from_task_visits(
            rewarded=rewarded,
            first_poke_in_time=arrival,
            final_poke_out_time=end,
            reward_on_time=reward_on,
        )

    if "outcome_known_time" in trials.columns:
        outcome = trials["outcome_known_time"].to_numpy(dtype=float)
        if expected_from_physical_clock is not None and not np.allclose(
            outcome,
            expected_from_physical_clock,
            rtol=0.0,
            atol=1e-9,
            equal_nan=True,
        ):
            raise ValueError(
                "outcome_known_time conflicts with reward-on/end-time semantics"
            )
    elif expected_from_physical_clock is not None:
        outcome = expected_from_physical_clock
    else:
        outcome = end.copy()

    invalid = (
        ~np.isfinite(end)
        | ~np.isfinite(arrival)
        | ~np.isfinite(outcome)
        | (outcome < arrival)
        | (outcome > end)
    )
    if invalid.any():
        bad = tuple(np.flatnonzero(invalid)[:10])
        raise ValueError(
            "trial outcome-known times must be finite and lie between "
            f"destination arrival and end_time; invalid row(s): {bad}"
        )
    return outcome
