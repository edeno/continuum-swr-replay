"""Canonical arrival and outcome-availability clocks for behavioral history."""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders.trial_clocks import (
    outcome_known_times_from_task_visits,
)
from continuum_swr_replay.tables.columns import require_columns


def trial_arrival_times(trials: pd.DataFrame) -> np.ndarray:
    """Per-trial destination-arrival timestamps for visit/location history.

    The arrival is ``destination_first_poke_in_time``: the moment the animal
    *arrives at* the goal well (poke-in, which precedes reward delivery). Every
    loaded trial frame carries it (``data_loaders.trials``). This clock controls
    destination, visit-recency, and switch history only. Reward-bearing history
    must use :func:`trial_outcome_times` instead.

    Parameters
    ----------
    trials : pd.DataFrame
        Movement trials with ``destination_first_poke_in_time``.

    Returns
    -------
    np.ndarray, shape (n_trials,)
        Each trial's first poke-in at its destination (s).

    Raises
    ------
    ValueError
        If the column is absent or a value is not finite.
    """
    require_columns(trials, ["destination_first_poke_in_time"], "trials")
    arrival = trials["destination_first_poke_in_time"].to_numpy(dtype=float)
    if not np.isfinite(arrival).all():
        bad = tuple(np.flatnonzero(~np.isfinite(arrival))[:10])
        raise ValueError(
            "destination_first_poke_in_time must be finite for every trial; "
            f"invalid row(s): {bad}"
        )
    return arrival


def trial_outcome_times(trials: pd.DataFrame) -> np.ndarray:
    """Per-trial timestamp at which reward versus omission is observable.

    Every trial table carries ``outcome_known_time``: first pump onset for a
    rewarded trial and ``min(arrival + OMISSION_KNOWN_LATENCY_SEC, destination
    final poke-out)`` for an omission. A frame without it is refused.

    Every returned value must be finite and lie within
    ``[destination_arrival, end_time]``. Invalid explicit timestamps raise
    rather than silently falling back to another clock.
    """
    if len(trials) == 0:
        return np.array([], dtype=float)
    require_columns(trials, ["end_time", "outcome_known_time"], "trials")

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
        # is_reward identifies the rewarded trials the reward clock applies to.
        require_columns(trials, ["is_reward"], f"trials with {reward_column}")
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
