"""Complete pandas fixtures for canonical port-experience APIs.

These helpers adapt hand-written movement-trial fixtures; production code must
always construct experiences from raw well visits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.data_loaders.trial_clocks import (
    outcome_known_times_from_task_visits,
)

PORT_EXPERIENCE_COLUMNS = [
    "experience_id",
    "movement_trial_number",
    "is_initial_experience",
    "is_boundary_derived",
    "outcome_observed",
    "port",
    "patch",
    "arrival_time",
    "departure_time",
    "reward_on_time",
    "outcome_known_time",
    "is_reward",
]


def empty_port_experiences() -> pd.DataFrame:
    """Return an empty frame with the complete canonical schema."""
    return pd.DataFrame(columns=PORT_EXPERIENCE_COLUMNS)


def _outcome_known_times(trials: pd.DataFrame) -> np.ndarray:
    """Each toy trial's outcome clock: its own, or the loader's derivation.

    A frame that carries ``outcome_known_time`` keeps it. Otherwise, with a
    reward-on column, it is pump onset for a reward and the pump-latency
    boundary after arrival (capped at the final poke-out) for an omission, as
    ``data_loaders.trials`` derives it; with no reward-on column, ``end_time``.
    """
    if "outcome_known_time" in trials:
        return pd.to_numeric(trials["outcome_known_time"], errors="coerce").to_numpy(
            dtype=float
        )
    arrival_col = (
        "destination_first_poke_in_time"
        if "destination_first_poke_in_time" in trials
        else "end_time"
    )
    arrival = pd.to_numeric(trials[arrival_col], errors="coerce").to_numpy(dtype=float)
    end = pd.to_numeric(trials["end_time"], errors="coerce").to_numpy(dtype=float)
    arrival = np.where(np.isfinite(arrival), arrival, end)
    if "reward_on_time" not in trials:
        return end
    reward = pd.to_numeric(trials["is_reward"], errors="coerce").to_numpy(dtype=float)
    given_reward_on = pd.to_numeric(trials["reward_on_time"], errors="coerce").to_numpy(
        dtype=float
    )
    return outcome_known_times_from_task_visits(
        rewarded=reward,
        first_poke_in_time=arrival,
        final_poke_out_time=end,
        reward_on_time=np.where(reward.astype(bool), given_reward_on, np.nan),
    )


def with_outcome_known_time(trials: pd.DataFrame) -> pd.DataFrame:
    """A toy trial frame with its ``outcome_known_time``, as the loaders give it.

    Every loaded trial frame carries the outcome clock, and the table builders
    require it; a hand-built frame without it gets the clock
    :func:`_outcome_known_times` derives.

    Parameters
    ----------
    trials : pd.DataFrame
        A toy movement-trial frame.

    Returns
    -------
    pd.DataFrame
        A copy with ``outcome_known_time``.
    """
    if "outcome_known_time" in trials or trials.empty:
        return trials
    return trials.assign(outcome_known_time=_outcome_known_times(trials))


def port_experiences_from_trials(
    trials: pd.DataFrame,
    *,
    prepend_initial: bool,
    initial_reward: bool | None = False,
) -> pd.DataFrame:
    """Adapt a complete toy movement-trial frame into port experiences.

    ``prepend_initial=True`` is required for tests of builders/context that
    validate one initial experience plus one destination per movement trial.
    Low-level history tests can set it false to preserve their deliberately
    minimal hand-oracle sequence.
    """
    if trials.empty:
        return empty_port_experiences()
    frame = trials.copy()
    arrival_col = (
        "destination_first_poke_in_time"
        if "destination_first_poke_in_time" in frame
        else "end_time"
    )
    arrival = pd.to_numeric(frame[arrival_col], errors="coerce").to_numpy(dtype=float)
    end = pd.to_numeric(frame["end_time"], errors="coerce").to_numpy(dtype=float)
    arrival = np.where(np.isfinite(arrival), arrival, end)
    reward = pd.to_numeric(frame["is_reward"], errors="coerce").to_numpy(dtype=float)
    outcome = _outcome_known_times(frame)
    reward_on = np.where(reward.astype(bool), outcome, np.nan)
    ports = (
        pd.to_numeric(frame["to_well"], errors="coerce").to_numpy(dtype=int)
        if "to_well" in frame
        else np.zeros(len(frame), dtype=int)
    )
    patches = np.asarray([1 + int(port) // 2 for port in ports], dtype=int)
    offset = int(prepend_initial)
    chronological_rank = np.empty(len(frame), dtype=int)
    chronological_order = np.argsort(arrival, kind="stable")
    chronological_rank[chronological_order] = np.arange(len(frame))
    destination = pd.DataFrame(
        {
            "experience_id": chronological_rank + offset,
            "movement_trial_number": pd.array(frame.index, dtype="Int64"),
            "is_initial_experience": not prepend_initial,
            "is_boundary_derived": False,
            "outcome_observed": True,
            "port": ports,
            "patch": patches,
            "arrival_time": arrival,
            "departure_time": end,
            "reward_on_time": reward_on,
            "outcome_known_time": outcome,
            "is_reward": reward.astype(bool),
        }
    )
    if not prepend_initial:
        destination["is_initial_experience"] = False
        initial_mask = destination["experience_id"].eq(0)
        destination.loc[initial_mask, "is_initial_experience"] = True
        destination["movement_trial_number"] = pd.array(
            destination["movement_trial_number"], dtype="Int64"
        )
        destination.loc[initial_mask, "movement_trial_number"] = pd.NA
        return destination[PORT_EXPERIENCE_COLUMNS]

    first_start = float(
        pd.to_numeric(frame["movement_start_time"], errors="coerce").min()
    )
    first_arrival = float(arrival[0])
    initial_outcome = first_start
    initial_arrival = min(first_start, initial_outcome) - 1.0
    first_row = int(np.argmin(arrival))
    initial_port = (
        int(frame.iloc[first_row]["from_well"]) if "from_well" in frame else 0
    )
    initial = pd.DataFrame(
        {
            "experience_id": [0],
            "movement_trial_number": pd.array([pd.NA], dtype="Int64"),
            "is_initial_experience": [True],
            "is_boundary_derived": [initial_reward is None],
            "outcome_observed": [initial_reward is not None],
            "port": [initial_port],
            "patch": [1 + initial_port // 2],
            "arrival_time": [initial_arrival],
            "departure_time": [first_start],
            "reward_on_time": [initial_outcome if initial_reward is True else np.nan],
            "outcome_known_time": [
                initial_outcome if initial_reward is not None else np.nan
            ],
            "is_reward": pd.array([initial_reward], dtype="boolean"),
        }
    )
    if initial_outcome >= first_arrival:
        initial.loc[0, "outcome_known_time"] = first_arrival - 1e-6
        initial.loc[0, "departure_time"] = first_arrival - 1e-6
    return pd.concat([initial, destination], ignore_index=True)[PORT_EXPERIENCE_COLUMNS]
