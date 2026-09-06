"""Physical task-visit clocks shared by behavioral table producers.

One task visit begins at the first destination poke-in and ends at the final
poke-out before a different port is visited. Consecutive same-port re-pokes do
not create new task outcomes. Keeping the outcome rule here prevents the trial,
qvals, and replay-history builders from assigning different availability times
to the same visit.
"""

from __future__ import annotations

import numpy as np

from continuum_swr_replay.data_loaders.constants import OMISSION_KNOWN_LATENCY_SEC


def outcome_known_times_from_task_visits(
    *,
    rewarded: np.ndarray,
    first_poke_in_time: np.ndarray,
    final_poke_out_time: np.ndarray,
    reward_on_time: np.ndarray,
) -> np.ndarray:
    """Return the physical time at which each task outcome is available.

    Rewards become available at the recorded pump onset. Omissions become
    available at the latest possible pump latency after arrival, capped by the
    collapsed task visit's final departure. All inputs must have identical
    one-dimensional shape. Rewarded visits require exactly one finite pump
    onset within the visit; omission visits require a missing pump onset.
    """
    rewarded_arr = np.asarray(rewarded)
    arrival = np.asarray(first_poke_in_time, dtype=float)
    departure = np.asarray(final_poke_out_time, dtype=float)
    reward_on = np.asarray(reward_on_time, dtype=float)
    arrays = (rewarded_arr, arrival, departure, reward_on)
    if any(array.ndim != 1 for array in arrays):
        raise ValueError("task-visit outcome-clock inputs must be one-dimensional")
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("task-visit outcome-clock inputs must have identical shape")

    try:
        rewarded_numeric = rewarded_arr.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("rewarded labels must be finite binary values") from exc
    if (
        not np.isfinite(rewarded_numeric).all()
        or not np.isin(rewarded_numeric, (0.0, 1.0)).all()
    ):
        raise ValueError("rewarded labels must be finite binary values")
    rewarded_mask = rewarded_numeric.astype(bool)

    if not np.isfinite(arrival).all():
        raise ValueError("task visit lacks finite first poke_in time")
    if not np.isfinite(departure).all():
        raise ValueError("task visit lacks finite final poke_out time")
    if (departure < arrival).any():
        raise ValueError("task visit has final poke_out before first poke_in")
    if (~np.isfinite(reward_on[rewarded_mask])).any():
        raise ValueError("rewarded task visit lacks finite reward_on time")
    if np.isfinite(reward_on[~rewarded_mask]).any():
        raise ValueError(
            "omission task visit unexpectedly carries finite reward_on time"
        )
    if (reward_on[rewarded_mask] < arrival[rewarded_mask]).any() or (
        reward_on[rewarded_mask] > departure[rewarded_mask]
    ).any():
        raise ValueError(
            "rewarded task visit reward_on time must lie within first poke_in and "
            "final poke_out"
        )

    observed_latency = reward_on[rewarded_mask] - arrival[rewarded_mask]
    if observed_latency.size and observed_latency.max() > OMISSION_KNOWN_LATENCY_SEC:
        raise ValueError(
            f"reward latency {observed_latency.max():.3f}s exceeds "
            f"OMISSION_KNOWN_LATENCY_SEC={OMISSION_KNOWN_LATENCY_SEC}s; the "
            "omission boundary would precede a possible pump onset"
        )

    return np.where(
        rewarded_mask,
        reward_on,
        np.minimum(arrival + OMISSION_KNOWN_LATENCY_SEC, departure),
    )


def outcome_known_time_from_task_visit(
    *,
    rewarded: bool,
    first_poke_in_time: float,
    final_poke_out_time: float,
    reward_on_time: float,
) -> float:
    """Scalar wrapper for :func:`outcome_known_times_from_task_visits`."""
    outcome = outcome_known_times_from_task_visits(
        rewarded=np.asarray([rewarded]),
        first_poke_in_time=np.asarray([first_poke_in_time], dtype=float),
        final_poke_out_time=np.asarray([final_poke_out_time], dtype=float),
        reward_on_time=np.asarray([reward_on_time], dtype=float),
    )
    return float(outcome[0])
