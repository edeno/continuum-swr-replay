"""Which times lie inside a well visit.

A well visit is one port experience: from the animal's first nose-poke in at a
reward well (``arrival_time``) to its final poke-out there (``departure_time``);
consecutive re-pokes at the same well are one visit. A time ``t`` is inside a
visit when ``arrival_time <= t < departure_time``, so the sample of the first
poke-in is inside and the sample of the final poke-out is outside.

This is the one statement of that rule. The per-sample ``occupancy_state`` of
the table builders (``behavior.covariates.context``) is made with it, and the
well-visit label of an event or a stop (``visit_context``) summarizes those
per-sample states; a quantity summed over time (seconds, SWRs, per-bin scores)
is restricted to the visits with :func:`in_well_visit` on its own time grid.
Immobility and excluded time are separate rules (``immobility``,
``analyzed_time``).
"""

from __future__ import annotations

import numpy as np

__all__ = ["in_well_visit", "well_visit_index"]


def well_visit_index(
    times: np.ndarray, arrival_times: np.ndarray, departure_times: np.ndarray
) -> np.ndarray:
    """The visit each time lies in.

    Parameters
    ----------
    times : np.ndarray, shape (n_times,)
        Times (s), in any order.
    arrival_times, departure_times : np.ndarray, shape (n_visits,)
        Each visit's first poke-in and final poke-out (s), in time order
        (``arrival_times`` ascending), as the port experiences give them.

    Returns
    -------
    np.ndarray of int, shape (n_times,)
        The position, in ``arrival_times``, of the last visit that began at or
        before the time, when the time is before that visit's final poke-out;
        -1 otherwise (before the first visit, or between two visits).

    Raises
    ------
    ValueError
        If the two visit arrays differ in length or ``arrival_times`` is not
        ascending.
    """
    times = np.asarray(times, dtype=float)
    arrivals = np.asarray(arrival_times, dtype=float)
    departures = np.asarray(departure_times, dtype=float)
    if arrivals.shape != departures.shape or arrivals.ndim != 1:
        raise ValueError(
            "arrival_times and departure_times must be one-dimensional and of "
            f"one length; got shapes {arrivals.shape} and {departures.shape}"
        )
    if (np.diff(arrivals) < 0).any():
        raise ValueError("arrival_times must be in ascending order")
    if arrivals.size == 0:
        return np.full(times.shape, -1, dtype=np.int64)
    last_begun = np.searchsorted(arrivals, times, side="right") - 1
    inside = (last_begun >= 0) & (times < departures[np.clip(last_begun, 0, None)])
    return np.where(inside, last_begun, -1).astype(np.int64)


def in_well_visit(
    times: np.ndarray, arrival_times: np.ndarray, departure_times: np.ndarray
) -> np.ndarray:
    """True where a time is inside a well visit (``arrival <= t < departure``).

    Parameters
    ----------
    times : np.ndarray, shape (n_times,)
        Times (s), for example a decode frame's index.
    arrival_times, departure_times : np.ndarray, shape (n_visits,)
        As :func:`well_visit_index`.

    Returns
    -------
    np.ndarray of bool, shape (n_times,)
    """
    return well_visit_index(times, arrival_times, departure_times) >= 0
