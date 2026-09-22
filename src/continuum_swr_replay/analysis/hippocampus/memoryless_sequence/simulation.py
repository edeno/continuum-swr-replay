"""Put a replay of known shape into a simulated recording.

Used to ask what a detector does to an event whose true path, duration and
firing rate are known. The recording itself comes from ``non_local_detector``'s
simulation; these helpers add an event to it and say where an event may safely
go.

Placement is not a detail. A window straddling the end of an immobile stretch
loses its center to the speed gate, and because jumps are forward the sequence
ends one window earlier still, so an event placed too near the edge measures
the speed gate rather than whatever it was meant to measure.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "InjectedReplay",
    "immobile_intervals",
    "inject_replay",
    "place_field_means",
    "placements",
    "without",
]

FIELD_STD_CM = np.sqrt(12.5)
"""Place field width the simulation package uses."""


@dataclass(frozen=True)
class InjectedReplay:
    """One replay to add to a recording.

    Attributes
    ----------
    start_cm : float
        Where the represented path begins.
    path_cm : float
        How far it travels. Its speed is this over ``duration_s``.
    duration_s : float
    peak_rate_hz : float
        Firing rate of a cell whose field the path passes over exactly.
    at_s : float
        When the event begins.
    seed : int

    """

    start_cm: float
    path_cm: float
    duration_s: float
    peak_rate_hz: float
    at_s: float
    seed: int = 0


def place_field_means(n_neurons: int, track_height_cm: float) -> NDArray:
    """Field centers tiled evenly along the track."""
    return np.linspace(0.0, track_height_cm, n_neurons)


def inject_replay(
    spike_times: list[NDArray],
    place_field_means: NDArray,
    start_cm: float,
    path_cm: float,
    duration_s: float,
    peak_rate_hz: float,
    at_s: float,
    sampling_frequency: float,
    seed: int = 0,
) -> tuple[list[NDArray], tuple[float, float]]:
    """Add a replay that travels a given distance in a given time.

    Each cell fires as a Poisson process whose rate follows its own place
    field, read at the position the replay represents at that moment. Spikes
    already inside the event's window are removed first, so the event is not
    read on top of whatever the recording was doing.

    Parameters
    ----------
    spike_times : list of ndarray
        One array per cell. Not modified.
    place_field_means : ndarray, shape (n_neurons,)
    start_cm, path_cm, duration_s, peak_rate_hz, at_s : float
        The replay, as in :class:`InjectedReplay`.
    sampling_frequency : float
        Rate at which the trajectory is sampled.
    seed : int

    Returns
    -------
    spike_times : list of ndarray
        A new list, sorted per cell.
    bounds : tuple of float
        When the event begins and ends.

    """
    n_samples = max(2, int(duration_s * sampling_frequency))
    step_s = duration_s / (n_samples - 1)
    at = at_s + np.arange(n_samples) * step_s
    position_cm = start_cm + path_cm * np.arange(n_samples) / (n_samples - 1)

    rate_hz = peak_rate_hz * np.exp(
        -0.5
        * (
            (position_cm[:, None] - np.asarray(place_field_means)[None, :])
            / FIELD_STD_CM
        )
        ** 2
    )
    fired = np.random.default_rng(seed).random(rate_hz.shape) < rate_hz * step_s

    injected = []
    for neuron, neuron_times in enumerate(spike_times):
        kept = neuron_times[(neuron_times < at[0]) | (neuron_times > at[-1])]
        injected.append(np.sort(np.concatenate([kept, at[fired[:, neuron]]])))
    return injected, (float(at[0]), float(at[-1]))


def immobile_intervals(
    time: NDArray,
    speed_cm_per_s: NDArray,
    threshold_cm_per_s: float,
    minimum_duration_s: float,
) -> NDArray:
    """Stretches the animal held still for long enough to hold an event.

    Parameters
    ----------
    time : ndarray, shape (n_time,)
    speed_cm_per_s : ndarray, shape (n_time,)
    threshold_cm_per_s : float
    minimum_duration_s : float

    Returns
    -------
    intervals : ndarray, shape (n_intervals, 2)
        First and last time of each stretch.

    """
    time = np.asarray(time, dtype=np.float64)
    still = np.asarray(speed_cm_per_s, dtype=np.float64) < threshold_cm_per_s
    edges = np.diff(still.astype(np.int8), prepend=0, append=0)
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    if starts.size == 0:
        return np.zeros((0, 2), dtype=np.float64)

    intervals = np.column_stack([time[starts], time[stops]])
    return intervals[np.diff(intervals, axis=1).ravel() >= minimum_duration_s]


def placements(
    intervals: NDArray,
    duration_s: float,
    margin_s: float,
    n: int,
    rng: np.random.Generator,
) -> NDArray:
    """Start times for an event of this length, clear of every interval's edge.

    The margin is the room a decode needs on each side. An event placed without
    it runs into the speed gate, which is a different measurement from the one
    intended.

    Parameters
    ----------
    intervals : ndarray, shape (n_intervals, 2)
    duration_s, margin_s : float
    n : int
        How many start times to draw.
    rng : numpy.random.Generator

    Returns
    -------
    at_s : ndarray, shape (n,)
        Drawn uniformly over the usable part of every wide enough interval.

    Raises
    ------
    ValueError
        No interval can hold the event and its margins.

    """
    intervals = np.asarray(intervals, dtype=np.float64).reshape(-1, 2)
    earliest = intervals[:, 0] + margin_s
    latest = intervals[:, 1] - margin_s - duration_s
    usable = latest > earliest
    if not usable.any():
        raise ValueError(
            f"No immobile stretch holds a {duration_s} s event with a {margin_s} s "
            "margin on each side."
        )

    earliest, latest = earliest[usable], latest[usable]
    width = latest - earliest
    chosen = rng.choice(earliest.size, size=n, p=width / width.sum())
    return earliest[chosen] + rng.random(n) * width[chosen]


def without(intervals: NDArray, excluded: NDArray, margin_s: float = 0.0) -> NDArray:
    """Remove spans from a set of intervals, splitting them where needed.

    Used to keep an injected event away from whatever the recording already
    holds without discarding the whole stretch it sits in. Dropping the whole
    stretch instead leaves very few places to inject, and if every stretch
    holds something the result is nothing at all.

    Parameters
    ----------
    intervals : ndarray, shape (n_intervals, 2)
    excluded : ndarray, shape (n_excluded, 2)
        Spans to remove. A zero-width span still removes its margin.
    margin_s : float
        Widens each excluded span on both sides.

    Returns
    -------
    kept : ndarray, shape (n_kept, 2)
        In order. An interval covered entirely is gone.

    """
    kept = [
        tuple(row) for row in np.asarray(intervals, dtype=np.float64).reshape(-1, 2)
    ]
    for start_s, end_s in np.asarray(excluded, dtype=np.float64).reshape(-1, 2):
        low, high = start_s - margin_s, end_s + margin_s
        split = []
        for interval_start, interval_end in kept:
            if high <= interval_start or low >= interval_end:
                split.append((interval_start, interval_end))
                continue
            if interval_start < low:
                split.append((interval_start, low))
            if high < interval_end:
                split.append((high, interval_end))
        kept = split
    return np.array(kept, dtype=np.float64).reshape(-1, 2)
