"""Put a replay of known shape into a simulated recording.

Used to ask what a detector does to an event whose true path, duration and
firing rate are known. The recording itself comes from
``non_local_detector.simulate.sorted_spikes_simulation.make_simulated_data``,
which is also what ``scripts/figures/generate_figure01.py`` uses; these helpers
add an event to it and say where an event may safely go.

Besides a single straight replay, the same machinery writes the activity the
detectors should not call replay, or should split: a burst whose firing carries
no place (``placeless_burst``), a burst whose represented place jumps at random
(``jumping_burst``), and replays made of several parts with a jump or a silent
gap between them (``piecewise_replay``). Each is an ``InjectedActivity``: when
it happens, which place it represents at each moment, how fast every cell fires,
which spans are its spatial parts, and what it truly is, as one of
``TRUTH_LABELS``: a continuous replay is non-local representation and a
sequence; a jumping burst is non-local representation but no sequence; local
coding and a place-less burst are neither.

Placement is not a detail. A window straddling the end of an immobile stretch
loses its center to the speed gate, and because jumps are forward the sequence
ends one window earlier still, so an event placed too near the edge measures
the speed gate rather than whatever it was meant to measure.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "TRUTH_LABELS",
    "Gap",
    "InjectedActivity",
    "InjectedReplay",
    "Segment",
    "field_rates",
    "immobile_intervals",
    "inject_activity",
    "inject_replay",
    "jumping_burst",
    "piecewise_replay",
    "place_field_means",
    "placeless_burst",
    "placements",
    "without",
]

TRUTH_LABELS = ("continuous_replay", "non_local", "local", "no_content")
"""What an event truly is, for the two targets scored separately.

``continuous_replay`` is non-local representation and a sequence; ``non_local``
is non-local representation without a sequence; ``local`` and ``no_content``
are neither."""

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
        How far it travels, signed: negative runs toward lower positions. Its
        speed is this over ``duration_s``.
    duration_s : float
        Positive.
    peak_rate_hz : float
        Firing rate of a cell whose field the path passes over exactly. Not
        negative.
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

    def __post_init__(self) -> None:
        for field in ("start_cm", "path_cm", "duration_s", "peak_rate_hz", "at_s"):
            value = getattr(self, field)
            if not math.isfinite(value):
                raise ValueError(f"{field} must be finite, got {value!r}.")
        if self.duration_s <= 0:
            raise ValueError(f"duration_s must be positive, got {self.duration_s}.")
        if self.peak_rate_hz < 0:
            raise ValueError(
                f"peak_rate_hz must not be negative, got {self.peak_rate_hz}."
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int | np.integer):
            raise ValueError(f"seed must be an integer, got {self.seed!r}.")

    @property
    def end_s(self) -> float:
        return self.at_s + self.duration_s


def place_field_means(n_neurons: int, track_height_cm: float) -> NDArray:
    """Field centers tiled evenly along the track."""
    return np.linspace(0.0, track_height_cm, n_neurons)


@dataclass(frozen=True)
class InjectedActivity:
    """Activity to write into a recording, and what it truly is.

    Attributes
    ----------
    time_s : ndarray, shape (n_samples,)
        Evenly spaced sample times.
    position_cm : ndarray, shape (n_samples,)
        The place represented at each sample; NaN where the activity is
        silent or represents no place.
    rate_hz : ndarray, shape (n_samples, n_neurons)
        Each cell's firing rate at each sample.
    parts : tuple of (float, float)
        When each spatial part begins and ends.
    truth : str
        One of ``TRUTH_LABELS``.

    """

    time_s: NDArray
    position_cm: NDArray
    rate_hz: NDArray
    parts: tuple[tuple[float, float], ...]
    truth: str

    def __post_init__(self) -> None:
        if self.truth not in TRUTH_LABELS:
            raise ValueError(
                f"truth must be one of {TRUTH_LABELS}, got {self.truth!r}."
            )
        if not self.rate_hz.shape[0] == self.time_s.size == self.position_cm.size:
            raise ValueError(
                f"{self.time_s.size} samples carry {self.position_cm.size} positions "
                f"and {self.rate_hz.shape[0]} rows of rates."
            )

    @property
    def bounds(self) -> tuple[float, float]:
        """When the activity begins and ends."""
        return float(self.time_s[0]), float(self.time_s[-1])


@dataclass(frozen=True)
class Segment:
    """One constant-speed part of a piecewise replay."""

    duration_s: float
    start_cm: float
    path_cm: float


@dataclass(frozen=True)
class Gap:
    """A silent stretch between two parts of a piecewise replay."""

    duration_s: float


def _sample_times(at_s: float, duration_s: float, sampling_frequency: float) -> NDArray:
    """Evenly spaced samples from ``at_s`` to ``at_s + duration_s``, both included."""
    n_samples = max(2, int(duration_s * sampling_frequency))
    step_s = duration_s / (n_samples - 1)
    return at_s + np.arange(n_samples) * step_s


def field_rates(
    position_cm: NDArray, place_field_means: NDArray, peak_rate_hz: float
) -> NDArray:
    """Each cell's rate at each represented place; zero where there is none.

    Parameters
    ----------
    position_cm : ndarray, shape (n_samples,)
    place_field_means : ndarray, shape (n_neurons,)
    peak_rate_hz : float

    Returns
    -------
    rate_hz : ndarray, shape (n_samples, n_neurons)

    """
    position = np.asarray(position_cm, dtype=np.float64)
    rate: NDArray = peak_rate_hz * np.exp(
        -0.5
        * ((position[:, None] - np.asarray(place_field_means)[None, :]) / FIELD_STD_CM)
        ** 2
    )
    return np.where(np.isfinite(position)[:, None], rate, 0.0)


def inject_activity(
    spike_times: list[NDArray], activity: InjectedActivity, seed: int
) -> list[NDArray]:
    """Replace every cell's spikes over the activity's span with draws from its rates.

    Firing is drawn once per sample: a cell fires at a sample with probability
    ``rate * step``, so it fires at most once per sample. Spikes already inside
    the span are removed first, so the activity is not read on top of whatever
    the recording was doing.

    Parameters
    ----------
    spike_times : list of ndarray
        One array per cell. Not modified.
    activity : InjectedActivity
    seed : int

    Returns
    -------
    spike_times : list of ndarray
        A new list, sorted per cell.

    Raises
    ------
    ValueError
        There is not one rate per cell, or a rate would fire at every sample.

    """
    at, rate = activity.time_s, activity.rate_hz
    if len(spike_times) != rate.shape[1]:
        raise ValueError(f"{len(spike_times)} cells but {rate.shape[1]} rates.")
    step_s = float(at[1] - at[0])
    if (rate * step_s).max(initial=0.0) >= 1.0:
        raise ValueError(
            f"A rate of {rate.max()} Hz sampled every {step_s} s would fire at "
            "every sample; raise the sampling frequency."
        )
    fired = np.random.default_rng(seed).random(rate.shape) < rate * step_s
    injected = []
    for neuron, neuron_times in enumerate(spike_times):
        kept = neuron_times[(neuron_times < at[0]) | (neuron_times > at[-1])]
        injected.append(np.sort(np.concatenate([kept, at[fired[:, neuron]]])))
    return injected


def inject_replay(
    spike_times: list[NDArray],
    place_field_means: NDArray,
    replay: InjectedReplay,
    sampling_frequency: float,
) -> tuple[list[NDArray], tuple[float, float]]:
    """Add a replay that travels a given distance in a given time.

    Each cell fires at a rate that follows its own place field, read at the
    position the replay represents at that moment; see :func:`inject_activity`
    for how firing is drawn.

    Parameters
    ----------
    spike_times : list of ndarray
        One array per cell. Not modified.
    place_field_means : ndarray, shape (n_neurons,)
    replay : InjectedReplay
    sampling_frequency : float
        Rate at which the trajectory is sampled.

    Returns
    -------
    spike_times : list of ndarray
        A new list, sorted per cell.
    bounds : tuple of float
        When the event begins and ends.

    Raises
    ------
    ValueError
        ``peak_rate_hz / sampling_frequency`` is 1 or more, so the per-sample
        firing probability would saturate and the rate would silently be
        lower than asked; or there is not one place field per cell.

    """
    if len(spike_times) != len(place_field_means):
        raise ValueError(
            f"{len(spike_times)} cells but {len(place_field_means)} place fields."
        )
    at = _sample_times(replay.at_s, replay.duration_s, sampling_frequency)
    step_s = float(at[1] - at[0])
    if replay.peak_rate_hz * step_s >= 1.0:
        raise ValueError(
            f"A peak rate of {replay.peak_rate_hz} Hz sampled every {step_s} s "
            "would fire at every sample; raise the sampling frequency."
        )
    position_cm = replay.start_cm + replay.path_cm * np.arange(at.size) / (at.size - 1)
    activity = InjectedActivity(
        time_s=at,
        position_cm=position_cm,
        rate_hz=field_rates(position_cm, place_field_means, replay.peak_rate_hz),
        parts=((float(at[0]), float(at[-1])),),
        truth="continuous_replay",
    )
    return inject_activity(spike_times, activity, replay.seed), activity.bounds


def piecewise_replay(
    pieces: list[Segment | Gap],
    at_s: float,
    sampling_frequency: float,
    place_field_means: NDArray,
    peak_rate_hz: float,
) -> InjectedActivity:
    """A replay of several constant-speed parts, with jumps or silent gaps between.

    A jump is two segments in a row whose second does not start where the
    first ended; a chain is segments separated by gaps. A sample on the
    boundary between two pieces belongs to the earlier one.

    Parameters
    ----------
    pieces : list of Segment or Gap
        In order; the first and last must be segments.
    at_s : float
        When the first segment begins.
    sampling_frequency : float
    place_field_means : ndarray, shape (n_neurons,)
    peak_rate_hz : float

    Returns
    -------
    activity : InjectedActivity
        Its parts are the segments' spans; truth ``continuous_replay``.

    """
    if (
        not pieces
        or not isinstance(pieces[0], Segment)
        or not isinstance(pieces[-1], Segment)
    ):
        raise ValueError("A piecewise replay must begin and end with a segment.")
    ends = at_s + np.cumsum([piece.duration_s for piece in pieces])
    starts = ends - [piece.duration_s for piece in pieces]
    at = _sample_times(at_s, float(ends[-1] - at_s), sampling_frequency)
    # the piece each sample falls in, a boundary sample going to the earlier
    piece_index = np.minimum(np.searchsorted(ends, at, side="left"), len(pieces) - 1)
    position_cm = np.full(at.size, np.nan)
    for index, piece in enumerate(pieces):
        if isinstance(piece, Gap):
            continue
        here = piece_index == index
        fraction = (at[here] - starts[index]) / piece.duration_s
        position_cm[here] = piece.start_cm + piece.path_cm * fraction
    return InjectedActivity(
        time_s=at,
        position_cm=position_cm,
        rate_hz=field_rates(position_cm, place_field_means, peak_rate_hz),
        parts=tuple(
            (float(starts[index]), float(ends[index]))
            for index, piece in enumerate(pieces)
            if isinstance(piece, Segment)
        ),
        truth="continuous_replay",
    )


def placeless_burst(
    duration_s: float,
    at_s: float,
    sampling_frequency: float,
    place_field_means: NDArray,
    peak_rate_hz: float,
    track_cm: float,
) -> InjectedActivity:
    """A population burst whose firing represents no place.

    Every cell fires at one rate, the population rate of a representation held
    at mid-track divided evenly among the cells, so the burst holds as many
    spikes as a replay of the same duration and peak rate there.
    """
    at = _sample_times(at_s, duration_s, sampling_frequency)
    population_hz = field_rates(
        np.array([track_cm / 2]), place_field_means, peak_rate_hz
    ).sum()
    n_neurons = len(place_field_means)
    return InjectedActivity(
        time_s=at,
        position_cm=np.full(at.size, np.nan),
        rate_hz=np.full((at.size, n_neurons), population_hz / n_neurons),
        parts=((float(at[0]), float(at[-1])),),
        truth="no_content",
    )


def jumping_burst(
    duration_s: float,
    at_s: float,
    sampling_frequency: float,
    place_field_means: NDArray,
    peak_rate_hz: float,
    track_cm: float,
    seed: int,
    hold_s: float = 0.015,
) -> InjectedActivity:
    """A burst representing a place drawn at random along the track every ``hold_s``.

    Cells fire by their fields at each held place, so the burst carries
    spatial information but no sequence. The places are drawn from their own
    stream of ``seed``, apart from the one that draws the spikes.
    """
    at = _sample_times(at_s, duration_s, sampling_frequency)
    hold = np.floor((at - at_s) / hold_s).astype(int)
    places = np.random.default_rng([seed, 1]).uniform(0.0, track_cm, hold.max() + 1)
    position_cm = places[hold]
    return InjectedActivity(
        time_s=at,
        position_cm=position_cm,
        rate_hz=field_rates(position_cm, place_field_means, peak_rate_hz),
        parts=((float(at[0]), float(at[-1])),),
        truth="non_local",
    )


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
        A NaN speed does not establish stillness, so it breaks a stretch.
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
