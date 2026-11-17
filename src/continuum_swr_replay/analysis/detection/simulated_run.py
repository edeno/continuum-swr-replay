"""The simulated run of Figure 1, decoded by the switching state-space model.

Figure 1 (``scripts/figures/generate_figure01.py``) shows the model at work on a
recording whose spiking is known: an animal runs back and forth on a linear
track, pausing at the far end, and during four of those pauses the spikes follow
a pattern the simulation puts there (:data:`PATTERNS`). :func:`simulate_and_decode`
builds that recording and treats it as the analyses treat a real one: the
detector is built from the real decodes' settings (:func:`make_example_detector`),
fitted by the same call (``estimate_parameters``), and its non-local events are
found by the same rule
(``event_definition.nonlocal_event_rule.detect_nonlocal_events``). What a real
HPC decode has and this does not: spike waveform features (the simulated cells
are sorted, so the detector is the sorted-spikes one the mPFC and OFC decodes
use), a track graph, and an artifact mask.

The recording is ``non_local_detector``'s ``make_simulated_data`` with three
changes. Its first sweep, which starts at the animal, is removed, and its second
sweep, which ends at the animal, is moved into that pause. A remote place held
still (:func:`splice_remote_stationary`) takes the pause the second sweep left.
Its silence is lengthened (:data:`SILENCE_DURATION_S`).

The memoryless comparison (``memoryless_sequence.comparison_simulation``) starts
from the same ``make_simulated_data`` and shares this module's decode grid
(:data:`DECODE_BIN_STRIDE`) and its remote stationary event, but it is a
different recording and a different decode: see that module.
"""

from dataclasses import dataclass

import numpy as np
import xarray as xr
from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.environment import Environment
from non_local_detector.simulate.sorted_spikes_simulation import (
    SAMPLING_FREQUENCY as SIMULATION_SAMPLING_FREQUENCY,
)
from non_local_detector.simulate.sorted_spikes_simulation import (
    TRACK_HEIGHT,
    make_simulated_data,
)

from continuum_swr_replay.analysis.event_definition.nonlocal_event_rule import (
    NON_LOCAL_STATES,
    detect_nonlocal_events,
    nonlocal_state_probability,
)
from continuum_swr_replay.decoding.config import NonLocalDecodingConfig
from continuum_swr_replay.decoding.fit import algorithm_params
from continuum_swr_replay.decoding.results import ENCODING_KEY
from continuum_swr_replay.decoding.transitions import nonlocal_detector_kwargs
from continuum_swr_replay.parameters import PLACE_BIN_SIZE_1D, SAMPLING_FREQUENCY

__all__ = [
    "DECODE_BIN_STRIDE",
    "PATTERNS",
    "REMOTE_SPIKE_INTERVAL_S",
    "REMOTE_STATIONARY_CM",
    "REMOTE_STATIONARY_S",
    "SILENCE_DURATION_S",
    "DecodedSimulation",
    "SimulatedRun",
    "estimate_state_space_model",
    "make_example_detector",
    "simulate_and_decode",
    "simulate_run",
    "splice_remote_stationary",
]

DECODE_BIN_STRIDE, _remainder = divmod(
    SIMULATION_SAMPLING_FREQUENCY, SAMPLING_FREQUENCY
)
"""Figure 1 strides the 1500 Hz simulation to the project's 500 Hz decode grid,
the grid the detector's transitions are built for."""
if _remainder:
    raise ValueError(
        f"the simulation's {SIMULATION_SAMPLING_FREQUENCY} Hz is not a whole "
        f"multiple of the decode's {SAMPLING_FREQUENCY} Hz"
    )

REMOTE_STATIONARY_CM = 42.5
"""The place a remote stationary event holds, far from the animal's 170 cm."""

REMOTE_STATIONARY_S = 0.300
REMOTE_SPIKE_INTERVAL_S = 0.012
"""One neuron firing at ~83 Hz, matching the other events' spike density."""

SILENCE_DURATION_S = 0.75
"""The simulator's silence lasts 0.5 s; lengthened so that the No-Spike state
clears the ramps the transitions put at its edges and reads at full strength."""

# Rows of ``make_simulated_data``'s ``event_times``. The simulator puts one event
# in each pause, in this order; its fifth, "local" event has zero duration.
_OUTBOUND_SWEEP_ROW, _FRAGMENTED_ROW, _INBOUND_SWEEP_ROW, _SILENCE_ROW = 0, 1, 2, 3

PATTERNS = ("sweep", "fragmented", "stationary", "silence")
"""The spiking the simulation puts in four pauses, in time order: a sweep along
the track that ends at the animal, places scattered over the track, a remote
place held still, and no spikes at all."""


def splice_remote_stationary(
    spike_times: list[np.ndarray],
    place_field_means: np.ndarray,
    start_s: float,
) -> tuple[list[np.ndarray], tuple[float, float]]:
    """Add a non-local representation that is held still rather than travelled.

    The package simulates travelling replay, a stationary local event and
    silence, but not a remote place held still. That case separates two things
    a sequence detector does for different reasons: it absorbs a stationary
    representation when the representation is the animal's own place, and it
    declines a stationary representation when it is somewhere else. Only the
    second is the method behaving as designed.

    Every spike in the event's span is removed, and the cell whose field lies
    nearest :data:`REMOTE_STATIONARY_CM` fires every
    :data:`REMOTE_SPIKE_INTERVAL_S` for :data:`REMOTE_STATIONARY_S`.

    Parameters
    ----------
    spike_times : list of np.ndarray
        Each cell's spike times, in seconds.
    place_field_means : np.ndarray, shape (n_neurons,)
        Each cell's place-field center, in cm.
    start_s : float
        When the event starts, in seconds.

    Returns
    -------
    spliced : list of np.ndarray
        The spike times with the event in place.
    bounds : tuple of float
        The event's start and end, in seconds.
    """
    end_s = start_s + REMOTE_STATIONARY_S
    neuron = int(np.argmin(np.abs(place_field_means - REMOTE_STATIONARY_CM)))
    spliced = [
        neuron_times[(neuron_times < start_s) | (neuron_times > end_s)]
        for neuron_times in spike_times
    ]
    spliced[neuron] = np.sort(
        np.concatenate(
            [spliced[neuron], np.arange(start_s, end_s, REMOTE_SPIKE_INTERVAL_S)]
        )
    )
    return spliced, (start_s, end_s)


def make_example_detector() -> NonLocalSortedSpikesDetector:
    """The unfitted detector of Figure 1, built as a real decode's is.

    The likelihood parameters, the discrete transitions (setting P30-B) and the
    Local kernel are the real decodes': ``decoding.config.NonLocalDecodingConfig``
    and ``decoding.transitions.nonlocal_detector_kwargs``, put together as
    ``decoding.fit.create_detector`` puts them together. The place bins are the
    real decodes' size; the simulated track is a line, so there is no track
    graph.

    Returns
    -------
    NonLocalSortedSpikesDetector
        The detector class the mPFC and OFC decodes use.
    """
    config = NonLocalDecodingConfig()
    return NonLocalSortedSpikesDetector(
        environments=Environment(place_bin_size=PLACE_BIN_SIZE_1D),
        **algorithm_params(config, "sorted_spikes"),
        **nonlocal_detector_kwargs(config),
    )


def estimate_state_space_model(
    time: np.ndarray,
    position: np.ndarray,
    spike_times: list[np.ndarray],
    is_training: np.ndarray,
) -> tuple[NonLocalSortedSpikesDetector, xr.Dataset]:
    """Fit Figure 1's detector to a simulated recording as a real decode is fitted.

    The encoding model and the transitions are estimated by EM
    (``estimate_parameters``, as in ``decoding.fit.perform_decoding``).

    EM also replaces the detector's initial conditions with the posterior at the
    recording's first time bin, and a detector left like that returns an all-zero
    posterior for any stretch that starts elsewhere (non_local_detector issue
    232). The initial conditions are therefore put back to the model's prior
    after the fit, so that the returned detector can ``predict`` on a window of
    the recording; ``results``, the decode of the whole recording, is the EM
    pass's own.

    Parameters
    ----------
    time : np.ndarray, shape (n_time,)
        The decode grid, in seconds.
    position : np.ndarray, shape (n_time,)
        The animal's position on it, in cm.
    spike_times : list of np.ndarray
        Each cell's spike times, in seconds.
    is_training : np.ndarray, shape (n_time,)
        The bins the place fields are fitted on.

    Returns
    -------
    detector : NonLocalSortedSpikesDetector
        Fitted, with the prior's initial conditions.
    results : xr.Dataset
        The decode of the recording it was fitted on.
    """
    detector = make_example_detector()
    prior = np.array(detector.discrete_initial_conditions)
    results = detector.estimate_parameters(
        position_time=time,
        position=position,
        spike_times=spike_times,
        time=time,
        is_training=is_training,
    )
    detector.discrete_initial_conditions = prior
    detector.initialize_initial_conditions()
    return detector, results


@dataclass(frozen=True)
class SimulatedRun:
    """The simulated run of Figure 1, before it is decoded.

    Every array with a time axis is on the decode grid, ``time``.
    """

    time: np.ndarray  # (n_time,), s
    position: np.ndarray  # (n_time,), cm
    speed: np.ndarray  # (n_time,), cm/s
    spike_times: list[np.ndarray]  # spike times (s) per cell, in simulator order
    place_field_means: np.ndarray  # (n_neurons,), cm, in the order of spike_times
    patterns: dict[str, tuple[float, float]]  # s; when each of PATTERNS occurs
    is_pattern: np.ndarray  # (n_time,), bool: inside a pattern or a simulator event


@dataclass(frozen=True)
class DecodedSimulation:
    """The simulated run and what the detector decoded from it.

    Every array with a time axis is on the decode grid, ``time``.
    """

    time: np.ndarray  # (n_time,), s
    position: np.ndarray  # (n_time,), cm
    speed: np.ndarray  # (n_time,), cm/s
    sorted_spikes: list[np.ndarray]  # spike times (s) per cell, by field position
    place_field_means: np.ndarray  # (n_neurons,), cm, in the order of sorted_spikes
    state_names: tuple[str, ...]  # the detector's states, (n_states,)
    state_probabilities: np.ndarray  # (n_time, n_states)
    posterior: np.ndarray  # (n_time, n_position): every state with a position
    non_local_posterior: np.ndarray  # (n_time, n_position): the non-local states
    position_bins: np.ndarray  # (n_position,), cm
    non_local_probability: np.ndarray  # (n_time,)
    patterns: dict[str, tuple[float, float]]  # s; when each of PATTERNS occurs
    non_local_events: list[tuple[float, float]]  # s; by the project's event rule


def simulate_run(n_neurons: int = 30, seed: int = 0) -> SimulatedRun:
    """Simulate the run of Figure 1: the simulator's recording with its patterns.

    Parameters
    ----------
    n_neurons : int, optional
        Number of place cells, by default 30.
    seed : int, optional
        Seed of ``make_simulated_data``'s random generator, by default 0.

    Returns
    -------
    SimulatedRun
    """
    (
        speed,
        position,
        spike_times,
        time,
        event_times,
        _sampling_frequency,
        is_event,
        _place_fields,
    ) = make_simulated_data(n_neurons=n_neurons, seed=seed)
    # (n_events, 2) -> one (start, end) per simulated event, in s
    event_times = [tuple(float(t) for t in row) for row in event_times]
    place_field_means = np.linspace(0, float(TRACK_HEIGHT), n_neurons)

    # The outbound sweep starts at the animal; the inbound sweep ends there. The
    # inbound sweep is moved into the outbound sweep's pause (the animal sits at
    # the same end of the track in both), and a remote place held still takes
    # the pause it leaves: the splice clears that pause's spikes with its span.
    outbound = event_times[_OUTBOUND_SWEEP_ROW]
    inbound = event_times[_INBOUND_SWEEP_ROW]
    sweep = (outbound[0], outbound[0] + (inbound[1] - inbound[0]))
    cleared_until = max(outbound[1], sweep[1])
    spike_times = [
        np.sort(
            np.concatenate(
                [
                    cell[(cell < outbound[0]) | (cell > cleared_until)],
                    cell[(cell >= inbound[0]) & (cell <= inbound[1])]
                    - inbound[0]
                    + sweep[0],
                ]
            )
        )
        for cell in spike_times
    ]
    spike_times, stationary = splice_remote_stationary(
        spike_times, place_field_means, inbound[0]
    )
    if inbound[1] > stationary[1]:
        raise ValueError(
            f"the simulator's second sweep ({inbound[0]:.3f}-{inbound[1]:.3f} s) "
            "outlasts the remote stationary event that replaces it"
        )

    silence_start = event_times[_SILENCE_ROW][0]
    silence = (silence_start, silence_start + SILENCE_DURATION_S)
    spike_times = [
        cell[(cell < silence[0]) | (cell > silence[1])] for cell in spike_times
    ]

    patterns = {
        "sweep": sweep,
        "fragmented": event_times[_FRAGMENTED_ROW],
        "stationary": stationary,
        "silence": silence,
    }
    for start, end in (sweep, stationary, silence):
        is_event = is_event | ((time >= start) & (time <= end))

    # the decode grid; the spike times keep their own resolution
    time, position, speed, is_event = (
        np.asarray(array[::DECODE_BIN_STRIDE]).squeeze()
        for array in (time, position, speed, is_event)
    )

    return SimulatedRun(
        time=time,
        position=position,
        speed=speed,
        spike_times=spike_times,
        place_field_means=place_field_means,
        patterns=patterns,
        is_pattern=is_event,
    )


def simulate_and_decode(n_neurons: int = 30, seed: int = 0) -> DecodedSimulation:
    """Simulate the run of Figure 1, decode it and find its non-local events.

    The model is fitted (:func:`estimate_state_space_model`) on the time outside
    the patterns: spikes inside a pattern do not shape the place fields, as a
    real decode's training mask keeps ripple times out.

    Parameters
    ----------
    n_neurons : int, optional
        Number of place cells, by default 30.
    seed : int, optional
        Seed of ``make_simulated_data``'s random generator, by default 0.

    Returns
    -------
    DecodedSimulation
    """
    run = simulate_run(n_neurons=n_neurons, seed=seed)
    time, speed = run.time, run.speed
    detector, results = estimate_state_space_model(
        time, run.position, run.spike_times, ~run.is_pattern
    )

    non_local_probability = nonlocal_state_probability(
        results.acausal_state_probabilities
    ).squeeze()
    events, _ = detect_nonlocal_events(non_local_probability, speed, time)

    # No-Spike has no position, so its bins are NaN and drop out of the sums.
    by_state = results["acausal_posterior"].unstack("state_bins")
    on_grid = ("time", "position")

    # cells in the order of their fitted place-field peaks, for the rasters
    encoding_model = detector.encoding_model_[ENCODING_KEY]
    order = np.argsort(np.argmax(np.asarray(encoding_model["place_fields"]), axis=1))

    return DecodedSimulation(
        time=time,
        position=run.position,
        speed=speed,
        sorted_spikes=[run.spike_times[i] for i in order],
        place_field_means=run.place_field_means[order],
        state_names=tuple(
            str(name) for name in results.acausal_state_probabilities.states.values
        ),
        state_probabilities=np.asarray(results.acausal_state_probabilities.values),
        posterior=np.asarray(by_state.sum("state").transpose(*on_grid).values),
        non_local_posterior=np.asarray(
            by_state.sel(state=list(NON_LOCAL_STATES))
            .sum("state")
            .transpose(*on_grid)
            .values
        ),
        position_bins=np.asarray(by_state.coords["position"].values),
        non_local_probability=np.asarray(non_local_probability.values),
        patterns=run.patterns,
        non_local_events=list(
            zip(events["start_time"].tolist(), events["end_time"].tolist(), strict=True)
        ),
    )
