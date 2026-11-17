"""The memoryless comparison's simulation: Figure 1's run with added events, decoded.

The supplemental memoryless comparison figure
(``scripts/figures/generate_supp_memoryless_comparison.py``) and the reference check
(``scripts/checks/compare_memoryless_to_reference.py``) read one simulation
(:func:`simulate_and_decode`): the simulated run of Figure 1
(``analysis.detection.simulated_run.simulate_run``, whose patterns are the first
rows of ``event_times``) with five replays added: a short path, a short
replay, one far from the animal, a sparse one and a fast one
(:data:`SHORT_PATH_EVENT` and the other ``*_EVENT`` constants name the rows). It is decoded twice from one set of place fields: window by window under
a uniform prior, as the sliding-window detector does, and by the switching
state-space model of Figure 1, fitted as a real decode is
(``simulated_run.estimate_state_space_model``), whose non-local events are
found by the project's rule (``nonlocal_event_rule.detect_nonlocal_events``).
The sliding-window detector's settings are :data:`CONFIG`.
"""

import numpy as np
from non_local_detector.simulate.sorted_spikes_simulation import (
    SAMPLING_FREQUENCY,
)

from continuum_swr_replay.analysis.detection.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.simulation import (
    InjectedReplay,
    added_spike_span,
    inject_replay,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.sweep import (
    CONFIG,
    N_NEURONS,
    SHUFFLE_RUN_SEED,
    run_sliding_window,
    sliding_window_fields,
)
from continuum_swr_replay.analysis.detection.simulated_run import (
    PATTERNS,
    estimate_state_space_model,
    read_state_space_decode,
    simulate_run,
)
from continuum_swr_replay.decoding.results import ENCODING_KEY

__all__ = [
    "CONFIG",
    "FAR_REPLAY",
    "FAR_REPLAY_EVENT",
    "FAST_REPLAY",
    "FAST_REPLAY_EVENT",
    "FRAGMENTED_EVENT",
    "N_NEURONS",
    "REMOTE_STATIONARY_EVENT",
    "REPLAY_EVENTS",
    "SHORT_PATH",
    "SHORT_PATH_EVENT",
    "SHORT_REPLAY",
    "SHORT_REPLAY_EVENT",
    "SHUFFLE_RUN_SEED",
    "SILENCE_EVENT",
    "SPARSE_REPLAY",
    "SPARSE_REPLAY_EVENT",
    "SWEEP_EVENT",
    "simulate_and_decode",
]

SHORT_REPLAY = InjectedReplay(
    start_cm=35.0,
    path_cm=100.0,
    duration_s=0.080,
    peak_rate_hz=120.0,
    at_s=44.40,
    seed=1,
)
"""A replay that travels far but lasts only 80 ms, firing strongly enough that
spike count is not the issue. The detector is expected to find it and report
it too long; the figure reads the ratio off the data rather than stating
it here. The window puts a floor under any duration the method can report."""

SHORT_PATH = InjectedReplay(
    start_cm=40.0, path_cm=30.0, duration_s=0.300, peak_rate_hz=40.0, at_s=20.50, seed=0
)
"""A replay that travels only 30 cm. The detector decodes it well, but the
dispersion criterion needs a path of about 42 cm, so it is expected to reject
the candidate."""

FAR_REPLAY = InjectedReplay(
    start_cm=20.0, path_cm=90.0, duration_s=0.300, peak_rate_hz=60.0, at_s=58.30, seed=2
)
"""A replay within the published definition's reach that stays away from the
animal: 90 cm in 300 ms, ending 60 cm short of where the animal rests. Both
detectors are expected to report it, at much the same times."""

SPARSE_REPLAY = InjectedReplay(
    start_cm=30.0, path_cm=90.0, duration_s=0.300, peak_rate_hz=20.0, at_s=7.40, seed=0
)
"""The same path and duration as :data:`FAR_REPLAY` at a third of its firing
rate: a handful of spikes. Its path and duration are within the published
definition's reach, but the detector is expected to decline it; the windows
hold too few spikes to decode a steady sequence."""

FAST_REPLAY = InjectedReplay(
    start_cm=20.0, path_cm=90.0, duration_s=0.120, peak_rate_hz=60.0, at_s=31.50, seed=0
)
"""The far replay's path and firing rate in 120 ms, at 7.5 m/s: longer than the
published 100 ms and far from the animal, but within one 80 ms window the
represented position crosses most of the path. The detector is expected to
decline it."""

SWEEP_EVENT, FRAGMENTED_EVENT, REMOTE_STATIONARY_EVENT, SILENCE_EVENT = (
    PATTERNS.index(name) for name in ("sweep", "fragmented", "stationary", "silence")
)
(
    SHORT_PATH_EVENT,
    SHORT_REPLAY_EVENT,
    FAR_REPLAY_EVENT,
    SPARSE_REPLAY_EVENT,
    FAST_REPLAY_EVENT,
) = (len(PATTERNS) + added for added in range(5))
"""Rows of ``event_times`` in :func:`simulate_and_decode`: Figure 1's patterns
(``simulated_run.PATTERNS``), then the events added here, in the order they are
added."""

REPLAY_EVENTS = (
    SWEEP_EVENT,
    FRAGMENTED_EVENT,
    REMOTE_STATIONARY_EVENT,
    SHORT_PATH_EVENT,
    SHORT_REPLAY_EVENT,
    FAR_REPLAY_EVENT,
    SPARSE_REPLAY_EVENT,
    FAST_REPLAY_EVENT,
)
"""Every labeled event in which the spikes represent a place other than the
animal's: all but the silence."""


def simulate_and_decode(n_neurons: int, config: MemorylessSequenceConfig) -> dict:
    """Simulate the comparison's recording and decode it both ways.

    Parameters
    ----------
    n_neurons : int
        Number of place cells; the figure's is :data:`N_NEURONS`.
    config : MemorylessSequenceConfig
        The sliding-window detector's settings.

    Returns
    -------
    dict
        The recording (``time``, ``position``, ``spike_times``,
        ``event_times``), the sliding-window decode and its candidate sequences
        (``posterior``, ``midpoint_s``, ``candidates``, ``shuffles``), and the
        state-space decode on ``time`` (``state_names``,
        ``state_probabilities``, ``state_space_posterior`` over
        ``state_space_position_bins``, and its events,
        ``state_space_intervals``; ``simulated_run.read_state_space_decode``).
    """
    run = simulate_run(n_neurons=n_neurons)
    time, position, speed = run.time, run.position, run.speed
    spike_times, place_field_means = run.spike_times, run.place_field_means
    event_times = np.array(list(run.patterns.values()))
    is_event = run.is_pattern.copy()

    # the replays added to the run, in the order of their rows of event_times
    # (SHORT_PATH_EVENT to FAST_REPLAY_EVENT)
    for replay in (SHORT_PATH, SHORT_REPLAY, FAR_REPLAY, SPARSE_REPLAY, FAST_REPLAY):
        with_replay, bounds = inject_replay(
            spike_times,
            place_field_means,
            replay,
            sampling_frequency=SAMPLING_FREQUENCY,
        )
        # an event is when its spikes are, as the run's patterns are
        event_times = np.vstack(
            [event_times, added_spike_span(spike_times, with_replay)]
        )
        is_event |= (time >= bounds[0]) & (time <= bounds[1])
        spike_times = with_replay

    # encode on non-event data only, the analogue of this project's no-ripple
    # training mask; spikes inside an event must not shape the place fields
    detector, results = estimate_state_space_model(
        time, position, spike_times, ~is_event
    )
    decode = read_state_space_decode(results, speed, time)

    # one set of place fields for both procedures, read from the detector that
    # fitted them, and the sliding window run as the sweeps run it
    fields = sliding_window_fields(detector.encoding_model_[ENCODING_KEY], config)
    sliding = run_sliding_window(
        fields, spike_times, (time, position, speed), "figure-one-simulation", config
    )

    return {
        "detector": detector,
        "time": time,
        # the grid spikes are binned against: the decode grid itself
        "binning_time": time,
        "position": position,
        "spike_times": spike_times,
        "event_times": event_times,
        "place_bin_centers_cm": fields["place_bin_centers_cm"],
        "posterior": sliding.posterior,
        "midpoint_s": sliding.midpoint_s,
        "window_speed": sliding.window_speed,
        "candidates": sliding.candidates,
        "shuffles": sliding.shuffles,
        "bin_distance_cm": fields["bin_distance_cm"],
        "state_names": decode.state_names,
        "state_probabilities": decode.state_probabilities,
        "state_space_posterior": decode.posterior,
        "state_space_position_bins": decode.position_bins,
        "state_space_intervals": decode.non_local_events,
    }
