"""The Figure 1 simulation with the memoryless comparison's added events, decoded.

The supplemental memoryless comparison figure
(``scripts/figures/generate_memoryless_comparison.py``) and the reference check
(``scripts/checks/compare_memoryless_to_reference.py``) read one simulation
(:func:`simulate_and_decode`): the package's Figure 1 simulation with a remote
stationary event, a short path, a short replay and a stray window added
(their rows of ``event_times`` are :data:`SHORT_PATH_EVENT` and the other
``*_EVENT`` constants), decoded window by window under a uniform prior and by
the switching state-space model fitted on the same non-event data. The figure's
numbers are those of :data:`N_NEURONS` cells decoded under :data:`CONFIG`.
"""

import numpy as np
from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.environment import Environment
from non_local_detector.simulate.sorted_spikes_simulation import (
    SAMPLING_FREQUENCY,
    TRACK_HEIGHT,
    make_simulated_data,
)

from continuum_swr_replay.analysis.detection.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.detect import (
    detect_candidate_sequences,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.likelihood import (
    interior_bin_distances_cm,
    sorted_spikes_epoch_log_likelihood,
    window_midpoints_s,
    with_reference_rate_floor,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.shuffle import (
    DEFAULT_N_SHUFFLES,
    ShuffleSeed,
    shuffle_candidate_sorted,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.simulation import (
    InjectedReplay,
    immobile_intervals,
    inject_replay,
    stray_window_spikes,
    without,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.window_statistics import (
    aggregate_windows,
    normalize_windows,
)
from continuum_swr_replay.decoding.results import ENCODING_KEY
from continuum_swr_replay.parameters import LOCAL_POSITION_STD

__all__ = [
    "CONFIG",
    "CONTINUOUS_INBOUND_EVENT",
    "CONTINUOUS_OUTBOUND_EVENT",
    "DECODE_BIN_STRIDE",
    "N_NEURONS",
    "REMOTE_STATIONARY_CM",
    "REMOTE_STATIONARY_EVENT",
    "REMOTE_STATIONARY_S",
    "SHORT_PATH",
    "SHORT_PATH_EVENT",
    "SHORT_REPLAY",
    "SHORT_REPLAY_EVENT",
    "SHUFFLE_RUN_SEED",
    "STRAY_WINDOW_AT_S",
    "STRAY_WINDOW_EVENT",
    "STRAY_WINDOW_N_LOCAL",
    "STRAY_WINDOW_STILL_S",
    "TRACK_HEIGHT_CM",
    "simulate_and_decode",
    "splice_remote_stationary",
]

TRACK_HEIGHT_CM = float(TRACK_HEIGHT)
"""The simulation package's own track length, read rather than restated."""

DECODE_BIN_STRIDE = 3
"""Figure 1 strides the 1500 Hz simulation to the project's 500 Hz decode grid."""

N_NEURONS = 25
"""Cells in the simulation the figure and its recorded numbers are made from,
and in the memoryless sweeps' recording (``sweep.build_recording``)."""

CONFIG = MemorylessSequenceConfig()
"""The sliding-window detector's settings on that 500 Hz grid, in the figure
and in the sweeps."""

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
it too long; the panel title reads the ratio off the data rather than stating
it here. The window puts a floor under any duration the method can report."""

SHORT_PATH = InjectedReplay(
    start_cm=40.0, path_cm=30.0, duration_s=0.300, peak_rate_hz=40.0, at_s=20.50, seed=0
)
"""A replay that travels only 30 cm. The detector decodes it well, but the
dispersion criterion needs a path of about 42 cm, so it is expected to reject
the candidate."""

SHORT_PATH_EVENT, SHORT_REPLAY_EVENT = 6, 7
CONTINUOUS_OUTBOUND_EVENT, CONTINUOUS_INBOUND_EVENT = 0, 2
REMOTE_STATIONARY_EVENT = 5
STRAY_WINDOW_EVENT = 8
"""Rows of ``event_times`` in :func:`simulate_and_decode`, in the order the
events are added."""

SHUFFLE_RUN_SEED = 20260922
"""Seeds the place-cell identity shuffle, as in the sweep."""

REMOTE_STATIONARY_CM = 42.5
"""The place a remote stationary event holds, far from the animal's 170 cm."""

REMOTE_STATIONARY_S = 0.300
REMOTE_SPIKE_INTERVAL_S = 0.012
"""One neuron firing at ~83 Hz, matching the other events' spike density."""

STRAY_WINDOW_AT_S = 31.40
"""Where the hand-placed stray-window event begins: in the stillness before the
inbound replay, among candidates of local coding that no caption claim reads.
The stillness before the outbound replay is no good: that replay's candidate
absorbs the local coding before it, so an event there changes its verdict."""

STRAY_WINDOW_N_LOCAL = 2
STRAY_WINDOW_STILL_S = 0.120
"""The still parts are carried by the two cells whose fields lie nearest the
animal, firing in turn for 120 ms either side of the far window. Cells further
along the track would pull the decode away from the animal, and the
state-space detector then reads those parts as non-local. The parts are as
long as the dispersion allows: the one far window keeps the event above 12 cm
however long the still parts around it run, until about 180 ms each."""

STRAY_WINDOW_PLACEMENT_MARGIN_S = 0.4
"""How far the event's cleared span must sit from every other event."""


def splice_remote_stationary(
    spike_times: list[np.ndarray],
    place_field_means: np.ndarray,
    start_s: float,
) -> tuple[list[np.ndarray], tuple[float, float]]:
    """Add a non-local representation that is held still rather than travelled.

    The package simulates travelling replay, a stationary local event and
    silence, but not a remote place held still. That case separates two things
    the detector does for different reasons: it absorbs a stationary
    representation when the representation is the animal's own place, and it
    declines a stationary representation when it is somewhere else. Only the
    second is the method behaving as designed.
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


def simulate_and_decode(n_neurons: int, config: MemorylessSequenceConfig) -> dict:
    """Decode the Figure 1 simulation window by window under a uniform prior."""
    (
        speed,
        position,
        spike_times,
        time,
        event_times,
        _sampling_frequency,
        is_event,
        _place_fields,
    ) = make_simulated_data(n_neurons=n_neurons)

    place_field_means = np.linspace(0, TRACK_HEIGHT_CM, n_neurons)

    # the package has no remote stationary event; add one in the last pause,
    # and hold it out of training like every other event
    spike_times, remote_stationary = splice_remote_stationary(
        spike_times, place_field_means, float(event_times[4, 0])
    )
    event_times = np.vstack([event_times, remote_stationary])
    is_event |= (time >= remote_stationary[0]) & (time <= remote_stationary[1])

    # two replays the detector is expected to mis-report, in the order
    # SHORT_PATH_EVENT, SHORT_REPLAY_EVENT
    for replay in (SHORT_PATH, SHORT_REPLAY):
        spike_times, bounds = inject_replay(
            spike_times,
            place_field_means,
            replay,
            sampling_frequency=SAMPLING_FREQUENCY,
        )
        event_times = np.vstack([event_times, bounds])
        is_event |= (time >= bounds[0]) & (time <= bounds[1])

    # a still event at the animal's place with one window decoded far away,
    # hand-placed in the first stillness clear of every other event. It is
    # written into the spikes that are decoded and nowhere else: the encoding
    # model is fitted on the recording as it was before, so adding this event
    # leaves every other panel exactly as it was
    training_spike_times, is_training_event = spike_times, is_event.copy()
    clear = without(
        immobile_intervals(
            time, speed, config.speed_threshold_cm_per_s, minimum_duration_s=1.0
        ),
        event_times,
        margin_s=STRAY_WINDOW_PLACEMENT_MARGIN_S,
    )

    stray_local_cm = float(np.interp(STRAY_WINDOW_AT_S, time, position))
    spike_times, stray = stray_window_spikes(
        spike_times,
        place_field_means,
        STRAY_WINDOW_AT_S,
        local_cm=stray_local_cm,
        still_s=STRAY_WINDOW_STILL_S,
        n_local=STRAY_WINDOW_N_LOCAL,
        window_s=config.window_duration_s,
        stride_s=config.stride_s,
    )
    if not (
        (clear[:, 0] <= stray.cleared[0]) & (stray.cleared[1] <= clear[:, 1])
    ).any():
        raise RuntimeError(
            f"The stray-window event at {STRAY_WINDOW_AT_S} s does not sit in "
            "stillness clear of every other event; move STRAY_WINDOW_AT_S."
        )
    event_times = np.vstack([event_times, stray.bounds])
    is_event |= (time >= stray.bounds[0]) & (time <= stray.bounds[1])

    time, position, speed, is_event, is_training_event = (
        array[::DECODE_BIN_STRIDE]
        for array in (time, position, speed, is_event, is_training_event)
    )

    # encode on non-event data only, the analogue of this project's no-ripple
    # training mask; spikes inside an event must not shape the place fields
    is_training = ~is_training_event

    # the grid both procedures decode on comes from the config, matching the sweep
    detector = NonLocalSortedSpikesDetector(
        environments=[Environment(place_bin_size=config.place_bin_size_cm)],
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={"position_std": 6.0, "block_size": int(2**8)},
        local_position_std=LOCAL_POSITION_STD,
    )
    # fit rather than estimate_parameters, matching the sweep: EM would
    # re-estimate the place fields and the transitions on the data it is then
    # read on, and the two scripts should not differ in that.
    detector.fit(
        position_time=time,
        position=position,
        spike_times=training_spike_times,
        is_training=is_training,
    )

    # one set of place fields for both procedures, read from the detector that
    # fitted them; the sliding window decodes them as the reference does, with
    # its floor under every field
    encoding_model = with_reference_rate_floor(
        detector.encoding_model_[ENCODING_KEY], config
    )
    environment = encoding_model["environment"]

    result = sorted_spikes_epoch_log_likelihood(
        time, position, spike_times, encoding_model, config
    )
    position, speed = position[: result.time.size], speed[: result.time.size]

    posterior = normalize_windows(
        aggregate_windows(
            result.log_likelihood, config.window_n_samples, config.stride_n_samples
        )
    )
    midpoint_s = window_midpoints_s(result.time, config)
    window_speed = np.interp(midpoint_s, result.time, speed)
    bin_distance_cm = interior_bin_distances_cm(environment)
    candidates = detect_candidate_sequences(
        posterior, midpoint_s, bin_distance_cm, window_speed, config
    )
    # the reference's own significance test, on the full decode grid: the
    # likelihood's grid has lost the final row, which spikes are binned against
    shuffles = {
        int(i): shuffle_candidate_sorted(
            int(candidates.runs[i, 0]),
            int(candidates.runs[i, 1]),
            time,
            spike_times,
            encoding_model,
            bin_distance_cm,
            config,
            seed=ShuffleSeed(run=SHUFFLE_RUN_SEED, session="figure-one-simulation"),
            n_shuffles=DEFAULT_N_SHUFFLES,
        )
        for i in np.flatnonzero(candidates.is_replay)
    }

    return {
        "detector": detector,
        "time": result.time,
        "is_event": is_event[: result.time.size],
        "position": position,
        "speed": speed,
        "spike_times": spike_times,
        "event_times": event_times,
        "place_bin_centers_cm": result.place_bin_centers_cm,
        "posterior": posterior,
        "midpoint_s": midpoint_s,
        "window_speed": window_speed,
        "candidates": candidates,
        "shuffles": shuffles,
        "bin_distance_cm": bin_distance_cm,
        "stray_window": stray,
        "stray_window_local_cm": stray_local_cm,
        "stray_window_remote_cm": float(place_field_means[stray.remote_cell]),
    }
