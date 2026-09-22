"""The memoryless detector on the simulation this project's Figure 1 explains.

``make_simulated_data`` produces one 65 s recording holding five labeled
events: a continuous outbound replay, a fragmented replay, a continuous
inbound replay, a silent stretch, and a local event. Figure 1 runs the
switching state-space model on it, so running the ported sliding-window
detector on the same recording puts both against one ground truth.

What these tests pin is the detector's blind spot rather than its accuracy.
It scores each window on posterior spread and on the jump to the next window,
and it carries no competing hypothesis for the animal's own position. Local
coding is smooth, narrow and slow, so it passes every window criterion and is
assembled into the same sequence as the replay that follows it.
"""

from dataclasses import dataclass

import numpy as np
import pytest
from non_local_detector.environment import Environment
from non_local_detector.likelihoods.sorted_spikes_kde import (
    fit_sorted_spikes_kde_encoding_model,
)

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.detect import (
    CandidateSequences,
    detect_candidate_sequences,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (
    interior_bin_distances_cm,
    sorted_spikes_epoch_log_likelihood,
    window_midpoints_s,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.statistics import (
    aggregate_windows,
    normalize_windows,
)

DECODE_BIN_STRIDE = 3
"""Figure 1 strides the 1500 Hz simulation to the project's 500 Hz decode grid."""

CONTINUOUS_OUTBOUND, FRAGMENTED, CONTINUOUS_INBOUND, NO_SPIKE, LOCAL = range(5)
REMOTE_STATIONARY = 5
"""A place held still, away from the animal. Added here; the package has none."""

TRACK_HEIGHT_CM = 170
REMOTE_STATIONARY_CM = 42.5
REMOTE_STATIONARY_S = 0.300
REMOTE_SPIKE_INTERVAL_S = 0.012
CONFIG = MemorylessSequenceConfig(sampling_frequency=500.0)
SAME_PLACE_CM = 15.0
"""How close a decoded center must be to count as the animal's own position."""


@dataclass
class Decoded:
    time: np.ndarray
    position: np.ndarray
    event_times: np.ndarray
    place_bin_centers_cm: np.ndarray
    midpoint_s: np.ndarray
    candidates: CandidateSequences
    encoding_model: dict
    spike_times: list[np.ndarray]
    posterior: np.ndarray
    bin_distance_cm: np.ndarray
    window_speed: np.ndarray


@pytest.fixture(scope="module")
def decoded() -> Decoded:
    from non_local_detector.simulate.sorted_spikes_simulation import make_simulated_data

    (
        _speed,
        position,
        spike_times,
        time,
        event_times,
        _sampling_frequency,
        is_event,
        _place_fields,
    ) = make_simulated_data(n_neurons=25)
    speed = _speed

    # a remote place held still, which the package does not simulate. It
    # separates two things the detector does for different reasons: absorbing a
    # stationary representation that is the animal's own place, and declining
    # one that is somewhere else.
    remote_start_s = float(event_times[4, 0])
    remote_end_s = remote_start_s + REMOTE_STATIONARY_S
    remote_neuron = int(
        np.argmin(
            np.abs(
                np.linspace(0, TRACK_HEIGHT_CM, len(spike_times)) - REMOTE_STATIONARY_CM
            )
        )
    )
    spike_times = [
        neuron[(neuron < remote_start_s) | (neuron > remote_end_s)]
        for neuron in spike_times
    ]
    spike_times[remote_neuron] = np.sort(
        np.concatenate(
            [
                spike_times[remote_neuron],
                np.arange(remote_start_s, remote_end_s, REMOTE_SPIKE_INTERVAL_S),
            ]
        )
    )
    event_times = np.vstack([event_times, [remote_start_s, remote_end_s]])
    is_event |= (time >= remote_start_s) & (time <= remote_end_s)

    time, position, speed, is_event = (
        array[::DECODE_BIN_STRIDE] for array in (time, position, speed, is_event)
    )
    environment = Environment(place_bin_size=CONFIG.place_bin_size_cm).fit_place_grid(
        position[:, None]
    )

    # encode on non-event data only: spikes inside an event must not shape the
    # place fields they are later decoded against
    is_training = ~is_event
    encoding_model = fit_sorted_spikes_kde_encoding_model(
        position_time=time[is_training],
        position=position[is_training, None],
        spike_times=[
            neuron[np.interp(neuron, time, is_training.astype(float)) > 0.5]
            for neuron in spike_times
        ],
        environment=environment,
        sampling_frequency=round(CONFIG.sampling_frequency),
        position_std=6.0,
        disable_progress_bar=True,
    )

    result = sorted_spikes_epoch_log_likelihood(
        time, position, spike_times, encoding_model, CONFIG
    )
    posterior = normalize_windows(
        aggregate_windows(
            result.log_likelihood, CONFIG.window_n_samples, CONFIG.stride_n_samples
        )
    )
    midpoint_s = window_midpoints_s(result.time, CONFIG)

    bin_distance_cm = interior_bin_distances_cm(environment)
    window_speed = np.interp(midpoint_s, result.time, speed[: result.time.size])

    return Decoded(
        time=result.time,
        position=position[: result.time.size],
        event_times=event_times,
        place_bin_centers_cm=result.place_bin_centers_cm,
        midpoint_s=midpoint_s,
        candidates=detect_candidate_sequences(
            posterior, midpoint_s, bin_distance_cm, window_speed, CONFIG
        ),
        encoding_model=encoding_model,
        spike_times=spike_times,
        posterior=posterior,
        bin_distance_cm=bin_distance_cm,
        window_speed=window_speed,
    )


def events_overlapping(decoded: Decoded, sequence: int) -> list[int]:
    """Which labeled events a detected sequence touches."""
    start, stop = decoded.candidates.runs[sequence]
    return [
        event
        for event, (event_start, event_end) in enumerate(decoded.event_times)
        if decoded.midpoint_s[start] < event_end
        and event_start < decoded.midpoint_s[stop]
    ]


def fraction_outside_its_event(decoded: Decoded, sequence: int) -> float:
    """Share of a detected sequence that lies outside the event it overlaps.

    Free of any "same place" tolerance, so it does not depend on a judgement
    call the way :func:`local_fraction` does.
    """
    start, stop = decoded.candidates.runs[sequence]
    midpoints = decoded.midpoint_s[start : stop + 1]
    event_start, event_end = decoded.event_times[
        events_overlapping(decoded, sequence)[0]
    ]
    return 1.0 - float(np.mean((midpoints >= event_start) & (midpoints <= event_end)))


def local_fraction(decoded: Decoded, sequence: int) -> float:
    """Share of a sequence's windows whose center is the animal's own place."""
    start, stop = decoded.candidates.runs[sequence]
    center = decoded.candidates.decisions.center_index[start : stop + 1]
    present = center >= 0
    center_cm = decoded.place_bin_centers_cm[center[present]]
    animal_cm = np.interp(
        decoded.midpoint_s[start : stop + 1][present], decoded.time, decoded.position
    )
    return float(np.mean(np.abs(center_cm - animal_cm) < SAME_PLACE_CM))


class TestWhatItAccepts:
    def test_both_continuous_replays_are_found_and_nothing_else_is(self, decoded):
        accepted = np.flatnonzero(decoded.candidates.is_replay)

        assert [events_overlapping(decoded, s) for s in accepted] == [
            [CONTINUOUS_OUTBOUND],
            [CONTINUOUS_INBOUND],
        ]

    def test_the_silent_and_local_events_are_not_accepted(self, decoded):
        for sequence in np.flatnonzero(decoded.candidates.is_replay):
            touched = events_overlapping(decoded, sequence)
            assert NO_SPIKE not in touched
            assert LOCAL not in touched

    def test_the_fragmented_event_is_too_short_to_be_a_sequence(self, decoded):
        """It lasts 36 ms, well under the 100 ms a sequence needs."""
        start_s, end_s = decoded.event_times[FRAGMENTED]

        assert end_s - start_s < CONFIG.minimum_duration_s


class TestLocalCodingIsAbsorbed:
    """The detector has no hypothesis that competes with a replay for a window."""

    def test_most_of_a_detected_event_lies_outside_the_event_it_found(self, decoded):
        """The measure that needs no "same place" judgement."""
        for sequence in np.flatnonzero(decoded.candidates.is_replay):
            assert fraction_outside_its_event(decoded, sequence) > 0.7

    def test_an_accepted_event_runs_far_longer_than_the_replay_in_it(self, decoded):
        for sequence in np.flatnonzero(decoded.candidates.is_replay):
            event = events_overlapping(decoded, sequence)[0]
            true_duration_s = float(
                decoded.event_times[event, 1] - decoded.event_times[event, 0]
            )

            assert decoded.candidates.duration_s[sequence] > 3 * true_duration_s

    def test_most_of_an_accepted_event_decodes_the_animal_s_own_position(self, decoded):
        for sequence in np.flatnonzero(decoded.candidates.is_replay):
            assert local_fraction(decoded, sequence) > 0.5

    def test_the_local_measure_does_not_turn_on_the_tolerance(self, decoded):
        """The decoded-to-animal distance is bimodal, so any cutoff between the
        two modes gives the same answer; only a cutoff below the grid's own
        resolution changes it."""
        sequence = np.flatnonzero(decoded.candidates.is_replay)[0]
        start, stop = decoded.candidates.runs[sequence]
        center = decoded.candidates.decisions.center_index[start : stop + 1]
        present = center >= 0
        distance_cm = np.abs(
            decoded.place_bin_centers_cm[center[present]]
            - np.interp(
                decoded.midpoint_s[start : stop + 1][present],
                decoded.time,
                decoded.position,
            )
        )

        for tolerance_cm in (8.0, 15.0, 30.0):
            assert 0.6 < np.mean(distance_cm < tolerance_cm) < 0.85

    def test_local_coding_passes_both_window_criteria(self, decoded):
        """Which is why it is absorbed: it is smooth, narrow and slow."""
        sequence = np.flatnonzero(decoded.candidates.is_replay)[0]
        start, stop = decoded.candidates.runs[sequence]
        replay_start_s = decoded.event_times[CONTINUOUS_OUTBOUND][0]
        before = (
            np.flatnonzero(decoded.midpoint_s[start : stop + 1] < replay_start_s)
            + start
        )

        gates = decoded.candidates.decisions.gates
        assert before.size > 100
        assert gates.spread_below_threshold[before].mean() > 0.8
        assert gates.jump_below_threshold[before].mean() > 0.8

    def test_the_merge_rule_carries_most_of_the_inflation(self, decoded):
        """Absorbing the local stretch takes the merge rule, not the window
        criteria alone: the stretch is broken by windows each criterion
        rejects, and merging bridges them."""
        from dataclasses import replace

        unmerged = detect_candidate_sequences(
            decoded.posterior,
            decoded.midpoint_s,
            decoded.bin_distance_cm,
            decoded.window_speed,
            replace(CONFIG, merge_gap_s=1e-6, merge_gap_cm=1e-6),
        )

        merged_duration_s = decoded.candidates.duration_s[
            np.flatnonzero(decoded.candidates.is_replay)
        ]
        unmerged_duration_s = unmerged.duration_s[np.flatnonzero(unmerged.is_replay)]
        assert unmerged_duration_s.size == merged_duration_s.size
        assert (unmerged_duration_s < 0.5 * merged_duration_s).all()
        assert (
            decoded.candidates.pieces[
                np.flatnonzero(decoded.candidates.is_replay)[0]
            ].size
            > 1
        )


class TestRemoteStationaryIsDeclined:
    """A non-local place held still. Outside what the method targets, by design."""

    def test_it_is_decoded_as_a_remote_place(self, decoded):
        """Not a failure to decode: the posterior is confident and far away."""
        event_start, event_end = decoded.event_times[REMOTE_STATIONARY]
        inside = (decoded.midpoint_s >= event_start) & (decoded.midpoint_s <= event_end)
        center = decoded.candidates.decisions.center_index[inside]
        center_cm = decoded.place_bin_centers_cm[center[center >= 0]]

        assert np.median(center_cm) == pytest.approx(REMOTE_STATIONARY_CM, abs=5.0)
        assert (
            np.nanmedian(decoded.candidates.decisions.spread_cm[inside])
            < CONFIG.spread_threshold_cm
        )

    def test_it_becomes_a_candidate_of_about_the_right_duration(self, decoded):
        touching = [
            i
            for i in range(len(decoded.candidates.runs))
            if REMOTE_STATIONARY in events_overlapping(decoded, i)
        ]

        assert touching
        assert max(decoded.candidates.duration_s[i] for i in touching) > 0.25

    def test_it_is_declined_for_going_nowhere(self, decoded):
        for i in range(len(decoded.candidates.runs)):
            if REMOTE_STATIONARY in events_overlapping(decoded, i):
                assert not decoded.candidates.is_replay[i]
                assert (
                    decoded.candidates.dispersion_cm[i] < CONFIG.dispersion_threshold_cm
                )

    def test_a_stationary_representation_is_absorbed_only_next_to_a_travelling_one(
        self, decoded
    ):
        """The local stretch is absorbed because the replay starts where the
        animal is, so the bridge is short. This one has no such neighbour."""
        absorbed = np.flatnonzero(decoded.candidates.is_replay)
        remote = [
            i
            for i in range(len(decoded.candidates.runs))
            if REMOTE_STATIONARY in events_overlapping(decoded, i)
        ]

        assert set(absorbed).isdisjoint(remote)


class TestSortedSpikesLikelihood:
    def test_windows_sum_to_a_direct_place_field_computation(self, decoded):
        """The sorted-spikes likelihood must be additive over time too.

        Independent of the library's row assignment: count each neuron's spikes
        in the window's own half-open support, weight the log place field by
        that count, and charge one silent-part term per sample.
        """
        place_fields = np.asarray(decoded.encoding_model["place_fields"])
        interior = np.asarray(decoded.encoding_model["is_track_interior"])
        silent_part = np.asarray(
            decoded.encoding_model["no_spike_part_log_likelihood"]
        )[interior]

        windows = aggregate_windows(
            sorted_spikes_epoch_log_likelihood(
                decoded.time,
                decoded.position,
                decoded.spike_times,
                decoded.encoding_model,
                CONFIG,
            ).log_likelihood,
            CONFIG.window_n_samples,
            CONFIG.stride_n_samples,
        )

        # windows that hold spikes: a silent one compares only the exposure
        # term and would not test the spike assignment at all
        rng = np.random.default_rng(0)
        tested = 0
        for window in rng.permutation(len(windows))[:300]:
            start = window * CONFIG.stride_n_samples
            start_s = decoded.time[start]
            end_s = decoded.time[start + CONFIG.window_n_samples]
            counts = np.array(
                [
                    ((neuron >= start_s) & (neuron < end_s)).sum()
                    for neuron in decoded.spike_times
                ]
            )
            if counts.sum() == 0:
                continue
            tested += 1
            direct = (
                counts @ np.log(place_fields[:, interior])
                - CONFIG.window_n_samples * silent_part
            )

            np.testing.assert_allclose(windows[window], direct, rtol=1e-4, atol=1e-3)

        assert tested > 20
