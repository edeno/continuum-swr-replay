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

The fixture is the recording the supplemental figure draws
(``scripts/figures/generate_memoryless_comparison.py``): the package's five
events plus a remote stationary event, a 30 cm replay, an 80 ms replay and a
hand-placed still event with one window decoded far away, in that order. The
last is written into the decoded spikes only; the encoding model is fitted on
the recording without it, as the figure fits it. Both detectors read one encoding model, fitted by the
state-space detector, as the figure and the sweep do.
"""

import os
from dataclasses import dataclass

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest
from non_local_detector import NonLocalSortedSpikesDetector
from non_local_detector.environment import Environment
from non_local_detector.simulate.sorted_spikes_simulation import (
    SAMPLING_FREQUENCY,
    TRACK_HEIGHT,
)

from continuum_swr_replay.analysis.detection.memoryless_sequence.comparison_simulation import (
    CONFIG,
    N_NEURONS,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.detect import (
    CandidateSequences,
    detect_candidate_sequences,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.likelihood import (
    ENCODING_KEY,
    interior_bin_distances_cm,
    likelihood_grid,
    sorted_spikes_epoch_log_likelihood,
    window_midpoints_s,
    with_reference_rate_floor,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.shuffle import (
    DEFAULT_N_SHUFFLES,
    ShuffleResult,
    ShuffleSeed,
    shuffle_candidate_sorted,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.simulation import (
    InjectedReplay,
    inject_replay,
    stray_window_spikes,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.window_statistics import (
    aggregate_windows,
    normalize_windows,
)
from continuum_swr_replay.parameters import LOCAL_POSITION_STD

pytestmark = pytest.mark.integration

DECODE_BIN_STRIDE = 3
"""Figure 1 strides the 1500 Hz simulation to the project's 500 Hz decode grid."""

CONTINUOUS_OUTBOUND, FRAGMENTED, CONTINUOUS_INBOUND, NO_SPIKE, LOCAL = range(5)
REMOTE_STATIONARY, SHORT_PATH, SHORT_REPLAY, STRAY_WINDOW = 5, 6, 7, 8
"""Events added here, in the order the figure adds them: a place held still
away from the animal, a 30 cm replay, an 80 ms replay, and a still event with
one window decoded far away."""

STRAY_WINDOW_AT_S, STRAY_WINDOW_STILL_S, STRAY_WINDOW_N_LOCAL = 31.40, 0.120, 2
"""Where the figure puts the hand-placed stray-window event, how long its still
parts last and how many cells carry them."""

TRACK_HEIGHT_CM = float(TRACK_HEIGHT)
REMOTE_STATIONARY_CM = 42.5
REMOTE_STATIONARY_S = 0.300
REMOTE_SPIKE_INTERVAL_S = 0.012
SHORT_PATH_REPLAY = InjectedReplay(
    start_cm=40.0, path_cm=30.0, duration_s=0.300, peak_rate_hz=40.0, at_s=20.50, seed=0
)
SHORT_REPLAY_REPLAY = InjectedReplay(
    start_cm=35.0,
    path_cm=100.0,
    duration_s=0.080,
    peak_rate_hz=120.0,
    at_s=44.40,
    seed=1,
)
SAME_PLACE_CM = 15.0
"""How close a decoded center must be to count as the animal's own position."""

NON_LOCAL_PROB_THRESHOLD = 0.8
NON_LOCAL_MINIMUM_DURATION_S = 0.015
"""The project's non-local event definition, as the figure applies it."""

STATE_SPACE_BOUND_TOLERANCE_S = 0.050
"""How far a state-space event bound may sit from the labeled one."""

SHUFFLE_SEED = ShuffleSeed(run=20260922, session="figure-one-simulation")
"""The figure's seed, so this fixture's verdicts are the figure's."""


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
    state_space_intervals: list[tuple[float, float]]
    shuffles: dict[int, ShuffleResult]


def state_space_intervals(
    detector, time, position, spike_times
) -> list[tuple[float, float]]:
    """When the state-space detector calls the representation non-local."""
    results = detector.predict(
        position_time=time, position=position, spike_times=spike_times, time=time
    )
    probability = np.asarray(
        results.acausal_state_probabilities.sel(
            states=["Non-Local Continuous", "Non-Local Fragmented"]
        )
        .sum("states")
        .values
    ).squeeze()
    assert np.isfinite(probability).all() and probability.max() > 0.0
    above = probability > NON_LOCAL_PROB_THRESHOLD
    edges = np.diff(above.astype(np.int8), prepend=0, append=0)
    return [
        (float(time[start]), float(time[stop]))
        for start, stop in zip(
            np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1, strict=True
        )
        if time[stop] - time[start] >= NON_LOCAL_MINIMUM_DURATION_S
    ]


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
    ) = make_simulated_data(n_neurons=N_NEURONS)
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

    # the two replays the figure injects, in its order
    place_field_means = np.linspace(0, TRACK_HEIGHT_CM, len(spike_times))
    for replay in (SHORT_PATH_REPLAY, SHORT_REPLAY_REPLAY):
        spike_times, bounds = inject_replay(
            spike_times,
            place_field_means,
            replay,
            sampling_frequency=SAMPLING_FREQUENCY,
        )
        event_times = np.vstack([event_times, bounds])
        is_event |= (time >= bounds[0]) & (time <= bounds[1])

    # the hand-placed stray-window event, in the decoded spikes only
    training_spike_times = spike_times
    spike_times, stray = stray_window_spikes(
        spike_times,
        place_field_means,
        STRAY_WINDOW_AT_S,
        local_cm=float(np.interp(STRAY_WINDOW_AT_S, time, position)),
        still_s=STRAY_WINDOW_STILL_S,
        n_local=STRAY_WINDOW_N_LOCAL,
        window_s=CONFIG.window_duration_s,
        stride_s=CONFIG.stride_s,
    )
    event_times = np.vstack([event_times, stray.bounds])

    time, position, speed, is_event = (
        array[::DECODE_BIN_STRIDE] for array in (time, position, speed, is_event)
    )

    # encode on non-event data only: spikes inside an event must not shape the
    # place fields they are later decoded against. The state-space detector
    # fits the model and the sliding window is handed the same object.
    is_training = ~is_event
    detector = NonLocalSortedSpikesDetector(
        environments=[Environment(place_bin_size=CONFIG.place_bin_size_cm)],
        sorted_spikes_algorithm="sorted_spikes_kde",
        sorted_spikes_algorithm_params={"position_std": 6.0, "block_size": int(2**8)},
        local_position_std=LOCAL_POSITION_STD,
    )
    detector.fit(
        position_time=time,
        position=position,
        spike_times=training_spike_times,
        is_training=is_training,
    )
    # decoded as the figure decodes it, with the reference's floor
    encoding_model = with_reference_rate_floor(
        detector.encoding_model_[ENCODING_KEY], CONFIG
    )
    environment = encoding_model["environment"]

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
    # the decode grid's rows, its final sample dropped as the likelihood drops
    # its own; the state-space detector predicts on these
    decode_rows = time[:-1]
    window_speed = np.interp(midpoint_s, decode_rows, speed[:-1])
    candidates = detect_candidate_sequences(
        posterior, midpoint_s, bin_distance_cm, window_speed, CONFIG
    )
    # the reference's own significance test on every accepted candidate, on
    # the full decode grid, as the figure runs it
    shuffles = {
        int(i): shuffle_candidate_sorted(
            int(candidates.runs[i, 0]),
            int(candidates.runs[i, 1]),
            time,
            spike_times,
            encoding_model,
            bin_distance_cm,
            CONFIG,
            seed=SHUFFLE_SEED,
            n_shuffles=DEFAULT_N_SHUFFLES,
        )
        for i in np.flatnonzero(candidates.is_replay)
    }

    return Decoded(
        time=decode_rows,
        position=position[:-1],
        event_times=event_times,
        place_bin_centers_cm=result.place_bin_centers_cm,
        midpoint_s=midpoint_s,
        candidates=candidates,
        shuffles=shuffles,
        encoding_model=encoding_model,
        spike_times=spike_times,
        posterior=posterior,
        bin_distance_cm=bin_distance_cm,
        window_speed=window_speed,
        state_space_intervals=state_space_intervals(
            detector, decode_rows, position[:-1], spike_times
        ),
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
    """Share of a sequence's kept windows whose center is the animal's own place."""
    start, stop = decoded.candidates.runs[sequence]
    center = decoded.candidates.decisions.center_index[start : stop + 1]
    kept = decoded.candidates.assembly.kept[start : stop + 1]
    center_cm = decoded.place_bin_centers_cm[center[kept]]
    animal_cm = np.interp(
        decoded.midpoint_s[start : stop + 1][kept], decoded.time, decoded.position
    )
    return float(np.mean(np.abs(center_cm - animal_cm) < SAME_PLACE_CM))


def candidates_overlapping(decoded: Decoded, event: int) -> list[int]:
    return [
        i
        for i in range(len(decoded.candidates.runs))
        if event in events_overlapping(decoded, i)
    ]


def accepted_overlapping(decoded: Decoded, event: int) -> list[int]:
    return [
        i
        for i in candidates_overlapping(decoded, event)
        if decoded.candidates.is_replay[i]
    ]


def true_duration_s(decoded: Decoded, event: int) -> float:
    return float(decoded.event_times[event, 1] - decoded.event_times[event, 0])


class TestWhatItAccepts:
    def test_the_continuous_replays_and_the_short_replay_are_found_and_nothing_else_is(
        self, decoded
    ):
        accepted = np.flatnonzero(decoded.candidates.is_replay)

        assert [events_overlapping(decoded, s) for s in accepted] == [
            [CONTINUOUS_OUTBOUND],
            [STRAY_WINDOW],
            [CONTINUOUS_INBOUND],
            [SHORT_REPLAY],
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


def absorbed_sequences(decoded: Decoded) -> list[int]:
    """The accepted sequences that begin at the animal's own place: the two
    continuous replays. The short replay is far from the animal and is
    inflated by the window alone, which the next class covers."""
    return [
        i
        for i in np.flatnonzero(decoded.candidates.is_replay)
        if events_overlapping(decoded, i)[0]
        in (CONTINUOUS_OUTBOUND, CONTINUOUS_INBOUND)
    ]


class TestLocalCodingIsAbsorbed:
    """The detector has no hypothesis that competes with a replay for a window."""

    def test_most_of_a_detected_event_lies_outside_the_event_it_found(self, decoded):
        """The measure that needs no "same place" judgement."""
        for sequence in absorbed_sequences(decoded):
            assert fraction_outside_its_event(decoded, sequence) > 0.7

    def test_an_accepted_event_runs_far_longer_than_the_replay_in_it(self, decoded):
        for sequence in absorbed_sequences(decoded):
            event = events_overlapping(decoded, sequence)[0]

            assert decoded.candidates.duration_s[sequence] > 3 * true_duration_s(
                decoded, event
            )

    def test_most_of_an_accepted_event_decodes_the_animal_s_own_position(self, decoded):
        for sequence in absorbed_sequences(decoded):
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

        absorbed = absorbed_sequences(decoded)
        merged_duration_s = decoded.candidates.duration_s[absorbed]
        unmerged_duration_s = np.array(
            [
                max(
                    unmerged.duration_s[i]
                    for i in np.flatnonzero(unmerged.is_replay)
                    if unmerged.midpoint_s[unmerged.runs[i][0]]
                    < decoded.event_times[events_overlapping(decoded, sequence)[0], 1]
                    and decoded.event_times[events_overlapping(decoded, sequence)[0], 0]
                    < unmerged.midpoint_s[unmerged.runs[i][1]]
                )
                for sequence in absorbed
            ]
        )
        assert (unmerged_duration_s < 0.5 * merged_duration_s).all()
        assert decoded.candidates.pieces[absorbed[0]].size > 1


class TestTheWindowPutsAFloorUnderDuration:
    """An 80 ms replay far from the animal: found, and reported too long with
    no local coding in it. This is the other inflation mechanism, from the
    window alone rather than from absorption."""

    def test_it_is_found(self, decoded):
        assert accepted_overlapping(decoded, SHORT_REPLAY)

    def test_it_is_reported_longer_than_it_is_but_not_absorbed(self, decoded):
        sequence = accepted_overlapping(decoded, SHORT_REPLAY)[0]
        ratio = decoded.candidates.duration_s[sequence] / true_duration_s(
            decoded, SHORT_REPLAY
        )

        assert 1.0 < ratio < 2.0
        assert decoded.candidates.pieces[sequence].size == 1

    def test_none_of_it_is_the_animal_s_own_place(self, decoded):
        sequence = accepted_overlapping(decoded, SHORT_REPLAY)[0]

        assert local_fraction(decoded, sequence) == 0.0


class TestAShortPathIsDeclined:
    """A 30 cm replay, decoded well, declined by the dispersion criterion: a
    uniform traverse of length L has dispersion L / sqrt(12), so 12 cm needs
    about 42 cm of path however well the event is decoded."""

    def test_it_becomes_a_candidate_that_is_not_accepted(self, decoded):
        assert candidates_overlapping(decoded, SHORT_PATH)
        assert not accepted_overlapping(decoded, SHORT_PATH)

    def test_every_candidate_falls_short_on_dispersion_not_duration(self, decoded):
        candidates = decoded.candidates
        for i in candidates_overlapping(decoded, SHORT_PATH):
            assert candidates.dispersion_cm[i] < CONFIG.dispersion_threshold_cm
        assert (
            max(
                candidates.duration_s[i]
                for i in candidates_overlapping(decoded, SHORT_PATH)
            )
            > CONFIG.minimum_duration_s
        )

    def test_its_windows_pass_the_spread_gate(self, decoded):
        """Not a failure to decode."""
        event_start, event_end = decoded.event_times[SHORT_PATH]
        inside = (decoded.midpoint_s >= event_start) & (decoded.midpoint_s <= event_end)

        assert (
            decoded.candidates.decisions.gates.spread_below_threshold[inside].mean()
            > 0.8
        )


class TestThePublishedSignificanceTest:
    """The reference's place-cell identity shuffle, on the figure's recording.

    The inbound continuous replay and the 80 ms replay pass both halves: the
    positive controls for the whole published definition. The outbound
    continuous replay is mostly absorbed local coding and is reported far too
    long; the shuffle is run on it, but whether it passes sits within a few
    draws of the threshold and moves with the step and the seed, so nothing
    here pins its verdict. The caption reads these from the data; these pin
    them so a change is noticed.
    """

    @pytest.mark.parametrize("event", [CONTINUOUS_INBOUND, SHORT_REPLAY])
    def test_a_coherent_replay_passes_both_halves(self, decoded, event):
        sequence = accepted_overlapping(decoded, event)[0]
        result = decoded.shuffles[sequence]

        assert result.testable
        assert result.p_spread <= 0.05
        assert result.p_jump <= 0.05
        assert result.passes

    def test_the_absorbed_outbound_replay_is_tested_as_mostly_local_and_too_long(
        self, decoded
    ):
        """What is robust about it: the event the shuffle is asked about is
        mostly the animal's own place, and far longer than the replay in it.
        Its pass or fail is not claimed."""
        sequence = accepted_overlapping(decoded, CONTINUOUS_OUTBOUND)[0]

        assert decoded.shuffles[sequence].testable
        assert local_fraction(decoded, sequence) > 0.5
        assert decoded.candidates.duration_s[sequence] > 3 * true_duration_s(
            decoded, CONTINUOUS_OUTBOUND
        )

    def test_only_accepted_candidates_are_tested(self, decoded):
        assert set(decoded.shuffles) == set(
            np.flatnonzero(decoded.candidates.is_replay)
        )


class TestTheStateSpaceDetectorFindsAllFour:
    """The comparison needs every event the sliding window mis-reports to be an
    event some detector finds, or the figure shows the limits of the data."""

    @pytest.mark.parametrize(
        "event", [CONTINUOUS_OUTBOUND, SHORT_REPLAY, SHORT_PATH, REMOTE_STATIONARY]
    )
    def test_it_reports_the_event_with_both_bounds_close(self, decoded, event):
        event_start, event_end = decoded.event_times[event]
        overlapping = [
            (a, b)
            for a, b in decoded.state_space_intervals
            if a < event_end and event_start < b
        ]

        assert overlapping
        assert abs(overlapping[0][0] - event_start) < STATE_SPACE_BOUND_TOLERANCE_S
        assert abs(overlapping[-1][1] - event_end) < STATE_SPACE_BOUND_TOLERANCE_S

    def test_it_does_not_report_the_silent_stretch(self, decoded):
        """The package's "local" entry is a zero-width marker at the pause the
        remote stationary event was spliced into, so it is not a negative
        control here; the silent stretch is."""
        event_start, event_end = decoded.event_times[NO_SPIKE]

        assert not any(
            a < event_end + 0.02 and event_start - 0.02 < b
            for a, b in decoded.state_space_intervals
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


class TestOneWindowFarAway:
    """A still event called replay on the strength of one far window.

    The window is kept, dropped before the merge as a run of one, and still
    counted in the merged event's dispersion, as the reference counts it.
    """

    @pytest.fixture
    def sequence(self, decoded):
        accepted = [
            i
            for i in np.flatnonzero(decoded.candidates.is_replay)
            if STRAY_WINDOW in events_overlapping(decoded, i)
        ]
        assert len(accepted) == 1
        return accepted[0]

    def outside(self, decoded, sequence):
        start, stop = decoded.candidates.runs[sequence]
        in_piece = np.zeros(stop - start + 1, dtype=bool)
        for piece in decoded.candidates.pieces[sequence]:
            first, last = decoded.candidates.pre_merge_runs[piece]
            in_piece[first - start : last - start + 1] = True
        kept = decoded.candidates.decisions.gates.kept[start : stop + 1]
        return start + np.flatnonzero(kept & ~in_piece), in_piece

    def test_exactly_one_kept_window_lies_outside_the_pieces_and_it_is_far(
        self, decoded, sequence
    ):
        outside, _ = self.outside(decoded, sequence)
        center = decoded.candidates.decisions.center_index[outside]

        assert outside.size == 1
        assert decoded.place_bin_centers_cm[center[0]] < 40.0

    def test_the_call_rests_on_that_window(self, decoded, sequence):
        from continuum_swr_replay.analysis.detection.memoryless_sequence.window_statistics import (
            NO_CENTER,
            dispersion_cm,
        )

        _, in_piece = self.outside(decoded, sequence)
        start, stop = decoded.candidates.runs[sequence]
        center = decoded.candidates.decisions.center_index[start : stop + 1]
        pieces_only = dispersion_cm(
            np.where(in_piece, center, NO_CENTER), decoded.bin_distance_cm
        )

        assert (
            decoded.candidates.dispersion_cm[sequence] > CONFIG.dispersion_threshold_cm
        )
        assert pieces_only < CONFIG.dispersion_threshold_cm

    def test_every_window_in_its_pieces_is_the_animals_own_place(
        self, decoded, sequence
    ):
        _, in_piece = self.outside(decoded, sequence)
        start, stop = decoded.candidates.runs[sequence]
        windows = np.arange(start, stop + 1)[in_piece]
        center_cm = decoded.place_bin_centers_cm[
            decoded.candidates.decisions.center_index[windows]
        ]
        animal_cm = np.interp(
            decoded.midpoint_s[windows], decoded.time, decoded.position
        )

        assert np.abs(center_cm - animal_cm).max() < SAME_PLACE_CM


class TestSortedSpikesLikelihood:
    def test_windows_sum_to_a_direct_place_field_computation(self, decoded):
        """The sorted-spikes likelihood must be additive over time too.

        Independent of the library's row assignment: count each neuron's spikes
        in the window's own half-open support, weight the log place field by
        that count, and charge one silent-part term per decode sample the
        window spans: 80 ms of 2 ms samples, 40, however many rows hold them.
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

        # the likelihood's own grid, derived from the decode rows it was handed
        grid = likelihood_grid(decoded.time, CONFIG)
        decode_samples = round(
            CONFIG.window_duration_s * CONFIG.decode_sampling_frequency
        )

        # windows that hold spikes: a silent one compares only the exposure
        # term and would not test the spike assignment at all
        rng = np.random.default_rng(0)
        tested = 0
        for window in rng.permutation(len(windows))[:300]:
            start = window * CONFIG.stride_n_samples
            start_s = grid[start]
            end_s = grid[start + CONFIG.window_n_samples]
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
                - decode_samples * silent_part
            )

            np.testing.assert_allclose(windows[window], direct, rtol=1e-4, atol=1e-3)

        assert tested > 20
