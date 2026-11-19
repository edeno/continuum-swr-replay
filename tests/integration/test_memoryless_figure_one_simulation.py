"""The two replay detectors on the labeled recording of the memoryless comparison.

The fixture is the recording the supplemental figure draws
(``comparison_simulation.simulate_and_decode``): Figure 1's simulated run, whose
four patterns are a sweep that ends at the animal, a fragmented event, a remote
place held still and a silence, plus four replays drawn in: a 30 cm one, an 80 ms
one, one far from the animal and a fast one. Both detectors read one set of
place fields, fitted by the state-space model.

What these tests pin is the sliding-window detector's blind spot rather than its
accuracy. It scores each window on posterior spread and on the jump to the next
window, and it carries no competing hypothesis for the animal's own position.
Local coding is smooth, narrow and slow, so it passes every window criterion and
is assembled into the same sequence as the sweep that ends beside it.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pytest

from continuum_swr_replay.analysis.detection.memoryless_sequence import (
    recording_statistics as rs,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.comparison_simulation import (
    CONFIG,
    FAR_REPLAY_EVENT,
    FAST_REPLAY_EVENT,
    FRAGMENTED_EVENT,
    N_NEURONS,
    REMOTE_STATIONARY_EVENT,
    SHORT_PATH_EVENT,
    SHORT_REPLAY_EVENT,
    SILENCE_EVENT,
    SWEEP_EVENT,
    simulate_and_decode,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.likelihood import (
    likelihood_grid,
    sorted_spikes_epoch_log_likelihood,
    with_reference_rate_floor,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.window_statistics import (
    aggregate_windows,
)
from continuum_swr_replay.analysis.detection.simulated_run import REMOTE_STATIONARY_CM
from continuum_swr_replay.decoding.results import ENCODING_KEY

pytestmark = pytest.mark.integration

STATE_SPACE_BOUND_TOLERANCE_S = 0.025
"""How far a state-space event bound may sit from the labeled one."""


@pytest.fixture(scope="module")
def decoded() -> dict:
    return simulate_and_decode(N_NEURONS, CONFIG)


def state_space_overlapping(decoded: dict, event: int) -> list[tuple[float, float]]:
    """The state-space model's events that overlap a labeled event."""
    event_start, event_end = decoded["event_times"][event]
    return [
        (start, end)
        for start, end in decoded["state_space_intervals"]
        if start < event_end and event_start < end
    ]


def windows_inside(decoded: dict, event: int) -> np.ndarray:
    """Which windows have their midpoint inside a labeled event."""
    event_start, event_end = decoded["event_times"][event]
    return (decoded["midpoint_s"] >= event_start) & (decoded["midpoint_s"] <= event_end)


class TestWhatTheSlidingWindowReports:
    @pytest.mark.parametrize(
        ("event", "expected"),
        [
            (SWEEP_EVENT, "found"),
            (FRAGMENTED_EVENT, "declined"),
            (REMOTE_STATIONARY_EVENT, "declined"),
            (SILENCE_EVENT, "declined"),
            (SHORT_PATH_EVENT, "declined"),
            (SHORT_REPLAY_EVENT, "found"),
            (FAR_REPLAY_EVENT, "found"),
            (FAST_REPLAY_EVENT, "declined"),
        ],
    )
    def test_verdict(self, decoded, event, expected):
        assert rs.verdict(decoded, event) == expected

    def test_it_accepts_one_sequence_over_each_of_three_events_and_no_other(
        self, decoded
    ):
        accepted = np.flatnonzero(decoded["candidates"].is_replay).tolist()

        assert accepted == [
            *rs.accepted_overlapping(decoded, SWEEP_EVENT),
            *rs.accepted_overlapping(decoded, SHORT_REPLAY_EVENT),
            *rs.accepted_overlapping(decoded, FAR_REPLAY_EVENT),
        ]
        assert len(accepted) == 3

    def test_the_two_detectors_agree_on_the_replay_far_from_the_animal(self, decoded):
        # the sliding window's report is nearer the replay's own duration than
        # its reports of the sweep that ends at the animal and the short replay
        far = rs.duration_ratio(decoded, FAR_REPLAY_EVENT)
        assert 1.0 < far < rs.duration_ratio(decoded, SHORT_REPLAY_EVENT)
        assert far < rs.duration_ratio(decoded, SWEEP_EVENT)
        assert rs.state_space_duration_ratio(
            decoded, FAR_REPLAY_EVENT
        ) == pytest.approx(1.0, abs=0.15)

    def test_only_accepted_candidates_are_tested(self, decoded):
        assert set(decoded["shuffles"]) == set(
            np.flatnonzero(decoded["candidates"].is_replay)
        )

    @pytest.mark.parametrize("event", [SWEEP_EVENT, SHORT_REPLAY_EVENT])
    def test_a_found_event_passes_both_halves_of_the_shuffle(self, decoded, event):
        result = rs.shuffle_of(decoded, event)

        assert result.testable
        assert result.p_spread <= 0.05
        assert result.p_jump <= 0.05


class TestLocalCodingIsAbsorbed:
    """The detector has no hypothesis that competes with a replay for a window:
    the sweep ends at the animal, and the local coding after it joins it."""

    @pytest.fixture
    def sequence(self, decoded):
        (sequence,) = rs.accepted_overlapping(decoded, SWEEP_EVENT)
        return sequence

    def test_the_sweep_is_reported_more_than_twice_too_long(self, decoded):
        assert 2.0 < rs.duration_ratio(decoded, SWEEP_EVENT) < 3.0

    def test_most_of_the_reported_event_lies_outside_the_sweep(self, decoded, sequence):
        """The measure that needs no "same place" judgment."""
        start, stop = decoded["candidates"].runs[sequence]
        inside = windows_inside(decoded, SWEEP_EVENT)[start : stop + 1]

        assert 1.0 - inside.mean() > 0.5

    def test_most_of_it_decodes_the_animal_s_own_position(self, decoded, sequence):
        """Whatever the tolerance: the decoded-to-animal distance is bimodal."""
        start, stop = decoded["candidates"].runs[sequence]
        distance_cm = np.abs(
            rs.decoded_center_cm(decoded)[start : stop + 1]
            - np.interp(
                decoded["midpoint_s"][start : stop + 1],
                decoded["time"],
                decoded["position"],
            )
        )
        distance_cm = distance_cm[np.isfinite(distance_cm)]

        assert rs.local_fraction(decoded, sequence) > 0.5
        for tolerance_cm in (8.0, 15.0, 30.0):
            assert 0.5 < np.mean(distance_cm < tolerance_cm) < 0.65

    def test_local_coding_passes_both_window_criteria(self, decoded, sequence):
        """Which is why it is absorbed: it is smooth, narrow and slow."""
        start, stop = decoded["candidates"].runs[sequence]
        sweep_end_s = decoded["event_times"][SWEEP_EVENT][1]
        after = start + np.flatnonzero(
            decoded["midpoint_s"][start : stop + 1] > sweep_end_s
        )

        gates = decoded["candidates"].decisions.gates
        assert after.size > 50
        assert gates.spread_below_threshold[after].mean() > 0.8
        assert gates.jump_below_threshold[after].mean() > 0.8


class TestTheWindowPutsAFloorUnderDuration:
    """An 80 ms replay far from the animal: found, and reported too long with
    no local coding in it. This is the other inflation mechanism, from the
    window alone rather than from absorption."""

    def test_it_is_reported_longer_than_it_is_but_not_absorbed(self, decoded):
        (sequence,) = rs.accepted_overlapping(decoded, SHORT_REPLAY_EVENT)

        assert 1.0 < rs.duration_ratio(decoded, SHORT_REPLAY_EVENT) < 2.0
        assert decoded["candidates"].pieces[sequence].size == 1
        assert rs.local_fraction(decoded, sequence) == 0.0


class TestAShortPathIsDeclined:
    """A 30 cm replay, decoded well, declined by the dispersion criterion: a
    uniform traverse of length L has dispersion L / sqrt(12), so 12 cm needs
    about 42 cm of path however well the event is decoded."""

    def test_it_falls_short_on_dispersion_not_duration(self, decoded):
        candidates = decoded["candidates"]
        overlapping = rs.candidates_overlapping(decoded, SHORT_PATH_EVENT)

        assert (
            rs.best_dispersion_cm(decoded, SHORT_PATH_EVENT)
            < CONFIG.dispersion_threshold_cm
        )
        assert (
            max(candidates.duration_s[i] for i in overlapping)
            > CONFIG.minimum_duration_s
        )

    def test_its_windows_pass_the_spread_gate(self, decoded):
        """Not a failure to decode."""
        inside = windows_inside(decoded, SHORT_PATH_EVENT)
        gates = decoded["candidates"].decisions.gates

        assert gates.spread_below_threshold[inside].mean() > 0.8


class TestRemoteStationaryIsDeclined:
    """A non-local place held still. Outside what the method targets, by design."""

    def test_it_is_decoded_as_a_remote_place(self, decoded):
        """Not a failure to decode: the posterior is confident and far away."""
        inside = windows_inside(decoded, REMOTE_STATIONARY_EVENT)
        spread_cm = decoded["candidates"].decisions.spread_cm[inside]

        assert np.nanmedian(rs.decoded_center_cm(decoded)[inside]) == pytest.approx(
            REMOTE_STATIONARY_CM, abs=5.0
        )
        assert np.nanmedian(spread_cm) < CONFIG.spread_threshold_cm

    def test_it_is_a_candidate_of_about_the_right_duration_that_goes_nowhere(
        self, decoded
    ):
        candidates = decoded["candidates"]
        overlapping = rs.candidates_overlapping(decoded, REMOTE_STATIONARY_EVENT)

        assert max(candidates.duration_s[i] for i in overlapping) > 0.25
        assert (
            rs.best_dispersion_cm(decoded, REMOTE_STATIONARY_EVENT)
            < CONFIG.dispersion_threshold_cm
        )


class TestTheStateSpaceModel:
    """The comparison needs every event the sliding window mis-reports to be an
    event some detector finds, or the figure shows the limits of the data."""

    @pytest.mark.parametrize(
        "event",
        [
            SWEEP_EVENT,
            FRAGMENTED_EVENT,
            REMOTE_STATIONARY_EVENT,
            SHORT_PATH_EVENT,
            SHORT_REPLAY_EVENT,
        ],
    )
    def test_it_reports_the_event_with_both_bounds_close(self, decoded, event):
        event_start, event_end = decoded["event_times"][event]
        overlapping = state_space_overlapping(decoded, event)

        assert len(overlapping) == 1
        ((start, end),) = overlapping
        assert abs(start - event_start) < STATE_SPACE_BOUND_TOLERANCE_S
        assert abs(end - event_end) < STATE_SPACE_BOUND_TOLERANCE_S

    def test_neither_detector_reports_the_silence(self, decoded):
        assert not state_space_overlapping(decoded, SILENCE_EVENT)
        assert not rs.accepted_overlapping(decoded, SILENCE_EVENT)

    def test_it_reports_nothing_outside_the_labeled_events(self, decoded):
        overlapping = [
            interval
            for event in range(len(decoded["event_times"]))
            for interval in state_space_overlapping(decoded, event)
        ]

        assert sorted(overlapping) == sorted(decoded["state_space_intervals"])


class TestSortedSpikesLikelihood:
    def test_windows_sum_to_a_direct_place_field_computation(self, decoded):
        """The sorted-spikes likelihood must be additive over time too.

        Independent of the library's row assignment: count each neuron's spikes
        in the window's own half-open support, weight the log place field by
        that count, and charge one silent-part term per decode sample the
        window spans: 80 ms of 2 ms samples, 40, however many rows hold them.
        """
        encoding_model = with_reference_rate_floor(
            decoded["detector"].encoding_model_[ENCODING_KEY], CONFIG
        )
        place_fields = np.asarray(encoding_model["place_fields"])
        interior = np.asarray(encoding_model["is_track_interior"])
        silent_part = np.asarray(encoding_model["no_spike_part_log_likelihood"])[
            interior
        ]

        windows = aggregate_windows(
            sorted_spikes_epoch_log_likelihood(
                decoded["time"],
                decoded["position"],
                decoded["spike_times"],
                encoding_model,
                CONFIG,
            ).log_likelihood,
            CONFIG.window_n_samples,
            CONFIG.stride_n_samples,
        )

        # the likelihood's own grid, derived from the decode rows it was handed
        grid = likelihood_grid(decoded["time"], CONFIG)
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
                    for neuron in decoded["spike_times"]
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
