"""The simulated run of Figure 1: what the simulation holds and what the decode finds."""

from __future__ import annotations

import numpy as np
import pytest

from continuum_swr_replay.analysis.detection.simulated_run import (
    PATTERNS,
    REMOTE_SPIKE_INTERVAL_S,
    REMOTE_STATIONARY_CM,
    REMOTE_STATIONARY_S,
    SILENCE_DURATION_S,
    DecodedSimulation,
    simulate_and_decode,
    splice_remote_stationary,
)
from continuum_swr_replay.analysis.event_definition.nonlocal_event_rule import (
    NON_LOCAL_STATES,
)
from continuum_swr_replay.parameters import DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC

# What each pattern should be decoded as.
EXPECTED_STATE = {
    "stationary": "Non-Local Continuous",
    "fragmented": "Non-Local Fragmented",
    "sweep": "Non-Local Continuous",
    "silence": "No-Spike",
}
NON_LOCAL_PATTERNS = ("sweep", "fragmented", "stationary")


def test_splice_remote_stationary_leaves_one_cell_firing_in_the_event():
    place_field_means = np.array([0.0, 40.0, 90.0, 170.0])
    spike_times = [np.array([0.9, 1.05, 1.2, 1.5]) for _ in place_field_means]

    spliced, (start, end) = splice_remote_stationary(
        spike_times, place_field_means, 1.0
    )

    assert (start, end) == (1.0, 1.0 + REMOTE_STATIONARY_S)
    for cell, times in enumerate(spliced):
        inside = times[(times >= start - 1e-9) & (times <= end + 1e-9)]
        if cell == 1:  # 40 cm is the field nearest REMOTE_STATIONARY_CM
            np.testing.assert_allclose(
                inside, np.arange(start, end, REMOTE_SPIKE_INTERVAL_S)
            )
        else:
            assert inside.size == 0
        # spikes outside the event are untouched
        np.testing.assert_array_equal(
            times[(times < start - 1e-9) | (times > end + 1e-9)], [0.9, 1.5]
        )
    assert abs(place_field_means[1] - REMOTE_STATIONARY_CM) < 5


@pytest.fixture(scope="module")
def sim() -> DecodedSimulation:
    return simulate_and_decode()


def _within(sim: DecodedSimulation, span: tuple[float, float]) -> np.ndarray:
    return (sim.time >= span[0]) & (sim.time <= span[1])


@pytest.mark.slow
class TestSimulatedRun:
    def test_arrays_share_the_decode_grid(self, sim):
        n_time = sim.time.size
        assert sim.position.shape == sim.speed.shape == (n_time,)
        assert sim.non_local_probability.shape == (n_time,)
        assert sim.state_probabilities.shape == (n_time, len(sim.state_names))
        assert sim.posterior.shape == (n_time, sim.position_bins.size)
        assert sim.non_local_posterior.shape == sim.posterior.shape
        # 1500 Hz strided to the 500 Hz decode grid
        np.testing.assert_allclose(np.diff(sim.time), 0.002, atol=1e-9)
        np.testing.assert_allclose(sim.state_probabilities.sum(axis=1), 1, atol=1e-4)

    def test_patterns_are_in_time_order_and_each_within_a_pause(self, sim):
        assert tuple(sim.patterns) == PATTERNS
        starts = [sim.patterns[name][0] for name in PATTERNS]
        assert starts == sorted(starts)
        for name in PATTERNS:
            still = sim.speed[_within(sim, sim.patterns[name])]
            assert (still < DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC).all(), name

    def test_stationary_pattern_is_one_cell_far_from_the_animal(self, sim):
        start, end = sim.patterns["stationary"]
        assert end - start == pytest.approx(REMOTE_STATIONARY_S)
        firing = [
            cell
            for cell, times in enumerate(sim.sorted_spikes)
            if ((times >= start) & (times <= end)).any()
        ]
        assert len(firing) == 1
        assert abs(sim.place_field_means[firing[0]] - REMOTE_STATIONARY_CM) < 5
        animal = sim.position[_within(sim, (start, end))]
        assert (np.abs(animal - REMOTE_STATIONARY_CM) > 100).all()

    def test_silence_has_no_spikes(self, sim):
        start, end = sim.patterns["silence"]
        assert end - start == pytest.approx(SILENCE_DURATION_S)
        for times in sim.sorted_spikes:
            assert not ((times >= start) & (times <= end)).any()

    def test_each_pattern_is_decoded_as_its_state(self, sim):
        for name, state in EXPECTED_STATE.items():
            mean = sim.state_probabilities[_within(sim, sim.patterns[name])].mean(
                axis=0
            )
            assert sim.state_names[int(np.argmax(mean))] == state, name
            assert mean.max() > 0.8, name

    def test_stationary_pattern_is_decoded_at_the_remote_place(self, sim):
        rows = _within(sim, sim.patterns["stationary"])
        decoded = sim.position_bins[np.argmax(sim.non_local_posterior[rows], axis=1)]
        assert np.abs(np.median(decoded) - REMOTE_STATIONARY_CM) < 10

    def test_one_event_per_non_local_pattern_and_no_other(self, sim):
        assert len(sim.non_local_events) == len(NON_LOCAL_PATTERNS)
        for name, (start, end) in zip(
            NON_LOCAL_PATTERNS, sim.non_local_events, strict=True
        ):
            pattern = sim.patterns[name]
            # the detected edges lie within 25 ms of the simulated ones
            assert abs(start - pattern[0]) < 0.025, name
            assert abs(end - pattern[1]) < 0.025, name
        silence = sim.patterns["silence"]
        for start, end in sim.non_local_events:
            assert end < silence[0] or start > silence[1]

    def test_non_local_probability_is_the_two_non_local_states(self, sim):
        columns = [sim.state_names.index(name) for name in NON_LOCAL_STATES]
        np.testing.assert_allclose(
            sim.non_local_probability,
            sim.state_probabilities[:, columns].sum(axis=1),
            atol=1e-6,
        )
