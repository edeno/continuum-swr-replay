"""Injecting replays into a simulated recording, and placing them safely."""

import numpy as np
import pytest

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.simulation import (
    immobile_intervals,
    inject_replay,
    place_field_means,
    placements,
    without,
)

SAMPLING_FREQUENCY = 500.0


class TestPlaceFieldMeans:
    def test_fields_span_the_track(self):
        means = place_field_means(25, 170.0)

        assert means.size == 25
        assert means[0] == 0.0
        assert means[-1] == pytest.approx(170.0)

    def test_fields_are_evenly_spaced(self):
        spacing = np.diff(place_field_means(25, 170.0))

        np.testing.assert_allclose(spacing, spacing[0])


class TestInjectReplay:
    def setup_method(self):
        self.means = place_field_means(25, 170.0)
        self.spike_times = [np.arange(0.0, 10.0, 0.05) for _ in range(25)]

    def inject(self, **kwargs):
        defaults = {
            "spike_times": self.spike_times,
            "place_field_means": self.means,
            "start_cm": 20.0,
            "path_cm": 100.0,
            "duration_s": 0.200,
            "peak_rate_hz": 80.0,
            "at_s": 4.0,
            "sampling_frequency": SAMPLING_FREQUENCY,
            "seed": 0,
        }
        return inject_replay(**(defaults | kwargs))

    def test_the_event_spans_the_requested_time(self):
        _, (start_s, end_s) = self.inject()

        assert start_s == pytest.approx(4.0)
        assert end_s == pytest.approx(4.2, abs=0.005)

    def test_spikes_already_there_are_cleared_from_the_event(self):
        """Comparing against the original spikes, not against the output: the
        output trivially contains itself."""
        injected, (start_s, end_s) = self.inject()

        original = np.concatenate(self.spike_times)
        inside = original[(original > start_s) & (original < end_s)]
        assert inside.size > 0

        survivors = sum(int(np.isin(inside, neuron).sum()) for neuron in injected)
        assert survivors == 0

    def test_spikes_outside_the_event_are_untouched(self):
        injected, (start_s, end_s) = self.inject()

        for before, after in zip(self.spike_times, injected, strict=True):
            outside = before[(before < start_s) | (before > end_s)]
            assert np.isin(outside, after).all()

    def test_the_cells_that_fire_are_the_ones_the_path_crosses(self):
        injected, _ = self.inject(start_cm=20.0, path_cm=40.0)

        fired = [
            i
            for i, neuron in enumerate(injected)
            if ((neuron >= 4.0) & (neuron <= 4.2)).any()
        ]
        # fields at 20-60 cm, plus the kernel's reach either side
        assert all(0.0 <= self.means[i] <= 90.0 for i in fired)

    def test_a_higher_rate_gives_more_spikes(self):
        sparse, _ = self.inject(peak_rate_hz=10.0)
        dense, _ = self.inject(peak_rate_hz=200.0)

        def count(injected):
            return sum(((n >= 4.0) & (n <= 4.2)).sum() for n in injected)

        assert count(dense) > 3 * count(sparse)

    def test_the_same_seed_gives_the_same_spikes(self):
        first, _ = self.inject(seed=7)
        second, _ = self.inject(seed=7)

        for a, b in zip(first, second, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_each_neuron_stays_sorted(self):
        injected, _ = self.inject()

        for neuron in injected:
            assert (np.diff(neuron) >= 0).all()

    def test_the_input_is_not_changed(self):
        before = [neuron.copy() for neuron in self.spike_times]

        self.inject()

        for original, unchanged in zip(self.spike_times, before, strict=True):
            np.testing.assert_array_equal(original, unchanged)


class TestImmobileIntervals:
    def test_it_finds_the_still_stretches(self):
        time = np.arange(1000) / SAMPLING_FREQUENCY
        speed = np.full(1000, 20.0)
        speed[100:400] = 0.5
        speed[600:900] = 0.5

        intervals = immobile_intervals(time, speed, 4.0, 0.100)

        assert intervals.shape == (2, 2)
        np.testing.assert_allclose(intervals[0], [time[100], time[399]], atol=1e-9)

    def test_stretches_shorter_than_the_minimum_are_dropped(self):
        time = np.arange(1000) / SAMPLING_FREQUENCY
        speed = np.full(1000, 20.0)
        speed[100:110] = 0.5  # 20 ms
        speed[600:900] = 0.5

        intervals = immobile_intervals(time, speed, 4.0, 0.100)

        assert intervals.shape == (1, 2)

    def test_a_stretch_touching_the_start_is_kept(self):
        time = np.arange(1000) / SAMPLING_FREQUENCY
        speed = np.full(1000, 20.0)
        speed[:300] = 0.5

        intervals = immobile_intervals(time, speed, 4.0, 0.100)

        assert intervals[0, 0] == pytest.approx(time[0])

    def test_no_still_stretch_gives_nothing(self):
        time = np.arange(100) / SAMPLING_FREQUENCY

        intervals = immobile_intervals(time, np.full(100, 20.0), 4.0, 0.100)

        assert intervals.shape == (0, 2)


class TestPlacements:
    intervals = np.array([[10.0, 13.0], [20.0, 20.5]])

    def test_every_placement_leaves_room_for_the_margin(self):
        chosen = placements(
            self.intervals,
            duration_s=0.100,
            margin_s=1.0,
            n=200,
            rng=np.random.default_rng(0),
        )

        assert chosen.size == 200
        assert (chosen - 1.0 >= 10.0 - 1e-9).all()
        assert (chosen + 0.100 + 1.0 <= 13.0 + 1e-9).all()

    def test_an_interval_too_short_for_the_margin_is_not_used(self):
        """The second interval cannot hold the event plus its margins."""
        chosen = placements(
            self.intervals,
            duration_s=0.100,
            margin_s=1.0,
            n=200,
            rng=np.random.default_rng(0),
        )

        assert not ((chosen >= 20.0) & (chosen <= 20.5)).any()

    def test_placements_are_spread_through_the_usable_interval(self):
        chosen = placements(
            self.intervals,
            duration_s=0.100,
            margin_s=1.0,
            n=400,
            rng=np.random.default_rng(0),
        )

        assert chosen.max() - chosen.min() > 0.7 * (13.0 - 10.0 - 2.0 - 0.100)

    def test_no_usable_interval_raises(self):
        with pytest.raises(ValueError, match="margin"):
            placements(
                self.intervals,
                duration_s=0.100,
                margin_s=5.0,
                n=10,
                rng=np.random.default_rng(0),
            )


class TestWithout:
    def test_an_excluded_span_splits_an_interval(self):
        kept = without(np.array([[0.0, 10.0]]), np.array([[4.0, 6.0]]))

        np.testing.assert_allclose(kept, [[0.0, 4.0], [6.0, 10.0]])

    def test_a_margin_widens_what_is_removed(self):
        kept = without(np.array([[0.0, 10.0]]), np.array([[4.0, 6.0]]), margin_s=1.0)

        np.testing.assert_allclose(kept, [[0.0, 3.0], [7.0, 10.0]])

    def test_a_zero_width_span_still_removes_its_margin(self):
        """The simulation's "local" entry has no width, and it still sits in a pause."""
        kept = without(np.array([[0.0, 10.0]]), np.array([[5.0, 5.0]]), margin_s=0.5)

        np.testing.assert_allclose(kept, [[0.0, 4.5], [5.5, 10.0]])

    def test_an_interval_swallowed_whole_disappears(self):
        kept = without(np.array([[4.0, 6.0]]), np.array([[0.0, 10.0]]))

        assert kept.shape == (0, 2)

    def test_other_intervals_are_untouched(self):
        kept = without(np.array([[0.0, 10.0], [20.0, 30.0]]), np.array([[4.0, 6.0]]))

        np.testing.assert_allclose(kept, [[0.0, 4.0], [6.0, 10.0], [20.0, 30.0]])

    def test_every_excluded_span_is_applied(self):
        """Removing one span must not stop the rest from being removed."""
        kept = without(np.array([[0.0, 30.0]]), np.array([[4.0, 6.0], [14.0, 16.0]]))

        np.testing.assert_allclose(kept, [[0.0, 4.0], [6.0, 14.0], [16.0, 30.0]])

    def test_nothing_excluded_returns_what_it_was_given(self):
        kept = without(np.array([[0.0, 10.0]]), np.zeros((0, 2)))

        np.testing.assert_allclose(kept, [[0.0, 10.0]])
