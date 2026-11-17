"""Injecting replays into a simulated recording, and placing them safely."""

import numpy as np
import pytest

from continuum_swr_replay.analysis.detection.memoryless_sequence.simulation import (
    Gap,
    InjectedActivity,
    InjectedReplay,
    Segment,
    field_rates,
    immobile_intervals,
    inject_activity,
    inject_replay,
    jumping_event,
    piecewise_replay,
    place_field_means,
    placeless_event,
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

    def inject(self, sampling_frequency=SAMPLING_FREQUENCY, **kwargs):
        defaults = {
            "start_cm": 20.0,
            "path_cm": 100.0,
            "duration_s": 0.200,
            "peak_rate_hz": 80.0,
            "at_s": 4.0,
            "seed": 0,
        }
        return inject_replay(
            self.spike_times,
            self.means,
            InjectedReplay(**(defaults | kwargs)),
            sampling_frequency=sampling_frequency,
        )

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

    def test_one_place_field_per_cell_is_required(self):
        with pytest.raises(ValueError, match="place fields"):
            inject_replay(
                self.spike_times[:3],
                self.means,
                InjectedReplay(20.0, 100.0, 0.2, 80.0, 4.0),
                sampling_frequency=SAMPLING_FREQUENCY,
            )

    def test_a_rate_that_would_fire_at_every_sample_is_refused(self):
        """Bernoulli-per-sample firing saturates; the rate would silently be
        lower than asked."""
        with pytest.raises(ValueError, match="every sample"):
            self.inject(peak_rate_hz=600.0, sampling_frequency=500.0)


class TestInjectedReplay:
    def test_a_replay_knows_when_it_ends(self):
        replay = InjectedReplay(0.0, 50.0, duration_s=0.2, peak_rate_hz=40.0, at_s=1.0)

        assert replay.end_s == pytest.approx(1.2)

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            ("duration_s", 0.0, "duration_s"),
            ("duration_s", -0.1, "duration_s"),
            ("peak_rate_hz", -5.0, "peak_rate_hz"),
            ("at_s", float("nan"), "finite"),
            ("path_cm", float("inf"), "finite"),
            ("seed", 0.5, "seed"),
        ],
    )
    def test_an_impossible_replay_raises(self, field, value, message):
        replay = {
            "start_cm": 0.0,
            "path_cm": 50.0,
            "duration_s": 0.2,
            "peak_rate_hz": 40.0,
            "at_s": 1.0,
            "seed": 0,
        }
        replay[field] = value

        with pytest.raises(ValueError, match=message):
            InjectedReplay(**replay)


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


class TestInjectReplayDraws:
    def test_a_small_replay_draws_the_spikes_it_always_has(self):
        """Pinned: rebuilding the injector must not change a replay's spikes."""
        spike_times = [np.array([0.0, 0.5, 2.0]) for _ in range(3)]

        injected, bounds = inject_replay(
            spike_times,
            np.array([0.0, 10.0, 20.0]),
            InjectedReplay(0.0, 20.0, 0.2, 40.0, 1.0, seed=4),
            sampling_frequency=100.0,
        )

        assert bounds == (1.0, 1.2)
        np.testing.assert_allclose(injected[0], [0.0, 0.5, 1.01052632, 2.0])
        np.testing.assert_allclose(injected[1], [0.0, 0.5, 1.09473684, 1.11578947, 2.0])
        np.testing.assert_allclose(injected[2], [0.0, 0.5, 1.18947368, 2.0])


class TestPiecewiseReplay:
    means = place_field_means(25, 170.0)

    def test_a_jump_leaves_one_place_and_starts_at_another_without_a_gap(self):
        activity = piecewise_replay(
            [Segment(0.12, 20.0, 60.0), Segment(0.12, 120.0, 60.0)],
            at_s=1.0,
            sampling_frequency=1000.0,
            place_field_means=self.means,
            peak_rate_hz=60.0,
        )

        (first_start, first_end), (second_start, second_end) = activity.parts
        assert first_end == pytest.approx(second_start)
        assert np.isfinite(activity.position_cm).all()
        first = activity.time_s <= first_end
        np.testing.assert_allclose(
            activity.position_cm[first][[0, -1]], [20.0, 80.0], atol=0.6
        )
        np.testing.assert_allclose(
            activity.position_cm[~first][[0, -1]], [120.0, 180.0], atol=0.6
        )

    def test_a_chain_is_silent_in_its_gap(self):
        activity = piecewise_replay(
            [Segment(0.15, 20.0, 60.0), Gap(0.06), Segment(0.15, 80.0, 60.0)],
            at_s=1.0,
            sampling_frequency=1000.0,
            place_field_means=self.means,
            peak_rate_hz=160.0,
        )
        (_, first_end), (second_start, _) = activity.parts
        in_gap = (activity.time_s > first_end) & (activity.time_s < second_start)

        injected = inject_activity(
            [np.array([1.1, 1.18, 1.3]) for _ in range(25)], activity, seed=0
        )

        assert second_start - first_end == pytest.approx(0.06)
        assert np.isnan(activity.position_cm[in_gap]).all()
        assert (activity.rate_hz[in_gap] == 0).all()
        spikes = np.concatenate(injected)
        assert not ((spikes > first_end) & (spikes < second_start)).any()

    def test_a_replay_must_begin_and_end_with_a_segment(self):
        with pytest.raises(ValueError, match="segment"):
            piecewise_replay(
                [Gap(0.05), Segment(0.1, 0.0, 10.0)], 0.0, 1000.0, self.means, 20.0
            )


class TestControlEvents:
    means = place_field_means(25, 170.0)

    def test_a_placeless_event_fires_every_cell_at_one_rate(self):
        activity = placeless_event(0.2, 1.0, 1000.0, self.means, 60.0, 170.0)

        assert np.unique(activity.rate_hz).size == 1
        assert np.isnan(activity.position_cm).all()
        assert activity.truth == "no_content"

    def test_a_placeless_event_holds_as_many_spikes_as_a_replay_there(self):
        activity = placeless_event(0.2, 1.0, 1000.0, self.means, 60.0, 170.0)
        held_at_mid_track = field_rates(
            np.full(activity.time_s.size, 85.0), self.means, 60.0
        )

        np.testing.assert_allclose(
            activity.rate_hz.sum(), held_at_mid_track.sum(), rtol=1e-12
        )

    def test_a_jumping_event_holds_each_place_for_its_hold(self):
        activity = jumping_event(0.2, 1.0, 1000.0, self.means, 60.0, 170.0, seed=2)
        hold = np.floor((activity.time_s - 1.0) / 0.015)

        for index in np.unique(hold):
            assert np.unique(activity.position_cm[hold == index]).size == 1
        assert np.unique(activity.position_cm).size == np.unique(hold).size
        assert activity.truth == "non_local"

    def test_a_jumping_event_draws_places_along_the_whole_track(self):
        places = np.concatenate(
            [
                np.unique(
                    jumping_event(
                        0.5, 0.0, 1000.0, self.means, 60.0, 170.0, seed=seed
                    ).position_cm
                )
                for seed in range(20)
            ]
        )

        assert places.min() >= 0.0 and places.max() <= 170.0
        assert places.min() < 10.0 and places.max() > 160.0
        assert abs(places.mean() - 85.0) < 10.0


class TestInjectedActivity:
    def test_an_unknown_truth_is_refused(self):
        with pytest.raises(ValueError, match="truth"):
            InjectedActivity(
                np.arange(3.0),
                1.0,
                np.zeros(3),
                np.zeros((3, 2)),
                ((0.0, 2.0),),
                "replay",
            )

    def test_rates_for_the_wrong_number_of_samples_are_refused(self):
        with pytest.raises(ValueError, match="samples"):
            InjectedActivity(
                np.arange(3.0),
                1.0,
                np.zeros(3),
                np.zeros((4, 2)),
                ((0.0, 2.0),),
                "local",
            )
