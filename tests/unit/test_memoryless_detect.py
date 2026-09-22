"""Bin filter, sequence assembly, merge, and replay criteria."""

import numpy as np
import pytest

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.detect import (
    apply_speed_gate,
    assemble_sequences,
    contiguous_runs,
    gate_windows,
    merge_runs,
    replay_mask,
    sequence_dispersion_cm,
    sequence_duration_s,
    window_decisions,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.statistics import (
    NO_CENTER,
    forward_jumps_cm,
    window_center_and_spread,
)


def straight_track(n_bins: int, bin_size: float = 2.0) -> np.ndarray:
    """Distances along a straight segment of evenly spaced bins."""
    centers = np.arange(n_bins) * bin_size
    return np.abs(centers[:, None] - centers[None, :])


def accumulated_track(n_bins: int, bin_size: float = 0.4) -> np.ndarray:
    """A track whose bin centers accumulate, as summed edge lengths do.

    At this bin size the distance ten 2 cm steps apart is 19.999999999999993
    rather than 20: the thresholds land on values floating point cannot
    represent exactly.
    """
    centers = np.cumsum(np.full(n_bins, bin_size))
    return np.abs(centers[:, None] - centers[None, :])


def window_times(n_windows: int, stride_s: float = 0.004) -> np.ndarray:
    """Midpoints of consecutive windows."""
    return np.arange(n_windows) * stride_s


class TestApplySpeedGate:
    def test_keeps_centers_below_the_threshold(self):
        center = np.array([3, 4, 5])

        gated = apply_speed_gate(center, np.array([0.0, 1.0, 3.9]), 4.0)

        np.testing.assert_array_equal(gated, [3, 4, 5])

    def test_blanks_centers_at_or_above_the_threshold(self):
        center = np.array([3, 4, 5])

        gated = apply_speed_gate(center, np.array([0.0, 4.0, 9.0]), 4.0)

        np.testing.assert_array_equal(gated, [3, NO_CENTER, NO_CENTER])

    def test_an_unknown_speed_blanks_the_center(self):
        """A window whose speed cannot be established is not eligible."""
        gated = apply_speed_gate(np.array([3, 4]), np.array([np.nan, 1.0]), 4.0)

        np.testing.assert_array_equal(gated, [NO_CENTER, 4])

    def test_leaves_the_input_unchanged(self):
        center = np.array([3, 4])

        apply_speed_gate(center, np.array([9.0, 9.0]), 4.0)

        np.testing.assert_array_equal(center, [3, 4])

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="speed"):
            apply_speed_gate(np.array([3, 4]), np.array([1.0]), 4.0)


class TestGateWindows:
    config = MemorylessSequenceConfig()

    def test_keeps_a_window_with_a_center_and_small_spread_and_jump(self):
        gates = gate_windows(
            np.array([5]), np.array([2.0]), np.array([3.0]), self.config
        )

        assert gates.kept.tolist() == [True]

    def test_a_window_without_a_center_is_rejected(self):
        gates = gate_windows(
            np.array([NO_CENTER]), np.array([np.nan]), np.array([np.nan]), self.config
        )

        assert gates.has_center.tolist() == [False]
        assert gates.kept.tolist() == [False]

    def test_a_wide_posterior_is_rejected(self):
        gates = gate_windows(
            np.array([5]), np.array([11.0]), np.array([3.0]), self.config
        )

        assert gates.spread_below_threshold.tolist() == [False]
        assert gates.kept.tolist() == [False]

    def test_a_large_jump_is_rejected(self):
        gates = gate_windows(
            np.array([5]), np.array([2.0]), np.array([25.0]), self.config
        )

        assert gates.jump_below_threshold.tolist() == [False]
        assert gates.kept.tolist() == [False]

    def test_a_window_whose_successor_has_no_center_is_rejected(self):
        """The forward jump is undefined, so the sequence ends here."""
        gates = gate_windows(
            np.array([5]), np.array([2.0]), np.array([np.nan]), self.config
        )

        assert gates.jump_below_threshold.tolist() == [False]
        assert gates.kept.tolist() == [False]

    def test_a_value_exactly_at_a_threshold_is_rejected(self):
        """The criteria are strict inequalities, as in the reference."""
        gates = gate_windows(
            np.array([5, 5]),
            np.array([10.0, 2.0]),
            np.array([3.0, 20.0]),
            self.config,
        )

        assert gates.kept.tolist() == [False, False]

    def test_a_jump_of_exactly_the_threshold_is_rejected_despite_rounding(self):
        """Summed edge lengths can fall just below a distance they equal exactly."""
        distance = accumulated_track(60)
        jump = distance[0, 50]

        assert jump < self.config.jump_threshold_cm  # 19.999999999999993
        gates = gate_windows(
            np.array([0]), np.array([2.0]), np.array([jump]), self.config
        )

        assert gates.kept.tolist() == [False]

    def test_gates_are_independent_of_each_other(self):
        """Each rejection reason is recorded even when another also applies."""
        gates = gate_windows(
            np.array([5]), np.array([11.0]), np.array([25.0]), self.config
        )

        assert gates.spread_below_threshold.tolist() == [False]
        assert gates.jump_below_threshold.tolist() == [False]
        assert gates.has_center.tolist() == [True]


class TestContiguousRuns:
    def test_no_windows_kept_gives_no_runs(self):
        runs = contiguous_runs(np.zeros(5, dtype=bool))

        assert runs.shape == (0, 2)

    def test_an_empty_mask_gives_no_runs(self):
        runs = contiguous_runs(np.zeros(0, dtype=bool))

        assert runs.shape == (0, 2)

    def test_boundaries_are_inclusive(self):
        runs = contiguous_runs(np.array([False, True, True, True, False]))

        np.testing.assert_array_equal(runs, [[1, 3]])

    def test_separated_runs_stay_separate(self):
        runs = contiguous_runs(np.array([True, True, False, True, False, True]))

        np.testing.assert_array_equal(runs, [[0, 1], [3, 3], [5, 5]])

    def test_a_run_touching_both_ends(self):
        runs = contiguous_runs(np.ones(4, dtype=bool))

        np.testing.assert_array_equal(runs, [[0, 3]])


class TestMergeRuns:
    config = MemorylessSequenceConfig()
    distance = straight_track(60)

    def merge(self, runs, center, time_s=None, config=None):
        runs = np.asarray(runs)
        time_s = window_times(len(center)) if time_s is None else time_s
        return merge_runs(
            runs, np.asarray(center), time_s, self.distance, config or self.config
        )

    def test_a_single_run_is_returned_unchanged(self):
        merged, pieces = self.merge([[0, 2]], [1, 1, 1])

        np.testing.assert_array_equal(merged, [[0, 2]])
        assert [p.tolist() for p in pieces] == [[0]]

    def test_neighbors_close_in_space_and_time_merge(self):
        center = np.array([1, 1, NO_CENTER, 2, 2])

        merged, pieces = self.merge([[0, 1], [3, 4]], center)

        np.testing.assert_array_equal(merged, [[0, 4]])
        assert [p.tolist() for p in pieces] == [[0, 1]]

    def test_a_distant_neighbor_does_not_merge(self):
        # bins 1 and 40 are 78 cm apart on a 2 cm grid
        center = np.array([1, 1, NO_CENTER, 40, 40])

        merged, pieces = self.merge([[0, 1], [3, 4]], center)

        np.testing.assert_array_equal(merged, [[0, 1], [3, 4]])
        assert [p.tolist() for p in pieces] == [[0], [1]]

    def test_a_long_gap_does_not_merge(self):
        center = np.full(40, 1)
        center[2:38] = NO_CENTER

        merged, _ = self.merge([[0, 1], [38, 39]], center)

        np.testing.assert_array_equal(merged, [[0, 1], [38, 39]])

    def test_the_bridge_is_measured_between_the_facing_endpoints(self):
        """Not between run centers: the reference uses the adjacent windows."""
        # the runs face each other at bins 30 and 31, 2 cm apart, but their
        # far endpoints are bins 0 and 59
        center = np.array([0, 30, NO_CENTER, 31, 59])

        merged, _ = self.merge([[0, 1], [3, 4]], center)

        np.testing.assert_array_equal(merged, [[0, 4]])

    def test_merging_is_transitive_across_a_chain(self):
        center = np.array([1, 1, NO_CENTER, 2, 2, NO_CENTER, 3, 3])

        merged, pieces = self.merge([[0, 1], [3, 4], [6, 7]], center)

        np.testing.assert_array_equal(merged, [[0, 7]])
        assert [p.tolist() for p in pieces] == [[0, 1, 2]]

    def test_a_broken_chain_merges_only_its_close_links(self):
        center = np.array([1, 1, NO_CENTER, 2, 2, NO_CENTER, 50, 50])

        merged, pieces = self.merge([[0, 1], [3, 4], [6, 7]], center)

        np.testing.assert_array_equal(merged, [[0, 4], [6, 7]])
        assert [p.tolist() for p in pieces] == [[0, 1], [2]]

    def test_a_gap_exactly_at_the_distance_threshold_does_not_merge(self):
        # bins 1 and 11 are exactly 20 cm apart
        center = np.array([1, 1, NO_CENTER, 11, 11])

        merged, _ = self.merge([[0, 1], [3, 4]], center)

        np.testing.assert_array_equal(merged, [[0, 1], [3, 4]])

    def test_a_gap_exactly_at_the_time_threshold_does_not_merge(self):
        """At the reference's 5 ms stride a 50 ms gap lands on a window."""
        config = MemorylessSequenceConfig(stride_s=0.005, sampling_frequency=1000.0)
        center = np.full(11, 1)
        center[1:10] = NO_CENTER
        time_s = window_times(11, stride_s=0.005)

        merged, _ = merge_runs(
            np.array([[0, 0], [10, 10]]), center, time_s, self.distance, config
        )

        np.testing.assert_array_equal(merged, [[0, 0], [10, 10]])

    def test_no_runs_gives_no_runs(self):
        merged, pieces = merge_runs(
            np.zeros((0, 2), dtype=int),
            np.zeros(0, dtype=int),
            np.zeros(0),
            self.distance,
            self.config,
        )

        assert merged.shape == (0, 2)
        assert pieces == []


class TestAssembleSequences:
    distance = straight_track(60)

    def assemble(self, kept, center=None, config=None):
        kept = np.asarray(kept, dtype=bool)
        config = config or MemorylessSequenceConfig()
        center = np.ones(kept.size, dtype=int) if center is None else np.asarray(center)
        return assemble_sequences(
            kept,
            center,
            window_times(kept.size, config.stride_s),
            self.distance,
            config,
        )

    def test_a_single_window_run_is_dropped_before_merging(self):
        """Its index length is zero, as in the reference."""
        kept = np.zeros(60, dtype=bool)
        kept[10] = True
        kept[20:50] = True

        assembly = self.assemble(kept)

        np.testing.assert_array_equal(assembly.pre_merge_runs, [[20, 49]])

    def test_the_duration_filter_keeps_twenty_six_windows_at_a_four_ms_stride(self):
        kept = np.zeros(40, dtype=bool)
        kept[:26] = True

        assembly = self.assemble(kept)

        np.testing.assert_array_equal(assembly.runs, [[0, 25]])

    def test_the_duration_filter_drops_twenty_five_windows_at_a_four_ms_stride(self):
        kept = np.zeros(40, dtype=bool)
        kept[:25] = True

        assembly = self.assemble(kept)

        assert assembly.runs.shape == (0, 2)

    def test_the_duration_filter_keeps_twenty_one_windows_at_a_five_ms_stride(self):
        config = MemorylessSequenceConfig(stride_s=0.005, sampling_frequency=1000.0)
        kept = np.zeros(40, dtype=bool)
        kept[:21] = True

        assembly = self.assemble(kept, config=config)

        np.testing.assert_array_equal(assembly.runs, [[0, 20]])

    def test_the_duration_filter_drops_twenty_windows_at_a_five_ms_stride(self):
        config = MemorylessSequenceConfig(stride_s=0.005, sampling_frequency=1000.0)
        kept = np.zeros(40, dtype=bool)
        kept[:20] = True

        assembly = self.assemble(kept, config=config)

        assert assembly.runs.shape == (0, 2)

    def test_the_length_filter_runs_after_merging(self):
        """Two short runs that merge survive; neither would alone."""
        kept = np.zeros(40, dtype=bool)
        kept[:14] = True
        kept[16:30] = True

        assembly = self.assemble(kept)

        np.testing.assert_array_equal(assembly.pre_merge_runs, [[0, 13], [16, 29]])
        np.testing.assert_array_equal(assembly.runs, [[0, 29]])

    def test_rejected_windows_inside_a_merged_gap_stay_rejected(self):
        kept = np.zeros(40, dtype=bool)
        kept[:14] = True
        kept[16:30] = True

        assembly = self.assemble(kept)

        assert assembly.kept[14:16].tolist() == [False, False]

    def test_nothing_kept_gives_no_sequences(self):
        assembly = self.assemble(np.zeros(40, dtype=bool))

        assert assembly.runs.shape == (0, 2)
        assert assembly.pre_merge_runs.shape == (0, 2)


class TestSequenceDuration:
    def test_duration_spans_the_first_and_last_window_midpoint(self):
        time_s = window_times(40)

        duration = sequence_duration_s(np.array([[0, 25], [30, 35]]), time_s)

        np.testing.assert_allclose(duration, [25 * 0.004, 5 * 0.004])

    def test_the_window_width_is_not_added(self):
        """Event bounds are midpoints; the decoding support is kept separately."""
        time_s = window_times(30)

        duration = sequence_duration_s(np.array([[0, 25]]), time_s)

        np.testing.assert_allclose(duration, [0.100])

    def test_no_runs_gives_an_empty_float_array(self):
        duration = sequence_duration_s(np.zeros((0, 2), dtype=int), window_times(10))

        assert duration.shape == (0,)
        assert duration.dtype == np.float64


class TestSequenceDispersion:
    distance = straight_track(60)

    def test_dispersion_uses_only_the_retained_centers(self):
        """Windows inside a merged gap are not decoded positions."""
        center = np.array([0, 0, 30, 10, 10])
        kept = np.array([True, True, False, True, True])

        dispersion = sequence_dispersion_cm(
            np.array([[0, 4]]), kept, center, self.distance
        )

        # centers at bins 0, 0, 10, 10 -> 0 cm and 20 cm, equally weighted
        np.testing.assert_allclose(dispersion, [10.0])

    def test_a_stationary_sequence_has_no_dispersion(self):
        center = np.full(5, 7)
        kept = np.ones(5, dtype=bool)

        dispersion = sequence_dispersion_cm(
            np.array([[0, 4]]), kept, center, self.distance
        )

        np.testing.assert_allclose(dispersion, [0.0])

    def test_each_run_is_measured_on_its_own(self):
        center = np.array([0, 0, 0, 20, 30])
        kept = np.ones(5, dtype=bool)

        dispersion = sequence_dispersion_cm(
            np.array([[0, 2], [3, 4]]), kept, center, self.distance
        )

        np.testing.assert_allclose(dispersion, [0.0, 10.0])


class TestReplayMask:
    config = MemorylessSequenceConfig()

    def test_a_long_and_dispersed_sequence_is_replay(self):
        mask = replay_mask(np.array([0.2]), np.array([20.0]), self.config)

        assert mask.tolist() == [True]

    def test_a_short_sequence_is_not_replay(self):
        mask = replay_mask(np.array([0.05]), np.array([20.0]), self.config)

        assert mask.tolist() == [False]

    def test_a_compact_sequence_is_not_replay(self):
        mask = replay_mask(np.array([0.2]), np.array([5.0]), self.config)

        assert mask.tolist() == [False]

    def test_the_thresholds_are_strict(self):
        """A duration of exactly 100 ms fails, as in the reference driver."""
        mask = replay_mask(np.array([0.100, 0.2]), np.array([20.0, 12.0]), self.config)

        assert mask.tolist() == [False, False]

    def test_a_hundred_millisecond_run_fails_on_an_accumulated_time_vector(self):
        """A time vector built by accumulation overshoots 100 ms by rounding."""
        time_s = np.cumsum(np.full(30, 0.004))
        duration = sequence_duration_s(np.array([[0, 25]]), time_s)

        assert duration[0] > 0.100  # 0.10000000000000006
        mask = replay_mask(duration, np.array([20.0]), self.config)

        assert mask.tolist() == [False]

    def test_a_duration_of_exactly_one_hundred_milliseconds_fails_on_the_grid(self):
        """The 100 ms threshold lands on a window boundary at a 4 ms stride."""
        duration = sequence_duration_s(np.array([[0, 25]]), window_times(30))

        mask = replay_mask(duration, np.array([20.0]), self.config)

        assert mask.tolist() == [False]


class TestWindowDecisions:
    """The speed gate must blank centers before jumps are computed."""

    config = MemorylessSequenceConfig()
    distance = straight_track(60)

    def posterior_at(self, bins: np.ndarray) -> np.ndarray:
        posterior = np.zeros((bins.size, 60))
        posterior[np.arange(bins.size), bins] = 1.0
        return posterior

    def test_a_window_before_a_speed_rejected_window_is_rejected(self):
        bins = np.array([10, 11, 12, 13])
        speed = np.array([0.0, 0.0, 0.0, 9.0])

        decisions = window_decisions(
            self.posterior_at(bins), self.distance, speed, self.config
        )

        assert decisions.gates.kept.tolist() == [True, True, False, False]

    def test_the_speed_rejected_window_has_no_center(self):
        bins = np.array([10, 11, 12, 13])
        speed = np.array([0.0, 0.0, 0.0, 9.0])

        decisions = window_decisions(
            self.posterior_at(bins), self.distance, speed, self.config
        )

        assert decisions.center_index.tolist() == [10, 11, 12, NO_CENTER]
        assert np.isnan(decisions.jump_cm[2])

    def jumps_before_speed_gate(self, posterior, speed):
        """The wrong order: jumps from every center, then the speed gate."""
        center, spread = window_center_and_spread(posterior, self.distance)
        jump = forward_jumps_cm(center, self.distance)
        center = apply_speed_gate(center, speed, self.config.speed_threshold_cm_per_s)
        return gate_windows(center, spread, jump, self.config)

    def test_jumps_taken_before_the_speed_gate_would_keep_one_more_window(self):
        """Pins the ordering in the spec's data-flow step 4."""
        bins = np.array([10, 11, 12, 13])
        speed = np.array([0.0, 0.0, 0.0, 9.0])
        posterior = self.posterior_at(bins)

        correct = window_decisions(posterior, self.distance, speed, self.config)
        wrong_order = self.jumps_before_speed_gate(posterior, speed)

        assert correct.gates.kept.tolist() == [True, True, False, False]
        assert wrong_order.kept.tolist() == [True, True, True, False]

    def test_the_ordering_changes_whether_a_candidate_is_accepted(self):
        """The extra window is the one the duration filter turns on."""
        bins = np.arange(10, 10 + 27)
        speed = np.zeros(27)
        speed[-1] = 9.0  # the last window is not immobile
        posterior = self.posterior_at(bins)
        time_s = window_times(27)

        def runs_for(kept, center):
            return assemble_sequences(
                kept, center, time_s, self.distance, self.config
            ).runs

        decisions = window_decisions(posterior, self.distance, speed, self.config)
        wrong_order = self.jumps_before_speed_gate(posterior, speed)

        # windows 0-24 survive the correct order: index length 24, one short
        assert runs_for(decisions.gates.kept, decisions.center_index).shape == (0, 2)
        np.testing.assert_array_equal(
            runs_for(wrong_order.kept, decisions.center_index), [[0, 25]]
        )
