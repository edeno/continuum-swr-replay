"""Bin filter, sequence assembly, merge, and replay criteria."""

import numpy as np
import pytest

from continuum_swr_replay.analysis.detection.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.detect import (
    SequenceAssembly,
    WindowDecisions,
    WindowGates,
    apply_speed_gate,
    assemble_sequences,
    detect_candidate_sequences,
    gate_windows,
    merge_runs,
    replay_mask,
    sequence_dispersion_cm,
    sequence_duration_s,
    window_decisions,
)
from continuum_swr_replay.analysis.detection.memoryless_sequence.window_statistics import (
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

    def test_a_speed_rejection_is_told_apart_from_an_unusable_posterior(self):
        """Real tracking has gaps; the output must say which windows were lost
        to speed and which to the decoder, not fold both into one flag."""
        gates = gate_windows(
            np.array([5, NO_CENTER, 5]),
            np.array([2.0, np.nan, 2.0]),
            np.array([3.0, np.nan, 3.0]),
            self.config,
            speed_eligible=np.array([False, True, True]),
        )

        assert gates.posterior_usable.tolist() == [True, False, True]
        assert gates.speed_eligible.tolist() == [False, True, True]
        assert gates.has_center.tolist() == [False, False, True]
        assert gates.kept.tolist() == [False, False, True]


class TestWindowGatesInvariants:
    def test_arrays_of_different_lengths_raise(self):
        with pytest.raises(ValueError, match="length"):
            WindowGates(
                np.ones(3, bool), np.ones(4, bool), np.ones(3, bool), np.ones(3, bool)
            )

    def test_kept_cannot_disagree_with_its_parts(self):
        """``kept`` is derived, so there is no way to store a contradiction."""
        gates = WindowGates(
            np.ones(2, bool),
            np.ones(2, bool),
            np.ones(2, bool),
            np.array([True, False]),
        )

        assert gates.kept.tolist() == [True, False]
        assert not hasattr(gates, "__dict__") or "kept" not in vars(gates)

    def test_a_center_that_the_gates_deny_raises(self):
        gates = WindowGates(
            np.array([True, False]),
            np.ones(2, bool),
            np.ones(2, bool),
            np.ones(2, bool),
        )

        with pytest.raises(ValueError, match="has_center"):
            WindowDecisions(np.array([3, 4]), np.zeros(2), np.zeros(2), gates)


class TestSequenceAssemblyInvariants:
    def test_a_sequence_must_be_bounded_by_its_pieces(self):
        with pytest.raises(ValueError, match="pieces"):
            SequenceAssembly(
                runs=np.array([[0, 9]]),
                pieces=(np.array([0]),),
                pre_merge_runs=np.array([[0, 5]]),
                kept=np.ones(10, bool),
            )

    def test_a_sequence_beyond_the_windows_raises(self):
        with pytest.raises(ValueError, match="only 5 windows"):
            SequenceAssembly(
                runs=np.array([[0, 9]]),
                pieces=(np.array([0]),),
                pre_merge_runs=np.array([[0, 9]]),
                kept=np.ones(5, bool),
            )


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

    def test_a_run_endpoint_without_a_center_raises(self):
        """NO_CENTER is -1, which would silently index the last bin and measure
        a gap that does not exist."""
        center = np.array([1, NO_CENTER, NO_CENTER, 40, 40])

        with pytest.raises(ValueError, match="no center"):
            self.merge([[0, 1], [3, 4]], center)

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

    def linked_by_one_window(self) -> np.ndarray:
        """Two 15-window runs 56 ms apart, and one kept window 28 ms from each."""
        kept = np.zeros(50, dtype=bool)
        kept[0:15] = True
        kept[21] = True
        kept[28:43] = True
        return kept

    def test_by_default_a_single_window_cannot_link_two_runs(self):
        assembly = self.assemble(self.linked_by_one_window())

        np.testing.assert_array_equal(assembly.pre_merge_runs, [[0, 14], [28, 42]])
        assert assembly.runs.shape == (0, 2)  # too far apart, too short alone

    def test_keeping_single_window_runs_lets_one_link_a_merge(self):
        """The sensitivity setting for the reference's unvendored ``seqle``,
        whose length convention decides whether such a window survives."""
        config = MemorylessSequenceConfig(drop_single_window_runs=False)

        assembly = self.assemble(self.linked_by_one_window(), config=config)

        np.testing.assert_array_equal(
            assembly.pre_merge_runs, [[0, 14], [21, 21], [28, 42]]
        )
        np.testing.assert_array_equal(assembly.runs, [[0, 42]])

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
        mask = replay_mask(np.array([[0, 50]]), np.array([20.0]), self.config)

        assert mask.tolist() == [True]

    def test_a_short_sequence_is_not_replay(self):
        mask = replay_mask(np.array([[0, 12]]), np.array([20.0]), self.config)

        assert mask.tolist() == [False]

    def test_a_compact_sequence_is_not_replay(self):
        mask = replay_mask(np.array([[0, 50]]), np.array([5.0]), self.config)

        assert mask.tolist() == [False]

    def test_the_thresholds_are_strict(self):
        """25 strides is exactly 100 ms and fails, as in the reference driver."""
        mask = replay_mask(
            np.array([[0, 25], [0, 50]]), np.array([20.0, 12.0]), self.config
        )

        assert mask.tolist() == [False, False]

    def test_one_stride_past_the_duration_threshold_is_replay(self):
        mask = replay_mask(np.array([[3, 29]]), np.array([20.0]), self.config)

        assert mask.tolist() == [True]

    def test_a_duration_between_strides_is_rounded_up(self):
        """50 ms is 12.5 strides of 4 ms: 13 strides is strictly longer."""
        config = MemorylessSequenceConfig(minimum_duration_s=0.050)

        mask = replay_mask(np.array([[0, 12], [0, 13]]), np.array([20.0, 20.0]), config)

        assert mask.tolist() == [False, True]


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
        """The speed gate removes a window before its jump is measured."""
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


class TestDetectCandidateSequences:
    """The whole per-window pipeline, in the one order that is correct."""

    config = MemorylessSequenceConfig()
    distance = straight_track(60)

    def moving_posterior(self, bins: np.ndarray) -> np.ndarray:
        posterior = np.zeros((bins.size, 60))
        posterior[np.arange(bins.size), bins] = 1.0
        return posterior

    def test_a_long_travelling_sequence_is_replay(self):
        bins = np.arange(40)
        midpoints = window_times(40)

        found = detect_candidate_sequences(
            self.moving_posterior(bins),
            midpoints,
            self.distance,
            np.zeros(40),
            self.config,
        )

        np.testing.assert_array_equal(found.runs, [[0, 39]])
        np.testing.assert_allclose(found.duration_s, [39 * 0.004])
        assert found.is_replay.tolist() == [True]

    @pytest.mark.parametrize("origin_s", [0.0, 1.58e9])
    def test_a_hundred_millisecond_sequence_fails_whatever_the_clock(self, origin_s):
        """On a recording clock 100 ms of midpoints subtracts to 0.100000143 s."""
        # every second sample of a 500 Hz grid, from the second window on
        midpoints = origin_s + np.arange(2, 54, 2) / 500.0
        assert origin_s == 0.0 or midpoints[-1] - midpoints[0] > 0.100

        found = detect_candidate_sequences(
            self.moving_posterior(np.arange(26)),
            midpoints,
            self.distance,
            np.zeros(26),
            self.config,
        )

        np.testing.assert_array_equal(found.runs, [[0, 25]])
        assert found.dispersion_cm[0] > self.config.dispersion_threshold_cm
        assert found.is_replay.tolist() == [False]

    def test_a_stationary_sequence_is_not_replay(self):
        bins = np.full(40, 20)

        found = detect_candidate_sequences(
            self.moving_posterior(bins),
            window_times(40),
            self.distance,
            np.zeros(40),
            self.config,
        )

        np.testing.assert_array_equal(found.runs, [[0, 39]])
        np.testing.assert_allclose(found.dispersion_cm, [0.0])
        assert found.is_replay.tolist() == [False]

    def test_the_speed_gate_precedes_the_jumps(self):
        bins = np.arange(27)
        speed = np.zeros(27)
        speed[-1] = 9.0

        found = detect_candidate_sequences(
            self.moving_posterior(bins),
            window_times(27),
            self.distance,
            speed,
            self.config,
        )

        assert found.decisions.gates.kept.sum() == 25
        assert found.runs.shape == (0, 2)

    def test_the_intermediate_stages_are_kept(self):
        bins = np.arange(40)

        found = detect_candidate_sequences(
            self.moving_posterior(bins),
            window_times(40),
            self.distance,
            np.zeros(40),
            self.config,
        )

        assert found.pre_merge_runs.shape == (1, 2)
        assert [piece.tolist() for piece in found.pieces] == [[0]]
        assert found.decisions.spread_cm.size == 40

    def test_a_posterior_and_midpoints_of_different_lengths_raise(self):
        with pytest.raises(ValueError, match="midpoint"):
            detect_candidate_sequences(
                self.moving_posterior(np.arange(40)),
                window_times(39),
                self.distance,
                np.zeros(40),
                self.config,
            )

    def test_no_windows_gives_empty_results_with_the_right_types(self):
        found = detect_candidate_sequences(
            np.zeros((0, 60)), np.zeros(0), self.distance, np.zeros(0), self.config
        )

        assert found.runs.shape == (0, 2)
        assert found.duration_s.dtype == np.float64
        assert found.is_replay.dtype == np.bool_

    def test_midpoints_out_of_order_raise(self):
        midpoints = window_times(40)
        midpoints[10] = 5.0

        with pytest.raises(ValueError, match="increasing"):
            detect_candidate_sequences(
                self.moving_posterior(np.arange(40)),
                midpoints,
                self.distance,
                np.zeros(40),
                self.config,
            )

    @pytest.mark.parametrize("bad", [np.nan, np.inf])
    def test_a_row_mixing_finite_and_non_finite_values_raises(self, bad):
        """Only an all-NaN row means "not evaluated"; anything else is broken
        input that would otherwise become a silent rejection."""
        posterior = self.moving_posterior(np.arange(40))
        posterior[3, 0] = bad

        with pytest.raises(ValueError, match="mixes"):
            detect_candidate_sequences(
                posterior, window_times(40), self.distance, np.zeros(40), self.config
            )

    def test_an_infinite_midpoint_raises(self):
        midpoints = window_times(40)
        midpoints[-1] = np.inf

        with pytest.raises(ValueError, match="finite"):
            detect_candidate_sequences(
                self.moving_posterior(np.arange(40)),
                midpoints,
                self.distance,
                np.zeros(40),
                self.config,
            )

    def test_an_unnormalized_posterior_raises(self):
        """Scaling a row leaves its center alone and inflates its spread, so
        it would flip the spread gate silently."""
        posterior = self.moving_posterior(np.arange(40))
        posterior[3] *= 10.0

        with pytest.raises(ValueError, match="normalize"):
            detect_candidate_sequences(
                posterior, window_times(40), self.distance, np.zeros(40), self.config
            )

    def test_a_window_the_decoder_could_not_evaluate_splits_the_run(self):
        """It has no center, so it and its predecessor are rejected, and the
        two halves must each clear the duration filter on their own."""
        posterior = self.moving_posterior(np.arange(60))
        posterior[30] = np.nan

        found = detect_candidate_sequences(
            posterior, window_times(60), self.distance, np.zeros(60), self.config
        )

        gates = found.decisions.gates
        assert not gates.posterior_usable[30]
        assert gates.speed_eligible[30]
        assert gates.kept[29:31].tolist() == [False, False]
        np.testing.assert_array_equal(found.pre_merge_runs, [[0, 28], [31, 59]])
        # the gap is 8 ms and 6 cm, so the merge rule bridges it again
        np.testing.assert_array_equal(found.runs, [[0, 59]])
        assert [piece.tolist() for piece in found.pieces] == [[0, 1]]

    def test_no_merge_bridges_a_window_with_missing_rows(self):
        """A window touching the artifact mask is not decoded, and no
        candidate may span it: the same 8 ms, 6 cm gap is not bridged, so the
        shuffle never re-decodes the masked rows."""
        posterior = self.moving_posterior(np.arange(60))
        posterior[30] = np.nan
        unobserved = np.zeros(60, dtype=bool)
        unobserved[30] = True

        found = detect_candidate_sequences(
            posterior,
            window_times(60),
            self.distance,
            np.zeros(60),
            self.config,
            unobserved=unobserved,
        )

        np.testing.assert_array_equal(found.pre_merge_runs, [[0, 28], [31, 59]])
        np.testing.assert_array_equal(found.runs, [[0, 28], [31, 59]])

    def test_a_wide_merge_gap_still_does_not_bridge_missing_rows(self):
        config = MemorylessSequenceConfig(merge_gap_s=1.0, merge_gap_cm=200.0)
        posterior = self.moving_posterior(np.r_[np.arange(30), np.arange(30)])
        posterior[30:40] = np.nan
        unobserved = np.zeros(60, dtype=bool)
        unobserved[30:40] = True

        bridged = detect_candidate_sequences(
            posterior, window_times(60), self.distance, np.zeros(60), config
        )
        found = detect_candidate_sequences(
            posterior,
            window_times(60),
            self.distance,
            np.zeros(60),
            config,
            unobserved=unobserved,
        )

        np.testing.assert_array_equal(bridged.runs, [[0, 59]])
        assert not any(unobserved[first : last + 1].any() for first, last in found.runs)

    def test_the_result_carries_what_produced_it(self):
        found = detect_candidate_sequences(
            self.moving_posterior(np.arange(40)),
            window_times(40),
            self.distance,
            np.zeros(40),
            self.config,
        )

        assert found.config is self.config
        np.testing.assert_allclose(
            found.duration_s,
            found.midpoint_s[found.runs[:, 1]] - found.midpoint_s[found.runs[:, 0]],
        )
        np.testing.assert_array_equal(
            found.is_replay,
            replay_mask(found.runs, found.dispersion_cm, found.config),
        )
