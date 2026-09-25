"""The reference's arithmetic, and the port's agreement with it on sorted cells."""

import numpy as np
import pytest

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.detect import (
    detect_candidate_sequences,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (
    sorted_spikes_epoch_log_likelihood,
    window_midpoints_s,
    with_reference_rate_floor,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.shuffle import (
    ShuffleSeed,
    cell_identity_permutation,
    sorted_candidate_window_log_likelihood,
    unmoved_center_pairs,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.statistics import (
    aggregate_windows,
    normalize_windows,
)
from scripts.qa import memoryless_reference as reference
from scripts.qa.compare_memoryless_to_reference import (
    fields_and_thresholds,
    window_counts,
)
from tests.unit._memoryless_clusterless_epoch import (
    SAMPLING_FREQUENCY,
    build_sorted_track,
)

CONFIG = MemorylessSequenceConfig(sampling_frequency=SAMPLING_FREQUENCY)
THRESHOLDS = reference.ReferenceThresholds(
    spread_cm=10.0,
    jump_cm=20.0,
    speed_cm_per_s=4.0,
    merge_gap_cm=20.0,
    merge_gap_s=0.05,
    minimum_duration_s=0.1,
    dispersion_cm=12.0,
    stride_s=0.004,
    bin_size_cm=2.0,
)


class TestDecodeWindows:
    def test_is_the_poisson_posterior_with_the_floor(self):
        rate_hz = np.array([[10.0, 0.0, 5.0]])
        counts = np.array([[2.0]])

        posterior = reference.decode_windows(counts, rate_hz, window_s=0.08)

        rate = rate_hz[0] + 1e-3
        expected = rate**2 * np.exp(-0.08 * rate)
        np.testing.assert_allclose(posterior[0], expected / expected.sum())

    def test_without_the_floor_a_nearly_silent_bin_barely_holds_a_spike(self):
        """The library clips its fields at 1e-15 per sample, not at a floor."""
        posterior = reference.decode_windows(
            np.array([[1.0]]), np.array([[10.0, 1e-12]]), 0.08, baseline_hz=0.0
        )

        assert posterior[0, 1] < 1e-12


class TestCenterAndSpread:
    def test_are_the_continuous_mean_and_its_standard_deviation(self):
        posterior = np.array([[0.0, 0.5, 0.0, 0.5]])

        center, spread = reference.center_and_spread(posterior)

        np.testing.assert_allclose(center, [2.0])  # between the two masses
        np.testing.assert_allclose(spread, [1.0])


class TestMergeRuns:
    center = np.arange(40, dtype=np.float64)  # one bin per window
    time_s = 0.004 * np.arange(40)

    def test_a_chain_of_close_runs_becomes_one(self):
        runs = np.array([[0, 4], [6, 10], [12, 16]])

        merged = reference.merge_runs(runs, self.center, self.time_s, THRESHOLDS)

        np.testing.assert_array_equal(merged, [[0, 16]])

    def test_a_gap_too_long_in_time_is_not_merged(self):
        runs = np.array([[0, 4], [20, 24]])  # 64 ms apart, at one place

        merged = reference.merge_runs(runs, np.zeros(40), self.time_s, THRESHOLDS)

        np.testing.assert_array_equal(merged, runs)

    def test_a_gap_too_far_in_place_is_not_merged(self):
        center = self.center.copy()
        center[6:] += 20.0  # 22 bins, 44 cm, against 20 cm
        runs = np.array([[0, 4], [6, 10]])

        merged = reference.merge_runs(runs, center, self.time_s, THRESHOLDS)

        np.testing.assert_array_equal(merged, runs)


class TestRunsOfKept:
    def test_a_run_of_one_window_is_dropped(self):
        kept = np.array([True, False, True, True, False, True])

        np.testing.assert_array_equal(reference.runs_of_kept(kept), [[2, 3]])


class TestShuffleStatistics:
    def test_an_exact_zero_jump_is_set_aside(self):
        """A window pair whose counts are the same decodes the same center."""
        rate_hz = np.array([[40.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 40.0]])
        counts = np.array([[3.0, 0.0], [3.0, 0.0], [0.0, 3.0]])

        _, mean_jump = reference.shuffle_statistics(counts, rate_hz, 0.08)

        center, _ = reference.center_and_spread(
            reference.decode_windows(counts, rate_hz, 0.08)
        )
        # the jumps are [0, d, d] after padding; the zero is set aside
        assert mean_jump == pytest.approx(abs(center[2] - center[1]))


@pytest.fixture(scope="module")
def sorted_track():
    return build_sorted_track()


class TestPortAgreesWithTheReference:
    """The port, decoding with the reference's floor, decides as the reference does.

    What remains between them is the binned center, which on this fixture
    changes no gate, candidate or verdict.
    """

    @pytest.fixture(scope="class")
    def both(self, sorted_track):
        result = sorted_spikes_epoch_log_likelihood(
            sorted_track.time,
            sorted_track.linear_position,
            sorted_track.spike_times,
            with_reference_rate_floor(sorted_track.encoding_model, CONFIG),
            CONFIG,
        )
        posterior = normalize_windows(
            aggregate_windows(
                result.log_likelihood, CONFIG.window_n_samples, CONFIG.stride_n_samples
            )
        )
        midpoint_s = window_midpoints_s(result.time, CONFIG)
        speed = np.zeros(midpoint_s.size)  # every window eligible
        port = detect_candidate_sequences(
            posterior, midpoint_s, sorted_track.bin_distance_cm, speed, CONFIG
        )
        rate_hz, _, thresholds = fields_and_thresholds(
            sorted_track.encoding_model, CONFIG
        )
        counts = window_counts(
            sorted_track.spike_times, sorted_track.time, result.time.size, CONFIG
        )
        reference_posterior = reference.decode_windows(
            counts, rate_hz, CONFIG.window_duration_s
        )
        ref = reference.detect(reference_posterior, midpoint_s, speed, thresholds)
        return posterior, port, reference_posterior, ref

    def test_the_posteriors_are_the_same(self, both):
        posterior, _, reference_posterior, _ = both

        np.testing.assert_allclose(reference_posterior, posterior, atol=1e-5)

    def test_the_same_windows_are_kept(self, both):
        _, port, _, ref = both

        assert port.decisions.gates.kept.sum() > 0
        np.testing.assert_array_equal(ref.kept, port.decisions.gates.kept)

    def test_the_same_sequences_are_found_and_judged_alike(self, both):
        _, port, _, ref = both

        assert len(port.runs) > 0 and port.is_replay.any()
        np.testing.assert_array_equal(ref.runs, port.runs)
        np.testing.assert_array_equal(ref.is_replay, port.is_replay)


class TestZeroJumpRuleMatchesTheReference:
    """The pairs set aside are the reference's, in the observed decode and every draw.

    The reference decodes each window from its own counts and sets aside a
    jump that is exactly zero on its continuous center of mass; the port must
    set aside the same pairs, including under permuted fields, where the
    pairs can differ from the observed decode's.
    """

    WINDOWS = (60, 260)
    N_DRAWS = 20

    def test_the_same_pairs_are_set_aside_in_every_draw(self, sorted_track):
        first, last = self.WINDOWS
        floored = with_reference_rate_floor(sorted_track.encoding_model, CONFIG)
        rate_hz, _, _ = fields_and_thresholds(sorted_track.encoding_model, CONFIG)
        n_rows = sorted_spikes_epoch_log_likelihood(
            sorted_track.time,
            sorted_track.linear_position,
            sorted_track.spike_times,
            floored,
            CONFIG,
        ).time.size
        counts = window_counts(
            sorted_track.spike_times, sorted_track.time, n_rows, CONFIG
        )[first : last + 1]
        seed = ShuffleSeed(run=1, session="zero-jump")
        permutations = [None] + [
            cell_identity_permutation(
                seed, candidate=first, draw=draw, n=rate_hz.shape[0]
            )
            for draw in range(1, self.N_DRAWS + 1)
        ]

        set_aside_somewhere = False
        for permutation in permutations:
            port = unmoved_center_pairs(
                sorted_candidate_window_log_likelihood(
                    first,
                    last,
                    sorted_track.time,
                    sorted_track.spike_times,
                    floored,
                    CONFIG,
                    permutation=permutation,
                )
            )
            ref = reference.set_aside_pairs(
                counts,
                rate_hz if permutation is None else rate_hz[permutation],
                CONFIG.window_duration_s,
            )
            np.testing.assert_array_equal(port, ref)
            set_aside_somewhere |= bool(ref.any())
        assert set_aside_somewhere
