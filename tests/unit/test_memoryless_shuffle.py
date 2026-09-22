"""The encoding-pair shuffle that tests a candidate sequence for significance.

Two epochs are used. The shared one from ``_memoryless_clusterless_epoch`` pins
the shuffled likelihood to the library's own decode. A second, built here, is
one tetrode whose units have distinct waveforms and place fields that tile the
track, so the mark is the only thing that says where the animal is. That
isolates what the shuffle tests: in the shared epoch every electrode's marks
are drawn independently of position, so a shuffle there changes nothing.
"""

import copy

import numpy as np
import pytest

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (
    clusterless_epoch_log_likelihood,
    sorted_spikes_epoch_log_likelihood,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.shuffle import (
    MINIMUM_N_SHUFFLES,
    ShuffleResult,
    ShuffleSeed,
    candidate_window_log_likelihood,
    cell_identity_permutation,
    encoding_permutation,
    shuffle_candidate,
    shuffle_candidate_sorted,
    shuffle_p_value,
    shuffle_statistics,
    sorted_candidate_window_log_likelihood,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.statistics import (
    aggregate_windows,
    normalize_windows,
    window_center_and_spread,
)
from tests.unit._memoryless_clusterless_epoch import (
    MOVING_WINDOWS,
    SAMPLING_FREQUENCY,
    SortedTrack,
    TilingTetrode,
    build_simulated_epoch,
    build_sorted_track,
    build_tiling_tetrode,
)

CONFIG = MemorylessSequenceConfig(sampling_frequency=SAMPLING_FREQUENCY)
SEED = ShuffleSeed(run=20260922, session="simulated_epoch")


def straight_track(n_bins: int, bin_size: float = 2.0) -> np.ndarray:
    centers = np.arange(n_bins) * bin_size
    return np.abs(centers[:, None] - centers[None, :])


@pytest.fixture(scope="module")
def epoch(tmp_path_factory):
    return build_simulated_epoch(tmp_path_factory.mktemp("shuffle"))


@pytest.fixture(scope="module")
def tetrode() -> TilingTetrode:
    return build_tiling_tetrode()


class TestPValue:
    """``(1 + nansum(shuffle < observed)) / (1 + numShuffles)``, as the reference
    computes it (``load_replayEvents_cellIDshuffle_sameEvents.m``, lines 78-79)."""

    def test_a_null_entirely_above_the_observed_gives_the_smallest_p(self):
        p = shuffle_p_value(1.0, np.full(100, 2.0))

        assert p == pytest.approx(1 / 101)

    def test_a_tie_does_not_count_against_the_candidate(self):
        """The reference counts draws strictly below the observed value, so a
        null of exact ties gives the smallest p, not the largest."""
        p = shuffle_p_value(0.0, np.zeros(100))

        assert p == pytest.approx(1 / 101)

    def test_counts_draws_strictly_below_the_observed(self):
        null = np.array([0.5, 1.0, 1.5, 2.0, 3.0])

        assert shuffle_p_value(1.0, null) == pytest.approx((1 + 1) / 6)

    def test_a_nan_observed_value_is_below_nothing_and_passes(self):
        """The reference's arithmetic; unreachable for a candidate that meets
        the dispersion criterion, and recorded by ``ShuffleResult.testable``."""
        assert shuffle_p_value(np.nan, np.ones(100)) == pytest.approx(1 / 101)

    def test_a_nan_draw_is_skipped_without_shrinking_the_denominator(self):
        """``nansum`` skips it; ``1 + numShuffles`` does not change."""
        null = np.full(100, 0.5)
        null[3] = np.nan

        assert shuffle_p_value(1.0, null) == pytest.approx((1 + 99) / 101)

    def test_an_empty_null_is_refused(self):
        with pytest.raises(ValueError, match="at least one"):
            shuffle_p_value(1.0, np.array([]))


class TestVerdict:
    def result(self, p_spread_rank: int, p_jump_rank: int) -> ShuffleResult:
        """Nulls placed so each p-value is (1 + rank) / 101."""
        spread_null = np.r_[
            np.full(p_spread_rank, 0.5), np.full(100 - p_spread_rank, 2)
        ]
        jump_null = np.r_[np.full(p_jump_rank, 0.5), np.full(100 - p_jump_rank, 2)]
        return ShuffleResult(
            observed_spread_cm=1.0,
            observed_jump_cm=1.0,
            null_spread_cm=spread_null,
            null_jump_cm=jump_null,
        )

    def test_passes_only_when_both_p_values_are_at_or_below_005(self):
        assert self.result(0, 0).passes
        assert not self.result(0, 10).passes
        assert not self.result(10, 0).passes

    def test_four_draws_below_still_pass_and_five_do_not(self):
        """5 / 101 <= 0.05 < 6 / 101 at 100 draws."""
        assert self.result(4, 4).passes
        assert not self.result(5, 0).passes

    def test_exactly_005_passes_as_the_reference_compares(self):
        """At 19 draws the smallest p is 1 / 20 = 0.05 and the reference's
        ``<= 0.05`` passes it."""
        result = ShuffleResult(1.0, 1.0, np.full(19, 2.0), np.full(19, 2.0))

        assert result.p_spread == pytest.approx(0.05)
        assert result.passes

    def test_a_nan_observed_statistic_is_recorded_and_passes_its_half(self):
        """Following the reference; ``testable`` says it happened."""
        result = ShuffleResult(
            observed_spread_cm=np.nan,
            observed_jump_cm=1.0,
            null_spread_cm=np.full(100, 2.0),
            null_jump_cm=np.full(100, 2.0),
        )

        assert not result.testable
        assert result.passes

    def test_reports_how_many_draws_it_used(self):
        assert self.result(0, 0).n_shuffles == 100

    def test_nulls_of_different_lengths_are_refused(self):
        with pytest.raises(ValueError, match="draws"):
            ShuffleResult(
                observed_spread_cm=1.0,
                observed_jump_cm=1.0,
                null_spread_cm=np.ones(100),
                null_jump_cm=np.ones(99),
            )

    def test_too_few_draws_for_a_pass_are_refused(self):
        with pytest.raises(ValueError, match="n_shuffles"):
            ShuffleResult(1.0, 1.0, np.ones(MINIMUM_N_SHUFFLES - 1), np.ones(18))

    def test_the_draws_are_copied_and_frozen(self):
        null = np.full(100, 2.0)
        result = ShuffleResult(1.0, 1.0, null, null.copy())
        null[:] = 0.5

        assert result.p_spread == pytest.approx(1 / 101)
        with pytest.raises(ValueError):
            result.null_spread_cm[0] = 0.0


class TestStatistics:
    def test_means_the_spread_and_the_forward_jump_over_every_window(self):
        distance = straight_track(5)
        posterior = np.zeros((3, 5))
        posterior[[0, 1, 2], [0, 1, 3]] = 1.0  # centers 0, 1, 3: jumps 2, 4, then 4

        spread, jump = shuffle_statistics(posterior, distance)

        assert spread == pytest.approx(0.0)
        assert jump == pytest.approx((2 + 4 + 4) / 3)

    def test_zero_jumps_are_set_aside_as_the_reference_does(self):
        """``jumps(jumps==0) = nan; nanmean(jumps)``: a window whose center
        repeats the next one's drops out of the jump mean."""
        posterior = np.zeros((4, 5))
        posterior[[0, 1, 2, 3], [0, 0, 2, 2]] = 1.0  # jumps 0, 4, 0, then 0

        _, jump = shuffle_statistics(posterior, straight_track(5))

        assert jump == pytest.approx(4.0)

    def test_a_stationary_candidate_has_no_jump_mean(self):
        posterior = np.zeros((4, 5))
        posterior[:, 2] = 1.0

        _, jump = shuffle_statistics(posterior, straight_track(5))

        assert np.isnan(jump)

    def test_an_unusable_window_is_skipped_as_nanmean_skips_it(self):
        posterior = np.zeros((3, 5))
        posterior[[0, 2], [1, 2]] = 1.0
        posterior[1] = np.nan

        spread, jump = shuffle_statistics(posterior, straight_track(5))

        assert spread == pytest.approx(0.0)
        # window 0's successor has no center, window 1 has none, window 2 repeats
        # window 1's undefined jump: no jump is defined
        assert np.isnan(jump)


class TestPermutation:
    def test_is_a_permutation_of_the_encoding_rows(self):
        permutation = encoding_permutation(SEED, candidate=3, draw=1, electrode=0, n=50)

        assert sorted(permutation.tolist()) == list(range(50))

    def test_the_same_key_gives_the_same_permutation(self):
        first = encoding_permutation(SEED, candidate=3, draw=1, electrode=0, n=50)
        again = encoding_permutation(SEED, candidate=3, draw=1, electrode=0, n=50)

        np.testing.assert_array_equal(first, again)

    @pytest.mark.parametrize(
        "change",
        [
            {"candidate": 4},
            {"draw": 2},
            {"electrode": 1},
        ],
    )
    def test_each_part_of_the_key_changes_the_permutation(self, change):
        key = {"candidate": 3, "draw": 1, "electrode": 0} | change
        base = encoding_permutation(SEED, candidate=3, draw=1, electrode=0, n=50)

        other = encoding_permutation(SEED, n=50, **key)

        assert not np.array_equal(base, other)

    @pytest.mark.parametrize(
        "seed",
        [
            ShuffleSeed(run=1, session="simulated_epoch"),
            ShuffleSeed(run=20260922, session="other"),
        ],
    )
    def test_the_run_and_the_session_change_the_permutation(self, seed):
        base = encoding_permutation(SEED, candidate=3, draw=1, electrode=0, n=50)

        other = encoding_permutation(seed, candidate=3, draw=1, electrode=0, n=50)

        assert not np.array_equal(base, other)

    def test_draw_zero_is_reserved_for_the_observed_decode(self):
        with pytest.raises(ValueError, match="draw"):
            encoding_permutation(SEED, candidate=3, draw=0, electrode=0, n=50)

    def test_the_session_must_be_a_string(self):
        with pytest.raises(ValueError, match="session"):
            ShuffleSeed(run=1, session=5)  # type: ignore[arg-type]


class TestShuffledLikelihood:
    """The re-decode must be the library's decode, permuted and nothing else."""

    WINDOWS = (10, 70)

    def test_the_identity_reproduces_the_epoch_decode(self, epoch):
        first, last = self.WINDOWS
        whole = clusterless_epoch_log_likelihood(
            epoch.time,
            epoch.linear_position,
            epoch.spike_times,
            epoch.spike_waveform_features,
            epoch.encoding_model,
            CONFIG,
        )
        expected = aggregate_windows(
            whole.log_likelihood, CONFIG.window_n_samples, CONFIG.stride_n_samples
        )[first : last + 1]

        windows = candidate_window_log_likelihood(
            first,
            last,
            epoch.time,
            epoch.spike_times,
            epoch.spike_waveform_features,
            epoch.encoding_model,
            CONFIG,
        )

        assert windows.shape == expected.shape
        np.testing.assert_allclose(windows, expected, rtol=1e-5, atol=1e-3)

    def test_a_permutation_equals_the_library_on_permuted_positions(self, epoch):
        """The shuffle permutes marks by the inverse, which is the same set of
        (mark, position) pairs; decoding with explicitly permuted positions
        through the library must therefore agree."""
        first, last = self.WINDOWS
        permutations = [
            np.random.default_rng(electrode).permutation(positions.shape[0])
            for electrode, positions in enumerate(
                epoch.encoding_model["encoding_positions"]
            )
        ]
        permuted_model = copy.copy(epoch.encoding_model)
        permuted_model["encoding_positions"] = [
            np.asarray(positions)[permutation]
            for positions, permutation in zip(
                epoch.encoding_model["encoding_positions"], permutations, strict=True
            )
        ]
        library = clusterless_epoch_log_likelihood(
            epoch.time,
            epoch.linear_position,
            epoch.spike_times,
            epoch.spike_waveform_features,
            permuted_model,
            CONFIG,
        )
        expected = aggregate_windows(
            library.log_likelihood, CONFIG.window_n_samples, CONFIG.stride_n_samples
        )[first : last + 1]

        windows = candidate_window_log_likelihood(
            first,
            last,
            epoch.time,
            epoch.spike_times,
            epoch.spike_waveform_features,
            epoch.encoding_model,
            CONFIG,
            permutations=permutations,
        )

        np.testing.assert_allclose(windows, expected, rtol=1e-5, atol=1e-3)

    def test_a_permutation_changes_the_decode(self, tetrode):
        first, last = self.WINDOWS
        permutations = [
            np.random.default_rng(electrode).permutation(positions.shape[0])
            for electrode, positions in enumerate(
                tetrode.encoding_model["encoding_positions"]
            )
        ]
        args = (
            first,
            last,
            tetrode.time,
            tetrode.spike_times,
            tetrode.spike_waveform_features,
            tetrode.encoding_model,
            CONFIG,
        )

        observed = candidate_window_log_likelihood(*args)
        shuffled = candidate_window_log_likelihood(*args, permutations=permutations)

        assert not np.allclose(observed, shuffled)

    def test_leaves_the_encoding_model_and_the_spikes_alone(self, epoch):
        positions = [np.array(p) for p in epoch.encoding_model["encoding_positions"]]
        marks = [
            np.array(m)
            for m in epoch.encoding_model["encoding_spike_waveform_features"]
        ]
        features = [np.array(f) for f in epoch.spike_waveform_features]
        permutations = [
            np.random.default_rng(0).permutation(p.shape[0]) for p in positions
        ]

        candidate_window_log_likelihood(
            *self.WINDOWS,
            epoch.time,
            epoch.spike_times,
            epoch.spike_waveform_features,
            epoch.encoding_model,
            CONFIG,
            permutations=permutations,
        )

        for before, after in zip(
            positions, epoch.encoding_model["encoding_positions"], strict=True
        ):
            np.testing.assert_array_equal(before, np.asarray(after))
        for before, after in zip(
            marks, epoch.encoding_model["encoding_spike_waveform_features"], strict=True
        ):
            np.testing.assert_array_equal(before, np.asarray(after))
        for before, after in zip(features, epoch.spike_waveform_features, strict=True):
            np.testing.assert_array_equal(before, after)

    def test_a_window_range_past_the_decodable_rows_is_refused(self, epoch):
        n_windows = (
            1
            + (epoch.time.size - 1 - CONFIG.window_n_samples) // CONFIG.stride_n_samples
        )
        with pytest.raises(ValueError, match="window"):
            candidate_window_log_likelihood(
                0,
                n_windows,
                epoch.time,
                epoch.spike_times,
                epoch.spike_waveform_features,
                epoch.encoding_model,
                CONFIG,
            )

    def test_one_permutation_per_electrode_is_required(self, epoch):
        with pytest.raises(ValueError, match="electrode"):
            candidate_window_log_likelihood(
                *self.WINDOWS,
                epoch.time,
                epoch.spike_times,
                epoch.spike_waveform_features,
                epoch.encoding_model,
                CONFIG,
                permutations=[np.arange(3)],
            )


class TestShuffleCandidate:
    WINDOWS = MOVING_WINDOWS

    def test_the_decode_follows_the_animal(self, tetrode):
        """The premise of the next test, checked rather than assumed."""
        first, last = self.WINDOWS
        windows = candidate_window_log_likelihood(
            first,
            last,
            tetrode.time,
            tetrode.spike_times,
            tetrode.spike_waveform_features,
            tetrode.encoding_model,
            CONFIG,
        )
        center, _ = window_center_and_spread(
            normalize_windows(windows), tetrode.bin_distance_cm
        )
        environment = tetrode.encoding_model["environment"]
        interior = np.asarray(environment.is_track_interior_).ravel()
        decoded_cm = np.asarray(environment.place_bin_centers_).ravel()[interior][
            center
        ]
        midpoint_s = (
            tetrode.time[0]
            + CONFIG.window_duration_s / 2
            + CONFIG.stride_s * np.arange(first, last + 1)
        )
        animal_cm = np.interp(midpoint_s, tetrode.time, tetrode.linear_position)

        assert np.median(np.abs(decoded_cm - animal_cm)) <= 4.0  # one bin
        assert np.ptp(decoded_cm) > 50.0  # and it travels

    def test_a_real_trajectory_passes_both_halves(self, tetrode):
        """Positive control, under the reference's conventions.

        Breaking which mark goes with which place leaves a broad posterior,
        so the spread half separates by a wide margin. The center of a broad
        posterior sits near the middle of the track and barely moves, so the
        shuffled jump means are small, and the trajectory's own mean jump
        equals the smallest of them; because the reference counts only draws
        strictly below the observed value, that tie does not count against
        it. Under the port's earlier tie rule the same candidate failed.
        """
        result = shuffle_candidate(
            *self.WINDOWS,
            tetrode.time,
            tetrode.spike_times,
            tetrode.spike_waveform_features,
            tetrode.encoding_model,
            tetrode.bin_distance_cm,
            CONFIG,
            seed=SEED,
            n_shuffles=50,
        )

        assert result.testable
        assert result.observed_spread_cm < result.null_spread_cm.min()
        assert result.p_spread <= 0.05
        assert result.observed_jump_cm <= result.null_jump_cm.min()
        assert result.p_jump <= 0.05
        assert result.passes

    def test_the_observed_statistics_are_the_unshuffled_decode(self, tetrode):
        first, last = self.WINDOWS
        windows = candidate_window_log_likelihood(
            first,
            last,
            tetrode.time,
            tetrode.spike_times,
            tetrode.spike_waveform_features,
            tetrode.encoding_model,
            CONFIG,
        )
        spread, jump = shuffle_statistics(
            normalize_windows(windows), tetrode.bin_distance_cm
        )

        result = shuffle_candidate(
            first,
            last,
            tetrode.time,
            tetrode.spike_times,
            tetrode.spike_waveform_features,
            tetrode.encoding_model,
            tetrode.bin_distance_cm,
            CONFIG,
            seed=SEED,
            n_shuffles=MINIMUM_N_SHUFFLES,
        )

        assert result.observed_spread_cm == pytest.approx(spread)
        assert result.observed_jump_cm == pytest.approx(jump)

    def test_is_reproducible_from_its_seed(self, tetrode):
        args = (
            *self.WINDOWS,
            tetrode.time,
            tetrode.spike_times,
            tetrode.spike_waveform_features,
            tetrode.encoding_model,
            tetrode.bin_distance_cm,
            CONFIG,
        )

        first = shuffle_candidate(*args, seed=SEED, n_shuffles=MINIMUM_N_SHUFFLES)
        again = shuffle_candidate(*args, seed=SEED, n_shuffles=MINIMUM_N_SHUFFLES)
        other = shuffle_candidate(
            *args,
            seed=ShuffleSeed(run=1, session=SEED.session),
            n_shuffles=MINIMUM_N_SHUFFLES,
        )

        np.testing.assert_array_equal(first.null_spread_cm, again.null_spread_cm)
        np.testing.assert_array_equal(first.null_jump_cm, again.null_jump_cm)
        assert not np.array_equal(first.null_spread_cm, other.null_spread_cm)

    def test_a_draw_does_not_depend_on_how_many_draws_there_are(self, tetrode):
        """Draw k is seeded by k alone, so a pilot's draws are a prefix of the
        full run's."""
        args = (
            *self.WINDOWS,
            tetrode.time,
            tetrode.spike_times,
            tetrode.spike_waveform_features,
            tetrode.encoding_model,
            tetrode.bin_distance_cm,
            CONFIG,
        )

        short = shuffle_candidate(*args, seed=SEED, n_shuffles=MINIMUM_N_SHUFFLES)
        longer = shuffle_candidate(*args, seed=SEED, n_shuffles=MINIMUM_N_SHUFFLES + 3)

        np.testing.assert_array_equal(
            short.null_spread_cm, longer.null_spread_cm[:MINIMUM_N_SHUFFLES]
        )

    def test_any_single_draw_can_be_regenerated_from_its_key(self, tetrode):
        """Draw k of the full run equals a fresh re-decode under draw k's
        permutations alone, so the draw loop shares no state between draws."""
        first, last = self.WINDOWS
        result = shuffle_candidate(
            first,
            last,
            tetrode.time,
            tetrode.spike_times,
            tetrode.spike_waveform_features,
            tetrode.encoding_model,
            tetrode.bin_distance_cm,
            CONFIG,
            seed=SEED,
            n_shuffles=MINIMUM_N_SHUFFLES,
        )
        for draw in (1, 7, MINIMUM_N_SHUFFLES):
            permutations = [
                encoding_permutation(
                    SEED,
                    candidate=first,
                    draw=draw,
                    electrode=electrode,
                    n=np.asarray(positions).shape[0],
                )
                for electrode, positions in enumerate(
                    tetrode.encoding_model["encoding_positions"]
                )
            ]
            windows = candidate_window_log_likelihood(
                first,
                last,
                tetrode.time,
                tetrode.spike_times,
                tetrode.spike_waveform_features,
                tetrode.encoding_model,
                CONFIG,
                permutations=permutations,
            )
            spread, jump = shuffle_statistics(
                normalize_windows(windows), tetrode.bin_distance_cm
            )

            assert result.null_spread_cm[draw - 1] == pytest.approx(spread, rel=1e-6)
            assert result.null_jump_cm[draw - 1] == pytest.approx(jump, rel=1e-6)

    def test_a_trajectory_with_no_place_information_does_not_pass(self, tetrode):
        """Negative control: scramble which mark goes with which spike time,
        so the decode carries no place at all, and the published definition
        must not call the same windows a replay."""
        rng = np.random.default_rng(5)
        scrambled = [
            rng.permutation(f, axis=0) for f in tetrode.spike_waveform_features
        ]

        result = shuffle_candidate(
            *self.WINDOWS,
            tetrode.time,
            tetrode.spike_times,
            scrambled,
            tetrode.encoding_model,
            tetrode.bin_distance_cm,
            CONFIG,
            seed=SEED,
            n_shuffles=50,
        )

        assert not result.passes

    def test_needs_enough_draws_for_p_at_or_below_005_to_be_possible(self, tetrode):
        """With n draws the smallest p is 1 / (n + 1); below 19 draws no
        candidate could pass, and a run that passed nothing would look like
        a result."""
        assert MINIMUM_N_SHUFFLES == 19
        with pytest.raises(ValueError, match="n_shuffles"):
            shuffle_candidate(
                *self.WINDOWS,
                tetrode.time,
                tetrode.spike_times,
                tetrode.spike_waveform_features,
                tetrode.encoding_model,
                tetrode.bin_distance_cm,
                CONFIG,
                seed=SEED,
                n_shuffles=MINIMUM_N_SHUFFLES - 1,
            )


@pytest.fixture(scope="module")
def sorted_track() -> SortedTrack:
    return build_sorted_track()


class TestCellIdentityShuffle:
    """The reference's own shuffle, for sorted cells: swap whole place fields.

    ``rateMap(ind_rand, :)`` in ``load_replayEvents_cellIDshuffle_sameEvents.m``.
    Each cell's spikes are decoded with another cell's field; every field is
    kept intact, and the fields summed over cells, which is the term for the
    spikes that did not arrive, is unchanged.
    """

    WINDOWS = (60, 260)

    def args(self, sorted_track):
        return (
            *self.WINDOWS,
            sorted_track.time,
            sorted_track.spike_times,
            sorted_track.encoding_model,
            CONFIG,
        )

    def test_the_identity_reproduces_the_epoch_decode(self, sorted_track):
        first, last = self.WINDOWS
        whole = sorted_spikes_epoch_log_likelihood(
            sorted_track.time,
            sorted_track.linear_position,
            sorted_track.spike_times,
            sorted_track.encoding_model,
            CONFIG,
        )
        expected = aggregate_windows(
            whole.log_likelihood, CONFIG.window_n_samples, CONFIG.stride_n_samples
        )[first : last + 1]

        windows = sorted_candidate_window_log_likelihood(*self.args(sorted_track))

        np.testing.assert_allclose(windows, expected, rtol=1e-5, atol=1e-3)

    def test_a_permutation_equals_the_library_on_swapped_fields(self, sorted_track):
        first, last = self.WINDOWS
        n_cells = len(sorted_track.spike_times)
        permutation = np.random.default_rng(3).permutation(n_cells)
        swapped = copy.copy(sorted_track.encoding_model)
        swapped["place_fields"] = np.asarray(
            sorted_track.encoding_model["place_fields"]
        )[permutation]
        library = sorted_spikes_epoch_log_likelihood(
            sorted_track.time,
            sorted_track.linear_position,
            sorted_track.spike_times,
            swapped,
            CONFIG,
        )
        expected = aggregate_windows(
            library.log_likelihood, CONFIG.window_n_samples, CONFIG.stride_n_samples
        )[first : last + 1]

        windows = sorted_candidate_window_log_likelihood(
            *self.args(sorted_track), permutation=permutation
        )

        np.testing.assert_allclose(windows, expected, rtol=1e-5, atol=1e-3)

    def test_spike_trains_must_match_the_cells_one_for_one(self, sorted_track):
        """The library zips them with strict=False and would decode quietly."""
        first, last = self.WINDOWS
        with pytest.raises(ValueError, match="cells"):
            sorted_candidate_window_log_likelihood(
                first,
                last,
                sorted_track.time,
                sorted_track.spike_times[:-1],
                sorted_track.encoding_model,
                CONFIG,
            )

    def test_the_decode_follows_the_animal(self, sorted_track):
        """The premise of the next test, checked rather than assumed."""
        first, last = self.WINDOWS
        windows = sorted_candidate_window_log_likelihood(*self.args(sorted_track))
        center, _ = window_center_and_spread(
            normalize_windows(windows), sorted_track.bin_distance_cm
        )
        environment = sorted_track.encoding_model["environment"]
        interior = np.asarray(environment.is_track_interior_).ravel()
        decoded_cm = np.asarray(environment.place_bin_centers_).ravel()[interior][
            center
        ]
        midpoint_s = sorted_track.time[
            CONFIG.window_n_samples // 2
            + CONFIG.stride_n_samples * np.arange(first, last + 1)
        ]
        animal_cm = np.interp(
            midpoint_s, sorted_track.time, sorted_track.linear_position
        )

        assert np.median(np.abs(decoded_cm - animal_cm)) <= 2.5  # one bin
        assert np.ptp(decoded_cm) > 70.0

    def test_the_spread_half_cannot_separate_swapped_fields(self, sorted_track):
        """A property of the reference's own shuffle, recorded on file.

        Swapping whole fields leaves each shuffled posterior sharp, since each
        spike still points at one field, so the spread null sits where the
        observed spread does and that half cannot pass. The sharp, wrong
        posterior hops between fields, so the trajectory's mean non-zero jump
        sits below most draws, but on twenty tiling cells at 30 Hz not below
        enough of them, and this smooth run does not pass the published test.
        """
        result = shuffle_candidate_sorted(
            *self.args(sorted_track)[:5],
            sorted_track.bin_distance_cm,
            CONFIG,
            seed=SEED,
            n_shuffles=100,
        )

        assert result.testable
        assert result.observed_jump_cm < np.median(result.null_jump_cm)
        assert result.p_spread > 0.05
        assert not result.passes

    def test_a_cell_permutation_is_seeded_apart_from_the_electrode_ones(self):
        cells = cell_identity_permutation(SEED, candidate=3, draw=1, n=50)
        electrodes = encoding_permutation(SEED, candidate=3, draw=1, electrode=0, n=50)

        assert sorted(cells.tolist()) == list(range(50))
        assert not np.array_equal(cells, electrodes)

    def test_is_reproducible_and_a_prefix_of_a_longer_run(self, sorted_track):
        args = (
            *self.args(sorted_track)[:5],
            sorted_track.bin_distance_cm,
            CONFIG,
        )
        short = shuffle_candidate_sorted(
            *args, seed=SEED, n_shuffles=MINIMUM_N_SHUFFLES
        )
        longer = shuffle_candidate_sorted(
            *args, seed=SEED, n_shuffles=MINIMUM_N_SHUFFLES + 5
        )

        np.testing.assert_array_equal(
            short.null_jump_cm, longer.null_jump_cm[:MINIMUM_N_SHUFFLES]
        )

    def test_needs_enough_draws(self, sorted_track):
        with pytest.raises(ValueError, match="n_shuffles"):
            shuffle_candidate_sorted(
                *self.args(sorted_track)[:5],
                sorted_track.bin_distance_cm,
                CONFIG,
                seed=SEED,
                n_shuffles=MINIMUM_N_SHUFFLES - 1,
            )
