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
    def test_a_null_entirely_above_the_observed_gives_the_smallest_p(self):
        p = shuffle_p_value(1.0, np.full(100, 2.0))

        assert p == pytest.approx(1 / 101)

    def test_ties_count_against_the_candidate(self):
        """Graph centers are discrete, so a shuffled jump can equal the observed
        one exactly; a tie must not help the candidate."""
        p = shuffle_p_value(0.0, np.zeros(100))

        assert p == pytest.approx(1.0)

    def test_counts_draws_at_or_below_the_observed(self):
        null = np.array([0.5, 1.0, 1.5, 2.0, 3.0])

        assert shuffle_p_value(1.0, null) == pytest.approx((1 + 2) / 6)

    def test_a_non_finite_observed_value_is_untestable(self):
        assert np.isnan(shuffle_p_value(np.nan, np.ones(100)))

    def test_a_non_finite_null_draw_makes_it_untestable_rather_than_dropped(self):
        """Dropping the draw would shrink the denominator and could manufacture
        a pass."""
        null = np.full(100, 2.0)
        null[3] = np.inf

        assert np.isnan(shuffle_p_value(1.0, null))

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

    def test_passes_only_when_both_p_values_are_below_005(self):
        assert self.result(0, 0).passes
        assert not self.result(0, 10).passes
        assert not self.result(10, 0).passes

    def test_four_draws_at_or_below_still_pass_and_five_do_not(self):
        """5 / 101 < 0.05 < 6 / 101: at 100 draws strict and inclusive agree."""
        assert self.result(4, 4).passes
        assert not self.result(5, 0).passes

    def test_an_untestable_candidate_never_passes(self):
        result = ShuffleResult(
            observed_spread_cm=np.nan,
            observed_jump_cm=1.0,
            null_spread_cm=np.full(100, 2.0),
            null_jump_cm=np.full(100, 2.0),
        )

        assert not result.testable
        assert not result.passes

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


class TestStatistics:
    def test_means_the_spread_and_the_forward_jump_over_every_window(self):
        distance = straight_track(5)
        posterior = np.zeros((3, 5))
        posterior[[0, 1, 2], [0, 1, 3]] = 1.0  # centers 0, 1, 3: jumps 2, 4, then 4

        spread, jump = shuffle_statistics(posterior, distance)

        assert spread == pytest.approx(0.0)
        assert jump == pytest.approx((2 + 4 + 4) / 3)

    def test_a_stationary_candidate_keeps_its_zero_jumps(self):
        """The reference drops zero jumps; this port keeps them, so a still
        representation has a mean jump of zero rather than none."""
        posterior = np.zeros((4, 5))
        posterior[:, 2] = 1.0

        _, jump = shuffle_statistics(posterior, straight_track(5))

        assert jump == 0.0

    def test_one_unusable_window_makes_the_statistics_non_finite(self):
        posterior = np.zeros((3, 5))
        posterior[[0, 2], [1, 2]] = 1.0
        posterior[1] = np.nan

        spread, jump = shuffle_statistics(posterior, straight_track(5))

        assert not np.isfinite(spread)
        assert not np.isfinite(jump)


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

    def test_the_spread_test_separates_and_the_jump_test_cannot(self, tetrode):
        """A property of the method, recorded so it is on file for the pilot.

        Breaking which mark goes with which place leaves a broad posterior,
        and the center of a broad posterior sits near the middle of the track
        and barely moves. The shuffled mean jump is then smaller than that of
        a trajectory that really moves, so "observed jump below the shuffled"
        favours a decode that stands still. Measured here at 80 Hz; at 40 Hz
        the null's jumps are smaller still. Whether the reference's cell-ID
        shuffle behaves the same way on its own data is not known.
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
        assert result.p_spread < 0.05
        assert result.p_jump > 0.05
        assert not result.passes

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

    def test_needs_enough_draws_for_p_below_005_to_be_possible(self, tetrode):
        """With n draws the smallest p is 1 / (n + 1); below 20 draws no
        candidate could pass, and a run that passed nothing would look like
        a result."""
        assert MINIMUM_N_SHUFFLES == 20
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

    def test_the_jump_test_separates_and_the_spread_test_cannot(self, sorted_track):
        """The mirror image of the encoding-pair shuffle, recorded on file.

        Swapping whole fields leaves each shuffled posterior sharp, since each
        spike still points at one field, so the spread null sits where the
        observed spread does. The sharp, wrong posterior hops between fields,
        so the jump null lies above a trajectory that moves smoothly. Measured
        here at 30 Hz peak rate; the same at 15 Hz.
        """
        result = shuffle_candidate_sorted(
            *self.args(sorted_track)[:5],
            sorted_track.bin_distance_cm,
            CONFIG,
            seed=SEED,
            n_shuffles=100,
        )

        assert result.testable
        assert result.observed_jump_cm < result.null_jump_cm.min()
        assert result.p_jump < 0.05
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
