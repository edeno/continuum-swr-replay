"""Per-window statistics for the memoryless sequence detector."""

import warnings

import numpy as np
import pytest

from continuum_swr_replay.analysis.detection.memoryless_sequence.window_statistics import (
    NO_CENTER,
    aggregate_windows,
    dispersion_cm,
    forward_jumps_cm,
    normalize_windows,
    window_center_and_spread,
)


def straight_track(n_bins: int, bin_size: float = 2.0) -> np.ndarray:
    """Distances along a straight segment of evenly spaced bins."""
    centers = np.arange(n_bins) * bin_size
    return np.abs(centers[:, None] - centers[None, :])


class TestAggregateWindows:
    def test_sums_the_rows_of_each_window(self):
        log_likelihood = np.arange(12, dtype=float).reshape(6, 2)

        windows = aggregate_windows(
            log_likelihood, window_n_samples=2, stride_n_samples=2
        )

        # rows (0,1), (2,3), (4,5)
        np.testing.assert_allclose(windows, [[2, 4], [10, 12], [18, 20]])

    def test_overlapping_windows_reuse_rows(self):
        log_likelihood = np.arange(10, dtype=float).reshape(5, 2)

        windows = aggregate_windows(
            log_likelihood, window_n_samples=3, stride_n_samples=1
        )

        assert windows.shape == (3, 2)
        np.testing.assert_allclose(windows[0], log_likelihood[:3].sum(axis=0))
        np.testing.assert_allclose(windows[1], log_likelihood[1:4].sum(axis=0))

    def test_a_partial_window_at_the_end_is_dropped(self):
        """Every window must have its full support."""
        log_likelihood = np.zeros((7, 2))

        windows = aggregate_windows(
            log_likelihood, window_n_samples=4, stride_n_samples=2
        )

        assert windows.shape == (2, 2)  # rows 0-3 and 2-5; 4-7 does not exist

    def test_too_few_rows_gives_no_windows_with_the_right_shape(self):
        windows = aggregate_windows(
            np.zeros((3, 5)), window_n_samples=4, stride_n_samples=2
        )

        assert windows.shape == (0, 5)

    def test_sums_in_float64(self):
        """float32 cumulative sums lose precision over a long epoch.

        The window must equal the float64 sum of its own float32 rows, not the
        float32 running total, which drifts as the epoch gets longer.
        """
        log_likelihood = np.full((40_000, 2), 0.1, dtype=np.float32)

        windows = aggregate_windows(
            log_likelihood, window_n_samples=40, stride_n_samples=2
        )

        assert windows.dtype == np.float64
        exact = log_likelihood[-40:].astype(np.float64).sum(axis=0)
        np.testing.assert_allclose(windows[-1], exact, rtol=0, atol=0)

    def test_a_window_holding_a_missing_row_is_missing(self):
        """Every window that holds a missing row is NaN, so it has no posterior;
        the others are their rows' sums, whatever the missing row held."""
        log_likelihood = np.arange(20, dtype=float).reshape(10, 2)
        missing = np.zeros(10, dtype=bool)
        missing[5] = True
        garbled = log_likelihood.copy()
        garbled[5] = [np.nan, np.inf]

        windows = aggregate_windows(
            garbled, window_n_samples=4, stride_n_samples=2, missing=missing
        )

        # rows 0-3, 2-5, 4-7 and 6-9: the middle two hold row 5
        assert np.isnan(windows[[1, 2]]).all()
        np.testing.assert_array_equal(
            windows[[0, 3]],
            [log_likelihood[0:4].sum(axis=0), log_likelihood[6:10].sum(axis=0)],
        )

    def test_no_missing_row_changes_nothing(self):
        log_likelihood = np.random.default_rng(0).normal(-50.0, 20.0, (500, 7))

        np.testing.assert_array_equal(
            aggregate_windows(log_likelihood, 40, 2, missing=np.zeros(500, bool)),
            aggregate_windows(log_likelihood, 40, 2),
        )

    def test_missing_rows_must_be_one_per_row(self):
        with pytest.raises(ValueError, match="missing"):
            aggregate_windows(np.zeros((10, 2)), 4, 2, missing=np.zeros(9, bool))


class TestNormalizeWindows:
    def test_each_window_sums_to_one(self):
        posterior = normalize_windows(np.array([[0.0, 1.0, 2.0], [10.0, 10.0, 10.0]]))

        np.testing.assert_allclose(posterior.sum(axis=1), 1.0)

    def test_a_uniform_prior_makes_equal_log_likelihoods_equal(self):
        posterior = normalize_windows(np.zeros((1, 4)))

        np.testing.assert_allclose(posterior, 0.25)

    def test_normalization_is_shift_invariant(self):
        """Adding a constant to a window changes nothing; it is a likelihood."""
        window = np.array([[0.0, 1.0, 2.0]])

        np.testing.assert_allclose(
            normalize_windows(window), normalize_windows(window + 500.0)
        )

    def test_large_log_likelihoods_do_not_overflow(self):
        posterior = normalize_windows(np.array([[1000.0, 999.0]]))

        assert np.isfinite(posterior).all()
        assert posterior[0, 0] > posterior[0, 1]

    def test_a_bin_of_zero_likelihood_gets_zero_mass(self):
        posterior = normalize_windows(np.array([[0.0, -np.inf, 0.0]]))

        np.testing.assert_allclose(posterior, [[0.5, 0.0, 0.5]])

    @pytest.mark.parametrize(
        "row",
        [
            [np.nan, np.nan, np.nan],
            [-np.inf, -np.inf, -np.inf],
            [0.0, np.nan, 1.0],  # one NaN spoils the window
            [0.0, np.inf, 1.0],  # so does an infinite likelihood
        ],
    )
    def test_a_window_the_decoder_could_not_evaluate_is_all_nan(self, row):
        """Not uniform: uninformative and unevaluable are different things."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            posterior = normalize_windows(np.array([row, [0.0, 1.0, 2.0]]))

        assert np.isnan(posterior[0]).all()
        np.testing.assert_allclose(posterior[1].sum(), 1.0)


class TestCenterAndSpread:
    def test_the_answer_does_not_depend_on_the_block_size(self, monkeypatch):
        """The search runs in blocks of windows to bound memory."""
        from continuum_swr_replay.analysis.detection.memoryless_sequence import (
            window_statistics,
        )

        rng = np.random.default_rng(0)
        posterior = rng.random((300, 12))
        posterior /= posterior.sum(axis=1, keepdims=True)
        posterior[[5, 77], :] = np.nan
        distance = straight_track(12)
        whole = window_center_and_spread(posterior, distance)

        monkeypatch.setattr(window_statistics, "_WINDOW_CHUNK", 7)
        blocked = window_center_and_spread(posterior, distance)

        np.testing.assert_array_equal(blocked[0], whole[0])
        np.testing.assert_allclose(blocked[1], whole[1], rtol=1e-12)

    def test_a_posterior_on_one_bin_has_that_center_and_no_spread(self):
        distance = straight_track(5)
        posterior = np.array([[0.0, 0.0, 1.0, 0.0, 0.0]])

        center, spread = window_center_and_spread(posterior, distance)

        assert center.tolist() == [2]
        np.testing.assert_allclose(spread, [0.0])

    def test_spread_is_the_standard_deviation_on_a_straight_segment(self):
        """Uniform over three bins 2 cm apart: SD is sqrt(8/3)."""
        distance = straight_track(3)
        posterior = np.full((1, 3), 1 / 3)

        center, spread = window_center_and_spread(posterior, distance)

        assert center.tolist() == [1]
        np.testing.assert_allclose(spread, [np.sqrt(8 / 3)])

    def test_the_center_minimizes_expected_squared_distance(self):
        rng = np.random.default_rng(0)
        distance = straight_track(9)
        posterior = rng.dirichlet(np.ones(9), size=20)

        center, spread = window_center_and_spread(posterior, distance)

        for window in range(20):
            costs = (posterior[window] * distance**2).sum(axis=1)
            assert center[window] == int(np.argmin(costs))
            np.testing.assert_allclose(spread[window], np.sqrt(costs.min()))

    def test_ties_take_the_lower_bin_and_ignore_time(self):
        """Two bins equally good: bin order decides, deterministically."""
        distance = straight_track(2, bin_size=10.0)
        posterior = np.array([[0.5, 0.5], [0.5, 0.5]])

        center, _ = window_center_and_spread(posterior, distance)

        assert center.tolist() == [0, 0]

    def test_mass_off_the_track_cannot_move_the_center(self):
        """Junction placeholder bins are excluded before this is called."""
        distance = straight_track(4)
        on_track = np.array([True, True, True, False])
        posterior = np.array([[0.1, 0.1, 0.1, 0.7]])

        center, _ = window_center_and_spread(
            posterior[:, on_track], distance[np.ix_(on_track, on_track)]
        )

        assert center.tolist() == [1]

    def test_an_all_nan_window_gives_nan(self):
        distance = straight_track(3)
        posterior = np.full((1, 3), np.nan)

        center, spread = window_center_and_spread(posterior, distance)

        assert center[0] == -1
        assert np.isnan(spread[0])

    def test_the_center_approaches_the_mean_as_the_grid_refines(self):
        """Bin restriction approximates the center of mass.

        The center is quantized to a bin, so it sits within half a bin of the
        center of mass. Once the grid is fine enough the remaining error is
        wherever the grid points happen to fall, not the spacing, so the
        property is the bound and not a monotone decrease.
        """
        errors, half_bins = [], []
        for n_bins in (11, 101, 1001):
            bin_size = 10.0 / (n_bins - 1)
            distance = straight_track(n_bins, bin_size)
            centers = np.arange(n_bins) * bin_size
            weights = np.exp(-0.5 * ((centers - 3.7) / 1.0) ** 2)
            posterior = (weights / weights.sum())[None, :]
            index, _ = window_center_and_spread(posterior, distance)
            errors.append(abs(centers[index[0]] - (posterior[0] * centers).sum()))
            half_bins.append(bin_size / 2)

        for error, half_bin in zip(errors, half_bins, strict=True):
            assert error <= half_bin + 1e-9
        assert errors[-1] < errors[0] / 100


class TestForwardJumps:
    def test_window_k_carries_the_jump_to_k_plus_one(self):
        distance = straight_track(5)
        center = np.array([0, 1, 3])

        jumps = forward_jumps_cm(center, distance)

        # bins are 2 cm apart: 0->1 is 2 cm, 1->3 is 4 cm
        np.testing.assert_allclose(jumps[:2], [2.0, 4.0])

    def test_the_last_window_repeats_the_previous_jump(self):
        distance = straight_track(5)
        center = np.array([0, 1, 3])

        jumps = forward_jumps_cm(center, distance)

        assert jumps[-1] == jumps[-2]

    def test_a_window_whose_successor_is_missing_is_rejected(self):
        """The speed gate blanks centers before jumps are taken."""
        distance = straight_track(5)
        center = np.array([0, 1, -1, 3])

        jumps = forward_jumps_cm(center, distance)

        assert np.isnan(jumps[1])  # its successor has no center
        assert np.isnan(jumps[2])  # it has no center itself

    def test_a_single_window_has_no_forward_jump(self):
        jumps = forward_jumps_cm(np.array([2]), straight_track(5))

        assert jumps.shape == (1,)
        assert np.isnan(jumps[0])

    def test_no_windows_gives_an_empty_float_array(self):
        jumps = forward_jumps_cm(np.array([], dtype=int), straight_track(5))

        assert jumps.shape == (0,)
        assert jumps.dtype == float


class TestDispersion:
    def test_dispersion_is_the_rms_distance_from_the_center_of_the_path(self):
        distance = straight_track(5)

        value = dispersion_cm(np.array([0, 4]), distance)

        # bins 0 and 4 are 8 cm apart; either endpoint minimizes at 4 cm
        np.testing.assert_allclose(value, 4.0)

    def test_centers_on_disconnected_pieces_of_track_have_no_dispersion(self):
        """Not inf, which the replay criterion would accept."""
        distance = straight_track(4)
        distance[:2, 2:] = np.inf
        distance[2:, :2] = np.inf

        assert np.isnan(dispersion_cm(np.array([0, 3]), distance))
        # bins 0 and 1 are 2 cm apart: RMS distance to either endpoint is sqrt(2)
        np.testing.assert_allclose(
            dispersion_cm(np.array([0, 1]), distance), np.sqrt(2.0)
        )

    def test_a_stationary_sequence_has_no_dispersion(self):
        value = dispersion_cm(np.array([2, 2, 2]), straight_track(5))

        np.testing.assert_allclose(value, 0.0)

    def test_missing_centers_are_ignored(self):
        """Rejected windows inside a merged gap do not join the trajectory."""
        distance = straight_track(5)

        with_gap = dispersion_cm(np.array([0, -1, 4]), distance)
        without = dispersion_cm(np.array([0, 4]), distance)

        np.testing.assert_allclose(with_gap, without)

    def test_no_finite_centers_gives_nan(self):
        assert np.isnan(dispersion_cm(np.array([-1, -1]), straight_track(5)))


class TestUnreachableBins:
    """A track graph in pieces puts inf in the distance matrix."""

    def disconnected(self) -> np.ndarray:
        """Two arms of three bins each, with no path between them."""
        distance = np.full((6, 6), np.inf)
        for block in (slice(0, 3), slice(3, 6)):
            centers = np.arange(3) * 2.0
            distance[block, block] = np.abs(centers[:, None] - centers[None, :])
        return distance

    def test_a_window_spanning_unreachable_bins_gets_no_center(self):
        """0 * inf is nan, so the cost row is all nan and argmin would pick
        the first nan rather than report no center."""
        posterior = np.zeros((1, 6))
        posterior[0, [0, 4]] = 0.5

        center, spread = window_center_and_spread(posterior, self.disconnected())

        assert center.tolist() == [NO_CENTER]
        assert np.isnan(spread[0])

    def test_a_window_inside_one_piece_still_works(self):
        posterior = np.zeros((1, 6))
        posterior[0, [0, 2]] = 0.5

        center, spread = window_center_and_spread(posterior, self.disconnected())

        assert center[0] == 1
        assert spread[0] == pytest.approx(2.0)

    def test_a_missing_center_leaves_its_neighbor_without_a_jump(self):
        """Otherwise the preceding window measures a jump to bin 0."""
        posterior = np.zeros((2, 6))
        posterior[0, 1] = 1.0
        posterior[1, [0, 4]] = 0.5

        center, _ = window_center_and_spread(posterior, self.disconnected())
        jumps = forward_jumps_cm(center, self.disconnected())

        assert center.tolist() == [1, NO_CENTER]
        assert np.isnan(jumps).all()
