"""Prior-excluded log-likelihood on the native grid, and its additivity.

The fixture fits a real clusterless KDE encoding model from ``non_local_detector``
on simulated spikes over a small Y-shaped track, so the tests run against the
library's own conventions rather than a stand-in for them.
"""

import copy
from dataclasses import replace

import joblib
import numpy as np
import pytest
from non_local_detector.likelihoods.clusterless_kde import (
    block_estimate_log_joint_mark_intensity,
    kde_distance,
)

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.likelihood import (
    CLUSTERLESS_LIKELIHOOD_KEYS,
    ENCODING_KEY,
    EpochLikelihood,
    _drop_the_unreachable_row,
    check_marks_match_encoding,
    clusterless_epoch_log_likelihood,
    interior_bin_distances_cm,
    load_clusterless_encoding_model,
    sorted_spikes_epoch_log_likelihood,
    validate_time_grid,
    window_midpoints_s,
    with_reference_rate_floor,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.statistics import (
    aggregate_windows,
)
from tests.unit._memoryless_clusterless_epoch import (
    BOUNDARY_SAMPLES,
    N_FEATURES,
    SAMPLING_FREQUENCY,
    SimulatedEpoch,
    _add_spikes_at,
    _StandInDetector,
    build_simulated_epoch,
    build_sorted_track,
)


@pytest.fixture(scope="module")
def epoch(tmp_path_factory) -> SimulatedEpoch:
    return build_simulated_epoch(tmp_path_factory.mktemp("decode"))


@pytest.fixture(scope="module")
def decoded(epoch) -> tuple[np.ndarray, np.ndarray]:
    """The epoch's log-likelihood and its time axis."""
    result = clusterless_epoch_log_likelihood(
        epoch.time,
        epoch.linear_position,
        epoch.spike_times,
        epoch.spike_waveform_features,
        load_clusterless_encoding_model(epoch.detector_path),
    )
    return result.time, result.log_likelihood


@pytest.fixture(scope="module")
def position_distances(epoch) -> list:
    """Encoding-spike to bin kernel distances, which do not depend on a window."""
    interior = np.asarray(epoch.environment.is_track_interior_).ravel()
    bin_centers = np.asarray(epoch.environment.place_bin_centers_)[interior]
    return [
        kde_distance(
            bin_centers, encoding_positions, std=epoch.encoding_model["position_std"]
        )
        for encoding_positions in epoch.encoding_model["encoding_positions"]
    ]


def direct_window_log_likelihood(
    epoch: SimulatedEpoch,
    position_distances: list,
    start_s: float,
    end_s: float,
    include_end: bool = False,
) -> np.ndarray:
    """A marked-process sum over an explicit support.

    Independent of the library's time-bin assignment and of the sliding-window
    cumulative sum: the exposure is one ground-process contribution per sample,
    and every spike in ``[start_s, end_s)`` contributes its own
    log joint mark intensity.

    Both bounds are grid samples rather than a start plus a duration: adding
    0.080 to a sample does not land on the sample 40 rows later, and a spike on
    that boundary would then fall inside the support either way.

    ``include_end`` closes the support at its right end. Only the epoch's final
    row needs it: every other row hands a spike on its right edge to the row
    after it, and the final row has no row after it.
    """
    model = epoch.encoding_model
    n_samples = round((end_s - start_s) * SAMPLING_FREQUENCY)

    total = -n_samples * np.asarray(
        model["summed_ground_process_intensity"], dtype=np.float64
    )
    for electrode, position_distance in enumerate(position_distances):
        times = epoch.spike_times[electrode]
        in_window = (times >= start_s) & (
            (times <= end_s) if include_end else (times < end_s)
        )
        if not in_window.any():
            continue
        total += np.asarray(
            block_estimate_log_joint_mark_intensity(
                epoch.spike_waveform_features[electrode][in_window],
                model["encoding_spike_waveform_features"][electrode],
                np.full(
                    N_FEATURES, float(np.asarray(model["waveform_std"]).ravel()[0])
                ),
                model["occupancy"],
                model["mean_rates"][electrode],
                position_distance,
                model["block_size"],
            ),
            dtype=np.float64,
        ).sum(axis=0)
    return total


class TestLoadEncodingModel:
    def test_returns_the_arguments_the_likelihood_call_needs(self, epoch):
        model = load_clusterless_encoding_model(epoch.detector_path)

        assert set(CLUSTERLESS_LIKELIHOOD_KEYS) <= set(model)

    def test_a_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="detector"):
            load_clusterless_encoding_model(tmp_path / "absent_detector.pkl")

    def test_a_missing_encoding_key_names_the_keys_that_are_there(self, tmp_path):
        path = tmp_path / "other_detector.pkl"
        joblib.dump(_StandInDetector({("other", 7): {}}), path)

        with pytest.raises(KeyError, match=r"other.*7"):
            load_clusterless_encoding_model(path)

    def test_an_environment_saved_without_the_fitted_flag_is_restored(
        self, epoch, tmp_path
    ):
        """Detectors saved by older library versions carry no flag, and the
        track-distance methods refuse to run without one."""
        model = dict(epoch.encoding_model)
        environment = copy.deepcopy(model["environment"])
        del environment._is_fitted
        model["environment"] = environment
        path = tmp_path / "old_detector.pkl"
        joblib.dump(_StandInDetector({ENCODING_KEY: model}), path)

        loaded = load_clusterless_encoding_model(path)

        distances = loaded["environment"].get_distances_to_interior_bins(
            np.array([[10.0]])
        )
        assert np.isfinite(np.asarray(distances)).all()

    def test_a_per_feature_waveform_deviation_is_reduced_to_a_scalar(
        self, epoch, tmp_path
    ):
        """Electrodes differ in how many features they have, so a vector cannot
        be shared between them; the library expands a scalar per electrode."""
        model = dict(epoch.encoding_model)
        model["waveform_std"] = np.full(N_FEATURES, 24.0)
        path = tmp_path / "vector_detector.pkl"
        joblib.dump(_StandInDetector({ENCODING_KEY: model}), path)

        loaded = load_clusterless_encoding_model(path)

        assert isinstance(loaded["waveform_std"], float)
        assert loaded["waveform_std"] == 24.0

    def test_a_waveform_deviation_that_differs_between_features_raises(
        self, epoch, tmp_path
    ):
        """One bandwidth cannot stand in for several; decoding with the first
        would be a different model with no message."""
        model = dict(epoch.encoding_model)
        model["waveform_std"] = np.array([24.0, 10.0])
        path = tmp_path / "ragged_detector.pkl"
        joblib.dump(_StandInDetector({ENCODING_KEY: model}), path)

        with pytest.raises(ValueError, match="waveform_std"):
            load_clusterless_encoding_model(path)

    def test_an_environment_with_no_place_grid_raises(self, epoch, tmp_path):
        """Restoring the fitted flag must not vouch for a grid that is not there."""
        model = dict(epoch.encoding_model)
        environment = copy.deepcopy(model["environment"])
        environment.place_bin_centers_ = None  # the class default: never fitted
        model["environment"] = environment
        path = tmp_path / "unfitted_detector.pkl"
        joblib.dump(_StandInDetector({ENCODING_KEY: model}), path)

        with pytest.raises(ValueError, match="place grid"):
            load_clusterless_encoding_model(path)


class TestValidateTimeGrid:
    config = MemorylessSequenceConfig()

    def test_a_uniform_grid_at_the_stated_rate_passes(self):
        validate_time_grid(np.arange(100) / 500.0, self.config)

    def test_a_gap_raises(self):
        time = np.delete(np.arange(100) / 500.0, 50)

        with pytest.raises(ValueError, match="uniform"):
            validate_time_grid(time, self.config)

    def test_a_different_sampling_rate_raises(self):
        with pytest.raises(ValueError, match="sampled every"):
            validate_time_grid(np.arange(100) / 250.0, self.config)

    def test_a_grid_too_short_for_one_window_raises(self):
        with pytest.raises(ValueError, match="window"):
            validate_time_grid(np.arange(10) / 500.0, self.config)

    def test_a_grid_of_exactly_one_window_raises(self):
        """Its final row is dropped, leaving no complete window at all."""
        with pytest.raises(ValueError, match="final row"):
            validate_time_grid(np.arange(40) / 500.0, self.config)

    def test_a_grid_of_one_window_plus_the_dropped_row_passes(self):
        validate_time_grid(np.arange(41) / 500.0, self.config)

    def test_float_jitter_within_tolerance_passes(self):
        time = np.cumsum(np.full(100, 1 / 500.0))

        validate_time_grid(time, self.config)

    UNIX_START_S = 1581532345.123
    """A recording's clock: seconds since 1970, where a float64 step is 2.4e-7 s."""

    def test_a_grid_on_a_recording_clock_passes(self):
        """At 1.6e9 s the 2 ms spacings come out as 1.99986 or 2.00009 ms."""
        validate_time_grid(self.UNIX_START_S + np.arange(5000) / 500.0, self.config)

    def test_a_gap_on_a_recording_clock_raises(self):
        time = np.delete(self.UNIX_START_S + np.arange(5000) / 500.0, 2500)

        with pytest.raises(ValueError, match="uniform"):
            validate_time_grid(time, self.config)

    def test_a_different_rate_on_a_recording_clock_raises(self):
        with pytest.raises(ValueError, match="sampled every"):
            validate_time_grid(self.UNIX_START_S + np.arange(5000) / 250.0, self.config)

    def test_a_long_recording_whose_clock_runs_a_few_ppm_fast_passes(self):
        """The decodes' clocks run 0.4-1.7 ppm fast of 500 Hz, uniformly.

        Over a 27-minute epoch that puts the last samples most of a sample
        from a grid at exactly 500 Hz, though every row spans the same time.
        """
        rate = 500.0 * (1 + 1.7e-6)
        validate_time_grid(self.UNIX_START_S + np.arange(800_000) / rate, self.config)

    def test_a_rate_off_by_more_than_the_rate_tolerance_raises(self):
        """A uniform grid at 500.25 Hz has 80 ms windows 40 us short."""
        rate = 500.0 * (1 + 5e-4)
        with pytest.raises(ValueError, match="sampled every"):
            validate_time_grid(self.UNIX_START_S + np.arange(5000) / rate, self.config)

    def test_jitter_about_a_fast_clock_raises(self):
        rate = 500.0 * (1 + 1.7e-6)
        time = self.UNIX_START_S + np.arange(5000) / rate
        time[2500] += 0.01 / 500.0

        with pytest.raises(ValueError, match="uniform"):
            validate_time_grid(time, self.config)


class TestEpochLogLikelihood:
    def test_one_column_per_interior_bin(self, epoch, decoded):
        _, log_likelihood = decoded
        interior = np.asarray(epoch.environment.is_track_interior_).ravel()

        assert log_likelihood.shape[1] == interior.sum()
        assert log_likelihood.shape[1] < interior.size  # placeholders were dropped

    def test_junction_placeholder_bins_are_absent(self, epoch, decoded):
        """The centers that survive are exactly the interior ones."""
        _, log_likelihood = decoded
        interior = np.asarray(epoch.environment.is_track_interior_).ravel()
        centers = np.asarray(epoch.environment.place_bin_centers_).ravel()
        nodes = epoch.environment.place_bin_centers_nodes_df_
        junction_centers = np.asarray(nodes.linear_position[nodes.edge_id == -1])

        assert junction_centers.size > 0
        assert not np.isin(centers[interior], junction_centers).any()
        assert log_likelihood.shape[1] == interior.sum()

    def test_the_final_row_is_dropped(self, epoch, decoded):
        """It carries exposure but can never be assigned a spike."""
        time, log_likelihood = decoded

        assert log_likelihood.shape[0] == epoch.time.size - 1
        assert time.size == log_likelihood.shape[0]
        np.testing.assert_array_equal(time, epoch.time[:-1])

    def test_the_bin_centers_label_the_columns(self, epoch):
        """Every downstream statistic indexes bins by column, so the two must agree."""
        result = clusterless_epoch_log_likelihood(
            epoch.time,
            epoch.linear_position,
            epoch.spike_times,
            epoch.spike_waveform_features,
            load_clusterless_encoding_model(epoch.detector_path),
        )
        interior = np.asarray(epoch.environment.is_track_interior_).ravel()
        centers = np.asarray(epoch.environment.place_bin_centers_).ravel()

        assert result.place_bin_centers_cm.size == result.log_likelihood.shape[1]
        np.testing.assert_array_equal(result.place_bin_centers_cm, centers[interior])

    def test_the_columns_match_the_distance_matrix(self, epoch, decoded):
        """The statistics index the posterior and the distances with one index."""
        _, log_likelihood = decoded

        distance = interior_bin_distances_cm(epoch.environment)

        assert distance.shape[0] == log_likelihood.shape[1]

    def test_every_value_is_finite(self, decoded):
        _, log_likelihood = decoded

        assert np.isfinite(log_likelihood).all()

    def test_a_grid_that_fails_validation_raises(self, epoch):
        with pytest.raises(ValueError, match="uniform"):
            clusterless_epoch_log_likelihood(
                np.delete(epoch.time, 100),
                np.delete(epoch.linear_position, 100),
                epoch.spike_times,
                epoch.spike_waveform_features,
                load_clusterless_encoding_model(epoch.detector_path),
            )


class TestAdditivity:
    """A window sum must equal a direct marked-process sum over its support."""

    config = MemorylessSequenceConfig()

    def direct_for(self, epoch, position_distances, time, window: int) -> np.ndarray:
        """The direct sum over window ``window``, bounded by grid samples."""
        start = window * self.config.stride_n_samples
        return direct_window_log_likelihood(
            epoch,
            position_distances,
            time[start],
            epoch.time[start + self.config.window_n_samples],
        )

    def windows(self, decoded) -> np.ndarray:
        _, log_likelihood = decoded
        return aggregate_windows(
            log_likelihood,
            self.config.window_n_samples,
            self.config.stride_n_samples,
        )

    def test_an_interior_window_matches_the_direct_sum(
        self, epoch, decoded, position_distances
    ):
        time, _ = decoded
        windows = self.windows(decoded)
        window = 40

        direct = self.direct_for(epoch, position_distances, time, window)

        np.testing.assert_allclose(windows[window], direct, rtol=1e-5, atol=1e-4)

    def test_every_window_matches_the_direct_sum(
        self, epoch, decoded, position_distances
    ):
        time, _ = decoded
        windows = self.windows(decoded)
        direct = np.stack(
            [
                self.direct_for(epoch, position_distances, time, window)
                for window in range(len(windows))
            ]
        )

        np.testing.assert_allclose(windows, direct, rtol=1e-5, atol=1e-4)

    def test_the_windows_together_carry_every_spike(
        self, epoch, decoded, position_distances
    ):
        """Every row that survives trimming carries the spikes of its own interval."""
        time, log_likelihood = decoded

        whole = log_likelihood.sum(axis=0)
        direct = direct_window_log_likelihood(
            epoch, position_distances, time[0], epoch.time[-1], include_end=True
        )

        assert log_likelihood.shape[0] == round(
            (epoch.time[-1] - epoch.time[0]) * SAMPLING_FREQUENCY
        )
        np.testing.assert_allclose(whole, direct, rtol=1e-5, atol=1e-3)

    def test_a_spike_sits_exactly_on_a_window_boundary(self, epoch):
        """Otherwise the half-open support is never actually exercised."""
        spikes = epoch.spike_times[0]

        for sample in BOUNDARY_SAMPLES:
            assert (spikes == epoch.time[sample]).any()
        assert (spikes == epoch.time[-1]).any()

    def test_a_boundary_spike_belongs_to_the_later_window_only(
        self, epoch, decoded, position_distances
    ):
        """It opens the window that starts there and closes none before it."""
        time, _ = decoded
        windows = self.windows(decoded)
        sample = BOUNDARY_SAMPLES[0]
        opens = sample // self.config.stride_n_samples
        # the last window whose support ends exactly at this spike's time
        ends = opens - self.config.window_n_samples // self.config.stride_n_samples

        for window in (ends, opens):
            direct = self.direct_for(epoch, position_distances, time, window)
            np.testing.assert_allclose(windows[window], direct, rtol=1e-5, atol=1e-4)

    def test_a_spike_on_the_final_timestamp_is_carried(
        self, epoch, decoded, position_distances
    ):
        """The library folds it into the last row that survives trimming.

        That row's support is closed at its right end, unlike every other row.
        Treating it as half-open drops the spike and leaves the epoch sum short.
        """
        time, log_likelihood = decoded
        assert (epoch.spike_times[0] == epoch.time[-1]).any()
        half_open = direct_window_log_likelihood(
            epoch, position_distances, time[0], epoch.time[-1]
        )
        closed = direct_window_log_likelihood(
            epoch, position_distances, time[0], epoch.time[-1], include_end=True
        )

        assert not np.allclose(half_open, closed)
        np.testing.assert_allclose(
            log_likelihood.sum(axis=0), closed, rtol=1e-5, atol=1e-3
        )

    def test_the_last_window_of_the_epoch_matches(
        self, epoch, decoded, position_distances
    ):
        """The tail is where the library's end-row fold would show up."""
        time, _ = decoded
        windows = self.windows(decoded)

        direct = self.direct_for(epoch, position_distances, time, len(windows) - 1)

        np.testing.assert_allclose(windows[-1], direct, rtol=1e-5, atol=1e-4)

    def test_a_window_holding_the_final_row_closes_its_support(
        self, epoch, position_distances
    ):
        """Whether one does depends on how the epoch length divides the stride.

        The final row has no row after it to hand a spike on its right edge to,
        so it keeps that spike. A window containing that row inherits the
        closed end; every other window is half-open.
        """
        # one sample shorter, so the final row falls inside the last window
        time = epoch.time[:-1]
        spike_times = list(epoch.spike_times)
        features = list(epoch.spike_waveform_features)
        spike_times[1], features[1] = _add_spikes_at(
            spike_times[1], features[1], time[[-1]]
        )

        result = clusterless_epoch_log_likelihood(
            time,
            epoch.linear_position[: time.size],
            spike_times,
            features,
            load_clusterless_encoding_model(epoch.detector_path),
        )
        windows = aggregate_windows(
            result.log_likelihood,
            self.config.window_n_samples,
            self.config.stride_n_samples,
        )
        last_start = (len(windows) - 1) * self.config.stride_n_samples
        assert last_start + self.config.window_n_samples == result.time.size

        epoch_with_extra = replace(
            epoch, spike_times=spike_times, spike_waveform_features=features
        )
        window_end_s = time[last_start + self.config.window_n_samples]
        half_open = direct_window_log_likelihood(
            epoch_with_extra, position_distances, result.time[last_start], window_end_s
        )
        closed = direct_window_log_likelihood(
            epoch_with_extra,
            position_distances,
            result.time[last_start],
            window_end_s,
            include_end=True,
        )

        assert not np.allclose(half_open, closed)
        np.testing.assert_allclose(windows[-1], closed, rtol=1e-5, atol=1e-4)


class TestWindowMidpoints:
    config = MemorylessSequenceConfig()

    def test_one_midpoint_per_window(self, decoded):
        time, log_likelihood = decoded
        windows = aggregate_windows(
            log_likelihood, self.config.window_n_samples, self.config.stride_n_samples
        )

        midpoints = window_midpoints_s(time, self.config)

        assert midpoints.size == len(windows)

    def test_a_midpoint_is_half_a_window_after_its_start(self, decoded):
        time, _ = decoded

        midpoints = window_midpoints_s(time, self.config)

        np.testing.assert_allclose(midpoints[0], time[0] + 0.040)
        np.testing.assert_allclose(
            midpoints[1], time[self.config.stride_n_samples] + 0.040
        )

    def test_midpoints_are_one_stride_apart(self, decoded):
        time, _ = decoded

        midpoints = window_midpoints_s(time, self.config)

        np.testing.assert_allclose(np.diff(midpoints), self.config.stride_s, rtol=1e-9)

    def test_a_midpoint_is_a_sample_of_the_grid(self):
        """Not a sample time plus 0.040 s: on a recording's clock the two differ
        by up to 2.4e-7 s, enough for an event bound to miss the speed sample
        it sits on or to touch an SWR it does not overlap."""
        time = 1581532345.123 + np.arange(5000) / 500.0

        midpoints = window_midpoints_s(time, self.config)

        starts = np.arange(midpoints.size) * self.config.stride_n_samples
        np.testing.assert_array_equal(
            midpoints, time[starts + self.config.window_n_samples // 2]
        )

    def test_a_grid_too_short_for_one_window_gives_none(self):
        midpoints = window_midpoints_s(np.arange(20) / 500.0, self.config)

        assert midpoints.shape == (0,)


class TestEpochLikelihoodInvariants:
    def test_times_and_rows_must_agree(self):
        with pytest.raises(ValueError, match="log_likelihood is"):
            EpochLikelihood(
                time=np.arange(5) / 500.0,
                log_likelihood=np.zeros((4, 3)),
                place_bin_centers_cm=np.arange(3.0),
            )

    def test_columns_and_bins_must_agree(self):
        with pytest.raises(ValueError, match="log_likelihood is"):
            EpochLikelihood(
                time=np.arange(4) / 500.0,
                log_likelihood=np.zeros((4, 3)),
                place_bin_centers_cm=np.arange(2.0),
            )

    def test_a_non_finite_row_raises_rather_than_becoming_a_rejection(self, epoch):
        """Summed into windows, one NaN row would silently reject every window
        it touches; the decoder is meant never to produce one."""
        values = np.zeros((6, int(np.sum(epoch.environment.is_track_interior_))))
        values[2, 0] = np.nan

        with pytest.raises(ValueError, match="non-finite"):
            _drop_the_unreachable_row(
                values, np.arange(6) / 500.0, epoch.encoding_model
            )


class TestReferenceRateFloor:
    """The reference adds 1e-3 Hz to every rate map before it decodes."""

    config = MemorylessSequenceConfig(sampling_frequency=SAMPLING_FREQUENCY)

    @pytest.fixture(scope="class")
    def track(self):
        return build_sorted_track()

    def test_every_field_is_raised_by_the_floor_per_sample(self, track):
        floored = with_reference_rate_floor(track.encoding_model, self.config)

        np.testing.assert_allclose(
            np.asarray(floored["place_fields"])
            - np.asarray(track.encoding_model["place_fields"]),
            1e-3 / SAMPLING_FREQUENCY,
            rtol=1e-6,
        )

    def test_decodes_as_poisson_on_the_floored_fields(self, track):
        """``sum_n k_n log(f_n + b) - sum_n (f_n + b)`` per row, every term floored."""
        from non_local_detector.likelihoods.common import get_spikecount_per_time_bin

        floored = with_reference_rate_floor(track.encoding_model, self.config)
        result = sorted_spikes_epoch_log_likelihood(
            track.time, track.linear_position, track.spike_times, floored, self.config
        )

        interior = np.asarray(track.encoding_model["is_track_interior"]).ravel()
        fields = (
            np.asarray(track.encoding_model["place_fields"], dtype=np.float64)[
                :, interior
            ]
            + 1e-3 / SAMPLING_FREQUENCY
        )
        counts = np.stack(
            [get_spikecount_per_time_bin(t, track.time) for t in track.spike_times],
            axis=1,
        )[: result.time.size]
        expected = counts @ np.log(fields) - fields.sum(axis=0)
        np.testing.assert_allclose(
            result.log_likelihood, expected, rtol=1e-4, atol=1e-3
        )

    def test_the_model_it_was_given_is_unchanged(self, track):
        before = copy.deepcopy(
            {
                key: np.asarray(track.encoding_model[key])
                for key in ("place_fields", "no_spike_part_log_likelihood")
            }
        )

        with_reference_rate_floor(track.encoding_model, self.config)

        for key, value in before.items():
            np.testing.assert_array_equal(np.asarray(track.encoding_model[key]), value)


class TestInteriorBinDistances:
    def test_a_square_matrix_over_the_interior_bins(self, epoch):
        interior = np.asarray(epoch.environment.is_track_interior_).ravel()

        distance = interior_bin_distances_cm(epoch.environment)

        assert distance.shape == (interior.sum(), interior.sum())

    def test_the_diagonal_is_zero_and_the_matrix_is_symmetric(self, epoch):
        distance = interior_bin_distances_cm(epoch.environment)

        np.testing.assert_allclose(np.diag(distance), 0.0, atol=1e-9)
        np.testing.assert_allclose(distance, distance.T, atol=1e-9)

    def test_bins_on_different_arms_are_further_apart_than_in_linear_position(
        self, epoch
    ):
        """The reason jumps are measured on the graph and not on the coordinate."""
        interior = np.asarray(epoch.environment.is_track_interior_).ravel()
        centers = np.asarray(epoch.environment.place_bin_centers_).ravel()[interior]
        distance = interior_bin_distances_cm(epoch.environment)
        linear_gap = np.abs(centers[:, None] - centers[None, :])

        assert (distance > linear_gap + 1e-9).any()


class TestMarksMatchEncoding:
    """The marks decoded must be the marks the encoding model was fitted on.

    The runner reloads marks through ``load_data`` long after the decode. A
    change to the spike loader, or electrodes in another order, would decode
    the wrong marks against the right model without any error; the library
    only checks the electrode count.
    """

    def test_the_marks_the_model_was_fitted_on_pass(self, epoch):
        check_marks_match_encoding(
            epoch.encoding_model, epoch.spike_times, epoch.spike_waveform_features
        )

    def test_electrodes_in_another_order_are_refused(self, epoch):
        order = [1, 0, 2]
        with pytest.raises(ValueError, match="Electrode 0"):
            check_marks_match_encoding(
                epoch.encoding_model,
                [epoch.spike_times[i] for i in order],
                [epoch.spike_waveform_features[i] for i in order],
            )

    def test_a_changed_mark_is_refused(self, epoch):
        features = [np.array(f) for f in epoch.spike_waveform_features]
        features[2][5, 0] += 1.0

        with pytest.raises(ValueError, match="Electrode 2"):
            check_marks_match_encoding(
                epoch.encoding_model, epoch.spike_times, features
            )

    def test_a_missing_electrode_is_refused(self, epoch):
        with pytest.raises(ValueError, match="electrodes"):
            check_marks_match_encoding(
                epoch.encoding_model,
                epoch.spike_times[:2],
                epoch.spike_waveform_features[:2],
            )

    def test_a_different_number_of_features_is_refused(self, epoch):
        features = list(epoch.spike_waveform_features)
        features[1] = features[1][:, :1]

        with pytest.raises(ValueError, match="features"):
            check_marks_match_encoding(
                epoch.encoding_model, epoch.spike_times, features
            )

    def test_extra_spikes_beyond_the_encoding_ones_are_allowed(self, epoch):
        """The encoding set is the training span's spikes, a subset."""
        times = [np.append(t, t[-1] + 1.0) for t in epoch.spike_times]
        features = [
            np.vstack([f, np.full((1, f.shape[1]), 999.0)])
            for f in epoch.spike_waveform_features
        ]

        check_marks_match_encoding(epoch.encoding_model, times, features)
