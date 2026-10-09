"""The artifact detector's features, each against its definition by hand.

The definitions are the prototype's (``.claude/research/refresh_artifact_detector/
scripts/artdet_lfp*.py``, ``artdet_spikes.py``, ``artdet_detector.py``):

- robust units: a statistic over its per-channel MAD x 1.4826;
- S: 150-250 Hz Welch power per 1-s window (15 Hann segments of 64 ms), each
  channel over its epoch median, and spec = 2nd largest / median over channels;
- J5(t) = median(x[t+1..t+5]) - median(x[t-5..t-1]);
- L20(t) = mean(x[t+5..t+24]) - mean(x[t-24..t-5]);
- spike lines on two 200-µs grids offset by half a bin;
- a mark's leave-self-out nearest-neighbour distance to a reference set.

Offline and deterministic.
"""

from __future__ import annotations

import numpy as np
import pytest

from continuum_swr_replay.analysis.hippocampus.artifacts.features import (
    channel_specificity,
    level_jump,
    level_shift,
    nearest_mark_distance,
    persistent_step_count,
    reference_jump,
    robust_jump,
    robust_scale,
    spike_lines,
    window_band_power,
    window_centers,
    window_valid,
)

FS = 1000.0


class TestRobustUnits:
    def test_mad_times_1_4826_per_column_over_valid_rows(self):
        values = np.array(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [1000.0, -5.0]]
        )
        valid = np.array([True, True, True, True, False])

        scale = robust_scale(values, valid)

        np.testing.assert_allclose(scale, 1.4826 * np.array([1.0, 10.0]))

    def test_every_row_when_no_validity_is_given(self):
        values = np.arange(10.0)[:, None]

        np.testing.assert_allclose(robust_scale(values), [1.4826 * 2.5])


class TestWindowValidity:
    def test_a_sample_is_valid_when_its_whole_window_is(self):
        valid = np.ones(100, dtype=bool)
        valid[50] = False

        out = window_valid(valid, 5)

        expected = np.ones(100, dtype=bool)
        expected[45:56] = False
        np.testing.assert_array_equal(out, expected)

    def test_samples_beyond_the_ends_count_as_valid(self):
        valid = np.ones(20, dtype=bool)

        np.testing.assert_array_equal(window_valid(valid, 24), valid)

    def test_a_run_of_invalid_samples_widens_by_the_reach_on_each_side(self):
        valid = np.ones(200, dtype=bool)
        valid[100:130] = False

        out = window_valid(valid, 24)

        assert out[75] and not out[76]
        assert not out[153] and out[154]


class TestLevelJump:
    def test_j5_is_the_difference_of_5_sample_medians(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(200, 3)).astype(np.float32)

        jump = level_jump(x)

        assert jump.shape == x.shape
        for t in range(5, 200 - 5):
            expected = np.median(x[t + 1 : t + 6], axis=0) - np.median(
                x[t - 5 : t], axis=0
            )
            np.testing.assert_allclose(jump[t], expected, rtol=1e-6)
        assert not jump[:3].any() and not jump[-3:].any()

    def test_robust_jump_is_absolute_j5_over_its_mad(self):
        jump = np.array([[-4.0], [0.0], [1.0], [2.0], [3.0]], dtype=np.float32)

        robust = robust_jump(jump, np.ones(5, dtype=bool))

        np.testing.assert_allclose(robust[:, 0], np.abs(jump[:, 0]) / 1.4826)

    def test_a_sample_without_a_j5_value_is_0_and_left_out_of_the_scale(self):
        rng = np.random.default_rng(3)
        jump = rng.normal(size=(300, 2)).astype(np.float32)
        jump[100:110] = 1e4  # values read across an invalid sample
        has_jump = np.ones(300, dtype=bool)
        has_jump[100:110] = False

        robust = robust_jump(jump, has_jump)

        assert not robust[100:110].any()
        clean = np.r_[0:100, 110:300]
        np.testing.assert_allclose(
            robust[clean], np.abs(jump[clean]) / robust_scale(jump[clean]), rtol=1e-6
        )

    def test_reference_jump_is_the_largest_over_references(self):
        x = np.zeros((100, 2), dtype=np.float32)
        x[50:, 1] = 100.0
        x[::2, :] += 1.0  # a little texture so the scale is not zero

        out = reference_jump(x, np.ones(100, dtype=bool))
        per_reference = robust_jump(level_jump(x), np.ones(100, dtype=bool))

        np.testing.assert_allclose(out, per_reference.max(axis=1))
        assert out.shape == (100,)

    def test_no_reference_electrode_gives_no_jump(self):
        out = reference_jump(np.zeros((100, 0)), np.ones(100, dtype=bool))

        np.testing.assert_array_equal(out, np.zeros(100))


class TestLevelShift:
    def test_l20_is_the_difference_of_20_sample_means_skipping_5(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(300, 2)).astype(np.float32)

        shift = level_shift(x)

        for t in range(24, 300 - 25):
            expected = x[t + 5 : t + 25].mean(axis=0) - x[t - 24 : t - 4].mean(axis=0)
            np.testing.assert_allclose(shift[t], expected, rtol=1e-5, atol=1e-6)
        assert not shift[:24].any() and not shift[300 - 25 :].any()


class TestPersistentStepCount:
    def test_a_channel_counts_when_large_same_signed_and_persistent(self):
        jump = np.array([[30.0, 30.0, -30.0, 30.0, 10.0]])
        robust = np.abs(jump) / np.array([1.0, 1.0, 1.0, 1.0, 0.1])
        shift = np.array([[20.0, -20.0, -20.0, 10.0, 10.0]])
        # 0: counts; 1: opposite sign; 2: counts (both negative);
        # 3: |L20| = 10 < 0.5 * 30; 4: robust 100, |L20| = 10 >= 5: counts

        count = persistent_step_count(
            jump, robust, shift, threshold=20.0, persistence=0.5
        )

        np.testing.assert_array_equal(count, [3])

    def test_the_threshold_is_inclusive(self):
        jump = np.array([[20.0]])
        count = persistent_step_count(
            jump, jump.copy(), np.array([[10.0]]), threshold=20.0, persistence=0.5
        )
        np.testing.assert_array_equal(count, [1])


class TestWindowSpectra:
    def test_power_is_welchs_mean_of_15_hann_segments(self):
        rng = np.random.default_rng(2)
        n_windows, n_channels = 3, 2
        x = rng.normal(size=(n_windows * 1000 + 400, n_channels)).astype(np.float32)

        power = window_band_power(x, FS, band_hz=(150.0, 250.0))

        freqs = np.fft.rfftfreq(64, 1 / FS)
        band = (freqs >= 150.0) & (freqs < 250.0)
        for w in range(n_windows):
            for c in range(n_channels):
                segments = []
                for s in range(15):
                    seg = x[w * 1000 + s * 64 : w * 1000 + (s + 1) * 64, c]
                    seg = (seg - seg.mean()) * np.hanning(64)
                    segments.append(np.abs(np.fft.rfft(seg)) ** 2)
                expected = np.mean(segments, axis=0)[band].sum()
                np.testing.assert_allclose(power[w, c], expected, rtol=1e-4)
        assert power.shape == (n_windows, n_channels)

    def test_a_window_holding_an_invalid_sample_has_no_power(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(4000, 3)).astype(np.float32)
        valid = np.ones(4000, dtype=bool)
        valid[2999] = False  # the last sample of window 2

        power = window_band_power(x, FS, valid=valid)

        assert np.isnan(power[2]).all()
        np.testing.assert_array_equal(
            power[[0, 1, 3]], window_band_power(x, FS)[[0, 1, 3]]
        )

    def test_window_centers_are_the_mean_sample_time(self):
        time = 1.6e9 + np.arange(3500) / FS

        centers = window_centers(time, FS)

        np.testing.assert_allclose(
            centers, time[:3000].reshape(3, 1000).mean(axis=1), rtol=0, atol=1e-6
        )

    def test_spec_is_second_largest_over_median_of_the_channel_ratios(self):
        power = np.ones((5, 6))
        power[2, [0, 1]] = [50.0, 30.0]  # window 2: two channels raised
        power[3, 0] = 100.0  # window 3: one channel raised

        spec = channel_specificity(power)

        # every channel's epoch median is 1, so ratios are the powers
        np.testing.assert_allclose(spec, [1.0, 1.0, 30.0, 1.0, 1.0])

    def test_a_window_without_power_has_no_specificity(self):
        rng = np.random.default_rng(5)
        power = rng.uniform(1.0, 2.0, (8, 6))
        power[[2, 5]] = np.nan

        spec = channel_specificity(power)

        assert np.isnan(spec[[2, 5]]).all()
        kept = [0, 1, 3, 4, 6, 7]
        np.testing.assert_allclose(spec[kept], channel_specificity(power[kept]))

    def test_no_window_with_power_is_refused(self):
        with pytest.raises(ValueError, match="no window"):
            channel_specificity(np.full((3, 4), np.nan))

    def test_an_swr_like_rise_on_every_channel_is_not_specific(self):
        power = np.ones((5, 6))
        power[2] = 40.0

        assert channel_specificity(power)[2] == pytest.approx(1.0)


class TestNearestMark:
    def test_the_spike_itself_is_left_out(self):
        marks = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0], [30.0, 0.0]])
        is_reference = np.array([True, True, True, False])

        distance = nearest_mark_distance(marks, is_reference)

        # 0 and 1 are exact duplicates: each the other's neighbour
        np.testing.assert_allclose(distance, [0.0, 0.0, 10.0, 20.0])

    def test_a_lone_reference_mark_has_no_neighbour(self):
        marks = np.array([[0.0, 0.0], [3.0, 4.0]])

        distance = nearest_mark_distance(marks, np.array([True, False]))

        np.testing.assert_allclose(distance, [np.inf, 5.0])

    def test_no_reference_means_every_mark_is_unsupported(self):
        marks = np.zeros((3, 4))

        distance = nearest_mark_distance(marks, np.zeros(3, dtype=bool))

        assert np.isinf(distance).all()


class TestSpikeLines:
    def test_a_line_across_a_bin_edge_is_whole_on_the_offset_grid(self):
        start = 1.6e9
        # 4 tetrodes within 40 µs around a grid-0 edge at start + 0.2 s
        times = start + 0.2 + np.array([-30e-6, -10e-6, 5e-6, 20e-6])
        tetrode = np.array([0, 1, 2, 3])
        is_ood = np.array([True, True, False, False])

        lines = spike_lines(
            times, tetrode, is_ood, n_tetrodes=8, start_time=start, bin_s=200e-6
        )

        best = lines.loc[lines["n_tetrodes"].idxmax()]
        assert best["n_tetrodes"] == 4
        assert best["tetrode_fraction"] == pytest.approx(0.5)
        assert best["ood_fraction"] == pytest.approx(0.5)
        assert best["n_spikes"] == 4
        # grid 0 splits it 2 + 2
        grid0 = lines[np.isclose((lines["time"] - start) % 200e-6, 100e-6, atol=1e-6)]
        assert grid0["n_tetrodes"].max() == 2

    def test_counts_are_distinct_tetrodes_but_every_spike(self):
        start = 0.0
        times = np.array([1.00001, 1.00002, 1.00003])
        tetrode = np.array([0, 0, 1])
        is_ood = np.array([True, False, False])

        lines = spike_lines(
            times, tetrode, is_ood, n_tetrodes=4, start_time=start, bin_s=200e-6
        )

        assert (lines["n_tetrodes"] == 2).all()
        assert (lines["n_spikes"] == 3).all()
        np.testing.assert_allclose(lines["ood_fraction"], 1 / 3)
        assert lines["time"].is_monotonic_increasing

    def test_bins_below_the_minimum_tetrode_count_are_left_out(self):
        times = np.array([1.0, 2.0, 2.00001])
        lines = spike_lines(
            times,
            np.array([0, 0, 1]),
            np.zeros(3, dtype=bool),
            n_tetrodes=4,
            start_time=0.0,
            bin_s=200e-6,
            min_tetrodes=2,
        )

        assert len(lines) >= 1
        assert (lines["n_tetrodes"] >= 2).all()
        assert (np.abs(lines["time"] - 2.0) < 2e-4).all()

    def test_no_spikes_give_no_lines(self):
        lines = spike_lines(
            np.zeros(0),
            np.zeros(0, dtype=int),
            np.zeros(0, dtype=bool),
            n_tetrodes=4,
            start_time=0.0,
            bin_s=200e-6,
        )

        assert lines.empty
        assert list(lines.columns) == [
            "time",
            "n_tetrodes",
            "tetrode_fraction",
            "n_spikes",
            "ood_fraction",
        ]
