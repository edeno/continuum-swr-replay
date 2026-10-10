"""The artifact detector's four rules on synthetic epochs.

Each case adds one artifact, or one SWR-like event that must not be masked, to
a 30-s epoch of white noise and Poisson spikes (``_artifact_synthetic``). The
detector masks only on an artifact signature (S, T-ref, T-ca1, T-spk), never on
power alone. Offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.hippocampus.artifacts import detect
from continuum_swr_replay.analysis.hippocampus.artifacts.detect import (
    detect_artifacts,
)
from continuum_swr_replay.analysis.hippocampus.artifacts.features import (
    level_jump,
    persistent_step_count,
    robust_jump,
)
from continuum_swr_replay.analysis.hippocampus.artifacts.io import CORE_COLUMNS
from continuum_swr_replay.decoding_common import DecodingConfig
from tests.unit._artifact_synthetic import FS, N_TETRODES, NOISE, SyntheticEpoch


def _detect(epoch: SyntheticEpoch, **kwargs) -> pd.DataFrame:
    return detect_artifacts(
        epoch.time,
        epoch.ca1,
        epoch.reference,
        epoch.spike_times,
        epoch.spike_marks,
        FS,
        **kwargs,
    )


class TestTheConstants:
    def test_the_rule_p_thresholds_are_the_margin_times_the_calibration_maxima(
        self,
    ):
        """C2's rule P: 1.5 x the largest value in the calibration SWR set
        (followup/c2_rule_check.csv, row "rule (all four)", divided by 1.5)."""
        assert detect.THRESHOLD_MARGIN == 1.5
        assert detect.S_CORE_CALIBRATION_MAX == 2.8723976612091064
        assert detect.T_REF_CALIBRATION_MAX == 11.715611457824707
        assert detect.T_CA1_K_CALIBRATION_MAX == 16.682405471801758
        assert detect.S_CORE == 4.30859649181366
        assert detect.T_REF == 17.57341718673706
        assert detect.T_CA1_K == 25.023608207702637

    def test_the_definitions_are_the_reports(self):
        """detector_report.md §1 and §5.1, verbatim; T-spk stays off rule P."""
        assert detect.S_CORE_RUN == 2
        assert detect.S_EXT == 2.0
        assert detect.S_GAP == 2
        assert detect.S_WINDOW_S == 1.0
        assert detect.S_SEGMENT_S == 0.064
        assert detect.S_BAND_HZ == (150.0, 250.0)
        assert detect.T_CA1_RHO == 0.5
        assert detect.T_CA1_FRAC == 0.15
        assert detect.T_CA1_MIN == 3
        assert detect.T_SPK_FRAC == 0.40
        assert detect.T_SPK_OOD == 0.20
        assert detect.T_SPK_BIN_S == 200e-6
        assert detect.T_SPK_OOD_NN == 4.0
        assert detect.T_MERGE_S == 0.05

    def test_the_own_event_window_is_1_ms_and_recorded(self):
        """A neuron's absolute refractory period: a definition, off rule P."""
        assert detect.T_SPK_OOD_OWN_EVENT_S == 1e-3
        assert detect.detector_constants()["t_spk_ood_own_event_s"] == 1e-3

    def test_the_ood_cut_is_read_from_the_decoders_settings(self):
        assert detect.WAVEFORM_STD == DecodingConfig().waveform_std


class TestNoArtifact:
    def test_a_clean_epoch_has_no_core(self):
        cores = _detect(SyntheticEpoch())

        assert cores.empty
        assert list(cores.columns) == list(CORE_COLUMNS)

    def test_the_inputs_are_left_as_they_were(self):
        epoch = SyntheticEpoch()
        epoch.add_step(10.5, [], 60 * NOISE, on_reference=True)
        ca1, reference = epoch.ca1.copy(), epoch.reference.copy()

        _detect(epoch)

        np.testing.assert_array_equal(epoch.ca1, ca1)
        np.testing.assert_array_equal(epoch.reference, reference)


class TestTRef:
    def test_a_reference_step_gives_a_t_ref_core(self):
        epoch = SyntheticEpoch()
        epoch.add_step(10.5, [], 60 * NOISE, on_reference=True)

        cores = _detect(epoch)

        assert len(cores) == 1
        core = cores.iloc[0]
        assert core["kind"] == "T"
        assert core["signatures"] == "ref"
        assert epoch.at(10.5) - 0.01 <= core["core_start"] <= epoch.at(10.5)
        assert epoch.at(10.5) <= core["core_end"] <= epoch.at(10.5) + 0.01
        assert core["ref_j5_max"] >= detect.T_REF
        assert np.isnan(core["ca1_frac_max"]) and np.isnan(core["spk_frac_max"])

    def test_a_reference_step_mirrored_on_ca1_names_both_signatures(self):
        epoch = SyntheticEpoch()
        epoch.add_step(10.5, [], 60 * NOISE, on_reference=True)
        epoch.add_step(10.5, list(range(20)), -60 * NOISE)

        cores = _detect(epoch)

        assert cores["signatures"].tolist() == ["ca1+ref"]
        assert cores["ca1_frac_max"].iloc[0] == pytest.approx(1.0)

    def test_a_reference_jump_below_15_robust_units_is_not_masked(self):
        epoch = SyntheticEpoch()
        epoch.add_step(10.5, [], 5 * NOISE, on_reference=True)
        robust = robust_jump(
            level_jump(epoch.reference), np.ones(epoch.time.size, bool)
        )
        assert 4.0 < robust.max() < detect.T_REF  # the step is real but small

        assert _detect(epoch).empty


class TestTCa1:
    def test_a_persistent_step_on_3_of_20_channels_gives_t_ca1(self):
        epoch = SyntheticEpoch()
        epoch.add_step(12.5, [0, 1, 2], 60 * NOISE)

        cores = _detect(epoch)

        assert cores["kind"].tolist() == ["T"]
        assert cores["signatures"].tolist() == ["ca1"]
        assert cores["ca1_frac_max"].iloc[0] == pytest.approx(3 / 20)
        assert abs(cores["core_start"].iloc[0] - epoch.at(12.5)) < 0.01

    def test_a_step_on_2_of_20_channels_is_not_enough(self):
        epoch = SyntheticEpoch()
        epoch.add_step(12.5, [0, 1], 60 * NOISE)

        assert _detect(epoch).empty

    def test_a_ripple_burst_on_every_channel_gives_no_core(self):
        """Its cycles jump >= 20 robust units on many channels, as a giant
        SWR's onset does, but the level does not persist."""
        epoch = SyntheticEpoch()
        epoch.add_ripple_burst(15.5, 60 * NOISE)
        robust = robust_jump(level_jump(epoch.ca1), np.ones(epoch.time.size, bool))
        assert (robust >= detect.T_CA1_K).sum(axis=1).max() >= 10  # the jumps are there

        assert _detect(epoch).empty


class TestS:
    def test_5_s_of_ripple_noise_on_4_channels_is_one_block_with_its_dip_filled(self):
        epoch = SyntheticEpoch()
        # windows 10-11 and 13-15 noisy, window 12 a one-window dip
        epoch.add_band_noise(10.0, 12.0, [0, 1, 2, 3], rms=5 * NOISE)
        epoch.add_band_noise(13.0, 16.0, [0, 1, 2, 3], rms=5 * NOISE, seed=2)

        cores = _detect(epoch)

        assert cores["kind"].tolist() == ["S"]
        core = cores.iloc[0]
        assert core["signatures"] == "S"
        assert core["n_windows"] == 6
        assert core["spec_max"] >= detect.S_CORE
        assert core["core_start"] == pytest.approx(epoch.at(10.0) - 0.0005, abs=1e-6)
        assert core["core_end"] == pytest.approx(epoch.at(16.0) - 0.0005, abs=1e-6)

    def test_a_single_noisy_window_is_not_s(self):
        epoch = SyntheticEpoch()
        epoch.add_band_noise(10.0, 11.0, [0, 1, 2, 3], rms=5 * NOISE)

        assert _detect(epoch).empty

    def test_a_gap_of_3_windows_keeps_two_blocks(self):
        epoch = SyntheticEpoch()
        epoch.add_band_noise(5.0, 7.0, [0, 1, 2, 3], rms=5 * NOISE)
        epoch.add_band_noise(10.0, 12.0, [0, 1, 2, 3], rms=5 * NOISE, seed=2)

        cores = _detect(epoch)

        assert cores["kind"].tolist() == ["S", "S"]
        assert cores["n_windows"].tolist() == [2, 2]


class TestTSpk:
    def test_a_line_on_half_the_tetrodes_with_ood_marks_gives_t_spk(self):
        epoch = SyntheticEpoch()
        epoch.add_spike_line(20.25, [0, 2, 4, 6, 8], out_of_distribution=True)

        cores = _detect(epoch)

        assert cores["signatures"].tolist() == ["spk"]
        core = cores.iloc[0]
        assert core["kind"] == "T"
        assert core["spk_frac_max"] == pytest.approx(0.5)
        assert core["spk_ood_max"] == pytest.approx(1.0)
        assert abs(core["core_start"] - epoch.at(20.25)) < 0.001

    def test_the_same_line_with_in_distribution_marks_gives_no_core(self):
        epoch = SyntheticEpoch()
        epoch.add_spike_line(20.25, [0, 2, 4, 6, 8], out_of_distribution=False)

        assert _detect(epoch).empty

    def test_a_line_detected_twice_per_tetrode_still_gives_t_spk(self):
        """The R9 miss (chimi20200227_02_r1 at 8.614 s): the artifact is
        detected twice, 0.15 ms apart, on each tetrode of the line. The twins
        share a mark, but neither is the other's reference: they are one
        event."""
        epoch = SyntheticEpoch()
        tetrodes = [0, 2, 4, 6, 8]
        epoch.add_spike_line(20.25, tetrodes, out_of_distribution=True)
        epoch.add_spike_line(20.25 + 0.15e-3, tetrodes, out_of_distribution=True)

        cores = _detect(epoch)

        assert cores["signatures"].tolist() == ["spk"]
        assert cores["spk_frac_max"].iloc[0] == pytest.approx(0.5)
        assert cores["spk_ood_max"].iloc[0] == pytest.approx(1.0)

    def test_twin_detections_are_ood_but_the_same_marks_2_ms_apart_are_not(self):
        cut = detect.T_SPK_OOD_NN * detect.WAVEFORM_STD
        artifact = np.full(4, 10 * cut)
        marks = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 0, 0, 0], artifact, artifact])
        empty = pd.DataFrame({"start_time": [], "end_time": []}, dtype=float)

        def ood(gap_s: float) -> list[bool]:
            times = np.array([1.0, 2.0, 5.0, 5.0 + gap_s])
            _, _, is_ood, _ = detect.spike_ood([times], [marks], (0.0, 9.0), empty)
            return is_ood.tolist()

        assert ood(0.15e-3) == [False, False, True, True]
        assert ood(2e-3) == [False, False, False, False]

    def test_an_ood_line_on_30_percent_of_tetrodes_is_not_enough(self):
        epoch = SyntheticEpoch()
        epoch.add_spike_line(20.25, [0, 2, 4], out_of_distribution=True)

        assert _detect(epoch).empty

    def test_the_ood_reference_leaves_out_marks_inside_lfp_signatures(self):
        """Every tetrode fires artifact-like marks around a reference step. Inside
        the step's decoder-padded interval they are not reference marks, so a
        later line with the same marks is out of distribution."""
        artifact_mark = np.full(4, 1500.0)
        burst = 5.5 + np.linspace(-0.25, 0.25, 30)

        def epoch_with(step: bool) -> SyntheticEpoch:
            epoch = SyntheticEpoch()
            if step:
                epoch.add_step(5.5, [], 60 * NOISE, on_reference=True)
            for tetrode in range(N_TETRODES):
                epoch.add_spikes(tetrode, burst + tetrode * 1e-3, artifact_mark)
            epoch.add_spike_line(20.25, [0, 2, 4, 6, 8], out_of_distribution=True)
            return epoch

        with_step = _detect(epoch_with(step=True))
        without_step = _detect(epoch_with(step=False))

        line = abs(with_step["core_start"] - epoch_with(step=True).at(20.25)) < 0.001
        assert with_step.loc[line, "signatures"].tolist() == ["spk"]
        # without an LFP signature the burst's marks are reference marks, so
        # the line is in distribution, as it was against the decoder's encoding
        assert not (
            abs(without_step["core_start"] - epoch_with(step=False).at(20.25)) < 0.001
        ).any()


class TestMergingT:
    def test_signatures_within_50_ms_merge_into_one_core(self):
        epoch = SyntheticEpoch()
        epoch.add_step(10.5, [], 60 * NOISE, on_reference=True)
        epoch.add_spike_line(10.53, [0, 2, 4, 6, 8], out_of_distribution=True)

        cores = _detect(epoch)

        assert cores["signatures"].tolist() == ["ref+spk"]
        assert cores["spk_frac_max"].iloc[0] == pytest.approx(0.5)

    def test_signatures_further_apart_stay_separate(self):
        epoch = SyntheticEpoch()
        epoch.add_step(10.5, [], 60 * NOISE, on_reference=True)
        epoch.add_spike_line(10.7, [0, 2, 4, 6, 8], out_of_distribution=True)

        cores = _detect(epoch)

        assert cores["signatures"].tolist() == ["ref", "spk"]
        assert cores["core_start"].is_monotonic_increasing


class TestInvalidSamples:
    """No feature reads across an invalid sample, and the gap itself is not
    masked. The probe: a DC offset turns zero-filled samples into steps at the
    run's edges unless the step features skip them."""

    GAP = slice(10000, 10300)
    """NaN at samples 10000-10299 (0.3 s at 10.0-10.3 s)."""

    @staticmethod
    def _offset_epoch() -> SyntheticEpoch:
        epoch = SyntheticEpoch()
        epoch.ca1 += 1000.0
        epoch.reference += 1000.0
        return epoch

    def test_non_finite_rows_are_counted_and_make_no_core(self):
        epoch = SyntheticEpoch()
        epoch.ca1[5000:5010, 3] = np.nan

        cores = _detect(epoch, valid=np.ones(epoch.time.size, dtype=bool))

        assert cores.attrs["counts"]["n_invalid_samples"] == 10
        assert cores.empty

    @pytest.mark.parametrize("how", ["nan", "valid"])
    def test_a_gap_in_offset_data_makes_no_core_at_its_edges(self, how):
        epoch = self._offset_epoch()
        valid = np.ones(epoch.time.size, dtype=bool)
        if how == "nan":
            epoch.ca1[self.GAP] = np.nan
        else:  # the stored band unobserved there, as the runner passes it
            valid[self.GAP] = False

        cores = _detect(epoch, valid=valid)

        assert cores.empty, cores[["signatures", "core_start"]]
        counts = cores.attrs["counts"]
        assert counts["n_invalid_samples"] == 300
        assert counts["n_windows_without_power"] == 1  # window 10 holds the gap

    def test_a_ca1_step_next_to_a_gap_is_still_found(self):
        """40 ms after the gap: its J5 and L20 windows do not reach it."""
        epoch = self._offset_epoch()
        epoch.ca1[self.GAP] = np.nan
        epoch.add_step(10.34, [0, 1, 2], 60 * NOISE)

        cores = _detect(epoch)

        assert cores["signatures"].tolist() == ["ca1"]
        assert abs(cores["core_start"].iloc[0] - epoch.at(10.34)) < 0.01

    def test_a_ca1_step_whose_l20_window_reaches_a_gap_is_not_found(self):
        """10 ms after the gap: its J5 window does not reach it, but its L20
        window (t - 24 .. t + 24 samples) does. L20 there would read the
        zero-filled gap, so the step has no L20 value and the robust J5 is
        zeroed (``detect_artifacts``): no persistent step, no T-ca1."""
        clean = self._offset_epoch()
        clean.add_step(10.31, [0, 1, 2], 60 * NOISE)
        assert _detect(clean)["signatures"].tolist() == ["ca1"]  # found without it

        epoch = self._offset_epoch()
        epoch.ca1[self.GAP] = np.nan
        epoch.add_step(10.31, [0, 1, 2], 60 * NOISE)

        cores = _detect(epoch)

        assert cores.empty, cores[["signatures", "core_start"]]

    def test_a_reference_step_next_to_a_gap_is_still_found(self):
        """10 ms after the gap: T-ref reads J5 alone, whose window does not
        reach it."""
        epoch = self._offset_epoch()
        epoch.ca1[self.GAP] = np.nan
        epoch.add_step(10.31, [], 60 * NOISE, on_reference=True)

        cores = _detect(epoch)

        assert cores["signatures"].tolist() == ["ref"]
        assert abs(cores["core_start"].iloc[0] - epoch.at(10.31)) < 0.01


class TestTheInputs:
    def test_a_reference_not_shaped_time_by_electrode_is_refused(self):
        epoch = SyntheticEpoch()
        two = np.column_stack([epoch.reference, epoch.reference])

        for reference in (two.T, epoch.reference[:, 0], epoch.reference[:-1]):
            with pytest.raises(ValueError, match="reference_lfp"):
                detect_artifacts(
                    epoch.time,
                    epoch.ca1,
                    reference,
                    epoch.spike_times,
                    epoch.spike_marks,
                    FS,
                )

    def test_a_ca1_lfp_not_shaped_time_by_channel_is_refused(self):
        epoch = SyntheticEpoch()
        with pytest.raises(ValueError, match="ca1_lfp"):
            detect_artifacts(
                epoch.time,
                epoch.ca1.T,
                epoch.reference,
                epoch.spike_times,
                epoch.spike_marks,
                FS,
            )

    def test_no_reference_electrode_disables_t_ref_and_says_so(self):
        epoch = SyntheticEpoch()
        epoch.add_step(10.5, [0, 1, 2], 60 * NOISE)
        no_reference = np.zeros((epoch.time.size, 0))

        cores = detect_artifacts(
            epoch.time,
            epoch.ca1,
            no_reference,
            epoch.spike_times,
            epoch.spike_marks,
            FS,
        )

        assert cores.attrs["counts"]["t_ref_disabled"] is True
        assert cores.attrs["counts"]["n_references"] == 0
        assert cores["signatures"].tolist() == ["ca1"]  # the other rules still run
        assert _detect(SyntheticEpoch()).attrs["counts"]["t_ref_disabled"] is False

    def test_timestamp_gaps_are_counted(self):
        epoch = SyntheticEpoch()
        epoch.time[5000:] += 0.01
        epoch.time[20000:] += 0.5

        counts = _detect(epoch).attrs["counts"]

        assert counts["n_time_gaps"] == 2
        assert _detect(SyntheticEpoch()).attrs["counts"]["n_time_gaps"] == 0


class TestBoundaries:
    """Each threshold at exactly its value: >= for T_REF, T_CA1_K, T_SPK_FRAC,
    T_SPK_OOD;
    a strict > for the nearest-mark cut; merges at exactly S_GAP windows and
    at exactly T_MERGE_S."""

    def test_t_ref_fires_at_exactly_its_threshold(self):
        robust = np.array(
            [np.nextafter(detect.T_REF, 0.0), detect.T_REF, detect.T_REF + 1e-3]
        )

        events = detect.t_ref_events(np.arange(3.0), robust)

        assert events["time"].tolist() == [1.0, 2.0]

    def test_t_ca1_counts_a_step_at_exactly_its_threshold(self):
        jump = np.full((2, 3), 30.0)
        robust = np.array(
            [[detect.T_CA1_K] * 3, [np.nextafter(detect.T_CA1_K, 0.0)] * 3]
        )
        shift = np.full((2, 3), 30.0)

        count = persistent_step_count(
            jump, robust, shift, threshold=detect.T_CA1_K, persistence=detect.T_CA1_RHO
        )

        np.testing.assert_array_equal(count, [3, 0])

    @pytest.mark.parametrize(
        ("n_tetrodes", "ood", "fires"),
        [(4, 1.0, True), (3, 1.0, False), (5, 0.2, True), (5, 0.19999, False)],
    )
    def test_t_spk_at_exactly_40_and_20_percent(self, n_tetrodes, ood, fires):
        lines = pd.DataFrame(
            {
                "time": [1.0],
                "n_tetrodes": [n_tetrodes],
                "tetrode_fraction": [n_tetrodes / 10],
                "n_spikes": [5],
                "ood_fraction": [ood],
            }
        )

        assert len(detect.t_spk_events(lines)) == int(fires)

    def test_a_line_on_exactly_4_of_10_tetrodes_is_caught(self):
        epoch = SyntheticEpoch()
        epoch.add_spike_line(20.25, [0, 3, 6, 9], out_of_distribution=True)

        cores = _detect(epoch)

        assert cores["signatures"].tolist() == ["spk"]
        assert cores["spk_frac_max"].iloc[0] == pytest.approx(0.4)

    def test_a_line_with_exactly_1_of_5_marks_ood_is_caught(self):
        epoch = SyntheticEpoch()
        epoch.add_spike_line(20.25, [0, 2, 4, 6], out_of_distribution=False)
        epoch.add_spike_line(20.25 + 32e-6, [8], out_of_distribution=True)

        cores = _detect(epoch)

        assert cores["signatures"].tolist() == ["spk"]
        assert cores["spk_ood_max"].iloc[0] == pytest.approx(0.2)

    @pytest.mark.parametrize(("offset", "ood"), [(0.0, False), (1e-3, True)])
    def test_the_nearest_mark_cut_is_strict(self, offset, ood):
        cut = detect.T_SPK_OOD_NN * detect.WAVEFORM_STD
        marks = np.array([[0.0, 0.0, 0.0, 0.0], [cut + offset, 0.0, 0.0, 0.0]])
        empty = pd.DataFrame({"start_time": [], "end_time": []}, dtype=float)

        _, _, is_ood, _ = detect.spike_ood(
            [np.array([1.0, 2.0])], [marks], (0.0, 3.0), empty
        )

        assert is_ood.tolist() == [ood, ood]

    @pytest.mark.parametrize(
        ("value", "n_blocks"),
        [(detect.S_CORE, 1), (np.nextafter(detect.S_CORE, 0.0), 0)],
        ids=["at-s_core", "just-below"],
    )
    def test_an_s_core_starts_at_exactly_s_core(self, value, n_blocks):
        spec = np.ones(30)
        spec[10:12] = value

        cores = detect.s_cores(np.arange(30) + 0.5, spec)

        assert len(cores) == n_blocks

    @pytest.mark.parametrize(("second_block", "n_blocks"), [(14, 1), (15, 2)])
    def test_s_blocks_exactly_2_windows_apart_merge(self, second_block, n_blocks):
        spec = np.ones(30)
        spec[10:12] = 10.0
        spec[second_block : second_block + 2] = 10.0

        cores = detect.s_cores(np.arange(30) + 0.5, spec)

        assert len(cores) == n_blocks
        if n_blocks == 1:
            assert cores["n_windows"].iloc[0] == 6  # the 2 gap windows included

    @pytest.mark.parametrize(("second", "n_cores"), [(0.05, 1), (0.0500001, 2)])
    def test_t_events_exactly_50_ms_apart_merge(self, second, n_cores):
        events = detect.t_ref_events(np.array([0.0, second]), np.array([20.0, 20.0]))

        assert len(detect.t_cores(events)) == n_cores
