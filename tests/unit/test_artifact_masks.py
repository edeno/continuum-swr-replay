"""The artifact masks: one padding table, closed intervals, merged and clipped.

The detector stores unpadded cores; each consumer pads them from
``CONSUMER_PADDING``. These pin the table's values (detector_report.md §5.1 and
the Phase C rulings), the merging and clipping, and the closed-interval
convention of ``interval_mask``. Offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.hippocampus.artifacts.masks import (
    CONSUMER_PADDING,
    Consumer,
    artifact_intervals,
    core_class,
    interval_mask,
    masked_seconds,
    subtract_intervals,
)

SPAN = (100.0, 200.0)


def _cores(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    """Cores from ``(signatures, core_start, core_end)``; kind S or T."""
    return pd.DataFrame(
        {
            "kind": ["S" if sig == "S" else "T" for sig, _, _ in rows],
            "signatures": [sig for sig, _, _ in rows],
            "core_start": [float(start) for _, start, _ in rows],
            "core_end": [float(end) for _, _, end in rows],
        }
    )


def _pairs(intervals: pd.DataFrame) -> list[tuple[float, float]]:
    assert list(intervals.columns) == ["start_time", "end_time"]
    return [(float(s), float(e)) for s, e in intervals.to_numpy()]


class TestThePaddingTable:
    def test_the_values_are_the_rulings(self):
        assert CONSUMER_PADDING == {
            Consumer.SWR: {"S": (0.0, 0.0), "T_lfp": (0.3, 0.3), "T_spike": (0.1, 0.1)},
            Consumer.DECODER: {
                "S": (1.0, 1.0),
                "T_lfp": (0.3, 0.3),
                "T_spike": (0.1, 0.1),
            },
            Consumer.ANALYSIS: {
                "S": (1.0, 1.0),
                "T_lfp": (0.3, 1.3),
                "T_spike": (0.1, 0.1),
            },
        }

    def test_the_table_cannot_be_changed_in_place(self):
        with pytest.raises(TypeError):
            CONSUMER_PADDING[Consumer.SWR] = {}  # type: ignore[index]
        with pytest.raises(TypeError):
            CONSUMER_PADDING[Consumer.SWR]["S"] = (5.0, 5.0)  # type: ignore[index]

    def test_analysis_is_decoder_plus_one_second_after_each_lfp_step(self):
        decoder = CONSUMER_PADDING[Consumer.DECODER]
        analysis = CONSUMER_PADDING[Consumer.ANALYSIS]
        assert analysis["S"] == decoder["S"]
        assert analysis["T_spike"] == decoder["T_spike"]
        assert analysis["T_lfp"] == (decoder["T_lfp"][0], decoder["T_lfp"][1] + 1.0)

    @pytest.mark.parametrize(
        ("signatures", "expected"),
        [
            ("S", "S"),
            ("ref", "T_lfp"),
            ("ca1", "T_lfp"),
            ("ca1+ref", "T_lfp"),
            ("ref+spk", "T_lfp"),
            ("ca1+ref+spk", "T_lfp"),
            ("spk", "T_spike"),
        ],
    )
    def test_each_core_falls_in_one_row_of_the_table(self, signatures, expected):
        cores = _cores([(signatures, 150.0, 150.0)])
        assert list(core_class(cores)) == [expected]

    def test_an_unknown_signature_is_refused(self):
        with pytest.raises(ValueError, match="emg"):
            core_class(_cores([("emg", 150.0, 150.0)]))


class TestEachConsumersIntervals:
    CORES = [
        ("S", 110.0, 115.0),  # S block
        ("ref", 130.0, 130.01),  # LFP step
        ("spk", 140.0, 140.0002),  # spike-only line
        ("spk", 140.15, 140.15),  # a second line 0.15 s later
        ("ca1+spk", 100.1, 100.12),  # a step near the span's start
        ("ref", 199.9, 199.9),  # a step near the span's end
    ]

    def test_swr(self):
        intervals = artifact_intervals(_cores(self.CORES), Consumer.SWR, SPAN)

        assert _pairs(intervals) == pytest.approx(
            [
                (100.0, 100.42),  # clipped at the span's start
                (110.0, 115.0),  # S unpadded
                (129.7, 130.31),
                (139.9, 140.25),  # the two lines merge once padded
                (199.6, 200.0),  # clipped at the span's end
            ]
        )

    def test_decoder(self):
        intervals = artifact_intervals(_cores(self.CORES), Consumer.DECODER, SPAN)

        assert _pairs(intervals) == pytest.approx(
            [
                (100.0, 100.42),
                (109.0, 116.0),  # S padded by 1 s
                (129.7, 130.31),
                (139.9, 140.25),
                (199.6, 200.0),
            ]
        )

    def test_analysis(self):
        intervals = artifact_intervals(_cores(self.CORES), Consumer.ANALYSIS, SPAN)

        assert _pairs(intervals) == pytest.approx(
            [
                (100.0, 101.42),  # 1.3 s after an LFP step
                (109.0, 116.0),
                (129.7, 131.31),
                (139.9, 140.25),  # spike-only: no extra second
                (199.6, 200.0),
            ]
        )

    def test_a_consumer_is_named_by_its_string_too(self):
        cores = _cores(self.CORES)
        pd.testing.assert_frame_equal(
            artifact_intervals(cores, "analysis", SPAN),
            artifact_intervals(cores, Consumer.ANALYSIS, SPAN),
        )

    def test_an_unknown_consumer_is_refused(self):
        with pytest.raises(ValueError, match="widloski"):
            artifact_intervals(_cores(self.CORES), "widloski", SPAN)

    def test_touching_intervals_merge(self):
        # closed intervals: [149, 156] and [156, 161] share 156
        cores = _cores([("S", 150.0, 155.0), ("S", 157.0, 160.0)])

        intervals = artifact_intervals(cores, Consumer.DECODER, SPAN)

        assert _pairs(intervals) == [(149.0, 161.0)]

    def test_overlapping_and_nested_intervals_merge(self):
        cores = _cores(
            [("S", 150.0, 160.0), ("ref", 155.0, 155.0), ("S", 159.0, 170.0)]
        )

        intervals = artifact_intervals(cores, Consumer.SWR, SPAN)

        assert _pairs(intervals) == [(150.0, 170.0)]

    def test_intervals_are_in_time_order_whatever_the_cores_order(self):
        cores = _cores(self.CORES)
        shuffled = cores.iloc[::-1].reset_index(drop=True)

        pd.testing.assert_frame_equal(
            artifact_intervals(shuffled, Consumer.DECODER, SPAN),
            artifact_intervals(cores, Consumer.DECODER, SPAN),
        )

    def test_no_cores_give_no_intervals(self):
        intervals = artifact_intervals(_cores([]), Consumer.ANALYSIS, SPAN)

        assert list(intervals.columns) == ["start_time", "end_time"]
        assert intervals.empty
        assert intervals.dtypes.tolist() == [np.float64, np.float64]

    def test_masked_seconds_is_the_merged_length(self):
        intervals = artifact_intervals(_cores(self.CORES), Consumer.SWR, SPAN)

        assert masked_seconds(intervals) == pytest.approx(
            0.42 + 5.0 + 0.61 + 0.35 + 0.4
        )


class TestTheMask:
    def test_intervals_are_closed(self):
        time = np.array([0.9, 1.0, 1.5, 2.0, 2.1, 3.0, 3.5, 4.0])
        intervals = pd.DataFrame({"start_time": [1.0, 3.0], "end_time": [2.0, 3.0]})

        mask = interval_mask(time, intervals)

        np.testing.assert_array_equal(
            mask, [False, True, True, True, False, True, False, False]
        )

    def test_unsorted_and_overlapping_intervals(self):
        time = np.linspace(0.0, 10.0, 1001)
        intervals = pd.DataFrame(
            {"start_time": [6.0, 1.0, 1.5], "end_time": [7.0, 3.0, 2.0]}
        )

        expected = ((time >= 1.0) & (time <= 3.0)) | ((time >= 6.0) & (time <= 7.0))
        np.testing.assert_array_equal(interval_mask(time, intervals), expected)

    def test_no_intervals_mask_nothing(self):
        intervals = pd.DataFrame({"start_time": [], "end_time": []}, dtype=float)

        assert not interval_mask(np.arange(5.0), intervals).any()

    def test_a_pandas_index_is_a_time_axis(self):
        intervals = pd.DataFrame({"start_time": [1.0], "end_time": [2.0]})

        mask = interval_mask(pd.Index([0.5, 1.0, 2.5]), intervals)

        np.testing.assert_array_equal(mask, [False, True, False])


class TestSubtractingIntervals:
    def test_the_rest_is_exactly_the_unmasked_time(self):
        """Closed stretches less closed intervals: a time is in the result if and
        only if it is in a stretch and in no interval, endpoints included."""
        stretches = np.array([[0.0, 10.0], [20.0, 30.0]])
        intervals = pd.DataFrame(
            {"start_time": [2.0, 5.0, 25.0], "end_time": [3.0, 5.0, 35.0]}
        )
        time = np.unique(
            np.r_[
                np.linspace(-1.0, 36.0, 3701),
                intervals.to_numpy().ravel(),
                np.nextafter(intervals.to_numpy().ravel(), np.inf),
                np.nextafter(intervals.to_numpy().ravel(), -np.inf),
            ]
        )

        rest = subtract_intervals(stretches, intervals)

        in_rest = np.zeros(time.size, dtype=bool)
        for start, end in rest:
            in_rest |= (time >= start) & (time <= end)
        in_stretch = ((time >= 0.0) & (time <= 10.0)) | (
            (time >= 20.0) & (time <= 30.0)
        )
        np.testing.assert_array_equal(
            in_rest, in_stretch & ~interval_mask(time, intervals)
        )
        assert rest.shape == (4, 2)

    def test_no_intervals_leave_the_stretches(self):
        stretches = np.array([[0.0, 10.0]])
        empty = pd.DataFrame({"start_time": [], "end_time": []}, dtype=float)

        np.testing.assert_array_equal(subtract_intervals(stretches, empty), stretches)

    def test_a_stretch_inside_an_interval_vanishes(self):
        stretches = np.array([[2.0, 3.0], [5.0, 6.0]])
        intervals = pd.DataFrame({"start_time": [1.0], "end_time": [4.0]})

        np.testing.assert_array_equal(
            subtract_intervals(stretches, intervals), [[5.0, 6.0]]
        )
