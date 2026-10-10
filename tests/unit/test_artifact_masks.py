"""The artifact mask: one padding table, closed intervals, merged and clipped.

The detector stores unpadded cores; one table, ``PADDING_S``, pads them into
the mask every stage uses. These pin the table's values (the lead's simplified
design: S ±1 s, a step with an LFP signature ±0.3 s, a spike-only line
±0.1 s), the merging and clipping, and the closed-interval convention of
``interval_mask``. Offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.hippocampus.artifacts.masks import (
    PADDING_S,
    artifact_intervals,
    core_class,
    covered_seconds,
    interval_mask,
    masked_seconds,
    overlaps_intervals,
    pad_intervals,
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
    def test_the_values_are_the_leads_design(self):
        assert PADDING_S == {"S": 1.0, "T_lfp": 0.3, "T_spike": 0.1}

    def test_the_table_cannot_be_changed_in_place(self):
        with pytest.raises(TypeError):
            PADDING_S["S"] = 0.0  # type: ignore[index]

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


class TestTheMaskIntervals:
    CORES = [
        ("S", 110.0, 115.0),  # S block
        ("ref", 130.0, 130.01),  # LFP step
        ("spk", 140.0, 140.0002),  # spike-only line
        ("spk", 140.15, 140.15),  # a second line 0.15 s later
        ("ca1+spk", 100.1, 100.12),  # a step near the span's start
        ("ref", 199.9, 199.9),  # a step near the span's end
    ]

    def test_each_row_pads_both_sides(self):
        intervals = artifact_intervals(_cores(self.CORES), SPAN)

        assert _pairs(intervals) == pytest.approx(
            [
                (100.0, 100.42),  # clipped at the span's start
                (109.0, 116.0),  # S padded by 1 s
                (129.7, 130.31),  # an LFP step by 0.3 s, after it too
                (139.9, 140.25),  # spike-only by 0.1 s: the two lines merge
                (199.6, 200.0),  # clipped at the span's end
            ]
        )

    def test_there_is_no_post_step_second(self):
        intervals = artifact_intervals(_cores([("ca1+ref", 150.0, 150.0)]), SPAN)

        assert _pairs(intervals) == pytest.approx([(149.7, 150.3)])

    def test_touching_intervals_merge(self):
        # closed intervals: [149, 156] and [156, 161] share 156
        cores = _cores([("S", 150.0, 155.0), ("S", 157.0, 160.0)])

        intervals = artifact_intervals(cores, SPAN)

        assert _pairs(intervals) == [(149.0, 161.0)]

    def test_overlapping_and_nested_intervals_merge(self):
        cores = _cores(
            [("S", 150.0, 160.0), ("ref", 155.0, 155.0), ("S", 159.0, 170.0)]
        )

        intervals = artifact_intervals(cores, SPAN)

        assert _pairs(intervals) == [(149.0, 171.0)]

    def test_intervals_are_in_time_order_whatever_the_cores_order(self):
        cores = _cores(self.CORES)
        shuffled = cores.iloc[::-1].reset_index(drop=True)

        pd.testing.assert_frame_equal(
            artifact_intervals(shuffled, SPAN),
            artifact_intervals(cores, SPAN),
        )

    def test_no_cores_give_no_intervals(self):
        intervals = artifact_intervals(_cores([]), SPAN)

        assert list(intervals.columns) == ["start_time", "end_time"]
        assert intervals.empty
        assert intervals.dtypes.tolist() == [np.float64, np.float64]

    def test_masked_seconds_is_the_merged_length(self):
        intervals = artifact_intervals(_cores(self.CORES), SPAN)

        assert masked_seconds(intervals) == pytest.approx(
            0.42 + 7.0 + 0.61 + 0.35 + 0.4
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


class TestPaddingTheMask:
    """``pad_intervals`` is the one way any stage widens the mask: the decoded
    events' reach, the history windows and the peri-event windows."""

    def test_both_sides_by_one_distance(self):
        intervals = pd.DataFrame({"start_time": [10.0, 20.0], "end_time": [11.0, 20.5]})

        assert _pairs(pad_intervals(intervals, 0.3)) == [(9.7, 11.3), (19.7, 20.8)]

    def test_before_and_after_apart(self):
        """A history window looks back only: it is reached from after the mask."""
        intervals = pd.DataFrame({"start_time": [10.0], "end_time": [11.0]})

        assert _pairs(pad_intervals(intervals, 0.0, 0.15)) == [(10.0, 11.15)]
        assert _pairs(pad_intervals(intervals, 0.25, 0.0)) == [(9.75, 11.0)]

    def test_padded_intervals_that_meet_merge(self):
        intervals = pd.DataFrame({"start_time": [10.0, 11.5], "end_time": [11.0, 12.0]})

        assert _pairs(pad_intervals(intervals, 0.25)) == [(9.75, 12.25)]

    def test_no_padding_is_the_mask_merged(self):
        intervals = pd.DataFrame({"start_time": [3.0, 1.0], "end_time": [4.0, 2.0]})

        assert _pairs(pad_intervals(intervals, 0.0)) == [(1.0, 2.0), (3.0, 4.0)]

    def test_no_intervals_stay_none(self):
        empty = pd.DataFrame({"start_time": [], "end_time": []}, dtype=float)

        assert _pairs(pad_intervals(empty, 0.3)) == []

    def test_a_negative_distance_is_refused(self):
        intervals = pd.DataFrame({"start_time": [1.0], "end_time": [2.0]})

        with pytest.raises(ValueError, match="distance"):
            pad_intervals(intervals, -0.1)


class TestOverlapAndCoverage:
    def test_an_interval_that_touches_overlaps(self):
        """Closed on both sides: sharing one instant counts."""
        intervals = pd.DataFrame({"start_time": [10.0, 20.0], "end_time": [11.0, 21.0]})
        starts = np.array([8.0, 9.0, 11.0, 11.0001, 15.0, 19.0, 21.0, 25.0])
        ends = np.array([9.9, 10.0, 12.0, 12.0, 30.0, 19.9999, 22.0, 26.0])

        np.testing.assert_array_equal(
            overlaps_intervals(starts, ends, intervals),
            [False, True, True, False, True, False, True, False],
        )

    def test_overlap_matches_the_brute_force_pairs(self):
        rng = np.random.default_rng(0)
        intervals = pd.DataFrame(
            {"start_time": rng.uniform(0, 100, 30), "end_time": np.zeros(30)}
        )
        intervals["end_time"] = intervals["start_time"] + rng.uniform(0, 3, 30)
        starts = rng.uniform(0, 100, 500)
        ends = starts + rng.uniform(0, 2, 500)
        a = intervals["start_time"].to_numpy()
        b = intervals["end_time"].to_numpy()
        brute = ((starts[:, None] <= b[None]) & (a[None] <= ends[:, None])).any(axis=1)

        np.testing.assert_array_equal(
            overlaps_intervals(starts, ends, intervals), brute
        )

    def test_covered_seconds_within_each_stretch(self):
        intervals = pd.DataFrame(
            {"start_time": [2.0, 4.0, 10.0], "end_time": [3.0, 6.0, 10.0]}
        )
        starts = np.array([0.0, 2.5, 5.0, 7.0, 0.0])
        ends = np.array([1.0, 4.5, 20.0, 9.0, 20.0])

        np.testing.assert_allclose(
            covered_seconds(starts, ends, intervals), [0.0, 1.0, 1.0, 0.0, 3.0]
        )

    def test_no_intervals_cover_exactly_zero(self):
        empty = pd.DataFrame({"start_time": [], "end_time": []}, dtype=float)
        starts = np.array([0.0, 1.0])

        covered = covered_seconds(starts, starts + 1.0, empty)

        assert covered.dtype == np.float64
        np.testing.assert_array_equal(covered, [0.0, 0.0])
        assert not overlaps_intervals(starts, starts + 1.0, empty).any()
