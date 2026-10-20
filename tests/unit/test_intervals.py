"""The interval algebra (``continuum_swr_replay.intervals``).

Interval sets: closed intervals in seconds. A set is sorted and disjoint:
overlapping or touching intervals are one, and an interval of no duration is
dropped. The closed-interval frame functions, the overlap conventions, the
runs of a mask and spike binning follow, each on the edge cases the
code-organization plan names: touching endpoints, zero-length and nested
intervals, unsorted and empty input.
"""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.hippocampus.event_detection import (
    strict_overlap_matrix,
)
from continuum_swr_replay.analysis.hippocampus.event_eligibility import (
    overlaps_dropped_events,
)
from continuum_swr_replay.intervals import (
    contains,
    contiguous_true_runs,
    covered_seconds,
    get_spikecount_per_time_bin,
    intersect,
    interval_mask,
    masked_seconds,
    merge_intervals,
    normalize,
    overlaps_intervals,
    overlaps_open,
    pad_intervals,
    subtract,
    subtract_intervals,
    total_duration,
    union,
)


def s(*pairs):
    return np.array(pairs, dtype=float).reshape(-1, 2)


class TestNormalize:
    def test_sorts_and_merges_overlapping_and_touching(self):
        np.testing.assert_array_equal(
            normalize(s((5, 7), (0, 2), (2, 3), (6, 9))), s((0, 3), (5, 9))
        )

    def test_drops_intervals_of_no_duration_and_invalid_ones(self):
        np.testing.assert_array_equal(
            normalize(s((1, 1), (4, 3), (np.nan, 2), (0, 0.5))), s((0, 0.5))
        )

    def test_empty_stays_empty_with_its_shape(self):
        assert normalize(np.empty((0, 2))).shape == (0, 2)

    def test_refuses_anything_but_pairs(self):
        with pytest.raises(ValueError, match="pairs"):
            normalize(np.zeros((3, 3)))


class TestSetOperations:
    a = s((0, 4), (6, 10))
    b = s((2, 7), (9, 12))

    def test_union(self):
        np.testing.assert_array_equal(union(self.a, self.b), s((0, 12)))

    def test_intersect(self):
        np.testing.assert_array_equal(
            intersect(self.a, self.b), s((2, 4), (6, 7), (9, 10))
        )

    def test_subtract(self):
        np.testing.assert_array_equal(subtract(self.a, self.b), s((0, 2), (7, 9)))

    def test_intersecting_with_nothing_is_nothing(self):
        assert intersect(self.a, np.empty((0, 2))).shape == (0, 2)

    def test_subtracting_nothing_changes_nothing(self):
        np.testing.assert_array_equal(subtract(self.a, np.empty((0, 2))), self.a)

    def test_touching_sets_intersect_in_nothing(self):
        """A shared endpoint has no duration."""
        assert intersect(s((0, 2)), s((2, 4))).shape == (0, 2)

    def test_total_duration_counts_overlap_once(self):
        assert total_duration(s((0, 4), (2, 6), (10, 11))) == pytest.approx(7.0)

    @pytest.mark.parametrize("seed", range(20))
    def test_agree_with_a_sampled_indicator(self, seed):
        """Checked against the slow, obvious version on a fine grid."""
        rng = np.random.default_rng(seed)
        grid = np.arange(0, 100, 0.01) + 0.005  # sample centers, off every bound

        def random_set():
            starts = rng.uniform(0, 95, 6).round(1)
            return s(
                *zip(starts, starts + rng.uniform(0.1, 8, 6).round(1), strict=True)
            )

        def indicator(intervals):
            inside = np.zeros(grid.size, dtype=bool)
            for start, end in intervals:
                inside |= (grid >= start) & (grid <= end)
            return inside

        a, b = random_set(), random_set()
        for got, want in (
            (union(a, b), indicator(a) | indicator(b)),
            (intersect(a, b), indicator(a) & indicator(b)),
            (subtract(a, b), indicator(a) & ~indicator(b)),
        ):
            np.testing.assert_array_equal(indicator(got), want)
            assert total_duration(got) == pytest.approx(want.sum() * 0.01, abs=0.05)


class TestContains:
    def test_an_event_wholly_inside_one_interval(self):
        within = contains(s((0, 4), (6, 10)), np.array([1, 3, 6]), np.array([2, 7, 10]))

        assert within.tolist() == [True, False, True]

    def test_an_event_spanning_two_touching_intervals_is_inside_their_union(self):
        """Contained in the set, not in any one of the pieces it was built from."""
        within = contains(s((0, 2), (2, 4)), np.array([1.0]), np.array([3.0]))

        assert within.tolist() == [True]

    def test_nothing_is_inside_an_empty_set(self):
        assert contains(
            np.empty((0, 2)), np.array([1.0]), np.array([2.0])
        ).tolist() == [False]


def test_an_inverted_query_is_refused_by_contains():
    with pytest.raises(ValueError, match="end < start"):
        contains(np.array([[0.0, 10.0]]), np.array([4.0]), np.array([2.0]))


def test_strict_normalization_refuses_what_lenient_normalization_drops():
    inverted = np.array([[5.0, 3.0], [1.0, 2.0]])

    np.testing.assert_allclose(normalize(inverted), [[1.0, 2.0]])
    with pytest.raises(ValueError, match="inverted"):
        normalize(inverted, strict=True)
    with pytest.raises(ValueError, match="inverted"):
        normalize(np.array([[np.nan, 1.0]]), strict=True)


def frame(*pairs):
    return pd.DataFrame(
        np.array(pairs, dtype=float).reshape(-1, 2), columns=["start_time", "end_time"]
    )


class TestClosedIntervalEdgeCases:
    def test_merge_joins_touching_and_nested_and_sorts(self):
        merged = merge_intervals(
            np.array([5.0, 0.0, 2.0, 1.0]), np.array([6.0, 2.0, 3.0, 1.5])
        )
        np.testing.assert_array_equal(merged, [[0.0, 3.0], [5.0, 6.0]])

    def test_merge_keeps_a_zero_length_interval(self):
        np.testing.assert_array_equal(
            merge_intervals(np.array([4.0, 1.0]), np.array([4.0, 2.0])),
            [[1.0, 2.0], [4.0, 4.0]],
        )

    def test_merge_of_nothing_is_empty(self):
        assert merge_intervals(np.array([]), np.array([])).shape == (0, 2)

    def test_membership_is_closed_on_a_zero_length_interval(self):
        mask = interval_mask(np.array([0.9, 1.0, 1.1]), frame((1.0, 1.0)))
        assert mask.tolist() == [False, True, False]

    def test_masked_seconds_counts_nested_and_touching_once(self):
        assert masked_seconds(frame((0, 4), (1, 2), (4, 5), (7, 7))) == 5.0
        assert masked_seconds(frame()) == 0.0

    def test_padding_a_zero_length_interval_gives_it_width(self):
        padded = pad_intervals(frame((3, 3), (0, 1)), 0.5)
        np.testing.assert_array_equal(padded.to_numpy(), [[-0.5, 1.5], [2.5, 3.5]])

    def test_subtracting_a_point_splits_a_stretch_at_the_floats_beside_it(self):
        pieces = subtract_intervals(np.array([[0.0, 2.0]]), frame((1, 1)))
        np.testing.assert_array_equal(
            pieces,
            [[0.0, np.nextafter(1.0, -np.inf)], [np.nextafter(1.0, np.inf), 2.0]],
        )

    def test_closed_overlap_on_the_edge_cases(self):
        intervals = frame((5, 6), (1, 2), (2, 3), (8, 8))
        starts = np.array([0.0, 3.0, 1.2, 8.0, 7.0, 3.5])
        ends = np.array([1.0, 3.0, 1.8, 8.0, 9.0, 4.5])
        # touch at 1, touch at 3, nested, zero-length on a zero-length
        # interval, a stretch containing a zero-length interval, a gap
        assert overlaps_intervals(starts, ends, intervals).tolist() == [
            True,
            True,
            True,
            True,
            True,
            False,
        ]
        assert not overlaps_intervals(starts, ends, frame()).any()

    def test_coverage_of_a_touch_or_a_point_is_zero_seconds(self):
        covered = covered_seconds(
            np.array([2.0, 0.0, 0.0]), np.array([3.0, 4.0, 1.0]), frame((1, 2), (3, 3))
        )
        np.testing.assert_array_equal(covered, [0.0, 1.0, 0.0])


class TestTheOpenOverlapTest:
    STARTS = np.array([1.0, 4.0])
    ENDS = np.array([2.0, 4.0])

    def test_abutting_is_not_an_overlap(self):
        assert not overlaps_open(2.0, 3.0, self.STARTS, self.ENDS)
        assert not overlaps_open(0.0, 1.0, self.STARTS, self.ENDS)

    def test_a_zero_length_window_strictly_inside_overlaps(self):
        assert overlaps_open(3.5, 4.5, self.STARTS, self.ENDS)
        assert overlaps_open(1.5, 1.5, self.STARTS, self.ENDS)

    def test_nested_and_empty(self):
        assert overlaps_open(1.2, 1.8, self.STARTS, self.ENDS)
        assert not overlaps_open(1.2, 1.8, np.array([]), np.array([]))


def test_the_four_overlap_conventions_side_by_side():
    """The module docstring's table, row by row: (query, other) pairs."""
    cases = {
        "touching": ((0.0, 1.0), (1.0, 2.0)),
        "zero-length query inside": ((1.5, 1.5), (1.0, 2.0)),
        "zero-length other inside": ((1.0, 2.0), (1.5, 1.5)),
        "nested": ((1.2, 1.8), (1.0, 2.0)),
    }
    got = {}
    for name, ((s, e), (o_s, o_e)) in cases.items():
        query = (np.array([s]), np.array([e]))
        got[name] = (
            bool(overlaps_intervals(*query, frame((o_s, o_e)))[0]),
            overlaps_open(s, e, np.array([o_s]), np.array([o_e])),
            bool(strict_overlap_matrix(*query, np.array([o_s]), np.array([o_e]))[0, 0]),
            bool(overlaps_dropped_events(*query, frame((o_s, o_e)))[0]),
        )
    # (closed, open, strict, dropped-event)
    assert got == {
        "touching": (True, False, False, False),
        "zero-length query inside": (True, True, False, True),
        "zero-length other inside": (True, True, False, False),
        "nested": (True, True, True, True),
    }


class TestContiguousTrueRuns:
    def test_runs_at_the_edges_and_inside(self):
        mask = np.array([True, True, False, True, False, False, True])
        np.testing.assert_array_equal(
            contiguous_true_runs(mask), [[0, 2], [3, 4], [6, 7]]
        )

    def test_empty_all_false_and_all_true(self):
        assert contiguous_true_runs(np.array([], dtype=bool)).shape == (0, 2)
        assert contiguous_true_runs(np.zeros(3, dtype=bool)).shape == (0, 2)
        np.testing.assert_array_equal(
            contiguous_true_runs(np.ones(3, dtype=bool)), [[0, 3]]
        )
        assert contiguous_true_runs(np.ones(3, dtype=bool)).dtype == np.int64


def test_spike_binning_uses_left_edges_and_keeps_the_last_bin():
    # bins start at 0, 1, 2, 3; spikes outside [time[0], time[-1]] are dropped
    counts = get_spikecount_per_time_bin(
        np.array([-1.0, 0.0, 0.5, 1.0, 2.5, 3.0, 3.5]), np.array([0.0, 1.0, 2.0, 3.0])
    )
    assert counts.tolist() == [2, 1, 1, 1]


class TestClosedIntervalsRefuseMalformedBounds:
    """NaN bounds and start > end raise, naming the rows, in every
    closed-interval function. Before, one NaN row merged into its neighbour:
    ``merge_intervals([nan, 4], [nan, 5])`` gave ``[[4, nan]]``, hiding
    ``[4, 5]`` from overlap, coverage and subtraction."""

    BAD = {
        "nan start": (np.nan, 2.0),
        "nan end": (1.0, np.nan),
        "inverted": (3.0, 1.0),
    }

    CALLS = {
        "merge_intervals": lambda iv: merge_intervals(
            iv.start_time.to_numpy(), iv.end_time.to_numpy()
        ),
        "interval_mask": lambda iv: interval_mask(np.array([4.5]), iv),
        "subtract_intervals": lambda iv: subtract_intervals(
            np.array([[0.0, 10.0]]), iv
        ),
        "masked_seconds": masked_seconds,
        "pad_intervals": lambda iv: pad_intervals(iv, 0.1),
        "overlaps_intervals": lambda iv: overlaps_intervals(
            np.array([4.5]), np.array([4.6]), iv
        ),
        "covered_seconds": lambda iv: covered_seconds(
            np.array([4.5]), np.array([4.6]), iv
        ),
    }

    @pytest.mark.parametrize("function", sorted(CALLS))
    @pytest.mark.parametrize("bad", sorted(BAD))
    def test_every_closed_function_refuses_a_bad_interval(self, bad, function):
        intervals = frame((4.0, 5.0), self.BAD[bad])
        with pytest.raises(ValueError, match=r"NaN bound or start > end.* 1: "):
            self.CALLS[function](intervals)

    @pytest.mark.parametrize("bad", sorted(BAD))
    def test_a_bad_queried_stretch_is_refused(self, bad):
        s, e = self.BAD[bad]
        starts, ends = np.array([0.0, s]), np.array([1.0, e])
        with pytest.raises(ValueError, match="overlaps_intervals stretches"):
            overlaps_intervals(starts, ends, frame((0, 1)))
        with pytest.raises(ValueError, match="covered_seconds stretches"):
            covered_seconds(starts, ends, frame((0, 1)))
        with pytest.raises(ValueError, match="subtract_intervals stretches"):
            subtract_intervals(np.column_stack([starts, ends]), frame((0, 1)))

    def test_the_message_counts_every_bad_row_and_shows_five(self):
        starts = np.arange(8.0)
        with pytest.raises(ValueError, match=r"8 interval\(s\).*, and 3 more"):
            merge_intervals(starts, starts - 1.0)

    def test_infinite_and_zero_length_bounds_are_well_formed(self):
        np.testing.assert_array_equal(
            merge_intervals(np.array([-np.inf, 2.0]), np.array([1.0, 2.0])),
            [[-np.inf, 1.0], [2.0, 2.0]],
        )

    def test_the_open_test_and_the_interval_sets_are_unchanged(self):
        # overlaps_open moved with its semantics; the interval sets drop
        # malformed rows by design (normalize), and refuse only when strict.
        assert not overlaps_open(np.nan, 1.0, np.array([0.0]), np.array([2.0]))
        np.testing.assert_array_equal(
            normalize(np.array([[np.nan, 1.0], [3.0, 1.0], [4.0, 5.0]])), [[4.0, 5.0]]
        )
