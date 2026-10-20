"""Interval sets: the arithmetic every availability and overlap count rests on.

Closed intervals in seconds. A set is sorted and disjoint: overlapping or
touching intervals are one, and an interval of no duration is dropped.
"""

import numpy as np
import pytest

from continuum_swr_replay.intervals import (
    contains,
    intersect,
    normalize,
    subtract,
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
