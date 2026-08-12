"""Regression tests for event-gallery interval selection."""

from __future__ import annotations

import numpy as np

from scripts.figures.generate_event_gallery import _intervals_overlap_any


def test_gallery_overlap_ignores_point_intervals_on_either_side():
    """Gallery categories use the same positive-duration overlap convention."""
    starts = np.array([0.0, 1.0, 4.0])
    ends = np.array([2.0, 1.0, 6.0])
    other_starts = np.array([1.0, 5.0])
    other_ends = np.array([1.0, 7.0])

    observed = _intervals_overlap_any(starts, ends, other_starts, other_ends)

    assert observed.tolist() == [False, False, True]
