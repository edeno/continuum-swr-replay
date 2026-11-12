"""The held-out sweep's labelling of events."""

import numpy as np
import pytest

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence import sweep
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.simulation import (
    Gap,
    Segment,
    piecewise_replay,
    place_field_means,
)
from continuum_swr_replay.analysis.inventory_comparison import heldout_sweep

REPLAY = piecewise_replay(
    [Segment(0.2, 40.0, 60.0)], 2.0, 1000.0, place_field_means(25, 170.0), 60.0
)
RECORDED = np.array([[5.0, 5.3]])


def event(start_s, end_s):
    times = np.linspace(start_s, end_s, 5)
    return sweep.ReportedEvent(start_s, end_s, times, np.full(5, 50.0))


class TestEventTruth:
    @pytest.mark.parametrize(
        ("bounds", "truth"),
        [
            ((2.0, 2.2), "continuous_replay"),  # all inside
            ((1.9, 2.2), "continuous_replay"),  # two thirds inside
            ((1.6, 2.1), "mixed"),  # a fifth inside
            ((3.0, 3.2), "local"),  # nowhere near
            ((5.1, 5.2), "recorded_event"),  # on the recording's own event
        ],
    )
    def test_is_judged_by_how_much_of_the_event_lies_in_the_activity(
        self, bounds, truth
    ):
        assert heldout_sweep.event_truth(event(*bounds), REPLAY, RECORDED) == truth

    def test_an_event_inside_a_silent_gap_is_not_replay(self):
        """The gap is inside the activity's bounds but represents nothing."""
        chain = piecewise_replay(
            [Segment(0.15, 30.0, 60.0), Gap(0.2), Segment(0.15, 90.0, 60.0)],
            2.0,
            1000.0,
            place_field_means(25, 170.0),
            60.0,
        )

        assert (
            heldout_sweep.event_truth(event(2.2, 2.3), chain, RECORDED) == "silent_gap"
        )
        assert heldout_sweep.event_truth(event(2.1, 2.3), chain, RECORDED) == "mixed"
        assert (
            heldout_sweep.event_truth(event(2.0, 2.15), chain, RECORDED)
            == "continuous_replay"
        )

    def test_without_an_injection_everything_is_local_or_recorded(self):
        assert heldout_sweep.event_truth(event(2.0, 2.2), None, RECORDED) == "local"
        assert (
            heldout_sweep.event_truth(event(5.1, 5.2), None, RECORDED)
            == "recorded_event"
        )
