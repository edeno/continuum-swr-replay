"""The held-out sweep's labelling of events and its summary."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.simulation import (
    Gap,
    Segment,
    piecewise_replay,
    place_field_means,
)
from scripts.detection import sweep_memoryless_detection as sweep
from scripts.detection import sweep_memoryless_heldout as heldout

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
        assert heldout.event_truth(event(*bounds), REPLAY, RECORDED) == truth

    def test_an_event_inside_a_silent_gap_is_not_replay(self):
        """The gap is inside the activity's bounds but represents nothing."""
        chain = piecewise_replay(
            [Segment(0.15, 30.0, 60.0), Gap(0.2), Segment(0.15, 90.0, 60.0)],
            2.0,
            1000.0,
            place_field_means(25, 170.0),
            60.0,
        )

        assert heldout.event_truth(event(2.2, 2.3), chain, RECORDED) == "silent_gap"
        assert heldout.event_truth(event(2.1, 2.3), chain, RECORDED) == "mixed"
        assert (
            heldout.event_truth(event(2.0, 2.15), chain, RECORDED)
            == "continuous_replay"
        )

    def test_without_an_injection_everything_is_local_or_recorded(self):
        assert heldout.event_truth(event(2.0, 2.2), None, RECORDED) == "local"
        assert heldout.event_truth(event(5.1, 5.2), None, RECORDED) == "recorded_event"


def rows(**overrides):
    base = {
        "fold": "A",
        "trial": "t",
        "detector": "state_space",
        "event": 0,
        "event_class": "ours_only",
        "truth": "continuous_replay",
        "estimator": "state_space",
        "coverage": 6.0,
        "gain_vs_animal": 1.0,
        "supported": True,
    }
    return base | overrides


class TestSummary:
    def summary(self, table):
        out = heldout.summarize(pd.DataFrame(table))
        return out.set_index(
            ["summary", "measure", "estimator", "detector", "event_class", "truth"]
        ).value

    def test_each_detector_counts_its_own_events(self):
        """A shared event is scored once per detector; neither count doubles."""
        table = [
            rows(event_class="shared", detector="state_space"),
            rows(event_class="shared", detector="sliding_window", supported=False),
        ]

        values = self.summary(table)

        assert (
            values[
                "by_class",
                "n_events",
                "state_space",
                "state_space",
                "shared",
                "continuous_replay",
            ]
            == 1
        )
        assert (
            values[
                "by_class",
                "support_rate",
                "state_space",
                "sliding_window",
                "shared",
                "continuous_replay",
            ]
            == 0
        )

    def test_a_floor_with_no_informative_event_is_written_as_none(self):
        values = self.summary([rows(coverage=3.0)])

        assert (
            values[
                "by_class",
                "n_events",
                "state_space",
                "state_space",
                "ours_only",
                "continuous_replay",
            ]
            == 0
        )
        assert np.isnan(
            values[
                "by_class",
                "support_rate",
                "state_space",
                "state_space",
                "ours_only",
                "continuous_replay",
            ]
        )
        assert (
            values[
                "by_class",
                "n_events_floor2",
                "state_space",
                "state_space",
                "ours_only",
                "continuous_replay",
            ]
            == 1
        )

    def test_false_support_counts_local_coding_and_contentless_bursts(self):
        table = [
            rows(event=0, truth="local", supported=True),
            rows(event=1, truth="no_content", supported=False),
            rows(event=2, truth="mixed", supported=True),
        ]

        values = self.summary(table)

        assert (
            values[
                "error_rates",
                "false_support_rate",
                "state_space",
                "state_space",
                "all",
                "",
            ]
            == 0.5
        )
        assert (
            values[
                "error_rates",
                "n_false_support_rate",
                "state_space",
                "state_space",
                "all",
                "",
            ]
            == 2
        )
        assert (
            values[
                "error_rates",
                "support_rate_mixed",
                "state_space",
                "state_space",
                "all",
                "",
            ]
            == 1.0
        )

    def test_agreement_compares_the_same_event_under_both_estimators(self):
        table = [
            rows(event=0, estimator="state_space", supported=True),
            rows(event=0, estimator="sliding_window", supported=True),
            rows(event=1, estimator="state_space", supported=True),
            rows(event=1, estimator="sliding_window", supported=False),
        ]

        values = self.summary(table)

        assert (
            values["agreement", "estimator_agreement", "both", "both", "all", ""] == 0.5
        )

    def test_an_event_in_a_silent_gap_counts_as_without_content(self):
        table = [
            rows(event=0, truth="silent_gap", supported=True),
            rows(event=1, truth="no_content", supported=False),
        ]

        values = self.summary(table)
        where = ("error_rates", "state_space", "state_space", "all", "")

        assert values[where[0], "n_false_support_rate", *where[1:]] == 2
        assert values[where[0], "false_support_rate", *where[1:]] == 0.5
