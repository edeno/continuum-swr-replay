"""One eligibility rule for every event cohort: immobile, and clear of the mask.

An event is eligible when the whole event is immobile (``event_immobility``)
and it shares no time with the one artifact mask. A decoded event (non-local,
ContFrag label, Widloski) must also lie farther than ``DECODER_REACH_S`` from
the mask. A dropped event keeps its reason. Offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.hippocampus.event_eligibility import (
    DECODER_REACH_S,
    DropReason,
    artifact_drop_reasons,
    drop_labels,
    event_drop_reasons,
    label_drop_reasons,
)

MASK = pd.DataFrame({"start_time": [10.0], "end_time": [11.0]})
NO_MASK = pd.DataFrame({"start_time": [], "end_time": []}, dtype=float)


def test_the_decoder_reach_is_three_tenths_of_a_second():
    assert DECODER_REACH_S == 0.3


class TestArtifactReasons:
    @pytest.mark.parametrize(
        ("start", "end", "reason"),
        [
            (10.2, 10.3, DropReason.ARTIFACT),  # inside
            (9.0, 10.0, DropReason.ARTIFACT),  # touches the start
            (11.0, 11.5, DropReason.ARTIFACT),  # touches the end
            (8.0, 12.0, DropReason.ARTIFACT),  # spans it
            (9.5, 9.7, DropReason.DECODER_REACH),  # ends 0.3 s before it
            (11.25, 11.4, DropReason.DECODER_REACH),
            (11.3, 11.4, DropReason.DECODER_REACH),  # starts exactly 0.3 s after
            (11.31, 11.4, None),
            (9.0, 9.69, None),
        ],
    )
    def test_a_decoded_event(self, start, end, reason):
        reasons = artifact_drop_reasons(
            np.array([start]), np.array([end]), MASK, reach_s=DECODER_REACH_S
        )

        assert reasons.tolist() == [None if reason is None else reason.value]

    def test_an_event_from_a_detector_has_no_reach(self):
        """SWR and MUA come from the LFP and spikes, not from the decoder."""
        reasons = artifact_drop_reasons(
            np.array([9.5, 10.5]), np.array([9.7, 10.6]), MASK, reach_s=0.0
        )

        assert reasons.tolist() == [None, DropReason.ARTIFACT.value]

    def test_no_mask_drops_nothing(self):
        reasons = artifact_drop_reasons(
            np.array([10.0]), np.array([11.0]), NO_MASK, reach_s=DECODER_REACH_S
        )

        assert reasons.tolist() == [None]


class TestTheWholeRule:
    TIME = np.round(np.arange(9.0, 13.0, 0.002), 6)

    def _reasons(self, events, speed, reach_s=0.0):
        return event_drop_reasons(
            pd.DataFrame(events, columns=["start_time", "end_time"]),
            speed=speed,
            time=self.TIME,
            artifacts=MASK,
            reach_s=reach_s,
        ).tolist()

    def test_speed_reasons_as_the_immobility_rule(self):
        speed = np.zeros(self.TIME.size)
        speed[(self.TIME >= 12.0) & (self.TIME <= 12.1)] = 10.0
        speed[(self.TIME >= 12.5) & (self.TIME <= 12.51)] = np.nan
        events = [(12.05, 12.08), (12.50, 12.52), (12.301, 12.3015), (12.2, 12.3)]

        assert self._reasons(events, speed) == [
            DropReason.MOVING.value,
            DropReason.SPEED_UNKNOWN.value,
            DropReason.NO_SPEED_SAMPLE.value,
            None,
        ]

    def test_the_mask_comes_first(self):
        """An event in the mask is unobserved, whatever the speed says."""
        speed = np.full(self.TIME.size, 10.0)

        assert self._reasons([(10.5, 10.6), (11.1, 11.2)], speed, DECODER_REACH_S) == [
            DropReason.ARTIFACT.value,
            DropReason.DECODER_REACH.value,
        ]


def _frame(labels: np.ndarray, time: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {"non_local_event": labels, "head_speed": 0.0},
        index=pd.Index(time, name="time"),
    )


class TestLabels:
    TIME = np.round(np.arange(9.0, 13.0, 0.002), 6)

    def _labels(self) -> np.ndarray:
        labels = np.zeros(self.TIME.size, dtype=np.int64)
        for number, (start, end) in enumerate(
            [(9.2, 9.3), (9.6, 9.8), (10.9, 11.1), (11.5, 11.6)], start=1
        ):
            labels[(self.TIME >= start) & (self.TIME <= end)] = number
        return labels

    def test_each_label_keeps_a_reason(self):
        drops = label_drop_reasons(
            _frame(self._labels(), self.TIME),
            "non_local_event",
            MASK,
            reach_s=DECODER_REACH_S,
        )

        assert drops["non_local_event"].tolist() == [1, 2, 3, 4]
        assert drops["drop_reason"].tolist() == [
            None,
            DropReason.DECODER_REACH.value,
            DropReason.ARTIFACT.value,
            None,
        ]
        assert drops.loc[1, "start_time"] == pytest.approx(9.6)
        assert drops.loc[1, "end_time"] == pytest.approx(9.8)

    def test_a_dropped_label_is_zeroed_whole_and_the_rest_keep_their_numbers(self):
        frame = _frame(self._labels(), self.TIME)
        drops = label_drop_reasons(
            frame, "non_local_event", MASK, reach_s=DECODER_REACH_S
        )

        out = drop_labels(frame, "non_local_event", drops)

        assert sorted(np.unique(out["non_local_event"])) == [0, 1, 4]
        kept = frame["non_local_event"].isin([1, 4]).to_numpy()
        np.testing.assert_array_equal(
            out["non_local_event"].to_numpy()[kept],
            frame["non_local_event"].to_numpy()[kept],
        )
        assert (out["non_local_event"].to_numpy()[~kept] == 0).all()
        # the caller's frame is not changed
        assert sorted(np.unique(frame["non_local_event"])) == [0, 1, 2, 3, 4]

    def test_no_mask_leaves_the_frame_as_it_was(self):
        frame = _frame(self._labels(), self.TIME)
        drops = label_drop_reasons(
            frame, "non_local_event", NO_MASK, reach_s=DECODER_REACH_S
        )

        out = drop_labels(frame, "non_local_event", drops)

        assert drops["drop_reason"].isna().all()
        pd.testing.assert_frame_equal(out, frame)

    def test_a_frame_with_no_label_has_no_rows(self):
        frame = _frame(np.zeros(self.TIME.size, dtype=np.int64), self.TIME)

        drops = label_drop_reasons(frame, "non_local_event", MASK, reach_s=0.3)

        assert list(drops.columns) == [
            "non_local_event",
            "start_time",
            "end_time",
            "drop_reason",
        ]
        assert drops.empty
