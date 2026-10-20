"""One analysed-time rule for every event cohort, exposure, bin and window.

Excluded time is the artifact mask widened by ``ARTIFACT_REACH_S`` on each
side. An event of any kind is analysed only when the whole event is immobile
(``events.immobility``) and lies wholly outside excluded time; a kept event that
overlaps an SWR or MUA event dropped for excluded time is dropped too. A
dropped event keeps its reason. Offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.events.eligibility import (
    ARTIFACT_REACH_S,
    EXCLUDED_TIME_REASONS,
    DropReason,
    artifact_drop_reasons,
    drop_labels,
    event_drop_reasons,
    excluded_time,
    label_drop_reasons,
    overlaps_dropped_events,
)

MASK = pd.DataFrame({"start_time": [10.0], "end_time": [11.0]})
NO_MASK = pd.DataFrame({"start_time": [], "end_time": []}, dtype=float)


def test_the_reach_is_three_tenths_of_a_second():
    assert ARTIFACT_REACH_S == 0.3
    assert EXCLUDED_TIME_REASONS == {"artifact", "near_artifact"}


def test_excluded_time_is_the_mask_widened_by_the_reach():
    mask = pd.DataFrame({"start_time": [10.0, 11.5], "end_time": [11.0, 12.0]})

    excluded = excluded_time(mask)

    np.testing.assert_allclose(excluded.to_numpy(), [[9.7, 12.3]])
    assert excluded_time(NO_MASK).empty


class TestArtifactReasons:
    @pytest.mark.parametrize(
        ("start", "end", "reason"),
        [
            (10.2, 10.3, DropReason.ARTIFACT),  # inside
            (9.0, 10.0, DropReason.ARTIFACT),  # touches the start
            (11.0, 11.5, DropReason.ARTIFACT),  # touches the end
            (8.0, 12.0, DropReason.ARTIFACT),  # spans it
            (9.5, 9.7, DropReason.NEAR_ARTIFACT),  # ends 0.3 s before it
            (11.25, 11.4, DropReason.NEAR_ARTIFACT),
            (11.3, 11.4, DropReason.NEAR_ARTIFACT),  # starts exactly 0.3 s after
            (11.31, 11.4, None),
            (9.0, 9.69, None),
        ],
    )
    def test_every_event_kind_has_the_one_reach(self, start, end, reason):
        reasons = artifact_drop_reasons(np.array([start]), np.array([end]), MASK)

        assert reasons.tolist() == [None if reason is None else reason.value]

    def test_no_mask_drops_nothing(self):
        reasons = artifact_drop_reasons(np.array([10.0]), np.array([11.0]), NO_MASK)

        assert reasons.tolist() == [None]


class TestOverlappingADroppedEvent:
    DROPPED = pd.DataFrame({"start_time": [5.0, 8.0], "end_time": [5.2, 8.0]})

    @pytest.mark.parametrize(
        ("start", "end", "overlaps"),
        [
            (5.1, 5.15, True),  # inside
            (4.9, 5.05, True),  # across its start
            (5.19, 5.3, True),  # across its end
            (4.0, 6.0, True),  # around it
            (5.2, 5.3, False),  # touches its end at an instant: class known
            (4.8, 5.0, False),  # touches its start
            (5.1, 5.1, True),  # a single-bin event inside it
            (7.9, 8.1, False),  # a zero-length dropped interval has no overlap
        ],
    )
    def test_positive_duration_overlap_as_is_swr(self, start, end, overlaps):
        assert overlaps_dropped_events(
            np.array([start]), np.array([end]), self.DROPPED
        ).tolist() == [overlaps]

    def test_matches_the_brute_force_pairs(self):
        rng = np.random.default_rng(1)
        dropped = pd.DataFrame({"start_time": rng.uniform(0, 100, 40)})
        dropped["end_time"] = dropped["start_time"] + rng.uniform(0, 2, 40)
        starts = rng.uniform(0, 100, 400)
        ends = starts + rng.uniform(0, 1, 400)
        a, b = dropped.start_time.to_numpy(), dropped.end_time.to_numpy()
        brute = (
            (b[None] > starts[:, None]) & (a[None] < ends[:, None]) & (b > a)[None]
        ).any(axis=1)

        np.testing.assert_array_equal(
            overlaps_dropped_events(starts, ends, dropped), brute
        )


class TestTheWholeRule:
    TIME = np.round(np.arange(9.0, 13.0, 0.002), 6)

    def _reasons(self, events, speed):
        return event_drop_reasons(
            pd.DataFrame(events, columns=["start_time", "end_time"]),
            speed=speed,
            time=self.TIME,
            artifacts=MASK,
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

    def test_excluded_time_comes_first(self):
        """An event in excluded time is unobserved, whatever the speed says."""
        speed = np.full(self.TIME.size, 10.0)

        assert self._reasons([(10.5, 10.6), (11.1, 11.2)], speed) == [
            DropReason.ARTIFACT.value,
            DropReason.NEAR_ARTIFACT.value,
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
            _frame(self._labels(), self.TIME), "non_local_event", MASK
        )

        assert drops["non_local_event"].tolist() == [1, 2, 3, 4]
        assert drops["drop_reason"].tolist() == [
            None,
            DropReason.NEAR_ARTIFACT.value,
            DropReason.ARTIFACT.value,
            None,
        ]
        assert drops.loc[1, "start_time"] == pytest.approx(9.6)
        assert drops.loc[1, "end_time"] == pytest.approx(9.8)

    def test_a_dropped_label_is_zeroed_whole_and_the_rest_keep_their_numbers(self):
        frame = _frame(self._labels(), self.TIME)
        drops = label_drop_reasons(frame, "non_local_event", MASK)

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
        drops = label_drop_reasons(frame, "non_local_event", NO_MASK)

        out = drop_labels(frame, "non_local_event", drops)

        assert drops["drop_reason"].isna().all()
        pd.testing.assert_frame_equal(out, frame)

    def test_a_frame_with_no_label_has_no_rows(self):
        frame = _frame(np.zeros(self.TIME.size, dtype=np.int64), self.TIME)

        drops = label_drop_reasons(frame, "non_local_event", MASK)

        assert list(drops.columns) == [
            "non_local_event",
            "start_time",
            "end_time",
            "drop_reason",
        ]
        assert drops.empty


def test_a_cohort_read_outside_the_source_takes_the_overlap_pass_too():
    """Events re-detected from a posterior (or Widloski's) get the same pass
    against the SWR and MUA events dropped for excluded time."""
    dropped = pd.DataFrame({"start_time": [9.5], "end_time": [9.75]})
    starts, ends = np.array([9.4, 10.2, 8.0]), np.array([9.55, 10.3, 8.5])

    reasons = artifact_drop_reasons(starts, ends, MASK, dropped)

    assert reasons.tolist() == [
        DropReason.OVERLAPS_DROPPED_EVENT.value,
        DropReason.ARTIFACT.value,
        None,
    ]
