"""One analyzed-time rule for every event cohort, exposure, bin and window.

Excluded time is the artifact mask widened by ``ARTIFACT_REACH_S`` on each
side. An event of any kind is analyzed only when the whole event is immobile
(``event_definition.immobility``) and lies wholly outside excluded time; a kept event that
overlaps an SWR or MUA event dropped for excluded time is dropped too. A
dropped event keeps its reason. Offline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_definition.eligibility import (
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


# --- The speed rule before it was computed once (frozen at 974e9a8d) ---------
# ``event_drop_reasons`` applied the whole-event rule twice: with the per-event
# loop of ``immobility.filter_event_intervals_by_speed`` and with the cumsum
# counts, raising when they disagreed. These are those functions as they were,
# copied verbatim (the loop's logging dropped), so the counts alone can be
# checked against the loop.


def _frozen_validate_speed_threshold(speed_threshold: float) -> float:
    threshold = float(speed_threshold)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError(
            f"speed_threshold must be finite and positive, got {speed_threshold!r}"
        )
    return threshold


def _frozen_event_speeds_are_immobile(speed, *, speed_threshold) -> bool:
    threshold = _frozen_validate_speed_threshold(speed_threshold)
    values = np.asarray(speed, dtype=float)
    if values.ndim != 1:
        raise ValueError("speed must be one-dimensional")
    return bool(
        values.size > 0 and np.isfinite(values).all() and (values < threshold).all()
    )


def _frozen_filter_event_intervals_by_speed(
    event_times, *, speed, time, speed_threshold
):
    threshold = _frozen_validate_speed_threshold(speed_threshold)
    missing = [
        column
        for column in ("start_time", "end_time")
        if column not in event_times.columns
    ]
    if missing:
        raise KeyError(f"event_times is missing required columns: {missing}")

    time_values = np.asarray(time, dtype=float)
    speed_values = np.asarray(speed, dtype=float)
    if time_values.ndim != 1 or speed_values.ndim != 1:
        raise ValueError("time and speed must be one-dimensional")
    if len(time_values) != len(speed_values):
        raise ValueError(
            "time and speed must have equal length, "
            f"got {len(time_values)} and {len(speed_values)}"
        )
    if not np.isfinite(time_values).all():
        raise ValueError("time must contain only finite values")
    if len(time_values) > 1 and not (np.diff(time_values) > 0).all():
        raise ValueError("time must be strictly increasing")

    starts = event_times["start_time"].to_numpy(dtype=float)
    ends = event_times["end_time"].to_numpy(dtype=float)
    if not (np.isfinite(starts).all() and np.isfinite(ends).all()):
        raise ValueError("event start_time and end_time must be finite")
    if (ends < starts).any():
        raise ValueError("event end_time must be greater than or equal to start_time")

    start_indices = np.searchsorted(time_values, starts, side="left")
    end_indices = np.searchsorted(time_values, ends, side="right")
    keep = np.fromiter(
        (
            _frozen_event_speeds_are_immobile(
                speed_values[start:end], speed_threshold=threshold
            )
            for start, end in zip(start_indices, end_indices, strict=True)
        ),
        dtype=bool,
        count=len(event_times),
    )
    return event_times.loc[keep].copy()


def _frozen_event_drop_reasons(events, *, speed, time, artifacts, speed_threshold=4.0):
    time = np.asarray(time, dtype=np.float64)
    values = np.asarray(speed, dtype=np.float64)
    starts = events["start_time"].to_numpy(dtype=np.float64)
    ends = events["end_time"].to_numpy(dtype=np.float64)
    numbered = pd.DataFrame(
        {"start_time": starts, "end_time": ends}, index=np.arange(len(events))
    )
    immobile = numbered.index.isin(
        _frozen_filter_event_intervals_by_speed(
            numbered, speed=values, time=time, speed_threshold=speed_threshold
        ).index
    )
    first = np.searchsorted(time, starts, side="left")
    stop = np.searchsorted(time, ends, side="right")
    finite = np.isfinite(values)
    fast = finite & (values >= speed_threshold)

    def _in_event(flags):
        cumulative = np.concatenate([[0], np.cumsum(flags, dtype=np.int64)])
        counts = cumulative[stop] - cumulative[first]
        return counts

    reasons = np.full(len(events), None, dtype=object)
    reasons[~immobile & (_in_event(~finite) > 0)] = DropReason.SPEED_UNKNOWN.value
    reasons[~immobile & (_in_event(fast) > 0)] = DropReason.MOVING.value
    reasons[~immobile & (stop <= first)] = DropReason.NO_SPEED_SAMPLE.value
    if (pd.isna(reasons) & ~immobile).any():
        raise RuntimeError(
            "an event failed the immobility rule for no reason this module names"
        )
    excluded = artifact_drop_reasons(starts, ends, artifacts)
    in_excluded = ~pd.isna(excluded)
    reasons[in_excluded] = excluded[in_excluded]
    return reasons


class TestTheSpeedRuleMatchesTheLoop:
    """The counts alone give every reason the loop and the counts gave."""

    TIME = np.round(np.arange(9.0, 13.0, 0.002), 6)

    def _both(self, events, speed, artifacts=MASK, threshold=4.0):
        frame = pd.DataFrame(events, columns=["start_time", "end_time"], dtype=float)
        kwargs = {
            "speed": speed,
            "time": self.TIME,
            "artifacts": artifacts,
            "speed_threshold": threshold,
        }
        return (
            event_drop_reasons(frame, **kwargs).tolist(),
            _frozen_event_drop_reasons(frame, **kwargs).tolist(),
        )

    def _speed(self):
        speed = np.full(self.TIME.size, 1.0)
        speed[(self.TIME >= 12.0) & (self.TIME <= 12.1)] = 10.0  # fast
        speed[self.TIME == 12.2] = 4.0  # exactly the threshold: fast
        speed[(self.TIME >= 12.5) & (self.TIME <= 12.51)] = np.nan
        speed[self.TIME == 12.6] = np.inf
        speed[self.TIME == 12.7] = -np.inf
        speed[(self.TIME >= 12.8) & (self.TIME <= 12.81)] = np.nan
        speed[self.TIME == 12.806] = 9.0  # fast beside NaN: moving wins
        return speed

    def test_no_events(self):
        new, old = self._both(np.empty((0, 2)), self._speed())
        assert new == old == []

    @pytest.mark.parametrize("artifacts", [MASK, NO_MASK])
    def test_named_cases(self, artifacts):
        events = [
            (12.05, 12.08),  # fast inside
            (12.1, 12.15),  # starts exactly on the last fast sample
            (11.95, 12.0),  # ends exactly on the first fast sample
            (12.102, 12.198),  # exactly between fast and threshold samples: kept
            (12.198, 12.2),  # ends exactly on the threshold sample
            (12.50, 12.52),  # NaN
            (12.6, 12.6),  # zero length on an inf sample
            (12.7, 12.7),  # zero length on a -inf sample
            (12.3, 12.3),  # zero length on an ordinary sample: kept
            (12.3011, 12.3015),  # between two samples: no sample
            (12.802, 12.808),  # NaN and fast
            (8.0, 8.5),  # before the first sample
            (13.5, 14.0),  # after the last sample
            (8.0, 9.0),  # ends exactly on the first sample
            (12.998, 14.0),  # starts exactly on the last sample
            (10.5, 10.6),  # in the mask
            (11.1, 11.2),  # near it
            (9.5, 9.7),  # ends exactly at the reach
        ]
        new, old = self._both(events, self._speed(), artifacts)
        assert new == old
        if artifacts is MASK:
            assert new[:11] == [
                "moving",
                "moving",
                "moving",
                None,
                "moving",
                "speed_unknown",
                "speed_unknown",
                "speed_unknown",
                None,
                "no_speed_sample",
                "moving",
            ]
            assert new[11:] == [
                "no_speed_sample",
                "no_speed_sample",
                None,
                None,
                "artifact",
                "near_artifact",
                "near_artifact",
            ]

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_random_events(self, seed):
        rng = np.random.default_rng(seed)
        speed = rng.choice(
            [0.5, 3.9, 4.0, 7.0, np.nan],
            size=self.TIME.size,
            p=[0.9, 0.04, 0.01, 0.03, 0.02],
        )
        starts = rng.uniform(8.5, 13.5, 500)
        # a quarter of the bounds land exactly on a sample
        on_sample = rng.random(500) < 0.25
        starts[on_sample] = rng.choice(self.TIME, on_sample.sum())
        lengths = rng.choice([0.0, 0.001, 0.004, 0.02, 0.1], 500)
        events = np.column_stack([starts, starts + lengths])
        new, old = self._both(events, speed)
        assert new == old
        assert {"moving", "speed_unknown", "no_speed_sample", None} <= set(new)


class TestMalformedInputsAreRefused:
    """Each input the loop refused is still refused, with its ValueError."""

    TIME = np.arange(0.0, 1.0, 0.1)

    def _call(self, events=((0.2, 0.4),), speed=None, time=None, threshold=4.0):
        time = self.TIME if time is None else time
        speed = np.zeros(len(time)) if speed is None else speed
        return event_drop_reasons(
            pd.DataFrame(list(events), columns=["start_time", "end_time"]),
            speed=speed,
            time=time,
            artifacts=NO_MASK,
            speed_threshold=threshold,
        )

    def test_well_formed_inputs_are_read(self):
        assert self._call().tolist() == [None]

    @pytest.mark.parametrize("threshold", [0.0, -1.0, np.nan, np.inf])
    def test_a_threshold_not_finite_and_positive(self, threshold):
        with pytest.raises(ValueError, match="finite and positive"):
            self._call(threshold=threshold)

    def test_arrays_not_one_dimensional(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            self._call(speed=np.zeros((len(self.TIME), 1)))

    def test_arrays_of_unequal_length(self):
        with pytest.raises(ValueError, match="equal length"):
            self._call(speed=np.zeros(len(self.TIME) - 1))

    def test_time_not_finite(self):
        time = self.TIME.copy()
        time[3] = np.nan
        with pytest.raises(ValueError, match="only finite values"):
            self._call(time=time)

    @pytest.mark.parametrize("time", [[0.0, 0.1, 0.1, 0.3], [0.0, 0.2, 0.1, 0.3]])
    def test_time_not_strictly_increasing(self, time):
        with pytest.raises(ValueError, match="strictly increasing"):
            self._call(time=np.asarray(time))

    @pytest.mark.parametrize(
        "event", [(np.nan, 0.4), (0.2, np.nan), (0.2, np.inf), (-np.inf, 0.4)]
    )
    def test_a_bound_not_finite(self, event):
        with pytest.raises(ValueError, match="must be finite"):
            self._call(events=[event])

    def test_an_event_ending_before_it_starts(self):
        with pytest.raises(ValueError, match="greater than or equal"):
            self._call(events=[(0.2, 0.4), (0.5, 0.45)])

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"threshold": 0.0},
            {"speed": np.zeros(9)},
            {"time": np.array([0.0, 0.1, 0.1, 0.3]), "speed": np.zeros(4)},
            {"events": [(0.5, 0.45)]},
            {"events": [(np.nan, 0.4)]},
        ],
    )
    def test_the_frozen_loop_refused_the_same_inputs(self, kwargs):
        events = kwargs.pop("events", ((0.2, 0.4),))
        time = kwargs.pop("time", self.TIME)
        speed = kwargs.pop("speed", np.zeros(len(time)))
        threshold = kwargs.pop("threshold", 4.0)
        frame = pd.DataFrame(list(events), columns=["start_time", "end_time"])
        for function in (event_drop_reasons, _frozen_event_drop_reasons):
            with pytest.raises(ValueError):
                function(
                    frame,
                    speed=speed,
                    time=time,
                    artifacts=NO_MASK,
                    speed_threshold=threshold,
                )
