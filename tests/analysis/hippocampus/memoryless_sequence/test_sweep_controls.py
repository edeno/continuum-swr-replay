"""The controls' truth, geometry and per-trial scoring."""

import numpy as np
import pytest

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence import (
    sweep,
    sweep_controls,
)
from continuum_swr_replay.analysis.hippocampus.memoryless_sequence.simulation import (
    Gap,
    Segment,
    piecewise_replay,
    place_field_means,
)

MEANS = place_field_means(25, 170.0)


def chain(first_cm=90.0, second_cm=10.0, gap_s=0.1):
    return piecewise_replay(
        [Segment(0.15, first_cm, 60.0), Gap(gap_s), Segment(0.15, second_cm, 60.0)],
        at_s=1.0,
        sampling_frequency=1000.0,
        place_field_means=MEANS,
        peak_rate_hz=60.0,
    )


def event(start_s, end_s, activity, offset_cm=0.0):
    """An event whose decoded positions sit on the activity's path, plus an offset."""
    times = np.linspace(start_s, end_s, 20)
    return sweep.ReportedEvent(
        start_s, end_s, times, sweep_controls.truth_path(activity)(times) + offset_cm
    )


def state_run(activity):
    time = np.linspace(activity.bounds[0] - 0.5, activity.bounds[1] + 0.5, 800)
    return sweep.StateSpaceRun(
        time=time,
        p_continuous=np.full(time.size, 0.25),
        p_fragmented=np.full(time.size, 0.5),
        canonical=[],
        continuous_only=[],
        non_local_posterior=np.ones((time.size, 3)) / 3,
    )


class TestTruthPath:
    def test_is_silent_in_the_gap_and_held_outside_the_activity(self):
        activity = chain()
        truth = sweep_controls.truth_path(activity)

        assert np.isnan(truth([1.2]))[0]
        assert truth([0.5])[0] == pytest.approx(90.0)
        assert truth([2.0])[0] == pytest.approx(70.0)
        assert truth([1.075])[0] == pytest.approx(120.0, abs=0.5)


class TestCoveredGround:
    def test_disjoint_parts_add_up(self):
        assert sweep_controls.covered_cm(chain(90.0, 10.0)) == pytest.approx(
            120.0, abs=0.5
        )

    def test_overlapping_parts_count_their_ground_once(self):
        activity = piecewise_replay(
            sweep_controls.jump_segments(0.12, 20.0), 1.0, 1000.0, MEANS, 60.0
        )

        assert sweep_controls.covered_cm(activity) == pytest.approx(100.0, abs=0.5)


class TestJumpGeometry:
    @pytest.mark.parametrize("jump_cm", sweep_controls.JUMP_CM)
    def test_the_far_end_is_the_same_for_every_jump(self, jump_cm):
        first, second = sweep_controls.jump_segments(0.12, jump_cm)

        assert max(
            first.start_cm + first.path_cm, second.start_cm + second.path_cm
        ) == (pytest.approx(sweep_controls.FAR_END_CM))
        assert second.start_cm == pytest.approx(
            first.start_cm + first.path_cm - jump_cm
        )
        assert min(first.start_cm, second.start_cm) >= 0.0


class TestScoreTrial:
    def test_one_event_over_both_parts_spans_them(self):
        activity = chain(30.0, 90.0, gap_s=0.01)
        spanning = event(*activity.bounds, activity)
        events = {detector: [spanning] for detector in sweep_controls.DETECTORS}

        rows = sweep_controls.score_trial(activity, events, state_run(activity))

        assert [row["segmentation"] for row in rows] == ["spanning"] * 4
        assert all(row["called"] and row["n_recovered"] == 2 for row in rows)

    def test_a_part_decoded_elsewhere_is_not_recovered(self):
        """Right on the first part, over 40 cm off on the second, with more of the
        event's positions on the first, so the whole-event median stays small."""
        activity = chain(30.0, 90.0, gap_s=0.01)
        (first_start, first_end), (second_start, second_end) = activity.parts
        times = np.concatenate(
            [
                np.linspace(first_start, first_end, 40),
                np.linspace(second_start, second_end, 10),
            ]
        )
        second = times >= second_start
        decoded = np.where(
            second,
            90.0 + 60.0 * (times - second_start) / (second_end - second_start) - 45.0,
            30.0 + 60.0 * (times - first_start) / (first_end - first_start),
        )
        wrong_second = sweep.ReportedEvent(*activity.bounds, times, decoded)
        events = {detector: [wrong_second] for detector in sweep_controls.DETECTORS}

        rows = sweep_controls.score_trial(activity, events, state_run(activity))

        assert [row["n_recovered"] for row in rows] == [1] * 4
        assert {row["segmentation"] for row in rows} == {"parts_missing"}

    def test_an_event_off_the_path_is_called_but_recovers_nothing(self):
        activity = chain()
        wrong = event(*activity.bounds, activity, offset_cm=50.0)
        events = {detector: [wrong] for detector in sweep_controls.DETECTORS}

        rows = sweep_controls.score_trial(activity, events, state_run(activity))

        assert all(row["called"] and row["n_recovered"] == 0 for row in rows)
        assert {row["segmentation"] for row in rows} == {"parts_missing"}

    def test_no_event_is_not_called(self):
        activity = chain()
        events = {detector: [] for detector in sweep_controls.DETECTORS}

        rows = sweep_controls.score_trial(activity, events, state_run(activity))

        assert not any(row["called"] for row in rows)
        assert rows[0]["continuous_share"] == pytest.approx(0.25)


class TestPartAlone:
    def test_keeps_the_events_spikes_inside_the_part_and_the_recordings_outside(self):
        original = [np.array([0.5, 1.05, 1.2, 1.3, 2.0])]
        injected = [np.array([0.5, 1.01, 1.02, 1.21, 1.22, 2.0])]

        alone = sweep_controls.part_alone_spikes(original, injected, (1.0, 1.15))

        np.testing.assert_array_equal(alone[0], [0.5, 1.01, 1.02, 1.2, 1.3, 2.0])
