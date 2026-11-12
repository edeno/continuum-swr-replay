"""The sweeps' shared grid, placement stretches, decode window and published events."""

from types import SimpleNamespace

import numpy as np

from continuum_swr_replay.analysis.hippocampus.memoryless_sequence import sweep


class TestReplayGrid:
    def test_keeps_the_paths_that_lie_on_the_track_in_grid_order(self, monkeypatch):
        """170 cm track: a 30 cm path at 160 cm and a 100 cm path at 20 cm run
        off it; a 170 cm path centered at 85 cm fits exactly."""
        monkeypatch.setattr(sweep, "PATH_CM", (30.0, 100.0, 170.0))
        monkeypatch.setattr(sweep, "MIDPOINT_CM", (20.0, 60.0, 85.0, 160.0))
        monkeypatch.setattr(sweep, "DURATION_S", (0.1,))
        monkeypatch.setattr(sweep, "PEAK_RATE_HZ", (20.0, 60.0))

        assert sweep.replay_grid() == [
            (30.0, 20.0, 0.1, 20.0),
            (30.0, 20.0, 0.1, 60.0),
            (30.0, 60.0, 0.1, 20.0),
            (30.0, 60.0, 0.1, 60.0),
            (30.0, 85.0, 0.1, 20.0),
            (30.0, 85.0, 0.1, 60.0),
            (100.0, 60.0, 0.1, 20.0),
            (100.0, 60.0, 0.1, 60.0),
            (100.0, 85.0, 0.1, 20.0),
            (100.0, 85.0, 0.1, 60.0),
            (170.0, 85.0, 0.1, 20.0),
            (170.0, 85.0, 0.1, 60.0),
        ]


class TestUsableIntervals:
    def test_still_stretches_of_a_second_less_each_event_and_its_margin(self):
        """Still below 4 cm/s (4.0 itself is moving) for at least 1 s: 0-3.5 s and
        6.5-10 s; 5.0-5.5 s is too short. The event at 8.0-8.2 s cuts
        7.75-8.45 s, and the zero-width one at 2.0 s cuts 1.75-2.25 s."""
        time = np.arange(0.0, 10.01, 0.5)
        speed = np.zeros_like(time)
        speed[np.isin(time, [4.0, 4.5])] = 10.0
        speed[time == 6.0] = 4.0
        recording = {
            "time": time,
            "speed": speed,
            "event_times": np.array([[8.0, 8.2], [2.0, 2.0]]),
        }

        np.testing.assert_allclose(
            sweep.usable_intervals(recording),
            [[0.0, 1.75], [2.25, 3.5], [6.5, 7.75], [8.45, 10.0]],
        )


class TestDecodeWindow:
    RECORDING = {
        "time": np.arange(10.0),
        "position": np.arange(10.0) * 10,
        "speed": np.arange(10.0) + 0.5,
    }

    def test_takes_the_margin_either_side_inclusive(self):
        time, position, speed = sweep.decode_window(self.RECORDING, (3.0, 5.0), 1.0)

        np.testing.assert_array_equal(time, [2.0, 3.0, 4.0, 5.0, 6.0])
        np.testing.assert_array_equal(position, [20.0, 30.0, 40.0, 50.0, 60.0])
        np.testing.assert_array_equal(speed, [2.5, 3.5, 4.5, 5.5, 6.5])

    def test_without_a_margin_is_the_stretch_itself(self):
        time, _, _ = sweep.decode_window(self.RECORDING, (3.0, 5.0), margin_s=0.0)

        np.testing.assert_array_equal(time, [3.0, 4.0, 5.0])

    def test_defaults_to_the_sweeps_margin(self):
        time, _, _ = sweep.decode_window(self.RECORDING, (4.0, 4.5))

        assert sweep.DECODE_MARGIN_S == 1.0
        np.testing.assert_array_equal(time, [3.0, 4.0, 5.0])


class TestPublishedEvents:
    def test_are_the_accepted_candidates_that_pass_the_shuffle_in_order(self):
        events = {i: f"event {i}" for i in (0, 2, 3, 5)}
        run = sweep.SlidingWindowRun(
            time=np.zeros(0),
            midpoint_s=np.zeros(0),
            posterior=np.zeros((0, 0)),
            candidates=None,
            accepted=[0, 2, 3, 5],
            events=events,
            shuffles={
                0: SimpleNamespace(passes=True),
                2: SimpleNamespace(passes=False),
                3: SimpleNamespace(passes=True),
                5: SimpleNamespace(passes=False),
            },
        )

        assert sweep.published_events(run) == ["event 0", "event 3"]

    def test_none_accepted_is_none_published(self):
        run = sweep.SlidingWindowRun(
            np.zeros(0), np.zeros(0), np.zeros((0, 0)), None, [], {}, {}
        )

        assert sweep.published_events(run) == []
