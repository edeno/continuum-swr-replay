"""Tests for multiunit HSE (High Synchrony Event) detection.

Tests get_multiunit_hse — a pure numpy function with no Spyglass dependency —
and the artifact intervals ``load_data`` passes it, with Spyglass stubbed.
"""

from __future__ import annotations

import importlib
import sys
import types

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.data_loaders.lfp import get_multiunit, get_multiunit_hse


class TestGetMultiunitHSE:
    """Test HSE detection from multiunit spike data."""

    @pytest.fixture
    def bin_time(self):
        """2 seconds at 500 Hz."""
        return np.arange(0.0, 2.0, 0.002)

    @pytest.fixture
    def low_speed(self, bin_time):
        """Stationary animal (speed below threshold)."""
        return np.full(len(bin_time), 1.0)

    @pytest.fixture
    def high_speed(self, bin_time):
        """Moving animal (speed above threshold)."""
        return np.full(len(bin_time), 10.0)

    def test_returns_three_dataframes(self, bin_time, low_speed):
        """Function should return (hse_times, firing_rate, zscore)."""
        rng = np.random.RandomState(42)
        spike_times = [np.sort(rng.uniform(0, 2, 50)) for _ in range(5)]

        hse_times, firing_rate, zscore = get_multiunit_hse(
            spike_times, low_speed, bin_time
        )

        assert isinstance(hse_times, pd.DataFrame)
        assert isinstance(firing_rate, pd.DataFrame)
        assert isinstance(zscore, pd.DataFrame)

    def test_firing_rate_shape_matches_bin_time(self, bin_time, low_speed):
        """Firing rate should have same length as bin_time."""
        rng = np.random.RandomState(42)
        spike_times = [np.sort(rng.uniform(0, 2, 50)) for _ in range(5)]

        _, firing_rate, zscore = get_multiunit_hse(spike_times, low_speed, bin_time)

        assert len(firing_rate) == len(bin_time)
        assert len(zscore) == len(bin_time)

    def test_firing_rate_index_is_bin_time(self, bin_time, low_speed):
        """Firing rate index should match bin_time values."""
        rng = np.random.RandomState(42)
        spike_times = [np.sort(rng.uniform(0, 2, 50)) for _ in range(5)]

        _, firing_rate, _ = get_multiunit_hse(spike_times, low_speed, bin_time)

        np.testing.assert_array_equal(firing_rate.index.to_numpy(), bin_time)

    def test_no_spikes_gives_zero_firing_rate(self, bin_time, low_speed):
        """Empty spike arrays should give zero firing rate."""
        spike_times = [np.array([]) for _ in range(5)]

        _, firing_rate, zscore = get_multiunit_hse(spike_times, low_speed, bin_time)

        assert firing_rate.values.max() == 0.0

    def test_constant_firing_gives_zero_zscore(self, bin_time, low_speed):
        """Constant firing rate should give zero z-score (std=0 guard)."""
        spike_times = [np.array([]) for _ in range(5)]

        _, _, zscore = get_multiunit_hse(spike_times, low_speed, bin_time)

        np.testing.assert_array_equal(zscore.values, 0.0)

    def test_no_spikes_gives_no_hse(self, bin_time, low_speed):
        """A spike-free area has a constant rate, so it holds no HSE.

        ripple_detection 2.0 raises on a rate with no scale, so the detector
        is skipped and an empty table comes back, as 1.7.0 returned.
        """
        spike_times = [np.array([]) for _ in range(5)]

        hse_times, _, _ = get_multiunit_hse(spike_times, low_speed, bin_time)

        assert len(hse_times) == 0
        assert list(hse_times.columns) == ["start_time", "end_time", "duration"]
        assert hse_times.index.name == "event_number"

    def test_high_speed_suppresses_hse_detection(self, bin_time, high_speed):
        """HSE events should not be detected when animal is moving fast."""
        rng = np.random.RandomState(42)
        # Create a burst of synchronous spikes
        burst_time = 1.0
        spike_times = [
            np.sort(
                np.concatenate(
                    [
                        rng.uniform(0, 2, 20),
                        np.full(10, burst_time) + rng.randn(10) * 0.001,
                    ]
                )
            )
            for _ in range(10)
        ]

        hse_times, _, _ = get_multiunit_hse(spike_times, high_speed, bin_time)

        # High speed should suppress detection
        assert len(hse_times) == 0

    def test_hse_times_has_start_end_columns(self, bin_time, low_speed):
        """HSE times DataFrame should have start_time and end_time columns."""
        rng = np.random.RandomState(42)
        spike_times = [np.sort(rng.uniform(0, 2, 50)) for _ in range(5)]

        hse_times, _, _ = get_multiunit_hse(spike_times, low_speed, bin_time)

        assert "start_time" in hse_times.columns
        assert "end_time" in hse_times.columns

    def test_hse_times_within_data_bounds(self, bin_time, low_speed):
        """Detected HSE events should be within the bin_time range."""
        rng = np.random.RandomState(42)
        # Lots of spikes to increase chance of HSE detection
        spike_times = [np.sort(rng.uniform(0, 2, 200)) for _ in range(10)]

        hse_times, _, _ = get_multiunit_hse(spike_times, low_speed, bin_time)

        if len(hse_times) > 0:
            assert hse_times["start_time"].min() >= bin_time[0]
            assert hse_times["end_time"].max() <= bin_time[-1]

    def test_spikes_outside_bin_time_ignored(self, bin_time, low_speed):
        """Spikes outside the bin_time range should not affect results."""
        rng = np.random.RandomState(42)
        # Spikes within range
        spike_times_in = [np.sort(rng.uniform(0, 2, 50)) for _ in range(5)]
        # Add spikes far outside range
        spike_times_out = [
            np.concatenate([st, rng.uniform(100, 200, 50)]) for st in spike_times_in
        ]

        _, rate_in, _ = get_multiunit_hse(spike_times_in, low_speed, bin_time)
        _, rate_out, _ = get_multiunit_hse(spike_times_out, low_speed, bin_time)

        # Firing rates should be identical since out-of-range spikes are filtered
        np.testing.assert_array_equal(rate_in.values, rate_out.values)

    def test_single_unit_works(self, bin_time, low_speed):
        """Single unit (list of 1 array) should work without error."""
        rng = np.random.RandomState(42)
        spike_times = [np.sort(rng.uniform(0, 2, 100))]

        hse_times, firing_rate, zscore = get_multiunit_hse(
            spike_times, low_speed, bin_time
        )

        assert len(firing_rate) == len(bin_time)

    def test_zscore_has_zero_mean_unit_variance(self, bin_time, low_speed):
        """Z-scored firing rate should have approximately zero mean and unit variance."""
        rng = np.random.RandomState(42)
        # Need enough spikes for non-trivial distribution
        spike_times = [np.sort(rng.uniform(0, 2, 500)) for _ in range(10)]

        _, _, zscore = get_multiunit_hse(spike_times, low_speed, bin_time)

        values = zscore.values.flatten()
        # Z-score should be close to mean=0, std=1
        assert abs(np.nanmean(values)) < 0.1
        assert abs(np.nanstd(values) - 1.0) < 0.1

    def test_uses_max_speed_threshold_from_parameters(self):
        """Verify speed_threshold matches the centralized parameter."""
        from continuum_swr_replay.parameters import MAX_SPEED_THRESHOLD_CM_PER_SEC

        # The function uses MAX_SPEED_THRESHOLD_CM_PER_SEC internally
        # Verify the parameter value is what we expect
        assert MAX_SPEED_THRESHOLD_CM_PER_SEC == 4.0


class TestArtifactIntervals:
    """MUA detection leaves the artifact intervals (the artifact mask) out of
    its normalization and drops the events that overlap them."""

    BURSTS_S = (5.0, 10.0)
    """Seconds: two synchronous bursts, real HSEs."""

    LINE_S = 15.0
    """Seconds: a spike line, 300 spikes per unit in 50 ms, an artifact."""

    ARTIFACT = pd.DataFrame({"start_time": [14.8], "end_time": [15.3]})
    """The spike line padded, as the artifact mask pads one."""

    @pytest.fixture
    def bin_time(self):
        """20 seconds at 500 Hz."""
        return np.arange(0.0, 20.0, 0.002)

    @pytest.fixture
    def speed(self, bin_time):
        return np.full(len(bin_time), 1.0)

    @staticmethod
    def units() -> list[np.ndarray]:
        """Ten units: 5 Hz background, the two bursts, and the spike line."""
        rng = np.random.default_rng(0)
        units = []
        for _ in range(10):
            background = rng.uniform(0, 20, 100)
            bursts = np.concatenate(
                [at + rng.normal(0, 0.01, 4) for at in TestArtifactIntervals.BURSTS_S]
            )
            line = TestArtifactIntervals.LINE_S + rng.uniform(0, 0.05, 300)
            units.append(np.sort(np.concatenate([background, bursts, line])))
        return units

    @pytest.fixture
    def spike_times(self):
        return self.units()

    @staticmethod
    def _found(hse_times: pd.DataFrame, at: float) -> bool:
        return bool(((hse_times.start_time <= at) & (hse_times.end_time >= at)).any())

    def test_the_artifact_hides_the_real_events_unmasked(
        self, spike_times, speed, bin_time
    ):
        hse_times, _, _ = get_multiunit_hse(spike_times, speed, bin_time)

        assert self._found(hse_times, self.LINE_S)
        assert not any(self._found(hse_times, at) for at in self.BURSTS_S)

    def test_masked_the_real_events_are_found_and_the_artifact_dropped(
        self, spike_times, speed, bin_time
    ):
        hse_times, _, zscore = get_multiunit_hse(
            spike_times, speed, bin_time, artifact_intervals=self.ARTIFACT
        )

        assert all(self._found(hse_times, at) for at in self.BURSTS_S)
        overlaps = (hse_times.start_time <= 15.3) & (hse_times.end_time >= 14.8)
        assert not overlaps.any()
        np.testing.assert_array_equal(hse_times.index, np.arange(1, len(hse_times) + 1))
        assert hse_times.index.name == "event_number"
        # the z-score is normalized over the unmasked bins, as the detector is
        outside = ~((bin_time >= 14.8) & (bin_time <= 15.3))
        values = zscore["firing_rate"].to_numpy()
        assert abs(values[outside].mean()) < 1e-9
        assert values[outside].std() == pytest.approx(1.0, abs=1e-9)

    def test_an_event_touching_an_interval_is_dropped(
        self, spike_times, speed, bin_time
    ):
        """Closed intervals: an interval ending on an event's first bin drops it."""
        found, _, _ = get_multiunit_hse(
            spike_times, speed, bin_time, artifact_intervals=self.ARTIFACT
        )
        first = found.loc[found.start_time < 6.0].iloc[0]
        touching = pd.concat(
            [
                self.ARTIFACT,
                pd.DataFrame(
                    {
                        "start_time": [first.start_time - 0.5],
                        "end_time": [first.start_time],
                    }
                ),
            ],
            ignore_index=True,
        )

        hse_times, _, _ = get_multiunit_hse(
            spike_times, speed, bin_time, artifact_intervals=touching
        )

        assert not self._found(hse_times, self.BURSTS_S[0])
        assert self._found(hse_times, self.BURSTS_S[1])

    def test_no_intervals_is_no_mask(self, spike_times, speed, bin_time):
        empty = pd.DataFrame({"start_time": [], "end_time": []}, dtype=float)

        masked = get_multiunit_hse(
            spike_times, speed, bin_time, artifact_intervals=empty
        )
        unmasked = get_multiunit_hse(spike_times, speed, bin_time)

        for got, expected in zip(masked, unmasked, strict=True):
            pd.testing.assert_frame_equal(got, expected)

    def test_get_multiunit_masks_only_the_areas_given(
        self, spike_times, speed, bin_time
    ):
        position_info = pd.DataFrame({"head_speed": speed}, index=bin_time)

        multiunit = get_multiunit(
            {"HPC": spike_times, "mPFC": spike_times},
            position_info,
            artifact_intervals={"HPC": self.ARTIFACT},
        )

        hpc, mpfc = multiunit["hse_times"]["HPC"], multiunit["hse_times"]["mPFC"]
        assert all(self._found(hpc, at) for at in self.BURSTS_S)
        assert not self._found(hpc, self.LINE_S)
        assert self._found(mpfc, self.LINE_S)


@pytest.fixture
def bandit_task(monkeypatch):
    """``bandit_task``, imported offline with ``spyglass.common`` stubbed, as
    ``test_position.py`` imports it; forgotten afterwards if imported here."""
    common = types.ModuleType("spyglass.common")
    common.PositionIntervalMap = type("PositionIntervalMap", (), {})
    monkeypatch.setitem(sys.modules, "spyglass", types.ModuleType("spyglass"))
    monkeypatch.setitem(sys.modules, "spyglass.common", common)

    name = "continuum_swr_replay.data_loaders.bandit_task"
    fresh = name not in sys.modules
    yield importlib.import_module(name)
    if fresh:
        sys.modules.pop(name, None)
        vars(sys.modules["continuum_swr_replay.data_loaders"]).pop("bandit_task", None)


def test_load_data_masks_the_mua_of_the_areas_it_is_given(bandit_task, monkeypatch):
    """``load_data``'s HSEs leave out the artifact intervals given for an area."""
    case = TestArtifactIntervals
    bin_time = np.arange(0.0, 20.0, 0.002)
    units = case.units()
    monkeypatch.setattr(
        bandit_task,
        "get_epoch_position_info",
        lambda nwb_file_name, epoch_name: {
            "position_info": pd.DataFrame(
                {"head_speed": np.full(bin_time.size, 1.0)},
                index=pd.Index(bin_time, name="time"),
            ),
            "track_graph": None,
            "linear_edge_order": None,
            "linear_edge_spacing": None,
        },
    )
    monkeypatch.setattr(
        bandit_task,
        "get_spike_data",
        lambda nwb_file_name: {
            "spike_times": {"HPC": units},
            "spike_waveform_features": {"HPC": [np.zeros((u.size, 4)) for u in units]},
        },
    )

    data = bandit_task.load_data(
        "sim20200101_.nwb",
        "02_r1",
        include=frozenset({"hse_times"}),
        mua_artifact_intervals={"HPC": case.ARTIFACT},
    )

    hse_times = data["hse_times"]["HPC"]
    assert all(case._found(hse_times, at) for at in case.BURSTS_S)
    assert not case._found(hse_times, case.LINE_S)


class TestTheLoadersIntervalRulesAreTheMasks:
    """``data_loaders`` may not import ``analysis``, so the loader keeps its own
    copies of the closed-interval rules. These tie them to the originals:
    ``intervals.interval_mask`` for membership, and
    ``swr_inventories.drop_overlapping_events`` and ``intervals.overlaps_intervals``
    (the event overlap of the one eligibility rule) for dropping."""

    TIME = np.round(np.arange(0.0, 2.0, 0.001), 3)
    """A 1 kHz grid, so interval and event bounds land on samples exactly."""

    @staticmethod
    def _cases() -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """(intervals, events) pairs: random, unsorted and overlapping
        intervals; zero-length ones; bounds on the grid's samples, at its
        edges, and events touching intervals at an instant; and no rows."""
        rng = np.random.default_rng(7)
        grid = TestTheLoadersIntervalRulesAreTheMasks.TIME
        cases = []
        for _ in range(200):
            n_intervals, n_events = rng.integers(0, 6), rng.integers(0, 8)
            starts = rng.choice(grid, n_intervals)
            ends = starts + rng.choice([0.0, 0.001, 0.05, 0.3], n_intervals)
            event_starts = rng.choice(grid, n_events)
            event_ends = event_starts + rng.choice([0.0, 0.002, 0.1], n_events)
            cases.append(
                (
                    pd.DataFrame({"start_time": starts, "end_time": ends}),
                    pd.DataFrame({"start_time": event_starts, "end_time": event_ends}),
                )
            )
        touching = pd.DataFrame({"start_time": [0.5, 0.0], "end_time": [0.7, 0.0]})
        cases.append(
            (
                touching,
                pd.DataFrame(
                    {
                        "start_time": [0.7, 0.3, 0.71, 0.0, 1.999],
                        "end_time": [0.8, 0.5, 0.9, 0.0, 1.999],
                    }
                ),
            )
        )
        cases.append((pd.DataFrame({"start_time": [], "end_time": []}), cases[0][1]))
        return cases

    def test_membership_is_interval_mask(self):
        from continuum_swr_replay.data_loaders.lfp import _in_intervals
        from continuum_swr_replay.intervals import interval_mask

        for intervals, _ in self._cases():
            np.testing.assert_array_equal(
                _in_intervals(self.TIME, intervals), interval_mask(self.TIME, intervals)
            )

    def test_event_overlap_is_the_detectors_and_the_sources(self):
        from continuum_swr_replay.analysis.hippocampus.swr_inventories import (
            drop_overlapping_events,
        )
        from continuum_swr_replay.data_loaders.lfp import _overlap_any
        from continuum_swr_replay.intervals import overlaps_intervals

        for intervals, events in self._cases():
            overlaps = _overlap_any(events, intervals)
            kept, n_dropped = drop_overlapping_events(events, intervals)
            assert n_dropped == int(overlaps.sum())
            np.testing.assert_array_equal(
                kept.to_numpy(), events.loc[~overlaps].to_numpy()
            )
            if len(events) and len(intervals):
                np.testing.assert_array_equal(
                    overlaps,
                    overlaps_intervals(
                        events.start_time.to_numpy(),
                        events.end_time.to_numpy(),
                        intervals,
                    ),
                )
