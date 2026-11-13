"""Tests for multiunit HSE (High Synchrony Event) detection.

Tests get_multiunit_hse — a pure numpy function with no Spyglass dependency —
and the artifact intervals ``load_data`` passes it, with Spyglass stubbed.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from ripple_detection import (
    get_multiunit_population_firing_rate,
    multiunit_HSE_detector,
)

from continuum_swr_replay.data_loaders.lfp import get_multiunit, get_multiunit_hse
from continuum_swr_replay.intervals import interval_mask, overlaps_intervals
from continuum_swr_replay.parameters import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    HSE_CLOSE_EVENT_THRESHOLD_SEC,
    HSE_MINIMUM_DURATION_SEC,
    HSE_ZSCORE_THRESHOLD,
)


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
        from continuum_swr_replay.parameters import (
            DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
        )

        # The function uses DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC internally
        # Verify the parameter value is what we expect
        assert DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC == 4.0


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
def bandit_task():
    """``data_loaders.bandit_task``; it imports Spyglass only inside its loaders."""
    return importlib.import_module("continuum_swr_replay.data_loaders.bandit_task")


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
    """The loader's closed membership rule is ``intervals.interval_mask``
    itself. Event dropping calls ``intervals.overlaps_intervals`` (the event
    overlap of the one eligibility rule), as
    ``swr_detectors.drop_overlapping_events`` does."""

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

    def test_event_overlap_is_the_detectors_and_the_sources(self):
        from continuum_swr_replay.analysis.detection.swr_detectors import (
            drop_overlapping_events,
        )
        from continuum_swr_replay.intervals import overlaps_intervals

        for intervals, events in self._cases():
            overlaps = overlaps_intervals(
                events.start_time.to_numpy(), events.end_time.to_numpy(), intervals
            )
            kept, n_dropped = drop_overlapping_events(events, intervals)
            assert n_dropped == int(overlaps.sum())
            np.testing.assert_array_equal(
                kept.to_numpy(), events.loc[~overlaps].to_numpy()
            )


# --- get_multiunit_hse before it binned with get_spikecount_per_time_bin -------
# (frozen at ba2961a5). The function as it was, copied verbatim (docstring
# dropped), with the inline digitize/bincount binning, so the rewrite's three
# outputs can be checked against it exactly.


def _frozen_get_multiunit_hse(
    spike_times: list[NDArray[np.float64]],
    speed: NDArray[np.float64],
    bin_time: NDArray[np.float64],
    artifact_intervals: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sampling_frequency = 1 / np.median(np.diff(bin_time))
    start_time, end_time = bin_time[[0, -1]]
    # ``np.digitize(t, bin_time[1:])`` assigns spikes to bin index
    # 0..N-1 where bin ``i`` covers ``[bin_time[i], bin_time[i+1])``
    # (and bin N-1 also catches ``t == bin_time[-1]``). The previous
    # ``bin_time[1:-1]`` form produced indices 0..N-2 only, so the
    # final output bin was unreachable — verified with
    # ``bin_time=[0,1,2,3]``: spikes at 2.5 and 3.0 both landed in
    # bin 2 and bin 3 stayed at zero.
    multiunit_spikes = np.stack(
        [
            np.bincount(
                np.digitize(
                    unit_spike_times[
                        np.logical_and(
                            unit_spike_times >= start_time,
                            unit_spike_times <= end_time,
                        )
                    ],
                    bin_time[1:],
                ),
                minlength=bin_time.shape[0],
            )
            for unit_spike_times in spike_times
        ],
        axis=1,
    )

    multiunit_firing_rate = pd.DataFrame(
        get_multiunit_population_firing_rate(multiunit_spikes, sampling_frequency),
        index=bin_time,
        columns=["firing_rate"],
    )
    is_artifact = (
        np.zeros(bin_time.shape, dtype=bool)
        if artifact_intervals is None
        else interval_mask(bin_time, artifact_intervals)
    )
    observed_rate = multiunit_firing_rate.to_numpy()[~is_artifact]
    mean = np.nanmean(observed_rate) if observed_rate.size else np.nan
    std = np.nanstd(observed_rate) if observed_rate.size else np.nan

    # Avoid division by near-zero if firing rate is constant
    if std < 1e-10 or np.isnan(std):
        multiunit_rate_zscore = pd.DataFrame(
            np.zeros_like(multiunit_firing_rate),
            index=bin_time,
            columns=["firing_rate"],
        )
        # A constant rate holds no HSE. ripple_detection 2.0 raises on a rate
        # with no scale, where 1.7.0 returned no events, so skip the detector.
        hse_times = pd.DataFrame(
            {"start_time": [], "end_time": [], "duration": []},
            index=pd.Index([], dtype=np.int64, name="event_number"),
            dtype=float,
        )
        return hse_times, multiunit_firing_rate, multiunit_rate_zscore

    multiunit_rate_zscore = (multiunit_firing_rate - mean) / std

    hse_times = multiunit_HSE_detector(
        bin_time,
        multiunit_spikes,
        speed,
        sampling_frequency,
        speed_threshold=DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
        minimum_duration=HSE_MINIMUM_DURATION_SEC,
        zscore_threshold=HSE_ZSCORE_THRESHOLD,
        close_event_threshold=HSE_CLOSE_EVENT_THRESHOLD_SEC,
        normalization_mask=None if not is_artifact.any() else ~is_artifact,
    )
    if artifact_intervals is not None and len(artifact_intervals):
        hse_times = hse_times.loc[
            ~overlaps_intervals(
                hse_times["start_time"].to_numpy(dtype=float),
                hse_times["end_time"].to_numpy(dtype=float),
                artifact_intervals,
            )
        ]
        hse_times.index = pd.RangeIndex(1, len(hse_times) + 1, name="event_number")

    return hse_times, multiunit_firing_rate, multiunit_rate_zscore


class TestMultiunitHSEMatchesFrozen:
    """``get_multiunit_hse``'s outputs are the frozen version's, exactly."""

    @staticmethod
    def _assert_same(spike_times, speed, bin_time, artifact_intervals=None):
        old = _frozen_get_multiunit_hse(
            spike_times, speed, bin_time, artifact_intervals
        )
        new = get_multiunit_hse(spike_times, speed, bin_time, artifact_intervals)
        for old_frame, new_frame in zip(old, new, strict=True):
            pd.testing.assert_frame_equal(new_frame, old_frame, check_exact=True)
        return new

    def test_spikes_on_bin_edges_and_at_the_ends(self):
        bin_time = np.arange(0.0, 2.0, 0.002)
        speed = np.zeros_like(bin_time)
        spike_times = [
            bin_time[[0, 1, 2, 500, 998, 999]],  # on edges, first and last
            np.array([-0.001, 0.0, 1.0, bin_time[-1], bin_time[-1] + 1e-9]),
            np.array([]),  # an empty unit
            np.array([1.0011, 0.5, 0.25, np.nan]),  # unsorted, with a NaN
        ]
        mua = self._assert_same(spike_times, speed, bin_time)[1]
        assert mua["firing_rate"].iloc[-1] > 0

    def test_simulated_sessions(self):
        rng = np.random.default_rng(3)
        bin_time = np.arange(0.0, 20.0, 0.002)
        speed = np.where((bin_time > 5.0) & (bin_time < 8.0), 20.0, 1.0)
        for n_units in (1, 5, 30):
            spike_times = [
                np.sort(rng.uniform(-0.5, 20.5, rng.poisson(400)))
                for _ in range(n_units)
            ]
            for burst in (3.0, 11.0, 16.0):
                spike_times = [
                    np.sort(np.append(times, rng.uniform(burst, burst + 0.08, 30)))
                    for times in spike_times
                ]
            artifacts = pd.DataFrame(
                {"start_time": [2.9, 15.0, 19.0], "end_time": [3.05, 15.0, 25.0]}
            )
            hse = self._assert_same(spike_times, speed, bin_time)[0]
            assert len(hse) > 0
            self._assert_same(spike_times, speed, bin_time, artifacts)

    def test_no_spikes(self):
        bin_time = np.arange(0.0, 1.0, 0.002)
        self._assert_same(
            [np.array([]), np.array([])], np.zeros_like(bin_time), bin_time
        )
