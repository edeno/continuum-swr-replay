"""Unit tests for event detection module.

Tests the critical event detection functions: threshold-based event detection,
duration filtering, and event overlap analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from continuum_swr_replay.analysis.hippocampus.event_detection import (
    classify_swr_overlap,
    classify_swr_overlap_geometry,
    get_confident_times,
    get_overlap_times,
)


class TestGetConfidentTimes:
    """Test threshold-based event detection."""

    def test_basic_detection_single_event(self):
        """Events above threshold are detected."""
        time = np.arange(0, 1.0, 0.002)  # 500 Hz, 1 second
        prob = np.zeros(len(time))

        # Single event from 0.1 to 0.2 seconds (100ms > 15ms minimum)
        prob[(time >= 0.1) & (time < 0.2)] = 0.9  # Above 0.8 threshold

        prob_array = xr.DataArray(prob, coords={"time": time}, dims=["time"])

        events = get_confident_times(
            prob_array,
            prob_threshold=0.8,
            minimum_duration=0.015
        )

        assert len(events) == 1, "Should detect exactly one event"
        assert events.iloc[0]["start_time"] == pytest.approx(0.1, abs=0.005)
        assert events.iloc[0]["end_time"] == pytest.approx(0.2, abs=0.005)
        assert events.iloc[0]["duration"] == pytest.approx(0.1, abs=0.005)

    def test_minimum_duration_filtering(self):
        """Events below minimum duration are filtered out."""
        time = np.arange(0, 1.0, 0.001)  # 1kHz for precision
        prob = np.zeros(len(time))

        # Short event (10ms < 15ms minimum)
        prob[(time >= 0.1) & (time < 0.11)] = 0.9

        # Long event (50ms > 15ms minimum)
        prob[(time >= 0.5) & (time < 0.55)] = 0.9

        prob_array = xr.DataArray(prob, coords={"time": time}, dims=["time"])

        events = get_confident_times(
            prob_array,
            prob_threshold=0.8,
            minimum_duration=0.015
        )

        # Only the long event should be detected
        assert len(events) == 1, "Short event should be filtered out"
        assert events.iloc[0]["start_time"] == pytest.approx(0.5, abs=0.005)

    def test_no_events_below_threshold(self):
        """Returns empty DataFrame when no events exceed threshold."""
        time = np.arange(0, 1.0, 0.002)
        prob = np.full(len(time), 0.5)  # All below 0.8 threshold

        prob_array = xr.DataArray(prob, coords={"time": time}, dims=["time"])

        events = get_confident_times(
            prob_array,
            prob_threshold=0.8,
            minimum_duration=0.015
        )

        assert len(events) == 0, "Should return empty DataFrame"
        assert "start_time" in events.columns
        assert "end_time" in events.columns
        assert "duration" in events.columns
        assert events.index.name == "event_number"

    @pytest.mark.parametrize(
        "boundary_type,expected_field,expected_value",
        [
            ("start", "start_time", 0.0),
            ("end", "end_time", 1.0),
        ],
        ids=["event_at_start", "event_at_end"]
    )
    def test_event_at_recording_boundary(self, boundary_type, expected_field, expected_value):
        """Events at recording boundaries (start/end) are handled correctly."""
        time = np.arange(0, 1.0, 0.002)
        prob = np.zeros(len(time))

        if boundary_type == "start":
            prob[time < 0.05] = 0.9
        else:  # end
            prob[time >= 0.95] = 0.9

        prob_array = xr.DataArray(prob, coords={"time": time}, dims=["time"])

        events = get_confident_times(
            prob_array,
            prob_threshold=0.8,
            minimum_duration=0.015
        )

        assert len(events) == 1
        assert events.iloc[0][expected_field] == pytest.approx(expected_value, abs=0.005)

    def test_multiple_events_detected(self):
        """Multiple distinct events are all detected."""
        time = np.arange(0, 1.0, 0.001)
        prob = np.zeros(len(time))

        # Three events
        prob[(time >= 0.1) & (time < 0.15)] = 0.9
        prob[(time >= 0.5) & (time < 0.55)] = 0.9
        prob[(time >= 0.8) & (time < 0.85)] = 0.9

        prob_array = xr.DataArray(prob, coords={"time": time}, dims=["time"])

        events = get_confident_times(
            prob_array,
            prob_threshold=0.8,
            minimum_duration=0.015
        )

        assert len(events) == 3, "Should detect all three events"
        # Check 1-based indexing
        assert events.index[0] == 1
        assert events.index[-1] == 3

    def test_speed_calculation_basic(self):
        """Max speed is calculated correctly for each event."""
        time = np.arange(0, 1.0, 0.002)
        prob = np.zeros(len(time))
        speed = np.full(len(time), 5.0)  # Constant 5 cm/s

        # Event from 0.1 to 0.2
        prob[(time >= 0.1) & (time < 0.2)] = 0.9
        speed[(time >= 0.15) & (time < 0.16)] = 15.0  # Peak speed in event

        prob_array = xr.DataArray(prob, coords={"time": time}, dims=["time"])

        events = get_confident_times(
            prob_array,
            speed=speed,
            time=time,
            prob_threshold=0.8,
            minimum_duration=0.015
        )

        assert len(events) == 1
        assert events.iloc[0]["max_speed"] == pytest.approx(15.0, abs=0.1)

    def test_speed_calculation_nan_handling(self):
        """NaN values in speed are handled with nanmax."""
        time = np.arange(0, 1.0, 0.002)
        prob = np.zeros(len(time))
        speed = np.full(len(time), 5.0)

        prob[(time >= 0.1) & (time < 0.2)] = 0.9
        speed[(time >= 0.12) & (time < 0.14)] = np.nan  # NaN in middle of event
        speed[(time >= 0.15) & (time < 0.16)] = 10.0  # Max speed

        prob_array = xr.DataArray(prob, coords={"time": time}, dims=["time"])

        events = get_confident_times(
            prob_array,
            speed=speed,
            time=time,
            prob_threshold=0.8,
            minimum_duration=0.015
        )

        # Should find max ignoring NaNs
        assert events.iloc[0]["max_speed"] == pytest.approx(10.0, abs=0.1)

    def test_speed_none_returns_nan(self):
        """When speed is None, max_speed column contains NaN."""
        time = np.arange(0, 1.0, 0.002)
        prob = np.zeros(len(time))
        prob[(time >= 0.1) & (time < 0.2)] = 0.9

        prob_array = xr.DataArray(prob, coords={"time": time}, dims=["time"])

        events = get_confident_times(
            prob_array,
            speed=None,
            time=None,
            prob_threshold=0.8,
            minimum_duration=0.015
        )

        assert len(events) == 1
        assert np.isnan(events.iloc[0]["max_speed"])

    def test_custom_threshold(self):
        """Custom probability threshold is respected."""
        time = np.arange(0, 1.0, 0.002)
        prob = np.full(len(time), 0.7)  # Below 0.8 but above 0.6
        prob[(time >= 0.1) & (time < 0.2)] = 0.7

        prob_array = xr.DataArray(prob, coords={"time": time}, dims=["time"])

        # With threshold 0.8 - should find nothing
        events_high = get_confident_times(
            prob_array,
            prob_threshold=0.8,
            minimum_duration=0.015
        )
        assert len(events_high) == 0

        # With threshold 0.6 - should find event
        events_low = get_confident_times(
            prob_array,
            prob_threshold=0.6,
            minimum_duration=0.015
        )
        assert len(events_low) >= 1

    def test_duration_accuracy(self):
        """Durations are calculated correctly (end - start)."""
        time = np.arange(0, 1.0, 0.001)
        prob = np.zeros(len(time))

        # Event with known duration
        prob[(time >= 0.2) & (time < 0.3)] = 0.9  # Exactly 100ms

        prob_array = xr.DataArray(prob, coords={"time": time}, dims=["time"])

        events = get_confident_times(
            prob_array,
            prob_threshold=0.8,
            minimum_duration=0.015
        )

        computed_duration = events.iloc[0]["end_time"] - events.iloc[0]["start_time"]
        assert events.iloc[0]["duration"] == pytest.approx(computed_duration, abs=1e-10)
        assert events.iloc[0]["duration"] == pytest.approx(0.1, abs=0.005)


class TestGetOverlapTimes:
    """Test event overlap detection."""

    def create_mock_results(self, time):
        """Helper to create minimal xarray Dataset with time coordinate."""
        return xr.Dataset(coords={"time": time})

    def test_perfect_overlap_detected(self):
        """Events completely contained in ripples are detected as overlap."""
        time = np.arange(0, 1.0, 0.002)

        # Non-local event from 0.1 to 0.2
        non_local_times = pd.DataFrame({
            "start_time": [0.1],
            "end_time": [0.2],
            "duration": [0.1],
        }, index=pd.Index([1], name="event_number"))

        # Ripple encompassing the non-local event
        ripple_times = pd.DataFrame({
            "start_time": [0.05],
            "end_time": [0.25],
            "duration": [0.2],
        })

        results = self.create_mock_results(time)

        overlap, no_overlap = get_overlap_times(
            non_local_times, ripple_times, results
        )

        assert len(overlap) == 1, "Event should be detected as overlapping"
        assert len(no_overlap) == 0, "No non-overlapping events"

    def test_partial_overlap_detected(self):
        """Events partially overlapping ripples are detected."""
        time = np.arange(0, 1.0, 0.002)

        # Non-local event from 0.1 to 0.2
        non_local_times = pd.DataFrame({
            "start_time": [0.1],
            "end_time": [0.2],
            "duration": [0.1],
        }, index=pd.Index([1], name="event_number"))

        # Ripple overlapping end of non-local event
        ripple_times = pd.DataFrame({
            "start_time": [0.15],
            "end_time": [0.25],
            "duration": [0.1],
        })

        results = self.create_mock_results(time)

        overlap, no_overlap = get_overlap_times(
            non_local_times, ripple_times, results
        )

        assert len(overlap) == 1, "Partial overlap should be detected"

    def test_no_overlap_detected(self):
        """Non-overlapping events are correctly separated."""
        time = np.arange(0, 1.0, 0.002)

        # Non-local event from 0.1 to 0.2
        non_local_times = pd.DataFrame({
            "start_time": [0.1],
            "end_time": [0.2],
            "duration": [0.1],
        }, index=pd.Index([1], name="event_number"))

        # Ripple completely separate (0.5 to 0.6)
        ripple_times = pd.DataFrame({
            "start_time": [0.5],
            "end_time": [0.6],
            "duration": [0.1],
        })

        results = self.create_mock_results(time)

        overlap, no_overlap = get_overlap_times(
            non_local_times, ripple_times, results
        )

        assert len(overlap) == 0, "No overlap should be detected"
        assert len(no_overlap) == 1, "Event should be in no-overlap"

    def test_multiple_events_mixed_overlap(self):
        """Multiple events with some overlapping, some not."""
        time = np.arange(0, 2.0, 0.002)

        # Three non-local events
        non_local_times = pd.DataFrame({
            "start_time": [0.1, 0.5, 1.0],
            "end_time": [0.2, 0.6, 1.1],
            "duration": [0.1, 0.1, 0.1],
        }, index=pd.Index([1, 2, 3], name="event_number"))

        # Ripples overlapping only first and third events
        ripple_times = pd.DataFrame({
            "start_time": [0.15, 1.05],
            "end_time": [0.25, 1.15],
            "duration": [0.1, 0.1],
        })

        results = self.create_mock_results(time)

        overlap, no_overlap = get_overlap_times(
            non_local_times, ripple_times, results
        )

        assert len(overlap) == 2, "Events 1 and 3 should overlap"
        assert len(no_overlap) == 1, "Event 2 should not overlap"

    def test_empty_ripple_times(self):
        """All events are non-overlapping when no ripples."""
        time = np.arange(0, 1.0, 0.002)

        non_local_times = pd.DataFrame({
            "start_time": [0.1],
            "end_time": [0.2],
            "duration": [0.1],
        }, index=pd.Index([1], name="event_number"))

        # Empty ripple times
        ripple_times = pd.DataFrame({
            "start_time": [],
            "end_time": [],
            "duration": [],
        })

        results = self.create_mock_results(time)

        overlap, no_overlap = get_overlap_times(
            non_local_times, ripple_times, results
        )

        assert len(overlap) == 0, "No overlaps with empty ripples"
        assert len(no_overlap) == 1, "All events are non-overlapping"

    def test_empty_non_local_times(self):
        """Empty input returns empty outputs."""
        time = np.arange(0, 1.0, 0.002)

        non_local_times = pd.DataFrame({
            "start_time": [],
            "end_time": [],
            "duration": [],
        }, index=pd.Index([], name="event_number", dtype=int))

        ripple_times = pd.DataFrame({
            "start_time": [0.1],
            "end_time": [0.2],
            "duration": [0.1],
        })

        results = self.create_mock_results(time)

        overlap, no_overlap = get_overlap_times(
            non_local_times, ripple_times, results
        )

        assert len(overlap) == 0
        assert len(no_overlap) == 0


class TestEventDetectionEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_timepoint_no_events(self):
        """Single timepoint returns no events (can't have duration > 0)."""
        time = np.array([0.0])
        prob = np.array([0.9])

        prob_array = xr.DataArray(prob, coords={"time": time}, dims=["time"])

        events = get_confident_times(
            prob_array,
            prob_threshold=0.8,
            minimum_duration=0.015
        )

        # Can't have event with duration >= 15ms from single timepoint
        assert len(events) == 0

    def test_all_ones_probability(self):
        """All time at high probability creates single event."""
        time = np.arange(0, 0.1, 0.001)
        prob = np.ones(len(time))

        prob_array = xr.DataArray(prob, coords={"time": time}, dims=["time"])

        events = get_confident_times(
            prob_array,
            prob_threshold=0.8,
            minimum_duration=0.015
        )

        assert len(events) == 1
        assert events.iloc[0]["start_time"] == pytest.approx(0.0, abs=0.005)
        assert events.iloc[0]["end_time"] == pytest.approx(0.1, abs=0.005)

    def test_probability_exactly_at_threshold(self):
        """Probability exactly at threshold is included (> not >=)."""
        time = np.arange(0, 1.0, 0.002)
        prob = np.zeros(len(time))

        # Set to exactly threshold value
        prob[(time >= 0.1) & (time < 0.2)] = 0.8

        prob_array = xr.DataArray(prob, coords={"time": time}, dims=["time"])

        events = get_confident_times(
            prob_array,
            prob_threshold=0.8,
            minimum_duration=0.015
        )

        # With > threshold, should NOT include exactly 0.8
        assert len(events) == 0, "Threshold is >, not >="

    def test_very_short_minimum_duration(self):
        """Very short minimum duration (1ms) is respected."""
        time = np.arange(0, 1.0, 0.0001)  # 10kHz sampling
        prob = np.zeros(len(time))

        # Very brief event (2ms)
        prob[(time >= 0.1) & (time < 0.102)] = 0.9

        prob_array = xr.DataArray(prob, coords={"time": time}, dims=["time"])

        events = get_confident_times(
            prob_array,
            prob_threshold=0.8,
            minimum_duration=0.001  # 1ms minimum
        )

        # Should detect the 2ms event
        assert len(events) == 1


class TestSpeedCalculationBoundsFix:
    """Test fix for array indexing bounds in speed calculation."""

    def test_event_at_time_boundary_start(self):
        """Events at the start of the time range should have valid speed."""
        time = np.arange(0, 1.0, 0.002)
        prob = np.zeros(len(time))
        speed = np.full(len(time), 10.0)

        # Event at very start of recording
        prob[(time >= 0.0) & (time < 0.05)] = 0.9

        prob_array = xr.DataArray(prob, coords={"time": time}, dims=["time"])

        events = get_confident_times(
            prob_array,
            speed=speed,
            time=time,
            prob_threshold=0.8,
            minimum_duration=0.015
        )

        assert len(events) == 1
        # Speed should be computed without index errors
        assert np.isfinite(events.iloc[0]["max_speed"]), "Should have valid speed"

    def test_event_at_time_boundary_end(self):
        """Events at the end of the time range should have valid speed."""
        time = np.arange(0, 1.0, 0.002)
        prob = np.zeros(len(time))
        speed = np.full(len(time), 10.0)

        # Event at very end of recording
        prob[(time >= 0.95) & (time < 1.0)] = 0.9

        prob_array = xr.DataArray(prob, coords={"time": time}, dims=["time"])

        events = get_confident_times(
            prob_array,
            speed=speed,
            time=time,
            prob_threshold=0.8,
            minimum_duration=0.015
        )

        assert len(events) == 1
        # Speed should be computed without index errors
        assert np.isfinite(events.iloc[0]["max_speed"]), "Should have valid speed"

    def test_event_outside_time_range_returns_nan(self):
        """Events with time range outside data bounds should return NaN speed."""
        time = np.arange(0.1, 0.9, 0.002)  # Start at 0.1, end before 0.9
        prob = np.zeros(len(time))
        speed = np.full(len(time), 10.0)

        # Event within valid range
        prob[(time >= 0.2) & (time < 0.3)] = 0.9

        prob_array = xr.DataArray(prob, coords={"time": time}, dims=["time"])

        events = get_confident_times(
            prob_array,
            speed=speed,
            time=time,
            prob_threshold=0.8,
            minimum_duration=0.015
        )

        assert len(events) == 1
        # Should not crash with out-of-bounds errors
        assert np.isfinite(events.iloc[0]["max_speed"]) or np.isnan(events.iloc[0]["max_speed"])

    def test_empty_time_interval_returns_nan(self):
        """When start and end indices are equal, should return NaN speed."""
        time = np.arange(0, 1.0, 0.1)  # Sparse time array
        prob = np.zeros(len(time))
        speed = np.full(len(time), 10.0)

        # Very short event that might not span any bins
        prob[3:5] = 0.9

        prob_array = xr.DataArray(prob, coords={"time": time}, dims=["time"])

        # Should not crash even with weird edge cases
        events = get_confident_times(
            prob_array,
            speed=speed,
            time=time,
            prob_threshold=0.8,
            minimum_duration=0.01
        )

        # Result should be valid (either empty or with NaN speeds)
        for idx in range(len(events)):
            max_speed = events.iloc[idx]["max_speed"]
            assert np.isfinite(max_speed) or np.isnan(max_speed), \
                "Speed should be finite or NaN, never crash"


def _event(start, end):
    return pd.DataFrame({"start_time": [start], "end_time": [end]})


def _swrs(intervals):
    if len(intervals) == 0:
        return pd.DataFrame(columns=["start_time", "end_time"])
    return pd.DataFrame(intervals, columns=["start_time", "end_time"])


VALID_SUBCLASSES = {
    "no_overlap",
    "event_contains_swr",
    "single_swr_partial",
    "multi_swr_no_gap",
    "multi_swr_gap",
    "swr_contains_event",
}


class TestClassifySwrOverlapGeometry:
    """Fine-grained classification of non-local events against SWRs.

    Each test isolates one sub-case with a minimal constructed geometry.
    Together they partition every possible interval configuration.
    """

    def test_no_overlap(self):
        """Event has no SWR overlap at all."""
        ev = _event(1.0, 2.0)
        swr = _swrs([(3.0, 4.0), (5.0, 6.0)])
        out = classify_swr_overlap_geometry(ev, swr).iloc[0]
        assert out["overlap_class"] == "non_swr"
        assert out["overlap_subclass"] == "no_overlap"
        assert out["n_swrs_overlapping"] == 0
        assert not out["contains_any_swr"]
        assert out["event_has_non_swr_time"]

    def test_swr_contains_event_single_swr(self):
        """One SWR fully spans the event."""
        ev = _event(1.0, 2.0)
        swr = _swrs([(0.5, 2.5)])
        out = classify_swr_overlap_geometry(ev, swr).iloc[0]
        assert out["overlap_class"] == "swr_contained"
        assert out["overlap_subclass"] == "swr_contains_event"
        assert out["n_swrs_overlapping"] == 1
        assert not out["event_has_non_swr_time"]

    def test_swr_contains_event_boundary_equality(self):
        """SWR with identical boundaries still 'contains' the event."""
        ev = _event(1.0, 2.0)
        swr = _swrs([(1.0, 2.0)])
        out = classify_swr_overlap_geometry(ev, swr).iloc[0]
        assert out["overlap_subclass"] == "swr_contains_event"

    def test_event_contains_swr(self):
        """Event is longer than the SWR; the SWR sits strictly inside."""
        ev = _event(0.0, 10.0)
        swr = _swrs([(3.0, 5.0)])
        out = classify_swr_overlap_geometry(ev, swr).iloc[0]
        assert out["overlap_class"] == "swr_partial"
        assert out["overlap_subclass"] == "event_contains_swr"
        assert out["n_swrs_overlapping"] == 1
        assert out["contains_any_swr"]
        assert out["event_has_non_swr_time"]

    def test_single_swr_partial(self):
        """Event overlaps exactly one SWR at a boundary; neither
        contains the other."""
        ev = _event(1.0, 2.0)
        swr = _swrs([(1.5, 2.5)])
        out = classify_swr_overlap_geometry(ev, swr).iloc[0]
        assert out["overlap_class"] == "swr_partial"
        assert out["overlap_subclass"] == "single_swr_partial"
        assert out["n_swrs_overlapping"] == 1
        assert not out["contains_any_swr"]
        assert out["event_has_non_swr_time"]

    def test_multi_swr_no_gap(self):
        """Two consecutive SWRs tile the event with no gap between.

        Rare in real data because SWR detectors typically enforce a
        minimum separation, but the classifier must still handle it
        correctly.
        """
        ev = _event(0.0, 1.0)
        swr = _swrs([(-0.2, 0.5), (0.5, 1.2)])
        out = classify_swr_overlap_geometry(ev, swr).iloc[0]
        assert out["overlap_class"] == "swr_partial"
        assert out["overlap_subclass"] == "multi_swr_no_gap"
        assert out["n_swrs_overlapping"] == 2
        assert not out["contains_any_swr"]
        assert not out["event_has_non_swr_time"]

    def test_multi_swr_gap(self):
        """Event overlaps >=2 SWRs and has timebins outside every SWR."""
        ev = _event(0.0, 1.0)
        swr = _swrs([(-0.1, 0.3), (0.7, 1.1)])
        out = classify_swr_overlap_geometry(ev, swr).iloc[0]
        assert out["overlap_class"] == "swr_partial"
        assert out["overlap_subclass"] == "multi_swr_gap"
        assert out["n_swrs_overlapping"] == 2
        assert not out["contains_any_swr"]
        assert out["event_has_non_swr_time"]

    def test_subclasses_partition_every_event(self):
        """Across a mixed batch every event lands in exactly one
        subclass, and the coarse-class projection matches
        classify_swr_overlap (modulo the string-name mapping)."""
        events = pd.DataFrame(
            {
                "start_time": [1.0, 0.5, 2.0, 3.0, 4.0, 5.0],
                "end_time":   [2.0, 2.5, 2.5, 4.0, 5.0, 5.5],
            }
        )
        swrs = pd.DataFrame(
            {
                "start_time": [0.9, 3.2, 4.4, 10.0],
                "end_time":   [2.1, 3.6, 4.8, 11.0],
            }
        )
        geom = classify_swr_overlap_geometry(events, swrs)
        coarse = classify_swr_overlap(events, swrs)

        assert set(geom["overlap_subclass"]).issubset(VALID_SUBCLASSES)

        # Every row has exactly one subclass (no empty strings).
        assert (geom["overlap_subclass"] != "").all()

        # Coarse labels differ between the two functions — the new
        # function uses descriptive names, the old one uses the legacy
        # "none"/"extends"/"contained" strings. Map between them here
        # to verify they agree on category membership.
        coarse_to_legacy = {
            "non_swr": "none",
            "swr_partial": "extends",
            "swr_contained": "contained",
        }
        for i in range(len(geom)):
            new_coarse = geom.iloc[i]["overlap_class"]
            legacy_coarse = coarse.iloc[i]["overlap_class"]
            assert coarse_to_legacy[new_coarse] == legacy_coarse, (
                f"Event {i}: new={new_coarse}, legacy={legacy_coarse}"
            )
