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
    merged_interval_duration,
)


def _prob_array(
    events: list[tuple[float, float]],
    *,
    dt: float = 0.002,
    duration: float = 1.0,
    event_value: float = 0.9,
    base_value: float = 0.0,
) -> tuple[xr.DataArray, np.ndarray]:
    """Build the (prob_array, time) pair fed into ``get_confident_times``.

    ``events`` is a list of ``(start, end)`` windows; each is set to
    ``event_value``. Outside the windows, the array sits at ``base_value``.
    Returns both the DataArray and the time axis since several tests need
    the raw ``time`` array (e.g. for ``speed`` alignment).
    """
    time = np.arange(0, duration, dt)
    prob = np.full(len(time), base_value)
    for start, end in events:
        prob[(time >= start) & (time < end)] = event_value
    return xr.DataArray(prob, coords={"time": time}, dims=["time"]), time


class TestGetConfidentTimes:
    """Test threshold-based event detection."""

    def test_basic_detection_single_event(self):
        """Events above threshold are detected."""
        dt = 0.002  # 500 Hz
        # Single event from 0.1 to 0.2 seconds (100ms > 15ms minimum).
        prob_array, _ = _prob_array([(0.1, 0.2)], dt=dt)

        events = get_confident_times(
            prob_array,
            prob_threshold=0.8,
            minimum_duration=0.015
        )

        # Tolerance is one bin (dt) — tighter tolerances would conflate
        # bin-edge rounding with sampling-rate drift bugs.
        assert len(events) == 1, "Should detect exactly one event"
        assert events.iloc[0]["start_time"] == pytest.approx(0.1, abs=dt + 1e-9)
        assert events.iloc[0]["end_time"] == pytest.approx(0.2, abs=dt + 1e-9)
        assert events.iloc[0]["duration"] == pytest.approx(0.1, abs=dt + 1e-9)

    def test_minimum_duration_filtering(self):
        """Events below minimum duration are filtered out."""
        dt = 0.001  # 1 kHz for precision
        # Short event (10ms < 15ms minimum) and long event (50ms > 15ms).
        prob_array, _ = _prob_array(
            [(0.1, 0.11), (0.5, 0.55)], dt=dt
        )

        events = get_confident_times(
            prob_array,
            prob_threshold=0.8,
            minimum_duration=0.015
        )

        # Only the long event should be detected
        assert len(events) == 1, "Short event should be filtered out"
        assert events.iloc[0]["start_time"] == pytest.approx(0.5, abs=dt + 1e-9)

    def test_no_events_below_threshold(self):
        """Returns empty DataFrame when no events exceed threshold."""
        # Whole recording at 0.5 — below 0.8 threshold everywhere.
        prob_array, _ = _prob_array([], dt=0.002, base_value=0.5)

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
        dt = 0.002
        window = (0.0, 0.05) if boundary_type == "start" else (0.95, 1.0 + dt)
        prob_array, _ = _prob_array([window], dt=dt)

        events = get_confident_times(
            prob_array,
            prob_threshold=0.8,
            minimum_duration=0.015
        )

        assert len(events) == 1
        assert events.iloc[0][expected_field] == pytest.approx(expected_value, abs=dt + 1e-9)

    def test_multiple_events_detected(self):
        """Multiple distinct events are all detected."""
        prob_array, _ = _prob_array(
            [(0.1, 0.15), (0.5, 0.55), (0.8, 0.85)],
            dt=0.001,
        )

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
        prob_array, time = _prob_array([(0.1, 0.2)], dt=0.002)
        speed = np.full(len(time), 5.0)  # Constant 5 cm/s baseline
        speed[(time >= 0.15) & (time < 0.16)] = 15.0  # Peak speed in event

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
        prob_array, time = _prob_array([(0.1, 0.2)], dt=0.002)
        speed = np.full(len(time), 5.0)
        speed[(time >= 0.12) & (time < 0.14)] = np.nan  # NaN in middle of event
        speed[(time >= 0.15) & (time < 0.16)] = 10.0  # Max speed

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
        prob_array, _ = _prob_array([(0.1, 0.2)], dt=0.002)

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
        dt = 0.001
        # Event with known duration (exactly 100 ms).
        prob_array, _ = _prob_array([(0.2, 0.3)], dt=dt)

        events = get_confident_times(
            prob_array,
            prob_threshold=0.8,
            minimum_duration=0.015
        )

        computed_duration = events.iloc[0]["end_time"] - events.iloc[0]["start_time"]
        assert events.iloc[0]["duration"] == pytest.approx(computed_duration, abs=1e-10)
        assert events.iloc[0]["duration"] == pytest.approx(0.1, abs=dt + 1e-9)


class TestGetOverlapTimes:
    """Test event overlap detection."""

    def test_perfect_overlap_detected(self):
        """Events completely contained in ripples are detected as overlap."""
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

        overlap, no_overlap = get_overlap_times(
            non_local_times, ripple_times
        )

        assert len(overlap) == 1, "Event should be detected as overlapping"
        assert len(no_overlap) == 0, "No non-overlapping events"

    def test_partial_overlap_detected(self):
        """Events partially overlapping ripples are detected."""

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

        overlap, no_overlap = get_overlap_times(
            non_local_times, ripple_times
        )

        assert len(overlap) == 1, "Partial overlap should be detected"

    def test_no_overlap_detected(self):
        """Non-overlapping events are correctly separated."""

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

        overlap, no_overlap = get_overlap_times(
            non_local_times, ripple_times
        )

        assert len(overlap) == 0, "No overlap should be detected"
        assert len(no_overlap) == 1, "Event should be in no-overlap"

    def test_multiple_events_mixed_overlap(self):
        """Multiple events with some overlapping, some not."""

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

        overlap, no_overlap = get_overlap_times(
            non_local_times, ripple_times
        )

        assert len(overlap) == 2, "Events 1 and 3 should overlap"
        assert len(no_overlap) == 1, "Event 2 should not overlap"

    def test_boundary_touch_at_event_end_is_not_overlap(self):
        """Regression: a ripple starting exactly at the non-local
        event's end shares zero positive duration with it. Pre-fix,
        ``get_labels`` labeled the boundary timebin as inside both
        intervals (inclusive) and ``get_overlap_times`` reported the
        event as overlapping; post-fix uses strict positive-duration
        intersection."""
        non_local_times = pd.DataFrame({
            "start_time": [1.0],
            "end_time": [2.0],
            "duration": [1.0],
        }, index=pd.Index([1], name="event_number"))
        ripple_times = pd.DataFrame({
            "start_time": [2.0],
            "end_time": [3.0],
            "duration": [1.0],
        })
        overlap, no_overlap = get_overlap_times(
            non_local_times, ripple_times
        )
        assert len(overlap) == 0, (
            f"Boundary-touch ripple was reported as overlap "
            f"(len(overlap)={len(overlap)}); ripple [2.0, 3.0] only "
            "touches event [1.0, 2.0] at t=2.0 and shares zero "
            "positive duration."
        )
        assert len(no_overlap) == 1

    def test_boundary_touch_at_event_start_is_not_overlap(self):
        """Mirror case: a ripple ending exactly at the non-local
        event's start time."""
        non_local_times = pd.DataFrame({
            "start_time": [1.0],
            "end_time": [2.0],
            "duration": [1.0],
        }, index=pd.Index([1], name="event_number"))
        ripple_times = pd.DataFrame({
            "start_time": [-1.0],
            "end_time": [1.0],
            "duration": [2.0],
        })
        overlap, no_overlap = get_overlap_times(
            non_local_times, ripple_times
        )
        assert len(overlap) == 0
        assert len(no_overlap) == 1

    def test_empty_ripple_times(self):
        """All events are non-overlapping when no ripples."""

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

        overlap, no_overlap = get_overlap_times(
            non_local_times, ripple_times
        )

        assert len(overlap) == 0, "No overlaps with empty ripples"
        assert len(no_overlap) == 1, "All events are non-overlapping"

    def test_empty_non_local_times(self):
        """Empty input returns empty outputs."""

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

        overlap, no_overlap = get_overlap_times(
            non_local_times, ripple_times
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
        # 0.1 s recording @ 1 kHz, prob=1.0 throughout (no event windows
        # carved out — the base value carries the whole array).
        prob_array, _ = _prob_array(
            [], dt=0.001, duration=0.1, base_value=1.0,
        )

        events = get_confident_times(
            prob_array,
            prob_threshold=0.8,
            minimum_duration=0.015
        )

        assert len(events) == 1
        dt = 0.002  # matches the recording-boundary fixture above
        assert events.iloc[0]["start_time"] == pytest.approx(0.0, abs=dt + 1e-9)
        assert events.iloc[0]["end_time"] == pytest.approx(0.1, abs=dt + 1e-9)

    def test_probability_exactly_at_threshold(self):
        """Probability exactly at threshold is included (> not >=)."""
        # Window set to exactly the threshold value (0.8).
        prob_array, _ = _prob_array([(0.1, 0.2)], dt=0.002, event_value=0.8)

        events = get_confident_times(
            prob_array,
            prob_threshold=0.8,
            minimum_duration=0.015
        )

        # With > threshold, should NOT include exactly 0.8
        assert len(events) == 0, "Threshold is >, not >="

    def test_very_short_minimum_duration(self):
        """Very short minimum duration (1ms) is respected."""
        # 10 kHz sampling, very brief event (2 ms).
        prob_array, _ = _prob_array([(0.1, 0.102)], dt=0.0001)

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
        prob_array, time = _prob_array([(0.0, 0.05)], dt=0.002)
        speed = np.full(len(time), 10.0)

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
        prob_array, time = _prob_array([(0.95, 1.0)], dt=0.002)
        speed = np.full(len(time), 10.0)

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

    def test_event_within_data_bounds_detected(self):
        """When data starts after t=0 (offset position index), the detector
        still finds an event whose probability mask is fully inside the
        recorded range. Build the array manually because ``_prob_array``
        assumes a zero-anchored time axis."""
        time = np.arange(0.1, 0.9, 0.002)
        prob = np.zeros(len(time))
        speed = np.full(len(time), 10.0)
        prob[(time >= 0.2) & (time < 0.3)] = 0.9
        prob_array = xr.DataArray(prob, coords={"time": time}, dims=["time"])

        events = get_confident_times(
            prob_array,
            speed=speed,
            time=time,
            prob_threshold=0.8,
            minimum_duration=0.015,
        )

        assert len(events) == 1
        assert events.iloc[0]["max_speed"] == pytest.approx(10.0)


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

        # Both functions now use the same coarse-class vocabulary,
        # so geom["overlap_class"] and coarse["overlap_class"] must
        # agree row-for-row.
        for i in range(len(geom)):
            assert geom.iloc[i]["overlap_class"] == coarse.iloc[i]["overlap_class"], (
                f"Event {i}: geometry={geom.iloc[i]['overlap_class']}, "
                f"classify={coarse.iloc[i]['overlap_class']}"
            )

    def test_boundary_touch_at_event_end_is_no_overlap(self):
        """Regression: a SWR starting exactly at the event's end
        shares zero positive duration with the event. Pre-fix used
        inclusive ``>=`` / ``<=`` and labeled this as
        ``swr_partial`` / ``single_swr_partial``, inflating
        SWR-coincident counts on back-to-back intervals."""
        ev = _event(1.0, 2.0)
        swr = _swrs([(2.0, 3.0)])
        out = classify_swr_overlap_geometry(ev, swr).iloc[0]
        assert out["overlap_class"] == "non_swr", (
            f"overlap_class={out['overlap_class']!r} on boundary "
            "touch — SWR [2.0, 3.0] only contacts event [1.0, 2.0] "
            "at t=2.0 and shares zero duration."
        )
        assert out["overlap_subclass"] == "no_overlap"
        assert out["n_swrs_overlapping"] == 0
        # And the coarse classifier must agree.
        coarse = classify_swr_overlap(ev, swr).iloc[0]
        assert coarse["overlap_class"] == "non_swr"

    def test_boundary_touch_at_event_start_is_no_overlap(self):
        """Mirror case: a SWR ending exactly at the event's start."""
        ev = _event(1.0, 2.0)
        swr = _swrs([(-1.0, 1.0)])
        out = classify_swr_overlap_geometry(ev, swr).iloc[0]
        assert out["overlap_class"] == "non_swr"
        assert out["overlap_subclass"] == "no_overlap"
        coarse = classify_swr_overlap(ev, swr).iloc[0]
        assert coarse["overlap_class"] == "non_swr"

    def test_minimal_positive_overlap_still_partial(self):
        """Sanity: any positive-duration overlap, however small, is
        still ``swr_partial`` — the strict-inequality fix must not
        also drop genuinely overlapping events."""
        ev = _event(1.0, 2.0)
        swr = _swrs([(1.999, 3.0)])  # 1ms of real overlap
        out = classify_swr_overlap_geometry(ev, swr).iloc[0]
        assert out["overlap_class"] == "swr_partial"
        assert out["overlap_subclass"] == "single_swr_partial"
        coarse = classify_swr_overlap(ev, swr).iloc[0]
        assert coarse["overlap_class"] == "swr_partial"


class TestMergedIntervalDuration:
    """Direct coverage for the interval union behind ``event_has_non_swr_time``.

    This logic was inline in ``classify_swr_overlap_geometry``'s per-event loop
    and reachable only through the multi-SWR overlap branch, so breaking the
    merge (never merging, always merging) left the whole suite green.
    """

    def test_single_interval_is_its_own_duration(self):
        assert merged_interval_duration(np.array([1.0]), np.array([3.0])) == 3.0 - 1.0

    def test_disjoint_intervals_add(self):
        got = merged_interval_duration(np.array([0.0, 5.0]), np.array([1.0, 7.0]))
        assert got == pytest.approx(1.0 + 2.0)

    def test_overlapping_intervals_count_the_overlap_once(self):
        # [0, 3] and [2, 4] cover [0, 4]; summing raw durations would give 5.
        got = merged_interval_duration(np.array([0.0, 2.0]), np.array([3.0, 4.0]))
        assert got == pytest.approx(4.0)

    def test_touching_intervals_give_their_full_span(self):
        """A contiguous train covers its whole span with no double count.

        Note this cannot discriminate the merge condition's ``<=`` from ``<``:
        touching intervals summed separately equal the same span merged, and
        any genuine overlap satisfies both predicates, so the two operators
        return identical durations on every input (checked over 200,000
        randomized interval sets on a grid that forces exact touching). The
        assertion pins the contract, not that operator.
        """
        assert merged_interval_duration(
            np.array([0.0, 1.0]), np.array([1.0, 2.0])
        ) == pytest.approx(2.0)

    def test_fully_contained_interval_adds_nothing(self):
        got = merged_interval_duration(np.array([0.0, 1.0]), np.array([9.0, 2.0]))
        assert got == pytest.approx(9.0)

    def test_unsorted_input_is_sorted_first(self):
        got = merged_interval_duration(np.array([5.0, 0.0]), np.array([7.0, 1.0]))
        assert got == pytest.approx(3.0)
