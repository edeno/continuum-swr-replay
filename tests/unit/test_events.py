"""Unit tests for DIO event processing.

Tests conversion of digital I/O events from binary signals to discrete
start/stop time intervals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.data_loaders.events import (
    convert_dio_events_to_start_stop_times,
)


class TestConvertDIOEventsToStartStopTimes:
    """Test DIO event conversion from binary to intervals."""

    def test_single_event(self):
        """Single DIO event should produce one start/stop interval."""
        # Create event data: single pulse at t=1.0
        times = np.array([0.0, 1.0, 2.0, 3.0])
        event_data = pd.DataFrame(
            {'signal': [0, 1, 0, 0]},
            index=times
        )

        result = convert_dio_events_to_start_stop_times(
            event_data['signal'],
            event_name="Poke1"
        )

        # Should have exactly one event
        assert len(result) == 1

        # Check timing
        assert result.loc[('Poke1', 1), 'start_time'] == 1.0
        assert result.loc[('Poke1', 1), 'end_time'] == 2.0

        # Check metadata (event_name is in the index, not a column)
        assert ('Poke1', 1) in result.index

    def test_multiple_events(self):
        """Multiple DIO events should produce multiple intervals."""
        times = np.arange(0.0, 10.0, 0.5)
        signal = np.zeros(len(times))
        signal[[2, 5, 8]] = 1  # Events at indices 2, 5, 8

        event_data = pd.DataFrame({'signal': signal}, index=times)

        result = convert_dio_events_to_start_stop_times(
            event_data['signal'],
            event_name="Poke1"
        )

        # Should have 3 events
        assert len(result) == 3

        # Check event numbering (1-indexed)
        assert ('Poke1', 1) in result.index
        assert ('Poke1', 2) in result.index
        assert ('Poke1', 3) in result.index

        # Check start times
        assert result.loc[('Poke1', 1), 'start_time'] == times[2]
        assert result.loc[('Poke1', 2), 'start_time'] == times[5]
        assert result.loc[('Poke1', 3), 'start_time'] == times[8]

    def test_no_events(self):
        """Empty event signal should produce empty DataFrame."""
        times = np.arange(0.0, 5.0, 1.0)
        signal = np.zeros(len(times))

        event_data = pd.DataFrame({'signal': signal}, index=times)

        result = convert_dio_events_to_start_stop_times(
            event_data['signal'],
            event_name="Poke1"
        )

        # Should be empty but have correct structure
        assert len(result) == 0
        assert 'start_time' in result.columns
        assert 'end_time' in result.columns
        # event_name is in the index, not a column
        assert result.index.names == ['event_name', 'event_number']

    def test_event_at_boundary_end(self):
        """Event at last time point should clamp end_time to avoid index error."""
        times = np.array([0.0, 1.0, 2.0, 3.0])
        signal = np.array([0, 0, 0, 1])  # Event at last index

        event_data = pd.DataFrame({'signal': signal}, index=times)

        result = convert_dio_events_to_start_stop_times(
            event_data['signal'],
            event_name="Poke1"
        )

        assert len(result) == 1

        # Start time should be at last index
        assert result.loc[('Poke1', 1), 'start_time'] == 3.0

        # End time should be clamped to last index (not out of bounds)
        assert result.loc[('Poke1', 1), 'end_time'] == 3.0

    def test_event_at_boundary_start(self):
        """Event at first time point should work correctly."""
        times = np.array([0.0, 1.0, 2.0, 3.0])
        signal = np.array([1, 0, 0, 0])  # Event at first index

        event_data = pd.DataFrame({'signal': signal}, index=times)

        result = convert_dio_events_to_start_stop_times(
            event_data['signal'],
            event_name="Poke1"
        )

        assert len(result) == 1
        assert result.loc[('Poke1', 1), 'start_time'] == 0.0
        assert result.loc[('Poke1', 1), 'end_time'] == 1.0

    def test_consecutive_events(self):
        """Consecutive event pulses should be treated as separate events."""
        times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        signal = np.array([0, 1, 1, 0, 0])  # Two consecutive 1s

        event_data = pd.DataFrame({'signal': signal}, index=times)

        result = convert_dio_events_to_start_stop_times(
            event_data['signal'],
            event_name="Poke1"
        )

        # Each 1 is treated as a separate event
        assert len(result) == 2

    def test_event_numbering_starts_at_one(self):
        """Event numbering should start at 1, not 0."""
        times = np.arange(0.0, 10.0, 1.0)
        signal = np.array([1, 0, 1, 0, 1, 0, 0, 0, 0, 0])

        event_data = pd.DataFrame({'signal': signal}, index=times)

        result = convert_dio_events_to_start_stop_times(
            event_data['signal'],
            event_name="Poke1"
        )

        # Check event numbers are 1, 2, 3
        assert ('Poke1', 1) in result.index
        assert ('Poke1', 2) in result.index
        assert ('Poke1', 3) in result.index
        assert ('Poke1', 0) not in result.index

    def test_multiindex_structure(self):
        """Result should have multi-index (event_name, event_number)."""
        times = np.array([0.0, 1.0, 2.0])
        signal = np.array([1, 0, 0])

        event_data = pd.DataFrame({'signal': signal}, index=times)

        result = convert_dio_events_to_start_stop_times(
            event_data['signal'],
            event_name="Poke1"
        )

        # Check index structure
        assert isinstance(result.index, pd.MultiIndex)
        assert result.index.names == ['event_name', 'event_number']

    def test_preserves_time_precision(self):
        """Should preserve floating point precision in timestamps."""
        times = np.array([0.0, 1.234567, 2.345678, 3.456789])
        signal = np.array([0, 1, 0, 0])

        event_data = pd.DataFrame({'signal': signal}, index=times)

        result = convert_dio_events_to_start_stop_times(
            event_data['signal'],
            event_name="Poke1"
        )

        # Times should be preserved precisely
        assert result.loc[('Poke1', 1), 'start_time'] == pytest.approx(1.234567)
        assert result.loc[('Poke1', 1), 'end_time'] == pytest.approx(2.345678)

    def test_different_event_names(self):
        """Event name should be correctly set in output."""
        times = np.array([0.0, 1.0])
        signal = np.array([1, 0])

        event_data = pd.DataFrame({'signal': signal}, index=times)

        for event_name in ["Poke1", "Pump2", "Light3"]:
            result = convert_dio_events_to_start_stop_times(
                event_data['signal'],
                event_name=event_name
            )

            # event_name is in the index, not a column
            assert (event_name, 1) in result.index

    def test_all_ones_signal(self):
        """Signal that is all ones should create event for each time point."""
        times = np.array([0.0, 1.0, 2.0, 3.0])
        signal = np.ones(len(times))

        event_data = pd.DataFrame({'signal': signal}, index=times)

        result = convert_dio_events_to_start_stop_times(
            event_data['signal'],
            event_name="Poke1"
        )

        # Should have one event per time point
        assert len(result) == 4

        # Last event should have clamped end_time
        assert result.loc[('Poke1', 4), 'end_time'] == 3.0

    def test_bool_dtype_signal(self):
        """Should handle boolean dtype signals."""
        times = np.array([0.0, 1.0, 2.0, 3.0])
        signal = np.array([False, True, False, True], dtype=bool)

        event_data = pd.DataFrame({'signal': signal}, index=times)

        result = convert_dio_events_to_start_stop_times(
            event_data['signal'],
            event_name="Poke1"
        )

        # Should detect both True values
        assert len(result) == 2

    def test_int_dtype_signal(self):
        """Should handle integer dtype signals (0/1)."""
        times = np.array([0.0, 1.0, 2.0, 3.0])
        signal = np.array([0, 1, 0, 2], dtype=int)  # 2 should also count as True

        event_data = pd.DataFrame({'signal': signal}, index=times)

        result = convert_dio_events_to_start_stop_times(
            event_data['signal'],
            event_name="Poke1"
        )

        # Both 1 and 2 should be detected (nonzero values)
        assert len(result) == 2

    def test_float_dtype_signal(self):
        """Should handle float dtype signals."""
        times = np.array([0.0, 1.0, 2.0, 3.0])
        signal = np.array([0.0, 1.0, 0.0, 0.5])

        event_data = pd.DataFrame({'signal': signal}, index=times)

        result = convert_dio_events_to_start_stop_times(
            event_data['signal'],
            event_name="Poke1"
        )

        # Both 1.0 and 0.5 should be detected (nonzero values)
        assert len(result) == 2

    def test_returns_correct_column_types(self):
        """Output columns should have correct data types."""
        times = np.array([0.0, 1.0, 2.0])
        signal = np.array([1, 0, 0])

        event_data = pd.DataFrame({'signal': signal}, index=times)

        result = convert_dio_events_to_start_stop_times(
            event_data['signal'],
            event_name="Poke1"
        )

        # start_time and end_time should be float (from timestamps)
        assert pd.api.types.is_numeric_dtype(result['start_time'])
        assert pd.api.types.is_numeric_dtype(result['end_time'])

        # event_name and event_number are in the index, not columns
        assert result.index.names == ['event_name', 'event_number']

        # Check index level types
        assert isinstance(result.index.get_level_values('event_name')[0], str)
        assert pd.api.types.is_integer_dtype(result.index.get_level_values('event_number'))

    def test_empty_dataframe_input(self):
        """Should handle empty DataFrame gracefully."""
        event_data = pd.DataFrame({'signal': []}, index=[])

        result = convert_dio_events_to_start_stop_times(
            event_data['signal'],
            event_name="Poke1"
        )

        assert len(result) == 0
        assert 'start_time' in result.columns


class TestLoadDIOEventTypeEmptyPattern:
    """Regression: ``_load_dio_event_type`` builds an ``events`` dict
    keyed by ``dio_event_name`` from ``DIOEvents.fetch_nwb()``. When
    no rows match the SQL ``LIKE`` pattern (e.g. a session that
    lacks any ``Light%`` events), ``events`` is empty and the
    subsequent ``pd.concat([])`` raised
    ``ValueError: No objects to concatenate`` — a missing optional
    DIO type then crashed the entire ``load_dios`` call. The fix
    short-circuits to a schema-complete empty frame.
    """

    def test_empty_pattern_returns_schema_complete_empty_frame(self, monkeypatch):
        # Stub ``spyglass.common.DIOEvents`` with an object whose
        # restriction operators no-op and ``fetch_nwb()`` returns
        # an empty list — matches the "no rows match this LIKE" case
        # without needing a live Spyglass connection.
        import sys
        import types

        spyglass_pkg = types.ModuleType("spyglass")
        common_mod = types.ModuleType("spyglass.common")

        class _NoMatchQuery:
            def __and__(self, _other):
                return self

            def fetch_nwb(self):
                return []

        common_mod.DIOEvents = _NoMatchQuery()
        monkeypatch.setitem(sys.modules, "spyglass", spyglass_pkg)
        monkeypatch.setitem(sys.modules, "spyglass.common", common_mod)

        from continuum_swr_replay.data_loaders.events import _load_dio_event_type

        out = _load_dio_event_type("any.nwb", "Light%", (0.0, 100.0))
        assert len(out) == 0
        assert list(out.columns) == ["start_time", "end_time"]
        assert list(out.index.names) == ["event_name", "event_number"]
