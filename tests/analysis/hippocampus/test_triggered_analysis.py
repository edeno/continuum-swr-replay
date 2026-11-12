"""Unit tests for event-triggered analysis utilities.

Tests functions for extracting signal windows around events for peri-event
time histogram analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.hippocampus.triggered_analysis import (
    compute_triggered_average_curves,
    get_signal_windows,
)


class TestGetSignalWindows:
    """Test extracting signal windows around event times."""

    @pytest.fixture
    def mock_data(self) -> pd.DataFrame:
        """Create mock time-series data with multiple signals.

        Creates 2 seconds of data at 500 Hz (1000 samples).
        """
        sampling_frequency = 500.0
        duration = 2.0
        n_samples = int(sampling_frequency * duration)
        time = np.arange(n_samples) / sampling_frequency

        # Create some realistic signals
        rng = np.random.default_rng(42)
        signal1 = np.sin(2 * np.pi * 8 * time) + rng.normal(0, 0.1, n_samples)
        signal2 = np.cos(2 * np.pi * 6 * time) + rng.normal(0, 0.1, n_samples)
        signal3 = rng.normal(0, 1, n_samples)

        return pd.DataFrame(
            {
                "animal": ["mouse1"] * n_samples,
                "day": [20200101] * n_samples,
                "epoch": ["01_r1"] * n_samples,
                "theta_power": signal1,
                "ripple_power": signal2,
                "speed": signal3,
            },
            index=time,
        )

    @pytest.fixture
    def mock_events(self) -> pd.DataFrame:
        """Create mock event times.

        Events at 0.5s, 1.0s, and 1.5s within the data range.
        """
        return pd.DataFrame(
            {
                "animal": ["mouse1", "mouse1", "mouse1"],
                "day": [20200101, 20200101, 20200101],
                "epoch": ["01_r1", "01_r1", "01_r1"],
                "event_id": [1, 2, 3],
                "start_time": [0.5, 1.0, 1.5],
            }
        )

    def test_normal_window_extraction(self, mock_data, mock_events):
        """Should extract windows correctly for events in data range."""
        result = get_signal_windows(
            mock_data,
            mock_events,
            signal_cols=["theta_power", "ripple_power", "speed"],
            time_col="start_time",
            window_sec=0.2,  # ±0.1s = 100ms on each side
            sampling_frequency=500.0,
        )

        # Check dimensions
        assert result.dims == ("event", "time", "signal")

        # Check shape: 3 events, 101 time points (±50 + center), 3 signals
        assert result.shape[0] == 3  # 3 events
        assert result.shape[1] == 101  # 2 * 50 + 1 samples
        assert result.shape[2] == 3  # 3 signals

        # Check coordinates
        assert list(result.coords["signal"].values) == [
            "theta_power",
            "ripple_power",
            "speed",
        ]

        # Check time axis is centered at 0
        time_values = result.coords["time"].values
        assert time_values[0] == pytest.approx(-0.1, abs=1e-6)
        assert time_values[50] == pytest.approx(0.0, abs=1e-6)
        assert time_values[-1] == pytest.approx(0.1, abs=1e-6)

        # Check no NaN values (all events well within data)
        assert not np.isnan(result.values).any()

    def test_events_near_boundaries_partial_windows(self, mock_data):
        """Events near data boundaries should have partial windows with NaN."""
        # Events very close to start and end of data
        boundary_events = pd.DataFrame(
            {
                "animal": ["mouse1", "mouse1"],
                "day": [20200101, 20200101],
                "epoch": ["01_r1", "01_r1"],
                "event_id": [1, 2],
                "start_time": [0.05, 1.95],  # Near start and end
            }
        )

        result = get_signal_windows(
            mock_data,
            boundary_events,
            signal_cols=["theta_power"],
            time_col="start_time",
            window_sec=0.2,  # ±0.1s window
            sampling_frequency=500.0,
        )

        # Should have 2 events
        assert result.shape[0] == 2

        # First event (at 0.05s) should have NaN at beginning
        # (trying to go to -0.05s which is before data start at 0.0s)
        first_event_window = result.isel(event=0, signal=0).values
        # Should have some NaN values
        assert np.isnan(first_event_window).any()

        # Last event (at 1.95s) should have NaN at end
        # (trying to go to 2.05s which is after data end at ~2.0s)
        last_event_window = result.isel(event=1, signal=0).values
        assert np.isnan(last_event_window).any()

    def test_events_outside_data_range_full_nan(self, mock_data):
        """Events completely outside data range should have all NaN windows."""
        outside_events = pd.DataFrame(
            {
                "animal": ["mouse1", "mouse1"],
                "day": [20200101, 20200101],
                "epoch": ["01_r1", "01_r1"],
                "event_id": [1, 2],
                "start_time": [-1.0, 5.0],  # Way before and after data
            }
        )

        result = get_signal_windows(
            mock_data,
            outside_events,
            signal_cols=["theta_power"],
            time_col="start_time",
            window_sec=0.2,
            sampling_frequency=500.0,
        )

        # Should have 2 events with full NaN windows
        assert result.shape[0] == 2

        # All values should be NaN
        assert np.isnan(result.values).all()

    def test_empty_events_returns_proper_structure(self, mock_data):
        """No events should return empty DataArray with proper structure."""
        empty_events = pd.DataFrame(
            {
                "animal": [],
                "day": [],
                "epoch": [],
                "start_time": [],
            }
        )

        result = get_signal_windows(
            mock_data,
            empty_events,
            signal_cols=["theta_power", "ripple_power"],
            time_col="start_time",
            window_sec=0.2,
            sampling_frequency=500.0,
        )

        # Should have proper dimensions but 0 events
        assert result.dims == ("event", "time", "signal")
        assert result.shape[0] == 0
        assert result.shape[1] == 101  # Window size still defined
        assert result.shape[2] == 2  # Signal count

        # Coordinates should still be defined
        assert len(result.coords["signal"]) == 2
        assert len(result.coords["time"]) == 101

    def test_invalid_sampling_frequency_raises_error(self, mock_data, mock_events):
        """Invalid sampling frequency should raise ValueError."""
        with pytest.raises(ValueError, match="sampling_frequency must be positive"):
            get_signal_windows(
                mock_data,
                mock_events,
                signal_cols=["theta_power"],
                time_col="start_time",
                window_sec=0.2,
                sampling_frequency=0.0,
            )

        with pytest.raises(ValueError, match="sampling_frequency must be positive"):
            get_signal_windows(
                mock_data,
                mock_events,
                signal_cols=["theta_power"],
                time_col="start_time",
                window_sec=0.2,
                sampling_frequency=-500.0,
            )

    def test_invalid_window_sec_raises_error(self, mock_data, mock_events):
        """Invalid window size should raise ValueError."""
        with pytest.raises(ValueError, match="window_sec must be positive"):
            get_signal_windows(
                mock_data,
                mock_events,
                signal_cols=["theta_power"],
                time_col="start_time",
                window_sec=0.0,
                sampling_frequency=500.0,
            )

        with pytest.raises(ValueError, match="window_sec must be positive"):
            get_signal_windows(
                mock_data,
                mock_events,
                signal_cols=["theta_power"],
                time_col="start_time",
                window_sec=-0.2,
                sampling_frequency=500.0,
            )

    def test_missing_data_columns_raise_error(self, mock_data, mock_events):
        """Missing required columns in data should raise ValueError."""
        with pytest.raises(ValueError, match="data missing required columns"):
            get_signal_windows(
                mock_data,
                mock_events,
                signal_cols=["nonexistent_signal"],
                time_col="start_time",
                window_sec=0.2,
                sampling_frequency=500.0,
            )

    def test_missing_event_columns_raise_error(self, mock_data, mock_events):
        """Missing required columns in event_times should raise ValueError."""
        # Missing time_col
        with pytest.raises(ValueError, match="event_times missing required columns"):
            get_signal_windows(
                mock_data,
                mock_events,
                signal_cols=["theta_power"],
                time_col="nonexistent_time",
                window_sec=0.2,
                sampling_frequency=500.0,
            )

    def test_event_id_col_creates_multiindex(self, mock_data, mock_events):
        """Using event_id_col should create MultiIndex for event coordinate."""
        result = get_signal_windows(
            mock_data,
            mock_events,
            signal_cols=["theta_power"],
            time_col="start_time",
            window_sec=0.2,
            sampling_frequency=500.0,
            event_id_col="event_id",
        )

        # Event coordinate should be a MultiIndex
        assert isinstance(result.coords["event"].values[0], tuple)

        # Should have (animal, day, epoch, event_id) tuples
        first_event_coord = result.coords["event"].values[0]
        assert len(first_event_coord) == 4
        assert first_event_coord[0] == "mouse1"
        assert first_event_coord[1] == 20200101
        assert first_event_coord[2] == "01_r1"
        assert first_event_coord[3] == 1  # First event_id

    def test_without_event_id_col_uses_integers(self, mock_data, mock_events):
        """Without event_id_col should use sequential integers."""
        result = get_signal_windows(
            mock_data,
            mock_events,
            signal_cols=["theta_power"],
            time_col="start_time",
            window_sec=0.2,
            sampling_frequency=500.0,
            event_id_col=None,
        )

        # Event coordinate should be integers
        assert np.issubdtype(result.coords["event"].dtype, np.integer)
        assert list(result.coords["event"].values) == [0, 1, 2]

    def test_multiple_sessions(self):
        """Should handle multiple sessions correctly."""
        # Create data for two sessions
        sampling_frequency = 500.0
        n_samples = 1000
        time = np.arange(n_samples) / sampling_frequency

        data_session1 = pd.DataFrame(
            {
                "animal": ["mouse1"] * n_samples,
                "day": [20200101] * n_samples,
                "epoch": ["01_r1"] * n_samples,
                "signal": np.sin(2 * np.pi * 8 * time),
            },
            index=time,
        )

        data_session2 = pd.DataFrame(
            {
                "animal": ["mouse1"] * n_samples,
                "day": [20200102] * n_samples,
                "epoch": ["02_r1"] * n_samples,
                "signal": np.cos(2 * np.pi * 8 * time),
            },
            index=time,
        )

        data = pd.concat([data_session1, data_session2])

        # Events in both sessions
        events = pd.DataFrame(
            {
                "animal": ["mouse1", "mouse1"],
                "day": [20200101, 20200102],
                "epoch": ["01_r1", "02_r1"],
                "start_time": [0.5, 0.5],
            }
        )

        result = get_signal_windows(
            data,
            events,
            signal_cols=["signal"],
            time_col="start_time",
            window_sec=0.2,
            sampling_frequency=500.0,
        )

        # Should have 2 events (one from each session)
        assert result.shape[0] == 2

        # No NaN values (both events well within their session data)
        assert not np.isnan(result.values).any()

    def test_single_signal(self, mock_data, mock_events):
        """Should work with a single signal."""
        result = get_signal_windows(
            mock_data,
            mock_events,
            signal_cols=["theta_power"],
            time_col="start_time",
            window_sec=0.2,
            sampling_frequency=500.0,
        )

        assert result.shape[2] == 1
        assert result.coords["signal"].values[0] == "theta_power"

    def test_different_time_col(self, mock_data):
        """Should work with different time column names."""
        events_with_end = pd.DataFrame(
            {
                "animal": ["mouse1"],
                "day": [20200101],
                "epoch": ["01_r1"],
                "end_time": [1.0],
            }
        )

        result = get_signal_windows(
            mock_data,
            events_with_end,
            signal_cols=["theta_power"],
            time_col="end_time",  # Use end_time instead of start_time
            window_sec=0.2,
            sampling_frequency=500.0,
        )

        assert result.shape[0] == 1
        assert not np.isnan(result.values).any()

    def test_window_size_calculation(self, mock_data, mock_events):
        """Window size should be calculated correctly."""
        # 0.4s window at 500 Hz = ±100 samples + center = 201 samples
        result = get_signal_windows(
            mock_data,
            mock_events,
            signal_cols=["theta_power"],
            time_col="start_time",
            window_sec=0.4,
            sampling_frequency=500.0,
        )

        assert result.shape[1] == 201

        # Time axis should span -0.2 to +0.2
        time_values = result.coords["time"].values
        assert time_values[0] == pytest.approx(-0.2, abs=1e-6)
        assert time_values[-1] == pytest.approx(0.2, abs=1e-6)


def test_triggered_average_curves_are_mean_and_sem_over_events_by_class():
    """Mean and SEM over events of each class's windows, lag by lag."""
    # 50 samples at 500 Hz. Events 1, 2 and 3 are one sample each (10, 25, 40);
    # the signal is 1, 4 and 9 around them, so each window is one constant.
    time = np.arange(50) / 500.0
    labels = np.zeros(50, dtype=int)
    labels[[10, 25, 40]] = [1, 2, 3]
    signal = np.repeat([1.0, 4.0, 9.0], [18, 15, 17])
    frame = pd.DataFrame(
        {
            "animal": "a",
            "day": 1,
            "epoch": "02_r1",
            "non_local_event": labels,
            "x": signal,
        },
        index=time,
    )
    event_types = {
        "non_local_event_immobile_no_overlap_SWR_event": frame[
            frame["non_local_event"].isin([1, 3])
        ],
        "non_local_event_immobile_overlap_SWR_event": frame[
            frame["non_local_event"] == 2
        ],
    }

    curves = compute_triggered_average_curves(
        frame, event_types, ["x"], window_sec=0.004
    )

    curve = curves["x"]
    np.testing.assert_allclose(curve["time"], [-0.002, 0.0, 0.002])
    # Events 1 and 3: the mean of 1 and 9 is 5; the SEM, std(ddof=1)/sqrt(2), 4.
    np.testing.assert_allclose(curve["no_overlap_mean"], [5.0, 5.0, 5.0])
    np.testing.assert_allclose(curve["no_overlap_sem"], [4.0, 4.0, 4.0])
    # Event 2 alone: its value, and no SEM from one event.
    np.testing.assert_allclose(curve["overlap_mean"], [4.0, 4.0, 4.0])
    assert np.isnan(curve["overlap_sem"]).all()
