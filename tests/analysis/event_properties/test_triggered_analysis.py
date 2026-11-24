"""Unit tests for event-triggered analysis utilities.

Tests functions for extracting signal windows around events for peri-event
time histogram analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_properties.triggered_analysis import (
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
        with pytest.raises(ValueError, match=r"^data lacks required column"):
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
        with pytest.raises(ValueError, match=r"^event_times lacks required column"):
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
    # quartiles over events 1 and 3 (values 1 and 9): 3, 5 and 7
    np.testing.assert_allclose(curve["no_overlap_q25"], [3.0, 3.0, 3.0])
    np.testing.assert_allclose(curve["no_overlap_median"], [5.0, 5.0, 5.0])
    np.testing.assert_allclose(curve["no_overlap_q75"], [7.0, 7.0, 7.0])
    np.testing.assert_allclose(curve["overlap_median"], [4.0, 4.0, 4.0])


def test_triggering_events_in_keeps_the_frames_rows_of_the_selected_events():
    from continuum_swr_replay.analysis.event_properties.triggered_analysis import (
        TRIGGERING_EVENT_TYPES,
        triggering_events_in,
    )

    frame = pd.DataFrame(
        {
            "animal": "a",
            "day": [1, 1, 1, 1, 2, 2],
            "epoch": "02_r1",
            # label 1 of day 1 has two rows; label 1 of day 2 is another event
            "non_local_event": [1, 1, 2, 3, 1, 3],
        }
    )
    no_swr, swr = TRIGGERING_EVENT_TYPES
    event_types = {
        no_swr: frame.iloc[[0, 1, 3, 4]],
        swr: frame.iloc[[2, 5]],
        "SWR_events": frame.iloc[[5]],
    }
    selected = pd.DataFrame(
        {
            "animal": ["a", "a"],
            "session_date": [1, 2],
            "epoch": ["02_r1", "02_r1"],
            "non_local_event": [1, 3],
        }
    )
    kept = triggering_events_in(event_types, selected)
    assert kept[no_swr].index.tolist() == [0, 1]
    assert kept[swr].index.tolist() == [5]
    assert kept["SWR_events"] is event_types["SWR_events"]
    assert len(event_types[no_swr]) == 4  # the input is not changed


def _two_animal_windows() -> tuple[np.ndarray, list[str], list[str]]:
    """Animal a: epoch e1 with five events, 1..5 at both lags, and epoch e2
    with one event, 10 at the first lag and no value at the second. Animal b:
    one epoch with two events, (0, 0) and (4, 8)."""
    windows = np.array(
        [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0],
         [10.0, np.nan], [0.0, 0.0], [4.0, 8.0]]
    )  # fmt: skip
    animals = ["a"] * 6 + ["b"] * 2
    epochs = ["e1"] * 5 + ["e2"] + ["e1"] * 2
    return windows, animals, epochs


def test_animal_quartiles_are_each_epochs_then_the_median_over_its_epochs():
    """First lag: a's e1 has quartiles 2, 3, 4 and e2 10, so a's are the
    medians of two epochs, 6, 6.5 and 7 (pooled over its six events they would
    be 2.25, 3.5 and 4.75). Second lag: e2 has no value, so it is not one of
    a's epochs there and a's quartiles are e1's, 2, 3, 4. b's one epoch of two
    events: 1, 2, 3 and 2, 4, 6."""
    from continuum_swr_replay.analysis.event_properties.triggered_analysis import (
        animal_quartiles,
    )

    quartiles = animal_quartiles(*_two_animal_windows())
    assert list(quartiles["median"].columns) == ["a", "b"]
    np.testing.assert_allclose(quartiles["q25"]["a"], [6.0, 2.0])
    np.testing.assert_allclose(quartiles["median"]["a"], [6.5, 3.0])
    np.testing.assert_allclose(quartiles["q75"]["a"], [7.0, 4.0])
    np.testing.assert_allclose(quartiles["q25"]["b"], [1.0, 2.0])
    np.testing.assert_allclose(quartiles["median"]["b"], [2.0, 4.0])
    np.testing.assert_allclose(quartiles["q75"]["b"], [3.0, 6.0])


def test_animal_balanced_quartiles_are_the_mean_over_animals():
    """The animals' quartiles of the test above, averaged: (6 + 1) / 2, ...
    Where only b has a value, b's value alone."""
    from continuum_swr_replay.analysis.event_properties.triggered_analysis import (
        animal_balanced_quartiles,
    )

    quartiles = animal_balanced_quartiles(*_two_animal_windows())
    np.testing.assert_allclose(quartiles["q25"], [3.5, 2.0])
    np.testing.assert_allclose(quartiles["median"], [4.25, 3.5])
    np.testing.assert_allclose(quartiles["q75"], [5.0, 5.0])
    only_b = animal_balanced_quartiles(
        np.array([[1.0, np.nan], [9.0, 3.0]]), ["a", "b"], ["e1", "e1"]
    )
    np.testing.assert_allclose(only_b["median"], [5.0, 3.0])


def test_animal_window_medians_are_each_animals_median_over_its_events():
    """Animal a's events 1 and 2 of day 1 and event 1 of day 2, b's one: a's
    median at each lag is the median of its two epochs' medians (day 1: 1.5,
    and 5 with the NaN left out; day 2: 3 and 7), b's is its own window."""
    import xarray as xr

    from continuum_swr_replay.analysis.event_properties.triggered_analysis import (
        animal_window_medians,
    )

    values = np.array([[1.0, 5.0], [2.0, np.nan], [3.0, 7.0], [10.0, 20.0]])[:, :, None]
    index = pd.MultiIndex.from_tuples(
        [("a", 1, "e", 1), ("a", 1, "e", 2), ("a", 2, "e", 1), ("b", 1, "e", 1)],
        names=["animal", "day", "epoch", "non_local_event"],
    )
    windows = xr.DataArray(
        values,
        dims=("event", "time", "signal"),
        coords={"event": index, "time": [-0.002, 0.0], "signal": ["MUA_zscore"]},
    )
    medians = animal_window_medians(windows, "MUA_zscore")
    assert list(medians.columns) == ["a", "b"]
    np.testing.assert_allclose(medians["a"], [2.25, 6.0])
    np.testing.assert_allclose(medians["b"], [10.0, 20.0])
    np.testing.assert_allclose(medians.index, [-0.002, 0.0])


def _windows(values: np.ndarray, animals: list[str], labels: list[int]):
    """Two-lag windows of the four triggered signals, the same values for each,
    their event index carrying ``animal``, ``day``, ``epoch`` and
    ``non_local_event``."""
    import xarray as xr

    from continuum_swr_replay.analysis.event_properties.triggered_analysis import (
        TRIGGERED_SIGNALS,
    )

    index = pd.MultiIndex.from_arrays(
        [animals, [1] * len(animals), ["02_r1"] * len(animals), labels],
        names=["animal", "day", "epoch", "non_local_event"],
    )
    return xr.DataArray(
        np.repeat(values[:, :, None], len(TRIGGERED_SIGNALS), axis=2),
        dims=("event", "time", "signal"),
        coords={
            "event": index,
            "time": [-0.002, 0.0, 0.002],
            "signal": list(TRIGGERED_SIGNALS),
        },
    )


def test_an_epochs_trigger_windows_are_stored_a_row_per_window_and_signal(
    monkeypatch,
):
    """Each event window of each signal and each No-Spike window is one row,
    its lags in columns by their offset in samples; the record holds the
    frame's events by SWR overlap and the MUA level outside the periods."""
    import xarray as xr

    from continuum_swr_replay.analysis.event_properties import triggered_analysis as ta

    frame = pd.DataFrame(
        {"animal": "a", "day": 1, "epoch": "02_r1"}, index=[0.0, 0.002, 0.004]
    )
    event_types = {
        "non_local_event_immobile": pd.DataFrame(
            {"animal": "a", "day": 1, "epoch": "02_r1", "non_local_event": [4, 4, 7]}
        ),
        "non_local_event_immobile_overlap_SWR_event": pd.DataFrame(
            {"animal": "a", "day": 1, "epoch": "02_r1", "non_local_event": [7]}
        ),
        "non_local_event_immobile_no_overlap_SWR_event": pd.DataFrame(
            {"animal": "a", "day": 1, "epoch": "02_r1", "non_local_event": [4, 4]}
        ),
    }
    monkeypatch.setattr(
        ta,
        "triggered_windows",
        lambda frame, types, signals, window_sec: {
            "no_overlap": _windows(np.array([[1.0, 2.0, 3.0]]), ["a"], [4]),
            "overlap": _windows(np.array([[4.0, 5.0, 6.0]]), ["a"], [7]),
        },
    )
    periods = pd.DataFrame(
        {"start_time": [0.002, 0.004], "end_time": [0.004, 0.006], "duration": 0.002}
    )
    monkeypatch.setattr(ta, "no_spike_periods", lambda *args: periods)
    monkeypatch.setattr(ta, "level_outside_periods", lambda *args: 0.25)

    def signal_windows(frame, event_times, signals, window_sec, event_id_col):
        index = pd.MultiIndex.from_arrays(
            [["a"] * 2, [1] * 2, ["02_r1"] * 2, event_times[event_id_col]],
            names=["animal", "day", "epoch", event_id_col],
        )
        return xr.DataArray(
            np.array([[7.0, 8.0, 9.0], [0.0, 1.0, 2.0]])[:, :, None],
            dims=("event", "time", "signal"),
            coords={"event": index, "time": [-0.002, 0.0, 0.002], "signal": signals},
        )

    monkeypatch.setattr(ta, "get_signal_windows", signal_windows)
    rows, record = ta.epoch_trigger_windows(
        frame, event_types, np.array([0.0]), np.array([1.0])
    )
    assert list(rows.columns) == [*ta.WINDOW_ROW_KEYS, "lag-001", "lag+000", "lag+001"]
    assert len(rows) == 2 * len(ta.TRIGGERED_SIGNALS) + 2
    no_spike = rows[rows["kind"] == ta.NO_SPIKE_KIND]
    assert no_spike["non_local_event"].tolist() == [0, 1]
    assert set(no_spike["signal"]) == {ta.NO_SPIKE_SIGNAL}
    overlap = rows[(rows["kind"] == "overlap") & (rows["signal"] == "SWR_zscore")]
    assert overlap["non_local_event"].tolist() == [7]
    np.testing.assert_allclose(overlap.iloc[0, 3:].to_numpy(float), [4.0, 5.0, 6.0])
    assert record == {
        "n_events": {"swr": 1, "non_swr": 1, "total": 2},
        "no_spike_level": 0.25,
        "n_no_spike_periods": 2,
    }


def _stored_rows() -> pd.DataFrame:
    """Stored windows of two animals: a's events 1 and 2 (no SWR) and 3 (SWR)
    in one epoch, b's event 1 (no SWR); and a No-Spike window of each."""
    from continuum_swr_replay.analysis.event_properties.triggered_analysis import (
        NO_SPIKE_KIND,
        NO_SPIKE_SIGNAL,
        TRIGGERED_SIGNALS,
    )

    rows = []
    for animal, kind, label, values in (
        ("a", "no_overlap", 1, [1.0, 2.0, 3.0]),
        ("a", "no_overlap", 2, [3.0, 4.0, 5.0]),
        ("a", "overlap", 3, [9.0, 9.0, 9.0]),
        ("b", "no_overlap", 1, [5.0, 6.0, 7.0]),
        ("b", "overlap", 2, [1.0, 1.0, 1.0]),
    ):
        for signal in TRIGGERED_SIGNALS:
            rows.append((animal, kind, signal, label, *values))
    rows.append(("a", NO_SPIKE_KIND, NO_SPIKE_SIGNAL, 0, 2.0, 0.0, -2.0))
    rows.append(("b", NO_SPIKE_KIND, NO_SPIKE_SIGNAL, 0, 4.0, 2.0, 0.0))
    table = pd.DataFrame(
        rows,
        columns=[
            "animal", "kind", "signal", "non_local_event",
            "lag-001", "lag+000", "lag+001",
        ],
    )  # fmt: skip
    return table.assign(session_date=1, epoch="02_r1")


def test_the_stored_windows_give_figure_2gs_curves_and_the_supplements_traces():
    """Figure 2g's quartiles are the animal-balanced quartiles of the stored
    windows (no SWR: a's median of 1, 3 and b's 5 -> mean (2 + 5) / 2 = 3.5 at
    the first lag), the supplement's traces each animal's median, and the time
    axis the offsets over the sampling frequency."""
    from continuum_swr_replay.analysis.event_properties import triggered_analysis as ta
    from continuum_swr_replay.parameters import SAMPLING_FREQUENCY

    rows = _stored_rows()
    curves = ta.stored_triggered_curves(rows)
    curve = curves["MUA_zscore"]
    np.testing.assert_allclose(curve["no_overlap_median"], [3.5, 4.5, 5.5])
    np.testing.assert_allclose(curve["overlap_median"], [5.0, 5.0, 5.0])
    np.testing.assert_allclose(curve["time"], np.array([-1, 0, 1]) / SAMPLING_FREQUENCY)
    traces = ta.stored_animal_window_medians(rows)
    np.testing.assert_allclose(traces["no_overlap"]["SWR_zscore"]["a"], [2.0, 3.0, 4.0])
    np.testing.assert_allclose(traces["overlap"]["SWR_zscore"]["b"], [1.0, 1.0, 1.0])
    epochs = pd.DataFrame(
        {"animal": ["a", "a", "b"], "no_spike_level": [0.0, 1.0, 2.0]}
    )
    trace = ta.stored_no_spike_trace(rows, epochs)
    np.testing.assert_allclose(trace["median"], [3.0, 1.0, -1.0])
    # a's median over its epochs 0.5, b's 2 -> 1.25
    assert trace["level"] == pytest.approx(1.25)


def test_the_stored_windows_are_reduced_epoch_by_epoch():
    """a's no-SWR events at the first lag: 1 on day 1; 3, 7 and 9 on day 2.
    Day 1's median is 1 and day 2's 7, so a's is 4 (pooled over its four
    events it would be 5); with b's 5 the mean is 4.5. A No-Spike window of a
    third day is one more of a's epochs."""
    from continuum_swr_replay.analysis.event_properties import triggered_analysis as ta

    rows = _stored_rows()
    a_two = (rows["animal"] == "a") & (rows["non_local_event"] == 2)
    rows.loc[a_two, "session_date"] = 2
    extra = rows[a_two & (rows["kind"] == "no_overlap")].copy()
    rows = pd.concat(
        [
            rows,
            extra.assign(non_local_event=4, **{"lag-001": 7.0}),
            extra.assign(non_local_event=5, **{"lag-001": 9.0}),
        ],
        ignore_index=True,
    )
    traces = ta.stored_animal_window_medians(rows)
    assert traces["no_overlap"]["MUA_zscore"]["a"].iloc[0] == pytest.approx(4.0)
    curves = ta.stored_triggered_curves(rows)
    # a 4, b 5
    assert curves["MUA_zscore"]["no_overlap_median"][0] == pytest.approx(4.5)
    no_spike = rows[rows["kind"] == ta.NO_SPIKE_KIND]
    rows = pd.concat(
        [
            rows,
            no_spike[no_spike["animal"] == "a"].assign(
                session_date=3, **{"lag-001": 6.0}
            ),
        ],
        ignore_index=True,
    )
    epochs = pd.DataFrame({"animal": ["a", "b"], "no_spike_level": [0.0, 0.0]})
    # a: the median of its epochs' 2 and 6, 4; b: 4
    assert ta.stored_no_spike_trace(rows, epochs)["median"][0] == pytest.approx(4.0)


def test_the_stored_windows_of_the_selected_events_are_kept_with_every_period():
    from continuum_swr_replay.analysis.event_properties import triggered_analysis as ta

    rows = _stored_rows()
    events = pd.DataFrame(
        {"animal": ["a", "b"], "session_date": 1, "epoch": "02_r1",
         "non_local_event": [3, 1]}
    )  # fmt: skip
    kept = ta.stored_windows_of(rows, events)
    assert set(
        zip(kept["animal"], kept["kind"], kept["non_local_event"], strict=True)
    ) == {
        ("a", "overlap", 3),
        ("b", "no_overlap", 1),
        ("a", ta.NO_SPIKE_KIND, 0),
        ("b", ta.NO_SPIKE_KIND, 0),
    }


def test_the_frames_events_are_counted_by_swr_overlap():
    from continuum_swr_replay.analysis.event_properties.triggered_analysis import (
        frame_event_counts,
    )

    rows = pd.DataFrame(
        {
            "animal": "a",
            "day": 1,
            "epoch": "e",
            # bins of three events; event 2 has two bins
            "non_local_event": [1, 2, 2, 3],
        }
    )
    counts = frame_event_counts(
        {
            "non_local_event_immobile": rows,
            "non_local_event_immobile_overlap_SWR_event": rows.iloc[1:3],
            "non_local_event_immobile_no_overlap_SWR_event": rows.iloc[[0, 3]],
        }
    )
    assert counts == {"swr": 1, "non_swr": 2, "total": 3}
