"""Tests for analysis constants module."""

import numpy as np
import pytest


def test_constants_importable():
    """Test that all constants can be imported."""

    # If we get here, all imports succeeded
    assert True


def test_default_non_local_threshold():
    """Test DEFAULT_NON_LOCAL_THRESHOLD has correct type and value."""
    from continuum_swr_replay.analysis.constants import DEFAULT_NON_LOCAL_THRESHOLD

    assert isinstance(DEFAULT_NON_LOCAL_THRESHOLD, float)
    assert 0.0 <= DEFAULT_NON_LOCAL_THRESHOLD <= 1.0
    assert DEFAULT_NON_LOCAL_THRESHOLD == 0.8


def test_default_immobility_threshold():
    """Test DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC has correct type and value."""
    from continuum_swr_replay.analysis.constants import (
        DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC,
    )

    assert isinstance(DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC, float)
    assert DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC > 0
    assert DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC == 4.0


def test_default_sampling_frequency():
    """Test DEFAULT_SAMPLING_FREQUENCY_HZ has correct type and value."""
    from continuum_swr_replay.analysis.constants import DEFAULT_SAMPLING_FREQUENCY_HZ

    assert isinstance(DEFAULT_SAMPLING_FREQUENCY_HZ, float)
    assert DEFAULT_SAMPLING_FREQUENCY_HZ > 0
    assert DEFAULT_SAMPLING_FREQUENCY_HZ == 500.0


def test_default_window_event_triggered():
    """Test DEFAULT_WINDOW_EVENT_TRIGGERED_SEC has correct type and value."""
    from continuum_swr_replay.analysis.constants import (
        DEFAULT_WINDOW_EVENT_TRIGGERED_SEC,
    )

    assert isinstance(DEFAULT_WINDOW_EVENT_TRIGGERED_SEC, float)
    assert DEFAULT_WINDOW_EVENT_TRIGGERED_SEC > 0
    assert DEFAULT_WINDOW_EVENT_TRIGGERED_SEC == 0.25


def test_bins_duration_ms():
    """Test BINS_DURATION_MS has correct type and structure."""
    from continuum_swr_replay.analysis.constants import BINS_DURATION_MS

    assert isinstance(BINS_DURATION_MS, np.ndarray)
    assert BINS_DURATION_MS.dtype == np.dtype("int64")
    assert len(BINS_DURATION_MS) > 0
    assert BINS_DURATION_MS[0] == 0
    assert BINS_DURATION_MS[-1] == 240
    # Check spacing is 10 ms
    assert np.all(np.diff(BINS_DURATION_MS) == 10)


def test_bins_probability():
    """Test BINS_PROBABILITY has correct type and structure."""
    from continuum_swr_replay.analysis.constants import BINS_PROBABILITY

    assert isinstance(BINS_PROBABILITY, np.ndarray)
    assert BINS_PROBABILITY.dtype == np.dtype("float64")
    assert len(BINS_PROBABILITY) > 0
    assert BINS_PROBABILITY[0] == pytest.approx(0.0)
    assert BINS_PROBABILITY[-1] == pytest.approx(1.0)
    # Check spacing is 0.05
    assert np.allclose(np.diff(BINS_PROBABILITY), 0.05)


def test_bins_distance_cm():
    """Test BINS_DISTANCE_CM has correct type and structure."""
    from continuum_swr_replay.analysis.constants import BINS_DISTANCE_CM

    assert isinstance(BINS_DISTANCE_CM, np.ndarray)
    assert BINS_DISTANCE_CM.dtype == np.dtype("int64")
    assert len(BINS_DISTANCE_CM) > 0
    assert BINS_DISTANCE_CM[0] == 0
    assert BINS_DISTANCE_CM[-1] == 228
    # Check spacing is 2 cm
    assert np.all(np.diff(BINS_DISTANCE_CM) == 2)


def test_bins_hpd_cm():
    """Test BINS_HPD_CM has correct type and structure."""
    from continuum_swr_replay.analysis.constants import BINS_HPD_CM

    assert isinstance(BINS_HPD_CM, np.ndarray)
    assert BINS_HPD_CM.dtype == np.dtype("int64")
    assert len(BINS_HPD_CM) > 0
    assert BINS_HPD_CM[0] == 0
    assert BINS_HPD_CM[-1] == 240
    # Check spacing is 10 cm
    assert np.all(np.diff(BINS_HPD_CM) == 10)


def test_bins_speed_cm_per_sec():
    """Test BINS_SPEED_CM_PER_SEC has correct type and structure."""
    from continuum_swr_replay.analysis.constants import BINS_SPEED_CM_PER_SEC

    assert isinstance(BINS_SPEED_CM_PER_SEC, np.ndarray)
    assert BINS_SPEED_CM_PER_SEC.dtype == np.dtype("float64")
    assert len(BINS_SPEED_CM_PER_SEC) > 0
    assert BINS_SPEED_CM_PER_SEC[0] == pytest.approx(0.0)
    assert BINS_SPEED_CM_PER_SEC[-1] == pytest.approx(19.5)
    # Check spacing is 0.5 cm/s
    assert np.allclose(np.diff(BINS_SPEED_CM_PER_SEC), 0.5)


def test_bins_are_numpy_arrays_not_lists():
    """Test that all bin definitions are numpy arrays, not lists."""
    from continuum_swr_replay.analysis.constants import (
        BINS_DISTANCE_CM,
        BINS_DURATION_MS,
        BINS_HPD_CM,
        BINS_PROBABILITY,
        BINS_SPEED_CM_PER_SEC,
    )

    # Ensure they are numpy arrays for efficient operations
    assert isinstance(BINS_DURATION_MS, np.ndarray)
    assert isinstance(BINS_PROBABILITY, np.ndarray)
    assert isinstance(BINS_DISTANCE_CM, np.ndarray)
    assert isinstance(BINS_HPD_CM, np.ndarray)
    assert isinstance(BINS_SPEED_CM_PER_SEC, np.ndarray)
