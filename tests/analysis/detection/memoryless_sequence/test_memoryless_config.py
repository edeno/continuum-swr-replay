"""Thresholds for the memoryless sliding-window sequence detector."""

import dataclasses

import pytest

from continuum_swr_replay.analysis.detection.memoryless_sequence.config import (
    MemorylessSequenceConfig,
)
from continuum_swr_replay.parameters import DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC


def test_defaults_are_the_retained_settings():
    """The settings kept from Widloski & Foster 2025."""
    config = MemorylessSequenceConfig()

    assert config.window_duration_s == 0.080
    assert config.place_bin_size_cm == 2.0
    assert config.jump_threshold_cm == 20.0
    assert config.merge_gap_cm == 20.0
    assert config.merge_gap_s == 0.050
    assert config.minimum_duration_s == 0.100
    assert config.dispersion_threshold_cm == 12.0


def test_adapted_settings_are_ours_not_theirs():
    """The settings adapted to this project's grid, immobility rule and track."""
    config = MemorylessSequenceConfig()

    assert config.stride_s == 0.004  # 5 ms is 2.5 samples at 500 Hz
    assert config.speed_threshold_cm_per_s == 4.0  # theirs is 5
    assert config.spread_threshold_cm == 10.0  # adapted, not equivalent


def test_the_speed_threshold_is_the_project_s_immobility_rule():
    """One rule for every cohort, read from one constant rather than copied."""
    assert (
        MemorylessSequenceConfig().speed_threshold_cm_per_s
        == DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC
    )


def test_bin_counts_follow_the_stride():
    """A threshold in windows must be derived, never carried over."""
    config = MemorylessSequenceConfig(sampling_frequency=500.0)

    assert config.window_n_samples == 40
    assert config.stride_n_samples == 2
    # a surviving run spans at least minimum_duration between first and last
    # midpoint, so its index length is minimum_duration / stride
    assert config.minimum_run_index_length == 25


def test_a_different_stride_moves_the_window_counts():
    """The constant that would silently become a 50 ms threshold."""
    at_500 = MemorylessSequenceConfig(sampling_frequency=500.0)
    at_1000 = MemorylessSequenceConfig(sampling_frequency=1000.0)

    assert at_1000.stride_n_samples == 2 * at_500.stride_n_samples
    assert at_1000.window_n_samples == 2 * at_500.window_n_samples
    assert at_1000.minimum_run_index_length == at_500.minimum_run_index_length


def test_the_reference_stride_gives_the_reference_counts():
    """At their 5 ms stride a surviving run holds at least 21 windows."""
    config = MemorylessSequenceConfig(stride_s=0.005, sampling_frequency=200.0)

    assert config.minimum_run_index_length == 20


def test_config_is_frozen():
    config = MemorylessSequenceConfig()

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.jump_threshold_cm = 30.0


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("window_duration_s", 0.0),
        ("stride_s", 0.0),
        ("sampling_frequency", 0.0),
        ("place_bin_size_cm", -1.0),
        ("minimum_duration_s", -0.1),
        ("window_duration_s", float("inf")),
        ("jump_threshold_cm", float("nan")),
        ("jump_threshold_cm", True),
    ],
)
def test_non_positive_or_non_finite_settings_raise(field, value):
    with pytest.raises(ValueError, match=field):
        MemorylessSequenceConfig(**{field: value})


def test_a_stride_longer_than_the_window_raises():
    """Windows must overlap; the reference steps 5 ms through 80 ms."""
    with pytest.raises(ValueError, match="stride_s"):
        MemorylessSequenceConfig(window_duration_s=0.080, stride_s=0.100)


def test_a_window_that_is_not_a_whole_number_of_samples_raises():
    with pytest.raises(ValueError, match="window_duration_s"):
        MemorylessSequenceConfig(window_duration_s=0.081, sampling_frequency=500.0)


def test_the_reference_stride_on_the_project_grid_raises():
    """5 ms is 2.5 samples at 500 Hz; the reason the stride was adapted."""
    with pytest.raises(ValueError, match="stride_s"):
        MemorylessSequenceConfig(stride_s=0.005, sampling_frequency=500.0)


@pytest.mark.parametrize(
    ("minimum_duration_s", "stride_s", "index_length"),
    [
        (0.101, 0.004, 26),  # 25.25 strides: 26, never 25 (100 ms)
        (0.100, 0.003, 34),  # 33.3 strides: 34, never 33 (99 ms)
        (0.050, 0.004, 13),  # the reference's 1D branch: 52 ms, not 48
        (0.100, 0.004, 25),  # exact
    ],
)
def test_a_minimum_duration_between_strides_rounds_up_never_down(
    minimum_duration_s, stride_s, index_length
):
    """A filter that rounded down would keep runs shorter than asked."""
    config = MemorylessSequenceConfig(
        minimum_duration_s=minimum_duration_s,
        stride_s=stride_s,
        sampling_frequency=1000.0,
    )

    assert config.minimum_run_index_length == index_length
    assert index_length * stride_s >= minimum_duration_s - 1e-12


def test_a_minimum_duration_shorter_than_one_stride_raises():
    with pytest.raises(ValueError, match="minimum_duration_s"):
        MemorylessSequenceConfig(minimum_duration_s=0.001, stride_s=0.004)


def test_a_window_of_an_odd_number_of_samples_is_refused():
    """Its midpoint would fall between two samples, and event bounds are samples."""
    with pytest.raises(ValueError, match="even"):
        MemorylessSequenceConfig(window_duration_s=0.082)


def test_single_window_runs_are_dropped_unless_asked_otherwise():
    assert MemorylessSequenceConfig().drop_single_window_runs is True
    assert (
        MemorylessSequenceConfig(drop_single_window_runs=False).drop_single_window_runs
        is False
    )
