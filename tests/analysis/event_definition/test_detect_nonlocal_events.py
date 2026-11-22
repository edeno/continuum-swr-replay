"""The whole non-local event rule, ``detect_nonlocal_events``, on a hand-made decode."""

from __future__ import annotations

import numpy as np
import xarray as xr

from continuum_swr_replay.analysis.event_definition.nonlocal_event_rule import (
    detect_nonlocal_events,
)

DT = 0.002


def _decode() -> tuple[xr.DataArray, np.ndarray, np.ndarray]:
    """One second at 500 Hz with three runs of high non-local probability.

    - 0.100-0.150 s, animal still: an event.
    - 0.300-0.306 s, animal still: 4 samples, shorter than the minimum duration.
    - 0.500-0.550 s, animal moving at 10 cm/s: not immobile.
    """
    time = np.arange(0, 1, DT)
    probability = np.zeros(time.size)
    speed = np.zeros(time.size)
    for start, end in ((0.100, 0.150), (0.300, 0.306), (0.500, 0.550)):
        probability[(time >= start - 1e-9) & (time <= end + 1e-9)] = 0.95
    speed[(time >= 0.45) & (time <= 0.60)] = 10.0
    return xr.DataArray(probability, dims="time", coords={"time": time}), speed, time


def test_only_the_long_immobile_run_is_an_event():
    probability, speed, time = _decode()

    events, labels = detect_nonlocal_events(probability, speed, time)

    assert len(events) == 1
    np.testing.assert_allclose(
        events[["start_time", "end_time"]].to_numpy(), [[0.100, 0.150]], atol=1e-9
    )
    inside = (time >= 0.100 - 1e-9) & (time <= 0.150 + 1e-9)
    assert (labels.to_numpy()[inside] == 1).all()
    assert (labels.to_numpy()[~inside] == 0).all()
    assert labels.name == "non_local_event"


def test_the_probability_is_read_on_the_time_axis_given():
    """A decode's own time coordinate may differ from the position clock by rounding."""
    probability, speed, time = _decode()
    drifted = probability.assign_coords(time=time + 1e-7)

    events, _ = detect_nonlocal_events(drifted, speed, time)

    np.testing.assert_allclose(
        events[["start_time", "end_time"]].to_numpy(), [[0.100, 0.150]], atol=1e-9
    )


def test_a_higher_threshold_drops_the_event():
    probability, speed, time = _decode()

    events, labels = detect_nonlocal_events(probability, speed, time, 0.99)

    assert len(events) == 0
    assert (labels == 0).all()


def test_a_shorter_minimum_keeps_the_short_run_and_a_longer_one_drops_the_event():
    """The 4-sample run is an event at a 4-sample minimum; the 26-sample event
    is not at a 27-sample minimum."""
    probability, speed, time = _decode()

    short, _ = detect_nonlocal_events(probability, speed, time, minimum_duration=4 * DT)
    long, _ = detect_nonlocal_events(probability, speed, time, minimum_duration=27 * DT)

    np.testing.assert_allclose(
        short[["start_time", "end_time"]].to_numpy(),
        [[0.100, 0.150], [0.300, 0.306]],
        atol=1e-9,
    )
    assert len(long) == 0
