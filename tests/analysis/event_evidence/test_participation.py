"""Tests for the tetrode participation and spike rate of non-local events."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_evidence import participation as pt
from continuum_swr_replay.analysis.event_evidence.state_evidence.windows import (
    event_bins,
)

BIN_S = 0.002


def test_event_bins_map_times_to_the_decode_grid():
    time = np.arange(100) * BIN_S
    s, e = event_bins(time, np.array([0.010, 0.050]), np.array([0.020, 0.050]))
    np.testing.assert_array_equal(s, [5, 25])
    np.testing.assert_array_equal(e, [10, 25])
    with pytest.raises(ValueError, match="decode's time bins"):
        event_bins(time, np.array([0.0105]), np.array([0.020]))
    # a hair after a bin is that bin, as a hair before is
    s, e = event_bins(time, np.array([0.010 + 5e-7]), np.array([0.020 - 5e-7]))
    np.testing.assert_array_equal(s, [5])
    np.testing.assert_array_equal(e, [10])


def test_local_rest_needs_local_immobile_and_no_event_nearby():
    local = np.array([0.9] * 10 + [0.5] + [0.9] * 9)
    immobile = np.ones(20, dtype=bool)
    immobile[15] = False
    in_event = np.zeros(20, dtype=bool)
    in_event[5] = True
    mask = pt.local_rest_mask(local, immobile, in_event, margin_bins=2)
    # bins 3-7 are within 2 bins of the event, 10 is not Local, 15 is moving
    expected = np.ones(20, dtype=bool)
    expected[[3, 4, 5, 6, 7, 10, 15]] = False
    np.testing.assert_array_equal(mask, expected)


def test_control_windows_have_the_events_lengths_and_lie_in_local_rest():
    mask = np.zeros(200, dtype=bool)
    mask[100:200] = True
    s = np.array([10, 40])
    e = np.array([14, 59])
    windows = pt.control_windows(s, e, mask, seed=(1, "epoch"))
    np.testing.assert_array_equal(windows["e"] - windows["s"], e - s)
    for a, b in zip(windows["s"], windows["e"], strict=True):
        assert mask[a : b + 1].all()
    # the same seed draws the same windows
    again = pt.control_windows(s, e, mask, seed=(1, "epoch"))
    pd.testing.assert_frame_equal(windows, again)


def test_tetrode_spike_counts_count_each_tetrode_in_half_open_windows():
    spikes = [np.array([0.001, 0.005, 0.010]), np.array([0.020]), np.array([])]
    counts = pt.tetrode_spike_counts(
        spikes, np.array([0.0, 0.010]), np.array([0.010, 0.030])
    )
    # [0, 10 ms) holds two of tetrode 0; [10, 30 ms) one of tetrode 0, one of 1
    np.testing.assert_array_equal(counts, [[2, 0, 0], [1, 1, 0]])


def test_participation_and_rate_from_counts():
    counts = np.array([[2, 0, 0], [1, 1, 0]])
    measures = pt.participation_and_rate(counts, np.array([0.010, 0.020]))
    np.testing.assert_allclose(measures["participation"], [1 / 3, 2 / 3])
    np.testing.assert_allclose(measures["spike_rate_hz"], [200.0, 100.0])
    np.testing.assert_array_equal(measures["n_active_tetrodes"], [1, 2])
    np.testing.assert_array_equal(measures["n_spikes"], [2, 2])


def test_participation_statistics_are_epoch_medians_averaged_over_animals():
    table = pd.DataFrame(
        {
            "animal": ["a", "a", "a", "a", "b", "b"],
            "epoch_id": ["a1", "a1", "a2", "a2", "b1", "b1"],
            "group": ["no SWR"] * 6,
            "kind": ["event"] * 6,
            "participation": [0.2, 0.4, 0.6, 0.6, 0.5, 0.7],
            "spike_rate_hz": [100.0, 300.0, 50.0, 50.0, 10.0, 30.0],
        }
    )
    stats = pt.participation_statistics(table)
    row = stats.loc[("no SWR", "event")]
    # a: epochs 0.3 and 0.6, mean 0.45; b: 0.6; animals' mean 0.525
    assert row["participation_animal_mean"] == pytest.approx(0.525)
    np.testing.assert_allclose(row["participation_animals"], [0.45, 0.6])
    assert row["spike_rate_hz_animal_mean"] == pytest.approx((125.0 + 20.0) / 2)
    assert row["n_animals"] == 2


def test_an_event_longer_than_any_local_rest_stretch_has_no_control():
    mask = np.zeros(100, dtype=bool)
    mask[10:30] = True
    windows = pt.control_windows(
        np.array([0, 40, 60]), np.array([4, 69, 64]), mask, seed=(1, "epoch")
    )
    # the 30-bin event has no 30-bin stretch of Local rest; the 5-bin ones do
    assert windows.loc[1, "s"] == -1
    assert windows.loc[1, "e"] == -1
    assert (windows.loc[[0, 2], "s"] >= 10).all()
    assert (windows.loc[[0, 2], "e"] < 30).all()


def test_window_stops_are_the_next_decoder_timestamp():
    time = np.array([0.0, 0.002, 0.0041, 0.006])
    np.testing.assert_allclose(
        pt.window_stops(time, np.array([0, 1, 3])), [0.002, 0.0041, 0.008]
    )


def test_participation_differences_count_the_animals_whose_events_exceed_rest():
    rows = []
    for animal, event, control in (("a", 0.6, 0.3), ("b", 0.4, 0.5)):
        for kind, value in (("event", event), ("control", control)):
            rows.append(
                {
                    "animal": animal,
                    "epoch_id": f"{animal}1",
                    "group": "no SWR",
                    "kind": kind,
                    "participation": value,
                    "spike_rate_hz": 10 * value,
                }
            )
    differences = pt.participation_differences(
        pt.participation_statistics(pd.DataFrame(rows))
    )
    row = differences.loc["no SWR"]
    assert row["participation_n_animals_above"] == 1
    assert row["spike_rate_hz_n_animals_above"] == 1
    assert row["n_animals"] == 2
