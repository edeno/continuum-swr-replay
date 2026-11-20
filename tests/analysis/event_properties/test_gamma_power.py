"""gamma_power: band power, z-scoring and event means, on simulated signals."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.event_properties.gamma_power import (
    GAMMA_BANDS,
    band_power,
    epoch_gamma,
    event_means,
    power_change_db,
    samples_in_intervals,
    stretch_band_power,
    zscore_on,
)

FS = 1000.0


def sines(freq_hz: float, n_channels: int = 3, seconds: float = 4.0) -> np.ndarray:
    t = np.arange(int(seconds * FS)) / FS
    return np.column_stack([np.sin(2 * np.pi * freq_hz * t)] * n_channels)


@pytest.mark.parametrize(
    "freq_hz, high, low",
    [(40.0, "slow_gamma", "fast_gamma"), (80.0, "fast_gamma", "slow_gamma")],
)
def test_a_sine_shows_up_in_its_own_band_only(freq_hz, high, low):
    lfp = sines(freq_hz)
    inside = band_power(lfp, FS, GAMMA_BANDS[high])[500:-500].mean()
    outside = band_power(lfp, FS, GAMMA_BANDS[low])[500:-500].mean()
    # a unit sine has power 1 (squared envelope); the other band passes almost none
    assert inside == pytest.approx(1.0, rel=0.05)
    assert outside < 0.05


def test_power_rises_during_a_burst():
    lfp = (
        sines(40.0, seconds=4.0)
        * np.r_[np.full(2000, 0.2), np.full(2000, 1.0)][:, None]
    )
    power = band_power(lfp, FS, GAMMA_BANDS["slow_gamma"])
    assert power[2500:3500].mean() > 10 * power[500:1500].mean()


def test_zscore_uses_only_the_kept_samples():
    values = np.r_[np.zeros(50), np.ones(50), np.full(10, 100.0)]
    keep = np.r_[np.ones(100, bool), np.zeros(10, bool)]
    z = zscore_on(values, keep)
    assert z[:100].mean() == pytest.approx(0.0)
    assert z[:100].std() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        zscore_on(np.ones(5), np.ones(5, bool))


def test_event_means_include_both_bounds_and_mark_empty_events():
    grid = np.arange(10.0)
    values = np.arange(10.0)
    means = event_means(grid, values, np.array([2.0, 4.5]), np.array([4.0, 4.6]))
    assert means[0] == pytest.approx(3.0)
    assert np.isnan(means[1])


def test_event_means_survive_missing_samples_elsewhere():
    """A missing sample makes the events that hold it unknown, and no other."""
    grid = np.arange(10.0)
    values = np.arange(10.0)
    values[7] = np.nan
    means = event_means(
        grid, values, np.array([2.0, 6.0, 8.0]), np.array([4.0, 8.0, 9.0])
    )
    assert means[0] == pytest.approx(3.0)
    assert np.isnan(means[1])
    assert means[2] == pytest.approx(8.5)


def test_band_power_is_filtered_within_each_gap_free_stretch():
    """Where the recording paused, the samples before and after are not joined:
    each stretch is filtered by itself, so a step across the pause leaves no
    ringing, and a stretch too short to filter has no power."""
    band = GAMMA_BANDS["slow_gamma"]
    time = np.arange(4000) / FS
    lfp = sines(40.0, seconds=4.0)
    lfp[time >= 2.5] += 50.0  # the signal comes back at another level
    kept = (time < 1.5) | (time >= 2.5)

    power = stretch_band_power(lfp[kept], time[kept], FS, band)

    expected = np.r_[
        band_power(lfp[time < 1.5], FS, band), band_power(lfp[time >= 2.5], FS, band)
    ]
    np.testing.assert_allclose(power, expected)
    # joined, the step rings through the band
    assert band_power(lfp[kept], FS, band)[1500] > 100 * power[1500]

    short = (time < 1.5) | ((time >= 2.0) & (time < 2.01)) | (time >= 2.5)
    power = stretch_band_power(lfp[short], time[short], FS, band)
    assert np.isnan(power[1500:1510]).all()
    assert np.isfinite(np.delete(power, np.arange(1500, 1510))).all()


@pytest.mark.parametrize("band, freq_hz", [("slow_gamma", 40.0), ("fast_gamma", 80.0)])
@pytest.mark.parametrize("gain", [2.0, 0.5, 1.0])
def test_a_known_power_ratio_comes_back_in_db(band, freq_hz, gain):
    # 20 s of a gamma sine in noise; in "event" seconds the sine's amplitude is
    # multiplied by `gain`. Far from the edges of each stretch, the band power
    # ratio is gain**2 (the noise adds the same small power to both), so the
    # change is close to 20 log10(gain) dB.
    rng = np.random.default_rng(0)
    t = np.arange(int(20 * FS)) / FS
    in_event = (t % 2.0) >= 1.0
    amplitude = np.where(in_event, gain, 1.0)
    lfp = (amplitude * np.sin(2 * np.pi * freq_hz * t))[:, None] + rng.normal(
        0, 0.05, (t.size, 1)
    )
    power = band_power(lfp, FS, GAMMA_BANDS[band])
    interior = ((t % 1.0) > 0.2) & ((t % 1.0) < 0.8) & (t > 1) & (t < 19)
    db = power_change_db(power, in_event & interior, ~in_event & interior)
    assert db == pytest.approx(20 * np.log10(gain), abs=0.15)


def test_power_change_db_needs_both_sets():
    power = np.array([1.0, 2.0, np.nan])
    assert np.isnan(
        power_change_db(power, np.array([False, False, True]), np.ones(3, bool))
    )
    assert power_change_db(
        power, np.array([False, True, False]), np.array([True, False, False])
    ) == pytest.approx(10 * np.log10(2))


def test_samples_in_intervals_matches_event_means_rule():
    grid = np.arange(10) * 0.002
    mask = samples_in_intervals(
        grid, np.array([0.002, 0.011]), np.array([0.006, 0.014])
    )
    np.testing.assert_array_equal(np.flatnonzero(mask), [1, 2, 3, 6, 7])


# --- one epoch ---------------------------------------------------------------------


def _epoch():
    """8 s of LFP at 1 kHz: a slow-gamma sine of amplitude 0.2, raised to 1
    during 3.0-3.5 s (an SWR event) and 100 during 7.0-7.5 s (excluded time).
    The decode grid steps by 2 ms over 0.5-7.9 s; the animal rests below
    4 cm/s except while running at 20 cm/s over 5-6 s. A non-SWR event at
    4.0-4.5 s holds no raised gamma."""
    time = np.arange(8000) / FS
    gain = np.full(time.size, 0.2)
    gain[(time >= 3.0) & (time < 3.5)] = 1.0
    gain[(time >= 7.0) & (time < 7.5)] = 100.0
    lfp = sines(40.0, n_channels=2, seconds=8.0) * gain[:, None]
    grid = np.round(np.arange(0.5, 7.9, 0.002), 6)
    label = ((grid >= 3.0) & (grid <= 3.5)) | ((grid >= 4.0) & (grid <= 4.5))
    dot = pd.DataFrame(
        {
            "head_speed": np.where((grid >= 5.0) & (grid < 6.0), 20.0, 1.0),
            "in_excluded_time": (grid >= 6.9) & (grid <= 7.6),
            "non_local_event": label.astype(int),
            "SWR_event": ((grid >= 3.0) & (grid <= 3.5)).astype(int),
            "MUA_event": 0,
        },
        index=grid,
    )
    excluded = pd.DataFrame({"start_time": [6.9], "end_time": [7.6]})
    events = pd.DataFrame(
        {
            "event_start_s": [3.0, 4.0],
            "event_end_s": [3.5, 4.5],
            "is_swr": [True, False],
        }
    )
    return lfp, time, dot, grid, excluded, events


def test_epoch_gamma_measures_events_and_references_on_analyzed_time():
    lfp, time, dot, grid, excluded, events = _epoch()
    before = dot.copy()

    columns, reference, on_grid = epoch_gamma(
        lfp, time, FS, dot, grid, excluded, events, 5, np.random.default_rng(0)
    )

    pd.testing.assert_frame_equal(dot, before)
    # the z-scored power of each band at every bin of the decode grid
    assert list(on_grid) == ["slow_gamma_z", "fast_gamma_z"]
    assert on_grid["slow_gamma_z"].shape == grid.shape
    assert list(columns) == [
        "slow_gamma_z",
        "slow_gamma_power",
        "fast_gamma_z",
        "fast_gamma_power",
    ]
    # power 1 in the SWR event, 0.04 elsewhere (a sine's squared envelope)
    np.testing.assert_allclose(columns["slow_gamma_power"], [1.0, 0.04], rtol=0.1)
    assert reference["slow_gamma_power_rest_median"] == pytest.approx(0.04, rel=0.05)
    assert reference["slow_gamma_power_running_median"] == pytest.approx(0.04, rel=0.05)
    assert reference["n_running_bins"] == 500
    # each class against the mean power over quiet rest: one SWR event, so its
    # samples' mean is the event's mean
    rest = reference["slow_gamma_power_rest_mean"]
    assert reference["slow_gamma_db_swr"] == pytest.approx(
        10 * np.log10(columns["slow_gamma_power"][0] / rest)
    )
    assert reference["slow_gamma_db_non_swr"] == pytest.approx(
        10 * np.log10(columns["slow_gamma_power"][1] / rest)
    )
    assert (
        reference["slow_gamma_db_swr"] > 12.0 > 1.0 > reference["slow_gamma_db_non_swr"]
    )
    # the burst in excluded time does not enter the z-score: with it (power
    # 10^4 for 0.5 s) the SWR event's z would be below 0
    z = columns["slow_gamma_z"]
    assert z[0] > 2.0
    assert z[1] < 0.0
    assert reference["slow_gamma_rest_median"] < 0
    assert list(reference)[-4:] == [
        "slow_gamma_db_swr",
        "slow_gamma_db_non_swr",
        "fast_gamma_db_swr",
        "fast_gamma_db_non_swr",
    ]


def test_epoch_gamma_is_unknown_where_the_lfp_was_not_recorded():
    """The recording pauses over 1.0-2.0 s, and the 0.1 s on each side of the
    pause are 30 times larger; excluded time covers those edges with the
    artifact rule's 0.3 s reach, not the middle of the pause. The gamma of the
    pause is unknown: no value on the grid, no rest bin, no part in a
    reference."""
    lfp, time, dot, grid, excluded, events = _epoch()
    _, whole, _ = epoch_gamma(
        lfp, time, FS, dot, grid, excluded, events, 5, np.random.default_rng(0)
    )
    lfp = lfp.copy()
    lfp[((time >= 0.9) & (time < 1.0)) | ((time >= 2.0) & (time < 2.1))] *= 30.0
    recorded = (time < 1.0) | (time >= 2.0)
    edges = pd.DataFrame({"start_time": [0.6, 1.7], "end_time": [1.3, 2.4]})
    excluded = pd.concat([edges, excluded], ignore_index=True)
    dot = dot.copy()
    for start, end in zip(edges["start_time"], edges["end_time"], strict=True):
        dot.loc[(grid >= start) & (grid <= end), "in_excluded_time"] = True
    paused = (grid > 1.0) & (grid < 2.0)

    columns, reference, on_grid = epoch_gamma(
        lfp[recorded], time[recorded], FS, dot, grid, excluded, events, 5,
        np.random.default_rng(0),
    )  # fmt: skip

    for band in on_grid.values():
        assert np.isnan(band[paused]).all()
        assert np.isfinite(band[~paused]).all()
    # the events and the rest level are what they are without the pause
    np.testing.assert_allclose(columns["slow_gamma_power"], [1.0, 0.04], rtol=0.1)
    # (within a quarter: the loud edges still leak a little past excluded time;
    # read across the pause, the mean would be many times larger)
    assert reference["slow_gamma_power_rest_mean"] == pytest.approx(
        whole["slow_gamma_power_rest_mean"], rel=0.25
    )
    assert reference["slow_gamma_power_rest_median"] == pytest.approx(0.04, rel=0.05)
    assert reference["slow_gamma_rest_matched_non_swr"] < 0
    rest = (
        (dot["head_speed"] < 4)
        & ~dot["in_excluded_time"]
        & (dot["non_local_event"] == 0)
        & ~paused
    )
    assert reference["n_rest_bins"] == int(rest.sum())


def _rest_only_epoch(n_events: int = 1500, event_s: float = 0.05):
    """240 s of white-noise LFP at 1 kHz with the animal at rest throughout. The
    non-SWR events, one every 0.15 s, are windows of that same rest: nothing
    happens in them."""
    rng = np.random.default_rng(0)
    time = np.arange(240_000) / FS
    lfp = rng.standard_normal((time.size, 2))
    grid = np.round(np.arange(1.0, 239.0, 0.002), 6)
    starts = np.round(1.5 + 0.15 * np.arange(n_events), 6)
    ends = np.round(starts + event_s, 6)
    label = np.zeros(grid.size, dtype=int)
    for k, (start, end) in enumerate(zip(starts, ends, strict=True), start=1):
        label[(grid >= start) & (grid <= end)] = k
    dot = pd.DataFrame(
        {
            "head_speed": 1.0,
            "in_excluded_time": False,
            "non_local_event": label,
            "SWR_event": 0,
            "MUA_event": 0,
        },
        index=grid,
    )
    excluded = pd.DataFrame({"start_time": [], "end_time": []})
    events = pd.DataFrame(
        {"event_start_s": starts, "event_end_s": ends, "is_swr": False}
    )
    return lfp, time, dot, grid, excluded, events


def test_an_events_reference_is_rest_windows_of_its_own_length():
    """Band power is skewed, so a mean over an event's bins sits above the median
    of single rest bins even when the event is itself rest. Against rest windows
    as long as the events, it does not."""
    lfp, time, dot, grid, excluded, events = _rest_only_epoch()
    columns, reference, _ = epoch_gamma(
        lfp, time, FS, dot, grid, excluded, events, 50, np.random.default_rng(1)
    )
    for band in GAMMA_BANDS:
        event_level = np.median(columns[f"{band}_z"])
        assert event_level - reference[f"{band}_rest_median"] > 0.08
        assert abs(event_level - reference[f"{band}_rest_matched_non_swr"]) < 0.03
        # no SWR event, and the animal never runs: no level to give
        assert np.isnan(reference[f"{band}_rest_matched_swr"])
        assert np.isnan(reference[f"{band}_running_matched_non_swr"])
