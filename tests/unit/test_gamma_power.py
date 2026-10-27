"""gamma_power: band power, z-scoring and event means, on simulated signals."""

import numpy as np
import pytest

from continuum_swr_replay.analysis.hippocampus.gamma_power import (
    GAMMA_BANDS,
    band_power,
    event_means,
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
