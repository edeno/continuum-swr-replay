"""Tests for ``swr_inventories``: the Zugaro filter and its channel choice, the
four detectors' settings and run, Yu's threshold diagnostics, the check that
the stored ripple band reproduces from its raw input, and the Kay version
check.

The filter is the Buzsaki-lab Databank files' ``FindRipples`` band, a
Butterworth band-pass of order 3 at 120-180 Hz applied forward and backward.
Its channel is chosen as buzcode's ``bz_GetBestRippleChan`` chooses it: a
140-180 Hz order-4 band-pass, a centered moving RMS, and the ratio of the mean
RMS to the median RMS. The gains, settling lengths and window pinned here are
the ones in the design spec (``docs/design/2026-10-03-full-pipeline-refresh-
design.md``, Stage 3, "The Zugaro filter"). Every "gain" is the zero-phase
amplitude gain ``abs(H) ** 2``: forward-backward filtering applies the
single-pass response twice, so the design edges sit at 0.5 (-6 dB).

All signals are synthetic and seeded; nothing is read or written.
"""

from __future__ import annotations

import dataclasses
import json
from functools import cache
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from ripple_detection import (
    Yu_ripple_detector,
    filter_ripple_band,
    noise_threshold_diagnostics,
)
from ripple_detection.core import _contiguous_valid_blocks
from ripple_detection.simulate import simulate_session, simulate_time
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, sosfilt, sosfiltfilt, sosfreqz

from continuum_swr_replay.analysis.hippocampus import swr_inventories
from continuum_swr_replay.analysis.hippocampus.swr_inventories import (
    CHANNEL_SELECTION_BAND_HZ,
    CHANNEL_SELECTION_FILTER_ORDER,
    CHANNEL_SELECTION_RMS_WINDOW_S,
    DEAD_CHANNEL_RMS,
    DETECTOR_SETTINGS,
    SETTLING_TOLERANCE,
    ZUGARO_BAND_HZ,
    ZUGARO_FILTER_ORDER,
    ZUGARO_MAXIMUM_DURATION_S,
    NoUsableChannelError,
    _centered_moving_rms,
    band_reproduction_error,
    best_ripple_channel,
    cap_zugaro_duration,
    design_bandpass_sos,
    drop_overlapping_events,
    gap_free_stretches,
    kay_version_check,
    odd_window_samples,
    run_swr_detectors,
    settling_samples,
    yu_diagnostics,
    zero_phase_bandpass,
)
from continuum_swr_replay.analysis.inventory_comparison.schema import SessionIdentity

FS = 1000.0
"""The nominal LFP rate every filter is designed at, in Hz."""

DESIGNS = {
    "zugaro": (ZUGARO_BAND_HZ, ZUGARO_FILTER_ORDER),
    "selection": (CHANNEL_SELECTION_BAND_HZ, CHANNEL_SELECTION_FILTER_ORDER),
}
"""The two band-pass designs: Zugaro detection and channel selection."""


def _regular_time(n_time: int) -> NDArray[np.float64]:
    """Timestamps of ``n_time`` samples at ``FS``, in seconds."""
    return np.arange(n_time) / FS


def _zero_phase_gain(sos: NDArray, frequencies: list[float]) -> dict[float, float]:
    """``abs(H) ** 2`` of ``sos`` at each frequency, as the zero-phase gain."""
    _, response = sosfreqz(sos, worN=np.asarray(frequencies, dtype=float), fs=FS)
    return dict(zip(frequencies, np.abs(response) ** 2, strict=True))


def _settling(design: str) -> int:
    """``settling_samples`` of one of ``DESIGNS`` at ``FS``."""
    band_hz, order = DESIGNS[design]
    return settling_samples(design_bandpass_sos(FS, band_hz, order), FS)


def _reference_rms_statistics(
    values: NDArray, time: NDArray
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Mean and median of the selection band's 13-sample moving RMS.

    An independent reference for one gap-free stretch. At the window's ends it
    uses ``uniform_filter1d``'s nearest-value extension rather than the
    module's truncated window. On these signals that moves the statistics by
    at most about 1e-5 of themselves (measured). That is inside the 1e-4 the
    bursty test allows its score and the 1 % margins the guard test leaves,
    and well under the 5e-4 that an 11- or 15-sample window would shift the
    bursty score.

    Parameters
    ----------
    values : NDArray, shape (n_time, n_channels)
    time : NDArray, shape (n_time,)

    Returns
    -------
    mean_rms, median_rms : NDArray, shape (n_channels,)
    """
    filtered = zero_phase_bandpass(
        values, time, FS, CHANNEL_SELECTION_BAND_HZ, CHANNEL_SELECTION_FILTER_ORDER
    )
    kept = np.isfinite(filtered).all(axis=1)
    rms = np.sqrt(
        uniform_filter1d(filtered[kept] ** 2, size=13, axis=0, mode="nearest")
    )
    return rms.mean(axis=0), np.median(rms, axis=0)


# ---------------------------------------------------------------------------
# The filters
# ---------------------------------------------------------------------------


def test_zugaro_settings_are_pinned():
    """The Global Constraints' Zugaro and channel-selection settings, verbatim."""
    assert ZUGARO_BAND_HZ == (120.0, 180.0)
    assert ZUGARO_FILTER_ORDER == 3
    assert CHANNEL_SELECTION_BAND_HZ == (140.0, 180.0)
    assert CHANNEL_SELECTION_FILTER_ORDER == 4
    assert CHANNEL_SELECTION_RMS_WINDOW_S == 15 / 1250 == 0.012
    assert DEAD_CHANNEL_RMS == 1.0
    assert SETTLING_TOLERANCE == 1e-3
    assert ZUGARO_MAXIMUM_DURATION_S == 0.150


def test_zugaro_band_gains():
    """The order-3 120-180 Hz design's zero-phase gains, from the design spec."""
    sos = design_bandpass_sos(FS, ZUGARO_BAND_HZ, ZUGARO_FILTER_ORDER)
    gain = _zero_phase_gain(sos, [60.0, 100.0, 120.0, 150.0, 180.0, 200.0, 250.0])

    assert gain[120.0] == pytest.approx(0.500, abs=1e-3)
    assert gain[180.0] == pytest.approx(0.500, abs=1e-3)
    assert gain[150.0] == pytest.approx(1.000, abs=1e-3)
    assert gain[100.0] == pytest.approx(0.0222, abs=5e-4)
    assert gain[200.0] == pytest.approx(0.0573, abs=5e-4)
    assert gain[250.0] <= 0.0011
    assert gain[60.0] <= 1e-4


def test_selection_band_gains():
    """The order-4 140-180 Hz selection design is at half gain on its edges."""
    sos = design_bandpass_sos(
        FS, CHANNEL_SELECTION_BAND_HZ, CHANNEL_SELECTION_FILTER_ORDER
    )
    gain = _zero_phase_gain(sos, [140.0, 180.0])

    assert gain[140.0] == pytest.approx(0.500, abs=1e-3)
    assert gain[180.0] == pytest.approx(0.500, abs=1e-3)


def test_settling_lengths():
    """90 samples for Zugaro, about 170 for selection, rounded up to 10 ms.

    Each length also meets the definition, at the default tolerance and at two
    others. From that sample on, the single-pass impulse response stays below
    the tolerance times its peak, and within the 10 ms before it, it was still
    at or above it. Two of the extra cases settle 2-3 samples past a multiple
    of 10 (Zugaro at 3e-2 after 42 samples, selection at 1e-4 after 213), where
    rounding to nearest instead of up would fall short.
    """
    assert _settling("zugaro") == 90
    selection = _settling("selection")
    assert 160 <= selection <= 180
    assert selection % 10 == 0

    impulse = np.zeros(int(10 * FS))
    impulse[0] = 1.0
    for design, (band_hz, order) in DESIGNS.items():
        sos = design_bandpass_sos(FS, band_hz, order)
        magnitude = np.abs(sosfilt(sos, impulse))
        for tolerance in (3e-2, SETTLING_TOLERANCE, 1e-4):
            n_settle = settling_samples(sos, FS, tolerance)
            floor = tolerance * magnitude.max()
            assert n_settle % 10 == 0, (design, tolerance)
            assert np.all(magnitude[n_settle:] < floor), (design, tolerance)
            assert np.any(magnitude[n_settle - 10 : n_settle] >= floor), (
                design,
                tolerance,
            )


def test_sos_matches_transfer_function_form():
    """Second-order sections give the transfer-function form's output."""
    signal = np.random.default_rng(0).standard_normal(200_000)
    sos = design_bandpass_sos(FS, ZUGARO_BAND_HZ, ZUGARO_FILTER_ORDER)
    b, a = butter(
        ZUGARO_FILTER_ORDER, ZUGARO_BAND_HZ, btype="bandpass", fs=FS, output="ba"
    )

    from_sos = sosfiltfilt(sos, signal, padtype="odd", padlen=90)
    from_ba = filtfilt(b, a, signal, padtype="odd", padlen=90)

    assert np.max(np.abs(from_sos - from_ba)) / np.max(np.abs(from_ba)) < 1e-10


def test_passes_ripples_and_rejects_neighbors_at_1000_hz():
    """A 150 Hz sinusoid keeps its amplitude; a 100 Hz one falls below 0.03."""
    time = _regular_time(10_000)
    sinusoids = np.column_stack(
        [np.sin(2 * np.pi * 150.0 * time), np.sin(2 * np.pi * 100.0 * time)]
    )

    filtered = zero_phase_bandpass(
        sinusoids, time, FS, ZUGARO_BAND_HZ, ZUGARO_FILTER_ORDER
    )
    kept = np.isfinite(filtered).all(axis=1)
    assert kept.sum() == 10_000 - 2 * 90

    ripple_amplitude = np.sqrt(2.0 * np.mean(filtered[kept, 0] ** 2))
    assert ripple_amplitude == pytest.approx(1.0, abs=0.01)
    assert np.max(np.abs(filtered[kept, 1])) < 0.03


@pytest.mark.parametrize("design", list(DESIGNS))
def test_stretch_is_filtered_with_odd_padding_of_settling_length(design):
    """Kept rows are ``sosfiltfilt`` with odd padding of the settling length.

    The edge exclusion would hide most of a wrong padding from the transient
    test below, so the recipe itself is pinned, bit for bit.
    """
    band_hz, order = DESIGNS[design]
    n_settle = _settling(design)
    values = np.random.default_rng(6).standard_normal((2_000, 2))

    filtered = zero_phase_bandpass(values, _regular_time(2_000), FS, band_hz, order)
    expected = sosfiltfilt(
        design_bandpass_sos(FS, band_hz, order),
        values,
        axis=0,
        padtype="odd",
        padlen=n_settle,
    )

    kept = slice(n_settle, 2_000 - n_settle)
    np.testing.assert_array_equal(filtered[kept], expected[kept])


def test_integer_input_is_filtered_in_float64():
    """int16 input gives the float64 result, even where odd padding would wrap.

    The odd extension computes ``2 * x[0] - x[k]``. In int16 that wraps for
    samples near the ends of the range, so the cast must come first.
    """
    time = _regular_time(3_000)
    signal = 20_000 * np.sin(2 * np.pi * 160.0 * time) + 10_000 * np.sin(
        2 * np.pi * 7.0 * time
    )
    counts = signal.astype(np.int16)[:, None]
    counts[[0, -1], 0] = [30_000, -30_000]

    from_counts = zero_phase_bandpass(
        counts, time, FS, ZUGARO_BAND_HZ, ZUGARO_FILTER_ORDER
    )
    from_float = zero_phase_bandpass(
        counts.astype(np.float64), time, FS, ZUGARO_BAND_HZ, ZUGARO_FILTER_ORDER
    )

    assert from_counts.dtype == np.float64
    np.testing.assert_array_equal(from_counts, from_float)


# ---------------------------------------------------------------------------
# Gaps and edges (Review Focus 4)
# ---------------------------------------------------------------------------


def test_stretches_split_at_nan_and_time_jump():
    """A 1.6x median step and a NaN row in one channel make three stretches."""
    time = _regular_time(1_000)
    time[400:] += 0.6 / FS
    values = np.ones((1_000, 2))
    values[700, 1] = np.nan

    stretches = gap_free_stretches(time, values)

    np.testing.assert_array_equal(stretches, [[0, 400], [400, 700], [701, 1_000]])


def test_stretches_match_the_package_rule():
    """The stretches are ``ripple_detection``'s blocks for the same rows.

    Every step is dyadic, so timestamps and their differences are exact: a step
    of exactly 1.5x the median does not split, as the package's strict
    inequality has it, while 1.625x and 3x steps do.
    """
    base_step = 2.0**-10
    steps = np.full(99, base_step)
    steps[[30, 60]] = [1.5 * base_step, 1.625 * base_step]
    time = np.concatenate([[0.0], np.cumsum(steps)])
    np.testing.assert_array_equal(
        gap_free_stretches(time, np.zeros((100, 1))), [[0, 61], [61, 100]]
    )

    rng = np.random.default_rng(3)
    n_time = 5_000
    steps = np.full(n_time - 1, base_step)
    for multiple in (1.5, 1.625, 3.0):
        steps[rng.choice(n_time - 1, size=8, replace=False)] = multiple * base_step
    time = np.concatenate([[0.0], np.cumsum(steps)])
    values = rng.standard_normal((n_time, 3))
    values[rng.choice(n_time, size=20, replace=False), rng.integers(0, 3, 20)] = np.nan
    values[rng.choice(n_time, size=3, replace=False), 0] = np.inf

    package_blocks = _contiguous_valid_blocks(np.isfinite(values).all(axis=1), time)

    np.testing.assert_array_equal(
        gap_free_stretches(time, values),
        np.asarray(package_blocks, dtype=int).reshape(-1, 2),
    )


@pytest.mark.parametrize("gap", ["nan_row", "time_jump"])
def test_filter_never_crosses_a_gap(gap):
    """A huge pulse ending stretch 1 leaves stretch 2 as if filtered alone."""
    n_time, gap_row = 3_000, 1_500
    time = _regular_time(n_time)
    values = np.random.default_rng(1).standard_normal((n_time, 2))
    if gap == "nan_row":
        values[gap_row] = np.nan
        second_start = gap_row + 1
    else:
        time[gap_row:] += 1.0
        second_start = gap_row
    values[gap_row - 1] = 1e9

    filtered = zero_phase_bandpass(
        values, time, FS, ZUGARO_BAND_HZ, ZUGARO_FILTER_ORDER
    )
    alone = zero_phase_bandpass(
        values[second_start:],
        time[second_start:],
        FS,
        ZUGARO_BAND_HZ,
        ZUGARO_FILTER_ORDER,
    )

    assert np.isfinite(alone).any()
    np.testing.assert_array_equal(filtered[second_start:], alone)


@pytest.mark.parametrize("design", list(DESIGNS))
def test_edges_are_unobserved(design):
    """Each stretch's first and last settling rows are NaN; its interior is not."""
    band_hz, order = DESIGNS[design]
    n_settle = _settling(design)
    time = _regular_time(4_000)
    time[2_600:] += 0.5
    values = np.random.default_rng(2).standard_normal((4_000, 2))
    values[1_200, 0] = np.nan

    filtered = zero_phase_bandpass(values, time, FS, band_hz, order)

    assert np.isnan(filtered[1_200]).all()
    for start, stop in [(0, 1_200), (1_201, 2_600), (2_600, 4_000)]:
        assert np.isnan(filtered[start : start + n_settle]).all()
        assert np.isnan(filtered[stop - n_settle : stop]).all()
        assert np.isfinite(filtered[start + n_settle : stop - n_settle]).all()


@pytest.mark.parametrize("design", list(DESIGNS))
def test_short_stretch_is_unobserved(design):
    """A stretch of at most twice the settling length is all NaN.

    Stretches of ``s - 1`` (shorter than the padding), ``s`` and ``2 s`` rows are
    all NaN; one of ``2 s + 1`` rows keeps exactly its middle row.
    """
    band_hz, order = DESIGNS[design]
    n_settle = _settling(design)
    lengths = [1_000, n_settle - 1, n_settle, 2 * n_settle, 2 * n_settle + 1, 1_000]
    bounds = []
    pieces = []
    rng = np.random.default_rng(4)
    for length in lengths:
        start = sum(len(piece) for piece in pieces)
        bounds.append((start, start + length))
        pieces.append(rng.standard_normal((length, 1)))
        pieces.append(np.full((1, 1), np.nan))
    values = np.concatenate(pieces)
    time = _regular_time(len(values))

    filtered = zero_phase_bandpass(values, time, FS, band_hz, order)

    for (start, stop), length in zip(bounds, lengths, strict=True):
        observed = np.flatnonzero(np.isfinite(filtered[start:stop, 0]))
        if length <= 2 * n_settle:
            assert observed.size == 0, length
        elif length == 2 * n_settle + 1:
            np.testing.assert_array_equal(observed, [n_settle])
        else:
            assert observed.size == length - 2 * n_settle


@pytest.mark.parametrize("design", list(DESIGNS))
def test_kept_rows_have_no_edge_transient(design):
    """A 3 s stretch filtered alone matches a 60 s signal over its kept rows."""
    band_hz, order = DESIGNS[design]
    time = _regular_time(60_000)
    signal = np.random.default_rng(5).standard_normal((60_000, 1))
    long_filtered = zero_phase_bandpass(signal, time, FS, band_hz, order)

    cut = slice(30_000, 33_000)
    short_filtered = zero_phase_bandpass(signal[cut], time[cut], FS, band_hz, order)
    kept = np.isfinite(short_filtered[:, 0])

    assert kept.sum() == 3_000 - 2 * _settling(design)
    difference = np.abs(short_filtered[kept, 0] - long_filtered[cut][kept, 0])
    assert difference.max() < 1e-3 * np.nanstd(long_filtered)


# ---------------------------------------------------------------------------
# Channel selection (bz_GetBestRippleChan)
# ---------------------------------------------------------------------------


def test_rms_window_is_13_samples_at_1000_hz():
    """12 ms rounds up to an odd 13 samples; Zugaro's 11/1250 s gives 9."""
    assert odd_window_samples(0.012, 1000.0) == 13
    assert odd_window_samples(CHANNEL_SELECTION_RMS_WINDOW_S, 1000.0) == 13
    assert odd_window_samples(11 / 1250, 1000.0) == 9


def test_moving_rms_is_centered_within_stretches():
    """The 13-sample RMS is centered and stays inside each run of observed rows.

    The runs are rows 0-99, then 102-149 after two NaN rows, then 150-199 after
    a timestamp gap. Each has unit spikes in a zero signal. A spike inside a run
    spreads power 1/13 over the 13 rows centered on it. A spike on a run's
    first or last row is averaged over the 7-13 rows of that run its window
    covers. Nothing reaches the neighboring run.
    """
    time = _regular_time(200)
    time[150:] += 1.0
    filtered = np.zeros((200, 1))
    filtered[[100, 101]] = np.nan
    filtered[[50, 102, 149]] = 1.0

    rms = _centered_moving_rms(filtered, time, 13)

    expected_power = np.zeros(200)
    expected_power[44:57] = 1 / 13
    expected_power[102:109] = 1 / np.arange(7, 14)
    expected_power[143:150] = 1 / np.arange(13, 6, -1)
    expected_power[[100, 101]] = np.nan
    np.testing.assert_allclose(rms[:, 0] ** 2, expected_power, rtol=1e-12, atol=0)


def test_bursty_channel_wins():
    """Sparse 150-170 Hz bursts beat steady 160 Hz power of equal energy.

    The bursty channel has the higher electrode id and the second column, so
    neither a lowest-id nor a first-column preference can pick it.
    """
    rng = np.random.default_rng(7)
    n_time = 20_000
    time = _regular_time(n_time)
    noise = 30.0 * rng.standard_normal((n_time, 2))
    noise_floor = zero_phase_bandpass(
        noise, time, FS, CHANNEL_SELECTION_BAND_HZ, CHANNEL_SELECTION_FILTER_ORDER
    )
    assert np.all(np.sqrt(np.nanmean(noise_floor**2, axis=0)) >= 5.0)

    bursty = noise[:, 1].copy()
    burst_time = np.arange(50) / FS
    for start in range(500, n_time - 500, 1_000):
        frequency = rng.uniform(150.0, 170.0)
        burst = np.hanning(50) * np.sin(2 * np.pi * frequency * burst_time)
        bursty[start : start + 50] += 400.0 * burst
    # the steady channel's 160 Hz amplitude solves sum(steady**2) == sum(bursty**2)
    tone = np.sin(2 * np.pi * 160.0 * time)
    quadratic = [
        np.sum(tone**2),
        2.0 * np.sum(noise[:, 0] * tone),
        np.sum(noise[:, 0] ** 2) - np.sum(bursty**2),
    ]
    steady = noise[:, 0] + np.max(np.roots(quadratic)) * tone
    assert np.sum(steady**2) == pytest.approx(np.sum(bursty**2), rel=1e-9)

    channels = np.column_stack([steady, bursty])
    choice = best_ripple_channel(channels, time, FS, np.array([3, 7]))

    assert choice.electrode_id == 7
    assert choice.excluded == {}
    assert set(choice.scores) == {3, 7}
    assert choice.scores[7] > choice.scores[3]
    mean_rms, median_rms = _reference_rms_statistics(channels, time)
    assert choice.scores[7] == pytest.approx(mean_rms[1] / median_rms[1], rel=1e-4)


@pytest.mark.parametrize("statistic", ["mean", "median"])
def test_dead_channel_guard_is_pinned(statistic):
    """A channel at 0.99 RMS is excluded and one at 1.01 is kept.

    For "mean", the signal's median RMS is above its mean, so only the mean
    guard can exclude it; for "median", the reverse.
    """
    n_time = 20_000
    time = _regular_time(n_time)
    if statistic == "mean":
        tone = np.sin(2 * np.pi * 160.0 * time)
        signal = np.where((np.arange(n_time) // 250) % 5 == 0, 0.0, tone)
    else:
        rng = np.random.default_rng(8)
        signal = rng.standard_normal(n_time)
        burst = 20.0 * np.hanning(50) * np.sin(2 * np.pi * 160.0 * np.arange(50) / FS)
        for start in range(500, n_time - 500, 1_000):
            signal[start : start + 50] += burst
    mean_rms, median_rms = _reference_rms_statistics(signal[:, None], time)
    level = (mean_rms if statistic == "mean" else median_rms)[0]
    other = (median_rms if statistic == "mean" else mean_rms)[0]
    assert other > 1.03 * level

    channels = np.column_stack([signal * 0.99 / level, signal * 1.01 / level])
    choice = best_ripple_channel(channels, time, FS, np.array([5, 9]))

    assert choice.electrode_id == 9
    assert set(choice.excluded) == {5}
    assert f"{statistic} RMS" in choice.excluded[5]
    other_name = "median" if statistic == "mean" else "mean"
    assert f"{other_name} RMS" not in choice.excluded[5]


def test_all_channels_excluded_raises():
    """Two flat channels leave nothing to choose, which is an error."""
    assert issubclass(NoUsableChannelError, ValueError)
    with pytest.raises(NoUsableChannelError, match="mean RMS"):
        best_ripple_channel(
            np.zeros((5_000, 2)), _regular_time(5_000), FS, np.array([2, 4])
        )


def test_tie_goes_to_lowest_electrode_id():
    """Two identical channels tie exactly; the lower id, listed second, wins."""
    signal = np.random.default_rng(9).standard_normal(5_000)

    choice = best_ripple_channel(
        np.column_stack([signal, signal]) * 30.0,
        _regular_time(5_000),
        FS,
        np.array([40, 8]),
    )

    assert choice.scores[40] == choice.scores[8]
    assert choice.electrode_id == 8


# ---------------------------------------------------------------------------
# The four detectors on shared inputs
# ---------------------------------------------------------------------------

INJECTED_RIPPLE_TIMES = [5, 15, 25, 35, 45]
"""Seconds: the five ripples ``simulate_session`` injects into 60 s."""

SIMULATED_ELECTRODE_IDS = np.array([11, 12, 13, 14])
"""Electrode ids of the four simulated channels, unlike their column numbers."""

COUNTS_PER_SIMULATED_UNIT = 1_000.0
"""Scale from ``simulate_session``'s units to int16-count-like stored units.

Unscaled, the simulated channels have about 0.09 RMS in the 140-180 Hz
selection band, below ``DEAD_CHANNEL_RMS`` (1, in the stored units: int16
counts for Spyglass LFP), so every channel would be excluded. The scaled LFP
feeds both the ripple band and Zugaro's channel. Every detector thresholds a
z-score or a percentile of its own trace, so the scale does not change what
it detects.
"""


@cache
def _simulated_epoch(ripple_frequency: tuple[float, float]) -> SimpleNamespace:
    """A 60 s simulated session at 1000 Hz, as ``run_swr_detectors`` takes it.

    Cached, so the arrays are shared: copy before changing them.

    Returns
    -------
    SimpleNamespace
        ``time`` (n_time,); ``band``, the ripple band of every channel
        (n_time, 4); ``zugaro_lfp``, the chosen channel Zugaro-filtered
        (n_time, 1); ``speed`` (n_time,), 0 throughout; and ``windows``
        (5, 2), each injected ripple's start and end.
    """
    time = simulate_time(60_000, 1000)
    session = simulate_session(
        time, INJECTED_RIPPLE_TIMES, rng=0, ripple_frequency=ripple_frequency
    )
    lfps = COUNTS_PER_SIMULATED_UNIT * session.lfps
    choice = best_ripple_channel(lfps, time, FS, SIMULATED_ELECTRODE_IDS)
    column = np.flatnonzero(SIMULATED_ELECTRODE_IDS == choice.electrode_id)
    return SimpleNamespace(
        time=time,
        band=filter_ripple_band(lfps, sampling_frequency=1000),
        zugaro_lfp=zero_phase_bandpass(
            lfps[:, column], time, FS, ZUGARO_BAND_HZ, ZUGARO_FILTER_ORDER
        ),
        speed=session.speed,
        windows=session.ripple_windows,
    )


def _run_on(epoch: SimpleNamespace) -> dict:
    """``run_swr_detectors`` on a simulated epoch."""
    return run_swr_detectors(epoch.time, epoch.band, epoch.zugaro_lfp, epoch.speed, FS)


def _n_ripples_found(events: pd.DataFrame, windows: NDArray) -> int:
    """Count the injected ripples an event shares positive duration with."""
    start = events.start_time.to_numpy()[:, None]
    end = events.end_time.to_numpy()[:, None]
    shared = np.minimum(end, windows[:, 1]) - np.maximum(start, windows[:, 0])
    return int(np.count_nonzero((shared > 0).any(axis=0)))


def _events_with_durations(durations: list[float]) -> pd.DataFrame:
    """A detector-like event table with these durations, a second apart."""
    start = 1.0 + np.arange(len(durations))
    return pd.DataFrame(
        {
            "start_time": start,
            "end_time": start + np.asarray(durations),
            "duration": durations,
            "max_zscore": 6.0 + np.arange(len(durations)),
        },
        index=pd.Index(np.arange(1, len(durations) + 1), name="event_number"),
    )


def test_detector_settings_are_pinned():
    """Spec Stage 3's four detector settings, verbatim and read-only."""
    assert isinstance(DETECTOR_SETTINGS, MappingProxyType)
    assert DETECTOR_SETTINGS == {
        "kay": {
            "speed_threshold": 4.0,
            "minimum_duration": 0.015,
            "zscore_threshold": 2.0,
            "smoothing_sigma": 0.004,
            "close_ripple_threshold": 0.0,
        },
        "karlsson": {
            "speed_threshold": 4.0,
            "minimum_duration": 0.015,
            "zscore_threshold": 3.0,
            "smoothing_sigma": 0.004,
            "close_ripple_threshold": 0.0,
        },
        "yu": {
            "speed_threshold": 4.0,
            "minimum_duration": 0.020,
            "percentile": 99.99,
            "smoothing_sigma": 0.004,
            "close_ripple_threshold": 0.0,
            "zscore_per_channel": True,
        },
        "zugaro": {
            "speed_threshold": 4.0,
            "low_threshold": 2.0,
            "high_threshold": 5.0,
            "minimum_inter_ripple_interval": 0.050,
            "minimum_duration": 0.0,
            "maximum_duration": None,
        },
    }
    for settings in DETECTOR_SETTINGS.values():
        assert isinstance(settings, MappingProxyType)
    with pytest.raises(TypeError):
        DETECTOR_SETTINGS["kay"]["zscore_threshold"] = 3.0


def test_zugaro_keeps_150_ms_and_drops_longer():
    """0.149 and 0.150 s are kept with their rows intact; 0.151 s is dropped."""
    events = _events_with_durations([0.149, 0.150, 0.151])

    kept, n_dropped = cap_zugaro_duration(events)

    assert n_dropped == 1
    pd.testing.assert_frame_equal(kept, events.loc[[1, 2]])


def test_detectors_find_injected_ripples_at_1000_hz():
    """Each detector overlaps at least 4 of the 5 injected 150-175 Hz ripples."""
    epoch = _simulated_epoch((150, 175))

    results = _run_on(epoch)

    assert set(results) == {"kay", "karlsson", "yu", "zugaro"}
    for name, events in results.items():
        assert isinstance(events, pd.DataFrame), (name, events)
        assert _n_ripples_found(events, epoch.windows) >= 4, name


def test_zugaro_ignores_a_230_hz_ripple():
    """Zugaro's 120-180 Hz band finds none of five 229-231 Hz ripples.

    Kay's 150-250 Hz band finds them in the same run, so the ripples are
    there to find.
    """
    epoch = _simulated_epoch((229, 231))

    results = _run_on(epoch)

    assert _n_ripples_found(results["zugaro"], epoch.windows) == 0
    assert _n_ripples_found(results["kay"], epoch.windows) >= 4


def test_a_failing_detector_does_not_hide_the_others(monkeypatch):
    """Yu raising is returned as its exception; the other three still report."""
    failure = ValueError("no immobility noise to estimate the threshold from")

    def failing_yu(*args, **kwargs):
        raise failure

    monkeypatch.setattr(swr_inventories, "Yu_ripple_detector", failing_yu)
    epoch = _simulated_epoch((150, 175))

    results = _run_on(epoch)

    assert results["yu"] is failure
    for name in ("kay", "karlsson", "zugaro"):
        assert isinstance(results[name], pd.DataFrame), name
        assert _n_ripples_found(results[name], epoch.windows) >= 4, name


def test_each_detector_reads_its_own_input_and_settings(monkeypatch):
    """Kay, Karlsson and Yu read the band, Zugaro its channel, each with its settings.

    Only Zugaro's table is capped at 150 ms, with the count dropped in its
    ``attrs``; Kay keeps its 200 ms event.
    """
    calls = {}

    def recorder(name, events):
        def detector(time, signal, speed, sampling_frequency, **settings):
            calls[name] = (time, signal, speed, sampling_frequency, settings)
            return events

        return detector

    detectors = {
        "kay": "Kay_ripple_detector",
        "karlsson": "Karlsson_ripple_detector",
        "yu": "Yu_ripple_detector",
        "zugaro": "Zugaro_ripple_detector",
    }
    zugaro_events = _events_with_durations([0.149, 0.150, 0.151])
    for name, attribute in detectors.items():
        events = zugaro_events if name == "zugaro" else _events_with_durations([0.2])
        monkeypatch.setattr(swr_inventories, attribute, recorder(name, events))
    time = _regular_time(100)
    band = np.ones((100, 3))
    zugaro_lfp = np.ones((100, 1))
    speed = np.zeros(100)

    results = run_swr_detectors(time, band, zugaro_lfp, speed, FS)

    assert set(calls) == set(detectors)
    for name, (got_time, signal, got_speed, rate, settings) in calls.items():
        assert got_time is time and got_speed is speed and rate == FS, name
        assert signal is (zugaro_lfp if name == "zugaro" else band), name
        assert settings == DETECTOR_SETTINGS[name], name
    pd.testing.assert_frame_equal(results["zugaro"], zugaro_events.loc[[1, 2]])
    assert results["zugaro"].attrs["n_dropped_over_maximum_duration"] == 1
    assert len(results["kay"]) == 1


# ---------------------------------------------------------------------------
# Artifact time: one mask for every detector's normalization
# ---------------------------------------------------------------------------

ARTIFACT_S = (50.0, 56.0)
"""Seconds: a loud 200 Hz burst after the last injected ripple."""

ARTIFACT_MASK_S = (49.7, 56.3)
"""The burst padded by 0.3 s on each side, as the ``swr`` mask pads an LFP step:
the band filter and the 4 ms smoothing carry the burst about 0.1 s past it."""


@cache
def _epoch_with_an_artifact() -> SimpleNamespace:
    """``_simulated_epoch((150, 175))`` with a burst 50 times the LFP's SD.

    Unmasked, the burst inflates every detector's normalization so far that
    none finds an injected ripple (Yu cannot even estimate its threshold).
    """
    time = simulate_time(60_000, 1000)
    session = simulate_session(
        time, INJECTED_RIPPLE_TIMES, rng=0, ripple_frequency=(150, 175)
    )
    lfps = COUNTS_PER_SIMULATED_UNIT * session.lfps
    burst = (time >= ARTIFACT_S[0]) & (time < ARTIFACT_S[1])
    lfps[burst] += 50 * np.std(lfps) * np.sin(2 * np.pi * 200 * time[burst])[:, None]
    choice = best_ripple_channel(lfps, time, FS, SIMULATED_ELECTRODE_IDS)
    column = np.flatnonzero(SIMULATED_ELECTRODE_IDS == choice.electrode_id)
    return SimpleNamespace(
        time=time,
        band=filter_ripple_band(lfps, sampling_frequency=1000),
        zugaro_lfp=zero_phase_bandpass(
            lfps[:, column], time, FS, ZUGARO_BAND_HZ, ZUGARO_FILTER_ORDER
        ),
        speed=session.speed,
        windows=session.ripple_windows,
        unobserved=(time >= ARTIFACT_MASK_S[0]) & (time <= ARTIFACT_MASK_S[1]),
    )


def test_the_artifact_mask_leaves_every_detectors_normalization(monkeypatch):
    """One artifact mask reaches all four detectors as their normalization mask.

    Each normalizes over its own sample less the artifact time. Kay, Karlsson
    and Zugaro z-score over every valid sample, so theirs is every sample
    outside the mask; Yu's noise sample is the immobile samples, so its is the
    immobile samples outside the mask, an unknown speed not being immobile.
    The other settings are the pinned ones.
    """
    calls = {}

    def recorder(name):
        def detector(time, signal, speed, sampling_frequency, **settings):
            calls[name] = settings
            return _events_with_durations([0.05])

        return detector

    for name, attribute in {
        "kay": "Kay_ripple_detector",
        "karlsson": "Karlsson_ripple_detector",
        "yu": "Yu_ripple_detector",
        "zugaro": "Zugaro_ripple_detector",
    }.items():
        monkeypatch.setattr(swr_inventories, attribute, recorder(name))
    time = _regular_time(100)
    speed = np.zeros(100)
    speed[60:70] = 10.0
    speed[70:75] = np.nan
    unobserved = np.zeros(100, dtype=bool)
    unobserved[20:40] = True

    run_swr_detectors(
        time,
        np.ones((100, 3)),
        np.ones((100, 1)),
        speed,
        FS,
        unobserved=unobserved,
    )

    immobile_yu = ~unobserved
    immobile_yu[60:75] = False
    expected = {
        "kay": ~unobserved,
        "karlsson": ~unobserved,
        "yu": immobile_yu,
        "zugaro": ~unobserved,
    }
    for name, settings in calls.items():
        settings = dict(settings)
        mask = settings.pop("normalization_mask")
        assert mask.dtype == bool and mask.shape == (100,), name
        np.testing.assert_array_equal(mask, expected[name], err_msg=name)
        assert not mask[unobserved].any(), name
        assert settings == DETECTOR_SETTINGS[name], name


def test_without_an_artifact_mask_no_normalization_mask_is_passed(monkeypatch):
    """With no mask the detectors keep their own default normalization."""
    calls = {}

    def kay(time, signal, speed, sampling_frequency, **settings):
        calls["kay"] = settings
        return _events_with_durations([0.05])

    monkeypatch.setattr(swr_inventories, "Kay_ripple_detector", kay)
    time = _regular_time(100)

    run_swr_detectors(time, np.ones((100, 3)), np.ones((100, 1)), np.zeros(100), FS)

    assert calls["kay"] == DETECTOR_SETTINGS["kay"]


def test_a_masked_artifact_no_longer_hides_the_ripples():
    """A loud burst hides every injected ripple from all four detectors; with
    its time masked out of their normalization, each finds them again."""
    epoch = _epoch_with_an_artifact()

    unmasked = _run_on(epoch)
    masked = run_swr_detectors(
        epoch.time,
        epoch.band,
        epoch.zugaro_lfp,
        epoch.speed,
        FS,
        unobserved=epoch.unobserved,
    )

    for name in ("kay", "karlsson", "zugaro"):
        assert _n_ripples_found(unmasked[name], epoch.windows) == 0, name
    assert isinstance(unmasked["yu"], ValueError)
    for name, events in masked.items():
        assert isinstance(events, pd.DataFrame), (name, events)
        assert _n_ripples_found(events, epoch.windows) >= 4, name


def test_drop_overlapping_events_drops_closed_overlaps_and_renumbers():
    """An event sharing any time with an interval goes, touching included; the
    rest keep their rows, renumbered 1..n, and the drops are counted."""
    events = _events_with_durations([0.1, 0.1, 0.1, 0.1])  # 1.0, 2.0, 3.0, 4.0
    intervals = pd.DataFrame(
        {"start_time": [2.05, 3.1, 9.0], "end_time": [2.07, 3.5, 9.5]}
    )  # inside event 2; touching event 3 at its end, 3.1

    kept, n_dropped = drop_overlapping_events(events, intervals)

    assert n_dropped == 2
    np.testing.assert_array_equal(kept.start_time, [1.0, 4.0])
    assert kept.index.name == "event_number"
    np.testing.assert_array_equal(kept.index, [1, 2])
    np.testing.assert_array_equal(kept.max_zscore, events.max_zscore.loc[[1, 4]])


def test_drop_overlapping_events_with_nothing_to_drop():
    """No intervals, or no events, leave the table as it was."""
    events = _events_with_durations([0.1, 0.1])
    no_intervals = pd.DataFrame({"start_time": [], "end_time": []}, dtype=float)

    kept, n_dropped = drop_overlapping_events(events, no_intervals)
    assert n_dropped == 0
    pd.testing.assert_frame_equal(kept, events)

    no_events = events.iloc[:0]
    kept, n_dropped = drop_overlapping_events(
        no_events, pd.DataFrame({"start_time": [1.0], "end_time": [2.0]})
    )
    assert n_dropped == 0
    assert kept.empty and kept.index.name == "event_number"


# ---------------------------------------------------------------------------
# Yu's threshold diagnostics
# ---------------------------------------------------------------------------


def test_yu_diagnostics_match_the_detector():
    """The diagnostics' threshold is the events' ``detection_threshold_zscore``.

    The simulated session gets what changes Yu's noise sample: a running bout
    after the last ripple and a stretch of unknown speed, both left out; and
    NaN rows followed by a timestamp jump, which leave a 10-sample block
    between them, shorter than the 20 samples of Yu's minimum duration, so
    the detector treats it as missing.
    """
    epoch = _simulated_epoch((150, 175))
    time, band, speed = epoch.time.copy(), epoch.band.copy(), epoch.speed.copy()
    speed[(time >= 50.0) & (time < 56.0)] = 10.0
    speed[20_000:20_100] = np.nan
    band[39_985:39_990] = np.nan
    time[40_000:] += 0.5

    with pytest.warns(UserWarning, match="shorter than the 20 samples"):
        events = Yu_ripple_detector(time, band, speed, FS, **DETECTOR_SETTINGS["yu"])
    diagnostics = yu_diagnostics(time, band, speed, FS)

    assert len(events) > 0
    np.testing.assert_allclose(
        events.detection_threshold_zscore,
        diagnostics["threshold_zscore"],
        rtol=0,
        atol=1e-9,
    )
    noise = np.isfinite(band).all(axis=1) & (speed <= 4.0)
    noise[39_990:40_000] = False
    assert diagnostics["n_values"] == np.count_nonzero(noise)
    assert diagnostics["threshold_zscore"] == pytest.approx(
        (diagnostics["threshold"] - diagnostics["normalization_center"])
        / diagnostics["normalization_scale"],
        rel=1e-12,
    )
    assert len(diagnostics["counts"]) == len(diagnostics["histogram_edges"])
    json.dumps(diagnostics, allow_nan=False)


def test_yu_diagnostics_leave_out_the_artifact_time():
    """With an artifact mask, the diagnostics estimate on the noise sample the
    detector is given: immobile samples outside the mask. Their threshold is
    the masked detector's, and the masked samples are not counted."""
    epoch = _epoch_with_an_artifact()
    noise = np.isfinite(epoch.band).all(axis=1) & (epoch.speed <= 4.0)
    noise &= ~epoch.unobserved

    events = Yu_ripple_detector(
        epoch.time,
        epoch.band,
        epoch.speed,
        FS,
        normalization_mask=noise,
        **DETECTOR_SETTINGS["yu"],
    )
    diagnostics = yu_diagnostics(
        epoch.time, epoch.band, epoch.speed, FS, unobserved=epoch.unobserved
    )

    assert len(events) > 0
    np.testing.assert_allclose(
        events.detection_threshold_zscore,
        diagnostics["threshold_zscore"],
        rtol=0,
        atol=1e-9,
    )
    assert diagnostics["n_values"] == np.count_nonzero(noise)


def test_yu_diagnostics_write_a_non_finite_scalar_as_none(monkeypatch):
    """An infinite ``flank_ratio`` (mean at or below the mode) comes back as None.

    The dict then serializes as strict JSON.
    """

    def infinite_flank_ratio(*args, **kwargs):
        diagnostics = noise_threshold_diagnostics(*args, **kwargs)
        return dataclasses.replace(diagnostics, flank_ratio=np.inf)

    monkeypatch.setattr(
        swr_inventories, "noise_threshold_diagnostics", infinite_flank_ratio
    )
    epoch = _simulated_epoch((150, 175))

    diagnostics = yu_diagnostics(epoch.time, epoch.band, epoch.speed, FS)

    assert diagnostics["flank_ratio"] is None
    assert np.isfinite(diagnostics["threshold_zscore"])
    json.dumps(diagnostics, allow_nan=False)


def test_yu_diagnostics_refuse_a_zscore_per_channel_they_do_not_follow(monkeypatch):
    """With ``zscore_per_channel`` False the diagnostics raise rather than diverge.

    The detector then estimates its threshold on the immobility-normalized
    trace, not on the median of per-channel z-scores.
    """
    settings = dict(DETECTOR_SETTINGS)
    settings["yu"] = {**DETECTOR_SETTINGS["yu"], "zscore_per_channel": False}
    monkeypatch.setattr(swr_inventories, "DETECTOR_SETTINGS", settings)
    epoch = _simulated_epoch((150, 175))

    with pytest.raises(ValueError, match="zscore_per_channel"):
        yu_diagnostics(epoch.time, epoch.band, epoch.speed, FS)


# ---------------------------------------------------------------------------
# The stored band reproduces from its raw input (Review Focus 3)
# ---------------------------------------------------------------------------

BAND_KERNEL = np.array([-0.25, 0.5, -0.25])
"""A fixed kernel standing in for the 535-tap ``ripple_150_250`` filter."""


def _fake_filter_data(
    timestamps, data, filter_coeff, valid_times, electrodes, decimation
):
    """``filter_data``'s contract, by ``np.convolve``.

    The ``electrodes`` columns of ``data`` within the valid times, each
    convolved with ``filter_coeff``, decimated, and returned in ``data``'s
    dtype with their timestamps.
    """
    rows = np.zeros(len(timestamps), dtype=bool)
    for start, stop in valid_times:
        rows |= (timestamps >= start) & (timestamps <= stop)
    columns = data[rows][:, electrodes].astype(np.float64)
    filtered = np.column_stack(
        [np.convolve(column, filter_coeff, mode="same") for column in columns.T]
    )
    return filtered[::decimation].astype(data.dtype), timestamps[rows][::decimation]


def _band_inputs(dtype: str) -> dict:
    """``RippleBandInputs`` whose stored band is ``_fake_filter_data``'s output.

    The LFP has six columns, and the band's three electrodes sit in columns
    4, 1 and 2, so a column number is not an electrode id. It runs at twice
    the band's rate, so the decimation is 2, and its samples run one past each
    end of the valid interval, as make's padded read does. The stored band
    lacks the filter's first three rows, as one made by another Spyglass
    version might, so only shared timestamps can be compared.
    """
    rng = np.random.default_rng(11)
    decimation = 2
    raw_timestamps = 1.6e9 + np.arange(3_000) / (decimation * FS)
    raw = (300.0 * rng.standard_normal((3_000, 6))).astype(dtype)
    electrode_index = np.array([4, 1, 2])
    interval = SimpleNamespace(
        times=np.array([[raw_timestamps[1], raw_timestamps[-2]]])
    )
    band, time = _fake_filter_data(
        raw_timestamps, raw, BAND_KERNEL, interval.times, electrode_index, decimation
    )
    band, time = band[3:], time[3:]
    return {
        "time": time,
        "ripple_band": band.astype(np.float64),
        "ripple_band_dtype": str(band.dtype),
        "raw_referenced": raw[np.isin(raw_timestamps, time)][:, electrode_index].astype(
            np.float64
        ),
        "raw_referenced_native": raw,
        "raw_timestamps": raw_timestamps,
        "reference_overflow_count": 0,
        "electrode_ids": np.array([3, 7, 9]),
        "reference_ids": np.full(3, -1),
        "lfp_band_elect_index": electrode_index,
        "band_valid_interval": interval,
        "decimation": decimation,
        "sampling_frequency": FS,
        "band_filter_coeff": BAND_KERNEL,
        "band_filter_name": "ripple_150_250",
        "lfp_filter_name": "LFP 0-400 Hz",
        "lfp_units": "volts",
    }


@pytest.mark.parametrize("dtype", ["int16", "float64"])
def test_band_check_passes_on_its_own_filter(dtype):
    """Re-filtering with the filter that made the band gives an error of 0.

    The filter receives ``LFPBandV1.make``'s six arguments, in make's order,
    and the tolerance is 1 for an integer band, else 1e-6 of its SD.
    """
    inputs = _band_inputs(dtype)
    calls = []

    def recording_filter(*args):
        calls.append(args)
        return _fake_filter_data(*args)

    error, tolerance = band_reproduction_error(inputs, fir_filter=recording_filter)

    assert error == 0.0
    expected_tolerance = (
        1.0 if dtype == "int16" else 1e-6 * np.std(inputs["ripple_band"])
    )
    assert tolerance == pytest.approx(expected_tolerance, rel=1e-12)
    (args,) = calls
    assert len(args) == 6
    expected = [
        inputs["raw_timestamps"],
        inputs["raw_referenced_native"],
        inputs["band_filter_coeff"],
        inputs["band_valid_interval"].times,
        inputs["lfp_band_elect_index"],
    ]
    for position, (got, want) in enumerate(zip(args[:5], expected, strict=True)):
        assert got is want, position
    assert args[5] == inputs["decimation"]


@pytest.mark.parametrize(
    "swapped_columns", [(4, 5), (2, 3)], ids=["first_electrode", "last_electrode"]
)
def test_band_check_catches_a_permuted_channel_map(swapped_columns):
    """Raw columns permuted against the band put the error far past tolerance.

    Swapping one band column with a column outside the band moves that one
    electrode onto another channel, as a column-to-electrode map that took
    positions for ids would. The band's first electrode sits in column 4 and
    its last in column 2, so either one alone is caught.
    """
    inputs = _band_inputs("int16")
    order = np.arange(6)
    order[list(swapped_columns)] = swapped_columns[::-1]
    inputs["raw_referenced_native"] = inputs["raw_referenced_native"][:, order]

    error, tolerance = band_reproduction_error(inputs, fir_filter=_fake_filter_data)

    assert tolerance == 1.0
    assert error > 100 * tolerance


@pytest.mark.parametrize("n_shared", [5, 0], ids=["sliver", "none"])
def test_band_check_refuses_a_partial_reproduction(n_shared):
    """A re-filter missing stored timestamps raises, naming both counts.

    The re-filter keeps only the stored band's first ``n_shared`` timestamps,
    and every other stored row is off by 10,000 counts. Compared on shared
    timestamps alone, the sliver would pass with an error of 0; with none
    shared there would be nothing to compare.
    """
    inputs = _band_inputs("int16")
    n_stored = len(inputs["time"])
    inputs["ripple_band"][n_shared:] += 10_000

    def partial_filter(*args):
        filtered, timestamps = _fake_filter_data(*args)
        kept = np.isin(timestamps, inputs["time"][:n_shared])
        return filtered[kept], timestamps[kept]

    with pytest.raises(
        ValueError,
        match=f"Only {n_shared} of the stored band's {n_stored} timestamps",
    ):
        band_reproduction_error(inputs, fir_filter=partial_filter)


# ---------------------------------------------------------------------------
# Kay version check
# ---------------------------------------------------------------------------

KAY_CHECK_SESSION = SessionIdentity(
    animal="simrat", day=20990101, epoch="02_r1", brain_area="HPC"
)
"""A synthetic epoch for the Kay check's event keys."""


def test_kay_check_statuses():
    """Matched, added and missed events by strict positive-duration overlap.

    The direct 1.0 s event and the canonical 1.05 s event match each other.
    Direct 2.0 overlaps nothing, and direct 5.0 only touches canonical 5.1 at
    one instant, so both are added; canonical 3.0 and 5.1 are missed. The
    direct 2.0 event lies in a running bout, and is kept, flagged ineligible.
    """
    direct = pd.DataFrame(
        {"start_time": [1.0, 2.0, 5.0], "end_time": [1.1, 2.1, 5.1]},
        index=pd.Index([1, 2, 3], name="event_number"),
    )
    canonical = pd.DataFrame(
        {"start_time": [1.05, 3.0, 5.1], "end_time": [1.12, 3.1, 5.2]}
    )
    speed_time = np.arange(6_000) / 1000.0
    speed = np.where((speed_time >= 1.95) & (speed_time <= 2.15), 10.0, 0.0)

    check = kay_version_check(
        direct,
        canonical,
        session=KAY_CHECK_SESSION,
        run_id="test",
        speed_time_s=speed_time,
        speed_cm_per_s=speed,
    )

    assert list(check.columns) == [
        "detector",
        "event_key",
        "status",
        "partner_event_key",
        "start_shift_s",
        "end_shift_s",
        "whole_event_speed_eligible",
    ]
    rows = check.set_index("event_key")
    d1, d2, d3 = (f"kay:simrat20990101_02_r1:{key}" for key in (1, 2, 3))
    c0, c1, c2 = (f"kay_spyglass:simrat20990101_02_r1:{key}" for key in (0, 1, 2))
    assert rows.detector.to_dict() == {
        d1: "kay",
        d2: "kay",
        d3: "kay",
        c0: "kay_spyglass",
        c1: "kay_spyglass",
        c2: "kay_spyglass",
    }
    assert rows.status.to_dict() == {
        d1: "matched",
        d2: "added",
        d3: "added",
        c0: "matched",
        c1: "missed",
        c2: "missed",
    }
    assert rows.partner_event_key[d1] == c0
    assert rows.partner_event_key[c0] == d1
    assert rows.partner_event_key.drop([d1, c0]).isna().all()
    # direct minus canonical, the same on both rows of the pair
    np.testing.assert_allclose(rows.start_shift_s[[d1, c0]], [-0.05, -0.05])
    np.testing.assert_allclose(rows.end_shift_s[[d1, c0]], [-0.02, -0.02])
    assert rows[["start_shift_s", "end_shift_s"]].drop([d1, c0]).isna().all().all()
    assert check.whole_event_speed_eligible.dtype == bool
    assert rows.whole_event_speed_eligible.to_dict() == {
        d1: True,
        d2: False,
        d3: True,
        c0: True,
        c1: True,
        c2: True,
    }


def test_kay_check_names_the_direct_inventory():
    """A named direct inventory carries its name in its rows and keys.

    The statuses, partners and shifts are those of the default name, so Kay
    run on another input can share one table with the direct Kay without a
    key naming two events.
    """
    direct = pd.DataFrame(
        {"start_time": [1.0, 2.0], "end_time": [1.1, 2.1]},
        index=pd.Index([1, 2], name="event_number"),
    )
    canonical = pd.DataFrame({"start_time": [1.05, 3.0], "end_time": [1.12, 3.1]})
    speed_time = np.arange(4_000) / 1000.0
    arguments = {
        "session": KAY_CHECK_SESSION,
        "run_id": "test",
        "speed_time_s": speed_time,
        "speed_cm_per_s": np.zeros_like(speed_time),
    }

    named = kay_version_check(
        direct, canonical, direct_detector="kay_spyglass_inputs", **arguments
    )
    default = kay_version_check(direct, canonical, **arguments)

    d1, d2 = (f"kay_spyglass_inputs:simrat20990101_02_r1:{key}" for key in (1, 2))
    c0 = "kay_spyglass:simrat20990101_02_r1:0"
    rows = named.set_index("event_key")
    assert rows.detector[[d1, d2]].tolist() == ["kay_spyglass_inputs"] * 2
    assert rows.status[[d1, d2, c0]].tolist() == ["matched", "added", "matched"]
    assert rows.partner_event_key[c0] == d1
    renamed = default.replace({"detector": {"kay": "kay_spyglass_inputs"}}).assign(
        event_key=lambda t: t.event_key.str.replace(
            "^kay:", "kay_spyglass_inputs:", regex=True
        ),
        partner_event_key=lambda t: t.partner_event_key.str.replace(
            "^kay:", "kay_spyglass_inputs:", regex=True
        ),
    )
    pd.testing.assert_frame_equal(named, renamed)


def test_kay_check_partner_is_the_longest_overlap():
    """A matched event's partner is its longest overlap, the earlier on a tie.

    Direct 1.0-1.2 s overlaps canonical 0.95-1.05 for 0.05 s and 1.1-1.3 for
    0.1 s, so its partner is the second, and each canonical event's partner
    is the direct one. Direct 3.0-3.25 overlaps two canonical events for
    exactly 0.125 s each, and takes the earlier. Shifts are taken against
    each row's own partner.
    """
    direct = pd.DataFrame({"start_time": [1.0, 3.0], "end_time": [1.2, 3.25]})
    canonical = pd.DataFrame(
        {
            "start_time": [0.95, 1.1, 2.875, 3.125],
            "end_time": [1.05, 1.3, 3.125, 3.5],
        }
    )
    speed_time = np.arange(5_000) / 1000.0

    check = kay_version_check(
        direct,
        canonical,
        session=KAY_CHECK_SESSION,
        run_id="test",
        speed_time_s=speed_time,
        speed_cm_per_s=np.zeros_like(speed_time),
    )

    rows = check.set_index("event_key")
    d0, d1 = (f"kay:simrat20990101_02_r1:{key}" for key in (0, 1))
    c0, c1, c2, c3 = (
        f"kay_spyglass:simrat20990101_02_r1:{key}" for key in (0, 1, 2, 3)
    )
    assert (rows.status == "matched").all()
    assert rows.partner_event_key.to_dict() == {
        d0: c1,
        d1: c2,
        c0: d0,
        c1: d0,
        c2: d1,
        c3: d1,
    }
    np.testing.assert_allclose(
        rows.start_shift_s[[d0, c0, c3]], [1.0 - 1.1, 1.0 - 0.95, 3.0 - 3.125]
    )
    np.testing.assert_allclose(
        rows.end_shift_s[[d0, c0, c3]], [1.2 - 1.3, 1.2 - 1.05, 3.25 - 3.5]
    )


@pytest.mark.parametrize(
    "canonical_bounds",
    [[(2.0, 2.1)], []],
    ids=["disjoint", "no_canonical_events"],
)
def test_kay_check_partner_keys_stay_object_when_nothing_matches(canonical_bounds):
    """With no match, ``partner_event_key`` is still object, every entry missing."""
    direct = pd.DataFrame({"start_time": [1.0], "end_time": [1.1]})
    canonical = pd.DataFrame(
        canonical_bounds, columns=["start_time", "end_time"], dtype=float
    )
    speed_time = np.arange(3_000) / 1000.0

    check = kay_version_check(
        direct,
        canonical,
        session=KAY_CHECK_SESSION,
        run_id="test",
        speed_time_s=speed_time,
        speed_cm_per_s=np.zeros_like(speed_time),
    )

    assert check.partner_event_key.dtype == object
    assert check.partner_event_key.isna().all()
    assert check.status.tolist() == ["added"] + ["missed"] * len(canonical_bounds)
    assert check[["start_shift_s", "end_shift_s"]].isna().all().all()
