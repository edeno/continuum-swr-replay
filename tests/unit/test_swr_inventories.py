"""Tests for the Zugaro filter and its channel choice in ``swr_inventories``.

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

import numpy as np
import pytest
from numpy.typing import NDArray
from ripple_detection.core import _contiguous_valid_blocks
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, sosfilt, sosfiltfilt, sosfreqz

from continuum_swr_replay.analysis.hippocampus.swr_inventories import (
    CHANNEL_SELECTION_BAND_HZ,
    CHANNEL_SELECTION_FILTER_ORDER,
    CHANNEL_SELECTION_RMS_WINDOW_S,
    DEAD_CHANNEL_RMS,
    SETTLING_TOLERANCE,
    ZUGARO_BAND_HZ,
    ZUGARO_FILTER_ORDER,
    ZUGARO_MAXIMUM_DURATION_S,
    NoUsableChannelError,
    _centered_moving_rms,
    best_ripple_channel,
    design_bandpass_sos,
    gap_free_stretches,
    odd_window_samples,
    settling_samples,
    zero_phase_bandpass,
)

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
