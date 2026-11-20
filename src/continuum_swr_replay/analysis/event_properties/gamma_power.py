"""Slow- and fast-gamma power of CA1 on the decode grid: z-scored per epoch, and its change against quiet rest in dB (``power_change_db``).

Is a non-local event accompanied by a rise in slow gamma (CA3 input) or fast
gamma (entorhinal input; Colgin et al., 2009, *Nature* 462:353)? This gives
each epoch's gamma power on the same footing as the decode frame's other
z-scored signals (``theta_power_zscore``, ``SWR_zscore``):

1. Each kept CA1 tetrode's referenced raw LFP (the electrodes rule R-T keeps
   for the ripple consensus) is band-passed with a zero-phase 4th-order
   Butterworth filter in each band of :data:`GAMMA_BANDS`, one gap-free stretch
   of the recording at a time (:func:`stretch_band_power`).
2. Its power is the squared Hilbert envelope; the epoch's power is the mean
   over the kept tetrodes.
3. It is z-scored over the decode span outside excluded time, and read onto the
   decode grid by linear interpolation. A bin of the grid where the recording
   paused has no value and is in neither regime (:func:`recorded_mask`).
4. On the band power itself (not z), :func:`power_change_db` gives a class of
   events' change against the epoch's quiet rest:
   ``10 log10(mean power over the events' samples / mean power over rest)``.

:func:`epoch_gamma` does all of it for one epoch, with each event's mean and the
epoch's running and quiet-rest levels (``brain_state_reference``).

An event's value is a mean over its bins, and band power is skewed: a mean over
a window sits above the median of single bins when nothing differs, and the more
so the longer the window. So each class of events also gets the level of rest
(and of running) windows as long as its events (:func:`matched_level`), which is
what its events' values are to be compared with.

The fast band stops at 100 Hz, below Colgin's upper bound (about 140 Hz), to
keep clear of the ripple band (150-250 Hz) and of the broadband rise spikes
leave in the LFP during SWRs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.signal import butter, hilbert, sosfiltfilt

from continuum_swr_replay.analysis.detection.swr_detectors import gap_free_stretches
from continuum_swr_replay.analysis.event_properties.brain_state_reference import (
    EVENT_COLUMNS,
    brain_state_reference,
    matched_window_means,
    regime_masks,
)
from continuum_swr_replay.intervals import interval_mask, max_step_s, sample_step

GAMMA_BANDS: dict[str, tuple[float, float]] = {
    "slow_gamma": (25.0, 55.0),
    "fast_gamma": (65.0, 100.0),
}
"""Each band's word and its edges in Hz."""

FILTER_ORDER = 4
"""Butterworth order of each band-pass (applied forward and back)."""

FILTER_PAD = 3 * (2 * FILTER_ORDER + 1)
"""Samples ``sosfiltfilt`` pads each end with for a band-pass of
:data:`FILTER_ORDER` (its default): a signal must be longer to be filtered."""


def band_power(
    lfp: NDArray[np.float64], sampling_frequency: float, band: tuple[float, float]
) -> NDArray[np.float64]:
    """Mean over channels of each channel's band power.

    Parameters
    ----------
    lfp : NDArray, shape (n_time, n_channels)
        Referenced raw LFP, one column per tetrode.
    sampling_frequency : float
        Hz.
    band : (float, float)
        Low and high edge in Hz.

    Returns
    -------
    power : NDArray, shape (n_time,)
        The squared Hilbert envelope of each channel's band-passed signal,
        averaged over channels.
    """
    sos = butter(
        FILTER_ORDER, band, btype="bandpass", fs=sampling_frequency, output="sos"
    )
    filtered = sosfiltfilt(sos, np.asarray(lfp, dtype=np.float64), axis=0)
    envelope = np.abs(hilbert(filtered, axis=0))
    power: NDArray[np.float64] = (envelope**2).mean(axis=1)
    return power


def stretch_band_power(
    lfp: NDArray[np.float64],
    lfp_time: NDArray[np.float64],
    sampling_frequency: float,
    band: tuple[float, float],
) -> NDArray[np.float64]:
    """:func:`band_power` of each gap-free stretch of a recording, by itself.

    A recording that paused holds the samples before and after the pause side by
    side. Filtered as one signal, the step between them rings through the band;
    so each stretch (``swr_detectors.gap_free_stretches``, the rule the SWR
    detectors split their filters at) is filtered alone.

    Parameters
    ----------
    lfp : NDArray, shape (n_time, n_channels)
    lfp_time : NDArray, shape (n_time,)
        Its sample times, seconds.
    sampling_frequency : float
        Hz.
    band : (float, float)
        Low and high edge in Hz.

    Returns
    -------
    power : NDArray, shape (n_time,)
        NaN over a stretch too short to filter (the zero-phase filter pads each
        end with :data:`FILTER_PAD` samples of the signal).
    """
    power = np.full(lfp_time.size, np.nan)
    for start, stop in gap_free_stretches(lfp_time, lfp):
        if stop - start > FILTER_PAD:
            power[start:stop] = band_power(lfp[start:stop], sampling_frequency, band)
    return power


def recorded_mask(
    grid_time: NDArray[np.float64],
    lfp_time: NDArray[np.float64],
    power: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Which grid times lie in a stretch of the recording that has band power.

    Parameters
    ----------
    grid_time : NDArray, shape (n_grid,)
        Seconds.
    lfp_time : NDArray, shape (n_time,)
        The recording's sample times, seconds.
    power : NDArray, shape (n_time,)
        :func:`stretch_band_power`: NaN over the stretches too short to filter.

    Returns
    -------
    recorded : NDArray of bool, shape (n_grid,)
        True from a stretch's first sample to its last, and as far beyond each
        as two grid times may be apart without a gap between them
        (``intervals.max_step_s`` of the grid's step): a grid that runs a bin
        past the recording's last sample still reads that sample. Farther from
        every stretch, the recording gives no value.
    """
    stretches = gap_free_stretches(lfp_time, power)
    if len(stretches) == 0:
        return np.zeros(grid_time.size, dtype=bool)
    reach = max_step_s(sample_step(grid_time))
    first = lfp_time[stretches[:, 0]] - reach
    last = lfp_time[stretches[:, 1] - 1] + reach
    index = np.searchsorted(first, grid_time, side="right") - 1
    recorded: NDArray[np.bool_] = (index >= 0) & (
        grid_time <= last[np.maximum(index, 0)]
    )
    return recorded


def zscore_on(
    values: NDArray[np.float64], keep: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """``values`` z-scored by the mean and standard deviation of ``values[keep]``.

    Raises
    ------
    ValueError
        No sample is kept, or the kept samples do not vary.
    """
    kept = values[keep]
    if kept.size == 0:
        raise ValueError("no sample to z-score on")
    scale = kept.std()
    if not scale > 0:
        raise ValueError("the kept samples do not vary")
    z: NDArray[np.float64] = (values - kept.mean()) / scale
    return z


def event_means(
    grid_time: NDArray[np.float64],
    values: NDArray[np.float64],
    starts: NDArray[np.float64],
    ends: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Mean of ``values`` over the grid samples ``start <= t <= end`` of each event.

    NaN for an event that holds no grid sample, or a sample without a value
    (the recording paused there): its mean is not known.
    """
    first = np.searchsorted(grid_time, starts, side="left")
    stop = np.searchsorted(grid_time, ends, side="right")
    known = np.isfinite(values)
    cumulative = np.r_[0.0, np.cumsum(np.where(known, values, 0.0))]
    n_unknown = np.r_[0, np.cumsum(~known)]
    n = stop - first
    with np.errstate(invalid="ignore", divide="ignore"):
        means = (cumulative[stop] - cumulative[first]) / n
    means = means.astype(np.float64)
    means[(n <= 0) | (n_unknown[stop] > n_unknown[first])] = np.nan
    return means


def samples_in_intervals(
    grid_time: NDArray[np.float64],
    starts: NDArray[np.float64],
    ends: NDArray[np.float64],
) -> NDArray[np.bool_]:
    """Which grid samples lie in any interval ``start <= t <= end`` (the rule of
    :func:`event_means`)."""
    first = np.searchsorted(grid_time, starts, side="left")
    stop = np.searchsorted(grid_time, ends, side="right")
    delta = np.zeros(grid_time.size + 1, dtype=np.int64)
    np.add.at(delta, first, 1)
    np.add.at(delta, stop, -1)
    return np.cumsum(delta[:-1]) > 0


def power_change_db(
    power: NDArray[np.float64],
    during: NDArray[np.bool_],
    reference: NDArray[np.bool_],
) -> float:
    """``10 log10(mean power during / mean power in the reference)``, in dB.

    On band power (not its z-score): an event's gamma against the same epoch's
    quiet rest. 0 dB is no change, +3 dB about twice the power.

    Parameters
    ----------
    power : NDArray, shape (n_time,)
        Band power (``band_power``), on any grid.
    during, reference : NDArray of bool, shape (n_time,)
        The samples of the events and of the reference; non-finite samples are
        left out.

    Returns
    -------
    float
        NaN when either set has no finite sample.
    """
    power = np.asarray(power, dtype=np.float64)
    finite = np.isfinite(power)
    a, b = power[during & finite], power[reference & finite]
    if a.size == 0 or b.size == 0:
        return float("nan")
    return float(10.0 * np.log10(a.mean() / b.mean()))


def matched_level(windows: NDArray[np.float64]) -> float:
    """The level a class's events would have if each were a window of a regime.

    A class's level in an epoch is the median over its events of each event's
    mean. ``windows`` holds, for each event, the means of windows of a regime
    (quiet rest, running) as long as the event: each column is one draw of
    "every event replaced by such a window". The level is the median over the
    events of each draw, averaged over the draws.

    Parameters
    ----------
    windows : NDArray, shape (n_events, n_windows)
        ``brain_state_reference.matched_window_means`` of the class's events;
        NaN where an event's length fits no run of the regime.

    Returns
    -------
    float
        NaN with no event, or when no event has a window.
    """
    if windows.size == 0 or not np.isfinite(windows).any():
        return float("nan")
    return float(np.nanmean(np.nanmedian(windows, axis=0)))


def epoch_gamma(
    lfp: NDArray[np.float64],
    lfp_time: NDArray[np.float64],
    sampling_frequency: float,
    dot: pd.DataFrame,
    grid_time: NDArray[np.float64],
    excluded: pd.DataFrame,
    events: pd.DataFrame,
    n_windows: int,
    rng: np.random.Generator,
) -> tuple[
    dict[str, NDArray[np.float64]], dict[str, float], dict[str, NDArray[np.float64]]
]:
    """One epoch's gamma: each event's mean, the running and rest levels, and each class's change.

    For each band of :data:`GAMMA_BANDS`: the band power (:func:`band_power`),
    z-scored over the LFP samples inside the decode span and outside excluded
    time (:func:`zscore_on`), each read onto the decode grid by linear
    interpolation, with no value where the recording paused
    (:func:`recorded_mask`); each event's mean of both (:func:`event_means`); the level
    of quiet-rest and of running windows as long as each class's events
    (:func:`matched_level`, on the z-score); and the change of each class of
    events (SWR, non-SWR) against quiet rest in dB (:func:`power_change_db`, on
    the band power).

    Parameters
    ----------
    lfp : NDArray, shape (n_lfp, n_channels)
        Referenced raw LFP of the kept CA1 tetrodes.
    lfp_time : NDArray, shape (n_lfp,)
        Its sample times, seconds.
    sampling_frequency : float
        Hz.
    dot : pd.DataFrame, shape (n_time, n_columns)
        The epoch's decode frame read through the SWR source, with
        ``head_speed``, ``in_excluded_time`` and the event labels
        (``brain_state_reference.EVENT_COLUMNS``). Not changed.
    grid_time : NDArray, shape (n_time,)
        The decode grid (the frame's index), seconds.
    excluded : pd.DataFrame
        The epoch's excluded time, ``start_time`` and ``end_time``.
    events : pd.DataFrame, shape (n_events, n_columns)
        The epoch's events: ``event_start_s``, ``event_end_s``, ``is_swr``.
    n_windows : int
        Windows of each regime drawn per event.
    rng : np.random.Generator
        Draws the windows.

    Returns
    -------
    columns : dict of NDArray, each shape (n_events,)
        ``{band}_z`` and ``{band}_power``: each event's mean.
    reference : dict
        The running and quiet-rest levels of every column
        (``brain_state_reference``; the z under the band's word, the power
        under ``{band}_power``); ``{band}_rest_matched_{class}`` and
        ``{band}_running_matched_{class}`` (class ``swr``, ``non_swr``), the
        z-score's level in windows as long as the class's events (NaN with no
        event of the class, or no run of the regime that long); then
        ``{band}_db_swr`` and ``{band}_db_non_swr``.
    on_grid : dict of NDArray, each shape (n_time,)
        ``{band}_z``: the z-scored band power at every bin of the decode grid,
        NaN where the LFP was not recorded.
    """
    in_span = (lfp_time >= grid_time[0]) & (lfp_time <= grid_time[-1])
    zscore_keep = in_span & ~interval_mask(lfp_time, excluded)

    starts = events["event_start_s"].to_numpy(dtype=np.float64)
    ends = events["event_end_s"].to_numpy(dtype=np.float64)
    is_swr = events["is_swr"].to_numpy(dtype=bool)
    powers = {
        word: stretch_band_power(lfp, lfp_time, sampling_frequency, band)
        for word, band in GAMMA_BANDS.items()
    }
    # Every band has power over the same samples: the stretches long enough to
    # filter. A bin of the grid outside them has no gamma, so it is in no regime.
    has_power = np.isfinite(powers[next(iter(powers))])
    recorded = recorded_mask(grid_time, lfp_time, powers[next(iter(powers))])
    signals = dot[["head_speed", "in_excluded_time", *EVENT_COLUMNS]].copy()
    signals["in_excluded_time"] = (
        signals["in_excluded_time"].to_numpy(dtype=bool) | ~recorded
    )
    regimes = regime_masks(signals)
    rest = regimes["rest"]
    # The bins of each event, by the rule of ``event_means``.
    n_bins = np.searchsorted(grid_time, ends, side="right") - np.searchsorted(
        grid_time, starts, side="left"
    )
    members = {"swr": is_swr, "non_swr": ~is_swr}
    in_class = {
        "swr": samples_in_intervals(grid_time, starts[is_swr], ends[is_swr]),
        "non_swr": samples_in_intervals(grid_time, starts[~is_swr], ends[~is_swr]),
    }
    columns = {}
    change_db = {}
    matched = {}
    z_on_grid = {}
    for word, power in powers.items():
        z = zscore_on(power, zscore_keep & has_power)
        on_grid = np.where(
            recorded, np.interp(grid_time, lfp_time[has_power], z[has_power]), np.nan
        )
        signals[f"{word}_z"] = on_grid
        z_on_grid[f"{word}_z"] = on_grid
        columns[f"{word}_z"] = event_means(grid_time, on_grid, starts, ends)
        for regime, eligible in regimes.items():
            windows = matched_window_means(on_grid, eligible, n_bins, n_windows, rng)
            for group, chosen in members.items():
                matched[f"{word}_{regime}_matched_{group}"] = matched_level(
                    windows[chosen]
                )
        power_on_grid = np.where(
            recorded,
            np.interp(grid_time, lfp_time[has_power], power[has_power]),
            np.nan,
        )
        signals[f"{word}_power"] = power_on_grid
        columns[f"{word}_power"] = event_means(grid_time, power_on_grid, starts, ends)
        for group, mask in in_class.items():
            change_db[f"{word}_db_{group}"] = power_change_db(power_on_grid, mask, rest)
    reference = brain_state_reference(
        signals,
        {
            **{word: f"{word}_z" for word in GAMMA_BANDS},
            **{f"{word}_power": f"{word}_power" for word in GAMMA_BANDS},
        },
    )
    return columns, {**reference, **matched, **change_db}, z_on_grid
