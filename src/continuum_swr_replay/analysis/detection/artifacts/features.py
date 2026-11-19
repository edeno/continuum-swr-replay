"""The artifact detector's features: S window spectra, J5 and L20 steps, spike lines.

Every LFP threshold is in robust per-channel units: a statistic over its own
epoch MAD x 1.4826 (:func:`robust_scale`), never absolute LFP units. The
definitions are the validated prototype's. The detector (``detect.py``) passes the LFP as float32, as the
prototype did, so its thresholds see the same numbers.

Shapes: ``n_time`` LFP samples, ``n_channels`` CA1 channels, ``n_references``
reference electrodes, ``n_windows`` 1-s windows, ``n_spikes`` spikes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation, median_filter
from scipy.spatial import cKDTree

__all__ = [
    "J5_REACH",
    "L20_REACH",
    "MADN_FACTOR",
    "SPIKE_LINE_COLUMNS",
    "channel_specificity",
    "level_jump",
    "level_shift",
    "nearest_mark_distance",
    "persistent_step_count",
    "reference_jump",
    "robust_jump",
    "robust_scale",
    "spike_lines",
    "window_band_power",
    "window_centers",
    "window_valid",
]

MADN_FACTOR = 1.4826
"""MAD x 1.4826 estimates a normal SD: the robust unit of every LFP threshold."""

SCALE_FLOOR = 1e-6
"""Smallest robust scale divided by, so a flat channel gives 0, not NaN
(the prototype's floor)."""

J5_REACH = 5
"""J5 at sample t reads the samples t-5 .. t+5."""
L20_REACH = 24
"""L20 at sample t reads the samples t-24 .. t+24."""

SPIKE_LINE_COLUMNS = [
    "time",
    "n_tetrodes",
    "tetrode_fraction",
    "n_spikes",
    "ood_fraction",
]


def robust_scale(
    values: NDArray[np.floating], valid: NDArray[np.bool_] | None = None
) -> NDArray[np.floating]:
    """Each column's MAD x 1.4826 over the valid rows.

    Parameters
    ----------
    values : ndarray, shape (n_rows, n_columns)
    valid : ndarray of bool, shape (n_rows,), optional
        Rows to use; all when None.

    Returns
    -------
    ndarray, shape (n_columns,)
        In `values`' dtype.
    """
    data = values if valid is None or bool(np.all(valid)) else values[valid]
    center = np.median(data, axis=0)
    scale: NDArray[np.floating] = MADN_FACTOR * np.median(np.abs(data - center), axis=0)
    return scale


def window_valid(valid: NDArray[np.bool_], reach: int) -> NDArray[np.bool_]:
    """True at each sample whose window ``t - reach .. t + reach`` holds only
    valid samples.

    A sample's step value (J5 with ``reach = 5``, L20 with ``reach = 24``) is
    defined only there: elsewhere its window reads across an invalid sample.
    Samples beyond the array's ends count as valid; the step features handle
    the ends themselves.

    Parameters
    ----------
    valid : ndarray of bool, shape (n_time,)
    reach : int
        Samples on each side.

    Returns
    -------
    ndarray of bool, shape (n_time,)
    """
    invalid = ~np.asarray(valid, dtype=bool)
    if not invalid.any():
        return np.ones(invalid.size, dtype=bool)
    near_invalid: NDArray[np.bool_] = binary_dilation(
        invalid, structure=np.ones(2 * reach + 1, dtype=bool)
    )
    return ~near_invalid


def window_centers(
    time: NDArray[np.float64], sampling_frequency: float, window_s: float = 1.0
) -> NDArray[np.float64]:
    """Each whole window's mean sample time.

    Windows are consecutive runs of ``round(window_s * sampling_frequency)``
    samples from the first sample; the samples after the last whole window are
    in none.

    Returns
    -------
    ndarray, shape (n_windows,)
    """
    n_window = int(round(window_s * sampling_frequency))
    n_windows = time.size // n_window
    centers: NDArray[np.float64] = (
        time[: n_windows * n_window].reshape(n_windows, n_window).mean(axis=1)
    )
    return centers


def window_band_power(
    lfp: NDArray[np.floating],
    sampling_frequency: float,
    *,
    window_s: float = 1.0,
    segment_s: float = 0.064,
    band_hz: tuple[float, float] = (150.0, 250.0),
    valid: NDArray[np.bool_] | None = None,
    windows_per_chunk: int = 256,
) -> NDArray[np.float32]:
    """Welch band power per window and channel.

    Each window (as :func:`window_centers`) is cut into as many whole segments
    of ``segment_s`` as fit (15 of 64 ms in 1 s, from the window's start). Each
    segment has its mean removed and a Hann taper applied; the window's
    spectrum is the mean of the segments' squared FFT magnitudes, and its band
    power is the sum over ``band_hz[0] <= f < band_hz[1]``. A window holding
    an invalid sample has no power (NaN).

    Parameters
    ----------
    lfp : ndarray, shape (n_time, n_channels)
    sampling_frequency : float
        Hz.
    window_s, segment_s : float
        Seconds.
    band_hz : tuple of float
        Lower bound inclusive, upper exclusive.
    valid : ndarray of bool, shape (n_time,), optional
        Which samples were observed; all when None.
    windows_per_chunk : int
        Windows transformed at once; only memory depends on it.

    Returns
    -------
    ndarray of float32, shape (n_windows, n_channels)
        NaN in every channel of a window that holds an invalid sample.
    """
    n_window = int(round(window_s * sampling_frequency))
    n_segment = int(round(segment_s * sampling_frequency))
    n_segments = n_window // n_segment
    n_windows = lfp.shape[0] // n_window
    n_channels = lfp.shape[1]
    freqs = np.fft.rfftfreq(n_segment, 1.0 / sampling_frequency)
    in_band = (freqs >= band_hz[0]) & (freqs < band_hz[1])
    taper = np.hanning(n_segment)[None, None, :, None]
    power = np.empty((n_windows, n_channels), dtype=np.float32)
    for first in range(0, n_windows, windows_per_chunk):
        last = min(first + windows_per_chunk, n_windows)
        chunk = lfp[first * n_window : last * n_window].reshape(
            last - first, n_window, n_channels
        )[:, : n_segments * n_segment]
        segments = chunk.reshape(last - first, n_segments, n_segment, n_channels)
        segments = segments - segments.mean(axis=2, keepdims=True)
        spectra = (
            (np.abs(np.fft.rfft(segments * taper, axis=2)) ** 2)
            .mean(axis=1)
            .astype(np.float32)
        )  # (windows, n_freqs, n_channels)
        power[first:last] = spectra[:, in_band].sum(axis=1)
    if valid is not None:
        unobserved = ~np.asarray(valid, dtype=bool)[: n_windows * n_window]
        power[unobserved.reshape(n_windows, n_window).any(axis=1)] = np.nan
    return power


def channel_specificity(power: NDArray[np.floating]) -> NDArray[np.floating]:
    """Per window: the 2nd-largest channel ratio over the median ratio.

    A channel's ratio is its window power over its own epoch-median window
    power, the median taken over the windows with power. Taking the 2nd-largest
    ratio means one channel alone never raises the specificity. Rule S tells
    sustained noise from SWRs by duration (:func:`~.detect.s_cores` needs 2
    consecutive windows), not by how many channels an SWR raises.

    Parameters
    ----------
    power : ndarray, shape (n_windows, n_channels)
        From :func:`window_band_power`; at least 2 channels. A window with a
        non-finite power has no specificity.

    Returns
    -------
    ndarray, shape (n_windows,)
        NaN for a window without power, which never reaches an S threshold.

    Raises
    ------
    ValueError
        If there are fewer than 2 channels, no window has power, or a
        channel's epoch-median power is not positive (a flat channel), which
        would make every ratio of that channel undefined.
    """
    if power.shape[1] < 2:
        raise ValueError("channel specificity needs at least 2 channels")
    observed = np.isfinite(power).all(axis=1)
    if not observed.any():
        raise ValueError("no window has ripple-band power on every channel")
    median_power = np.median(power[observed], axis=0)
    if not (median_power > 0).all():
        flat = np.flatnonzero(~(median_power > 0)).tolist()
        raise ValueError(
            f"CA1 channel columns {flat} have no ripple-band power in most "
            "windows, so the channel specificity is undefined"
        )
    ratio = power[observed] / median_power
    second = -np.partition(-ratio, 1, axis=1)[:, 1]
    specificity: NDArray[np.floating] = np.full(
        power.shape[0], np.nan, dtype=ratio.dtype
    )
    specificity[observed] = second / np.median(ratio, axis=1)
    return specificity


def level_jump(lfp: NDArray[np.floating]) -> NDArray[np.floating]:
    """J5: the 5-ms median level after a sample minus the one before it.

    ``J5[t] = median(x[t+1..t+5]) - median(x[t-5..t-1])``, from a 5-sample
    running median with the edge sample repeated (``mode="nearest"``); 0 for
    the first and last 3 samples.

    Parameters
    ----------
    lfp : ndarray, shape (n_time, n_channels)

    Returns
    -------
    ndarray, shape (n_time, n_channels)
        In `lfp`'s units and dtype.
    """
    running = median_filter(lfp, size=(5, 1), mode="nearest")
    jump = np.zeros_like(lfp)
    jump[3:-3] = running[6:] - running[:-6]
    return jump


def level_shift(lfp: NDArray[np.floating]) -> NDArray[np.floating]:
    """L20: the 20-ms mean level after a sample minus the one before it.

    ``L20[t] = mean(x[t+5..t+24]) - mean(x[t-24..t-5])``, skipping 4 samples on
    each side of ``t``, for ``t = 24 .. n_time - 26``; 0 for the first 24 and
    the last 25 samples. Both windows would fit at ``t = n_time - 25`` too, but
    the prototype left it 0 and so does this. Sums are taken in float64.

    Parameters
    ----------
    lfp : ndarray, shape (n_time, n_channels)

    Returns
    -------
    ndarray, shape (n_time, n_channels)
        In `lfp`'s units and dtype.
    """
    n_time = lfp.shape[0]
    shift = np.zeros_like(lfp)
    if n_time < 50:
        return shift
    for channel in range(lfp.shape[1]):  # one channel at a time bounds memory
        total = np.zeros(n_time + 1)
        np.cumsum(lfp[:, channel], dtype=np.float64, out=total[1:])
        # t runs over 24 .. n_time - 26; total[k] is the sum of x[0..k-1], so
        # after = x[t+5..t+24] = total[t+25] - total[t+5] and
        # before = x[t-24..t-5] = total[t-4] - total[t-24]
        after = (total[49:n_time] - total[29 : n_time - 20]) / 20.0
        before = (total[20 : n_time - 29] - total[: n_time - 49]) / 20.0
        shift[24 : n_time - 25, channel] = after - before
    return shift


def robust_jump(
    jump: NDArray[np.floating], valid: NDArray[np.bool_] | None = None
) -> NDArray[np.floating]:
    """|J5| in robust units: over each channel's MAD x 1.4826 of J5.

    Parameters
    ----------
    jump : ndarray, shape (n_time, n_channels)
        From :func:`level_jump`.
    valid : ndarray of bool, shape (n_time,), optional
        The samples with a J5 value (``window_valid(observed, J5_REACH)``):
        the scale is estimated from them, and every other sample is 0, so it
        cannot reach a threshold. All samples when None.

    Returns
    -------
    ndarray, shape (n_time, n_channels)
    """
    scale = robust_scale(jump, valid)
    robust: NDArray[np.floating] = np.abs(jump) / np.maximum(scale, SCALE_FLOOR)
    if valid is not None:
        robust[~np.asarray(valid, dtype=bool)] = 0
    return robust


def reference_jump(
    reference_lfp: NDArray[np.floating], valid: NDArray[np.bool_] | None = None
) -> NDArray[np.floating]:
    """The largest robust |J5| over the reference electrodes, per sample.

    Parameters
    ----------
    reference_lfp : ndarray, shape (n_time, n_references)
        Raw, unreferenced reference electrodes.
    valid : ndarray of bool, shape (n_time,), optional
        The samples with a J5 value, as :func:`robust_jump` takes them.

    Returns
    -------
    ndarray, shape (n_time,)
        0 everywhere when there is no reference electrode.
    """
    if reference_lfp.shape[1] == 0:
        return np.zeros(reference_lfp.shape[0], dtype=reference_lfp.dtype)
    largest: NDArray[np.floating] = robust_jump(level_jump(reference_lfp), valid).max(
        axis=1
    )
    return largest


def persistent_step_count(
    jump: NDArray[np.floating],
    robust: NDArray[np.floating],
    shift: NDArray[np.floating],
    *,
    threshold: float,
    persistence: float,
) -> NDArray[np.int64]:
    """How many channels make a persistent step at each sample.

    A channel's step persists when ``robust >= threshold``, L20 has J5's sign,
    and ``|L20| >= persistence * |J5|``: the level stays shifted for tens of ms,
    as after a step artifact, and not as after a ripple cycle.

    Parameters
    ----------
    jump, robust, shift : ndarray, shape (n_time, n_channels)
        J5, robust |J5| and L20.
    threshold : float
        Robust units.
    persistence : float
        The fraction of |J5| that |L20| must keep.

    Returns
    -------
    ndarray of int, shape (n_time,)
    """
    persistent = (
        (robust >= threshold)
        & (np.sign(shift) == np.sign(jump))
        & (np.abs(shift) >= persistence * np.abs(jump))
    )
    count: NDArray[np.int64] = persistent.sum(axis=1, dtype=np.int64)
    return count


def nearest_mark_distance(
    marks: NDArray[np.floating],
    is_reference: NDArray[np.bool_],
    times: NDArray[np.floating],
    *,
    own_event_s: float,
) -> NDArray[np.float64]:
    """Each spike's Euclidean distance to the nearest reference mark outside its own event.

    The prototype's mark space and metric: the tetrode's raw marks (µV),
    Euclidean. A spike's own event is every mark of the same tetrode at most
    `own_event_s` from it in time, the spike itself included (Δt = 0); those
    marks are left out of its search. Within a neuron's absolute refractory
    period (about 1 ms) two detections on one tetrode are either two units,
    whose marks differ and are not each other's nearest neighbor among the
    tetrode's marks, or one waveform detected twice. Leaving them out stops an
    artifact detected twice from holding its own twin in distribution; on the
    23 acceptance epochs it made 20 other spikes OOD and changed no other
    core. An exact duplicate of a mark more
    than `own_event_s` away still counts as support.

    The search is exact: the k nearest reference marks are found with k = 1 +
    the most reference marks inside any spike's own event (capped at the
    reference size), so they always hold the nearest one outside it.

    Parameters
    ----------
    marks : ndarray, shape (n_spikes, n_features)
        One tetrode's marks.
    is_reference : ndarray of bool, shape (n_spikes,)
        Which spikes form the reference set.
    times : ndarray, shape (n_spikes,)
        The spikes' times (seconds), in any order.
    own_event_s : float
        A reference mark at most this far from the spike in time is its own
        event; only marks strictly farther away are searched.

    Returns
    -------
    ndarray, shape (n_spikes,)
        In the marks' units; inf when no reference mark lies outside the
        spike's own event.
    """
    marks = np.asarray(marks, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    n_spikes = marks.shape[0]
    distance = np.full(n_spikes, np.inf)
    reference_index = np.flatnonzero(is_reference)
    if reference_index.size == 0 or n_spikes == 0:
        return distance
    reference_time = times[reference_index]
    # Reference marks inside each spike's own event, counted on a window a few
    # float spacings wider, so rounding in t +- own_event_s cannot undercount
    # (a larger k stays exact).
    sorted_time = np.sort(reference_time)
    reach = own_event_s + 4 * np.spacing(np.abs(times).max())
    n_own = np.searchsorted(sorted_time, times + reach, side="right")
    n_own -= np.searchsorted(sorted_time, times - reach, side="left")
    k = int(min(n_own.max() + 1, reference_index.size))
    tree = cKDTree(marks[reference_index])
    found, neighbor = tree.query(marks, k=k)
    found = found.reshape(n_spikes, k)
    neighbor = neighbor.reshape(n_spikes, k)
    outside = np.abs(reference_time[neighbor] - times[:, None]) > own_event_s
    first = outside.argmax(axis=1)  # the nearest outside the own event
    nearest: NDArray[np.float64] = np.where(
        outside.any(axis=1), found[np.arange(n_spikes), first], np.inf
    )
    return nearest


def spike_lines(
    spike_times: NDArray[np.float64],
    tetrode: NDArray[np.integer],
    is_ood: NDArray[np.bool_],
    *,
    n_tetrodes: int,
    start_time: float,
    bin_s: float,
    min_tetrodes: int = 1,
) -> pd.DataFrame:
    """Spike synchrony in short bins, on two grids offset by half a bin.

    Bins are ``bin_s`` wide from ``start_time`` (grid 0) and from
    ``start_time + bin_s / 2`` (grid 1), so a line across one grid's bin edge
    is whole on the other's.

    Parameters
    ----------
    spike_times : ndarray, shape (n_spikes,)
        Seconds.
    tetrode : ndarray of int, shape (n_spikes,)
        Each spike's tetrode, ``0 .. n_tetrodes - 1``.
    is_ood : ndarray of bool, shape (n_spikes,)
        Whether each spike's mark is out of distribution.
    n_tetrodes : int
        The epoch's tetrodes with any spike: the fraction's denominator.
    start_time, bin_s : float
        Seconds.
    min_tetrodes : int
        Bins with fewer distinct tetrodes are left out.

    Returns
    -------
    pd.DataFrame
        One row per bin with a spike on at least `min_tetrodes` tetrodes, both
        grids, sorted by ``time`` (the bin's center): ``n_tetrodes`` (distinct
        tetrodes), ``tetrode_fraction`` (over `n_tetrodes`), ``n_spikes``
        (every spike in the bin) and ``ood_fraction`` (OOD spikes over
        ``n_spikes``).
    """
    times = np.asarray(spike_times, dtype=np.float64)
    tetrode = np.asarray(tetrode, dtype=np.int64)
    weights = np.asarray(is_ood, dtype=np.float64)
    frames = []
    for offset in (0.0, bin_s / 2):
        bins = np.floor((times - start_time - offset) / bin_s).astype(np.int64)
        # distinct (bin, tetrode) pairs; floor division recovers the bin
        pairs = np.unique(bins * n_tetrodes + tetrode)
        line_bins, n_distinct = np.unique(pairs // n_tetrodes, return_counts=True)
        position = np.searchsorted(line_bins, bins)
        n_spikes = np.bincount(position, minlength=line_bins.size)
        n_ood = np.bincount(position, weights=weights, minlength=line_bins.size)
        keep = n_distinct >= min_tetrodes
        frames.append(
            pd.DataFrame(
                {
                    "time": start_time + offset + (line_bins[keep] + 0.5) * bin_s,
                    "n_tetrodes": n_distinct[keep].astype(np.int64),
                    "tetrode_fraction": n_distinct[keep] / n_tetrodes,
                    "n_spikes": n_spikes[keep].astype(np.int64),
                    "ood_fraction": n_ood[keep] / np.maximum(n_spikes[keep], 1),
                },
                columns=SPIKE_LINE_COLUMNS,
            )
        )
    lines = pd.concat(frames, ignore_index=True)
    return lines.sort_values("time", kind="stable").reset_index(drop=True)
