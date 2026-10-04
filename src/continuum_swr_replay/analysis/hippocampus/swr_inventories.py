"""Direct SWR inventories: the Zugaro filter and the channel it reads.

Four ``ripple_detection`` detectors run on one epoch's shared inputs (design
spec ``docs/design/2026-10-03-full-pipeline-refresh-design.md``, Stage 3).
Kay, Karlsson and Yu read the stored 150-250 Hz ripple band. Zugaro, the
``FindRipples`` lineage, reads one raw referenced CA1 channel, chosen and
filtered by this module.

The Zugaro filter
-----------------
The Buzsaki-lab Databank files' band: a Butterworth band-pass of order 3 at
120-180 Hz, designed at the nominal 1000 Hz LFP rate in second-order sections
and run forward and backward with ``sosfiltfilt``. Forward-backward filtering
squares the single-pass gain, so the design edges sit at half amplitude
(-6 dB) and 123.6-175.4 Hz is above -3 dB. It is the method's own filter, so
it is not redesigned.

Gaps and edges
--------------
Each gap-free stretch is filtered on its own, in float64. A row with any
non-finite value, or a timestamp step over 1.5 times the median step, ends a
stretch, as in ``ripple_detection``. The padding (odd extension) and the edge
exclusion both come from the filter's settling time. That is the number of
samples its single-pass impulse response takes to stay below 1e-3 of its
peak, rounded up to a whole 10 ms. It gives 90 samples for the Zugaro design
at 1000 Hz and 170 for the channel-selection design, where SciPy's default
padding would be 21 and 27. The first and last that many rows of each stretch
are NaN (unobserved), as is every row of a stretch too short for both.

Channel selection
-----------------
Follows buzcode's ``bz_GetBestRippleChan`` (commit ``0969ddf``):

1. band-pass 140-180 Hz with a Butterworth filter of order 4, zero-phase,
   filtered per stretch as above;
2. take a centered moving RMS over 12 ms, the original's 15 samples at
   1250 Hz, rounded up to an odd 13 samples at 1000 Hz;
3. score each channel as mean RMS / median RMS over its observed samples, so
   sparse large bursts over a low baseline score high;
4. leave out channels whose mean or median RMS is below 1, in the stored LFP
   units (buzcode's dead-channel guard);
5. the highest score wins, and an exact tie goes to the lowest electrode id.
   If every channel is left out, there is no channel, and that is an error.

The original's own comment calls it a placeholder for a real signal-to-noise
measure; it is used because it is the lineage's automated rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import convolve1d
from scipy.signal import butter, sosfilt, sosfiltfilt

ZUGARO_BAND_HZ: tuple[float, float] = (120.0, 180.0)
"""Hz: the Databank files' ``FindRipples`` band (single-pass -3 dB edges)."""

ZUGARO_FILTER_ORDER = 3
"""Butterworth order of the Zugaro band-pass."""

CHANNEL_SELECTION_BAND_HZ: tuple[float, float] = (140.0, 180.0)
"""Hz: ``bz_GetBestRippleChan``'s band."""

CHANNEL_SELECTION_FILTER_ORDER = 4
"""Butterworth order of the channel-selection band-pass."""

CHANNEL_SELECTION_RMS_WINDOW_S = 15 / 1250
"""Seconds (0.012): ``bz_GetBestRippleChan``'s 15-sample RMS window at 1250 Hz."""

DEAD_CHANNEL_RMS = 1.0
"""``bz_GetBestRippleChan``'s dead-channel guard, in the stored LFP units."""

SETTLING_TOLERANCE = 1e-3
"""A filter has settled once its impulse response stays below this fraction of
its peak."""

ZUGARO_MAXIMUM_DURATION_S = 0.150
"""Seconds: the Databank files' maximum ripple duration. Longer Zugaro events
are dropped after merging; an event of exactly 0.150 s is kept."""

_GAP_STEP_RATIO = 1.5
"""A timestamp step over this multiple of the median step is a gap
(``ripple_detection``'s rule)."""

_SETTLING_ROUNDING_S = 0.010
"""Settling lengths are rounded up to a whole multiple of this, in seconds."""

_IMPULSE_RESPONSE_S = 10.0
"""Seconds of single-pass impulse response computed to find a settling length."""


class NoUsableChannelError(ValueError):
    """No candidate channel is usable, so Zugaro has no channel for the epoch."""


@dataclass(frozen=True)
class ChannelChoice:
    """The channel ``best_ripple_channel`` chose, with every candidate's score.

    Attributes
    ----------
    electrode_id : int
        The chosen channel's electrode.
    scores : dict[int, float]
        Every candidate's mean RMS / median RMS, keyed by electrode id. NaN
        where the median RMS is zero (such a channel is always excluded).
    excluded : dict[int, str]
        The candidates the dead-channel guard left out, keyed by electrode id,
        each with its reason.
    """

    electrode_id: int
    scores: dict[int, float]
    excluded: dict[int, str]


def design_bandpass_sos(
    sampling_frequency: float, band_hz: tuple[float, float], order: int
) -> NDArray[np.float64]:
    """Design a Butterworth band-pass as second-order sections.

    Parameters
    ----------
    sampling_frequency : float
        Design rate in Hz, the nominal LFP rate.
    band_hz : tuple[float, float]
        Lower and upper band edges in Hz, where the single-pass gain is -3 dB
        and the zero-phase gain is -6 dB.
    order : int
        Butterworth order. The band-pass has ``2 * order`` poles.

    Returns
    -------
    sos : NDArray[np.float64], shape (order, 6)
        ``butter(order, band_hz, btype="bandpass", fs=sampling_frequency,
        output="sos")``.
    """
    sos: NDArray[np.float64] = butter(
        order, band_hz, btype="bandpass", fs=sampling_frequency, output="sos"
    )
    return sos


def settling_samples(
    sos: NDArray, sampling_frequency: float, tolerance: float = SETTLING_TOLERANCE
) -> int:
    """Count the samples a filter takes to settle, rounded up to a whole 10 ms.

    The single-pass impulse response is computed over 10 s. The settling
    length is the first sample from which its magnitude stays below
    ``tolerance`` times its peak, rounded up to a whole 10 ms at
    ``sampling_frequency``. ``zero_phase_bandpass`` pads each stretch by this
    many samples and leaves out this many rows at each end.

    At 1000 Hz, the Zugaro design last reaches the tolerance at sample 86
    (86 ms), giving 90. The channel-selection design last reaches it at sample
    166, giving 170.

    Parameters
    ----------
    sos : NDArray, shape (n_sections, 6)
        A stable filter in second-order sections.
    sampling_frequency : float
        Rate in Hz that ``sos`` was designed at.
    tolerance : float, optional
        Fraction of the response's peak magnitude. Default ``SETTLING_TOLERANCE``.

    Returns
    -------
    n_samples : int
        The settling length in samples.

    Raises
    ------
    ValueError
        If the response is still at or above the tolerance after the first 5
        of the 10 s computed. The tail then cannot be shown to stay below it.
    """
    n_response = math.ceil(_IMPULSE_RESPONSE_S * sampling_frequency)
    impulse = np.zeros(n_response)
    impulse[0] = 1.0
    magnitude = np.abs(sosfilt(sos, impulse))
    n_settle = int(np.flatnonzero(magnitude >= tolerance * magnitude.max())[-1]) + 1
    if n_settle > n_response // 2:
        msg = (
            f"The impulse response is still at or above {tolerance:g} of its peak "
            f"{n_settle / sampling_frequency:.3g} s in, past half the "
            f"{_IMPULSE_RESPONSE_S:g} s computed, so its settling time is not known."
        )
        raise ValueError(msg)
    samples_per_rounding = _SETTLING_ROUNDING_S * sampling_frequency
    return math.ceil(math.ceil(n_settle / samples_per_rounding) * samples_per_rounding)


def odd_window_samples(window_s: float, sampling_frequency: float) -> int:
    """Convert a window to whole samples, rounded up to an odd count.

    ``round(window_s * sampling_frequency)``, plus one if that is even. This
    is how ``ripple_detection`` sizes Zugaro's moving average: 11/1250 s is 9
    samples at 1000 Hz. An odd count lets a window be centered on its sample.

    Parameters
    ----------
    window_s : float
        Window length in seconds.
    sampling_frequency : float
        Rate in Hz.

    Returns
    -------
    n_samples : int
        The odd window length, 13 for the 12 ms RMS window at 1000 Hz.
    """
    n_samples = round(window_s * sampling_frequency)
    return n_samples + 1 if n_samples % 2 == 0 else n_samples


def gap_free_stretches(time: NDArray, values: NDArray) -> NDArray[np.int64]:
    """Find the stretches of consecutive, finite rows with no timestamp gap.

    A row with any non-finite value, or a timestamp step over 1.5 times the
    median step, ends a stretch. This is the rule ``ripple_detection`` splits
    its filters and detectors at. The median is taken over every step of
    ``time``, so an overstated sampling rate cannot make every sample a
    stretch of its own.

    Parameters
    ----------
    time : NDArray, shape (n_time,)
        Nondecreasing timestamps in seconds.
    values : NDArray, shape (n_time, n_channels)
        Samples. A row with any NaN or infinity belongs to no stretch.

    Returns
    -------
    stretches : NDArray[np.int64], shape (n_stretches, 2)
        ``[start, stop)`` row indices of each stretch, in time order.

    Raises
    ------
    ValueError
        If ``time`` does not have one timestamp per row of ``values``, holds
        a non-finite or decreasing timestamp, or has a median step of zero.
        The package refuses the last two as well.
    """
    time = np.asarray(time, dtype=float)
    values = np.asarray(values)
    n_time = values.shape[0]
    if time.shape != (n_time,):
        msg = (
            f"time must have shape ({n_time},), one timestamp per row of values, "
            f"got {time.shape}."
        )
        raise ValueError(msg)
    steps = np.diff(time)
    if not np.all(np.isfinite(time)) or np.any(steps < 0):
        msg = "time must contain finite, nondecreasing timestamps."
        raise ValueError(msg)

    is_valid = np.all(np.isfinite(values), axis=tuple(range(1, values.ndim)))
    boundary = np.zeros(n_time + 1, dtype=bool)
    boundary[[0, -1]] = True
    boundary[1:-1] = is_valid[1:] != is_valid[:-1]
    if n_time > 1:
        median_step = np.median(steps)
        if median_step <= 0:
            msg = (
                "time must have a positive median step, so timestamps must not repeat."
            )
            raise ValueError(msg)
        boundary[1:-1] |= steps > _GAP_STEP_RATIO * median_step
    edges = np.flatnonzero(boundary)
    starts, stops = edges[:-1], edges[1:]
    keep = is_valid[starts]
    return np.column_stack([starts[keep], stops[keep]])


def zero_phase_bandpass(
    values: NDArray,
    time: NDArray,
    sampling_frequency: float,
    band_hz: tuple[float, float],
    order: int,
) -> NDArray[np.float64]:
    """Band-pass each gap-free stretch forward and backward, in float64.

    The Butterworth design from ``design_bandpass_sos`` is run with
    ``sosfiltfilt(padtype="odd", padlen=n_settle)`` over each stretch from
    ``gap_free_stretches`` on its own, so it never filters across a gap.
    ``n_settle`` is the design's ``settling_samples``. The same count is then
    left out at each end of a stretch, so no kept row carries a padding
    transient.

    Parameters
    ----------
    values : NDArray, shape (n_time, n_channels)
        Signals, in any real dtype. Cast to float64 before filtering.
    time : NDArray, shape (n_time,)
        Nondecreasing timestamps in seconds.
    sampling_frequency : float
        Design rate in Hz, the nominal LFP rate.
    band_hz : tuple[float, float]
        Band edges in Hz, at -6 dB zero-phase gain.
    order : int
        Butterworth order.

    Returns
    -------
    filtered : NDArray[np.float64], shape (n_time, n_channels)
        The zero-phase band-passed signals. NaN marks unobserved rows: rows
        outside every stretch, the first and last ``n_settle`` rows of each
        stretch, and every row of a stretch of at most ``2 * n_settle`` rows.
    """
    values = np.asarray(values, dtype=np.float64)
    sos = design_bandpass_sos(sampling_frequency, band_hz, order)
    n_settle = settling_samples(sos, sampling_frequency)
    filtered = np.full(values.shape, np.nan)
    for start, stop in gap_free_stretches(time, values):
        n_rows = stop - start
        if n_rows <= 2 * n_settle:
            continue
        stretch = sosfiltfilt(
            sos, values[start:stop], axis=0, padtype="odd", padlen=n_settle
        )
        filtered[start + n_settle : stop - n_settle] = stretch[
            n_settle : n_rows - n_settle
        ]
    return filtered


def _centered_moving_rms(
    filtered: NDArray, time: NDArray, window: int
) -> NDArray[np.float64]:
    """Compute a centered moving RMS within each run of observed rows.

    Parameters
    ----------
    filtered : NDArray, shape (n_time, n_channels)
        Band-passed signals, NaN where unobserved.
    time : NDArray, shape (n_time,)
        Nondecreasing timestamps in seconds.
    window : int
        Odd window length in samples.

    Returns
    -------
    rms : NDArray[np.float64], shape (n_time, n_channels)
        The root of the mean square over the ``window`` rows centered on each
        row. NaN where ``filtered`` is unobserved. Near the ends of a run of
        observed rows the window is truncated to the run's own rows, so it
        never reaches across a gap or an edge.
    """
    rms = np.full(filtered.shape, np.nan)
    kernel = np.ones(window)
    for start, stop in gap_free_stretches(time, filtered):
        power = convolve1d(filtered[start:stop] ** 2, kernel, axis=0, mode="constant")
        n_summed = convolve1d(np.ones(stop - start), kernel, mode="constant")
        rms[start:stop] = np.sqrt(power / n_summed[:, np.newaxis])
    return rms


def best_ripple_channel(
    values: NDArray,
    time: NDArray,
    sampling_frequency: float,
    electrode_ids: NDArray,
) -> ChannelChoice:
    """Choose Zugaro's channel as buzcode's ``bz_GetBestRippleChan`` does.

    Each candidate is band-passed by ``zero_phase_bandpass`` at
    ``CHANNEL_SELECTION_BAND_HZ`` with order ``CHANNEL_SELECTION_FILTER_ORDER``.
    Its moving RMS is then taken over ``odd_window_samples(
    CHANNEL_SELECTION_RMS_WINDOW_S, sampling_frequency)`` samples (13 at
    1000 Hz), centered, within each stretch. The score is its mean RMS over
    its median RMS, across the observed rows. A candidate whose mean or median
    RMS is below ``DEAD_CHANNEL_RMS`` is excluded. The highest score among the
    rest wins, and an exact tie goes to the lowest electrode id.

    Parameters
    ----------
    values : NDArray, shape (n_time, n_channels)
        The candidates' referenced raw LFP, in the stored units the
        dead-channel guard applies in (int16 counts for Spyglass LFP).
    time : NDArray, shape (n_time,)
        Nondecreasing timestamps in seconds.
    sampling_frequency : float
        Nominal rate in Hz, used for the filter design and the window.
    electrode_ids : NDArray, shape (n_channels,)
        Each column's electrode id. Must be unique.

    Returns
    -------
    ChannelChoice
        The winner, every candidate's score, and the excluded candidates with
        their reasons.

    Raises
    ------
    ValueError
        If ``values`` is not 2-D or ``electrode_ids`` is not one unique id per
        column.
    NoUsableChannelError
        If every candidate is excluded, or no row is observed after edge
        exclusion.
    """
    values = np.asarray(values)
    electrode_ids = np.asarray(electrode_ids)
    if values.ndim != 2 or electrode_ids.shape != (values.shape[1],):
        msg = (
            "values must be (n_time, n_channels) with one electrode id per column; "
            f"got values {values.shape} and electrode_ids {electrode_ids.shape}."
        )
        raise ValueError(msg)
    if np.unique(electrode_ids).size != electrode_ids.size:
        msg = f"electrode_ids must be unique, got {electrode_ids.tolist()}."
        raise ValueError(msg)

    filtered = zero_phase_bandpass(
        values,
        time,
        sampling_frequency,
        CHANNEL_SELECTION_BAND_HZ,
        CHANNEL_SELECTION_FILTER_ORDER,
    )
    window = odd_window_samples(CHANNEL_SELECTION_RMS_WINDOW_S, sampling_frequency)
    rms = _centered_moving_rms(filtered, time, window)
    observed_rms = rms[np.all(np.isfinite(rms), axis=1)]
    if len(observed_rms) == 0:
        msg = (
            "No LFP row is observed after edge exclusion: no gap-free stretch is "
            "longer than twice the channel-selection filter's settling length."
        )
        raise NoUsableChannelError(msg)

    mean_rms = observed_rms.mean(axis=0)
    median_rms = np.median(observed_rms, axis=0)
    ratio = np.divide(
        mean_rms, median_rms, out=np.full_like(mean_rms, np.nan), where=median_rms > 0
    )
    scores = {int(e): float(r) for e, r in zip(electrode_ids, ratio, strict=True)}
    excluded: dict[int, str] = {}
    for electrode_id, mean, median in zip(
        electrode_ids, mean_rms, median_rms, strict=True
    ):
        reasons = [
            f"{name} RMS {level:.4g} < {DEAD_CHANNEL_RMS:g}"
            for name, level in (("mean", mean), ("median", median))
            if level < DEAD_CHANNEL_RMS
        ]
        if reasons:
            excluded[int(electrode_id)] = "; ".join(reasons)

    usable = [electrode_id for electrode_id in scores if electrode_id not in excluded]
    if not usable:
        msg = (
            f"Every candidate channel is below the dead-channel guard "
            f"({DEAD_CHANNEL_RMS:g}, in the stored LFP units): {excluded}"
        )
        raise NoUsableChannelError(msg)
    best_score = max(scores[electrode_id] for electrode_id in usable)
    winner = min(
        electrode_id for electrode_id in usable if scores[electrode_id] == best_score
    )
    return ChannelChoice(electrode_id=winner, scores=scores, excluded=excluded)
