"""Direct SWR inventories: four detectors on shared inputs, and their checks.

Four ``ripple_detection`` detectors run on one epoch's shared inputs (design
spec ``docs/design/2026-10-03-full-pipeline-refresh-design.md``, Stage 3).
Kay, Karlsson and Yu read the stored 150-250 Hz ripple band. Zugaro, the
``FindRipples`` lineage, reads one raw referenced CA1 channel, chosen and
filtered by this module. The module also runs the four with their pinned
settings, recomputes Yu's threshold diagnostics, checks that the stored band
reproduces from its raw input, and compares direct Kay events with the
canonical Spyglass ones.

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

The detectors
-------------
``run_swr_detectors`` runs Kay, Karlsson, Yu and Zugaro, each with its own
``DETECTOR_SETTINGS`` and every one with ``speed_threshold=4.0``: an event is
kept only if speed at its first and last sample is at most 4 cm/s, and an
unknown speed fails. What they share is the input: the electrodes, time, speed
and the nominal sampling rate. Zugaro's events longer than 150 ms are then
dropped. A detector that raises is reported as its exception, so one failure
does not hide the others' results. ``yu_diagnostics`` recomputes Yu's
threshold and the histogram behind it, so both are known for an epoch with no
events too.

Checks
------
``band_reproduction_error`` re-filters the raw LFP as ``LFPBandV1.make`` did
and compares the result with the stored band. Agreement shows that Zugaro's
candidates are the channels the other three read, with the same referencing.
``kay_version_check`` marks each event of a direct Kay inventory and of the
canonical Spyglass one matched, added or missed, by strict positive-duration
overlap.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from ripple_detection import (
    Karlsson_ripple_detector,
    Kay_ripple_detector,
    Yu_ripple_detector,
    Zugaro_ripple_detector,
    get_Yu_ripple_consensus_trace,
    minimum_sample_count,
    noise_threshold_diagnostics,
)
from scipy.ndimage import convolve1d
from scipy.signal import butter, sosfilt, sosfiltfilt

from continuum_swr_replay.analysis.inventory_comparison.overlap import overlap_pairs
from continuum_swr_replay.analysis.inventory_comparison.schema import (
    SessionIdentity,
    inventory_from_intervals,
)
from continuum_swr_replay.types import RippleBandInputs

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


DETECTOR_SETTINGS: Mapping[str, Mapping[str, object]] = MappingProxyType(
    {
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
)
"""Each detector's keyword settings, from the design spec's Stage 3 table.

- Kay: Spyglass's canonical settings, equal to the package defaults.
- Karlsson: the package defaults.
- Yu: spec 01, step 4.
- Zugaro: the Buzsaki-lab Databank files' ``FindRipples`` settings, merging
  with no length cap and with no minimum duration. ``cap_zugaro_duration``
  then applies their 150 ms maximum.

A setting left out takes the package default: Kay, Karlsson and Zugaro
z-score over all valid samples, Yu normalizes its consensus over the immobile
ones, and Zugaro's moving average is 11/1250 s (9 samples at 1000 Hz).
"""

_DIRECT_KAY = "kay"
"""``kay_version_check``'s name for the direct Kay inventory."""

_CANONICAL_KAY = "kay_spyglass"
"""``kay_version_check``'s name for the canonical Spyglass Kay inventory."""


def cap_zugaro_duration(events: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop Zugaro events longer than ``ZUGARO_MAXIMUM_DURATION_S``.

    The Databank files' 150 ms maximum, applied after merging: the buzcode
    lineage merges with no length cap, then drops events over the maximum.
    An event of exactly 0.150 s is kept.

    Parameters
    ----------
    events : pd.DataFrame
        ``Zugaro_ripple_detector``'s table. Its ``duration`` column is the
        time from the first to the last sample, in seconds.

    Returns
    -------
    kept : pd.DataFrame
        The rows with ``duration <= ZUGARO_MAXIMUM_DURATION_S``, unchanged,
        in their order and with their index.
    n_dropped : int
        The number of rows dropped.
    """
    keep = events["duration"].to_numpy() <= ZUGARO_MAXIMUM_DURATION_S
    return events.loc[keep], int(np.count_nonzero(~keep))


def run_swr_detectors(
    time: NDArray,
    ripple_band: NDArray,
    zugaro_lfp: NDArray,
    speed: NDArray,
    sampling_frequency: float,
) -> dict[str, pd.DataFrame | Exception]:
    """Run Kay, Karlsson, Yu and Zugaro on one epoch's shared inputs.

    Each runs with its own ``DETECTOR_SETTINGS``. Kay, Karlsson and Yu read
    the stored ripple band; Zugaro reads its one filtered channel. A detector
    that raises is returned as its exception, so the others still report.

    Parameters
    ----------
    time : NDArray, shape (n_time,)
        Timestamps in seconds, shared by every input.
    ripple_band : NDArray, shape (n_time, n_channels)
        The stored 150-250 Hz band of the CA1 electrodes.
    zugaro_lfp : NDArray, shape (n_time, 1)
        Zugaro's channel, already filtered by ``zero_phase_bandpass`` at
        ``ZUGARO_BAND_HZ``. NaN rows are unobserved, and Zugaro leaves them
        out of its z-score and events.
    speed : NDArray, shape (n_time,)
        Head speed in cm/s. NaN is unknown, and fails the speed rule at an
        event's first or last sample.
    sampling_frequency : float
        The nominal rate in Hz, passed to every detector.

    Returns
    -------
    results : dict[str, pd.DataFrame | Exception]
        Keyed ``"kay"``, ``"karlsson"``, ``"yu"`` and ``"zugaro"``: the
        detector's own event table, or the exception it raised. Zugaro's
        table is capped by ``cap_zugaro_duration``, and its
        ``attrs["n_dropped_over_maximum_duration"]`` holds the number of
        events the cap dropped.
    """
    # Looked up when called, so a test can replace a detector in this module.
    detectors: dict[str, tuple[Callable[..., pd.DataFrame], NDArray]] = {
        "kay": (Kay_ripple_detector, ripple_band),
        "karlsson": (Karlsson_ripple_detector, ripple_band),
        "yu": (Yu_ripple_detector, ripple_band),
        "zugaro": (Zugaro_ripple_detector, zugaro_lfp),
    }
    results: dict[str, pd.DataFrame | Exception] = {}
    for name, (detector, signal) in detectors.items():
        try:
            events = detector(
                time, signal, speed, sampling_frequency, **DETECTOR_SETTINGS[name]
            )
        except Exception as error:  # one detector's failure must not stop the rest
            results[name] = error
            continue
        if name == "zugaro":
            events, n_dropped = cap_zugaro_duration(events)
            events.attrs["n_dropped_over_maximum_duration"] = n_dropped
        results[name] = events
    return results


def yu_diagnostics(
    time: NDArray, ripple_band: NDArray, speed: NDArray, sampling_frequency: float
) -> dict:
    """Recompute Yu's threshold as ``Yu_ripple_detector`` does, with its diagnostics.

    With ``DETECTOR_SETTINGS["yu"]``, the consensus is
    ``get_Yu_ripple_consensus_trace`` over the detector's blocks: finite rows
    split at timestamp gaps, where a block shorter than the minimum duration
    is treated as missing. ``noise_threshold_diagnostics`` then runs on the
    consensus over the detector's noise sample, the valid samples whose speed
    is at or below the speed threshold (an unknown speed is not), with the
    percentile. The threshold is converted to the detector's units with the
    mean and standard deviation (ddof=0) of that same sample. Nothing depends
    on the events, so an epoch with none still has its diagnostics.

    Parameters
    ----------
    time : NDArray, shape (n_time,)
        Timestamps in seconds.
    ripple_band : NDArray, shape (n_time, n_channels)
        The ripple band Yu reads. NaN rows are missing.
    speed : NDArray, shape (n_time,)
        Head speed in cm/s; NaN is unknown.
    sampling_frequency : float
        The nominal rate in Hz.

    Returns
    -------
    diagnostics : dict
        JSON-ready, in Python scalars and lists. Every field of
        ``NoiseThresholdDiagnostics``: ``threshold`` (in the consensus
        trace's units, the median of per-channel z-scored envelopes),
        ``mode``, ``mode_index``, ``mean``, ``min``, ``flank_ratio`` (may be
        infinite), ``histogram_edges``, ``counts``, ``smoothed_counts``,
        ``mirrored_positions``, ``mirrored_counts``, ``out_of_grid_fraction``,
        ``n_values`` and ``n_in_grid``. Then ``normalization_center`` and
        ``normalization_scale``, and ``threshold_zscore``, ``(threshold -
        center) / scale``: the events' ``detection_threshold_zscore``.

    Raises
    ------
    ValueError
        Where the detector cannot estimate its threshold either: no valid
        sample at or below the speed threshold, or a noise histogram that
        does not resolve the percentile.
    """
    settings: Mapping[str, Any] = DETECTOR_SETTINGS["yu"]
    time = np.asarray(time, dtype=float)
    band = np.asarray(ripple_band, dtype=float)
    speed = np.asarray(speed, dtype=float)
    n_minimum = minimum_sample_count(time, settings["minimum_duration"])
    short = [
        (start, stop)
        for start, stop in gap_free_stretches(time, band)
        if stop - start < n_minimum
    ]
    if short:
        band = band.copy()
        for start, stop in short:
            band[start:stop] = np.nan
    consensus = get_Yu_ripple_consensus_trace(
        band,
        sampling_frequency,
        settings["smoothing_sigma"],
        settings["zscore_per_channel"],
        time=time,
    )
    is_noise = np.all(np.isfinite(band), axis=1) & (
        speed <= settings["speed_threshold"]
    )
    noise = consensus[is_noise]
    diagnostics = noise_threshold_diagnostics(noise, percentile=settings["percentile"])
    center = float(np.mean(noise))
    scale = float(np.std(noise))
    return {
        field.name: np.asarray(getattr(diagnostics, field.name)).tolist()
        for field in fields(diagnostics)
    } | {
        "normalization_center": center,
        "normalization_scale": scale,
        "threshold_zscore": (diagnostics.threshold - center) / scale,
    }


def band_reproduction_error(
    inputs: RippleBandInputs, fir_filter: Callable | None = None
) -> tuple[float, float]:
    """Re-filter the raw LFP as ``LFPBandV1.make`` did; compare with the stored band.

    ``fir_filter`` is called as make calls ``FirFilterParameters().filter_data``
    (spyglass ``lfp/analysis/v1/lfp_band.py``), with make's six arguments in
    make's order: ``raw_timestamps``; ``raw_referenced_native``, every LFP
    column with the band electrodes referenced in the stored dtype;
    ``band_filter_coeff``; ``band_valid_interval.times``;
    ``lfp_band_elect_index``, the band electrodes' columns; and
    ``decimation``. Agreement shows that the referenced columns Zugaro
    chooses from are the channels the band was made from. A
    column-to-electrode map that took positions for ids would filter other
    channels.

    Parameters
    ----------
    inputs : RippleBandInputs
        From ``get_ripple_band_inputs``.
    fir_filter : Callable, optional
        Takes those six arguments and returns ``(filtered, timestamps)``, as
        ``filter_data`` does. Default None: Spyglass's own
        ``FirFilterParameters().filter_data``, imported only then, so that
        importing this module needs no database.

    Returns
    -------
    max_abs_difference : float
        The largest ``|re-filtered - stored|`` over every channel and every
        timestamp the two share, in the stored band's units. A timestamp held
        by only one of the two is not compared.
    tolerance : float
        1.0 for an integer band dtype, else 1e-6 times the stored band's
        standard deviation.
    """
    if fir_filter is None:
        from spyglass.common import FirFilterParameters

        fir_filter = FirFilterParameters().filter_data
    valid_interval: Any = inputs["band_valid_interval"]  # a Spyglass Interval
    refiltered, refiltered_time = fir_filter(
        inputs["raw_timestamps"],
        inputs["raw_referenced_native"],
        inputs["band_filter_coeff"],
        valid_interval.times,
        inputs["lfp_band_elect_index"],
        inputs["decimation"],
    )
    refiltered = np.asarray(refiltered, dtype=np.float64)
    stored = np.asarray(inputs["ripple_band"], dtype=np.float64)
    _, refiltered_rows, stored_rows = np.intersect1d(
        np.asarray(refiltered_time), inputs["time"], return_indices=True
    )
    difference = refiltered[refiltered_rows] - stored[stored_rows]
    if np.issubdtype(np.dtype(inputs["ripple_band_dtype"]), np.integer):
        tolerance = 1.0
    else:
        tolerance = 1e-6 * float(np.std(stored))
    return float(np.max(np.abs(difference))), tolerance


def kay_version_check(
    direct: pd.DataFrame,
    canonical: pd.DataFrame,
    *,
    session: SessionIdentity,
    run_id: str,
    speed_time_s: NDArray,
    speed_cm_per_s: NDArray,
) -> pd.DataFrame:
    """Mark each event of two Kay inventories matched, added or missed.

    ``direct`` is Kay at the pinned ``ripple_detection`` commit (detector
    ``"kay"``) and ``canonical`` the canonical Spyglass events
    (``"kay_spyglass"``). Two events match when they share time of positive
    duration, ``overlap_pairs``' strict rule; touching at an instant is no
    match. A direct event that matches nothing is ``added``, a canonical one
    ``missed``. A matched event's partner is the event of the other set it
    overlaps longest, the earlier one on a tie, so a split event can leave
    several matched events sharing one partner. Matching uses every event of
    both sets, eligible or not; ``whole_event_speed_eligible`` lets the report
    split them.

    Parameters
    ----------
    direct, canonical : pd.DataFrame
        Event tables with ``start_time`` and ``end_time`` in seconds, indexed
        by the detector's own event ids (the package's ``event_number``).
    session : SessionIdentity
        The epoch, named in the event keys.
    run_id : str
        The run, recorded on the inventories the keys come from.
    speed_time_s, speed_cm_per_s : NDArray, shape (n_samples,)
        The speed trace eligibility is read from, with strictly increasing
        times.

    Returns
    -------
    check : pd.DataFrame
        One row per event, the direct events first, each set in start order:

        - ``detector``: ``"kay"`` or ``"kay_spyglass"``;
        - ``event_key``: ``"<detector>:<animal><day>_<epoch>:<event id>"``,
          as ``inventory_from_intervals`` makes it;
        - ``status``: ``"matched"``, ``"added"`` or ``"missed"``;
        - ``partner_event_key``: the partner's key; NaN unless matched;
        - ``start_shift_s``, ``end_shift_s``: the direct event's bound minus
          the canonical one's, over the event and its partner, so both rows
          of a mutual pair agree; NaN unless matched;
        - ``whole_event_speed_eligible``: every speed sample from the event's
          start to its end, of which there must be one, finite and below
          4 cm/s, the canonical rule (``event_speeds_are_immobile``).
    """
    events = pd.concat(
        [
            inventory_from_intervals(
                table["start_time"].to_numpy(),
                table["end_time"].to_numpy(),
                native_keys=table.index.to_numpy(),
                detector=detector,
                session=session,
                run_id=run_id,
                speed_time_s=speed_time_s,
                speed_cm_per_s=speed_cm_per_s,
            )
            for detector, table in ((_DIRECT_KAY, direct), (_CANONICAL_KAY, canonical))
        ],
        ignore_index=True,
    )
    pairs = overlap_pairs(events)
    # every overlap, seen from each of its two events
    links = pd.concat(
        [
            pd.DataFrame(
                {
                    "event_key": pairs[f"event_key_{own}"],
                    "partner_event_key": pairs[f"event_key_{other}"],
                    "intersection_s": pairs["intersection_s"],
                }
            )
            for own, other in (("a", "b"), ("b", "a"))
        ],
        ignore_index=True,
    )
    longest = links.loc[links.groupby("event_key")["intersection_s"].idxmax()]
    partners = longest.set_index("event_key")["partner_event_key"]

    check = events[["detector", "event_key", "whole_event_speed_eligible"]].copy()
    check["partner_event_key"] = check["event_key"].map(partners)
    is_direct = (check["detector"] == _DIRECT_KAY).to_numpy()
    check["status"] = np.where(
        check["partner_event_key"].notna(),
        "matched",
        np.where(is_direct, "added", "missed"),
    )
    bounds = events.set_index("event_key")[["event_start_s", "event_end_s"]]
    own_bounds = bounds.loc[check["event_key"]].to_numpy()
    partner_bounds = bounds.reindex(check["partner_event_key"]).to_numpy()
    shift = np.where(is_direct, 1.0, -1.0)[:, np.newaxis] * (
        own_bounds - partner_bounds
    )
    check["start_shift_s"] = shift[:, 0]
    check["end_shift_s"] = shift[:, 1]
    return check[
        [
            "detector",
            "event_key",
            "status",
            "partner_event_key",
            "start_shift_s",
            "end_shift_s",
            "whole_event_speed_eligible",
        ]
    ]
