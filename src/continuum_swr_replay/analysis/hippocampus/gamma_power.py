"""Slow- and fast-gamma power of CA1, z-scored per epoch, on the decode grid.

Is a non-local event accompanied by a rise in slow gamma (CA3 input) or fast
gamma (entorhinal input; Colgin et al., 2009, *Nature* 462:353)? This gives
each epoch's gamma power on the same footing as the decode frame's other
z-scored signals (``theta_power_zscore``, ``SWR_zscore``):

1. Each kept CA1 tetrode's referenced raw LFP (the electrodes rule R-T keeps
   for the ripple consensus) is band-passed with a zero-phase 4th-order
   Butterworth filter in each band of :data:`GAMMA_BANDS`.
2. Its power is the squared Hilbert envelope; the epoch's power is the mean
   over the kept tetrodes.
3. It is z-scored over the decode span outside excluded time, and read onto the
   decode grid by linear interpolation.

The fast band stops at 100 Hz, below Colgin's upper bound (about 140 Hz), to
keep clear of the ripple band (150-250 Hz) and of the broadband rise spikes
leave in the LFP during SWRs.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, hilbert, sosfiltfilt

GAMMA_BANDS: dict[str, tuple[float, float]] = {
    "slow_gamma": (25.0, 55.0),
    "fast_gamma": (65.0, 100.0),
}
"""Each band's word and its edges in Hz."""

FILTER_ORDER = 4
"""Butterworth order of each band-pass (applied forward and back)."""


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

    NaN for an event that holds no grid sample.
    """
    first = np.searchsorted(grid_time, starts, side="left")
    stop = np.searchsorted(grid_time, ends, side="right")
    cumulative = np.r_[0.0, np.cumsum(values)]
    n = stop - first
    with np.errstate(invalid="ignore", divide="ignore"):
        means = (cumulative[stop] - cumulative[first]) / n
    means = means.astype(np.float64)
    means[n <= 0] = np.nan
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
