"""When an mPFC peri-event response peaks, and when it starts.

The shipped curves are ridge-penalised B-spline kernels on the report window
(``+/-200 ms``, one value per 2 ms lag). Two times are read off each:

- **peak**: the lag of the largest ``|curve|``, the convention of
  :func:`~continuum_swr_replay.analysis.pfc.modulation.peak_log_modulation`;
  its sign says whether the unit is enhanced or suppressed;
- **onset**: the half-maximum onset. From the peak, walk toward earlier lags
  while the curve keeps the peak's sign and at least ``fraction`` of its size;
  the onset is the earliest lag reached. A run that reaches the first lag of
  the window is flagged (``onset_at_edge``): its true onset is at or before
  the window edge. With ``baseline="window_start"`` the threshold is half way
  from the curve's value at the first lag to the peak instead of half way from
  zero (a rise onset), for curves that sit away from zero across the window.

A smooth kernel moves a response's apparent start by an amount that depends
on the basis and the prior; ``scripts/pfc/calibrate_pfc_kernel_timing.py``
measures it on simulated responses of known onset. Read onsets beside it.

The population curve is the mean over units of each curve multiplied by the
sign of its own peak (so a suppression counts as a response of the same
size), with a percentile interval from resampling sessions.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class PeakOnset(NamedTuple):
    """Peak and half-maximum onset of each curve.

    Attributes
    ----------
    peak : NDArray, shape (n_units,)
        Signed value at the largest ``|curve|`` (log-rate modulation).
    peak_lag : NDArray, shape (n_units,)
        Lag of the peak, in seconds.
    onset_lag : NDArray, shape (n_units,)
        Earliest lag of the contiguous run, ending at the peak, over which the
        curve keeps the peak's sign and at least ``fraction`` of its size, in
        seconds.
    onset_at_edge : NDArray of bool, shape (n_units,)
        True where that run reaches the first lag of the window. Only the
        ``"zero"`` baseline can set it: measured from its own first-lag value,
        a curve's first lag belongs to the run only if it is the peak.

    All are NaN (False for ``onset_at_edge``) for a curve that is not finite
    or is identically zero.
    """

    peak: NDArray
    peak_lag: NDArray
    onset_lag: NDArray
    onset_at_edge: NDArray


def peak_and_onset(
    curves: NDArray,
    lag_time: NDArray,
    *,
    fraction: float = 0.5,
    baseline: str = "zero",
) -> PeakOnset:
    """Peak and half-maximum onset of each curve.

    Parameters
    ----------
    curves : NDArray, shape (n_lags,) or (n_lags, n_units)
        Peri-event log-rate modulation, one column per unit, on the window
        whose lags ``lag_time`` gives.
    lag_time : NDArray, shape (n_lags,)
        Lag of each row, in seconds, ascending.
    fraction : float
        Share of the peak's size the curve must keep to belong to the run.
    baseline : {"zero", "window_start"}
        Where the size is measured from: zero (the half-maximum onset), or
        the curve's value at the first lag (the rise onset). With
        ``"window_start"`` a lag belongs to the run when the curve, signed by
        the peak, is at least ``fraction`` of the way from its first-lag value
        to the peak.

    Returns
    -------
    PeakOnset
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if baseline not in ("zero", "window_start"):
        raise ValueError(f"baseline must be 'zero' or 'window_start', got {baseline!r}")
    lag_time = np.asarray(lag_time, dtype=float)
    values = np.asarray(curves, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] != lag_time.shape[0]:
        raise ValueError(
            f"curves have {values.shape[0]} lags and lag_time {lag_time.shape[0]}"
        )
    by_unit = values.T  # (n_units, n_lags)
    n_units, n_lags = by_unit.shape
    filled = np.nan_to_num(by_unit)
    at = np.argmax(np.abs(filled), axis=1)
    peak = filled[np.arange(n_units), at]
    usable = np.isfinite(by_unit).all(axis=1) & (peak != 0)

    if baseline == "zero":
        above = (np.sign(filled) == np.sign(peak)[:, None]) & (
            np.abs(filled) >= fraction * np.abs(peak)[:, None]
        )
    else:
        aligned = filled * np.sign(peak)[:, None]
        start = aligned[:, :1]
        above = aligned >= start + fraction * (np.abs(peak)[:, None] - start)
    lag_index = np.arange(n_lags)
    breaks = ~above & (lag_index[None, :] < at[:, None])
    last_break = np.where(breaks, lag_index[None, :], -1).max(axis=1)
    onset_index = last_break + 1

    nan = np.full(n_units, np.nan)
    return PeakOnset(
        peak=np.where(usable, peak, nan),
        peak_lag=np.where(usable, lag_time[at], nan),
        onset_lag=np.where(usable, lag_time[onset_index], nan),
        onset_at_edge=usable & (onset_index == 0),
    )


def sign_aligned_mean(curves: NDArray, peak: NDArray) -> NDArray:
    """Mean over units of each curve times the sign of its own peak.

    Parameters
    ----------
    curves : NDArray, shape (n_lags, n_units)
    peak : NDArray, shape (n_units,)
        Signed peaks, from :func:`peak_and_onset`.

    Returns
    -------
    NDArray, shape (n_lags,)
        NaN when no unit is given.
    """
    curves = np.asarray(curves, dtype=float)
    signs = np.sign(np.asarray(peak, dtype=float))
    if curves.shape[1] == 0:
        return np.full(curves.shape[0], np.nan)
    return (curves * signs[None, :]).mean(axis=1)


class PopulationTiming(NamedTuple):
    """Peak and onset of the sign-aligned population curve, with intervals.

    Attributes
    ----------
    curve : NDArray, shape (n_lags,)
        The sign-aligned mean curve.
    peak_lag, onset_lag : float
        Its peak lag and half-maximum onset, in seconds.
    peak_lag_interval, onset_lag_interval : tuple of float
        2.5th and 97.5th percentiles over session resamples, in seconds.
    n_units, n_groups : int
        Units and sessions the curve averages.
    """

    curve: NDArray
    peak_lag: float
    onset_lag: float
    peak_lag_interval: tuple[float, float]
    onset_lag_interval: tuple[float, float]
    n_units: int
    n_groups: int


def population_timing(
    curves: NDArray,
    peak: NDArray,
    groups: NDArray,
    lag_time: NDArray,
    *,
    n_boot: int = 1_000,
    seed: int = 0,
    fraction: float = 0.5,
    baseline: str = "zero",
) -> PopulationTiming:
    """Timing of the sign-aligned population curve, resampling sessions.

    Each bootstrap draw resamples the sessions (``groups``) with replacement
    and averages every unit of every drawn session, a session drawn twice
    counting twice; the aligned mean curve of the draw gives one peak lag and
    one onset.

    Parameters
    ----------
    curves : NDArray, shape (n_lags, n_units)
    peak : NDArray, shape (n_units,)
        Signed peaks, from :func:`peak_and_onset`.
    groups : NDArray, shape (n_units,)
        Session label of each unit.
    lag_time : NDArray, shape (n_lags,)
    n_boot : int
        Session resamples.
    seed : int
        Seed of the resampling.
    fraction, baseline
        Onset rule, as in :func:`peak_and_onset`.

    Returns
    -------
    PopulationTiming
    """
    curves = np.asarray(curves, dtype=float)
    aligned = curves * np.sign(np.asarray(peak, dtype=float))[None, :]
    labels, group_index = np.unique(np.asarray(groups), return_inverse=True)
    n_groups = len(labels)
    nan_pair = (float("nan"), float("nan"))
    if n_groups == 0:
        empty = np.full(curves.shape[0], np.nan)
        return PopulationTiming(empty, np.nan, np.nan, nan_pair, nan_pair, 0, 0)

    # Per-session sums make every resample one weighted average.
    sums = np.zeros((n_groups, curves.shape[0]))
    np.add.at(sums, group_index, aligned.T)
    sizes = np.bincount(group_index, minlength=n_groups).astype(float)
    curve = sums.sum(axis=0) / sizes.sum()
    observed = peak_and_onset(curve, lag_time, fraction=fraction, baseline=baseline)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_groups, size=(n_boot, n_groups))
    weights = np.zeros((n_boot, n_groups))
    np.add.at(weights, (np.arange(n_boot)[:, None], draws), 1.0)
    boot_curves = (weights @ sums) / (weights @ sizes)[:, None]
    boot = peak_and_onset(boot_curves.T, lag_time, fraction=fraction, baseline=baseline)

    def interval(values: NDArray) -> tuple[float, float]:
        low, high = np.nanpercentile(values, [2.5, 97.5])
        return float(low), float(high)

    return PopulationTiming(
        curve=curve,
        peak_lag=float(observed.peak_lag[0]),
        onset_lag=float(observed.onset_lag[0]),
        peak_lag_interval=interval(boot.peak_lag),
        onset_lag_interval=interval(boot.onset_lag),
        n_units=int(curves.shape[1]),
        n_groups=n_groups,
    )
