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
size), with a percentile interval from resampling the units' groups (their
sessions, or their recording days).

The timing statistics of the units called for an event class
(:func:`called_response_timing`, :func:`called_response_timing_by_animal`,
:func:`population_rise_timing`, :func:`paired_peak_lag_statistics`,
:func:`peak_before_onset_fraction`) are the ones the mpfc summary reports; the
lags they return are in ms, where the functions above them return seconds.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.pfc.modulation import REPORT_HALF_WINDOW_S


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
    baseline: str | tuple[float, float] = "zero",
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
    baseline : {"zero", "window_start"} or (float, float)
        Where the size is measured from: zero (the half-maximum onset), the
        curve's value at the first lag (the rise onset), or the curve's mean
        over the lags ``[start, stop]`` in seconds (for a window wide enough
        to hold a pre-event reference). Other than ``"zero"``, a lag belongs to
        the run when the curve, signed by the peak, is at least ``fraction``
        of the way from the reference level to the peak.

    Returns
    -------
    PeakOnset
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    lag_time = np.asarray(lag_time, dtype=float)
    reference_lags = _reference_lags(baseline, lag_time)
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

    if reference_lags is None:
        above = (np.sign(filled) == np.sign(peak)[:, None]) & (
            np.abs(filled) >= fraction * np.abs(peak)[:, None]
        )
    else:
        aligned = filled * np.sign(peak)[:, None]
        start = aligned[:, reference_lags].mean(axis=1, keepdims=True)
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


def _reference_lags(
    baseline: str | tuple[float, float], lag_time: NDArray
) -> NDArray | None:
    """Rows the reference level is averaged over; None for a zero reference."""
    if isinstance(baseline, str):
        if baseline == "zero":
            return None
        if baseline == "window_start":
            return np.array([0])
        raise ValueError(
            f"baseline must be 'zero', 'window_start' or (start, stop), got {baseline!r}"
        )
    start, stop = baseline
    rows = np.flatnonzero((lag_time >= start - 1e-12) & (lag_time <= stop + 1e-12))
    if rows.size == 0:
        raise ValueError(f"no lag in the reference window {baseline}")
    return rows


def response_size(
    curves: NDArray,
    lag_time: NDArray,
    baseline: str | tuple[float, float] = "zero",
) -> NDArray:
    """Size of each curve's peak above (or below) a reference level.

    Parameters
    ----------
    curves : NDArray, shape (n_lags,) or (n_lags, n_units)
    lag_time : NDArray, shape (n_lags,)
    baseline : {"zero", "window_start"} or (float, float)
        Reference level, as in :func:`peak_and_onset`.

    Returns
    -------
    NDArray, shape (n_units,)
        Signed peak minus the reference level (negative for a decrease that
        goes below its reference), on the log-rate scale; NaN for a curve
        with no peak.
    """
    lag_time = np.asarray(lag_time, dtype=float)
    values = np.asarray(curves, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    timing = peak_and_onset(values, lag_time)
    rows = _reference_lags(baseline, lag_time)
    reference = 0.0 if rows is None else values[rows].mean(axis=0)
    return timing.peak - reference


def steepest_approach_lag(curves: NDArray, lag_time: NDArray) -> NDArray:
    """Lag of the steepest change toward each curve's peak, before the peak.

    The lag between two samples at which the curve moves fastest in the
    direction of its own peak (up for an increase, down for a decrease), over
    the lags before the peak. It does not depend on the curve's level, so a
    constant offset across the window leaves it unchanged.

    Parameters
    ----------
    curves : NDArray, shape (n_lags,) or (n_lags, n_units)
    lag_time : NDArray, shape (n_lags,)

    Returns
    -------
    NDArray, shape (n_units,)
        Seconds (midpoint of the steepest step); NaN for a curve with no peak
        or a peak at the first lag.
    """
    lag_time = np.asarray(lag_time, dtype=float)
    values = np.asarray(curves, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    timing = peak_and_onset(values, lag_time)
    at = np.searchsorted(lag_time, np.nan_to_num(timing.peak_lag, nan=lag_time[0]))
    step = (
        np.diff(np.nan_to_num(values.T), axis=1)
        * np.sign(np.nan_to_num(timing.peak))[:, None]
    )
    before = np.arange(step.shape[1])[None, :] < at[:, None]
    step = np.where(before, step, -np.inf)
    best = np.argmax(step, axis=1)
    midpoints = 0.5 * (lag_time[1:] + lag_time[:-1])
    ok = np.isfinite(timing.peak) & (at > 0)
    return np.where(ok, midpoints[best], np.nan)


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
        Its peak lag (the lag of its largest absolute value) and half-maximum
        onset, in seconds.
    peak_lag_interval, onset_lag_interval : tuple of float
        2.5th and 97.5th percentiles over the resamples of the groups, in
        seconds.
    steepest_lag : float
        Its steepest approach to the peak (:func:`steepest_approach_lag`), s.
    steepest_lag_interval : tuple of float
    size : float
        Its value at the peak less the reference level of ``baseline``
        (:func:`response_size`), log-rate units. Each unit's curve is signed
        by its own peak, so an increase and a decrease both count as positive;
        the size is negative only if the mean curve's largest absolute value
        is on the other side of zero from the units' own peaks.
    size_interval : tuple of float
    n_units, n_groups : int
        Units and groups the curve averages.
    """

    curve: NDArray
    peak_lag: float
    onset_lag: float
    peak_lag_interval: tuple[float, float]
    onset_lag_interval: tuple[float, float]
    steepest_lag: float
    steepest_lag_interval: tuple[float, float]
    size: float
    size_interval: tuple[float, float]
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
    baseline: str | tuple[float, float] = "zero",
) -> PopulationTiming:
    """Timing of the sign-aligned population curve, resampling groups of units.

    Each bootstrap draw resamples the groups (``groups``: the units' sessions,
    or their recording days) with replacement and averages every unit of every
    drawn group, a group drawn twice counting twice; the aligned mean curve of
    the draw gives one peak lag and one onset.

    Parameters
    ----------
    curves : NDArray, shape (n_lags, n_units)
    peak : NDArray, shape (n_units,)
        Signed peaks, from :func:`peak_and_onset`.
    groups : NDArray, shape (n_units,)
        The label each unit is resampled by.
    lag_time : NDArray, shape (n_lags,)
    n_boot : int
        Resamples of the groups.
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
        return PopulationTiming(
            empty,
            np.nan,
            np.nan,
            nan_pair,
            nan_pair,
            np.nan,
            nan_pair,
            np.nan,
            nan_pair,
            0,
            0,
        )

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

    observed_steepest = steepest_approach_lag(curve, lag_time)
    boot_steepest = steepest_approach_lag(boot_curves.T, lag_time)
    observed_size = response_size(curve, lag_time, baseline=baseline)
    boot_size = response_size(boot_curves.T, lag_time, baseline=baseline)

    def interval(values: NDArray) -> tuple[float, float]:
        low, high = np.nanpercentile(values, [2.5, 97.5])
        return float(low), float(high)

    return PopulationTiming(
        curve=curve,
        peak_lag=float(observed.peak_lag[0]),
        onset_lag=float(observed.onset_lag[0]),
        peak_lag_interval=interval(boot.peak_lag),
        onset_lag_interval=interval(boot.onset_lag),
        steepest_lag=float(observed_steepest[0]),
        steepest_lag_interval=interval(boot_steepest),
        size=float(observed_size[0]),
        size_interval=interval(boot_size),
        n_units=int(curves.shape[1]),
        n_groups=n_groups,
    )


# --- timing statistics of the units called for an event class ------------------------

#: The sign of a unit's peak, by the word for its direction.
DIRECTIONS = {"enhanced": 1.0, "suppressed": -1.0}

_MS = 1e3

#: The columns of :func:`called_response_timing`.
_CALLED_TIMING_COLUMNS = (
    "n_units",
    "peak_lag_median_ms",
    "peak_lag_quartile_low_ms",
    "peak_lag_quartile_high_ms",
    "peak_before_onset_fraction",
    "onset_median_ms",
    "onset_before_onset_fraction",
    "onset_at_edge_fraction",
    "peak_at_edge_fraction",
    "rise_onset_median_ms",
    "steepest_median_ms",
    "population_pre_event_level_fraction",
    "population_peak_ms",
    "population_peak_ms_ci_low",
    "population_peak_ms_ci_high",
    "population_onset_ms",
    "population_onset_ms_ci_low",
    "population_onset_ms_ci_high",
    "population_rise_onset_ms",
    "population_rise_onset_ms_ci_low",
    "population_rise_onset_ms_ci_high",
    "population_steepest_ms",
    "population_steepest_ms_ci_low",
    "population_steepest_ms_ci_high",
    "population_size",
    "population_size_ci_low",
    "population_size_ci_high",
    "population_rise_size",
    "population_rise_size_ci_low",
    "population_rise_size_ci_high",
    "population_size_vs_control",
    "population_control_pre_event_level_fraction",
    "n_animals",
    "n_animals_onset_before_event",
    "n_animals_steepest_before_event",
    "n_animals_rise_onset_before_event",
)


def called_response_timing_by_animal(
    curves: NDArray, lag_time: NDArray, called: NDArray, animals: NDArray
) -> pd.DataFrame:
    """When each animal's own mean response starts, by direction of the peak.

    The called units are split by the sign of their peak; within each
    direction, every animal with such a unit has its own sign-aligned mean
    curve (:func:`sign_aligned_mean`), whose start is read by three rules.

    Parameters
    ----------
    curves : NDArray, shape (n_lags, n_units)
        Each unit's peri-event curve.
    lag_time : NDArray, shape (n_lags,)
        Lag of each row, in seconds, ascending.
    called : NDArray of bool, shape (n_units,)
        The units called for the event class.
    animals : NDArray, shape (n_units,)
        Each unit's animal.

    Returns
    -------
    timing : pd.DataFrame, shape (n_direction_animals, 3)
        Indexed by ``direction`` (a key of :data:`DIRECTIONS`) and ``animal``,
        animals in sorted order. Each lag is in ms, relative to event onset:

        - ``onset_ms``: the half-maximum onset of the animal's mean curve.
        - ``rise_onset_ms``: its rise onset, measured from the curve's value
          at the first lag.
        - ``steepest_ms``: its steepest approach to the peak
          (:func:`steepest_approach_lag`); NaN when the mean curve peaks at
          the first lag.
    """
    timing = peak_and_onset(curves, lag_time)
    directions, names, rows = [], [], []
    for direction, sign in DIRECTIONS.items():
        chosen = called & (np.sign(timing.peak) == sign)
        for animal in np.unique(animals[chosen]):
            own = chosen & (animals == animal)
            own_curve = sign_aligned_mean(curves[:, own], timing.peak[own])
            onset = peak_and_onset(own_curve, lag_time).onset_lag[0]
            rise = peak_and_onset(own_curve, lag_time, baseline="window_start")
            steepest = steepest_approach_lag(own_curve, lag_time)[0]
            directions.append(direction)
            names.append(animal)
            rows.append(
                {
                    "onset_ms": onset * _MS,
                    "rise_onset_ms": rise.onset_lag[0] * _MS,
                    "steepest_ms": steepest * _MS,
                }
            )
    return pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_arrays(
            [directions, names], names=["direction", "animal"]
        ),
        columns=["onset_ms", "rise_onset_ms", "steepest_ms"],
    )


def called_response_timing(
    curves: NDArray,
    control: NDArray,
    lag_time: NDArray,
    called: NDArray,
    groups: NDArray,
    animals: NDArray,
    *,
    n_boot: int = 1_000,
    seed: int = 0,
    half_window_s: float = REPORT_HALF_WINDOW_S,
) -> pd.DataFrame:
    """When the response of the units called for an event class peaks and starts.

    The called units are split by the sign of their peak (:data:`DIRECTIONS`).
    For each direction, the timing is given per unit (medians and shares over
    units) and for the population: the sign-aligned mean curve of those units
    (:func:`population_timing`), with 95 % intervals from resampling
    ``groups``. Lags are reported in ms, relative to event onset.

    Parameters
    ----------
    curves : NDArray, shape (n_lags, n_units)
        Each unit's peri-event curve for the event class.
    control : NDArray, shape (n_lags, n_units)
        Each unit's matched-control curve; a unit's may be NaN.
    lag_time : NDArray, shape (n_lags,)
        Lag of each row, in seconds, ascending.
    called : NDArray of bool, shape (n_units,)
        The units called for the event class.
    groups : NDArray, shape (n_units,)
        The label each unit is resampled by (its recording day).
    animals : NDArray, shape (n_units,)
        Each unit's animal.
    n_boot, seed : int
        Resamples of ``groups`` behind the intervals, and their seed.
    half_window_s : float
        Half-width of the window the curves are on, in seconds.

    Returns
    -------
    statistics : pd.DataFrame, shape (2, n_statistics)
        Indexed by direction. Every statistic but ``n_units`` is NaN when the
        direction has no unit.

        Over units:

        - ``n_units``: the called units of that direction.
        - ``peak_lag_median_ms``, ``peak_lag_quartile_low_ms``,
          ``peak_lag_quartile_high_ms``: the median and the 25th and 75th
          percentiles of the peak lag.
        - ``peak_before_onset_fraction``: the share whose peak lag is negative.
        - ``onset_median_ms``: the median half-maximum onset
          (:func:`peak_and_onset`); ``onset_before_onset_fraction``: the share
          whose onset is negative; ``onset_at_edge_fraction``: the share whose
          half-maximum run reaches the first lag.
        - ``peak_at_edge_fraction``: the share whose peak is at either edge of
          the window.
        - ``rise_onset_median_ms``: the median rise onset (the onset measured
          from the curve's value at the first lag).
        - ``steepest_median_ms``: the median lag of the steepest approach to
          the peak (:func:`steepest_approach_lag`).

        For the sign-aligned mean curve (each with ``_ci_low`` and
        ``_ci_high``, the 2.5th and 97.5th percentiles over the resamples,
        except the two ``fraction`` statistics and ``size_vs_control``):

        - ``population_peak_ms``, ``population_onset_ms``,
          ``population_rise_onset_ms``, ``population_steepest_ms``: its peak
          lag, half-maximum onset, rise onset and steepest approach.
        - ``population_size``: its peak, from zero; ``population_rise_size``:
          its peak less its value at the first lag (log-rate scale).
        - ``population_pre_event_level_fraction``: its value at the first lag
          as a share of its largest value, clipped to [0, 1].
        - ``population_size_vs_control``: its largest value less, at that lag,
          the mean of the same units' control curves signed the same way, over
          the units with a finite control curve (NaN when there is none).
        - ``population_control_pre_event_level_fraction``: that control mean
          at the first lag as a share of the mean curve's largest value,
          clipped to [0, 1].

        Over animals (:func:`called_response_timing_by_animal`):

        - ``n_animals``: animals with a unit of that direction.
        - ``n_animals_onset_before_event``,
          ``n_animals_rise_onset_before_event``,
          ``n_animals_steepest_before_event``: those whose own mean curve has
          a negative half-maximum onset, rise onset and steepest approach.
    """
    timing = peak_and_onset(curves, lag_time)
    by_animal = called_response_timing_by_animal(curves, lag_time, called, animals)
    rows = {}
    for direction, sign in DIRECTIONS.items():
        chosen = called & (np.sign(timing.peak) == sign)
        row: dict[str, float] = dict.fromkeys(_CALLED_TIMING_COLUMNS, np.nan)
        row["n_units"] = int(chosen.sum())
        rows[direction] = row
        if not chosen.any():
            continue
        peak_ms = timing.peak_lag[chosen] * _MS
        onset_ms = timing.onset_lag[chosen] * _MS
        row["peak_lag_median_ms"] = np.median(peak_ms)
        low, high = np.percentile(peak_ms, [25, 75])
        row["peak_lag_quartile_low_ms"] = low
        row["peak_lag_quartile_high_ms"] = high
        row["peak_before_onset_fraction"] = float((peak_ms < 0).mean())
        row["onset_median_ms"] = np.median(onset_ms)
        row["onset_before_onset_fraction"] = float((onset_ms < 0).mean())
        row["onset_at_edge_fraction"] = float(timing.onset_at_edge[chosen].mean())
        row["peak_at_edge_fraction"] = float(
            (np.abs(peak_ms) >= half_window_s * _MS - 1e-6).mean()
        )
        rise_ms = (
            peak_and_onset(
                curves[:, chosen], lag_time, baseline="window_start"
            ).onset_lag
            * _MS
        )
        row["rise_onset_median_ms"] = np.nanmedian(rise_ms)
        population_curve = population_timing(
            curves[:, chosen],
            timing.peak[chosen],
            groups[chosen],
            lag_time,
            n_boot=n_boot,
            seed=seed,
        )
        population_rise = population_timing(
            curves[:, chosen],
            timing.peak[chosen],
            groups[chosen],
            lag_time,
            n_boot=n_boot,
            seed=seed,
            baseline="window_start",
        )
        row["population_pre_event_level_fraction"] = float(
            np.clip(
                population_curve.curve[0] / np.max(population_curve.curve),
                0.0,
                1.0,
            )
        )
        row["population_rise_onset_ms"] = population_rise.onset_lag * _MS
        row["population_rise_onset_ms_ci_low"] = (
            population_rise.onset_lag_interval[0] * _MS
        )
        row["population_rise_onset_ms_ci_high"] = (
            population_rise.onset_lag_interval[1] * _MS
        )
        for what, point, (ci_low, ci_high) in (
            ("peak", population_curve.peak_lag, population_curve.peak_lag_interval),
            ("onset", population_curve.onset_lag, population_curve.onset_lag_interval),
        ):
            row[f"population_{what}_ms"] = point * _MS
            row[f"population_{what}_ms_ci_low"] = ci_low * _MS
            row[f"population_{what}_ms_ci_high"] = ci_high * _MS
        steepest_ms = steepest_approach_lag(curves[:, chosen], lag_time) * _MS
        row["steepest_median_ms"] = np.nanmedian(steepest_ms)
        row["population_steepest_ms"] = population_curve.steepest_lag * _MS
        for bound, value in zip(
            ("low", "high"), population_curve.steepest_lag_interval, strict=True
        ):
            row[f"population_steepest_ms_ci_{bound}"] = value * _MS
        for rule, result in (
            ("size", population_curve),
            ("rise_size", population_rise),
        ):
            row[f"population_{rule}"] = result.size
            for bound, value in zip(("low", "high"), result.size_interval, strict=True):
                row[f"population_{rule}_ci_{bound}"] = value
        chosen_control = control[:, chosen]
        usable = np.isfinite(chosen_control).all(axis=0)
        if usable.any():
            at_peak = int(np.argmax(population_curve.curve))
            control_mean = sign_aligned_mean(
                chosen_control[:, usable], timing.peak[chosen][usable]
            )
            row["population_size_vs_control"] = (
                population_curve.curve[at_peak] - control_mean[at_peak]
            )
            row["population_control_pre_event_level_fraction"] = float(
                np.clip(control_mean[0] / np.max(population_curve.curve), 0, 1)
            )
        own = by_animal.loc[direction]
        row["n_animals"] = len(own)
        for rule in ("onset", "steepest", "rise_onset"):
            row[f"n_animals_{rule}_before_event"] = int((own[f"{rule}_ms"] < 0).sum())
    return pd.DataFrame.from_dict(
        rows, orient="index", columns=list(_CALLED_TIMING_COLUMNS)
    )


def paired_peak_lag_statistics(
    first: NDArray,
    second: NDArray,
    lag_time: NDArray,
    first_called: NDArray,
    second_called: NDArray,
) -> dict[str, float]:
    """Whether units called for two event classes peak later for the second.

    Parameters
    ----------
    first, second : NDArray, shape (n_lags, n_units)
        Each unit's curve for the two classes.
    lag_time : NDArray, shape (n_lags,)
        Lag of each row, in seconds, ascending.
    first_called, second_called : NDArray of bool, shape (n_units,)
        The units called for each class.

    Returns
    -------
    statistics : dict
        - ``n_paired``: units called for both classes whose two peaks have the
          same sign.
        - ``difference_median_ms``: over those units, the median of the second
          class's peak lag minus the first's, in ms.
        - ``second_later_fraction``: the share of those units whose second
          peak is later than their first.

        The last two are NaN when there is no such unit.
    """
    first_timing = peak_and_onset(first, lag_time)
    second_timing = peak_and_onset(second, lag_time)
    paired = (
        first_called
        & second_called
        & (np.sign(first_timing.peak) == np.sign(second_timing.peak))
    )
    difference = (second_timing.peak_lag[paired] - first_timing.peak_lag[paired]) * _MS
    statistics = {
        "n_paired": int(paired.sum()),
        "difference_median_ms": np.nan,
        "second_later_fraction": np.nan,
    }
    if paired.any():
        statistics["difference_median_ms"] = np.median(difference)
        statistics["second_later_fraction"] = float((difference > 0).mean())
    return statistics


def peak_before_onset_fraction(curves: NDArray, lag_time: NDArray) -> float:
    """The share of curves whose peak lag is negative.

    For matched-control curves, which hold no event, this is what a peak lag
    drawn by noise gives.

    Parameters
    ----------
    curves : NDArray, shape (n_lags, n_units)
    lag_time : NDArray, shape (n_lags,)
        Lag of each row, in seconds, ascending.

    Returns
    -------
    fraction : float
        Over the curves with a peak (finite and not identically zero); NaN
        when there is none.
    """
    timing = peak_and_onset(curves, lag_time)
    usable = np.isfinite(timing.peak_lag)
    return float((timing.peak_lag[usable] < 0).mean())
