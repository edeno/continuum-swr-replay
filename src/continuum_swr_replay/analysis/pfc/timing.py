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
measures it on simulated responses of known onset, read by
:func:`timing_rule_calibration`. Read onsets beside it.

The population curve is the mean over units of each curve multiplied by the
sign of its own peak (so a suppression counts as a response of the same
size), with a 95 % interval from the nested bootstrap
(:func:`~continuum_swr_replay.analysis.shared_statistics.nested_bootstrap_draw`:
the animals, then each drawn animal's groups of units, its sessions or its
recording days, then each drawn group's units): the statistic plus or minus a
small-sample t correction over the animals times the draws' standard deviation
(:func:`~continuum_swr_replay.analysis.shared_statistics.bootstrap_interval`). An animal's own curve is the
mean over its groups of each group's sign-aligned mean curve.

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

from continuum_swr_replay.analysis.shared_statistics import (
    animal_balanced_curve,
    bootstrap_interval,
    nested_bootstrap_draw,
    nested_codes,
)


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
        95 % intervals from the nested resamples
        (:func:`~continuum_swr_replay.analysis.shared_statistics.bootstrap_interval`
        over the animals), in seconds; NaN when the curve has no such lag or
        there are fewer than two animals.
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
    n_units, n_groups, n_animals : int
        Units, groups and animals the curve averages.
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
    n_animals: int


#: Nested-bootstrap replicates behind the reported population timing intervals,
#: and their seed.
TIMING_BOOTSTRAP_DRAWS = 1_000
TIMING_BOOTSTRAP_SEED = 0


def population_timing(
    curves: NDArray,
    peak: NDArray,
    groups: NDArray,
    animals: NDArray,
    lag_time: NDArray,
    *,
    n_boot: int = TIMING_BOOTSTRAP_DRAWS,
    seed: int = TIMING_BOOTSTRAP_SEED,
    fraction: float = 0.5,
    baseline: str | tuple[float, float] = "zero",
) -> PopulationTiming:
    """Timing of the sign-aligned population curve, with nested-bootstrap intervals.

    Each bootstrap draw is one replicate of the nested bootstrap
    (:func:`~continuum_swr_replay.analysis.shared_statistics.nested_bootstrap_draw`):
    the animals with replacement, then each drawn animal's groups (``groups``:
    the units' sessions, or their recording days), then each drawn group's
    units; the mean of the drawn units' aligned curves, a unit drawn twice
    counting twice, gives one peak lag and one onset.

    Parameters
    ----------
    curves : NDArray, shape (n_lags, n_units)
    peak : NDArray, shape (n_units,)
        Signed peaks, from :func:`peak_and_onset`.
    groups : NDArray, shape (n_units,)
        Each unit's group, the middle level of the resampling; a group belongs
        to one animal.
    animals : NDArray, shape (n_units,)
        Each unit's animal.
    lag_time : NDArray, shape (n_lags,)
    n_boot : int
        Bootstrap replicates.
    seed : int
        Seed of the resampling.
    fraction, baseline
        Onset rule, as in :func:`peak_and_onset`.

    Returns
    -------
    PopulationTiming
    """
    curves = np.asarray(curves, dtype=float)
    resampled = _resampled_mean_curves(
        curves, peak, groups, animals, n_boot=n_boot, seed=seed
    )
    if resampled is None:
        nan_pair = (float("nan"), float("nan"))
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
            0,
        )
    return _mean_curve_timing(
        *resampled,
        lag_time,
        n_units=curves.shape[1],
        fraction=fraction,
        baseline=baseline,
    )


def _resampled_mean_curves(
    curves: NDArray,
    peak: NDArray,
    groups: NDArray,
    animals: NDArray,
    *,
    n_boot: int,
    seed: int,
) -> tuple[NDArray, NDArray, int, int] | None:
    """The sign-aligned mean curve, shape (n_lags,), its ``n_boot`` nested
    resamples (animals, groups, units), shape (n_boot, n_lags), and the numbers of
    groups and of animals; None when there is no unit."""
    aligned = curves * np.sign(np.asarray(peak, dtype=float))[None, :]
    if aligned.shape[1] == 0:
        return None
    animal_codes, group_codes = nested_codes(animals, groups)
    n_groups = int(group_codes.max()) + 1
    # The mean over units, summed group by group.
    sums = np.zeros((n_groups, curves.shape[0]))
    np.add.at(sums, group_codes, aligned.T)
    curve = sums.sum(axis=0) / aligned.shape[1]

    rng = np.random.default_rng(seed)
    counts = np.stack(
        [
            nested_bootstrap_draw(animal_codes, group_codes, rng).rows
            for _ in range(n_boot)
        ]
    ).astype(float)  # (n_boot, n_units)
    boot_curves = (counts @ aligned.T) / counts.sum(axis=1)[:, None]
    return curve, boot_curves, n_groups, int(animal_codes.max()) + 1


def _mean_curve_timing(
    curve: NDArray,
    boot_curves: NDArray,
    n_groups: int,
    n_animals: int,
    lag_time: NDArray,
    *,
    n_units: int,
    fraction: float,
    baseline: str | tuple[float, float],
) -> PopulationTiming:
    """:func:`population_timing` of a mean curve and its resamples
    (:func:`_resampled_mean_curves`)."""
    observed = peak_and_onset(curve, lag_time, fraction=fraction, baseline=baseline)
    boot = peak_and_onset(boot_curves.T, lag_time, fraction=fraction, baseline=baseline)

    observed_steepest = steepest_approach_lag(curve, lag_time)
    boot_steepest = steepest_approach_lag(boot_curves.T, lag_time)
    observed_size = response_size(curve, lag_time, baseline=baseline)
    boot_size = response_size(boot_curves.T, lag_time, baseline=baseline)

    lag_bounds = (float(lag_time[0]), float(lag_time[-1]))

    def interval(
        estimate: float, draws: NDArray, bounds: tuple[float, float] | None = None
    ) -> tuple[float, float]:
        low, high, _ = bootstrap_interval(estimate, draws, n_animals, bounds)
        return low, high

    peak_lag = float(observed.peak_lag[0])
    onset_lag = float(observed.onset_lag[0])
    steepest_lag = float(observed_steepest[0])
    size = float(observed_size[0])
    return PopulationTiming(
        curve=curve,
        peak_lag=peak_lag,
        onset_lag=onset_lag,
        peak_lag_interval=interval(peak_lag, boot.peak_lag, lag_bounds),
        onset_lag_interval=interval(onset_lag, boot.onset_lag, lag_bounds),
        steepest_lag=steepest_lag,
        steepest_lag_interval=interval(steepest_lag, boot_steepest, lag_bounds),
        size=size,
        size_interval=interval(size, boot_size),
        n_units=n_units,
        n_groups=n_groups,
        n_animals=n_animals,
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


def population_rise_timing(
    curves: NDArray,
    peak: NDArray,
    groups: NDArray,
    animals: NDArray,
    lag_time: NDArray,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, float]:
    """When the sign-aligned mean curve of a set of units peaks and starts to rise.

    The timing of the mean curve measured from its value at the first lag
    (:func:`population_timing` with ``baseline="window_start"``), with 95 %
    intervals from the nested bootstrap of animals, ``groups`` and units.

    Parameters
    ----------
    curves : NDArray, shape (n_lags, n_units)
        Each unit's peri-event curve.
    peak : NDArray, shape (n_units,)
        Signed peaks, from :func:`peak_and_onset`.
    groups : NDArray, shape (n_units,)
        Each unit's group, the middle level of the resampling.
    animals : NDArray, shape (n_units,)
        Each unit's animal.
    lag_time : NDArray, shape (n_lags,)
        Lag of each row, in seconds, ascending.
    n_boot, seed : int
        Bootstrap replicates behind the intervals, and their seed.

    Returns
    -------
    statistics : dict
        - ``peak_ms``: the lag of the mean curve's largest absolute value, in
          ms.
        - ``rise_onset_ms``: its rise onset, in ms: the earliest lag of the
          run, ending at the peak, over which the curve is at least half way
          from its value at the first lag to the peak.
        - ``rise_size``: its value at the peak less its value at the first
          lag, on the log-rate scale.
        - ``rise_onset_ms_ci_low``, ``rise_onset_ms_ci_high``,
          ``rise_size_ci_low``, ``rise_size_ci_high``: the 95 % intervals of
          the rise onset and the rise size from the resamples, a small-sample
          t correction over the animals (NaN with one animal).

        One unit is enough: the mean curve is then that unit's own. With no
        unit, every statistic is NaN.
    """
    return _rise_statistics(
        population_timing(
            curves,
            peak,
            groups,
            animals,
            lag_time,
            n_boot=n_boot,
            seed=seed,
            baseline="window_start",
        )
    )


def _rise_statistics(rise: PopulationTiming) -> dict[str, float]:
    """The statistics of :func:`population_rise_timing` from the timing of the
    mean curve measured from its value at the first lag."""
    return {
        "peak_ms": rise.peak_lag * _MS,
        "rise_onset_ms": rise.onset_lag * _MS,
        "rise_onset_ms_ci_low": rise.onset_lag_interval[0] * _MS,
        "rise_onset_ms_ci_high": rise.onset_lag_interval[1] * _MS,
        "rise_size": rise.size,
        "rise_size_ci_low": rise.size_interval[0],
        "rise_size_ci_high": rise.size_interval[1],
    }


def called_response_timing_by_animal(
    curves: NDArray,
    lag_time: NDArray,
    called: NDArray,
    groups: NDArray,
    animals: NDArray,
) -> pd.DataFrame:
    """When each animal's own mean response starts, by direction of the peak.

    The called units are split by the sign of their peak; within each
    direction, every animal with such a unit has its own curve, unit -> group
    -> animal: the mean over its groups of each group's sign-aligned mean
    curve (:func:`sign_aligned_mean`;
    :func:`~continuum_swr_replay.analysis.shared_statistics.animal_balanced_curve`),
    so every group of an animal weighs the same however many units it holds.
    Its start is read by three rules.

    Parameters
    ----------
    curves : NDArray, shape (n_lags, n_units)
        Each unit's peri-event curve.
    lag_time : NDArray, shape (n_lags,)
        Lag of each row, in seconds, ascending.
    called : NDArray of bool, shape (n_units,)
        The units called for the event class.
    groups : NDArray, shape (n_units,)
        Each unit's group (its session, or its recording day).
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
        if not chosen.any():
            continue
        by_group = {
            (animal, group): sign_aligned_mean(curves[:, mine], timing.peak[mine])
            for animal, group in sorted(
                set(zip(animals[chosen], groups[chosen], strict=True))
            )
            for mine in [chosen & (animals == animal) & (groups == group)]
        }
        _, per_animal = animal_balanced_curve(
            by_group, {key: key[0] for key in by_group}
        )
        for animal, own_curve in per_animal.items():
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
    n_boot: int = TIMING_BOOTSTRAP_DRAWS,
    seed: int = TIMING_BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """When the response of the units called for an event class peaks and starts.

    The called units are split by the sign of their peak (:data:`DIRECTIONS`);
    a called unit whose curve is not finite or is zero everywhere has no peak
    and is in neither direction. For each direction, the timing is given per
    unit (medians and shares over units) and for the population: the
    sign-aligned mean curve of those units (:func:`population_timing`), with
    95 % intervals from the nested bootstrap of animals, ``groups`` and units.
    Lags are reported in ms, relative to event onset.

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
        Each unit's group (its recording day): the middle level of the
        resampling and of each animal's own curve.
    animals : NDArray, shape (n_units,)
        Each unit's animal.
    n_boot, seed : int
        Bootstrap replicates behind the intervals, and their seed.

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
        - ``peak_at_edge_fraction``: the share whose peak is at the first or
          the last lag of ``lag_time``.
        - ``rise_onset_median_ms``: the median rise onset (the onset measured
          from the curve's value at the first lag).
        - ``steepest_median_ms``: the median lag of the steepest approach to
          the peak (:func:`steepest_approach_lag`), over the units whose peak
          is not at the first lag (a peak there has no approach).

        For the sign-aligned mean curve (each with ``_ci_low`` and
        ``_ci_high``, its 95 % interval from the replicates with a
        small-sample t correction over the animals, except the two
        ``fraction`` statistics and ``size_vs_control``):

        - ``population_peak_ms``, ``population_onset_ms``,
          ``population_rise_onset_ms``, ``population_steepest_ms``: its peak
          lag, half-maximum onset, rise onset and steepest approach.
        - ``population_size``: its peak, from zero; ``population_rise_size``:
          its peak less its value at the first lag (log-rate scale). The rise
          onset and the rise size are those of :func:`population_rise_timing`.
        - ``population_pre_event_level_fraction``: its value at the first lag
          as a share of its value at its peak (the lag of its largest absolute
          value, the peak of ``population_peak_ms`` and ``population_size``),
          clipped to the range 0 to 1: a first-lag value on the other side of
          zero from the peak is reported as 0.
        - ``population_size_vs_control``: over the units of that direction
          with a finite control curve, their sign-aligned mean curve at its
          peak less, at that lag, the mean of their control curves, each
          signed by its unit's peak (NaN when no unit has a control curve).
        - ``population_control_pre_event_level_fraction``: that control mean
          at the first lag as a share of the same units' mean curve at its
          peak, clipped to the range 0 to 1.

        Over animals (:func:`called_response_timing_by_animal`):

        - ``n_animals``: animals with a unit of that direction.
        - ``n_animals_onset_before_event``,
          ``n_animals_rise_onset_before_event``,
          ``n_animals_steepest_before_event``: those whose own mean curve has
          a negative half-maximum onset, rise onset and steepest approach.

    by_animal : pd.DataFrame
        The per-animal timing those counts are read from
        (:func:`called_response_timing_by_animal` of the same units).
    """
    timing = peak_and_onset(curves, lag_time)
    by_animal = called_response_timing_by_animal(
        curves, lag_time, called, groups, animals
    )
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
            np.isin(timing.peak_lag[chosen], lag_time[[0, -1]]).mean()
        )
        rise_ms = (
            peak_and_onset(
                curves[:, chosen], lag_time, baseline="window_start"
            ).onset_lag
            * _MS
        )
        row["rise_onset_median_ms"] = np.nanmedian(rise_ms)
        steepest_ms = steepest_approach_lag(curves[:, chosen], lag_time) * _MS
        row["steepest_median_ms"] = np.nanmedian(steepest_ms)
        # One resampling, read from zero and from the first lag.
        resampled = _resampled_mean_curves(
            curves[:, chosen],
            timing.peak[chosen],
            groups[chosen],
            animals[chosen],
            n_boot=n_boot,
            seed=seed,
        )
        population_curve, rise_curve = (
            _mean_curve_timing(
                *resampled,
                lag_time,
                n_units=int(chosen.sum()),
                fraction=0.5,
                baseline=baseline,
            )
            for baseline in ("zero", "window_start")
        )
        for what, point, (ci_low, ci_high) in (
            ("peak", population_curve.peak_lag, population_curve.peak_lag_interval),
            ("onset", population_curve.onset_lag, population_curve.onset_lag_interval),
            (
                "steepest",
                population_curve.steepest_lag,
                population_curve.steepest_lag_interval,
            ),
        ):
            row[f"population_{what}_ms"] = point * _MS
            row[f"population_{what}_ms_ci_low"] = ci_low * _MS
            row[f"population_{what}_ms_ci_high"] = ci_high * _MS
        row["population_size"] = population_curve.size
        row["population_size_ci_low"] = population_curve.size_interval[0]
        row["population_size_ci_high"] = population_curve.size_interval[1]
        rise = _rise_statistics(rise_curve)
        for name in (
            "rise_onset_ms",
            "rise_onset_ms_ci_low",
            "rise_onset_ms_ci_high",
            "rise_size",
            "rise_size_ci_low",
            "rise_size_ci_high",
        ):
            row[f"population_{name}"] = rise[name]
        mean_curve = population_curve.curve
        at_peak = int(np.argmax(np.abs(mean_curve)))
        row["population_pre_event_level_fraction"] = float(
            np.clip(mean_curve[0] / mean_curve[at_peak], 0.0, 1.0)
        )
        with_control = np.isfinite(control[:, chosen]).all(axis=0)
        if with_control.any():
            # Both means are over the units with a control curve: the mean
            # curve itself when every unit has one.
            own_peak = timing.peak[chosen][with_control]
            if not with_control.all():
                mean_curve = sign_aligned_mean(
                    curves[:, chosen][:, with_control], own_peak
                )
                at_peak = int(np.argmax(np.abs(mean_curve)))
            control_mean = sign_aligned_mean(
                control[:, chosen][:, with_control], own_peak
            )
            row["population_size_vs_control"] = (
                mean_curve[at_peak] - control_mean[at_peak]
            )
            row["population_control_pre_event_level_fraction"] = float(
                np.clip(control_mean[0] / mean_curve[at_peak], 0.0, 1.0)
            )
        own = by_animal.loc[direction]
        row["n_animals"] = len(own)
        for rule in ("onset", "steepest", "rise_onset"):
            row[f"n_animals_{rule}_before_event"] = int((own[f"{rule}_ms"] < 0).sum())
    statistics = pd.DataFrame.from_dict(
        rows, orient="index", columns=list(_CALLED_TIMING_COLUMNS)
    )
    return statistics, by_animal


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


# --- what the estimator does to a response of known timing ---------------------------


def timing_rule_readings(
    curves: NDArray,
    lag_time: NDArray,
    reference_lags: tuple[float, float] | None = None,
) -> dict[str, NDArray]:
    """Every timing and size rule of this module, read on each curve.

    Parameters
    ----------
    curves : NDArray, shape (n_lags,) or (n_lags, n_units)
    lag_time : NDArray, shape (n_lags,)
        Lag of each row, in seconds, ascending.
    reference_lags : (float, float), optional
        Lags, in seconds, of a pre-event reference level (a window wide enough
        to hold one); adds the rules measured from it.

    Returns
    -------
    readings : dict of str to NDArray, each shape (n_units,)
        Lags in ms, sizes on the log-rate scale:

        - ``peak``: the peak lag (:func:`peak_and_onset`).
        - ``onset_zero``, ``onset_rise``: the half-maximum onset measured from
          zero and from the value at the first lag.
        - ``onset_reference``: from the mean over ``reference_lags`` (only
          with ``reference_lags``).
        - ``steepest``: the steepest approach to the peak
          (:func:`steepest_approach_lag`).
        - ``size_zero``, ``size_rise``, ``size_reference``: the size
          (:func:`response_size`) from the same three references.
    """
    onsets: dict[str, str | tuple[float, float]] = {
        "onset_zero": "zero",
        "onset_rise": "window_start",
    }
    sizes: dict[str, str | tuple[float, float]] = {
        "size_zero": "zero",
        "size_rise": "window_start",
    }
    if reference_lags is not None:
        onsets["onset_reference"] = reference_lags
        sizes["size_reference"] = reference_lags
    readings = {"peak": peak_and_onset(curves, lag_time).peak_lag * _MS}
    for name, baseline in onsets.items():
        readings[name] = (
            peak_and_onset(curves, lag_time, baseline=baseline).onset_lag * _MS
        )
    readings["steepest"] = steepest_approach_lag(curves, lag_time) * _MS
    for name, baseline in sizes.items():
        readings[name] = response_size(curves, lag_time, baseline=baseline)
    return readings


def timing_rule_calibration(
    curves: NDArray,
    lag_time: NDArray,
    truth: NDArray | None,
    reference_lags: tuple[float, float] | None = None,
) -> dict:
    """How each timing rule reads fitted curves of a response of known shape.

    Each rule of :func:`timing_rule_readings` is read on every curve and on
    their sign-aligned mean (:func:`sign_aligned_mean`, each curve signed by
    its own peak), and compared with the same rule read on the truth's
    sign-aligned curve on the same lags, so a bias is the estimator's
    displacement and not the shape's own rise.

    Parameters
    ----------
    curves : NDArray, shape (n_lags, n_units)
        The fitted curves.
    lag_time : NDArray, shape (n_lags,)
        Lag of each row, in seconds, ascending.
    truth : NDArray, shape (n_lags,), or None
        The planted response on the same lags; None when nothing was planted,
        in which case a size's truth is 0 and a lag's has none.
    reference_lags : (float, float), optional
        As in :func:`timing_rule_readings`.

    Returns
    -------
    calibration : dict
        - ``fraction_onset_before_zero``, ``fraction_peak_before_zero``: the
          share of curves whose half-maximum onset from zero, or peak lag, is
          negative (a curve with no peak counts as not negative).
        - ``fraction_peak_sign_negative``: the share whose peak is a decrease.
        - ``rules``: per rule, ``truth`` (the rule on the truth; None for a lag
          with no truth), ``unit_median``, ``unit_p2_5`` and ``unit_p97_5``
          (the median and the 2.5th and 97.5th percentiles over curves, NaN
          left out), ``population`` (the rule on the sign-aligned mean curve),
          and ``unit_bias`` and ``population_bias`` (``unit_median`` and
          ``population`` less the truth; None without one). The truth is read
          sign-aligned, so for a planted decrease a size's truth is positive
          and ``unit_bias`` compares it with the curves' own, negative, sizes.
    """
    curves = np.asarray(curves, dtype=float)
    units = timing_rule_readings(curves, lag_time, reference_lags)
    peaks = peak_and_onset(curves, lag_time).peak
    population = {
        name: float(value[0])
        for name, value in timing_rule_readings(
            sign_aligned_mean(curves, peaks), lag_time, reference_lags
        ).items()
    }
    if truth is None:
        truth_read: dict[str, float] = {}
    else:
        truth_curve = np.asarray(truth, dtype=float)[:, None]
        truth_read = {
            name: float(value[0])
            for name, value in timing_rule_readings(
                sign_aligned_mean(
                    truth_curve, peak_and_onset(truth_curve, lag_time).peak
                ),
                lag_time,
                reference_lags,
            ).items()
        }
    rules = {}
    for name, values in units.items():
        low, mid, high = np.nanpercentile(values, [2.5, 50, 97.5])
        true = truth_read.get(name, 0.0 if name.startswith("size") else None)
        rules[name] = {
            "truth": true,
            "unit_median": float(mid),
            "unit_p2_5": float(low),
            "unit_p97_5": float(high),
            "population": population[name],
            "unit_bias": None if true is None else float(mid - true),
            "population_bias": None if true is None else float(population[name] - true),
        }
    return {
        "fraction_onset_before_zero": float(np.mean(units["onset_zero"] < 0)),
        "fraction_peak_before_zero": float(np.mean(units["peak"] < 0)),
        "fraction_peak_sign_negative": float(np.mean(peaks < 0)),
        "rules": rules,
    }
