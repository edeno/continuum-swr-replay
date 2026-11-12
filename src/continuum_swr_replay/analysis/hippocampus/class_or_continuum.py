"""One class or two: do SWR-coupled and non-SWR non-local events differ by kind or
by degree of ripple power?

An event's ``is_swr`` label is set by overlap with a Kay SWR, and a Kay SWR is a
run of the z-scored consensus ripple power at or above
:data:`KAY_ZSCORE_THRESHOLD` that lasts the detector's minimum duration,
extended to where the power returns to its mean. So the label is roughly a
thresholded version of a continuous quantity. Two tests separate "two classes"
from "one continuum cut by a threshold":

1. **One mode or two.** Over all non-local events, is the distribution of event
   ripple power unimodal? :func:`dip_test` (Hartigan's dip, with its p-value
   against the uniform, the least favourable unimodal law), :func:`sarle_bimodality`
   (descriptive only: a skewed unimodal law can exceed its 0.555 cutoff) and
   :func:`kde_modes`.
2. **A step at the threshold.** For an event property ``y``, does the label add
   anything beyond a smooth function of ripple power? :func:`fit_step` compares
   ``y ~ s(power) [+ covariates]`` with the same plus a 0/1 term, by held-out
   squared error (K folds grouped by epoch) and by the size of the 0/1 term
   with an epoch-clustered interval. Two versions:

   - *event power* (the event's peak or mean ripple power, the term ``is_swr``,
     the rows in the power range where both labels occur,
     :func:`common_support`). The power inside the event window is only part of
     the ripple the label sees (an SWR runs on to where the power returns to
     its mean), so the label carries information about the ripple that the
     window's power misses. :func:`simulate_ripple_events` shows that this
     version finds a "step" even in one continuum; it cannot decide.
   - *the detector's criterion* (:func:`kay_criterion`, the level the ripple
     the event touches sustains for the minimum duration; the term is
     ``criterion >= threshold``, rows in a window around the threshold). The
     label is a step function of this criterion, so this is a sharp regression
     discontinuity: one continuum gives no step whatever the noise in the
     power, and a class boundary at the threshold gives one.

The simulations check both tests on data whose answer is known:
:func:`matched_unimodal_sample`, :func:`matched_lognormal_sample` and
:func:`two_class_mixture_sample` for test 1; :func:`simulate_step_data` (a
planted step or none) and :func:`simulate_ripple_events` (one continuum, or one
with a planted step, labelled by the Kay rule on a simulated ripple-power
trace) for test 2.
"""

from __future__ import annotations

from dataclasses import dataclass

import diptest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import brentq
from scipy.signal import find_peaks
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import SplineTransformer

from continuum_swr_replay.analysis.hippocampus.swr_detectors import DETECTOR_SETTINGS

KAY_ZSCORE_THRESHOLD = float(DETECTOR_SETTINGS["kay"]["zscore_threshold"])
"""The z-score the Kay detector's SWRs must reach (2.0)."""

KAY_MINIMUM_DURATION_S = float(DETECTOR_SETTINGS["kay"]["minimum_duration"])
"""How long the power must stay at or above the threshold (0.015 s)."""

SARLE_CUTOFF = 5.0 / 9.0
"""Sarle's coefficient of a uniform law; above it a sample is often called bimodal."""


# --- test 1: one mode or two -------------------------------------------------


def sarle_bimodality(x: NDArray) -> float:
    """Sarle's bimodality coefficient ``(g^2 + 1) / (k + 3 (n-1)^2 / ((n-2)(n-3)))``.

    ``g`` is the sample skewness and ``k`` the sample excess kurtosis (both
    bias-corrected). Above :data:`SARLE_CUTOFF` a sample is often called
    bimodal, but a skewed unimodal law exceeds it too (an exponential gives
    5/9), so it is descriptive only.

    Parameters
    ----------
    x : NDArray, shape (n,)
        Finite values, n > 3.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    g = stats.skew(x, bias=False)
    k = stats.kurtosis(x, fisher=True, bias=False)
    return float((g**2 + 1.0) / (k + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))))


def dip_test(x: NDArray) -> tuple[float, float]:
    """Hartigan's dip statistic and its p-value (``diptest``, table interpolation).

    The null is the uniform law, the unimodal law with the largest dip, so a
    small p-value rejects every unimodal law.

    Parameters
    ----------
    x : NDArray, shape (n,)
        Finite values.

    Returns
    -------
    dip, p_value : float
    """
    dip, p_value = diptest.diptest(np.asarray(x, dtype=np.float64))
    return float(dip), float(p_value)


def kde_modes(
    x: NDArray,
    *,
    n_grid: int = 1024,
    trim: float = 0.005,
    prominence: float = 0.05,
) -> NDArray[np.float64]:
    """The modes of a Gaussian KDE (Silverman bandwidth), highest first.

    The KDE is read on ``n_grid`` points between the ``trim`` and ``1 - trim``
    quantiles. A local maximum counts as a mode when it stands at least
    ``prominence`` times the highest density above the deeper of the troughs
    that separate it from higher ground (``scipy.signal.find_peaks``'
    prominence), so the small ripples of a long tail are not modes.

    Parameters
    ----------
    x : NDArray, shape (n,)
        Finite values.

    Returns
    -------
    modes : NDArray, shape (n_modes,)
        The modes' locations, in ``x``'s units, by decreasing density; the
        grid's highest point when the density is monotone on it.
    """
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.quantile(x, [trim, 1.0 - trim])
    grid = np.linspace(lo, hi, n_grid)
    density = stats.gaussian_kde(x, bw_method="silverman")(grid)
    # pad with zeros so a maximum at a grid edge can count
    padded = np.concatenate([[0.0], density, [0.0]])
    peaks, _ = find_peaks(padded, prominence=prominence * density.max())
    peaks = peaks - 1
    if peaks.size == 0:
        peaks = np.array([int(np.argmax(density))])
    order = np.argsort(density[peaks])[::-1]
    return grid[peaks[order]]


def _gamma_parameters(
    mean: float, sd: float, skewness: float
) -> tuple[float, float, float]:
    """Shape, location and scale of the three-parameter gamma with these moments."""
    shape = 4.0 / skewness**2
    scale = sd / np.sqrt(shape)
    return shape, mean - shape * scale, scale


def matched_unimodal_sample(
    rng: np.random.Generator, n: int, mean: float, sd: float, skewness: float
) -> NDArray[np.float64]:
    """A unimodal sample with the given mean, SD and (positive) skewness.

    A three-parameter gamma (shape ``4 / skewness^2``): with shape below 1 its
    density falls from its lower bound, still one mode.
    """
    if skewness <= 0:
        raise ValueError(f"skewness must be positive, got {skewness}")
    shape, loc, scale = _gamma_parameters(mean, sd, skewness)
    return loc + rng.gamma(shape, scale, size=n)


def matched_lognormal_sample(
    rng: np.random.Generator, n: int, mean: float, sd: float, skewness: float
) -> NDArray[np.float64]:
    """A shifted lognormal sample with the given mean, SD and (positive) skewness."""
    if skewness <= 0:
        raise ValueError(f"skewness must be positive, got {skewness}")
    # skewness = (w + 2) sqrt(w - 1), w = exp(sigma^2): solve for w
    w = float(brentq(lambda w: (w + 2) * np.sqrt(w - 1) - skewness, 1 + 1e-12, 1e6))
    sigma = np.sqrt(np.log(w))
    unit = rng.lognormal(0.0, sigma, size=n)
    unit_mean = np.exp(sigma**2 / 2)
    unit_sd = np.sqrt((w - 1) * w)
    return mean + sd * (unit - unit_mean) / unit_sd


def two_class_mixture_sample(
    rng: np.random.Generator,
    n: int,
    locations: tuple[float, float],
    spreads: tuple[float, float],
    weight_second: float,
) -> NDArray[np.float64]:
    """A sample of two Gaussian classes: ``n`` draws, each from the second class
    with probability ``weight_second``."""
    second = rng.random(n) < weight_second
    return np.where(
        second,
        rng.normal(locations[1], spreads[1], n),
        rng.normal(locations[0], spreads[0], n),
    )


# --- test 2: a step at the threshold -----------------------------------------


def common_support(
    power: NDArray, is_swr: NDArray, *, tail: float = 0.025
) -> tuple[float, float]:
    """The ripple-power range where both labels occur.

    From the ``tail`` quantile of the SWR events' power to the ``1 - tail``
    quantile of the non-SWR events' power.

    Parameters
    ----------
    power : NDArray, shape (n_events,)
    is_swr : NDArray of bool, shape (n_events,)

    Returns
    -------
    low, high : float
        ``low < high`` when the labels overlap; otherwise ``low >= high`` and
        no event lies in the range.
    """
    power = np.asarray(power, dtype=np.float64)
    is_swr = np.asarray(is_swr, dtype=bool)
    low = float(np.quantile(power[is_swr], tail))
    high = float(np.quantile(power[~is_swr], 1.0 - tail))
    return low, high


def _spline(x: NDArray, n_knots: int) -> NDArray[np.float64]:
    """A cubic B-spline basis of ``x`` with knots at its quantiles, no constant."""
    transformer = SplineTransformer(
        n_knots=n_knots, degree=3, knots="quantile", include_bias=False
    )
    return np.asarray(
        transformer.fit_transform(np.asarray(x, dtype=np.float64)[:, None])
    )


@dataclass(frozen=True)
class StepFit:
    """One property's step test in one animal.

    Attributes
    ----------
    n_events, n_groups : int
        Events and epochs in the fit.
    n_labelled : int
        Events whose 0/1 term is 1.
    step, step_ci_low, step_ci_high : float
        The 0/1 term's coefficient (in ``y``'s units) and its 95 % interval,
        clustered by epoch.
    raw_difference : float
        Mean ``y`` of the events labelled 1 less that of the others, same rows.
    step_share : float
        ``step / raw_difference``: the part of the label difference that a
        step at the label carries once power (and any covariate) is smooth.
    label_vif : float
        ``1 / (1 - R^2)`` of the 0/1 term regressed on the smooth terms.
    label_power_correlation : float
        Point-biserial correlation of the 0/1 term with power.
    mse_smooth, mse_step : float
        Held-out mean squared error without and with the 0/1 term.
    heldout_gain : float
        ``(mse_smooth - mse_step) / mse_smooth``; above 0 the term helps.
    folds_step_better : int
        Folds in which the term lowers the held-out error.
    n_folds : int
    """

    n_events: int
    n_groups: int
    n_labelled: int
    step: float
    step_ci_low: float
    step_ci_high: float
    raw_difference: float
    step_share: float
    label_vif: float
    label_power_correlation: float
    mse_smooth: float
    mse_step: float
    heldout_gain: float
    folds_step_better: int
    n_folds: int

    @property
    def has_step(self) -> bool:
        """The interval excludes 0 and the term lowers the held-out error."""
        excludes_zero = self.step_ci_low > 0 or self.step_ci_high < 0
        return bool(excludes_zero and self.heldout_gain > 0)


def fit_step(
    y: NDArray,
    power: NDArray,
    label: NDArray,
    groups: NDArray,
    *,
    covariate: NDArray | None = None,
    n_knots: int = 10,
    covariate_knots: int = 6,
    n_splits: int = 5,
) -> StepFit:
    """``y ~ s(power) [+ s(covariate)]`` against the same ``+ label``.

    Ordinary least squares on all given rows (the caller chooses them: the
    common support, or a window around the threshold). ``s`` is a cubic
    B-spline with knots at quantiles of the rows' power, smooth everywhere, so
    the ``label`` term is the only place a jump can enter. The interval of the
    term is clustered by ``groups``; the held-out comparison uses
    ``n_splits`` folds that keep each group whole (``GroupKFold``). The knots
    read no ``y``.

    Parameters
    ----------
    y : NDArray, shape (n_events,)
        The property, already on the scale to compare on (e.g. log duration).
    power : NDArray, shape (n_events,)
        The ripple-power variable.
    label : NDArray of bool, shape (n_events,)
        ``is_swr``, or ``power >= threshold`` for the discontinuity design.
    groups : NDArray, shape (n_events,)
        Each event's epoch.
    covariate : NDArray, shape (n_events,), optional
        A second smooth term (e.g. log duration).
    n_knots, covariate_knots : int
        Knots of the two cubic B-splines.
    n_splits : int
        Folds; fewer when there are fewer groups.

    Returns
    -------
    StepFit
    """
    y = np.asarray(y, dtype=np.float64)
    power = np.asarray(power, dtype=np.float64)
    label = np.asarray(label, dtype=bool)
    groups = np.asarray(groups)
    columns = [np.ones((y.size, 1)), _spline(power, n_knots)]
    if covariate is not None:
        columns.append(_spline(covariate, covariate_knots))
    smooth = np.hstack(columns)
    full = np.hstack([smooth, label[:, None].astype(np.float64)])
    _, group_codes = np.unique(groups, return_inverse=True)

    fit = sm.OLS(y, full).fit(cov_type="cluster", cov_kwds={"groups": group_codes})
    step = float(fit.params[-1])
    ci_low, ci_high = (float(v) for v in fit.conf_int()[-1])

    label_fit = sm.OLS(label.astype(np.float64), smooth).fit()
    vif = 1.0 / max(1.0 - float(label_fit.rsquared), 1e-12)
    correlation = float(np.corrcoef(label.astype(np.float64), power)[0, 1])

    n_groups = int(np.unique(group_codes).size)
    n_folds = min(n_splits, n_groups)
    sse_smooth = sse_step = 0.0
    better = 0
    for train, test in GroupKFold(n_splits=n_folds).split(full, y, group_codes):
        errors = []
        for design in (smooth, full):
            coef, *_ = np.linalg.lstsq(design[train], y[train], rcond=None)
            errors.append(float(np.sum((y[test] - design[test] @ coef) ** 2)))
        sse_smooth += errors[0]
        sse_step += errors[1]
        better += int(errors[1] < errors[0])
    mse_smooth = sse_smooth / y.size
    mse_step = sse_step / y.size
    raw = float(y[label].mean() - y[~label].mean())
    return StepFit(
        n_events=int(y.size),
        n_groups=n_groups,
        n_labelled=int(label.sum()),
        step=step,
        step_ci_low=ci_low,
        step_ci_high=ci_high,
        raw_difference=raw,
        step_share=step / raw if raw != 0 else float("nan"),
        label_vif=vif,
        label_power_correlation=correlation,
        mse_smooth=mse_smooth,
        mse_step=mse_step,
        heldout_gain=(mse_smooth - mse_step) / mse_smooth,
        folds_step_better=better,
        n_folds=n_folds,
    )


def overall_step_share(fit: StepFit, y: NDArray, is_swr: NDArray) -> dict[str, float]:
    """A step as a share of the SWR minus non-SWR difference over all the events.

    :attr:`StepFit.step_share` divides by the label difference over the rows the
    fit compared on; this divides by the difference over every event with the
    property (``y``), whichever rows the fit used.

    Parameters
    ----------
    fit : StepFit
        The step test (:func:`fit_step`).
    y : NDArray, shape (n_events,)
        The property of every event, on the scale the fit compared on.
    is_swr : NDArray of bool, shape (n_events,)

    Returns
    -------
    share : dict
        - ``overall_difference``: mean ``y`` of the SWR events less that of the
          others.
        - ``step_share_overall``: ``fit.step`` over it.
        - ``share_overall_ci_low``, ``share_overall_ci_high``: the step's
          interval over it, lower bound first (a negative difference swaps the
          bounds).

    Raises
    ------
    ZeroDivisionError
        If the difference is zero.
    """
    y = np.asarray(y, dtype=np.float64)
    is_swr = np.asarray(is_swr, dtype=bool)
    overall = float(y[is_swr].mean() - y[~is_swr].mean())
    bounds = (fit.step_ci_low / overall, fit.step_ci_high / overall)
    return {
        "overall_difference": overall,
        "step_share_overall": fit.step / overall,
        "share_overall_ci_low": min(bounds),
        "share_overall_ci_high": max(bounds),
    }


# --- simulations for test 2 ----------------------------------------------------


def simulate_step_data(
    rng: np.random.Generator,
    *,
    n_groups: int = 40,
    n_per_group: int = 300,
    step: float = 0.0,
    label_slope: float = 2.0,
    noise_sd: float = 1.0,
) -> pd.DataFrame:
    """Events whose property is a smooth function of power, plus ``step`` at the label.

    Power is a skewed (gamma) variable with an epoch offset; the label is
    ``Bernoulli(logistic(label_slope * (power - KAY_ZSCORE_THRESHOLD)))``, so
    both labels occur near the threshold; ``y = sin(power) + 0.5 power + step *
    is_swr + epoch effect + noise``.

    Returns
    -------
    pd.DataFrame
        ``power``, ``is_swr``, ``y``, ``group``.
    """
    n = n_groups * n_per_group
    group = np.repeat(np.arange(n_groups), n_per_group)
    offset = rng.normal(0.0, 0.3, n_groups)[group]
    power = -0.5 + rng.gamma(1.5, 1.5, n) + offset
    p_swr = 1.0 / (1.0 + np.exp(-label_slope * (power - KAY_ZSCORE_THRESHOLD)))
    is_swr = rng.random(n) < p_swr
    epoch_effect = rng.normal(0.0, 0.2, n_groups)[group]
    y = (
        np.sin(power)
        + 0.5 * power
        + step * is_swr
        + epoch_effect
        + rng.normal(0.0, noise_sd, n)
    )
    return pd.DataFrame({"power": power, "is_swr": is_swr, "y": y, "group": group})


GRID_MINIMUM_SAMPLES = 8
"""The Kay minimum duration on the 2 ms decode grid. At the detector's 1 kHz,
15 ms is 15 consecutive samples (``ripple_detection.core.minimum_sample_count``),
which span 14 ms; on the decode frame's 2 ms grid (every other sample of the
same trace) such a run leaves 7 or 8 samples. 8 (14 ms) reproduces the
inventory's labels better than 7 on a check of one epoch per animal (7 calls
more brief crossings SWRs); the choice reads the labels only, no property."""


def _runs(mask: NDArray[np.bool_]) -> NDArray[np.int64]:
    """First and last index (inclusive) of each run of True, shape (n_runs, 2)."""
    edges = np.diff(np.concatenate([[0], np.asarray(mask, dtype=np.int8), [0]]))
    return np.column_stack(
        [np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1]
    ).astype(np.int64)


def sustained_level(z: NDArray, minimum_samples: int) -> NDArray[np.float64]:
    """Each sample's sustained level: the minimum of ``z`` over the
    ``minimum_samples`` samples starting there.

    A window holding a NaN, or running past the end, is ``-inf``. A run of
    ``minimum_samples`` samples at or above a threshold exists exactly where
    this reaches the threshold.

    Returns
    -------
    level : NDArray, shape (n_time,)
    """
    z = np.where(np.isfinite(z), np.asarray(z, dtype=np.float64), -np.inf)
    level = np.full(z.shape, -np.inf)
    if z.size >= minimum_samples:
        windows = np.lib.stride_tricks.sliding_window_view(z, minimum_samples)
        level[: windows.shape[0]] = windows.min(axis=1)
    return level


def kay_like_intervals(
    z: NDArray,
    *,
    threshold: float = KAY_ZSCORE_THRESHOLD,
    minimum_samples: int = GRID_MINIMUM_SAMPLES,
) -> NDArray[np.int64]:
    """The Kay rule on a z-scored trace: each run of ``z > 0`` that holds
    ``minimum_samples`` consecutive samples at or above ``threshold`` (a run
    above threshold, extended to where the trace returns to its mean). NaN is
    neither.

    Returns
    -------
    intervals : NDArray[np.int64], shape (n_swr, 2)
        First and last sample of each SWR (inclusive).
    """
    z = np.asarray(z, dtype=np.float64)
    positive = _runs(np.isfinite(z) & (z > 0))
    level = sustained_level(z, minimum_samples)
    if positive.size == 0:
        return positive
    keep = [level[a : b + 1].max() >= threshold for a, b in positive]
    return positive[np.asarray(keep, dtype=bool)]


def kay_criterion(
    z: NDArray,
    first: NDArray,
    last: NDArray,
    *,
    minimum_samples: int = GRID_MINIMUM_SAMPLES,
) -> NDArray[np.float64]:
    """The Kay detector's own criterion around each event: the highest level
    the ripple power sustains for ``minimum_samples`` samples in the event or
    in any run of ``z > 0`` the event overlaps.

    The event overlaps a :func:`kay_like_intervals` SWR exactly when this is
    at or above the threshold, so ``is_swr`` is a step function of it: the
    running variable of a sharp regression discontinuity. Below the threshold
    it grows with the ripple the event touches. A window that leaves the span
    holds a sample at or below 0; one holding a NaN is ``-inf``.

    Parameters
    ----------
    z : NDArray, shape (n_time,)
        The z-scored ripple power on the grid.
    first, last : NDArray of int, shape (n_events,)
        Each event's first and last row (inclusive), ``first <= last``.

    Returns
    -------
    criterion : NDArray, shape (n_events,)
        ``-inf`` where no window of the span is finite.
    """
    z = np.asarray(z, dtype=np.float64)
    first = np.asarray(first, dtype=np.int64)
    last = np.asarray(last, dtype=np.int64)
    positive = _runs(np.isfinite(z) & (z > 0))
    level = sustained_level(z, minimum_samples)
    # the runs overlapping [first, last]: starts at or before last, ends at or
    # after first; runs are sorted and disjoint
    lo = np.searchsorted(positive[:, 1], first, side="left")
    hi = np.searchsorted(positive[:, 0], last, side="right")
    span_first = np.where(
        hi > lo,
        np.minimum(first, positive[np.minimum(lo, len(positive) - 1), 0]),
        first,
    )
    span_last = np.where(
        hi > lo, np.maximum(last, positive[np.maximum(hi - 1, 0), 1]), last
    )
    criterion = np.empty(first.shape)
    for k, (a, b) in enumerate(zip(span_first, span_last, strict=True)):
        criterion[k] = level[a : max(b - minimum_samples + 2, a + 1)].max()
    return criterion


def simulate_ripple_events(
    rng: np.random.Generator,
    *,
    duration_s: float = 600.0,
    sample_interval_s: float = 0.002,
    burst_rate_hz: float = 1.0,
    amplitude_sigma: float = 0.8,
    background_sd: float = 1.0,
    jitter_fraction: float = 0.5,
    property_noise_sd: float = 0.5,
    step: float = 0.0,
) -> pd.DataFrame:
    """One continuum of events, labelled by the Kay rule on a simulated ripple trace.

    The trace is a smoothed noise background plus Gaussian ripple bursts at
    Poisson times, with lognormal (unimodal, continuous) amplitudes ``A`` and
    widths; it is z-scored and :func:`kay_like_intervals` gives the SWRs.
    Every burst carries one non-local event, centred near the burst (jitter
    ``jitter_fraction`` of its width), lasting longer for larger bursts. Each
    event's ``power_peak`` and ``power_mean`` are the trace's maximum and mean
    over its samples, as the real events' are, and ``power_criterion`` is
    :func:`kay_criterion`; ``is_swr`` is a strict overlap with an SWR. Its
    property ``y = log A + noise`` depends only on the burst (one continuum),
    plus ``step * is_swr`` when a step is planted;
    ``log_duration`` is the log of its duration in seconds.

    Returns
    -------
    pd.DataFrame
        One row per event: ``power_peak``, ``power_mean``, ``power_criterion``,
        ``is_swr``, ``y``,
        ``log_duration``, ``log_amplitude``, ``group`` (0).
    """
    n_time = int(round(duration_s / sample_interval_s))
    time = np.arange(n_time) * sample_interval_s
    kernel_t = np.arange(-15, 16) * sample_interval_s
    kernel = np.exp(-0.5 * (kernel_t / 0.004) ** 2)
    background = np.abs(
        np.convolve(rng.normal(0.0, 1.0, n_time), kernel / kernel.sum(), mode="same")
    )
    background *= background_sd / background.std()
    n_bursts = rng.poisson(burst_rate_hz * duration_s)
    centers = rng.uniform(0.5, duration_s - 0.5, n_bursts)
    amplitude = rng.lognormal(0.0, amplitude_sigma, n_bursts)
    width = 0.015 * np.exp(rng.normal(0.0, 0.3, n_bursts)) * amplitude**0.25
    trace = background.copy()
    for c, a, w in zip(centers, amplitude, width, strict=True):
        lo, hi = np.searchsorted(time, [c - 5 * w, c + 5 * w])
        trace[lo:hi] += a * np.exp(-0.5 * ((time[lo:hi] - c) / w) ** 2)
    z = (trace - trace.mean()) / trace.std()
    swr = kay_like_intervals(z)

    half = width * np.exp(rng.normal(0.0, 0.3, n_bursts))
    middle = centers + rng.normal(0.0, jitter_fraction, n_bursts) * width
    first = np.clip(np.searchsorted(time, middle - half), 0, n_time - 1)
    last = np.clip(np.searchsorted(time, middle + half), 0, n_time - 1)
    last = np.maximum(last, first + 6)  # at least 7 samples, as the real minimum
    cumulative = np.concatenate([[0.0], np.cumsum(z)])
    peak = np.array([z[a : b + 1].max() for a, b in zip(first, last, strict=True)])
    mean = (cumulative[last + 1] - cumulative[first]) / (last - first + 1)
    # strict overlap: some sample of the event is inside an SWR, not only touching
    in_swr = np.zeros(n_time + 1, dtype=np.int64)
    for a, b in swr:
        in_swr[a] += 1
        in_swr[b + 1] -= 1
    covered = np.concatenate([[0], np.cumsum(np.cumsum(in_swr)[:-1] > 0)])
    is_swr = (covered[last + 1] - covered[first]) > 0
    criterion = kay_criterion(z, first, last)
    log_amplitude = np.log(amplitude)
    y = log_amplitude + rng.normal(0.0, property_noise_sd, n_bursts) + step * is_swr
    return pd.DataFrame(
        {
            "power_peak": peak,
            "power_mean": mean,
            "power_criterion": criterion,
            "is_swr": is_swr,
            "y": y,
            "log_duration": np.log((last - first + 1) * sample_interval_s),
            "log_amplitude": log_amplitude,
            "group": 0,
        }
    )
