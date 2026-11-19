"""Simulated responses of known onset and size, fit as the mPFC kernels are.

The Figure 4 curves are ridge-penalized cubic B-spline kernels with knots every
50 ms, fit on +/-250 ms and read on +/-200 ms. :func:`simulate_and_fit` plants a
response in simulated Poisson units, fits it with the same knot spacing, prior
and fitter as ``run_session``, and reads the fit with every rule of
:func:`~continuum_swr_replay.analysis.pfc.timing.timing_rule_calibration`, so
a bias is the estimator's displacement and not the shape's own rise.
:func:`scenario_list` is the grid of scenarios the calibration reports
(``scripts/pfc/calibrate_pfc_kernel_timing.py`` runs it):

- ``response``: a 100 ms boxcar or an alpha function peaking 30 ms after its
  start; log-rate gains +0.7, -0.7 and -0.3; true starts -50, 0 and +50 ms;
- ``stop_offset``: no event-locked response, but the session is cut into 20 s
  stretches, 40 % of them at a log rate 0.3 lower, and 80 % of the events fall
  in those lower stretches;
- ``stop_offset_plus_response``: the same, plus a -0.7 boxcar from 0 ms;

each crossed over two windows (``canonical``: fit +/-0.25 s, read +/-0.2 s;
``wide``: fit +/-1.05 s, read +/-1.0 s).

Limits: every bin is fit (no immobility mask), one event class is fit (the
shipped curves come from two-class models), there is one baseline rate and one
event count, and units are not selected by a held-out test.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from continuum_swr_replay.analysis.pfc.batched_glm import (
    fit_poisson_glms,
    make_ridge_penalty,
)
from continuum_swr_replay.analysis.pfc.design import (
    convolve_event_basis,
    event_basis_functions,
    symmetric_lags,
)
from continuum_swr_replay.analysis.pfc.event_sets import DEFAULT_BASIS_HALF_WINDOW_S
from continuum_swr_replay.analysis.pfc.modulation import (
    REPORT_HALF_WINDOW_S,
    in_report_window,
)
from continuum_swr_replay.analysis.pfc.timing import timing_rule_calibration
from continuum_swr_replay.parameters import SAMPLING_FREQUENCY

#: Fitted bins and events of the simulated session (the median canonical
#: session's fitted bins and per-class event count).
DEFAULT_N_BINS = 267_000
DEFAULT_N_EVENTS = 100
MIN_EVENT_SPACING_S = 0.6
SHAPES = ("boxcar", "alpha")
GAINS = (0.7, -0.7, -0.3)
TRUE_ONSETS_MS = (-50.0, 0.0, 50.0)
BOXCAR_WIDTH_S = 0.100
ALPHA_PEAK_S = 0.030

#: Fit and read half-windows, in seconds.
WINDOWS = {
    "canonical": (DEFAULT_BASIS_HALF_WINDOW_S, REPORT_HALF_WINDOW_S),
    "wide": (1.05, 1.00),
}
#: Lags (s) the wide window's pre-event reference level is averaged over.
REFERENCE_LAGS_S = (-1.0, -0.6)

STRETCH_S = 20.0
LOW_STRETCH_FRACTION = 0.4
LOW_STRETCH_LOG_RATE = -0.3
EVENTS_IN_LOW_STRETCHES = 0.8


def response_shape(shape: str, lag_time: NDArray, onset_s: float) -> NDArray:
    """Unit-height response ``h(lag - onset)`` on ``lag_time`` (seconds)."""
    tau = lag_time - onset_s
    if shape == "boxcar":
        return ((tau >= 0) & (tau < BOXCAR_WIDTH_S)).astype(float)
    if shape == "alpha":
        scaled = np.clip(tau, 0.0, None) / ALPHA_PEAK_S
        return np.where(tau >= 0, scaled * np.exp(1.0 - scaled), 0.0)
    raise ValueError(f"unknown shape {shape!r}")


def event_onsets(n_bins: int, n_events: int, rng: np.random.Generator) -> NDArray:
    """Onset bins, uniformly placed, at least ``MIN_EVENT_SPACING_S`` apart."""
    margin = int(1.1 * SAMPLING_FREQUENCY)
    spacing = int(MIN_EVENT_SPACING_S * SAMPLING_FREQUENCY)
    slack = n_bins - 2 * margin - spacing * (n_events - 1)
    if slack <= 0:
        raise ValueError("too many events for the session length")
    gaps = np.sort(rng.integers(0, slack, size=n_events))
    return margin + gaps + spacing * np.arange(n_events)


def stretch_offset_and_onsets(
    n_bins: int, n_events: int, rng: np.random.Generator
) -> tuple[NDArray, NDArray]:
    """A stretch-wise log-rate offset and onsets crowded into the low stretches.

    Returns
    -------
    log_offset : NDArray, shape (n_bins,)
    onsets : NDArray of int, shape (n_events,)
    """
    stretch = int(STRETCH_S * SAMPLING_FREQUENCY)
    n_stretches = n_bins // stretch
    low = rng.permutation(n_stretches) < round(LOW_STRETCH_FRACTION * n_stretches)
    log_offset = np.zeros(n_bins)
    for index in np.flatnonzero(low):
        log_offset[index * stretch : (index + 1) * stretch] = LOW_STRETCH_LOG_RATE
    n_low = int(round(EVENTS_IN_LOW_STRETCHES * n_events))
    spacing = int(MIN_EVENT_SPACING_S * SAMPLING_FREQUENCY)
    margin = int(1.1 * SAMPLING_FREQUENCY)
    onsets: list[int] = []
    for want_low, count in ((True, n_low), (False, n_events - n_low)):
        pool = np.flatnonzero(low == want_low)
        placed = 0
        while placed < count:
            index = int(rng.choice(pool))
            onset = int(
                rng.integers(index * stretch + margin, (index + 1) * stretch - margin)
            )
            if all(abs(onset - other) >= spacing for other in onsets):
                onsets.append(onset)
                placed += 1
    return log_offset, np.sort(np.asarray(onsets))


def simulate_and_fit(
    *,
    window: str,
    scenario: str,
    shape: str,
    gain: float,
    onset_ms: float,
    n_bins: int,
    n_events: int,
    n_units: int,
    rate_hz: float,
    sigma: float,
    seed: int,
) -> dict:
    """One scenario: plant, simulate, fit, and read every rule
    (:func:`~continuum_swr_replay.analysis.pfc.timing.timing_rule_calibration`)."""
    rng = np.random.default_rng(seed)
    if scenario == "response":
        log_rate = np.zeros(n_bins)
        onsets = event_onsets(n_bins, n_events, rng)
    else:
        log_rate, onsets = stretch_offset_and_onsets(n_bins, n_events, rng)
    planted_gain = 0.0 if scenario == "stop_offset" else gain
    truth_lags = np.arange(
        -int(1.1 * SAMPLING_FREQUENCY), int(1.1 * SAMPLING_FREQUENCY)
    )
    truth = planted_gain * response_shape(
        shape, truth_lags / SAMPLING_FREQUENCY, onset_ms / 1e3
    )
    for onset in onsets:
        log_rate[onset + truth_lags] += truth
    mean = rate_hz / SAMPLING_FREQUENCY * np.exp(log_rate)
    counts = rng.poisson(mean[:, None] * np.ones((1, n_units))).astype(float)

    fit_half, read_half = WINDOWS[window]
    lags = symmetric_lags(fit_half, SAMPLING_FREQUENCY)
    basis, lag_time, _ = event_basis_functions(lags, SAMPLING_FREQUENCY)
    indicator = np.zeros(n_bins)
    indicator[onsets] = 1.0
    design = np.concatenate(
        [np.ones((n_bins, 1)), convolve_event_basis(indicator, basis, lags)], axis=1
    )
    fit = fit_poisson_glms(
        design,
        counts,
        penalty=make_ridge_penalty(design.shape[1], sigma),
        compute_covariance=False,
    )
    report = in_report_window(lag_time, read_half)
    report_lags = lag_time[report]
    curves = basis[report] @ np.asarray(fit.params)[:, 1:].T  # (n_lags, n_units)
    truth_curve = (
        planted_gain * response_shape(shape, report_lags, onset_ms / 1e3)
        if planted_gain != 0.0
        else None
    )
    calibration = timing_rule_calibration(
        curves,
        report_lags,
        truth_curve,
        reference_lags=REFERENCE_LAGS_S if window == "wide" else None,
    )
    return {
        "window": window,
        "scenario": scenario,
        "shape": shape,
        "gain": float(planted_gain),
        "true_onset_ms": float(onset_ms),
        "n_units": int(n_units),
        "fraction_converged": float(np.mean(np.asarray(fit.converged))),
        **calibration,
    }


def scenario_list(quick: bool) -> list[dict]:
    """The calibration's scenarios, as keyword arguments of :func:`simulate_and_fit`.

    Parameters
    ----------
    quick : bool
        Only the canonical-window boxcar increase starting at 0 and +50 ms.

    Returns
    -------
    list of dict
        ``window``, ``scenario``, ``shape``, ``gain`` and ``onset_ms`` of each:
        every window x shape x gain x true onset of the ``response`` scenario,
        then, per window, ``stop_offset`` and ``stop_offset_plus_response``
        (a -0.7 boxcar from 0 ms).
    """
    if quick:
        return [
            {
                "window": "canonical",
                "scenario": "response",
                "shape": "boxcar",
                "gain": 0.7,
                "onset_ms": onset,
            }
            for onset in (0.0, 50.0)
        ]
    scenarios = [
        {
            "window": window,
            "scenario": "response",
            "shape": shape,
            "gain": gain,
            "onset_ms": onset,
        }
        for window in WINDOWS
        for shape in SHAPES
        for gain in GAINS
        for onset in TRUE_ONSETS_MS
    ]
    scenarios += [
        {
            "window": window,
            "scenario": scenario,
            "shape": "boxcar",
            "gain": -0.7,
            "onset_ms": 0.0,
        }
        for window in WINDOWS
        for scenario in ("stop_offset", "stop_offset_plus_response")
    ]
    return scenarios
