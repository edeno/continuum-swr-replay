"""What the mPFC kernel estimator does to the timing and size of a known response.

The Figure 4 curves are ridge-penalised (sigma = 3) cubic B-spline kernels
with knots every 50 ms, fit on +/-250 ms and read on +/-200 ms. This script
plants responses of known onset and size in simulated Poisson units, fits them
as ``run_session`` does (same knot spacing, prior and fitter), and reads them
with every rule in ``analysis/pfc/timing.py``:

- onsets: half-maximum from zero (``onset_zero``), half way from the -200 ms
  value (``onset_rise``), half way from the mean over -1.0 to -0.6 s
  (``onset_reference``, wide window only), and the steepest approach to the
  peak (``steepest``);
- the peak lag (``peak``);
- sizes: the peak itself (``size_zero``), the peak less the value at the
  window's first lag (``size_rise``), the peak less the -1.0 to -0.6 s mean
  (``size_reference``, wide window only).

Each rule is read on every unit (median over units) and on the population
curve (the mean of the curves signed by their own peaks), and compared with
the same rule applied to the planted truth on the same lags, so a bias is the
estimator's displacement and not the shape's own rise. Those statistics are
``timing.timing_rule_calibration``'s; this script simulates and fits.

Scenarios, crossed over two windows (``canonical``: fit +/-0.25 s, read
+/-0.2 s; ``wide``: fit +/-1.05 s, read +/-1.0 s):

- ``response``: a 100 ms boxcar or an alpha function peaking 30 ms after its
  start; log-rate gains +0.7, -0.7 and -0.3 (a shallow decrease); true starts
  -50, 0 and +50 ms;
- ``stop_offset``: no event-locked response at all, but the session is cut
  into 20 s stretches, 40 % of them at a log rate 0.3 lower, and 80 % of the
  events fall in those lower stretches (events occur where the unit fires
  less);
- ``stop_offset_plus_response``: the same, plus a -0.7 boxcar from 0 ms.

Simulated session: 500 Hz bins, ``--n-bins`` bins (default: the median
canonical session's fitted bins), ``--n-events`` onsets (default: the median
per-class count) at least 0.6 s apart, baseline ``--rate-hz``,
``--n-units`` independent units per scenario.

Limits: every bin is fit (no immobility mask), one event class is fit (the
shipped curves come from two-class models), there is one baseline rate and
one event count, and units are not selected by a held-out test.

Writes ``<output>`` (JSON): per scenario its settings, the truth read by each
rule, and each rule's unit median, 2.5th/97.5th unit percentiles, population
value and bias, plus the share of units whose apparent onset (zero rule) or
peak lag is negative.

Usage
-----
JAX_PLATFORMS=cpu uv run python scripts/pfc/calibrate_pfc_kernel_timing.py
JAX_PLATFORMS=cpu uv run python scripts/pfc/calibrate_pfc_kernel_timing.py --quick \
    --output /tmp/timing_calibration.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

DEFAULT_N_BINS = 267_000
DEFAULT_N_EVENTS = 100
SAMPLING_FREQUENCY = 500.0
MIN_EVENT_SPACING_S = 0.6
SHAPES = ("boxcar", "alpha")
GAINS = (0.7, -0.7, -0.3)
TRUE_ONSETS_MS = (-50.0, 0.0, 50.0)
BOXCAR_WIDTH_S = 0.100
ALPHA_PEAK_S = 0.030

#: Fit and read half-windows, in seconds.
WINDOWS = {"canonical": (0.25, 0.20), "wide": (1.05, 1.00)}
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
    from continuum_swr_replay.analysis.pfc.batched_glm import (
        fit_poisson_glms,
        make_ridge_penalty,
    )
    from continuum_swr_replay.analysis.pfc.design import (
        convolve_event_basis,
        event_basis_functions,
        symmetric_lags,
    )
    from continuum_swr_replay.analysis.pfc.timing import timing_rule_calibration

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
    report = np.abs(lag_time) <= read_half + 1e-12
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


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    from continuum_swr_replay.paths import processed_data_dir

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON to write (default: <processed data>/pfc_response_followups/"
        "timing_calibration.json)",
    )
    parser.add_argument("--n-bins", type=int, default=DEFAULT_N_BINS)
    parser.add_argument("--n-events", type=int, default=DEFAULT_N_EVENTS)
    parser.add_argument("--n-units", type=int, default=200)
    parser.add_argument("--rate-hz", type=float, default=5.0)
    parser.add_argument("--penalty-sigma", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="20 units; the canonical-window boxcar increase at 0 and +50 ms only",
    )
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = (
            processed_data_dir() / "pfc_response_followups" / "timing_calibration.json"
        )
    return args


def main(argv: Sequence[str] | None = None) -> None:
    if "cuda" in os.environ.get("JAX_PLATFORMS", "cuda"):
        sys.exit("run on CPU: set JAX_PLATFORMS=cpu (the fits are small)")
    args = parse_arguments(argv)
    from continuum_swr_replay.runs.provenance import described_commit

    if args.quick:
        args.n_units = 20
    results = []
    for index, scenario in enumerate(scenario_list(args.quick)):
        result = simulate_and_fit(
            **scenario,
            n_bins=args.n_bins,
            n_events=args.n_events,
            n_units=args.n_units,
            rate_hz=args.rate_hz,
            sigma=args.penalty_sigma,
            seed=args.seed + index,
        )
        rules = result["rules"]
        print(
            f"{scenario['window']:9s} {scenario['scenario']:26s} {scenario['shape']:6s} "
            f"gain {scenario['gain']:+.1f} start {scenario['onset_ms']:+4.0f} ms -> "
            f"population onset_rise {rules['onset_rise']['population']:+6.0f}, "
            f"steepest {rules['steepest']['population']:+6.0f}, "
            f"peak {rules['peak']['population']:+6.0f} ms",
            flush=True,
        )
        results.append(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "written_at": datetime.now(UTC).isoformat(),
                "commit": described_commit(),
                "arguments": {
                    key: (str(value) if isinstance(value, Path) else value)
                    for key, value in vars(args).items()
                },
                "sampling_frequency_hz": SAMPLING_FREQUENCY,
                "windows_s": {
                    k: {"fit": f, "read": r} for k, (f, r) in WINDOWS.items()
                },
                "reference_lags_s": list(REFERENCE_LAGS_S),
                "scenarios": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
