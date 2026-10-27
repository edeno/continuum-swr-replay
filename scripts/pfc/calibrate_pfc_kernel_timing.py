"""How far the shipped mPFC kernel estimator moves a response's onset and peak.

The Figure 4 curves are ridge-penalised (sigma = 3) cubic B-spline kernels
with knots every 50 ms, fit on +/-250 ms and read on +/-200 ms. A smooth
kernel spreads a response into earlier lags, so the half-maximum onset
(``analysis/pfc/timing.py``) of a fitted curve can precede the response's true
start. This script plants responses of known onset in simulated Poisson spike
trains, fits them exactly as ``run_session`` does (same lags, basis, prior and
fitter), and records the apparent onset and peak lag.

Simulated session: 500 Hz bins, ``--n-bins`` fitted bins (default: the median
canonical session's), ``--n-events`` onsets (default: the median per-class
count) at least 0.6 s apart, baseline ``--rate-hz``. Each scenario plants a
log-rate gain ``gain * h(t - onset)`` with ``h`` a 100 ms boxcar or an alpha
function peaking 30 ms after its start, for gains +/-0.7 and true onsets
-50, 0, +25, +50 ms, in ``--n-units`` independent units.

Writes ``<output>`` (JSON): per scenario the true onset and the median and
2.5th/97.5th percentiles of the apparent onset and peak lag, in ms.

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
GAINS = (0.7, -0.7)
TRUE_ONSETS_MS = (-50.0, 0.0, 25.0, 50.0)
BOXCAR_WIDTH_S = 0.100
ALPHA_PEAK_S = 0.030


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
    margin = int(0.3 * SAMPLING_FREQUENCY)
    spacing = int(MIN_EVENT_SPACING_S * SAMPLING_FREQUENCY)
    slack = n_bins - 2 * margin - spacing * (n_events - 1)
    if slack <= 0:
        raise ValueError("too many events for the session length")
    gaps = np.sort(rng.integers(0, slack, size=n_events))
    return margin + gaps + spacing * np.arange(n_events)


def simulate_and_fit(
    shape: str,
    gain: float,
    onset_ms: float,
    *,
    n_bins: int,
    n_events: int,
    n_units: int,
    rate_hz: float,
    sigma: float,
    seed: int,
) -> dict[str, float]:
    """One scenario: plant, simulate, fit, read the timing."""
    from continuum_swr_replay.analysis.pfc.batched_glm import (
        fit_poisson_glms,
        make_ridge_penalty,
    )
    from continuum_swr_replay.analysis.pfc.design import (
        convolve_event_basis,
        event_basis_functions,
        symmetric_lags,
    )
    from continuum_swr_replay.analysis.pfc.event_sets import (
        DEFAULT_BASIS_HALF_WINDOW_S,
    )
    from continuum_swr_replay.analysis.pfc.modulation import REPORT_HALF_WINDOW_S
    from continuum_swr_replay.analysis.pfc.timing import peak_and_onset

    rng = np.random.default_rng(seed)
    onsets = event_onsets(n_bins, n_events, rng)
    indicator = np.zeros(n_bins)
    indicator[onsets] = 1.0

    # The truth: a log-rate modulation that is zero outside the response.
    truth_lags = np.arange(
        -int(0.3 * SAMPLING_FREQUENCY), int(0.3 * SAMPLING_FREQUENCY)
    )
    truth = gain * response_shape(
        shape, truth_lags / SAMPLING_FREQUENCY, onset_ms / 1e3
    )
    log_gain = np.zeros(n_bins)
    for onset in onsets:
        positions = onset + truth_lags
        valid = (positions >= 0) & (positions < n_bins)
        log_gain[positions[valid]] += truth[valid]
    mean = rate_hz / SAMPLING_FREQUENCY * np.exp(log_gain)
    counts = rng.poisson(mean[:, None] * np.ones((1, n_units))).astype(float)

    lags = symmetric_lags(DEFAULT_BASIS_HALF_WINDOW_S, SAMPLING_FREQUENCY)
    basis, lag_time, _ = event_basis_functions(lags, SAMPLING_FREQUENCY)
    design = np.concatenate(
        [np.ones((n_bins, 1)), convolve_event_basis(indicator, basis, lags)], axis=1
    )
    fit = fit_poisson_glms(
        design,
        counts,
        penalty=make_ridge_penalty(design.shape[1], sigma),
        compute_covariance=False,
    )
    params = np.asarray(fit.params)[:, 1:]
    report = np.abs(lag_time) <= REPORT_HALF_WINDOW_S + 1e-12
    curves = basis[report] @ params.T  # (n_report_lags, n_units)
    timing = peak_and_onset(curves, lag_time[report])

    onset_ms_apparent = timing.onset_lag * 1e3
    peak_ms_apparent = timing.peak_lag * 1e3

    def summary(values: NDArray) -> dict[str, float]:
        low, mid, high = np.nanpercentile(values, [2.5, 50, 97.5])
        return {"median": float(mid), "p2_5": float(low), "p97_5": float(high)}

    return {
        "shape": shape,
        "gain": float(gain),
        "true_onset_ms": float(onset_ms),
        "true_peak_ms": float(
            onset_ms
            + (ALPHA_PEAK_S * 1e3 if shape == "alpha" else BOXCAR_WIDTH_S * 5e2)
        ),
        "apparent_onset_ms": summary(onset_ms_apparent),
        "apparent_peak_ms": summary(peak_ms_apparent),
        "onset_bias_ms_median": float(np.nanmedian(onset_ms_apparent) - onset_ms),
        "fraction_onset_at_edge": float(np.mean(timing.onset_at_edge)),
        "fraction_converged": float(np.mean(np.asarray(fit.converged))),
        "fraction_peak_sign_correct": float(
            np.mean(np.sign(timing.peak) == np.sign(gain))
        ),
        "n_units": int(n_units),
    }


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
        help="20 units, the boxcar enhanced scenarios at onsets 0 and +50 ms only",
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
        scenarios = [("boxcar", 0.7, 0.0), ("boxcar", 0.7, 50.0)]
    else:
        scenarios = [
            (shape, gain, onset)
            for shape in SHAPES
            for gain in GAINS
            for onset in TRUE_ONSETS_MS
        ]
    results = []
    for index, (shape, gain, onset) in enumerate(scenarios):
        result = simulate_and_fit(
            shape,
            gain,
            onset,
            n_bins=args.n_bins,
            n_events=args.n_events,
            n_units=args.n_units,
            rate_hz=args.rate_hz,
            sigma=args.penalty_sigma,
            seed=args.seed + index,
        )
        print(
            f"{shape:6s} gain {gain:+.1f} onset {onset:+5.0f} ms -> apparent onset "
            f"{result['apparent_onset_ms']['median']:+6.1f} ms, peak "
            f"{result['apparent_peak_ms']['median']:+6.1f} ms",
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
                "onset_rule": "half-maximum run ending at the peak "
                "(analysis/pfc/timing.py:peak_and_onset)",
                "scenarios": results,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
