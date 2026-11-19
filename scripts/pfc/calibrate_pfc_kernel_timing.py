"""What the mPFC kernel estimator does to the timing and size of a known response.

The Figure 4 curves are ridge-penalized (sigma = 3) cubic B-spline kernels
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
``timing.timing_rule_calibration``'s, and the simulation, the fit and the
scenarios are ``analysis/pfc/timing_calibration.py``'s; this script runs the
scenarios and writes what they give.

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

from continuum_swr_replay.analysis.pfc.timing_calibration import (
    DEFAULT_N_BINS,
    DEFAULT_N_EVENTS,
    REFERENCE_LAGS_S,
    WINDOWS,
    scenario_list,
    simulate_and_fit,
)
from continuum_swr_replay.parameters import SAMPLING_FREQUENCY


def parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    from continuum_swr_replay.paths import PFC_TIMING_CALIBRATION, processed_data_dir

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
        args.output = processed_data_dir() / PFC_TIMING_CALIBRATION
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
                "sampling_frequency_hz": float(SAMPLING_FREQUENCY),
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
