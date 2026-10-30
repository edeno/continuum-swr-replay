"""The duration-aware simulation check at the real duration spread. Simulation only.

``event_durations.json`` puts the non-local streams' durations at medians 94 ms
(with SWR) and 48 ms (without), log-SD 0.67 and 0.70: wider than the 0.45 the
2026-10-26 simulation used. This reruns its four scenarios with those
durations (4,000 stops each, both response streams, both designs) and writes
``real_spread_check.json`` next to this file.

Usage::

    uv run python .claude/docs/reviews/2026-10-27-duration-aware-history-refit/run_real_spread_check.py
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from continuum_swr_replay.analysis.replay.duration_history_simulation import (
    HISTORY_BLOCKS,
    SCENARIOS,
    STREAMS,
    fit_history_direction,
    simulate_interval_streams,
    summarize_direction,
    true_lag_kernel,
)

OUT = Path(__file__).resolve().parent
N_STOPS = 4000
MEDIANS = {"swr": 0.094, "no_swr": 0.048}
SIGMAS = {"swr": 0.67, "no_swr": 0.70}
MIN_SUPPORT = 10.0


def _one(item: tuple[int, str]) -> tuple[str, dict]:
    warnings.filterwarnings("ignore")
    seed, name = item
    sc = dataclasses.replace(
        SCENARIOS[name], duration_median_s=MEDIANS, duration_sigma=SIGMAS
    )
    stops, events = simulate_interval_streams(sc, n_stops=N_STOPS, seed=seed)
    rec = {
        "n_events": events["stream"].value_counts().to_dict(),
        "duration_log_sd": events.groupby("stream")["duration_s"]
        .apply(lambda d: float(np.log(d).std()))
        .to_dict(),
    }
    for response in STREAMS:
        for aware in (False, True):
            r = fit_history_direction(
                stops, events, response=response, duration_aware=aware
            )
            s = summarize_direction(r)
            peaks = {}
            for block, kernel in r["kernels"][r["best"]].items():
                _, anchor, rel = HISTORY_BLOCKS[block]
                ok = r["support"][block] >= MIN_SUPPORT
                if not ok.any():
                    continue
                fitted = np.where(ok, kernel, np.nan)
                lag = int(np.nanargmax(fitted))
                source = response if rel == "self" else r["cross"]
                truth = true_lag_kernel(
                    sc, target=response, source=source, anchor=anchor
                )
                peaks[block] = {
                    "peak_lag_ms": (lag + 1) * 15,
                    "peak_rr": float(np.exp(fitted[lag])),
                    "truth_peak_rr": float(np.exp(truth.max())),
                    "truth_peak_lag_ms": int(np.argmax(truth) + 1) * 15,
                }
            if not aware:
                k = r["kernels"]["self+cross"]
                peaks["current_self_onset_rr_by_lag"] = (
                    np.exp(k["self_onset"]).round(3).tolist()
                )
                peaks["current_cross_onset_rr_by_lag"] = (
                    np.exp(k["cross_onset"]).round(3).tolist()
                )
            rec[f"{response}/{'duration_aware' if aware else 'current'}"] = {
                **{
                    k: (round(v, 2) if isinstance(v, float) else v)
                    for k, v in s.items()
                },
                "collinearity": r["collinearity"],
                "best_kernels": peaks,
            }
    return name, rec


def main() -> None:
    items = [(3000 + i, name) for i, name in enumerate(SCENARIOS)]
    with ProcessPoolExecutor(max_workers=len(items)) as pool:
        out = dict(pool.map(_one, items))
    result = {
        "n_stops": N_STOPS,
        "medians_s": MEDIANS,
        "log_sd": SIGMAS,
        "scenarios": out,
    }
    (OUT / "real_spread_check.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
