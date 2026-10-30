"""Null replicates at the real duration spread. Simulation only.

Scenario (a) (no dependence) with the real durations of
``event_durations.json`` (medians 94 / 48 ms, log-SD 0.67 / 0.70), 10 seeds of
4,000 stops, both response streams, duration-aware design. Records the
fixed-block gains, so the review can say how often the design reads history
into independent events at the real spread. Writes ``real_spread_null.json``.

Usage::

    uv run python .claude/docs/reviews/2026-10-27-duration-aware-history-refit/run_real_spread_null.py
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from continuum_swr_replay.analysis.replay.duration_history_simulation import (
    SCENARIOS,
    STREAMS,
    fit_history_direction,
    simulate_interval_streams,
    summarize_direction,
)

OUT = Path(__file__).resolve().parent
SEEDS = tuple(range(4000, 4010))
N_STOPS = 4000
MEDIANS = {"swr": 0.094, "no_swr": 0.048}
SIGMAS = {"swr": 0.67, "no_swr": 0.70}


def _one(seed: int) -> list[dict]:
    warnings.filterwarnings("ignore")
    sc = dataclasses.replace(
        SCENARIOS["a_independent"], duration_median_s=MEDIANS, duration_sigma=SIGMAS
    )
    stops, events = simulate_interval_streams(sc, n_stops=N_STOPS, seed=seed)
    rows = []
    for response in STREAMS:
        r = fit_history_direction(stops, events, response=response, duration_aware=True)
        s = summarize_direction(r)
        rows.append(
            {
                "seed": seed,
                "response": response,
                **{
                    k: (round(v, 2) if isinstance(v, float) else v)
                    for k, v in s.items()
                },
            }
        )
    return rows


def main() -> None:
    with ProcessPoolExecutor(max_workers=len(SEEDS)) as pool:
        rows = [row for rows in pool.map(_one, SEEDS) for row in rows]
    d_self = np.array([r["delta_aic_self"] for r in rows])
    d_cross = np.array([r["delta_aic_cross"] for r in rows])
    summary = {
        "n_fits": len(rows),
        "delta_aic_self_max": float(d_self.max()),
        "delta_aic_self_n_positive": int((d_self > 0).sum()),
        "delta_aic_self_n_above_4": int((d_self > 4).sum()),
        "delta_aic_cross_max": float(d_cross.max()),
        "delta_aic_cross_n_positive": int((d_cross > 0).sum()),
    }
    out = {"seeds": SEEDS, "n_stops": N_STOPS, "summary": summary, "fits": rows}
    (OUT / "real_spread_null.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
