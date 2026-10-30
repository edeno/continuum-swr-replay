"""Null replicates for the 2026-10-26 duration-aware history review. Simulation only.

Scenario (a) (no dependence) at the realistic base rates, one long session
(6,000 stops) per seed, both response streams, duration-aware design only.
Records every rung's AIC gain over the history-free rung and which self rung
AIC chose, so the review can state how often the duration-aware design reads
history into independent events. Writes ``null_replicates.json`` next to this
file.

Usage::

    uv run python .claude/docs/reviews/2026-10-26-duration-aware-history-simulation/run_null_replicates.py
"""

from __future__ import annotations

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
SEEDS = tuple(range(11, 21))
N_STOPS = 6000


def _one(seed: int) -> list[dict]:
    warnings.filterwarnings("ignore")
    stops, events = simulate_interval_streams(
        SCENARIOS["a_independent"], n_stops=N_STOPS, seed=seed
    )
    rows = []
    for response in STREAMS:
        r = fit_history_direction(stops, events, response=response, duration_aware=True)
        aic = r["aic"]
        rows.append(
            {
                "seed": seed,
                "response": response,
                "n_events": r["n_events"],
                "gain_over_none": {
                    k: round(aic["none"] - v, 2) for k, v in aic.items()
                },
                **summarize_direction(r),
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
        "delta_aic_cross_max": float(d_cross.max()),
        "delta_aic_cross_n_positive": int((d_cross > 0).sum()),
        "n_best_self_not_none": sum(r["best_self"] != "none" for r in rows),
        "n_best_not_none": sum(r["best"] != "none" for r in rows),
        "max_gain_any_rung": float(
            max(max(r["gain_over_none"].values()) for r in rows)
        ),
    }
    out = {"seeds": SEEDS, "n_stops": N_STOPS, "summary": summary, "fits": rows}
    (OUT / "null_replicates.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
