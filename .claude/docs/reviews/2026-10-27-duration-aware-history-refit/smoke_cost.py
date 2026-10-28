"""Smoke test and cost estimate for the duration-aware history refit (no writes
to the model directory).

Fits ``_fit_one_session`` of ``scripts/rate/fit_duration_aware_history_rate.py``
on three at-well sessions (fewest, median and most stops), records wall time
and peak RSS, and extrapolates to the cohort by a least-squares line of time on
the session's stop count. Writes ``smoke_cost.json`` next to this file.

Usage::

    CSR_PROCESSED_DATA_DIR=/cumulus/edeno/continuum-swr-replay/Processed-Data \\
        uv run python .claude/docs/reviews/2026-10-27-duration-aware-history-refit/smoke_cost.py
"""

from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import numpy as np

from continuum_swr_replay.analysis.events.cohort_selection import select_stop_cohort
from continuum_swr_replay.analysis.replay import attach_is_swr_to_onsets
from continuum_swr_replay.paths import table_path
from continuum_swr_replay.tables.io import read_table
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS
from scripts.rate.fit_duration_aware_history_rate import (
    _fit_one_session,
    attach_event_ends,
)

OUT = Path(__file__).resolve().parent


def _rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def main() -> None:
    t0 = time.perf_counter()
    stop = read_table(table_path("stop_table"))
    onset_table = read_table(table_path("event_onset_table"))
    event_table = read_table(table_path("event_table"))
    onsets = attach_event_ends(
        attach_is_swr_to_onsets(onset_table, event_table), event_table
    )
    t_prep = time.perf_counter() - t0
    stop, _ = select_stop_cohort(stop, cohort="at_well")
    n_stops = stop.groupby(list(SESSION_COLUMNS)).size().sort_values()
    rss_loaded = _rss_gb()
    picks = [n_stops.index[0], n_stops.index[len(n_stops) // 2], n_stops.index[-1]]
    onset_groups = dict(tuple(onsets.groupby(list(SESSION_COLUMNS), sort=False)))
    stop_groups = dict(tuple(stop.groupby(list(SESSION_COLUMNS), sort=False)))
    sessions = []
    for keys in picks:
        t = time.perf_counter()
        _, rows, _ = _fit_one_session(keys, stop_groups[keys], onset_groups[keys])
        elapsed = time.perf_counter() - t
        ok = sum(r["fit_ok"] for r in rows)
        sessions.append(
            {
                "session": [str(k) for k in keys],
                "n_stops": int(n_stops[keys]),
                "n_fits": len(rows),
                "n_fits_ok": int(ok),
                "seconds": round(elapsed, 1),
                "peak_rss_gb_after": round(_rss_gb(), 2),
                "n_onsets_not_at_risk": {
                    r["outcome"]: r["n_onsets_not_at_risk"]
                    for r in rows
                    if r["model"] == "Mnohist_dur"
                },
                "canonical_corr_self": {
                    r["outcome"]: round(r.get("canonical_corr_self", np.nan), 3)
                    for r in rows
                    if r["model"] == "Mnohist_dur"
                },
            }
        )
        print(sessions[-1], flush=True)
    x = np.array([s["n_stops"] for s in sessions], dtype=float)
    y = np.array([s["seconds"] for s in sessions], dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    total_s = float(np.sum(np.maximum(intercept + slope * n_stops.to_numpy(), 0)))
    out = {
        "n_sessions_cohort": int(len(n_stops)),
        "n_stops_cohort": int(n_stops.sum()),
        "prep_seconds": round(t_prep, 1),
        "peak_rss_gb_after_load": round(rss_loaded, 2),
        "sessions": sessions,
        "seconds_per_stop": float(slope),
        "seconds_intercept": float(intercept),
        "cpu_hours_single_worker": round(total_s / 3600, 2),
    }
    (OUT / "smoke_cost.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "sessions"}, indent=1))


if __name__ == "__main__":
    main()
