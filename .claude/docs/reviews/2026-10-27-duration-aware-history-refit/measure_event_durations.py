"""Spread of real event durations, per class and animal (read-only).

For the duration-aware history refit: the simulation's identifiability depends
on the log-SD of event durations (``2026-10-26-duration-aware-history-
simulation.md``, "Identifiability limits"). Classes:

- ``swr``: the Kay SWRs in the onset table (analysed stops), durations from the
  SWR inventory matched on start time;
- ``non_local_swr`` / ``non_local_no_swr``: ``event_table`` events split by
  ``is_swr`` (``event_duration_s``).

Also the gap from a non-local event's end to the next non-local onset in the
same stop (fragmentation shows as an excess of short gaps). Writes
``event_durations.json`` next to this file.

Usage::

    CSR_PROCESSED_DATA_DIR=/cumulus/edeno/continuum-swr-replay/Processed-Data \\
        uv run python .claude/docs/reviews/2026-10-27-duration-aware-history-refit/measure_event_durations.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.events.swr_source import (
    read_swr_inventory,
    swr_epoch_id_of_session,
    swr_source_name,
)
from continuum_swr_replay.paths import table_path
from continuum_swr_replay.tables.io import read_table
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

OUT = Path(__file__).resolve().parent


def _spread(d: np.ndarray) -> dict:
    log_d = np.log(d)
    q25, q75 = np.percentile(log_d, [25, 75])
    return {
        "n": int(len(d)),
        "median_ms": float(np.median(d) * 1000),
        "log_sd": float(log_d.std(ddof=1)),
        "log_sd_iqr": float((q75 - q25) / 1.349),
        "p05_ms": float(np.percentile(d, 5) * 1000),
        "p95_ms": float(np.percentile(d, 95) * 1000),
    }


def _swr_durations(onsets: pd.DataFrame) -> pd.DataFrame:
    swr = onsets[onsets["event_type"] == "SWR"]
    rows = []
    for keys, g in swr.groupby(list(SESSION_COLUMNS), sort=False):
        inv = read_swr_inventory(swr_epoch_id_of_session(*keys))
        starts = inv["start_time"].to_numpy(dtype=float)
        dur = (inv["end_time"] - inv["start_time"]).to_numpy(dtype=float)
        idx = np.searchsorted(starts, g["onset_time"].to_numpy(dtype=float))
        idx = np.clip(idx, 0, len(starts) - 1)
        # nearest start on either side
        left = np.clip(idx - 1, 0, len(starts) - 1)
        t = g["onset_time"].to_numpy(dtype=float)
        pick = np.where(np.abs(starts[left] - t) < np.abs(starts[idx] - t), left, idx)
        err = np.abs(starts[pick] - t)
        rows.append(
            pd.DataFrame(
                {"animal": keys[0], "duration_s": dur[pick], "match_err_s": err}
            )
        )
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    onsets = read_table(table_path("event_onset_table"))
    events = read_table(table_path("event_table"))
    swr = _swr_durations(onsets)
    max_err = float(swr["match_err_s"].max())
    classes = {
        "swr": swr[["animal", "duration_s"]],
        "non_local_swr": events.loc[
            events["is_swr"].eq(True), ["animal", "event_duration_s"]
        ].rename(columns={"event_duration_s": "duration_s"}),
        "non_local_no_swr": events.loc[
            events["is_swr"].eq(False), ["animal", "event_duration_s"]
        ].rename(columns={"event_duration_s": "duration_s"}),
    }
    out: dict = {
        "swr_source": swr_source_name(),
        "swr_start_match_max_error_s": max_err,
        "classes": {},
    }
    for name, df in classes.items():
        d = df["duration_s"].to_numpy(dtype=float)
        out["classes"][name] = {
            "all": _spread(d[d > 0]),
            "per_animal": {
                a: _spread(g["duration_s"].to_numpy(dtype=float))
                for a, g in df[df["duration_s"] > 0].groupby("animal")
            },
        }
    # gaps: a non-local event's end to the next non-local onset in its stop
    ev = events.sort_values([*SESSION_COLUMNS, "stop_id", "event_start_s"])
    same = (ev["stop_id"].to_numpy()[1:] == ev["stop_id"].to_numpy()[:-1]) & (
        ev["epoch"].to_numpy()[1:] == ev["epoch"].to_numpy()[:-1]
    )
    gaps = (ev["event_start_s"].to_numpy()[1:] - ev["event_end_s"].to_numpy()[:-1])[
        same
    ]
    edges_ms = [0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 250, 500, 1000, np.inf]
    counts, _ = np.histogram(gaps * 1000, bins=edges_ms)
    out["non_local_gaps"] = {
        "n": int(len(gaps)),
        "n_negative": int((gaps < 0).sum()),
        "min_ms": float(gaps.min() * 1000),
        "median_ms": float(np.median(gaps) * 1000),
        "pct_below_60_ms": float(100 * np.mean(gaps < 0.06)),
        "pct_below_150_ms": float(100 * np.mean(gaps < 0.15)),
        "histogram_edges_ms": [float(e) for e in edges_ms],
        "histogram_counts": counts.tolist(),
    }
    (OUT / "event_durations.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
