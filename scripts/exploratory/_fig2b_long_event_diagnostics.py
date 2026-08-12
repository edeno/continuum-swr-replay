#!/usr/bin/env python
"""Are long non-SWR non-local events real, or a model-specification artifact?

Tests three competing accounts of the long-duration non-SWR tail, per-event,
using what actually drives the posterior during the event window:

  A. genuine non-local trajectory  -> elevated MUA, MAP position MOVES
  B. spike desert / No-Spike should win -> MUA at/below baseline, high No-Spike
     belief competing, few spikes; Non-Local call is prior- not evidence-driven
  C. sticky continuity prior        -> MAP FROZEN (near-zero map_speed, low
     positional variance) riding through low evidence

For each immobile non-local event we aggregate its data_over_time bins and
compare three groups: long non-SWR (>1 s and >0.5 s), short non-SWR (<0.1 s),
and SWR-overlapping. Writes a group summary + per-long-event table.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from continuum_swr_replay.analysis.hippocampus import (
    build_classified_nonlocal_event_table,
    load_data_over_time,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
OUT = Path(__file__).resolve().parents[2] / ".claude/research/fig2b_outputs"
KEYS = ["animal", "day", "epoch", "non_local_event"]


def main() -> None:
    logging.info("Loading data_over_time ...")
    dot = load_data_over_time(PROCESSED_DATA_DIR)
    assert int(dot["non_local_event"].min()) == 0, "expected 0 = no event"

    logging.info("Building classified table ...")
    tbl = build_classified_nonlocal_event_table(dot)
    immob = tbl[tbl["whole_event_speed_eligible"].astype(bool)].copy()

    # ── per-event posterior diagnostics straight from the event bins ──
    logging.info("Aggregating per-event posterior diagnostics ...")
    ev_bins = dot[dot["non_local_event"] > 0].copy()
    # MAP movement: positional spread + per-bin speed of the decoded estimate
    g = ev_bins.groupby(KEYS, sort=False)
    diag = g.agg(
        n_bins=("non_local_event", "size"),
        nospike_mean=("No-Spike_prob", "mean"),
        nospike_max=("No-Spike_prob", "max"),
        local_mean=("Local_prob", "mean"),
        nonlocal_mean=("Non-Local_prob", "mean"),
        frag_mean=("Non-Local_Fragmented_prob", "mean"),
        cont_mean=("Non-Local_Continuous_prob", "mean"),
        mua_z_mean=("MUA_zscore", "mean"),
        mua_z_max=("MUA_zscore", "max"),
        swr_z_mean=("SWR_zscore", "mean"),
        map_pos_std=("map_estimate", "std"),     # cm; frozen MAP -> ~0
        map_pos_range=("map_estimate", lambda s: float(s.max() - s.min())),
        map_speed_mean=("map_speed", "mean"),
        hpd_mean=("hpd", "mean"),
    ).reset_index()

    m = immob.merge(diag, on=KEYS, how="left", validate="one_to_one")

    nonswr = m["overlap_class"] == "non_swr"
    groups = {
        "non-SWR >1s": m[nonswr & (m["event_duration"] > 1.0)],
        "non-SWR 0.5-1s": m[nonswr & m["event_duration"].between(0.5, 1.0)],
        "non-SWR <0.1s": m[nonswr & (m["event_duration"] < 0.1)],
        "SWR-overlap (all)": m[~nonswr],
    }

    metrics = [
        ("n_bins", "bins"),
        ("mua_z_mean", "MUA z (mean)"),
        ("mua_z_max", "MUA z (max)"),
        ("nospike_mean", "No-Spike prob"),
        ("nospike_max", "No-Spike max"),
        ("map_pos_std", "MAP pos std cm"),
        ("map_speed_mean", "MAP speed"),
        ("hpd_mean", "HPD"),
        ("frag_mean", "Frag prob"),
        ("cont_mean", "Cont prob"),
    ]

    lines = []
    def emit(x=""):
        lines.append(x)
        print(x)

    emit("=" * 92)
    emit("LONG NON-SWR EVENTS: real vs model-spec artifact — per-event posterior diagnostics")
    emit("=" * 92)
    emit("Group sizes: " + ", ".join(f"{k}={len(v)}" for k, v in groups.items()))
    emit("")
    header = f"{'metric':16s}" + "".join(f"{k:>20s}" for k in groups)
    emit(header)
    emit("-" * len(header))
    for col, lbl in metrics:
        row = f"{lbl:16s}"
        for v in groups.values():
            row += f"{v[col].median():>20.3f}"
        emit(row + "   (median)")
    emit("")
    emit("Interpretation guide:")
    emit("  A real        -> long-group MUA z >> 0 and MAP speed/std large (moves)")
    emit("  B spike desert -> long-group MUA z near/below SWR & short groups, "
         "No-Spike max high")
    emit("  C frozen prior -> long-group MAP pos std ~0 and MAP speed ~0 despite "
         "long duration")
    emit("")

    # fraction of long events that look frozen / spike-poor
    long1 = groups["non-SWR >1s"]
    short = groups["non-SWR <0.1s"]
    swr = groups["SWR-overlap (all)"]
    def frac(df, mask):
        return float(mask.mean()) if len(df) else np.nan
    emit(f"Diagnostic flags on the >1s non-SWR events (n={len(long1)}):")
    emit(f"  MUA z_mean < 0 (below-baseline spiking): "
         f"{frac(long1, long1['mua_z_mean'] < 0):.0%}  "
         f"(short non-SWR: {frac(short, short['mua_z_mean'] < 0):.0%}, "
         f"SWR-ovlp: {frac(swr, swr['mua_z_mean'] < 0):.0%})")
    emit(f"  MAP pos std < 2 cm (near-frozen): "
         f"{frac(long1, long1['map_pos_std'] < 2):.0%}  "
         f"(short non-SWR: {frac(short, short['map_pos_std'] < 2):.0%}, "
         f"SWR-ovlp: {frac(swr, swr['map_pos_std'] < 2):.0%})")
    emit(f"  No-Spike max > 0.3 (No-Spike competed): "
         f"{frac(long1, long1['nospike_max'] > 0.3):.0%}  "
         f"(short non-SWR: {frac(short, short['nospike_max'] > 0.3):.0%})")
    emit("")

    # per-long-event detail
    cols = KEYS + ["event_duration", "n_bins", "mua_z_mean", "mua_z_max",
                   "nospike_max", "map_pos_std", "map_speed_mean", "hpd_mean",
                   "frag_mean", "cont_mean", "max_speed"]
    cols = [c for c in cols if c in long1.columns]
    detail = long1.sort_values("event_duration", ascending=False)[cols]
    emit("All >1s non-SWR events:")
    emit(detail.to_string(index=False))
    detail.to_csv(OUT / "long_nonswr_diagnostics.csv", index=False)

    (OUT / "long_event_diagnostics_summary.txt").write_text("\n".join(lines))
    emit(f"\nWrote long_nonswr_diagnostics.csv, long_event_diagnostics_summary.txt to {OUT}")


if __name__ == "__main__":
    main()
