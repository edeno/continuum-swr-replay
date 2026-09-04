#!/usr/bin/env python
"""Does the amount of non-local activity vary by date within each animal?

Premise to test: SWRs decrease over learning. If non-local events ride on SWRs,
non-local *rate* might fall over days too — and the non-SWR fraction (Fig 2b)
might drift. We compute, per session, immobile recording time, the immobile
non-local event rate, the SWR rate, and the non-SWR fraction, then look at the
trend vs ordinal date *within each animal* (Spearman, robust to spacing).

Rates use immobile minutes as the denominator (events/SWRs are immobility
phenomena), so a longer session does not inflate the count.

Writes per-session CSV + per-animal trend summary to
.claude/research/fig2b_outputs/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from continuum_swr_replay.analysis.hippocampus import (
    build_classified_nonlocal_event_table,
    load_data_over_time,
)
from continuum_swr_replay.analysis.hippocampus.event_immobility import (
    filter_event_labels_by_speed,
)
from continuum_swr_replay.analysis.hippocampus.event_table import (
    DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC as SPEED_THR,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.session_keys import SESSION_COLUMNS_DAY

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
OUT = Path(__file__).resolve().parents[2] / ".claude/research/fig2b_outputs"
OUT.mkdir(parents=True, exist_ok=True)
SESSION = list(SESSION_COLUMNS_DAY)


def main() -> None:
    logging.info("Loading data_over_time ...")
    dot = load_data_over_time(PROCESSED_DATA_DIR)
    dt = float(np.median(np.diff(dot.index.values[:5000])))  # ~2 ms
    logging.info("bin dt = %.4f s; threshold = %g cm/s", dt, SPEED_THR)

    logging.info("Building classified event table ...")
    tbl = build_classified_nonlocal_event_table(dot)
    immob = tbl[tbl["whole_event_speed_eligible"].astype(bool)]

    # per-session non-local immobile counts + non_swr fraction
    nl = (
        immob.groupby(SESSION)
        .agg(
            n_nonlocal=("event_id", "size"),
            n_non_swr=("overlap_class", lambda s: int((s == "non_swr").sum())),
        )
        .reset_index()
    )
    nl["f_nonswr"] = nl["n_non_swr"] / nl["n_nonlocal"]

    # per-session durations + SWR counts straight from data_over_time
    logging.info("Per-session durations and SWR counts ...")
    swr_eligible_dot = filter_event_labels_by_speed(
        dot,
        ["SWR_event"],
        speed_threshold=SPEED_THR,
    )
    recs = []
    for (animal, day, epoch), g in swr_eligible_dot.groupby(SESSION, sort=False):
        immobile_min = float((g["head_speed"] < SPEED_THR).sum()) * dt / 60.0
        total_min = float(len(g)) * dt / 60.0
        # distinct SWRs = unique nonzero SWR_event labels in this session
        swr_labels = g["SWR_event"].to_numpy()
        n_swr = int(np.unique(swr_labels[swr_labels > 0]).size)
        recs.append(
            {"animal": animal, "day": day, "epoch": epoch,
             "total_min": total_min, "immobile_min": immobile_min,
             "n_swr": n_swr}
        )
    dur = pd.DataFrame(recs)

    s = nl.merge(dur, on=SESSION, how="left")
    s["nonlocal_rate"] = s["n_nonlocal"] / s["immobile_min"]      # per immobile min
    s["swr_rate"] = s["n_swr"] / s["immobile_min"]                # per immobile min
    s = s.sort_values(SESSION).reset_index(drop=True)
    s.to_csv(OUT / "per_session_dated.csv", index=False)

    lines = []
    def emit(x=""):
        lines.append(x)
        print(x)

    emit("=" * 76)
    emit("NON-LOCAL AMOUNT vs DATE, PER ANIMAL")
    emit("=" * 76)
    emit(f"{len(s)} sessions. Rates per immobile-minute. Trend = Spearman vs "
         f"ordinal date within animal (sessions same-day share a rank tie via day).")
    emit("")

    metrics = ["nonlocal_rate", "swr_rate", "f_nonswr", "n_nonlocal", "immobile_min"]
    summary = []
    for animal, g in s.groupby("animal"):
        g = g.sort_values(["day", "epoch"])
        # ordinal date index within animal (1 per distinct day)
        day_rank = g["day"].rank(method="dense").to_numpy()
        n_days = int(g["day"].nunique())
        emit(f"--- {animal}  ({len(g)} sessions over {n_days} days: "
             f"{g['day'].min()}..{g['day'].max()}) ---")
        row = {"animal": animal, "n_sessions": len(g), "n_days": n_days}
        for m in metrics:
            rho, p = spearmanr(day_rank, g[m].to_numpy())
            row[f"{m}_rho"] = rho
            row[f"{m}_p"] = p
            first = g[g["day"] == g["day"].min()][m].mean()
            last = g[g["day"] == g["day"].max()][m].mean()
            star = "*" if p < 0.05 else " "
            emit(f"    {m:14s} rho={rho:+.3f} p={p:.3f}{star}  "
                 f"first-day mean={first:8.3f}  last-day mean={last:8.3f}")
        summary.append(row)
        emit("")

    summ = pd.DataFrame(summary)
    summ.to_csv(OUT / "per_animal_date_trends.csv", index=False)

    # pooled-within-animal sign tally for the headline metrics
    emit("=" * 76)
    emit("DIRECTION TALLY across 5 animals (sign of Spearman rho vs date):")
    for m in ["swr_rate", "nonlocal_rate", "f_nonswr"]:
        rhos = summ[f"{m}_rho"]
        sig = summ[f"{m}_p"] < 0.05
        emit(f"    {m:14s}: {int((rhos<0).sum())} down / {int((rhos>0).sum())} up "
             f"({int((sig & (rhos<0)).sum())} sig-down, "
             f"{int((sig & (rhos>0)).sum())} sig-up)  "
             f"rho range [{rhos.min():+.2f}, {rhos.max():+.2f}]")

    (OUT / "date_trends_summary.txt").write_text("\n".join(lines))
    emit("")
    emit(f"Wrote per_session_dated.csv, per_animal_date_trends.csv, "
         f"date_trends_summary.txt to {OUT}")


if __name__ == "__main__":
    main()
