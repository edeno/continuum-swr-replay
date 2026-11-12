#!/usr/bin/env python
"""Are the long non-SWR non-local events concentrated in certain animals?

For each animal, summarise the duration distribution of immobile non-SWR
non-local events (and compare to SWR-overlapping events), and quantify the long
tail (counts/fractions above 0.5/1/2/5 s) plus where the single longest events
live. Writes a per-animal table, the top-N longest events, and a histogram.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.events.classified_table import (
    build_classified_nonlocal_event_table,
)
from continuum_swr_replay.analysis.hippocampus import load_data_over_time
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.visualization.figure_utilities import save_figure

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)
OUT = Path(__file__).resolve().parents[2] / ".claude/research/fig2b_outputs"
OUT.mkdir(parents=True, exist_ok=True)
THRESHOLDS = [0.5, 1.0, 2.0, 5.0]  # seconds


def main() -> None:
    logging.info("Loading data_over_time ...")
    dot = load_data_over_time(PROCESSED_DATA_DIR)
    tbl = build_classified_nonlocal_event_table(dot)
    immob = tbl[tbl["whole_event_speed_eligible"].astype(bool)].copy()
    nonswr = immob[immob["overlap_class"] == "non_swr"].copy()
    swr = immob[immob["overlap_class"] != "non_swr"].copy()
    d = "event_duration"

    lines = []

    def emit(x=""):
        lines.append(x)
        print(x)

    emit("=" * 84)
    emit("DURATION OF NON-SWR NON-LOCAL EVENTS, PER ANIMAL (immobile)")
    emit("=" * 84)
    emit(f"non-SWR events: {len(nonswr):,}   SWR-overlapping: {len(swr):,}")
    emit(
        f"non-SWR duration (s): median={nonswr[d].median():.3f}  "
        f"p90={nonswr[d].quantile(0.9):.3f}  p99={nonswr[d].quantile(0.99):.3f}  "
        f"max={nonswr[d].max():.3f}"
    )
    emit(
        f"SWR-ovlp duration (s): median={swr[d].median():.3f}  "
        f"p90={swr[d].quantile(0.9):.3f}  p99={swr[d].quantile(0.99):.3f}  "
        f"max={swr[d].max():.3f}"
    )
    emit("(non-SWR events are expected longer: SWRs cap the overlapping ones.)")
    emit("")

    # per-animal duration summary for non-SWR
    rows = []
    for animal, g in nonswr.groupby("animal"):
        r = {
            "animal": animal,
            "n_nonswr": len(g),
            "median_s": g[d].median(),
            "p90_s": g[d].quantile(0.9),
            "p99_s": g[d].quantile(0.99),
            "max_s": g[d].max(),
        }
        for t in THRESHOLDS:
            n = int((g[d] > t).sum())
            r[f"n_gt_{t}s"] = n
            r[f"frac_gt_{t}s"] = n / len(g)
        rows.append(r)
    per_animal = pd.DataFrame(rows).set_index("animal")
    per_animal.to_csv(OUT / "nonswr_durations_per_animal.csv")

    emit("Per-animal non-SWR duration summary:")
    emit(
        f"{'animal':8s} {'n':>7s} {'med':>6s} {'p90':>6s} {'p99':>6s} "
        f"{'max':>7s} {'>1s':>7s} {'>2s':>7s} {'>5s':>6s}"
    )
    for a, r in per_animal.iterrows():
        emit(
            f"{a:8s} {int(r.n_nonswr):7d} {r.median_s:6.3f} {r.p90_s:6.3f} "
            f"{r.p99_s:6.3f} {r.max_s:7.3f} "
            f"{int(r['n_gt_1.0s']):4d}({r['frac_gt_1.0s']:.1%}) "
            f"{int(r['n_gt_2.0s']):4d}({r['frac_gt_2.0s']:.1%}) "
            f"{int(r['n_gt_5.0s']):3d}"
        )
    emit("")

    # where do the absolute-longest non-SWR events live?
    emit("Share of the longest non-SWR events by animal:")
    for t in THRESHOLDS:
        long = nonswr[nonswr[d] > t]
        if len(long) == 0:
            continue
        vc = long["animal"].value_counts()
        share = (vc / len(long) * 100).round(1)
        top = ", ".join(f"{a} {share[a]:.0f}%" for a in vc.index)
        emit(f"  > {t}s: {len(long):4d} events | by animal: {top}")
    emit("")

    # animal share of non-SWR events overall (baseline to compare the tail to)
    base = (nonswr["animal"].value_counts() / len(nonswr) * 100).round(1)
    emit(
        "Baseline animal share of ALL non-SWR events: "
        + ", ".join(f"{a} {base[a]:.0f}%" for a in base.index)
    )
    emit(
        "(compare the >Xs shares above to this baseline: over-share => that "
        "animal carries the long tail)"
    )
    emit("")

    # top 15 longest non-SWR events
    emit("Top 15 longest non-SWR non-local events:")
    cols = [
        "animal",
        "day",
        "epoch",
        "event_duration",
        "peak_nonlocal_prob",
        "animal_max_speed_cm_s",
        "n_swrs_overlapping",
    ]
    cols = [c for c in cols if c in nonswr.columns]
    top = nonswr.sort_values(d, ascending=False).head(15)[cols]
    emit(top.to_string(index=False))
    top.to_csv(OUT / "nonswr_longest_events.csv", index=False)

    # histogram (log-x) per animal
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.logspace(np.log10(nonswr[d].min() + 1e-3), np.log10(nonswr[d].max()), 50)
    for a, g in nonswr.groupby("animal"):
        ax.hist(g[d], bins=bins, histtype="step", lw=1.6, label=a, density=True)
    ax.set_xscale("log")
    ax.set_xlabel("non-SWR non-local event duration (s)")
    ax.set_ylabel("density")
    ax.set_title("Non-SWR non-local event durations by animal")
    ax.legend()
    fig.tight_layout()
    save_figure(
        fig,
        "nonswr_durations",
        base_dir=OUT,
        dpi=140,
        formats=["png"],
        transparent=False,
    )

    (OUT / "nonswr_durations_summary.txt").write_text("\n".join(lines))
    emit(
        f"\nWrote nonswr_durations_per_animal.csv, nonswr_longest_events.csv, "
        f"nonswr_durations.png, nonswr_durations_summary.txt to {OUT}"
    )


if __name__ == "__main__":
    main()
