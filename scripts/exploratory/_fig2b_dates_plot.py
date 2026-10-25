#!/usr/bin/env python
"""Diagnostic plot: non-local rate, SWR rate, non-SWR fraction vs date/animal."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from continuum_swr_replay.figure_utilities import save_figure

OUT = Path(__file__).resolve().parents[2] / ".claude/research/fig2b_outputs"
s = pd.read_csv(OUT / "per_session_dated.csv")
animals = sorted(s["animal"].unique())
metrics = [("nonlocal_rate", "non-local /immob-min"),
           ("swr_rate", "SWR /immob-min"),
           ("f_nonswr", "non-SWR fraction")]

fig, axes = plt.subplots(3, len(animals), figsize=(3.0 * len(animals), 8),
                         sharex="col")
for j, animal in enumerate(animals):
    g = s[s["animal"] == animal].copy()
    g["di"] = g["day"].rank(method="dense").astype(int)
    dm = g.groupby("di").mean(numeric_only=True)
    for i, (m, lbl) in enumerate(metrics):
        ax = axes[i, j]
        ax.scatter(g["di"], g[m], s=14, alpha=0.45, color="0.5")
        ax.plot(dm.index, dm[m], "-o", color="C3", ms=4, lw=1.5)
        if m == "f_nonswr":
            ax.axhline(0.5, ls=":", color="k", lw=0.8)
            ax.set_ylim(0, 1)
        if i == 0:
            ax.set_title(f"{animal}\n({g['day'].nunique()} days)")
        if j == 0:
            ax.set_ylabel(lbl)
        if i == 2:
            ax.set_xlabel("day index")
fig.suptitle("Non-local activity & SWR rate over dates, per animal "
             "(grey=session, red=day-mean)", y=1.0)
fig.tight_layout()
save_figure(fig, "dates_trends", base_dir=OUT, dpi=140, formats=["png"], transparent=False)
print("wrote", OUT / "dates_trends.png")
