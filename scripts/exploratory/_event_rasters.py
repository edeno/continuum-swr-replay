#!/usr/bin/env python
"""Basic event-train rasters + PSTH + autocorrelogram (the point-process triptych).

Stop-onset-aligned rasters (rows = stops sorted by immobility duration, x =
time-since-stop-onset, one tick per event onset), the column-mean rate (PSTH,
survival-corrected exposure), and the within-stop inter-onset-interval histogram
(autocorrelogram view: refractoriness + burstiness). For SWR vs non-SWR non-local.
"""

from __future__ import annotations

import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
from continuum_swr_replay.figure_utilities import save_figure  # noqa: E402
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR as P  # noqa: E402
from continuum_swr_replay.paths import (  # noqa: E402
    REPLAY_CONTENT_DIR,  # noqa: E402
    table_path,
)
from continuum_swr_replay.tables.covariate_registry import (  # noqa: E402
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table  # noqa: E402

RC = REPLAY_CONTENT_DIR
XMAX = 8.0
N_ROWS = 1500  # stops sampled for the raster


def main() -> None:
    onset = read_table(table_path("event_onset_table", RC))
    ev = read_table(table_path("event_table", RC))
    st = read_table(table_path("stop_table", RC))
    require_covariate_schema(onset, table="event_onset")
    require_covariate_schema(ev, table="event")
    require_covariate_schema(st, table="stop")
    ev = ev[["stop_id", "event_start_s", "is_swr"]].copy()
    ev["rank"] = ev.sort_values("event_start_s").groupby("stop_id").cumcount()
    nl = onset[onset.event_type == "non_local"].copy()
    nl["rank"] = nl.sort_values("onset_time").groupby("stop_id").cumcount()
    nl = nl.merge(ev[["stop_id", "rank", "is_swr"]], on=["stop_id", "rank"], how="left")

    pops = {
        "SWR": onset[onset.event_type == "SWR"],
        "non-local NO SWR": nl[nl.is_swr.eq(False)],
    }
    st = st[["stop_id", "immobility_duration_s"]]
    dur = st.set_index("stop_id")["immobility_duration_s"]

    # Sample stops that have >=1 onset of either population, sorted by duration.
    have = pd.Index(pops["SWR"].stop_id.unique()).union(pops["non-local NO SWR"].stop_id.unique())
    cand = dur.loc[dur.index.isin(have) & (dur > 0)].sort_values(ascending=False)
    if len(cand) > N_ROWS:
        cand = cand.iloc[:: max(1, len(cand) // N_ROWS)]
    row_of = pd.Series(np.arange(len(cand)), index=cand.index)  # rank 0 = longest

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {"SWR": "k", "non-local NO SWR": "C0"}

    # --- rasters (top row), same stops/rows in both ---
    for ax, (name, src) in zip(axes[0], pops.items(), strict=False):
        s = src[src.stop_id.isin(row_of.index) & (src.t_since_stop_onset < XMAX)]
        y = row_of.loc[s.stop_id].to_numpy()
        ax.scatter(s.t_since_stop_onset, y, s=1.5, color=colors[name], alpha=0.5, marker="|")
        ax.plot(cand.clip(upper=XMAX).to_numpy(), np.arange(len(cand)), color="C3", lw=1, label="stop ends (duration)")
        ax.set_xlim(0, XMAX)
        ax.set_ylim(len(cand), 0)
        ax.set_xlabel("time since stop onset (s)")
        ax.set_ylabel("stop (sorted by duration →)")
        ax.set_title(f"Raster: {name}  ({len(s):,} onsets)")
        ax.legend(loc="lower right", fontsize=8, frameon=False)

    # --- PSTH (bottom-left): survival-corrected rate vs time-since-stop ---
    tb = np.arange(0, XMAX + 0.1, 0.1)
    tc = 0.5 * (tb[:-1] + tb[1:])
    durv = dur[dur > 0].to_numpy()
    expo = np.array([(durv > t).sum() for t in tc]) * 0.1  # stop-seconds available per bin
    axp = axes[1, 0]
    for name, src in pops.items():
        h, _ = np.histogram(src.t_since_stop_onset.to_numpy(), tb)
        axp.plot(tc, h / np.where(expo > 0, expo, np.nan), color=colors[name], lw=2, label=name)
    axp.set_xlabel("time since stop onset (s)")
    axp.set_ylabel("onset rate (events / s)")
    axp.set_title("PSTH (column-mean rate)")
    axp.legend(frameon=False)
    axp.set_xlim(0, XMAX)

    # --- autocorrelogram (bottom-right): within-stop inter-onset intervals ---
    axa = axes[1, 1]
    ib = np.arange(0, 0.5, 0.015)
    for name, src in pops.items():
        s = src.sort_values(["stop_id", "onset_time"])
        isi = s.groupby("stop_id")["onset_time"].diff().dropna()
        axa.hist(isi[isi < 0.5], bins=ib, histtype="step", color=colors[name], lw=2, density=True, label=name)
    axa.axvspan(0, 0.06, color="0.9", zorder=0)
    axa.set_xlabel("inter-onset interval within stop (s)")
    axa.set_ylabel("density")
    axa.set_title("Autocorrelogram (within-stop ISI; shaded = refractory <60 ms)")
    axa.legend(frameon=False)

    fig.suptitle("Event-train triptych: raster · PSTH · autocorrelogram", y=1.0)
    fig.tight_layout()
    out = P.parent / ".claude" / "research" / "event_raster_outputs" / "event_rasters_triptych.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out.stem, base_dir=out.parent, dpi=150, formats=["png"], transparent=False)
    print("saved", out)
    for name, src in pops.items():
        print(f"  {name}: {len(src):,} onsets, {src.stop_id.nunique():,} stops")


if __name__ == "__main__":
    main()
