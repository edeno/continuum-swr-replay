#!/usr/bin/env python
"""Peri-departure event RATE of non-local events by type (exploratory).

Align every non-local event to the animal's departure (`stop_end`) and plot a
hazard-style rate: events/s among stops still immobile at that lead time. The
denominator at lead time tau = (# stops with immobility_time >= tau) * dt, so the
curve is a proper rate, not a raw histogram (which would just track exposure and
misread short-stop concentration as a pre-departure ramp).

Groups (2x2): {non-SWR, SWR} x {content on / off the current arm}, where
"on-current" = the event's dominant content arm (argmax arm_count_hard) equals
the animal's current_arm.

Writes to .claude/research/peri_departure_outputs/.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from continuum_swr_replay.parameters import PROCESSED_DATA_DIR  # noqa: E402
from continuum_swr_replay.table_io import read_table

RC = PROCESSED_DATA_DIR / "replay_content"
OUT = Path(__file__).resolve().parents[2] / ".claude/research/peri_departure_outputs"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    ea = read_table(RC / "event_arm_table.pkl")
    et = read_table(RC / "event_table.pkl")
    stop = read_table(RC / "stop_table.pkl")

    dom = ea.loc[ea.groupby("event_id")["arm_count_hard"].idxmax(),
                 ["event_id", "arm"]].rename(columns={"arm": "dominant_arm"})
    ev = et.merge(dom, on="event_id", how="left")
    ev["on_current"] = ev["dominant_arm"] == ev["current_arm"]
    ev["grp"] = (np.where(ev["is_swr"], "SWR", "nonSWR") + " "
                 + np.where(ev["on_current"], "on-current", "off-current"))
    ev = ev.merge(stop[["stop_id", "stop_end"]], on="stop_id", how="left")
    ev["t_to_depart"] = ev["stop_end"] - ev["event_midpoint_time"]

    dt = 0.2
    edges = np.arange(0, 6 + dt, dt)
    centers = edges[:-1] + dt / 2
    D = stop["immobility_time"].to_numpy()
    exposure = np.array([(D >= c).sum() for c in centers]) * dt  # stop-seconds/bin

    groups = {
        "nonSWR on-current": ("#d1495b", 2.6),
        "nonSWR off-current": ("#edae49", 1.8),
        "SWR on-current": ("#00798c", 1.8),
        "SWR off-current": ("#003d5b", 1.8),
    }

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    table = {"lead_time_s": -centers}
    for label, (color, lw) in groups.items():
        t = ev.loc[ev["grp"] == label, "t_to_depart"].to_numpy()
        t = t[(t >= 0) & (t < 6)]
        rate = np.histogram(t, bins=edges)[0] / exposure
        table[label] = rate
        ax.plot(-centers, rate, color=color, lw=lw, label=f"{label} (n={len(t)})")

    ax.axvline(0, color="k", lw=1.2, ls="--", alpha=0.7)
    ax.text(-0.05, ax.get_ylim()[1] * 0.97, "departure", rotation=90,
            va="top", ha="right", fontsize=9, alpha=0.7)
    ax.set_xlabel("time relative to departure (s)   [0 = animal leaves]")
    ax.set_ylabel("event rate (events / s, among still-immobile stops)")
    ax.set_title("Peri-departure rate of non-local events by type")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "peri_departure_rate.png", dpi=140, bbox_inches="tight")
    pd.DataFrame(table).to_csv(OUT / "peri_departure_rate.csv", index=False)
    print("wrote", OUT / "peri_departure_rate.png")
    print("wrote", OUT / "peri_departure_rate.csv")


if __name__ == "__main__":
    main()
