#!/usr/bin/env python
"""Time-resolved event-onset rate vs time-since-reward-pump (Part B core).

The confound-free version of the lag-1 consummatory boost: per-bin onset rate
as a function of t_since_pump over ALL immobility bins (small t = post-reward
consummatory; large t = far from reward). Numerator = onset counts per t-bin
(event_onset_table); denominator = immobility-bin exposure per t-bin
(stop_timebin_table). Non-local onsets are split by canonical event-level
is_swr (matched within stop by time-rank).

rate(t) = onsets(t) / (exposure_bins(t) * bin_s)   [events / s of immobility]

NB: does NOT use is_reward_stop (a pump fires within ~0.1 s of the poke, so
almost none fall strictly inside a later bout). The canonical clock for
"how long since the outcome was available" is the stop table's
``time_since_outcome_known_s`` / ``outcome_known_time_s``; use those rather
than re-deriving a pump offset here.
"""

from __future__ import annotations

import warnings

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

warnings.filterwarnings("ignore")
from continuum_swr_replay.analysis.replay import attach_is_swr_to_onsets  # noqa: E402
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
BIN_S = 0.004
TBINS = np.arange(0, 10.01, 0.25)
CENT = 0.5 * (TBINS[:-1] + TBINS[1:])
COLORS = {"detected SWR": "k", "non-local w/ SWR": "C3", "non-local NO SWR": "C0"}


def main() -> None:
    dense = read_table(table_path("stop_timebin_table", RC))
    require_covariate_schema(dense, table="stop_timebin")
    exp_h, _ = np.histogram(dense.t_since_pump.dropna().to_numpy(), TBINS)
    sec = exp_h * BIN_S  # seconds of immobility exposure per t-bin

    onset_source = read_table(table_path("event_onset_table", RC))
    events = read_table(table_path("event_table", RC))
    require_covariate_schema(onset_source, table="event_onset")
    require_covariate_schema(events, table="event")
    onset = attach_is_swr_to_onsets(onset_source, events)
    nl = onset[onset.event_type == "non_local"]

    pops = {
        "detected SWR": onset.loc[onset.event_type == "SWR", "t_since_pump"],
        "non-local w/ SWR": nl.loc[nl.is_swr.eq(True), "t_since_pump"],
        "non-local NO SWR": nl.loc[nl.is_swr.eq(False), "t_since_pump"],
    }

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    print("population        | n      | peak rate | t_peak(s) | rate 7-10s | peak/tail")
    for name, t in pops.items():
        h, _ = np.histogram(t.dropna().to_numpy(), TBINS)
        rate = h / np.where(sec > 0, sec, np.nan)
        ax.plot(CENT, rate, color=COLORS[name], lw=2, label=f"{name} (n={int(t.notna().sum()):,})")
        pk = np.nanmax(rate)
        tpk = CENT[np.nanargmax(rate)]
        tail = h[(CENT >= 7) & (CENT < 10)].sum() / max(sec[(CENT >= 7) & (CENT < 10)].sum(), 1e-9)
        print(f"{name:17s} | {int(t.notna().sum()):6,} | {pk:9.3f} | {tpk:9.2f} | {tail:10.3f} | {pk / max(tail, 1e-9):.2f}")
    ax.axvspan(0, 1.0, color="0.9", zorder=0)
    ax.text(0.5, ax.get_ylim()[1] * 0.04, "consumption", ha="center", fontsize=8, color="0.4")
    ax.set_xlabel("time since reward pump (s)")
    ax.set_ylabel("onset rate (events / s of immobility)")
    ax.set_title("Time-resolved onset rate vs time since reward\n(equal-exposure 4 ms bins, all immobility)")
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(0, 10)
    fig.tight_layout()
    out = P.parent / ".claude" / "research" / "rate_psth_outputs" / "rate_timecourse_vs_pump.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
