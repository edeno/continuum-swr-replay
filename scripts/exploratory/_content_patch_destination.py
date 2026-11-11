#!/usr/bin/env python
"""Where does off-current content go? Patch-level destination on switch events.

TEMPORARY / exploratory. On patch-SWITCH events (current patch != previous patch,
so current/prev/other are three distinct patches), normalize each event's content
to a distribution over the 3 patches and report the mean content fraction on
current / prev / other (third) patch, split by SWR. Also: is the prev patch the
last-rewarded one (explaining its avoidance)?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.paths import REPLAY_CONTENT_DIR, table_path
from continuum_swr_replay.tables.covariate_registry import (
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table

RC = REPLAY_CONTENT_DIR


def main() -> None:
    pt = read_table(table_path("event_patch_table", RC))
    require_covariate_schema(pt, table="event_patch")
    tot = pt.groupby("event_id")["patch_count_hard"].transform("sum")
    pt = pt[tot > 0].copy()
    pt["frac"] = pt["patch_count_hard"] / tot[tot > 0]
    cur = pt[pt.is_current_patch == 1].groupby("event_id")["patch"].first()
    prev = pt[pt.is_prev_patch == 1].groupby("event_id")["patch"].first()
    ev = pd.DataFrame({"cur": cur, "prev": prev}).dropna()
    ev = ev[ev.cur != ev.prev]  # SWITCH events: 3 distinct patch roles
    pt = pt[pt.event_id.isin(ev.index)].merge(ev, left_on="event_id", right_index=True)
    pt["role"] = np.where(
        pt.patch == pt.cur, "current", np.where(pt.patch == pt.prev, "prev", "other")
    )

    print(
        f"switch events: {ev.shape[0]}  (content distribution over current/prev/other)\n"
    )
    print(f"{'pop':<10}{'current':>10}{'prev':>10}{'other':>10}{'n_events':>10}")
    for lab, sub in [("non-SWR", pt[~pt.is_swr]), ("with-SWR", pt[pt.is_swr])]:
        m = sub.groupby("role")["frac"].mean()
        print(
            f"{lab:<10}{m.get('current', 0):>10.3f}{m.get('prev', 0):>10.3f}"
            f"{m.get('other', 0):>10.3f}{sub.event_id.nunique():>10}"
        )

    print("\n--- of OFF-current content, is the patch the last-rewarded one? ---")
    off = pt[pt.role != "current"]
    for lab, sub in [("non-SWR", off[~off.is_swr]), ("with-SWR", off[off.is_swr])]:
        lr = sub.groupby("role")["is_last_rewarded_patch"].mean()
        print(
            f"{lab:<10} frac last-rewarded:  prev={lr.get('prev', np.nan):.2f}"
            f"  other={lr.get('other', np.nan):.2f}"
        )


if __name__ == "__main__":
    main()
