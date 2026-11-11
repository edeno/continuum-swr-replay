"""TEMP: extract per-event peak SWR/MUA z-score for the trunk dose-response.

For each of the 218 non_local sessions, load the saved *_HPC_80_data_over_time
pkl, and for every event in event_table take the peak SWR_zscore / MUA_zscore
over the event window. Saves a compact per-event table keyed by event_id.
Delete after the trunk-zscore investigation.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.tables.covariate_registry import (
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table

PD = Path("Processed-Data")
TRUNK = {6, 7, 8}


def main():
    eat = read_table(PD / "replay_content" / "event_arm_table.parquet")
    et = read_table(PD / "replay_content" / "event_table.parquet")
    require_covariate_schema(eat, table="event_arm")
    require_covariate_schema(et, table="event")

    # trunk share + n_unique_arms (recompute n_unique from arms with mass)
    eat = eat.dropna(subset=["arm_count_hard"]).copy()
    tot = eat.groupby("event_id")["arm_count_hard"].transform("sum")
    eat = eat[tot > 0].copy()
    eat["is_trunk"] = eat["candidate_arm"].isin(TRUNK)
    grp = eat.groupby("event_id")
    trunk = grp.apply(
        lambda g: g.loc[g.is_trunk, "arm_count_hard"].sum() / g["arm_count_hard"].sum(),
        include_groups=False,
    ).rename("trunk_share")
    n_arms_mass = grp.apply(
        lambda g: int((g["arm_count_hard"] > 0).sum()), include_groups=False
    ).rename("n_arms_with_mass")
    per_event = pd.concat([trunk, n_arms_mass], axis=1).reset_index()

    et = et.merge(per_event, on="event_id", how="inner")

    rows = []
    sessions = list(et.groupby(["animal", "session_date", "epoch"]))
    t0 = time.time()
    for i, ((a, d, e), es) in enumerate(sessions):
        fn = PD / f"{a}{d}_{e}_HPC_80_data_over_time.pkl"
        if not fn.exists():
            print(f"  MISSING {fn.name}", flush=True)
            continue
        dot = pd.read_pickle(fn)[["SWR_zscore", "MUA_zscore"]].sort_index()
        idx = dot.index.values
        sw = dot["SWR_zscore"].values
        mu = dot["MUA_zscore"].values
        for _, r in es.iterrows():
            lo = np.searchsorted(idx, r.event_start_s)
            hi = np.searchsorted(idx, r.event_end_s, side="right")
            if hi <= lo:
                continue
            rows.append(
                {
                    "event_id": r.event_id,
                    "animal": a,
                    "session_date": d,
                    "epoch": e,
                    "is_swr": bool(r.is_swr),
                    "is_mua": bool(r.is_mua),
                    "trunk_share": r.trunk_share,
                    "n_arms_with_mass": r.n_arms_with_mass,
                    "swr_z": float(np.nanmax(sw[lo:hi])),
                    "mua_z": float(np.nanmax(mu[lo:hi])),
                }
            )
        if (i + 1) % 25 == 0:
            print(
                f"  {i + 1}/{len(sessions)} sessions, {len(rows)} events, "
                f"{time.time() - t0:.0f}s",
                flush=True,
            )

    out = pd.DataFrame(rows)
    out_path = PD / "replay_content" / "_trunk_zscore_per_event.pkl"
    out.to_pickle(out_path)
    print(
        f"\nDONE: {len(out)} events from {out['epoch'].count()} rows -> {out_path} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
