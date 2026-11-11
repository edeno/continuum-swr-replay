"""TEMP (b): timebin-level instantaneous z-score -> trunk-coding probability.

For each non_local session: load data_over_time + environment, map
map_estimate -> arm, keep immobile non-local timebins, and relate each
timebin's instantaneous SWR/MUA z-score to whether the decoded arm is a
trunk arm. Computes a pooled z-binned curve, per-session Spearman, and a
WITHIN-EVENT (event-demeaned) version that asks whether the high-z
timebins *inside the same event* are the trunk-coding ones.

Aggregates to binned counts + per-session stats (never holds all
timebins). Delete after the trunk-zscore investigation.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from continuum_swr_replay.analysis.replay.spatial_mapping import map_estimate_to_arm
from continuum_swr_replay.tables.covariate_registry import (
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table

PD = Path("Processed-Data")
TRUNK = {6, 7, 8}
NONLOCAL_THRESH = 0.8
SPEED_THRESH = 4.0
BIN_EDGES = np.arange(-3.0, 14.01, 0.5)


def main():
    et = read_table(PD / "replay_content" / "event_table.parquet")
    require_covariate_schema(et, table="event")
    sessions = list(et.groupby(["animal", "session_date", "epoch"]))

    # global binned accumulators: n and n_trunk per z-bin, for SWR and MUA
    nb = len(BIN_EDGES) - 1
    acc = {k: np.zeros(nb) for k in ["swr_n", "swr_trunk", "mua_n", "mua_trunk"]}
    per_session = []
    t0 = time.time()

    for i, ((a, d, e), es) in enumerate(sessions):
        dot_fn = PD / f"{a}{d}_{e}_HPC_80_data_over_time.pkl"
        env_fn = PD / f"{a}{d}_{e}_HPC_environment.pkl"
        if not dot_fn.exists() or not env_fn.exists():
            print(f"  MISSING {a}{d}_{e}", flush=True)
            continue
        cols = [
            "Non-Local_Continuous_prob",
            "Non-Local_Fragmented_prob",
            "map_estimate",
            "SWR_zscore",
            "MUA_zscore",
            "head_speed",
        ]
        dot = pd.read_pickle(dot_fn)[cols].sort_index()
        env = pd.read_pickle(env_fn)

        nonlocal_prob = (
            dot["Non-Local_Continuous_prob"].to_numpy()
            + dot["Non-Local_Fragmented_prob"].to_numpy()
        )
        mask = (nonlocal_prob >= NONLOCAL_THRESH) & (
            dot["head_speed"].to_numpy() < SPEED_THRESH
        )
        if mask.sum() < 50:
            continue
        sub_t = dot.index.values[mask]
        arm = map_estimate_to_arm(dot["map_estimate"].to_numpy()[mask], env)
        valid = ~np.isnan(arm)
        arm = arm[valid]
        sub_t = sub_t[valid]
        is_trunk = np.isin(arm.astype(int), list(TRUNK)).astype(float)
        swr_z = dot["SWR_zscore"].to_numpy()[mask][valid]
        mua_z = dot["MUA_zscore"].to_numpy()[mask][valid]

        # global binned counts
        for zc, key in [(swr_z, "swr"), (mua_z, "mua")]:
            bi = np.clip(np.digitize(zc, BIN_EDGES) - 1, 0, nb - 1)
            acc[f"{key}_n"] += np.bincount(bi, minlength=nb)
            acc[f"{key}_trunk"] += np.bincount(bi, weights=is_trunk, minlength=nb)

        # per-session marginal Spearman
        sp_swr = spearmanr(swr_z, is_trunk).statistic if is_trunk.std() else np.nan
        sp_mua = spearmanr(mua_z, is_trunk).statistic if is_trunk.std() else np.nan

        # WITHIN-EVENT: assign each timebin to the event window containing it
        ev = es[["event_start_s", "event_end_s"]].to_numpy()
        order = np.argsort(ev[:, 0])
        ev = ev[order]
        starts = ev[:, 0]
        j = np.searchsorted(starts, sub_t, side="right") - 1
        in_ev = (j >= 0) & (j < len(ev))
        in_ev[in_ev] &= sub_t[in_ev] <= ev[j[in_ev], 1]
        we_sp_swr = we_sp_mua = np.nan
        n_we_events = 0
        if in_ev.sum() > 50:
            eid = j[in_ev]
            tk = is_trunk[in_ev]
            zs = swr_z[in_ev]
            zm = mua_z[in_ev]
            df = pd.DataFrame({"eid": eid, "tk": tk, "zs": zs, "zm": zm})
            # demean within event
            for c in ["tk", "zs", "zm"]:
                df[c + "_d"] = df[c] - df.groupby("eid")[c].transform("mean")
            # keep only events with >1 timebin and within-event variation
            df = df[df.groupby("eid")["eid"].transform("size") > 1]
            n_we_events = df["eid"].nunique()
            if len(df) > 50 and df["tk_d"].std() > 0:
                if df["zs_d"].std() > 0:
                    we_sp_swr = spearmanr(df["zs_d"], df["tk_d"]).statistic
                if df["zm_d"].std() > 0:
                    we_sp_mua = spearmanr(df["zm_d"], df["tk_d"]).statistic

        per_session.append(
            {
                "animal": a,
                "session_date": d,
                "epoch": e,
                "n_timebins": len(is_trunk),
                "trunk_rate": float(is_trunk.mean()),
                "spearman_swr": sp_swr,
                "spearman_mua": sp_mua,
                "we_spearman_swr": we_sp_swr,
                "we_spearman_mua": we_sp_mua,
                "n_we_events": n_we_events,
            }
        )
        if (i + 1) % 25 == 0:
            print(
                f"  {i + 1}/{len(sessions)} sessions, {time.time() - t0:.0f}s",
                flush=True,
            )

    centers = 0.5 * (BIN_EDGES[:-1] + BIN_EDGES[1:])
    binned = pd.DataFrame(
        {
            "z_center": centers,
            "swr_n": acc["swr_n"],
            "swr_trunk": acc["swr_trunk"],
            "mua_n": acc["mua_n"],
            "mua_trunk": acc["mua_trunk"],
        }
    )
    ps = pd.DataFrame(per_session)
    binned.to_pickle(PD / "replay_content" / "_trunk_zscore_timebin_binned.pkl")
    ps.to_pickle(PD / "replay_content" / "_trunk_zscore_timebin_persession.pkl")
    print(f"\nDONE: {len(ps)} sessions, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
