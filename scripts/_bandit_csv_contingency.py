#!/usr/bin/env python
"""Canonical per-stop contingency/value from the LorenFrankLab/SpatialBanditTask CSVs.

TEMPORARY / exploratory (underscore-prefixed). Builds a per-stop_id table of
nominal value (p_reward_current, rpe, value_diff_patch, block, trials-since-block)
from the lab's canonical parsed behavior CSVs, to validate and ultimately replace
scripts/_statescript_contingency.py.

The CSVs are trial-level (one row per port visit) with no timestamps; our
stop_table is finer (many immobility bouts per visit). Mapping, per (animal, day):
  1. collapse each epoch's stops into consecutive-arm WELL VISITS;
  2. match each epoch to the CSV session whose visit/trial ARM sequence aligns best
     (difflib ratio; greedy, unmatched epochs allowed -- the CSV is a cleaned subset);
  3. align visit->trial within the matched pair (difflib matching blocks) and read
     each visit's contingency. Because contingency is block-constant, only the block
     needs to be right, so the alignment is highly tolerant; gaps inherit the
     nearest aligned trial's block (forward/back fill).
Each visit's value fields broadcast to all its stops.

Cross-check vs the StateScript parser's contingency_table.pkl on shared stop_ids.

Usage: python scripts/_bandit_csv_contingency.py [--animal A] [--write]
"""

from __future__ import annotations

import argparse
import difflib

import numpy as np
import pandas as pd

from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

RC = PROCESSED_DATA_DIR / "replay_content"
CSV_DIR = PROCESSED_DATA_DIR / "spatial_bandit_behavior" / "data"
ANIMALS = ("chimi", "j16", "peanut", "senor", "wilbur")
# wells 0-5 -> patches 0,0,1,1,2,2 (WELL_PATCH_MAPPING, 0-indexed patches)
WELL_PATCH = np.array([0, 0, 1, 1, 2, 2])


def _decode_contingency(codes: pd.Series) -> np.ndarray:
    """(n_trials, 6) reward probability per port from the 12-digit code."""
    s = codes.astype("int64").astype(str).str.zfill(12)
    return np.array([[int(t[i : i + 2]) for i in range(0, 12, 2)] for t in s]) / 100.0


def _patch_value_diff(ports: np.ndarray) -> np.ndarray:
    """best-minus-second-best patch value (mean p(R) over each patch's 2 wells)."""
    patch_val = np.stack(
        [ports[:, WELL_PATCH == p].mean(axis=1) for p in range(3)], axis=1
    )
    s = np.sort(patch_val, axis=1)
    return s[:, -1] - s[:, -2]


def _prep_csv(animal: str) -> pd.DataFrame:
    df = pd.read_csv(CSV_DIR / f"{animal}_clean_contingencies_only_parsed_data.csv", index_col=0)
    ports = _decode_contingency(df["contingency"])
    arm = df["leaf"].to_numpy() - 1  # 1-indexed leaf -> 0-indexed arm
    patch_val = np.stack([ports[:, WELL_PATCH == p].mean(axis=1) for p in range(3)], axis=1)
    best_patch = patch_val.argmax(axis=1)
    df = df.assign(
        arm=arm,
        p_reward_current=ports[np.arange(len(df)), arm],
        p_reward_best=ports.max(axis=1),
        is_best_patch=(WELL_PATCH[arm] == best_patch),
        value_diff_patch=_patch_value_diff(ports),
    )
    df["rpe"] = df["reward"].to_numpy() - df["p_reward_current"].to_numpy()
    # block = contingency-change index within (date, session); position within block
    out = []
    for _, g in df.groupby(["date", "session"], sort=False):
        g = g.sort_values("trial").copy()
        chg = (g["contingency"].to_numpy()[1:] != g["contingency"].to_numpy()[:-1])
        block = np.concatenate([[0], np.cumsum(chg)])
        g["block_number"] = block
        g["trials_since_block_change"] = g.groupby(block).cumcount().to_numpy()
        out.append(g)
    return pd.concat(out, ignore_index=True)


def _visits(arms: list) -> tuple[list, list[list[int]]]:
    """Consecutive-arm run-length encode -> (visit arms, stop-index groups)."""
    v_arm, groups = [], []
    for i, a in enumerate(arms):
        if pd.isna(a):
            continue
        if not v_arm or a != v_arm[-1]:
            v_arm.append(a)
            groups.append([])
        groups[-1].append(i)
    return v_arm, groups


def build_animal(animal: str) -> pd.DataFrame:
    csv = _prep_csv(animal)
    st = pd.read_pickle(RC / "stop_table.pkl")
    st = st[st.animal == animal]
    val_cols = ["arm", "p_reward_current", "p_reward_best", "is_best_patch", "rpe",
                "value_diff_patch", "block_number", "trials_since_block_change"]
    rows = []
    for day, st_day in st.groupby("day"):
        csv_day = csv[csv.date == day]
        if csv_day.empty:
            continue
        sessions = {s: g.sort_values("trial").reset_index(drop=True)
                    for s, g in csv_day.groupby("session")}
        # match each epoch to its best CSV session by arm-sequence ratio (greedy)
        epochs = {}
        for ep, g in st_day.groupby("epoch"):
            g = g.sort_values("stop_idx")
            v_arm, groups = _visits(g["current_arm"].tolist())
            epochs[ep] = (g, [int(x) for x in v_arm], groups)
        used = set()
        for ep, (g, v_arm, groups) in epochs.items():
            best_s, best_r = None, 0.0
            for s, sg in sessions.items():
                if s in used:
                    continue
                r = difflib.SequenceMatcher(
                    None, v_arm, sg["arm"].tolist(), autojunk=False
                ).ratio()
                if r > best_r:
                    best_r, best_s = r, s
            if best_s is None or best_r < 0.5:
                continue  # epoch with no clean CSV session
            used.add(best_s)
            sg = sessions[best_s]
            # align visit -> trial; fill gaps with nearest aligned trial (block-constant)
            sm = difflib.SequenceMatcher(None, v_arm, sg["arm"].tolist(), autojunk=False)
            v2t = np.full(len(v_arm), -1)
            for a, b, size in sm.get_matching_blocks():
                for k in range(size):
                    v2t[a + k] = b + k
            v2t = pd.Series(v2t).replace(-1, np.nan).ffill().bfill().astype("Int64")
            g = g.reset_index(drop=True)
            for vi, stop_pos in enumerate(groups):
                ti = v2t[vi]
                if pd.isna(ti):
                    continue
                trow = sg.iloc[int(ti)]
                for pos in stop_pos:
                    rows.append({
                        "stop_id": g.iloc[pos]["stop_id"],
                        "epoch": ep, "csv_session": best_s, "match_ratio": round(best_r, 3),
                        **{c: trow[c] for c in val_cols},
                    })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--animal", choices=ANIMALS, default=None)
    p.add_argument("--write", action="store_true")
    args = p.parse_args()
    animals = [args.animal] if args.animal else list(ANIMALS)

    parts = []
    for a in animals:
        t = build_animal(a)
        st = pd.read_pickle(RC / "stop_table.pkl")
        n_stop = (st.animal == a).sum()
        print(f"{a:7s} mapped {len(t):5d}/{n_stop:5d} stops "
              f"({100*len(t)/n_stop:.0f}%)  median match_ratio={t['match_ratio'].median():.2f}")
        parts.append(t)
    tab = pd.concat(parts, ignore_index=True)

    # cross-check vs StateScript parser
    old = pd.read_pickle(RC / "models_per_bin_rate" / "contingency_table.pkl")
    m = tab.merge(old[["stop_id", "p_reward_current"]], on="stop_id",
                  how="inner", suffixes=("_csv", "_ss"))
    agree = np.isclose(m["p_reward_current_csv"].astype(float),
                       m["p_reward_current_ss"].astype(float), atol=1e-6)
    print(f"\ncross-check vs StateScript parser: {len(m)} shared stops, "
          f"p_reward_current agree = {100*agree.mean():.1f}%")
    if (~agree).any():
        bad = m[~agree]
        print("  disagreements by |diff|:")
        print((bad["p_reward_current_csv"].astype(float)
               - bad["p_reward_current_ss"].astype(float)).round(2).value_counts().head())

    if args.write:
        out = RC / "models_per_bin_rate" / "contingency_table_csv.pkl"
        tab.to_pickle(out)
        print(f"\nwrote {len(tab)} rows -> {out}")


if __name__ == "__main__":
    main()
