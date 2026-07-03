"""Decisive test for the swr_x_value sign flip (beta vs HMM value construct).

Hypothesis: CS5 omits swr_x_current, and beta assigns the current arm a high
positive value while HMM assigns it ~0, so beta's swr_x_q_arm aliases the strong
off-current shift (swr_x_current<0) and inherits its sign. Adding swr_x_current as
a control should de-alias beta and converge the two constructs.

Fits per session (conditional Poisson, 9 arms), reports median swr_x_value coef +
per-animal sign consistency for 4 cells: {beta,hmm} x {no control, +swr_x_current}.
"""

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay.fitting import fit_conditional_poisson
from continuum_swr_replay.analysis.replay.model_specs import (
    SWR_CONTRAST_CONTENT_BASEC as BASEc,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

RC = PROCESSED_DATA_DIR / "replay_content"
SC = ["animal", "day", "epoch"]
ev = pd.read_pickle(RC / "event_arm_table.pkl")
ev = ev[ev["q_arm"].notna() & ev["q_arm_hmm"].notna()].copy()
ev["swr_x_q_arm"] = ev["is_swr"].astype(int).to_numpy() * ev["q_arm"].to_numpy()
# arm fixed-effect dummies (arm_fe_k = indicator arm==k; arm 8 = reference)
for k in range(8):
    ev[f"arm_fe_{k}"] = (ev["arm"].to_numpy() == k).astype(float)

CELLS = {
    "beta_noctl":  ([*BASEc, "q_arm", "swr_x_q_arm"], "swr_x_q_arm"),
    "beta_+swrcur": ([*BASEc, "swr_x_current", "q_arm", "swr_x_q_arm"], "swr_x_q_arm"),
    "hmm_noctl":   ([*BASEc, "q_arm_hmm", "swr_x_q_arm_hmm"], "swr_x_q_arm_hmm"),
    "hmm_+swrcur": ([*BASEc, "swr_x_current", "q_arm_hmm", "swr_x_q_arm_hmm"], "swr_x_q_arm_hmm"),
}

rows = {k: [] for k in CELLS}
for keys, g in ev.groupby(SC):
    if g["event_id"].nunique() < 18:
        continue
    for cell, (preds, term) in CELLS.items():
        bad = g.loc[g[[*preds, "arm_count_hard"]].isna().any(axis=1), "event_id"]
        d = g[~g["event_id"].isin(bad)]
        if d["event_id"].nunique() < 18:
            continue
        r = fit_conditional_poisson(d, predictors=preds, response_col="arm_count_hard",
                                    event_col="event_id", n_categories=9)
        if r and f"coef_{term}" in r and np.isfinite(r[f"coef_{term}"]):
            rows[cell].append((keys[0], r[f"coef_{term}"]))

print(f"\n{'cell':13s} {'n_sess':>6s} {'median_coef':>12s} {'n_animals_pos':>14s}")
for cell in CELLS:
    vals = [c for _, c in rows[cell] if abs(c) < 10]  # drop quasi-separated
    if not vals:
        print(f"{cell:13s} {'-':>6s}"); continue
    df = pd.DataFrame(rows[cell], columns=["animal", "coef"])
    df = df[df.coef.abs() < 10]
    npos = int((df.groupby("animal").coef.median() > 0).sum())
    print(f"{cell:13s} {len(vals):>6d} {np.median(vals):>12.3f} {npos:>10d}/5")
