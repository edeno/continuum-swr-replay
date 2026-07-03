"""Augment the NonLocal content arm table with the HMM per-arm value.

Adds ``q_arm_hmm`` (+ its ``swr_x_q_arm_hmm`` interaction) to the existing
``event_arm_table.pkl`` by the SAME ``attach_hmm_arm_value`` mapping the table
builder now uses, sourced from the regenerated ``qvals_table.pkl`` (which carries
``hmm_q_port{0..5}``). This is rebuild-equivalent to a full ``tables.py`` rebuild
for these two columns, but avoids the expensive decode-derived rebuild -- a bridge
until the table is next regenerated from scratch (which will emit them natively).

Run AFTER `python -m continuum_swr_replay.analysis.behavior.qvals_table --write`.
"""

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.replay.covariates import attach_hmm_arm_value
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

RC = PROCESSED_DATA_DIR / "replay_content"
qv = pd.read_pickle(RC / "models_per_bin_rate" / "qvals_table.pkl")
eat_path = RC / "event_arm_table.pkl"
eat = pd.read_pickle(eat_path)

q_hmm = attach_hmm_arm_value(
    eat["arm"].to_numpy(), eat["stop_id"].to_numpy(), qv
)
eat["q_arm_hmm"] = q_hmm
eat["swr_x_q_arm_hmm"] = eat["is_swr"].astype(int).to_numpy() * q_hmm

# Sanity: q_arm_hmm should be present wherever the beta q_arm is (both key off the
# same stop_id contingency presence), and land in the same value range as Q.
both_present = eat["q_arm"].notna() & eat["q_arm_hmm"].notna()
print(f"rows: {len(eat)}  q_arm present: {int(eat['q_arm'].notna().sum())}  "
      f"q_arm_hmm present: {int(eat['q_arm_hmm'].notna().sum())}  "
      f"both: {int(both_present.sum())}")
print(f"q_arm_hmm range: {np.nanmin(q_hmm):.3f} .. {np.nanmax(q_hmm):.3f}")
# leaf arms only (trunks are structural 0.0); correlation with beta q_arm as a
# loose sanity check (same construct family, different model -> positive but < 1)
leaf = (eat["arm"].to_numpy() < 6) & both_present.to_numpy()
r = np.corrcoef(eat.loc[leaf, "q_arm"], eat.loc[leaf, "q_arm_hmm"])[0, 1]
print(f"leaf-arm corr(q_arm_beta, q_arm_hmm) = {r:.3f}  (n={int(leaf.sum())})")

eat.to_pickle(eat_path)
print(f"wrote {eat_path} with q_arm_hmm + swr_x_q_arm_hmm")
