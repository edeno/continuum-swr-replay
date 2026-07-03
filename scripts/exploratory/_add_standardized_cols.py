"""Add standardized (global 1-SD z-score) columns for the continuous covariates.

Best practice for comparable effect sizes: standardize continuous predictors to
unit SD so a coefficient is 'effect per 1 SD of predictor', leaving binary (0/1)
predictors raw. For these Poisson/conditional-Poisson GLMs the standardized
coefficient equals raw_coef * SD(x) exactly (SEs too), so this also documents the
SD reference used to convert existing raw fits to standardized units.

Adds ``<name>_z`` to event_arm_table (per-arm covariates) and qvals_table
(event-scalar covariates), each z-scored by its GLOBAL (pooled) mean/SD so
per-session coefficients share one scale. Writes the SD reference to JSON.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

RC = PROCESSED_DATA_DIR / "replay_content"
OUTREF = Path(".claude/research/swr_contrast_coefs/covariate_sd_reference.json")
OUTREF.parent.mkdir(parents=True, exist_ok=True)

# continuous covariates by table (binary predictors deliberately excluded)
ARM_CONT = ["q_arm", "q_arm_hmm", "reward_rate_arm_k10", "p_reward_arm"]
QV_CONT = ["hmm_relative_value", "hmm_rpe", "reward_entropy", "hmm_value_update",
           "state_entropy", "trials_since_switch", "trials_since_block_change"]

sd_ref = {}


def add_z(path, cols):
    df = pd.read_pickle(path)
    bak = path.with_suffix(path.suffix + ".bak_prestd")
    if not bak.exists():
        df.to_pickle(bak)
    for c in cols:
        if c not in df.columns:
            print(f"  (skip {c}: absent)")
            continue
        x = df[c].to_numpy(dtype=float)
        mu, sd = np.nanmean(x), np.nanstd(x)
        sd_ref[c] = float(sd)
        df[c + "_z"] = (x - mu) / sd if sd > 0 else np.nan
        print(f"  {c:24s} mean={mu:+.3f} SD={sd:.3f} -> {c}_z")
    df.to_pickle(path)


print("event_arm_table.pkl:")
add_z(RC / "event_arm_table.pkl", ARM_CONT)
print("qvals_table.pkl:")
add_z(RC / "models_per_bin_rate" / "qvals_table.pkl", QV_CONT)
# recent_switch_rate lives in stop_table (add_uncertainty_covariates); grab its SD too
try:
    from continuum_swr_replay.analysis.replay._helpers import add_uncertainty_covariates
    st = add_uncertainty_covariates(pd.read_pickle(RC / "stop_table.pkl"))
    if "recent_switch_rate" in st:
        sd_ref["recent_switch_rate"] = float(np.nanstd(st["recent_switch_rate"].to_numpy(float)))
        print(f"  recent_switch_rate SD={sd_ref['recent_switch_rate']:.3f} (from stop_table)")
except Exception as e:  # noqa: BLE001
    print("  (recent_switch_rate SD skipped:", type(e).__name__, ")")

OUTREF.write_text(json.dumps(sd_ref, indent=2, sort_keys=True))
print("\nwrote SD reference ->", OUTREF)
