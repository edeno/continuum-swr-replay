#!/usr/bin/env python
"""Re-verify the documented content findings against freshly re-fit models.

TEMPORARY / exploratory. Reads the re-fit content-model outputs (arm + patch,
all/SWR-split) and reports the claim-relevant coefficients (median across
sessions, FDR Wilcoxon p, %positive, per-animal sign consistency), checked
against the documented findings in paper/claims_table.md.
"""

from __future__ import annotations

import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

RC = PROCESSED_DATA_DIR / "replay_content"
SC = ["animal", "day", "epoch"]

# (config label, models dir, models to report)
CONFIGS = [
    ("arm ALL", "models", ["M1", "M5", "M6", "M7"]),
    ("arm SWR=T (with-SWR)", "models_swr_T", ["M1", "M5"]),
    ("arm SWR=F (non-SWR)", "models_swr_F", ["M1", "M5"]),
    ("patch ALL", "models_patch", ["P1", "P4"]),
    ("patch SWR=T", "models_patch_swr_T", ["P1"]),
    ("patch SWR=F", "models_patch_swr_F", ["P1"]),
]
# documented expectations (from claims_table.md) to flag
EXPECT = {
    "is_current": "+ large (current-arm bias)",
    "is_last_rewarded_elsewhere": "- (avoid last-rewarded; ~-1.66, non-SWR)",
    "reward_rate_arm_k10": "~0 n.s. (NOT value-selective, non-SWR)",
    "is_prev": "context",
    "swr_x_current": "SWR x current (broadening)",
    "swr_x_prev": "SWR x prev",
    "is_current_patch": "+ (patch current)",
    "is_prev_patch": "- post-switch",
}


def _report(mdf: pd.DataFrame, model: str) -> None:
    sub = mdf[mdf.model == model]
    if sub.empty:
        print(f"    {model}: (no rows)")
        return
    coef_cols = [c for c in sub.columns if c.startswith("coef_")
                 and not c.startswith("coef_arm_fe")]
    stats = []
    for c in coef_cols:
        s = sub[c].dropna()
        if len(s) < 5:
            continue
        p = wilcoxon(s)[1]
        n_an = (sub.groupby("animal")[c].median() > 0).sum()
        stats.append((c[5:], s.median(), (s > 0).mean() * 100, n_an, p, len(s)))
    if not stats:
        return
    _, fdr, _, _ = multipletests([t[4] for t in stats], method="fdr_bh")
    print(f"    {model} (n={len(sub)} sessions):")
    print(f"      {'predictor':<32}{'med':>8}{'%pos':>7}{'an':>5}{'p(FDR)':>10}  note")
    for (name, med, pos, n_an, _p, _n), q in zip(stats, fdr, strict=True):
        note = next((v for k, v in EXPECT.items() if k == name), "")
        print(f"      {name:<32}{med:>+8.2f}{pos:>6.0f}%{f'{n_an}/5':>5}{q:>10.1e}  {note}")


def main() -> None:
    for label, subdir, models in CONFIGS:
        path = RC / subdir / "all_models.pkl"
        print(f"\n{'=' * 72}\n{label}   [{subdir}/all_models.pkl]\n{'=' * 72}")
        if not path.exists():
            print("    (missing — not re-fit yet)")
            continue
        mdf = pd.read_pickle(path)
        if "model" not in mdf.columns:
            print("    (no 'model' column)")
            continue
        for m in models:
            _report(mdf, m)


if __name__ == "__main__":
    main()
