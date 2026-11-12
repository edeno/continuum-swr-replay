"""Factorial diagnostic: which change flips CS5 (swr x arm value)?
Axes: response (count vs fraction) x base design (BASEc_NT, +TSC, BASEc full).
Reuses the SWR-contrast runner's cohort assembly and _run, so leaf-only handling
and cohort match the shipped fit."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location(
    "swrc", "scripts/behavior/content/fit_swr_contrast_content.py"
)
m = importlib.util.module_from_spec(spec)
sys.modules["swrc"] = m
spec.loader.exec_module(m)
from continuum_swr_replay.analysis.behavior.contrast_aggregation import (  # noqa: E402
    assemble_arm,
    assemble_patch,
    contrast_fit_inputs,
    prepare_cohorts,
)
from continuum_swr_replay.analysis.behavior.model_specs import (  # noqa: E402
    _TSC,
    BASEc,
    BASEc_NT,
)
from continuum_swr_replay.paths import replay_content_dir  # noqa: E402

inputs = contrast_fit_inputs()
ev, _ = assemble_arm(inputs, replay_content_dir() / "standardization_reference.json")
pt, _ = assemble_patch(ev, inputs)
ev, pt, _, _ = prepare_cohorts(ev, pt, outcome_stratum="all", max_per_animal=None)
print("cohort events", ev.event_id.nunique(), "rows", len(ev), flush=True)
KEY = "swr_x_reward_belief_mean_candidate_arm_preoutcome"
VAL = "reward_belief_mean_candidate_arm_preoutcome"
designs = {"NT": BASEc_NT, "NT+TSC": [*BASEc_NT, _TSC], "full": BASEc}
out = []
for resp in ("arm_count_hard", "arm_fraction"):
    for dname, base in designs.items():
        ladder = {"CS5_armvalue": ([*base, VAL, KEY], "pooled")}
        res, _ = m._run(ev, ladder, response_col=resp, n_categories=9)
        res = res[res.model == "CS5_armvalue"].copy()
        sd = res[f"sd_{KEY}"] if f"sd_{KEY}" in res else np.nan
        ok = res[f"coef_{KEY}"].notna()
        if "separated_terms" in res:
            ok &= ~res["separated_terms"].apply(lambda t: KEY in (t or []))
        ok &= (res[f"coef_{KEY}"].abs() * sd < 10) if f"sd_{KEY}" in res else True
        r = res[ok]
        per_animal = r.groupby("animal")[f"coef_{KEY}"].median()
        se_col = f"bse_cluster_{KEY}" if f"bse_cluster_{KEY}" in r else f"bse_{KEY}"
        out.append(
            {
                "response": resp,
                "design": dname,
                "n_sessions": len(r),
                "median_coef": r[f"coef_{KEY}"].median(),
                "n_animals_pos": int((per_animal > 0).sum()),
                "per_animal": per_animal.round(3).to_dict(),
                "median_se": r[se_col].median(),
                "se_col": se_col,
            }
        )
        print(out[-1], flush=True)
pd.DataFrame(out).to_csv(
    Path(".claude/docs/reviews/2026-09-05-cs5-response-by-design-factorial.csv"),
    index=False,
)
