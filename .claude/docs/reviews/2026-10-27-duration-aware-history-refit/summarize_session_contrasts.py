"""Session-level AIC contrasts of the duration-aware history refit (read-only).

For each cohort's ``duration_aware_history_models{suffix}.pkl`` and each
population, per session (both rungs fit):

- self: no history -> onset-only / offset-only / both anchors; onset-only vs
  offset-only; what each anchor adds to the other;
- cross (the two non-local streams): self (both) -> + cross onset-only /
  offset-only / both; onset-only vs offset-only.

A contrast is ``AIC(simpler) - AIC(richer)`` (positive favors the richer, or
for ``*_offset_vs_onset`` the offset anchor). Aggregation: session median and
share positive; per animal the median and the summed likelihood-ratio
statistic (``sum(dAIC + 2 * dk)``, chi-square with ``sum(dk)`` df under no
effect, valid however sparse a session is); animals positive of 5 (animal
median > 0) and animals significant (summed-LR p < 0.05). One-anchor-vs-one-
anchor contrasts are non-nested and get no LR test.

At-well stops only. Fit failures and separation flags are counted per rung.
Writes ``session_contrasts.json``.

Usage::

    CSR_PROCESSED_DATA_DIR=/cumulus/edeno/continuum-swr-replay/Processed-Data \\
        uv run python .claude/docs/reviews/2026-10-27-duration-aware-history-refit/summarize_session_contrasts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from continuum_swr_replay.paths import model_dir
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

OUT = Path(__file__).resolve().parent
# At-well stops only (lead's decision 2026-10-27). The shipped all-stops
# ``cross_history_models.pkl`` is a different cohort and is not compared.
COHORTS = {"at_well": ""}
#: name -> (simpler rung, richer rung, nested)
CONTRASTS: dict[str, tuple[str, str, bool]] = {
    "self_both": ("Mnohist_dur", "Mself_dur", True),
    "self_onset_only": ("Mnohist_dur", "Mself_onset_dur", True),
    "self_offset_only": ("Mnohist_dur", "Mself_offset_dur", True),
    "self_offset_adds_to_onset": ("Mself_onset_dur", "Mself_dur", True),
    "self_onset_adds_to_offset": ("Mself_offset_dur", "Mself_dur", True),
    "self_offset_vs_onset": ("Mself_onset_dur", "Mself_offset_dur", False),
    "cross_both": ("Mself_dur", "Mselfcross_dur", True),
    "cross_onset_only": ("Mself_dur", "Mselfcross_onset_dur", True),
    "cross_offset_only": ("Mself_dur", "Mselfcross_offset_dur", True),
    "cross_offset_vs_onset": ("Mselfcross_onset_dur", "Mselfcross_offset_dur", False),
}


def _aggregate(per_session: pd.DataFrame, nested: bool) -> dict:
    """``per_session``: animal, d_aic, d_k per session."""
    d = per_session["d_aic"].to_numpy(dtype=float)
    out = {
        "n_sessions": int(len(d)),
        "session_median": float(np.median(d)),
        "session_pct_positive": float(100 * np.mean(d > 0)),
        "animals": {},
    }
    n_pos = n_sig = 0
    for animal, g in per_session.groupby("animal"):
        med = float(g["d_aic"].median())
        rec = {"n_sessions": int(len(g)), "median": med}
        n_pos += med > 0
        if nested:
            lr = float((g["d_aic"] + 2 * g["d_k"]).sum())
            df = float(g["d_k"].sum())
            p = float(chi2.sf(lr, df))
            rec |= {"summed_lr": lr, "df": df, "p": p}
            n_sig += p < 0.05
        out["animals"][animal] = rec
    out["animals_positive_of_5"] = int(n_pos)
    if nested:
        out["animals_significant_of_5"] = int(n_sig)
    return out


def _contrasts(models: pd.DataFrame) -> dict:
    ok = models[models["fit_ok"]]
    keys = [*SESSION_COLUMNS, "outcome"]
    aic = ok.pivot_table(index=keys, columns="model", values="aic")
    k = ok.pivot_table(index=keys, columns="model", values="n_params")
    out: dict = {}
    for outcome in sorted(ok["outcome"].unique()):
        a = aic.xs(outcome, level="outcome")
        kk = k.xs(outcome, level="outcome")
        out[outcome] = {}
        for name, (simple, rich, nested) in CONTRASTS.items():
            if simple not in a or rich not in a:
                continue
            both = a[[simple, rich]].dropna()
            if both.empty:
                continue
            per = pd.DataFrame(
                {
                    "animal": both.index.get_level_values("animal"),
                    "d_aic": (both[simple] - both[rich]).to_numpy(),
                    "d_k": (
                        kk.loc[both.index, rich] - kk.loc[both.index, simple]
                    ).to_numpy(),
                }
            )
            out[outcome][name] = _aggregate(per, nested)
    return out


def _fit_counts(models: pd.DataFrame) -> dict:
    out = {}
    for (outcome, model), g in models.groupby(["outcome", "model"]):
        ok = g[g["fit_ok"]]
        out.setdefault(outcome, {})[model] = {
            "n_sessions": int(len(g)),
            "n_failed": int((~g["fit_ok"]).sum()),
            "n_separated_any_term": int((ok["separated_terms"] != "").sum()),
            "n_separated_history_term": int(
                ok["separated_terms"].str.contains("hist_rcos").sum()
            ),
        }
    return out


def main() -> None:
    result: dict = {"cohorts": {}}
    for cohort, suffix in COHORTS.items():
        path = model_dir("per_bin_rate") / f"duration_aware_history_models{suffix}.pkl"
        models = pd.read_pickle(path)
        result["cohorts"][cohort] = {
            "path": str(path),
            "n_sessions": int(models.groupby(list(SESSION_COLUMNS)).ngroups),
            "contrasts": _contrasts(models),
            "fits": _fit_counts(models),
        }
    (OUT / "session_contrasts.json").write_text(json.dumps(result, indent=1))
    for cohort, c in result["cohorts"].items():
        print(f"\n## {cohort} ({c['n_sessions']} sessions)")
        for outcome, cs in c["contrasts"].items():
            for name, s in cs.items():
                sig = s.get("animals_significant_of_5", "-")
                print(
                    f"  {outcome:17s} {name:27s} n={s['n_sessions']:3d} "
                    f"med={s['session_median']:+7.2f} pos={s['session_pct_positive']:5.1f}% "
                    f"animals+={s['animals_positive_of_5']} sig={sig}"
                )


if __name__ == "__main__":
    main()
