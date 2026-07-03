#!/usr/bin/env python
"""Fit the HMM-harmonized replay-CONTENT ladder (arm + patch), is_swr version.

The content analog of ``scripts/fit_swr_contrast_rate.py``. Where the rate ladder
models HOW OFTEN non-local events occur (Poisson count per time bin), this models
WHICH ARM / PATCH an event represents: a conditional Poisson over the 9 arms (or 3
patches) within each event (``strata(event_id)`` ≡ clogit). Design spec:
``docs/design/content_model_design.md``.

Structural rule: in a conditional-over-arms likelihood a covariate CONSTANT within
an event cancels, so all event-level scalars (reward, rpe, value, entropy, context,
is_swr) enter ONLY as interactions with arm-features (``:is_current``). Per-arm
covariates (``reward_rate_arm_k10``, ``q_arm``, ``is_last_rewarded_elsewhere``) enter
as main effects.

Ladders (arm; fit per session, per is_swr split {all, SWR, non-SWR}):
  C0..C2  spatial baseline (ARMFE -> is_current -> +autocorrelation controls)
  C3..C5  per-arm value/reward CONTENT (main effects)
  C6..C9  state modulation of current-bias (rpe/value/uncert/update : is_current)
  C10..C12 CONTEXT split (periswitch/block/switchrate : is_current)
  CS1..CS5 is_swr:feature contrasts (pooled, 2-way; the dissociation)
Event-scalar dissociations (C6-C12) come from comparing the split, not a pooled 3-way.

Aggregation: across-session median coef + Wilcoxon signed-rank + per-animal sign
consistency (n/5), identical to the rate side. NOTE: ``q_arm`` here is the existing
(beta-model) per-arm value; HMM per-arm value harmonization is a deferred follow-up.

Usage:
    uv run python scripts/fit_swr_contrast_content.py               # full
    uv run python scripts/fit_swr_contrast_content.py --max-per-animal 2   # smoke
"""

from __future__ import annotations

import argparse
import logging
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from continuum_swr_replay.analysis.replay._helpers import (
    add_arm_fixed_effects,
    add_uncertainty_covariates,
)
from continuum_swr_replay.analysis.replay.fitting import fit_conditional_poisson
from continuum_swr_replay.analysis.replay.model_specs import (
    SWR_CONTRAST_CONTENT_ARM_KEY_TERM as ARM_KEY_TERM,
)
from continuum_swr_replay.analysis.replay.model_specs import (
    SWR_CONTRAST_CONTENT_ARM_LADDER as ARM_LADDER,
)
from continuum_swr_replay.analysis.replay.model_specs import (
    SWR_CONTRAST_CONTENT_PATCH_KEY_TERM as PATCH_KEY_TERM,
)
from continuum_swr_replay.analysis.replay.model_specs import (
    SWR_CONTRAST_CONTENT_PATCH_LADDER as PATCH_LADDER,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RC = PROCESSED_DATA_DIR / "replay_content"
SC = ["animal", "day", "epoch"]
MIN_EVENTS = 18


# event-level scalars z-scored (so :is_current coefs are per-SD & comparable)
_ZSCALARS = [
    "hmm_rpe",
    "hmm_relative_value",
    "reward_entropy",
    "hmm_value_update",
    "trials_since_switch",
    "trials_since_block_change",
    "recent_switch_rate",
]


def _z(x: np.ndarray) -> np.ndarray:
    s = np.nanstd(x)
    return (x - np.nanmean(x)) / s if s > 0 else x * 0.0


def _assemble_arm() -> pd.DataFrame:
    ev = pd.read_pickle(RC / "event_arm_table.pkl")
    add_arm_fixed_effects(ev)
    stop = add_uncertainty_covariates(pd.read_pickle(RC / "stop_table.pkl"))
    q = pd.read_pickle(RC / "models_per_bin_rate" / "qvals_table.pkl")
    ev = ev.merge(
        stop[["stop_id", "reward_lag1", "recent_switch_rate"]], on="stop_id", how="left"
    )
    ev = ev.merge(
        q[
            [
                "stop_id",
                "hmm_rpe",
                "hmm_relative_value",
                "reward_entropy",
                "hmm_value_update",
                "trials_since_switch",
                "is_post_switch",
                "trials_since_block_change",
            ]
        ],
        on="stop_id",
        how="left",
    )
    for c in _ZSCALARS:
        ev[c + "_z"] = _z(ev[c].to_numpy(dtype=float))
    cur = ev["is_current"].to_numpy(dtype=float)
    swr = ev["is_swr"].to_numpy(dtype=float)
    # state-modulation interactions (event-scalar_z : is_current)
    ev["rpe_x_current"] = cur * ev["hmm_rpe_z"]
    ev["relval_x_current"] = cur * ev["hmm_relative_value_z"]
    ev["entropy_x_current"] = cur * ev["reward_entropy_z"]
    ev["update_x_current"] = cur * ev["hmm_value_update_z"]
    # context interactions
    ev["tss_x_current"] = cur * ev["trials_since_switch_z"]
    ev["postsw_x_current"] = cur * ev["is_post_switch"].to_numpy(dtype=float)
    ev["tsspostsw_x_current"] = (
        cur * ev["trials_since_switch_z"] * ev["is_post_switch"].to_numpy(dtype=float)
    )
    ev["block_x_current"] = cur * ev["trials_since_block_change_z"]
    ev["switchrate_x_current"] = cur * ev["recent_switch_rate_z"]
    # within-stop-time duration control (z(log1p t) : is_current)
    tsc = np.log1p(ev["time_since_stop"].to_numpy(dtype=float))
    ev["tsc_x_current"] = cur * _z(tsc)
    # is_swr:q_arm contrast (the one swr_x_* not pre-materialized)
    ev["swr_x_q_arm"] = swr * ev["q_arm"].to_numpy(dtype=float)
    return ev


def _assemble_patch(arm_ev: pd.DataFrame) -> pd.DataFrame:
    pt = pd.read_pickle(RC / "event_patch_table.pkl")
    # patch table lacks stop_id: map event_id -> stop_id from the arm table
    e2s = arm_ev[["event_id", "stop_id"]].drop_duplicates("event_id")
    pt = pt.merge(e2s, on="event_id", how="left")
    stop = add_uncertainty_covariates(pd.read_pickle(RC / "stop_table.pkl"))
    q = pd.read_pickle(RC / "models_per_bin_rate" / "qvals_table.pkl")
    pt = pt.merge(stop[["stop_id", "recent_switch_rate"]], on="stop_id", how="left")
    pt = pt.merge(q[["stop_id", "trials_since_switch"]], on="stop_id", how="left")
    pt["trials_since_switch_z"] = _z(pt["trials_since_switch"].to_numpy(float))
    pt["recent_switch_rate_z"] = _z(pt["recent_switch_rate"].to_numpy(float))
    cp = pt["is_current_patch"].to_numpy(float)
    lr = pt["is_last_rewarded_patch_elsewhere"].to_numpy(float)
    swr = pt["is_swr"].to_numpy(float)
    pt["currentpatch_x_tss"] = cp * pt["trials_since_switch_z"]
    pt["lastrewpatch_x_tss"] = lr * pt["trials_since_switch_z"]
    pt["currentpatch_x_switchrate"] = cp * pt["recent_switch_rate_z"]
    pt["swr_x_current_patch"] = swr * cp
    pt["swr_x_lastrewpatch_tss"] = swr * lr * pt["trials_since_switch_z"]
    return pt


def _fit_session(
    g: pd.DataFrame, predictors: list[str], *, response_col: str, n_categories: int
) -> dict | None:
    d = g.dropna(subset=predictors)
    if d["event_id"].nunique() < MIN_EVENTS:
        return None
    return fit_conditional_poisson(
        d,
        predictors=predictors,
        response_col=response_col,
        event_col="event_id",
        n_categories=n_categories,
    )


def _run(
    ev: pd.DataFrame, ladder: dict, *, response_col: str, n_categories: int
) -> pd.DataFrame:
    rows = []
    splits = {"all": ev, "SWR": ev[ev.is_swr], "nonSWR": ev[~ev.is_swr]}
    for name, (preds, scope) in ladder.items():
        pops = ["all"] if scope == "pooled" else ["all", "SWR", "nonSWR"]
        for pop in pops:
            for keys, g in splits[pop].groupby(SC, sort=False):
                r = _fit_session(
                    g, preds, response_col=response_col, n_categories=n_categories
                )
                if r is None:
                    continue
                rows.append(
                    {
                        "model": name,
                        "population": pop,
                        **dict(zip(SC, keys, strict=True)),
                        **{k: v for k, v in r.items() if k.startswith("coef_")},
                        "n_events": r["n_events"],
                    }
                )
    return pd.DataFrame(rows)


def _summ(res: pd.DataFrame, ladder: dict, key_term: dict) -> None:
    print(
        f"\n{'model':18s}{'pop':8s}{'term':30s}{'n':>4}{'medCoef':>9}"
        f"{'%pos':>6}{'anim':>6}{'signed-rank p':>14}"
    )
    for name in ladder:
        term = key_term.get(name)
        if term is None:
            continue
        col = f"coef_{term}"
        scope = ladder[name][1]
        for pop in ["all"] if scope == "pooled" else ["all", "SWR", "nonSWR"]:
            s = res[(res.model == name) & (res.population == pop)]
            if col not in s.columns:
                continue
            x = s[col].dropna()
            x = x[x.abs() < 10]  # drop quasi-separated sessions (coefs are O(1))
            if len(x) < 5:
                continue
            nan = (s.groupby("animal")[col].median() > 0).sum()
            p = wilcoxon(x)[1] if len(x) > 5 else np.nan
            star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""
            print(
                f"{name:18s}{pop:8s}{term:30s}{len(x):4d}{x.median():+9.3f}"
                f"{100 * (x > 0).mean():5.0f}%{f'{nan}/5':>6}{p:14.2e} {star}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--max-per-animal",
        type=int,
        default=None,
        help="smoke: keep first N sessions per animal",
    )
    args = ap.parse_args()

    sfx = "_smoke" if args.max_per_animal else ""
    (RC / "models").mkdir(parents=True, exist_ok=True)
    ev = _assemble_arm()
    pt = _assemble_patch(ev)
    if args.max_per_animal is not None:
        keep = (
            ev[SC]
            .drop_duplicates()
            .groupby("animal", sort=False)
            .head(args.max_per_animal)
        )
        ev = ev.merge(keep, on=SC)
        pt = pt.merge(keep, on=SC)

    logger.info(
        "ARM ladder: %d rows, %d events, %d sessions",
        len(ev),
        ev["event_id"].nunique(),
        ev.groupby(SC).ngroups,
    )
    arm = _run(ev, ARM_LADDER, response_col="arm_count_hard", n_categories=9)
    arm.to_pickle(RC / "models" / f"swr_contrast_content_arm{sfx}.pkl")

    logger.info(
        "PATCH ladder: %d rows, %d events, %d sessions",
        len(pt),
        pt["event_id"].nunique(),
        pt.groupby(SC).ngroups,
    )
    patch = _run(pt, PATCH_LADDER, response_col="patch_count_hard", n_categories=3)
    patch.to_pickle(RC / "models" / f"swr_contrast_content_patch{sfx}.pkl")

    logger.info("saved arm=%d patch=%d rows -> %s", len(arm), len(patch), RC / "models")
    print("\n########## ARM ##########")
    _summ(arm, ARM_LADDER, ARM_KEY_TERM)
    print("\n########## PATCH ##########")
    _summ(patch, PATCH_LADDER, PATCH_KEY_TERM)


if __name__ == "__main__":
    main()
