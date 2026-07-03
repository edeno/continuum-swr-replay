#!/usr/bin/env python
"""Fit the HMM-harmonized replay-CONTENT ladder (arm + patch), is_swr version.

The content analog of ``scripts/rate/fit_swr_contrast_rate.py``. Where the rate ladder
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

Aggregation is AUDITED. Per-session conditional-Poisson coefficients carry SE and
p-values. **Primary population evidence is at the animal level (n=5)**: the animal-
balanced median coef and per-animal sign consistency (``n_animals_pos`` / ``n_animals``).
The Wilcoxon signed-rank across sessions is reported as a **descriptive** consistency
statistic ONLY — sessions within an animal are not independent, so it is not a valid
population p-value — with a Benjamini-Hochberg FDR across the rung family per population.
``typical_session_se`` is the median within-session SE, not the aggregate's uncertainty.
A provenance sidecar (git hash, params) is written. DEFERRED follow-ups (in the
provenance): the **hierarchical Poisson with animal random effects** that would replace
the descriptive session signed-rank with rigorous population inference; a common
complete-case cohort; within-group-centered VIF (`fitting.py` uses the raw design); and
HMM per-arm ``q_arm`` (this runner still uses the beta-model per-arm value).

Usage:
    uv run python scripts/content/fit_swr_contrast_content.py               # full
    uv run python scripts/content/fit_swr_contrast_content.py --max-per-animal 2   # smoke
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import subprocess
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from continuum_swr_replay.analysis.replay._helpers import (
    add_arm_fixed_effects,
    add_uncertainty_covariates,
)
from continuum_swr_replay.analysis.replay.contrast_aggregation import (
    aggregate_contrast,
    specs_from_ladder,
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
    # time_since_stop is event-level (constant across arms/patches within an event);
    # carry it for the within-stop-time duration control (currentpatch_x_tsc).
    e2t = arm_ev[["event_id", "time_since_stop"]].drop_duplicates("event_id")
    pt = pt.merge(e2t, on="event_id", how="left")
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
    # within-stop-time duration control (z(log1p t) : is_current_patch), the patch
    # analog of the arm tsc_x_current — used to time-control the SWR patch contrast.
    pt["currentpatch_x_tsc"] = cp * _z(np.log1p(pt["time_since_stop"].to_numpy(float)))
    pt["lastrewpatch_x_tss"] = lr * pt["trials_since_switch_z"]
    pt["currentpatch_x_switchrate"] = cp * pt["recent_switch_rate_z"]
    pt["swr_x_current_patch"] = swr * cp
    pt["swr_x_lastrewpatch_tss"] = swr * lr * pt["trials_since_switch_z"]
    return pt


def _fit_session(
    g: pd.DataFrame, predictors: list[str], *, response_col: str, n_categories: int
) -> dict | None:
    required = [response_col, "event_id", *predictors]
    missing = [c for c in required if c not in g.columns]
    if missing:
        return fit_conditional_poisson(
            g,
            predictors=predictors,
            response_col=response_col,
            event_col="event_id",
            n_categories=n_categories,
        )

    # Preserve the full candidate set within every retained event. Row-wise
    # NaN drops can leave an event with fewer than 9 arms / 3 patches, changing
    # the conditional likelihood while the null still assumes n_categories.
    events_with_nan = g.loc[g[required].isna().any(axis=1), "event_id"].unique()
    d = g[~g["event_id"].isin(events_with_nan)]
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
                        **{
                            k: v
                            for k, v in r.items()
                            if k.startswith(("coef_", "bse_", "pval_"))
                        },
                        "n_events": r["n_events"],
                        "n_params": r["n_params"],
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


def _aggregate(
    res: pd.DataFrame, ladder: dict, key_term: dict, level: str, sd: dict | None = None
) -> pd.DataFrame:
    """Audited across-session aggregation (shared engine); one row per rung x pop.

    See :mod:`continuum_swr_replay.analysis.replay.contrast_aggregation` for the
    column semantics (animal-level primary, session signed-rank descriptive, BH-FDR,
    standardized per-SD columns). This wrapper just expands the content ladder into
    (model, population, term) specs so the content and rate ladders mirror.
    """
    return aggregate_contrast(res, specs_from_ladder(ladder, key_term), level, sd)


def _provenance(args: argparse.Namespace, n_sessions: int) -> dict:
    """Reproducibility record saved alongside the audited results."""
    try:
        git_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        git_hash = "unknown"
    return {
        "git_hash": git_hash,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_sessions": int(n_sessions),
        "smoke_max_per_animal": args.max_per_animal,
        "inference": "per-session conditional Poisson; across-session Wilcoxon "
        "signed-rank across sessions is DESCRIPTIVE only (sessions within an animal "
        "are not independent); primary population evidence is per-animal sign "
        "consistency (n_animals_pos / n_animals, n=5). BH-FDR across the rung family "
        "per population is applied to the descriptive session signed-rank p.",
        "q_arm_source": "beta-model per-arm value (HMM q_arm harmonization deferred)",
        "deferred": [
            "mixed conditional logit with animal random effects (the rigorous "
            "population inference; supersedes the descriptive session signed-rank p). "
            "NOT available in statsmodels (ConditionalPoisson takes no random effects "
            "or cluster-robust cov); realistic path is NumPyro (JAX already a dep). "
            "Caveat: the animal random-effect variance is estimated from n=5 groups, "
            "so this does honest bookkeeping, not a power upgrade.",
            "common complete-case cohort across rungs",
            "within-group-centered VIF (fitting.py:420 uses the raw design)",
            "HMM per-arm q_arm",
        ],
    }


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

    # Audited aggregation: SE, per-session significance, per-animal sign, signed-rank
    # p, BH-FDR across the rung family, and standardized (per-SD) effect sizes — plus a
    # provenance sidecar. SDs are the global (pooled) SD of the raw per-arm covariates.
    sd_arm = {
        c: float(np.nanstd(ev[c].to_numpy(dtype=float)))
        for c in ("q_arm", "q_arm_hmm", "reward_rate_arm_k10", "p_reward_arm")
        if c in ev.columns
    }
    agg = pd.concat(
        [
            _aggregate(arm, ARM_LADDER, ARM_KEY_TERM, "arm", sd_arm),
            _aggregate(patch, PATCH_LADDER, PATCH_KEY_TERM, "patch", {}),
        ],
        ignore_index=True,
    )
    agg.to_pickle(RC / "models" / f"swr_contrast_content_audited{sfx}.pkl")
    prov = _provenance(args, ev.groupby(SC).ngroups)
    (RC / "models" / f"swr_contrast_content_provenance{sfx}.json").write_text(
        json.dumps(prov, indent=2)
    )
    logger.info("saved audited table (%d rows) + provenance -> %s", len(agg), RC / "models")
    print(
        "\n########## AUDITED  (animal-level = primary; session signed-rank = descriptive) ##########"
    )
    cols = [
        "level", "model", "population", "term", "n_sessions", "n_animals",
        "coef_animal_median", "n_animals_pos", "coef_session_median",
        "typical_session_se", "pct_sig_session", "session_signrank_p",
        "session_signrank_fdr_q",
    ]
    with pd.option_context("display.max_rows", None, "display.width", 220):
        print(agg[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
