#!/usr/bin/env python
"""Fit the per-bin non-local rate models, `is_swr` version (see docs/models_table.md).

Two ladders on the canonical (stop x bin) design from `aggregate_to_timebins`
(15 ms bins, `log(exposure)` offset, Poisson), fit per session:

  * STRATIFIED (G0-G8): each spec fit to `non_local`, `non_local_swr`,
    `non_local_no_swr` separately. Comparing a coefficient across the three
    populations = the descriptive SWR-vs-non-SWR dissociation.
  * is_swr CONTRAST (S0-S10, design B): the `non_local_swr` and `non_local_no_swr`
    per-bin designs are STACKED (identical (stop,bin) rows + covariates, differing
    only in `count`), tagged by `is_swr`, and `is_swr:X` interactions test whether a
    task effect differs by SWR status. Self-history (`hist_rcos_*`) is stream-specific
    only in its raw per-stream values; it is still wrapped inside `is_swr * BASE`
    (S2-S10), so `is_swr:hist_rcos_k` refractoriness-by-SWR interactions ARE fit.
    Time/reward/task covariates are shared across the two rows of a bin. The within-bin
    pairing is carried by the per-session fit + across-session aggregation (no GLMM),
    matching the rest of Family B.

Covariate blocks BASE/PERISWITCH/BLOCK/SWITCHRATE/VALUE/RPE/UNCERT/UPDATE/REWXTIME
are defined below. The task
dimensions are the paper's (Comrie et al.), all from the beta-bernoulli model:
relative value (`switch_advantage_mu_preoutcome_sess`; Stay-vs-Switch ΔV), RPE
(`rpe` = `reward - q_chosen`), belief entropy (`beta_reward_entropy`, mean per-leaf
Beta differential entropy), global value update (`global_mu_update_abs_postoutcome_sess`);
+ CONTEXT (peri-switch/
block) from `qvals_table` and `recent_switch_rate` from `add_uncertainty_covariates`.
One tidy row per (population, model, session), each carrying per-term coef + SE
(``bse_*``) + p-value (``pval_*``); aggregate across sessions downstream (per-animal
sign consistency primary, session signed-rank descriptive — see the content runner).

Usage:
    uv run python scripts/rate/fit_swr_contrast_rate.py --max-per-animal 1   # smoke (~4 min)
    uv run python scripts/rate/fit_swr_contrast_rate.py --workers 16         # full
"""

from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from patsy import PatsyError  # noqa: E402
from statsmodels.tools.sm_exceptions import PerfectSeparationError  # noqa: E402

from continuum_swr_replay.analysis.replay import (  # noqa: E402
    BIN_S,
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    N_REWARD_LAGS,
    POPULATIONS,
    TMAX,
    add_uncertainty_covariates,
    aggregate_to_timebins,
    attach_is_swr_to_onsets,
)
from continuum_swr_replay.analysis.replay._fit_diagnostics import (  # noqa: E402
    capture_fit_warnings,
    count_fit_quality_warnings,
    fit_status,
)
from continuum_swr_replay.analysis.replay.cohort_audit import (  # noqa: E402
    cohort_record,
    stop_own_missing,
    term_diagnostics,
    write_cohort_audit,
)
from continuum_swr_replay.analysis.replay.contrast_aggregation import (  # noqa: E402
    aggregate_contrast,
)
from continuum_swr_replay.analysis.replay.covariate_timing import (  # noqa: E402
    assert_predictor_timing,
)
from continuum_swr_replay.analysis.replay.model_specs import (  # noqa: E402
    DESCRIPTIVE_RATE_RUNGS,
)
from continuum_swr_replay.analysis.replay.model_specs import (  # noqa: E402
    SWR_CONTRAST_RATE_G_SPECS as G_SPECS,
)
from continuum_swr_replay.analysis.replay.model_specs import (  # noqa: E402
    SWR_CONTRAST_RATE_S_SPECS as S_SPECS,
)
from continuum_swr_replay.analysis.replay.model_specs import (  # noqa: E402
    SWR_CONTRAST_RATE_STOP_COVARS as STOP_COVARS,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR  # noqa: E402

# Only the benign exp()-overflow RuntimeWarning (quasi-separated coefs -> rate ratios)
# is silenced. Convergence/separation are recorded per fit as flags inside _fit (see
# analysis.replay._fit_diagnostics), not blanket-ignored.
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

SESSION_COLS = ["animal", "day", "epoch"]

# Audited-aggregation key terms, mirroring the content ladder. G-rungs add one
# dimension each (fit per population); S-rungs carry the is_swr:<dimension> contrast.
_G_KEY = {
    "G0_base": "reward_lag1", "G1_periswitch": "trials_since_switch",
    "G2_block": "trials_since_block_change", "G3_switchrate": "recent_switch_rate",
    "G4_value": "switch_advantage_mu_preoutcome_sess", "G5_rpe": "rpe",
    "G6_uncert": "beta_reward_entropy", "G7_update": "global_mu_update_abs_postoutcome_sess",
}
_S_KEY = {
    "S1_rew1": "is_swr:reward_lag1", "S3_periswitch": "is_swr:trials_since_switch",
    "S4_block": "is_swr:trials_since_block_change",
    "S5_switchrate": "is_swr:recent_switch_rate",
    "S6_value": "is_swr:switch_advantage_mu_preoutcome_sess", "S7_rpe": "is_swr:rpe",
    "S8_uncert": "is_swr:beta_reward_entropy",
    "S9_update": "is_swr:global_mu_update_abs_postoutcome_sess",
}
# raw fit population -> friendly label shared with the content table
_POP_LABEL = {"non_local": "all", "non_local_swr": "SWR", "non_local_no_swr": "nonSWR"}
_RATE_CONT_COVARS = [
    "trials_since_switch", "trials_since_block_change", "recent_switch_rate",
    "switch_advantage_mu_preoutcome_sess", "rpe", "beta_reward_entropy",
    "global_mu_update_abs_postoutcome_sess",
]
RC = PROCESSED_DATA_DIR / "replay_content"
STRATA = ("non_local", "non_local_swr", "non_local_no_swr")
MIN_BINS, MIN_EVENTS = 200, 20
_EXPECTED = (
    np.linalg.LinAlgError,
    ValueError,
    RuntimeError,
    PerfectSeparationError,
    PatsyError,
)


def _fit(
    design: pd.DataFrame, formula: str, need_cols: list[str]
) -> tuple[dict | None, str | None]:
    """Poisson GLM `count ~ formula` with a log(exposure) offset. NaN-drop then
    offset (mirrors fit_per_bin_rate_model to keep the offset aligned).

    Returns ``(row, reason)``: ``reason`` is ``None`` on success, else a short tag
    for the skip -- ``missing:<cols>`` (a needed column is absent -> a config/upstream
    bug, distinct from a data skip), ``too_few``, ``no_reward_contrast``, or the
    caught exception's class name. The caller tallies these so drop-outs are visible
    (a silently-shrunk per-model session count would bias the downstream aggregate).
    """
    missing = [c for c in need_cols if c not in design.columns]
    if missing:
        return None, f"missing:{','.join(missing)}"
    df = design.dropna(subset=[*need_cols, "exposure"])
    df = df[df["exposure"] > 0]
    n_events = int((df["count"] > 0).sum())
    if len(df) < MIN_BINS or n_events < MIN_EVENTS:
        return None, "too_few"
    if "reward_lag1" in need_cols and df["reward_lag1"].nunique() < 2:
        return None, "no_reward_contrast"  # reward terms unidentified without contrast
    offset = np.log(df["exposure"].to_numpy(dtype=float))
    try:
        with capture_fit_warnings() as caught:
            res = smf.glm(
                f"count ~ {formula}", data=df, family=sm.families.Poisson(), offset=offset
            ).fit()
    except _EXPECTED as e:
        return None, type(e).__name__
    status = fit_status(res, "glm")
    out = {f"coef_{k}": v for k, v in res.params.items()}
    # Within-session coefficient uncertainty: carry SE + p per term so downstream
    # aggregation can weight/annotate (mirrors the content contrast runner). These
    # are model-based (non-robust) SEs; clustered/robust inference is a follow-up.
    out.update({f"bse_{k}": v for k, v in res.bse.items()})
    out.update({f"pval_{k}": v for k, v in res.pvalues.items()})
    out.update(
        n_bins=int(res.nobs), n_events=n_events, llf=float(res.llf), aic=float(res.aic),
        n_params=int(len(res.params)),
        # distinct stops that contributed >=1 retained bin -- the stop-grain modeled count.
        n_stops=int(df["stop_id"].nunique()),
        converged=status["converged"], separated=status["separated"],
        n_convergence_warnings=count_fit_quality_warnings(caught),
    )
    return out, None


def _fit_one_session(keys, stops_g, onsets_g) -> tuple[tuple, list[dict], dict]:
    rows: list[dict] = []
    skips: dict = {}  # (model, population, reason) -> count
    designs = {
        pop: aggregate_to_timebins(
            stops_g,
            POPULATIONS[pop](onsets_g),
            bin_s=BIN_S,
            tmax=TMAX,
            reward_lags=N_REWARD_LAGS,
            self_history_lags=N_HISTORY_LAGS,
            self_history_basis=N_HISTORY_BASIS,
            stop_covariates=STOP_COVARS,
        )
        for pop in STRATA
    }
    base = dict(zip(SESSION_COLS, keys, strict=True))
    # A. stratified
    for pop in STRATA:
        for name, (formula, need) in G_SPECS.items():
            fit, reason = _fit(designs[pop], formula, need)
            if fit is not None:
                rows.append({"population": pop, "model": name, **base, **fit})
            else:
                skips[(name, pop, reason)] = skips.get((name, pop, reason), 0) + 1
    # B. is_swr contrast: stack the two non-local streams
    stacked = pd.concat(
        [
            designs["non_local_swr"].assign(is_swr=1.0),
            designs["non_local_no_swr"].assign(is_swr=0.0),
        ],
        ignore_index=True,
    )
    for name, (formula, need) in S_SPECS.items():
        fit, reason = _fit(stacked, formula, [*need, "is_swr"])
        if fit is not None:
            rows.append({"population": "contrast", "model": name, **base, **fit})
        else:
            skips[(name, "contrast", reason)] = (
                skips.get((name, "contrast", reason), 0) + 1
            )
    return keys, rows, skips


def _load() -> tuple[pd.DataFrame, pd.DataFrame, int]:
    stop = add_uncertainty_covariates(pd.read_pickle(RC / "stop_table.pkl"))
    q = pd.read_pickle(RC / "models_per_bin_rate" / "qvals_table.pkl")
    qcols = [
        "stop_id",
        "trials_since_switch",
        "is_post_switch",
        "trials_since_block_change",
        "switch_advantage_mu_preoutcome_sess",
        "rpe",
        "beta_reward_entropy",
        "global_mu_update_abs_postoutcome_sess",
    ]
    missing = [c for c in qcols if c not in q.columns]
    if missing:
        raise ValueError(
            f"qvals_table missing {missing}; regenerate it "
            "(python -m continuum_swr_replay.analysis.behavior.qvals_table --write)"
        )
    n0 = len(stop)
    stop = stop.merge(q[qcols], on="stop_id", how="inner", validate="many_to_one")
    # fail loudly if a task covariate the specs need never made it onto the stop
    # table (rename / add_uncertainty_covariates / qvals regression) -- otherwise
    # every model using it would silently drop out of the output.
    missing_cov = [c for c in STOP_COVARS if c not in stop.columns]
    if missing_cov:
        raise ValueError(
            f"stop table missing task covariates {missing_cov} "
            "(check add_uncertainty_covariates + the qvals merge)"
        )
    logger.info(
        "stop x qvals join: %d -> %d stops (%.0f%% kept)",
        n0,
        len(stop),
        100 * len(stop) / n0,
    )
    onsets = attach_is_swr_to_onsets(
        pd.read_pickle(RC / "event_onset_table.pkl"),
        pd.read_pickle(RC / "event_table.pkl"),
    )
    # Stops dropped by the inner qvals join are the mechanism-B "qvals gap" (no fitted
    # HMM/value state) -- reported per-model as n_drop_qvals_gap in the cohort audit.
    n_qvals_dropped = int(n0 - len(stop))
    return stop, onsets, n_qvals_dropped


def _subset_max_per_animal(stop: pd.DataFrame, n: int) -> pd.DataFrame:
    keep = stop[SESSION_COLS].drop_duplicates().groupby("animal", sort=False).head(n)
    return stop.merge(keep, on=SESSION_COLS)


def _cohort_records(
    models: pd.DataFrame, stop: pd.DataFrame, n_qvals_dropped: int
) -> list[dict]:
    """One stop-grain cohort record per (model, population).

    Grain is ``stop`` (the plan audits the rate family at stop grain — the timebin
    design is aggregated per-session inside the fitter). Buckets:
    ``n_drop_qvals_gap`` = the inner qvals-join drop (mechanism B, stops with no fitted
    value state, already removed in ``_load``); ``n_drop_own_predictor_missing`` =
    post-join stops with ``immobility<=0`` or a NaN stop-level required predictor
    (e.g. ``reward_lag1``, model-specific); ``n_modeled`` = the fitter's distinct
    modeled-stop count (``n_stops``); the rest (bin-level drops, session gates) go to
    ``n_excluded_session``. Reconciles by construction.
    """
    total = int(len(stop))  # post-join stops
    n_sessions_in = stop.groupby(SESSION_COLS, sort=False).ngroups
    need_by_model = {
        **{m: need for m, (_f, need) in G_SPECS.items()},
        **{m: need for m, (_f, need) in S_SPECS.items()},
    }
    pairs = [(m, p) for m in G_SPECS for p in STRATA]
    pairs += [(m, "contrast") for m in S_SPECS]
    records = []
    for name, pop in pairs:
        sub = (
            models[(models.model == name) & (models.population == pop)]
            if len(models) else models
        )
        fit_keys = {tuple(r) for r in sub[SESSION_COLS].to_numpy()} if len(sub) else set()
        n_modeled = int(sub["n_stops"].sum()) if "n_stops" in sub else 0
        stop_level = [c for c in need_by_model.get(name, ()) if c in stop.columns]
        n_own = int(stop_own_missing(stop, stop_level).sum())
        records.append(cohort_record(
            name, _POP_LABEL.get(pop, pop), "stop",
            n_in=total + int(n_qvals_dropped), n_modeled=n_modeled,
            sessions_in=n_sessions_in, sessions_modeled=len(fit_keys),
            animals_modeled=len({k[0] for k in fit_keys}),
            drop_notes=(
                "qvals-join drop -> qvals_gap (mechanism B); immobility<=0 / NaN "
                "stop-level predictor -> own; bin-level + session gates -> excluded"
            ),
            n_drop_qvals_gap=int(n_qvals_dropped),
            n_drop_own_predictor_missing=n_own,
            n_excluded_session=total - n_own - n_modeled,
            n_sessions_flagged_nonconverged=(
                int((sub["converged"] == False).sum()) if "converged" in sub else 0  # noqa: E712
            ),
            n_sessions_flagged_separated=(
                int((sub["separated"] == True).sum()) if "separated" in sub else 0  # noqa: E712
            ),
        ))
    return records


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--max-per-animal", type=int, default=None, help="smoke: N sessions/animal"
    )
    ap.add_argument(
        "--workers", type=int, default=1, help="per-session process fan-out"
    )
    args = ap.parse_args()

    # Causal-hygiene gate (phase 4): each rung's covariates must be pre-outcome, or the
    # rung is an explicitly-declared descriptive rung (RPE/value association or peri-switch
    # alignment). A NEW leaky covariate on an undeclared rung fails here. S-rungs carry
    # is_swr as a gating modifier (a SAFE_MODIFIER), so asserting `need` covers the
    # is_swr:<covariate> expansion.
    for specs, descr in ((G_SPECS, DESCRIPTIVE_RATE_RUNGS), (S_SPECS, DESCRIPTIVE_RATE_RUNGS)):
        for name, (_formula, need) in specs.items():
            assert_predictor_timing(need, descriptive=(name in descr))

    stop, onsets, n_qvals_dropped = _load()
    if args.max_per_animal is not None:
        stop = _subset_max_per_animal(stop, args.max_per_animal)

    onset_groups = dict(tuple(onsets.groupby(SESSION_COLS, sort=False)))
    items = [
        (k, s, onset_groups[k])
        for k, s in stop.groupby(SESSION_COLS, sort=False)
        if k in onset_groups
    ]
    logger.info(
        "Fitting SWR-contrast per-bin models (G0-G8 x %s + S0-S10) over %d sessions "
        "at BIN_S=%.3f with %d worker(s)...",
        list(STRATA),
        len(items),
        BIN_S,
        args.workers,
    )

    t0 = time.perf_counter()
    rows: list[dict] = []
    skips: Counter = Counter()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_fit_one_session, k, s, o) for k, s, o in items]
            for fut in as_completed(futs):
                _, r, sk = fut.result()
                rows.extend(r)
                skips.update(sk)
    else:
        for k, s, o in items:
            _, r, sk = _fit_one_session(k, s, o)
            rows.extend(r)
            skips.update(sk)
            logger.info("  %s: %d fits", "_".join(map(str, k)), len(r))

    # log skipped (model, population, reason) cells so a silently-shrunk per-model
    # session count is visible before the downstream signed-rank aggregate.
    if skips:
        cfg = {
            f"{m}/{p}": r
            for (m, p, r), _ in skips.items()
            if str(r).startswith("missing")
        }
        if cfg:
            logger.warning("MISSING-COLUMN skips (config/upstream bug): %s", cfg)
        logger.info(
            "skipped (model/pop -> reason: count): %s",
            {f"{m}/{p}:{r}": c for (m, p, r), c in sorted(skips.items())},
        )

    models = pd.DataFrame(rows)
    out_dir = RC / "models_per_bin_rate"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (
        "swr_contrast_rate_smoke.pkl"
        if args.max_per_animal
        else "swr_contrast_rate.pkl"
    )
    models.to_pickle(out)

    # Cohort-audit table (stop grain; reconciles n_in - Σdrops == n_modeled) +
    # term-level quasi-separation diagnostics, mirroring the content contrast runner.
    csfx = "_smoke" if args.max_per_animal else ""
    # The qvals-join drop is measured on the full stop table in _load(); in smoke mode
    # the per-animal subset happens after, so attributing the full-data drop against a
    # tiny subset n_in would be meaningless. Report 0 there (smoke is not a reported run).
    audit_qvals_drop = 0 if args.max_per_animal else n_qvals_dropped
    write_cohort_audit(
        _cohort_records(models, stop, audit_qvals_drop),
        out_dir / f"swr_contrast_rate_cohort_audit{csfx}.csv",
    )
    if len(models):
        td_specs = [(g, p, t) for g, t in _G_KEY.items() for p in STRATA]
        td_specs += [(s, "contrast", t) for s, t in _S_KEY.items()]
        term_diagnostics(models, td_specs).to_csv(
            out_dir / f"swr_contrast_rate_term_diagnostics{csfx}.csv", index=False
        )
    logger.info(
        "Saved %d rows (%d sessions, populations=%s, models=%s) to %s in %.1fs",
        len(models),
        models.groupby(SESSION_COLS).ngroups if len(models) else 0,
        sorted(models["population"].unique()) if len(models) else [],
        sorted(models["model"].unique()) if len(models) else [],
        out,
        time.perf_counter() - t0,
    )

    if len(models):
        # Audited aggregation mirroring the content ladder (per-animal primary,
        # session signed-rank, BH-FDR, standardized per-SD columns). G-rungs are
        # relabeled non_local/_swr/_no_swr -> all/SWR/nonSWR to match the content table.
        qv = pd.read_pickle(out_dir / "qvals_table.pkl")
        st = add_uncertainty_covariates(pd.read_pickle(RC / "stop_table.pkl"))
        sd = {}
        for c in _RATE_CONT_COVARS:
            src = qv if c in qv.columns else st if c in st.columns else None
            if src is not None:
                sd[c] = float(np.nanstd(src[c].to_numpy(dtype=float)))
        specs = [(g, p, t) for g, t in _G_KEY.items() for p in _POP_LABEL]
        specs += [(s, "contrast", t) for s, t in _S_KEY.items()]
        agg = aggregate_contrast(models, specs, "rate", sd)
        agg["population"] = agg["population"].map(lambda p: _POP_LABEL.get(p, p))
        asfx = "_smoke" if args.max_per_animal else ""
        agg.to_pickle(out_dir / f"swr_contrast_rate_audited{asfx}.pkl")
        # Persist the exact per-SD factors this run used for the std_* columns so the
        # standardized coefficients are reproducible. These SDs are computed over the
        # FULL qvals/stop tables (re-read above), not the per-model fitted cohort -- the
        # two differ negligibly on full data (~0.2% for recent_switch_rate). This is the
        # runner's own reference; it is NOT the builder's pooled-global table _z.
        (out_dir / f"swr_contrast_rate_meta{asfx}.json").write_text(json.dumps(
            {"fit_time_sd_reference": {k: float(v) for k, v in sd.items()},
             "sd_scope": "full qvals_table + stop_table (std_factor = raw_coef * SD)",
             "note": "the exact SD factors aggregate_contrast used for the std_* columns; "
                     "NOT the builder's pooled-global table _z"},
            indent=2, sort_keys=True,
        ))
        logger.info("Saved audited rate table (%d rows) + meta -> %s", len(agg), out_dir)


if __name__ == "__main__":
    main()
