#!/usr/bin/env python
"""Fit the per-bin non-local rate models, `is_swr` version (see docs/models_table.md).

Two ladders on the canonical (stop x bin) design from `aggregate_to_timebins`
(15 ms bins, `log(exposure)` offset, Poisson), fit per session. The design is the
at-risk one shared with `fit_per_bin_rate_models.py`: a bin inside an ongoing
non-local event is dropped, and self-history is counted from event ends
(`PER_BIN_HISTORY_ANCHORS`).

  * STRATIFIED (G0-G8): each spec fit to `non_local`, `non_local_swr`,
    `non_local_no_swr` separately. Comparing a coefficient across the three
    populations = the descriptive SWR-vs-non-SWR dissociation.
  * is_swr CONTRAST (S0-S10, design B): the `non_local_swr` and `non_local_no_swr`
    per-bin designs are STACKED (identical (stop,bin) rows + covariates, differing
    only in `count`), tagged by `is_swr`, and `is_swr:X` interactions test whether a
    task effect differs by SWR status. Self-history (`offhist_rcos_*`) is
    stream-specific only in its raw per-stream values; it is still wrapped inside
    `is_swr * BASE` (S2-S10), so `is_swr:offhist_rcos_k` history-by-SWR
    interactions ARE fit.
    Time/reward/task covariates are shared across the two rows of a bin. The within-bin
    pairing is carried by the per-session fit + across-session aggregation (no GLMM),
    matching the rest of Family B.

TWO COHORTS per (session, population). `rpe_postoutcome` and
`reward_belief_mean_update_l1_all_ports_postoutcome` are undefined on a stop that
starts before its trial's outcome is known, so they are NaN'd there. The rungs that
need them (`G5_rpe`, `G7_update`, `S7_rpe`, `S9_update`) are therefore fit on a
POST_OUTCOME cohort -- the primary design re-frozen on those columns -- alongside a
refit of their ladder's null (`G0_base` / `S0_rate`), so their AIC comparison shares
rows. Every other rung shares the PRIMARY cohort, which the gate does not shrink.
Each fitted row carries a `cohort` label; `(session, population, model, cohort)` is
the key.

Covariate blocks BASE/PERISWITCH/BLOCK/SWITCHRATE/VALUE/RPE/UNCERT/UPDATE/REWXTIME
are defined below. The task
dimensions are the paper's (Comrie et al.), all from the beta-bernoulli model:
`relative_value_switch_minus_stay_preoutcome`, `rpe_postoutcome`,
`reward_belief_differential_entropy_leaf_mean_preoutcome`, and
`reward_belief_mean_update_l1_all_ports_postoutcome`; plus peri-switch/block
context from `qvals_table` and
`patch_switch_fraction_last5_completed_trips` from the canonical stop table.
One tidy row per (population, model, session), each carrying per-term coef + SE
(``bse_*``) + p-value (``pval_*``); aggregate across sessions downstream (per-animal
sign consistency primary, session signed-rank descriptive — see the content runner).

Usage:
    uv run python scripts/behavior/rate/fit_swr_contrast_rate.py --max-per-animal 1   # smoke (~4 min)
    uv run python scripts/behavior/rate/fit_swr_contrast_rate.py --workers 16         # full
    uv run python scripts/behavior/rate/fit_swr_contrast_rate.py --aggregate-only     # re-audit
"""

from __future__ import annotations

from collections.abc import Sequence

from continuum_swr_replay.runs.threads import pin_blas_threads

pin_blas_threads()

import argparse  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from patsy import PatsyError  # noqa: E402
from statsmodels.tools.sm_exceptions import PerfectSeparationError  # noqa: E402

from continuum_swr_replay.analysis.behavior import (  # noqa: E402
    BIN_S,
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    POPULATIONS,
    TMAX,
    add_uncertainty_covariates,
    aggregate_to_timebins,
    attach_is_swr_to_onsets,
    freeze_per_bin_cohort,
)
from continuum_swr_replay.analysis.behavior._fit_diagnostics import (  # noqa: E402
    NUISANCE_TERMS,
    capture_fit_warnings,
    count_fit_quality_warnings,
    fit_flag_counts,
    fit_status,
)
from continuum_swr_replay.analysis.behavior.cohort_audit import (  # noqa: E402
    cohort_record,
    stop_own_missing,
    term_diagnostics,
    write_cohort_audit,
)
from continuum_swr_replay.analysis.behavior.contrast_aggregation import (  # noqa: E402
    aggregate_contrast,
    sd_reference,
)
from continuum_swr_replay.analysis.behavior.covariate_timing import (  # noqa: E402
    assert_predictor_timing,
)
from continuum_swr_replay.analysis.behavior.covariates import (  # noqa: E402
    assert_qvals_aligned,
)
from continuum_swr_replay.analysis.behavior.fit_provenance import (  # noqa: E402
    record_inputs,
    write_fit_provenance,
)
from continuum_swr_replay.analysis.behavior.model_specs import (  # noqa: E402
    DESCRIPTIVE_RATE_RUNGS,
    PER_BIN_DESIGN,
    POST_OUTCOME_COHORT,
    POST_OUTCOME_COVARIATES,
    POST_OUTCOME_RATE_RUNGS,
    PRIMARY_COHORT,
    REWARD_HISTORY_COLUMNS,
    longest_history_window_s,
)
from continuum_swr_replay.analysis.behavior.model_specs import (  # noqa: E402
    SWR_CONTRAST_RATE_G_SPECS as G_SPECS,
)
from continuum_swr_replay.analysis.behavior.model_specs import (  # noqa: E402
    SWR_CONTRAST_RATE_S_SPECS as S_SPECS,
)
from continuum_swr_replay.analysis.behavior.model_specs import (  # noqa: E402
    SWR_CONTRAST_RATE_STOP_COVARS as STOP_COVARS,
)
from continuum_swr_replay.analysis.behavior.rate.duration_aware_history import (  # noqa: E402
    at_risk_events,
    attach_event_ends,
)
from continuum_swr_replay.analysis.events.analysed_time import (  # noqa: E402
    require_swr_source,
)
from continuum_swr_replay.analysis.events.cohort_selection import (  # noqa: E402
    STOP_COHORTS,
    select_stop_cohort,
    stop_cohort_suffix,
)
from continuum_swr_replay.analysis.events.swr_source import (  # noqa: E402
    swr_source_name,
)
from continuum_swr_replay.paths import (  # noqa: E402
    model_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.tables.covariate_registry import (  # noqa: E402
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table  # noqa: E402
from continuum_swr_replay.tables.session_keys import (  # noqa: E402
    SESSION_COLUMNS,
    subset_max_per_animal,
)

# Only the benign exp()-overflow RuntimeWarning (quasi-separated coefs -> rate ratios)
# is silenced. Convergence/separation are recorded per fit as flags inside _fit (see
# analysis.behavior._fit_diagnostics), not blanket-ignored.
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


# Audited-aggregation key terms, mirroring the content ladder. G-rungs add one
# dimension each (fit per population); S-rungs carry the is_swr:<dimension> contrast.
_G_KEY = {
    "G0_base": REWARD_HISTORY_COLUMNS[0],
    "G1_periswitch": "periswitch_index_trials",
    "G2_block": "trials_since_block_change",
    "G3_switchrate": "patch_switch_fraction_last5_completed_trips",
    "G4_value": "relative_value_switch_minus_stay_preoutcome",
    "G5_rpe": "rpe_postoutcome",
    "G6_uncert": "reward_belief_differential_entropy_leaf_mean_preoutcome",
    "G7_update": "reward_belief_mean_update_l1_all_ports_postoutcome",
}
_S_KEY = {
    "S1_rew": f"is_swr:{REWARD_HISTORY_COLUMNS[0]}",
    "S3_periswitch": "is_swr:periswitch_index_trials",
    "S4_block": "is_swr:trials_since_block_change",
    "S5_switchrate": "is_swr:patch_switch_fraction_last5_completed_trips",
    "S6_value": "is_swr:relative_value_switch_minus_stay_preoutcome",
    "S7_rpe": "is_swr:rpe_postoutcome",
    "S8_uncert": "is_swr:reward_belief_differential_entropy_leaf_mean_preoutcome",
    "S9_update": "is_swr:reward_belief_mean_update_l1_all_ports_postoutcome",
}
# raw fit population -> friendly label shared with the content table
_POP_LABEL = {"non_local": "all", "non_local_swr": "SWR", "non_local_no_swr": "nonSWR"}
_RATE_CONT_COVARS = [
    "periswitch_index_trials",
    "trials_since_block_change",
    "patch_switch_fraction_last5_completed_trips",
    "relative_value_switch_minus_stay_preoutcome",
    "rpe_postoutcome",
    "reward_belief_differential_entropy_leaf_mean_preoutcome",
    "reward_belief_mean_update_l1_all_ports_postoutcome",
]


def fit_input_paths() -> dict[str, Path]:
    """The tables `_load` reads, recorded in each artifact's provenance sidecar;
    under ``paths.replay_content_dir()``, read when called."""
    rc = replay_content_dir()
    return {
        "stop_table": table_path("stop_table", rc),
        "qvals_table": table_path("qvals_table", rc),
        "event_onset_table": table_path("event_onset_table", rc),
        "event_table": table_path("event_table", rc),
    }


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
    """Poisson GLM `count ~ formula` with a log(exposure) offset.

    ``design`` must already be frozen to the ladder's common cohort
    (:func:`freeze_per_bin_cohort` over the union of every rung's columns), so
    every rung of a ladder is fit to the same rows and their AIC / deviance are
    comparable. An incomplete design is a caller bug and raises rather than
    being quietly subset here.

    Returns ``(row, reason)``: ``reason`` is ``None`` on success, else a short tag
    for the skip -- ``missing:<cols>`` (a needed column is absent -> a config/upstream
    bug, distinct from a data skip), ``too_few``, ``no_reward_contrast``, or the
    caught exception's class name. The caller tallies these so drop-outs are visible
    (a silently-shrunk per-model session count would bias the downstream aggregate).
    """
    missing = [c for c in need_cols if c not in design.columns]
    if missing:
        return None, f"missing:{','.join(missing)}"
    nan_cols = {
        c: int(design[c].isna().sum())
        for c in (*need_cols, "exposure")
        if design[c].isna().any()
    }
    if nan_cols:
        raise ValueError(
            f"is_swr-contrast rate design has missing values in {nan_cols}; "
            "freeze the ladder's common cohort with freeze_per_bin_cohort() "
            "before fitting (a per-rung drop would split the ladder's cohort)."
        )
    n_bad = int((design["exposure"] <= 0).sum())
    if n_bad:
        raise ValueError(
            f"is_swr-contrast rate design has {n_bad} row(s) with exposure <= 0; "
            "freeze the ladder's common cohort with freeze_per_bin_cohort()."
        )
    df = design
    n_events = int((df["count"] > 0).sum())
    if len(df) < MIN_BINS or n_events < MIN_EVENTS:
        return None, "too_few"
    if (
        REWARD_HISTORY_COLUMNS[0] in need_cols
        and df[REWARD_HISTORY_COLUMNS[0]].nunique() < 2
    ):
        return None, "no_reward_contrast"  # reward terms unidentified without contrast
    offset = np.log(df["exposure"].to_numpy(dtype=float))
    try:
        with capture_fit_warnings() as caught:
            res = smf.glm(
                f"count ~ {formula}",
                data=df,
                family=sm.families.Poisson(),
                offset=offset,
            ).fit()
    except _EXPECTED as e:
        return None, type(e).__name__
    predictor_sd = np.asarray(res.model.exog, dtype=float).std(axis=0)
    term_names = [str(t) for t in res.params.index]
    status = fit_status(
        res,
        "glm",
        predictor_sd=predictor_sd,
        term_names=term_names,
        nuisance_terms=NUISANCE_TERMS,
    )
    out = {f"coef_{k}": v for k, v in res.params.items()}
    # Within-session coefficient uncertainty: carry SE + p per term so downstream
    # aggregation can weight/annotate (mirrors the content contrast runner). These
    # are model-based (non-robust) SEs; clustered/robust inference is a follow-up.
    out.update({f"bse_{k}": v for k, v in res.bse.items()})
    out.update({f"pval_{k}": v for k, v in res.pvalues.items()})
    out.update(
        {
            f"sd_{name}": float(s)
            for name, s in zip(term_names, predictor_sd, strict=True)
        }
    )
    out.update(
        n_bins=int(res.nobs),
        n_events=n_events,
        llf=float(res.llf),
        aic=float(res.aic),
        n_params=int(len(res.params)),
        # distinct stops that contributed >=1 retained bin -- the stop-grain modeled count.
        n_stops=int(df["stop_id"].nunique()),
        converged=status["converged"],
        separated=status["separated"],
        # "|"-joined so the tuple survives a round trip through pickle and CSV.
        separated_terms="|".join(status["separated_terms"]),
        n_convergence_warnings=count_fit_quality_warnings(caught),
    )
    return out, None


# Each ladder's null rung: the companion fit that gives a post-outcome rung a
# same-rows AIC comparison on the post-outcome cohort.
LADDER_NULL = {"G": "G0_base", "S": "S0_rate"}


def _primary_union(specs: dict) -> list[str]:
    """Design columns of every rung of ``specs`` that is not a post-outcome rung."""
    return list(
        dict.fromkeys(
            c
            for name, (_f, need) in specs.items()
            if name not in POST_OUTCOME_RATE_RUNGS
            for c in need
        )
    )


def _fit_post_outcome_cohort(
    design: pd.DataFrame,
    specs: dict,
    ladder: str,
    population: str,
    base: dict,
    tag_cols: tuple[str, ...] = (),
) -> tuple[list[dict], dict, int]:
    """Fit a ladder's post-outcome rungs and its null on one re-frozen cohort.

    The post-outcome covariates are undefined on stops that begin before the
    outcome is known (see :func:`_mask_post_outcome_covariates`), so the rungs
    that carry them cannot be fit on the ladder's primary cohort. They get a
    second cohort: the primary design re-frozen on their own columns, shared by
    every post-outcome rung of the ladder (both covariates are NaN on the same
    rows) and by the ladder's null rung, which is refit here so the rungs'
    AIC / deviance comparison against the null is made on identical rows.

    Parameters
    ----------
    design : pd.DataFrame
        The ladder's PRIMARY-frozen (stop x bin) design.
    specs : dict
        ``SWR_CONTRAST_RATE_G_SPECS`` or ``SWR_CONTRAST_RATE_S_SPECS``.
    ladder : {"G", "S"}
        Selects the companion null rung from :data:`LADDER_NULL`.
    population : str
        Fit population recorded on each row (a stratum, or ``"contrast"``).
    base : dict
        Session key columns and the design label copied onto each row.
    tag_cols : tuple of str
        Extra required columns of the design (``("is_swr",)`` for the stacked
        contrast design).

    Returns
    -------
    rows : list of dict
        One row per fitted rung, tagged ``cohort=POST_OUTCOME_COHORT``.
    skips : dict
        ``(model, population, cohort, reason) -> count`` for rungs not fit.
    n_stops_dropped : int
        Stops the re-freeze removed from the primary cohort.
    """
    rungs = [n for n in specs if n in POST_OUTCOME_RATE_RUNGS]
    need_union = list(dict.fromkeys(c for n in rungs for c in specs[n][1]))
    frozen, freeze = freeze_per_bin_cohort(
        design, (), extra_columns=[*need_union, *tag_cols]
    )
    n_stops_dropped = freeze["n_stops_before"] - freeze["n_stops_after"]
    rows: list[dict] = []
    skips: dict = {}
    for name in (*rungs, LADDER_NULL[ladder]):
        formula, need = specs[name]
        fit, reason = _fit(frozen, formula, [*need, *tag_cols])
        if fit is not None:
            rows.append(
                {
                    "population": population,
                    "model": name,
                    "cohort": POST_OUTCOME_COHORT,
                    **base,
                    **fit,
                }
            )
        else:
            key = (name, population, POST_OUTCOME_COHORT, reason)
            skips[key] = skips.get(key, 0) + 1
    return rows, skips, n_stops_dropped


# The ladder's longest history window (both ladders share one cohort per
# stratum), from the one catalog of ladders.
HISTORY_WINDOW_S = longest_history_window_s([*G_SPECS.values(), *S_SPECS.values()])


def _fit_one_session(keys, stops_g, onsets_g) -> tuple[tuple, list[dict], dict, dict]:
    """Fit both ladders for one session on TWO frozen cohorts each.

    The primary cohort carries every rung whose covariates are all pre-outcome;
    those rungs share one frozen cohort per (session, population), so their AIC /
    deviance comparison is valid. The post-outcome rungs (``G5_rpe`` /
    ``G7_update`` / ``S7_rpe`` / ``S9_update``) are fit on a second cohort --
    the primary design re-frozen on their own columns -- together with their
    ladder's null rung, so their comparison is also made on identical rows.
    Every row carries a ``cohort`` label, so ``(session, population, model,
    cohort)`` is unique.

    Returns ``(keys, rows, skips, cohort_drops)``. ``cohort_drops`` maps a
    population (each stratum plus ``"contrast"``) to the stops the primary
    freeze removed, and ``(population, POST_OUTCOME_COHORT)`` to the stops both
    freezes removed together; the audit reports them as
    ``n_drop_common_cohort_other`` for the rungs that do not need the missing
    column themselves.
    """
    rows: list[dict] = []
    skips: dict = {}  # (model, population, cohort, reason) -> count
    cohort_drops: dict = {}
    designs = {
        pop: aggregate_to_timebins(
            stops_g,
            POPULATIONS[pop](onsets_g),
            bin_s=BIN_S,
            tmax=TMAX,
            reward_basis=True,
            self_history_lags=N_HISTORY_LAGS,
            self_history_basis=N_HISTORY_BASIS,
            stop_covariates=STOP_COVARS,
            history_window_s=HISTORY_WINDOW_S,
            # Every stratum is at risk outside every non-local event, so the
            # two streams' designs keep the same bins.
            at_risk_events=at_risk_events(onsets_g, pop),
            offset_history=True,
        )
        for pop in STRATA
    }
    # ONE primary cohort per ladder: the union of the columns of every rung that
    # is not a post-outcome rung, for each stratum. The union is identical across
    # strata (the covariates are the same per-stop values; only ``count``
    # differs), so the stacked contrast design below keeps its within-bin
    # pairing. The post-outcome covariates are deliberately NOT in this union:
    # they are NaN on every stop that begins before its outcome is known, which
    # would cost the whole ladder those stops.
    g_union = _primary_union(G_SPECS)
    for pop in STRATA:
        frozen, freeze = freeze_per_bin_cohort(designs[pop], (), extra_columns=g_union)
        cohort_drops[pop] = freeze["n_stops_before"] - freeze["n_stops_after"]
        designs[pop] = frozen
    base = {
        **dict(zip(SESSION_COLUMNS, keys, strict=True)),
        # The design every row was fit on, so a reader can refuse a fit from
        # the old all-bins design.
        "per_bin_design": PER_BIN_DESIGN,
    }
    # A. stratified, primary cohort
    for pop in STRATA:
        for name, (formula, need) in G_SPECS.items():
            if name in POST_OUTCOME_RATE_RUNGS:
                continue
            fit, reason = _fit(designs[pop], formula, need)
            if fit is not None:
                rows.append(
                    {
                        "population": pop,
                        "model": name,
                        "cohort": PRIMARY_COHORT,
                        **base,
                        **fit,
                    }
                )
            else:
                key = (name, pop, PRIMARY_COHORT, reason)
                skips[key] = skips.get(key, 0) + 1
    # A2. stratified, post-outcome cohort (each rung + the ladder's null)
    for pop in STRATA:
        po_rows, po_skips, po_dropped = _fit_post_outcome_cohort(
            designs[pop], G_SPECS, "G", pop, base
        )
        rows.extend(po_rows)
        for key, count in po_skips.items():
            skips[key] = skips.get(key, 0) + count
        cohort_drops[(pop, POST_OUTCOME_COHORT)] = cohort_drops[pop] + po_dropped
    # B. is_swr contrast: stack the two non-local streams
    stacked = pd.concat(
        [
            designs["non_local_swr"].assign(is_swr=1.0),
            designs["non_local_no_swr"].assign(is_swr=0.0),
        ],
        ignore_index=True,
    )
    s_union = _primary_union(S_SPECS)
    stacked, s_freeze = freeze_per_bin_cohort(
        stacked, (), extra_columns=[*s_union, "is_swr"]
    )
    cohort_drops["contrast"] = s_freeze["n_stops_before"] - s_freeze["n_stops_after"]
    for name, (formula, need) in S_SPECS.items():
        if name in POST_OUTCOME_RATE_RUNGS:
            continue
        fit, reason = _fit(stacked, formula, [*need, "is_swr"])
        if fit is not None:
            rows.append(
                {
                    "population": "contrast",
                    "model": name,
                    "cohort": PRIMARY_COHORT,
                    **base,
                    **fit,
                }
            )
        else:
            key = (name, "contrast", PRIMARY_COHORT, reason)
            skips[key] = skips.get(key, 0) + 1
    po_rows, po_skips, po_dropped = _fit_post_outcome_cohort(
        stacked, S_SPECS, "S", "contrast", base, tag_cols=("is_swr",)
    )
    rows.extend(po_rows)
    for key, count in po_skips.items():
        skips[key] = skips.get(key, 0) + count
    cohort_drops[("contrast", POST_OUTCOME_COHORT)] = (
        cohort_drops["contrast"] + po_dropped
    )
    return keys, rows, skips, cohort_drops


def _mask_post_outcome_covariates(
    stop: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """NaN the post-outcome covariates on stops that begin before the outcome.

    ``rpe_postoutcome`` and ``reward_belief_mean_update_l1_all_ports_postoutcome``
    are defined only once the trial's outcome is known (``outcome_known_time``).
    A stop whose ``stop_start`` precedes that instant spans the pre-outcome
    interval, so attaching the trial's post-outcome value to it would let the
    model condition on information the animal did not yet have. Those values are
    set to NaN here, before any freeze, so the cohort freezes -- not a silent
    per-rung drop inside the fitter -- decide which rows each rung is fit on.

    Parameters
    ----------
    stop : pd.DataFrame
        Post-qvals-join stop table; must carry ``stop_start``,
        ``outcome_known_time``, ``occupancy_state``, and both post-outcome
        covariates.

    Returns
    -------
    gated : pd.DataFrame
        Copy of ``stop`` with the post-outcome covariates set to NaN on
        pre-outcome stops.
    n_masked_by_state : pd.Series
        Number of masked stops per ``occupancy_state`` category.
    """
    required = ("stop_start", "outcome_known_time", "occupancy_state")
    missing = [
        c for c in (*required, *POST_OUTCOME_COVARIATES) if c not in stop.columns
    ]
    if missing:
        raise ValueError(
            f"post-outcome gate needs {missing} on the joined stop table "
            "(stop_start + occupancy_state from the stop table, "
            "outcome_known_time from qvals_table)"
        )
    stop_start = stop["stop_start"].to_numpy(dtype=float)
    outcome_known = stop["outcome_known_time"].to_numpy(dtype=float)
    pre_outcome = stop_start < outcome_known
    gated = stop.copy()
    gated.loc[pre_outcome, list(POST_OUTCOME_COVARIATES)] = np.nan
    n_masked_by_state = (
        stop.loc[pre_outcome, "occupancy_state"].value_counts().astype(int)
    )
    return gated, n_masked_by_state


def _load(cohort: str) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    inputs = fit_input_paths()
    stop = read_table(inputs["stop_table"])
    q = read_table(inputs["qvals_table"])
    onset_table = read_table(inputs["event_onset_table"])
    event_table = read_table(inputs["event_table"])
    require_covariate_schema(stop, table="stop")
    require_covariate_schema(q, table="qvals")
    require_covariate_schema(onset_table, table="event_onset")
    require_covariate_schema(event_table, table="event")
    # The bins are dropped by the configured source's excluded time, so the
    # tables must come from that source (its artifact run included).
    require_swr_source(stop, what=str(inputs["stop_table"]))
    require_swr_source(onset_table, what=str(inputs["event_onset_table"]))
    stop = add_uncertainty_covariates(stop)
    qcols = [
        "stop_id",
        "outcome_known_time",
        "periswitch_index_trials",
        "is_periswitch_post",
        "trials_since_block_change",
        "relative_value_switch_minus_stay_preoutcome",
        "rpe_postoutcome",
        "reward_belief_differential_entropy_leaf_mean_preoutcome",
        "reward_belief_mean_update_l1_all_ports_postoutcome",
    ]
    missing = [c for c in qcols if c not in q.columns]
    if missing:
        raise ValueError(
            f"qvals_table missing {missing}; regenerate it "
            "(uv run python scripts/behavior/build_qvals_table.py --write)"
        )
    # Guard the stop_id join before the inner join hides its casualties
    # (see assert_qvals_aligned). It must see the WHOLE stop table: it claims
    # every qvals row falling inside the epoch's span, so checking it against a
    # cohort-restricted table would read the cohort's own stops as missing.
    assert_qvals_aligned(stop, q)
    # Cohort next, so the qvals gap below is measured within the population
    # actually fit rather than across one this run never touches.
    stop, cohort_report = select_stop_cohort(stop, cohort=cohort)
    logger.info(
        "stop cohort = %s (%s): %d -> %d stops (%.1f%% retained); by state %s",
        cohort_report["cohort"],
        cohort_report["rule"],
        cohort_report["n_stops_before"],
        cohort_report["n_stops_after"],
        100 * cohort_report["n_stops_after"] / max(cohort_report["n_stops_before"], 1),
        cohort_report["counts_before"],
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
    stop, n_masked_by_state = _mask_post_outcome_covariates(stop)
    n_masked = int(n_masked_by_state.sum())
    logger.info(
        "post-outcome gate: %d of %d stops (%.1f%%) start before the outcome is "
        "known; %s set to NaN there. Share by occupancy_state: %s",
        n_masked,
        len(stop),
        100 * n_masked / max(len(stop), 1),
        ", ".join(POST_OUTCOME_COVARIATES),
        {
            state: f"{count} ({100 * count / max(n_masked, 1):.0f}% of gated)"
            for state, count in n_masked_by_state.items()
            if count
        },
    )
    require_swr_source(event_table, what=str(inputs["event_table"]))
    onsets = attach_event_ends(
        attach_is_swr_to_onsets(onset_table, event_table), event_table
    )
    # Stops dropped by the inner qvals join are the mechanism-B "qvals gap" (no fitted
    # beta/value state) -- reported per-model as n_drop_qvals_gap in the cohort audit.
    n_qvals_dropped = int(n0 - len(stop))
    return stop, onsets, n_qvals_dropped


def _home_cohort(model: str) -> str:
    """The cohort a rung's own coefficients are reported from."""
    return POST_OUTCOME_COHORT if model in POST_OUTCOME_RATE_RUNGS else PRIMARY_COHORT


def _cohort_records(
    models: pd.DataFrame,
    stop: pd.DataFrame,
    n_qvals_dropped: int,
    cohort_stop_drops: dict | None = None,
) -> list[dict]:
    """One stop-grain cohort record per (model, population, cohort).

    Grain is ``stop`` (the rate family is audited at stop grain — the timebin
    design is aggregated per-session inside the fitter). Buckets:
    ``n_drop_qvals_gap`` = the inner qvals-join drop (mechanism B, stops with no fitted
    value state, already removed in ``_load``, so the cohort freeze cannot see them
    and they are never counted twice); ``n_drop_own_predictor_missing`` =
    post-join stops with ``immobility<=0`` or a NaN stop-level required predictor
    (e.g. ``reward_history_rcos_0``, model-specific);
    ``n_drop_common_cohort_other`` = post-join stops this rung's own predictors
    cover but that the frozen ladder cohort still removes because ANOTHER rung
    needs a column that is NaN there; ``n_modeled`` = the fitter's distinct
    modeled-stop count (``n_stops``); the rest (bin-level drops, session gates) go to
    ``n_excluded_session``. Reconciles by construction.

    ``cohort_stop_drops`` maps a fit population (and, for the post-outcome
    cohort, ``(population, POST_OUTCOME_COHORT)``) to the stops the freezes
    actually removed; it must not exceed what the stop-level attribution
    accounts for.

    The two cohorts are distinguished by ``cohort_mode``: ``common`` for the
    primary ladder, ``common_post_outcome`` for the post-outcome rungs and the
    null refit beside them, so the ladder null appears once per cohort rather
    than colliding on ``(model, population)``.
    """
    total = int(len(stop))  # post-join stops
    n_sessions_in = stop.groupby(list(SESSION_COLUMNS), sort=False).ngroups
    if len(models) and "cohort" not in models.columns:
        raise ValueError(
            "the fitted rows must carry a 'cohort' column; the ladder null is "
            "fit once per cohort and would otherwise be audited twice against "
            "the same rows"
        )
    need_by_model = {
        **{m: need for m, (_f, need) in G_SPECS.items()},
        **{m: need for m, (_f, need) in S_SPECS.items()},
    }
    frozen_drops = cohort_stop_drops or {}
    # The union a rung's cohort was frozen on: the primary ladder's columns, plus
    # the post-outcome columns for the rungs fit on the post-outcome cohort. A
    # stop this rung's own predictors cover but that its cohort's union removes
    # is a common-cohort drop, so the null refit beside a post-outcome rung books
    # the gated stops there while the post-outcome rung itself books them as its
    # own missing predictor.
    union_by_ladder = {
        ("G", PRIMARY_COHORT): _primary_union(G_SPECS),
        ("S", PRIMARY_COHORT): _primary_union(S_SPECS),
        ("G", POST_OUTCOME_COHORT): [c for _f, need in G_SPECS.values() for c in need],
        ("S", POST_OUTCOME_COHORT): [c for _f, need in S_SPECS.values() for c in need],
    }
    union_missing = {
        key: stop_own_missing(
            stop, list(dict.fromkeys(c for c in cols if c in stop.columns))
        )
        for key, cols in union_by_ladder.items()
    }
    primary_g = [m for m in G_SPECS if m not in POST_OUTCOME_RATE_RUNGS]
    primary_s = [m for m in S_SPECS if m not in POST_OUTCOME_RATE_RUNGS]
    post_g = [m for m in G_SPECS if m in POST_OUTCOME_RATE_RUNGS]
    post_s = [m for m in S_SPECS if m in POST_OUTCOME_RATE_RUNGS]
    pairs = [(m, p, PRIMARY_COHORT) for m in primary_g for p in STRATA]
    pairs += [(m, "contrast", PRIMARY_COHORT) for m in primary_s]
    pairs += [
        (m, p, POST_OUTCOME_COHORT) for m in (*post_g, LADDER_NULL["G"]) for p in STRATA
    ]
    pairs += [(m, "contrast", POST_OUTCOME_COHORT) for m in (*post_s, LADDER_NULL["S"])]
    records = []
    for name, pop, cohort in pairs:
        sub = (
            models[
                (models.model == name)
                & (models.population == pop)
                & (models.cohort == cohort)
            ]
            if len(models)
            else models
        )
        fit_keys = (
            {tuple(r) for r in sub[list(SESSION_COLUMNS)].to_numpy()}
            if len(sub)
            else set()
        )
        n_modeled = int(sub["n_stops"].sum()) if "n_stops" in sub else 0
        stop_level = [c for c in need_by_model.get(name, ()) if c in stop.columns]
        own_missing = stop_own_missing(stop, stop_level)
        n_own = int(own_missing.sum())
        ladder = "S" if pop == "contrast" else "G"
        n_common = int((union_missing[(ladder, cohort)] & ~own_missing).sum())
        n_frozen_out = int(
            frozen_drops.get(pop if cohort == PRIMARY_COHORT else (pop, cohort), 0)
        )
        if n_frozen_out > n_own + n_common:
            raise AssertionError(
                f"{name}/{pop}: the cohort freeze removed {n_frozen_out} stops "
                f"but the stop-level attribution only accounts for "
                f"{n_own + n_common} (own {n_own} + common {n_common})"
            )
        cohort_mode = (
            "common" if cohort == PRIMARY_COHORT else f"common_{POST_OUTCOME_COHORT}"
        )
        records.append(
            cohort_record(
                name,
                _POP_LABEL.get(pop, pop),
                "stop",
                cohort_mode=cohort_mode,
                n_in=total + int(n_qvals_dropped),
                n_modeled=n_modeled,
                sessions_in=n_sessions_in,
                sessions_modeled=len(fit_keys),
                animals_modeled=len({k[0] for k in fit_keys}),
                drop_notes=(
                    "qvals-join drop -> qvals_gap (mechanism B); immobility<=0 / NaN "
                    "own stop-level predictor -> own; NaN only in another rung's "
                    "column of this cohort's union -> common; bin-level + session "
                    "gates -> excluded"
                ),
                n_drop_qvals_gap=int(n_qvals_dropped),
                n_drop_own_predictor_missing=n_own,
                n_drop_common_cohort_other=n_common,
                n_excluded_session=total - n_own - n_common - n_modeled,
                **fit_flag_counts(sub),
            )
        )
    return records


def _covariate_sds(qv: pd.DataFrame, st: pd.DataFrame) -> dict[str, float]:
    """Per-covariate SD over one (qvals, stop) pair.

    Each name in :data:`_RATE_CONT_COVARS` is taken from whichever table carries
    it, qvals first; a name absent from both is omitted rather than recorded as
    ``nan``. ``aggregate_contrast`` then refuses (``std_factor`` raises) any
    fitted term whose covariate has no SD here.

    Parameters
    ----------
    qv, st : pandas.DataFrame
        qvals and stop tables, already restricted to the scope the SDs describe.

    Returns
    -------
    dict of str to float
        ``covariate -> SD``
        (:func:`~continuum_swr_replay.analysis.behavior.contrast_aggregation.sd_reference`),
        in the order of :data:`_RATE_CONT_COVARS`.
    """
    not_in_qvals = [c for c in _RATE_CONT_COVARS if c not in qv.columns]
    sd = {**sd_reference(st, not_in_qvals), **sd_reference(qv, _RATE_CONT_COVARS)}
    return {c: sd[c] for c in _RATE_CONT_COVARS if c in sd}


def _max_relative_sd_difference(
    sd: dict[str, float], sd_full: dict[str, float]
) -> tuple[str | None, float]:
    """Largest signed relative difference between two per-SD references.

    Reported as ``(sd - sd_full) / sd_full`` so the sign says whether the cohort
    is narrower (negative) or wider than the full tables; the covariate returned
    is the one with the largest magnitude.

    Parameters
    ----------
    sd : dict of str to float
        SDs actually used for the ``std_*`` columns (the fitted cohort).
    sd_full : dict of str to float
        SDs over the full tables, for comparison only.

    Returns
    -------
    covariate : str or None
        Name of the largest-magnitude difference, ``None`` if none is comparable.
    relative_difference : float
        Its signed relative difference; ``0.0`` when nothing is comparable.
    """
    diffs = {
        c: (v - sd_full[c]) / sd_full[c]
        for c, v in sd.items()
        if c in sd_full and np.isfinite(sd_full[c]) and sd_full[c] > 0
    }
    if not diffs:
        return None, 0.0
    worst = max(diffs, key=lambda c: abs(diffs[c]))
    return worst, float(diffs[worst])


def _write_aggregation(
    models: pd.DataFrame,
    out_dir: Path,
    suffix: str,
    fit_inputs: dict,
    cli: list[str],
) -> int:
    """Aggregate the per-session fits and write the audited artifacts.

    Writes ``swr_contrast_rate_term_diagnostics<suffix>.csv``,
    ``swr_contrast_rate_audited<suffix>.pkl`` with its provenance sidecar, and
    ``swr_contrast_rate_meta<suffix>.json``. Split out of the fitting path so
    ``--aggregate-only`` can re-derive them from the persisted per-session fits
    after an aggregation rule changes, without refitting 218 sessions.

    The aggregation mirrors the content ladder (per-animal primary, session
    signed-rank, BH-FDR, standardized per-SD columns). G-rungs are relabeled
    ``non_local``/``_swr``/``_no_swr`` -> ``all``/``SWR``/``nonSWR`` to match the
    content table. Each key term is aggregated over the cohort its rung was fit
    on; the null refit on the post-outcome cohort exists for the AIC comparison
    only and is never aggregated as a second copy of the ladder null.

    The audited rows and their sidecar carry the ``stop_cohort`` the per-session
    fits were made on, so a sensitivity bundle can never be read as the
    canonical one.  That same label scopes the per-SD reference behind the
    ``std_*`` columns: the SDs are taken over the stop cohort the fits name, not
    over the full tables.

    Parameters
    ----------
    models : pd.DataFrame
        Per-session fits, one row per (population, model, session, cohort),
        carrying the ``stop_cohort`` label the run was fit under.
    out_dir : Path
        ``models_per_bin_rate``.
    suffix : str
        ``"_smoke"`` for a smoke run, ``""`` otherwise.
    fit_inputs : dict
        Input records from :func:`record_inputs` for the audited sidecar.
    cli : list of str
        Command line recorded in the sidecar.

    Returns
    -------
    int
        Number of audited rows written (0 when ``models`` is empty).

    Raises
    ------
    ValueError
        If the per-session fits do not all name exactly one ``stop_cohort``.
    """
    if not len(models):
        return 0

    # The audited table is the artifact a reader quotes, so it has to say which
    # stop population it is about; the label comes from the fits it aggregates
    # rather than from the CLI, so `--aggregate-only` cannot relabel a bundle.
    stop_cohorts = sorted(models["stop_cohort"].dropna().unique())
    if len(stop_cohorts) != 1:
        raise ValueError(
            "per-session fits must all name one stop cohort; "
            f"found {stop_cohorts or ['none']}"
        )

    td_specs = [(g, p, t) for g, t in _G_KEY.items() for p in STRATA]
    td_specs += [(s, "contrast", t) for s, t in _S_KEY.items()]
    pd.concat(
        [
            term_diagnostics(
                models[models.cohort == cohort],
                [sp for sp in td_specs if _home_cohort(sp[0]) == cohort],
            ).assign(cohort=cohort)
            for cohort in (PRIMARY_COHORT, POST_OUTCOME_COHORT)
        ],
        ignore_index=True,
    ).to_csv(out_dir / f"swr_contrast_rate_term_diagnostics{suffix}.csv", index=False)

    inputs = fit_input_paths()
    qv_full = read_table(inputs["qvals_table"])
    st_full = read_table(inputs["stop_table"])
    require_covariate_schema(qv_full, table="qvals")
    require_covariate_schema(st_full, table="stop")
    require_swr_source(st_full, what=str(inputs["stop_table"]))
    st_full = add_uncertainty_covariates(st_full)
    # A std_* column rescales a coefficient fit on ONE stop cohort, so its SD has
    # to come from that same cohort: dividing an at-well coefficient by an
    # all-stops SD reports a per-SD effect for a population the rung was never fit
    # on. The label comes from the fits, so a sensitivity bundle standardizes on
    # its own population. `all_stops` keeps every stop, so the restriction is a
    # no-op there and that bundle's SDs are unchanged.
    st, _ = select_stop_cohort(st_full, cohort=stop_cohorts[0])
    qv = qv_full[qv_full["stop_id"].isin(set(st["stop_id"]))]
    sd = _covariate_sds(qv, st)
    sd_full = _covariate_sds(qv_full, st_full)
    max_sd_covar, max_sd_rel = _max_relative_sd_difference(sd, sd_full)
    logger.info(
        "per-SD reference on the %s cohort (%d of %d stops); largest relative "
        "difference vs the full tables: %s %+.2f%%",
        stop_cohorts[0],
        len(st),
        len(st_full),
        max_sd_covar,
        100 * max_sd_rel,
    )
    specs = [(g, p, t) for g, t in _G_KEY.items() for p in _POP_LABEL]
    specs += [(s, "contrast", t) for s, t in _S_KEY.items()]
    agg = pd.concat(
        [
            aggregate_contrast(
                models[models.cohort == cohort],
                [sp for sp in specs if _home_cohort(sp[0]) == cohort],
                "rate",
                sd,
            ).assign(cohort=cohort)
            for cohort in (PRIMARY_COHORT, POST_OUTCOME_COHORT)
        ],
        ignore_index=True,
    )
    agg["population"] = agg["population"].map(lambda p: _POP_LABEL.get(p, p))
    agg["stop_cohort"] = stop_cohorts[0]
    audited = out_dir / f"swr_contrast_rate_audited{suffix}.pkl"
    agg.to_pickle(audited)
    write_fit_provenance(
        audited,
        fit_inputs,
        "poisson_glm",
        cli,
        extra={"stop_cohort": stop_cohorts[0], "swr_source": swr_source_name()},
    )
    # Persist the exact per-SD factors this run used for the std_* columns so the
    # standardized coefficients are reproducible. They are computed over the stop
    # cohort the fits name, not over the full tables; the recorded
    # max_relative_sd_difference_vs_full_tables says how far the two scopes are
    # apart on this run. This is the runner's own reference; it is NOT the
    # builder's pooled-global table _z.
    (out_dir / f"swr_contrast_rate_meta{suffix}.json").write_text(
        json.dumps(
            {
                "fit_time_sd_reference": {k: float(v) for k, v in sd.items()},
                "sd_scope": (
                    f"{stop_cohorts[0]} stop cohort of the qvals_table + "
                    "stop_table (std_factor = raw_coef * SD)"
                ),
                "max_relative_sd_difference_vs_full_tables": {
                    "covariate": max_sd_covar,
                    "relative_difference": max_sd_rel,
                },
                "note": (
                    "the exact SD factors aggregate_contrast used for the std_* "
                    "columns; NOT the builder's pooled-global table _z"
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )
    logger.info("Saved audited rate table (%d rows) + meta -> %s", len(agg), out_dir)
    return len(agg)


def _aggregate_only(smoke: bool, cohort: str, cli: list[str]) -> None:
    """Re-derive the audited artifacts from the persisted per-session fits.

    Reads ``swr_contrast_rate<suffix>.pkl`` and rewrites the audited table, the
    term-diagnostics CSV and the meta JSON. The audited sidecar records that
    pickle as its input, so the audited table is tied to the fit it was derived
    from rather than to the tables that fit read.

    Parameters
    ----------
    smoke : bool
        Read/write the ``_smoke`` artifacts.
    cohort : {"at_well", "all_stops", "in_transit"}
        Which cohort's bundle to re-aggregate; picks the artifact suffix.
    cli : list of str
        The command line to record in the audited sidecar.

    Raises
    ------
    SystemExit
        If the per-session pickle is absent, or predates the per-term separation
        flag (no ``separated_terms`` column) -- aggregating it would silently
        apply the per-term drop rule to fits that carry no per-term information.
    """
    suffix = stop_cohort_suffix(cohort, smoke=smoke)
    out_dir = model_dir("per_bin_rate")
    fits_path = out_dir / f"swr_contrast_rate{suffix}.pkl"
    if not fits_path.exists():
        raise SystemExit(f"no per-session fits at {fits_path}; run the fit first")
    models = pd.read_pickle(fits_path)
    if "separated_terms" not in models.columns:
        raise SystemExit(
            f"{fits_path} has no 'separated_terms' column: it predates the "
            "per-term separation flag, and re-aggregating it would apply the "
            "per-term drop rule to fits that carry no per-term information. "
            "Refit before aggregating."
        )
    n_rows = _write_aggregation(
        models,
        out_dir,
        suffix,
        # Named by the file, not by the runner: the freshness sweep resolves an
        # input name to a file beside the sidecar, so recording a sensitivity
        # bundle under the bare name would point it at the canonical fit and
        # report the aggregate as permanently stale.
        record_inputs({fits_path.stem: fits_path}),
        cli,
    )
    logger.info(
        "Re-aggregated %d per-session fits from %s -> %d audited rows",
        len(models),
        fits_path,
        n_rows,
    )


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--max-per-animal", type=int, default=None, help="smoke: N sessions/animal"
    )
    ap.add_argument(
        "--workers", type=int, default=1, help="per-session process fan-out"
    )
    ap.add_argument(
        "--stop-cohort",
        choices=STOP_COHORTS,
        default="at_well",
        help=(
            "Stop population to fit. 'at_well' (default) is the canonical cohort "
            "shared with the content family and publishes the unsuffixed "
            "artifacts; 'all_stops' and 'in_transit' are sensitivity bundles "
            "published under _allstops / _intransit."
        ),
    )
    ap.add_argument(
        "--aggregate-only",
        action="store_true",
        help=(
            "skip fitting: re-aggregate the persisted per-session fits into the "
            "audited table, term diagnostics and meta JSON"
        ),
    )
    argv = sys.argv[1:] if argv is None else list(argv)
    args = ap.parse_args(argv)
    cli = [sys.argv[0], *argv]
    # `--max-per-animal 0` is still smoke mode, so test the flag, not its value.
    smoke = args.max_per_animal is not None
    if args.aggregate_only:
        _aggregate_only(smoke, args.stop_cohort, cli)
        return

    # Causal-hygiene gate: each rung's covariates must be pre-outcome, or the
    # rung is an explicitly-declared descriptive rung (RPE/value association or peri-switch
    # alignment). A NEW leaky covariate on an undeclared rung fails here. S-rungs carry
    # is_swr as a gating modifier (a SAFE_MODIFIER), so asserting `need` covers the
    # is_swr:<covariate> expansion.
    for specs, descr in (
        (G_SPECS, DESCRIPTIVE_RATE_RUNGS),
        (S_SPECS, DESCRIPTIVE_RATE_RUNGS),
    ):
        for name, (_formula, need) in specs.items():
            assert_predictor_timing(need, descriptive=(name in descr))

    stop, onsets, n_qvals_dropped = _load(args.stop_cohort)
    # Hash the inputs now, not at write time: the fan-out runs long enough that a
    # table could be regenerated mid-fit and be certified as this fit's input.
    fit_inputs = record_inputs(fit_input_paths())
    if args.max_per_animal is not None:
        stop = subset_max_per_animal(stop, args.max_per_animal)

    onset_groups = dict(tuple(onsets.groupby(list(SESSION_COLUMNS), sort=False)))
    items = [
        (k, s, onset_groups[k])
        for k, s in stop.groupby(list(SESSION_COLUMNS), sort=False)
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
    cohort_stop_drops: Counter = Counter()
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_fit_one_session, k, s, o) for k, s, o in items]
            for fut in as_completed(futs):
                _, r, sk, drops = fut.result()
                rows.extend(r)
                skips.update(sk)
                cohort_stop_drops.update(drops)
    else:
        for k, s, o in items:
            _, r, sk, drops = _fit_one_session(k, s, o)
            rows.extend(r)
            skips.update(sk)
            cohort_stop_drops.update(drops)
            logger.info("  %s: %d fits", "_".join(map(str, k)), len(r))

    # log skipped (model, population, reason) cells so a silently-shrunk per-model
    # session count is visible before the downstream signed-rank aggregate.
    if skips:
        cfg = {
            f"{m}/{p}/{c}": r
            for (m, p, c, r), _ in skips.items()
            if str(r).startswith("missing")
        }
        if cfg:
            logger.warning("MISSING-COLUMN skips (config/upstream bug): %s", cfg)
        logger.info(
            "skipped (model/pop/cohort -> reason: count): %s",
            {f"{m}/{p}/{c}:{r}": n for (m, p, c, r), n in sorted(skips.items())},
        )

    models = pd.DataFrame(rows)
    # Workers finish in any order, so the rows are sorted by their full key
    # (unique per row) before they are saved and aggregated.
    if len(models):
        models = models.sort_values(
            [*SESSION_COLUMNS, "population", "model", "cohort"], ignore_index=True
        )
    # Cohort sizes: the post-outcome cohort is the primary one minus the stops
    # gated above, so the gap between the two lines is the cost of the gate.
    if len(models):
        for (pop, cohort), grp in models.groupby(["population", "cohort"]):
            logger.info(
                "cohort %-12s %-16s: %d fits, median n_stops=%.0f, n_bins=%.0f",
                cohort,
                pop,
                len(grp),
                grp["n_stops"].median(),
                grp["n_bins"].median(),
            )
    # Which population every row was fit on, carried on the row itself: a
    # coefficient read out of a concatenated table would otherwise not say, and
    # the model comparison invalidates a cross-cohort step on this column.
    models["stop_cohort"] = args.stop_cohort

    out_dir = model_dir("per_bin_rate")
    out_dir.mkdir(parents=True, exist_ok=True)
    csfx = stop_cohort_suffix(args.stop_cohort, smoke=smoke)
    out = out_dir / f"swr_contrast_rate{csfx}.pkl"
    models.to_pickle(out)
    write_fit_provenance(
        out,
        fit_inputs,
        "poisson_glm",
        cli,
        extra={"stop_cohort": args.stop_cohort, "swr_source": swr_source_name()},
    )

    # Cohort-audit table (stop grain; reconciles n_in - Σdrops == n_modeled) +
    # term-level quasi-separation diagnostics, mirroring the content contrast runner.
    # The qvals-join drop is measured on the cohort's stop table in _load(); in smoke
    # mode
    # the per-animal subset happens after, so attributing the full-data drop against a
    # tiny subset n_in would be meaningless. Report 0 there (smoke is not a reported run).
    audit_qvals_drop = 0 if smoke else n_qvals_dropped
    write_cohort_audit(
        _cohort_records(models, stop, audit_qvals_drop, dict(cohort_stop_drops)),
        out_dir / f"swr_contrast_rate_cohort_audit{csfx}.csv",
    )
    logger.info(
        "Saved %d rows (%d sessions, populations=%s, models=%s) to %s in %.1fs",
        len(models),
        models.groupby(list(SESSION_COLUMNS)).ngroups if len(models) else 0,
        sorted(models["population"].unique()) if len(models) else [],
        sorted(models["model"].unique()) if len(models) else [],
        out,
        time.perf_counter() - t0,
    )

    _write_aggregation(models, out_dir, csfx, fit_inputs, cli)


if __name__ == "__main__":
    main()
