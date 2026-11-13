"""The per-bin ``is_swr``-contrast rate ladders, fit over sessions, and their audit.

Two ladders on the at-risk (stop x bin) design shared with the per-bin rate
ladder (``per_bin_rate_ladder``), fit per session:

- STRATIFIED (``SWR_CONTRAST_RATE_G_SPECS``): each rung fit to ``non_local``,
  ``non_local_swr`` and ``non_local_no_swr`` separately;
- ``is_swr`` CONTRAST (``SWR_CONTRAST_RATE_S_SPECS``): the ``non_local_swr``
  and ``non_local_no_swr`` designs stacked, tagged by ``is_swr``, so the
  ``is_swr:X`` interactions test whether a task effect differs by SWR status.

Each (session, population) has two cohorts: the PRIMARY one, frozen over every
rung that is not a post-outcome rung, and the POST_OUTCOME one, the primary
design re-frozen on the post-outcome rungs' columns, on which those rungs and
their ladder's null are fit (:func:`fit_contrast_session`).
:func:`fit_contrast_sessions` runs the sessions; :func:`join_contrast_qvals`
builds the stop table they are fit on; :func:`contrast_cohort_records` is the
stop-grain cohort audit; :func:`contrast_term_diagnostics`,
:func:`cohort_sd_reference` and :func:`aggregate_contrast_fits` are the
audited aggregation over sessions.
"""

from __future__ import annotations

import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.events.cohort_selection import select_stop_cohort
from continuum_swr_replay.tables.columns import require_columns
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

from ..cohort_audit import fitted_cohort_record, stop_own_missing, term_diagnostics
from ..contrast_aggregation import aggregate_contrast, sd_reference
from ..covariates import add_uncertainty_covariates
from ..model_specs import (
    BIN_S,
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    PER_BIN_DESIGN,
    POST_OUTCOME_COHORT,
    POST_OUTCOME_COVARIATES,
    POST_OUTCOME_RATE_RUNGS,
    PRIMARY_COHORT,
    REWARD_HISTORY_COLUMNS,
    TMAX,
    longest_history_window_s,
)
from ..model_specs import SWR_CONTRAST_RATE_G_SPECS as G_SPECS
from ..model_specs import SWR_CONTRAST_RATE_S_SPECS as S_SPECS
from ..model_specs import SWR_CONTRAST_RATE_STOP_COVARS as STOP_COVARS
from ..per_bin_design import POPULATIONS, aggregate_to_timebins
from .duration_aware_history import at_risk_events
from .per_bin_rate_model import fit_per_bin_glm, freeze_per_bin_cohort
from .post_outcome import mask_post_outcome_covariates

logger = logging.getLogger(__name__)

# Audited-aggregation key terms, mirroring the content ladder. G-rungs add one
# dimension each (fit per population); S-rungs carry the is_swr:<dimension> contrast.
G_KEY = {
    "G0_base": REWARD_HISTORY_COLUMNS[0],
    "G1_periswitch": "periswitch_index_trials",
    "G2_block": "trials_since_block_change",
    "G3_switchrate": "patch_switch_fraction_last5_completed_trips",
    "G4_value": "relative_value_switch_minus_stay_preoutcome",
    "G5_rpe": "rpe_postoutcome",
    "G6_uncert": "reward_belief_differential_entropy_leaf_mean_preoutcome",
    "G7_update": "reward_belief_mean_update_l1_all_ports_postoutcome",
}
S_KEY = {
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
POP_LABEL = {"non_local": "all", "non_local_swr": "SWR", "non_local_no_swr": "nonSWR"}
RATE_CONT_COVARS = [
    "periswitch_index_trials",
    "trials_since_block_change",
    "patch_switch_fraction_last5_completed_trips",
    "relative_value_switch_minus_stay_preoutcome",
    "rpe_postoutcome",
    "reward_belief_differential_entropy_leaf_mean_preoutcome",
    "reward_belief_mean_update_l1_all_ports_postoutcome",
]


STRATA = ("non_local", "non_local_swr", "non_local_no_swr")


def _fit_rung(
    design: pd.DataFrame, formula: str, need_cols: list[str], name: str
) -> tuple[dict | None, str | None]:
    """One rung's tidy row and skip reason (:func:`fit_per_bin_glm`).

    ``design`` must already be frozen to the ladder's common cohort; the
    per-bin ``predicted`` series is not tidy data and is left out of the row.
    """
    fit, reason = fit_per_bin_glm(design, f"count ~ {formula}", need_cols, name=name)
    if fit is None:
        return None, reason
    return {k: v for k, v in fit.items() if k != "predicted"}, None


# Each ladder's null rung: the companion fit that gives a post-outcome rung a
# same-rows AIC comparison on the post-outcome cohort.
LADDER_NULL = {"G": "G0_base", "S": "S0_rate"}


def primary_union(specs: dict) -> list[str]:
    """Design columns of every rung of ``specs`` that is not a post-outcome rung."""
    return list(
        dict.fromkeys(
            c
            for name, (_f, need) in specs.items()
            if name not in POST_OUTCOME_RATE_RUNGS
            for c in need
        )
    )


def fit_post_outcome_cohort(
    design: pd.DataFrame,
    specs: dict,
    ladder: str,
    population: str,
    base: dict,
    tag_cols: tuple[str, ...] = (),
) -> tuple[list[dict], dict, int]:
    """Fit a ladder's post-outcome rungs and its null on one re-frozen cohort.

    The post-outcome covariates are undefined on stops that begin before the
    outcome is known (see :func:`~continuum_swr_replay.analysis.behavior.rate.post_outcome.mask_post_outcome_covariates`), so the rungs
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
        fit, reason = _fit_rung(frozen, formula, [*need, *tag_cols], name)
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


def fit_contrast_session(
    keys, stops_g, onsets_g
) -> tuple[tuple, list[dict], dict, dict]:
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
    g_union = primary_union(G_SPECS)
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
            fit, reason = _fit_rung(designs[pop], formula, need, name)
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
        po_rows, po_skips, po_dropped = fit_post_outcome_cohort(
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
    s_union = primary_union(S_SPECS)
    stacked, s_freeze = freeze_per_bin_cohort(
        stacked, (), extra_columns=[*s_union, "is_swr"]
    )
    cohort_drops["contrast"] = s_freeze["n_stops_before"] - s_freeze["n_stops_after"]
    for name, (formula, need) in S_SPECS.items():
        if name in POST_OUTCOME_RATE_RUNGS:
            continue
        fit, reason = _fit_rung(stacked, formula, [*need, "is_swr"], name)
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
    po_rows, po_skips, po_dropped = fit_post_outcome_cohort(
        stacked, S_SPECS, "S", "contrast", base, tag_cols=("is_swr",)
    )
    rows.extend(po_rows)
    for key, count in po_skips.items():
        skips[key] = skips.get(key, 0) + count
    cohort_drops[("contrast", POST_OUTCOME_COHORT)] = (
        cohort_drops["contrast"] + po_dropped
    )
    return keys, rows, skips, cohort_drops


def fit_contrast_sessions(
    stop: pd.DataFrame, onsets: pd.DataFrame, *, workers: int = 1
) -> tuple[pd.DataFrame, Counter, dict]:
    """Both ladders for every session that has onsets (:func:`fit_contrast_session`).

    Parameters
    ----------
    stop : pd.DataFrame
        The joined stop table (:func:`join_contrast_qvals`).
    onsets : pd.DataFrame
        Event onsets carrying ``is_swr`` and each event's end.
    workers : int
        Processes for the per-session fan-out; 1 fits serially.

    Returns
    -------
    models : pd.DataFrame
        One row per (session, population, model, cohort), sorted by that key
        (workers finish in any order).
    skips : Counter
        ``(model, population, cohort, reason) -> count`` of rungs not fit.
    cohort_stop_drops : dict
        The stops the freezes removed, summed over sessions, keyed as
        :func:`fit_contrast_session`'s ``cohort_drops``.
    """
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
        workers,
    )
    rows: list[dict] = []
    skips: Counter = Counter()
    cohort_stop_drops: Counter = Counter()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(fit_contrast_session, k, s, o) for k, s, o in items]
            for fut in as_completed(futs):
                _, r, sk, drops = fut.result()
                rows.extend(r)
                skips.update(sk)
                cohort_stop_drops.update(drops)
    else:
        for k, s, o in items:
            _, r, sk, drops = fit_contrast_session(k, s, o)
            rows.extend(r)
            skips.update(sk)
            cohort_stop_drops.update(drops)
            logger.info("  %s: %d fits", "_".join(map(str, k)), len(r))
    models = pd.DataFrame(rows)
    if len(models):
        models = models.sort_values(
            [*SESSION_COLUMNS, "population", "model", "cohort"], ignore_index=True
        )
    return models, skips, dict(cohort_stop_drops)


# The qvals columns the contrast ladders read, joined onto the stop table.
QVALS_COLUMNS = [
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


def join_contrast_qvals(
    stop: pd.DataFrame, q: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """The stop table the contrast ladders are fit on.

    Inner-joins the :data:`QVALS_COLUMNS` of ``q`` onto ``stop`` (a stop with
    no qvals row has no fitted value state: the mechanism-B "qvals gap"), then
    sets the post-outcome covariates to NaN on the stops that begin before
    their outcome is known (:func:`mask_post_outcome_covariates`).

    Parameters
    ----------
    stop : pd.DataFrame
        The stop table of the fitted stop cohort, with the uncertainty
        covariates (``add_uncertainty_covariates``).
    q : pd.DataFrame
        The ``qvals_table``.

    Returns
    -------
    stop : pd.DataFrame
        The joined and masked stop table.
    n_qvals_dropped : int
        Stops the inner join removed, reported per model as
        ``n_drop_qvals_gap`` in the cohort audit.

    Raises
    ------
    ValueError
        If ``q`` lacks a :data:`QVALS_COLUMNS` column, or a covariate the
        ladders need (``SWR_CONTRAST_RATE_STOP_COVARS``) is not on the joined
        table.
    """
    require_columns(q, QVALS_COLUMNS, "qvals_table")
    n0 = len(stop)
    stop = stop.merge(
        q[QVALS_COLUMNS], on="stop_id", how="inner", validate="many_to_one"
    )
    # fail loudly if a task covariate the specs need never made it onto the stop
    # table (rename / add_uncertainty_covariates / qvals regression) -- otherwise
    # every model using it would silently drop out of the output.
    require_columns(stop, STOP_COVARS, "the stop table joined to qvals_table")
    logger.info(
        "stop x qvals join: %d -> %d stops (%.0f%% kept)",
        n0,
        len(stop),
        100 * len(stop) / n0,
    )
    stop, n_masked_by_state = mask_post_outcome_covariates(stop)
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
    return stop, int(n0 - len(stop))


def home_cohort(model: str) -> str:
    """The cohort a rung's own coefficients are reported from."""
    return POST_OUTCOME_COHORT if model in POST_OUTCOME_RATE_RUNGS else PRIMARY_COHORT


def contrast_cohort_records(
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
    if len(models):
        # The ladder null is fit once per cohort; without the column it would
        # be audited twice against the same rows.
        require_columns(models, ["cohort"], "the fitted rows")
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
        ("G", PRIMARY_COHORT): primary_union(G_SPECS),
        ("S", PRIMARY_COHORT): primary_union(S_SPECS),
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
        stop_level = [c for c in need_by_model.get(name, ()) if c in stop.columns]
        ladder = "S" if pop == "contrast" else "G"
        cohort_mode = (
            "common" if cohort == PRIMARY_COHORT else f"common_{POST_OUTCOME_COHORT}"
        )
        records.append(
            fitted_cohort_record(
                name,
                POP_LABEL.get(pop, pop),
                "stop",
                sub,
                count_column="n_stops",
                cohort_mode=cohort_mode,
                n_in=total + int(n_qvals_dropped),
                sessions_in=n_sessions_in,
                own_missing=stop_own_missing(stop, stop_level),
                common_missing=union_missing[(ladder, cohort)],
                n_qvals_gap=int(n_qvals_dropped),
                n_frozen_out=int(
                    frozen_drops.get(
                        pop if cohort == PRIMARY_COHORT else (pop, cohort), 0
                    )
                ),
                drop_notes=(
                    "qvals-join drop -> qvals_gap (mechanism B); immobility<=0 / NaN "
                    "own stop-level predictor -> own; NaN only in another rung's "
                    "column of this cohort's union -> common; bin-level + session "
                    "gates -> excluded"
                ),
            )
        )
    return records


def covariate_sds(qv: pd.DataFrame, st: pd.DataFrame) -> dict[str, float]:
    """Per-covariate SD over one (qvals, stop) pair.

    Each name in :data:`RATE_CONT_COVARS` is taken from whichever table carries
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
        in the order of :data:`RATE_CONT_COVARS`.
    """
    not_in_qvals = [c for c in RATE_CONT_COVARS if c not in qv.columns]
    sd = {**sd_reference(st, not_in_qvals), **sd_reference(qv, RATE_CONT_COVARS)}
    return {c: sd[c] for c in RATE_CONT_COVARS if c in sd}


def max_relative_sd_difference(
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


def contrast_term_diagnostics(models: pd.DataFrame) -> pd.DataFrame:
    """Term-level quasi-separation diagnostics of every key term, by cohort.

    Each key term (:data:`G_KEY` per stratum, :data:`S_KEY` on the contrast) is
    diagnosed on the cohort its rung was fit on (:func:`home_cohort`), with
    :func:`~continuum_swr_replay.analysis.behavior.cohort_audit.term_diagnostics`.

    Parameters
    ----------
    models : pd.DataFrame
        Per-session fits, one row per (population, model, session, cohort).

    Returns
    -------
    pd.DataFrame
        ``term_diagnostics``' rows with a ``cohort`` column.
    """
    td_specs = [(g, p, t) for g, t in G_KEY.items() for p in STRATA]
    td_specs += [(s, "contrast", t) for s, t in S_KEY.items()]
    return pd.concat(
        [
            term_diagnostics(
                models[models.cohort == cohort],
                [sp for sp in td_specs if home_cohort(sp[0]) == cohort],
            ).assign(cohort=cohort)
            for cohort in (PRIMARY_COHORT, POST_OUTCOME_COHORT)
        ],
        ignore_index=True,
    )


def cohort_sd_reference(
    qv_full: pd.DataFrame, st_full: pd.DataFrame, stop_cohort: str
) -> dict:
    """The per-SD reference of the ``std_*`` columns: the fitted stop cohort's SDs.

    A ``std_*`` column rescales a coefficient fit on ONE stop cohort, so its SD
    has to come from that same cohort: dividing an at-well coefficient by an
    all-stops SD reports a per-SD effect for a population the rung was never fit
    on. ``all_stops`` keeps every stop, so the restriction is a no-op there.

    Parameters
    ----------
    qv_full, st_full : pd.DataFrame
        The whole ``qvals_table`` and ``stop_table``.
    stop_cohort : str
        The stop cohort the fits name.

    Returns
    -------
    dict
        ``sd`` (:func:`covariate_sds` over the cohort's stops), ``sd_full``
        (over the full tables, for comparison only), ``max_sd_covariate`` and
        ``max_sd_relative_difference``
        (:func:`max_relative_sd_difference` of the two), and ``n_stops`` /
        ``n_stops_full``.
    """
    st_full = add_uncertainty_covariates(st_full)
    st, _ = select_stop_cohort(st_full, cohort=stop_cohort)
    qv = qv_full[qv_full["stop_id"].isin(set(st["stop_id"]))]
    sd = covariate_sds(qv, st)
    sd_full = covariate_sds(qv_full, st_full)
    max_sd_covar, max_sd_rel = max_relative_sd_difference(sd, sd_full)
    return {
        "sd": sd,
        "sd_full": sd_full,
        "max_sd_covariate": max_sd_covar,
        "max_sd_relative_difference": max_sd_rel,
        "n_stops": len(st),
        "n_stops_full": len(st_full),
    }


def aggregate_contrast_fits(
    models: pd.DataFrame, sd: dict[str, float], stop_cohort: str
) -> pd.DataFrame:
    """The audited aggregation of the per-session contrast fits.

    Each key term is aggregated (:func:`~continuum_swr_replay.analysis.behavior.contrast_aggregation.aggregate_contrast`,
    per-animal primary, session signed-rank, BH-FDR, ``std_*`` columns from
    ``sd``) over the cohort its rung was fit on; the null refit on the
    post-outcome cohort exists for the AIC comparison only and is never
    aggregated as a second copy of the ladder null. The G-rung populations are
    relabeled ``non_local``/``_swr``/``_no_swr`` -> ``all``/``SWR``/``nonSWR``
    (:data:`POP_LABEL`), as in the content table.

    Parameters
    ----------
    models : pd.DataFrame
        Per-session fits, one row per (population, model, session, cohort).
    sd : dict of str to float
        The per-SD reference (:func:`cohort_sd_reference`'s ``sd``).
    stop_cohort : str
        The stop cohort the fits were made on, recorded on every row.

    Returns
    -------
    pd.DataFrame
        One row per (rung, population, term, cohort).
    """
    specs = [(g, p, t) for g, t in G_KEY.items() for p in POP_LABEL]
    specs += [(s, "contrast", t) for s, t in S_KEY.items()]
    agg = pd.concat(
        [
            aggregate_contrast(
                models[models.cohort == cohort],
                [sp for sp in specs if home_cohort(sp[0]) == cohort],
                "rate",
                sd,
            ).assign(cohort=cohort)
            for cohort in (PRIMARY_COHORT, POST_OUTCOME_COHORT)
        ],
        ignore_index=True,
    )
    agg["population"] = agg["population"].map(lambda p: POP_LABEL.get(p, p))
    agg["stop_cohort"] = stop_cohort
    return agg
