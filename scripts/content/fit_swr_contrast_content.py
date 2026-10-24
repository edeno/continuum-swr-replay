#!/usr/bin/env python
"""Fit the beta-bernoulli replay-CONTENT ladder (arm + patch), is_swr version.

The content analog of ``scripts/rate/fit_swr_contrast_rate.py``. Where the rate ladder
models HOW OFTEN non-local events occur (Poisson count per time bin), this models
WHICH ARM / PATCH an event represents: a conditional Poisson over the 9 arms (or 3
patches) within each event (``strata(event_id)`` ≡ clogit). Design spec:
``docs/design/content_model_design.md``.

Structural rule: in a conditional-over-arms likelihood a covariate CONSTANT within
an event cancels, so all event-level scalars (reward, rpe, value, entropy, context,
SWR status) enter ONLY as interactions with arm features. Per-arm covariates
(``reward_fraction_candidate_arm_last10_visits``,
``reward_belief_mean_candidate_arm_preoutcome``,
``is_last_rewarded_noncurrent_arm``)
enter as main effects.

Ladders (arm; fit per session, per is_swr split {all, SWR, non-SWR}); the rungs
are defined in ``model_specs.SWR_CONTRAST_CONTENT_ARM_LADDER`` /
``SWR_CONTRAST_CONTENT_PATCH_LADDER``:
  C0..C2  spatial baseline (ARMFE -> is_current -> +autocorrelation controls)
  C3..C5  per-arm value/reward CONTENT (main effects)
  C6..C9  state modulation of current-bias (rpe/value/uncert/update : is_current)
  C10..C12 CONTEXT split (periswitch/block/switchrate : is_current)
  C13     per-arm belief uncertainty (split)
  CS1..CS6 is_swr:feature contrasts (pooled, 2-way; the dissociation): CS1-CS5,
          CS5b (value net of the previous goal), CS6 (belief uncertainty), and the
          ``nt`` twins CS1nt / CS4nt / CS5nt / CS6nt
  patch   CP0..CP4 dimension rungs; CPS1 / CPS1nt / CPS2 contrasts
Event-scalar dissociations (C6-C12) come from comparing the split, not a pooled 3-way.

Every rung built on the base (``BASEc`` / ``BASEc_PATCH``) carries a within-stop-time
control AND an event-length control -- z(log ``n_nonlocal_bins``) crossed with the
current and previous arm/patch indicators. SWR-coupled events are the longer ones and
a longer trajectory visits more arms, so without it ``swr_x_is_current_arm`` partly
reads "longer events leave the current arm". ``CS1nt_current`` / ``CPS1nt_current``
are the deliberately uncontrolled sensitivity twins. Because rewarded and omitted
stops differ in their SWR share, ``--outcome-stratum {all,rewarded,omitted}`` refits
the ladder within one current-outcome stratum (an event whose outcome is not yet
known is in neither) into ``_rewarded`` / ``_omitted`` artifacts.

Aggregation is AUDITED. Per-session conditional-Poisson coefficients carry SE and
p-values. **Primary population evidence is at the animal level (n=5)**: the animal-
balanced median coef and per-animal sign consistency (``n_animals_pos`` / ``n_animals``).
The Wilcoxon signed-rank across sessions is reported as a **descriptive** consistency
statistic ONLY — sessions within an animal are not independent, so it is not a valid
population p-value — with a Benjamini-Hochberg FDR across the rung family per population.
``typical_session_se`` is the median within-session SE, not the aggregate's uncertainty.
Per-arm value uses the beta-bernoulli construct
(``reward_belief_mean_candidate_arm_preoutcome``). The response is the per-event
allocation fraction (``--response hard_count`` refits the timebin counts into
separate ``_hard_count`` artifacts). A provenance sidecar (git hash, params) is
written, plus a ``.provenance.json`` beside each fitted pickle recording the
tables it was fit on and a ``.z_reference.json`` recording the fit-cohort
standardization moments. DEFERRED follow-ups (in the provenance): the **hierarchical Poisson
with animal random effects** that would replace the descriptive session signed-rank with
rigorous population inference; a common complete-case cohort; and within-group-centered
VIF (`fitting.py` uses the raw design).

Usage:
    uv run python scripts/content/fit_swr_contrast_content.py               # full
    uv run python scripts/content/fit_swr_contrast_content.py --max-per-animal 2   # smoke
    uv run python scripts/content/fit_swr_contrast_content.py --estimator dirichlet_multinomial
    uv run python scripts/content/fit_swr_contrast_content.py --outcome-stratum rewarded
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from continuum_swr_replay.analysis.replay.cohort_audit import (
    cohort_record,
    drop_profile,
    estimator_skip_note,
    term_diagnostics,
    write_cohort_audit,
)
from continuum_swr_replay.analysis.replay.content_response import (
    ESTIMATOR_CHOICES,
    RESPONSE_CHOICES,
    default_response_for_estimator,
    renormalize_fraction_over_candidates,
    resolve_response_choice,
    resolve_response_column,
    validate_fraction_response,
)
from continuum_swr_replay.analysis.replay.contrast_aggregation import (
    CONTENT_FIT_INPUTS,
    LOG_EVENT_LENGTH,
    OUTCOME_STRATA,
    aggregate_contrast,
    assemble_arm,
    assemble_patch,
    prepare_cohorts,
    prepare_session,
    specs_from_ladder,
)
from continuum_swr_replay.analysis.replay.covariate_timing import (
    assert_predictor_timing,
)
from continuum_swr_replay.analysis.replay.dirichlet_multinomial import (
    DEFAULT_MIN_EVENTS_SESSION_PHI,
    DEFAULT_N_JOBS,
    ESTIMATOR_SKIP_CLASSES,
    concentration_cols_for_cohort,
    fit_sessions_dirichlet_multinomial,
)
from continuum_swr_replay.analysis.replay.dirichlet_multinomial import (
    ESTIMATOR as DIRICHLET_MULTINOMIAL,
)
from continuum_swr_replay.analysis.replay.fit_provenance import (
    git_hash as loaded_commit,
)
from continuum_swr_replay.analysis.replay.fit_provenance import (
    record_inputs,
    write_fit_provenance,
)
from continuum_swr_replay.analysis.replay.fitting import (
    fit_conditional_poisson,
    poisson_estimator_name,
)
from continuum_swr_replay.analysis.replay.model_specs import (
    DESCRIPTIVE_CONTENT_RUNGS,
    LEAF_ONLY_CONTRAST_RUNGS,
    LEAF_ONLY_N_CATEGORIES,
    leaf_only_predictors,
)
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
from continuum_swr_replay.paths import REPLAY_CONTENT_DIR, model_dir
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS
from continuum_swr_replay.tables.standardization import (
    save_reference,
    z_reference_path,
)

# Quasi-separated coefficients legitimately overflow exp() when rate ratios are formed
# outside the fit (RuntimeWarning). Convergence/perfect-separation warnings are recorded
# per fit as converged/separated/n_convergence_warnings flags inside the fitters (see
# analysis.replay._fit_diagnostics), so only the benign RuntimeWarning noise is silenced
# here -- NOT a blanket ignore that would hide fit-quality signals.
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RC = REPLAY_CONTENT_DIR
SC = list(SESSION_COLUMNS)

#: Concentration contrast the Dirichlet-multinomial adds to every rung.
LOG_PHI_IS_SWR = "log_phi_is_swr"

#: Event-length term of the length-adjusted concentration design
#: (``--concentration-length``). Reported per SD of ``log n_nonlocal_bins``.
LOG_PHI_EVENT_LENGTH = "log_phi_log_n_nonlocal_bins_z"

# Skip classes that are the ESTIMATOR refusing a session rather than the cohort
# rules excluding it. These are rare and actionable, so the audit names the
# session; ``too_few`` / ``missing`` are routine and are summarized by class.
# Shared with the content runner so the two report skips the same way.
_ESTIMATOR_SKIP_CLASSES = ESTIMATOR_SKIP_CLASSES


def resolve_concentration_cols(
    frame: pd.DataFrame, *, with_event_length: bool
) -> tuple[str, ...]:
    """Event-level columns entering ``log phi`` for one population's rows.

    The Dirichlet-multinomial likelihood uses an event's length ``L_e`` only as
    the multinomial index: ``log phi_e`` carries no length term, so the fitted
    concentration is assumed to be the same for a 25-bin event and a 90-bin one.
    SWR events run about 1.7x longer (median 45 vs 26 bins) than non-SWR events, so the shipped
    ``log_phi_is_swr`` is the TOTAL concentration contrast -- whatever an SWR
    event does to within-event redundancy, length included. Adding
    ``log_n_nonlocal_bins_z`` to the concentration design gives the
    length-adjusted twin, exactly parallel to the total/direct distinction the
    ``nt`` rungs already draw on the mean structure.

    The length term is added only where the cohort identifies ``is_swr`` at all:
    inside an SWR (or non-SWR) split there is no concentration contrast to
    decompose, and the fits there must keep the same design as before.

    Parameters
    ----------
    frame : pd.DataFrame
        The rows about to be fitted.
    with_event_length : bool
        Whether to append the z-scored log event length.

    Returns
    -------
    tuple of str

    Raises
    ------
    ValueError
        When the length term is requested and the cohort does not carry it. A
        silently dropped term would publish a ``_dm_len`` artifact identical to
        the unadjusted one under a name that claims otherwise.
    """
    base = concentration_cols_for_cohort(frame)
    if not with_event_length or base != ("is_swr",):
        return base
    column = f"{LOG_EVENT_LENGTH}_z"
    if column not in frame.columns:
        raise ValueError(
            f"--concentration-length needs {column!r} on the cohort; the "
            "assembled arm and patch frames standardize it in "
            "contrast_aggregation.assemble_arm / assemble_patch"
        )
    return (*base, column)


def _artifact_suffix(args: argparse.Namespace, response: str) -> str:
    """Artifact-name suffix: smoke subset, response, estimator, and stratum.

    The response is named only where it is not the estimator's own default, so
    a Dirichlet-multinomial artifact says ``_dm`` rather than repeating in
    ``_hard_count`` the one response that estimator has. ``_dm`` comes before
    the stratum suffix, so the stratified artifacts of one estimator sort
    together, and the length-adjusted concentration twin is ``_dm_len`` -- a
    different estimand of ``log_phi_is_swr``, so it must never overwrite the
    total one.
    """
    estimator = args.estimator
    smoke = "_smoke" if args.max_per_animal is not None else ""
    default_response = default_response_for_estimator(estimator)
    response_suffix = "" if response == default_response else f"_{response}"
    dm = "_dm" if estimator == "dirichlet_multinomial" else ""
    if dm and args.concentration_length:
        dm += "_len"
    stratum = "" if args.outcome_stratum == "all" else f"_{args.outcome_stratum}"
    return smoke + response_suffix + dm + stratum


def _session_audit(prep: dict, r: dict | None, *, reason: str = "") -> dict:
    """Close one session's audit record around whatever the fitter returned.

    Any survivor not modeled -- the fitter's ``n < 2 * n_params`` gate, a failed
    fit, or a design the estimator refused -- is a session/fit-level exclusion,
    which is what makes ``n_in - Σdrops == n_modeled`` reconcile exactly.
    """
    n_modeled = int(r["n_events"]) if r is not None else 0
    if r is None:
        reason = reason or prep["reason"] or "fit_none"
    else:
        reason = ""
    return _audit(
        prep["n_in"],
        prep["n_qvals"],
        prep["n_own"],
        prep["n_zero"],
        prep["n_surv"] - n_modeled,
        n_modeled,
        reason,
        prep["dropped"],
    )


def _fit_session(
    g: pd.DataFrame, predictors: list[str], *, response_col: str, n_categories: int
) -> tuple[dict | None, dict]:
    """Fit one session's conditional Poisson and return ``(row_or_None, audit)``."""
    surv, prep = prepare_session(g, predictors, response_col=response_col)
    if surv is None:
        return None, _session_audit(prep, None)
    r = fit_conditional_poisson(
        surv,
        predictors=predictors,
        response_col=response_col,
        event_col="event_id",
        n_categories=n_categories,
    )
    return r, _session_audit(prep, r)


def _audit(n_in, n_qvals, n_own, n_zero, n_excluded, n_modeled, reason, dropped):
    return {
        "n_in": n_in,
        "n_drop_qvals_gap": n_qvals,
        "n_drop_own_predictor_missing": n_own,
        "n_drop_zero_response": n_zero,
        "n_excluded_session": n_excluded,
        "n_modeled": n_modeled,
        "reason": reason,
        "dropped": dropped,
    }


def _fit_population(
    pop_frame: pd.DataFrame,
    predictors: list[str],
    *,
    estimator: str,
    response_col: str,
    n_categories: int,
    n_jobs: int,
    min_events_session_phi: int,
    concentration_length: bool = False,
) -> list[tuple[tuple, dict | None, dict]]:
    """Fit every session of one (rung, population), in groupby order.

    Returns ``(session_key, row_or_None, audit)`` per session. Both estimators
    see exactly the same prepared rows; only the fit differs. The
    Dirichlet-multinomial fits are batched so they can share a process pool and
    a pooled concentration intercept for the short sessions, and a session whose
    within-event design it refuses becomes an audited skip
    (``reason="dm_design:..."``) rather than an exception out of the ladder.
    """
    if estimator != "dirichlet_multinomial":
        return [
            (
                tuple(keys),
                *_fit_session(
                    g,
                    predictors,
                    response_col=response_col,
                    n_categories=n_categories,
                ),
            )
            for keys, g in pop_frame.groupby(SC, sort=False)
        ]

    prepared: list[tuple[tuple, pd.DataFrame | None, dict]] = [
        (tuple(keys), *prepare_session(g, predictors, response_col=response_col))
        for keys, g in pop_frame.groupby(SC, sort=False)
    ]
    animal_position = SC.index("animal")
    fits, skips = fit_sessions_dirichlet_multinomial(
        [
            (key, key[animal_position], surv)
            for key, surv, _prep in prepared
            if surv is not None
        ],
        predictors,
        min_events=min_events_session_phi,
        n_jobs=n_jobs,
        response_col=response_col,
        event_col="event_id",
        concentration_cols=resolve_concentration_cols(
            pop_frame, with_event_length=concentration_length
        ),
        n_categories=n_categories,
    )
    rows = dict(fits)
    reasons = dict(skips)
    return [
        (
            key,
            rows.get(key),
            _session_audit(prep, rows.get(key), reason=reasons.get(key, "")),
        )
        for key, _surv, prep in prepared
    ]


def _run(
    ev: pd.DataFrame,
    ladder: dict,
    *,
    response_col: str,
    n_categories: int,
    estimator: str = "conditional_poisson",
    n_jobs: int = 1,
    min_events_session_phi: int = DEFAULT_MIN_EVENTS_SESSION_PHI,
    concentration_length: bool = False,
) -> tuple[pd.DataFrame, list[dict]]:
    """Fit the ladder per session and build one cohort-audit record per (model, pop).

    Returns ``(results, cohort_records)``; the audit records reconcile
    ``n_in - Σdrops == n_modeled`` (grain = event) via :func:`cohort_record`.
    """
    rows = []
    audits = []
    splits = {"all": ev, "SWR": ev[ev.is_swr], "nonSWR": ev[~ev.is_swr]}
    for name, (preds, scope) in ladder.items():
        pops = ["all"] if scope == "pooled" else ["all", "SWR", "nonSWR"]
        for pop in pops:
            acc = dict.fromkeys(
                (
                    "n_in",
                    "n_drop_qvals_gap",
                    "n_drop_own_predictor_missing",
                    "n_drop_zero_response",
                    "n_excluded_session",
                    "n_modeled",
                ),
                0,
            )
            sessions_in = sessions_modeled = n_nonconv = n_sep = 0
            animals_modeled: set = set()
            session_skips: list[tuple[str, str]] = []
            dropped_frames = []
            # Belief rungs are fitted over the leaf arms only: the trunks carry a
            # structural zero for every leaf-derived belief covariate, so they add
            # no information about those constructs. Persisted z units are also
            # defined on leaf support. See LEAF_ONLY_CONTRAST_RUNGS.
            leaf_only = name in LEAF_ONLY_CONTRAST_RUNGS
            pop_frame = splits[pop]
            fit_categories = n_categories
            fit_preds = preds
            renormalized = False
            if leaf_only:
                pop_frame = pop_frame[
                    pop_frame["candidate_arm"] < LEAF_ONLY_N_CATEGORIES
                ]
                fit_categories = LEAF_ONLY_N_CATEGORIES
                fit_preds = leaf_only_predictors(preds)
                # A fraction is a share of the event's whole nine-arm mass, and
                # the conditional likelihood weights an event by the total it
                # carries, so dropping the trunks unrescaled would weight an
                # event by its leaf share -- and SWR events put far more of
                # their bins on the trunks than non-SWR events do, which is the
                # very contrast being estimated. Rescale to unit mass over the
                # six leaves. Counts keep their timebin weighting by design.
                renormalized = response_col.endswith("_fraction")
                if renormalized:
                    pop_frame = renormalize_fraction_over_candidates(
                        pop_frame, response_col=response_col
                    )
            for keys, r, a in _fit_population(
                pop_frame,
                list(fit_preds),
                estimator=estimator,
                response_col=response_col,
                n_categories=fit_categories,
                n_jobs=n_jobs,
                min_events_session_phi=min_events_session_phi,
                concentration_length=concentration_length,
            ):
                sessions_in += 1
                for k in acc:
                    acc[k] += a[k]
                if len(a["dropped"]):
                    dropped_frames.append(a["dropped"])
                if r is None:
                    if a["reason"]:
                        session_skips.append(("/".join(map(str, keys)), a["reason"]))
                    continue
                sessions_modeled += 1
                animals_modeled.add(keys[SC.index("animal")])
                n_nonconv += r.get("converged") is False
                n_sep += bool(r.get("separated"))
                rows.append(
                    {
                        "model": name,
                        "population": pop,
                        **dict(zip(SC, keys, strict=True)),
                        # ``sd_<term>`` travels with the coefficients because the
                        # aggregation's identifiability backstop is read on the
                        # effect scale (|coef| x sd). Without it a runaway fit is
                        # judged by its raw coefficient, which drops identified
                        # sessions whose predictor has a small SD and keeps
                        # unidentified ones whose predictor has a large one.
                        **{
                            k: v
                            for k, v in r.items()
                            if k.startswith(("coef_", "bse_", "pval_", "sd_"))
                        },
                        # Per-term separation flags, so a fit whose self-history
                        # coefficient ran away is not dropped for every term.
                        "separated_terms": r.get("separated_terms"),
                        "n_events": r["n_events"],
                        # Events on each side of the is_swr concentration
                        # modifier. A modifier resting on one or two SWR events
                        # is finite, unflagged, and invisible to every
                        # identifiability rule; only these counts expose it.
                        # NaN under the conditional Poisson and in the within-SWR
                        # splits, where there is no such modifier.
                        "n_events_swr": r.get("n_events_swr", np.nan),
                        "n_events_nonswr": r.get("n_events_nonswr", np.nan),
                        "n_params": r["n_params"],
                        # Which allocation the row was fit on, and therefore at
                        # which grain its llf/AIC/SEs are stated.
                        "estimator": r.get("estimator"),
                        "response_col": response_col,
                        # True where the trunk rows were dropped and the
                        # remaining fractions rescaled to unit mass per event.
                        "response_renormalized": renormalized,
                        # Whether the row's llf carries the response's
                        # normalizing constants, so a later comparison cannot
                        # difference a full likelihood against a conditional one.
                        "llf_includes_constants": r.get("llf_includes_constants"),
                        # Where this fit's concentration intercept came from:
                        # its own likelihood, or the animal-pooled median that a
                        # session too short to estimate it is pinned to.
                        "phi_source": r.get("phi_source"),
                        # fit-diagnostics flags (additive; drive the aggregation's
                        # session-level nonconverged/separated counts).
                        "converged": r.get("converged"),
                        "separated": r.get("separated"),
                        "n_convergence_warnings": r.get("n_convergence_warnings"),
                        # Whether the fit reached its optimum, for the bundle's
                        # gates: the conditional-Poisson fitter's Newton finish
                        # (NaN for an estimator without one). Terms the session
                        # cannot inform are listed, "|"-joined, with NaN
                        # coefficients.
                        "llf": r.get("llf", np.nan),
                        "newton_decrement": r.get("newton_decrement", np.nan),
                        "max_abs_grad": r.get("max_abs_grad", np.nan),
                        "n_newton_steps": r.get("n_newton_steps", np.nan),
                        "unidentified_terms": r.get("unidentified_terms", ""),
                    }
                )
            dropped = (
                pd.concat(dropped_frames, ignore_index=True)
                if dropped_frames
                else pd.DataFrame()
            )
            notes = drop_profile(dropped)
            # Sessions that produced no fit are already counted under
            # n_excluded_session; naming why keeps a design the estimator
            # refused distinguishable from a session that was merely too short.
            # The routine classes are summarized; a refusal by the estimator
            # itself names the session it happened to. Same helper (and so the
            # same policy) as the content runner's cohort audit.
            skip_note = estimator_skip_note(session_skips, _ESTIMATOR_SKIP_CLASSES)
            if skip_note:
                notes = "; ".join(filter(None, [notes, skip_note]))
            audits.append(
                cohort_record(
                    name,
                    pop,
                    "event",
                    n_in=acc["n_in"],
                    n_modeled=acc["n_modeled"],
                    sessions_in=sessions_in,
                    sessions_modeled=sessions_modeled,
                    animals_modeled=len(animals_modeled),
                    drop_notes=notes,
                    n_drop_qvals_gap=acc["n_drop_qvals_gap"],
                    n_drop_own_predictor_missing=acc["n_drop_own_predictor_missing"],
                    n_drop_zero_response=acc["n_drop_zero_response"],
                    n_excluded_session=acc["n_excluded_session"],
                    n_sessions_flagged_nonconverged=n_nonconv,
                    n_sessions_flagged_separated=n_sep,
                )
            )
    return pd.DataFrame(rows), audits


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


#: Concentration terms aggregated on every rung. ``log_phi_log_n_nonlocal_bins_z``
#: is only fitted under ``--concentration-length``; on a run without it the
#: coefficient column is simply absent and ``aggregate_contrast`` emits no row.
_CONCENTRATION_TERMS = (LOG_PHI_IS_SWR, LOG_PHI_EVENT_LENGTH)


def _concentration_specs(ladder: dict) -> list[tuple[str, str, str]]:
    """Aggregation specs for the concentration contrast, one per rung x population.

    ``log_phi_is_swr`` is not any rung's key term -- it is the same quantity on
    every rung, the difference in how tightly an SWR event's mass is
    concentrated -- so it is aggregated as its own family rather than folded
    into the beta contrast's BH-FDR. The same holds for the event-length term of
    the length-adjusted design, which is what the ``is_swr`` contrast is being
    adjusted for and must be readable next to it. Specs are emitted for every
    rung, population and concentration term; the ones whose fits carry no such
    coefficient (the conditional Poisson everywhere, the within-SWR splits where
    the flag is constant, and the length term on a run that did not fit it)
    produce no row.
    """
    return [
        (name, pop, term)
        for name in ladder
        for pop in (
            ["all"] if ladder[name][1] == "pooled" else ["all", "SWR", "nonSWR"]
        )
        for term in _CONCENTRATION_TERMS
    ]


def _renormalized_rungs(*results: pd.DataFrame) -> list[str]:
    """Rung names whose fitted rows were produced on a rescaled response.

    Read off the rows that were actually fitted rather than off the leaf-only
    rung set, so a run that skipped or never requested a rung cannot claim it.

    Parameters
    ----------
    *results : pd.DataFrame
        Per-session result frames carrying ``model`` and, where the fit rescaled
        its response to the fitted candidate set, ``response_renormalized``.

    Returns
    -------
    list of str
        Sorted rung names, empty when no fitted row was rescaled.
    """
    names: set[str] = set()
    for res in results:
        if res.empty or "response_renormalized" not in res.columns:
            continue
        rescaled = res["response_renormalized"].fillna(False).astype(bool)
        names |= set(res.loc[rescaled, "model"])
    return sorted(names)


def _provenance(
    args: argparse.Namespace,
    n_sessions: int,
    z_reference: dict | None = None,
    outcome_stratum_audit: dict | None = None,
    per_sd_scale: dict | None = None,
    renormalized_rungs: list[str] | None = None,
    concentration_cols: list[str] | None = None,
) -> dict:
    """Reproducibility record saved alongside the audited results."""
    return {
        # The commit this process loaded, suffixed -dirty if its tracked files
        # were edited then: the value the fit sidecars record. A refit runs for
        # hours in a checkout that other work commits in, so the HEAD at write
        # time can be newer than the code that ran.
        "git_hash": loaded_commit(),
        # Fit-time _z reference (event-cohort mean/SD for the event scalars). This is
        # the runner's OWN standardization -- NOT the builder's pooled-global table _z.
        "fit_time_z_reference": z_reference or {},
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_sessions": int(n_sessions),
        "smoke_max_per_animal": args.max_per_animal,
        # Which per-session estimator produced the coefficients, and therefore
        # whether the table carries a concentration contrast at all.
        "estimator": args.estimator,
        "min_events_session_phi": args.min_events_session_phi,
        # Event-level columns entering log phi on the pooled populations. With
        # the length term the is_swr coefficient is the LENGTH-ADJUSTED
        # concentration contrast; without it, the total one.
        "concentration_cols": list(concentration_cols or []),
        "concentration_length_adjusted": (
            f"{LOG_EVENT_LENGTH}_z" in (concentration_cols or [])
        ),
        # Cohort selection: which current-outcome stratum was fit, and how many
        # events had no known outcome and were therefore in neither stratum.
        "outcome_stratum": args.outcome_stratum,
        "outcome_stratum_audit": outcome_stratum_audit or {},
        # Per-SD units the audited table's standardized effects are stated in.
        # Frozen over the full cohort, so they do not move between strata.
        "per_sd_scale": per_sd_scale or {},
        "inference": "per-session conditional Poisson; across-session Wilcoxon "
        "signed-rank across sessions is DESCRIPTIVE only (sessions within an animal "
        "are not independent); primary population evidence is per-animal sign "
        "consistency (n_animals_pos / n_animals, n=5). BH-FDR across the rung family "
        "per population is applied to the descriptive session signed-rank p.",
        # Rungs whose fitted rows were produced on a response rescaled to the
        # candidate set they conditioned on (the leaf-only rungs, on the
        # fraction response). Read off the rows that were fitted, so a rung that
        # produced none is not named here.
        "response_renormalized_rungs": list(renormalized_rungs or []),
        # Legacy key name, kept because existing sidecars use it; the value is
        # the arm-value construct (reward_belief_mean_candidate_arm_preoutcome).
        "q_arm_source": (
            "beta-bernoulli per-arm value (reward_belief_mean_candidate_arm_preoutcome)"
        ),
        "deferred": [
            "mixed conditional logit with animal random effects (the rigorous "
            "population inference; supersedes the descriptive session signed-rank p). "
            "NOT available in statsmodels (ConditionalPoisson takes no random effects "
            "or cluster-robust cov); realistic path is NumPyro (JAX already a dep). "
            "Caveat: the animal random-effect variance is estimated from n=5 groups, "
            "so this does honest bookkeeping, not a power upgrade.",
            "common complete-case cohort across rungs",
            "within-group-centered VIF (fitting.py:420 uses the raw design)",
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
    ap.add_argument(
        "--estimator",
        type=str,
        default="conditional_poisson",
        choices=list(ESTIMATOR_CHOICES),
        help=(
            "Per-session estimator. 'conditional_poisson' (default, shipped) "
            "fits the softmax allocation. 'dirichlet_multinomial' adds an "
            "explicit within-event concentration, so an event's autocorrelated "
            "timebins are not counted as independent draws and every rung also "
            f"reports coef_{LOG_PHI_IS_SWR} -- a direct test of whether SWR "
            "events spread their mass over more arms. It is defined on the "
            "integer counts, so it forces --response hard_count, and it writes "
            "'_dm' artifacts of its own."
        ),
    )
    ap.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help=(
            "Worker processes for the Dirichlet-multinomial per-session fits "
            f"(default: {DEFAULT_N_JOBS}). Ignored by the conditional Poisson."
        ),
    )
    ap.add_argument(
        "--min-events-session-phi",
        type=int,
        default=DEFAULT_MIN_EVENTS_SESSION_PHI,
        help=(
            "Modelled events a session needs to estimate its own "
            "Dirichlet-multinomial concentration intercept (default: "
            f"{DEFAULT_MIN_EVENTS_SESSION_PHI}); shorter sessions are pinned to "
            "the median intercept of that animal's longer sessions. This "
            "changes the fitted COHORT, not just a nuisance parameter: a "
            "session just under the threshold is refit with its intercept "
            "pinned, and if that refit fails it is dropped (reason "
            "'pinned_fit_none') even though its free fit had converged."
        ),
    )
    ap.add_argument(
        "--concentration-length",
        action="store_true",
        help=(
            "Dirichlet-multinomial only: add z(log n_nonlocal_bins) to the "
            "concentration design, giving the LENGTH-ADJUSTED twin of "
            f"coef_{LOG_PHI_IS_SWR}. The likelihood otherwise treats log phi as "
            "length-invariant, and SWR events run ~1.7x longer, so the default "
            "coefficient is the TOTAL concentration contrast with length "
            "included. Writes '_dm_len' artifacts of its own and also reports "
            f"coef_{LOG_PHI_EVENT_LENGTH} per SD of log event length."
        ),
    )
    ap.add_argument(
        "--response",
        type=str,
        default=None,
        choices=list(RESPONSE_CHOICES),
        help=(
            "Per-event response. Defaults to the estimator's own: 'fraction' "
            "for the conditional Poisson, 'hard_count' for the "
            "Dirichlet-multinomial (which has no fraction form). 'fraction' fits "
            "arm_fraction / patch_fraction, so each event contributes one "
            "observation of unit mass. 'hard_count' fits the timebin counts, "
            "weighting an event by its length and treating its autocorrelated "
            "bins as independent draws; it writes '_hard_count' artifacts so "
            "the two never overwrite each other."
        ),
    )
    ap.add_argument(
        "--outcome-stratum",
        type=str,
        default="all",
        choices=list(OUTCOME_STRATA),
        help=(
            "Current-outcome stratum to fit. Rewarded and omitted stops differ in "
            "their SWR share, so 'rewarded' / 'omitted' refit the ladder within "
            "one outcome (events whose outcome is not yet known are in neither) "
            "and write '_rewarded' / '_omitted' artifacts."
        ),
    )
    args = ap.parse_args()
    if args.concentration_length and args.estimator != "dirichlet_multinomial":
        # The conditional Poisson has no concentration parameter at all, so
        # there is nothing for a length term to enter; accepting the flag would
        # write an artifact whose name promises an adjustment it never made.
        ap.error(
            "--concentration-length is a Dirichlet-multinomial option; the "
            f"{args.estimator!r} estimator has no concentration design"
        )
    # One resolution point for the response: the estimator's own default, and a
    # refusal where the two cannot be combined.
    response = resolve_response_choice(args.estimator, args.response)
    arm_response = resolve_response_column("arm", response)
    patch_response = resolve_response_column("patch", response)
    estimator = (
        DIRICHLET_MULTINOMIAL
        if args.estimator == "dirichlet_multinomial"
        else poisson_estimator_name(arm_response)
    )
    fit_kwargs = {
        "estimator": args.estimator,
        "n_jobs": args.n_jobs,
        "min_events_session_phi": args.min_events_session_phi,
        "concentration_length": args.concentration_length,
    }
    # The concentration design as requested. Within an SWR / non-SWR split the
    # flag is constant and ``resolve_concentration_cols`` drops the whole design;
    # this records what the pooled populations were asked for.
    concentration_cols = (
        []
        if args.estimator != "dirichlet_multinomial"
        else ["is_swr", f"{LOG_EVENT_LENGTH}_z"]
        if args.concentration_length
        else ["is_swr"]
    )
    fit_extra = {"concentration_cols": concentration_cols}

    # Causal-hygiene gate: every rung's predictors must be pre-outcome, OR the
    # rung must be an explicitly-declared descriptive rung (RPE/value association or
    # peri-switch alignment). A NEW leaky covariate on an undeclared rung fails here.
    for ladder in (ARM_LADDER, PATCH_LADDER):
        for name, (preds, _scope) in ladder.items():
            assert_predictor_timing(
                preds, descriptive=(name in DESCRIPTIVE_CONTENT_RUNGS)
            )

    sfx = _artifact_suffix(args, response)
    # Not ``content_arm``: the arm content ladder publishes that directory by
    # replacing it whole, which would delete these fits.
    (model_dir("swr_contrast", RC)).mkdir(parents=True, exist_ok=True)
    # The standardization references are taken over the FULL assembled cohorts,
    # before the outcome stratum and the smoke subset, so every stratum's
    # coefficients are stated in the same per-SD unit.
    ev, arm_z_reference = assemble_arm()
    pt, patch_z_reference = assemble_patch(ev)
    # Hash the tables now, while they are the ones being fit: the ladder runs
    # for long enough that a table could be regenerated before the artifacts are
    # written, and a sidecar hashed at write time would certify bytes no fit saw.
    fit_inputs = record_inputs(CONTENT_FIT_INPUTS)
    if response == "fraction":
        validate_fraction_response(
            ev, arm_response, table_label=CONTENT_FIT_INPUTS["event_arm_table"]
        )
        validate_fraction_response(
            pt, patch_response, table_label=CONTENT_FIT_INPUTS["event_patch_table"]
        )
    logger.info("response = %s / %s (%s)", arm_response, patch_response, estimator)
    fit_z_reference = {"arm_cohort": arm_z_reference, "patch_cohort": patch_z_reference}
    ev, pt, per_sd_scale, stratum_audit = prepare_cohorts(
        ev,
        pt,
        outcome_stratum=args.outcome_stratum,
        max_per_animal=args.max_per_animal,
    )
    logger.info(
        "outcome stratum %s: %d of %d events kept; %d excluded for an unknown "
        "outcome (%d SWR / %d non-SWR)",
        stratum_audit["stratum"],
        stratum_audit["n_events_kept"],
        stratum_audit["n_events_in"],
        stratum_audit["n_events_unknown_outcome"],
        stratum_audit["n_events_unknown_outcome_swr"],
        stratum_audit["n_events_unknown_outcome_non_swr"],
    )

    logger.info(
        "ARM ladder: %d rows, %d events, %d sessions",
        len(ev),
        ev["event_id"].nunique(),
        ev.groupby(SC).ngroups,
    )
    arm, arm_audit = _run(
        ev, ARM_LADDER, response_col=arm_response, n_categories=9, **fit_kwargs
    )
    arm_path = model_dir("swr_contrast", RC) / f"swr_contrast_content_arm{sfx}.pkl"
    arm.to_pickle(arm_path)
    write_fit_provenance(arm_path, fit_inputs, estimator, sys.argv, extra=fit_extra)
    save_reference(arm_z_reference, z_reference_path(arm_path))

    logger.info(
        "PATCH ladder: %d rows, %d events, %d sessions",
        len(pt),
        pt["event_id"].nunique(),
        pt.groupby(SC).ngroups,
    )
    patch, patch_audit = _run(
        pt, PATCH_LADDER, response_col=patch_response, n_categories=3, **fit_kwargs
    )
    patch_path = model_dir("swr_contrast", RC) / f"swr_contrast_content_patch{sfx}.pkl"
    patch.to_pickle(patch_path)
    write_fit_provenance(patch_path, fit_inputs, estimator, sys.argv, extra=fit_extra)
    save_reference(patch_z_reference, z_reference_path(patch_path))

    # Cohort-audit table (one row per model x population; reconciles n_in - Σdrops =
    # n_modeled at the event grain) + term-level quasi-separation diagnostics.
    write_cohort_audit(
        arm_audit + patch_audit,
        model_dir("swr_contrast", RC) / f"swr_contrast_content_cohort_audit{sfx}.csv",
    )
    term_diag = pd.concat(
        [
            term_diagnostics(arm, specs_from_ladder(ARM_LADDER, ARM_KEY_TERM)),
            term_diagnostics(patch, specs_from_ladder(PATCH_LADDER, PATCH_KEY_TERM)),
        ],
        ignore_index=True,
    )
    term_diag.to_csv(
        model_dir("swr_contrast", RC)
        / f"swr_contrast_content_term_diagnostics{sfx}.csv",
        index=False,
    )

    logger.info(
        "saved arm=%d patch=%d rows -> %s",
        len(arm),
        len(patch),
        model_dir("swr_contrast", RC),
    )
    print("\n########## ARM ##########")
    _summ(arm, ARM_LADDER, ARM_KEY_TERM)
    print("\n########## PATCH ##########")
    _summ(patch, PATCH_LADDER, PATCH_KEY_TERM)

    # Audited aggregation: SE, per-session significance, per-animal sign, signed-rank
    # p, BH-FDR across the rung family, and standardized (per-SD) effect sizes — plus a
    # provenance sidecar. The per-SD units are the pooled SDs frozen over the full
    # cohort by prepare_cohorts, so a stratified refit is on the pooled unit.
    agg = pd.concat(
        [
            _aggregate(
                arm,
                ARM_LADDER,
                ARM_KEY_TERM,
                "candidate_arm",
                per_sd_scale["arm_cohort"],
            ),
            _aggregate(
                patch,
                PATCH_LADDER,
                PATCH_KEY_TERM,
                "patch",
                per_sd_scale["patch_cohort"],
            ),
            # The concentration contrast, next to the beta contrast it modifies:
            # under the Dirichlet-multinomial every rung also estimates how much
            # more (or less) an SWR event's mass is spread out, which is the
            # "SWR broadens" claim tested directly rather than read off a
            # coefficient on the mean structure.
            aggregate_contrast(arm, _concentration_specs(ARM_LADDER), "candidate_arm"),
            aggregate_contrast(patch, _concentration_specs(PATCH_LADDER), "patch"),
        ],
        ignore_index=True,
    )
    audited_path = (
        model_dir("swr_contrast", RC) / f"swr_contrast_content_audited{sfx}.pkl"
    )
    agg.to_pickle(audited_path)
    write_fit_provenance(audited_path, fit_inputs, estimator, sys.argv, extra=fit_extra)
    prov = _provenance(
        args,
        ev.groupby(SC).ngroups,
        fit_z_reference,
        stratum_audit,
        per_sd_scale,
        _renormalized_rungs(arm, patch),
        concentration_cols,
    )
    (
        model_dir("swr_contrast", RC) / f"swr_contrast_content_provenance{sfx}.json"
    ).write_text(json.dumps(prov, indent=2))
    logger.info(
        "saved audited table (%d rows) + provenance -> %s",
        len(agg),
        model_dir("swr_contrast", RC),
    )
    print(
        "\n########## AUDITED  (animal-level = primary; session signed-rank = descriptive) ##########"
    )
    cols = [
        "level",
        "model",
        "population",
        "term",
        "n_sessions",
        "n_animals",
        "coef_animal_median",
        "n_animals_pos",
        "coef_session_median",
        "typical_session_se",
        "pct_sig_session",
        "session_signrank_p",
        "session_signrank_fdr_q",
    ]
    with pd.option_context("display.max_rows", None, "display.width", 220):
        print(agg[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
