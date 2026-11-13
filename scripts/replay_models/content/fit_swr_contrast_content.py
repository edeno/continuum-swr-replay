#!/usr/bin/env python
"""Fit the beta-bernoulli replay-CONTENT ladder (arm + patch), is_swr version.

The content analog of ``scripts/replay_models/rate/fit_swr_contrast_rate.py``. Where the rate ladder
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
    uv run python scripts/replay_models/content/fit_swr_contrast_content.py               # full
    uv run python scripts/replay_models/content/fit_swr_contrast_content.py --max-per-animal 2   # smoke
    uv run python scripts/replay_models/content/fit_swr_contrast_content.py --estimator dirichlet_multinomial
    uv run python scripts/replay_models/content/fit_swr_contrast_content.py --outcome-stratum rewarded
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import warnings
from collections.abc import Sequence

import pandas as pd

from continuum_swr_replay.analysis.event_tables.content_response import (
    ESTIMATOR_CHOICES,
    RESPONSE_CHOICES,
    default_response_for_estimator,
    resolve_response_choice,
    resolve_response_column,
    validate_fraction_response,
)
from continuum_swr_replay.analysis.replay_models.cohort_audit import (
    term_diagnostics,
    write_cohort_audit,
)
from continuum_swr_replay.analysis.replay_models.content.dirichlet_multinomial import (
    DEFAULT_MIN_EVENTS_SESSION_PHI,
    DEFAULT_N_JOBS,
)
from continuum_swr_replay.analysis.replay_models.content.dirichlet_multinomial import (
    ESTIMATOR as DIRICHLET_MULTINOMIAL,
)
from continuum_swr_replay.analysis.replay_models.content.session_fits import (
    fit_contrast_ladder,
)
from continuum_swr_replay.analysis.replay_models.contrast_aggregation import (
    LOG_EVENT_LENGTH,
    LOG_PHI_EVENT_LENGTH,
    LOG_PHI_IS_SWR,
    OUTCOME_STRATA,
    assemble_arm,
    assemble_patch,
    audited_content_contrast,
    contrast_fit_inputs,
    gated_term_statistics,
    prepare_cohorts,
    renormalized_rungs,
    specs_from_ladder,
)
from continuum_swr_replay.analysis.replay_models.covariate_timing import (
    assert_predictor_timing,
)
from continuum_swr_replay.analysis.replay_models.fitting import poisson_estimator_name
from continuum_swr_replay.analysis.replay_models.model_specs import (
    DESCRIPTIVE_CONTENT_RUNGS,
)
from continuum_swr_replay.analysis.replay_models.model_specs import (
    SWR_CONTRAST_CONTENT_ARM_KEY_TERM as ARM_KEY_TERM,
)
from continuum_swr_replay.analysis.replay_models.model_specs import (
    SWR_CONTRAST_CONTENT_ARM_LADDER as ARM_LADDER,
)
from continuum_swr_replay.analysis.replay_models.model_specs import (
    SWR_CONTRAST_CONTENT_PATCH_KEY_TERM as PATCH_KEY_TERM,
)
from continuum_swr_replay.analysis.replay_models.model_specs import (
    SWR_CONTRAST_CONTENT_PATCH_LADDER as PATCH_LADDER,
)
from continuum_swr_replay.paths import model_dir
from continuum_swr_replay.runs.fit_provenance import (
    git_hash as loaded_commit,
)
from continuum_swr_replay.runs.fit_provenance import (
    record_inputs,
    write_fit_provenance,
)
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS
from continuum_swr_replay.tables.standardization import (
    save_reference,
    z_reference_path,
)

# Quasi-separated coefficients legitimately overflow exp() when rate ratios are formed
# outside the fit (RuntimeWarning). Convergence/perfect-separation warnings are recorded
# per fit as converged/separated/n_convergence_warnings flags inside the fitters (see
# analysis.replay_models._fit_diagnostics), so only the benign RuntimeWarning noise is silenced
# here -- NOT a blanket ignore that would hide fit-quality signals.
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

SC = list(SESSION_COLUMNS)


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


def _summ(res: pd.DataFrame, ladder: dict, key_term: dict) -> None:
    """One line per rung and population: the key term over the sessions the
    per-term gate keeps (``gated_term_statistics``); a term the gate
    refuses is printed with the gate's reason and no numbers."""
    print(
        f"\n{'model':18s}{'pop':8s}{'term':30s}{'n':>4}{'medCoef':>9}"
        f"{'%pos':>6}{'anim':>6}{'signed-rank p':>14}"
    )
    for name in ladder:
        term = key_term.get(name)
        if term is None:
            continue
        scope = ladder[name][1]
        for pop in ["all"] if scope == "pooled" else ["all", "SWR", "nonSWR"]:
            s = res[(res.model == name) & (res.population == pop)]
            if f"coef_{term}" not in s.columns:
                continue
            gate, stats = gated_term_statistics(s, term)
            if gate.refusal is not None:
                print(f"{name:18s}{pop:8s}{term:30s} (not reported: {gate.refusal})")
                continue
            p = stats["signed_rank_p"]
            star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else ""
            animals = f"{stats['n_animals_positive']}/{stats['n_animals']}"
            print(
                f"{name:18s}{pop:8s}{term:30s}{stats['n_sessions']:4d}"
                f"{stats['median']:+9.3f}{100 * stats['positive_fraction']:5.0f}%"
                f"{animals:>6}{p:14.2e} {star}"
            )


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


def main(argv: Sequence[str] | None = None) -> None:
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
    argv = sys.argv[1:] if argv is None else list(argv)
    args = ap.parse_args(argv)
    cli = [sys.argv[0], *argv]
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
        "response": response,
        "estimator": args.estimator,
        "n_jobs": args.n_jobs,
        "min_events_session_phi": args.min_events_session_phi,
        "concentration_length": args.concentration_length,
    }
    # The concentration design as requested. Within an SWR / non-SWR split the
    # flag is constant and ``session_fits.resolve_concentration_cols`` drops the
    # whole design;
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
    (model_dir("swr_contrast")).mkdir(parents=True, exist_ok=True)
    # The standardization references are taken over the FULL assembled cohorts,
    # before the outcome stratum and the smoke subset, so every stratum's
    # coefficients are stated in the same per-SD unit.
    contrast_inputs = contrast_fit_inputs()
    ev, arm_z_reference = assemble_arm(contrast_inputs)
    pt, patch_z_reference = assemble_patch(ev, contrast_inputs)
    # Hash the tables now, while they are the ones being fit: the ladder runs
    # for long enough that a table could be regenerated before the artifacts are
    # written, and a sidecar hashed at write time would certify bytes no fit saw.
    fit_inputs = record_inputs(contrast_inputs)
    if response == "fraction":
        validate_fraction_response(
            ev, arm_response, table_label=contrast_inputs["event_arm_table"]
        )
        validate_fraction_response(
            pt, patch_response, table_label=contrast_inputs["event_patch_table"]
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
    arm, arm_audit = fit_contrast_ladder(
        ev, ARM_LADDER, response_col=arm_response, n_categories=9, **fit_kwargs
    )
    arm_path = model_dir("swr_contrast") / f"swr_contrast_content_arm{sfx}.pkl"
    arm.to_pickle(arm_path)
    write_fit_provenance(arm_path, fit_inputs, estimator, cli, extra=fit_extra)
    save_reference(arm_z_reference, z_reference_path(arm_path))

    logger.info(
        "PATCH ladder: %d rows, %d events, %d sessions",
        len(pt),
        pt["event_id"].nunique(),
        pt.groupby(SC).ngroups,
    )
    patch, patch_audit = fit_contrast_ladder(
        pt, PATCH_LADDER, response_col=patch_response, n_categories=3, **fit_kwargs
    )
    patch_path = model_dir("swr_contrast") / f"swr_contrast_content_patch{sfx}.pkl"
    patch.to_pickle(patch_path)
    write_fit_provenance(patch_path, fit_inputs, estimator, cli, extra=fit_extra)
    save_reference(patch_z_reference, z_reference_path(patch_path))

    # Cohort-audit table (one row per model x population; reconciles n_in - Σdrops =
    # n_modeled at the event grain) + term-level quasi-separation diagnostics.
    write_cohort_audit(
        arm_audit + patch_audit,
        model_dir("swr_contrast") / f"swr_contrast_content_cohort_audit{sfx}.csv",
    )
    term_diag = pd.concat(
        [
            term_diagnostics(arm, specs_from_ladder(ARM_LADDER, ARM_KEY_TERM)),
            term_diagnostics(patch, specs_from_ladder(PATCH_LADDER, PATCH_KEY_TERM)),
        ],
        ignore_index=True,
    )
    term_diag.to_csv(
        model_dir("swr_contrast") / f"swr_contrast_content_term_diagnostics{sfx}.csv",
        index=False,
    )

    logger.info(
        "saved arm=%d patch=%d rows -> %s",
        len(arm),
        len(patch),
        model_dir("swr_contrast"),
    )
    print("\n########## ARM ##########")
    _summ(arm, ARM_LADDER, ARM_KEY_TERM)
    print("\n########## PATCH ##########")
    _summ(patch, PATCH_LADDER, PATCH_KEY_TERM)

    # Audited aggregation: SE, per-session significance, per-animal sign, signed-rank
    # p, BH-FDR across the rung family, and standardized (per-SD) effect sizes — plus a
    # provenance sidecar. The per-SD units are the pooled SDs frozen over the full
    # cohort by prepare_cohorts, so a stratified refit is on the pooled unit.
    agg = audited_content_contrast(
        arm,
        patch,
        arm_ladder=ARM_LADDER,
        arm_key_term=ARM_KEY_TERM,
        patch_ladder=PATCH_LADDER,
        patch_key_term=PATCH_KEY_TERM,
        per_sd_scale=per_sd_scale,
    )
    audited_path = model_dir("swr_contrast") / f"swr_contrast_content_audited{sfx}.pkl"
    agg.to_pickle(audited_path)
    write_fit_provenance(audited_path, fit_inputs, estimator, cli, extra=fit_extra)
    prov = _provenance(
        args,
        ev.groupby(SC).ngroups,
        fit_z_reference,
        stratum_audit,
        per_sd_scale,
        renormalized_rungs(arm, patch),
        concentration_cols,
    )
    (
        model_dir("swr_contrast") / f"swr_contrast_content_provenance{sfx}.json"
    ).write_text(json.dumps(prov, indent=2))
    logger.info(
        "saved audited table (%d rows) + provenance -> %s",
        len(agg),
        model_dir("swr_contrast"),
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
