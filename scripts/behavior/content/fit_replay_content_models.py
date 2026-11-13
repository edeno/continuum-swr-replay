#!/usr/bin/env python
"""Fit the replay content model hierarchy on pre-computed data.

The arm ladder is ``M0``, ``M0a``, ``M1``-``M6`` by default, with the opt-in
value/uncertainty rungs ``Mv``, ``Mu``, ``Mvu``; the patch ladder is ``P0``-``P3``
with the opt-in ``Pv`` (``--hierarchy patch``). The rungs are defined in
``analysis/behavior/model_specs.py``.

Primary model: Conditional Poisson regression (multinomial-count softmax).
Robustness check: Conditional Logit on dominant arm.

Loads the master long-format dataset from analysis/behavior/tables/prepare_data.py
(built by scripts/tables/prepare_replay_content_data.py),
fits per-session models for each model in the hierarchy, compares nested
models via AIC (primary) and LRT, and saves results.

Inference note — read the standard errors honestly. On the default
``--grain timebin`` fit, a single session's coefficient standard errors are
OPTIMISTIC: the ~32 autocorrelated per-event timebins are treated as
independent draws, so one session's SEs are too small by roughly 5x. A
per-session SE is therefore not a calibrated uncertainty. Population inference
comes from the across-session distribution of the per-session coefficients
(the sign-rank / FDR aggregation over sessions), never from a single fit's SE;
``--grain event`` collapses each event to one observation for an honest
per-session n, at the cost of the within-event distribution.

The shipped response is the per-event allocation FRACTION (``arm_fraction`` /
``patch_fraction``): each event contributes one observation of unit mass, so the
likelihood, AIC and standard errors are stated at the event grain rather than
weighting an event by its number of decoded timebins. ``--response hard_count``
fits the counts instead, for the retired-count comparison, and publishes to its
own ``_hard_count`` directory.

Any run that is not the full default ladder is fit on its own complete-case
cohort, so it publishes into a directory keyed by the rungs it requested
(``models_M5-Mv-Mu-Mvu/``) rather than into the canonical ``models/`` bundle.
``--leaf-only`` conditions on the six leaf arms rather than all nine candidates
-- the Beta-belief covariates are structural zeros on the trunks -- which is a
six-category likelihood and so gets its own ``_leaf`` bundle; the belief rungs'
parents must be requested in the same run to keep the AIC comparisons nested.
Every pickle in a published bundle carries a ``.provenance.json`` sidecar naming
the table it was fit on and that table's SHA-256 as of the read, which the
figure and summary scripts check before reading the fit. A bundle is published
by replacing its directory whole, so nothing else may write into one: the
is_swr-contrast fits live in ``models_swr_contrast/`` for that reason.

``--estimator dirichlet_multinomial`` replaces the conditional Poisson with a
softmax whose within-event concentration is estimated rather than assumed, so an
event's ~32 autocorrelated timebins stop counting as independent draws and
``coef_log_phi_is_swr`` says directly whether SWR events spread their mass over
more arms. It is defined on the integer counts, so it forces ``--response
hard_count``, and it publishes to its own ``_dm`` bundle: its likelihood keeps
the multinomial coefficient the conditional one drops, so its AIC is on a
different scale and must never be differenced against the conditional bundle's.

Usage:
    uv run python scripts/behavior/content/fit_replay_content_models.py                # base ladder
    uv run python scripts/behavior/content/fit_replay_content_models.py --response hard_count
    uv run python scripts/behavior/content/fit_replay_content_models.py --estimator dirichlet_multinomial
    uv run python scripts/behavior/content/fit_replay_content_models.py --models M0,M1,M2   # -> models_M0-M1-M2/
    uv run python scripts/behavior/content/fit_replay_content_models.py --models M5,Mv,Mu,Mvu
    uv run python scripts/behavior/content/fit_replay_content_models.py --leaf-only --models M0,M0a,M1,M2,M3,M4,Mv,Mu,Mvu
    uv run python scripts/behavior/content/fit_replay_content_models.py --cv           # + CV
    uv run python scripts/behavior/content/fit_replay_content_models.py --logit        # + logit robustness
"""

import argparse
import logging
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.behavior._fit_diagnostics import fit_flag_counts
from continuum_swr_replay.analysis.behavior.coefficient_statistics import (
    coefficient_family_statistics,
)
from continuum_swr_replay.analysis.behavior.cohort_audit import (
    write_cohort_audit,
)
from continuum_swr_replay.analysis.behavior.content.cohort import (
    content_cohort_record,
    prepare_canonical_fit_request,
    select_content_cohort,
)
from continuum_swr_replay.analysis.behavior.content.dirichlet_multinomial import (
    DEFAULT_MIN_EVENTS_SESSION_PHI,
    DEFAULT_N_JOBS,
)
from continuum_swr_replay.analysis.behavior.content.fit_outputs import (
    BRAIN_AREA_CHOICES,
    EVENT_SOURCE_CHOICES,
    GRAIN_CHOICES,
    HIERARCHY_CHOICES,
    SWR_SUBSET_CHOICES,
    canonical_output_subdir,
    canonical_output_subdirs,
    frame_estimator,
    partial_ladder_subdir,
    resolve_contfrag_paths,
    scale_label,
)
from continuum_swr_replay.analysis.behavior.content.model_comparison import (
    fit_metric_statistics,
    run_full_hierarchy_comparison,
)
from continuum_swr_replay.analysis.behavior.content.session_fits import (
    LOGIT_SKIP_COLUMNS,
    fit_conditional_logit_robustness,
    fit_content_ladder,
)
from continuum_swr_replay.analysis.behavior.content_response import (
    ESTIMATOR_CHOICES,
    RESPONSE_CHOICES,
    resolve_response_choice,
)
from continuum_swr_replay.analysis.behavior.model_specs import (
    LEAF_ONLY_N_CATEGORIES,
    MODEL_HIERARCHY,
    MODEL_HIERARCHY_MUA,
    MODEL_HIERARCHY_SWR,
    MODEL_PATCH,
    MODEL_PATCH_MUA,
    MODEL_PATCH_SWR,
    ModelSpec,
    default_content_model_names,
    rung_predictors,
)
from continuum_swr_replay.analysis.behavior.tables.table_contracts import (
    validate_requested_standardization,
)
from continuum_swr_replay.analysis.events.analysed_time import require_swr_source
from continuum_swr_replay.paths import contfrag_replay_content_dir, replay_content_dir
from continuum_swr_replay.runs.fit_provenance import (
    record_inputs,
    write_fit_provenance,
)
from continuum_swr_replay.tables.covariate_registry import (
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table, table_exists
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

# Map event source to model hierarchy and input directory (the accessor that
# resolves it, called when the run starts)
_EVENT_SOURCE_CONFIG = {
    "non_local": {
        "hierarchy": MODEL_HIERARCHY,
        "input_dir": replay_content_dir,
        "table_name": "event_arm_table",
    },
    "SWR": {
        "hierarchy": MODEL_HIERARCHY_SWR,
        "input_dir": contfrag_replay_content_dir,
        "table_name": "event_arm_table_swr",
    },
    "MUA": {
        "hierarchy": MODEL_HIERARCHY_MUA,
        "input_dir": contfrag_replay_content_dir,
        "table_name": "event_arm_table_mua",
    },
}

# When --hierarchy patch is selected, the table and response column are
# different but the input directory is shared with the corresponding
# event-source. event_id remains the grouping key (globally unique).
# ``table_name`` here is informational only — the actual table name is
# resolved by ``fit_outputs.resolve_contfrag_paths``, which appends the
# brain-area suffix for the ContFrag sources. The SWR/MUA patch tables
# are built by ``prepare_contfrag_replay_content_data.py``.
_PATCH_OVERRIDES = {
    "non_local": {
        "hierarchy": MODEL_PATCH,
        "table_name": "event_patch_table",
    },
    "SWR": {"hierarchy": MODEL_PATCH_SWR, "table_name": "event_patch_table_swr"},
    "MUA": {"hierarchy": MODEL_PATCH_MUA, "table_name": "event_patch_table_mua"},
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)


def content_fit_inputs(input_dir: Path, table_name: str) -> dict[str, Path]:
    """Name -> path of every table one content-ladder fit is built from.

    The ladder reads a single canonical candidate table, resolved from the event
    source and hierarchy. Recorded in the provenance sidecar of every artifact
    the run publishes, and re-derived by the readers of those artifacts, so a
    regenerated table cannot be read through a fit that predates it.

    Parameters
    ----------
    input_dir : Path
        The run's candidate-table root.
    table_name : str
        Table stem from :func:`resolve_contfrag_paths`.

    Returns
    -------
    dict of str to Path
        ``{table_name: <input_dir>/<table_name>.parquet}``.
    """
    return {table_name: Path(input_dir) / f"{table_name}.parquet"}


def _publish_staged_directory(staged_dir: Path, output_dir: Path) -> None:
    """Swap a fully written staged bundle into place, restoring on failure.

    Directories cannot be swapped in a single rename, so there is a window
    between moving the old bundle aside and moving the new one in. Exceptions
    unwind it; a kill signal cannot. If the process dies inside that window the
    output directory is absent and the previous bundle is intact next to it, in
    a directory named ``.<output-name>.previous-*`` -- rename it back.
    """
    backup_root: Path | None = None
    backup_dir: Path | None = None
    if output_dir.exists():
        backup_root = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.previous-", dir=output_dir.parent
            )
        )
        backup_dir = backup_root / "output"
        output_dir.replace(backup_dir)

    try:
        staged_dir.replace(output_dir)
    except BaseException:
        if backup_dir is not None and backup_dir.exists():
            backup_dir.replace(output_dir)
        if backup_root is not None and backup_root.exists():
            shutil.rmtree(backup_root)
        raise
    else:
        if backup_root is not None:
            try:
                shutil.rmtree(backup_root)
            except OSError:
                logging.warning(
                    "Published %s but could not remove previous-output backup %s",
                    output_dir,
                    backup_root,
                    exc_info=True,
                )


def _write_complete_output_bundle(
    *,
    output_dir: Path,
    model_results: Mapping[str, pd.DataFrame],
    combined: pd.DataFrame,
    cohort_audits: list[dict],
    logit_results: Mapping[str, pd.DataFrame],
    fit_inputs: Mapping[str, object],
    logit_skips: pd.DataFrame | None = None,
    estimator: str,
    cli: Sequence[str],
    response_renormalized: bool = False,
) -> None:
    """Stage every artifact, then replace the output directory as one bundle.

    Every pickle gets a provenance sidecar naming the table the fit read, its
    SHA-256 as of the read, the estimator that produced it, and whether the
    response was rescaled to the fitted candidate set, so a reader can refuse a
    fit whose input has moved since and can tell a leaf-only fraction from the
    table's own nine-arm one. The sidecars are written into the staging
    directory, so they are swapped in atomically with the pickles they
    describe -- a bundle can never be published without them.
    """
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staged_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    cli = list(cli)

    def _write(frame: pd.DataFrame, path: Path) -> None:
        frame.to_pickle(path)
        write_fit_provenance(
            path,
            dict(fit_inputs),
            estimator,
            cli,
            extra={"response_renormalized": bool(response_renormalized)},
        )

    try:
        for model_name, result in model_results.items():
            _write(result, staged_dir / f"{model_name}.pkl")
        _write(combined, staged_dir / "all_models.pkl")
        write_cohort_audit(cohort_audits, staged_dir / "all_models_cohort_audit.csv")
        if logit_results:
            logit_dir = staged_dir / "logit"
            logit_dir.mkdir()
            for model_name, result in logit_results.items():
                _write(result, logit_dir / f"{model_name}_logit.pkl")
            # Written even when empty, so "nothing was skipped" is on disk.
            skips = (
                logit_skips
                if logit_skips is not None
                else pd.DataFrame(columns=LOGIT_SKIP_COLUMNS)
            )
            skips.to_csv(logit_dir / "logit_skipped_sessions.csv", index=False)
        _publish_staged_directory(staged_dir, output_dir)
    finally:
        if staged_dir.exists():
            shutil.rmtree(staged_dir)


def print_model_results(
    model_name: str,
    mdf: pd.DataFrame,
    predictors: list[str],
    hierarchy: dict | None = None,
) -> None:
    """Print summary for one model."""
    if hierarchy is None:
        hierarchy = MODEL_HIERARCHY
    print(f"\n{'=' * 60}")
    spec = hierarchy[model_name]
    print(f"{model_name}: {spec.description}")
    print(f"{'=' * 60}")

    estimator = frame_estimator(mdf)
    print(f"\nFit quality (n={len(mdf)} sessions; {scale_label(estimator)}):")
    for metric in ["aic", "bic", "llf"]:
        if metric in mdf.columns:
            stats = fit_metric_statistics(mdf, metric)
            if stats["n_sessions"] > 0:
                print(f"  {metric}: median={stats['median']:.2f}")
    if "pseudo_r2" in mdf.columns:
        stats = fit_metric_statistics(mdf, "pseudo_r2")
        if stats["n_sessions"] > 0:
            # McFadden's ratio is not comparable across estimators: the DM's
            # llf and llf_null both carry the multinomial constant that a
            # conditional likelihood omits, and its null refits the
            # concentration, so the ratio sits on a different scale.
            note = (
                f" ({scale_label(estimator)}; llf and llf_null both include the "
                "multinomial constant, so NOT on a conditional fit's scale)"
                if estimator == "dirichlet_multinomial"
                else ""
            )
            print(f"  pseudo_r2: median={stats['median']:.3f}{note}")

    if not predictors:
        print("  (null model — uniform allocation)")
        return

    # PRIMARY evidence is at the animal level, matching the SWR-contrast ladder
    # (see contrast_aggregation's module docstring). Sessions within an animal are
    # not independent and the corpus is unbalanced -- two animals supply over half
    # the sessions and one supplies 6% -- so a session-level median is largely a
    # statement about those two, and a session-level p treats correlated
    # observations as independent. The per-animal median is balanced across
    # animals, and with n=5 the sign consistency IS the claim; the signed-rank p
    # (FDR-corrected over the rung's predictors) stays for cross-session
    # consistency but is descriptive, not the population test.
    print(
        f"\nCoefficients (animal-level primary, n={mdf['animal'].nunique()} animals; "
        "session median/p descriptive):"
    )
    # ``anim>0``: animals whose own median is positive, of the animals.
    # ``SE``: the median session SE over the sessions with a coefficient, the
    # event-clustered one where the fit wrote a finite one and the model SE
    # otherwise (``contrast_aggregation.typical_session_se``).
    # ``unident``: sessions that cannot inform the predictor at all (NaN
    # coefficient, named in ``unidentified_terms``); they are in no median here.
    header = (
        f"  {'Predictor':<25} {'AnimalMed':>10} {'anim>0':>7} {'SessMed':>9} "
        f"{'SE':>8} {'sess p':>10} {'unident':>7}"
    )
    print(header)
    print(f"  {'-' * (len(header) - 2)}")
    for pred, row in coefficient_family_statistics(mdf, predictors).iterrows():
        p = row["signed_rank_p_adjusted"]
        p_str = f"{p:.2e}" if np.isfinite(p) else "—"
        positive = f"{int(row['n_animals_positive'])}/{int(row['n_animals'])}"
        print(
            f"  {pred:<25} {row['animal_median']:10.3f} {positive:>7} "
            f"{row['median']:9.3f} {row['bse_median']:8.3f} {p_str:>10} "
            f"{int(row['n_unidentified']):>7d}"
        )


def format_comparison_table(comparison: pd.DataFrame) -> list[str]:
    """Render the nested-comparison headline block, one string per line.

    What the headline may show depends on the scale the rungs were fit at. On a
    count response every event contributes ~32 autocorrelated timebins as if
    they were independent draws, so the per-session counts of AIC wins and of
    significant likelihood-ratio tests are anticonservative by roughly that
    factor: they are columns of the comparison but are not printed as a
    headline result. On the fraction (and on the dominant-choice
    logit) an event is one observation, so the same counts are printed, marked
    as the event-scale quasi-likelihood quantities they are.

    Parameters
    ----------
    comparison : pd.DataFrame
        Output of ``run_full_hierarchy_comparison``; one row per step.

    Returns
    -------
    list of str
        Lines to print. Empty when there is nothing to compare.
    """
    if comparison.empty:
        return []
    estimator = frame_estimator(comparison)
    label = scale_label(estimator)
    event_scale = label.startswith("event-scale")

    lines = [
        "=" * 60,
        f"MODEL COMPARISON (ΔAIC = child − parent, negative = child better; {label})",
        "=" * 60,
        "",
    ]
    header = f"  {'Step':<12} {'ΔAIC animal':>12} {'animals':>8} {'ΔAIC sess':>11}"
    if event_scale:
        header += f" {'Child wins':>12} {'LRT sig':>10}"
    header += f" {'sess p':>11}"
    lines.append(header)
    lines.append(f"  {'-' * (len(header) - 2)}")

    for _, row in comparison.iterrows():
        step = f"{row['parent']}→{row['child']}"
        wp = f"{row['wilcoxon_p']:.2e}" if pd.notna(row["wilcoxon_p"]) else "—"
        am = row.get("delta_aic_animal_median", float("nan"))
        am_str = f"{am:.1f}" if pd.notna(am) else "—"
        agree = (
            f"{int(row.get('n_animals_child_better', 0))}/"
            f"{int(row.get('n_animals', 0))}"
            if row.get("n_animals", 0)
            else "—"
        )
        line = f"  {step:<12} {am_str:>12} {agree:>8} {row['delta_aic_median']:>11.1f}"
        if event_scale:
            line += (
                f" {row['child_better_aic']:>8}/{row['n_sessions']:<3}"
                f" {row['lrt_significant']:>6}/{row['n_sessions']:<3}"
            )
        line += f" {wp:>11}"
        lines.append(line)

    if event_scale:
        # Read off the label rather than hardcoded: the conditional fits' counts
        # are quasi-likelihood ones, the Dirichlet-multinomial's are not.
        lines.append(f"  Child wins / LRT sig are {label} counts.")
    else:
        lines.append(
            "  Per-session child_better_aic / lrt_significant are timebin-scale "
            "(events' bins counted as independent draws) and are not reported "
            "here."
        )
    return lines


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "Comma-separated model names (default: the arm ladder "
            "M0,M0a,M1-M6 or the patch ladder P0-P3; the value/uncertainty "
            "rungs Mv,Mu,Mvu,Pv are opt-in and must be named here)"
        ),
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run event-based cross-validation per session (slower)",
    )
    parser.add_argument(
        "--grain",
        type=str,
        default="timebin",
        choices=list(GRAIN_CHOICES),
        help=(
            "Observation grain. 'timebin' (default) fits the conditional "
            "Poisson over per-arm timebin counts. 'event' collapses each event "
            "to a single observation -- which arm won -- and fits the "
            "conditional logit, so n is the number of events rather than the "
            "number of timebins. The timebin grain treats ~32 autocorrelated "
            "bins per event as independent draws and its SEs are optimistic by "
            "roughly 5x; the event grain is honest about n but cannot see the "
            "within-event distribution. Written to a '_event_grain' output "
            "directory so the two never overwrite each other."
        ),
    )
    parser.add_argument(
        "--logit",
        action="store_true",
        help="Also fit conditional logit on dominant arm (robustness)",
    )
    parser.add_argument(
        "--event-source",
        type=str,
        default="non_local",
        choices=list(EVENT_SOURCE_CHOICES),
        help="Event source: non_local (model-detected), SWR, or MUA (default: non_local)",
    )
    parser.add_argument(
        "--hierarchy",
        type=str,
        default="arm",
        choices=list(HIERARCHY_CHOICES),
        help=(
            "Resolution: 'arm' (9 candidate arms, M0,M0a,M1-M6 plus opt-in "
            "Mv/Mu/Mvu) or 'patch' (3 patches, P0-P3 plus opt-in Pv). Patch "
            "fits use event_patch_table.parquet and patch_count_hard."
        ),
    )
    parser.add_argument(
        "--estimator",
        type=str,
        default="conditional_poisson",
        choices=list(ESTIMATOR_CHOICES),
        help=(
            "Per-session estimator. 'conditional_poisson' (default, shipped) "
            "fits the softmax allocation at the grain --response selects. "
            "'dirichlet_multinomial' fits the same softmax mean structure with "
            "an explicit within-event concentration, so an event's "
            "autocorrelated timebins are not counted as independent draws and "
            "coef_log_phi_is_swr measures directly whether SWR events spread "
            "their mass more widely. It is defined on the integer counts, so it "
            "forces --response hard_count, and it publishes to a '_dm' "
            "directory: its likelihood carries the multinomial coefficient the "
            "conditional one drops, so its AIC is not on the conditional "
            "bundle's scale."
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=DEFAULT_N_JOBS,
        help=(
            "Worker processes for the Dirichlet-multinomial per-session fits "
            f"(default: {DEFAULT_N_JOBS}). Ignored by the conditional Poisson."
        ),
    )
    parser.add_argument(
        "--min-events-session-phi",
        type=int,
        default=DEFAULT_MIN_EVENTS_SESSION_PHI,
        help=(
            "Modelled events a session needs to estimate its own "
            "Dirichlet-multinomial concentration intercept (default: "
            f"{DEFAULT_MIN_EVENTS_SESSION_PHI}). Shorter sessions are pinned to "
            "the median intercept of that animal's longer sessions, and record "
            "which pooled value they used in phi_source. This changes the "
            "fitted COHORT, not just a nuisance parameter: a session just under "
            "the threshold is refit with its intercept pinned, and if that "
            "refit fails it is dropped (reason 'pinned_fit_none') even though "
            "its free fit had converged."
        ),
    )
    parser.add_argument(
        "--response",
        type=str,
        default=None,
        choices=list(RESPONSE_CHOICES),
        help=(
            "Per-event response. Defaults to the estimator's own: 'fraction' "
            "for the conditional Poisson, 'hard_count' for the "
            "Dirichlet-multinomial (which has no fraction form, so asking for "
            "one is an error). 'fraction' (the shipped response) fits "
            "arm_fraction / patch_fraction, so every event contributes one "
            "observation of unit mass and the likelihood, AIC and SEs are at "
            "the event grain. 'hard_count' fits arm_count_hard / "
            "patch_count_hard, which weights an event by its number of decoded "
            "timebins and treats those autocorrelated bins as independent "
            "draws; it is kept for the retired-count comparison and writes to "
            "a '_hard_count' output directory so the two never overwrite each "
            "other."
        ),
    )
    parser.add_argument(
        "--leaf-only",
        action="store_true",
        help=(
            "Condition on the six leaf arms instead of all nine candidates. "
            "The Beta-belief covariates (Mv/Mu/Mvu) are structural zeros on the "
            "three trunks, which inflates their within-event collinearity when "
            "the trunks are carried; fitting them leaf-only requires their "
            "parent rungs to be leaf-only too, so request the whole chain in "
            "one run (--leaf-only --models M0,M0a,M1,M2,M3,M4,Mv,Mu,Mvu). The "
            "arm dummies for arms 5-7 are dropped (arm 5 is the reference) and "
            "the null baseline becomes log(6). Arm hierarchy only; publishes to "
            "a '_leaf' output directory."
        ),
    )
    parser.add_argument(
        "--swr-subset",
        type=str,
        default="all",
        choices=list(SWR_SUBSET_CHOICES),
        help=(
            "Restrict to the SWR-overlap subset of events. 'T' keeps "
            "events with is_swr=True; 'F' keeps is_swr=False. With a "
            "subset the hierarchy is reduced to MODEL_HIERARCHY_SWR / "
            "MODEL_PATCH_SWR (i.e. M0-M5 / P0-P3) since swr_x_* "
            "interactions are constant within a subset. Output is "
            "written to a suffixed subdirectory (e.g. models_swr_T)."
        ),
    )
    parser.add_argument(
        "--brain-area",
        type=str,
        default="HPC",
        choices=list(BRAIN_AREA_CHOICES),
        help=(
            "Brain area suffix used when resolving ContFrag prep "
            "outputs (event_arm_table_{source}_{brain_area}.pkl). "
            "Ignored for the non_local event source, which is built "
            "by ``prepare_replay_content_data.py`` and is not "
            "partitioned per area."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Explicit candidate-table root. Defaults to the canonical root for "
            "--event-source. Use this for staging validation."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Exact model-output directory. Defaults to the source/hierarchy "
            "subdirectory under --input-dir. Existing outputs are replaced only "
            "after the complete requested bundle validates."
        ),
    )
    return parser


class _RunConfig(NamedTuple):
    """Resolved paths, model hierarchy, and requested rung names for one run."""

    hierarchy: dict[str, ModelSpec]
    input_dir: Path
    output_dir: Path
    table_name: str
    model_names: Sequence[str]
    # ``--response`` after the estimator default is applied, so no downstream
    # step has to re-derive it (or forget to).
    response: str = "fraction"


class _ContentCohort(NamedTuple):
    """The loaded, subset, and complete-case-frozen fitting cohort."""

    long_df: pd.DataFrame
    long_df_prefilter: pd.DataFrame
    valid_model_names: Sequence[str]
    union_predictors: Sequence[str]
    response_col: str
    event_col: str
    n_events_before_complete_case: int
    fit_inputs: dict[str, object]
    response_renormalized: bool


def _resolve_run_config(args: argparse.Namespace) -> _RunConfig:
    """Resolve the hierarchy, input/output paths, and requested model names."""
    # Select hierarchy and input paths based on event source. The
    # --hierarchy flag overrides the model dict and table name when
    # 'patch' is selected; the input directory follows the event source.
    source_config = _EVENT_SOURCE_CONFIG[args.event_source]
    estimator = args.estimator
    # One resolution point for the response: ``--response`` defaults to the
    # estimator's own, and a combination the estimator cannot fit is refused
    # here rather than silently overridden.
    response = resolve_response_choice(estimator, args.response)
    if estimator == "dirichlet_multinomial":
        if args.grain != "timebin":
            raise ValueError(
                "--grain event collapses each event to the arm that won, which "
                "discards the within-event distribution the "
                "Dirichlet-multinomial concentration is estimated from. Drop "
                "--grain event or use --estimator conditional_poisson."
            )
        if args.cv:
            raise ValueError(
                "--cv runs the conditional-Poisson cross-validation, whose "
                "held-out score is not on the Dirichlet-multinomial likelihood "
                "scale. Drop --cv or use --estimator conditional_poisson."
            )
    if args.leaf_only and args.hierarchy == "patch":
        raise ValueError(
            "--leaf-only restricts the nine candidate arms to the six leaves; "
            "the patch hierarchy's three candidates are patches, not arms, and "
            "have no leaf subset. Drop --leaf-only or use --hierarchy arm."
        )
    if args.hierarchy == "patch":
        override = _PATCH_OVERRIDES[args.event_source]
        hierarchy = override["hierarchy"]
        table_name = override["table_name"]
        if hierarchy is None:
            raise NotImplementedError(
                f"Patch hierarchy not defined for event source {args.event_source!r}. "
                "Add an entry to _PATCH_OVERRIDES pointing at a MODEL_PATCH "
                "variant in model_specs.py."
            )
    else:
        hierarchy = source_config["hierarchy"]
        table_name = source_config["table_name"]
        if hierarchy is None:
            raise NotImplementedError(
                f"Event source {args.event_source!r} has no MODEL_HIERARCHY defined "
                "in analysis/behavior/model_specs.py. Define it before running."
            )
    # The (table_name, output_subdir) contract with the prep scripts
    # lives in ``resolve_contfrag_paths`` so the path-resolution logic
    # has a direct unit-test target rather than only being exercised
    # at the end of the long ``main()``. The hierarchy / source
    # validity gates above are unchanged; this only resolves names.
    table_name, _output_subdir = resolve_contfrag_paths(
        args.event_source, args.brain_area, args.hierarchy
    )

    input_dir = (
        args.input_dir.resolve()
        if args.input_dir is not None
        else source_config["input_dir"]()
    )
    # SWR subset: restrict to is_swr=True or =False events. Within a
    # subset is_swr is constant, so the swr_x_* interactions (M6/M7,
    # P4) collapse onto their main effects and are unidentifiable —
    # use the *_SWR variants (M0-M5 / P0-P3) which already drop them.
    # The subset and the event grain each get their own bundle
    # directory (see ``canonical_output_subdir``) so neither publishes
    # over the unrestricted timebin fits.
    if args.swr_subset != "all":
        hierarchy = (
            MODEL_PATCH_SWR if args.hierarchy == "patch" else MODEL_HIERARCHY_SWR
        )
        logging.info(
            "SWR subset = %s: hierarchy narrowed to %s",
            args.swr_subset,
            list(hierarchy.keys()),
        )
    output_dir = input_dir / canonical_output_subdir(
        args.event_source,
        args.brain_area,
        args.hierarchy,
        args.swr_subset,
        args.grain,
        response,
        args.leaf_only,
        estimator,
    )
    if args.leaf_only:
        logging.info(
            "candidate set = %d leaf arms -> %s", LEAF_ONLY_N_CATEGORIES, output_dir
        )
    if response != "fraction":
        logging.info("response = %s -> %s", response, output_dir)
    if estimator != "conditional_poisson":
        logging.info("estimator = %s -> %s", estimator, output_dir)
    if args.grain == "event":
        logging.info("grain = event (one observation per event) -> %s", output_dir)

    # Determine which models to fit
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
    else:
        # Belief-dependent rungs require contingency/qvals columns, which are NaN
        # for unmapped stops. Including Mv/Mu/Mvu/Pv in a default run would shrink
        # the shared complete-case set for the whole base ladder. Request them
        # explicitly on their own matched cohort (e.g. --models M5,Mv,Mu,Mvu or
        # --models P3,Pv).
        model_names = list(default_content_model_names(hierarchy))

    # A run that fits anything other than the full default ladder is fit on its
    # own complete-case cohort, so its rungs are not comparable with the
    # canonical bundle's and must not be written next to them (or over them).
    # It gets its own directory, keyed by the rungs it fits.
    canonical_dir = output_dir
    is_partial_ladder = set(model_names) != set(default_content_model_names(hierarchy))
    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
        # Any canonical bundle, not just this invocation's: an --output-dir can
        # name the bundle of another hierarchy, SWR subset, grain, event source
        # or brain area just as easily as its own.
        if is_partial_ladder and output_dir.name in canonical_output_subdirs():
            raise ValueError(
                f"--models {','.join(model_names)} is not the full "
                f"{canonical_dir.name}/ ladder "
                f"({','.join(default_content_model_names(hierarchy))}); writing it "
                f"to {output_dir}, the canonical {output_dir.name}/ bundle, would "
                "replace part of a cohort-matched bundle with rungs fit on a "
                "different event set. Drop --output-dir to get a directory keyed "
                "by the requested rungs, or name a directory of your own."
            )
        logging.info("explicit model output -> %s", output_dir)
    elif is_partial_ladder:
        output_dir = canonical_dir.with_name(
            partial_ladder_subdir(canonical_dir.name, model_names)
        )
        logging.info("partial ladder (%s) -> %s", ", ".join(model_names), output_dir)

    return _RunConfig(
        hierarchy=hierarchy,
        input_dir=input_dir,
        output_dir=output_dir,
        table_name=table_name,
        model_names=model_names,
        response=response,
    )


def _load_and_filter_cohort(
    args: argparse.Namespace, config: _RunConfig
) -> _ContentCohort:
    """Load and check the canonical table, then select the fitting cohort.

    The selection is :func:`~continuum_swr_replay.analysis.behavior.content.cohort.select_content_cohort`.
    """
    hierarchy = config.hierarchy
    input_dir = config.input_dir
    table_name = config.table_name
    model_names = config.model_names

    # Canonical inputs are mandatory. The former implicit ``long_count.pkl``
    # fallback could fit M0-M3 from a stale schema and write those partial
    # results into the canonical ``models/`` directory.
    fit_input_paths = content_fit_inputs(input_dir, table_name)
    canonical_path = fit_input_paths[table_name]
    legacy_path = input_dir / "long_count.pkl"
    if not table_exists(canonical_path):
        legacy_note = (
            f" Found legacy artifact {legacy_path}, but it cannot be used "
            "for canonical fits."
            if legacy_path.exists()
            else ""
        )
        raise FileNotFoundError(
            f"Canonical content table {canonical_path} was not found.{legacy_note} "
            "Legacy fallback is disabled; regenerate the canonical table with "
            "the appropriate prepare script."
        )

    logging.info("Loading %s...", canonical_path)
    long_df = read_table(canonical_path)
    # Hash the table now, while the fit is reading it: the ladder runs long
    # enough that the table could be regenerated before the bundle is written,
    # and a sidecar hashed at write time would certify bytes the fit never saw.
    fit_inputs = record_inputs(fit_input_paths)
    require_covariate_schema(
        long_df, table="event_arm" if args.hierarchy == "arm" else "event_patch"
    )
    # Its events and is_swr must come from the configured SWR source, artifact
    # run included.
    require_swr_source(long_df, what=str(canonical_path))
    (
        long_df,
        valid_model_names,
        union_predictors,
        response_col,
        event_col,
    ) = prepare_canonical_fit_request(
        long_df,
        hierarchy=hierarchy,
        model_names=model_names,
        hierarchy_kind=args.hierarchy,
        table_label=canonical_path,
        response=config.response,
    )
    logging.info("response = %s (%s)", response_col, config.response)
    validate_requested_standardization(
        long_df,
        union_predictors=union_predictors,
        input_dir=input_dir,
        table_label=canonical_path,
    )
    logging.info(
        "Loaded %d rows, %d sessions",
        len(long_df),
        long_df.groupby(list(SESSION_COLUMNS)).ngroups,
    )

    selection = select_content_cohort(
        long_df,
        hierarchy_kind=args.hierarchy,
        union_predictors=union_predictors,
        response_col=response_col,
        response=config.response,
        table_label=canonical_path,
        event_col=event_col,
        swr_subset=args.swr_subset,
        leaf_only=args.leaf_only,
    )
    return _ContentCohort(
        long_df=selection.long_df,
        long_df_prefilter=selection.long_df_prefilter,
        valid_model_names=valid_model_names,
        union_predictors=union_predictors,
        response_col=response_col,
        event_col=event_col,
        n_events_before_complete_case=selection.n_events_before_complete_case,
        fit_inputs=fit_inputs,
        response_renormalized=selection.response_renormalized,
    )


def _fit_content_ladder(
    args: argparse.Namespace, config: _RunConfig, cohort: _ContentCohort
) -> tuple[dict[str, pd.DataFrame], dict[str, list[dict]]]:
    """The run's options and cohort, passed to ``session_fits.fit_content_ladder``."""
    return fit_content_ladder(
        cohort.long_df,
        config.hierarchy,
        cohort.valid_model_names,
        hierarchy_kind=args.hierarchy,
        leaf_only=args.leaf_only,
        response_col=cohort.response_col,
        event_col=cohort.event_col,
        response_renormalized=cohort.response_renormalized,
        run_cv=args.cv,
        grain=args.grain,
        estimator=args.estimator,
        n_jobs=args.n_jobs,
        min_events_session_phi=args.min_events_session_phi,
    )


def _audit_compare_and_publish(
    args: argparse.Namespace,
    config: _RunConfig,
    cohort: _ContentCohort,
    model_results: dict[str, pd.DataFrame],
    estimator_skips: dict[str, list[dict]] | None = None,
    *,
    cli: list[str],
) -> None:
    """Build the cohort audit, run nested comparisons and optional conditional-
    logit robustness fits, then stage and publish the complete output bundle.
    ``cli`` is the command line recorded in the provenance sidecars."""
    hierarchy = config.hierarchy
    output_dir = config.output_dir
    valid_model_names = cohort.valid_model_names
    estimator_skips = estimator_skips or {}

    # Build every audit record only after the model frames have been restricted
    # to the exact same successfully fitted session cohort.
    cohort_audits: list[dict] = []
    population = getattr(args, "swr_subset", None) or "all"
    for model_name, mdf in model_results.items():
        predictors = rung_predictors(hierarchy[model_name], args.leaf_only)
        cohort_audits.append(
            content_cohort_record(
                model_name,
                mdf,
                long_df_prefilter=cohort.long_df_prefilter,
                predictors=predictors,
                union_predictors=cohort.union_predictors,
                response_col=cohort.response_col,
                event_col=cohort.event_col,
                n_events_in=cohort.n_events_before_complete_case,
                population=population,
                flag_counts=fit_flag_counts(mdf),
                estimator_skips=estimator_skips.get(model_name, []),
            )
        )

        print_model_results(model_name, mdf, predictors, hierarchy)

    combined = pd.concat(model_results.values(), ignore_index=True)
    combined_models = tuple(combined["model"].drop_duplicates())
    if combined_models != tuple(valid_model_names):
        raise RuntimeError(
            f"Combined fit contains models {combined_models}, expected "
            f"{tuple(valid_model_names)}; no output will be published"
        )

    # The nested comparison is printed, not written: the figure and the summary
    # recompute it from the fits.
    if len(model_results) >= 2:
        logging.info("Running nested model comparisons...")
        comparison = run_full_hierarchy_comparison(combined, hierarchy=hierarchy)
        headline = format_comparison_table(comparison)
        if headline:
            print("\n" + "\n".join(headline))

    # Optional conditional-logit robustness fits on the dominant arm, on the
    # primary ladder's shared sessions, collected before the bundle is staged.
    logit_outputs: dict[str, pd.DataFrame] = {}
    logit_skips = pd.DataFrame(columns=LOGIT_SKIP_COLUMNS)
    if args.logit:
        logit_outputs, logit_skips = fit_conditional_logit_robustness(
            cohort.long_df,
            next(iter(model_results.values())),
            hierarchy,
            valid_model_names,
            leaf_only=args.leaf_only,
            response_col=cohort.response_col,
            event_col=cohort.event_col,
            response_renormalized=cohort.response_renormalized,
        )

    estimator = frame_estimator(combined)
    if estimator is None:
        raise RuntimeError(
            "Combined fit does not carry a single estimator label; the bundle's "
            "provenance could not say how its numbers were produced, so no "
            "output will be published"
        )

    # Only now, after all requested fits and derived artifacts are valid, write
    # to a sibling staging directory and swap the complete directory into place.
    _write_complete_output_bundle(
        output_dir=output_dir,
        model_results=model_results,
        combined=combined,
        cohort_audits=cohort_audits,
        logit_results=logit_outputs,
        logit_skips=logit_skips,
        fit_inputs=cohort.fit_inputs,
        estimator=str(estimator),
        cli=cli,
        response_renormalized=cohort.response_renormalized,
    )
    logging.info(
        "Published %d model files and complete audit bundle to %s",
        len(model_results),
        output_dir,
    )

    logging.info("Done!")


def run_content_fit(args: argparse.Namespace, cli: list[str]) -> None:
    """Load the content tables, fit the ladder, compare, and publish the bundle.

    ``cli`` is the command line recorded in the provenance sidecars.
    """
    config = _resolve_run_config(args)
    cohort = _load_and_filter_cohort(args, config)
    model_results, estimator_skips = _fit_content_ladder(args, config, cohort)
    _audit_compare_and_publish(
        args, config, cohort, model_results, estimator_skips, cli=cli
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_arg_parser()
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(argv)
    run_content_fit(args, [sys.argv[0], *argv])


if __name__ == "__main__":
    main()
