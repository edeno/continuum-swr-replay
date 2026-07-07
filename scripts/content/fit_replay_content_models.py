#!/usr/bin/env python
"""Fit the replay content model hierarchy (M0-M7) on pre-computed data.

Primary model: Conditional Poisson regression (multinomial-count softmax).
Robustness check: Conditional Logit on dominant arm.

Loads the master long-format dataset from analysis/replay/prepare_data.py,
fits per-session models for each model in the hierarchy, compares nested
models via AIC (primary) and LRT, and saves results.

Usage:
    uv run python scripts/content/fit_replay_content_models.py                # fit all
    uv run python scripts/content/fit_replay_content_models.py --models M0,M1,M2
    uv run python scripts/content/fit_replay_content_models.py --cv           # + CV
    uv run python scripts/content/fit_replay_content_models.py --logit        # + logit robustness
"""

import argparse
import logging
import time

import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from continuum_swr_replay.analysis.replay._helpers import add_arm_fixed_effects
from continuum_swr_replay.analysis.replay.cohort_audit import (
    cohort_record,
    drop_profile,
    event_drop_buckets,
    write_cohort_audit,
)
from continuum_swr_replay.analysis.replay.covariate_timing import (
    assert_predictor_timing,
)
from continuum_swr_replay.analysis.replay.fitting import (
    fit_conditional_logit,
    fit_conditional_poisson,
)
from continuum_swr_replay.analysis.replay.model_checking import (
    cross_validate_conditional_poisson,
)
from continuum_swr_replay.analysis.replay.model_comparison import (
    run_full_hierarchy_comparison,
)
from continuum_swr_replay.analysis.replay.model_specs import (
    MODEL_HIERARCHY,
    MODEL_HIERARCHY_MUA,
    MODEL_HIERARCHY_SWR,
    MODEL_PATCH,
    MODEL_PATCH_MUA,
    MODEL_PATCH_SWR,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

# Map event source to model hierarchy and input directory
_EVENT_SOURCE_CONFIG = {
    "non_local": {
        "hierarchy": MODEL_HIERARCHY,
        "input_dir_name": "replay_content",
        "table_name": "event_arm_table",
    },
    "SWR": {
        "hierarchy": MODEL_HIERARCHY_SWR,
        "input_dir_name": "contfrag_replay_content",
        "table_name": "event_arm_table_swr",
    },
    "MUA": {
        "hierarchy": MODEL_HIERARCHY_MUA,
        "input_dir_name": "contfrag_replay_content",
        "table_name": "event_arm_table_mua",
    },
}

# When --hierarchy patch is selected, the table and response column are
# different but the input directory is shared with the corresponding
# event-source. event_id remains the grouping key (globally unique).
# ``table_name`` here is informational only — the actual table name is
# resolved by ``resolve_contfrag_paths`` further down, which appends the
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

SESSION_COLS = ["animal", "day", "epoch"]


def resolve_contfrag_paths(
    event_source: str,
    brain_area: str,
    hierarchy_kind: str,
) -> tuple[str, str]:
    """Return ``(table_name, output_subdir)`` for a fit invocation.

    Pure helper extracted so the prep ↔ fitter filename contract has a
    direct unit-test target. ``event_source`` is one of
    ``"non_local"`` / ``"SWR"`` / ``"MUA"``; ``hierarchy_kind`` is
    ``"arm"`` / ``"patch"``.

    Non-local fits stay unsuffixed (their prep is single-area /
    single-source). ContFrag fits append ``_{brain_area}`` to the
    table name and ``_{source_lower}_{brain_area}`` to the output
    subdir so SWR/HPC, SWR/mPFC, MUA/HPC … never overwrite each
    other's ``M0.pkl`` / ``all_models.pkl``.
    """
    if event_source == "non_local":
        if hierarchy_kind == "patch":
            return "event_patch_table", "models_patch"
        return "event_arm_table", "models"

    base_table = (
        f"event_patch_table_{event_source.lower()}"
        if hierarchy_kind == "patch"
        else f"event_arm_table_{event_source.lower()}"
    )
    table_name = f"{base_table}_{brain_area}"

    base_subdir = "models_patch" if hierarchy_kind == "patch" else "models"
    output_subdir = f"{base_subdir}_{event_source.lower()}_{brain_area}"
    return table_name, output_subdir


def fit_model_per_session(
    long_df: pd.DataFrame,
    model_name: str,
    predictors: list[str],
    run_cv: bool = False,
    response_col: str = "arm_count",
    event_col: str = "event_uid",
    n_categories: int = 9,
) -> pd.DataFrame:
    """Fit conditional Poisson model across all sessions.

    Parameters
    ----------
    long_df : pd.DataFrame
        Event-arm table (canonical ``event_arm_table.pkl`` or legacy
        ``long_count.pkl``).
    model_name : str
        Model identifier (e.g., "M0").
    predictors : list[str]
        Predictor columns for this model.
    run_cv : bool
        If True, also run event-based cross-validation per session.
    response_col : str
        Column name for the count response (``"arm_count_hard"``
        canonical, ``"arm_count"`` legacy).
    event_col : str
        Column name for the event grouping key (``"event_id"``
        canonical, ``"event_uid"`` legacy).

    Returns
    -------
    pd.DataFrame
        One row per session with coefficients and fit statistics.
    """
    results = []

    # Minimum 2 events per session, scaled by candidate-set size:
    # n_categories=9 (arm) → 18 rows, n_categories=3 (patch) → 6 rows.
    # The previous hardcoded 18 silently skipped valid 2–5 event patch
    # sessions before fit_conditional_poisson could apply its own
    # n_events < 2 * len(predictors) gate.
    min_rows = n_categories * 2
    for keys, group in long_df.groupby(SESSION_COLS):
        if len(group) < min_rows:
            continue

        r = fit_conditional_poisson(
            group,
            predictors=list(predictors),
            response_col=response_col,
            event_col=event_col,
            n_categories=n_categories,
        )
        if r is None:
            continue

        if run_cv:
            cv = cross_validate_conditional_poisson(
                group,
                predictors=list(predictors),
                event_col=event_col,
                response_col=response_col,
                n_categories=n_categories,
            )
            if cv is not None:
                r.update(cv)

        r["model"] = model_name
        for col, val in zip(SESSION_COLS, keys, strict=True):
            r[col] = val

        results.append(r)

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


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

    print(f"\nFit quality (n={len(mdf)} sessions):")
    for metric in ["aic", "bic", "llf"]:
        if metric in mdf.columns:
            vals = mdf[metric].dropna()
            if len(vals) > 0:
                print(f"  {metric}: median={vals.median():.2f}")

    if not predictors:
        print("  (null model — uniform allocation)")
        return

    # Coefficients with FDR correction
    raw_pvals = []
    coef_info = {}
    for pred in predictors:
        col = f"coef_{pred}"
        if col not in mdf.columns:
            continue
        vals = mdf[col].dropna()
        coef_info[pred] = vals
        if len(vals) >= 5:
            try:
                _, p = wilcoxon(vals)
                raw_pvals.append((pred, p))
            except ValueError:
                pass

    corrected_map = {}
    if raw_pvals:
        _, corrected, _, _ = multipletests([x[1] for x in raw_pvals], method="fdr_bh")
        corrected_map = {x[0]: c for x, c in zip(raw_pvals, corrected, strict=True)}

    print("\nCoefficients (median across sessions, FDR-corrected Wilcoxon p):")
    print(f"  {'Predictor':<25} {'Median':>8} {'SE':>8} {'!=0 p':>10}")
    print(f"  {'-' * 55}")
    for pred in predictors:
        if pred not in coef_info:
            continue
        vals = coef_info[pred]
        se_col = f"bse_{pred}"
        se = mdf[se_col].median() if se_col in mdf.columns else float("nan")
        p = corrected_map.get(pred)
        p_str = f"{p:.2e}" if p is not None else "—"
        print(f"  {pred:<25} {vals.median():8.3f} {se:8.3f} {p_str:>10}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names (default: all M0-M7)",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run event-based cross-validation per session (slower)",
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
        choices=["non_local", "SWR", "MUA"],
        help="Event source: non_local (model-detected), SWR, or MUA (default: non_local)",
    )
    parser.add_argument(
        "--hierarchy",
        type=str,
        default="arm",
        choices=["arm", "patch"],
        help=(
            "Resolution: 'arm' (9 arms, M0-M7) or 'patch' (3 patches, "
            "P0-P4). Patch fits use event_patch_table.pkl and "
            "patch_count_hard."
        ),
    )
    parser.add_argument(
        "--swr-subset",
        type=str,
        default="all",
        choices=["all", "T", "F"],
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
        choices=["HPC", "mPFC", "OFC"],
        help=(
            "Brain area suffix used when resolving ContFrag prep "
            "outputs (event_arm_table_{source}_{brain_area}.pkl). "
            "Ignored for the non_local event source, which is built "
            "by ``prepare_replay_content_data.py`` and is not "
            "partitioned per area."
        ),
    )
    args = parser.parse_args()

    # Select hierarchy and input paths based on event source. The
    # --hierarchy flag overrides the model dict and table name when
    # 'patch' is selected; the input directory follows the event source.
    source_config = _EVENT_SOURCE_CONFIG[args.event_source]
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
                "in analysis/replay/model_specs.py. Define it before running."
            )
    # The (table_name, output_subdir) contract with the prep scripts
    # lives in ``resolve_contfrag_paths`` so the path-resolution logic
    # has a direct unit-test target rather than only being exercised
    # at the end of the long ``main()``. The hierarchy / source
    # validity gates above are unchanged; this only resolves names.
    table_name, output_subdir = resolve_contfrag_paths(
        args.event_source, args.brain_area, args.hierarchy
    )

    input_dir = PROCESSED_DATA_DIR / source_config["input_dir_name"]
    output_dir = input_dir / output_subdir

    # SWR subset: restrict to is_swr=True or =False events. Within a
    # subset is_swr is constant, so the swr_x_* interactions (M6/M7,
    # P4) collapse onto their main effects and are unidentifiable —
    # use the *_SWR variants (M0-M5 / P0-P3) which already drop them.
    # Output goes to a suffixed subdirectory so the unrestricted fits
    # are not overwritten.
    if args.swr_subset != "all":
        if args.hierarchy == "patch":
            hierarchy = MODEL_PATCH_SWR
        else:
            hierarchy = MODEL_HIERARCHY_SWR
        output_dir = input_dir / f"{output_subdir}_swr_{args.swr_subset}"
        logging.info(
            "SWR subset = %s: hierarchy narrowed to %s, output -> %s",
            args.swr_subset, list(hierarchy.keys()), output_dir,
        )

    # Determine which models to fit
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
    else:
        # Value-block rungs (Mv/Pv) require p_reward_arm/q_arm, which are NaN for
        # stops absent from the contingency artifact. Including them in a default
        # run would shrink the shared complete-case event set for the *whole*
        # ladder (M0-M7/P0-P3 would then be fit on fewer events). Keep the base
        # ladder as the default; request the value block explicitly to compare it
        # against its own base (e.g. --models M5,Mv or --models P3,Pv).
        value_rungs = {"Mv", "Pv"}
        model_names = [m for m in hierarchy if m not in value_rungs]

    # Load the input table. For non_local arm fits, fall back to legacy naming.
    canonical_path = input_dir / f"{table_name}.pkl"
    legacy_path = input_dir / "long_count.pkl"
    if canonical_path.exists():
        count_path = canonical_path
        canonical = True
    elif (
        args.event_source == "non_local"
        and args.hierarchy == "arm"
        and legacy_path.exists()
    ):
        count_path = legacy_path
        canonical = False
        logging.warning(
            "using legacy long_count.pkl — run prepare_replay_content_data.py "
            "to regenerate canonical event_arm_table.pkl"
        )
    else:
        logging.error(
            "%s not found. Run the appropriate prepare script first.",
            canonical_path,
        )
        return

    logging.info("Loading %s...", count_path)
    long_df = pd.read_pickle(count_path)
    logging.info(
        "Loaded %d rows, %d sessions",
        len(long_df),
        long_df.groupby(SESSION_COLS).ngroups,
    )

    # Apply the SWR subset filter (--swr-subset T/F). is_swr is
    # event-level on the event_arm_table and patch table. Drop whole
    # events / patch-rows where is_swr doesn't match.
    if args.swr_subset != "all":
        if "is_swr" not in long_df.columns:
            logging.error(
                "--swr-subset requires an 'is_swr' column; not found in %s",
                count_path,
            )
            return
        keep = long_df["is_swr"] if args.swr_subset == "T" else ~long_df["is_swr"]
        n_before_rows = len(long_df)
        long_df = long_df.loc[keep].reset_index(drop=True)
        logging.info(
            "SWR subset filter (%s): %d -> %d rows",
            args.swr_subset, n_before_rows, len(long_df),
        )

    # Arm fixed effects are derived at fit time rather than stored in
    # the canonical table. Patch tables have no ``arm`` column and the
    # patch hierarchy does not use arm FE, so this is arm-only.
    if args.hierarchy == "arm":
        if "arm" in long_df.columns:
            add_arm_fixed_effects(long_df)
        else:
            logging.warning(
                "no 'arm' column — arm fixed-effect models (M0a+) will be "
                "skipped as missing-predictor"
            )

    # Collect the union of all predictors across requested models so we
    # can restrict to a common complete-case event set. This ensures that
    # M0 through M7 are all fit on the *same* events per session, which
    # is required for valid AIC deltas and LRTs.
    all_predictors: set[str] = set()
    valid_model_names = []
    for model_name in model_names:
        if model_name not in hierarchy:
            logging.warning("Unknown model: %s", model_name)
            continue
        spec = hierarchy[model_name]
        missing = [p for p in spec.predictors if p not in long_df.columns]
        if missing:
            logging.warning(
                "Model %s: missing predictors %s — skipping", model_name, missing
            )
            continue
        all_predictors.update(spec.predictors)
        valid_model_names.append(model_name)

    if not valid_model_names:
        logging.error("No valid models to fit")
        return

    # Resolve response/event column names. Patch hierarchy uses
    # patch_count_hard; arm hierarchy keeps the legacy fallback for
    # non_local + long_count.pkl.
    if args.hierarchy == "patch":
        response_col = "patch_count_hard"
        event_col = "event_id"
    else:
        response_col = "arm_count_hard" if canonical else "arm_count"
        event_col = "event_id" if canonical else "event_uid"

    # Pre-filter to common complete-case event set: drop entire events
    # (all 9 arm rows) if ANY predictor in the union is NaN for ANY row.
    # Keep the pre-filter frame for the cohort audit (per-model drop attribution).
    long_df_prefilter = long_df
    union_predictors = sorted(all_predictors)
    required_cols = [response_col, event_col] + union_predictors
    available_cols = [c for c in required_cols if c in long_df.columns]
    has_nan = long_df[available_cols].isna().any(axis=1)
    events_with_nan = long_df.loc[has_nan, event_col].unique()
    n_before = long_df[event_col].nunique()
    long_df = long_df[~long_df[event_col].isin(events_with_nan)].reset_index(drop=True)
    n_after = long_df[event_col].nunique()
    if n_before != n_after:
        logging.info(
            "Filtered to common complete-case events: %d → %d events "
            "(%d dropped due to NaN in %s)",
            n_before,
            n_after,
            n_before - n_after,
            sorted(all_predictors),
        )
    if n_after == 0:
        # A predictor that is structurally NaN on some arm of every
        # event (e.g. a stale ``recent_reward_rate`` column predating
        # the trunk-arm 0.0 fix) wipes out the whole event set under
        # event-wise complete-case filtering. Fail loudly rather than
        # fitting M0 on zero events.
        logging.error(
            "Complete-case filtering removed ALL %d events — a predictor "
            "in %s is NaN on some arm of every event. Regenerate the "
            "event-arm table with the current prepare script.",
            n_before,
            sorted(all_predictors),
        )
        return

    # Also drop zero-count events
    event_totals = long_df.groupby(event_col)[response_col].transform("sum")
    long_df = long_df[event_totals > 0].reset_index(drop=True)

    logging.info(
        "Fitting on %d rows, %d sessions, %d events",
        len(long_df),
        long_df.groupby(SESSION_COLS).ngroups,
        long_df[event_col].nunique(),
    )

    # Fit each model
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results = []
    cohort_audits: list[dict] = []
    population = getattr(args, "swr_subset", None) or "all"
    sessions_in = long_df_prefilter.groupby(SESSION_COLS).ngroups

    for model_name in valid_model_names:
        spec = hierarchy[model_name]
        predictors = list(spec.predictors)
        # Causal-hygiene gate (phase 4): the base content hierarchy carries no
        # post/future-outcome covariates, so no descriptive exception is expected here;
        # a future leak fails loud rather than becoming a silent predictive claim.
        assert_predictor_timing(predictors)

        cv_label = " + CV" if args.cv else ""
        logging.info(
            "Fitting %s (%d predictors%s)...", model_name, len(predictors), cv_label
        )
        t0 = time.time()

        # Patch hierarchy collapses 9 arms to 3 patches; the null /
        # pseudo-R² / CV baselines must use log(3) instead of log(9)
        # for P0–P3 to be on the right chance footing.
        n_categories = 3 if args.hierarchy == "patch" else 9

        mdf = fit_model_per_session(
            long_df,
            model_name,
            predictors,
            run_cv=args.cv,
            response_col=response_col,
            event_col=event_col,
            n_categories=n_categories,
        )

        elapsed = time.time() - t0
        logging.info("  %s: %d session fits in %.1fs", model_name, len(mdf), elapsed)

        # Cohort-audit record (grain=event, common complete-case cohort). Attribute
        # the union-filter drops to this rung's own vs the other rungs' predictors.
        buckets = event_drop_buckets(
            long_df_prefilter, event_col=event_col, response_col=response_col,
            own_predictors=predictors, union_predictors=union_predictors,
        )
        n_modeled = int(mdf["n_events"].sum()) if len(mdf) else 0
        n_drops = sum(
            buckets[k] for k in (
                "n_drop_qvals_gap", "n_drop_own_predictor_missing",
                "n_drop_common_cohort_other", "n_drop_zero_response",
            )
        )
        dropped = long_df_prefilter[
            long_df_prefilter[event_col].isin(buckets["dropped_event_ids"])
        ].drop_duplicates(event_col)
        cohort_audits.append(cohort_record(
            model_name, population, "event", cohort_mode="common",
            n_in=n_before, n_modeled=n_modeled,
            sessions_in=sessions_in, sessions_modeled=len(mdf),
            animals_modeled=int(mdf["animal"].nunique()) if len(mdf) else 0,
            drop_notes=drop_profile(dropped),
            n_drop_qvals_gap=buckets["n_drop_qvals_gap"],
            n_drop_own_predictor_missing=buckets["n_drop_own_predictor_missing"],
            n_drop_common_cohort_other=buckets["n_drop_common_cohort_other"],
            n_drop_zero_response=buckets["n_drop_zero_response"],
            n_excluded_session=n_before - n_drops - n_modeled,
            n_sessions_flagged_nonconverged=(
                int((mdf["converged"] == False).sum())  # noqa: E712
                if "converged" in mdf.columns else 0
            ),
            n_sessions_flagged_separated=(
                int((mdf["separated"] == True).sum())  # noqa: E712
                if "separated" in mdf.columns else 0
            ),
        ))

        if len(mdf) > 0:
            model_path = output_dir / f"{model_name}.pkl"
            mdf.to_pickle(model_path)
            logging.info("  Saved to %s", model_path)

            print_model_results(model_name, mdf, predictors, hierarchy)
            all_results.append(mdf)

    # Cohort-audit table: one row per model (grain=event, common cohort); every row
    # reconciles n_in - Σdrops == n_modeled (asserted by cohort_record).
    if cohort_audits:
        write_cohort_audit(cohort_audits, output_dir / "all_models_cohort_audit.csv")
        logging.info(
            "Saved cohort audit (%d rows) to %s",
            len(cohort_audits), output_dir / "all_models_cohort_audit.csv",
        )

    if not all_results:
        logging.error("No models produced results")
        return

    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)
    combined_path = output_dir / "all_models.pkl"
    combined.to_pickle(combined_path)
    logging.info("Saved combined results to %s", combined_path)

    # Run hierarchy comparison
    if len(all_results) >= 2:
        logging.info("Running nested model comparisons...")
        comparison = run_full_hierarchy_comparison(combined, hierarchy=hierarchy)
        if len(comparison) > 0:
            comp_path = output_dir / "model_comparison.pkl"
            comparison.to_pickle(comp_path)
            logging.info("Saved comparison to %s", comp_path)

            print(f"\n{'=' * 60}")
            print("MODEL COMPARISON (positive ΔAIC = child better)")
            print(f"{'=' * 60}")
            print(
                f"\n  {'Step':<12} {'ΔAIC med':>10} {'Child wins':>12} "
                f"{'Parent wins':>13} {'LRT sig':>10} {'Wilcoxon p':>12}"
            )
            print(f"  {'-' * 70}")
            for _, row in comparison.iterrows():
                step = f"{row['parent']}→{row['child']}"
                wp = f"{row['wilcoxon_p']:.2e}" if pd.notna(row["wilcoxon_p"]) else "—"
                print(
                    f"  {step:<12} {row['delta_aic_median']:>10.1f} "
                    f"{row['child_better_aic']:>8}/{row['n_sessions']:<3} "
                    f"{row['parent_better_aic']:>9}/{row['n_sessions']:<3} "
                    f"{row['lrt_significant']:>6}/{row['n_sessions']:<3} "
                    f"{wp:>12}"
                )

    # Conditional logit robustness check
    if args.logit:
        logging.info("Fitting conditional logit robustness checks...")
        logit_dir = output_dir / "logit"
        logit_dir.mkdir(parents=True, exist_ok=True)

        for model_name in valid_model_names:
            spec = hierarchy[model_name]
            if not spec.predictors:
                continue
            predictors = list(spec.predictors)

            logit_results = []
            for keys, group in long_df.groupby(SESSION_COLS):
                # Forward the resolved response column. Patch fits use
                # ``patch_count_hard``; without this, the logit call
                # silently fell back to ``arm_count_hard`` (the
                # ``fit_conditional_logit`` default), found no such
                # column on patch tables, and returned ``None`` for
                # every session — producing an empty ``logit/`` dir
                # with no robustness outputs.
                r = fit_conditional_logit(
                    group,
                    predictors=predictors,
                    event_col=event_col,
                    response_col=response_col,
                )
                if r is not None:
                    r["model"] = model_name
                    for col, val in zip(SESSION_COLS, keys, strict=True):
                        r[col] = val
                    logit_results.append(r)

            if logit_results:
                logit_df = pd.DataFrame(logit_results)
                logit_path = logit_dir / f"{model_name}_logit.pkl"
                logit_df.to_pickle(logit_path)
                logging.info("  %s logit: %d sessions saved", model_name, len(logit_df))

    logging.info("Done!")


if __name__ == "__main__":
    main()
