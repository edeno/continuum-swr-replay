#!/usr/bin/env python
"""Fit the replay content model hierarchy (M0-M7) on pre-computed data.

Primary model: Conditional Poisson regression (multinomial-count softmax).
Robustness check: Conditional Logit on dominant arm.

Loads the master long-format dataset from prepare_replay_content_data.py,
fits per-session models for each model in the hierarchy, compares nested
models via AIC (primary) and LRT, and saves results.

Usage:
    uv run python scripts/fit_replay_content_models.py                # fit all
    uv run python scripts/fit_replay_content_models.py --models M0,M1,M2
    uv run python scripts/fit_replay_content_models.py --cv           # + CV
    uv run python scripts/fit_replay_content_models.py --logit        # + logit robustness
"""

import argparse
import logging
import time

import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from continuum_swr_replay.analysis.hippocampus.replay_content.fitting import (
    fit_conditional_logit,
    fit_conditional_poisson,
)
from continuum_swr_replay.analysis.hippocampus.replay_content.model_checking import (
    cross_validate_conditional_poisson,
)
from continuum_swr_replay.analysis.hippocampus.replay_content.model_comparison import (
    run_full_hierarchy_comparison,
)
from continuum_swr_replay.analysis.hippocampus.replay_content.model_specs import (
    MODEL_HIERARCHY,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)

INPUT_DIR = PROCESSED_DATA_DIR / "replay_content"
OUTPUT_DIR = PROCESSED_DATA_DIR / "replay_content" / "models"
SESSION_COLS = ["animal", "day", "epoch"]


def fit_model_per_session(
    long_df: pd.DataFrame,
    model_name: str,
    predictors: list[str],
    run_cv: bool = False,
) -> pd.DataFrame:
    """Fit conditional Poisson model across all sessions.

    Parameters
    ----------
    long_df : pd.DataFrame
        Master count-based long-format data with event_uid.
    model_name : str
        Model identifier (e.g., "M0").
    predictors : list[str]
        Predictor columns for this model.
    run_cv : bool
        If True, also run event-based cross-validation per session.

    Returns
    -------
    pd.DataFrame
        One row per session with coefficients and fit statistics.
    """
    results = []

    for keys, group in long_df.groupby(SESSION_COLS):
        # Do NOT drop rows here — fit_conditional_poisson handles
        # event-wise NaN filtering internally, preserving the 9-arm
        # candidate set for every retained event.
        if len(group) < 18:  # minimum 2 events (9 arms * 2)
            continue

        r = fit_conditional_poisson(
            group,
            predictors=list(predictors),
            response_col="arm_count",
            event_col="event_uid",
        )
        if r is None:
            continue

        # Cross-validation
        if run_cv:
            cv = cross_validate_conditional_poisson(
                group,
                predictors=list(predictors),
                event_col="event_uid",
                response_col="arm_count",
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
) -> None:
    """Print summary for one model."""
    print(f"\n{'=' * 60}")
    spec = MODEL_HIERARCHY[model_name]
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
    args = parser.parse_args()

    # Determine which models to fit
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
    else:
        model_names = list(MODEL_HIERARCHY.keys())

    # Load count-based long format
    count_path = INPUT_DIR / "long_count.pkl"
    if not count_path.exists():
        # Fall back to legacy path
        count_path = INPUT_DIR / "long.pkl"
    if not count_path.exists():
        logging.error(
            "Long-format data not found. Run prepare_replay_content_data.py first."
        )
        return

    logging.info("Loading %s...", count_path)
    long_df = pd.read_pickle(count_path)
    logging.info(
        "Loaded %d rows, %d sessions",
        len(long_df),
        long_df.groupby(SESSION_COLS).ngroups,
    )

    # Collect the union of all predictors across requested models so we
    # can restrict to a common complete-case event set. This ensures that
    # M0 through M7 are all fit on the *same* events per session, which
    # is required for valid AIC deltas and LRTs.
    all_predictors: set[str] = set()
    valid_model_names = []
    for model_name in model_names:
        if model_name not in MODEL_HIERARCHY:
            logging.warning("Unknown model: %s", model_name)
            continue
        spec = MODEL_HIERARCHY[model_name]
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

    # Pre-filter to common complete-case event set: drop entire events
    # (all 9 arm rows) if ANY predictor in the union is NaN for ANY row.
    required_cols = ["arm_count", "event_uid"] + sorted(all_predictors)
    available_cols = [c for c in required_cols if c in long_df.columns]
    has_nan = long_df[available_cols].isna().any(axis=1)
    events_with_nan = long_df.loc[has_nan, "event_uid"].unique()
    n_before = long_df["event_uid"].nunique()
    long_df = long_df[~long_df["event_uid"].isin(events_with_nan)].reset_index(
        drop=True
    )
    n_after = long_df["event_uid"].nunique()
    if n_before != n_after:
        logging.info(
            "Filtered to common complete-case events: %d → %d events "
            "(%d dropped due to NaN in %s)",
            n_before,
            n_after,
            n_before - n_after,
            sorted(all_predictors),
        )

    # Also drop zero-count events
    event_totals = long_df.groupby("event_uid")["arm_count"].transform("sum")
    long_df = long_df[event_totals > 0].reset_index(drop=True)

    logging.info(
        "Fitting on %d rows, %d sessions, %d events",
        len(long_df),
        long_df.groupby(SESSION_COLS).ngroups,
        long_df["event_uid"].nunique(),
    )

    # Fit each model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for model_name in valid_model_names:
        spec = MODEL_HIERARCHY[model_name]
        predictors = list(spec.predictors)

        cv_label = " + CV" if args.cv else ""
        logging.info(
            "Fitting %s (%d predictors%s)...", model_name, len(predictors), cv_label
        )
        t0 = time.time()

        mdf = fit_model_per_session(long_df, model_name, predictors, run_cv=args.cv)

        elapsed = time.time() - t0
        logging.info("  %s: %d session fits in %.1fs", model_name, len(mdf), elapsed)

        if len(mdf) > 0:
            model_path = OUTPUT_DIR / f"{model_name}.pkl"
            mdf.to_pickle(model_path)
            logging.info("  Saved to %s", model_path)

            print_model_results(model_name, mdf, predictors)
            all_results.append(mdf)

    if not all_results:
        logging.error("No models produced results")
        return

    # Combine all results
    combined = pd.concat(all_results, ignore_index=True)
    combined_path = OUTPUT_DIR / "all_models.pkl"
    combined.to_pickle(combined_path)
    logging.info("Saved combined results to %s", combined_path)

    # Run hierarchy comparison
    if len(all_results) >= 2:
        logging.info("Running nested model comparisons...")
        comparison = run_full_hierarchy_comparison(combined)
        if len(comparison) > 0:
            comp_path = OUTPUT_DIR / "model_comparison.pkl"
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
        logit_dir = OUTPUT_DIR / "logit"
        logit_dir.mkdir(parents=True, exist_ok=True)

        for model_name in model_names:
            if model_name not in MODEL_HIERARCHY:
                continue
            spec = MODEL_HIERARCHY[model_name]
            if not spec.predictors:
                continue
            predictors = list(spec.predictors)
            missing = [p for p in predictors if p not in long_df.columns]
            if missing:
                continue

            logit_results = []
            for keys, group in long_df.groupby(SESSION_COLS):
                r = fit_conditional_logit(
                    group, predictors=predictors, event_col="event_uid"
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
                logging.info(
                    "  %s logit: %d sessions saved", model_name, len(logit_df)
                )

    logging.info("Done!")


if __name__ == "__main__":
    main()
