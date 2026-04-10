#!/usr/bin/env python
"""Fit the replay content model hierarchy (M0-M7) on pre-computed data.

Loads the master long-format dataset from prepare_replay_content_data.py,
fits per-session GLMs for each model, compares nested models, and saves
results.

Usage:
    uv run python scripts/fit_replay_content_models.py                # fit all
    uv run python scripts/fit_replay_content_models.py --models M0,M1,M2  # subset
"""

import argparse
import logging
import pickle
import time

import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from continuum_swr_replay.analysis.hippocampus.replay_content.fitting import (
    fit_glm,
    fit_mixed_effects,
)
from continuum_swr_replay.analysis.hippocampus.replay_content.model_checking import (
    aggregate_diagnostics,
    cross_validate_session,
    print_diagnostic_summary,
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
    """Fit one model across all sessions, returning per-session results.

    Parameters
    ----------
    long_df : pd.DataFrame
        Master long-format data.
    model_name : str
        Model identifier (e.g., "M0").
    predictors : list[str]
        Predictor columns for this model.
    run_cv : bool
        If True, also run 5-fold cross-validation per session.

    Returns
    -------
    pd.DataFrame
        One row per session with coefficients and diagnostics.
    """
    results = []

    for keys, group in long_df.groupby(SESSION_COLS):
        # Drop rows with NaN in required predictors
        if predictors:
            sub = group.dropna(subset=predictors + ["replay_here"])
        else:
            sub = group.dropna(subset=["replay_here"])

        if len(sub) < 18:  # minimum 2 events worth of data (9 arms * 2)
            continue

        r = fit_glm(sub, predictors=predictors)
        if r is None:
            continue

        # Cross-validation
        if run_cv:
            cv = cross_validate_session(sub, predictors=predictors)
            if cv is not None:
                r.update(cv)

        r["model"] = model_name
        for col, val in zip(SESSION_COLS, keys, strict=True):
            r[col] = val
        r["n_events"] = len(sub) // 9

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
    for metric in ["pseudo_r2", "brier", "aic"]:
        if metric in mdf.columns:
            vals = mdf[metric].dropna()
            if len(vals) > 0:
                print(f"  {metric}: median={vals.median():.4f}")

    if not predictors:
        print("  (null model — intercept only)")
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

    print("\nCoefficients (FDR-corrected p-values):")
    print(f"  {'Predictor':<25} {'Median':>8} {'!=0 p':>10}")
    print(f"  {'-' * 45}")
    for pred in predictors:
        if pred not in coef_info:
            continue
        vals = coef_info[pred]
        p = corrected_map.get(pred)
        p_str = f"{p:.2e}" if p is not None else "—"
        print(f"  {pred:<25} {vals.median():8.3f} {p_str:>10}")

    # Key diagnostics
    for metric, label in [
        ("cond_number", "Cond#"),
        ("max_abs_coef", "Max|coef|"),
    ]:
        if metric in mdf.columns:
            vals = mdf[metric].dropna()
            if len(vals) > 0:
                print(f"\n  {label}: median={vals.median():.1f}")


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
        help="Run 5-fold cross-validation per session (slower)",
    )
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="Also fit mixed-effects models (slowest)",
    )
    args = parser.parse_args()

    # Determine which models to fit
    if args.models:
        model_names = [m.strip() for m in args.models.split(",")]
    else:
        model_names = list(MODEL_HIERARCHY.keys())

    # Load data
    long_path = INPUT_DIR / "long.pkl"
    if not long_path.exists():
        logging.error(
            "Long-format data not found at %s. Run prepare_replay_content_data.py first.",
            long_path,
        )
        return

    logging.info("Loading %s...", long_path)
    long_df = pd.read_pickle(long_path)
    logging.info(
        "Loaded %d rows, %d sessions",
        len(long_df),
        long_df.groupby(SESSION_COLS).ngroups,
    )

    # Fit each model
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for model_name in model_names:
        if model_name not in MODEL_HIERARCHY:
            logging.warning("Unknown model: %s", model_name)
            continue

        spec = MODEL_HIERARCHY[model_name]
        predictors = spec.predictors

        # Check that all predictors exist in the data
        missing = [p for p in predictors if p not in long_df.columns]
        if missing:
            logging.warning(
                "Model %s: missing predictors %s — skipping", model_name, missing
            )
            continue

        cv_label = " + CV" if args.cv else ""
        logging.info(
            "Fitting %s (%d predictors%s)...", model_name, len(predictors), cv_label
        )
        t0 = time.time()

        mdf = fit_model_per_session(long_df, model_name, predictors, run_cv=args.cv)

        elapsed = time.time() - t0
        logging.info("  %s: %d session fits in %.1fs", model_name, len(mdf), elapsed)

        if len(mdf) > 0:
            # Save per-model results
            model_path = OUTPUT_DIR / f"{model_name}.pkl"
            mdf.to_pickle(model_path)
            logging.info("  Saved to %s", model_path)

            # Print summary
            print_model_results(model_name, mdf, predictors)

            # Print aggregated diagnostics
            diag = aggregate_diagnostics(mdf, predictors)
            print_diagnostic_summary(diag, model_name, predictors)

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

    # Mixed-effects fitting
    if args.mixed:
        mixed_dir = OUTPUT_DIR / "mixed_effects"
        mixed_dir.mkdir(parents=True, exist_ok=True)

        for model_name in model_names:
            if model_name not in MODEL_HIERARCHY:
                continue
            spec = MODEL_HIERARCHY[model_name]
            if not spec.predictors:
                logging.info("Skipping mixed-effects %s (no predictors)", model_name)
                continue
            predictors = spec.predictors
            missing = [p for p in predictors if p not in long_df.columns]
            if missing:
                continue

            logging.info(
                "Fitting mixed-effects %s (%d rows, %d sessions)...",
                model_name,
                len(long_df),
                long_df.groupby(SESSION_COLS).ngroups,
            )
            t0 = time.time()
            me_result = fit_mixed_effects(long_df, predictors)
            elapsed = time.time() - t0

            if me_result is not None:
                me_path = mixed_dir / f"{model_name}_mixed.pkl"
                with open(me_path, "wb") as f:
                    pickle.dump(me_result, f)
                logging.info(
                    "  %s mixed: %ds, LL=%.0f", model_name, elapsed, me_result["ll"]
                )

                print(f"\n  MIXED-EFFECTS {model_name}:")
                for name, mean, sd in zip(
                    me_result["fe_names"],
                    me_result["fe_mean"],
                    me_result["fe_sd"],
                    strict=True,
                ):
                    z = abs(mean / sd) if sd > 0 else 0
                    sig = (
                        "***"
                        if z > 3.29
                        else "**"
                        if z > 2.58
                        else "*"
                        if z > 1.96
                        else ""
                    )
                    print(f"    {name:<30}: {mean:8.4f} ± {sd:.4f} (z={z:.1f}) {sig}")
            else:
                logging.warning("  %s mixed: FAILED", model_name)

    logging.info("Done!")


if __name__ == "__main__":
    main()
