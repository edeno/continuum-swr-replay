#!/usr/bin/env python
"""Stage 6: replay content × reward outcome → behavioral change.

Reads the canonical tables under ``PROCESSED_DATA_DIR/replay_content/``
and runs:

1. :func:`stop_replay_summary` — per-stop replay-content aggregates.
2. :func:`fit_is_stay_logistic_per_session` — for each role in
   ``{current, prev, last_rewarded, best}``, fit
   ``is_stay ~ is_after_omission * frac_on_<role>`` per session.

Outputs (next to the input tables):

- ``stop_replay_summary.pkl`` — one row per stop.
- ``stage6_is_stay_logistic.pkl`` — one row per (session, role).

Usage::

    uv run python scripts/pfc/run_stage6_learning.py
    uv run python scripts/pfc/run_stage6_learning.py --content-stratum swr
    uv run python scripts/pfc/run_stage6_learning.py --content-stratum non_swr
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from continuum_swr_replay.analysis.replay.learning import (
    fit_is_stay_logistic_per_session,
    fit_is_stay_pooled_gee,
    stop_replay_summary,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)

INPUT_DIR = PROCESSED_DATA_DIR / "replay_content"
SUMMARY_PATH = INPUT_DIR / "stop_replay_summary.pkl"
LOGIT_PATH = INPUT_DIR / "stage6_is_stay_logistic.pkl"
GEE_PATH = INPUT_DIR / "stage6_is_stay_pooled_gee.pkl"

ARM_ROLES = ("current", "prev", "last_rewarded", "best")
PATCH_ROLES = ("current_patch", "prev_patch", "last_rewarded_patch", "best_patch")
ROLES = ARM_ROLES + PATCH_ROLES
SESSION_COLS = ["animal", "day", "epoch"]


def _content_col_for(role: str, stratum: str) -> str:
    if stratum == "all":
        return f"frac_on_{role}"
    return f"frac_on_{role}_{stratum}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--content-stratum",
        type=str,
        default="all",
        choices=["all", "swr", "non_swr", "mua"],
        help="Which event subset's replay content to use as the predictor.",
    )
    parser.add_argument(
        "--min-stops",
        type=int,
        default=30,
        help="Minimum stops per session for fitting.",
    )
    parser.add_argument(
        "--rebuild-summary",
        action="store_true",
        help="Rebuild stop_replay_summary even if it exists on disk.",
    )
    args = parser.parse_args()

    if not SUMMARY_PATH.exists() or args.rebuild_summary:
        logging.info("Building stop_replay_summary from canonical tables...")
        st = pd.read_pickle(INPUT_DIR / "stop_table.pkl")
        ev = pd.read_pickle(INPUT_DIR / "event_table.pkl")
        ea = pd.read_pickle(INPUT_DIR / "event_arm_table.pkl")
        logging.info(
            "  stops=%d  events=%d  event_arm rows=%d",
            len(st),
            len(ev),
            len(ea),
        )
        summary = stop_replay_summary(st, ev, ea)
        summary.to_pickle(SUMMARY_PATH)
        logging.info("  saved %d rows → %s", len(summary), SUMMARY_PATH)
    else:
        logging.info("Loading cached stop_replay_summary from %s", SUMMARY_PATH)
        summary = pd.read_pickle(SUMMARY_PATH)

    logging.info(
        "Stops with replay events: %d / %d",
        int(summary["n_events_with_replay"].fillna(0).gt(0).sum()),
        len(summary),
    )

    # Fit per-session logistic for each role + pooled GEE.
    all_results = []
    gee_rows = []
    for role in ROLES:
        col = _content_col_for(role, args.content_stratum)
        if col not in summary.columns:
            logging.warning("Skipping role=%s: column %s missing", role, col)
            continue
        usable = summary.dropna(subset=[col])
        logging.info(
            "Role=%-25s stratum=%-7s usable stops=%d",
            role,
            args.content_stratum,
            len(usable),
        )
        coefs = fit_is_stay_logistic_per_session(
            usable,
            content_col=col,
            outcome_col="is_after_omission",
            target_col="is_stay",
            min_stops=args.min_stops,
        )
        coefs["role"] = role
        coefs["content_stratum"] = args.content_stratum
        coefs["content_col"] = col
        all_results.append(coefs)

        # Pooled GEE on the same data.
        gee_result = fit_is_stay_pooled_gee(
            usable,
            content_col=col,
            outcome_col="is_after_omission",
            target_col="is_stay",
        )
        gee_result.update({"role": role, "content_stratum": args.content_stratum})
        gee_rows.append(gee_result)

    out = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    if not out.empty:
        out_path = (
            LOGIT_PATH
            if args.content_stratum == "all"
            else INPUT_DIR / f"stage6_is_stay_logistic_{args.content_stratum}.pkl"
        )
        out.to_pickle(out_path)
        logging.info("Saved %d rows → %s", len(out), out_path)

    gee_df = pd.DataFrame(gee_rows)
    if not gee_df.empty:
        gee_path = (
            GEE_PATH
            if args.content_stratum == "all"
            else INPUT_DIR / f"stage6_is_stay_pooled_gee_{args.content_stratum}.pkl"
        )
        gee_df.to_pickle(gee_path)
        logging.info("Saved pooled-GEE %d rows → %s", len(gee_df), gee_path)

    # ── Cross-session summary, FDR-corrected across roles.
    if out.empty:
        return
    converged = out[out["status"] == "ok"].copy()
    logging.info(
        "===== Cross-session summary (stratum=%s, converged sessions only) =====",
        args.content_stratum,
    )
    pvals = []
    role_results = []
    for role in ROLES:
        sub = converged[converged["role"] == role].dropna(subset=["beta_inter"])
        if len(sub) < 5:
            logging.info("  %s: too few sessions (%d)", role, len(sub))
            continue
        med = float(sub["beta_inter"].median())
        n_pos = int((sub["beta_inter"] > 0).sum())
        n_neg = int((sub["beta_inter"] < 0).sum())
        try:
            stat, p = wilcoxon(sub["beta_inter"].dropna())
        except ValueError:
            p = float("nan")
        role_results.append((role, len(sub), med, n_pos, n_neg, p))
        pvals.append(p)

    if pvals:
        # ``multipletests`` with NaN inputs returns all-NaN adjusted
        # p-values (verified on this branch's SciPy/statsmodels).
        # That contaminates every role's FDR when a single role
        # hits the wilcoxon ValueError path (e.g. all-zero β_inter
        # samples on current SciPy that don't always raise).
        # Correct only the finite p-values and pass NaN through for
        # the failed roles.
        pvals_arr = np.asarray(pvals, dtype=float)
        ok = np.isfinite(pvals_arr)
        p_fdr = np.full_like(pvals_arr, np.nan)
        if ok.any():
            _, p_fdr_ok, _, _ = multipletests(pvals_arr[ok], method="fdr_bh")
            p_fdr[ok] = p_fdr_ok
        for (role, n, med, n_pos, n_neg, p), p_corr in zip(
            role_results, p_fdr, strict=True
        ):
            logging.info(
                "  role=%-25s n=%d  med β_inter=%+.3f  pos=%d  neg=%d  p=%.2e  p_FDR=%.2e",
                role,
                n,
                med,
                n_pos,
                n_neg,
                p,
                p_corr,
            )

    # ── Pooled GEE summary (population-averaged estimates with cluster-
    # robust SE on session). One row per role; FDR across roles.
    if gee_df.empty:
        return
    converged_gee = gee_df[gee_df["status"] == "ok"].copy()
    if converged_gee.empty:
        logging.info("No pooled-GEE fits converged.")
        return
    logging.info(
        "===== Pooled GEE (Binomial, cluster-robust SE), stratum=%s =====",
        args.content_stratum,
    )
    p_inter = converged_gee["p_inter"].fillna(1.0).to_numpy()
    _, p_fdr_gee, _, _ = multipletests(p_inter, method="fdr_bh")
    for (_, row), p_corr in zip(converged_gee.iterrows(), p_fdr_gee, strict=True):
        logging.info(
            "  role=%-25s n_sess=%d  n_stops=%d  β_inter=%+.3f (SE=%.3f)  p=%.2e  p_FDR=%.2e",
            row["role"],
            row["n_sessions"],
            row["n_stops"],
            row["beta_inter"],
            row["se_inter"],
            row["p_inter"],
            p_corr,
        )


if __name__ == "__main__":
    main()
