#!/usr/bin/env python
"""Stage 6: replay content × reward outcome → behavioral change.

Reads the canonical tables under ``PROCESSED_DATA_DIR/replay_content/``
and runs:

1. :func:`stop_replay_summary` — per-stop replay-content aggregates.
2. :func:`fit_is_stay_logistic_per_session` — for each role in
   ``{current, prev, last_rewarded, best}``, fit
   ``is_stay ~ is_omitted_outcome * frac_on_<role>`` per session.

Both steps are read on one population. ``--stop-cohort at_well`` (the default)
keeps the well-anchored stops the hippocampal rate and content ladders are fit
on, so a Stage 6 statement and a content statement are about the same stops;
``all_stops`` applies no occupancy restriction and publishes its artifacts under
``_allstops`` names, so a sensitivity bundle cannot be mistaken for the
canonical one.

At this grain the choice makes no difference to any fitted number. ``is_stay``
is derived from the stop's ``occupancy_well_arm`` (``tables_stop.py``), which is
finite only for strictly ``at_well`` stops, so the response is missing for every
``mixed``, ``in_transit`` and ``unknown`` stop and the effective population is
the 67,606 strictly-at-well stops that carry an ``is_stay`` either way. The
cohort is applied regardless, so the population is stated rather than implied.

Outputs (next to the input tables):

- ``stop_replay_summary.pkl`` — one row per stop.
- ``stage6_is_stay_logistic.pkl`` — one row per (session, role).

Usage::

    uv run python scripts/behavior/run_stay_leave.py
    uv run python scripts/behavior/run_stay_leave.py --content-stratum swr
    uv run python scripts/behavior/run_stay_leave.py --content-stratum non_swr
    uv run python scripts/behavior/run_stay_leave.py --stop-cohort all_stops
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

from continuum_swr_replay.analysis.events.cohort_selection import (
    select_stop_cohort,
    stop_cohort_suffix,
)
from continuum_swr_replay.analysis.events.swr_source import require_swr_source
from continuum_swr_replay.analysis.replay.fit_provenance import (
    record_inputs,
    write_fit_provenance,
)
from continuum_swr_replay.analysis.replay.learning import (
    fit_is_stay_logistic_per_session,
    fit_is_stay_pooled_gee,
    stop_replay_summary,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR
from continuum_swr_replay.tables.covariate_registry import (
    COVARIATE_SCHEMA_VERSION,
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)

INPUT_DIR = PROCESSED_DATA_DIR / "replay_content"

#: The canonical tables the per-stop summary is built from, hashed into every
#: provenance sidecar this script writes.
FIT_INPUTS = {
    "stop_table": INPUT_DIR / "stop_table.pkl",
    "event_table": INPUT_DIR / "event_table.pkl",
    "event_arm_table": INPUT_DIR / "event_arm_table.pkl",
}

#: Populations Stage 6 may be read on. ``at_well`` is the canonical shared
#: cohort and publishes the unsuffixed artifact names; ``all_stops`` applies no
#: occupancy restriction and publishes ``_allstops`` names. ``in_transit`` is
#: not offered: every role this stage conditions on (current/prev/last-rewarded
#: arm) is defined from a well the animal has arrived at.
STOP_COHORTS = ("at_well", "all_stops")

ARM_ROLES = ("current", "prev", "last_rewarded", "best")
PATCH_ROLES = ("current_patch", "prev_patch", "last_rewarded_patch", "best_patch")
ROLES = ARM_ROLES + PATCH_ROLES


def summary_path(stop_cohort: str) -> Path:
    """Cached per-stop summary for one population."""
    return INPUT_DIR / f"stop_replay_summary{stop_cohort_suffix(stop_cohort)}.pkl"


def fit_path(kind: str, stratum: str, stop_cohort: str) -> Path:
    """Output path for one (fit kind, content stratum, population).

    The canonical at-well bundle keeps the unsuffixed names every reader already
    hard-codes; the cohort suffix comes last, so a stratified sensitivity bundle
    reads as ``stage6_is_stay_logistic_swr_allstops.pkl``.
    """
    stratum_part = "" if stratum == "all" else f"_{stratum}"
    suffix = stop_cohort_suffix(stop_cohort)
    return INPUT_DIR / f"stage6_is_stay_{kind}{stratum_part}{suffix}.pkl"


def write_summary_provenance(
    cache_path: Path,
    fit_inputs: dict[str, Any],
    stop_cohort: str,
    cohort_report: dict[str, Any],
) -> Path:
    """Write the per-stop summary's own provenance sidecar.

    The summary is an artifact in its own right -- it is cached, reused across
    strata, and read months later -- so it states the population it holds and
    the tables it was cut from without the reader having to open the pickle.
    """
    return write_fit_provenance(
        cache_path,
        fit_inputs,
        "per_stop_summary",
        sys.argv,
        extra={
            "stop_cohort": stop_cohort,
            "stop_cohort_report": cohort_report,
            "stop_table_sha256": fit_inputs.get("stop_table", {}).get("sha256"),
        },
    )


def _content_col_for(role: str, stratum: str) -> str:
    if stratum == "all":
        return f"frac_on_{role}"
    return f"frac_on_{role}_{stratum}"


def cross_session_role_summary(coefs: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon on each role's per-session interaction, FDR-corrected across roles.

    This is the statistic Stage 6 is read on, so it lives in one function: the
    run logs it, and an audit re-derives it from a saved bundle -- including a
    bundle fit on another population -- with the same code rather than a second
    implementation that can drift.

    Parameters
    ----------
    coefs : pandas.DataFrame
        Per-session fits as ``fit_is_stay_logistic_per_session`` returns them,
        concatenated over roles: ``role``, ``status`` and ``beta_inter``.

    Returns
    -------
    pandas.DataFrame
        One row per role in :data:`ROLES`, in that order:
        ``role``, ``n_sessions`` (converged sessions with a finite
        ``beta_inter``), ``median_beta_inter``, ``n_pos``, ``n_neg``, ``p``,
        ``p_fdr``, and ``status`` (``"ok"`` or ``"too_few_sessions"``). A role
        with fewer than five sessions is not testable and is excluded from the
        correction as well as from the test.
    """
    converged = coefs[coefs["status"] == "ok"]
    rows = []
    for role in ROLES:
        sub = converged[converged["role"] == role].dropna(subset=["beta_inter"])
        if len(sub) < 5:
            rows.append(
                {
                    "role": role,
                    "n_sessions": len(sub),
                    "median_beta_inter": np.nan,
                    "n_pos": int((sub["beta_inter"] > 0).sum()),
                    "n_neg": int((sub["beta_inter"] < 0).sum()),
                    "p": np.nan,
                    "status": "too_few_sessions",
                }
            )
            continue
        try:
            _, p = wilcoxon(sub["beta_inter"])
        except ValueError:
            # SciPy refuses an all-zero sample; that is a failed test, not a
            # p-value, and must not become one below.
            p = float("nan")
        rows.append(
            {
                "role": role,
                "n_sessions": len(sub),
                "median_beta_inter": float(sub["beta_inter"].median()),
                "n_pos": int((sub["beta_inter"] > 0).sum()),
                "n_neg": int((sub["beta_inter"] < 0).sum()),
                "p": p,
                "status": "ok",
            }
        )
    summary = pd.DataFrame(rows)
    summary["p_fdr"] = np.nan
    # ``multipletests`` returns an all-NaN vector if ANY input is NaN, so a
    # single role that failed its test would otherwise erase every other role's
    # FDR. Correct the finite p-values and pass NaN through for the rest.
    testable = (summary["status"] == "ok") & np.isfinite(summary["p"])
    if testable.any():
        _, p_fdr, _, _ = multipletests(
            summary.loc[testable, "p"].to_numpy(dtype=float), method="fdr_bh"
        )
        summary.loc[testable, "p_fdr"] = p_fdr
    return summary


def log_cross_session_role_summary(summary: pd.DataFrame) -> None:
    """Log one line per role of :func:`cross_session_role_summary`."""
    for row in summary.itertuples():
        if row.status != "ok":
            logging.info("  %s: too few sessions (%d)", row.role, row.n_sessions)
            continue
        logging.info(
            "  role=%-25s n=%d  med β_inter=%+.3f  pos=%d  neg=%d  p=%.2e  p_FDR=%.2e",
            row.role,
            row.n_sessions,
            row.median_beta_inter,
            row.n_pos,
            row.n_neg,
            row.p,
            row.p_fdr,
        )


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
    parser.add_argument(
        "--stop-cohort",
        type=str,
        default="at_well",
        choices=list(STOP_COHORTS),
        help=(
            "Population to read (default: at_well). 'at_well' keeps only the "
            "well-anchored stops the hippocampal rate and content ladders are "
            "fit on, and publishes the canonical artifact names; 'all_stops' "
            "applies no occupancy restriction and publishes '_allstops' names."
        ),
    )
    args = parser.parse_args()

    cache_path = summary_path(args.stop_cohort)
    if not cache_path.exists() or args.rebuild_summary:
        logging.info("Building stop_replay_summary from canonical tables...")
        st = read_table(FIT_INPUTS["stop_table"])
        ev = read_table(FIT_INPUTS["event_table"])
        ea = read_table(FIT_INPUTS["event_arm_table"])
        # Hash the tables while this run is reading them, not at write time.
        fit_inputs = record_inputs(FIT_INPUTS)
        require_covariate_schema(st, table="stop")
        require_covariate_schema(ev, table="event")
        require_covariate_schema(ea, table="event_arm")
        # the content strata split on is_swr
        require_swr_source(st, what=str(FIT_INPUTS["stop_table"]))
        require_swr_source(ev, what=str(FIT_INPUTS["event_table"]))
        require_swr_source(ea, what=str(FIT_INPUTS["event_arm_table"]))
        logging.info(
            "  stops=%d  events=%d  event_arm rows=%d",
            len(st),
            len(ev),
            len(ea),
        )
        # Restrict the population BEFORE summarising: a stop that is not in the
        # cohort must not reach a fit, and an event inside it must not reach a
        # stop that is.
        st, cohort_report = select_stop_cohort(st, cohort=args.stop_cohort)
        logging.info(
            "  cohort=%s (%s): kept %d/%d stops",
            cohort_report["cohort"],
            cohort_report["rule"],
            cohort_report["n_stops_after"],
            cohort_report["n_stops_before"],
        )
        summary = stop_replay_summary(st, ev, ea)
        summary.attrs["input_covariate_schema_version"] = COVARIATE_SCHEMA_VERSION
        summary.attrs["stop_cohort"] = args.stop_cohort
        summary.attrs["stop_cohort_report"] = cohort_report
        summary.attrs["inputs"] = fit_inputs
        summary.to_pickle(cache_path)
        write_summary_provenance(
            cache_path, fit_inputs, args.stop_cohort, cohort_report
        )
        logging.info("  saved %d rows → %s", len(summary), cache_path)
    else:
        logging.info("Loading cached stop_replay_summary from %s", cache_path)
        summary = pd.read_pickle(cache_path)
        if (
            summary.attrs.get("input_covariate_schema_version")
            != COVARIATE_SCHEMA_VERSION
        ):
            raise ValueError(
                "stop_replay_summary uses a legacy covariate schema; rerun with "
                "--rebuild-summary before fitting Stage 6"
            )
        # A summary cut before the cohort existed carries the canonical name and
        # every stop in the epoch. It states no population, so it cannot be
        # accepted as this one.
        cached_cohort = summary.attrs.get("stop_cohort")
        if cached_cohort != args.stop_cohort:
            raise ValueError(
                f"{cache_path.name} was built on stop cohort {cached_cohort!r}, "
                f"not {args.stop_cohort!r}; rerun with --rebuild-summary"
            )
        cohort_report = summary.attrs.get("stop_cohort_report", {})
        fit_inputs = summary.attrs.get("inputs", {})
        # The cohort names a population; the hashes name the bytes it was cut
        # from. A regenerated stop or event table changes which stops and events
        # the summary holds without changing its name, so the cache is only
        # usable while the tables it read are the tables on disk.
        current_inputs = record_inputs(FIT_INPUTS)
        moved = sorted(
            name
            for name, record in current_inputs.items()
            if fit_inputs.get(name, {}).get("sha256") != record["sha256"]
        )
        if moved:
            raise ValueError(
                f"{cache_path.name} was built from a different generation of "
                f"{', '.join(moved)} than the table(s) on disk now; rerun with "
                "--rebuild-summary before fitting Stage 6"
            )

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
            outcome_col="is_omitted_outcome",
            target_col="is_stay",
            min_stops=args.min_stops,
        )
        coefs["role"] = role
        coefs["content_stratum"] = args.content_stratum
        coefs["content_col"] = col
        coefs["stop_cohort"] = args.stop_cohort
        all_results.append(coefs)

        # Pooled GEE on the same data.
        gee_result = fit_is_stay_pooled_gee(
            usable,
            content_col=col,
            outcome_col="is_omitted_outcome",
            target_col="is_stay",
        )
        gee_result.update(
            {
                "role": role,
                "content_stratum": args.content_stratum,
                "stop_cohort": args.stop_cohort,
            }
        )
        gee_rows.append(gee_result)

    provenance_extra = {
        "stop_cohort": args.stop_cohort,
        "stop_cohort_report": cohort_report,
        "stop_table_sha256": fit_inputs.get("stop_table", {}).get("sha256"),
        "content_stratum": args.content_stratum,
        "min_stops": args.min_stops,
        "n_stops": int(len(summary)),
        "summary_artifact": cache_path.name,
    }

    out = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    if not out.empty:
        out_path = fit_path("logistic", args.content_stratum, args.stop_cohort)
        out.to_pickle(out_path)
        write_fit_provenance(
            out_path, fit_inputs, "logistic_glm", sys.argv, extra=provenance_extra
        )
        logging.info("Saved %d rows → %s", len(out), out_path)

    gee_df = pd.DataFrame(gee_rows)
    if not gee_df.empty:
        gee_path = fit_path("pooled_gee", args.content_stratum, args.stop_cohort)
        gee_df.to_pickle(gee_path)
        write_fit_provenance(
            gee_path, fit_inputs, "binomial_gee", sys.argv, extra=provenance_extra
        )
        logging.info("Saved pooled-GEE %d rows → %s", len(gee_df), gee_path)

    # ── Cross-session summary, FDR-corrected across roles.
    if out.empty:
        return
    logging.info(
        "===== Cross-session summary (stratum=%s, cohort=%s, converged "
        "sessions only) =====",
        args.content_stratum,
        args.stop_cohort,
    )
    log_cross_session_role_summary(cross_session_role_summary(out))

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
