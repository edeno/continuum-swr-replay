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

The ladders' fits, the qvals join, the cohort audit and the audited aggregation
are ``analysis/behavior/rate/swr_contrast_rate.py``; this script reads the
tables, selects the stop cohort, and writes.

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
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

from continuum_swr_replay.analysis.behavior import (  # noqa: E402
    add_uncertainty_covariates,
    attach_is_swr_to_onsets,
)
from continuum_swr_replay.analysis.behavior.cohort_audit import (  # noqa: E402
    write_cohort_audit,
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
)
from continuum_swr_replay.analysis.behavior.model_specs import (  # noqa: E402
    SWR_CONTRAST_RATE_G_SPECS as G_SPECS,
)
from continuum_swr_replay.analysis.behavior.model_specs import (  # noqa: E402
    SWR_CONTRAST_RATE_S_SPECS as S_SPECS,
)
from continuum_swr_replay.analysis.behavior.rate.duration_aware_history import (  # noqa: E402
    attach_event_ends,
)
from continuum_swr_replay.analysis.behavior.rate.swr_contrast_rate import (  # noqa: E402
    aggregate_contrast_fits,
    cohort_sd_reference,
    contrast_cohort_records,
    contrast_term_diagnostics,
    fit_contrast_sessions,
    join_contrast_qvals,
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
# is silenced. Convergence/separation are recorded per fit as flags by the fitter (see
# analysis.behavior._fit_diagnostics), not blanket-ignored.
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


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
    # Stops dropped by the inner qvals join are the mechanism-B "qvals gap" (no fitted
    # beta/value state) -- reported per-model as n_drop_qvals_gap in the cohort audit.
    stop, n_qvals_dropped = join_contrast_qvals(stop, q)
    require_swr_source(event_table, what=str(inputs["event_table"]))
    onsets = attach_event_ends(
        attach_is_swr_to_onsets(onset_table, event_table), event_table
    )
    return stop, onsets, n_qvals_dropped


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

    contrast_term_diagnostics(models).to_csv(
        out_dir / f"swr_contrast_rate_term_diagnostics{suffix}.csv", index=False
    )

    inputs = fit_input_paths()
    qv_full = read_table(inputs["qvals_table"])
    st_full = read_table(inputs["stop_table"])
    require_covariate_schema(qv_full, table="qvals")
    require_covariate_schema(st_full, table="stop")
    require_swr_source(st_full, what=str(inputs["stop_table"]))
    reference = cohort_sd_reference(qv_full, st_full, stop_cohorts[0])
    sd = reference["sd"]
    max_sd_covar = reference["max_sd_covariate"]
    max_sd_rel = reference["max_sd_relative_difference"]
    logger.info(
        "per-SD reference on the %s cohort (%d of %d stops); largest relative "
        "difference vs the full tables: %s %+.2f%%",
        stop_cohorts[0],
        reference["n_stops"],
        reference["n_stops_full"],
        max_sd_covar,
        100 * max_sd_rel,
    )
    agg = aggregate_contrast_fits(models, sd, stop_cohorts[0])
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

    t0 = time.perf_counter()
    models, skips, cohort_stop_drops = fit_contrast_sessions(
        stop, onsets, workers=args.workers
    )

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
        contrast_cohort_records(models, stop, audit_qvals_drop, cohort_stop_drops),
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
