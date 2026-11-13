#!/usr/bin/env python
"""Fit the per-bin replay rate vs behavioral-uncertainty / value, by population.

Tests whether non-local replay *rate* is differentially modulated by the
animal's behavioral uncertainty and the task's nominal value structure across
SWR-coupled vs non-SWR non-local events, controlling for the within-stop time
course, raised-cosine self-history, the reward kernel, and the recent reward
rate. Rate-only -- replay direction is out of scope.

Per session: load ``stop_table``; validate its canonical
``patch_switch_fraction_last5_completed_trips``
and derive stop-local summaries (:func:`add_uncertainty_covariates`); inner-merge
the nominal value + real
behavioral-model covariates from ``qvals_table`` on ``stop_id`` (one exactly
stop-start-aligned source -- no difflib CSV); drop the epoch register's
first-exposure sessions (``epoch_register.first_exposure_epochs``); attach the
canonical ``is_swr`` to the onsets.
Because the fitted block contains current-trip RPE/value updates, retain only
physical response bins whose left edge is at or after ``outcome_known_time``
(reward onset for rewarded trials; latest possible pump latency, capped at
the collapsed task visit's final departure, for omissions).
Per population in ``{swr, non_local_swr, non_local_no_swr}`` build the per-bin
design (carrying the uncertainty + value covariates as ``stop_covariates``) on
the at-risk bins shared with ``fit_per_bin_rate_models.py`` (a bin inside an
ongoing event of the population's detector is dropped) and fit a single spec -- ``MU_uncertainty``, defined in
``model_specs.MODEL_UNCERTAINTY_RATE`` (not in ``MODEL_PER_BIN_RATE`` /
``all_models.pkl``). The spec is M7r (time spline +
raised-cosine self-history from event ends + reward kernel) plus the
uncertainty/value block, so the block is tested *beyond* those confounds.

Collect one tidy row per (session, population) with every covariate's
``coef_`` / ``bse_`` / ``pval_`` / ``rr_`` ->
``Processed-Data/replay_content/models_per_bin_rate/uncertainty_rate.pkl``.

The fit loop, the qvals join, the first-exposure drop, the RL z-scores and the
cohort audit are ``analysis/replay_models/rate/uncertainty_rate.py``; this script
reads the tables, selects the stop cohort, and writes.

Usage::

    uv run python scripts/replay_models/rate/fit_uncertainty_rate.py                    # full
    uv run python scripts/replay_models/rate/fit_uncertainty_rate.py --max-per-animal 1 # smoke
"""

from __future__ import annotations

from collections.abc import Sequence

from continuum_swr_replay.runs.threads import pin_blas_threads

# Pin BLAS to one thread BEFORE numpy import (matches the Family B orchestrator).
pin_blas_threads()

import argparse  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

from continuum_swr_replay.analysis.behavior.covariates import (  # noqa: E402
    CANONICAL_SWITCH_WINDOW,
    add_uncertainty_covariates,
    assert_qvals_aligned,
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
from continuum_swr_replay.analysis.replay_models.coefficient_statistics import (  # noqa: E402
    coefficient_statistics,
)
from continuum_swr_replay.analysis.replay_models.cohort_audit import (  # noqa: E402
    write_cohort_audit,
)
from continuum_swr_replay.analysis.replay_models.model_specs import (  # noqa: E402
    RATE_BIN_S,
)
from continuum_swr_replay.analysis.replay_models.per_bin_design import (  # noqa: E402
    attach_is_swr_to_onsets,
)
from continuum_swr_replay.analysis.replay_models.rate.duration_aware_history import (  # noqa: E402
    attach_event_ends,
)
from continuum_swr_replay.analysis.replay_models.rate.uncertainty_rate import (  # noqa: E402
    FITTED_COVARIATES,
    MU_UNCERTAINTY,
    POPULATIONS_TO_FIT,
    drop_exposure_sessions,
    fit_uncertainty_rate,
    join_uncertainty_qvals,
    standardize_rl_covariates,
    uncertainty_cohort_records,
)
from continuum_swr_replay.paths import (  # noqa: E402
    model_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.runs.fit_provenance import (  # noqa: E402
    record_inputs,
    write_fit_provenance,
)
from continuum_swr_replay.tables.covariate_registry import (  # noqa: E402
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table  # noqa: E402
from continuum_swr_replay.tables.session_keys import (  # noqa: E402
    SESSION_COLUMNS,
    subset_max_per_animal,
)
from continuum_swr_replay.tables.standardization import (  # noqa: E402
    Reference,
    save_reference,
    z_reference_path,
)

# separated history coefs legitimately overflow exp(); convergence/separation are
# recorded per fit as flags in per_bin_rate_model (analysis.replay_models._fit_diagnostics),
# so only the benign RuntimeWarning is silenced (not a blanket ignore).
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def fit_input_paths() -> dict[str, Path]:
    """The tables `main` reads, recorded in the artifact's provenance sidecar;
    under ``paths.replay_content_dir()``, read when called."""
    rc = replay_content_dir()
    return {
        "stop_table": table_path("stop_table", rc),
        "event_onset_table": table_path("event_onset_table", rc),
        "event_table": table_path("event_table", rc),
        "qvals_table": table_path("qvals_table", rc),
    }


def _write_outputs(
    out_dir: Path,
    *,
    models: pd.DataFrame,
    z_reference: Reference,
    gate_audit: pd.DataFrame,
    cohort_records: list[dict],
    fit_inputs: dict,
    argv: list[str],
    smoke: bool,
    stop_cohort: str,
) -> Path:
    """Write the fitted rows, their z reference, the provenance, and the audits.

    The fit-cohort standardization moments go beside the model as
    ``<stem>.z_reference.json``, so a per-SD coefficient can be converted back to
    raw units without refitting. ``stop_cohort`` picks the artifact suffix (the
    canonical at-well bundle keeps the unsuffixed names) and is recorded in the
    provenance sidecar. Returns the model path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = stop_cohort_suffix(stop_cohort, smoke=smoke)
    out = out_dir / f"uncertainty_rate{sfx}.pkl"
    models.to_pickle(out)
    save_reference(z_reference, z_reference_path(out))
    write_fit_provenance(
        out,
        fit_inputs,
        "poisson_glm",
        argv,
        extra={"stop_cohort": stop_cohort, "swr_source": swr_source_name()},
    )
    gate_audit.to_csv(
        out_dir / f"uncertainty_rate_post_outcome_gate_audit{sfx}.csv",
        index=False,
    )
    write_cohort_audit(
        cohort_records, out_dir / f"uncertainty_rate_cohort_audit{sfx}.csv"
    )
    return out


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-per-animal",
        type=int,
        default=None,
        help="Smoke mode: keep only the first N (post-exposure) sessions per animal.",
    )
    parser.add_argument(
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
    parser.add_argument(
        "--switch-window",
        type=int,
        choices=(CANONICAL_SWITCH_WINDOW,),
        default=CANONICAL_SWITCH_WINDOW,
        help=(
            "Completed-trial window for recent_switch_rate. The canonical "
            "tables currently support only 5; other values require rebuilding "
            "from the full trial stream."
        ),
    )
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(argv)
    # `--max-per-animal 0` is still smoke mode, so test the flag, not its value.
    smoke = args.max_per_animal is not None

    t0 = time.perf_counter()
    inputs = fit_input_paths()
    stop = read_table(inputs["stop_table"])
    onset_table = read_table(inputs["event_onset_table"])
    event_table = read_table(inputs["event_table"])
    require_covariate_schema(stop, table="stop")
    require_covariate_schema(onset_table, table="event_onset")
    require_covariate_schema(event_table, table="event")
    # The bins are dropped by the configured source's excluded time, so the
    # tables must come from that source (its artifact run included).
    require_swr_source(stop, what=str(inputs["stop_table"]))
    require_swr_source(onset_table, what=str(inputs["event_onset_table"]))
    require_swr_source(event_table, what=str(inputs["event_table"]))
    onsets = attach_event_ends(
        attach_is_swr_to_onsets(onset_table, event_table), event_table
    )

    # add_uncertainty_covariates validates the upstream recent_switch_rate (the
    # only model-free covariate still fit) and derives stop-local summaries. Its
    # leaky_q_value_diff / omission_streak / wsls_deviation stand-ins are no
    # longer used here.
    stop = add_uncertainty_covariates(stop, switch_window=args.switch_window)
    # Nominal schedule + real model covariates, ALL from qvals_table -- one
    # exactly stop-start-aligned source (retires the difflib contingency_table_csv
    # for this model). Keep qvals' canonical RPE name. Inner join: keep
    # only stops mapped to a trial (post-final-trial tails ~1-3% drop out); model
    # DVs are NaN on the first trial of a session -> listwise-dropped at fit.
    qvals = read_table(inputs["qvals_table"])
    require_covariate_schema(qvals, table="qvals")
    # Hash the inputs now, not at write time: the fit runs long enough that a
    # table could be regenerated mid-run and be certified as this fit's input.
    fit_inputs = record_inputs(inputs)
    # Guard the stop_id join before the inner join hides its casualties
    # (see assert_qvals_aligned). It must see the WHOLE stop table: it claims
    # every qvals row falling inside the epoch's span, so checking it against a
    # cohort-restricted table would read the cohort's own stops as missing.
    assert_qvals_aligned(stop, qvals)
    # Cohort next, so the qvals gap below is measured within the population
    # actually fit rather than across one this run never touches.
    stop, cohort_report = select_stop_cohort(stop, cohort=args.stop_cohort)
    logger.info(
        "stop cohort = %s (%s): %d -> %d stops (%.1f%% retained); by state %s",
        cohort_report["cohort"],
        cohort_report["rule"],
        cohort_report["n_stops_before"],
        cohort_report["n_stops_after"],
        100 * cohort_report["n_stops_after"] / max(cohort_report["n_stops_before"], 1),
        cohort_report["counts_before"],
    )
    n_before = len(stop)
    stop = join_uncertainty_qvals(stop, qvals)
    n_postjoin = len(stop)  # inner-join (qvals-gap, mechanism B) drop = n_before - this
    # Session count BEFORE the first-exposure drop, so the audit's sessions_in
    # matches n_in's stop population (n_in includes the first-exposure stops as
    # excluded, not removed).
    n_sessions_postjoin = stop.groupby(list(SESSION_COLUMNS)).ngroups
    stop = drop_exposure_sessions(stop)
    if args.max_per_animal is not None:
        stop = subset_max_per_animal(stop, args.max_per_animal, sort_sessions=True)
    # z-score the RL decision variables AFTER the drop/subset so the per-SD scale
    # is the SD of exactly the rows that get fit.
    stop, z_reference = standardize_rl_covariates(stop)

    n_sessions = stop.groupby(list(SESSION_COLUMNS)).ngroups
    logger.info(
        "Fitting %s over %d sessions x %s populations (RATE_BIN_S=%.3f, "
        "switch_window=%d)...",
        MU_UNCERTAINTY.name,
        n_sessions,
        list(POPULATIONS_TO_FIT),
        RATE_BIN_S,
        args.switch_window,
    )
    models = fit_uncertainty_rate(stop, onsets)
    gate_audit = pd.DataFrame(models.attrs["post_outcome_gate_audit"])

    # Cohort audit (stop grain). The inner qvals join is the mechanism-B qvals-gap;
    # immobility<=0 / NaN stop-level predictors on the fitted frame -> own; the
    # first-exposure drop, subset, and per-bin gates -> excluded.
    audit_qvals = 0 if smoke else (n_before - n_postjoin)  # smoke: N/A
    audit_records = uncertainty_cohort_records(
        models,
        stop,
        n_postjoin=n_postjoin,
        n_sessions_postjoin=n_sessions_postjoin,
        qvals_gap=audit_qvals,
    )
    # Which population every row was fit on, carried on the row itself: a
    # coefficient read out of a concatenated table would otherwise not say, and
    # the model comparison invalidates a cross-cohort step on this column.
    models["stop_cohort"] = args.stop_cohort
    out = _write_outputs(
        model_dir("per_bin_rate"),
        models=models,
        z_reference=z_reference,
        gate_audit=gate_audit,
        cohort_records=audit_records,
        fit_inputs=fit_inputs,
        argv=[sys.argv[0], *argv],
        smoke=smoke,
        stop_cohort=args.stop_cohort,
    )
    if models.empty:
        raise RuntimeError(
            "No uncertainty-rate cell retained both rewarded and omission bins "
            "after the physical post-outcome and complete-case gates; audits "
            "were written, but a one-outcome survivor model was not fit."
        )
    logger.info(
        "Saved %d rows (%d sessions, populations=%s) to %s in %.1fs",
        len(models),
        models.groupby(list(SESSION_COLUMNS)).ngroups if len(models) else 0,
        sorted(models["outcome"].unique()) if len(models) else [],
        out,
        time.perf_counter() - t0,
    )
    if len(models):
        for cov in FITTED_COVARIATES:
            col = f"coef_{cov}"
            if col not in models:
                continue
            summary = {
                outcome: coefficient_statistics(fits, col)["exp_median"]
                for outcome, fits in models.groupby("outcome")
            }
            logger.info("median RR  %-26s %s", cov, summary)


if __name__ == "__main__":
    main()
