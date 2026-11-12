#!/usr/bin/env python
"""Fit the per-bin conditional-intensity ladder (Family B) and persist it.

Family B is the point-process counterpart to the per-stop rate hierarchy
(``fit_per_stop_rate_models.py``). Where Family A fits one Poisson GLM per stop
with a ``log(immobility_duration_s)`` dwell offset, here every model is fit on the
canonical (stop x bin) design from ``aggregate_to_timebins`` with a
``log(exposure)`` offset, modelling the within-stop time course explicitly.

The design is the at-risk one (``PER_BIN_DESIGN``): a bin inside an ongoing
event of the outcome's detector cannot hold an onset, so it is dropped
(``duration_aware_history.at_risk_events``), and the self-history control is
counted from event ends (``PER_BIN_HISTORY_ANCHORS``). Each rung's rate is
therefore the rate at which events start while none is ongoing, not events per
second of stop (Family A's quantity).

The WHOLE ladder runs at one bin width (``BIN_S = 0.015``): for each
(session, outcome) the design is built ONCE — carrying the reward-history
basis and the self-history lags — then frozen to one cohort with
``freeze_per_bin_cohort`` over the union of every rung's design columns, and
every ``MODEL_PER_BIN_RATE`` model is fit against that frozen design (each
``formula()`` selects its own columns): the stop-clock ladder M0_const, M1_rew,
M5_time, M3r_histcos, M6r_time_histcos, M7r_time_histcos_rew and
M9r_rew_x_time, and the arrival-clock sub-ladder M0a_const_arrival,
M5a_time_arrival and M7ra_time_arrival_histcos_rew, which is fit on the arrival
window carved out of the same frozen cohort. The freeze is what makes the
nested AIC / CV / GOF comparison valid: AIC scales with the number of rows, so
a rung fit to fewer bins would win mechanically. The
fitters raise rather than drop rows, and the ladder additionally asserts that
freezing it removes no stop but those without an arrival clock (every other
column it needs is complete on the canonical stop table by construction).
Designs are built per session so all sessions' bins are never resident at once.
The fit loop and the cohort audit are
``analysis/behavior/rate/per_bin_rate_ladder.py``; this script reads the
tables, selects the stop cohort, and writes.

Each (session, outcome, model) contributes one tidy row with the fit
coefficients, the 10-fold stop-based cross-validation (``cv_bits_per_event``),
and the point-process goodness-of-fit (``ks_statistic`` etc.), mirroring the
arm/patch/per-stop ``all_models.pkl`` so the families concatenate on
``[outcome, model, animal, session_date, epoch]``.

The four outcomes (``POPULATIONS``) are detected SWR, all non-local, and the
non-local SWR / no-SWR split — the pooled ``non_local`` is the one-population
null tested against its split.

How this family (B) differs from Family A on the same stop is in the
``fit_per_stop_rate_models.py`` docstring. Family B never reads the dense
``stop_timebin_table``: ``aggregate_to_timebins`` builds the 15 ms bins from
``stop_table`` and ``event_onset_table``.

Outputs: ``models_per_bin_rate/all_models{sfx}.pkl`` (one row per (session,
outcome, model)), its ``.provenance.json`` sidecar, and
``per_bin_rate_cohort_audit{sfx}.csv``; ``sfx`` is empty for the default
``at_well`` cohort, ``_allstops`` / ``_intransit`` otherwise, with ``_smoke``
appended in smoke mode.

Inference note. The self-history block absorbs within-stop autocorrelation, so
the per-bin likelihood is better calibrated than a naive independent-bin count
model. Even so, a single session's coefficient standard errors are a
within-session quantity, not the inferential unit: population claims come from
the across-session distribution of the per-session coefficients, and the nested
AIC / cross-validation / point-process goodness-of-fit comparison is what the
ladder is read on. Do not quote one session's SE as a calibrated population
uncertainty.

Usage:
    uv run python scripts/behavior/rate/fit_per_bin_rate_models.py               # full, serial
    uv run python scripts/behavior/rate/fit_per_bin_rate_models.py --workers 16  # full, parallel
    uv run python scripts/behavior/rate/fit_per_bin_rate_models.py --max-per-animal 1  # smoke
"""

from __future__ import annotations

from collections.abc import Sequence

from continuum_swr_replay.runs.threads import pin_blas_threads

# Pin BLAS to one thread per process BEFORE numpy is imported: with the
# per-session process fan-out (--workers), multi-threaded BLAS would
# oversubscribe the cores and run far slower than single-threaded workers.
pin_blas_threads()

import argparse  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

from continuum_swr_replay.analysis.behavior import (  # noqa: E402
    BIN_S,
    MODEL_PER_BIN_RATE,
    POPULATIONS,
    attach_is_swr_to_onsets,
)
from continuum_swr_replay.analysis.behavior.cohort_audit import (  # noqa: E402
    write_cohort_audit,
)
from continuum_swr_replay.analysis.behavior.covariate_timing import (  # noqa: E402
    assert_predictor_timing,
)
from continuum_swr_replay.analysis.behavior.fit_provenance import (  # noqa: E402
    record_inputs,
    write_fit_provenance,
)
from continuum_swr_replay.analysis.behavior.rate.duration_aware_history import (  # noqa: E402
    attach_event_ends,
)
from continuum_swr_replay.analysis.behavior.rate.per_bin_rate_ladder import (  # noqa: E402
    fit_per_bin_rate_models,
    per_bin_cohort_records,
)
from continuum_swr_replay.analysis.behavior.rate.per_bin_rate_model import (  # noqa: E402
    _design_columns,
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

# Separated self-history coefficients legitimately overflow exp() (-> 0 / inf
# RR); these are median-robust at read time and are not fit failures. Convergence /
# perfect-separation are recorded per fit as flags in per_bin_rate_model (see
# analysis.behavior._fit_diagnostics), so only the benign RuntimeWarning is silenced.
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def fit_input_paths() -> dict[str, Path]:
    """The tables every fit in this family is built from, by name.

    The provenance sidecar records their hashes, and the summarizer/plotter
    check the same mapping, so a regenerated table cannot be read through a fit
    that predates it. Under ``paths.replay_content_dir()``, read when called.
    """
    rc = replay_content_dir()
    return {
        "stop_table": table_path("stop_table", rc),
        "event_onset_table": table_path("event_onset_table", rc),
        "event_table": table_path("event_table", rc),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-per-animal",
        type=int,
        default=None,
        help="Smoke mode: keep only the first N sessions per animal.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Process workers for the per-session fan-out (default serial).",
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
    parser.add_argument("--no-cv", action="store_true", help="Skip cross-validation.")
    parser.add_argument("--no-gof", action="store_true", help="Skip goodness-of-fit.")
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(argv)

    # Causal-hygiene gate: the per-bin design basis (spline, reward kernel,
    # self-/cross-history) is backward/within-stop safe; a future leak fails loud.
    for spec in MODEL_PER_BIN_RATE.values():
        assert_predictor_timing(c for c in _design_columns(spec) if c != "count")

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
    # Hash the inputs now, not at write time: the ladder runs long enough that a
    # table could be regenerated mid-fit and be certified as this fit's input.
    fit_inputs = record_inputs(inputs)
    require_swr_source(event_table, what=str(inputs["event_table"]))
    onsets = attach_event_ends(
        attach_is_swr_to_onsets(onset_table, event_table), event_table
    )
    # The cohort is applied to the whole stop table BEFORE the smoke subset (so a
    # smoke session is a cohort session) and before the per-session common-cohort
    # freeze (so the freeze reports what the fitted population is missing, not
    # what some other population is).
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
    if args.max_per_animal is not None:
        stop = subset_max_per_animal(stop, args.max_per_animal)

    logger.info(
        "Fitting per-bin ladder %s (+CV=%s, +GOF=%s) x %s over %d sessions "
        "at BIN_S=%.3f with %d worker(s)...",
        list(MODEL_PER_BIN_RATE),
        not args.no_cv,
        not args.no_gof,
        list(POPULATIONS),
        stop.groupby(list(SESSION_COLUMNS)).ngroups,
        BIN_S,
        args.workers,
    )
    t0 = time.perf_counter()
    models = fit_per_bin_rate_models(
        stop,
        onsets,
        run_cv=not args.no_cv,
        run_gof=not args.no_gof,
        workers=args.workers,
    )

    # Which population every row was fit on, carried on the row itself: a
    # coefficient read out of a concatenated table would otherwise not say, and
    # the model comparison invalidates a cross-cohort step on this column.
    models["stop_cohort"] = args.stop_cohort

    out_dir = model_dir("per_bin_rate")
    out_dir.mkdir(parents=True, exist_ok=True)
    # `--max-per-animal 0` is still smoke mode, so test the flag, not its value.
    smoke = args.max_per_animal is not None
    csfx = stop_cohort_suffix(args.stop_cohort, smoke=smoke)
    out = out_dir / f"all_models{csfx}.pkl"
    models.to_pickle(out)
    write_fit_provenance(
        out,
        fit_inputs,
        "poisson_glm",
        [sys.argv[0], *argv],
        extra={"stop_cohort": args.stop_cohort, "swr_source": swr_source_name()},
    )
    write_cohort_audit(
        per_bin_cohort_records(models, stop, models.attrs.get("cohort_stop_drops")),
        out_dir / f"per_bin_rate_cohort_audit{csfx}.csv",
    )
    logger.info(
        "Saved %d rows (%d sessions, outcomes=%s, models=%s) to %s in %.1fs",
        len(models),
        models.groupby(list(SESSION_COLUMNS)).ngroups if len(models) else 0,
        sorted(models["outcome"].unique()) if len(models) else [],
        sorted(models["model"].unique()) if len(models) else [],
        out,
        time.perf_counter() - t0,
    )


if __name__ == "__main__":
    main()
