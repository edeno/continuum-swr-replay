#!/usr/bin/env python
"""Fit the per-bin conditional-intensity ladder (Family B) and persist it.

Family B is the point-process counterpart to the per-stop rate hierarchy
(``fit_per_stop_rate_models.py``). Where Family A fits one Poisson GLM per stop
with a ``log(immobility_time)`` dwell offset, here every model is fit on the
canonical (stop x bin) design from ``aggregate_to_timebins`` with a
``log(exposure)`` offset, modelling the within-stop time course explicitly.

The WHOLE ladder runs at one bin width (``BIN_S = 0.015``): for each
(session, outcome) the design is built ONCE — carrying the reward-history
basis and the self-history lags — then frozen to one cohort with
``freeze_per_bin_cohort`` over the union of every rung's design columns, and
every ``MODEL_PER_BIN_RATE`` model (M0_const..M9r_rew_x_time) is fit against
that frozen design (each ``formula()`` selects its own columns). The freeze is
what makes the nested AIC / CV / GOF comparison valid: AIC scales with the
number of rows, so a rung fit to fewer bins would win mechanically. The
fitters raise rather than drop rows, and this runner additionally asserts that
freezing the shipped ladder removes zero stops (every column it needs is
complete on the canonical stop table by construction). Designs are built per
session so all sessions' bins are never resident at once.

Each (session, outcome, model) contributes one tidy row with the fit
coefficients, the 10-fold stop-based cross-validation (``cv_bits_per_event``),
and the point-process goodness-of-fit (``ks_statistic`` etc.), mirroring the
arm/patch/per-stop ``all_models.pkl`` so the families concatenate on
``[outcome, model, animal, session_date, epoch]``.

The four outcomes (``POPULATIONS``) are detected SWR, all non-local, and the
non-local SWR / no-SWR split — the pooled ``non_local`` is the one-population
null tested against its split.

Inference note. The self-history block absorbs within-stop autocorrelation, so
the per-bin likelihood is better calibrated than a naive independent-bin count
model. Even so, a single session's coefficient standard errors are a
within-session quantity, not the inferential unit: population claims come from
the across-session distribution of the per-session coefficients, and the nested
AIC / cross-validation / point-process goodness-of-fit comparison is what the
ladder is read on. Do not quote one session's SE as a calibrated population
uncertainty.

Usage:
    uv run python scripts/rate/fit_per_bin_rate_models.py               # full, serial
    uv run python scripts/rate/fit_per_bin_rate_models.py --workers 16  # full, parallel
    uv run python scripts/rate/fit_per_bin_rate_models.py --max-per-animal 1  # smoke
"""

from __future__ import annotations

import os

# Pin BLAS to one thread per process BEFORE numpy is imported: with the
# per-session process fan-out (--workers), multi-threaded BLAS would
# oversubscribe the cores and run far slower than single-threaded workers.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import logging  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import pandas as pd  # noqa: E402

from continuum_swr_replay.analysis.hippocampus.swr_source import (  # noqa: E402
    excluded_intervals_by_session,
    require_swr_source,
    swr_source_name,
)
from continuum_swr_replay.analysis.replay import (  # noqa: E402
    ARRIVAL_CLOCK_COLUMN,
    BIN_S,
    MODEL_PER_BIN_RATE,
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    POPULATIONS,
    STOP_COVARIATES,
    TMAX,
    aggregate_to_timebins,
    apply_arrival_window,
    attach_is_swr_to_onsets,
    cross_validate_per_bin_rate_model,
    fit_per_bin_rate_model,
    freeze_per_bin_cohort,
    per_bin_rate_gof,
)
from continuum_swr_replay.analysis.replay.cohort_audit import (  # noqa: E402
    cohort_record,
    stop_own_missing,
    write_cohort_audit,
)
from continuum_swr_replay.analysis.replay.cohort_selection import (  # noqa: E402
    STOP_COHORTS,
    select_stop_cohort,
    stop_cohort_suffix,
)
from continuum_swr_replay.analysis.replay.covariate_registry import (  # noqa: E402
    require_covariate_schema,
)
from continuum_swr_replay.analysis.replay.covariate_timing import (  # noqa: E402
    DERIVED_COVARIATE_SOURCES,
    assert_predictor_timing,
)
from continuum_swr_replay.analysis.replay.fit_provenance import (  # noqa: E402
    record_inputs,
    write_fit_provenance,
)
from continuum_swr_replay.analysis.replay.model_specs import (  # noqa: E402
    longest_history_window_s,
)
from continuum_swr_replay.analysis.replay.per_bin_rate_model import (  # noqa: E402
    _design_columns,
)
from continuum_swr_replay.parameters import PROCESSED_DATA_DIR  # noqa: E402
from continuum_swr_replay.session_keys import SESSION_COLUMNS  # noqa: E402
from continuum_swr_replay.table_io import read_table  # noqa: E402

# Separated self-history coefficients legitimately overflow exp() (-> 0 / inf
# RR); these are median-robust at read time and are not fit failures. Convergence /
# perfect-separation are recorded per fit as flags in per_bin_rate_model (see
# analysis.replay._fit_diagnostics), so only the benign RuntimeWarning is silenced.
warnings.filterwarnings("ignore", category=RuntimeWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

RC = PROCESSED_DATA_DIR / "replay_content"

# The tables every fit in this family is built from. The provenance sidecar
# records their hashes, and the summarizer/plotter check the same mapping, so a
# regenerated table cannot be read through a fit that predates it.
FIT_INPUTS = {
    "stop_table": RC / "stop_table.pkl",
    "event_onset_table": RC / "event_onset_table.pkl",
    "event_table": RC / "event_table.pkl",
}

# The ladder is fit on one frozen cohort, so its bins are dropped by its
# longest-reaching history window: a bin whose window touches excluded
# mask reads unobserved time.
HISTORY_WINDOW_S = longest_history_window_s(MODEL_PER_BIN_RATE.values())


def _fit_one_session(
    keys: tuple,
    stops_g: pd.DataFrame,
    onsets_g: pd.DataFrame,
    run_cv: bool,
    run_gof: bool,
) -> tuple[tuple, list[dict], dict, dict, dict, dict, float]:
    """Fit all (outcome, model) cells for one session on ONE frozen cohort.

    Returns ``(keys, rows, none_counts, cv_none_counts, cohort_drops,
    arrival_counts, elapsed)``. ``none_counts`` maps ``(model, outcome) ->
    count`` of cells whose GLM fit returned ``None`` (too little data /
    unidentified / numerical failure); ``cv_none_counts`` maps the same key to
    cells that fit but whose cross-validation returned ``None`` (a fold failed
    or the design was not CV-able), which would otherwise drop the ``cv_*``
    columns silently. ``cohort_drops`` maps ``outcome -> stops removed by the
    cohort freeze``, which the cohort audit reports as
    ``n_drop_common_cohort_other``; apart from the stops with no arrival clock
    the shipped ladder requires it to be zero (asserted below).
    ``arrival_counts`` maps ``outcome -> Counter`` of the arrival-clock cohort
    tallies (what the arrival column cost the common cohort, and what its
    window then retained). Top-level (picklable) so it can run in a worker
    process.
    """
    t0 = time.perf_counter()
    rows: list[dict] = []
    none_counts: dict = {}
    cv_none_counts: dict = {}
    cohort_drops: dict = {}
    arrival_counts: dict = {}
    has_arrival_rungs = any(
        spec.time_clock == "arrival" for spec in MODEL_PER_BIN_RATE.values()
    )
    # excluded time (the artifact mask widened by 0.3 s) is not analysed: its
    # bins, and those whose history reaches it, leave every rung
    excluded = excluded_intervals_by_session(stops_g)
    for outcome, selector in POPULATIONS.items():
        design = aggregate_to_timebins(
            stops_g,
            selector(onsets_g),
            bin_s=BIN_S,
            tmax=TMAX,
            reward_basis=True,
            self_history_lags=N_HISTORY_LAGS,
            self_history_basis=N_HISTORY_BASIS,
            stop_covariates=STOP_COVARIATES,
            excluded_intervals=excluded,
            history_window_s=HISTORY_WINDOW_S,
        )
        if design.attrs.get("n_bins_excluded") or design.attrs.get(
            "n_bins_excluded_history"
        ):
            logger.info(
                "  %s/%s: excluded time dropped %d bins and %d more by "
                "their %.3f s history window",
                "_".join(map(str, keys)),
                outcome,
                design.attrs["n_bins_excluded"],
                design.attrs["n_bins_excluded_history"],
                HISTORY_WINDOW_S,
            )
        # ONE cohort for the whole ladder, frozen before the first fit.
        frozen, freeze = freeze_per_bin_cohort(design, MODEL_PER_BIN_RATE.values())
        n_dropped = freeze["n_stops_before"] - freeze["n_stops_after"]
        cohort_drops[outcome] = n_dropped
        # The arrival clock is the ONE column of the union that is legitimately
        # missing on the canonical stop table: a stop that began before the
        # session's first well arrival has none. Because the freeze is common to
        # the ladder those bins leave every rung, so the cost is reported rather
        # than absorbed -- but it is not the upstream-table change the assertion
        # below is looking for, and is subtracted from it.
        n_dropped_arrival = int(freeze.get("n_stops_dropped_arrival_clock", 0))
        if n_dropped - n_dropped_arrival:
            offenders = {
                col: int(design[col].isna().sum())
                for col in freeze["required_columns"]
                if col != ARRIVAL_CLOCK_COLUMN and design[col].isna().any()
            }
            raise ValueError(
                f"{'_'.join(map(str, keys))}/{outcome}: freezing the shipped "
                f"per-bin ladder removed {n_dropped - n_dropped_arrival} of "
                f"{freeze['n_stops_before']} stops for reasons other than the "
                "arrival clock. Every other column the ladder needs is complete "
                "on the canonical stop table by construction (the reward-history "
                "basis is zero-filled and flagged by reward_history_incomplete, "
                "the history lags are zero-filled), so such a drop means an "
                f"upstream table changed. Missing values per column: {offenders}"
            )
        design = frozen
        # The arrival rungs are fit on the window their spline's boundary knots
        # span, taken from the frozen cohort so they still share the ladder's
        # bins; a stop-clock rung is never fit on it.
        arrival_design, window = (None, {})
        if has_arrival_rungs:
            arrival_design, window = apply_arrival_window(design)
        arrival_counts[outcome] = Counter(
            {
                "n_bins_frozen": int(freeze["n_bins_after"]),
                "n_stops_frozen": int(freeze["n_stops_after"]),
                "n_event_bins_frozen": int(freeze["n_event_bins_after"]),
                "n_bins_dropped_arrival_clock": int(
                    freeze.get("n_bins_dropped_arrival_clock", 0)
                ),
                "n_stops_dropped_arrival_clock": n_dropped_arrival,
                "n_bins_arrival_window": int(window.get("n_bins_arrival_window", 0)),
                "n_stops_arrival_window": int(window.get("n_stops_arrival_window", 0)),
                "n_event_bins_arrival_window": int(
                    window.get("n_event_bins_arrival_window", 0)
                ),
            }
        )
        for name, spec in MODEL_PER_BIN_RATE.items():
            fit_design = arrival_design if spec.time_clock == "arrival" else design
            fit = fit_per_bin_rate_model(fit_design, spec)
            if fit is None:
                none_counts[(name, outcome)] = none_counts.get((name, outcome), 0) + 1
                continue
            row = {
                "outcome": outcome,
                "model": name,
                **dict(zip(SESSION_COLUMNS, keys, strict=True)),
                # Which clock this rung's rows are on. Rungs on different clocks
                # are fit to different rows, so this travels with every fit and
                # invalidates a cross-clock comparison downstream.
                "time_clock": spec.time_clock,
                # 'predicted' is a per-bin Series (GOF input), not tidy data.
                **{k: v for k, v in fit.items() if k != "predicted"},
            }
            if spec.time_clock == "arrival":
                row.update(window)
            if run_gof:
                gof = per_bin_rate_gof(fit_design, fit)
                # n_events / dispersion duplicate the fit dict (same rows);
                # keep the fit's to avoid a silent, fragile overwrite.
                gof.pop("n_events", None)
                gof.pop("dispersion", None)
                row.update(gof)
            if run_cv:
                cv = cross_validate_per_bin_rate_model(fit_design, spec)
                if cv is not None:
                    row.update(cv)
                else:
                    cv_none_counts[(name, outcome)] = (
                        cv_none_counts.get((name, outcome), 0) + 1
                    )
            rows.append(row)
    return (
        keys,
        rows,
        none_counts,
        cv_none_counts,
        cohort_drops,
        arrival_counts,
        time.perf_counter() - t0,
    )


def fit_per_bin_rate_models(
    stop_table: pd.DataFrame,
    onsets: pd.DataFrame,
    *,
    run_cv: bool = True,
    run_gof: bool = True,
    workers: int = 1,
) -> pd.DataFrame:
    """One row per (outcome, model, session): coefficients + CV + GOF.

    ``onsets`` must already carry the canonical ``is_swr`` column (run
    :func:`attach_is_swr_to_onsets` once before calling). ``workers > 1``
    fans the per-session work out over processes (sessions are independent).

    Every rung of one (session, outcome) cell is fit to the frozen cohort, so
    all its stop-clock rungs report the same ``n_bins`` / ``n_stops``; the
    arrival-clock rungs are fit on the arrival window carved out of that same
    cohort, so they share an ``n_bins`` / ``n_stops`` with each other. The
    number of stops that freeze removed per outcome is attached as
    ``result.attrs["cohort_stop_drops"]`` for the cohort audit; apart from the
    stops with no arrival clock the shipped ladder requires it to be zero. The
    arrival-clock cohort tallies (what requiring the arrival clock cost the
    common cohort, and what the window then retained, per outcome) are attached
    as ``result.attrs["arrival_clock_cohort"]``.
    """
    onset_groups = dict(tuple(onsets.groupby(list(SESSION_COLUMNS), sort=False)))
    items = [
        (keys, stops_g, onset_groups[keys])
        for keys, stops_g in stop_table.groupby(list(SESSION_COLUMNS), sort=False)
        if keys in onset_groups
    ]

    rows: list[dict] = []
    none_counts: Counter = Counter()
    cv_none_counts: Counter = Counter()
    cohort_stop_drops: Counter = Counter()
    arrival_cohort: dict[str, Counter] = {outcome: Counter() for outcome in POPULATIONS}

    def _collect(keys, r, nc, cvnc, drops, arrival, elapsed):
        rows.extend(r)
        none_counts.update(nc)
        cv_none_counts.update(cvnc)
        cohort_stop_drops.update(drops)
        for outcome, counts in arrival.items():
            arrival_cohort[outcome].update(counts)
        logger.info("  %s: %d fits in %.1fs", "_".join(map(str, keys)), len(r), elapsed)

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(_fit_one_session, k, s, o, run_cv, run_gof)
                for k, s, o in items
            ]
            for fut in as_completed(futures):
                _collect(*fut.result())
    else:
        for k, s, o in items:
            _collect(*_fit_one_session(k, s, o, run_cv, run_gof))

    if none_counts:
        logger.info(
            "None (unfit) counts per (model, outcome): %s",
            {f"{m}/{o}": c for (m, o), c in sorted(none_counts.items())},
        )
    if cv_none_counts:
        logger.info(
            "CV-None (fit ok, CV dropped) counts per (model, outcome): %s",
            {f"{m}/{o}": c for (m, o), c in sorted(cv_none_counts.items())},
        )
    for outcome, counts in arrival_cohort.items():
        if not counts["n_bins_frozen"]:
            continue
        kept = counts["n_bins_frozen"] + counts["n_bins_dropped_arrival_clock"]
        logger.info(
            "  %s: the arrival clock cost the common cohort %d/%d bins (%.2f%%) "
            "and %d stops; its window then kept %d/%d bins (%.1f%%) and "
            "%d/%d event bins (%.1f%%)",
            outcome,
            counts["n_bins_dropped_arrival_clock"],
            kept,
            100.0 * counts["n_bins_dropped_arrival_clock"] / kept,
            counts["n_stops_dropped_arrival_clock"],
            counts["n_bins_arrival_window"],
            counts["n_bins_frozen"],
            100.0 * counts["n_bins_arrival_window"] / counts["n_bins_frozen"],
            counts["n_event_bins_arrival_window"],
            counts["n_event_bins_frozen"],
            100.0
            * counts["n_event_bins_arrival_window"]
            / max(counts["n_event_bins_frozen"], 1),
        )
    result = pd.DataFrame(rows)
    # Workers finish in any order. Each session's rows arrive together, in
    # ladder order, so a stable sort by session fixes the row order of the
    # saved frame without reordering the ladder within a session.
    if len(result):
        result = result.sort_values(
            list(SESSION_COLUMNS), kind="stable", ignore_index=True
        )
    result.attrs["cohort_stop_drops"] = {
        outcome: int(cohort_stop_drops.get(outcome, 0)) for outcome in POPULATIONS
    }
    result.attrs["arrival_clock_cohort"] = {
        outcome: dict(counts) for outcome, counts in arrival_cohort.items()
    }
    return result


def _subset_max_per_animal(stop_table: pd.DataFrame, n: int) -> pd.DataFrame:
    keep = (
        stop_table[list(SESSION_COLUMNS)]
        .drop_duplicates()
        .groupby("animal", sort=False)
        .head(n)
    )
    return stop_table.merge(keep, on=list(SESSION_COLUMNS))


def _stop_level_design_columns(spec, stop: pd.DataFrame) -> list[str]:
    """A spec's design columns as they exist on the stop table.

    A per-bin design column that the design *derives* from a stop column (the
    arrival clock, built as bin centre minus ``last_well_arrival_time_s``) is
    resolved to that source, so a stop whose clock is missing is attributed to
    the rung that needs it instead of vanishing from the audit.
    """
    cols = [
        source
        for column in _design_columns(spec)
        for source in DERIVED_COVARIATE_SOURCES.get(column, [column])
        if source in stop.columns
    ]
    return list(dict.fromkeys(cols))


def _cohort_records(
    models: pd.DataFrame,
    stop: pd.DataFrame,
    cohort_stop_drops: dict | None = None,
) -> list[dict]:
    """Stop-grain cohort record per (model, population) for the per-bin ladder.

    Family B has no qvals merge, and every rung shares one frozen cohort
    (``freeze_per_bin_cohort``), so a stop leaves the modelled set for one of
    two reasons. It is an **own-predictor** drop when ``immobility_time<=0`` or
    one of *this* rung's stop-level required predictors is NaN; it is a
    **common-cohort** drop (``n_drop_common_cohort_other``) when this rung's own
    predictors are all present but another rung of the ladder needs a column
    that is NaN — the freeze removes it from every rung. ``n_modeled`` is the
    fitter's actual distinct-modeled-stop count (``n_stops``); remaining
    survivors of skipped / gated sessions and any bin-level drops fall into
    ``n_excluded_session``. Reconciles by construction.

    ``cohort_stop_drops`` is the runner's per-outcome count of stops the freeze
    actually removed (``result.attrs["cohort_stop_drops"]``); it must not exceed
    what the stop-level attribution accounts for. A per-bin design column the
    design derives from a stop column (the arrival clock from
    ``last_well_arrival_time_s``) is attributed through its source, since that
    is where the missingness lives.
    """
    total = int(len(stop))
    n_sessions_in = stop.groupby(list(SESSION_COLUMNS), sort=False).ngroups
    frozen_drops = cohort_stop_drops or {}
    union_cols = [
        c
        for spec in MODEL_PER_BIN_RATE.values()
        for c in _stop_level_design_columns(spec, stop)
    ]
    union_missing = stop_own_missing(stop, list(dict.fromkeys(union_cols)))
    recs = []
    for name, spec in MODEL_PER_BIN_RATE.items():
        stop_level = _stop_level_design_columns(spec, stop)
        own_missing = stop_own_missing(stop, stop_level)
        n_own = int(own_missing.sum())
        n_common = int((union_missing & ~own_missing).sum())
        for outcome in POPULATIONS:
            sub = (
                models[(models.model == name) & (models.outcome == outcome)]
                if len(models)
                else models
            )
            fit_keys = (
                {tuple(r) for r in sub[list(SESSION_COLUMNS)].to_numpy()}
                if len(sub)
                else set()
            )
            n_modeled = int(sub["n_stops"].sum()) if "n_stops" in sub else 0
            n_frozen_out = int(frozen_drops.get(outcome, 0))
            if n_frozen_out > n_own + n_common:
                raise AssertionError(
                    f"{name}/{outcome}: the cohort freeze removed {n_frozen_out} "
                    f"stops but the stop-level attribution only accounts for "
                    f"{n_own + n_common} (own {n_own} + common {n_common})"
                )
            recs.append(
                cohort_record(
                    name,
                    outcome,
                    "stop",
                    cohort_mode="common",
                    n_in=total,
                    n_modeled=n_modeled,
                    sessions_in=n_sessions_in,
                    sessions_modeled=len(fit_keys),
                    animals_modeled=len({k[0] for k in fit_keys}),
                    drop_notes=(
                        "immobility<=0 / NaN own stop-level predictor -> own; "
                        "NaN only in another rung's column (frozen cohort) -> common; "
                        "min_bins/min_events/reward-contrast gates + bin-level -> excluded"
                        + (
                            "; a stop whose every bin is past the arrival window is "
                            "excluded too"
                            if spec.time_clock == "arrival"
                            else ""
                        )
                    ),
                    n_drop_own_predictor_missing=n_own,
                    n_drop_common_cohort_other=n_common,
                    n_excluded_session=total - n_own - n_common - n_modeled,
                    n_sessions_flagged_nonconverged=(
                        int((sub["converged"] == False).sum())  # noqa: E712
                        if "converged" in sub
                        else 0
                    ),
                    n_sessions_flagged_separated=(
                        int((sub["separated"] == True).sum())  # noqa: E712
                        if "separated" in sub
                        else 0
                    ),
                )
            )
    return recs


def main() -> None:
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
    args = parser.parse_args()

    # Causal-hygiene gate: the per-bin design basis (spline, reward kernel,
    # self-/cross-history) is backward/within-stop safe; a future leak fails loud.
    for spec in MODEL_PER_BIN_RATE.values():
        assert_predictor_timing(c for c in _design_columns(spec) if c != "count")

    stop = read_table(RC / "stop_table.pkl")
    onset_table = read_table(RC / "event_onset_table.pkl")
    event_table = read_table(RC / "event_table.pkl")
    require_covariate_schema(stop, table="stop")
    require_covariate_schema(onset_table, table="event_onset")
    require_covariate_schema(event_table, table="event")
    # The bins are dropped by the configured source's excluded time, so the
    # tables must come from that source (its artifact run included).
    require_swr_source(stop, what=str(FIT_INPUTS["stop_table"]))
    require_swr_source(onset_table, what=str(FIT_INPUTS["event_onset_table"]))
    # Hash the inputs now, not at write time: the ladder runs long enough that a
    # table could be regenerated mid-fit and be certified as this fit's input.
    fit_inputs = record_inputs(FIT_INPUTS)
    require_swr_source(event_table, what=str(FIT_INPUTS["event_table"]))
    onsets = attach_is_swr_to_onsets(onset_table, event_table)
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
        stop = _subset_max_per_animal(stop, args.max_per_animal)

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

    out_dir = RC / "models_per_bin_rate"
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
        sys.argv,
        extra={"stop_cohort": args.stop_cohort, "swr_source": swr_source_name()},
    )
    write_cohort_audit(
        _cohort_records(models, stop, models.attrs.get("cohort_stop_drops")),
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
