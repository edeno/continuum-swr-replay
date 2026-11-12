"""The per-bin rate ladder (Family B) fit over sessions, and its cohort audit.

For each (session, outcome) the at-risk design is built once
(:func:`ladder_design`), frozen to one cohort over the union of every
``MODEL_PER_BIN_RATE`` rung's columns, and every rung is fit to it
(:func:`fit_ladder_session`); the arrival-clock rungs are fit on the arrival
window carved out of that cohort. :func:`fit_per_bin_rate_models` runs the
sessions, serially or over processes, and returns one tidy row per (session,
outcome, model) with the freeze's tallies in ``attrs``;
:func:`per_bin_cohort_records` turns them into the stop-grain cohort audit.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd

from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

from ..cohort_audit import fitted_cohort_record, stop_own_missing
from ..covariate_timing import DERIVED_COVARIATE_SOURCES
from ..model_specs import (
    ARRIVAL_CLOCK_COLUMN,
    BIN_S,
    MODEL_PER_BIN_RATE,
    N_HISTORY_BASIS,
    N_HISTORY_LAGS,
    PER_BIN_DESIGN,
    STOP_COVARIATES,
    TMAX,
    longest_history_window_s,
)
from ..per_bin_design import POPULATIONS, aggregate_to_timebins
from .duration_aware_history import at_risk_events
from .per_bin_rate_model import (
    _design_columns,
    apply_arrival_window,
    cross_validate_per_bin_rate_model,
    fit_per_bin_rate_model,
    freeze_per_bin_cohort,
    per_bin_rate_gof,
)
from .rate_fit_statistics import arrival_window_statistics

logger = logging.getLogger(__name__)

# The ladder is fit on one frozen cohort, so its bins are dropped by its
# longest-reaching history window: a bin whose window touches excluded
# mask reads unobserved time.
HISTORY_WINDOW_S = longest_history_window_s(MODEL_PER_BIN_RATE.values())


def ladder_design(
    stops_g: pd.DataFrame,
    onsets_g: pd.DataFrame,
    outcome: str,
    *,
    stop_covariates: Sequence[str] = STOP_COVARIATES,
) -> pd.DataFrame:
    """One session's at-risk per-bin design for ``outcome``, before the freeze.

    Parameters
    ----------
    stops_g, onsets_g : pd.DataFrame
        One session's stop rows, and its onset rows carrying ``is_swr`` and
        each event's end.
    outcome : str
        A key of ``POPULATIONS``.
    stop_covariates : sequence of str
        Stop columns carried onto every bin; the ladder's own by default. An
        extra column (``stop_start``, to order the bins) changes no fit.

    Returns
    -------
    pd.DataFrame, shape (n_bins, n_columns)
        The design :func:`fit_ladder_session` freezes and fits.
    """
    return aggregate_to_timebins(
        stops_g,
        POPULATIONS[outcome](onsets_g),
        bin_s=BIN_S,
        tmax=TMAX,
        reward_basis=True,
        self_history_lags=N_HISTORY_LAGS,
        self_history_basis=N_HISTORY_BASIS,
        stop_covariates=stop_covariates,
        history_window_s=HISTORY_WINDOW_S,
        at_risk_events=at_risk_events(onsets_g, outcome),
        offset_history=True,
    )


def fit_ladder_session(
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
    for outcome in POPULATIONS:
        design = ladder_design(stops_g, onsets_g, outcome)
        # An onset of the response is at risk in its own bin by construction;
        # one in a dropped bin means two of the detector's events overlap.
        if design.attrs["n_onsets_not_at_risk"]:
            raise ValueError(
                f"{'_'.join(map(str, keys))}/{outcome}: "
                f"{design.attrs['n_onsets_not_at_risk']} onsets fell in bins "
                "with no at-risk time"
            )
        n_bins_not_at_risk = int(design.attrs["n_bins_not_at_risk"])
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
                # The design the row was fit on, so a reader can refuse a fit
                # from the old all-bins design.
                "per_bin_design": PER_BIN_DESIGN,
                "n_bins_not_at_risk": n_bins_not_at_risk,
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

    ``onsets`` must already carry the canonical ``is_swr`` column and each
    event's end (run ``attach_is_swr_to_onsets`` and ``attach_event_ends``
    once before calling). ``workers > 1`` fans the per-session work out over
    processes (sessions are independent).

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
    stop_groups = stop_table.groupby(list(SESSION_COLUMNS), sort=False)
    items = [
        (keys, stops_g, onset_groups[keys])
        for keys, stops_g in stop_groups
        if keys in onset_groups
    ]
    # A stop-session with no onset rows is not fit at all; name it rather than
    # let it vanish from the ladder unremarked.
    no_onsets = [keys for keys in stop_groups.groups if keys not in onset_groups]
    if no_onsets:
        logger.warning(
            "Skipping %d stop-session(s) with no event onsets: %s",
            len(no_onsets),
            ", ".join("_".join(map(str, keys)) for keys in no_onsets),
        )

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
                ex.submit(fit_ladder_session, k, s, o, run_cv, run_gof)
                for k, s, o in items
            ]
            for fut in as_completed(futures):
                _collect(*fut.result())
    else:
        for k, s, o in items:
            _collect(*fit_ladder_session(k, s, o, run_cv, run_gof))

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
        # a population without event bins (a smoke run) has no window shares
        if not counts["n_event_bins_frozen"]:
            continue
        window = arrival_window_statistics(counts)
        logger.info(
            "  %s: the arrival clock cost the common cohort %d/%d bins (%.2f%%) "
            "and %d stops; its window then kept %d/%d bins (%.1f%%) and "
            "%d/%d event bins (%.1f%%)",
            outcome,
            counts["n_bins_dropped_arrival_clock"],
            window["n_bins_before"],
            100.0 * window["dropped_bin_fraction"],
            counts["n_stops_dropped_arrival_clock"],
            counts["n_bins_arrival_window"],
            counts["n_bins_frozen"],
            100.0 * window["bin_share"],
            counts["n_event_bins_arrival_window"],
            counts["n_event_bins_frozen"],
            100.0 * window["event_share"],
        )
    result = pd.DataFrame(rows)
    # Workers finish in any order, so the rows are sorted by their full key
    # (unique per row) before the frame is saved.
    if len(result):
        result = result.sort_values(
            [*SESSION_COLUMNS, "outcome", "model"], ignore_index=True
        )
    result.attrs["cohort_stop_drops"] = {
        outcome: int(cohort_stop_drops.get(outcome, 0)) for outcome in POPULATIONS
    }
    result.attrs["arrival_clock_cohort"] = {
        outcome: dict(counts) for outcome, counts in arrival_cohort.items()
    }
    return result


def stop_level_design_columns(spec, stop: pd.DataFrame) -> list[str]:
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


def per_bin_cohort_records(
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

    ``cohort_stop_drops`` is the per-outcome count of stops the freeze
    actually removed (``result.attrs["cohort_stop_drops"]`` of
    :func:`fit_per_bin_rate_models`); it must not exceed what the stop-level
    attribution accounts for. A per-bin design column the design derives from a
    stop column (the arrival clock from ``last_well_arrival_time_s``) is
    attributed through its source, since that is where the missingness lives.
    """
    total = int(len(stop))
    n_sessions_in = stop.groupby(list(SESSION_COLUMNS), sort=False).ngroups
    frozen_drops = cohort_stop_drops or {}
    union_cols = [
        c
        for spec in MODEL_PER_BIN_RATE.values()
        for c in stop_level_design_columns(spec, stop)
    ]
    union_missing = stop_own_missing(stop, list(dict.fromkeys(union_cols)))
    recs = []
    for name, spec in MODEL_PER_BIN_RATE.items():
        own_missing = stop_own_missing(stop, stop_level_design_columns(spec, stop))
        for outcome in POPULATIONS:
            sub = (
                models[(models.model == name) & (models.outcome == outcome)]
                if len(models)
                else models
            )
            recs.append(
                fitted_cohort_record(
                    name,
                    outcome,
                    "stop",
                    sub,
                    count_column="n_stops",
                    cohort_mode="common",
                    n_in=total,
                    sessions_in=n_sessions_in,
                    own_missing=own_missing,
                    common_missing=union_missing,
                    n_frozen_out=int(frozen_drops.get(outcome, 0)),
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
                )
            )
    return recs
