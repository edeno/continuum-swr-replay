#!/usr/bin/env python
"""Fit the per-stop replay-rate hierarchy (R0-R1) and persist it.

The rate models are ordinary Poisson GLMs on per-stop event counts with a
``log(immobility_duration_s)`` offset (see ``per_stop_rate_model.fit_per_stop_rate_model``) — a
different model family from the conditional arm/patch hierarchies, so they
live outside ``fit_replay_content_models.py``. This script fits R0-R1
per session from the saved ``stop_table``, adds K-fold cross-validated
per-stop log-likelihood (``cross_validate_per_stop_rate_model``), and saves a tidy
one-row-per-(outcome, model, session) table mirroring the arm/patch
``all_models.pkl``.

The response is fit on four per-stop count outcomes (tagged by ``outcome``),
all read directly from the canonical ``stop_table`` columns:

- ``swr``              -> ``n_swr_events``              : detected ripples (LFP) — the
                                                  Ambrose-2016 positive control.
- ``non_local``        -> ``n_selected_events``        : all non-local decoding events.
- ``non_local_swr``    -> ``n_nonlocal_swr_events``    : non-local events overlapping a SWR.
- ``non_local_no_swr`` -> ``n_nonlocal_non_swr_events`` : non-local events without a SWR.

Fitting each MODEL_PER_STOP_RATE model on each outcome is an outcome *variant* of the specs,
not a new model. Tests the canonical SWR-rate-vs-reward relation on detected
ripples through recent reward history only -- R1 has no current-trip reward /
omission term, because those are post-outcome quantities (``model_specs``,
"Rate models") -- then asks whether the non-SWR non-local events share that
rate effect.

How Family A (this script) and Family B (``fit_per_bin_rate_models.py``) differ
on the same stop. Family A gives the stop one row: its whole-stop event count
(the ``stop_table`` columns above), a ``log(immobility_duration_s)`` offset, and
the reward-history basis evaluated once at bout start, so the whole stop carries
one history. Family B cuts the same stop into ``BIN_S`` = 15 ms bins over its
first ``TMAX`` = 4 s only, counts onsets per bin from ``event_onset_table`` with
a ``log(exposure)`` offset, adds a time-since-stop spline and self-history, and
in the bins at or past the stop's outcome boundary (``outcome_known_time_s``)
shifts the reward-history lag vector by one so they see the trip's own reward
or omission (a bout that spans a well arrival is never shifted). Onsets after
the first 4 s count toward A but not toward B.

Outputs: ``models_per_stop_rate/all_models{sfx}.pkl`` (one row per (outcome,
model, session) for R0-R1), its ``.provenance.json`` sidecar, and
``all_models{sfx}_cohort_audit.csv``; ``sfx`` is empty for the default
``at_well`` cohort, ``_allstops`` / ``_intransit`` otherwise, with ``_smoke``
appended in smoke mode.

Usage:
    uv run python scripts/rate/fit_per_stop_rate_models.py                    # full
    uv run python scripts/rate/fit_per_stop_rate_models.py --max-per-animal 1  # smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.events.analysed_time import require_swr_source
from continuum_swr_replay.analysis.events.cohort_selection import (
    STOP_COHORTS,
    freeze_common_cohort,
    select_stop_cohort,
    stop_cohort_suffix,
)
from continuum_swr_replay.analysis.replay.cohort_audit import (
    cohort_record,
    write_cohort_audit,
)
from continuum_swr_replay.analysis.replay.covariate_timing import (
    assert_predictor_timing,
)
from continuum_swr_replay.analysis.replay.fit_provenance import (
    record_inputs,
    write_fit_provenance,
)
from continuum_swr_replay.analysis.replay.model_specs import (
    MODEL_PER_STOP_RATE,
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_INCOMPLETE,
)
from continuum_swr_replay.analysis.replay.per_stop_rate_model import (
    PER_STOP_RATE_OUTCOMES,
    cross_validate_per_stop_rate_model,
    fit_per_stop_rate_model,
)
from continuum_swr_replay.paths import model_dir, table_path
from continuum_swr_replay.tables.covariate_registry import (
    require_covariate_schema,
)
from continuum_swr_replay.tables.io import read_table
from continuum_swr_replay.tables.session_keys import (
    SESSION_COLUMNS,
    subset_max_per_animal,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def fit_input_paths() -> dict[str, Path]:
    """The table this family is fit on, recorded in the artifact's provenance
    sidecar; under ``paths.replay_content_dir()``, read when called."""
    return {"stop_table": table_path("stop_table")}


# Per-stop count outcomes (canonical stop_table columns), tagged by `outcome`.
RESPONSE_COLS = PER_STOP_RATE_OUTCOMES
OFFSET_COL = "immobility_duration_s"
# Union of every rung's predictors: the ladder is frozen to one cohort over
# these, so R0 and R1 are fit to the same stops and their AIC is comparable.
ALL_PREDICTORS = list(
    dict.fromkeys(c for spec in MODEL_PER_STOP_RATE.values() for c in spec.predictors)
)


def _incomplete(group: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Rows a complete-case freeze would remove for ``columns`` (+ the offset).

    Mirrors :func:`freeze_common_cohort`'s first three filters at row grain:
    missing value, non-finite numeric value, nonpositive offset. The stop table
    is one row per stop, so row grain and unit grain coincide.
    """
    present = [c for c in columns if c in group.columns]
    bad = (
        group[present].isna().any(axis=1)
        if present
        else pd.Series(False, index=group.index)
    )
    numeric = [c for c in present if pd.api.types.is_numeric_dtype(group[c])]
    if numeric:
        values = group[numeric].to_numpy(dtype=float)
        bad = bad | ~np.isfinite(values).all(axis=1)
    if OFFSET_COL in group.columns:
        bad = bad | (group[OFFSET_COL] <= 0)
    return bad


def fit_per_stop_rate_hierarchy(
    stop_table: pd.DataFrame, run_cv: bool = True
) -> tuple[pd.DataFrame, list[dict]]:
    """One row per (outcome, model, session): R0-R1 fit stats, coefs, CV-LL.

    Every rung of a session is fit to ONE cohort, frozen once per session over
    the union of the ladder's predictors, the outcome columns present, and a
    positive ``immobility_duration_s`` — so R0 and R1 report the same
    ``n_stops`` and their AIC / BIC / LRT are comparable.

    Also returns one stop-grain cohort-audit record per (model, outcome): a
    stop missing THIS rung's own predictor (or with ``immobility<=0``) goes to
    ``n_drop_own_predictor_missing``; a stop the freeze removes only because
    ANOTHER rung's predictor is missing there goes to
    ``n_drop_common_cohort_other``; a skipped session's remaining stops
    (min_stops / rank gate) go to ``n_excluded_session``; the fitted
    ``n_stops`` sum is ``n_modeled``. Reconciles by construction.
    """
    missing_predictors = [c for c in ALL_PREDICTORS if c not in stop_table.columns]
    if missing_predictors:
        raise KeyError(
            "stop table is missing per-stop rate predictor(s) "
            f"{missing_predictors}; the ladder cannot be frozen to one cohort."
        )
    responses = [c for c in RESPONSE_COLS.values() if c in stop_table.columns]

    rows = []
    acc: dict = {}  # (model, outcome) -> counts
    for keys, group in stop_table.groupby(list(SESSION_COLUMNS), sort=False):
        # ONE cohort for the whole ladder and every outcome of this session.
        cohort, _selection = freeze_common_cohort(
            group,
            unit_col="stop_id",
            required_cols=[*ALL_PREDICTORS, *responses],
            offset_col=OFFSET_COL,
        )
        dropped = ~group["stop_id"].isin(cohort["stop_id"])
        for outcome, response_col in RESPONSE_COLS.items():
            if response_col not in group.columns:
                continue
            for name, spec in MODEL_PER_STOP_RATE.items():
                preds = list(spec.predictors)
                a = acc.setdefault(
                    (name, outcome),
                    {
                        "n_in": 0,
                        "n_own": 0,
                        "n_common": 0,
                        "n_modeled": 0,
                        "sessions_in": 0,
                        "sessions_modeled": 0,
                        "animals": set(),
                        "n_nonconv": 0,
                        "n_sep": 0,
                    },
                )
                a["sessions_in"] += 1
                a["n_in"] += len(group)
                own_bad = _incomplete(group, [response_col, *preds])
                a["n_own"] += int(own_bad.sum())
                a["n_common"] += int((dropped & ~own_bad).sum())

                res = fit_per_stop_rate_model(cohort, preds, response_col=response_col)
                if res is None:
                    continue
                a["n_modeled"] += int(res["n_stops"])
                a["sessions_modeled"] += 1
                a["animals"].add(keys[0])
                a["n_nonconv"] += res.get("converged") is False
                a["n_sep"] += bool(res.get("separated"))
                row = {
                    "outcome": outcome,
                    "model": name,
                    **dict(zip(SESSION_COLUMNS, keys, strict=True)),
                    **res,
                }
                if run_cv:
                    cv = cross_validate_per_stop_rate_model(
                        cohort, preds, response_col=response_col
                    )
                    if cv is not None:
                        row.update(cv)
                rows.append(row)

    records = [
        cohort_record(
            name,
            outcome,
            "stop",
            cohort_mode="common",
            n_in=a["n_in"],
            n_modeled=a["n_modeled"],
            sessions_in=a["sessions_in"],
            sessions_modeled=a["sessions_modeled"],
            animals_modeled=len(a["animals"]),
            drop_notes=(
                "own NaN predictor / immobility<=0 -> own; NaN only in another "
                "rung's predictor (frozen cohort) -> common; session gates -> "
                "excluded"
            ),
            n_drop_own_predictor_missing=a["n_own"],
            n_drop_common_cohort_other=a["n_common"],
            n_excluded_session=(
                a["n_in"] - a["n_own"] - a["n_common"] - a["n_modeled"]
            ),
            n_sessions_flagged_nonconverged=a["n_nonconv"],
            n_sessions_flagged_separated=a["n_sep"],
        )
        for (name, outcome), a in acc.items()
    ]
    return pd.DataFrame(rows), records


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-per-animal",
        type=int,
        default=None,
        help="Smoke mode: keep only the first N sessions per animal.",
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
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parser.parse_args(argv)

    # Causal-hygiene gate: R1 uses the past-only reward-history basis, so the
    # per-stop ladder is pre-outcome throughout; a future leak fails loud.
    for spec in MODEL_PER_STOP_RATE.values():
        assert_predictor_timing(spec.predictors)

    inputs = fit_input_paths()
    stop = read_table(inputs["stop_table"])
    require_covariate_schema(stop, table="stop")
    # Its SWR counts (is_swr_stop, n_swr_events, ...) must come from the
    # configured source, artifact run included.
    require_swr_source(stop, what=str(inputs["stop_table"]))
    # Hash the input now, not at write time, so the sidecar describes the bytes
    # this fit read rather than whatever is on disk when it finishes.
    fit_inputs = record_inputs(inputs)
    missing = [c for c in RESPONSE_COLS.values() if c not in stop.columns]
    if missing:
        raise SystemExit(
            f"stop_table missing outcome columns {missing}; "
            "regenerate it first (scripts/tables/regenerate_stop_table.py)."
        )
    basis_cols = [*REWARD_HISTORY_COLUMNS, REWARD_HISTORY_INCOMPLETE]
    missing_basis = [c for c in basis_cols if c not in stop.columns]
    if missing_basis:
        raise SystemExit(
            f"stop_table missing reward-history basis columns {missing_basis}; "
            "regenerate it first (scripts/tables/regenerate_stop_table.py)."
        )
    # The cohort is applied to the whole stop table BEFORE the smoke subset and
    # before the ladder's common-cohort freeze, so the freeze reports what the
    # fitted population is missing rather than what some other population is.
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
        "Fitting rate hierarchy %s (CV=%s) x %s over %d sessions...",
        list(MODEL_PER_STOP_RATE),
        not args.no_cv,
        list(RESPONSE_COLS),
        stop.groupby(list(SESSION_COLUMNS)).ngroups,
    )
    rate, cohort_audits = fit_per_stop_rate_hierarchy(stop, run_cv=not args.no_cv)

    # Which population every row was fit on, carried on the row itself: a
    # coefficient read out of a concatenated table would otherwise not say, and
    # the model comparison invalidates a cross-cohort step on this column.
    rate["stop_cohort"] = args.stop_cohort

    out_dir = model_dir("per_stop_rate")
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = stop_cohort_suffix(args.stop_cohort, smoke=args.max_per_animal is not None)
    out = out_dir / f"all_models{sfx}.pkl"
    rate.to_pickle(out)
    write_fit_provenance(
        out,
        fit_inputs,
        "poisson_glm",
        [sys.argv[0], *argv],
        extra={"stop_cohort": args.stop_cohort},
    )
    write_cohort_audit(cohort_audits, out_dir / f"all_models{sfx}_cohort_audit.csv")
    logger.info(
        "Saved %d rows (%d sessions, outcomes=%s) to %s",
        len(rate),
        rate.groupby(list(SESSION_COLUMNS)).ngroups,
        sorted(rate["outcome"].unique()),
        out,
    )


if __name__ == "__main__":
    main()
