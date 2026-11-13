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
    uv run python scripts/behavior/rate/fit_per_stop_rate_models.py                    # full
    uv run python scripts/behavior/rate/fit_per_stop_rate_models.py --max-per-animal 1  # smoke
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from continuum_swr_replay.analysis.behavior.cohort_audit import write_cohort_audit
from continuum_swr_replay.analysis.behavior.covariate_timing import (
    assert_predictor_timing,
)
from continuum_swr_replay.analysis.behavior.model_specs import (
    MODEL_PER_STOP_RATE,
    REWARD_HISTORY_COLUMNS,
    REWARD_HISTORY_INCOMPLETE,
)
from continuum_swr_replay.analysis.behavior.rate.per_stop_rate_model import (
    PER_STOP_RATE_OUTCOMES,
    fit_per_stop_rate_hierarchy,
)
from continuum_swr_replay.analysis.events.analysed_time import require_swr_source
from continuum_swr_replay.analysis.events.cohort_selection import (
    STOP_COHORTS,
    select_stop_cohort,
    stop_cohort_suffix,
)
from continuum_swr_replay.paths import model_dir, table_path
from continuum_swr_replay.runs.fit_provenance import (
    record_inputs,
    write_fit_provenance,
)
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
    missing = [c for c in PER_STOP_RATE_OUTCOMES.values() if c not in stop.columns]
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
        list(PER_STOP_RATE_OUTCOMES),
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
