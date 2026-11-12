#!/usr/bin/env python
"""Bivariate per-bin conditional intensity: does one non-local stream's history
drive the other's rate?

Family B (per-bin point process) extended to a bivariate coupling. For each
session we fit the non-local-**no-SWR** onset rate per (stop x bin) with a
within-stop time spline + reward kernel + its own self-history, and ask whether
adding the **with-SWR** non-local stream's recent history (a cross-history block)
improves the fit -- i.e. is a non-SWR non-local event made more/less likely by a
recent with-SWR non-local event? The nested comparison (self vs self+cross)
isolates the cross contribution; the cross-history coefficients give its sign and
timescale. The reverse direction (with-SWR rate ~ ... + non-SWR cross history) is
fit too, so the coupling is reported both ways.

Per session, per direction (specs: ``model_specs.MODEL_CROSS_HISTORY_RATE``):
  Mself      = bs(tc) + reward kernel + self-history (raised cosine)
  Mselfcross = Mself + cross-history (raised cosine)
We report the cross-history coefficients, the self->cross AIC difference
(``delta_aic`` = AIC(Mselfcross) - AIC(Mself); negative = cross history helps),
and the cross cv-bits/event gain; then aggregate across sessions (median, %
improved, per-animal sign consistency, Benjamini-Hochberg q of the signed-rank p).
The two rungs are frozen on their own columns, so a session-direction whose two
cohorts differ in bins, event bins or stops
(``fit_comparability.why_not_comparable``) is not fit. The fit loop is
``analysis/behavior/rate/cross_history.py``; this script reads the tables,
prints the summary and writes.

Usage: python scripts/behavior/rate/fit_cross_history_rate.py [--max-per-animal N] [--write]
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

import pandas as pd

from continuum_swr_replay.analysis.behavior import (
    attach_is_swr_to_onsets,
)
from continuum_swr_replay.analysis.behavior.coefficient_statistics import (
    coefficient_family_statistics,
    coefficient_statistics,
)
from continuum_swr_replay.analysis.behavior.rate.cross_history import (
    CROSS_COEFS,
    fit_cross_history,
)
from continuum_swr_replay.analysis.events.analysed_time import (
    require_swr_source,
)
from continuum_swr_replay.analysis.events.swr_source import (
    swr_source_name,
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


def _summary(models: pd.DataFrame) -> None:
    """Print each direction's summary over sessions.

    The AIC and cv-bits differences are ``coefficient_statistics``' (the
    sessions' median and signs, and the signed-rank p, NaN below six non-zero
    differences); the cross-history coefficients are
    ``coefficient_family_statistics``', their q values Benjamini-Hochberg over
    the basis terms some session has.
    """
    for direction, g in models.groupby("direction"):
        print(f"\n=== {direction}  (n={len(g)} sessions) ===")
        aic = coefficient_statistics(g, "delta_aic")
        print(
            f"  self->+cross AIC difference (+cross minus self; negative = "
            f"cross helps): median {aic['median']:+.2f}  "
            f"%improved {100 * aic['negative_fraction']:.0f}%  "
            f"p={aic['signed_rank_p']:.1e}"
        )
        if "delta_cv_bits" in g:
            cv = coefficient_statistics(g, "delta_cv_bits")
            print(
                f"  cross cv-bits/event gain: median {cv['median']:+.4f}  "
                f"%+ {100 * cv['positive_fraction']:.0f}%  (n={cv['n_sessions']})"
            )
        # cross-history coefficients (FDR across the basis)
        family = coefficient_family_statistics(
            g, [c.removeprefix("coef_") for c in CROSS_COEFS]
        )
        family = family[family["n_sessions"] > 0] if len(family) else family
        if len(family):
            print("  cross-history basis coefficients (early->late lag):")
            # itertuples keeps each column's dtype: the animal counts print as ints
            for row in family.itertuples():
                animals = f"{row.n_animals_positive}/{row.n_animals}"
                print(
                    f"    {row.Index:<18}{row.median:>+8.3f}"
                    f"{row.positive_fraction * 100:>6.0f}%{animals:>7}"
                    f"  q={row.signed_rank_p_adjusted:.1e}"
                )


def main(argv: Sequence[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--max-per-animal",
        type=int,
        default=None,
        help="smoke: cap sessions per animal",
    )
    p.add_argument("--write", action="store_true")
    args = p.parse_args(argv)

    stop_path = table_path("stop_table")
    onset_path = table_path("event_onset_table")
    event_path = table_path("event_table")
    stop = read_table(stop_path)
    onset_table = read_table(onset_path)
    event_table = read_table(event_path)
    require_covariate_schema(stop, table="stop")
    require_covariate_schema(onset_table, table="event_onset")
    require_covariate_schema(event_table, table="event")
    # The bins are dropped by the configured source's excluded time, so the
    # tables must come from that source (its artifact run included).
    require_swr_source(stop, what=str(stop_path))
    require_swr_source(onset_table, what=str(onset_path))
    require_swr_source(event_table, what=str(event_path))
    onsets = attach_is_swr_to_onsets(onset_table, event_table)
    if args.max_per_animal is not None:
        stop = subset_max_per_animal(stop, args.max_per_animal)
        logger.info(
            "smoke: %d sessions", len(stop[list(SESSION_COLUMNS)].drop_duplicates())
        )

    models = fit_cross_history(stop, onsets)
    _summary(models)
    if args.write:
        out = model_dir("per_bin_rate") / "cross_history_models.pkl"
        models.attrs["swr_source"] = swr_source_name()
        models.to_pickle(out)
        print(f"\nwrote {len(models)} rows -> {out}")


if __name__ == "__main__":
    main()
