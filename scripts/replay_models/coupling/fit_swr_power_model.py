#!/usr/bin/env python
"""Per-event SWR-coupling strength (ripple power) vs behavioral covariates.

For each non-local event, the SWR-coupling strength is summarized by
``ripple_power_z`` (the event-mean of the z-scored consensus ripple power -- the
same signal the Kay detector thresholds, so ``is_swr`` is a thresholded version of
this continuous quantity). This script asks, across non-local events:

  1. Distribution shape of the SWR power -- bimodal (categorical SWR vs non-SWR)
     or unimodal (the SWR/non-SWR split is a cut on a continuum)?
  2. Duration confound -- is the (event-mean) power correlated with event duration?
  3. Power model -- does reward/value/exploration predict the SWR power of an
     event (power ~ covariates), i.e. acting on SWR-COUPLING strength rather than
     on whether an event occurs (occurrence is the separate per-bin rate model)?

The model contains the current trial's RPE, so its fit cohort is restricted to
events beginning at or after ``outcome_known_time`` (reward onset for rewarded
trials; latest possible pump latency, capped at the collapsed task visit's final
departure, for omissions).
Distribution and duration
descriptives remain summaries of all non-local events.

Scope / caveat: this fits only the *conditional power-vs-covariates* relationship
by per-event OLS pooled across sessions. It is NOT a full marked point process --
it does not jointly model the event intensity, does not test whether the power
depends on event history (the conditional-independence assumption), and has no
shared-latent coupling between occurrence and power. Those are deferred to a joint
treatment. ``ripple_power_z`` here is the event-mean; a peri-onset peak (the
detector's actual criterion) is the better summary and is left as a follow-up.

Output: ``models_per_event_swr_power/swr_power_model.pkl`` (per-session
coefficients), its z reference, two audits and its ``.provenance.json``
sidecar (the hashes of the three tables it read) beside it.

Usage::

    uv run python scripts/replay_models/coupling/fit_swr_power_model.py                    # full
    uv run python scripts/replay_models/coupling/fit_swr_power_model.py --max-per-animal 1 # smoke
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_properties.class_or_continuum import (
    SARLE_CUTOFF,
)
from continuum_swr_replay.analysis.events.analysed_time import require_swr_source
from continuum_swr_replay.analysis.events.swr_source import swr_source_name
from continuum_swr_replay.analysis.replay_models.coefficient_statistics import (
    coefficient_statistics,
)
from continuum_swr_replay.analysis.replay_models.cohort_audit import write_cohort_audit
from continuum_swr_replay.analysis.replay_models.coupling.swr_power import (
    COHORT_EXTRA_COLUMNS,
    POWER_COL,
    POWER_PERCENTILES,
    PREDICTORS,
    assemble_power_frame,
    duration_power_correlation,
    fit_power_model,
    power_cohort_records,
    power_distribution_statistics,
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
from continuum_swr_replay.tables.session_keys import subset_max_per_animal
from continuum_swr_replay.tables.standardization import (
    Reference,
    save_reference,
    z_reference_path,
)


def _assemble() -> pd.DataFrame:
    ev = read_table(table_path("event_table"))
    stop = read_table(table_path("stop_table"))
    # qvals_table (exact stop-start alignment). These are the behavioral
    # model's experienced value and RPE, replacing retired nominal/leaky-Q
    # stand-ins. Keep their canonical names through fitting and persistence.
    cont = read_table(table_path("qvals_table"))
    require_covariate_schema(ev, table="event")
    require_covariate_schema(stop, table="stop")
    require_covariate_schema(cont, table="qvals")
    # is_swr and ripple_power_z must come from the configured SWR source
    require_swr_source(ev, what=str(table_path("event_table")))
    require_swr_source(stop, what=str(table_path("stop_table")))
    return assemble_power_frame(ev, stop, cont)


def _write_outputs(
    out_dir: Path,
    *,
    coefs: pd.DataFrame,
    z_reference: dict[str, Reference],
    gate_audit: pd.DataFrame,
    cohort_records: list[dict],
    smoke: bool,
) -> Path:
    """Write the fitted coefficients, their z reference, and the two audits.

    The per-session standardization moments go beside the model as
    ``<stem>.z_reference_by_session.json``, so a standardized coefficient can be
    converted back to raw units without refitting. Returns the model path.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = "_smoke" if smoke else ""
    out = out_dir / f"swr_power_model{sfx}.pkl"
    coefs.to_pickle(out)
    save_reference(z_reference, z_reference_path(out, by_session=True))
    gate_audit.to_csv(
        out_dir / f"swr_power_post_outcome_gate_audit{sfx}.csv",
        index=False,
    )
    write_cohort_audit(
        cohort_records,
        out_dir / f"swr_power_cohort_audit{sfx}.csv",
        extra_columns=COHORT_EXTRA_COLUMNS,
    )
    return out


def build_parser() -> argparse.ArgumentParser:
    """The command line: smoke mode and the output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--max-per-animal",
        type=int,
        default=None,
        help=(
            "Smoke mode: keep only the first N sessions per animal and write "
            "`_smoke`-suffixed artifacts, never the canonical ones."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=model_dir("per_event_swr_power"),
        help="Directory for the fitted model, its z reference, and the audits.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    # `--max-per-animal 0` is still smoke mode, so test the flag, not its value.
    smoke = args.max_per_animal is not None

    df = _assemble()
    fit_inputs = record_inputs(
        {
            name: table_path(name)
            for name in ("event_table", "stop_table", "qvals_table")
        }
    )
    if smoke:
        df = subset_max_per_animal(df, args.max_per_animal, sort_sessions=True)
    m = df[POWER_COL].to_numpy()
    distribution = power_distribution_statistics(df[POWER_COL], df["is_swr"])
    print(
        f"=== SWR power distribution ({POWER_COL}, "
        f"n={distribution['n_events']} non-local events) ==="
    )
    print(
        f"  pctiles {'/'.join(map(str, POWER_PERCENTILES))}:",
        np.round(distribution["percentiles"], 2),
    )
    print(
        f"  Sarle bimodality coef = {distribution['sarle_bimodality']:.3f}"
        f"  (>{SARLE_CUTOFF:.3f} => bimodal)"
    )
    for lab, key in [("non-SWR events", "non_swr"), ("SWR events", "swr")]:
        s = distribution[key]
        print(f"  {lab:16s} n={s['n']:6d}  mean={s['mean']:+.2f}  sd={s['sd']:.2f}")
    lo, hi = np.percentile(m, [1, 99])
    h, edges = np.histogram(m, bins=np.linspace(lo, hi, 21))
    print("  histogram (1-99 pctile):")
    for c, cnt in zip((edges[:-1] + edges[1:]) / 2, h, strict=True):
        print(f"    {c:+5.1f} | {'#' * int(40 * cnt / h.max())}")

    duration = duration_power_correlation(df["event_duration_s"], df[POWER_COL])
    print(
        f"\n=== duration confound: corr(power, event_duration_s) ="
        f" {duration['corr']:+.3f}"
        f"  (log-dur {duration['log_duration_corr']:+.3f}) ==="
    )

    coefs = fit_power_model(df)
    gate_audit = pd.DataFrame(coefs.attrs["post_outcome_gate_audit"])
    out = _write_outputs(
        args.output_dir,
        coefs=coefs,
        z_reference=coefs.attrs["z_reference"],
        gate_audit=gate_audit,
        cohort_records=power_cohort_records(df, coefs),
        smoke=smoke,
    )
    write_fit_provenance(
        out,
        fit_inputs,
        "ols",
        [sys.argv[0], *(sys.argv[1:] if argv is None else argv)],
        extra={"swr_source": swr_source_name()},
    )
    if coefs.empty:
        raise RuntimeError(
            "No SWR-power session retained both rewarded and omission events "
            "after the physical post-outcome and complete-case gates; audits "
            "were written, but a one-outcome survivor model was not fit."
        )
    print(f"\n=== power model: {POWER_COL} ~ standardized covariates (per session) ===")
    print(
        f"  {len(coefs)} sessions, {coefs.animal.nunique()} animals; saved -> {out}\n"
    )
    print(f"  {'covariate':<22}{'med coef':>10}{'%pos':>7}{'animals':>9}{'p':>10}")
    for c in PREDICTORS:
        stats = coefficient_statistics(coefs, c)
        animals = f"{stats['n_animals_positive']}/{stats['n_animals']}"
        print(
            f"  {c:<22}{stats['median']:>+10.3f}"
            f"{stats['positive_fraction'] * 100:>6.0f}%"
            f"{animals:>9}{stats['signed_rank_p']:>10.1e}"
        )


if __name__ == "__main__":
    main()
