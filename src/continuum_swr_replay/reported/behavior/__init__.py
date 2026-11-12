"""The behavior summary: how the events relate to the task, SWR against non-SWR.

Every value is read from an output a fit already wrote; nothing is refitted
here. This package names and describes the values. The statistics are computed
in ``analysis.behavior`` (``coefficient_statistics``, ``rate_fit_statistics``,
``history_kernel_statistics``, ``stop_statistics``, ``content_statistics`` and
``learning``); here only the size of an input is counted and a statistic an
input already holds is copied. A value read from a fit made on other tables
than the current ones is pending (:func:`~.inputs.fit_staleness`). The inputs
(:class:`~.inputs.BehaviorInputs`), with their runs in
``scripts/workflows/run_refits.py``:

- the per-bin rate ladder, Family B (``models_per_bin_rate/all_models.pkl``,
  ``rate.per_bin.at_well``; ``all_models_allstops.pkl``,
  ``rate.per_bin.all_stops``, the sensitivity cohort);
- the per-stop rate ladder, Family A (``models_per_stop_rate/all_models.pkl``,
  ``rate.per_stop.at_well``);
- the content ``is_swr``-contrast ladders' audited aggregates
  (``models_swr_contrast/swr_contrast_content_audited*.pkl``,
  ``content.swr_contrast.*`` and ``content.breadth_contrast.*``);
- the three-estimator comparison (``content_estimator_comparison/``,
  ``content.estimator_comparison``);
- the leaf-only arm ladder (``models_leaf_M0-M0a-M1-M2-M3-M4-Mv-Mu-Mvu/``,
  ``content.arm_ladder.leaf_arms``);
- the event-source suite's choice block
  (``event_source_comparison/non_local_HPC_comparison_fits.pkl``,
  ``behavior.event_source_suite.non_local``);
- the stay/leave fits (``stage6_is_stay_*.pkl``, ``behavior.stay_leave.*``);
- the SWR-power model (``models_per_event_swr_power/swr_power_model.pkl``,
  ``coupling.ripple_power``);
- the duration-aware history refit (``models_per_bin_rate/duration_aware_history_
  pooled.pkl``, ``rate.duration_aware_history.pooled_at_well``),
  the source of the self-history values, and its ridge cross-validated
  cross-history comparison
  (``models_per_bin_rate/cross_history_ridge_cv/cross_history_ridge_cv.pkl``,
  ``rate.cross_history_ridge_cv.at_well``), the source of the cross values;
- the within-arm horse race (``uncertainty_content/horserace_summary.csv``,
  ``content.uncertainty``);
- the canonical event and stop tables, for event lengths and cohort sizes.

Each aggregate is computed with the rule the producing script or its audit
states, named in the value's description. The content-ladder, leaf-ladder and
choice coefficients aggregated for this summary are read through the shipped
per-term identifiability gate
(:func:`~continuum_swr_replay.analysis.behavior.contrast_aggregation.gate_term_sessions`).
The per-bin rate values pass the same per-term gate: a reward kernel is read
over the sessions whose three reward-basis terms all pass it, and a single
coefficient over the sessions where that term does; the counts left out, and
the kept fits that did not converge, are reported beside them. The reward
effect is the reconstructed kernel; a single basis coefficient is reported only
as itself (:data:`~.rate.BASIS_COEFFICIENT_NOTE`). The time courses drop fits
flagged separated. The
self-history kernel is many sessions' separated term, so its values carry a
pending note and a gated reading beside them. "Animals positive" counts the
animals whose own median is above zero.

The values are made by subject, as ``analysis.behavior`` is divided:
:mod:`.rate` (the per-bin and per-stop rate ladders, the rate ``is_swr``
contrast and the cohort audits), :mod:`.history` (the duration-aware self- and
cross-history kernels), :mod:`.stops` (the stop- and onset-table facts the rate
docs quote), :mod:`.content` (the content contrasts, the estimator comparison,
the leaf ladder, the horse race and the trunk shares), :mod:`.choice` (the
choice block and the stay/leave fits) and :mod:`.coupling` (the SWR-power
model). :mod:`.inputs` holds the files read and the staleness rule, and
:mod:`.common` the tags table and the per-term gate's pending note.
:func:`summarize`, here, reads the inputs and calls each part.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.behavior.coefficient_statistics import fit_rows
from continuum_swr_replay.analysis.behavior.content import content_statistics
from continuum_swr_replay.analysis.events.cohort_selection import select_event_cohort
from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported.behavior.choice import (
    STAY_STRATA,
    choice_values,
    stay_leave_values,
)
from continuum_swr_replay.reported.behavior.common import TAGS
from continuum_swr_replay.reported.behavior.content import (
    CONTENT_SPECS,
    LEAF_RUNGS,
    breadth_derived_values,
    content_contrast_values,
    current_arm_paired_values,
    estimator_values,
    horserace_values,
    leaf_values,
    trunk_values,
)
from continuum_swr_replay.reported.behavior.coupling import power_values
from continuum_swr_replay.reported.behavior.history import duration_aware_history_values
from continuum_swr_replay.reported.behavior.inputs import (
    BehaviorInputs,
    _read_per_bin_fits,
    default_inputs,
    fit_staleness,
)
from continuum_swr_replay.reported.behavior.rate import (
    RATE_POPULATIONS,
    cohort_audit_values,
    per_bin_detail_values,
    per_bin_values,
    per_stop_values,
    rate_contrast_audit_values,
    rate_contrast_separation_values,
)
from continuum_swr_replay.reported.behavior.stops import stop_table_values
from continuum_swr_replay.reported.values import (
    Summary,
    file_record,
    pend,
    rel,
    value_factory,
)
from continuum_swr_replay.tables.session_keys import n_sessions

__all__ = ["default_inputs", "summarize"]


def summarize(inputs: BehaviorInputs, produced_by: str) -> Summary:
    """The behavior summary from ``inputs``."""
    summary = Summary(name="behavior", produced_by=produced_by)

    def record(*paths: Path) -> None:
        summary.inputs.extend(file_record(p, ROOT_DIR) for p in paths)

    rate_fits = (
        inputs.per_bin,
        inputs.per_bin_all_stops,
        inputs.per_bin_in_transit,
        inputs.rate_contrast,
        inputs.per_stop,
    )
    contrast_audited = inputs.rate_contrast.with_name("swr_contrast_rate_audited.pkl")
    summary.add(
        *pend(
            rate_contrast_audit_values(
                pd.read_pickle(contrast_audited),
                f"{rel(contrast_audited)} (rate.swr_contrast.at_well)",
            ),
            fit_staleness(contrast_audited),
        )
    )
    record(contrast_audited)
    summary.add(
        *pend(
            rate_contrast_separation_values(
                _read_per_bin_fits(inputs.rate_contrast),
                f"{rel(inputs.rate_contrast)} (rate.swr_contrast.at_well)",
            ),
            fit_staleness(inputs.rate_contrast),
        )
    )
    in_transit = _read_per_bin_fits(inputs.per_bin_in_transit)
    in_transit_stale = fit_staleness(inputs.per_bin_in_transit)
    make_transit = value_factory(
        summarize,
        f"{rel(inputs.per_bin_in_transit)} (rate.per_bin.in_transit)",
        TAGS["rate"],
        in_transit_stale,
    )
    summary.add(
        make_transit(
            "rate_intransit_sessions",
            n_sessions(in_transit),
            "count",
            "count",
            "sessions with any per-bin fit on the in-transit stops (the sample gates "
            "leave the rest out)",
        ),
        *(
            make_transit(
                f"rate_intransit_sessions_{word}",
                n_sessions(fit_rows(in_transit, outcome=outcome)),
                "count",
                "count",
                f"{label}: sessions with a per-bin fit on the in-transit stops",
            )
            for word, (outcome, label) in RATE_POPULATIONS.items()
        ),
    )
    summary.add(*cohort_audit_values(inputs))
    record(inputs.per_bin_audit, inputs.per_stop_audit)
    per_bin = _read_per_bin_fits(inputs.per_bin)
    per_bin_all = _read_per_bin_fits(inputs.per_bin_all_stops)
    per_bin_stale = fit_staleness(inputs.per_bin, inputs.per_bin_all_stops)
    summary.add(
        *pend(
            per_bin_values(
                per_bin,
                per_bin_all,
                f"{rel(inputs.per_bin)} (rate.per_bin.at_well) and {rel(inputs.per_bin_all_stops)} (rate.per_bin.all_stops)",
            ),
            per_bin_stale,
        )
    )
    # pending on all three fits: the notes join as fit_staleness of the three would
    summary.add(
        *pend(
            per_bin_detail_values(
                per_bin,
                per_bin_all,
                in_transit,
                f"{rel(inputs.per_bin)}, {rel(inputs.per_bin_all_stops)} and "
                f"{rel(inputs.per_bin_in_transit)} (rate.per_bin.*)",
            ),
            per_bin_stale,
            in_transit_stale,
        )
    )
    summary.add(
        *pend(
            per_stop_values(
                pd.read_pickle(inputs.per_stop),
                f"{rel(inputs.per_stop)} (rate.per_stop.at_well)",
            ),
            fit_staleness(inputs.per_stop),
        )
    )
    record(*rate_fits)

    summary.add(
        *pend(
            duration_aware_history_values(
                pd.read_pickle(inputs.cross_ridge_cv),
                pd.read_pickle(inputs.duration_aware_pooled),
                f"{rel(inputs.duration_aware_pooled)} "
                "(rate.duration_aware_history.pooled_at_well) and "
                f"{rel(inputs.cross_ridge_cv)} (rate.cross_history_ridge_cv.at_well)",
            ),
            fit_staleness(inputs.duration_aware_pooled, inputs.cross_ridge_cv),
        )
    )
    record(inputs.duration_aware_pooled, inputs.cross_ridge_cv)

    audited, sources = {}, {}
    for suffix in sorted({spec[0] for spec in CONTENT_SPECS.values()}):
        path = inputs.content_contrast_dir / f"swr_contrast_content_audited{suffix}.pkl"
        audited[suffix] = pd.read_pickle(path)
        sources[suffix] = f"{rel(path)} (audited aggregate)"
        record(path)
    content_stale = fit_staleness(
        *(
            inputs.content_contrast_dir / f"swr_contrast_content_audited{suffix}.pkl"
            for suffix in audited
        )
    )
    content = content_contrast_values(audited, sources)
    summary.add(*pend(content, content_stale))
    events = pd.read_parquet(
        inputs.event_table, columns=["animal", "is_swr", "n_nonlocal_bins"]
    )
    summary.add(
        *pend(
            breadth_derived_values(
                {v.key: v for v in content},
                events,
                f"{sources['_dm']}, {sources['_dm_len']} and {rel(inputs.event_table)}",
            ),
            content_stale,
        )
    )
    record(inputs.event_table)

    summary.add(
        *pend(
            estimator_values(
                pd.read_csv(inputs.estimator_summary),
                pd.read_csv(inputs.estimator_bootstrap),
                f"{rel(inputs.estimator_summary.parent)} (content.estimator_comparison)",
            ),
            fit_staleness(
                inputs.estimator_summary.with_name(
                    "content_estimator_comparison.parquet"
                )
            ),
        )
    )
    record(inputs.estimator_summary, inputs.estimator_bootstrap)

    leaf_fits = [inputs.leaf_dir / f"{n}.pkl" for n in LEAF_RUNGS]
    summary.add(
        *pend(
            leaf_values(
                inputs.leaf_dir,
                f"{rel(inputs.leaf_dir)} (content.arm_ladder.leaf_arms)",
            ),
            fit_staleness(*leaf_fits),
        )
    )
    record(*leaf_fits)

    summary.add(
        *pend(
            horserace_values(
                pd.read_csv(inputs.horserace_summary),
                f"{rel(inputs.horserace_summary)} (content.uncertainty)",
            ),
            fit_staleness(inputs.horserace_summary),
        )
    )
    record(inputs.horserace_summary)

    summary.add(
        *pend(
            choice_values(
                pd.read_pickle(inputs.choice_fits),
                f"{rel(inputs.choice_fits)} (behavior.event_source_suite.non_local)",
            ),
            fit_staleness(inputs.choice_fits),
        )
    )
    record(inputs.choice_fits)

    bundles = {}
    stay_leave_fits = []
    for stratum, sfx in STAY_STRATA.items():
        logistic = inputs.stay_leave_dir / f"stage6_is_stay_logistic{sfx}.pkl"
        pooled = inputs.stay_leave_dir / f"stage6_is_stay_pooled_gee{sfx}.pkl"
        stay_leave_fits += [logistic, pooled]
        bundles[stratum] = (pd.read_pickle(logistic), pd.read_pickle(pooled))
        record(logistic, pooled)
    stops = pd.read_parquet(inputs.stop_table)
    summary.add(
        *pend(
            stay_leave_values(
                bundles,
                stops,
                f"{rel(inputs.stay_leave_dir)}/stage6_is_stay_*.pkl (behavior.stay_leave.at_well*) and {rel(inputs.stop_table)}",
            ),
            fit_staleness(*stay_leave_fits),
        )
    )
    record(inputs.stop_table)
    onsets = pd.read_parquet(
        inputs.onset_table, columns=["event_type", "t_since_stop_onset", "stop_id"]
    )
    summary.add(
        *stop_table_values(
            stops, onsets, f"{rel(inputs.stop_table)} and {rel(inputs.onset_table)}"
        )
    )
    record(inputs.onset_table)
    arm, _ = select_event_cohort(
        pd.read_parquet(
            inputs.event_arm_table,
            columns=[
                *content_statistics.TRUNK_COLUMNS,
                content_statistics.TRUNK_RESPONSE,
                "occupancy_state",
            ],
        ),
        cohort="at_well",
    )
    summary.add(*trunk_values(arm, f"{rel(inputs.event_arm_table)} (at-well events)"))
    record(inputs.event_arm_table)
    split_fits = inputs.content_contrast_dir / "swr_contrast_content_arm.pkl"
    summary.add(
        *pend(
            current_arm_paired_values(
                pd.read_pickle(split_fits), f"{rel(split_fits)} (per-session fits)"
            ),
            fit_staleness(split_fits),
        )
    )
    record(split_fits)

    summary.add(
        *pend(
            power_values(
                pd.read_pickle(inputs.swr_power),
                f"{rel(inputs.swr_power)} (coupling.ripple_power)",
            ),
            fit_staleness(inputs.swr_power),
        )
    )
    record(inputs.swr_power)
    return summary
