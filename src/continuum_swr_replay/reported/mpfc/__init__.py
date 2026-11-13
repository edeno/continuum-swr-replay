"""The mpfc summary: how mPFC units respond to SWR and non-SWR non-local events.

Every value is read from an output an analysis already wrote; nothing is
refitted here. This package names and describes the values. The statistics are
computed in ``analysis.pfc.population_statistics`` (the unit population) and
``analysis.pfc.timing`` (when the responses peak and start), with the smallest
empirical p-value in ``analysis.pfc.heldout``; here only the size of an input
(its bundles, units, animals) is counted, and a value that a provenance sidecar,
the timing calibration, the marked-family summary or the stop-context fits
already hold is copied out. The inputs (:class:`~.inputs.MpfcInputs`):

- the at-well mPFC unit bundles (``Processed-Data/{epoch}_mPFC_modulation.pkl``
  and their ``*.provenance.json`` sidecars, from
  ``scripts/pfc/run_pfc_analysis_bandit.py``), read through Figure 4's loader
  (:func:`~continuum_swr_replay.analysis.pfc.load_population.load_population_data`,
  no firing-rate cutoff: every unit with a usable fit) for every per-unit value,
  with a 1 Hz cutoff as a check;
- the PFC cohort record the runner writes (``pfc_cohort_mPFC.json``);
- the canonical event and stop tables, only to check that the bundles and the
  follow-up fits below were made from the current ones;
- the timing calibration (``pfc_response_followups/timing_calibration.json``,
  from ``scripts/pfc/calibrate_pfc_kernel_timing.py``);
- the extended-event-set bundles with SWRs without replay
  (``pfc_response_followups/swr_without_replay``, from
  ``scripts/pfc/run_pfc_analysis_bandit.py --event-set with_swr_without_replay``);
- the marked-family summary of the follow-up models and the bundle sidecars
  beside it (``pfc_response_followups/marked_family_well_unknown``, from
  ``scripts/pfc/run_pfc_marked_model_family.py`` and
  ``scripts/pfc/summarize_pfc_marked_family.py``), pending when a sidecar
  records other tables (:func:`~.followups.marked_family_staleness`);
- the stop-context fits (``pfc_response_followups/stop_context``, from
  ``scripts/pfc/run_pfc_stop_context.py``), pending when a bundle, cached
  session table or table they read has changed
  (:func:`~.followups.stop_context_staleness`).

Figure 4 (``scripts/figures/generate_figure04.py``) draws the unit fractions
with the same functions. ``scripts/pfc/compare_pfc_cohorts.py`` calls them for
the called fractions and ``scripts/pfc/measure_pfc_null_geometry.py`` for the
units called without improvement; the cohort, fit-count, prior-binding and
other null-geometry numbers those two reports print are computed in the
scripts. Unit fractions are descriptive and pooled over sessions and animals;
they are not population inference.

Every value answers question 4.1, so the values are made by what they describe
rather than by question: :mod:`.population` (the cohort, the estimator and its
fits, the called fractions of Figure 4d, the null's geometry, the overlap and
size of the responses), :mod:`.timing` (when the responses peak and start),
:mod:`.swr_without_replay` (the extended event set), :mod:`.followups` (the
marked-family models and the stop-context fits, with their staleness rules) and
:mod:`.rate_cutoff` (the 1 Hz check). :mod:`.inputs` holds the files read and
the bundles' staleness rule, and :mod:`.common` the tags table and the words
the keys share. :func:`summarize`, here, reads the inputs and calls each part.
"""

from __future__ import annotations

import json

from continuum_swr_replay.analysis.pfc.load_population import (
    MIN_FIRING_RATE_HZ,
    load_population_data,
)
from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported.mpfc.common import TAGS
from continuum_swr_replay.reported.mpfc.followups import followup_values
from continuum_swr_replay.reported.mpfc.inputs import (
    STEM,
    MpfcInputs,
    default_inputs,
    read_sidecars,
    staleness,
)
from continuum_swr_replay.reported.mpfc.population import (
    called_fraction_values,
    cohort_values,
    estimator_values,
    null_geometry_values,
    overlap_and_effect_values,
    prior_binding_values,
)
from continuum_swr_replay.reported.mpfc.rate_cutoff import rate_cutoff_check_values
from continuum_swr_replay.reported.mpfc.swr_without_replay import (
    SWR_WITHOUT_REPLAY_CURVES,
    swr_without_replay_values,
)
from continuum_swr_replay.reported.mpfc.timing import timing_values
from continuum_swr_replay.reported.values import (
    Summary,
    file_record,
    files_record,
    rel,
    value_factory,
)
from continuum_swr_replay.runs.staleness import table_hashes

__all__ = ["default_inputs", "summarize"]


def summarize(inputs: MpfcInputs, produced_by: str) -> Summary:
    """The mpfc summary from ``inputs``."""
    summary = Summary(name="mpfc", produced_by=produced_by)
    sidecars = read_sidecars(inputs.bundle_dir)
    current = table_hashes(inputs.event_table, inputs.stop_table)
    pending = staleness(sidecars, current)
    population, curves, lag_time = load_population_data(
        inputs.bundle_dir, min_firing_rate=MIN_FIRING_RATE_HZ
    )
    cohort = json.loads(inputs.pfc_cohort.read_text())
    source = (
        f"{rel(inputs.bundle_dir)}/*{STEM}.pkl and their provenance sidecars "
        f"(scripts/pfc/run_pfc_analysis_bandit.py), {len(sidecars)} bundles, "
        "read by Figure 4's loader (analysis/pfc/load_population.py, no "
        "firing-rate cutoff)"
    )
    main = TAGS["modulation"]
    summary.add(
        *cohort_values(
            sidecars,
            population,
            cohort,
            value_factory(cohort_values, source, main, pending),
        ),
        *estimator_values(
            sidecars, value_factory(estimator_values, source, main, pending)
        ),
        *prior_binding_values(
            population, value_factory(prior_binding_values, source, main, pending)
        ),
        *called_fraction_values(
            population,
            value_factory(called_fraction_values, source, TAGS["figure_4d"], pending),
        ),
        *null_geometry_values(
            population,
            sidecars,
            value_factory(null_geometry_values, source, main, pending),
        ),
        *overlap_and_effect_values(
            population, value_factory(overlap_and_effect_values, source, main, pending)
        ),
    )
    calibration = (
        json.loads(inputs.timing_calibration.read_text())
        if inputs.timing_calibration.exists()
        else None
    )
    summary.add(
        *timing_values(
            population,
            curves,
            lag_time,
            calibration,
            value_factory(timing_values, source, main, pending),
        )
    )
    values, records = followup_values(inputs, current, cohort["n_pfc_cohort"])
    summary.add(*values)
    summary.inputs += records
    extended_dir = inputs.swr_without_replay_dir
    extended_bundles = sorted(extended_dir.glob(f"*{STEM}.pkl"))
    extended_check = None
    if extended_bundles:
        extended_sidecars = read_sidecars(extended_dir)
        incomplete = len(extended_sidecars) < len(sidecars)
        extended_pending = staleness(extended_sidecars, current) or (
            f"the extended-event-set batch holds {len(extended_sidecars)} of "
            f"{len(sidecars)} sessions; refresh when it completes"
            if incomplete
            else None
        )
        extended_population, extended_curves, _ = load_population_data(
            extended_dir,
            min_firing_rate=MIN_FIRING_RATE_HZ,
            extra_curves=SWR_WITHOUT_REPLAY_CURVES,
        )
        extended_check = (
            extended_population,
            extended_curves,
            lag_time,
            extended_sidecars,
        )
        summary.add(
            *swr_without_replay_values(
                extended_population,
                extended_curves,
                lag_time,
                extended_sidecars,
                value_factory(
                    swr_without_replay_values,
                    f"{rel(extended_dir)}/*{STEM}.pkl and sidecars "
                    "(scripts/pfc/run_pfc_analysis_bandit.py --event-set "
                    f"with_swr_without_replay), {len(extended_sidecars)} bundles, "
                    "Figure 4's loader (no firing-rate cutoff)",
                    TAGS["modulation"],
                    extended_pending,
                ),
            )
        )
        summary.inputs += [
            files_record(f"*{STEM}.pkl", extended_bundles, ROOT_DIR),
            files_record(
                f"*{STEM}.*.npy",
                extended_dir.glob(f"*{STEM}.*.npy"),
                ROOT_DIR,
            ),
        ]
    summary.add(
        *rate_cutoff_check_values(
            population,
            extended_check,
            value_factory(rate_cutoff_check_values, source, main, pending),
        )
    )

    data = inputs.bundle_dir
    summary.inputs += [
        files_record(f"*{STEM}.pkl", data.glob(f"*{STEM}.pkl"), ROOT_DIR),
        files_record(f"*{STEM}.*.npy", data.glob(f"*{STEM}.*.npy"), ROOT_DIR),
        files_record(
            f"*{STEM}.provenance.json", data.glob(f"*{STEM}.provenance.json"), ROOT_DIR
        ),
        file_record(inputs.pfc_cohort, ROOT_DIR),
        file_record(inputs.event_table, ROOT_DIR),
        file_record(inputs.stop_table, ROOT_DIR),
        *(
            [file_record(inputs.timing_calibration, ROOT_DIR)]
            if inputs.timing_calibration.exists()
            else []
        ),
    ]
    return summary
