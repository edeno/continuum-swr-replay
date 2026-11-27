"""The events summary: what the events are, and how they compare with SWRs and Widloski events.

Every value is read from an output an analysis already wrote; nothing is
refitted here. This package names and describes the values. The statistics are
computed in ``analysis.event_properties.event_statistics``,
``analysis.decode_quality.quality_metrics`` and
``analysis.detector_comparison.nonlocal_comparison``; here only the size of an
input (its epochs, its animals) is counted. The inputs (:class:`~continuum_swr_replay.reported.inputs.EventsInputs`):

- the canonical event table (``replay_content/event_table.parquet``): the
  immobile non-local events, their SWR-overlap class, timing against the
  matched SWR, and their per-event properties;
- an event features run (``replay_detection_comparison/event_features/<run>``):
  every compared event of the other detectors, whether it overlaps a non-local
  event (``pair_membership`` against ``state_space``), and its state shares
  (``*_state_shares.parquet``, from ``scripts/detector_comparison/add_state_shares.py``);
- the spike-evidence run that scores every event
  (``Processed-Data/event_spike_evidence``, one folder per epoch, from
  ``scripts/event_evidence/run_event_spike_evidence.sh``), read over the
  selected events (each scored event matched to its ``event_id``,
  ``spike_evidence.with_event_ids``) and the quiet windows that start inside a
  well visit (``spike_evidence.in_analyzed_subset``);
- the held-out tetrode test split by SWR (``nl_bouts_by_swr_stats.csv`` beside
  the state-evidence run the detection summary reads, from
  ``scripts/event_evidence/split_state_evidence_by_swr.py``), over the bouts
  that start inside a well visit, which its provenance records;
- each event time bin's content placed on the track, with its track distance to
  the animal and to the nearest well (``Processed-Data/content_along_arms``, from
  ``scripts/event_properties/measure_content_along_arms.py``);
- the one-class-or-two tests (``Processed-Data/ripple_class_structure``, from
  ``scripts/event_properties/measure_ripple_class_structure.py``);
- each event's start distance and duration-matched quiet-rest theta
  (``Processed-Data/event_start_rest_theta``, from
  ``scripts/event_properties/measure_event_start_and_rest_theta.py``);
- the nested bootstrap of the property contrasts
  (``Processed-Data/event_property_bootstrap``, from
  ``scripts/event_properties/bootstrap_event_properties.py``);
- the brain-state reference (``Processed-Data/brain_state_reference``, from
  ``scripts/event_properties/measure_brain_state_reference.py``) and the gamma pass of
  ``scripts/event_properties/measure_event_gamma.py`` (``Processed-Data/event_gamma``):
  the z-scored gamma per event and epoch, and the band power's change against
  quiet rest in dB (its ``reference.parquet``);
- each epoch's immobile time bins by ripple power, by MUA and by non-local
  probability (``Processed-Data/nonlocal_by_signal``, from
  ``scripts/event_properties/measure_nonlocal_by_signal.py``);
- the held-out state calibration over the immobile time inside the well visits
  (``Processed-Data/state_calibration_in_visits``, from
  ``scripts/event_evidence/summarize_state_calibration_in_visits.py``);
- the event rule's sensitivity check (``Processed-Data/event_rule_sweep``, from
  ``scripts/event_properties/sweep_event_rule.py``);
- the decoder quality of each analyzed epoch (``Processed-Data/decoder_quality``,
  from ``scripts/decoding/measure_decoder_quality.py``): the Local-state
  decoding error, the state fractions moving and immobile, and the calibration
  curve, summarized by ``analysis.decode_quality.quality_metrics``;
- the non-local restriction run (``Processed-Data/event_nonlocal_restriction``,
  from ``scripts/event_evidence/score_nonlocal_restriction.py``): each epoch's
  per-bin scores of the local-only, SWR-only and full decoders, each fold's bins
  under the whole decoder fit on the other folds (a run of the earlier design,
  which held out only the encodings, is refused, :func:`~continuum_swr_replay.reported.inputs.require_fold_fit`),
  summed over the windows of the canonical events and their Local-rest
  controls (``Processed-Data/event_nonlocal_restriction_windows``, from
  ``scripts/event_evidence/sum_nonlocal_restriction_windows.py``), and over the
  events detected by the canonical rule from the run's held-out state
  probabilities, with which held-out and canonical events share a bin and the
  bins summed by category, well visit and held-out event window
  (``Processed-Data/event_nonlocal_restriction_heldout_events``, from
  ``scripts/event_evidence/detect_heldout_events.py``, which Figure 2k draws);
  each bin's spikes and well-visit flag from ``Processed-Data/event_bin_facts``
  (``scripts/event_evidence/measure_bin_facts.py``); summarized by
  ``analysis.event_evidence.restriction_windows``,
  ``analysis.event_evidence.restriction_totals`` and
  ``analysis.event_evidence.heldout_events`` on the canonical events that start
  inside a well visit, the held-out events that start inside one
  (:data:`~.common.HELDOUT_EVENTS_SUBSET`) and the time inside a well visit.

Every value is over the analyses' one subset: the events that start inside a
well visit (``cohort_selection.select_visit_context``, applied once, where the
event table is read; a stored per-event output is read over the same events,
``cohort_selection.select_events_in``), and time, SWRs and the other detectors'
events inside a well visit. Each value's description says which
(:data:`~.common.EVENTS_SUBSET`, :data:`~.common.TIME_SUBSET`,
:data:`~.common.DETECTOR_SUBSET`), and the
summary counts the events of every well-visit label, kept and left out
(:func:`~.cohort.visit_context_values`). An output re-run over the subset records it
(``cohort_selection.ANALYZED_SUBSET``); one that does not makes the values read
from it pending (``cohort_selection.subset_staleness``). The decoder's quality is over each
epoch's decode, not over events.

Each derived output above that reads the canonical event table records the
table's SHA-256 in its manifest; an output made from another table than the
current one, or recording none, makes the values read from it pending
(:func:`~continuum_swr_replay.reported.inputs.event_table_staleness`). The restriction window values are also pending
when the window run read another event table than the held-out tetrode run whose
spike counts it adds (:func:`~.restriction.window_run_note_staleness`); the values of the
held-out event run when it read another restriction run, per-bin facts run or
event table than the current ones (:func:`~continuum_swr_replay.reported.inputs.read_heldout_restriction`); and those
that read the well visits through the per-bin facts (the time sums inside a
visit, the held-out events' start, the window counts after a visit) when the
facts run was not made from the current session cache (``prepare_data.session_cache_staleness``).
Every restriction value, over the canonical windows or the held-out events and
time, is also pending when the restriction run's recorded decode frames or SWR
source are not the current ones (:func:`~continuum_swr_replay.reported.inputs.restriction_input_staleness`: the
source by name, the decode frames against the event features run's record of
them). The held-out tetrode values and
the restriction window values are against Local-rest controls inside one well
visit (:data:`~.common.CONTROL_SUBSET`): their runs must have read the current
participation tables, and they are pending, as the participation values are,
when the participation run records no subset or another, or other visits than
the current session cache's (the participation run records the subset only
since its controls lie inside one visit).

Pooled values sum events over sessions and animals; "animals in direction"
counts the animals whose own value points the way the claim states, an
animal's value going event -> epoch -> animal (a median of its epochs'
medians, a mean of its epochs' means).

The values are made by subject, one module for each part of the summary, in
the order the summary holds them: :mod:`.cohort` (the well-visit labels, the
cohort and the boundary timing against SWRs), :mod:`.properties` (the event
properties, the start distance, rest theta and the property intervals),
:mod:`.signals` (theta, ripple power, MUA and gamma during events),
:mod:`.detectors` (the other detectors and what an analysis without the
non-local detector leaves out), :mod:`.spike_evidence` (the spike-evidence run
and the held-out test split by SWR), :mod:`.content` (where content sits along
the arms), :mod:`.tetrode_holdout` (the participation, likelihood-advantage and
held-out tetrode runs), :mod:`.restriction` (the restriction decoders),
:mod:`.decoder` (the decoder's quality), :mod:`.class_structure` (one class or
two, and the immobile time by signal and state) and :mod:`.validation` (the
immobile time by non-local probability, the rule sweep and the state
calibration). :mod:`.common` holds the tags table, the subsets the
descriptions name, the reads every part takes (:class:`~.common.SummaryReads`)
and the helpers several parts call; the readers shared with the figures stay
in :mod:`continuum_swr_replay.reported.inputs`. :func:`summarize`, here, reads
the event table and the visits and calls each part.
"""

from __future__ import annotations

from continuum_swr_replay.analysis.event_definition.analyzed_time import read_swr_table
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    check_visit_context_starts,
    select_visit_context,
)
from continuum_swr_replay.analysis.event_tables.prepare_data import well_visit_times
from continuum_swr_replay.data_loaders.epoch_register import analyzed_epoch_ids
from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported.events.class_structure import add_class_structure
from continuum_swr_replay.reported.events.cohort import add_cohort
from continuum_swr_replay.reported.events.common import TAGS, SummaryReads
from continuum_swr_replay.reported.events.content import add_content
from continuum_swr_replay.reported.events.decoder import add_decoder
from continuum_swr_replay.reported.events.detectors import add_detectors
from continuum_swr_replay.reported.events.properties import add_properties
from continuum_swr_replay.reported.events.restriction import add_restriction
from continuum_swr_replay.reported.events.signals import add_signals
from continuum_swr_replay.reported.events.spike_evidence import add_spike_evidence
from continuum_swr_replay.reported.events.tetrode_holdout import add_tetrode_holdout
from continuum_swr_replay.reported.events.validation import add_validation
from continuum_swr_replay.reported.inputs import EventsInputs, default_inputs
from continuum_swr_replay.reported.values import Summary, file_record, rel
from continuum_swr_replay.runs.io import ok_epochs, read_manifest

__all__ = ["TAGS", "default_inputs", "summarize"]


def summarize(inputs: EventsInputs, produced_by: str) -> Summary:
    """The events summary from ``inputs``: each part's values, in this order."""
    summary = Summary(name="events", produced_by=produced_by)

    # the one selection: the events that start inside a well visit
    table = read_swr_table(inputs.event_table)
    events = select_visit_context(table)
    # the time masks read the visits themselves: they and the labels must agree
    run = inputs.event_features_run
    features_manifest = read_manifest(run)
    run_epochs = ok_epochs(features_manifest)
    visits = {identifier: well_visit_times(identifier) for identifier in run_epochs}
    check_visit_context_starts(table, visits)
    event_table_record = file_record(inputs.event_table, ROOT_DIR)
    reads = SummaryReads(
        summary=summary,
        inputs=inputs,
        table=table,
        events=events,
        run=run,
        features_manifest=features_manifest,
        run_epochs=run_epochs,
        visits=visits,
        event_table_record=event_table_record,
        current_sha256=event_table_record["sha256"],
        # every per-epoch input must hold the epoch register's analyzed epochs
        n_epochs=len(analyzed_epoch_ids()),
        source=f"{rel(inputs.event_table)} (canonical event table)",
    )

    add_cohort(reads)
    add_properties(reads)
    add_signals(reads)
    add_detectors(reads)
    spike_evidence_record = add_spike_evidence(reads)
    add_content(reads)
    participation = add_tetrode_holdout(reads)
    add_restriction(reads, participation)
    add_decoder(reads)
    add_class_structure(reads, spike_evidence_record)
    add_validation(reads)
    return summary
