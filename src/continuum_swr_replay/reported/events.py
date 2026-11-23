"""The events summary: what the events are, and how they compare with SWRs and Widloski events.

Every value is read from an output an analysis already wrote; nothing is
refitted here. This module names and describes the values. The statistics are
computed in ``analysis.event_properties.event_statistics``,
``analysis.decode_quality.quality_metrics`` and
``analysis.detector_comparison.nonlocal_comparison``; here only the size of an
input (its epochs, its animals) is counted. The inputs (:class:`EventsInputs`):

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
  which held out only the encodings, is refused, :func:`require_fold_fit`),
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
  (:data:`HELDOUT_EVENTS_SUBSET`) and the time inside a well visit.

Every value is over the analyses' one subset: the events that start inside a
well visit (``cohort_selection.select_visit_context``, applied once, where the
event table is read; a stored per-event output is read over the same events,
``cohort_selection.select_events_in``), and time, SWRs and the other detectors'
events inside a well visit. Each value's description says which
(:data:`EVENTS_SUBSET`, :data:`TIME_SUBSET`, :data:`DETECTOR_SUBSET`), and the
summary counts the events of every well-visit label, kept and left out
(:func:`visit_context_values`). An output re-run over the subset records it
(``cohort_selection.ANALYZED_SUBSET``); one that does not makes the values read
from it pending (``cohort_selection.subset_staleness``). The decoder's quality is over each
epoch's decode, not over events.

Each derived output above that reads the canonical event table records the
table's SHA-256 in its manifest; an output made from another table than the
current one, or recording none, makes the values read from it pending
(:func:`event_table_staleness`). The restriction window values are also pending
when the window run read another event table than the held-out tetrode run whose
spike counts it adds (:func:`window_run_note_staleness`); the values of the
held-out event run when it read another restriction run, per-bin facts run or
event table than the current ones (:func:`read_heldout_restriction`); and those
that read the well visits through the per-bin facts (the time sums inside a
visit, the held-out events' start, the window counts after a visit) when the
facts run was not made from the current session cache (``prepare_data.session_cache_staleness``).
Every restriction value, over the canonical windows or the held-out events and
time, is also pending when the restriction run's recorded decode frames or SWR
source are not the current ones (:func:`restriction_input_staleness`: the
source by name, the decode frames against the event features run's record of
them). The held-out tetrode values and
the restriction window values are against Local-rest controls inside one well
visit (:data:`CONTROL_SUBSET`): their runs must have read the current
participation tables, and they are pending, as the participation values are,
when the participation run records no subset or another, or other visits than
the current session cache's (the participation run records the subset only
since its controls lie inside one visit).

Pooled values sum events over sessions and animals; "animals in direction"
counts the animals whose own value points the way the claim states, an
animal's value going event -> epoch -> animal (a median of its epochs'
medians, a mean of its epochs' means).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.constants import REMOTE_CM
from continuum_swr_replay.analysis.decode_quality import (
    nonlocal_probability,
    quality_metrics,
)
from continuum_swr_replay.analysis.detector_comparison import nonlocal_comparison
from continuum_swr_replay.analysis.detector_comparison.feature_run import (
    EVENT_FEATURES_RUN,
    STATE_SPACE_FEATURES_RUN,
)
from continuum_swr_replay.analysis.event_definition import swr_source
from continuum_swr_replay.analysis.event_definition.analyzed_time import read_swr_table
from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    check_visit_context_starts,
    count_visit_context,
    select_events_in,
    select_visit_context,
    subset_staleness,
)
from continuum_swr_replay.analysis.event_evidence import (
    heldout_events,
    likelihood_advantage,
    participation,
    restriction_totals,
    restriction_windows,
    shared_content,
    tetrode_holdout,
)
from continuum_swr_replay.analysis.event_evidence.nonlocal_restriction import (
    ALL_CONTRASTS,
    FAMILY_SUFFIX,
    FIT,
    SWR_ALLOWED_CONTRASTS,
)
from continuum_swr_replay.analysis.event_evidence.restriction_totals import (
    OUTSIDE_SWR_EVENT_WINDOW_GROUPS,
    OUTSIDE_SWR_EVENT_WINDOWS,
    RESTRICTION_FIGURE_EVENT_GROUP,
    RESTRICTION_FIGURE_FAMILY,
    RESTRICTION_FIGURE_INSIDE_SWR,
    RESTRICTION_FIGURE_OUTSIDE_EVENTS,
    RESTRICTION_FIGURE_OUTSIDE_SWR,
    SWR_ALLOWED_FIGURE_DECODERS,
    SWR_ALLOWED_FIGURE_DIFFERENCES,
    SWR_ALLOWED_TIME_CONTRASTS,
)
from continuum_swr_replay.analysis.event_evidence.spike_evidence import (
    BASELINE_GROUP,
    SPIKE_EVIDENCE_METRICS,
    in_analyzed_subset,
    with_event_ids,
)
from continuum_swr_replay.analysis.event_evidence.state_evidence.windows import (
    SWR_LABELS,
)
from continuum_swr_replay.analysis.event_evidence.tetrode_holdout import TRIAL_DESIGN
from continuum_swr_replay.analysis.event_properties import (
    content_along_arms,
    event_statistics,
    low_rate_places,
    nonlocal_by_signal,
    resolved_content,
    rule_sweep,
    state_by_signal,
)
from continuum_swr_replay.analysis.event_properties.class_or_continuum import (
    KAY_ZSCORE_THRESHOLD,
)
from continuum_swr_replay.analysis.event_properties.content_place_rate import (
    LOW_RATE_RANK,
)
from continuum_swr_replay.analysis.event_properties.property_contrast import (
    MAX_NAN_REPLICATE_SHARE,
)
from continuum_swr_replay.analysis.event_tables.prepare_data import (
    behavioral_context_cache_path,
    session_cache_staleness,
    time_mask_staleness,
    well_visit_times,
)
from continuum_swr_replay.analysis.regression.folds import MAX_SEGMENT_S, N_FOLDS
from continuum_swr_replay.data_loaders.epoch_register import (
    analyzed_animals,
    analyzed_epoch_ids,
    require_analyzed_epochs,
)
from continuum_swr_replay.parameters import DEFAULT_NON_LOCAL_THRESHOLD
from continuum_swr_replay.paths import (
    BRAIN_STATE_REFERENCE_DIR,
    CANONICAL_EVENTS_FOUND_NAME,
    CONTENT_ALONG_ARMS_DIR,
    CONTENT_PLACE_RATE_NAME,
    DECODER_QUALITY_DIR,
    EVENT_BIN_FACTS_DIR,
    EVENT_CONTENT_PLACE_RATE_DIR,
    EVENT_GAMMA_DIR,
    EVENT_LIKELIHOOD_ADVANTAGE_DIR,
    EVENT_NONLOCAL_RESTRICTION_DIR,
    EVENT_NONLOCAL_RESTRICTION_HELDOUT_EVENTS_DIR,
    EVENT_NONLOCAL_RESTRICTION_SWR_ALLOWED_DIR,
    EVENT_NONLOCAL_RESTRICTION_SWR_ALLOWED_HELDOUT_EVENTS_DIR,
    EVENT_NONLOCAL_RESTRICTION_WINDOWS_DIR,
    EVENT_PARTICIPATION_DIR,
    EVENT_PROPERTY_BOOTSTRAP_DIR,
    EVENT_RULE_SWEEP_DIR,
    EVENT_START_REST_THETA_DIR,
    EVENT_TETRODE_HOLDOUT_DIR,
    EVENT_TETRODE_HOLDOUT_EXTRA_DIR,
    EXPOSURE_IN_VISITS_DIR,
    HELDOUT_EVENTS_NAME,
    HELDOUT_TIME_SUMS_NAME,
    NL_BOUTS_BY_SWR_STEM,
    NONLOCAL_BY_SIGNAL_DIR,
    NONLOCAL_PROBABILITY_DIR,
    NONLOCAL_RESTRICTION_WINDOWS_NAME,
    PARTICIPATION_NAME,
    RIPPLE_CLASS_STRUCTURE_DIR,
    ROOT_DIR,
    STATE_BY_SIGNAL_DIR,
    STATE_CALIBRATION_IN_VISITS_DIR,
    STATE_EVIDENCE_DIR,
    STATE_EVIDENCE_RUN,
    TETRODE_HOLDOUT_NAME,
    event_features_root,
    event_spike_evidence_dir,
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.reported.formatting import (
    MS_PER_S,
    format_number,
    milliseconds,
)
from continuum_swr_replay.reported.values import (
    Summary,
    Tags,
    Value,
    file_record,
    files_record,
    over,
    pend,
    rel,
    value_factory,
)
from continuum_swr_replay.runs.epoch_records import MANIFEST
from continuum_swr_replay.runs.io import (
    files_sha256,
    ok_epochs,
    ok_table_paths,
    read_complete_run,
    read_manifest,
)
from continuum_swr_replay.runs.staleness import read_output_manifest, recorded_sha256
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

TAGS = {
    "decoding_error": Tags(("2a.1",)),
    "calibration": Tags(("2a.1",), ("S2d",)),
    "nonlocal_probability": Tags(("2a.6",), ("S2c",)),
    "state_fractions": Tags(("2a.6",), ("decoder_validation",)),
    "state_fractions_immobile": Tags(("2a.6",), ("2a",)),
    "cohort": Tags(("2b.2",), ("2d",)),
    "cohort_pooled": Tags(("2b.2",)),
    "visit_context": Tags(("2b.2",)),
    "left_out": Tags(("2b.2",)),
    "other_detectors_main": Tags(("2b.3",), ("2",)),
    "other_detectors": Tags(("2b.3",)),
    "nonlocal_found": Tags(("2b.3",), ("2e",)),
    "swr_content": Tags(("2b.3",), ("2c",)),
    "boundaries": Tags(("2b.4",), ("2f",)),
    "boundary_timing": Tags(("2b.4",), ("S2f",)),
    "boundary_timing_both": Tags(("2b.4",)),
    "boundary_timing_pooled": Tags(("2b.4",)),
    "ripple": Tags(("2d.8", "2b.5"), ("2g",)),
    "mua": Tags(("2d.8",), ("2g",)),
    "theta": Tags(("2d.8",), ("2g",)),
    "gamma_z": Tags(("2d.8",), ("S3",)),
    "brain_state": Tags(("2d.8",)),
    "class_distribution": Tags(("2b.5",)),
    "class_step": Tags(("2b.5", "2d.2", "2d.3", "2d.5", "2d.6")),
    "class_step_evidence": Tags(("2b.5",)),
    "nonlocal_power": Tags(("2b.5",), ("2i",)),
    "nonlocal_mua": Tags(("2b.5",), ("S2g",)),
    "state_by_signal": Tags(("2b.5",)),
    "rule_sweep": Tags(("2a.5",), ("S2e",)),
    "rule_sweep_pooled": Tags(("2a.5",)),
    "state_calibration": Tags(("2a.7",), ("S2a",)),
    "participation": Tags(("2d.9",), ("2h",)),
    "spike_rate": Tags(("2d.9",)),
    "state_composition": Tags(("2d.10",), ("3b",)),
    "likelihood_advantage": Tags(("2c.3",)),
    "tetrode_holdout": Tags(("2c.3",), ("2j", "S2h")),
    "tetrode_holdout_split": Tags(("2c.3",), ("S2h",)),
    "tetrode_holdout_text": Tags(("2c.3",)),
    "tetrode_holdout_extra_figure": Tags(("2c.3",), ("2j",)),
    "nonlocal_restriction": Tags(("2c.3",)),
    "nonlocal_restriction_figure": Tags(("2c.3",), ("2k",)),
    "shared_content": Tags(("2c.3",)),
    "shared_content_raw": Tags(("2c.3",)),
    "shared_content_text": Tags(("2c.3",)),
    "nonlocal_restriction_distance": Tags(("2c.3",), ("S2i",)),
    "nonlocal_restriction_swr_allowed": Tags(("2c.3",), ("S2l",)),
    "spike": Tags(("2d.9", "2c.3")),
    "spike_count": Tags(("2d.9",), ("2",)),
    "spike_evidence": Tags(("2c.3",), ("2",)),
    "spike_gpi": Tags(("2c.4",)),
    "content_occupancy": Tags(("2d.1",), ("3f",)),
    "content_by_animal": Tags(("2d.1",)),
    "duration": Tags(("2d.6",), ("3a",)),
    "extent": Tags(("2d.3",), ("3c",)),
    "arms": Tags(("2d.3",)),
    "speed": Tags(("2d.2",), ("3d",)),
    "certainty": Tags(("2d.5",)),
    "hpd": Tags(("2d.5",), ("S2k",)),
    "content_distance": Tags(("2d.1",), ("S3c",)),
    "content_distance_pooled": Tags(("2d.1",)),
    "stillness": Tags(("2d.4",)),
    "location": Tags(("2d.1",)),
}
"""Each group of values: the questions it answers and the figures that show it."""

#: The other detectors of question 2b.3, by the word their keys use. Kay is the main
#: figure's; the rest go to the supplement.
DETECTORS = {
    "kay": "kay",
    "karlsson": "karlsson",
    "yu": "yu",
    "zugaro": "zugaro",
    "carey": "carey",
    "mua": "mua",
    "widloski": "memoryless",
    "widloski_criteria": "memoryless_criteria",
}
MAIN_DETECTOR = "kay"

EVENTS_SUBSET = (
    "the immobile non-local events that start inside a well visit (from the first "
    "poke-in at a well to the final poke-out; visit_context in_visit or "
    "spans_visit_end)"
)
"""What the values of the events are over, as their descriptions say."""

HELDOUT_EVENTS_SUBSET = (
    "the immobile non-local events detected from the held-out decode (each fold's "
    "bins under the decoder fit on the other folds, by the canonical event rule; "
    "a small share span two folds, each bin still under a fit that did not see "
    "it) "
    "that start inside a well visit (their first bin is inside one, from the first "
    "poke-in at a well to the final poke-out)"
)
"""What the values of the events detected from the held-out decode are over."""

CONTROL_SUBSET = (
    f"{EVENTS_SUBSET}, against Local-rest control windows of each event's length "
    "lying wholly inside one well visit"
)
"""What the participation values are over: their controls are Local rest inside
one well visit (``event_evidence.participation.CONTROL_RULE``)."""

CANONICAL_WINDOWS_SUBSET = (
    "the canonical events, chosen from the decode fit on the whole epoch (the "
    "scores are held out; the choice of events is not; Figure 2k's values use "
    f"the events detected from the held-out decode instead): {CONTROL_SUBSET}"
)
"""What the restriction values over the canonical windows are over
(:func:`restriction_window_values`)."""

TIME_SUBSET = (
    "time inside a well visit (from the first poke-in at a well to the final poke-out)"
)
"""What the values of time (rest, immobile bins) are over."""

DETECTOR_SUBSET = (
    "the other detectors' events whose first sample is inside a well visit and "
    "the non-local events that start inside one, each selected by its own start; "
    "whether an event overlaps an event of the other side is read against all of "
    "that side's events, so its partner may start outside a visit"
)
"""What the values of the detector comparison are over."""

LEFT_OUT_SUBSET = (
    f"{EVENTS_SUBSET}, and the non-local detector's exposure inside a well visit; "
    "an event is left out when no event of the other inventory overlaps it, "
    "wherever that event starts"
)
"""What the values of what another inventory leaves out are over."""

SPIKE_EVIDENCE_SUBSET = (
    f"{EVENTS_SUBSET}, against the matched quiet windows whose first sample is "
    "inside a well visit"
)
"""What the values of the spike-evidence run are over."""

BOUT_SUBSET = (
    "the decoder's immobile Non-Local Continuous bouts whose first sample is inside "
    "a well visit (from the first poke-in at a well to the final poke-out); the "
    "bouts' matched windows are drawn from all of the epoch's time in the regime, "
    "and the values reported here do not use them"
)
"""What the values of the held-out test split by SWR are over."""


@dataclass(frozen=True)
class EventsInputs:
    """The files the events summary reads."""

    event_table: Path
    event_features_run: Path
    spike_evidence_dir: Path
    heldout_by_swr: Path
    content_along_arms: Path
    brain_state_reference: Path
    event_gamma: Path
    class_structure: Path
    nonlocal_by_signal: Path
    nonlocal_probability: Path
    state_by_signal: Path
    event_rule_sweep: Path
    state_calibration_in_visits: Path
    event_start_rest_theta: Path
    property_bootstrap: Path
    decoder_quality: Path
    event_participation: Path
    likelihood_advantage: Path
    tetrode_holdout: Path
    tetrode_holdout_extra: Path
    content_place_rate: Path
    nonlocal_restriction: Path
    nonlocal_restriction_windows: Path
    nonlocal_restriction_heldout_events: Path
    nonlocal_restriction_swr_allowed: Path
    nonlocal_restriction_swr_allowed_heldout_events: Path
    bin_facts: Path
    exposure_in_visits: Path


def default_inputs() -> EventsInputs:
    return EventsInputs(
        event_table=table_path("event_table", replay_content_dir()),
        event_features_run=event_features_root() / EVENT_FEATURES_RUN,
        spike_evidence_dir=event_spike_evidence_dir(),
        heldout_by_swr=processed_data_dir()
        / STATE_EVIDENCE_RUN
        / STATE_EVIDENCE_DIR
        / f"{NL_BOUTS_BY_SWR_STEM}_stats.csv",
        content_along_arms=processed_data_dir() / CONTENT_ALONG_ARMS_DIR,
        brain_state_reference=processed_data_dir()
        / BRAIN_STATE_REFERENCE_DIR
        / "reference.parquet",
        event_gamma=processed_data_dir() / EVENT_GAMMA_DIR,
        class_structure=processed_data_dir() / RIPPLE_CLASS_STRUCTURE_DIR,
        nonlocal_by_signal=processed_data_dir() / NONLOCAL_BY_SIGNAL_DIR,
        nonlocal_probability=processed_data_dir() / NONLOCAL_PROBABILITY_DIR,
        state_by_signal=processed_data_dir() / STATE_BY_SIGNAL_DIR,
        event_rule_sweep=processed_data_dir() / EVENT_RULE_SWEEP_DIR,
        state_calibration_in_visits=processed_data_dir()
        / STATE_CALIBRATION_IN_VISITS_DIR,
        event_start_rest_theta=processed_data_dir() / EVENT_START_REST_THETA_DIR,
        property_bootstrap=processed_data_dir() / EVENT_PROPERTY_BOOTSTRAP_DIR,
        decoder_quality=processed_data_dir() / DECODER_QUALITY_DIR,
        event_participation=processed_data_dir() / EVENT_PARTICIPATION_DIR,
        likelihood_advantage=processed_data_dir() / EVENT_LIKELIHOOD_ADVANTAGE_DIR,
        tetrode_holdout=processed_data_dir() / EVENT_TETRODE_HOLDOUT_DIR,
        tetrode_holdout_extra=processed_data_dir() / EVENT_TETRODE_HOLDOUT_EXTRA_DIR,
        content_place_rate=processed_data_dir() / EVENT_CONTENT_PLACE_RATE_DIR,
        nonlocal_restriction=processed_data_dir() / EVENT_NONLOCAL_RESTRICTION_DIR,
        nonlocal_restriction_windows=processed_data_dir()
        / EVENT_NONLOCAL_RESTRICTION_WINDOWS_DIR,
        nonlocal_restriction_heldout_events=processed_data_dir()
        / EVENT_NONLOCAL_RESTRICTION_HELDOUT_EVENTS_DIR,
        nonlocal_restriction_swr_allowed=processed_data_dir()
        / EVENT_NONLOCAL_RESTRICTION_SWR_ALLOWED_DIR,
        nonlocal_restriction_swr_allowed_heldout_events=processed_data_dir()
        / EVENT_NONLOCAL_RESTRICTION_SWR_ALLOWED_HELDOUT_EVENTS_DIR,
        bin_facts=processed_data_dir() / EVENT_BIN_FACTS_DIR,
        exposure_in_visits=processed_data_dir() / EXPOSURE_IN_VISITS_DIR,
    )


def _input_staleness(
    manifest: dict, name: str, current_sha256: str, what: str, adjective: str, noun: str
) -> str | None:
    """Why an output is stale for one of its inputs, or None.

    ``manifest`` records the input's SHA-256 under ``name`` or ``inputs.<name>``
    (``runs.staleness.recorded_sha256``); an output recording none cannot be
    tied to the current input and is treated as stale.
    """
    recorded = recorded_sha256(manifest, name)
    if recorded is None:
        return f"{what} records no {adjective} SHA-256; rerun it before citing"
    if recorded != current_sha256:
        return (
            f"{what} was made from another {noun} than the current one "
            f"(sha256 {recorded[:12]}, current {current_sha256[:12]}); rerun it "
            "before citing"
        )
    return None


def event_table_staleness(manifest: dict, current_sha256: str, what: str) -> str | None:
    """Why an output made from the canonical event table is stale, or None."""
    return _input_staleness(
        manifest, "event_table", current_sha256, what, "event-table", "event table"
    )


def window_run_note_staleness(manifest: dict, what: str) -> str | None:
    """Why the restriction window run's values are pending for its note, or None.

    The window run records a note (``restriction_windows.EVENT_TABLE_NOTE``, in
    its settings) when the event table it read is not the one the held-out
    tetrode run read, whose spike counts it adds; its windows still matched that
    run's window by window, but the two runs are not on one table.
    """
    note = manifest.get("settings", {}).get(restriction_windows.EVENT_TABLE_NOTE)
    if note is None:
        return None
    theirs = note["tetrode_holdout_event_table_sha256"]
    return (
        f"{what} read another event table (sha256 "
        f"{note['event_table_sha256'][:12]}) than the held-out tetrode run whose "
        f"spike counts it adds (sha256 {'none' if theirs is None else theirs[:12]}); "
        "rerun the held-out tetrode run and then the window run before citing"
    )


def require_trial_folds(manifest: dict, what: str, rerun: str) -> None:
    """Refuse a held-out run whose folds are not whole trials.

    A held-out run records its fold design in its settings (``fold_design``).
    A run of an older design (contiguous blocks of time, or no design recorded)
    cannot give values that describe folds of whole trials.

    Raises
    ------
    ValueError
        If ``manifest``'s settings do not record ``"fold_design": "trials"``;
        the message ends with ``rerun``, what to run instead.
    """
    design = manifest.get("settings", {}).get("fold_design")
    if design != TRIAL_DESIGN:
        raise ValueError(
            f"{what} is not held out in folds of whole trials (its settings "
            f"record fold_design {design!r}, not {TRIAL_DESIGN!r}). {rerun}"
        )


def require_fold_fit(manifest: dict, what: str) -> None:
    """Refuse a restriction run that is not the fold fit on folds of whole trials.

    The values describe scores under a decoder fit on the other folds, five
    folds of whole trials balanced by destination well; the fold-fit run's
    settings record ``"fit": "folds"`` and ``"fold_design": "trials"``. A run of
    the earlier design, which held out only the encodings and ran the models
    with the transitions fit on the whole epoch, records no ``fit``; a run of
    another fold design is refused by :func:`require_trial_folds`. Neither can
    give these values.

    Raises
    ------
    ValueError
        If ``manifest``'s settings do not record ``"fit": "folds"`` and
        ``"fold_design": "trials"``.
    """
    fit = manifest.get("settings", {}).get("fit")
    rerun = (
        "Put the fold-fit run (scripts/event_evidence/score_nonlocal_restriction.py) "
        "in this folder, then rerun scripts/event_evidence/measure_bin_facts.py, "
        "scripts/event_evidence/detect_heldout_events.py and "
        "scripts/event_evidence/sum_nonlocal_restriction_windows.py on it."
    )
    if fit != FIT:
        raise ValueError(
            f"{what} is not the fold fit (its settings record fit {fit!r}, not "
            f"{FIT!r}): it is the earlier design, which held out only the "
            f"encodings. {rerun}"
        )
    require_trial_folds(manifest, what, rerun)


def require_tetrode_holdout_folds(manifest: dict, what: str) -> None:
    """Refuse a held-out tetrode run whose folds are not whole trials
    (:func:`require_trial_folds`).

    Raises
    ------
    ValueError
        If ``manifest``'s settings do not record ``"fold_design": "trials"``.
    """
    require_trial_folds(
        manifest,
        what,
        "Put the held-out tetrode run on folds of whole trials "
        "(scripts/event_evidence/score_tetrode_holdout.py) in this folder, then "
        "rerun scripts/event_evidence/sum_nonlocal_restriction_windows.py on it.",
    )


def spike_evidence_staleness(
    manifest: dict, current_sha256: str, what: str
) -> str | None:
    """Why an output made from the spike-evidence run is stale, or None.

    ``current_sha256`` is the one digest over the run's per-epoch metrics
    (``runs.io.files_sha256``), which the output's manifest records under
    ``inputs.spike_evidence`` and this summary records for the same files.
    """
    return _input_staleness(
        manifest,
        "spike_evidence",
        current_sha256,
        what,
        "spike-evidence",
        "spike-evidence run",
    )


# --- the decoder --------------------------------------------------------------

#: The ten equal Local-probability bins of the calibration curve, in order.
CALIBRATION_BINS = (
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
)

#: Each decoder state's word in a state-fraction key.
_STATE_WORDS = {
    "Local_prob": "local",
    "No-Spike_prob": "no_spike",
    "Non-Local_Continuous_prob": "continuous",
    "Non-Local_Fragmented_prob": "fragmented",
}


def decoding_error_values(sessions: pd.DataFrame, source: str) -> list[Value]:
    """Question 2a.1: the Local-state decoding error over sessions."""
    make = value_factory(decoding_error_values, source, TAGS["decoding_error"])
    stats = quality_metrics.local_decoding_error_statistics(sessions)
    return [
        make(
            "decoding_error_median_cm",
            stats["median_error_cm"],
            "quantity",
            "sig2",
            "median over sessions of each session's median Local-state decoding "
            "error (the track-graph distance from the decoded to the actual "
            "position, over time bins with Local probability above 0.5 outside "
            "excluded time), sessions pooled",
            "cm",
        ),
        make(
            "decoding_error_sessions",
            stats["n_sessions"],
            "count",
            "count",
            "sessions with a Local-state decoding error",
        ),
    ]


def calibration_values(calibration: pd.DataFrame, source: str) -> list[Value]:
    """Question 2a.1 and Figure 2's supplement: how far the decoded place is
    from the animal in each of ten Local-probability bins, over all analyzed
    time (``quality_metrics.animal_distance_by_local_probability``: each
    session's median distance between the animal and the maximum of the
    position estimate conditional on the non-local states, the median over an
    animal's sessions, then the mean over animals, with the smallest and
    largest animal). The keys keep their old name; the quantity is not a calibration."""
    make = value_factory(calibration_values, source, TAGS["calibration"])
    _, curve = quality_metrics.animal_distance_by_local_probability(calibration)
    if len(curve) != len(CALIBRATION_BINS):
        raise ValueError(
            f"the distance by Local probability has {len(curve)} bins, not "
            f"{len(CALIBRATION_BINS)}"
        )
    values = []
    for index, (word, (center, row)) in enumerate(
        zip(CALIBRATION_BINS, curve.iterrows(), strict=True), start=1
    ):
        where = (
            f"time bins whose Local probability is in the 0.1-wide bin centered on "
            f"{format_number(center, 'dec2')} (bin {index} of "
            f"{len(CALIBRATION_BINS)}), outside excluded time, moving and immobile"
        )
        what = (
            "each session's median distance along the track between the animal "
            "and the maximum of the position estimate conditional on the "
            f"non-local states, at {where}"
        )
        key = f"calibration_error_bin_{word}"
        values += [
            make(
                key,
                row["animal_mean"],
                "quantity",
                "sig2",
                f"mean over animals of each animal's median over its sessions of {what}",
                "cm",
            ),
            make(
                f"{key}_animal_min",
                row["animal_min"],
                "quantity",
                "sig2",
                f"smallest animal's value of that mean over sessions of {what}",
                "cm",
            ),
            make(
                f"{key}_animal_max",
                row["animal_max"],
                "quantity",
                "sig2",
                f"largest animal's value of that mean over sessions of {what}",
                "cm",
            ),
        ]
    return values


def _state_fraction_values(
    make: Callable[..., Value], sessions: pd.DataFrame, movement: str, where: str
) -> list[Value]:
    """One movement state's four state fractions (``movement`` of
    ``quality_metrics.MOVEMENT_STATES``), made by ``make``."""
    fractions = quality_metrics.state_fraction_statistics(sessions)
    return [
        make(
            f"state_fraction_{movement}_{_STATE_WORDS[state]}",
            fractions.loc[movement, state],
            "fraction",
            "pct1",
            f"median over sessions of each session's mean {state} over its "
            f"{where} outside excluded time, sessions pooled",
        )
        for state in quality_metrics.STATE_PROBABILITY_COLUMNS
    ]


def state_fraction_values(sessions: pd.DataFrame, source: str) -> list[Value]:
    """Question 2a.6: the decoder's state fractions while the animal moves (the
    decoder-validation figure's bars)."""
    make = value_factory(state_fraction_values, source, TAGS["state_fractions"])
    return _state_fraction_values(make, sessions, "moving", "bins faster than 4 cm/s")


def immobile_state_fraction_values(sessions: pd.DataFrame, source: str) -> list[Value]:
    """Question 2a.6: the decoder's state fractions while the animal is still
    inside a well visit, beside Figure 2a, where everything else is inside the
    visits (the stored immobile fractions are over the immobile bins inside a
    visit, ``measure_decoder_quality.py``)."""
    make = value_factory(
        immobile_state_fraction_values, source, TAGS["state_fractions_immobile"]
    )
    return _state_fraction_values(
        make, sessions, "immobile",
        "bins slower than 4 cm/s inside a well visit (from the first poke-in at "
        "a well to the final poke-out)",
    )  # fmt: skip


# --- how common the events are, and how they sit against SWRs ----------------


def event_cohort_values(events: pd.DataFrame, source: str) -> list[Value]:
    """Question 2b.2: how many immobile non-local events carry no SWR."""
    make = value_factory(event_cohort_values, source, TAGS["cohort"])
    stats = event_statistics.overlap_class_statistics(events)
    return [
        make(
            "nonlocal_events",
            stats["n_events"],
            "count",
            "count",
            "immobile non-local events in the canonical event table (whole event below 4 cm/s)",
        ),
        make(
            "nonlocal_events_non_swr",
            stats["n_non_swr"],
            "count",
            "count",
            "of those, events overlapping no SWR (overlap_class non_swr)",
        ),
        make(
            "nonlocal_events_swr_contained",
            stats["n_swr_contained"],
            "count",
            "count",
            "events lying inside an SWR (overlap_class swr_contained)",
        ),
        make(
            "nonlocal_events_swr_partial",
            stats["n_swr_partial"],
            "count",
            "count",
            "events partly overlapping an SWR (overlap_class swr_partial)",
        ),
        make(
            "non_swr_fraction",
            stats["non_swr_fraction"],
            "fraction",
            "pct",
            "share of immobile non-local events overlapping no SWR, pooled over "
            "every event (no figure draws it)",
            tags=TAGS["cohort_pooled"],
        ),
        make(
            "non_swr_fraction_animal_mean",
            stats["non_swr_fraction_animal_mean"],
            "fraction",
            "pct",
            "share of immobile non-local events overlapping no SWR, the mean over "
            "animals of each animal's share (an animal's share is the mean over its "
            "epochs of the epoch's share; Figure 2d's bar)",
        ),
        make(
            "swr_contained_fraction",
            stats["swr_contained_fraction"],
            "fraction",
            "pct",
            "share of immobile non-local events lying inside an SWR, pooled over "
            "every event (no figure draws it)",
            tags=TAGS["cohort_pooled"],
        ),
        make(
            "swr_contained_fraction_animal_mean",
            stats["swr_contained_fraction_animal_mean"],
            "fraction",
            "pct",
            "share of immobile non-local events lying inside an SWR, the mean over "
            "animals of each animal's share (the mean over its epochs; Figure 2d's "
            "bar)",
        ),
        make(
            "swr_partial_fraction",
            stats["swr_partial_fraction"],
            "fraction",
            "pct",
            "share of immobile non-local events partly overlapping an SWR, pooled "
            "over every event (no figure draws it)",
            tags=TAGS["cohort_pooled"],
        ),
        make(
            "swr_partial_fraction_animal_mean",
            stats["swr_partial_fraction_animal_mean"],
            "fraction",
            "pct",
            "share of immobile non-local events partly overlapping an SWR, the mean "
            "over animals of each animal's share (the mean over its epochs; Figure "
            "2d's bar)",
        ),
        make(
            "non_swr_fraction_animal_min",
            stats["non_swr_fraction_animal_min"],
            "fraction",
            "pctnum",
            "lowest per-animal share of non-SWR events (an animal's share is the "
            "mean over its epochs of the epoch's share)",
        ),
        make(
            "non_swr_fraction_animal_max",
            stats["non_swr_fraction_animal_max"],
            "fraction",
            "pct",
            "highest per-animal share of non-SWR events (the mean over the animal's "
            "epochs)",
        ),
        make(
            "animals",
            stats["n_animals"],
            "count",
            "count",
            "animals in the event table",
        ),
        make(
            "epochs", stats["n_epochs"], "count", "count", "epochs in the event table"
        ),
        make(
            "animals_non_swr_plurality",
            stats["n_animals_non_swr_plurality"],
            "count",
            "count",
            "animals whose share of non-SWR events is the largest of their three "
            "overlap-class shares (each the mean over the animal's epochs)",
        ),
        make(
            "animals_non_swr_majority",
            stats["n_animals_non_swr_majority"],
            "count",
            "count",
            "animals whose share of non-SWR events (the mean over their epochs) is "
            "above one half",
        ),
    ]


def visit_context_values(table: pd.DataFrame, source: str) -> list[Value]:
    """Question 2b.2: the event table's events by well-visit label, and how many
    the analyses keep (those that start inside a well visit) and leave out, of
    each kind (``cohort_selection.count_visit_context``)."""
    make = value_factory(visit_context_values, source, TAGS["visit_context"])
    counts = count_visit_context(table)
    values = [
        make(
            f"events_label_{label}",
            row["n"],
            "count",
            "count",
            f"immobile non-local events of the canonical event table with "
            f"visit_context {label} ({'kept' if row['kept'] else 'left out'}: the "
            "analyses keep the events that start inside a well visit)",
        )
        for label, row in counts.iterrows()
    ]
    for word, name, rows in (
        ("", "", table),
        ("swr_", "SWR ", table[table["is_swr"]]),
        ("non_swr_", "non-SWR ", table[~table["is_swr"]]),
    ):
        kind = count_visit_context(rows)
        values += [
            make(
                f"{word}events_in_subset",
                kind.loc[kind["kept"], "n"].sum(),
                "count",
                "count",
                f"immobile non-local {name}events that start inside a well visit "
                "(visit_context in_visit or spans_visit_end): the events the "
                "analyses read",
            ),
            make(
                f"{word}events_left_out",
                kind.loc[~kind["kept"], "n"].sum(),
                "count",
                "count",
                f"immobile non-local {name}events that start outside every well "
                "visit, left out of the analyses",
            ),
        ]
    return values


def detector_visit_values(visit_counts: pd.DataFrame, source: str) -> list[Value]:
    """Question 2b.3: each other detector's compared events left out because
    their first sample is outside every well visit."""
    make = value_factory(detector_visit_values, source, TAGS["other_detectors"])
    return [
        make(
            f"{word}_events_outside_visit",
            visit_counts.loc[detector, "n_outside_visit"],
            "count",
            "count",
            f"{detector} events compared with the non-local detector whose first "
            "sample is outside every well visit, left out of the comparison",
        )
        for word, detector in DETECTORS.items()
    ]


def boundary_offset_values(events: pd.DataFrame, source: str) -> list[Value]:
    """Question 2b.4: non-local event bounds against the matched SWR's (Figure 2f)."""
    make = value_factory(boundary_offset_values, source, TAGS["boundaries"])
    stats = event_statistics.boundary_offset_statistics(
        event_statistics.boundary_offsets(events)
    )
    return [
        make(
            "swr_overlapping_events",
            stats["n_overlapping"],
            "count",
            "count",
            "immobile non-local events overlapping an SWR",
        ),
        make(
            "start_offset_median_ms",
            milliseconds(stats["start_offset_median_s"]),
            "quantity",
            "int",
            "median of event start minus matched SWR start",
            "ms",
        ),
        make(
            "end_offset_median_ms",
            milliseconds(stats["end_offset_median_s"]),
            "quantity",
            "int",
            "median of event end minus matched SWR end",
            "ms",
        ),
        *(
            make(
                f"{bound}_offset_beyond_{name}_fraction",
                stats[f"{bound}_offset_beyond_{name}_fraction"],
                "fraction",
                "pct",
                f"share of SWR-overlapping events whose {bound} differs from the "
                f"matched SWR's {bound} by more than {1000 * tail_s:g} ms",
            )
            for bound in ("start", "end")
            for name, tail_s in event_statistics.OFFSET_TAILS_S.items()
        ),
    ]


def boundary_timing_values(events: pd.DataFrame, source: str) -> list[Value]:
    """Question 2b.4 and Figure 2's supplement, panel S2f: the share of
    SWR-overlapping events in each of the nine combinations of their start and
    end against their matched SWR's (earlier, together within 25 ms, later;
    ``event_statistics.boundary_timing_statistics``): the mean over animals of
    each animal's share, with the lowest and highest animal, and the share
    pooled over events (no figure draws it); and the share whose start and end
    both differ (the four corner combinations together,
    ``event_statistics.boundary_both_ends_differ_statistics``). The panel draws
    the nine combinations, a row each, which carry its tag; the share with both
    bounds apart is for the text."""
    make = value_factory(boundary_timing_values, source, TAGS["boundary_timing"])
    make_both = value_factory(
        boundary_timing_values, source, TAGS["boundary_timing_both"]
    )
    offsets = event_statistics.boundary_offsets(events)
    table = event_statistics.boundary_timing_statistics(offsets, events)
    tolerance_ms = (
        1000 * event_statistics.OFFSET_TAILS_S[event_statistics.TIMING_TOLERANCE]
    )
    words = {
        "earlier": f"more than {tolerance_ms:g} ms earlier than",
        "together": f"within {tolerance_ms:g} ms of",
        "later": f"more than {tolerance_ms:g} ms later than",
    }
    values = []
    for (start, end), row in table.iterrows():
        key = f"timing_start_{start}_end_{end}"
        what = (
            f"SWR-overlapping events that start {words[start]} the matched SWR's "
            f"start and end {words[end]} its end"
        )
        values += _animal_range_values(
            make,
            f"{key}_share",
            (row["share"], row["animal_min"], row["animal_max"]),
            (
                f"share of the {what}, pooled over every SWR-overlapping event",
                f"lowest animal's share of the {what} (an animal's share is the "
                "mean over its epochs of the epoch's share)",
                f"highest animal's share of the {what} (the mean over its epochs)",
            ),
            animal_mean=(
                row["share_animal_mean"],
                f"share of the {what}, the mean over animals of each animal's "
                "share (the mean over its epochs of the epoch's share)",
            ),
            pooled_tags=TAGS["boundary_timing_pooled"],
        )
    both = event_statistics.boundary_both_ends_differ_statistics(offsets, events)
    what = (
        "SWR-overlapping events whose start and end both differ from the matched "
        f"SWR's by more than {tolerance_ms:g} ms (the four corner combinations "
        "together)"
    )
    key = "timing_both_ends_differ_share"
    values += [
        make_both(
            f"{key}_animal_mean",
            both["share_animal_mean"],
            "fraction",
            "pct",
            f"share of the {what}, the mean over animals of each animal's share "
            "(the mean over its epochs of the epoch's share)",
        ),
        make_both(
            f"{key}_animal_min",
            both["animal_min"],
            "fraction",
            "pctnum",
            f"lowest animal's share of the {what} (the mean over its epochs)",
        ),
        make_both(
            f"{key}_animal_max",
            both["animal_max"],
            "fraction",
            "pct",
            f"highest animal's share of the {what} (the mean over its epochs)",
        ),
    ]
    return values


def other_detector_values(events: pd.DataFrame, source: str) -> list[Value]:
    """Question 2b.3: the other side of the overlap, for each other detector.

    For each detector: the share of its events that overlap no non-local event,
    how those divide by content, and, for the events that do overlap, the
    shares of their time in each state. A row is in a state when that state's
    probability is above 0.8; "mostly" is more than half the event's rows.
    """
    values: list[Value] = []
    statistics = nonlocal_comparison.other_detector_statistics(events)
    for word, detector in DETECTORS.items():
        main = word == MAIN_DETECTOR
        make = value_factory(
            other_detector_values,
            source,
            TAGS["other_detectors_main"] if main else TAGS["other_detectors"],
        )
        stats = statistics.loc[detector]
        values += [
            make(
                f"{word}_events",
                stats["n_events"],
                "count",
                "count",
                f"{detector} events compared with the non-local detector",
            ),
            make(
                f"{word}_no_nonlocal_fraction",
                stats["no_nonlocal_fraction"],
                "fraction",
                "pct",
                f"share of {detector} events that overlap no non-local event, pooled "
                "over events",
                tags=TAGS["other_detectors"] if main else None,
            ),
            make(
                f"{word}_mostly_local_fraction",
                stats["mostly_local_fraction"],
                "fraction",
                "pct",
                f"share of {detector} events that overlap no non-local event and have "
                "Local above 0.8 in more than half their rows, pooled over events",
                tags=TAGS["other_detectors"] if main else None,
            ),
            make(
                f"{word}_no_nonlocal_mostly_local_fraction",
                stats["no_nonlocal_mostly_local_fraction"],
                "fraction",
                "pct",
                f"of {detector} events overlapping no non-local event, the share with Local "
                "above 0.8 in more than half their rows",
            ),
            make(
                f"{word}_no_nonlocal_mostly_no_spike_fraction",
                stats["no_nonlocal_mostly_no_spike_fraction"],
                "fraction",
                "pct",
                f"of {detector} events overlapping no non-local event, the share with No-Spike "
                "above 0.8 in more than half their rows",
            ),
            make(
                f"{word}_no_nonlocal_mixed_fraction",
                stats["no_nonlocal_mixed_fraction"],
                "fraction",
                "pct",
                f"of {detector} events overlapping no non-local event, the share with "
                "neither state in more than half their rows",
            ),
            make(
                f"{word}_overlap_local_share_median",
                stats["overlap_local_share_median"],
                "fraction",
                "pct",
                f"median over {detector} events overlapping a non-local event of the share "
                "of their rows with Local above 0.8",
            ),
            make(
                f"{word}_overlap_no_spike_share_median",
                stats["overlap_no_spike_share_median"],
                "fraction",
                "pct",
                f"median over {detector} events overlapping a non-local event of the share "
                "of their rows with No-Spike above 0.8",
            ),
            make(
                f"{word}_overlap_nonlocal_share_median",
                stats["overlap_nonlocal_share_median"],
                "fraction",
                "pct",
                f"median over {detector} events overlapping a non-local event of the share "
                "of their rows with non-local above 0.8",
            ),
            make(
                f"{word}_overlap_uncertain_share_median",
                stats["overlap_uncertain_share_median"],
                "fraction",
                "pct",
                f"median over {detector} events overlapping a non-local event of the share "
                "of their rows with no state above 0.8",
            ),
            make(
                f"{word}_overlap_any_local_fraction",
                stats["overlap_any_local_fraction"],
                "fraction",
                "pct",
                f"share of {detector} events overlapping a non-local event that hold at "
                "least one row with Local above 0.8",
            ),
            make(
                f"{word}_overlap_any_no_spike_fraction",
                stats["overlap_any_no_spike_fraction"],
                "fraction",
                "pct",
                f"share of {detector} events overlapping a non-local event that hold at "
                "least one row with No-Spike above 0.8",
            ),
            make(
                f"{word}_overlap_all_nonlocal_fraction",
                stats["overlap_all_nonlocal_fraction"],
                "fraction",
                "pct",
                f"share of {detector} events overlapping a non-local event whose every row "
                "is non-local above 0.8",
            ),
        ]
        if main:
            values += [
                make(
                    f"{word}_no_nonlocal_fraction_animal_mean",
                    stats["no_nonlocal_fraction_animal_mean"],
                    "fraction",
                    "pct",
                    f"share of {detector} events that overlap no non-local event, the "
                    "mean over animals of each animal's share (the mean over its "
                    "epochs of the epoch's share; Figure 2c's bar is 1 minus it)",
                    tags=TAGS["swr_content"],
                ),
                make(
                    f"{word}_mostly_local_fraction_animal_mean",
                    stats["mostly_local_fraction_animal_mean"],
                    "fraction",
                    "pct",
                    f"share of {detector} events that overlap no non-local event and "
                    "have Local above 0.8 in more than half their rows, the mean over "
                    "animals of each animal's share (the mean over its epochs; Figure "
                    "2c's bar)",
                    tags=TAGS["swr_content"],
                ),
                make(
                    f"{word}_no_nonlocal_fraction_animal_min",
                    stats["no_nonlocal_fraction_animal_min"],
                    "fraction",
                    "pctnum",
                    f"lowest per-animal share of {detector} events overlapping no non-local "
                    "event (an animal's share is the mean over its epochs of the epoch's "
                    "share)",
                ),
                make(
                    f"{word}_no_nonlocal_fraction_animal_max",
                    stats["no_nonlocal_fraction_animal_max"],
                    "fraction",
                    "pct",
                    f"highest per-animal share of {detector} events overlapping no non-local "
                    "event (the mean over the animal's epochs)",
                ),
            ]
    return values


#: The detectors whose share of the non-local events found is a value for the
#: text, beside Figure 2e's: the Widloski detector's criteria-only inventory
#: (its candidate windows before the shuffle test), which asks whether the
#: Widloski detector finds few non-local events because of its shuffle test.
TEXT_FOUND_DETECTORS = ("memoryless_criteria",)


def nonlocal_found_values(memberships: pd.DataFrame, source: str) -> list[Value]:
    """Question 2b.3 and Figure 2e's right bars: for each detector the panel
    draws (``nonlocal_comparison.FIGURE_DETECTORS``), the share of the
    selected non-local events that overlap one of its events: the mean over
    animals of each animal's share (the panel's bar), the lowest and highest
    animal's share, and the share pooled over events (no figure draws it)
    (``nonlocal_comparison.nonlocal_events_overlapped``, which the figure
    draws); and the same, for the text and with no figure, for
    :data:`TEXT_FOUND_DETECTORS`.

    Parameters
    ----------
    memberships : pd.DataFrame
        ``nonlocal_comparison.nonlocal_event_memberships`` of the selected
        non-local events.
    source : str
    """
    make_figure = value_factory(nonlocal_found_values, source, TAGS["nonlocal_found"])
    make_text = value_factory(nonlocal_found_values, source, TAGS["other_detectors"])
    words = {detector: word for word, detector in DETECTORS.items()}
    overlap = nonlocal_comparison.nonlocal_events_overlapped(
        memberships, (*nonlocal_comparison.FIGURE_DETECTORS, *TEXT_FOUND_DETECTORS)
    )
    values = []
    for detector, row in overlap.iterrows():
        make = make_text if detector in TEXT_FOUND_DETECTORS else make_figure
        key = f"nonlocal_events_found_by_{words[detector]}_fraction"
        values += _animal_range_values(
            make,
            key,
            (
                row["pooled"],
                float(np.min(row["animals"])),
                float(np.max(row["animals"])),
            ),
            (
                f"share of the non-local events that a {detector} event overlaps "
                "(of those whose overlap is known, inside the two detectors' "
                "shared exposure), pooled over events",
                f"lowest per-animal share of the non-local events that a {detector} "
                "event overlaps (an animal's share is the mean over its epochs of "
                "the epoch's share)",
                f"highest per-animal share of the non-local events that a {detector} "
                "event overlaps (the mean over the animal's epochs)",
            ),
            animal_mean=(
                row["animal_mean"],
                f"share of the non-local events that a {detector} event overlaps, "
                "the mean over animals of each animal's share (the mean over its "
                "epochs of the epoch's share)",
            ),
            pooled_tags=TAGS["other_detectors"],
        )
    return values


def detections_with_nonlocal_values(events: pd.DataFrame, source: str) -> list[Value]:
    """Question 2b.3: the share of a detector's events that overlap a non-local
    event (``nonlocal_comparison.events_overlapping_nonlocal``). For each
    detector Figure 2e's left bars draw (``nonlocal_comparison.FIGURE_DETECTORS``),
    tagged to the panel: the mean over animals of each animal's share (the
    bar), and the lowest and highest animal's (the pooled share is 1 minus
    ``{word}_no_nonlocal_fraction``). For each of :data:`TEXT_FOUND_DETECTORS`,
    for the text: the share pooled over events, the mean over animals and the
    lowest and highest animal's.

    Parameters
    ----------
    events : pd.DataFrame
        ``nonlocal_comparison.read_comparison``'s ``others``.
    source : str
    """
    make = value_factory(
        detections_with_nonlocal_values, source, TAGS["other_detectors"]
    )
    make_figure = value_factory(
        detections_with_nonlocal_values, source, TAGS["nonlocal_found"]
    )
    words = {detector: word for word, detector in DETECTORS.items()}
    values = []
    figure_shares = nonlocal_comparison.events_overlapping_nonlocal(
        events, nonlocal_comparison.FIGURE_DETECTORS
    )
    for detector, row in figure_shares.iterrows():
        key = f"{words[detector]}_with_nonlocal_fraction"
        values += [
            make_figure(
                f"{key}_animal_mean",
                row["animal_mean"],
                "fraction",
                "pct",
                f"share of the {detector} events that overlap a non-local event, the "
                "mean over animals of each animal's share (an animal's share is the "
                "mean over its epochs of the epoch's share)",
            ),
            make_figure(
                f"{key}_animal_min",
                float(np.min(row["animals"])),
                "fraction",
                "pctnum",
                f"lowest per-animal share of the {detector} events that overlap a "
                "non-local event (the mean over the animal's epochs)",
            ),
            make_figure(
                f"{key}_animal_max",
                float(np.max(row["animals"])),
                "fraction",
                "pct",
                f"highest per-animal share of the {detector} events that overlap a "
                "non-local event (the mean over the animal's epochs)",
            ),
        ]
    shares = nonlocal_comparison.events_overlapping_nonlocal(
        events, TEXT_FOUND_DETECTORS
    )
    for detector, row in shares.iterrows():
        values += _animal_range_values(
            make,
            f"{words[detector]}_with_nonlocal_fraction",
            (
                row["pooled"],
                float(np.min(row["animals"])),
                float(np.max(row["animals"])),
            ),
            (
                f"share of the {detector} events that overlap a non-local event, "
                "pooled over events",
                f"lowest per-animal share of the {detector} events that overlap a "
                "non-local event (an animal's share is the mean over its epochs of "
                "the epoch's share)",
                f"highest per-animal share of the {detector} events that overlap a "
                "non-local event (the mean over the animal's epochs)",
            ),
            animal_mean=(
                row["animal_mean"],
                f"share of the {detector} events that overlap a non-local event, the "
                "mean over animals of each animal's share (the mean over its epochs "
                "of the epoch's share)",
            ),
        )
    return values


# --- what an analysis without the non-local detector leaves out -----------------


def left_out_values(
    run_dir: Path,
    inventory_dir: Path,
    event_ids: Iterable[object],
    visits: Mapping[str, tuple[np.ndarray, np.ndarray]],
    exposure: pd.DataFrame,
    source: str,
) -> list[Value]:
    """Question 2b.2: how many of the selected non-local events, and how much of
    their time, an analysis that starts from SWRs (or from any LFP- or
    rate-based trigger, or from the Widloski detector) does not include, in
    total and per hour of rest inside the well visits
    (``nonlocal_comparison.left_out_statistics``)."""
    make = value_factory(left_out_values, source, TAGS["left_out"])
    stats = nonlocal_comparison.left_out_statistics(
        run_dir, inventory_dir, event_ids, visits, exposure
    )
    values = [
        make(
            "rest_hours",
            stats["rest_hours"],
            "quantity",
            "sig2",
            "hours of analyzed immobility inside a well visit the non-local detector "
            "observed (its exposure inside the visits), summed over epochs",
            "h",
        ),
        make(
            "nonlocal_time_s",
            stats["nonlocal_time_s"],
            "quantity",
            "int",
            "seconds of non-local event time (compared events), summed over epochs",
            "s",
        ),
        make(
            "kay_time_s",
            stats["kay_time_s"],
            "quantity",
            "int",
            "seconds the Kay SWRs whose first sample is inside a well visit span "
            "(compared events): what an SWR-triggered analysis examines",
            "s",
        ),
    ]
    for word, label in (
        ("kay", "Kay's SWRs"),
        ("swr", "the five SWR definitions together"),
        ("trigger", "the five SWR definitions and MUA together"),
        ("widloski", "the Widloski detector's events"),
    ):
        values += [
            make(
                f"nonlocal_time_outside_{word}_s",
                stats[f"nonlocal_time_outside_{word}_s"],
                "quantity",
                "int",
                f"seconds of non-local event time that {label} do not cover",
                "s",
            ),
            make(
                f"nonlocal_time_outside_{word}_fraction",
                stats[f"nonlocal_time_outside_{word}_fraction"],
                "fraction",
                "pct",
                f"share of non-local event time that {label} do not cover",
            ),
        ]
    for word, label in (
        ("kay", "no Kay SWR overlaps"),
        ("widloski", "no Widloski event overlaps"),
    ):
        values += [
            make(
                f"nonlocal_events_outside_{word}",
                stats[f"nonlocal_events_outside_{word}"],
                "count",
                "count",
                f"non-local events {label} (detector comparison memberships)",
            ),
            make(
                f"nonlocal_events_outside_{word}_fraction",
                stats[f"nonlocal_events_outside_{word}_fraction"],
                "fraction",
                "pct",
                f"share of the non-local events with a known membership that {label}",
            ),
            make(
                f"nonlocal_events_outside_{word}_per_hour",
                stats[f"nonlocal_events_outside_{word}_per_hour"],
                "quantity",
                "sig2",
                f"non-local events {label}, per hour of analyzed immobility inside a "
                "well visit",
                "per h",
            ),
        ]
    values += [
        make(
            "nonlocal_events_outside_trigger",
            stats["nonlocal_events_outside_trigger"],
            "count",
            "count",
            "non-local events no SWR definition and no MUA event overlaps "
            "(cell sequence_matched_by_no_trigger)",
        ),
        make(
            "nonlocal_events_outside_trigger_fraction",
            stats["nonlocal_events_outside_trigger_fraction"],
            "fraction",
            "pct",
            "share of the non-local events with a known membership that no SWR "
            "definition and no MUA event overlaps",
        ),
        make(
            "nonlocal_events_outside_trigger_per_hour",
            stats["nonlocal_events_outside_trigger_per_hour"],
            "quantity",
            "sig2",
            "non-local events no SWR definition and no MUA event overlaps, per hour of "
            "analyzed immobility inside a well visit",
            "per h",
        ),
    ]
    return values


# --- theta, ripple and MUA during events ------------------------------------------

_SIGNAL_EVENT_COLUMNS = {
    "theta": "theta_z",
    "ripple": "ripple_power_z",
    "mua": "mua_rate_z",
    "slow_gamma": "slow_gamma_z",
    "fast_gamma": "fast_gamma_z",
}
_SIGNAL_LABELS = {
    "theta": "theta power",
    "ripple": "ripple power",
    "mua": "MUA rate",
    "slow_gamma": "slow-gamma power (25-55 Hz)",
    "fast_gamma": "fast-gamma power (65-100 Hz)",
}


def brain_state_values(
    events: pd.DataFrame, reference: pd.DataFrame, source: str
) -> list[Value]:
    """Questions 2d.8 and 2b.5: theta, ripple power and MUA during events, against
    running and quiet rest.

    Each event's value is its mean over its bins of the epoch's z-scored signal
    (the event table's ``theta_z``, ``ripple_power_z``, ``mua_rate_z``); the
    references are each epoch's median during running (>10 cm/s) and quiet rest
    (<4 cm/s, in no event), on the same scale (``brain_state_reference``). An
    event is theta-like when its theta reaches its epoch's running median.
    """
    makes = {
        word: value_factory(
            brain_state_values,
            source,
            {"theta": TAGS["theta"], "ripple": TAGS["ripple"], "mua": TAGS["mua"]}[
                word
            ],
        )
        for word in ("theta", "ripple", "mua")
    }
    return _signal_values(makes, events, reference)


def gamma_z_values(
    events: pd.DataFrame, reference: pd.DataFrame, source: str
) -> list[Value]:
    """Question 2d.8: slow and fast gamma during events, against quiet-rest and
    running windows as long as the events.

    On the gamma pass's ``slow_gamma_z`` and ``fast_gamma_z``, joined on
    ``event_id``. An event's value is the mean over its bins, and band power is
    skewed, so it is compared with the level of windows of the same length
    (``gamma_power.matched_level``), each class with its own: not with the
    median of single bins, which a window's mean exceeds when nothing differs.
    """
    make = value_factory(gamma_z_values, source, TAGS["gamma_z"])
    words = ["slow_gamma", "fast_gamma"]
    levels = event_statistics.matched_reference_medians(reference, words)
    statistics = event_statistics.event_signal_statistics(
        events,
        reference,
        {word: _SIGNAL_EVENT_COLUMNS[word] for word in words},
        matched=True,
    )
    values: list[Value] = []
    for word in words:
        label = _SIGNAL_LABELS[word]
        for group, is_swr in (("non_swr", False), ("swr", True)):
            stats = statistics.loc[(word, is_swr)]
            name = "non-SWR" if group == "non_swr" else "SWR-overlapping"
            animal_value = (
                f"animals whose median {name} event {label} (the median over their "
                "epochs of each epoch's median)"
            )
            values.append(
                make(
                    f"{word}_median_{group}",
                    stats["median"],
                    "quantity",
                    "dec2",
                    f"median z-scored {label} of immobile {name} non-local events (event mean)",
                    "z",
                )
            )
            for regime, words_of_regime in (
                ("rest", "quiet-rest"),
                ("running", "running"),
            ):
                values.append(
                    make(
                        f"{word}_{regime}_matched_reference_{group}",
                        levels.loc[(word, is_swr), regime],
                        "quantity",
                        "dec2",
                        f"median over epochs of the epoch's z-scored {label} in "
                        f"{words_of_regime} windows as long as its {name} events (the "
                        "median over the events of one window's mean each, averaged "
                        "over the draws)",
                        "z",
                    )
                )
            values += [
                make(
                    f"{word}_animals_below_running_{group}",
                    stats["n_animals_below_running"],
                    "count",
                    "count",
                    f"{animal_value} is below the median over the same epochs of the "
                    "epoch's level in running windows as long as those events",
                ),
                make(
                    f"{word}_animals_above_rest_{group}",
                    stats["n_animals_above_rest"],
                    "count",
                    "count",
                    f"{animal_value} is above the median over the same epochs of the "
                    "epoch's level in quiet-rest windows as long as those events",
                ),
            ]
    return values


def _signal_values(
    makes: dict[str, Callable[..., Value]],
    events: pd.DataFrame,
    reference: pd.DataFrame,
) -> list[Value]:
    """The values of each signal in ``makes``, made by its maker: its running and
    rest references, and each class's median and animal counts (and, for theta,
    the theta-like share)."""
    words = list(makes)
    levels = event_statistics.reference_medians(reference, words)
    statistics = event_statistics.event_signal_statistics(
        events, reference, {word: _SIGNAL_EVENT_COLUMNS[word] for word in words}
    )
    values: list[Value] = []
    for word, make in makes.items():
        label = _SIGNAL_LABELS[word]
        values += [
            make(
                f"{word}_running_reference",
                levels.loc[word, "running"],
                "quantity",
                "dec2",
                f"median over epochs of the epoch's median z-scored {label} while running",
                "z",
            ),
            make(
                f"{word}_rest_reference",
                levels.loc[word, "rest"],
                "quantity",
                "dec2",
                f"median over epochs of the epoch's median z-scored {label} in quiet rest",
                "z",
            ),
        ]
        for group, is_swr in (("non_swr", False), ("swr", True)):
            stats = statistics.loc[(word, is_swr)]
            name = "non-SWR" if group == "non_swr" else "SWR-overlapping"
            values += [
                make(
                    f"{word}_median_{group}",
                    stats["median"],
                    "quantity",
                    "dec2",
                    f"median z-scored {label} of immobile {name} non-local events (event mean)",
                    "z",
                ),
                make(
                    f"{word}_animals_below_running_{group}",
                    stats["n_animals_below_running"],
                    "count",
                    "count",
                    f"animals whose median {name} event {label} (the median over their "
                    "epochs of each epoch's median) is below the median over the same "
                    "epochs of the epoch's median running level",
                ),
                make(
                    f"{word}_animals_above_rest_{group}",
                    stats["n_animals_above_rest"],
                    "count",
                    "count",
                    f"animals whose median {name} event {label} (the median over their "
                    "epochs of each epoch's median) is above the median over the same "
                    "epochs of the epoch's median quiet-rest level",
                ),
            ]
            if word == "theta":
                values.append(
                    make(
                        f"theta_like_fraction_{group}",
                        stats["reaches_running_fraction"],
                        "fraction",
                        "pct",
                        f"share of immobile {name} non-local events whose theta power "
                        "reaches their epoch's median running level",
                    )
                )
    return values


def gamma_db_values(reference: pd.DataFrame, source: str) -> list[Value]:
    """Question 2d.8: slow and fast gamma during events, as a change in band power
    against the same epoch's quiet rest, in dB.

    For each epoch and class: ``10 log10(mean band power over the class's event
    samples / mean band power over quiet rest)`` (below 4 cm/s, in no non-local,
    SWR or MUA event, outside excluded time), on the same band power the z
    values are made from (``scripts/event_properties/measure_event_gamma.py``). The
    median over epochs, and the animals whose median over their epochs is above
    0 dB. The z values (``gamma_z_values``) stay as they are.
    """
    make = value_factory(gamma_db_values, source, TAGS["brain_state"])
    bands = ["slow_gamma", "fast_gamma"]
    statistics = event_statistics.power_change_statistics(reference, bands)
    values: list[Value] = []
    for word in bands:
        label = _SIGNAL_LABELS[word]
        for group, is_swr in (("non_swr", False), ("swr", True)):
            name = "non-SWR" if group == "non_swr" else "SWR-overlapping"
            stats = statistics.loc[(word, is_swr)]
            values += [
                make(
                    f"{word}_db_median_{group}",
                    stats["median"],
                    "quantity",
                    "sig2s",
                    f"median over epochs of the change in {label} during {name} events "
                    "against the epoch's quiet rest: 10 log10 of the ratio of mean band "
                    "powers",
                    "dB",
                ),
                make(
                    f"{word}_db_animals_above_rest_{group}",
                    stats["n_animals_above_rest"],
                    "count",
                    "count",
                    f"animals whose median over epochs of that change ({name} events) "
                    "is above 0 dB",
                ),
            ]
    return values


# --- spike support ----------------------------------------------------------

_SPIKE_EVIDENCE_GROUPS = {
    "swr_remote": "SWR-remote",
    "non_swr_remote": "nonSWR-remote",
    "non_swr_close": "nonSWR-close",
}


def spike_evidence_metrics(per_epoch_dir: Path, n_epochs: int) -> pd.DataFrame:
    """The spike-evidence run's per-event metrics, one table per epoch, concatenated.

    Parameters
    ----------
    per_epoch_dir : Path
        The run's folder, with one folder per epoch.
    n_epochs : int
        The epochs the run must hold: it scores every analyzed epoch.

    Raises
    ------
    ValueError
        The run does not hold ``n_epochs`` epochs, or an epoch's table names
        another epoch.
    """
    paths = sorted(per_epoch_dir.glob(f"*/{SPIKE_EVIDENCE_METRICS}"))
    if len(paths) != n_epochs:
        raise ValueError(
            f"{per_epoch_dir} holds {len(paths)} epochs' metrics, not {n_epochs}"
        )
    tables = []
    for path in paths:
        table = pd.read_parquet(path)
        if set(table["epoch_id"]) != {path.parent.name}:
            raise ValueError(f"{path} holds epochs {sorted(set(table['epoch_id']))}")
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def spike_evidence_values(metrics: pd.DataFrame, source: str) -> list[Value]:
    """Questions 2c.3, 2c.4 and 2d.9: each event group's spike evidence against matched
    quiet windows (``baseline``), the negative control. Event-level medians, pooled
    over epochs; the run scores every event and every quiet window with at least one
    spike (no cap; the evidence is per spike, so a quiet window without spikes has
    none)."""
    make = value_factory(spike_evidence_values, source, TAGS["spike"])
    values = [
        make(
            "evidence_epochs",
            metrics["epoch_id"].nunique(),
            "count",
            "count",
            "epochs in the spike-evidence run",
        ),
    ]
    statistics = event_statistics.spike_evidence_statistics(metrics, "baseline")
    groups = {"baseline": "baseline", **_SPIKE_EVIDENCE_GROUPS}
    for word, group in groups.items():
        stats = statistics.loc[group]
        label = "matched quiet windows" if word == "baseline" else f"{group} events"
        values += [
            make(
                f"{word}_scored",
                stats["n_scored"],
                "count",
                "count",
                f"{label} scored",
                tags=TAGS["spike"],
            ),
            make(
                f"{word}_spikes_median",
                stats["spikes_median"],
                "quantity",
                "int",
                f"median spike count of {label}",
                "spikes",
                tags=TAGS["spike_count"],
            ),
            make(
                f"{word}_tetrodes_median",
                stats["tetrodes_median"],
                "quantity",
                "int",
                f"median number of tetrodes with a spike in {label}",
                "tetrodes",
                tags=TAGS["spike_count"],
            ),
            make(
                f"{word}_dll_per_spike_median",
                stats["dll_per_spike_median"],
                "quantity",
                "dec2",
                f"median log-likelihood ratio per spike, decoded location against the "
                f"animal's, of {label}",
                "nats per spike",
                tags=TAGS["spike_evidence"],
            ),
            make(
                f"{word}_gpi_distance_median_cm",
                stats["gpi_distance_median_cm"],
                "quantity",
                "int",
                f"median track distance from the MAP of {label} to the ground-process "
                "intensity peak",
                "cm",
                tags=TAGS["spike_gpi"],
            ),
        ]
        if word == "baseline":
            continue
        values += [
            make(
                f"{word}_epochs_above_baseline",
                stats["n_epochs_above_baseline"],
                "count",
                "count",
                f"epochs whose median {group} event scores above the epoch's median quiet "
                "window, per spike",
                tags=TAGS["spike_evidence"],
            ),
            make(
                f"{word}_epochs_scored",
                stats["n_epochs_scored"],
                "count",
                "count",
                f"epochs with a scored {group} event",
                tags=TAGS["spike_evidence"],
            ),
            make(
                f"{word}_animals_above_baseline",
                stats["n_animals_above_baseline"],
                "count",
                "count",
                f"animals whose median {group} event scores above their median quiet "
                "window, per spike, each the median over the animal's epochs of the "
                "epoch's median",
                tags=TAGS["spike_evidence"],
            ),
        ]
    values.append(
        make(
            "evidence_animals",
            metrics["animal"].nunique(),
            "count",
            "count",
            "animals in the spike-evidence run",
        )
    )
    return values


def spike_evidence_visit_values(
    metrics: pd.DataFrame, kept: np.ndarray, source: str
) -> list[Value]:
    """Questions 2c.3 and 2d.9: the scored events and quiet windows of each group
    left out because they start outside every well visit
    (``spike_evidence.in_analyzed_subset``)."""
    make = value_factory(spike_evidence_visit_values, source, TAGS["spike"])
    left_out = metrics.loc[~kept, "group"].value_counts()
    groups = {"baseline": BASELINE_GROUP, **_SPIKE_EVIDENCE_GROUPS}
    return [
        make(
            f"{word}_left_out",
            int(left_out.get(group, 0)),
            "count",
            "count",
            (
                "matched quiet windows scored whose first sample is outside every "
                "well visit, left out"
                if word == "baseline"
                else f"{group} events scored that start outside every well visit, "
                "left out"
            ),
        )
        for word, group in groups.items()
    ]


def heldout_by_swr_values(stats: pd.DataFrame, source: str) -> list[Value]:
    """Question 2c.3: the held-out tetrode test of immobile Non-Local Continuous bouts,
    split by SWR, per held-out spike against the animal's position."""
    make = value_factory(heldout_by_swr_values, source, TAGS["spike_evidence"])
    rows = stats[
        (stats["regime"] == "immobile") & (stats["metric"] == "runcv_nl_anim_mark_ps")
    ]
    values = []
    for word, state in (
        ("swr", "Non-Local Continuous | SWR"),
        ("non_swr", "Non-Local Continuous | no SWR"),
    ):
        (row,) = rows[rows["state"] == state].itertuples()
        positive, total = (int(n) for n in row.animals_pos.split("/"))
        label = "with an SWR" if word == "swr" else "without an SWR"
        values += [
            make(
                f"heldout_{word}_per_spike",
                row.animal_estimate,
                "quantity",
                "dec2",
                f"held-out evidence per held-out spike for immobile Non-Local Continuous "
                f"bouts {label}, against the animal's position: mean over animals",
                "nats per spike",
            ),
            make(
                f"heldout_{word}_per_spike_ci_low",
                row.animal_ci_lo,
                "quantity",
                "dec2",
                f"lower bound of its 95 % interval (nested bootstrap of the animals "
                f"and their epochs, small-sample t correction over the {total} "
                "animals)",
                "nats per spike",
            ),
            make(
                f"heldout_{word}_per_spike_ci_high",
                row.animal_ci_hi,
                "quantity",
                "dec2",
                "upper bound of that interval",
                "nats per spike",
            ),
            make(
                f"heldout_{word}_animals_positive",
                positive,
                "count",
                "count",
                "animals with a positive mean",
            ),
            make(
                f"heldout_{word}_animals",
                total,
                "count",
                "count",
                "animals in that test",
            ),
        ]
    return values


def heldout_bout_values(bouts: pd.DataFrame, source: str) -> list[Value]:
    """Question 2c.3: the immobile Non-Local Continuous bouts of the held-out test
    split by SWR that start inside a well visit and are read, those left out, and
    the read ones whose matched window starts outside every visit (the matched
    windows were drawn from all immobile time). ``bouts`` is the split's count
    table (``state_evidence.bout_visit_counts``, recorded in its provenance)."""
    make = value_factory(heldout_bout_values, source, TAGS["spike_evidence"])
    rows = bouts[bouts["regime"] == "immobile"].set_index("state")
    values = []
    for word, state in (
        ("swr", "Non-Local Continuous | SWR"),
        ("non_swr", "Non-Local Continuous | no SWR"),
    ):
        row = rows.loc[state]
        label = "with an SWR" if word == "swr" else "without an SWR"
        values += [
            make(
                f"heldout_{word}_bouts_in_subset",
                row["n_kept"],
                "count",
                "count",
                f"immobile Non-Local Continuous bouts {label} whose first sample is "
                "inside a well visit, read",
            ),
            make(
                f"heldout_{word}_bouts_left_out",
                row["n_left_out"],
                "count",
                "count",
                f"immobile Non-Local Continuous bouts {label} whose first sample is "
                "outside every well visit, left out",
            ),
            make(
                f"heldout_{word}_bouts_matched_outside_visit",
                row["n_kept_matched_outside_visit"],
                "count",
                "count",
                f"of the read bouts {label}, those whose matched window starts "
                "outside every well visit",
            ),
        ]
    return values


# --- what the events are -----------------------------------------------------------

#: The event groups of the per-event runs, by the word their keys use.
_RUN_GROUPS = {"non_swr": SWR_LABELS[False], "swr": SWR_LABELS[True]}


def state_composition_values(events: pd.DataFrame, source: str) -> list[Value]:
    """Question 2d.10: the mean share of an SWR or no-SWR event in Continuous and in
    Fragmented (Figure 3b)."""
    make = value_factory(state_composition_values, source, TAGS["state_composition"])
    statistics = event_statistics.state_composition_statistics(events)
    values = []
    for word, group in _RUN_GROUPS.items():
        for state in event_statistics.STATE_SHARE_COLUMNS:
            values.append(
                make(
                    f"{word}_{state}_share",
                    statistics.loc[group, f"{state}_animal_mean"],
                    "fraction",
                    "pct",
                    f"mean share of a {group} event in {state.capitalize()} (each "
                    "event weighted alike, whatever its length): the mean "
                    "over animals of each animal's mean over epochs of each epoch's "
                    "mean over events of the event's mean state probability",
                )
            )
    return values


# --- population events and spatial evidence --------------------------------------


def participation_values(table: pd.DataFrame, source: str) -> list[Value]:
    """Question 2d.9: tetrode participation and spike rate of the events against
    Local-rest windows of their length (Figure 2h shows
    the participation)."""
    makers = {
        "participation": value_factory(
            participation_values, source, TAGS["participation"]
        ),
        "spike_rate_hz": value_factory(
            participation_values, source, TAGS["spike_rate"]
        ),
    }
    statistics = participation.participation_statistics(table)
    differences = participation.participation_differences(statistics)
    values = []
    for word, group in _RUN_GROUPS.items():
        for measure, style, what, unit in (
            ("participation", "pct", "share of tetrodes with a spike", ""),
            ("spike_rate_hz", "int", "spike rate summed over tetrodes", "spikes per s"),
        ):
            kind = "fraction" if measure == "participation" else "quantity"
            make = makers[measure]
            for window, label in (
                ("event", "events"),
                ("control", "Local-rest windows"),
            ):
                values.append(
                    make(
                        f"{word}_{window}_{measure}",
                        statistics.loc[(group, window), f"{measure}_animal_mean"],
                        kind,
                        style,
                        f"{what} of {group} {label}: mean over animals of each "
                        "animal's median over epochs of each epoch's median",
                        unit,
                    )
                )
            values.append(
                make(
                    f"{word}_{measure}_animals_above_rest",
                    differences.loc[group, f"{measure}_n_animals_above"],
                    "count",
                    "count",
                    f"animals whose {group} events' {what} is above their "
                    "Local-rest windows'",
                )
            )
        values.append(
            makers["participation"](
                f"{word}_participation_animals",
                differences.loc[group, "n_animals"],
                "count",
                "count",
                f"animals with {group} events and Local-rest windows",
            )
        )
    values.append(
        makers["participation"](
            "participation_events_without_control",
            int((table["kind"] == "event").sum() - (table["kind"] == "control").sum()),
            "count",
            "count",
            "events longer than any stretch of Local rest inside one well visit of "
            "their epoch (no control)",
        )
    )
    return values


def _per_spike_interval(
    make: Callable[..., Value],
    key: str,
    row: pd.Series,
    description: str,
    style: str = "dec2",
    unit: str = "nats per spike",
) -> list[Value]:
    """A per-animal estimate, per spike unless ``unit`` says otherwise, and the
    bounds of its 95 % interval (``likelihood_advantage_statistics``'
    ``estimate``, ``ci_low``, ``ci_high``)."""
    return [
        make(key, row["estimate"], "quantity", style, description, unit),
        make(
            f"{key}_ci_low",
            row["ci_low"],
            "quantity",
            style,
            "lower bound of its 95 % interval (nested bootstrap of the animals "
            "and their epochs, small-sample t correction over the "
            f"{row['n_animals']} animals)",
            unit,
        ),
        make(
            f"{key}_ci_high",
            row["ci_high"],
            "quantity",
            style,
            "upper bound of that interval",
            unit,
        ),
    ]


def likelihood_advantage_values(table: pd.DataFrame, source: str) -> list[Value]:
    """Question 2c.3: the spike likelihood at the decoded positions minus at the
    animal's, per spike, over each whole event (in sample; the held-out tetrode
    supplementary figure shows the held-out score)."""
    make = value_factory(
        likelihood_advantage_values, source, TAGS["likelihood_advantage"]
    )
    statistics = likelihood_advantage.likelihood_advantage_statistics(table)
    values = []
    for word, group in _RUN_GROUPS.items():
        row = statistics.loc[group]
        values += _per_spike_interval(
            make,
            f"{word}_likelihood_advantage_per_spike",
            row,
            f"log-likelihood of {group} events' spikes at the decoded positions "
            "minus at the animal's, summed over each event and divided by its "
            "spikes: mean over animals of each animal's median over epochs of "
            "each epoch's median",
        )
        values += [
            make(
                f"{word}_likelihood_advantage_events_scored",
                row["n_scored"],
                "count",
                "count",
                f"{group} events with a per-spike score (at least one spike)",
            ),
            make(
                f"{word}_likelihood_advantage_events",
                row["n_events"],
                "count",
                "count",
                f"{group} events scored",
            ),
        ]
    return values


_TETRODE_HELD_OUT = (
    "each tetrode is left out of the decoder in turn; at a scored bin, that "
    "tetrode's and the other tetrodes' place-field encodings are fit on the other "
    f"folds ({N_FOLDS} folds of whole trials balanced by "
    f"destination well, a segment longer than {MAX_SEGMENT_S:g} s "
    "cut into pieces, training bins within "
    f"{tetrode_holdout.TRIAL_MARGIN_S:g} s of the fold left out); the state "
    "transitions and initial conditions are the canonical decoder's, fit on the "
    "whole epoch with every tetrode; the posterior is smoothed over the whole "
    "epoch; the windows are the canonical events and their controls, chosen from "
    "the decode of every tetrode, so the choice of time is not held out"
)
"""How the held-out tetrode score is held out, in words
(``tetrode_holdout.holdout_scores``)."""


#: Each held-out tetrode measure reported: its column of
#: ``tetrode_holdout.window_scores`` and the score in words.
_TETRODE_MEASURES = {
    "gain": (
        "gain_per_spike",
        "each tetrode's log predictive score under the other tetrodes' position "
        "distribution minus under local coding around the animal",
    ),
    "spiking_gain": (
        "spiking_gain_per_spike",
        "each tetrode's log predictive score under the other tetrodes' position "
        "distribution minus under local coding around the animal in the bins "
        "where the scored tetrode spiked (its spikes' waveform terms with the "
        "expected-firing term of those bins); with the silent bins' part it adds "
        "to the gain for each event, but each of the three is a median over "
        "events, so over the cohort they need not add up",
    ),
    "silent_gain": (
        "silent_gain_per_spike",
        "each tetrode's log predictive score under the other tetrodes' position "
        "distribution minus under local coding around the animal in the bins "
        "where the scored tetrode was silent (the expected-firing term alone: "
        "the gain from the absence of spikes); with the spiking bins' part it "
        "adds to the gain for each event, but each of the three is a median over "
        "events, so over the cohort they need not add up",
    ),
    "agreement": (
        "agreement_per_spike",
        "the agreement, each tetrode's log predictive score under the other "
        "tetrodes' position distribution minus under no place preference (the "
        "mean over place bins): above zero when the other tetrodes' "
        "distribution predicts where the scored tetrode points better than no "
        "preference does, that is when the tetrodes share content, local or "
        "not (the gain over local coding is above zero for any non-local "
        "activity the distribution covers, coherent or not)",
    ),
}
_CONTROL_AGREEMENT = (
    "; in the Local-rest controls the tetrodes share content, the animal's own "
    "place, so the controls are not a zero reference for the agreement (they are "
    "one for the gain, which is measured against local coding)"
)
"""What the controls' agreement values add to their description."""


def tetrode_holdout_values(windows: pd.DataFrame, source: str) -> list[Value]:
    """Question 2c.3: the held-out tetrode gain over local coding and the
    agreement, per spike, of the events and their Local-rest controls. The
    events' gain is drawn on Figure 2j (the fit to the decoded place over the
    animal's place) and, with its parts in the bins with and without a spike,
    on Figure 2's supplement (S2h), and carries their tags; the controls' gain
    and the agreement are drawn on no figure."""
    make_figure = value_factory(tetrode_holdout_values, source, TAGS["tetrode_holdout"])
    make_split = value_factory(
        tetrode_holdout_values, source, TAGS["tetrode_holdout_split"]
    )
    make_text = value_factory(
        tetrode_holdout_values, source, TAGS["tetrode_holdout_text"]
    )
    values = []
    estimates = (
        ("gain", "gain", "event", "events"),
        ("spiking_gain", "spiking_gain", "event", "events"),
        ("silent_gain", "silent_gain", "event", "events"),
        ("control_gain", "gain", "control", "Local-rest controls"),
        ("agreement", "agreement", "event", "events"),
        ("control_agreement", "agreement", "control", "Local-rest controls"),
    )
    for name, measure, kind, label in estimates:
        column, words = _TETRODE_MEASURES[measure]
        note = _CONTROL_AGREEMENT if (measure, kind) == ("agreement", "control") else ""
        if measure == "agreement" or kind == "control":
            make = make_text
        elif name == "gain":
            make = make_figure
        else:
            make = make_split
        statistics = tetrode_holdout.holdout_statistics(windows, kind, column)
        for word, group in _RUN_GROUPS.items():
            row = statistics.loc[group]
            values += _per_spike_interval(
                make,
                f"{word}_heldout_{name}_per_spike",
                row,
                f"held-out tetrode score of {group} {label}: {words}, summed over "
                "tetrodes and bins and divided by the window's spikes; "
                f"{_TETRODE_HELD_OUT}{note}; mean over animals of each animal's "
                "median over epochs of each epoch's median",
                # the controls' values are near zero: two significant digits
                style="sig2" if kind == "control" else "dec2",
            )
            if name == "gain":
                values.append(
                    make(
                        f"{word}_heldout_events_scored",
                        row["n_scored"],
                        "count",
                        "count",
                        f"{group} events with a held-out score per spike",
                    )
                )
    return values


#: Each width bin's word in a key (which holds no digit) and its range in words,
#: in the bins' order (``shared_content.WIDTH_BIN_LABELS``).
_WIDTH_BINS = dict(
    zip(
        shared_content.WIDTH_BIN_LABELS,
        (
            ("below_twenty_five_cm", "under 25 cm"),
            ("twenty_five_to_fifty_cm", "25 to 50 cm"),
            ("fifty_to_hundred_cm", "50 to 100 cm"),
            ("hundred_to_two_hundred_cm", "100 to 200 cm"),
            ("two_hundred_cm_or_more", "200 cm or more"),
        ),
        strict=True,
    )
)
_WIDTH_WORDS = (
    "the event's width: the mean over its left-out tetrodes of the mean over the "
    "window's scored bins of the 95 % HPD length of the other tetrodes' position "
    "distribution"
)
_REFERENCE_WORDS = (
    "the shuffled-event reference: the window's spikes scored under the other "
    "tetrodes' position distribution of another event window of the same group "
    "and fold (bins matched by offset from the window's start), minus under no "
    "place preference, summed over tetrodes and the bins scored and divided by "
    "their spikes; a window without a donor has none"
)
_UNMATCHED_DONOR = (
    "the donor is any event window of the same group and fold, not matched for "
    "the width of its estimate, so over all events the reference is a "
    "permutation reference; it is lower the wider the donor's estimate"
)
_MATCHED_DONOR = (
    "width-matched: only the windows whose donor's estimate (its 95 % HPD "
    "length, as the event's width) falls in the same width bin, a random subset "
    "of the bin's windows since the donor is drawn at random within the group "
    "and fold; the reference depends on the donor's width, not the window's"
)
_EXTRA_RUN = (
    "the held-out tetrode run made again with its extra scores "
    "(score_tetrode_holdout.py --extra-scores --null-seed 0)"
)


def shared_content_values(windows: pd.DataFrame, source: str) -> list[Value]:
    """Question 2c.3: during an event, do the tetrodes point to
    the same remote place? Over the events of each SWR group by the width of
    the other tetrodes' position distribution in fixed bins
    (``shared_content.shared_content_statistics``, on ``windows`` of a run with
    ``--extra-scores``): the held-out tetrode agreement per spike, its
    width-matched shuffled-event reference and the agreement minus it event by
    event (over the windows whose donor's estimate is in the same width bin,
    ``shared_content.width_matched``), each with its interval, and the
    matched windows; the share of events whose agreement is above zero; and
    the share of the group's events in the bin, each animal's then their mean
    with the lowest and highest animal, and pooled. No figure draws these by
    width (Figure 2j draws the difference over all widths, from
    :func:`tetrode_holdout_extra_values`); they are for the text."""
    make = value_factory(shared_content_values, source, TAGS["shared_content"])
    make_raw = value_factory(shared_content_values, source, TAGS["shared_content_raw"])
    make_text = value_factory(
        shared_content_values, source, TAGS["shared_content_text"]
    )
    statistics = shared_content.shared_content_statistics(
        windows[windows["kind"] == "event"]
    )
    _, words = _TETRODE_MEASURES["agreement"]
    reduction = (
        "mean over animals of each animal's median over epochs of each epoch's median"
    )
    values = []
    for label, (bin_word, width) in _WIDTH_BINS.items():
        for word, group in _RUN_GROUPS.items():
            row = statistics.agreement.loc[(label, group)]
            reference = statistics.reference.loc[(label, group)]
            difference = statistics.difference.loc[(label, group)]
            key = f"{word}_shared_content_{bin_word}"
            where = f"{group} events whose width is {width} ({_WIDTH_WORDS})"
            values += _per_spike_interval(
                make_raw,
                f"{key}_agreement_per_spike",
                row,
                f"held-out tetrode agreement of the {where}: {words}, summed over "
                "tetrodes and bins and divided by the event's spikes; above zero "
                "only when the left-out tetrode's activity is where the other "
                f"tetrodes point; {_TETRODE_HELD_OUT}; {_EXTRA_RUN}; {reduction}",
            )
            values += _per_spike_interval(
                make_raw,
                f"{key}_reference_per_spike",
                reference,
                f"{_REFERENCE_WORDS}, of the {where} whose donor's estimate is "
                f"in the same width bin ({_MATCHED_DONOR}); {_TETRODE_HELD_OUT}; "
                f"{_EXTRA_RUN}; {reduction}",
            )
            values += _interval_and_animals_positive(
                make,
                f"{key}_agreement_minus_reference_per_spike",
                difference,
                "held-out tetrode agreement minus the shuffled-event reference, "
                f"event by event, of the {where} whose donor's estimate is in the "
                f"same width bin ({_MATCHED_DONOR}; {words}; {_REFERENCE_WORDS}); "
                f"{_TETRODE_HELD_OUT}; {_EXTRA_RUN}; {reduction}",
            )
            values += [
                make(
                    f"{key}_share_positive",
                    row["share_positive"],
                    "fraction",
                    "pct",
                    f"share of the {where} whose agreement per spike is above zero: "
                    "per epoch, then each animal's median over its epochs, then the "
                    "mean over animals",
                ),
                make(
                    f"{key}_share_positive_animal_min",
                    row["share_positive_animal_min"],
                    "fraction",
                    "pctnum",
                    "lowest of the animals' values of that share",
                ),
                make(
                    f"{key}_share_positive_animal_max",
                    row["share_positive_animal_max"],
                    "fraction",
                    "pct",
                    "highest of the animals' values of that share",
                ),
                make_text(
                    f"{key}_share_of_events",
                    row["share_of_events"],
                    "fraction",
                    "pct",
                    f"share of the {group} events with a held-out score per spike "
                    f"whose width is {width}, pooled over every event",
                ),
                make(
                    f"{key}_share_of_events_animal_mean",
                    row["share_of_events_animal_mean"],
                    "fraction",
                    "pct",
                    f"share of the {group} events with a held-out score per spike "
                    f"whose width is {width}: each epoch's share, each animal's mean "
                    "over its epochs, then the mean over animals",
                ),
                make(
                    f"{key}_share_of_events_animal_min",
                    row["share_of_events_animal_min"],
                    "fraction",
                    "pctnum",
                    "lowest of the animals' values of that share",
                ),
                make(
                    f"{key}_share_of_events_animal_max",
                    row["share_of_events_animal_max"],
                    "fraction",
                    "pct",
                    "highest of the animals' values of that share",
                ),
                make(
                    f"{key}_epochs",
                    row["n_epochs"],
                    "count",
                    "count",
                    f"epochs with {group} events whose width is {width}",
                ),
                make(
                    f"{key}_events",
                    row["n_scored"],
                    "count",
                    "count",
                    f"{group} events with a held-out score per spike whose width "
                    f"is {width}",
                ),
                make(
                    f"{key}_width_matched_events",
                    reference["n_scored"],
                    "count",
                    "count",
                    "of those, events with a width-matched shuffled-event "
                    "reference (a donor whose estimate is in the same width bin "
                    "and a spike in the bins it scored)",
                ),
            ]
    return values


#: Each measure of the held-out tetrode run with its extra scores: its column
#: of ``tetrode_holdout.window_scores`` and the score in words.
_EXTRA_MEASURES = {
    "gain_rate": (
        "gain_rate_per_spike",
        "the part of the gain over local coding that the firing-rate term gives "
        "(the expected spikes and the spike count, not the waveform marks)",
    ),
    "gain_mark": (
        "gain_mark_per_spike",
        "the part of the gain over local coding that only the waveform marks "
        "give (the gain minus its firing-rate part)",
    ),
    "agreement_rate": (
        "agreement_rate_per_spike",
        "the part of the agreement that the firing-rate term gives",
    ),
    "agreement_mark": (
        "agreement_mark_per_spike",
        "the part of the agreement that only the waveform marks give",
    ),
    "agreement_reference": (
        "agreement_null_per_spike",
        f"{_REFERENCE_WORDS}; {_UNMATCHED_DONOR}",
    ),
    "agreement_minus_reference": (
        "agreement_minus_null_per_spike",
        "the agreement minus the shuffled-event reference, event by event (an "
        f"event without a donor has no value); {_UNMATCHED_DONOR}",
    ),
    "mark_information": (
        "mark_information_per_spike",
        "the place information of the scored spikes' marks: for each spike, log "
        "of the places minus the entropy of the conditional density of its mark "
        "over place (the rate map divided out), under its fold's encoding, "
        "summed over the window's spikes and divided by the spikes counted",
    ),
    "spike_information": (
        "spike_information_per_spike",
        "the place information of the scored spikes: for each spike, log of the "
        "places minus the entropy of its whole per-spike term over place (rate "
        "map and mark together), under its fold's encoding, summed over the "
        "window's spikes and divided by the spikes counted",
    ),
}


def tetrode_holdout_extra_values(windows: pd.DataFrame, source: str) -> list[Value]:
    """Question 2c.3: what the held-out tetrode evidence of the events is made
    of, from the run with its extra scores (``windows``,
    ``tetrode_holdout.window_scores``): the gain over local coding and the
    agreement each split into their firing-rate and mark parts, the agreement's
    shuffled-event reference and the agreement minus it, and the place
    information of the scored spikes in the events, in their Local-rest
    controls and the event minus its control
    (``low_rate_places.event_minus_control``), by group, with the statistic of
    the neighboring held-out tetrode values. Figure 2j draws the agreement minus
    the reference (the fit to the decoded place over another event's place)
    and carries its tag; no figure draws the others."""
    make = value_factory(
        tetrode_holdout_extra_values, source, TAGS["tetrode_holdout_text"]
    )
    make_figure = value_factory(
        tetrode_holdout_extra_values, source, TAGS["tetrode_holdout_extra_figure"]
    )
    reduction = (
        "mean over animals of each animal's median over epochs of each epoch's median"
    )
    values = []
    for name in (
        "gain_rate",
        "gain_mark",
        "agreement_rate",
        "agreement_mark",
        "agreement_reference",
        "agreement_minus_reference",
        "mark_information",
        "spike_information",
    ):
        column, words = _EXTRA_MEASURES[name]
        kinds = [("event", "events", "")]
        if name.endswith("information"):
            kinds.append(("control", "Local-rest controls", "control_"))
        for kind, label, prefix in kinds:
            statistics = tetrode_holdout.holdout_statistics(windows, kind, column)
            for word, group in _RUN_GROUPS.items():
                description = (
                    f"held-out tetrode score of {group} {label}: {words}; "
                    f"{_TETRODE_HELD_OUT}; {_EXTRA_RUN}; {reduction}"
                )
                key = f"{word}_heldout_{prefix}{name}_per_spike"
                row = statistics.loc[group]
                if name == "agreement_minus_reference":
                    values += _interval_and_animals_positive(
                        make_figure, key, row, description
                    )
                else:
                    values += _per_spike_interval(make, key, row, description)
        if name.endswith("information"):
            paired = low_rate_places.event_minus_control(windows, column)
            statistics = tetrode_holdout.holdout_statistics(
                paired, "event", f"{column}_minus_control"
            )
            for word, group in _RUN_GROUPS.items():
                values += _interval_and_animals_positive(
                    make,
                    f"{word}_heldout_{name}_minus_control_per_spike",
                    statistics.loc[group],
                    f"{words}, in each {group} event minus in its Local-rest "
                    f"control (paired on the event); {_TETRODE_HELD_OUT}; "
                    f"{_EXTRA_RUN}; {reduction}",
                )
    return values


#: The tenths of the place-rate rank, by their word in a key (which holds no
#: digit), lowest first.
_TENTHS = (
    "first", "second", "third", "fourth", "fifth",
    "sixth", "seventh", "eighth", "ninth", "tenth",
)  # fmt: skip
#: Each place group of ``low_rate_places.PLACES``: its word in a key and in text.
_PLACES = {
    "low_rate": ("low_rate", "decoded at a low-rate place"),
    "other": ("other_place", "decoded elsewhere"),
}
_PLACE_RATE_WORDS = (
    "an event's two places are the track's real places nearest its replay start "
    "and end positions, each ranked among the real places by the encoding "
    "model's summed ground intensity (0 the lowest rate); a low-rate event has "
    f"both places ranked below {LOW_RATE_RANK:g} (an event with one position is "
    "judged on that one)"
)


def low_rate_values(
    place_rate: pd.DataFrame,
    events: pd.DataFrame,
    windows: pd.DataFrame,
    source: str,
) -> list[Value]:
    """Question 2c.3: the events decoded at low-rate places
    (``low_rate_places.low_rate_statistics``). The share of each group's events
    that are low-rate events, each animal's then their mean, with the lowest
    and highest animal and the counts; the mean share in each tenth of the
    rank; the share of events with no SWR
    without the low-rate events (``event_statistics.overlap_class_statistics``
    of ``low_rate_places.events_outside_low_rate``); and the held-out tetrode
    scores of ``low_rate_places.LOW_RATE_MEASURES`` for low-rate and other
    events of each group (``windows``, a run with ``--extra-scores``). No
    figure draws these."""
    make_text = value_factory(low_rate_values, source, TAGS["tetrode_holdout_text"])
    statistics = low_rate_places.low_rate_statistics(windows, place_rate)
    values = []
    for word, group in _RUN_GROUPS.items():
        row = statistics.shares.loc[group]
        share = f"share of the {group} events that are low-rate events ({_PLACE_RATE_WORDS})"
        values += [
            make_text(
                f"{word}_low_rate_share_animal_mean",
                row["share_animal_mean"],
                "fraction",
                "pct",
                f"{share}: each epoch's share, each animal's mean over its epochs, "
                "then the mean over animals",
            ),
            make_text(
                f"{word}_low_rate_share_animal_min",
                row["share_animal_min"],
                "fraction",
                "pctnum",
                "lowest of the animals' values of that share",
            ),
            make_text(
                f"{word}_low_rate_share_animal_max",
                row["share_animal_max"],
                "fraction",
                "pct",
                "highest of the animals' values of that share",
            ),
            make_text(
                f"{word}_low_rate_events",
                row["n_low_rate_events"],
                "count",
                "count",
                f"{group} low-rate events",
            ),
            make_text(
                f"{word}_place_rate_ranked_events",
                row["n_events"],
                "count",
                "count",
                f"{group} events with a ranked place",
            ),
        ]
        for k, tenth in enumerate(_TENTHS):
            values.append(
                make_text(
                    f"{word}_place_rate_{tenth}_tenth_share",
                    statistics.tenths.loc[(group, k), "animal_mean"],
                    "fraction",
                    "pct",
                    f"share of the {group} events' places whose rank is in the "
                    f"{tenth} tenth from the lowest ({_PLACE_RATE_WORDS}; each "
                    "event counts its two places half each, its one place whole): "
                    "each epoch's share, each animal's mean over its epochs, then "
                    "the mean over animals; chance is the share of the track's "
                    "places in the tenth, about 10 %",
                )
            )
    outside = event_statistics.overlap_class_statistics(
        low_rate_places.events_outside_low_rate(events, place_rate)
    )
    values += [
        make_text(
            "non_swr_fraction_animal_mean_without_low_rate",
            outside["non_swr_fraction_animal_mean"],
            "fraction",
            "pct",
            "share of immobile non-local events overlapping no SWR, the mean over "
            "animals of each animal's share (the mean over its epochs of the "
            "epoch's share), over the events that are not low-rate events "
            f"({_PLACE_RATE_WORDS})",
        ),
        make_text(
            "nonlocal_events_without_low_rate",
            outside["n_events"],
            "count",
            "count",
            "immobile non-local events that are not low-rate events",
        ),
    ]
    reduction = (
        "mean over animals of each animal's median over epochs of each epoch's median"
    )
    names = {column: name for name, (column, _) in _EXTRA_MEASURES.items()}
    names["gain_per_spike"] = "gain"
    for column in low_rate_places.LOW_RATE_MEASURES:
        name = names[column]
        words = (
            _TETRODE_MEASURES["gain"][1] if name == "gain" else _EXTRA_MEASURES[name][1]
        )
        make = make_text
        split = statistics.measures[column]
        for place, (place_word, place_words) in _PLACES.items():
            for word, group in _RUN_GROUPS.items():
                row = split.loc[(place, group)]
                key = f"{word}_{place_word}_heldout_{name}_per_spike"
                values += _interval_and_animals_positive(
                    make,
                    key,
                    row,
                    f"held-out tetrode score of the {group} events {place_words} "
                    f"({_PLACE_RATE_WORDS}): {words}; {_TETRODE_HELD_OUT}; "
                    f"{_EXTRA_RUN}; {reduction}",
                )
                if name == "gain":
                    values.append(
                        make_text(
                            f"{word}_{place_word}_heldout_events_scored",
                            row["n_scored"],
                            "count",
                            "count",
                            f"{group} events {place_words} with a held-out score "
                            "per spike",
                        )
                    )
    return values


def resolved_content_values(
    events: pd.DataFrame, windows: pd.DataFrame, source: str
) -> list[Value]:
    """Question 2b.2: the share of events with no SWR among the events whose
    content is resolved (the other tetrodes' estimate under
    ``resolved_content.RESOLVED_WIDTH_CM`` wide, in the held-out tetrode run
    with its extra scores, ``windows``) and among the wider ones, by the rule
    of ``non_swr_fraction_animal_mean``; the share of events that are
    resolved; the events of each and those without a width
    (``resolved_content.resolved_share_statistics``). No figure draws these."""
    make = value_factory(resolved_content_values, source, TAGS["cohort_pooled"])
    statistics = resolved_content.resolved_share_statistics(events, windows)
    cut = f"{resolved_content.RESOLVED_WIDTH_CM:g} cm"
    rule = (
        "the mean over animals of each animal's share (the mean over its epochs "
        "of the epoch's share)"
    )
    width = (
        "the event's width: the mean over its left-out tetrodes and bins of the "
        "95 % HPD length of the other tetrodes' position estimate"
    )
    values = []
    for name, group, words in (
        ("resolved", "resolved", f"whose width is under {cut} ({width})"),
        ("wide", "wide", f"whose width is {cut} or more ({width})"),
    ):
        row = statistics[group]
        values += [
            make(
                f"non_swr_fraction_animal_mean_{name}",
                row["non_swr_fraction_animal_mean"],
                "fraction",
                "pct",
                f"share of the immobile non-local events {words} that overlap no "
                f"SWR: {rule}",
            ),
            make(
                f"non_swr_fraction_animal_min_{name}",
                row["non_swr_fraction_animal_min"],
                "fraction",
                "pctnum",
                "lowest of the animals' values of that share",
            ),
            make(
                f"non_swr_fraction_animal_max_{name}",
                row["non_swr_fraction_animal_max"],
                "fraction",
                "pct",
                "highest of the animals' values of that share",
            ),
            make(
                f"nonlocal_events_{name}",
                row["n_events"],
                "count",
                "count",
                f"immobile non-local events {words}",
            ),
        ]
    share = f"share of the immobile non-local events with a width whose width is under {cut} ({width})"
    values += [
        make(
            "resolved_share_animal_mean",
            statistics["resolved_share_animal_mean"],
            "fraction",
            "pct",
            f"{share}: {rule}",
        ),
        make(
            "resolved_share_animal_min",
            statistics["resolved_share_animal_min"],
            "fraction",
            "pctnum",
            "lowest of the animals' values of that share",
        ),
        make(
            "resolved_share_animal_max",
            statistics["resolved_share_animal_max"],
            "fraction",
            "pct",
            "highest of the animals' values of that share",
        ),
        make(
            "nonlocal_events_without_width",
            statistics["n_without_width"],
            "count",
            "count",
            "immobile non-local events without a width in the held-out tetrode "
            "run (left out of the shares by width)",
        ),
    ]
    return values


#: The three decoders of the restriction run (``nonlocal_restriction.MODELS``),
#: in words.
_RESTRICTION_MODELS = {
    "none": "the local-only decoder (Local and No-Spike states at every bin)",
    "swr": "the SWR-only decoder (only the non-local states in SWR bins, only the "
    "local states elsewhere)",
    "full": "the full decoder (every state at every bin)",
    "swr_allowed": "the SWR-allowed decoder (every state in SWR bins, only the "
    "local states elsewhere)",
}
#: The two score families (``nonlocal_restriction.FAMILY_SUFFIX``), in words.
_RESTRICTION_SCORES = {
    "filter": "one-step-ahead (filter) predictive log score of each 2 ms bin's spikes",
    "two_sided": "two-sided predictive log score of each 2 ms bin's spikes (from "
    "every other bin; summed, two-sided scores are not a log likelihood)",
}
_RESTRICTION_HELD_OUT = (
    "with each of five folds of whole trials (balanced by destination well; a "
    "trial's run to a well with its whole stay there, when longer than "
    f"{MAX_SEGMENT_S:g} s, is cut into pieces of at most "
    f"{MAX_SEGMENT_S:g} s that can fall in different folds) "
    "scored under a decoder whose place-field encodings were fit on the decode's "
    "training bins (outside every SWR and artifact) of the other folds, more than "
    "1 s from the fold, and whose state transitions and initial conditions were "
    "fit by EM with the fold's bins missing"
)
#: The window comparisons reported: kind of window, group, contrast.
_RESTRICTION_WINDOWS = (
    ("event", SWR_LABELS[False], "full_minus_swr"),
    ("event", SWR_LABELS[True], "swr_minus_none"),
    ("event", SWR_LABELS[True], "full_minus_none"),
    ("event", SWR_LABELS[True], "full_minus_swr"),
    ("control", SWR_LABELS[False], "full_minus_swr"),
    ("control", SWR_LABELS[True], "full_minus_swr"),
)
_RESTRICTION_KINDS = {
    "event": ("events", "events"),
    "control": ("controls", "matched Local-rest controls"),
}
#: Each unit of a restriction value: its word in a key, what the sum is divided
#: by, and the value's unit.
_RESTRICTION_UNITS = (
    ("per_spike", "their spikes", "nats per spike"),
    ("per_s", "their duration", "nats per s"),
)
#: Each group's bins outside SWRs, in words.
_OUTSIDE_SWR_WHERE = {
    SWR_LABELS[False]: "over each of the events with no SWR (its bins outside "
    "SWRs: all but a few inside an SWR the SWR-only decoder used)",
    SWR_LABELS[True]: "over the bins outside the SWR of each of the events that "
    "overlap an SWR",
}
#: The bins every value over the immobile time outside SWRs is summed over, in
#: words (``restriction_windows.outside_swr_bins``).
_OUTSIDE_SWR_BINS = (
    "immobile bins outside SWRs (outside every SWR of the source, where SWR "
    "detection was observed)"
)
#: How a value over time is reduced over epochs and animals, in words.
_EPOCH_REDUCTION = (
    "mean over animals of each animal's median over its epochs; 95 % interval "
    "from the nested bootstrap of the animals and their epochs, small-sample t "
    "correction over the animals"
)
#: Each SWR group's word in a key (the inverse of ``_RUN_GROUPS``).
_GROUP_WORDS = {group: word for word, group in _RUN_GROUPS.items()}
#: The full decoder's score minus the SWR-only decoder's, in words.
_FULL_MINUS_SWR = (
    f"under {_RESTRICTION_MODELS['full']} minus under {_RESTRICTION_MODELS['swr']}"
)


def _interval_and_animals_positive(
    make: Callable[..., Value],
    key: str,
    row: pd.Series,
    description: str,
    unit: str = "nats per spike",
) -> list[Value]:
    """:func:`_per_spike_interval` to two significant digits, and the animals
    whose value is above zero (``n_animals_positive``)."""
    return [
        *_per_spike_interval(make, key, row, description, style="sig2", unit=unit),
        make(
            f"{key}_animals_positive",
            row["n_animals_positive"],
            "count",
            "count",
            "animals whose value is above zero",
        ),
    ]


def _full_minus_swr_values(
    make: Callable[..., Value],
    key: str,
    row: pd.Series,
    unit: tuple[str, str, str],
    scored: str,
    time: str,
    epochs: str,
    positive: str = "epochs whose gain per second is above zero",
    difference: str = _FULL_MINUS_SWR,
) -> list[Value]:
    """The full minus the SWR-only decoder (or another ``difference``, in
    words) over ``time``, per ``unit`` (an entry of ``_RESTRICTION_UNITS``):
    :func:`_interval_and_animals_positive`, and per second also the epochs
    whose value is above zero (``positive``) and the epochs with a value
    (``epochs``, what they have)."""
    name, divided, words = unit
    values = _interval_and_animals_positive(
        make,
        f"{key}_{name}",
        row,
        f"{scored}, {difference}, summed over {time} and divided by "
        f"{divided}: {_EPOCH_REDUCTION}",
        unit=words,
    )
    if name == "per_s":
        values += [
            make(
                f"{key}_epochs_positive",
                row["n_epochs_positive"],
                "count",
                "count",
                positive,
            ),
            make(
                f"{key}_epochs",
                row["n_epochs"],
                "count",
                "count",
                f"epochs with {epochs}",
            ),
        ]
    return values


def _animal_range_values(
    make: Callable[..., Value],
    key: str,
    shares: tuple[float, float, float],
    descriptions: tuple[str, str, str],
    lowest_style: str = "pctnum",
    *,
    animal_mean: tuple[float, str] | None = None,
    pooled_tags: Tags | None = None,
) -> list[Value]:
    """A share, then the lowest and the highest animal's (``{key}_animal_min``,
    ``{key}_animal_max``): ``shares`` and ``descriptions`` in that order, each
    printed as a percent (the lowest in ``lowest_style``).

    With ``animal_mean`` (the mean over animals of the animals' shares and its
    description), ``{key}_animal_mean`` follows the share; ``pooled_tags``
    then replaces the share's tags (a pooled share that no panel draws)."""
    values = [
        make(key_, share, "fraction", style, description)
        for key_, share, style, description in zip(
            (key, f"{key}_animal_min", f"{key}_animal_max"),
            shares,
            ("pct", lowest_style, "pct"),
            descriptions,
            strict=True,
        )
    ]
    if animal_mean is None:
        return values
    values[0] = make(
        key, shares[0], "fraction", "pct", descriptions[0], tags=pooled_tags
    )
    mean, description = animal_mean
    return [
        values[0],
        make(f"{key}_animal_mean", mean, "fraction", "pct", description),
        *values[1:],
    ]


def _share_interval(
    make: Callable[..., Value], key: str, row: pd.Series, description: str
) -> list[Value]:
    """A share and the bounds of its 95 % interval (``estimate``, ``ci_low``,
    ``ci_high``)."""
    return [
        make(key, row["estimate"], "fraction", "pct", description),
        make(
            f"{key}_ci_low",
            row["ci_low"],
            "fraction",
            "pct",
            "lower bound of its 95 % interval",
        ),
        make(
            f"{key}_ci_high",
            row["ci_high"],
            "fraction",
            "pct",
            "upper bound of that interval",
        ),
    ]


def run_staleness(manifest: dict, folder: Path, name: str, run_dir: Path) -> str | None:
    """Why the output in ``folder`` (its ``manifest``) is stale for the run
    ``name`` in ``run_dir`` that it read, or None: it records no digest of that
    run's manifest, or another than the current one."""
    return _input_staleness(
        manifest,
        name,
        file_record(run_dir / MANIFEST, ROOT_DIR)["sha256"],
        rel(folder),
        f"{name} run",
        f"{name} run",
    )


@dataclass(frozen=True)
class HeldoutRestriction:
    """The held-out event run's inputs to the restriction values and Figure 2k
    (:func:`read_heldout_restriction`), and why a value read from them is
    pending (each None when it is not)."""

    events: pd.DataFrame
    """The events detected from the held-out decode that start inside a well
    visit (``starts_in_well_visit``)."""
    time_sums: pd.DataFrame
    """The time sums of every epoch (``restriction_totals.restriction_time_sums``)."""
    bin_s: pd.Series
    """Each epoch's decode time step (s), indexed by ``epoch_id``, as the run read
    it from the decode frame's time (checked there against the per-bin table's)."""
    restriction_run: str | None
    """The run was not made from the current restriction run."""
    bin_facts_run: str | None
    """The run was not made from the current per-bin facts run."""
    session_cache: str | None
    """The per-bin facts, which give the time sums, the events' start and the
    windows' bins their well-visit flag, were not made from the current
    session cache."""
    event_table: str | None
    """The run was not made from the current event table."""


def restriction_inputs_note(
    restriction_dir: Path,
    epochs: Iterable[str],
    label: str | None = None,
    features_manifest: dict | None = None,
) -> str | None:
    """Why the restriction run in ``restriction_dir`` was not made from the
    current decode frames and SWR source, or None: :func:`restriction_input_staleness`
    with the decode frames the event features run records for each epoch
    (``features_manifest``, read from the canonical run when None). ``label``
    names the run in the note (by default ``the restriction run (<folder>)``).
    The figures that draw the restriction run refuse it then
    (:func:`heldout_restriction_refusal`)."""
    if features_manifest is None:
        features_manifest = read_manifest(event_features_root() / EVENT_FEATURES_RUN)
    return restriction_input_staleness(
        read_output_manifest(restriction_dir),
        {
            epoch_id: record.get("decode", {}).get("sha256")
            for epoch_id, record in features_manifest["epochs"].items()
        },
        epochs,
        f"the restriction run ({restriction_dir})" if label is None else label,
    )


def checked_reference(reference: pd.DataFrame, folder: Path, name: str) -> pd.DataFrame:
    """A per-epoch measurement (the brain-state reference, the counts by signal
    of Figure 2i and its supplement), if it was made over the time inside the well visits, from the
    current session cache's visits and from the decodes the event features run
    records for every analyzed epoch.

    Parameters
    ----------
    reference : pd.DataFrame
        The measurement, with each row's ``epoch_id`` and ``decode_sha256``.
    folder : Path
        Its folder, whose ``manifest.json`` records the subset it is over and
        the session-cache entries it read the visits from.
    name : str
        What it is, for the error.

    Raises
    ------
    ValueError
        If it was not made over the analyses' subset or from the current
        session cache (``prepare_data.time_mask_staleness``), an analyzed epoch
        is missing or extra, or its decode hash differs.
    """
    note = time_mask_staleness(
        read_output_manifest(folder), reference["epoch_id"], f"the {name} ({folder})"
    )
    if note is not None:
        raise ValueError(note)
    manifest = read_manifest(event_features_root() / EVENT_FEATURES_RUN)
    expected = {
        epoch_id: manifest["epochs"][epoch_id]["decode"]["sha256"]
        for epoch_id in analyzed_epoch_ids()
    }
    recorded = dict(zip(reference["epoch_id"], reference["decode_sha256"], strict=True))
    if recorded != expected:
        differ = sorted(
            epoch_id
            for epoch_id in expected.keys() | recorded.keys()
            if expected.get(epoch_id) != recorded.get(epoch_id)
        )
        raise ValueError(
            f"the {name} was not made from the decodes of event "
            f"features run {EVENT_FEATURES_RUN}: {len(differ)} epochs differ, "
            f"e.g. {differ[:3]}"
        )
    return reference


def restriction_input_staleness(
    manifest: dict,
    decode_sha256: Mapping[str, str],
    epoch_ids: Iterable[str],
    what: str,
) -> str | None:
    """Why the restriction run was not made from the current decode frames and
    SWR source, or None.

    The run's settings record the SWR source by name (``swr_source``, which
    names the SWR inventory run and the artifact run) and each epoch's record
    the SHA-256 of the decode frame it read (``epoch_inputs.data_over_time``).
    The current source is ``swr_source.swr_source_name()``. The current decode
    frames are the ones the event features run records (``decode_sha256``),
    which the figures also take as the decodes (:func:`checked_reference`); no decode frame is hashed here, so a decode remade
    without a new event features run is not seen.

    Parameters
    ----------
    manifest : dict
        The restriction run's manifest.
    decode_sha256 : Mapping of str to str
        Each epoch's decode frame's SHA-256, as the event features run records it.
    epoch_ids : iterable of str
        The epochs read; one the run has no record of, or the features run no
        digest of, counts as another decode.
    what : str
        The restriction run, for the note.
    """
    notes = []
    recorded = manifest.get("settings", {}).get("swr_source")
    current = swr_source.swr_source_name()
    if recorded != current:
        notes.append(
            f"its settings record the SWR source {recorded!r}, not the current "
            f"{current!r}"
        )
    records = manifest.get("epochs", {})

    def recorded_decode(epoch_id: str) -> str | None:
        inputs = records.get(epoch_id, {}).get("epoch_inputs", {})
        return inputs.get("data_over_time", {}).get("sha256")

    differ = sorted(
        epoch_id
        for epoch_id in epoch_ids
        if recorded_decode(epoch_id) is None
        or recorded_decode(epoch_id) != decode_sha256.get(epoch_id)
    )
    if differ:
        notes.append(
            f"{len(differ)} of its epochs record another decode frame than the "
            f"one event features run {EVENT_FEATURES_RUN} records (e.g. "
            f"{differ[:3]})"
        )
    if not notes:
        return None
    return (
        f"{what} was not made from the current decodes and SWR source "
        f"({'; '.join(notes)}); rerun scripts/event_evidence/"
        "score_nonlocal_restriction.py and the runs made from it before citing"
    )


HELDOUT_EVENTS_RERUN = "scripts/event_evidence/detect_heldout_events.py"
HELDOUT_RESTRICTION_RERUN = {
    "restriction_inputs": "scripts/event_evidence/score_nonlocal_restriction.py "
    "and the runs made from it",
    "restriction_run": HELDOUT_EVENTS_RERUN,
    "bin_facts_run": HELDOUT_EVENTS_RERUN,
    "session_cache": "scripts/event_evidence/measure_bin_facts.py, then "
    + HELDOUT_EVENTS_RERUN,
}
"""The runners to make again when a figure refuses the held-out event run, by
what is not current (:func:`heldout_restriction_refusal`)."""


def heldout_restriction_refusal(
    heldout: HeldoutRestriction, restriction_inputs: str | None
) -> tuple[str, str] | None:
    """Why a figure drawing the held-out event run refuses it, or None.

    Parameters
    ----------
    heldout : HeldoutRestriction
        The run's inputs and reasons (:func:`read_heldout_restriction`); its
        event-table reason is the summary's alone.
    restriction_inputs : str or None
        Why the restriction run was not made from the current decodes and SWR
        source (:func:`restriction_inputs_note`).

    Returns
    -------
    tuple of (str, str) or None
        The first reason that is not None, without its closing advice to rerun
        before citing, and the runners to make again
        (:data:`HELDOUT_RESTRICTION_RERUN`).
    """
    for name, reason in (
        ("restriction_inputs", restriction_inputs),
        ("restriction_run", heldout.restriction_run),
        ("bin_facts_run", heldout.bin_facts_run),
        ("session_cache", heldout.session_cache),
    ):
        if reason is not None:
            return reason.split("; rerun ")[0], HELDOUT_RESTRICTION_RERUN[name]
    return None


def read_heldout_restriction(
    heldout_dir: Path,
    restriction_dir: Path,
    bin_facts_dir: Path,
    epoch_ids: Iterable[str],
    event_table_sha256: str,
) -> HeldoutRestriction:
    """The held-out event run's events, time sums and time steps, with the
    reasons a value read from them is pending.

    Parameters
    ----------
    heldout_dir, restriction_dir, bin_facts_dir : Path
        The held-out event run, the restriction run and the per-bin facts run.
    epoch_ids : iterable of str
        The epochs read (every analyzed epoch with an event).
    event_table_sha256 : str
        The current event table's SHA-256.

    Raises
    ------
    ValueError
        If the restriction run is not the fold fit on folds of whole trials
        (:func:`require_fold_fit`).
    """
    require_fold_fit(read_output_manifest(restriction_dir), rel(restriction_dir))
    epoch_ids = list(epoch_ids)
    manifest = read_output_manifest(heldout_dir)
    events = read_complete_run(heldout_dir, HELDOUT_EVENTS_NAME, epoch_ids)
    facts_manifest = read_output_manifest(bin_facts_dir)
    return HeldoutRestriction(
        events=events[events["starts_in_well_visit"].astype(bool)],
        time_sums=read_complete_run(heldout_dir, HELDOUT_TIME_SUMS_NAME, epoch_ids),
        bin_s=pd.Series(
            {
                epoch_id: record["bin_s"]
                for epoch_id, record in manifest["epochs"].items()
            }
        ),
        restriction_run=run_staleness(
            manifest, heldout_dir, "nonlocal_restriction", restriction_dir
        ),
        bin_facts_run=run_staleness(manifest, heldout_dir, "bin_facts", bin_facts_dir),
        session_cache=session_cache_staleness(
            facts_manifest, ok_epochs(facts_manifest), rel(bin_facts_dir)
        ),
        event_table=event_table_staleness(
            manifest, event_table_sha256, rel(heldout_dir)
        ),
    )


class WindowValues(NamedTuple):
    """The restriction window values, by what they read."""

    windows: list[Value]
    """The values that read the window run only."""
    after_visit: list[Value]
    """The counts of events by their bins after the final poke-out of their
    visit: of the window values, only these read the well visits (through the
    per-bin facts)."""


def restriction_window_values(windows: pd.DataFrame, source: str) -> WindowValues:
    """Question 2c.3: the three restriction decoders compared per spike over the
    canonical events of ``windows`` (those that start inside a well visit,
    ``cohort_selection.select_events_in``) and their Local-rest controls, in
    both score families, and the full decoder against the SWR-only decoder per
    spike and per second over the events' bins outside SWRs. No figure draws
    these, so none carries a figure's tag. Every key starts ``restriction_canonical_``, and the summary adds
    :data:`CANONICAL_WINDOWS_SUBSET` to every description."""
    make = value_factory(
        restriction_window_values, source, TAGS["nonlocal_restriction"]
    )
    # only the per-spike rows are read; each row's bootstrap is seeded on its
    # own, so the rows do not depend on which others are computed
    statistics = restriction_windows.restriction_window_statistics(
        windows, units=["per_spike"]
    ).set_index(["family", "contrast", "unit", "kind", "group"])
    values, after_visit = [], []
    for family, suffix in FAMILY_SUFFIX.items():
        for kind, group, contrast in _RESTRICTION_WINDOWS:
            row = statistics.loc[(family, contrast, "per_spike", kind, group)]
            kind_word, label = _RESTRICTION_KINDS[kind]
            a, b = restriction_windows.CONTRASTS[contrast]
            values += _interval_and_animals_positive(
                make,
                f"restriction_canonical_{_GROUP_WORDS[group]}_{kind_word}_{contrast}"
                f"{suffix}_per_spike",
                row,
                f"{_RESTRICTION_SCORES[family]}, {_RESTRICTION_HELD_OUT}, under "
                f"{_RESTRICTION_MODELS[a]} minus under {_RESTRICTION_MODELS[b]}, "
                f"summed over each of the {group} {label} and divided by its "
                "spikes: mean over animals of each animal's "
                "median over epochs of each epoch's median",
            )
    # full - swr per spike and per second over the events' bins outside SWRs
    outside = restriction_windows.restriction_window_statistics(
        windows,
        **{
            **OUTSIDE_SWR_EVENT_WINDOWS,
            "families": list(FAMILY_SUFFIX),
            "units": ["per_spike", "per_s"],
        },
    ).set_index(["family", "unit", "group"])
    counts = restriction_windows.outside_swr_window_counts(windows)
    a, b = restriction_windows.CONTRASTS["full_minus_swr"]
    for family, suffix in FAMILY_SUFFIX.items():
        for unit, divided, words in _RESTRICTION_UNITS:
            for group in OUTSIDE_SWR_EVENT_WINDOW_GROUPS:
                row = outside.loc[(family, unit, group)]
                word = _GROUP_WORDS[group]
                values += _interval_and_animals_positive(
                    make,
                    f"restriction_canonical_{word}_events_outside_swr_full_minus_swr"
                    f"{suffix}_{unit}",
                    row,
                    f"{_RESTRICTION_SCORES[family]}, {_RESTRICTION_HELD_OUT}, under "
                    f"{_RESTRICTION_MODELS[a]} minus under {_RESTRICTION_MODELS[b]}, "
                    f"summed {_OUTSIDE_SWR_WHERE[group]} and divided by {divided}: "
                    "mean over animals of each animal's median over epochs of each "
                    "epoch's median",
                    unit=words,
                )
                if family == RESTRICTION_FIGURE_FAMILY:
                    values.append(
                        make(
                            f"restriction_canonical_{word}_events_outside_swr_scored"
                            f"_{unit}",
                            row["n_scored"],
                            "count",
                            "count",
                            f"{group} events with a value "
                            + (
                                "per spike outside SWRs (immobile bins there and a "
                                "spike in them)"
                                if unit == "per_spike"
                                else "per second outside SWRs (immobile bins there)"
                            ),
                        )
                    )
    for group in OUTSIDE_SWR_EVENT_WINDOW_GROUPS:
        row = counts.loc[("event", group)]
        word = _GROUP_WORDS[group]
        values.append(
            make(
                f"restriction_canonical_{word}_events_outside_swr_without_spikes",
                row["n_without_spikes_outside_swr"],
                "count",
                "count",
                f"{group} events with immobile bins outside SWRs but no spike in "
                "them: no value per spike there",
            )
        )
        after_visit.append(
            make(
                f"restriction_canonical_{word}_events_outside_swr_after_visit",
                row["n_with_bins_outside_swr_outside_visit"],
                "count",
                "count",
                f"{group} events with immobile bins outside SWRs after the final "
                "poke-out of their visit (left out of "
                "the shares of immobile time inside a well visit)",
            )
        )
    # the windows and those with spikes are the same for every contrast and family
    for kind, (kind_word, label) in _RESTRICTION_KINDS.items():
        for word, group in _RUN_GROUPS.items():
            row = statistics.loc[("filter", "full_minus_swr", "per_spike", kind, group)]
            values += [
                make(
                    f"restriction_canonical_{word}_{kind_word}",
                    row["n_events"],
                    "count",
                    "count",
                    f"{group} {label} summed",
                ),
                make(
                    f"restriction_canonical_{word}_{kind_word}_scored",
                    row["n_scored"],
                    "count",
                    "count",
                    f"{group} {label} with a per-spike value (at least one spike)",
                ),
            ]
    values.append(
        make(
            "restriction_canonical_animals",
            row["n_animals"],
            "count",
            "count",
            "animals in the restriction comparison",
        )
    )
    return WindowValues(values, after_visit)


def restriction_heldout_event_values(windows: pd.DataFrame, source: str) -> list[Value]:
    """Question 2c.3: the full decoder minus the SWR-only decoder per spike
    over the bins outside SWRs of the events detected from the held-out decode
    (``windows``, those that start inside a well visit), events with no SWR and
    events that overlap one, in :data:`RESTRICTION_FIGURE_FAMILY`
    (:data:`OUTSIDE_SWR_EVENT_WINDOWS`); with the events of each group, those with a
    value and those with bins outside SWRs but no spike in them. The gain per
    spike in the events with no SWR is drawn on Figure 2k's second axis
    (``restriction_totals.RestrictionFigureStatistics.events``) and carries its
    tag, the counts with it; the rest is for the text."""
    make_text = value_factory(
        restriction_heldout_event_values, source, TAGS["nonlocal_restriction"]
    )
    make_figure = value_factory(
        restriction_heldout_event_values, source, TAGS["nonlocal_restriction_figure"]
    )
    statistics = restriction_windows.restriction_window_statistics(
        windows, **OUTSIDE_SWR_EVENT_WINDOWS
    ).set_index("group")
    counts = restriction_windows.outside_swr_window_counts(windows)
    a, b = restriction_windows.CONTRASTS["full_minus_swr"]
    values = []
    for group in OUTSIDE_SWR_EVENT_WINDOW_GROUPS:
        row = statistics.loc[group]
        word = _GROUP_WORDS[group]
        make = make_figure if group == RESTRICTION_FIGURE_EVENT_GROUP else make_text
        values += _interval_and_animals_positive(
            make,
            f"restriction_heldout_{word}_events_outside_swr_full_minus_swr_per_spike",
            row,
            f"{_RESTRICTION_SCORES[RESTRICTION_FIGURE_FAMILY]}, "
            f"{_RESTRICTION_HELD_OUT}, under {_RESTRICTION_MODELS[a]} minus under "
            f"{_RESTRICTION_MODELS[b]}, summed {_OUTSIDE_SWR_WHERE[group]} and "
            "divided by their spikes: mean over animals of each animal's median "
            "over epochs of each epoch's median",
        )
        values += [
            make(
                f"restriction_heldout_{word}_events",
                row["n_events"],
                "count",
                "count",
                f"{group} events detected from the held-out decode",
            ),
            make(
                f"restriction_heldout_{word}_events_outside_swr_scored_per_spike",
                row["n_scored"],
                "count",
                "count",
                f"{group} events with a value per spike outside SWRs (immobile "
                "bins there and a spike in them)",
            ),
            make(
                f"restriction_heldout_{word}_events_outside_swr_without_spikes",
                counts.loc[("event", group), "n_without_spikes_outside_swr"],
                "count",
                "count",
                f"{group} events with immobile bins outside SWRs but no spike in "
                "them: no value per spike there",
            ),
        ]
    return values


def restriction_agreement_values(
    heldout: pd.DataFrame, canonical: pd.DataFrame, source: str
) -> list[Value]:
    """Question 2c.3: how the events detected from the held-out decode and the
    canonical events agree. Of the canonical events of ``canonical``, the share
    a held-out event shares a bin with; of the held-out events of ``heldout``,
    the share a canonical event shares a bin with; each over every event and
    by SWR group, pooled and the lowest and highest animal
    (``heldout_events.found_again_statistics``); and the held-out events."""
    make = value_factory(
        restriction_agreement_values, source, TAGS["nonlocal_restriction"]
    )
    words = {"all": "", SWR_LABELS[False]: "_non_swr", SWR_LABELS[True]: "_swr"}
    values = []
    for events, found, stem, what, partner in (
        (
            canonical,
            "shares_bin_with_heldout",
            "restriction_canonical{}_events_found_again",
            "canonical events (found from the decode fit on all the data)",
            "an event detected from the held-out decode",
        ),
        (
            heldout,
            "shares_bin_with_canonical",
            "restriction_heldout{}_events_with_canonical",
            "events detected from the held-out decode",
            "a canonical event",
        ),
    ):
        statistics = heldout_events.found_again_statistics(events, found)
        for group, word in words.items():
            row = statistics.loc[group]
            key = stem.format(word)
            which = what if group == "all" else f"{group} {what}"
            values += [
                make(
                    key,
                    row["share"],
                    "fraction",
                    "pct",
                    f"share of the {which} that share at least one bin with "
                    f"{partner} (of any well-visit start), pooled over the events "
                    "of every epoch and animal",
                ),
                make(
                    f"{key}_animal_min",
                    row["share_animal_min"],
                    "fraction",
                    "pctnum",
                    "lowest of the animals' shares, each animal's share the mean "
                    "over its epochs of the epoch's share (the pooled share, over "
                    "events, can lie outside the animals' range)",
                ),
                make(
                    f"{key}_animal_max",
                    row["share_animal_max"],
                    "fraction",
                    "pct",
                    "highest of the animals' shares, each animal's share the mean "
                    "over its epochs of the epoch's share (the pooled share, over "
                    "events, can lie outside the animals' range)",
                ),
            ]
    row = heldout_events.found_again_statistics(
        heldout, "shares_bin_with_canonical"
    ).loc["all"]
    values.append(
        make(
            "restriction_heldout_events",
            row["n_events"],
            "count",
            "count",
            "events detected from the held-out decode",
        )
    )
    return values


class OutsideSwrValues(NamedTuple):
    """The outside-SWR values, by what they are over."""

    with_events: list[Value]
    """The values over the time and the selected held-out events."""
    outside_events: list[Value]
    """The values over the time no held-out event holds: the held-out event run
    marks that time from every event detected from the held-out decode,
    wherever it starts, so these are over the time inside a well visit alone."""


def restriction_outside_swr_values(
    time_sums: pd.DataFrame, bin_s: pd.Series, windows: pd.DataFrame, source: str
) -> OutsideSwrValues:
    """Question 2c.3: the full decoder against the SWR-only decoder over the
    immobile time outside SWRs inside a well visit (time chosen by speed, ripple
    detection and the well visits, not by the decoder), per spike and per
    second, the share of it in the events of ``windows`` (the events detected
    from the held-out decode that start inside a well visit), and the rest
    outside every held-out event (``time_sums``' ``in_event_window``, which
    marks every held-out event, wherever it starts). The gain per spike and
    the three shares (in the no-SWR events, in the time outside the SWR of the
    events that overlap one, and the rest outside those events) in
    :data:`RESTRICTION_FIGURE_FAMILY` is drawn on Figure 2k and carries its
    tag, as does the gain per spike outside every held-out event (its second
    axis); the shares are for the text."""
    make_text = value_factory(
        restriction_outside_swr_values, source, TAGS["nonlocal_restriction"]
    )
    make_figure = value_factory(
        restriction_outside_swr_values, source, TAGS["nonlocal_restriction_figure"]
    )
    statistics = restriction_totals.outside_swr_statistics(time_sums, bin_s, windows)
    time = f"every one of the epoch's {_OUTSIDE_SWR_BINS}"
    values, outside_events = [], []
    for family, suffix in FAMILY_SUFFIX.items():
        key = f"restriction_outside_swr_full_minus_swr{suffix}"
        scored = f"{_RESTRICTION_SCORES[family]}, {_RESTRICTION_HELD_OUT}"
        figure = family == RESTRICTION_FIGURE_FAMILY
        for unit in _RESTRICTION_UNITS:
            name, divided, words = unit
            values += _full_minus_swr_values(
                make_figure
                if figure and f"gain_{name}" == RESTRICTION_FIGURE_OUTSIDE_SWR
                else make_text,
                key,
                statistics.loc[(family, f"gain_{name}")],
                unit,
                scored,
                time,
                _OUTSIDE_SWR_BINS,
            )
            row = statistics.loc[(family, f"outside_events_gain_{name}")]
            outside_events += _interval_and_animals_positive(
                make_figure
                if figure
                and f"outside_events_gain_{name}" == RESTRICTION_FIGURE_OUTSIDE_EVENTS
                else make_text,
                f"{key}_outside_heldout_events_{name}",
                row,
                f"{scored}, {_FULL_MINUS_SWR}, summed over the epoch's immobile bins "
                "outside SWRs that no event detected from the held-out decode "
                f"holds (wherever it starts) and divided by {divided}: "
                f"{_EPOCH_REDUCTION}",
                unit=words,
            )
        for measure, name, part in (
            (
                "share_in_swr_events_outside_swr",
                "share_in_heldout_swr_events_outside_swr",
                "the time outside the SWR of the events detected from the held-out "
                "decode that overlap an SWR",
            ),
            (
                "share_in_no_swr_events",
                "share_in_heldout_non_swr_events",
                "the events detected from the held-out decode with no SWR",
            ),
        ):
            values += _share_interval(
                make_text,
                f"{key}_{name}",
                statistics.loc[(family, measure)],
                f"{scored}; the gain is the score {_FULL_MINUS_SWR}. Per epoch, "
                f"the gain summed over {part} (their {_OUTSIDE_SWR_BINS} "
                "inside a well visit; an event that runs past the final poke-out "
                "adds its bins before it only), divided by the gain summed over "
                f"every one of the epoch's {_OUTSIDE_SWR_BINS} inside a well "
                "visit (no value when that sum is not positive). Both are signed sums, so a share can "
                "be above 1 or below 0 and the two shares need not add to 1. "
                f"Over the cohort: {_EPOCH_REDUCTION}",
            )
        # a share that is below 0 where the full decoder does worse: a
        # unitless quantity, as a fraction must lie in [0, 1]
        row = statistics.loc[(family, "share_outside_events")]
        values += [
            make_text(
                f"{key}_share_outside_heldout_events",
                row["estimate"],
                "quantity",
                "dec2",
                f"{scored}; the gain is the score {_FULL_MINUS_SWR}. Per epoch, 1 "
                "minus the share of the gain in the events detected from the "
                "held-out decode with no SWR and minus its share in the time "
                "outside the SWR of those that overlap an SWR, so the three shares "
                f"add to 1: the share of the gain over the epoch's {_OUTSIDE_SWR_BINS} "
                "inside a well visit that lies outside those events (no value when "
                "the gain summed over those bins is not positive), as a fraction of "
                "1, below 0 when the full decoder does worse than the SWR-only "
                f"decoder there. Over the cohort: {_EPOCH_REDUCTION}",
            ),
            make_text(
                f"{key}_share_outside_heldout_events_ci_low",
                row["ci_low"],
                "quantity",
                "dec2",
                "lower bound of its 95 % interval",
            ),
            make_text(
                f"{key}_share_outside_heldout_events_ci_high",
                row["ci_high"],
                "quantity",
                "dec2",
                "upper bound of that interval",
            ),
        ]
    return OutsideSwrValues(values, outside_events)


#: Each part of the time outside SWRs: its word in a key (which holds no digit)
#: and what it keeps, in words, by kind of part and threshold (s).
_OUTSIDE_SWR_PARTS = {
    "beyond_swr_margin": {
        threshold: (
            f"beyond_{word}",
            f"farther than {threshold * 1000:g} ms (on each side) from every bin "
            "of the scoring mask (every SWR of the source)",
        )
        for threshold, word in zip(
            restriction_totals.OUTSIDE_SWR_MARGINS_S,
            ("hundred_ms", "quarter_s", "half_s"),
            strict=True,
        )
    },
    "place_sampling": {
        threshold: (
            f"sampled_{word}",
            f"whose place was sampled for at least {threshold:g} s in the "
            "training data of the bin's own fold (the seconds of the decode's "
            "training bins outside the fold and its margin at the animal's place "
            "bin)",
        )
        for threshold, word in zip(
            restriction_totals.PLACE_SAMPLING_THRESHOLDS_S,
            ("ten_s", "minute"),
            strict=True,
        )
    },
}


def restriction_outside_swr_part_values(
    time_sums: pd.DataFrame, bin_s: pd.Series, source: str
) -> list[Value]:
    """Question 2c.3: the full decoder against the SWR-only decoder over parts
    of the immobile time outside SWRs inside a well visit
    (``restriction_totals.outside_swr_subset_statistics``): away from the SWRs
    (farther than a margin from every bin of the scoring mask) and where the
    animal's place was well sampled in training, per spike and per second,
    with the epochs above zero and the share of the bins each part keeps; and
    the immobile bins outside every SWR where SWR detection was not observed,
    which every value outside SWRs leaves out
    (``restriction_totals.outside_swr_not_observed_counts``). The gains per
    spike in :data:`RESTRICTION_FIGURE_FAMILY` away from the SWRs are drawn on
    Figure 2's supplement and carry its panel's tag (S2i); the gain at
    well-sampled places is for the text."""
    make = value_factory(
        restriction_outside_swr_part_values, source, TAGS["nonlocal_restriction"]
    )
    make_distance = value_factory(
        restriction_outside_swr_part_values,
        source,
        TAGS["nonlocal_restriction_distance"],
    )
    counts = restriction_totals.outside_swr_not_observed_counts(time_sums)
    values = [
        make(
            "restriction_outside_swr_not_observed_bins",
            counts["n_bins"],
            "count",
            "count",
            "immobile bins outside every SWR of the source where SWR detection "
            "was not observed (a gap in the stored ripple band): not known to be "
            "outside an SWR, so every value over the immobile time outside SWRs "
            "leaves them out",
        ),
        make(
            "restriction_outside_swr_not_observed_epochs",
            counts["n_epochs"],
            "count",
            "count",
            "epochs with immobile bins outside every SWR where SWR detection was "
            "not observed",
        ),
    ]
    statistics = restriction_totals.outside_swr_subset_statistics(
        time_sums, bin_s
    ).set_index(["subset", "threshold_s", "family", "measure"])
    shares = restriction_totals.outside_swr_subset_shares(time_sums, bin_s)
    for subset, parts in _OUTSIDE_SWR_PARTS.items():
        for threshold, (word, kept) in parts.items():
            time = f"the epoch's {_OUTSIDE_SWR_BINS} {kept}"
            for family, suffix in FAMILY_SUFFIX.items():
                key = f"restriction_outside_swr_{word}_full_minus_swr{suffix}"
                scored = f"{_RESTRICTION_SCORES[family]}, {_RESTRICTION_HELD_OUT}"
                for unit in _RESTRICTION_UNITS:
                    drawn = (
                        subset == "beyond_swr_margin"
                        and family == RESTRICTION_FIGURE_FAMILY
                        and unit[0] == "per_spike"
                    )
                    values += _full_minus_swr_values(
                        make_distance if drawn else make,
                        key,
                        statistics.loc[(subset, threshold, family, f"gain_{unit[0]}")],
                        unit,
                        scored,
                        time,
                        f"{_OUTSIDE_SWR_BINS} {kept}",
                    )
            row = shares.loc[(subset, threshold)]
            where = f"that lie {kept}" if subset == "beyond_swr_margin" else kept
            share = f"share of the epoch's {_OUTSIDE_SWR_BINS} {where}"
            values += [
                make(
                    f"restriction_outside_swr_{word}_share_of_bins",
                    row["estimate"],
                    "fraction",
                    "pct",
                    f"{share}: mean over animals of each animal's median over its "
                    "epochs; no interval is given (a t interval of a share near 1 "
                    "can pass 1), the lowest animal's value is given beside it",
                ),
                make(
                    f"restriction_outside_swr_{word}_share_of_bins_animal_min",
                    row["animal_min"],
                    "fraction",
                    "pct",
                    f"lowest of the animals' values of the {share}, each animal's "
                    "the median over its epochs",
                ),
            ]
    return values


_INSIDE_SWR_BINS = (
    "immobile bins inside SWRs (inside an SWR of the source, where the SWR-only "
    "decoder must use the non-local states and the full decoder may use any state)"
)


def restriction_inside_swr_values(
    time_sums: pd.DataFrame, bin_s: pd.Series, source: str
) -> list[Value]:
    """Question 2c.3: the full decoder against the SWR-only decoder over the
    immobile time inside SWRs inside a well visit, per spike and per second,
    with the statistic, interval and epoch counts of the gain outside SWRs
    (``restriction_totals.inside_swr_gain_statistics``). The gain per spike in
    :data:`RESTRICTION_FIGURE_FAMILY` is drawn on Figure 2k's first axis and
    carries its tag; the rest is for the text."""
    make_text = value_factory(
        restriction_inside_swr_values, source, TAGS["nonlocal_restriction"]
    )
    make_figure = value_factory(
        restriction_inside_swr_values, source, TAGS["nonlocal_restriction_figure"]
    )
    statistics = restriction_totals.inside_swr_gain_statistics(
        restriction_totals.epoch_inside_swr_gains(time_sums, bin_s)
    ).set_index(["family", "measure"])
    values = []
    for family, suffix in FAMILY_SUFFIX.items():
        key = f"restriction_inside_swr_full_minus_swr{suffix}"
        scored = f"{_RESTRICTION_SCORES[family]}, {_RESTRICTION_HELD_OUT}"
        for unit in _RESTRICTION_UNITS:
            drawn = (
                family == RESTRICTION_FIGURE_FAMILY
                and f"gain_{unit[0]}" == RESTRICTION_FIGURE_INSIDE_SWR
            )
            values += _full_minus_swr_values(
                make_figure if drawn else make_text,
                key,
                statistics.loc[(family, f"gain_{unit[0]}")],
                unit,
                scored,
                f"every one of the epoch's {_INSIDE_SWR_BINS} inside a well visit",
                f"{_INSIDE_SWR_BINS} inside a well visit",
                "epochs whose gain per second inside SWRs is above zero",
            )
    return values


def restriction_immobile_values(
    time_sums: pd.DataFrame, bin_s: pd.Series, source: str
) -> list[Value]:
    """Question 2c.3: the SWR-only and full decoders against the local-only
    decoder over all immobile time inside a well visit (time chosen by speed and
    the well visits, not by the decoder or the SWRs), per spike and per second,
    their difference, and the share of the full decoder's gain the SWR-only
    decoder captures. No panel of Figure 2 draws these (Figure 2's supplement,
    panel l, draws the decoders over the local-only decoder of the run with
    the fourth decoder)."""
    make_text = value_factory(
        restriction_immobile_values, source, TAGS["nonlocal_restriction"]
    )
    statistics = restriction_totals.immobile_gain_statistics(
        restriction_totals.epoch_immobile_gains(time_sums, bin_s)
    ).set_index(["family", "measure"])
    values = []
    for family, suffix in FAMILY_SUFFIX.items():
        scored = (
            f"{_RESTRICTION_SCORES[family]}, {_RESTRICTION_HELD_OUT}, summed over "
            "every immobile bin of the epoch (inside and outside SWRs)"
        )
        for unit, divided, words in _RESTRICTION_UNITS:
            for contrast in ("swr_minus_none", "full_minus_none", "full_minus_swr"):
                measure = f"{contrast}_{unit}"
                a, b = restriction_windows.CONTRASTS[contrast]
                values += _interval_and_animals_positive(
                    make_text,
                    f"restriction_immobile_{contrast}{suffix}_{unit}",
                    statistics.loc[(family, measure)],
                    f"{scored}, under {_RESTRICTION_MODELS[a]} minus under "
                    f"{_RESTRICTION_MODELS[b]}, divided by {divided}: "
                    f"{_EPOCH_REDUCTION}",
                    unit=words,
                )
        row = statistics.loc[(family, "full_minus_swr_per_spike")]
        values.append(
            make_text(
                f"restriction_immobile_epochs_full_above_swr_only{suffix}",
                row["n_epochs_full_above_swr_only"],
                "count",
                "count",
                f"epochs whose summed score ({family}) over their immobile bins "
                "is higher under "
                f"{_RESTRICTION_MODELS['full']} than under "
                f"{_RESTRICTION_MODELS['swr']}",
            )
        )
        values += _share_interval(
            make_text,
            f"restriction_immobile_share_captured_by_swr_only{suffix}",
            statistics.loc[(family, "share_captured_by_swr_only")],
            f"{scored}: the gain of {_RESTRICTION_MODELS['swr']} over "
            f"{_RESTRICTION_MODELS['none']} as a share of the gain of "
            f"{_RESTRICTION_MODELS['full']} over it: {_EPOCH_REDUCTION}",
        )
    row = statistics.loc[(RESTRICTION_FIGURE_FAMILY, "full_minus_swr_per_spike")]
    values.append(
        make_text(
            "restriction_immobile_epochs",
            row["n_epochs"],
            "count",
            "count",
            "epochs with immobile bins in the restriction comparison",
        )
    )
    return values


#: Each time of ``restriction_totals.CONTRAST_TIMES``, in words: the bins a
#: contrast is summed over.
_CONTRAST_TIME_BINS = {
    "inside_swr": f"{_INSIDE_SWR_BINS} inside a well visit",
    "outside_swr": f"{_OUTSIDE_SWR_BINS} inside a well visit",
    "immobile": "immobile bins inside a well visit (inside SWRs, outside SWRs, "
    "and where SWR detection was not observed, which the sum outside SWRs "
    "leaves out)",
}
_SWR_ALLOWED_RUN = (
    "every decoder scored in one pass under the stored fold fits "
    "(scripts/event_evidence/score_nonlocal_restriction.py --from-stored-fit)"
)


def restriction_swr_allowed_values(
    windows: pd.DataFrame, time_sums: pd.DataFrame, bin_s: pd.Series, source: str
) -> list[Value]:
    """Question 2c.3: the SWR-allowed decoder (every state inside SWRs, only the
    local states outside them) against the other three, from a restriction run
    that scored all four in one pass (``restriction_totals.swr_allowed_statistics``).

    Over the immobile time inside SWRs, outside SWRs and all of it inside a well
    visit (``restriction_totals.CONTRAST_TIMES``), each contrast of
    ``nonlocal_restriction.SWR_ALLOWED_CONTRASTS`` per spike and per second, in
    both score families, with the statistic, interval and epoch counts of the
    gain inside SWRs (keys ``restriction_<time>_<contrast>...``); over all the
    immobile time also the SWR-only and full decoders' gains over the local-only
    decoder of the same pass (keys ``restriction_swr_allowed_run_immobile_...``);
    and the full minus the SWR-allowed decoder per spike over the bins outside
    SWRs of the no-SWR events detected from that run's held-out decode
    (``windows``, those that start inside a well visit), with the events and
    those with a value. The values Figure 2's supplement draws on its
    fourth-decoder panel (``restriction_totals.SWR_ALLOWED_FIGURE_*``, in
    :data:`RESTRICTION_FIGURE_FAMILY` per spike) carry its tag (S2l); the rest
    is for the text."""
    make_text = value_factory(
        restriction_swr_allowed_values, source, TAGS["nonlocal_restriction"]
    )
    make_figure = value_factory(
        restriction_swr_allowed_values,
        source,
        TAGS["nonlocal_restriction_swr_allowed"],
    )
    statistics = restriction_totals.swr_allowed_statistics(windows, time_sums, bin_s)
    values = []
    for time, contrasts in SWR_ALLOWED_TIME_CONTRASTS.items():
        bins = _CONTRAST_TIME_BINS[time]
        drawn = (
            *SWR_ALLOWED_FIGURE_DIFFERENCES,
            *(SWR_ALLOWED_FIGURE_DECODERS if time == "immobile" else ()),
        )
        for contrast in contrasts:
            a, b = ALL_CONTRASTS[contrast]
            prefix = (
                "restriction"
                if contrast in SWR_ALLOWED_CONTRASTS
                else "restriction_swr_allowed_run"
            )
            for family, suffix in FAMILY_SUFFIX.items():
                for unit in _RESTRICTION_UNITS:
                    measure = f"{contrast}_{unit[0]}"
                    figure = family == RESTRICTION_FIGURE_FAMILY and measure in drawn
                    values += _full_minus_swr_values(
                        make_figure if figure else make_text,
                        f"{prefix}_{time}_{contrast}{suffix}",
                        statistics.times.loc[(time, family, measure)],
                        unit,
                        f"{_RESTRICTION_SCORES[family]}, {_RESTRICTION_HELD_OUT}, "
                        f"{_SWR_ALLOWED_RUN}",
                        f"every one of the epoch's {bins}",
                        bins,
                        "epochs whose value per second is above zero",
                        difference=f"under {_RESTRICTION_MODELS[a]} minus under "
                        f"{_RESTRICTION_MODELS[b]}",
                    )
    group = RESTRICTION_FIGURE_EVENT_GROUP
    row = statistics.events.loc[group]
    word = _GROUP_WORDS[group]
    a, b = ALL_CONTRASTS[restriction_totals.SWR_ALLOWED_FIGURE_EVENT_CONTRAST]
    values += _interval_and_animals_positive(
        make_figure,
        f"restriction_heldout_{word}_events_outside_swr_full_minus_swr_allowed"
        "_per_spike",
        row,
        f"{_RESTRICTION_SCORES[RESTRICTION_FIGURE_FAMILY]}, "
        f"{_RESTRICTION_HELD_OUT}, {_SWR_ALLOWED_RUN}, under "
        f"{_RESTRICTION_MODELS[a]} minus under {_RESTRICTION_MODELS[b]}, summed "
        f"{_OUTSIDE_SWR_WHERE[group]} and divided by their spikes: mean over "
        "animals of each animal's median over epochs of each epoch's median",
    )
    values += [
        make_figure(
            f"restriction_heldout_swr_allowed_run_{word}_events",
            row["n_events"],
            "count",
            "count",
            f"{group} events detected from the held-out decode of the run that "
            "scored the SWR-allowed decoder",
        ),
        make_figure(
            f"restriction_heldout_swr_allowed_run_{word}_events_outside_swr_scored"
            "_per_spike",
            row["n_scored"],
            "count",
            "count",
            f"{group} events of that run with a value per spike outside SWRs "
            "(immobile bins there and a spike in them)",
        ),
    ]
    return values


# --- one class or two -----------------------------------------------------------

#: The properties of the step test (``class_or_continuum.STEP_PROPERTIES``)
#: other than spike evidence, by their word in ``step_fits.csv``.
OTHER_STEP_PROPERTIES = {
    "duration": "duration",
    "extent": "spatial extent",
    "speed": "replay speed",
    "posterior": "posterior peak",
    "hpd": "HPD coverage",
    "spikes": "spike count",
    "breadth": "arms visited",
}
STEP_KNOTS = 8
STEP_WINDOW_Z = 1.5
"""The step test's main settings (``measure_ripple_class_structure.py``'s defaults)."""


def class_distribution_values(distribution: pd.DataFrame, source: str) -> list[Value]:
    """Question 2b.5, test 1: is the events' peak ripple power unimodal, and where is
    its mode against the Kay threshold?

    Per animal, Hartigan's dip test and the KDE's main mode of the event's peak
    ripple power (the z-scored Kay consensus on the decode grid). The dip
    p-values, the other power measures and the simulation checks stay in the
    analysis's CSV outputs. No figure draws these: the dip test has no power
    at the observed separation (the analysis's simulation); the step test
    (:func:`class_step_values`) is for the text too.
    """
    make = value_factory(class_distribution_values, source, TAGS["class_distribution"])
    stats = event_statistics.peak_ripple_mode_statistics(
        distribution, n_animals=len(analyzed_animals())
    )
    return [
        make(
            "ripple_peak_animals_unimodal",
            stats["n_animals_unimodal"],
            "count",
            "count",
            "animals in which Hartigan's dip test does not reject one mode of the "
            "events' peak ripple power (p at or above 0.05)",
        ),
        make(
            "ripple_peak_mode_animal_min",
            stats["mode_animal_min"],
            "quantity",
            "dec2",
            "lowest per-animal location of the main KDE mode of the events' peak "
            "ripple power",
            "z",
        ),
        make(
            "ripple_peak_mode_animal_max",
            stats["mode_animal_max"],
            "quantity",
            "dec2",
            "highest per-animal location of that mode",
            "z",
        ),
    ]


def nonlocal_power_values(counts: pd.DataFrame, source: str) -> list[Value]:
    """Question 2b.5: how much of the non-local time has ripple power below the
    Kay threshold (Figure 2i)?

    Over the immobile time bins decoded non-local, each animal's bins pooled
    over its epochs (``counts``: the stored per-epoch counts by signal and
    non-local probability, ``nonlocal_by_signal.rest_signal_counts``).
    """
    make = value_factory(nonlocal_power_values, source, TAGS["nonlocal_power"])
    stats = event_statistics.nonlocal_below_threshold_statistics(
        nonlocal_by_signal.animal_nonlocal_counts(counts, "ripple_power"),
        KAY_ZSCORE_THRESHOLD,
    )
    key = "nonlocal_time_below_swr_threshold_fraction"
    return [
        make(
            key,
            stats["fraction_animal_mean"],
            "fraction",
            "pct",
            "share of the immobile time bins decoded non-local (non-local "
            f"probability above {DEFAULT_NON_LOCAL_THRESHOLD:g}) whose ripple power "
            f"is below the Kay SWR threshold ({KAY_ZSCORE_THRESHOLD:g} z): mean over "
            "animals of each animal's share of its bins",
        ),
        make(
            f"{key}_animal_min",
            stats["fraction_animal_min"],
            "fraction",
            "pct",
            "lowest per-animal share of non-local time bins below the Kay threshold",
        ),
        make(
            f"{key}_animal_max",
            stats["fraction_animal_max"],
            "fraction",
            "pct",
            "highest per-animal share",
        ),
    ]


def nonlocal_probability_values(counts: pd.DataFrame, source: str) -> list[Value]:
    """Question 2a.6 and Figure 2's supplement (S2c): how the immobile time
    inside the well visits is spread over the decoder's non-local probability,
    as three shares of the time, at most 0.2, between 0.2 and 0.8 (the event
    rule's cut), and above 0.8, each with the smallest and largest animal
    (``nonlocal_probability.probability_share_statistics``: each epoch's share,
    each animal's mean over its epochs, then the mean over animals)."""
    make = value_factory(
        nonlocal_probability_values, source, TAGS["nonlocal_probability"]
    )
    statistics = nonlocal_probability.probability_share_statistics(counts)
    reduction = (
        "each epoch's share, each animal's mean over its epochs, the mean over animals"
    )
    values = []
    for part, words in nonlocal_probability.SHARES.items():
        row = statistics.loc[part]
        what = (
            "share of the immobile time inside the well visits whose non-local "
            f"probability (Continuous plus Fragmented) is {words}"
        )
        values += _animal_range_values(
            make,
            f"nonlocal_probability_{part}_share",
            (row["animal_mean"], row["animal_min"], row["animal_max"]),
            (
                f"{what}: {reduction}",
                f"smallest animal's {what} (the mean over its epochs)",
                f"largest animal's {what} (the mean over its epochs)",
            ),
        )
    return values


#: Each signal of ``state_by_signal``: its word in a key, its name in text and
#: its detection level (z), an edge of its bins.
_STATE_SIGNALS = {
    "ripple_power": (
        "ripple",
        "ripple power",
        KAY_ZSCORE_THRESHOLD,
        "the SWR threshold",
    ),
    "mua": ("mua", "MUA", nonlocal_by_signal.MUA_LEVEL_Z, "the MUA event threshold"),
}
#: Each measure of ``state_by_signal.detection_level_statistics``: its word in
#: a key, its style and what it is (``{signal}`` the signal, ``{level}`` the
#: detection level in words).
_STATE_MEASURES = {
    "nonlocal_share_at_level": (
        "nonlocal_share_at_detection_level", "pct",
        "share of the confident non-local time (non-local probability above 0.8) "
        "with {signal} at or above {level}",
    ),
    "local_share_at_level": (
        "local_share_at_detection_level", "pct1",
        "share of the confident Local time (Local probability above 0.8) with "
        "{signal} at or above {level}",
    ),
}  # fmt: skip


def state_by_signal_values(counts: pd.DataFrame, source: str) -> list[Value]:
    """Question 2b.5: how much of the confident non-local time and of the
    confident Local time has ripple power, or MUA, at or above the detection
    level (the SWR threshold; the MUA event threshold), over the immobile time
    inside the well visits (``state_by_signal.detection_level_statistics``:
    each epoch's share, each animal's mean over its epochs, the mean over
    animals with the smallest and largest animal). No figure draws them (the
    top strips of Figure 2i and of its supplement's MUA panel draw the two
    distributions)."""
    make = value_factory(state_by_signal_values, source, TAGS["state_by_signal"])
    reduction = (
        "each epoch's share, each animal's mean over its epochs, the mean over animals"
    )
    values = []
    for signal, (word, name, level, level_words) in _STATE_SIGNALS.items():
        statistics = state_by_signal.detection_level_statistics(counts, signal, level)
        for measure, (key_word, style, what) in _STATE_MEASURES.items():
            row = statistics.loc[measure]
            text = what.format(signal=name, level=f"{level_words} ({level:g} z)")
            key = f"{word}_{key_word}"
            values += [
                make(key, row["animal_mean"], "fraction", style, f"{text}: {reduction}"),
                make(
                    f"{key}_animal_min", row["animal_min"], "fraction",
                    "pctnum" if style == "pct" else style,
                    f"smallest animal's {text} (the mean over its epochs)",
                ),
                make(
                    f"{key}_animal_max", row["animal_max"], "fraction", style,
                    f"largest animal's {text} (the mean over its epochs)",
                ),
            ]  # fmt: skip
    return values


def nonlocal_mua_values(counts: pd.DataFrame, source: str) -> list[Value]:
    """Non-local time against MUA, the MUA panel of Figure 2's supplement: the
    share of the immobile time bins with MUA at or above
    ``nonlocal_by_signal.MUA_LEVEL_Z`` that are decoded non-local, and the
    share of the bins decoded non-local whose MUA is below it
    (``event_statistics.nonlocal_share_above_statistics`` and
    ``nonlocal_below_threshold_statistics`` of the stored counts, as the panel
    reads them). Most non-local time is below the MUA level, and nearly all
    time above it is decoded non-local; this does not show that non-local
    detection is independent of population bursts (the held-out tetrode values
    address whether the spikes point to the decoded place)."""
    make = value_factory(nonlocal_mua_values, source, TAGS["nonlocal_mua"])
    animals = nonlocal_by_signal.animal_nonlocal_counts(counts, "mua")
    level = nonlocal_by_signal.MUA_LEVEL_Z
    above = event_statistics.nonlocal_share_above_statistics(animals, level)
    below = event_statistics.nonlocal_below_threshold_statistics(animals, level)
    nonlocal_ = f"non-local probability above {DEFAULT_NON_LOCAL_THRESHOLD:g}"
    level_words = (
        f"{level:g} z of MUA (MUA_zscore, the z-scored multiunit rate of the decoded "
        "tetrodes; the value of the MUA event detector's threshold, which z-scores "
        "its own smoothed rate)"
    )
    key_above = "nonlocal_share_mua_above_level"
    key_below = "nonlocal_time_below_mua_level_fraction"
    return [
        make(
            "mua_level_z",
            level,
            "quantity",
            "int",
            f"the MUA level of the values below: {level_words}",
            "z",
        ),
        *_animal_range_values(
            make,
            key_above,
            (
                above["share_animal_mean"],
                above["share_animal_min"],
                above["share_animal_max"],
            ),
            (
                f"share of the immobile time bins with MUA at or above {level_words} "
                f"that are decoded non-local ({nonlocal_}): mean over animals of each "
                "animal's share of its bins. Nearly all time above the MUA level is "
                "decoded non-local: the decoder needs spikes, so this does not show "
                "that non-local detection is independent of population bursts",
                f"lowest per-animal share of the bins with MUA at or above {level:g} z "
                "that are decoded non-local",
                "highest per-animal share",
            ),
            lowest_style="pct",
        ),
        *_animal_range_values(
            make,
            key_below,
            (
                below["fraction_animal_mean"],
                below["fraction_animal_min"],
                below["fraction_animal_max"],
            ),
            (
                f"share of the immobile time bins decoded non-local ({nonlocal_}) whose "
                f"MUA is below {level_words}: mean over animals of each animal's share "
                "of its bins. Most non-local time is below the MUA level",
                f"lowest per-animal share of non-local time bins with MUA below {level:g} z",
                "highest per-animal share",
            ),
            lowest_style="pct",
        ),
    ]


#: Each swept setting's word in a key and in a description, and each value's
#: word in a key (which holds no digit).
_SWEEP_SETTINGS = {
    "probability_threshold": (
        "probability_threshold",
        "the event rule's non-local probability an event stays strictly above",
        dict(
            zip(
                rule_sweep.PROBABILITY_THRESHOLDS,
                ("half", "point_six", "point_seven", "point_eight", "point_nine",
                 "point_nine_five"),
                strict=True,
            )
        ),
    ),
    "minimum_samples": (
        "minimum_samples",
        "the event rule's minimum run length (2 ms decode samples; 7 samples, "
        "the rule's 15 ms, is a shortest event of 14 ms, and n samples one of "
        "2n ms)",
        dict(
            zip(
                rule_sweep.MINIMUM_SAMPLES,
                ("seven", "ten", "fifteen", "twenty_five", "thirty_eight", "fifty"),
                strict=True,
            )
        ),
    ),
    "swr_threshold_z": (
        "swr_threshold",
        "the SWR detector's threshold, not a setting of the event rule (z of the "
        "Kay consensus ripple power; an event is counted as overlapping an SWR "
        "when its Kay criterion on the decode grid is at or above it; the share "
        "with no SWR changes strongly with it, as a threshold on one "
        "distribution must)",
        dict(
            zip(
                rule_sweep.SWR_THRESHOLDS_Z,
                ("zero", "half", "one", "one_and_half", "two", "two_and_half",
                 "three", "three_and_half", "four", "four_and_half", "five",
                 "five_and_half", "six"),
                strict=True,
            )
        ),
    ),
}  # fmt: skip


def rule_sweep_values(counts: pd.DataFrame, source: str) -> list[Value]:
    """Question 2a.5 and the sweep panel of Figure 2's supplement: the share of
    the immobile non-local events that start inside a well visit with no SWR,
    and the events against the rule's, at each setting of the event rule
    (``rule_sweep.sweep_statistics``, which the panel draws), and how often the
    Kay criterion's label at 2 z is the canonical one
    (``rule_sweep.criterion_agreement``). A sensitivity check: the canonical
    events and every other value are the rule's. The share with no SWR changes
    little with the event rule's two settings and strongly with the SWR
    detector's threshold, as a threshold on one distribution must (the text
    cites the threshold axis with the ripple-power panels); about a third of
    the events have no SWR even at 1 z."""
    make = value_factory(rule_sweep_values, source, TAGS["rule_sweep"])
    values = []
    for row in rule_sweep.sweep_statistics(counts).itertuples():
        setting_word, words, value_words = _SWEEP_SETTINGS[row.setting]
        value = int(row.value) if row.setting == "minimum_samples" else row.value
        key = f"rule_sweep_{setting_word}_{value_words[value]}"
        at = f"with {words} at {value:g}" + (
            " (the value the paper uses)" if row.is_rule else ""
        )
        values += [
            *_animal_range_values(
                make,
                f"{key}_no_swr_fraction",
                (row.no_swr_fraction, row.animal_min, row.animal_max),
                (
                    f"share of the events detected again {at} that overlap no SWR, "
                    "pooled over events",
                    f"lowest per-animal share with no SWR {at} (an animal's share is the "
                    "mean over its epochs of the epoch's share)",
                    "highest per-animal share",
                ),
                animal_mean=(
                    row.no_swr_fraction_animal_mean,
                    f"share of the events detected again {at} that overlap no SWR, "
                    "the mean over animals of each animal's share (the mean over its "
                    "epochs of the epoch's share; the panel's bold line)",
                ),
                pooled_tags=TAGS["rule_sweep_pooled"],
            ),
            make(
                f"{key}_events_vs_rule",
                row.events_vs_rule,
                "quantity",
                "dec2",
                f"the events detected again {at}, as a multiple of the rule's",
            ),
            make(
                f"{key}_events",
                row.n_events,
                "count",
                "count",
                f"the events detected again {at}, summed over the analyzed epochs",
            ),
        ]
    values.append(
        make(
            "rule_sweep_swr_criterion_agreement",
            rule_sweep.criterion_agreement(counts),
            "fraction",
            "pct",
            "share of the events whose Kay criterion on the decode grid gives their "
            "canonical SWR label at the detector's 2 z (the stand-in by which the "
            "SWR threshold is swept, since the SWRs are not detected again)",
        )
    )
    return values


#: Each calibrated state's word in a key, and what its evidence compares.
_CALIBRATED_STATES = {
    "Local": (
        "local",
        "the held-out tetrodes' spikes scored at the animal's place against at "
        "mismatched places (other windows' animal positions at least 20 cm away)",
    ),
    "Non-Local Continuous": (
        "continuous",
        "the place decoded from one half of the tetrodes scored on the other half "
        "against mismatched places (other windows' decoded places)",
    ),
    "Non-Local Fragmented": (
        "fragmented",
        "the place decoded from one half of the tetrodes scored on the other half "
        "against mismatched places (other windows' decoded places)",
    ),
}
_CALIBRATION_CAVEATS = (
    "the probability bins come from the decode with every tetrode, so the "
    "scored half's spikes helped make the state probability; only the remote "
    "place is held out by the tetrode split (one half chooses it, the other "
    "scores it); the place-field encodings are fit on running time in other "
    "time blocks (alternating 30 s blocks), so a window's own spikes are not in "
    "its encoding"
)


def state_calibration_values(
    bins: pd.DataFrame, trends: pd.DataFrame, source: str
) -> list[Value]:
    """Question 2a.7 and the state calibration panel of Figure 2's supplement:
    over the immobile calibration windows inside a well visit, binned by a
    state's probability, the held-out evidence for what the state claims, in
    the top and the bottom bins, and the rise over the bins (each epoch's
    slope), as ``state_evidence.calibration_statistics`` wrote them (the
    animal-level estimate and its nested bootstrap interval)."""
    make = value_factory(state_calibration_values, source, TAGS["state_calibration"])
    reduction = (
        "per epoch the median over its windows, then the mean over animals of "
        "each animal's mean epoch value; 95 % interval from the nested bootstrap "
        "of the animals and their epochs, small-sample t correction"
    )
    values = []
    for state, (word, what) in _CALIBRATED_STATES.items():
        mine = bins[bins["state"] == state].set_index("pbin")
        trend = trends[trends["state"] == state].iloc[0]
        key = f"state_calibration_{word}"
        for end, label, place in (
            ("top_bin", mine.index[-1], "the top"),
            ("bottom_bin", mine.index[0], "the bottom"),
        ):
            row = mine.loc[label]
            values.append(
                make(
                    f"{key}_{end}_per_spike",
                    row["animal_estimate"],
                    "quantity",
                    "sig2",
                    f"held-out evidence for {state} over the random 10-bin "
                    f"immobile windows whose {state} probability is in {place} bin "
                    f"({label}): {what}, the spike-mark log likelihood ratio per "
                    f"spike of the scored tetrodes (each run's ratio over the scored "
                    f"half's spikes, averaged over the runs in which that half "
                    f"spiked; a window in which it never spiked has no value); "
                    f"{reduction}. Caveats: {_CALIBRATION_CAVEATS}",
                    "nats per spike",
                )
            )
            if end == "top_bin":
                values += [
                    make(f"{key}_top_bin_per_spike_ci_low", row["animal_ci_lo"], "quantity",
                         "sig2", "lower bound of its 95 % interval", "nats per spike"),
                    make(f"{key}_top_bin_per_spike_ci_high", row["animal_ci_hi"], "quantity",
                         "sig2", "upper bound of that interval", "nats per spike"),
                ]  # fmt: skip
        values += [
            make(
                f"{key}_slope_per_spike",
                trend["slope_animal_estimate"],
                "quantity",
                "sig2",
                f"rise of that held-out evidence for {state} with the state's "
                "probability: each epoch's least-squares slope over its windows "
                "(nats per spike per unit of probability), then the mean over "
                "animals of each animal's mean epoch slope",
                "nats per spike per unit probability",
            ),
            make(f"{key}_slope_per_spike_ci_low", trend["slope_animal_ci_lo"], "quantity",
                 "sig2", "lower bound of its 95 % interval (nested bootstrap)",
                 "nats per spike per unit probability"),
            make(f"{key}_slope_per_spike_ci_high", trend["slope_animal_ci_hi"], "quantity",
                 "sig2", "upper bound of that interval",
                 "nats per spike per unit probability"),
        ]  # fmt: skip
    first = bins[bins["state"] == next(iter(_CALIBRATED_STATES))]
    values.append(
        make(
            "state_calibration_windows",
            int(first["n"].sum()),
            "count",
            "count",
            "random 10-bin immobile calibration windows that start inside a well "
            "visit and have a held-out score for Local (a scored spike), over the "
            "analyzed "
            "epochs",
        )
    )
    return values


def _step_statistics(steps: pd.DataFrame) -> pd.DataFrame:
    """The step test's statistics per property, at the main settings."""
    return event_statistics.threshold_step_statistics(
        steps,
        n_knots=STEP_KNOTS,
        window_z=STEP_WINDOW_Z,
        n_animals=len(analyzed_animals()),
    )


def class_step_values(steps: pd.DataFrame, source: str) -> list[Value]:
    """Question 2b.5 and the property questions, test 2: is there a step at the threshold?

    A sharp discontinuity design on the Kay criterion around each event (the
    level the ripple it touches sustains for 14 ms): per animal and property,
    ``y ~ s(criterion)`` against ``y ~ s(criterion) + (criterion >= 2 z)``
    (cubic B-spline, 8 knots) on the events within 1.5 z of the threshold. An
    animal has a step when the term's 95 % interval (clustered by epoch)
    excludes 0 and it lowers the held-out squared error (5 folds grouped by
    epoch). The share is the term over the animal's SWR minus non-SWR difference
    over all its events. The per-property counts, the version with log
    duration, the version on the event's own power, the sensitivity settings and
    the simulation checks stay in the CSV outputs.

    This is the value over the properties other than spike evidence, which are
    read from the event table and the event features run;
    :func:`class_step_evidence_values` holds spike evidence's. No figure draws
    them.
    """
    most, attaining = event_statistics.most_steps(
        _step_statistics(steps), list(OTHER_STEP_PROPERTIES)
    )
    # the note names the property (or properties) that attain the maximum
    attained = ", ".join(OTHER_STEP_PROPERTIES[p] for p in attaining)
    make = value_factory(class_step_values, source, TAGS["class_step"])
    return [
        make(
            "step_other_properties_criterion_animals_max",
            most,
            "count",
            "count",
            "the most animals with a step at the threshold in any property but spike "
            "evidence (duration, spatial extent, replay speed, posterior peak, HPD "
            "coverage, spike count, arms visited): the term's interval excludes 0 and "
            f"it lowers the held-out error; attained by {attained}",
        ),
    ]


def class_step_evidence_values(steps: pd.DataFrame, source: str) -> list[Value]:
    """Question 2b.5, test 2 (:func:`class_step_values`) for spike
    evidence per spike: the one property of the step test read from the
    spike-evidence run, so these values are stale when that run has changed
    since the step fits were made (:func:`spike_evidence_staleness`)."""
    evidence = _step_statistics(steps).loc["evidence"]
    make_evidence = value_factory(
        class_step_evidence_values, source, TAGS["class_step_evidence"]
    )
    return [
        make_evidence(
            "step_evidence_criterion_animals",
            evidence["n_animals_with_step"],
            "count",
            "count",
            "animals with a step at the threshold in spike evidence per spike (remote "
            "events): the term's interval excludes 0 and it lowers the held-out error",
        ),
        make_evidence(
            "step_evidence_criterion_animals_same_sign",
            evidence["n_animals_same_sign"],
            "count",
            "count",
            "animals whose jump in spike evidence per spike at the threshold points "
            "the way of their SWR minus non-SWR difference, whatever its interval",
        ),
        make_evidence(
            "step_evidence_criterion_share_median",
            evidence["step_share_median"],
            "quantity",
            "sig2s",
            "median over animals of the jump in spike evidence per spike at the "
            "threshold, as a share of the animal's SWR minus non-SWR difference",
        ),
    ]


def class_step_values_with_staleness(
    manifest: dict,
    spike_evidence_sha256: str,
    event_table_stale: str | None,
    steps: pd.DataFrame,
    source: str,
    what: str,
) -> list[Value]:
    """The step test's values (:func:`class_step_values` and
    :func:`class_step_evidence_values`), pending where the step fits are stale.

    Every value is pending with ``event_table_stale`` (the step fits were made
    from another event table); spike evidence's values are also pending when the
    step fits' ``manifest`` does not record the current spike-evidence run
    (``spike_evidence_sha256``, :func:`spike_evidence_staleness`; ``what`` names
    the step fits' folder in the note).
    """
    evidence_stale = spike_evidence_staleness(manifest, spike_evidence_sha256, what)
    return [
        *pend(class_step_values(steps, source), event_table_stale),
        *pend(
            class_step_evidence_values(steps, source), event_table_stale, evidence_stale
        ),
    ]


# --- properties ---------------------------------------------------------------

#: (word, column, scale, unit, style, the expected direction for non-SWR against
#: SWR, tags)
_PROPERTIES = (
    (
        "duration",
        "event_duration_s",
        MS_PER_S,
        "ms",
        "int",
        "lower",
        TAGS["duration"],
    ),
    (
        "extent",
        "spatial_extent_cm",
        1.0,
        "cm",
        "sig2",
        "lower",
        TAGS["extent"],
    ),
    (
        "speed",
        "replay_speed_median_cm_s",
        1.0,
        "cm/s",
        "sig2",
        "lower",
        TAGS["speed"],
    ),
    (
        "peak",
        "posterior_peak_mean",
        1.0,
        "",
        "dec2",
        "higher",
        TAGS["certainty"],
    ),
    ("hpd", "hpd_coverage_cm", 1.0, "cm", "sig2", "lower", TAGS["certainty"]),
    (
        "pause",
        "pause_fraction",
        1.0,
        "",
        "dec2",
        "higher",
        TAGS["stillness"],
    ),
    (
        "max_step",
        "step_distance_max_cm",
        1.0,
        "cm",
        "sig2",
        "lower",
        TAGS["stillness"],
    ),
    (
        "reversals",
        "n_reversals",
        1.0,
        "",
        "int",
        "lower",
        TAGS["stillness"],
    ),
    (
        "multimodal",
        "multimodal_fraction",
        1.0,
        "",
        "dec2",
        "lower",
        TAGS["certainty"],
    ),
    (
        "start_distance",
        "start_distance_cm",
        1.0,
        "cm",
        "sig2",
        "lower",
        TAGS["location"],
    ),
    (
        "arms",
        "n_unique_arms",
        1.0,
        "",
        "int",
        "lower",
        TAGS["arms"],
    ),
)


_PROPERTY_NAMES = {
    "duration": "event duration",
    "extent": "spatial extent of the decoded content",
    "speed": "replay speed (median over the event)",
    "peak": "posterior peak (mean over the event)",
    "hpd": "highest-density-region coverage",
    "pause": "pause fraction",
    "max_step": "largest step of the decoded position",
    "reversals": "number of direction reversals",
    "multimodal": "multimodal fraction of the posterior",
    "start_distance": "start distance (track-graph distance from the animal to the "
    "event's first decoded position)",
    "arms": "number of arms the decoded position visits",
}
"""How each property's values describe it."""


def property_values(events: pd.DataFrame, source: str) -> list[Value]:
    """Questions 2d.1–2d.6: per-event properties, non-SWR against SWR.

    Plain medians of all events, and the number of animals whose own medians
    differ in the claim's direction. Not duration-controlled. The properties
    are the event table's own columns; the start distance, read from another
    output, is made by :func:`start_distance_values`.
    """
    values = []
    for prop in _PROPERTIES:
        if prop[0] != "start_distance":
            values += _class_median_values(
                value_factory(property_values, source, prop[6]), events, prop
            )
    return values


def start_distance_values(events: pd.DataFrame, source: str) -> list[Value]:
    """Question 2d.1: the start distance of events, non-SWR against SWR, as
    :func:`property_values`; read from the start-distance output's
    ``start_distance_cm``."""
    (prop,) = (p for p in _PROPERTIES if p[0] == "start_distance")
    return _class_median_values(
        value_factory(start_distance_values, source, prop[6]), events, prop
    )


def _class_median_values(
    make: Callable[..., Value], events: pd.DataFrame, prop: tuple
) -> list[Value]:
    """One property's class medians and animals in direction, made by ``make``;
    ``prop`` is its row of :data:`_PROPERTIES`."""
    word, column, scale, unit, style, direction, _ = prop
    stats = event_statistics.class_medians(events, column)
    name = _PROPERTY_NAMES[word]
    return [
        make(
            f"{word}_median_non_swr",
            scale * stats["median_non_swr"],
            "quantity",
            style,
            f"median {name} of immobile non-SWR non-local events",
            unit,
        ),
        make(
            f"{word}_median_swr",
            scale * stats["median_swr"],
            "quantity",
            style,
            f"median {name} of immobile SWR-overlapping non-local events",
            unit,
        ),
        make(
            f"{word}_animals_in_direction",
            event_statistics.n_animals_in_direction(events, column, direction),
            "count",
            "count",
            f"animals whose non-SWR median {name} is {direction} than their SWR "
            "median, each the median over the animal's epochs of the epoch's median",
        ),
    ]


def hpd_tight_values(events: pd.DataFrame, source: str) -> list[Value]:
    """Question 2d.5 and the posterior-spread panel of Figure 2's supplement (S2k):
    the share of each class's events with a tight posterior, HPD coverage below
    ``event_statistics.HPD_TIGHT_CM``
    (``event_statistics.hpd_tight_share_statistics``), the animals in which the
    no-SWR share is the higher, and the cut. Each class holds a tight group of
    events and a diffuse one, so a median falls between them; the pooled class
    medians (``hpd_median_*``, :func:`property_values`) carry no figure tag."""
    make = value_factory(hpd_tight_values, source, TAGS["hpd"])
    result = event_statistics.hpd_tight_share_statistics(events)
    rows = result["rows"]
    cut = event_statistics.HPD_TIGHT_CM
    what = (
        f"share of the events whose HPD coverage, the length of track that holds "
        f"95 % of the decoded probability averaged over the event's bins, is below "
        f"{cut:g} cm"
    )
    reduction = (
        "each epoch's share, each animal's median over its epochs, the mean over "
        "animals; 95 % interval from the nested bootstrap of the animals and "
        "their epochs, small-sample t correction"
    )
    values = [
        make(
            "hpd_tight_cut_cm",
            cut,
            "quantity",
            "int",
            "the HPD coverage below which an event's posterior counts as tight, "
            "between the tight and the diffuse group of events",
            "cm",
        )
    ]
    for group, word, name in (
        ("no SWR", "non_swr", "non-SWR"),
        ("SWR", "swr", "SWR-overlapping"),
    ):
        row = rows.loc[group]
        key = f"hpd_tight_share_{word}"
        values += _share_interval(
            make, key, row, f"{what}, of immobile {name} non-local events: {reduction}"
        )
    values.append(
        make(
            "hpd_tight_share_animals_non_swr_higher",
            result["n_animals_non_swr_higher"],
            "count",
            "count",
            "animals whose share of non-SWR events with a tight posterior (the "
            "median over its epochs) is above their share of SWR events",
        )
    )
    return values


#: Each distance band's word in a key and what it holds
#: (``event_statistics.DISTANCE_BANDS_CM``).
_DISTANCE_BANDS = {
    "near": (
        "under",
        f"under {event_statistics.NEAR_CM:g} cm from the animal",
    ),
    "middle": (
        "between",
        f"{event_statistics.NEAR_CM:g} cm to under {REMOTE_CM:g} cm from the animal",
    ),
    "remote": ("beyond", f"{REMOTE_CM:g} cm or more from the animal"),
}


def content_distance_values(events: pd.DataFrame, source: str) -> list[Value]:
    """Question 2d.1 and the distance panel of Figure 3's supplement (S3c): for
    each class, the share of the events whose content lies under 20 cm, 20 to
    100 cm, and 100 cm or more from the animal (the mean over the event's bins
    of the absolute distance along the track of the decoded place from the
    animal's; ``event_statistics.content_distance_band_statistics``): the mean
    over animals of each animal's share with the lowest and highest animal,
    tagged to the panel, and the share pooled over events (no figure draws
    it)."""
    make = value_factory(content_distance_values, source, TAGS["content_distance"])
    table = event_statistics.content_distance_band_statistics(events)
    values = []
    for group, word, name in (
        ("no SWR", "non_swr", "non-SWR"),
        ("SWR", "swr", "SWR-overlapping"),
    ):
        for band, (band_word, where) in _DISTANCE_BANDS.items():
            row = table.loc[(group, band)]
            what = (
                f"share of the immobile {name} non-local events whose content "
                f"(the mean over the event's bins of the distance along the track "
                f"of the decoded place from the animal's) lies {where}"
            )
            values += _animal_range_values(
                make,
                f"content_distance_{band_word}_{word}_fraction",
                (row["share"], row["animal_min"], row["animal_max"]),
                (
                    f"{what}, pooled over events",
                    f"lowest animal's {what} (an animal's share is the mean over "
                    "its epochs of the epoch's share)",
                    f"highest animal's {what} (the mean over its epochs)",
                ),
                animal_mean=(
                    row["share_animal_mean"],
                    f"{what}, the mean over animals of each animal's share (the "
                    "mean over its epochs of the epoch's share)",
                ),
                pooled_tags=TAGS["content_distance_pooled"],
            )
    values += [
        make(
            "content_distance_near_cm",
            event_statistics.NEAR_CM,
            "quantity",
            "int",
            "the distance of the content from the animal under which it counts as near",
            "cm",
        ),
        make(
            "content_distance_remote_cm",
            REMOTE_CM,
            "quantity",
            "int",
            "the distance of the content from the animal at or beyond which it "
            "counts as remote (the spike-evidence analysis's remote place)",
            "cm",
        ),
    ]
    return values


def property_interval_values(intervals: pd.DataFrame, source: str) -> list[Value]:
    """Questions 2d.1–2d.6: the uncertainty of each property contrast, and
    the contrast at matched duration.

    A nested bootstrap (animals, then epochs within animal, then events within
    epoch; 1000 replicates, fixed seed) of the SWR minus non-SWR difference of
    class medians, and of the duration-matched difference: within ten quantile
    strata of the pooled event durations, averaged by stratum size
    (``analysis.event_properties.property_contrast``). 95 % intervals: the estimate
    plus or minus a small-sample t correction over the animals times the
    replicates' standard deviation. The class medians' own intervals stay in
    the bootstrap's ``intervals.csv``.
    """
    if set(intervals.index) != {p[0] for p in _PROPERTIES}:
        raise ValueError(
            f"the bootstrap holds {sorted(intervals.index)}, not the events summary's properties"
        )
    nan_columns = [c for c in intervals.columns if c.endswith("_nan_replicates")]
    if not nan_columns:
        raise ValueError("the bootstrap does not count its replicates without estimate")
    if "n_animals" not in intervals.columns:
        raise ValueError(
            "the bootstrap does not record its animals: its intervals are percentile "
            "intervals, not the small-sample t rule"
        )
    share = intervals[nan_columns].div(intervals["n_replicates"], axis=0)
    if (share > MAX_NAN_REPLICATE_SHARE).any(axis=None):
        raise ValueError(f"too many bootstrap replicates without estimate:\n{share}")
    values: list[Value] = []
    for word, _, _, unit, _, _, tags in _PROPERTIES:
        make = value_factory(property_interval_values, source, tags)
        row = intervals.loc[word]
        name = _PROPERTY_NAMES[word]
        interval = (
            "95 % interval (nested bootstrap, small-sample t correction over the "
            f"{format_number(row.n_animals, 'int')} animals)"
        )
        skipped = (
            f"; {format_number(row.matched_replicates_skipping_strata, 'int')} of "
            f"{format_number(row.n_replicates, 'int')} replicates left out a stratum "
            "that drew no event of a class, "
            f"{format_number(row.matched_difference_nan_replicates, 'int')} had no "
            "estimate"
        )
        values += [
            make(
                f"{word}_difference_low",
                row.difference_low,
                "quantity",
                "sig2s",
                f"lower bound of the {interval} of the SWR minus non-SWR difference "
                f"of median {name}",
                unit,
            ),
            make(
                f"{word}_difference_high",
                row.difference_high,
                "quantity",
                "sig2s",
                "upper bound of that interval",
                unit,
            ),
        ]
        if word == "duration":
            continue
        values += [
            make(
                f"{word}_matched_difference",
                row.matched_difference,
                "quantity",
                "sig2s",
                f"SWR minus non-SWR difference of median {name} at matched duration "
                "(within ten duration-quantile strata, averaged by stratum size)",
                unit,
            ),
            make(
                f"{word}_matched_difference_low",
                row.matched_difference_low,
                "quantity",
                "sig2s",
                f"lower bound of its {interval}{skipped}",
                unit,
            ),
            make(
                f"{word}_matched_difference_high",
                row.matched_difference_high,
                "quantity",
                "sig2s",
                f"upper bound of its {interval}",
                unit,
            ),
        ]
    return values


def rest_theta_values(events: pd.DataFrame, source: str) -> list[Value]:
    """Question 2d.8: the theta-like share of events against duration-matched quiet rest.

    For each event, ten windows of quiet rest (below 4 cm/s, in no non-local,
    SWR or MUA event, outside excluded time) with as many bins as the event,
    drawn uniformly in its epoch; a window is theta-like when its mean theta
    reaches the epoch's median running theta, as an event is. The reference
    share of a class is the mean over its events of their windows' theta-like
    share, so it carries the class's durations. Every event must have its
    windows: an event whose length no rest stretch of its epoch holds would
    have none, and raises rather than leaving the comparison silently.
    """
    make = value_factory(rest_theta_values, source, TAGS["brain_state"])
    statistics = event_statistics.rest_matched_theta_statistics(events)
    values: list[Value] = []
    for group, is_swr in (("non_swr", False), ("swr", True)):
        stats = statistics.loc[is_swr]
        name = "non-SWR" if group == "non_swr" else "SWR-overlapping"
        values += [
            make(
                f"theta_like_fraction_rest_matched_{group}",
                stats["rest_theta_like_fraction"],
                "fraction",
                "pct",
                f"share of quiet-rest windows, matched in duration to the {name} events, "
                "whose theta reaches the epoch's median running theta",
            ),
            make(
                f"theta_like_animals_above_rest_matched_{group}",
                stats["n_animals_above_rest"],
                "count",
                "count",
                f"animals whose share of theta-like {name} events is above that of "
                "their duration-matched quiet-rest windows, each the mean over the "
                "animal's epochs of the epoch's share",
            ),
        ]
    return values


# --- where content sits -------------------------------------------------------


def content_occupancy_values(bins: pd.DataFrame, source: str) -> list[Value]:
    """Question 2d.1: where content sits against the animal and the wells.

    ``bins``: each event time bin with its kind, its content's track distance
    to the animal and to the nearest well (a track-graph end point) and its arm
    with the arm's length, as :func:`content_along_arms_bins` reads them. The
    shares at a remote well, at the animal's own well, on a remote path and near
    the animal off the wells (``content_along_arms.content_place_shares``): each
    epoch's share of its time bins, averaged over the animal's epochs, then over
    animals. And the share at a well of content spread evenly over the track
    (``content_along_arms.even_content``), which those shares are read against.
    """
    make = value_factory(content_occupancy_values, source, TAGS["content_occupancy"])
    # The places that depend on where the animal is are in no panel: content
    # against the animal is the behavior figure's.
    make_by_animal = value_factory(
        content_occupancy_values, source, TAGS["content_by_animal"]
    )
    statistics = content_along_arms.place_share_statistics(
        content_along_arms.content_place_shares(bins)
    )
    values = [
        make(
            "occupancy_epochs",
            bins["epoch_id"].nunique(),
            "count",
            "count",
            "epochs whose event time bins place content on the track",
        ),
    ]
    for word, is_swr, name in (("non_swr", False, "non-SWR"), ("swr", True, "SWR")):
        for column in event_statistics.OCCUPANCY_PLACES:
            values.append(
                (make if column == "at_well" else make_by_animal)(
                    f"{word}_content_{column}_fraction",
                    statistics["means"].loc[is_swr, column],
                    "fraction",
                    "pct",
                    f"share of {name} non-local content time "
                    f"{column.replace('_', ' ')} (wells are the track graph's end "
                    "points; each epoch's share, averaged over the animal's epochs, "
                    "then over animals)",
                )
            )
    values.append(
        make(
            "animals_non_swr_more_at_well",
            statistics["n_animals_non_swr_more_at_well"],
            "count",
            "count",
            "animals whose non-SWR content is at a well more often than their SWR "
            "content, each share the mean over the animal's epochs",
        )
    )
    values.append(
        make(
            "even_content_at_well_fraction",
            content_along_arms.even_content(bins, event_statistics.AT_WELL_CM)[
                "at_well_fraction"
            ],
            "fraction",
            "pct",
            "share of the track within "
            f"{event_statistics.AT_WELL_CM:.0f} cm of a well (a track-graph end "
            "point): the share at a well of content spread evenly over the track",
        )
    )
    return values


def content_along_arms_bins(
    events: pd.DataFrame, folder: Path, n_epochs: int, columns: list[str]
) -> tuple[pd.DataFrame, list[Path]]:
    """Every time bin of the events of ``events`` in the content-along-arms run,
    with ``columns``, each with its event's ``animal`` and ``is_swr``; and the
    files read. The run holds the bins of every event of the table; the bins of
    the events in ``events`` are read (``cohort_selection.select_events_in``).

    Raises
    ------
    ValueError
        If ``folder`` does not hold ``n_epochs`` epochs, or its bins do not cover
        every event of ``events``.
    """
    paths = sorted(folder.glob("*.parquet"))
    if len(paths) != n_epochs:
        raise ValueError(f"{folder} holds {len(paths)} epochs, not {n_epochs}")
    bins = select_events_in(
        pd.concat(
            pd.read_parquet(path, columns=["event_id", "epoch_id", *columns])
            for path in paths
        ),
        events["event_id"],
    )
    n_bins = len(bins)
    bins = bins.merge(
        events[["event_id", "animal", "is_swr"]], on="event_id", validate="m:1"
    )
    if len(bins) != n_bins or bins["event_id"].nunique() != len(events):
        raise ValueError(f"the bins of {folder} do not cover the event table's events")
    return bins, paths


# --- the summary ---------------------------------------------------------------


def with_start_and_rest_theta(
    events: pd.DataFrame, folder: Path, n_epochs: int
) -> tuple[pd.DataFrame, list[Path]]:
    """``events`` with each one's start distance and rest-theta columns
    (``Processed-Data/event_start_rest_theta``), and the files read.

    Raises
    ------
    ValueError
        If ``folder`` does not hold ``n_epochs`` epochs, or an event has no row.
    """
    paths = sorted(folder.glob("*.parquet"))
    if len(paths) != n_epochs:
        raise ValueError(f"{folder} holds {len(paths)} epochs, not {n_epochs}")
    extras = pd.concat(map(pd.read_parquet, paths), ignore_index=True)
    joined = events.merge(
        extras.drop(columns=["epoch_id", "decode_sha256"]),
        on="event_id",
        how="left",
        validate="1:1",
    )
    if joined.n_bins.isna().any():
        raise ValueError("an event has no start-distance and rest-theta row")
    return joined, paths


def with_brain_state(
    events: pd.DataFrame, inputs: EventsInputs, n_epochs: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``events`` with their slow and fast gamma, and the brain-state reference
    with the gamma reference's columns joined; also the gamma reference itself.

    Raises
    ------
    ValueError
        If a reference does not hold ``n_epochs`` epochs, or an event has no
        gamma value.
    """
    reference = pd.read_parquet(inputs.brain_state_reference)
    gamma_reference = pd.read_parquet(inputs.event_gamma / "reference.parquet")
    if len(reference) != n_epochs or len(gamma_reference) != n_epochs:
        raise ValueError(
            f"the brain-state reference holds {len(reference)} epochs and the gamma "
            f"reference {len(gamma_reference)}, not {n_epochs}"
        )
    gamma_columns = [c for c in gamma_reference.columns if "gamma" in c]
    reference = reference.merge(
        gamma_reference[[*SESSION_COLUMNS, *gamma_columns]],
        on=list(SESSION_COLUMNS),
        how="inner",
        validate="1:1",
    )
    gamma_events = pd.concat(
        pd.read_parquet(path, columns=["event_id", "slow_gamma_z", "fast_gamma_z"])
        for path in sorted(inputs.event_gamma.glob("*_events.parquet"))
    )
    joined = events.merge(gamma_events, on="event_id", how="left", validate="1:1")
    if joined[["slow_gamma_z", "fast_gamma_z"]].isna().all(axis=1).any():
        raise ValueError("an event has no gamma value")
    return joined, reference, gamma_reference


def summarize(inputs: EventsInputs, produced_by: str) -> Summary:
    """The events summary from ``inputs``."""
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
    current_sha256 = event_table_record["sha256"]
    # every per-epoch input must hold the epoch register's analyzed epochs
    n_epochs = len(analyzed_epoch_ids())

    def stale(folder: Path) -> str | None:
        return event_table_staleness(
            read_output_manifest(folder), current_sha256, rel(folder)
        )

    def off_subset(folder: Path) -> str | None:
        return subset_staleness(read_output_manifest(folder), rel(folder))

    def off_visits(folder: Path, epoch_ids: Iterable[str]) -> str | None:
        """Why a time-masked output was not made from the current visits."""
        return session_cache_staleness(
            read_output_manifest(folder), epoch_ids, rel(folder)
        )

    def off_time_mask(folder: Path, epoch_ids: Iterable[str]) -> str | None:
        """Why a time-masked output is not over the subset and the current visits."""
        return time_mask_staleness(read_output_manifest(folder), epoch_ids, rel(folder))

    source = f"{rel(inputs.event_table)} (canonical event table)"
    summary.add(*visit_context_values(table, source))
    summary.add(*over(event_cohort_values(events, source), EVENTS_SUBSET))
    summary.add(*over(boundary_offset_values(events, source), EVENTS_SUBSET))
    summary.add(*over(boundary_timing_values(events, source), EVENTS_SUBSET))
    summary.add(*over(hpd_tight_values(events, source), EVENTS_SUBSET))
    summary.add(*over(content_distance_values(events, source), EVENTS_SUBSET))
    summary.add(*over(state_composition_values(events, source), EVENTS_SUBSET))
    summary.inputs.append(event_table_record)
    events_with_extras, extra_paths = with_start_and_rest_theta(
        events, inputs.event_start_rest_theta, n_epochs
    )
    extras_source = (
        f"{source} and {rel(inputs.event_start_rest_theta)} "
        "(scripts/event_properties/measure_event_start_and_rest_theta.py)"
    )
    extras_stale = stale(inputs.event_start_rest_theta)
    summary.add(
        *over(property_values(events_with_extras, extras_source), EVENTS_SUBSET)
    )
    summary.add(
        *over(
            pend(
                start_distance_values(events_with_extras, extras_source),
                extras_stale,
            ),
            EVENTS_SUBSET,
        )
    )
    summary.add(
        *over(
            pend(
                rest_theta_values(events_with_extras, extras_source),
                extras_stale,
                off_time_mask(
                    inputs.event_start_rest_theta, (path.stem for path in extra_paths)
                ),
            ),
            f"{EVENTS_SUBSET}, against quiet rest in {TIME_SUBSET}",
        )
    )
    summary.inputs += [
        files_record("*.parquet", extra_paths, ROOT_DIR),
        file_record(inputs.event_start_rest_theta / "manifest.json", ROOT_DIR),
    ]
    intervals = pd.read_csv(inputs.property_bootstrap / "intervals.csv", index_col=0)
    summary.add(
        *over(
            pend(
                property_interval_values(
                    intervals,
                    f"{rel(inputs.property_bootstrap)} "
                    "(scripts/event_properties/bootstrap_event_properties.py)",
                ),
                stale(inputs.property_bootstrap),
                off_subset(inputs.property_bootstrap),
            ),
            EVENTS_SUBSET,
        )
    )
    summary.inputs += [
        file_record(inputs.property_bootstrap / name, ROOT_DIR)
        for name in ("intervals.csv", "manifest.json")
    ]
    events_with_gamma, reference, gamma_reference = with_brain_state(
        events, inputs, n_epochs
    )
    signal_source = (
        f"{rel(inputs.event_table)} (event theta_z, ripple_power_z, mua_rate_z) "
        f"and {rel(inputs.brain_state_reference)} (scripts/event_properties/"
        "measure_brain_state_reference.py) and the gamma pass "
        f"{rel(inputs.event_gamma)} (scripts/event_properties/measure_event_gamma.py)"
    )
    against_rest = f"{EVENTS_SUBSET}, against quiet rest in {TIME_SUBSET}"
    summary.add(
        *over(
            pend(
                brain_state_values(events_with_gamma, reference, signal_source),
                off_time_mask(
                    inputs.brain_state_reference.parent, reference["epoch_id"]
                ),
            ),
            against_rest,
        )
    )
    summary.add(
        *over(
            pend(
                gamma_z_values(events_with_gamma, reference, signal_source),
                stale(inputs.event_gamma),
                off_subset(inputs.brain_state_reference.parent),
                off_subset(inputs.event_gamma),
                off_visits(inputs.brain_state_reference.parent, reference["epoch_id"]),
                off_visits(inputs.event_gamma, gamma_reference["epoch_id"]),
            ),
            against_rest,
        )
    )
    summary.inputs += [
        file_record(inputs.brain_state_reference, ROOT_DIR),
        file_record(inputs.event_gamma / "manifest.json", ROOT_DIR),
    ]
    summary.add(
        *over(
            pend(
                gamma_db_values(
                    gamma_reference,
                    f"{rel(inputs.event_gamma)}/reference.parquet "
                    "(scripts/event_properties/measure_event_gamma.py)",
                ),
                stale(inputs.event_gamma),
                off_time_mask(inputs.event_gamma, gamma_reference["epoch_id"]),
            ),
            against_rest,
        )
    )
    summary.inputs.append(
        file_record(inputs.event_gamma / "reference.parquet", ROOT_DIR)
    )
    summary.inputs.append(
        files_record(
            "*_events.parquet", inputs.event_gamma.glob("*_events.parquet"), ROOT_DIR
        )
    )

    comparison = nonlocal_comparison.read_comparison(run, visits)
    others = comparison.others
    inventory_dir, directional_path = (
        comparison.inventory_dir,
        comparison.directional_path,
    )
    detector_source = (
        f"event features run {run.name}: pair_membership against state_space and "
        "state_shares (scripts/detector_comparison/add_state_shares.py), with each "
        "epoch's well visits (the session cache's port experiences)"
    )
    summary.add(*over(other_detector_values(others, detector_source), DETECTOR_SUBSET))
    summary.add(
        *over(detections_with_nonlocal_values(others, detector_source), DETECTOR_SUBSET)
    )
    summary.add(
        *over(
            nonlocal_found_values(
                nonlocal_comparison.nonlocal_event_memberships(
                    run, comparison.directional, events["event_id"]
                ),
                detector_source,
            ),
            DETECTOR_SUBSET,
        )
    )
    summary.add(*detector_visit_values(comparison.visit_counts, detector_source))
    summary.inputs.append(
        files_record(
            "*.pkl",
            [
                behavioral_context_cache_path({"session_id": identifier})
                for identifier in run_epochs
            ],
            ROOT_DIR,
            "each epoch's well visits",
        )
    )
    folder = inputs.exposure_in_visits
    exposure = pd.read_parquet(folder / "exposure.parquet")
    summary.add(
        *over(
            pend(
                left_out_values(
                    run,
                    inventory_dir,
                    events["event_id"],
                    visits,
                    exposure,
                    f"event features run {run.name} (every inventory's compared "
                    f"events), inventory comparison run {inventory_dir.name} (its "
                    f"observations and cells) and {rel(folder)} (scripts/"
                    "detector_comparison/measure_exposure_in_well_visits.py: the "
                    "non-local detector's exposure inside the well visits)",
                ),
                off_time_mask(folder, exposure["epoch_id"]),
            ),
            LEFT_OUT_SUBSET,
        )
    )
    summary.inputs += [
        file_record(folder / name, ROOT_DIR)
        for name in ("exposure.parquet", "manifest.json")
    ]

    # the run's tables are read, and so recorded, for the epochs its manifest marks ok
    def run_tables(name: str) -> dict[str, object]:
        return files_record(f"*_{name}.parquet", ok_table_paths(run, name), ROOT_DIR)

    summary.inputs += [
        run_tables("features"),
        file_record(inventory_dir / "manifest.json", ROOT_DIR),
        *(
            files_record(
                f"*_{name}.parquet", ok_table_paths(inventory_dir, name), ROOT_DIR
            )
            for name in nonlocal_comparison.INVENTORY_TABLES
        ),
    ]
    summary.inputs += [
        run_tables("state_shares"),
        run_tables("pair_membership"),
        file_record(directional_path, ROOT_DIR),
    ]

    # the run scores every event: each scored event takes its event_id, and the
    # events the selection keeps and the quiet windows in a visit are read
    metrics = with_event_ids(
        spike_evidence_metrics(inputs.spike_evidence_dir, n_epochs), table
    )
    in_subset = in_analyzed_subset(metrics, events["event_id"], visits)
    evidence_source = (
        f"{rel(inputs.spike_evidence_dir)}/*/{SPIKE_EVIDENCE_METRICS} (every event, "
        "no cap), each scored event matched to the event table by epoch and start, "
        "with each epoch's well visits"
    )
    summary.add(
        *over(
            spike_evidence_values(metrics[in_subset], evidence_source),
            SPIKE_EVIDENCE_SUBSET,
        )
    )
    summary.add(*spike_evidence_visit_values(metrics, in_subset, evidence_source))
    spike_evidence_record = files_record(
        SPIKE_EVIDENCE_METRICS,
        inputs.spike_evidence_dir.glob(f"*/{SPIKE_EVIDENCE_METRICS}"),
        ROOT_DIR,
    )
    summary.inputs.append(spike_evidence_record)

    stats = pd.read_csv(inputs.heldout_by_swr)
    heldout_provenance_path = inputs.heldout_by_swr.with_name(
        f"{NL_BOUTS_BY_SWR_STEM}.provenance.json"
    )
    heldout_provenance = json.loads(heldout_provenance_path.read_text())
    heldout_off_subset = time_mask_staleness(
        heldout_provenance, heldout_provenance["epochs"], rel(heldout_provenance_path)
    )
    summary.add(
        *over(
            pend(
                heldout_by_swr_values(stats, rel(inputs.heldout_by_swr)),
                heldout_off_subset,
            ),
            BOUT_SUBSET,
        )
    )
    if heldout_off_subset is None:
        summary.add(
            *heldout_bout_values(
                pd.DataFrame(heldout_provenance["bouts"]), rel(heldout_provenance_path)
            )
        )
    summary.inputs += [
        file_record(inputs.heldout_by_swr, ROOT_DIR),
        file_record(heldout_provenance_path, ROOT_DIR),
    ]

    along_bins, along_paths = content_along_arms_bins(
        events,
        inputs.content_along_arms,
        n_epochs,
        [
            "distance_to_animal_cm",
            "distance_to_well_cm",
            "arm",
            "arm_length_cm",
            "n_place_bins",
        ],
    )
    summary.add(
        *over(
            pend(
                content_occupancy_values(
                    along_bins,
                    f"{rel(inputs.content_along_arms)} "
                    "(scripts/event_properties/measure_content_along_arms.py)",
                ),
                stale(inputs.content_along_arms),
            ),
            EVENTS_SUBSET,
        )
    )
    summary.inputs += [
        files_record("*.parquet", along_paths, ROOT_DIR),
        file_record(inputs.content_along_arms / "manifest.json", ROOT_DIR),
    ]

    # every analyzed epoch with an event: an epoch with none has no table
    analyzed = swr_source.event_table_epochs(table, analyzed_epoch_ids())
    selected = events["event_id"]
    participation_table = select_events_in(
        read_complete_run(inputs.event_participation, PARTICIPATION_NAME, analyzed),
        selected,
    )
    advantage_table = select_events_in(
        read_complete_run(
            inputs.likelihood_advantage, "likelihood_advantage", analyzed
        ),
        selected,
    )
    require_tetrode_holdout_folds(
        read_output_manifest(inputs.tetrode_holdout), rel(inputs.tetrode_holdout)
    )
    holdout_table = select_events_in(
        read_complete_run(inputs.tetrode_holdout, TETRODE_HOLDOUT_NAME, analyzed),
        selected,
    )
    # the held-out tetrode run with its extra scores (Figure 2j, the parts of
    # the evidence and the low-rate events): held out as the canonical run is,
    # and with the extra columns
    require_tetrode_holdout_folds(
        read_output_manifest(inputs.tetrode_holdout_extra),
        rel(inputs.tetrode_holdout_extra),
    )
    extra_table = select_events_in(
        read_complete_run(inputs.tetrode_holdout_extra, TETRODE_HOLDOUT_NAME, analyzed),
        selected,
    )
    tetrode_holdout.require_extra_columns(
        extra_table, rel(inputs.tetrode_holdout_extra)
    )
    # the place-rate run: every event of the table, read over the selected ones
    place_rate_table = select_events_in(
        read_complete_run(inputs.content_place_rate, CONTENT_PLACE_RATE_NAME, analyzed),
        selected,
    )
    participation_paths = ok_table_paths(inputs.event_participation, PARTICIPATION_NAME)
    participation_sha256 = files_sha256(participation_paths)

    def participation_off_subset() -> str | None:
        """Why the participation run's controls are not Local rest inside the
        current well visits, or None: its settings record no subset or another,
        or it read other visits than the current session cache's."""
        manifest = read_output_manifest(inputs.event_participation)
        what = rel(inputs.event_participation)
        notes = [
            subset_staleness(manifest.get("settings", {}), what),
            session_cache_staleness(manifest, analyzed, what),
        ]
        return "; ".join(note for note in notes if note) or None

    def participation_staleness(folder: Path) -> str | None:
        """Why a run made from the participation tables is stale, or None."""
        return _input_staleness(
            read_output_manifest(folder),
            "event_participation",
            participation_sha256,
            rel(folder),
            "participation",
            "participation run",
        )

    summary.add(
        *over(
            pend(
                participation_values(
                    participation_table,
                    f"{rel(inputs.event_participation)} "
                    "(scripts/event_evidence/measure_event_participation.py)",
                ),
                stale(inputs.event_participation),
                participation_off_subset(),
            ),
            CONTROL_SUBSET,
        )
    )
    summary.add(
        *over(
            pend(
                likelihood_advantage_values(
                    advantage_table,
                    f"{rel(inputs.likelihood_advantage)} "
                    "(scripts/event_evidence/score_event_likelihood_advantage.py)",
                ),
                stale(inputs.likelihood_advantage),
                participation_staleness(inputs.likelihood_advantage),
            ),
            EVENTS_SUBSET,
        )
    )
    holdout_windows = tetrode_holdout.window_scores(holdout_table)
    summary.add(
        *over(
            pend(
                tetrode_holdout_values(
                    holdout_windows,
                    f"{rel(inputs.tetrode_holdout)} "
                    "(scripts/event_evidence/score_tetrode_holdout.py)",
                ),
                stale(inputs.tetrode_holdout),
                participation_staleness(inputs.tetrode_holdout),
                participation_off_subset(),
            ),
            CONTROL_SUBSET,
        )
    )
    extra_windows = tetrode_holdout.window_scores(extra_table)
    extra_source = (
        f"{rel(inputs.tetrode_holdout_extra)} "
        "(scripts/event_evidence/score_tetrode_holdout.py --extra-scores)"
    )
    extra_stale = (
        stale(inputs.tetrode_holdout_extra),
        participation_staleness(inputs.tetrode_holdout_extra),
    )
    # the shared content: the events alone, by the width of the others' estimate
    summary.add(
        *over(
            pend(shared_content_values(extra_windows, extra_source), *extra_stale),
            EVENTS_SUBSET,
        )
    )
    # the parts of the evidence: the events and their controls
    summary.add(
        *over(
            pend(
                tetrode_holdout_extra_values(extra_windows, extra_source),
                *extra_stale,
                participation_off_subset(),
            ),
            CONTROL_SUBSET,
        )
    )
    # the share with no SWR among the events with a resolved place
    summary.add(
        *over(
            pend(
                resolved_content_values(events, extra_windows, extra_source),
                *extra_stale,
            ),
            EVENTS_SUBSET,
        )
    )
    # the events at low-rate places: the place-rate run and the extra run
    summary.add(
        *over(
            pend(
                low_rate_values(
                    place_rate_table,
                    events,
                    extra_windows,
                    f"{rel(inputs.content_place_rate)} "
                    "(scripts/event_properties/measure_content_place_rate.py) and "
                    f"{extra_source}",
                ),
                stale(inputs.content_place_rate),
                *extra_stale,
            ),
            EVENTS_SUBSET,
        )
    )
    summary.inputs += [
        files_record(
            f"*_{TETRODE_HOLDOUT_NAME}.parquet",
            ok_table_paths(inputs.tetrode_holdout, TETRODE_HOLDOUT_NAME),
            ROOT_DIR,
        ),
        file_record(inputs.tetrode_holdout / "manifest.json", ROOT_DIR),
        files_record(
            f"*_{TETRODE_HOLDOUT_NAME}.parquet",
            ok_table_paths(inputs.tetrode_holdout_extra, TETRODE_HOLDOUT_NAME),
            ROOT_DIR,
        ),
        file_record(inputs.tetrode_holdout_extra / "manifest.json", ROOT_DIR),
        files_record(
            f"*_{CONTENT_PLACE_RATE_NAME}.parquet",
            ok_table_paths(inputs.content_place_rate, CONTENT_PLACE_RATE_NAME),
            ROOT_DIR,
        ),
        file_record(inputs.content_place_rate / "manifest.json", ROOT_DIR),
        files_record(f"*_{PARTICIPATION_NAME}.parquet", participation_paths, ROOT_DIR),
        file_record(inputs.event_participation / "manifest.json", ROOT_DIR),
        files_record(
            "*_likelihood_advantage.parquet",
            ok_table_paths(inputs.likelihood_advantage, "likelihood_advantage"),
            ROOT_DIR,
        ),
        file_record(inputs.likelihood_advantage / "manifest.json", ROOT_DIR),
    ]

    # the restriction scores over the canonical events' windows and their
    # controls: the window run sums the restriction run's bins and takes its
    # spike counts from the held-out tetrode run, so it must be made from both
    # runs as they are now, and on the held-out tetrode run's event table (the
    # window run's note when it is not; the event-table check above covers the
    # window run's own table)
    restriction_dir = inputs.nonlocal_restriction
    restriction_windows_dir = inputs.nonlocal_restriction_windows
    heldout_dir = inputs.nonlocal_restriction_heldout_events
    # the held-out event run (Figure 2k): its events that start inside a well
    # visit, its time sums and time steps, and why a value read from them is
    # pending; a restriction run of the earlier design is refused (the window
    # and held-out event runs must have read this run, staleness below)
    heldout = read_heldout_restriction(
        heldout_dir, restriction_dir, inputs.bin_facts, analyzed, current_sha256
    )
    # every restriction value reads the restriction run: pending when it was not
    # made from the current decode frames (as the event features run records
    # them) and SWR source
    restriction_inputs = restriction_inputs_note(
        restriction_dir, analyzed, rel(restriction_dir), features_manifest
    )
    # the run holds the windows of every event: those of the selected events
    # (and their controls, under their event's id) are read
    restriction_windows_table = select_events_in(
        read_complete_run(
            restriction_windows_dir, NONLOCAL_RESTRICTION_WINDOWS_NAME, analyzed
        ),
        selected,
    )
    windows_manifest = read_output_manifest(restriction_windows_dir)
    restriction_stale = (
        stale(restriction_windows_dir),
        participation_staleness(restriction_windows_dir),
        run_staleness(
            windows_manifest,
            restriction_windows_dir,
            "tetrode_holdout",
            inputs.tetrode_holdout,
        ),
        run_staleness(
            windows_manifest,
            restriction_windows_dir,
            "nonlocal_restriction",
            restriction_dir,
        ),
        run_staleness(
            windows_manifest, restriction_windows_dir, "bin_facts", inputs.bin_facts
        ),
        window_run_note_staleness(windows_manifest, rel(restriction_windows_dir)),
        restriction_inputs,
    )
    restriction_source = (
        f"{rel(restriction_windows_dir)} "
        "(scripts/event_evidence/sum_nonlocal_restriction_windows.py, from "
        f"{rel(restriction_dir)}, scripts/event_evidence/score_nonlocal_restriction.py, "
        f"and {rel(inputs.bin_facts)}, scripts/event_evidence/measure_bin_facts.py)"
    )
    # the windows' controls come from the participation tables the held-out
    # tetrode run read (sum_nonlocal_restriction_windows.check_holdout_inputs):
    # Local rest inside one well visit only when that run records the subset
    window_values = restriction_window_values(
        restriction_windows_table, restriction_source
    )
    window_stale = (*restriction_stale, participation_off_subset())
    summary.add(
        *over(pend(window_values.windows, *window_stale), CANONICAL_WINDOWS_SUBSET)
    )
    # of the window values, only the counts after the visit read the visits
    # (through the per-bin facts)
    summary.add(
        *over(
            pend(window_values.after_visit, *window_stale, heldout.session_cache),
            CANONICAL_WINDOWS_SUBSET,
        )
    )

    # the canonical events found again, over the selected events
    found_table = select_events_in(
        read_complete_run(heldout_dir, CANONICAL_EVENTS_FOUND_NAME, analyzed),
        selected,
    )
    heldout_selected = heldout.events
    restriction_time_sums = heldout.time_sums
    restriction_bin_s = heldout.bin_s
    heldout_runs_stale = (
        heldout.restriction_run,
        heldout.bin_facts_run,
        restriction_inputs,
    )
    heldout_stale = (heldout.event_table, *heldout_runs_stale, heldout.session_cache)
    heldout_source = (
        f"{rel(heldout_dir)} (scripts/event_evidence/detect_heldout_events.py, from "
        f"{rel(restriction_dir)}, scripts/event_evidence/score_nonlocal_restriction.py, "
        f"and {rel(inputs.bin_facts)}, scripts/event_evidence/measure_bin_facts.py)"
    )
    summary.add(
        *over(
            pend(
                restriction_heldout_event_values(heldout_selected, heldout_source),
                *heldout_stale,
            ),
            HELDOUT_EVENTS_SUBSET,
        )
    )
    summary.add(
        *over(
            pend(
                restriction_agreement_values(
                    heldout_selected,
                    found_table,
                    f"{heldout_source} (its held-out events and the canonical "
                    "events found again)",
                ),
                *heldout_stale,
            ),
            f"{HELDOUT_EVENTS_SUBSET}, and {EVENTS_SUBSET}; whether an event "
            "shares a bin with an event of the other set is read against all of "
            "that set's events, so its partner may start outside a visit",
        )
    )
    outside_swr_values = restriction_outside_swr_values(
        restriction_time_sums,
        restriction_bin_s,
        heldout_selected,
        f"{heldout_source} (its held-out events and time sums)",
    )
    summary.add(
        *over(
            pend(outside_swr_values.with_events, *heldout_stale),
            f"{TIME_SUBSET}, and {HELDOUT_EVENTS_SUBSET}",
        )
    )
    # the time no event holds is outside every held-out event, selected or not:
    # those values are over the time alone
    summary.add(
        *over(pend(outside_swr_values.outside_events, *heldout_stale), TIME_SUBSET)
    )
    # over parts of the time outside SWRs, from the time sums alone: no event
    # enters
    summary.add(
        *over(
            pend(
                restriction_outside_swr_part_values(
                    restriction_time_sums,
                    restriction_bin_s,
                    f"{heldout_source} (its time sums)",
                ),
                *heldout_runs_stale,
                heldout.session_cache,
            ),
            TIME_SUBSET,
        )
    )
    # inside SWRs, from the time sums alone: no event enters
    summary.add(
        *over(
            pend(
                restriction_inside_swr_values(
                    restriction_time_sums,
                    restriction_bin_s,
                    f"{heldout_source} (its time sums)",
                ),
                *heldout_runs_stale,
                heldout.session_cache,
            ),
            TIME_SUBSET,
        )
    )
    # over immobile time only: no event enters
    summary.add(
        *over(
            pend(
                restriction_immobile_values(
                    restriction_time_sums,
                    restriction_bin_s,
                    f"{heldout_source} (its time sums)",
                ),
                *heldout_runs_stale,
                heldout.session_cache,
            ),
            TIME_SUBSET,
        )
    )
    # the fourth decoder (SWR-allowed): from the run that scored all four
    # decoders in one pass under the stored fold fits and the held-out event
    # run made from it, read and checked as the canonical runs are
    swr_allowed_dir = inputs.nonlocal_restriction_swr_allowed
    swr_allowed_heldout_dir = inputs.nonlocal_restriction_swr_allowed_heldout_events
    swr_allowed = read_heldout_restriction(
        swr_allowed_heldout_dir,
        swr_allowed_dir,
        inputs.bin_facts,
        analyzed,
        current_sha256,
    )
    summary.add(
        *over(
            pend(
                restriction_swr_allowed_values(
                    swr_allowed.events,
                    swr_allowed.time_sums,
                    swr_allowed.bin_s,
                    f"{rel(swr_allowed_heldout_dir)} "
                    "(scripts/event_evidence/detect_heldout_events.py, from "
                    f"{rel(swr_allowed_dir)}, scripts/event_evidence/"
                    "score_nonlocal_restriction.py --from-stored-fit, and "
                    f"{rel(inputs.bin_facts)}, scripts/event_evidence/"
                    "measure_bin_facts.py; its held-out events and time sums)",
                ),
                swr_allowed.event_table,
                swr_allowed.restriction_run,
                swr_allowed.bin_facts_run,
                restriction_inputs_note(
                    swr_allowed_dir, analyzed, rel(swr_allowed_dir), features_manifest
                ),
                swr_allowed.session_cache,
            ),
            f"{TIME_SUBSET}, and {HELDOUT_EVENTS_SUBSET}",
        )
    )
    summary.inputs += [
        file_record(restriction_dir / MANIFEST, ROOT_DIR),
        file_record(swr_allowed_dir / MANIFEST, ROOT_DIR),
        *(
            files_record(
                f"*_{name}.parquet",
                ok_table_paths(swr_allowed_heldout_dir, name),
                ROOT_DIR,
            )
            for name in (HELDOUT_EVENTS_NAME, HELDOUT_TIME_SUMS_NAME)
        ),
        file_record(swr_allowed_heldout_dir / MANIFEST, ROOT_DIR),
        files_record(
            f"*_{NONLOCAL_RESTRICTION_WINDOWS_NAME}.parquet",
            ok_table_paths(restriction_windows_dir, NONLOCAL_RESTRICTION_WINDOWS_NAME),
            ROOT_DIR,
        ),
        file_record(restriction_windows_dir / MANIFEST, ROOT_DIR),
        *(
            files_record(
                f"*_{name}.parquet", ok_table_paths(heldout_dir, name), ROOT_DIR
            )
            for name in (
                HELDOUT_EVENTS_NAME,
                CANONICAL_EVENTS_FOUND_NAME,
                HELDOUT_TIME_SUMS_NAME,
            )
        ),
        file_record(heldout_dir / MANIFEST, ROOT_DIR),
        file_record(inputs.bin_facts / MANIFEST, ROOT_DIR),
    ]

    folder = inputs.decoder_quality
    quality_sessions = pd.read_parquet(folder / "sessions.parquet")
    calibration = pd.read_parquet(folder / "calibration.parquet")
    for name, quality_table in (
        ("sessions.parquet", quality_sessions),
        ("calibration.parquet", calibration),
    ):
        require_analyzed_epochs(quality_table["epoch_id"], str(folder / name))
    quality_source = (
        f"{rel(folder)} (scripts/decoding/measure_decoder_quality.py: each analyzed "
        "epoch's decode)"
    )
    summary.add(*decoding_error_values(quality_sessions, quality_source))
    summary.add(*calibration_values(calibration, quality_source))
    summary.add(*state_fraction_values(quality_sessions, quality_source))
    # the immobile fractions are over the time inside the well visits
    summary.add(
        *over(
            pend(
                immobile_state_fraction_values(quality_sessions, quality_source),
                off_time_mask(folder, quality_sessions["epoch_id"]),
            ),
            f"the immobile {TIME_SUBSET}",
        )
    )
    summary.inputs += [
        file_record(folder / name, ROOT_DIR)
        for name in ("sessions.parquet", "calibration.parquet", "manifest.json")
    ]

    folder = inputs.class_structure
    names = ("distribution.csv", "step_fits.csv", "manifest.json")
    source = (
        f"{rel(folder)} (scripts/event_properties/measure_ripple_class_structure.py: the "
        f"event table, event features run {STATE_SPACE_FEATURES_RUN}, the decode frames "
        "and the spike-evidence run)"
    )
    class_stale = (
        "; ".join(note for note in (stale(folder), off_subset(folder)) if note) or None
    )
    summary.add(
        *over(
            pend(
                class_distribution_values(
                    pd.read_csv(folder / "distribution.csv"), source
                ),
                class_stale,
            ),
            EVENTS_SUBSET,
        )
    )
    summary.add(
        *over(
            class_step_values_with_staleness(
                read_output_manifest(folder),
                spike_evidence_record["sha256"],
                class_stale,
                pd.read_csv(folder / "step_fits.csv"),
                source,
                rel(folder),
            ),
            EVENTS_SUBSET,
        )
    )
    summary.inputs += [file_record(folder / name, ROOT_DIR) for name in names]

    folder = inputs.nonlocal_by_signal
    signal_counts = pd.read_parquet(folder / "counts.parquet")
    require_analyzed_epochs(signal_counts["epoch_id"], str(folder / "counts.parquet"))
    counts_source = (
        f"{rel(folder)} (scripts/event_properties/measure_nonlocal_by_signal.py: "
        "each analyzed epoch's decode)"
    )
    for builder_values in (nonlocal_power_values, nonlocal_mua_values):
        summary.add(
            *over(
                pend(
                    builder_values(signal_counts, counts_source),
                    off_time_mask(folder, signal_counts["epoch_id"]),
                ),
                f"the immobile {TIME_SUBSET}",
            )
        )
    summary.inputs += [
        file_record(folder / name, ROOT_DIR)
        for name in ("counts.parquet", "manifest.json")
    ]

    folder = inputs.state_by_signal
    state_counts = pd.read_parquet(folder / "counts.parquet")
    require_analyzed_epochs(state_counts["epoch_id"], str(folder / "counts.parquet"))
    summary.add(
        *over(
            pend(
                state_by_signal_values(
                    state_counts,
                    f"{rel(folder)} (scripts/event_properties/measure_state_by_signal.py: "
                    "each analyzed epoch's decode)",
                ),
                off_time_mask(folder, state_counts["epoch_id"].unique()),
            ),
            f"the immobile {TIME_SUBSET}",
        )
    )
    summary.inputs += [
        file_record(folder / name, ROOT_DIR)
        for name in ("counts.parquet", "manifest.json")
    ]

    folder = inputs.nonlocal_probability
    probability_counts = pd.read_parquet(folder / "counts.parquet")
    require_analyzed_epochs(
        probability_counts["epoch_id"], str(folder / "counts.parquet")
    )
    summary.add(
        *over(
            pend(
                nonlocal_probability_values(
                    probability_counts,
                    f"{rel(folder)} (scripts/decoding/measure_nonlocal_probability.py: "
                    "each analyzed epoch's decode)",
                ),
                off_time_mask(folder, probability_counts["epoch_id"].unique()),
            ),
            f"the immobile {TIME_SUBSET}",
        )
    )
    summary.inputs += [
        file_record(folder / name, ROOT_DIR)
        for name in ("counts.parquet", "manifest.json")
    ]

    folder = inputs.event_rule_sweep
    sweep_counts = pd.read_parquet(folder / "counts.parquet")
    require_analyzed_epochs(sweep_counts["epoch_id"], str(folder / "counts.parquet"))
    summary.add(
        *over(
            pend(
                rule_sweep_values(
                    sweep_counts,
                    f"{rel(folder)} (scripts/event_properties/sweep_event_rule.py: "
                    "each analyzed epoch's decode, detected again at each setting)",
                ),
                stale(folder),
                off_time_mask(folder, sweep_counts["epoch_id"].unique()),
            ),
            EVENTS_SUBSET,
        )
    )
    summary.inputs += [
        file_record(folder / name, ROOT_DIR)
        for name in ("counts.parquet", "manifest.json")
    ]

    folder = inputs.state_calibration_in_visits
    calibration_epochs = list(read_output_manifest(folder)["epochs"])
    require_analyzed_epochs(calibration_epochs, str(folder))
    names = ("calibration_bins.csv", "calibration_trends.csv", "manifest.json")
    summary.add(
        *over(
            pend(
                state_calibration_values(
                    pd.read_csv(folder / names[0]),
                    pd.read_csv(folder / names[1]),
                    f"{rel(folder)} (scripts/event_evidence/"
                    "summarize_state_calibration_in_visits.py, from the held-out "
                    f"tetrode run {STATE_EVIDENCE_RUN}/{STATE_EVIDENCE_DIR})",
                ),
                off_time_mask(folder, calibration_epochs),
            ),
            f"the immobile {TIME_SUBSET}",
        )
    )
    summary.inputs += [file_record(folder / name, ROOT_DIR) for name in names]
    return summary
