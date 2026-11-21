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
- each epoch's immobile time bins, and its non-local ones, by ripple power
  (``Processed-Data/nonlocal_by_ripple_power``, from
  ``scripts/event_properties/measure_nonlocal_by_ripple_power.py``);
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
  ``scripts/event_evidence/detect_heldout_events.py``, which Figure 2g draws);
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
facts run was not made from the current session cache (``prepare_data.session_cache_staleness``). The held-out tetrode values and
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

from continuum_swr_replay.analysis.decode_quality import quality_metrics
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
    tetrode_holdout,
)
from continuum_swr_replay.analysis.event_evidence.nonlocal_restriction import (
    FAMILY_SUFFIX,
    FIT,
)
from continuum_swr_replay.analysis.event_evidence.restriction_totals import (
    RESTRICTION_FIGURE,
    RESTRICTION_FIGURE_DECODERS,
    RESTRICTION_FIGURE_DIFFERENCE,
    RESTRICTION_FIGURE_FAMILY,
    RESTRICTION_FIGURE_GROUPS,
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
)
from continuum_swr_replay.analysis.event_properties.class_or_continuum import (
    KAY_ZSCORE_THRESHOLD,
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
from continuum_swr_replay.data_loaders.epoch_register import (
    analyzed_animals,
    analyzed_epoch_ids,
)
from continuum_swr_replay.parameters import DEFAULT_NON_LOCAL_THRESHOLD
from continuum_swr_replay.paths import (
    BRAIN_STATE_REFERENCE_DIR,
    CANONICAL_EVENTS_FOUND_NAME,
    CONTENT_ALONG_ARMS_DIR,
    DECODER_QUALITY_DIR,
    EVENT_BIN_FACTS_DIR,
    EVENT_GAMMA_DIR,
    EVENT_LIKELIHOOD_ADVANTAGE_DIR,
    EVENT_NONLOCAL_RESTRICTION_DIR,
    EVENT_NONLOCAL_RESTRICTION_HELDOUT_EVENTS_DIR,
    EVENT_NONLOCAL_RESTRICTION_WINDOWS_DIR,
    EVENT_PARTICIPATION_DIR,
    EVENT_PROPERTY_BOOTSTRAP_DIR,
    EVENT_START_REST_THETA_DIR,
    EVENT_TETRODE_HOLDOUT_DIR,
    EXPOSURE_IN_VISITS_DIR,
    HELDOUT_EVENTS_NAME,
    HELDOUT_TIME_SUMS_NAME,
    NL_BOUTS_BY_SWR_STEM,
    NONLOCAL_BY_RIPPLE_POWER_DIR,
    NONLOCAL_RESTRICTION_WINDOWS_NAME,
    PARTICIPATION_NAME,
    RIPPLE_CLASS_STRUCTURE_DIR,
    ROOT_DIR,
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
    "calibration": Tags(("2a.1",), ("decoder_validation",)),
    "state_fractions": Tags(("2a.6",), ("decoder_validation",)),
    "cohort": Tags(("2b.2",), ("2d",)),
    "visit_context": Tags(("2b.2",)),
    "left_out": Tags(("2b.2",)),
    "other_detectors_main": Tags(("2b.3",), ("2",)),
    "other_detectors": Tags(("2b.3",)),
    "swr_content": Tags(("2b.3",), ("2c",)),
    "boundaries": Tags(("2b.4",), ("2f",)),
    "ripple": Tags(("2d.8", "2b.5"), ("2i",)),
    "mua": Tags(("2d.8",), ("2i",)),
    "theta": Tags(("2d.8",), ("2i",)),
    "gamma_z": Tags(("2d.8",), ("event_gamma",)),
    "brain_state": Tags(("2d.8",)),
    "class_distribution": Tags(("2b.5",)),
    "class_step": Tags(("2b.5", "2d.2", "2d.3", "2d.5", "2d.6")),
    "class_step_evidence": Tags(("2b.5",)),
    "nonlocal_power": Tags(("2b.5",), ("2j",)),
    "participation": Tags(("2d.9",), ("2h",)),
    "spike_rate": Tags(("2d.9",)),
    "state_composition": Tags(("2d.10",), ("3b",)),
    "likelihood_advantage": Tags(("2c.3",)),
    "tetrode_holdout": Tags(("2c.3",), ("heldout_tetrode",)),
    "nonlocal_restriction": Tags(("2c.3",)),
    "nonlocal_restriction_figure": Tags(("2c.3",), ("2g",)),
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
    nonlocal_by_ripple_power: Path
    event_start_rest_theta: Path
    property_bootstrap: Path
    decoder_quality: Path
    event_participation: Path
    likelihood_advantage: Path
    tetrode_holdout: Path
    nonlocal_restriction: Path
    nonlocal_restriction_windows: Path
    nonlocal_restriction_heldout_events: Path
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
        nonlocal_by_ripple_power=processed_data_dir() / NONLOCAL_BY_RIPPLE_POWER_DIR,
        event_start_rest_theta=processed_data_dir() / EVENT_START_REST_THETA_DIR,
        property_bootstrap=processed_data_dir() / EVENT_PROPERTY_BOOTSTRAP_DIR,
        decoder_quality=processed_data_dir() / DECODER_QUALITY_DIR,
        event_participation=processed_data_dir() / EVENT_PARTICIPATION_DIR,
        likelihood_advantage=processed_data_dir() / EVENT_LIKELIHOOD_ADVANTAGE_DIR,
        tetrode_holdout=processed_data_dir() / EVENT_TETRODE_HOLDOUT_DIR,
        nonlocal_restriction=processed_data_dir() / EVENT_NONLOCAL_RESTRICTION_DIR,
        nonlocal_restriction_windows=processed_data_dir()
        / EVENT_NONLOCAL_RESTRICTION_WINDOWS_DIR,
        nonlocal_restriction_heldout_events=processed_data_dir()
        / EVENT_NONLOCAL_RESTRICTION_HELDOUT_EVENTS_DIR,
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


def require_fold_fit(manifest: dict, what: str) -> None:
    """Refuse a restriction run that is not the fold fit on folds of whole trials.

    The values describe scores under a decoder fit on the other folds, five
    folds of whole trials balanced by destination well; the fold-fit run's
    settings record ``"fit": "folds"`` and ``"fold_design": "trials"``. A run of
    the earlier design, which held out only the encodings and ran the models
    with the transitions fit on the whole epoch, records no ``fit``; a fold fit
    on contiguous blocks of time records ``"fold_design": "contiguous"``.
    Neither can give these values.

    Raises
    ------
    ValueError
        If ``manifest``'s settings do not record ``"fit": "folds"`` and
        ``"fold_design": "trials"``.
    """
    settings = manifest.get("settings", {})
    fit, design = settings.get("fit"), settings.get("fold_design")
    rerun = (
        "scripts/event_evidence/score_nonlocal_restriction.py) in this folder, "
        "then rerun scripts/event_evidence/measure_bin_facts.py, "
        "scripts/event_evidence/detect_heldout_events.py and "
        "scripts/event_evidence/sum_nonlocal_restriction_windows.py on it."
    )
    if fit != FIT:
        raise ValueError(
            f"{what} is not the fold fit (its settings record fit {fit!r}, not "
            f"{FIT!r}): it is the earlier design, which held out only the "
            f"encodings. Put the fold-fit run ({rerun}"
        )
    if design != TRIAL_DESIGN:
        raise ValueError(
            f"{what} is not the fold fit on folds of whole trials (its settings "
            f"record fold_design {design!r}, not {TRIAL_DESIGN!r}). Put the fold-fit "
            f"run with --fold-design {TRIAL_DESIGN} ({rerun}"
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
    """Question 2a.1: the calibration curve, decoding error against the decoder's
    Local probability (the decoder-validation figure's line): the median over
    sessions of each session's median error in each of the ten Local-probability
    bins. The interquartile band the figure also draws is not reported."""
    make = value_factory(calibration_values, source, TAGS["calibration"])
    curve = quality_metrics.calibration_curve_statistics(calibration)
    if len(curve) != len(CALIBRATION_BINS):
        raise ValueError(
            f"the calibration curve has {len(curve)} bins, not {len(CALIBRATION_BINS)}"
        )
    return [
        make(
            f"calibration_error_bin_{word}",
            median,
            "quantity",
            "sig2",
            f"median over sessions of each session's median decoding error at time "
            f"bins whose Local probability is in the 0.1-wide bin centered on "
            f"{format_number(center, 'dec2')} (bin {index} of "
            f"{len(CALIBRATION_BINS)}), outside excluded time, sessions pooled",
            "cm",
        )
        for index, (word, (center, median)) in enumerate(
            zip(CALIBRATION_BINS, curve["median"].items(), strict=True), start=1
        )
    ]


def state_fraction_values(sessions: pd.DataFrame, source: str) -> list[Value]:
    """Question 2a.6: the decoder's state fractions while the animal moves and
    while it is still (the decoder-validation figure's bars)."""
    make = value_factory(state_fraction_values, source, TAGS["state_fractions"])
    fractions = quality_metrics.state_fraction_statistics(sessions)
    where = {
        "moving": "bins faster than 4 cm/s",
        "immobile": "bins slower than 4 cm/s",
    }
    return [
        make(
            f"state_fraction_{movement}_{_STATE_WORDS[state]}",
            fractions.loc[movement, state],
            "fraction",
            "pct1",
            f"median over sessions of each session's mean {state} over its "
            f"{where[movement]} outside excluded time, sessions pooled",
        )
        for movement in quality_metrics.MOVEMENT_STATES
        for state in quality_metrics.STATE_PROBABILITY_COLUMNS
    ]


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
            "share of immobile non-local events overlapping no SWR, pooled over animals",
        ),
        make(
            "swr_contained_fraction",
            stats["swr_contained_fraction"],
            "fraction",
            "pct",
            "share of immobile non-local events lying inside an SWR, pooled over "
            "animals",
        ),
        make(
            "swr_partial_fraction",
            stats["swr_partial_fraction"],
            "fraction",
            "pct",
            "share of immobile non-local events partly overlapping an SWR, pooled "
            "over animals",
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
                f"share of {detector} events that overlap no non-local event",
            ),
            make(
                f"{word}_mostly_local_fraction",
                stats["mostly_local_fraction"],
                "fraction",
                "pct",
                f"share of {detector} events that overlap no non-local event and have "
                "Local above 0.8 in more than half their rows",
                tags=TAGS["swr_content"] if main else None,
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
    Local-rest windows of their length (Figure 2h shows the participation)."""
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


def tetrode_holdout_values(windows: pd.DataFrame, source: str) -> list[Value]:
    """Question 2c.3: the held-out tetrode gain over local coding, per spike, of
    the events and their Local-rest controls (the held-out tetrode
    supplementary figure)."""
    make = value_factory(tetrode_holdout_values, source, TAGS["tetrode_holdout"])
    values = []
    estimates = (
        ("gain", "event", "gain_per_spike", "events", ""),
        (
            "spiking_gain",
            "event",
            "spiking_gain_per_spike",
            "events",
            " in the bins where the scored tetrode spiked",
        ),
        ("control_gain", "control", "gain_per_spike", "Local-rest controls", ""),
    )
    for name, kind, column, label, spiking in estimates:
        statistics = tetrode_holdout.holdout_statistics(windows, kind, column)
        for word, group in _RUN_GROUPS.items():
            row = statistics.loc[group]
            values += _per_spike_interval(
                make,
                f"{word}_heldout_{name}_per_spike",
                row,
                f"held-out tetrode score of {group} {label}: each tetrode's "
                "log predictive score under the other tetrodes' position "
                "distribution minus under local coding around the animal"
                f"{spiking}, summed over tetrodes and bins and divided by the "
                "window's spikes; mean over animals of each animal's median "
                "over epochs of each epoch's median",
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


#: The three decoders of the restriction run (``nonlocal_restriction.MODELS``),
#: in words.
_RESTRICTION_MODELS = {
    "none": "the local-only decoder (Local and No-Spike states at every bin)",
    "swr": "the SWR-only decoder (only the non-local states in SWR bins, only the "
    "local states elsewhere)",
    "full": "the full decoder (every state at every bin)",
}
#: The two score families (``nonlocal_restriction.FAMILY_SUFFIX``), in words.
_RESTRICTION_SCORES = {
    "filter": "one-step-ahead (filter) predictive log score of each 2 ms bin's spikes",
    "two_sided": "two-sided predictive log score of each 2 ms bin's spikes (from "
    "every other bin)",
}
_RESTRICTION_HELD_OUT = (
    "with each of five folds of whole trials (balanced by destination well; a "
    "trial's run to a well with its whole stay there, when longer than "
    f"{tetrode_holdout.MAX_SEGMENT_S:g} s, is cut into pieces of at most "
    f"{tetrode_holdout.MAX_SEGMENT_S:g} s that can fall in different folds) "
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
#: How a value over time is reduced over epochs and animals, in words.
_EPOCH_REDUCTION = (
    "mean over animals of each animal's median over its epochs; 95 % interval "
    "from the nested bootstrap of the animals and their epochs, small-sample t "
    "correction over the animals"
)
#: Each SWR group's word in a key (the inverse of ``_RUN_GROUPS``).
_GROUP_WORDS = {group: word for word, group in _RUN_GROUPS.items()}


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
    """The held-out event run's inputs to the restriction values and Figure 2g
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
    spike and per second over the events' bins outside SWRs. Figure 2g draws
    the latter over the events detected from the held-out decode instead
    (:func:`restriction_heldout_event_values`), so none of these carries its
    tag."""
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
                f"restriction_{_GROUP_WORDS[group]}_{kind_word}_{contrast}{suffix}"
                "_per_spike",
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
            **RESTRICTION_FIGURE,
            "families": list(FAMILY_SUFFIX),
            "units": ["per_spike", "per_s"],
        },
    ).set_index(["family", "unit", "group"])
    counts = restriction_windows.outside_swr_window_counts(windows)
    a, b = restriction_windows.CONTRASTS["full_minus_swr"]
    for family, suffix in FAMILY_SUFFIX.items():
        for unit, divided, words in _RESTRICTION_UNITS:
            for group in RESTRICTION_FIGURE_GROUPS:
                row = outside.loc[(family, unit, group)]
                word = _GROUP_WORDS[group]
                values += _interval_and_animals_positive(
                    make,
                    f"restriction_{word}_events_outside_swr_full_minus_swr{suffix}"
                    f"_{unit}",
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
                            f"restriction_{word}_events_outside_swr_scored"
                            + ("_per_spike" if unit == "per_spike" else ""),
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
    for group in RESTRICTION_FIGURE_GROUPS:
        row = counts.loc[("event", group)]
        word = _GROUP_WORDS[group]
        values.append(
            make(
                f"restriction_{word}_events_outside_swr_without_spikes",
                row["n_without_spikes_outside_swr"],
                "count",
                "count",
                f"{group} events with immobile bins outside SWRs but no spike in "
                "them: no value per spike there",
            )
        )
        after_visit.append(
            make(
                f"restriction_{word}_events_outside_swr_after_visit",
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
                    f"restriction_{word}_{kind_word}",
                    row["n_events"],
                    "count",
                    "count",
                    f"{group} {label} summed",
                ),
                make(
                    f"restriction_{word}_{kind_word}_scored",
                    row["n_scored"],
                    "count",
                    "count",
                    f"{group} {label} with a per-spike value (at least one spike)",
                ),
            ]
    values.append(
        make(
            "restriction_animals",
            row["n_animals"],
            "count",
            "count",
            "animals in the restriction comparison",
        )
    )
    return WindowValues(values, after_visit)


def restriction_heldout_event_values(windows: pd.DataFrame, source: str) -> list[Value]:
    """Question 2c.3 and Figure 2g's second axis: the full decoder minus the
    SWR-only decoder per spike over the bins outside SWRs of the events
    detected from the held-out decode (``windows``, those that start inside a
    well visit), events with no SWR and events that overlap one, in
    :data:`RESTRICTION_FIGURE_FAMILY` (:data:`RESTRICTION_FIGURE`); with the
    events of each group, those with a value and those with bins outside SWRs
    but no spike in them. Every value carries that panel's tag."""
    make = value_factory(
        restriction_heldout_event_values,
        source,
        TAGS["nonlocal_restriction_figure"],
    )
    statistics = restriction_windows.restriction_window_statistics(
        windows, **RESTRICTION_FIGURE
    ).set_index("group")
    counts = restriction_windows.outside_swr_window_counts(windows)
    a, b = restriction_windows.CONTRASTS["full_minus_swr"]
    values = []
    for group in RESTRICTION_FIGURE_GROUPS:
        row = statistics.loc[group]
        word = _GROUP_WORDS[group]
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
                    f"{partner} (of any well-visit start), pooled over animals",
                ),
                make(
                    f"{key}_animal_min",
                    row["share_animal_min"],
                    "fraction",
                    "pctnum",
                    "lowest per-animal share (an animal's share is the mean over "
                    "its epochs of the epoch's share)",
                ),
                make(
                    f"{key}_animal_max",
                    row["share_animal_max"],
                    "fraction",
                    "pct",
                    "highest per-animal share (the mean over the animal's epochs)",
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
    marks every held-out event, wherever it starts)."""
    make = value_factory(
        restriction_outside_swr_values, source, TAGS["nonlocal_restriction"]
    )
    statistics = restriction_totals.outside_swr_gain_statistics(
        restriction_totals.epoch_outside_swr_gains(time_sums, bin_s, windows)
    ).set_index(["family", "measure"])
    full_minus_swr = (
        f"under {_RESTRICTION_MODELS['full']} minus under {_RESTRICTION_MODELS['swr']}"
    )
    time = "every immobile bin outside SWRs of the epoch"
    values, outside_events = [], []
    for family, suffix in FAMILY_SUFFIX.items():
        key = f"restriction_outside_swr_full_minus_swr{suffix}"
        scored = f"{_RESTRICTION_SCORES[family]}, {_RESTRICTION_HELD_OUT}"
        for unit, divided, words in _RESTRICTION_UNITS:
            row = statistics.loc[(family, f"gain_{unit}")]
            values += _interval_and_animals_positive(
                make,
                f"{key}_{unit}",
                row,
                f"{scored}, {full_minus_swr}, summed over {time} and divided by "
                f"{divided}: {_EPOCH_REDUCTION}",
                unit=words,
            )
            if unit == "per_s":
                values += [
                    make(
                        f"{key}_epochs_positive",
                        row["n_epochs_positive"],
                        "count",
                        "count",
                        "epochs whose gain per second is above zero",
                    ),
                    make(
                        f"{key}_epochs",
                        row["n_epochs"],
                        "count",
                        "count",
                        "epochs with immobile bins outside SWRs",
                    ),
                ]
            row = statistics.loc[(family, f"outside_events_gain_{unit}")]
            outside_events += _interval_and_animals_positive(
                make,
                f"{key}_outside_heldout_events_{unit}",
                row,
                f"{scored}, {full_minus_swr}, summed over the epoch's immobile bins "
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
                make,
                f"{key}_{name}",
                statistics.loc[(family, measure)],
                f"share of that gain ({family} score), summed over {time}, that "
                f"lies in {part}: the events' gain "
                "over the same kind of bins (immobile, outside SWRs, inside a "
                "well visit; an event that runs past the final poke-out adds "
                f"its bins before it only) over the epoch's: {_EPOCH_REDUCTION}",
            )
    return OutsideSwrValues(values, outside_events)


def restriction_immobile_values(
    time_sums: pd.DataFrame, bin_s: pd.Series, source: str
) -> list[Value]:
    """Question 2c.3: the SWR-only and full decoders against the local-only
    decoder over all immobile time inside a well visit (time chosen by speed and
    the well visits, not by the decoder or the SWRs), per spike and per second,
    their difference, and the share of the full decoder's gain the SWR-only
    decoder captures. The gains per spike and their difference in
    :data:`RESTRICTION_FIGURE_FAMILY` are drawn on Figure 2g's first axis and
    carry that panel's tag."""
    make_text = value_factory(
        restriction_immobile_values, source, TAGS["nonlocal_restriction"]
    )
    make_figure = value_factory(
        restriction_immobile_values, source, TAGS["nonlocal_restriction_figure"]
    )
    statistics = restriction_totals.immobile_gain_statistics(
        restriction_totals.epoch_immobile_gains(time_sums, bin_s)
    ).set_index(["family", "measure"])
    drawn = (*RESTRICTION_FIGURE_DECODERS, RESTRICTION_FIGURE_DIFFERENCE)
    values = []
    for family, suffix in FAMILY_SUFFIX.items():
        scored = (
            f"{_RESTRICTION_SCORES[family]}, {_RESTRICTION_HELD_OUT}, summed over "
            "every immobile bin of the epoch (inside and outside SWRs)"
        )
        for unit, divided, words in _RESTRICTION_UNITS:
            for contrast in ("swr_minus_none", "full_minus_none", "full_minus_swr"):
                measure = f"{contrast}_{unit}"
                figure = family == RESTRICTION_FIGURE_FAMILY and measure in drawn
                a, b = restriction_windows.CONTRASTS[contrast]
                values += _interval_and_animals_positive(
                    make_figure if figure else make_text,
                    f"restriction_immobile_{contrast}{suffix}_{unit}",
                    statistics.loc[(family, measure)],
                    f"{scored}, under {_RESTRICTION_MODELS[a]} minus under "
                    f"{_RESTRICTION_MODELS[b]}, divided by {divided}: "
                    f"{_EPOCH_REDUCTION}",
                    unit=words,
                )
        row = statistics.loc[(family, RESTRICTION_FIGURE_DIFFERENCE)]
        make = make_figure if family == RESTRICTION_FIGURE_FAMILY else make_text
        values.append(
            make(
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
    row = statistics.loc[(RESTRICTION_FIGURE_FAMILY, RESTRICTION_FIGURE_DIFFERENCE)]
    values.append(
        make_figure(
            "restriction_immobile_epochs",
            row["n_epochs"],
            "count",
            "count",
            "epochs with immobile bins in the restriction comparison",
        )
    )
    return values


# --- one class or two -----------------------------------------------------------

#: The properties of the step test (``class_or_continuum.STEP_PROPERTIES``)
#: other than spike evidence, by their word in ``step_fits.csv``.
_OTHER_STEP_PROPERTIES = {
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
    analysis's CSV outputs.
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
    Kay threshold (Figure 2j)?

    Over the immobile time bins decoded non-local, each animal's bins pooled
    over its epochs.
    """
    make = value_factory(nonlocal_power_values, source, TAGS["nonlocal_power"])
    stats = event_statistics.nonlocal_below_threshold_statistics(
        event_statistics.animal_power_counts(counts), KAY_ZSCORE_THRESHOLD
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
    :func:`class_step_evidence_values` holds spike evidence's.
    """
    most, attaining = event_statistics.most_steps(
        _step_statistics(steps), list(_OTHER_STEP_PROPERTIES)
    )
    # the note names the property (or properties) that attain the maximum
    attained = ", ".join(_OTHER_STEP_PROPERTIES[p] for p in attaining)
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
    run_epochs = ok_epochs(read_manifest(run))
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

    source = f"{rel(inputs.event_table)} (canonical event table)"
    summary.add(*visit_context_values(table, source))
    summary.add(*over(event_cohort_values(events, source), EVENTS_SUBSET))
    summary.add(*over(boundary_offset_values(events, source), EVENTS_SUBSET))
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
                off_subset(inputs.event_start_rest_theta),
                off_visits(
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
                off_subset(inputs.brain_state_reference.parent),
                off_visits(inputs.brain_state_reference.parent, reference["epoch_id"]),
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
                off_subset(inputs.event_gamma),
                off_visits(inputs.event_gamma, gamma_reference["epoch_id"]),
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
                off_subset(folder),
                off_visits(folder, exposure["epoch_id"]),
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
    holdout_table = select_events_in(
        read_complete_run(inputs.tetrode_holdout, TETRODE_HOLDOUT_NAME, analyzed),
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
    summary.add(
        *over(
            pend(
                tetrode_holdout_values(
                    tetrode_holdout.window_scores(holdout_table),
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
    summary.inputs += [
        files_record(
            f"*_{TETRODE_HOLDOUT_NAME}.parquet",
            ok_table_paths(inputs.tetrode_holdout, TETRODE_HOLDOUT_NAME),
            ROOT_DIR,
        ),
        file_record(inputs.tetrode_holdout / "manifest.json", ROOT_DIR),
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
    # the held-out event run (Figure 2g): its events that start inside a well
    # visit, its time sums and time steps, and why a value read from them is
    # pending; a restriction run of the earlier design is refused (the window
    # and held-out event runs must have read this run, staleness below)
    heldout = read_heldout_restriction(
        heldout_dir, restriction_dir, inputs.bin_facts, analyzed, current_sha256
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
    summary.add(*over(pend(window_values.windows, *window_stale), CONTROL_SUBSET))
    # of the window values, only the counts after the visit read the visits
    # (through the per-bin facts)
    summary.add(
        *over(
            pend(window_values.after_visit, *window_stale, heldout.session_cache),
            CONTROL_SUBSET,
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
    heldout_runs_stale = (heldout.restriction_run, heldout.bin_facts_run)
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
    summary.inputs += [
        file_record(restriction_dir / MANIFEST, ROOT_DIR),
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
        if set(quality_table["epoch_id"]) != set(analyzed_epoch_ids()):
            raise ValueError(
                f"{folder / name} holds {quality_table['epoch_id'].nunique()} "
                f"epochs, not the {n_epochs} analyzed ones"
            )
    quality_source = (
        f"{rel(folder)} (scripts/decoding/measure_decoder_quality.py: each analyzed "
        "epoch's decode)"
    )
    summary.add(*decoding_error_values(quality_sessions, quality_source))
    summary.add(*calibration_values(calibration, quality_source))
    summary.add(*state_fraction_values(quality_sessions, quality_source))
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

    folder = inputs.nonlocal_by_ripple_power
    power_counts = pd.read_parquet(folder / "counts.parquet")
    if set(power_counts["epoch_id"]) != set(analyzed_epoch_ids()):
        raise ValueError(
            f"{folder / 'counts.parquet'} holds {power_counts['epoch_id'].nunique()} "
            f"epochs, not the {n_epochs} analyzed ones"
        )
    summary.add(
        *over(
            pend(
                nonlocal_power_values(
                    power_counts,
                    f"{rel(folder)} (scripts/event_properties/"
                    "measure_nonlocal_by_ripple_power.py: each analyzed epoch's decode)",
                ),
                off_subset(folder),
                off_visits(folder, power_counts["epoch_id"]),
            ),
            f"the immobile {TIME_SUBSET}",
        )
    )
    summary.inputs += [
        file_record(folder / name, ROOT_DIR)
        for name in ("counts.parquet", "manifest.json")
    ]
    return summary
