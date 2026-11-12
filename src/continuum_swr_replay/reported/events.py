"""The events summary: what the events are, and how they compare with SWRs and Widloski events.

Every value is read from an output an analysis already wrote; nothing is
refitted here. This module names and describes the values. The statistics are
computed in ``analysis.hippocampus.event_statistics`` and
``analysis.inventory_comparison.nonlocal_comparison``; here only the size of an
input (its epochs, its animals) is counted. The inputs (:class:`EventsInputs`):

- the canonical event table (``replay_content/event_table.parquet``): the
  immobile non-local events, their SWR-overlap class, timing against the
  matched SWR, and their per-event properties;
- an event features run (``replay_detection_comparison/event_features/<run>``):
  every compared event of the other detectors, whether it overlaps a non-local
  event (``pair_membership`` against ``state_space``), and its state shares
  (``*_state_shares.parquet``, from ``scripts/detection/add_state_shares.py``);
- the spike-evidence run that scores every event
  (``Processed-Data/event_spike_evidence``, one folder per epoch, from
  ``scripts/detection/run_event_spike_evidence.sh``);
- the held-out tetrode test split by SWR (``nl_bouts_by_swr_stats.csv`` beside
  the state-evidence run the detection summary reads, from
  ``scripts/detection/split_state_evidence_by_swr.py``);
- the content-occupancy table (``Processed-Data/content_occupancy``, from
  ``scripts/content/analyze_content_occupancy.py``);
- the one-class-or-two tests (``Processed-Data/ripple_class_structure``, from
  ``scripts/detection/measure_ripple_class_structure.py``);
- each event's start distance and duration-matched quiet-rest theta
  (``Processed-Data/event_start_rest_theta``, from
  ``scripts/detection/measure_event_start_and_rest_theta.py``);
- the nested bootstrap of the property contrasts
  (``Processed-Data/event_property_bootstrap``, from
  ``scripts/detection/bootstrap_event_properties.py``);
- the brain-state reference (``Processed-Data/brain_state_reference``, from
  ``scripts/detection/measure_brain_state_reference.py``) and the two gamma
  passes of ``scripts/detection/measure_event_gamma.py``: the z-scored gamma per
  event and epoch (``Processed-Data/event_gamma``) and the band power's change
  against quiet rest in dB (``Processed-Data/event_gamma_power``).

Each derived output above that reads the canonical event table records the
table's SHA-256 in its manifest; an output made from another table than the
current one, or recording none, makes the values read from it pending
(:func:`event_table_staleness`).

Pooled values sum events over sessions and animals; "animals in direction"
counts the animals whose own value points the way the claim states.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.hippocampus import event_statistics
from continuum_swr_replay.analysis.hippocampus.property_contrast import (
    MAX_NAN_REPLICATE_SHARE,
)
from continuum_swr_replay.analysis.inventory_comparison import (
    feature_run,
    nonlocal_comparison,
)
from continuum_swr_replay.data_loaders.epoch_register import (
    analysed_animals,
    analysed_epoch_ids,
)
from continuum_swr_replay.paths import (
    ROOT_DIR,
    content_occupancy_dir,
    event_features_root,
    event_spike_evidence_dir,
    inventory_comparison_root,
    processed_data_dir,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.reported.detection import STATE_EVIDENCE_RUN
from continuum_swr_replay.reported.values import (
    Summary,
    Tags,
    Value,
    file_record,
    files_record,
    pend,
    rel,
    value_factory,
)
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS

TAGS = {
    "cohort": Tags(("2b.2",), ("2b",)),
    "left_out": Tags(("2b.2",), ("2",)),
    "other_detectors_main": Tags(("2b.3",), ("2",)),
    "other_detectors": Tags(("2b.3",)),
    "boundaries": Tags(("2b.4",), ("2d",)),
    "ripple": Tags(("2c.1", "2b.5"), ("3b",)),
    "brain_state": Tags(("2c.1",)),
    "class_distribution": Tags(("2b.5",)),
    "class_step": Tags(("2b.5", "2d.2", "2d.3", "2d.5", "2d.6")),
    "class_step_evidence": Tags(("2b.5", "2c.3")),
    "spike": Tags(("2c.2", "2c.3", "2c.5"), ("2",)),
    "spike_count": Tags(("2c.2",), ("2",)),
    "spike_evidence": Tags(("2c.3",), ("2",)),
    "spike_evidence_baseline": Tags(("2c.3", "2c.5"), ("2",)),
    "spike_gpi": Tags(("2c.4",)),
    "spike_remote": Tags(("2c.7",)),
    "content_occupancy": Tags(("2d.1", "2c.8"), ("2",)),
    "duration": Tags(("2d.6",)),
    "extent": Tags(("2d.3",), ("3",)),
    "speed": Tags(("2d.2",), ("3c",)),
    "certainty": Tags(("2d.5",)),
    "stillness": Tags(("2d.4",), ("3",)),
    "location": Tags(("2d.1",), ("3",)),
}
"""Each group of values: the questions it answers and the figures that show it. The
remote events' spike-support values also answer question 2c.7: their tags are a
spike group's joined with ``spike_remote``."""

EVENT_FEATURES_RUN = "30cf1cc432e9"
"""The event features run the events summary reads (242 epochs, all ``ok``). The
scripts that measure the summary's derived inputs from that run
(``scripts/detection/measure_event_start_and_rest_theta.py``,
``scripts/detection/measure_ripple_class_structure.py``) read its id here."""

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


@dataclass(frozen=True)
class EventsInputs:
    """The files the events summary reads."""

    event_table: Path
    event_features_run: Path
    spike_evidence_dir: Path
    heldout_by_swr: Path
    content_occupancy: Path
    brain_state_reference: Path
    event_gamma: Path
    class_structure: Path
    event_start_rest_theta: Path
    property_bootstrap: Path
    event_gamma_power: Path


def default_inputs() -> EventsInputs:
    return EventsInputs(
        event_table=table_path("event_table", replay_content_dir()),
        event_features_run=event_features_root() / EVENT_FEATURES_RUN,
        spike_evidence_dir=event_spike_evidence_dir(),
        heldout_by_swr=processed_data_dir()
        / STATE_EVIDENCE_RUN
        / "state_evidence"
        / "nl_bouts_by_swr_stats.csv",
        content_occupancy=content_occupancy_dir() / "content_occupancy_per_session.csv",
        brain_state_reference=processed_data_dir()
        / "brain_state_reference"
        / "reference.parquet",
        event_gamma=processed_data_dir() / "event_gamma",
        class_structure=processed_data_dir() / "ripple_class_structure",
        event_start_rest_theta=processed_data_dir() / "event_start_rest_theta",
        property_bootstrap=processed_data_dir() / "event_property_bootstrap",
        event_gamma_power=processed_data_dir() / "event_gamma_power",
    )


def event_table_staleness(manifest: dict, current_sha256: str, what: str) -> str | None:
    """Why an output made from the canonical event table is stale, or None.

    ``manifest`` records the table's SHA-256 under ``event_table`` or
    ``inputs.event_table``; an output recording none cannot be tied to the
    current table and is treated as stale.
    """
    recorded = (
        manifest.get("event_table")
        or manifest.get("inputs", {}).get("event_table")
        or {}
    ).get("sha256")
    if recorded is None:
        return f"{what} records no event-table SHA-256; rerun it before citing"
    if recorded != current_sha256:
        return (
            f"{what} was made from another event table than the current one "
            f"(sha256 {recorded[:12]}, current {current_sha256[:12]}); rerun it "
            "before citing"
        )
    return None


def _manifest(folder: Path) -> dict:
    return json.loads((folder / "manifest.json").read_text())


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
            "non_swr_fraction_animal_min",
            stats["non_swr_fraction_animal_min"],
            "fraction",
            "pctnum",
            "lowest per-animal share of non-SWR events",
        ),
        make(
            "non_swr_fraction_animal_max",
            stats["non_swr_fraction_animal_max"],
            "fraction",
            "pct",
            "highest per-animal share of non-SWR events",
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
            "animals in which non-SWR is the largest of the three overlap classes",
        ),
        make(
            "animals_non_swr_majority",
            stats["n_animals_non_swr_majority"],
            "count",
            "count",
            "animals in which more than half the events are non-SWR",
        ),
    ]


def boundary_offset_values(events: pd.DataFrame, source: str) -> list[Value]:
    """Question 2b.4: non-local event bounds against the matched SWR's (Figure 2d)."""
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
            1000 * stats["start_offset_median_s"],
            "quantity",
            "int",
            "median of event start minus matched SWR start",
            "ms",
        ),
        make(
            "end_offset_median_ms",
            1000 * stats["end_offset_median_s"],
            "quantity",
            "int",
            "median of event end minus matched SWR end",
            "ms",
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
                    f"lowest per-animal share of {detector} events overlapping no non-local event",
                ),
                make(
                    f"{word}_no_nonlocal_fraction_animal_max",
                    stats["no_nonlocal_fraction_animal_max"],
                    "fraction",
                    "pct",
                    f"highest per-animal share of {detector} events overlapping no non-local event",
                ),
            ]
    return values


# --- what an analysis without the non-local detector leaves out -----------------


def left_out_values(run_dir: Path, report_dir: Path, source: str) -> list[Value]:
    """Question 2b.2: how many non-local events, and how much of their time, an
    analysis that starts from SWRs (or from any LFP- or rate-based trigger, or
    from the Widloski detector) does not include, in total and per hour of rest.
    """
    make = value_factory(left_out_values, source, TAGS["left_out"])
    stats = nonlocal_comparison.left_out_statistics(run_dir, report_dir)
    values = [
        make(
            "rest_hours",
            stats["rest_hours"],
            "quantity",
            "sig2",
            "hours of analysed immobility the non-local detector observed (exposure), "
            "summed over epochs",
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
            "seconds Kay's SWRs span (compared events): what an SWR-triggered analysis "
            "examines",
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
                f"non-local events {label}, per hour of analysed immobility",
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
            "analysed immobility",
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
    """Questions 2c.1 and 2b.5: theta, ripple power and MUA during events, against
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
            TAGS["ripple"] if word == "ripple" else TAGS["brain_state"],
        )
        for word in ("theta", "ripple", "mua")
    }
    return _signal_values(makes, events, reference)


def gamma_z_values(
    events: pd.DataFrame, reference: pd.DataFrame, source: str
) -> list[Value]:
    """Question 2c.1: slow and fast gamma during events, against running and quiet
    rest.

    As :func:`brain_state_values`, on the gamma pass's ``slow_gamma_z`` and
    ``fast_gamma_z``, joined on ``event_id``, and its epochs' running and
    quiet-rest levels on the same scale.
    """
    make = value_factory(gamma_z_values, source, TAGS["brain_state"])
    return _signal_values({"slow_gamma": make, "fast_gamma": make}, events, reference)


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
                    f"animals whose median {name} event {label} is below the median, over "
                    "the same events, of each event's own epoch's median running level",
                ),
                make(
                    f"{word}_animals_above_rest_{group}",
                    stats["n_animals_above_rest"],
                    "count",
                    "count",
                    f"animals whose median {name} event {label} is above the median, over "
                    "the same events, of each event's own epoch's median quiet-rest level",
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
    """Question 2c.1: slow and fast gamma during events, as a change in band power
    against the same epoch's quiet rest, in dB.

    For each epoch and class: ``10 log10(mean band power over the class's event
    samples / mean band power over quiet rest)`` (below 4 cm/s, in no non-local,
    SWR or MUA event, outside excluded time), on the same band power the z
    values are made from (``scripts/detection/measure_event_gamma.py``). The
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


SPIKE_EVIDENCE_METRICS = "event_spike_evidence_metrics.parquet"
"""An epoch's table of the spike-evidence run, in the epoch's folder."""


def spike_evidence_metrics(per_epoch_dir: Path, n_epochs: int) -> pd.DataFrame:
    """The spike-evidence run's per-event metrics, one table per epoch, concatenated.

    Parameters
    ----------
    per_epoch_dir : Path
        The run's folder, with one folder per epoch.
    n_epochs : int
        The epochs the run must hold: it scores every analysed epoch.

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


def _remote_too(tags: Tags, word: str) -> Tags:
    """``tags``, joined with ``spike_remote`` when ``word`` names a remote group:
    the remote groups' values also answer question 2c.7."""
    return tags | TAGS["spike_remote"] if word.endswith("remote") else tags


def spike_evidence_values(metrics: pd.DataFrame, source: str) -> list[Value]:
    """Questions 2c.2, 2c.3, 2c.4 and 2c.5: each event group's spike evidence against matched
    quiet windows (``baseline``). Event-level medians, pooled over epochs; the
    run scores every event and quiet window (no cap). The remote groups' values
    also answer question 2c.7, whether remote events pass these tests."""
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
        scored_tags = _remote_too(TAGS["spike"], word)
        count_tags = _remote_too(TAGS["spike_count"], word)
        evidence_tags = _remote_too(TAGS["spike_evidence"], word)
        gpi_tags = _remote_too(TAGS["spike_gpi"], word)
        values += [
            make(
                f"{word}_scored",
                stats["n_scored"],
                "count",
                "count",
                f"{label} scored",
                tags=scored_tags,
            ),
            make(
                f"{word}_spikes_median",
                stats["spikes_median"],
                "quantity",
                "int",
                f"median spike count of {label}",
                "spikes",
                tags=count_tags,
            ),
            make(
                f"{word}_tetrodes_median",
                stats["tetrodes_median"],
                "quantity",
                "int",
                f"median number of tetrodes with a spike in {label}",
                "tetrodes",
                tags=count_tags,
            ),
            make(
                f"{word}_dll_per_spike_median",
                stats["dll_per_spike_median"],
                "quantity",
                "dec2",
                f"median log-likelihood ratio per spike, decoded location against the "
                f"animal's, of {label}",
                "nats per spike",
                tags=TAGS["spike_evidence_baseline"]
                if word == "baseline"
                else evidence_tags,
            ),
            make(
                f"{word}_gpi_distance_median_cm",
                stats["gpi_distance_median_cm"],
                "quantity",
                "int",
                f"median track distance from the MAP of {label} to the ground-process "
                "intensity peak",
                "cm",
                tags=gpi_tags,
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
                tags=evidence_tags,
            ),
            make(
                f"{word}_epochs_scored",
                stats["n_epochs_scored"],
                "count",
                "count",
                f"epochs with a scored {group} event",
                tags=evidence_tags,
            ),
            make(
                f"{word}_animals_above_baseline",
                stats["n_animals_above_baseline"],
                "count",
                "count",
                f"animals whose median {group} event scores above their median quiet "
                "window, per spike",
                tags=evidence_tags,
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
                "lower bound of its 95 % t-interval over animals",
                "nats per spike",
            ),
            make(
                f"heldout_{word}_per_spike_ci_high",
                row.animal_ci_hi,
                "quantity",
                "dec2",
                "upper bound of its 95 % t-interval over animals",
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


# --- one class or two -----------------------------------------------------------

#: The properties of the step test (``measure_ripple_class_structure.PROPERTIES``)
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
        distribution, n_animals=len(analysed_animals())
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
    """
    statistics = event_statistics.threshold_step_statistics(
        steps,
        n_knots=STEP_KNOTS,
        window_z=STEP_WINDOW_Z,
        n_animals=len(analysed_animals()),
    )
    most, attaining = event_statistics.most_steps(
        statistics, list(_OTHER_STEP_PROPERTIES)
    )
    # the note names the property (or properties) that attain the maximum
    attained = ", ".join(_OTHER_STEP_PROPERTIES[p] for p in attaining)
    make = value_factory(class_step_values, source, TAGS["class_step"])
    evidence = statistics.loc["evidence"]
    make_evidence = value_factory(
        class_step_values, source, TAGS["class_step_evidence"]
    )
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


# --- properties ---------------------------------------------------------------

#: (word, column, scale, unit, style, the expected direction for non-SWR against
#: SWR, tags)
_PROPERTIES = (
    (
        "duration",
        "event_duration_s",
        1000.0,
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
        TAGS["extent"],
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
    stats = event_statistics.class_medians(events, column, direction)
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
            stats["n_animals_in_direction"],
            "count",
            "count",
            f"animals whose non-SWR median {name} is {direction} than their SWR median",
        ),
    ]


def property_interval_values(intervals: pd.DataFrame, source: str) -> list[Value]:
    """Questions 2d.1–2d.6: the uncertainty of each property contrast, and
    the contrast at matched duration.

    A nested bootstrap (animals, then epochs within animal, then events within
    epoch; 1000 replicates, fixed seed) of the SWR minus non-SWR difference of
    class medians, and of the duration-matched difference: within ten quantile
    strata of the pooled event durations, averaged by stratum size
    (``analysis.hippocampus.property_contrast``). 95 % percentile intervals. The
    class medians' own intervals stay in the bootstrap's ``intervals.csv``.
    """
    if set(intervals.index) != {p[0] for p in _PROPERTIES}:
        raise ValueError(
            f"the bootstrap holds {sorted(intervals.index)}, not the events summary's properties"
        )
    nan_columns = [c for c in intervals.columns if c.endswith("_nan_replicates")]
    if not nan_columns:
        raise ValueError("the bootstrap does not count its replicates without estimate")
    share = intervals[nan_columns].div(intervals["n_replicates"], axis=0)
    if (share > MAX_NAN_REPLICATE_SHARE).any(axis=None):
        raise ValueError(f"too many bootstrap replicates without estimate:\n{share}")
    values: list[Value] = []
    for word, _, _, unit, _, _, tags in _PROPERTIES:
        make = value_factory(property_interval_values, source, tags)
        row = intervals.loc[word]
        name = _PROPERTY_NAMES[word]
        skipped = (
            f"; {row.matched_replicates_skipping_strata:.0f} of "
            f"{row.n_replicates:.0f} replicates left out a stratum that drew no "
            f"event of a class, {row.matched_difference_nan_replicates:.0f} had no "
            "estimate"
        )
        values += [
            make(
                f"{word}_difference_low",
                row.difference_low,
                "quantity",
                "sig2s",
                f"lower bound of the 95 % nested-bootstrap interval of the SWR minus "
                f"non-SWR difference of median {name}",
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
                f"lower bound of its 95 % nested-bootstrap interval{skipped}",
                unit,
            ),
            make(
                f"{word}_matched_difference_high",
                row.matched_difference_high,
                "quantity",
                "sig2s",
                "upper bound of its 95 % nested-bootstrap interval",
                unit,
            ),
        ]
    return values


def rest_theta_values(events: pd.DataFrame, source: str) -> list[Value]:
    """Question 2c.1: the theta-like share of events against duration-matched quiet rest.

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
                "their duration-matched quiet-rest windows",
            ),
        ]
    return values


# --- where content sits -------------------------------------------------------


def content_occupancy_values(occupancy: pd.DataFrame, source: str) -> list[Value]:
    """Question 2d.1: where content sits against the animal and the reward wells.

    Time-bin-weighted shares over all sessions (as the analysis prints them):
    at a remote well, at the animal's own well, on a remote path, near the
    animal off the wells.
    """
    make = value_factory(content_occupancy_values, source, TAGS["content_occupancy"])
    shares = event_statistics.content_occupancy_shares(occupancy)
    values = [
        make(
            "occupancy_epochs",
            occupancy["epoch_id"].nunique(),
            "count",
            "count",
            "epochs in the content-occupancy table",
        ),
    ]
    for word, group in (("non_swr", "nonSWR"), ("swr", "SWR")):
        for column in event_statistics.OCCUPANCY_PLACES:
            values.append(
                make(
                    f"{word}_content_{column}_fraction",
                    shares.loc[group, column],
                    "fraction",
                    "pct",
                    f"share of {group} non-local time bins whose content is "
                    f"{column.replace('_', ' ')}, weighted by time bins over sessions",
                )
            )
    values.append(
        make(
            "animals_non_swr_more_at_well",
            event_statistics.n_animals_non_swr_more_at_well(occupancy),
            "count",
            "count",
            "animals whose non-SWR content is at a well more often than their SWR content",
        )
    )
    return values


# --- the summary ---------------------------------------------------------------


def summarize(inputs: EventsInputs, produced_by: str) -> Summary:
    """The events summary from ``inputs``."""
    summary = Summary(name="events", produced_by=produced_by)

    events = pd.read_parquet(inputs.event_table)
    event_table_record = file_record(inputs.event_table, ROOT_DIR)
    current_sha256 = event_table_record["sha256"]
    # every per-epoch input must hold the epoch register's analysed epochs
    n_epochs = len(analysed_epoch_ids())

    def stale(folder: Path) -> str | None:
        return event_table_staleness(_manifest(folder), current_sha256, rel(folder))

    source = f"{rel(inputs.event_table)} (canonical event table)"
    summary.add(*event_cohort_values(events, source))
    summary.add(*boundary_offset_values(events, source))
    summary.inputs.append(event_table_record)
    extra_paths = sorted(inputs.event_start_rest_theta.glob("*.parquet"))
    if len(extra_paths) != n_epochs:
        raise ValueError(
            f"{inputs.event_start_rest_theta} holds {len(extra_paths)} epochs, "
            f"not {n_epochs}"
        )
    extras = pd.concat(map(pd.read_parquet, extra_paths), ignore_index=True)
    events_with_extras = events.merge(
        extras.drop(columns=["epoch_id", "decode_sha256"]),
        on="event_id",
        how="left",
        validate="1:1",
    )
    if events_with_extras.n_bins.isna().any():
        raise ValueError("an event has no start-distance and rest-theta row")
    extras_source = (
        f"{source} and {rel(inputs.event_start_rest_theta)} "
        "(scripts/detection/measure_event_start_and_rest_theta.py)"
    )
    extras_stale = stale(inputs.event_start_rest_theta)
    summary.add(*property_values(events_with_extras, extras_source))
    summary.add(
        *pend(start_distance_values(events_with_extras, extras_source), extras_stale)
    )
    summary.add(
        *pend(rest_theta_values(events_with_extras, extras_source), extras_stale)
    )
    summary.inputs += [
        files_record("*.parquet", extra_paths, ROOT_DIR),
        file_record(inputs.event_start_rest_theta / "manifest.json", ROOT_DIR),
    ]
    intervals = pd.read_csv(inputs.property_bootstrap / "intervals.csv", index_col=0)
    summary.add(
        *pend(
            property_interval_values(
                intervals,
                f"{rel(inputs.property_bootstrap)} "
                "(scripts/detection/bootstrap_event_properties.py)",
            ),
            stale(inputs.property_bootstrap),
        )
    )
    summary.inputs += [
        file_record(inputs.property_bootstrap / name, ROOT_DIR)
        for name in ("intervals.csv", "manifest.json")
    ]
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
    events_with_gamma = events.merge(
        gamma_events, on="event_id", how="left", validate="1:1"
    )
    if events_with_gamma[["slow_gamma_z", "fast_gamma_z"]].isna().all(axis=1).any():
        raise ValueError("an event has no gamma value")
    signal_source = (
        f"{rel(inputs.event_table)} (event theta_z, ripple_power_z, mua_rate_z) "
        f"and {rel(inputs.brain_state_reference)} (scripts/detection/"
        "measure_brain_state_reference.py) and the gamma pass "
        f"{rel(inputs.event_gamma)} (scripts/detection/measure_event_gamma.py)"
    )
    summary.add(*brain_state_values(events_with_gamma, reference, signal_source))
    summary.add(
        *pend(
            gamma_z_values(events_with_gamma, reference, signal_source),
            stale(inputs.event_gamma),
        )
    )
    summary.inputs += [
        file_record(inputs.brain_state_reference, ROOT_DIR),
        file_record(inputs.event_gamma / "manifest.json", ROOT_DIR),
    ]
    gamma_power = pd.read_parquet(inputs.event_gamma_power / "reference.parquet")
    if len(gamma_power) != n_epochs:
        raise ValueError(
            f"{inputs.event_gamma_power} holds {len(gamma_power)} epochs, not {n_epochs}"
        )
    summary.add(
        *pend(
            gamma_db_values(
                gamma_power,
                f"{rel(inputs.event_gamma_power)}/reference.parquet "
                "(scripts/detection/measure_event_gamma.py --out-name event_gamma_power)",
            ),
            stale(inputs.event_gamma_power),
        )
    )
    summary.inputs += [
        file_record(inputs.event_gamma_power / name, ROOT_DIR)
        for name in ("reference.parquet", "manifest.json")
    ]
    summary.inputs.append(
        files_record(
            "*_events.parquet", inputs.event_gamma.glob("*_events.parquet"), ROOT_DIR
        )
    )

    run = inputs.event_features_run
    others = nonlocal_comparison.other_detector_events(run)
    directional_path = (
        inventory_comparison_root()
        / _input_run(run, "inventory")
        / "report"
        / "directional_by_animal.csv"
    )
    nonlocal_comparison.check_against_directional(others, pd.read_csv(directional_path))
    summary.add(
        *other_detector_values(
            others,
            f"event features run {run.name}: pair_membership against state_space and "
            "state_shares (scripts/detection/add_state_shares.py)",
        )
    )
    report_dir = directional_path.parent
    summary.add(
        *left_out_values(
            run,
            report_dir,
            f"event features run {run.name} (every inventory's compared events) and the "
            f"detector comparison's report ({rel(report_dir)}: counts_by_animal.csv, "
            "cells_by_animal.csv)",
        )
    )
    summary.inputs += [
        files_record("*_features.parquet", run.glob("*_features.parquet"), ROOT_DIR),
        file_record(report_dir / "counts_by_animal.csv", ROOT_DIR),
        file_record(report_dir / "cells_by_animal.csv", ROOT_DIR),
    ]
    summary.inputs += [
        files_record(
            "*_state_shares.parquet", run.glob("*_state_shares.parquet"), ROOT_DIR
        ),
        files_record(
            "*_pair_membership.parquet", run.glob("*_pair_membership.parquet"), ROOT_DIR
        ),
        file_record(directional_path, ROOT_DIR),
    ]

    metrics = spike_evidence_metrics(inputs.spike_evidence_dir, n_epochs)
    summary.add(
        *spike_evidence_values(
            metrics,
            f"{rel(inputs.spike_evidence_dir)}/*/{SPIKE_EVIDENCE_METRICS} "
            "(every event, no cap)",
        )
    )
    summary.inputs.append(
        files_record(
            SPIKE_EVIDENCE_METRICS,
            inputs.spike_evidence_dir.glob(f"*/{SPIKE_EVIDENCE_METRICS}"),
            ROOT_DIR,
        )
    )

    stats = pd.read_csv(inputs.heldout_by_swr)
    summary.add(*heldout_by_swr_values(stats, rel(inputs.heldout_by_swr)))
    summary.inputs += [
        file_record(inputs.heldout_by_swr, ROOT_DIR),
        file_record(
            inputs.heldout_by_swr.with_name("nl_bouts_by_swr.provenance.json"),
            ROOT_DIR,
        ),
    ]

    occupancy = pd.read_csv(inputs.content_occupancy)
    summary.add(*content_occupancy_values(occupancy, rel(inputs.content_occupancy)))
    summary.inputs.append(file_record(inputs.content_occupancy, ROOT_DIR))

    folder = inputs.class_structure
    names = ("distribution.csv", "step_fits.csv", "manifest.json")
    source = (
        f"{rel(folder)} (scripts/detection/measure_ripple_class_structure.py: the "
        f"event table, event features run {EVENT_FEATURES_RUN}, the decode frames and "
        "the spike-evidence run)"
    )
    class_stale = stale(folder)
    summary.add(
        *pend(
            class_distribution_values(pd.read_csv(folder / "distribution.csv"), source),
            class_stale,
        )
    )
    summary.add(
        *pend(
            class_step_values(pd.read_csv(folder / "step_fits.csv"), source),
            class_stale,
        )
    )
    summary.inputs += [file_record(folder / name, ROOT_DIR) for name in names]
    return summary


def _input_run(run_dir: Path, name: str) -> str:
    """The id of the input run ``name`` that the event features run read."""
    run: str = feature_run.read_manifest(run_dir)["settings"]["input_runs"][name]
    return run
