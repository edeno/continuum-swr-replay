"""The events summary's detector comparison (questions 2b.2, 2b.3): how the
other detectors' events overlap the non-local events, the non-local events they
find, and what an analysis without the non-local detector leaves out. The
statistics are computed in ``analysis.detector_comparison.nonlocal_comparison``."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.detector_comparison import nonlocal_comparison
from continuum_swr_replay.analysis.event_tables.prepare_data import (
    behavioral_context_cache_path,
)
from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported.events.common import (
    DETECTOR_SUBSET,
    LEFT_OUT_SUBSET,
    TAGS,
    SummaryReads,
    _animal_range_values,
)
from continuum_swr_replay.reported.values import (
    Value,
    file_record,
    files_record,
    over,
    pend,
    rel,
    value_factory,
)
from continuum_swr_replay.runs.io import ok_table_paths

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


def other_detector_values(events: pd.DataFrame, source: str) -> list[Value]:
    """Question 2b.3: the other side of the overlap, for each other detector.

    For each detector: the share of its events that overlap no non-local event,
    how those divide by content, and, for the events that do overlap, the
    shares of their time in each state. A row is in a state when that state's
    probability is above 0.8; "mostly" is more than half the event's rows.
    Every value is event -> epoch -> animal -> mean over animals
    (``nonlocal_comparison.other_detector_statistics``): a share over a
    subset of a detector's events is each epoch's share over its events of
    the subset, each animal's mean over its epochs with such an event, then
    the mean over the animals with one, so a subset with few events gives a
    mean of few epochs' shares, each of few events.
    """
    shared = (
        "; each epoch's share, each animal's mean over its epochs with such an "
        "event, then the mean over the animals with one (an animal with none is "
        "left out; an epoch's share can rest on few events)"
    )
    median = (
        "; each epoch's median, each animal's median over its epochs with such an "
        "event, then the mean over the animals with one"
    )
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
                f"share of {detector} events that overlap no non-local event: each "
                "epoch's share, each animal's mean over its epochs, the mean over "
                f"animals (the same number as {word}_no_nonlocal_fraction_animal_mean)",
                tags=TAGS["other_detectors"] if main else None,
            ),
            make(
                f"{word}_mostly_local_fraction",
                stats["mostly_local_fraction"],
                "fraction",
                "pct",
                f"share of {detector} events that overlap no non-local event and have "
                "Local above 0.8 in more than half their rows: each epoch's share, "
                "each animal's mean over its epochs, the mean over animals (the same "
                f"number as {word}_mostly_local_fraction_animal_mean)",
                tags=TAGS["other_detectors"] if main else None,
            ),
            make(
                f"{word}_no_nonlocal_mostly_local_fraction",
                stats["no_nonlocal_mostly_local_fraction"],
                "fraction",
                "pct",
                f"of {detector} events overlapping no non-local event, the share with Local "
                f"above 0.8 in more than half their rows{shared}",
            ),
            make(
                f"{word}_no_nonlocal_mostly_no_spike_fraction",
                stats["no_nonlocal_mostly_no_spike_fraction"],
                "fraction",
                "pct",
                f"of {detector} events overlapping no non-local event, the share with No-Spike "
                f"above 0.8 in more than half their rows{shared}",
            ),
            make(
                f"{word}_no_nonlocal_mixed_fraction",
                stats["no_nonlocal_mixed_fraction"],
                "fraction",
                "pct",
                f"of {detector} events overlapping no non-local event, the share with "
                f"neither state in more than half their rows{shared}",
            ),
            make(
                f"{word}_overlap_local_share_median",
                stats["overlap_local_share_median"],
                "fraction",
                "pct",
                f"median share of the rows with Local above 0.8 of {detector} events "
                f"overlapping a non-local event{median}",
            ),
            make(
                f"{word}_overlap_no_spike_share_median",
                stats["overlap_no_spike_share_median"],
                "fraction",
                "pct",
                f"median share of the rows with No-Spike above 0.8 of {detector} events "
                f"overlapping a non-local event{median}",
            ),
            make(
                f"{word}_overlap_nonlocal_share_median",
                stats["overlap_nonlocal_share_median"],
                "fraction",
                "pct",
                f"median share of the rows with non-local above 0.8 of {detector} events "
                f"overlapping a non-local event{median}",
            ),
            make(
                f"{word}_overlap_uncertain_share_median",
                stats["overlap_uncertain_share_median"],
                "fraction",
                "pct",
                f"median share of the rows with no state above 0.8 of {detector} events "
                f"overlapping a non-local event{median}",
            ),
            make(
                f"{word}_overlap_any_local_fraction",
                stats["overlap_any_local_fraction"],
                "fraction",
                "pct",
                f"share of {detector} events overlapping a non-local event that hold at "
                f"least one row with Local above 0.8{shared}",
            ),
            make(
                f"{word}_overlap_any_no_spike_fraction",
                stats["overlap_any_no_spike_fraction"],
                "fraction",
                "pct",
                f"share of {detector} events overlapping a non-local event that hold at "
                f"least one row with No-Spike above 0.8{shared}",
            ),
            make(
                f"{word}_overlap_all_nonlocal_fraction",
                stats["overlap_all_nonlocal_fraction"],
                "fraction",
                "pct",
                f"share of {detector} events overlapping a non-local event whose every row "
                f"is non-local above 0.8{shared}",
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
    animal's share, and the same mean under the headline key (no figure draws
    it) (``nonlocal_comparison.nonlocal_events_overlapped``, which the figure
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
                row["animal_mean"],
                float(np.min(row["animals"])),
                float(np.max(row["animals"])),
            ),
            (
                f"share of the non-local events that a {detector} event overlaps "
                "(of those whose overlap is known, inside the two detectors' "
                "shared exposure): the mean over animals of each animal's share "
                "(the mean over its epochs of the epoch's share), the same number "
                f"as {key}_animal_mean",
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
    for the text: the mean over animals (under two keys, the headline and
    ``_animal_mean``) and the lowest and highest animal's.

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
                row["animal_mean"],
                float(np.min(row["animals"])),
                float(np.max(row["animals"])),
            ),
            (
                f"share of the {detector} events that overlap a non-local event: the "
                "mean over animals of each animal's share (the mean over its epochs "
                "of the epoch's share), the same number as "
                f"{words[detector]}_with_nonlocal_fraction_animal_mean",
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
                f"share of non-local event time that {label} do not cover: "
                "each epoch's share, each animal's mean over its epochs, the mean over animals",
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
                f"share of the non-local events with a known membership that {label}: "
                "each epoch's share, each animal's mean over its epochs, the mean over animals",
            ),
            make(
                f"nonlocal_events_outside_{word}_per_hour",
                stats[f"nonlocal_events_outside_{word}_per_hour"],
                "quantity",
                "sig2",
                f"non-local events {label}, per hour of analyzed immobility inside a "
                "well visit: each epoch's events per hour of its exposure, each animal's mean over its epochs, the mean over animals",
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
            "definition and no MUA event overlaps: each epoch's share, each animal's mean over its epochs, the mean over animals",
        ),
        make(
            "nonlocal_events_outside_trigger_per_hour",
            stats["nonlocal_events_outside_trigger_per_hour"],
            "quantity",
            "sig2",
            "non-local events no SWR definition and no MUA event overlaps, per hour of "
            "analyzed immobility inside a well visit: each epoch's events per hour of its exposure, each animal's mean over its epochs, the mean over animals",
            "per h",
        ),
    ]
    return values


def add_detectors(reads: SummaryReads) -> None:
    """Add the other detectors' comparison and what they leave out."""
    summary = reads.summary
    inputs = reads.inputs
    events = reads.events
    run = reads.run
    run_epochs = reads.run_epochs
    visits = reads.visits
    off_time_mask = reads.off_time_mask
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
