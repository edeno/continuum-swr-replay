"""The events summary's cohort values (questions 2b.2, 2b.4): how many events
there are in each class and well-visit label, and how their boundaries sit
against their SWRs. The statistics are computed in
``analysis.event_properties.event_statistics``."""

from __future__ import annotations

import pandas as pd

from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    count_visit_context,
)
from continuum_swr_replay.analysis.event_properties import event_statistics
from continuum_swr_replay.reported.events.common import (
    EVENTS_SUBSET,
    TAGS,
    SummaryReads,
    _animal_range_values,
)
from continuum_swr_replay.reported.formatting import milliseconds
from continuum_swr_replay.reported.values import Value, over, value_factory


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
            "share of immobile non-local events overlapping no SWR: the mean over "
            "animals of each animal's share (the mean over its epochs of the "
            "epoch's share), the same number as non_swr_fraction_animal_mean",
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
            "share of immobile non-local events lying inside an SWR: the mean over "
            "animals of each animal's share (the mean over its epochs of the "
            "epoch's share), the same number as swr_contained_fraction_animal_mean",
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
            "share of immobile non-local events partly overlapping an SWR: the mean "
            "over animals of each animal's share (the mean over its epochs of the "
            "epoch's share), the same number as swr_partial_fraction_animal_mean",
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


def boundary_offset_values(events: pd.DataFrame, source: str) -> list[Value]:
    """Question 2b.4: non-local event bounds against the matched SWR's (Figure 2f)."""
    make = value_factory(boundary_offset_values, source, TAGS["boundaries"])
    stats = event_statistics.boundary_offset_statistics(
        event_statistics.boundary_offsets(events), events
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
            "median of event start minus matched SWR start: each epoch's median, "
            "each animal's median over its epochs, the mean over animals",
            "ms",
        ),
        make(
            "end_offset_median_ms",
            milliseconds(stats["end_offset_median_s"]),
            "quantity",
            "int",
            "median of event end minus matched SWR end: each epoch's median, each "
            "animal's median over its epochs, the mean over animals",
            "ms",
        ),
        *(
            make(
                f"{bound}_offset_beyond_{name}_fraction",
                stats[f"{bound}_offset_beyond_{name}_fraction"],
                "fraction",
                "pct",
                f"share of SWR-overlapping events whose {bound} differs from the "
                f"matched SWR's {bound} by more than {1000 * tail_s:g} ms: each "
                "epoch's share, each animal's mean over its epochs, the mean over "
                "animals",
            )
            for bound in ("start", "end")
            for name, tail_s in event_statistics.OFFSET_TAILS_S.items()
        ),
    ]


def boundary_timing_values(events: pd.DataFrame, source: str) -> list[Value]:
    """Question 2b.4 and Figure 2's supplement, panel S2f: the share of
    SWR-overlapping events in each of the nine combinations of their start and
    end against their matched SWR's (earlier, together within 10 ms, later;
    ``event_statistics.boundary_timing_statistics``): the mean over animals of
    each animal's share, with the lowest and highest animal, and the same mean
    under the headline key (no figure draws it); and the share whose start and end
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
                f"share of the {what}: the mean over animals of each animal's share "
                f"(the mean over its epochs of the epoch's share), the same number as "
                f"{key}_share_animal_mean",
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


def add_cohort(reads: SummaryReads) -> None:
    """Add the well-visit labels, the cohort and the boundary timing."""
    summary = reads.summary
    table = reads.table
    events = reads.events
    source = reads.source
    summary.add(*visit_context_values(table, source))
    summary.add(*over(event_cohort_values(events, source), EVENTS_SUBSET))
    summary.add(*over(boundary_offset_values(events, source), EVENTS_SUBSET))
    summary.add(*over(boundary_timing_values(events, source), EVENTS_SUBSET))
