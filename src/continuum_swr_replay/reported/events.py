"""The events summary: what the events are, and how they compare with SWRs and Widloski events.

Every value is read from an output an analysis already wrote; nothing is
refitted here. The inputs (:class:`EventsInputs`):

- the canonical event table (``replay_content/event_table.parquet``): the
  immobile non-local events, their SWR-overlap class, timing against the
  matched SWR, and their per-event properties;
- an event features run (``replay_detection_comparison/event_features/<run>``):
  every compared event of the other detectors, whether it overlaps a non-local
  event (``pair_membership`` against ``state_space``), and its state shares
  (``*_state_shares.parquet``, from ``scripts/detection/add_state_shares.py``);
- the spike-evidence run that scores every event (``paper/figures/reality_all``, from
  ``scripts/detection/analyze_event_spike_evidence.py``);
- the held-out tetrode test split by SWR (``nl_bouts_by_swr_stats.csv`` beside
  the state-evidence run the detection summary reads, from
  ``scripts/detection/split_state_evidence_by_swr.py``);
- the content-occupancy table (``paper/figures/content_occupancy``, from
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

import hashlib
import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay import intervals
from continuum_swr_replay.analysis.hippocampus.property_contrast import (
    MAX_NAN_REPLICATE_SHARE,
)
from continuum_swr_replay.analysis.inventory_comparison.features import (
    STATE_SHARE_COLUMNS,
)
from continuum_swr_replay.paths import (
    ROOT_DIR,
    event_features_root,
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
)

_FIGURE_2 = ("paper", "figure: 2")
TAGS = {
    "cohort": Tags(("2b.2",), ("paper", "figure: 2b")),
    "left_out": Tags(("2b.2",), _FIGURE_2),
    "other_detectors_main": Tags(("2b.3",), _FIGURE_2),
    "other_detectors": Tags(("2b.3",), ("supplement",)),
    "boundaries": Tags(("2b.4",), ("paper", "figure: 2d")),
    "ripple": Tags(("2c.1", "2b.5"), ("paper", "figure: 3b")),
    "brain_state": Tags(("2c.1",), ("supplement",)),
    "rest_theta": Tags(("2c.1",), ("paper",)),
    "class_distribution": Tags(("2b.5",), ("supplement",)),
    "class_step": Tags(("2b.5", "2d.2", "2d.3", "2d.5", "2d.6"), ("paper",)),
    "class_step_evidence": Tags(("2b.5", "2c.3"), ("paper",)),
    "spike": Tags(("2c.2", "2c.3", "2c.5"), _FIGURE_2),
    "spike_remote": Tags(("2c.2", "2c.3", "2c.5", "2c.7"), _FIGURE_2),
    "spike_count": Tags(("2c.2",), _FIGURE_2),
    "spike_count_remote": Tags(("2c.2", "2c.7"), _FIGURE_2),
    "spike_evidence": Tags(("2c.3",), _FIGURE_2),
    "spike_evidence_remote": Tags(("2c.3", "2c.7"), _FIGURE_2),
    "spike_evidence_baseline": Tags(("2c.3", "2c.5"), _FIGURE_2),
    "spike_gpi": Tags(("2c.4",), ("supplement",)),
    "spike_gpi_remote": Tags(("2c.4", "2c.7"), ("supplement",)),
    "content_occupancy": Tags(("2d.1", "2c.8"), _FIGURE_2),
    "duration": Tags(("2d.6",), ("supplement",)),
    "extent": Tags(("2d.3",), ("paper", "figure: 3")),
    "speed": Tags(("2d.2",), ("paper", "figure: 3c")),
    "certainty": Tags(("2d.5",), ("supplement",)),
    "stillness": Tags(("2d.4",), ("paper", "figure: 3")),
    "location": Tags(("2d.1",), ("paper", "figure: 3")),
}
"""Each group of values: the questions it answers and where it is used. The
``_remote`` groups are the remote events' spike support, which also answers
question 2c.7."""

EVENT_FEATURES_RUN = "30cf1cc432e9"
"""The event features run the events summary reads (242 epochs, all ``ok``)."""

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
    reality_dir: Path
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
        reality_dir=ROOT_DIR / "paper/figures/reality_all/per_epoch",
        heldout_by_swr=processed_data_dir()
        / STATE_EVIDENCE_RUN
        / "state_evidence"
        / "nl_bouts_by_swr_stats.csv",
        content_occupancy=ROOT_DIR
        / "paper/figures/content_occupancy/content_occupancy_per_session.csv",
        brain_state_reference=processed_data_dir()
        / "brain_state_reference"
        / "reference.parquet",
        event_gamma=processed_data_dir() / "event_gamma",
        class_structure=processed_data_dir() / "ripple_class_structure",
        event_start_rest_theta=processed_data_dir() / "event_start_rest_theta",
        property_bootstrap=processed_data_dir() / "event_property_bootstrap",
        event_gamma_power=processed_data_dir() / "event_gamma_power",
    )


def _value_factory(source: str, tags: Tags):
    """A :class:`Value` builder with the shared source and tags filled in;
    ``tags_`` overrides the tags of one value."""

    def make(
        key: str,
        value: float | int,
        kind: str,
        style: str,
        description: str,
        unit: str = "",
        *,
        pending: str | None = None,
        tags_: Tags | None = None,
    ) -> Value:
        if kind == "count" and float(value).is_integer():
            value = int(value)
        elif kind != "count":
            value = float(value)
        return Value(
            key=key,
            value=value,
            kind=kind,
            style=style,
            unit=unit,
            description=description,
            questions=(tags_ or tags).questions,
            used_in=(tags_ or tags).used_in,
            source=source,
            pending=pending,
        )

    return make


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


def pend(
    values: Iterable[Value],
    note: str | None,
    selected: Callable[[Value], bool] = lambda value: True,
) -> list[Value]:
    """``values`` with ``note`` added to the pending note of the selected ones."""
    if note is None:
        return list(values)
    return [
        replace(v, pending=note if v.pending is None else f"{v.pending}; {note}")
        if selected(v)
        else v
        for v in values
    ]


def _manifest(folder: Path) -> dict:
    return json.loads((folder / "manifest.json").read_text())


# --- 2a: how common the events are, and how they sit against SWRs ------------


def event_cohort_values(events: pd.DataFrame, source: str) -> list[Value]:
    """Question 2b.2: how many immobile non-local events carry no SWR."""
    make = _value_factory(source, TAGS["cohort"])
    if not events["whole_event_speed_eligible"].all():
        raise ValueError(
            "the event table holds events that are not whole-event immobile"
        )
    classes = events["overlap_class"].value_counts()
    shares = (
        events.groupby("animal")["overlap_class"].value_counts(normalize=True).unstack()
    )
    non_swr = shares["non_swr"]
    plurality = (non_swr > shares.drop(columns="non_swr").max(axis=1)).sum()
    n_epochs = events.groupby(["animal", "session_date", "epoch"]).ngroups
    return [
        make(
            "nonlocal_events",
            len(events),
            "count",
            "count",
            "immobile non-local events in the canonical event table (whole event below 4 cm/s)",
        ),
        make(
            "nonlocal_events_non_swr",
            classes["non_swr"],
            "count",
            "count",
            "of those, events overlapping no SWR (overlap_class non_swr)",
        ),
        make(
            "nonlocal_events_swr_contained",
            classes["swr_contained"],
            "count",
            "count",
            "events lying inside an SWR (overlap_class swr_contained)",
        ),
        make(
            "nonlocal_events_swr_partial",
            classes["swr_partial"],
            "count",
            "count",
            "events partly overlapping an SWR (overlap_class swr_partial)",
        ),
        make(
            "non_swr_fraction",
            classes["non_swr"] / len(events),
            "fraction",
            "pct",
            "share of immobile non-local events overlapping no SWR, pooled over animals",
        ),
        make(
            "non_swr_fraction_animal_min",
            non_swr.min(),
            "fraction",
            "pctnum",
            "lowest per-animal share of non-SWR events",
        ),
        make(
            "non_swr_fraction_animal_max",
            non_swr.max(),
            "fraction",
            "pct",
            "highest per-animal share of non-SWR events",
        ),
        make("animals", len(shares), "count", "count", "animals in the event table"),
        make("epochs", n_epochs, "count", "count", "epochs in the event table"),
        make(
            "animals_non_swr_plurality",
            plurality,
            "count",
            "count",
            "animals in which non-SWR is the largest of the three overlap classes",
        ),
        make(
            "animals_non_swr_majority",
            (non_swr > 0.5).sum(),
            "count",
            "count",
            "animals in which more than half the events are non-SWR",
        ),
    ]


def boundary_offset_values(events: pd.DataFrame, source: str) -> list[Value]:
    """Question 2b.4: non-local event bounds against the matched SWR's (Figure 2d)."""
    make = _value_factory(source, TAGS["boundaries"])
    overlap = events[events["overlap_class"] != "non_swr"]
    start_ms = 1000 * (overlap["event_start_s"] - overlap["matched_swr_start"]).dropna()
    end_ms = 1000 * (overlap["event_end_s"] - overlap["matched_swr_end"]).dropna()
    if len(start_ms) != len(overlap) or len(end_ms) != len(overlap):
        raise ValueError("an SWR-overlapping event has no matched SWR bounds")
    return [
        make(
            "swr_overlapping_events",
            len(overlap),
            "count",
            "count",
            "immobile non-local events overlapping an SWR",
        ),
        make(
            "start_offset_median_ms",
            start_ms.median(),
            "quantity",
            "int",
            "median of event start minus matched SWR start",
            "ms",
        ),
        make(
            "end_offset_median_ms",
            end_ms.median(),
            "quantity",
            "int",
            "median of event end minus matched SWR end",
            "ms",
        ),
    ]


def _read_epochs(run_dir: Path, name: str, **kwargs) -> pd.DataFrame:
    paths = sorted(run_dir.glob(f"*_{name}.parquet"))
    if not paths:
        raise FileNotFoundError(f"no {name} tables in {run_dir}")
    return pd.concat((pd.read_parquet(p, **kwargs) for p in paths), ignore_index=True)


def other_detector_events(run_dir: Path) -> pd.DataFrame:
    """Each other detector's compared events: whether it overlaps a non-local
    event (``matched``), and its four state shares.

    Only events with a known membership (wholly inside the pair's shared
    exposure) are kept, as the comparison's ``directional`` table counts them.
    """
    keys = ["animal", "day", "epoch", "detector", "event_key"]
    shares = _read_epochs(run_dir, "state_shares")
    membership = _read_epochs(
        run_dir, "pair_membership", filters=[("other_detector", "==", "state_space")]
    )
    events = shares.merge(
        membership[[*keys, "membership"]], on=keys, how="inner", validate="1:1"
    )
    events = events[events["membership"].isin(["matched", "unmatched"])]
    return events.assign(matched=events["membership"] == "matched")


def check_against_directional(events: pd.DataFrame, directional: pd.DataFrame) -> None:
    """Raise unless the per-event memberships give the comparison's own counts."""
    mine = events.groupby("detector")["matched"].agg(n_eligible="size", n_matched="sum")
    theirs = (
        directional[directional["target_detector"] == "state_space"]
        .groupby("source_detector")[["n_eligible", "n_matched"]]
        .sum()
    )
    joined = theirs.join(mine, rsuffix="_events", how="outer")
    if not (
        (joined["n_eligible"] == joined["n_eligible_events"]).all()
        and (joined["n_matched"] == joined["n_matched_events"]).all()
    ):
        raise ValueError(f"memberships do not give the directional counts:\n{joined}")


def other_detector_values(events: pd.DataFrame, source: str) -> list[Value]:
    """Question 2b.3: the other side of the overlap, for each other detector.

    For each detector: the share of its events that overlap no non-local event,
    how those divide by content, and, for the events that do overlap, the
    shares of their time in each state. A row is in a state when that state's
    probability is above 0.8; "mostly" is more than half the event's rows.
    """
    values: list[Value] = []
    local, no_spike, non_local, uncertain = STATE_SHARE_COLUMNS
    for word, detector in DETECTORS.items():
        main = word == MAIN_DETECTOR
        make = _value_factory(
            source, TAGS["other_detectors_main"] if main else TAGS["other_detectors"]
        )
        mine = events[events["detector"] == detector]
        alone = mine[~mine["matched"]]
        overlapping = mine[mine["matched"]]
        alone_known = alone.dropna(subset=list(STATE_SHARE_COLUMNS))
        over_known = overlapping.dropna(subset=list(STATE_SHARE_COLUMNS))
        mostly_local = alone_known[local] > 0.5
        mostly_no_spike = alone_known[no_spike] > 0.5
        values += [
            make(
                f"{word}_events",
                len(mine),
                "count",
                "count",
                f"{detector} events compared with the non-local detector",
            ),
            make(
                f"{word}_no_nonlocal_fraction",
                len(alone) / len(mine),
                "fraction",
                "pct",
                f"share of {detector} events that overlap no non-local event",
            ),
            make(
                f"{word}_no_nonlocal_mostly_local_fraction",
                mostly_local.mean(),
                "fraction",
                "pct",
                f"of {detector} events overlapping no non-local event, the share with Local "
                "above 0.8 in more than half their rows",
            ),
            make(
                f"{word}_no_nonlocal_mostly_no_spike_fraction",
                mostly_no_spike.mean(),
                "fraction",
                "pct",
                f"of {detector} events overlapping no non-local event, the share with No-Spike "
                "above 0.8 in more than half their rows",
            ),
            make(
                f"{word}_no_nonlocal_mixed_fraction",
                (~mostly_local & ~mostly_no_spike).mean(),
                "fraction",
                "pct",
                f"of {detector} events overlapping no non-local event, the share with "
                "neither state in more than half their rows",
            ),
            make(
                f"{word}_overlap_local_share_median",
                over_known[local].median(),
                "fraction",
                "pct",
                f"median over {detector} events overlapping a non-local event of the share "
                "of their rows with Local above 0.8",
            ),
            make(
                f"{word}_overlap_no_spike_share_median",
                over_known[no_spike].median(),
                "fraction",
                "pct",
                f"median over {detector} events overlapping a non-local event of the share "
                "of their rows with No-Spike above 0.8",
            ),
            make(
                f"{word}_overlap_nonlocal_share_median",
                over_known[non_local].median(),
                "fraction",
                "pct",
                f"median over {detector} events overlapping a non-local event of the share "
                "of their rows with non-local above 0.8",
            ),
            make(
                f"{word}_overlap_uncertain_share_median",
                over_known[uncertain].median(),
                "fraction",
                "pct",
                f"median over {detector} events overlapping a non-local event of the share "
                "of their rows with no state above 0.8",
            ),
            make(
                f"{word}_overlap_any_local_fraction",
                (over_known[local] > 0).mean(),
                "fraction",
                "pct",
                f"share of {detector} events overlapping a non-local event that hold at "
                "least one row with Local above 0.8",
            ),
            make(
                f"{word}_overlap_any_no_spike_fraction",
                (over_known[no_spike] > 0).mean(),
                "fraction",
                "pct",
                f"share of {detector} events overlapping a non-local event that hold at "
                "least one row with No-Spike above 0.8",
            ),
            make(
                f"{word}_overlap_all_nonlocal_fraction",
                (over_known[non_local] == 1).mean(),
                "fraction",
                "pct",
                f"share of {detector} events overlapping a non-local event whose every row "
                "is non-local above 0.8",
            ),
        ]
        if main:
            per_animal = mine.groupby("animal")["matched"].mean()
            values += [
                make(
                    f"{word}_no_nonlocal_fraction_animal_min",
                    1 - per_animal.max(),
                    "fraction",
                    "pctnum",
                    f"lowest per-animal share of {detector} events overlapping no non-local event",
                ),
                make(
                    f"{word}_no_nonlocal_fraction_animal_max",
                    1 - per_animal.min(),
                    "fraction",
                    "pct",
                    f"highest per-animal share of {detector} events overlapping no non-local event",
                ),
            ]
    return values


# --- 2a: what an analysis without the non-local detector leaves out -------------

#: The other inventories an SWR- or rate-triggered analysis would use.
SWR_DEFINITIONS = ("kay", "karlsson", "yu", "zugaro", "carey")
TRIGGERS = (*SWR_DEFINITIONS, "mua")


def _intervals(table: pd.DataFrame, detectors: tuple[str, ...]) -> np.ndarray:
    rows = table[table["detector"].isin(detectors)]
    return rows[["event_start_s", "event_end_s"]].to_numpy(dtype=np.float64)


def left_out_seconds(run_dir: Path) -> dict[str, float]:
    """Seconds of non-local event time outside each other inventory, summed over epochs.

    Each epoch's compared events (``*_features.parquet``: every inventory's
    whole-event-immobile events): the non-local detector's event time, and the
    part of it that no event of Kay, of any of the five SWR definitions, of any
    trigger (the five and MUA), or of the Widloski detector covers; and the
    time Kay's SWRs span.
    """
    names = (
        "nonlocal",
        "kay",
        "outside_kay",
        "outside_swr",
        "outside_trigger",
        "outside_widloski",
    )
    totals = dict.fromkeys(names, 0.0)
    for path in sorted(run_dir.glob("*_features.parquet")):
        table = pd.read_parquet(
            path, columns=["detector", "event_start_s", "event_end_s"]
        )
        nonlocal_ = _intervals(table, ("state_space",))
        totals["nonlocal"] += intervals.total_duration(nonlocal_)
        totals["kay"] += intervals.total_duration(_intervals(table, ("kay",)))
        for name, detectors in (
            ("outside_kay", ("kay",)),
            ("outside_swr", SWR_DEFINITIONS),
            ("outside_trigger", TRIGGERS),
            ("outside_widloski", ("memoryless",)),
        ):
            outside = intervals.subtract(nonlocal_, _intervals(table, detectors))
            totals[name] += intervals.total_duration(outside)
    return totals


def left_out_values(run_dir: Path, report_dir: Path, source: str) -> list[Value]:
    """Question 2b.2: how many non-local events, and how much of their time, an
    analysis that starts from SWRs (or from any LFP- or rate-based trigger, or
    from the Widloski detector) does not include, in total and per hour of rest.
    """
    make = _value_factory(source, TAGS["left_out"])
    counts = pd.read_csv(report_dir / "counts_by_animal.csv")
    exposure_s = float(
        counts.loc[counts["detector"] == "state_space", "exposure_s"].sum()
    )
    hours = exposure_s / 3600.0
    cells = pd.read_csv(report_dir / "cells_by_animal.csv")
    no_trigger = cells[
        (cells["cell"] == "sequence_matched_by_no_trigger")
        & (cells["detector"] == "state_space")
    ][["n_events", "n_unmatched", "n_unknown"]].sum()
    membership = _read_epochs(
        run_dir, "pair_membership", filters=[("detector", "==", "state_space")]
    )
    known = membership[membership["membership"].isin(["matched", "unmatched"])]
    unmatched = (
        known.assign(alone=known["membership"] == "unmatched")
        .groupby("other_detector")["alone"]
        .agg(["sum", "size"])
    )
    seconds = left_out_seconds(run_dir)
    values = [
        make(
            "rest_hours",
            hours,
            "quantity",
            "sig2",
            "hours of analysed immobility the non-local detector observed (exposure), "
            "summed over epochs",
            "h",
        ),
        make(
            "nonlocal_time_s",
            seconds["nonlocal"],
            "quantity",
            "int",
            "seconds of non-local event time (compared events), summed over epochs",
            "s",
        ),
        make(
            "kay_time_s",
            seconds["kay"],
            "quantity",
            "int",
            "seconds Kay's SWRs span (compared events): what an SWR-triggered analysis "
            "examines",
            "s",
        ),
    ]
    for word, key, label in (
        ("kay", "outside_kay", "Kay's SWRs"),
        ("swr", "outside_swr", "the five SWR definitions together"),
        (
            "trigger",
            "outside_trigger",
            "the five SWR definitions and MUA together",
        ),
        (
            "widloski",
            "outside_widloski",
            "the Widloski detector's events",
        ),
    ):
        values += [
            make(
                f"nonlocal_time_outside_{word}_s",
                seconds[key],
                "quantity",
                "int",
                f"seconds of non-local event time that {label} do not cover",
                "s",
            ),
            make(
                f"nonlocal_time_outside_{word}_fraction",
                seconds[key] / seconds["nonlocal"],
                "fraction",
                "pct",
                f"share of non-local event time that {label} do not cover",
            ),
        ]
    for word, other, label in (
        ("kay", "kay", "no Kay SWR overlaps"),
        ("widloski", "memoryless", "no Widloski event overlaps"),
    ):
        n, total = int(unmatched.loc[other, "sum"]), int(unmatched.loc[other, "size"])
        values += [
            make(
                f"nonlocal_events_outside_{word}",
                n,
                "count",
                "count",
                f"non-local events {label} (detector comparison memberships)",
            ),
            make(
                f"nonlocal_events_outside_{word}_fraction",
                n / total,
                "fraction",
                "pct",
                f"share of the non-local events with a known membership that {label}",
            ),
            make(
                f"nonlocal_events_outside_{word}_per_hour",
                n / hours,
                "quantity",
                "sig2",
                f"non-local events {label}, per hour of analysed immobility",
                "per h",
            ),
        ]
    n_no_trigger = int(no_trigger["n_unmatched"])
    n_known = int(no_trigger["n_events"] - no_trigger["n_unknown"])
    values += [
        make(
            "nonlocal_events_outside_trigger",
            n_no_trigger,
            "count",
            "count",
            "non-local events no SWR definition and no MUA event overlaps "
            "(cell sequence_matched_by_no_trigger)",
        ),
        make(
            "nonlocal_events_outside_trigger_fraction",
            n_no_trigger / n_known,
            "fraction",
            "pct",
            "share of the non-local events with a known membership that no SWR "
            "definition and no MUA event overlaps",
        ),
        make(
            "nonlocal_events_outside_trigger_per_hour",
            n_no_trigger / hours,
            "quantity",
            "sig2",
            "non-local events no SWR definition and no MUA event overlaps, per hour of "
            "analysed immobility",
            "per h",
        ),
    ]
    return values


# --- 2b: theta, ripple and MUA during events --------------------------------------

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
    """Questions 2c.1 and 2b.5: theta, ripple power, MUA and slow and fast gamma during
    events, against running and quiet rest.

    Each event's value is its mean over its bins of the epoch's z-scored signal
    (the event table's ``theta_z``, ``ripple_power_z``, ``mua_rate_z``, and the
    gamma pass's ``slow_gamma_z``, ``fast_gamma_z``, joined on ``event_id``); the
    references are each epoch's median during running (>10 cm/s) and quiet rest
    (<4 cm/s, in no event), on the same scale (``brain_state_reference``). An
    event is theta-like when its theta reaches its epoch's running median.
    """
    keys = ["animal", "session_date", "epoch"]
    joined = events.merge(reference, on=keys, how="left", validate="m:1")
    if joined["theta_running_median"].isna().any():
        raise ValueError("an event's epoch has no running reference")
    values: list[Value] = []
    for word, column in _SIGNAL_EVENT_COLUMNS.items():
        make = _value_factory(
            source, TAGS["ripple"] if word == "ripple" else TAGS["brain_state"]
        )
        label = _SIGNAL_LABELS[word]
        values += [
            make(
                f"{word}_running_reference",
                reference[f"{word}_running_median"].median(),
                "quantity",
                "dec2",
                f"median over epochs of the epoch's median z-scored {label} while running",
                "z",
            ),
            make(
                f"{word}_rest_reference",
                reference[f"{word}_rest_median"].median(),
                "quantity",
                "dec2",
                f"median over epochs of the epoch's median z-scored {label} in quiet rest",
                "z",
            ),
        ]
        for group, is_swr in (("non_swr", False), ("swr", True)):
            rows = joined[joined["is_swr"] == is_swr].dropna(subset=[column])
            name = "non-SWR" if group == "non_swr" else "SWR-overlapping"
            per_animal = rows.groupby("animal").agg(
                event=(column, "median"),
                running=(f"{word}_running_median", "median"),
                rest=(f"{word}_rest_median", "median"),
            )
            values += [
                make(
                    f"{word}_median_{group}",
                    rows[column].median(),
                    "quantity",
                    "dec2",
                    f"median z-scored {label} of immobile {name} non-local events (event mean)",
                    "z",
                ),
                make(
                    f"{word}_animals_below_running_{group}",
                    (per_animal["event"] < per_animal["running"]).sum(),
                    "count",
                    "count",
                    f"animals whose median {name} event {label} is below their median running "
                    "level",
                ),
                make(
                    f"{word}_animals_above_rest_{group}",
                    (per_animal["event"] > per_animal["rest"]).sum(),
                    "count",
                    "count",
                    f"animals whose median {name} event {label} is above their median "
                    "quiet-rest level",
                ),
            ]
            if word == "theta":
                values.append(
                    make(
                        f"theta_like_fraction_{group}",
                        (rows[column] >= rows["theta_running_median"]).mean(),
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
    0 dB. The z values (``brain_state_values``) stay as they are.
    """
    make = _value_factory(source, TAGS["brain_state"])
    values: list[Value] = []
    for word in ("slow_gamma", "fast_gamma"):
        label = _SIGNAL_LABELS[word]
        for group in ("non_swr", "swr"):
            name = "non-SWR" if group == "non_swr" else "SWR-overlapping"
            column = f"{word}_db_{group}"
            if reference[column].isna().any():
                raise ValueError(
                    f"{int(reference[column].isna().sum())} epochs have no {column}"
                )
            rows = reference
            per_animal = rows.groupby("animal")[column].median()
            values += [
                make(
                    f"{word}_db_median_{group}",
                    rows[column].median(),
                    "quantity",
                    "sig2s",
                    f"median over epochs of the change in {label} during {name} events "
                    "against the epoch's quiet rest: 10 log10 of the ratio of mean band "
                    "powers",
                    "dB",
                ),
                make(
                    f"{word}_db_animals_above_rest_{group}",
                    (per_animal > 0).sum(),
                    "count",
                    "count",
                    f"animals whose median over epochs of that change ({name} events) "
                    "is above 0 dB",
                ),
            ]
    return values


# --- 2b: spike support ------------------------------------------------------

_REALITY_GROUPS = {
    "swr_remote": "SWR-remote",
    "non_swr_remote": "nonSWR-remote",
    "non_swr_close": "nonSWR-close",
}


REALITY_METRICS = "event_spike_evidence_metrics.parquet"
REALITY_EPOCHS = 242
"""The spike-evidence run scores every analysed epoch."""


def reality_metrics(per_epoch_dir: Path) -> pd.DataFrame:
    """The spike-evidence run's per-event metrics, one table per epoch, concatenated.

    Raises
    ------
    ValueError
        The run does not hold :data:`REALITY_EPOCHS` epochs, or an epoch's
        table names another epoch.
    """
    paths = sorted(per_epoch_dir.glob(f"*/{REALITY_METRICS}"))
    if len(paths) != REALITY_EPOCHS:
        raise ValueError(
            f"{per_epoch_dir} holds {len(paths)} epochs' metrics, not {REALITY_EPOCHS}"
        )
    tables = []
    for path in paths:
        table = pd.read_parquet(path)
        if set(table["epoch_id"]) != {path.parent.name}:
            raise ValueError(f"{path} holds epochs {sorted(set(table['epoch_id']))}")
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def spike_evidence_values(metrics: pd.DataFrame, source: str) -> list[Value]:
    """Questions 2c.2, 2c.3, 2c.4 and 2c.5: each event group's spike evidence against matched
    quiet windows (``baseline``). Event-level medians, pooled over epochs; the
    run scores every event and quiet window (no cap). The remote groups' values
    also answer question 2c.7, whether remote events pass these tests."""
    make = _value_factory(source, TAGS["spike"])
    values = [
        make(
            "evidence_epochs",
            metrics["epoch_id"].nunique(),
            "count",
            "count",
            "epochs in the spike-evidence run",
        ),
    ]
    base = metrics[metrics["group"] == "baseline"]
    by_epoch_base = base.groupby("epoch_id")["dll_per_spike"].median()
    by_animal_base = base.groupby("animal")["dll_per_spike"].median()
    groups = {"baseline": "baseline", **_REALITY_GROUPS}
    for word, group in groups.items():
        rows = metrics[metrics["group"] == group]
        label = "matched quiet windows" if word == "baseline" else f"{group} events"
        remote = word.endswith("remote")

        def pick(plain: Tags, far: Tags, remote: bool = remote) -> Tags:
            """The remote groups' values also answer question 2c.7."""
            return far if remote else plain

        values += [
            make(
                f"{word}_scored",
                len(rows),
                "count",
                "count",
                f"{label} scored",
                tags_=pick(TAGS["spike"], TAGS["spike_remote"]),
            ),
            make(
                f"{word}_spikes_median",
                rows["n_spikes"].median(),
                "quantity",
                "int",
                f"median spike count of {label}",
                "spikes",
                tags_=pick(TAGS["spike_count"], TAGS["spike_count_remote"]),
            ),
            make(
                f"{word}_tetrodes_median",
                rows["n_tetrodes"].median(),
                "quantity",
                "int",
                f"median number of tetrodes with a spike in {label}",
                "tetrodes",
                tags_=pick(TAGS["spike_count"], TAGS["spike_count_remote"]),
            ),
            make(
                f"{word}_dll_per_spike_median",
                rows["dll_per_spike"].median(),
                "quantity",
                "dec2",
                f"median log-likelihood ratio per spike, decoded location against the "
                f"animal's, of {label}",
                "nats per spike",
                tags_=TAGS["spike_evidence_baseline"]
                if word == "baseline"
                else pick(TAGS["spike_evidence"], TAGS["spike_evidence_remote"]),
            ),
            make(
                f"{word}_gpi_distance_median_cm",
                rows["dist_to_gpi_peak"].median(),
                "quantity",
                "int",
                f"median track distance from the MAP of {label} to the ground-process "
                "intensity peak",
                "cm",
                tags_=pick(TAGS["spike_gpi"], TAGS["spike_gpi_remote"]),
            ),
        ]
        if word == "baseline":
            continue
        epoch_med = rows.groupby("epoch_id")["dll_per_spike"].median()
        animal_med = rows.groupby("animal")["dll_per_spike"].median()
        values += [
            make(
                f"{word}_epochs_above_baseline",
                (epoch_med > by_epoch_base.reindex(epoch_med.index)).sum(),
                "count",
                "count",
                f"epochs whose median {group} event scores above the epoch's median quiet "
                "window, per spike",
                tags_=pick(TAGS["spike_evidence"], TAGS["spike_evidence_remote"]),
            ),
            make(
                f"{word}_epochs_scored",
                len(epoch_med),
                "count",
                "count",
                f"epochs with a scored {group} event",
                tags_=pick(TAGS["spike_evidence"], TAGS["spike_evidence_remote"]),
            ),
            make(
                f"{word}_animals_above_baseline",
                (animal_med > by_animal_base.reindex(animal_med.index)).sum(),
                "count",
                "count",
                f"animals whose median {group} event scores above their median quiet "
                "window, per spike",
                tags_=pick(TAGS["spike_evidence"], TAGS["spike_evidence_remote"]),
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
    make = _value_factory(source, TAGS["spike_evidence"])
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


# --- 2b: one class or two -------------------------------------------------------

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
    make = _value_factory(source, TAGS["class_distribution"])
    peak = distribution[
        (distribution.animal != "pooled") & (distribution.power == "peak")
    ]
    return [
        make(
            "ripple_peak_animals_unimodal",
            (peak.dip_p >= 0.05).sum(),
            "count",
            "count",
            "animals in which Hartigan's dip test does not reject one mode of the "
            "events' peak ripple power (p at or above 0.05)",
        ),
        make(
            "ripple_peak_mode_animal_min",
            peak.main_mode_z.min(),
            "quantity",
            "dec2",
            "lowest per-animal location of the main KDE mode of the events' peak "
            "ripple power",
            "z",
        ),
        make(
            "ripple_peak_mode_animal_max",
            peak.main_mode_z.max(),
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
    main = steps[
        (steps.n_knots == STEP_KNOTS)
        & (steps.power == "criterion")
        & np.isclose(steps.window_z, STEP_WINDOW_Z)
        & (steps.covariate == "none")
    ]
    by_property = main.groupby("property").has_step.agg(["sum", "size"])
    if not (by_property["size"] == 5).all():
        raise ValueError(f"not every property has 5 animals:\n{by_property}")
    others = by_property.loc[list(_OTHER_STEP_PROPERTIES), "sum"]
    # the note names the property (or properties) that attain the maximum
    attained = ", ".join(
        _OTHER_STEP_PROPERTIES[p] for p in others.index[others == others.max()]
    )
    make = _value_factory(source, TAGS["class_step"])
    evidence = main[main.property == "evidence"]
    make_evidence = _value_factory(source, TAGS["class_step_evidence"])
    return [
        make(
            "step_other_properties_criterion_animals_max",
            others.max(),
            "count",
            "count",
            "the most animals with a step at the threshold in any property but spike "
            "evidence (duration, spatial extent, replay speed, posterior peak, HPD "
            "coverage, spike count, arms visited): the term's interval excludes 0 and "
            f"it lowers the held-out error; attained by {attained}",
        ),
        make_evidence(
            "step_evidence_criterion_animals",
            by_property.loc["evidence", "sum"],
            "count",
            "count",
            "animals with a step at the threshold in spike evidence per spike (remote "
            "events): the term's interval excludes 0 and it lowers the held-out error",
        ),
        make_evidence(
            "step_evidence_criterion_animals_same_sign",
            (evidence.step_share_overall > 0).sum(),
            "count",
            "count",
            "animals whose jump in spike evidence per spike at the threshold points "
            "the way of their SWR minus non-SWR difference, whatever its interval",
        ),
        make_evidence(
            "step_evidence_criterion_share_median",
            evidence.step_share_overall.median(),
            "quantity",
            "sig2s",
            "median over animals of the jump in spike evidence per spike at the "
            "threshold, as a share of the animal's SWR minus non-SWR difference",
        ),
    ]


# --- 2c: properties -----------------------------------------------------------

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
    differ in the claim's direction. Not duration-controlled.
    """
    values = []
    for word, column, scale, unit, style, direction, tags in _PROPERTIES:
        make = _value_factory(source, tags)
        x = events[[column, "is_swr", "animal"]].dropna(subset=[column])
        x = x.assign(v=x[column] * scale)
        med = x.groupby("is_swr")["v"].median()
        by_animal = x.groupby(["animal", "is_swr"])["v"].median().unstack()
        compare: Callable[[pd.Series, pd.Series], pd.Series] = (
            pd.Series.lt if direction == "lower" else pd.Series.gt
        )
        in_direction = compare(by_animal[False], by_animal[True]).sum()
        name = _PROPERTY_NAMES[word]
        values += [
            make(
                f"{word}_median_non_swr",
                med[False],
                "quantity",
                style,
                f"median {name} of immobile non-SWR non-local events",
                unit,
            ),
            make(
                f"{word}_median_swr",
                med[True],
                "quantity",
                style,
                f"median {name} of immobile SWR-overlapping non-local events",
                unit,
            ),
            make(
                f"{word}_animals_in_direction",
                in_direction,
                "count",
                "count",
                f"animals whose non-SWR median {name} is {direction} than their SWR median",
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
        make = _value_factory(source, tags)
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
    make = _value_factory(source, TAGS["rest_theta"])
    if (events.rest_windows <= 0).any():
        raise ValueError(
            f"{int((events.rest_windows <= 0).sum())} events have no duration-matched "
            "quiet-rest window"
        )
    usable = events.assign(
        theta_like=events.theta_z >= events.theta_running_median,
    )
    values: list[Value] = []
    for group, is_swr in (("non_swr", False), ("swr", True)):
        rows = usable[usable.is_swr == is_swr]
        per_animal = rows.groupby("animal").agg(
            events=("theta_like", "mean"), rest=("rest_theta_like_share", "mean")
        )
        name = "non-SWR" if group == "non_swr" else "SWR-overlapping"
        values += [
            make(
                f"theta_like_fraction_rest_matched_{group}",
                rows.rest_theta_like_share.mean(),
                "fraction",
                "pct",
                f"share of quiet-rest windows, matched in duration to the {name} events, "
                "whose theta reaches the epoch's median running theta",
            ),
            make(
                f"theta_like_animals_above_rest_matched_{group}",
                (per_animal.events > per_animal.rest).sum(),
                "count",
                "count",
                f"animals whose share of theta-like {name} events is above that of "
                "their duration-matched quiet-rest windows",
            ),
        ]
    return values


# --- 2c: where content sits ---------------------------------------------------


def content_occupancy_values(occupancy: pd.DataFrame, source: str) -> list[Value]:
    """Question 2d.1: where content sits against the animal and the reward wells.

    Time-bin-weighted shares over all sessions (as the analysis prints them):
    at a remote well, at the animal's own well, on a remote path, near the
    animal off the wells.
    """
    make = _value_factory(source, TAGS["content_occupancy"])
    occupancy = occupancy.assign(
        at_well=occupancy["remote_well"] + occupancy["own_well"]
    )

    def weighted(frame: pd.DataFrame, column: str) -> float:
        return float((frame[column] * frame["n"]).sum() / frame["n"].sum())

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
        rows = occupancy[occupancy["group"] == group]
        for column in ("remote_well", "own_well", "at_well", "remote_path", "near_off"):
            values.append(
                make(
                    f"{word}_content_{column}_fraction",
                    weighted(rows, column),
                    "fraction",
                    "pct",
                    f"share of {group} non-local time bins whose content is "
                    f"{column.replace('_', ' ')}, weighted by time bins over sessions",
                )
            )
    per_animal = (
        occupancy.groupby(["animal", "group"])
        .apply(lambda g: weighted(g, "at_well"), include_groups=False)
        .unstack()
    )
    values.append(
        make(
            "animals_non_swr_more_at_well",
            (per_animal["nonSWR"] > per_animal["SWR"]).sum(),
            "count",
            "count",
            "animals whose non-SWR content is at a well more often than their SWR content",
        )
    )
    return values


# --- the summary ---------------------------------------------------------------


def summarize(inputs: EventsInputs, produced_by: str, commit: str) -> Summary:
    """The events summary from ``inputs``."""
    summary = Summary(name="events", produced_by=produced_by, commit=commit)

    events = pd.read_parquet(inputs.event_table)
    current_sha256 = hashlib.sha256(inputs.event_table.read_bytes()).hexdigest()

    def stale(folder: Path) -> str | None:
        return event_table_staleness(_manifest(folder), current_sha256, rel(folder))

    source = f"{rel(inputs.event_table)} (canonical event table)"
    summary.add(*event_cohort_values(events, source))
    summary.add(*boundary_offset_values(events, source))
    summary.inputs.append(file_record(inputs.event_table, ROOT_DIR))
    extra_paths = sorted(inputs.event_start_rest_theta.glob("*.parquet"))
    if len(extra_paths) != REALITY_EPOCHS:
        raise ValueError(
            f"{inputs.event_start_rest_theta} holds {len(extra_paths)} epochs"
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
    summary.add(
        *pend(
            property_values(events_with_extras, extras_source),
            extras_stale,
            lambda v: v.key.startswith("start_distance_"),
        )
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
    if len(reference) != REALITY_EPOCHS or len(gamma_reference) != REALITY_EPOCHS:
        raise ValueError("the brain-state or gamma reference does not hold every epoch")
    gamma_columns = [c for c in gamma_reference.columns if "gamma" in c]
    reference = reference.merge(
        gamma_reference[["animal", "session_date", "epoch", *gamma_columns]],
        on=["animal", "session_date", "epoch"],
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
    summary.add(
        *pend(
            brain_state_values(
                events_with_gamma,
                reference,
                f"{rel(inputs.event_table)} (event theta_z, ripple_power_z, mua_rate_z) "
                f"and {rel(inputs.brain_state_reference)} (scripts/detection/"
                "measure_brain_state_reference.py) and the gamma pass "
                f"{rel(inputs.event_gamma)} (scripts/detection/measure_event_gamma.py)",
            ),
            stale(inputs.event_gamma),
            lambda v: v.key.startswith(("slow_gamma_", "fast_gamma_")),
        )
    )
    summary.inputs += [
        file_record(inputs.brain_state_reference, ROOT_DIR),
        file_record(inputs.event_gamma / "manifest.json", ROOT_DIR),
    ]
    gamma_power = pd.read_parquet(inputs.event_gamma_power / "reference.parquet")
    if len(gamma_power) != REALITY_EPOCHS:
        raise ValueError(f"{inputs.event_gamma_power} does not hold every epoch")
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
    others = other_detector_events(run)
    directional_path = (
        inventory_comparison_root()
        / _input_run(run, "inventory")
        / "report"
        / "directional_by_animal.csv"
    )
    check_against_directional(others, pd.read_csv(directional_path))
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

    metrics = reality_metrics(inputs.reality_dir)
    summary.add(
        *spike_evidence_values(
            metrics,
            f"{rel(inputs.reality_dir)}/*/{REALITY_METRICS} (every event, no cap)",
        )
    )
    summary.inputs.append(
        files_record(
            REALITY_METRICS, inputs.reality_dir.glob(f"*/{REALITY_METRICS}"), ROOT_DIR
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


def rel(path: Path) -> str:
    """``path`` relative to the repository when it is under it."""
    return (
        str(path.relative_to(ROOT_DIR)) if path.is_relative_to(ROOT_DIR) else str(path)
    )


def _input_run(run_dir: Path, name: str) -> str:
    """The id of the input run ``name`` that the event features run read."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return manifest["settings"]["input_runs"][name]
