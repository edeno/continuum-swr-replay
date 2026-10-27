"""Part 2's reported values: what the events are, and how they compare with SWRs and Widloski events.

Every value is read from an output an analysis already wrote; nothing is
refitted here. The inputs (:class:`Part2Inputs`):

- the canonical event table (``replay_content/event_table.parquet``): the
  immobile non-local events, their SWR-overlap class, timing against the
  matched SWR, and their per-event properties;
- an event features run (``replay_detection_comparison/event_features/<run>``):
  every compared event of the other detectors, whether it overlaps a non-local
  event (``pair_membership`` against ``state_space``), and its state shares
  (``*_state_shares.parquet``, from ``scripts/detection/add_state_shares.py``);
- the spike-evidence run (``paper/figures/reality_r11``, from
  ``scripts/detection/analyze_event_spike_evidence.py``);
- the held-out tetrode test split by SWR, from an untracked research script
  (every value from it is marked pending);
- the content-occupancy table (``paper/figures/content_occupancy``, from
  ``scripts/content/analyze_content_occupancy.py``).

Pooled values sum events over sessions and animals; "animals in direction"
counts the animals whose own value points the way the claim states.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.inventory_comparison.features import (
    STATE_SHARE_COLUMNS,
)
from continuum_swr_replay.paths import (
    ROOT_DIR,
    event_features_root,
    inventory_comparison_root,
    replay_content_dir,
    table_path,
)
from continuum_swr_replay.reported.values import (
    Summary,
    Value,
    file_record,
    files_record,
)

EVENT_FEATURES_RUN = "30cf1cc432e9"
"""The event features run Part 2 reads (242 epochs, all ``ok``)."""

WIDLOSKI_PENDING = "Widloski detector: pending the Phase 2 pilot checks"
HELDOUT_PENDING = (
    "computed by an untracked research script "
    "(.claude/research/refresh_followup/f1_scripts/state_evidence_by_swr.py); "
    "promote it to a tracked script before citing"
)

#: The other detectors of claim 23, by the word their keys use. Kay is the main
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
class Part2Inputs:
    """The files Part 2 reads."""

    event_table: Path
    event_features_run: Path
    reality_metrics: Path
    heldout_by_swr: Path
    content_occupancy: Path


def default_inputs() -> Part2Inputs:
    return Part2Inputs(
        event_table=table_path("event_table", replay_content_dir()),
        event_features_run=event_features_root() / EVENT_FEATURES_RUN,
        reality_metrics=ROOT_DIR
        / "paper/figures/reality_r11/event_spike_evidence_metrics.parquet",
        heldout_by_swr=ROOT_DIR
        / ".claude/research/refresh_followup/f1_outputs/full240_nl_bouts_by_swr_stats.csv",
        content_occupancy=ROOT_DIR
        / "paper/figures/content_occupancy/content_occupancy_per_session.csv",
    )


def _value_factory(source: str, claims: tuple[str, ...], used_in: tuple[str, ...]):
    """A :class:`Value` builder with the shared source, claims and use filled in."""

    def make(
        key: str,
        value: float | int,
        kind: str,
        style: str,
        description: str,
        unit: str = "",
        *,
        pending: str | None = None,
        claims_: tuple[str, ...] | None = None,
        used_in_: tuple[str, ...] | None = None,
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
            claims=claims_ or claims,
            used_in=used_in_ or used_in,
            source=source,
            pending=pending,
        )

    return make


# --- 2a: how common the events are, and how they sit against SWRs ------------


def event_cohort_values(events: pd.DataFrame, source: str) -> list[Value]:
    """Claim 2: how many immobile non-local events carry no SWR."""
    make = _value_factory(source, ("2", "21"), ("paper", "figure: 2b"))
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
    """Claim 22: non-local event bounds against the matched SWR's (Figure 2d)."""
    make = _value_factory(source, ("22",), ("paper", "figure: 2d"))
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
    """Claim 23: the other side of the overlap, for each other detector.

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
            source,
            ("23", "4"),
            ("paper", "figure: main, other side of the overlap")
            if main
            else ("supplement",),
        )
        pending = WIDLOSKI_PENDING if detector.startswith("memoryless") else None
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
                pending=pending,
            ),
            make(
                f"{word}_no_nonlocal_fraction",
                len(alone) / len(mine),
                "fraction",
                "pct",
                f"share of {detector} events that overlap no non-local event",
                pending=pending,
            ),
            make(
                f"{word}_no_nonlocal_mostly_local_fraction",
                mostly_local.mean(),
                "fraction",
                "pct",
                f"of {detector} events overlapping no non-local event, the share with Local "
                "above 0.8 in more than half their rows",
                pending=pending,
            ),
            make(
                f"{word}_no_nonlocal_mostly_no_spike_fraction",
                mostly_no_spike.mean(),
                "fraction",
                "pct",
                f"of {detector} events overlapping no non-local event, the share with No-Spike "
                "above 0.8 in more than half their rows",
                pending=pending,
            ),
            make(
                f"{word}_no_nonlocal_mixed_fraction",
                (~mostly_local & ~mostly_no_spike).mean(),
                "fraction",
                "pct",
                f"of {detector} events overlapping no non-local event, the share with "
                "neither state in more than half their rows",
                pending=pending,
            ),
            make(
                f"{word}_overlap_local_share_median",
                over_known[local].median(),
                "fraction",
                "pct",
                f"median over {detector} events overlapping a non-local event of the share "
                "of their rows with Local above 0.8",
                pending=pending,
            ),
            make(
                f"{word}_overlap_no_spike_share_median",
                over_known[no_spike].median(),
                "fraction",
                "pct",
                f"median over {detector} events overlapping a non-local event of the share "
                "of their rows with No-Spike above 0.8",
                pending=pending,
            ),
            make(
                f"{word}_overlap_nonlocal_share_median",
                over_known[non_local].median(),
                "fraction",
                "pct",
                f"median over {detector} events overlapping a non-local event of the share "
                "of their rows with non-local above 0.8",
                pending=pending,
            ),
            make(
                f"{word}_overlap_uncertain_share_median",
                over_known[uncertain].median(),
                "fraction",
                "pct",
                f"median over {detector} events overlapping a non-local event of the share "
                "of their rows with no state above 0.8",
                pending=pending,
            ),
            make(
                f"{word}_overlap_any_local_fraction",
                (over_known[local] > 0).mean(),
                "fraction",
                "pct",
                f"share of {detector} events overlapping a non-local event that hold at "
                "least one row with Local above 0.8",
                pending=pending,
            ),
            make(
                f"{word}_overlap_any_no_spike_fraction",
                (over_known[no_spike] > 0).mean(),
                "fraction",
                "pct",
                f"share of {detector} events overlapping a non-local event that hold at "
                "least one row with No-Spike above 0.8",
                pending=pending,
            ),
            make(
                f"{word}_overlap_all_nonlocal_fraction",
                (over_known[non_local] == 1).mean(),
                "fraction",
                "pct",
                f"share of {detector} events overlapping a non-local event whose every row "
                "is non-local above 0.8",
                pending=pending,
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


# --- 2b: spike support ------------------------------------------------------

_REALITY_GROUPS = {
    "swr_remote": "SWR-remote",
    "non_swr_remote": "nonSWR-remote",
    "non_swr_close": "nonSWR-close",
}


def spike_evidence_values(metrics: pd.DataFrame, source: str) -> list[Value]:
    """Claims 6, 7, 7b, 7c: each event group's spike evidence against matched
    quiet windows (``baseline``). Event-level medians, pooled over epochs; the
    run scores at most 30 events per group per epoch."""
    make = _value_factory(source, ("6", "7", "7c"), ("paper", "figure: spike support"))
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
        values += [
            make(
                f"{word}_scored",
                len(rows),
                "count",
                "count",
                f"{label} scored (at most 30 per epoch)",
            ),
            make(
                f"{word}_spikes_median",
                rows["n_spikes"].median(),
                "quantity",
                "int",
                f"median spike count of {label}",
                "spikes",
                claims_=("6",),
            ),
            make(
                f"{word}_tetrodes_median",
                rows["n_tetrodes"].median(),
                "quantity",
                "int",
                f"median number of tetrodes with a spike in {label}",
                "tetrodes",
                claims_=("6",),
            ),
            make(
                f"{word}_dll_per_spike_median",
                rows["dll_per_spike"].median(),
                "quantity",
                "dec2",
                f"median log-likelihood ratio per spike, decoded location against the "
                f"animal's, of {label}",
                "nats per spike",
                claims_=("7", "7c") if word == "baseline" else ("7",),
            ),
            make(
                f"{word}_gpi_distance_median_cm",
                rows["dist_to_gpi_peak"].median(),
                "quantity",
                "int",
                f"median track distance from the MAP of {label} to the ground-process "
                "intensity peak",
                "cm",
                claims_=("7b",),
                used_in_=("supplement",),
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
                claims_=("7",),
            ),
            make(
                f"{word}_epochs_scored",
                len(epoch_med),
                "count",
                "count",
                f"epochs with a scored {group} event",
                claims_=("7",),
            ),
            make(
                f"{word}_animals_above_baseline",
                (animal_med > by_animal_base.reindex(animal_med.index)).sum(),
                "count",
                "count",
                f"animals whose median {group} event scores above their median quiet "
                "window, per spike",
                claims_=("7",),
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
    """Claim 7: the held-out tetrode test of immobile Non-Local Continuous bouts,
    split by SWR, per held-out spike against the animal's position."""
    make = _value_factory(source, ("7",), ("paper", "figure: spike support"))
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
                pending=HELDOUT_PENDING,
            ),
            make(
                f"heldout_{word}_per_spike_ci_low",
                row.animal_ci_lo,
                "quantity",
                "dec2",
                "lower bound of its 95 % t-interval over animals",
                "nats per spike",
                pending=HELDOUT_PENDING,
            ),
            make(
                f"heldout_{word}_per_spike_ci_high",
                row.animal_ci_hi,
                "quantity",
                "dec2",
                "upper bound of its 95 % t-interval over animals",
                "nats per spike",
                pending=HELDOUT_PENDING,
            ),
            make(
                f"heldout_{word}_animals_positive",
                positive,
                "count",
                "count",
                "animals with a positive mean",
                pending=HELDOUT_PENDING,
            ),
            make(
                f"heldout_{word}_animals",
                total,
                "count",
                "count",
                "animals in that test",
                pending=HELDOUT_PENDING,
            ),
        ]
    return values


# --- 2c: properties -----------------------------------------------------------

#: (word, column, scale, unit, style, the claim's direction for non-SWR against
#: SWR, claims, use)
_PROPERTIES = (
    (
        "duration",
        "event_duration_s",
        1000.0,
        "ms",
        "int",
        "lower",
        ("18",),
        ("supplement",),
    ),
    (
        "extent",
        "spatial_extent_cm",
        1.0,
        "cm",
        "sig2",
        "lower",
        ("16",),
        ("paper", "figure: 3"),
    ),
    (
        "speed",
        "replay_speed_median_cm_s",
        1.0,
        "cm/s",
        "sig2",
        "lower",
        ("15",),
        ("paper", "figure: 3c"),
    ),
    (
        "peak",
        "posterior_peak_mean",
        1.0,
        "",
        "dec2",
        "higher",
        ("17",),
        ("supplement",),
    ),
    ("hpd", "hpd_coverage_cm", 1.0, "cm", "sig2", "lower", ("17",), ("supplement",)),
    (
        "pause",
        "pause_fraction",
        1.0,
        "",
        "dec2",
        "higher",
        ("16b",),
        ("paper", "figure: 3"),
    ),
    (
        "max_step",
        "step_distance_max_cm",
        1.0,
        "cm",
        "sig2",
        "lower",
        ("16b",),
        ("paper", "figure: 3"),
    ),
    (
        "reversals",
        "n_reversals",
        1.0,
        "",
        "int",
        "lower",
        ("16b",),
        ("paper", "figure: 3"),
    ),
    (
        "multimodal",
        "multimodal_fraction",
        1.0,
        "",
        "dec2",
        "lower",
        ("17",),
        ("supplement",),
    ),
)


def property_values(events: pd.DataFrame, source: str) -> list[Value]:
    """Claims 15–18 and 16b: per-event properties, non-SWR against SWR.

    Plain medians of all events, and the number of animals whose own medians
    differ in the claim's direction. Not duration-controlled.
    """
    values = []
    for word, column, scale, unit, style, direction, claims, used_in in _PROPERTIES:
        make = _value_factory(source, claims, used_in)
        x = events[[column, "is_swr", "animal"]].dropna(subset=[column])
        x = x.assign(v=x[column] * scale)
        med = x.groupby("is_swr")["v"].median()
        by_animal = x.groupby(["animal", "is_swr"])["v"].median().unstack()
        compare: Callable[[pd.Series, pd.Series], pd.Series] = (
            pd.Series.lt if direction == "lower" else pd.Series.gt
        )
        in_direction = compare(by_animal[False], by_animal[True]).sum()
        name = column.replace("_", " ")
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


# --- 2c: where content sits ---------------------------------------------------


def content_occupancy_values(occupancy: pd.DataFrame, source: str) -> list[Value]:
    """Claim 14: where content sits against the animal and the reward wells.

    Time-bin-weighted shares over all sessions (as the analysis prints them):
    at a remote well, at the animal's own well, on a remote path, near the
    animal off the wells.
    """
    make = _value_factory(
        source, ("14", "13"), ("paper", "figure: 3, content occupancy")
    )
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


def summarize(inputs: Part2Inputs, produced_by: str, commit: str) -> Summary:
    """Part 2's summary from ``inputs``."""
    summary = Summary(part=2, produced_by=produced_by, commit=commit)

    events = pd.read_parquet(inputs.event_table)
    source = f"{rel(inputs.event_table)} (canonical event table)"
    summary.add(*event_cohort_values(events, source))
    summary.add(*boundary_offset_values(events, source))
    summary.add(*property_values(events, source))
    summary.inputs.append(file_record(inputs.event_table, ROOT_DIR))

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
    summary.inputs += [
        files_record(
            "*_state_shares.parquet", run.glob("*_state_shares.parquet"), ROOT_DIR
        ),
        files_record(
            "*_pair_membership.parquet", run.glob("*_pair_membership.parquet"), ROOT_DIR
        ),
        file_record(directional_path, ROOT_DIR),
    ]

    metrics = pd.read_parquet(inputs.reality_metrics)
    summary.add(
        *spike_evidence_values(
            metrics, f"{rel(inputs.reality_metrics)} (242-epoch run)"
        )
    )
    summary.inputs.append(file_record(inputs.reality_metrics, ROOT_DIR))

    stats = pd.read_csv(inputs.heldout_by_swr)
    summary.add(*heldout_by_swr_values(stats, rel(inputs.heldout_by_swr)))
    summary.inputs.append(file_record(inputs.heldout_by_swr, ROOT_DIR))

    occupancy = pd.read_csv(inputs.content_occupancy)
    summary.add(*content_occupancy_values(occupancy, rel(inputs.content_occupancy)))
    summary.inputs.append(file_record(inputs.content_occupancy, ROOT_DIR))
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
