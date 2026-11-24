"""The events summary's content values (question 2d.1): where the events'
content sits along the arms, as posterior mass, and the readers of the content
that the figures share. The statistics are computed in
``analysis.event_properties.content_along_arms``."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from continuum_swr_replay.analysis.event_definition.cohort_selection import (
    select_events_in,
)
from continuum_swr_replay.analysis.event_properties import (
    content_along_arms,
    event_statistics,
)
from continuum_swr_replay.paths import ROOT_DIR
from continuum_swr_replay.reported.events.common import (
    EVENTS_SUBSET,
    TAGS,
    SummaryReads,
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


def content_occupancy_values(bins: pd.DataFrame, source: str) -> list[Value]:
    """Question 2d.1: where content sits against the animal and the wells.

    ``bins``: the content time of each kind at each place of each epoch (the
    events' position distributions summed over their time bins), with the
    place's track distance to the animal and to the nearest well (a track-graph
    end point) and its arm with the arm's length, as
    :func:`content_mass_along_arms` reads them. The shares at a remote well, at
    the animal's own well, on a remote path and near the animal off the wells
    (``content_along_arms.content_place_shares``): each epoch's share of its
    content time, averaged over the animal's epochs, then over animals. And the share at a well of content spread evenly over the track
    (``content_along_arms.even_content``), which those shares are read against.
    The shares at a well, at a remote well and at the animal's own each have a
    95 % interval, and so does the no-SWR minus SWR difference of each
    (``content_along_arms.place_share_intervals``).
    """
    make = value_factory(content_occupancy_values, source, TAGS["content_occupancy"])
    # The places that depend on where the animal is: the supplement's panel c
    # draws the remote well and the animal's own; the other two are in no panel.
    make_by_animal = value_factory(
        content_occupancy_values, source, TAGS["content_by_animal"]
    )
    make_well_side = value_factory(
        content_occupancy_values, source, TAGS["content_well_side"]
    )
    makers = {
        "at_well": make,
        "remote_well": make_well_side,
        "own_well": make_well_side,
    }
    statistics = content_along_arms.place_share_statistics(
        content_along_arms.content_place_shares(bins)
    )
    values = [
        make(
            "occupancy_epochs",
            bins["epoch_id"].nunique(),
            "count",
            "count",
            "epochs whose events place content on the track",
        ),
    ]
    for word, is_swr, name in (("non_swr", False, "non-SWR"), ("swr", True, "SWR")):
        for column in event_statistics.OCCUPANCY_PLACES:
            values.append(
                makers.get(column, make_by_animal)(
                    f"{word}_content_{column}_fraction",
                    statistics["means"].loc[is_swr, column],
                    "fraction",
                    "pct",
                    f"share of {name} non-local content time "
                    f"{column.replace('_', ' ')} (content time is the events' "
                    "position distributions, the posterior of the two non-local "
                    "states over the place bins, summed over their time bins; wells "
                    "are the track graph's end points; each epoch's share, averaged "
                    "over the animal's epochs, then over animals)",
                )
            )
    # the headline shares' 95 % intervals, and the no-SWR minus SWR differences
    intervals = content_along_arms.place_share_intervals(bins)
    names = {"non_swr": "non-SWR", "swr": "SWR"}
    how = (
        "nested bootstrap of the animals and their epochs, small-sample t "
        "correction over the animals"
    )
    for place in content_along_arms.INTERVAL_PLACES:
        made = makers[place]
        where = place.replace("_", " ")
        for word, name in names.items():
            row = intervals.loc[(word, place)]
            for bound in ("low", "high"):
                values.append(
                    made(
                        f"{word}_content_{place}_fraction_ci_{bound}",
                        row[f"ci_{bound}"],
                        "fraction",
                        "pct",
                        f"{'lower' if bound == 'low' else 'upper'} bound of the 95 % "
                        f"interval of the share of {name} non-local content time "
                        f"{where} ({how})",
                    )
                )
        row = intervals.loc[("difference", place)]
        text = (
            f"non-SWR minus SWR share of non-local content time {where}, in "
            "percentage points: each epoch's difference of its two shares, "
            "averaged over the animal's epochs, then over animals"
        )
        values.append(
            made(
                f"content_{place}_fraction_difference",
                row["estimate"],
                "fraction",
                "pct",
                text,
            )
        )
        for bound in ("low", "high"):
            values.append(
                made(
                    f"content_{place}_fraction_difference_ci_{bound}",
                    row[f"ci_{bound}"],
                    "fraction",
                    "pct",
                    f"{'lower' if bound == 'low' else 'upper'} bound of the 95 % "
                    f"interval of that difference {where} ({how})",
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


_MASS_PLACE = [
    "arm",
    "arm_length_cm",
    "place_bin",
    "n_place_bins",
    "distance_to_well_cm",
    "far",
]
"""What names a place, and its side of the animal, in the content-mass run."""


def content_mass_along_arms(
    events: pd.DataFrame, folder: Path, n_epochs: int
) -> tuple[pd.DataFrame, list[Path]]:
    """The content time of the events of ``events`` at each place, per epoch and
    kind, from the content-mass run; and the files read.

    The run holds every event of the table; the rows of the events in
    ``events`` are read (``cohort_selection.select_events_in``) and summed per
    epoch, kind (``is_swr``), place and side of the animal.

    Returns
    -------
    bins : pd.DataFrame
        One row per epoch, kind, place and side: ``epoch_id``, ``animal``,
        ``is_swr``, the place (``arm``, ``arm_length_cm``, ``place_bin``,
        ``n_place_bins``, ``distance_to_well_cm``), ``far``, ``weight`` (the
        content time, in time bins) and ``distance_to_animal_cm`` (its
        weighted mean), as ``content_along_arms``'s statistics take a weighted
        table.
    paths : list of Path

    Raises
    ------
    ValueError
        If ``folder`` does not hold ``n_epochs`` epochs, or its rows do not
        cover every event of ``events``.
    """
    paths = sorted(folder.glob("*.parquet"))
    if len(paths) != n_epochs:
        raise ValueError(f"{folder} holds {len(paths)} epochs, not {n_epochs}")
    kinds = events[["event_id", "animal", "is_swr"]]
    tables, n_events = [], 0
    for path in paths:
        rows = select_events_in(
            pd.read_parquet(
                path,
                columns=[
                    "event_id",
                    "epoch_id",
                    *_MASS_PLACE,
                    "weight",
                    "distance_to_animal_cm",
                ],
            ),
            events["event_id"],
        ).merge(kinds, on="event_id", validate="m:1")
        n_events += rows["event_id"].nunique()
        rows["moment"] = rows["weight"].astype(float) * rows["distance_to_animal_cm"]
        rows["weight"] = rows["weight"].astype(float)
        summed = rows.groupby(
            ["epoch_id", "animal", "is_swr", *_MASS_PLACE], as_index=False
        )[["weight", "moment"]].sum()
        summed["distance_to_animal_cm"] = summed.pop("moment") / summed["weight"]
        tables.append(summed)
    if n_events != len(events):
        raise ValueError(f"the rows of {folder} do not cover the event table's events")
    return pd.concat(tables, ignore_index=True), paths


def content_mass_by_patch(
    events: pd.DataFrame, folder: Path, n_epochs: int
) -> tuple[pd.DataFrame, list[Path]]:
    """The content time of the events of ``events`` on each arm, at the wells
    and off them, with the patch the animal is in, per epoch and kind, from the
    content-mass run; and the files read
    (``content_along_arms.patch_content_shares`` takes the rows).

    Raises
    ------
    ValueError
        If ``folder`` does not hold ``n_epochs`` epochs, or an event of
        ``events`` has no patch (``current_patch``).
    """
    paths = sorted(folder.glob("*.parquet"))
    if len(paths) != n_epochs:
        raise ValueError(f"{folder} holds {len(paths)} epochs, not {n_epochs}")
    if events["current_patch"].isna().any():
        raise ValueError("an event has no current patch")
    kinds = events[["event_id", "animal", "is_swr", "current_patch"]]
    place = ["arm", "distance_to_well_cm"]
    tables = []
    for path in paths:
        rows = select_events_in(
            pd.read_parquet(path, columns=["event_id", "epoch_id", *place, "weight"]),
            events["event_id"],
        ).merge(kinds, on="event_id", validate="m:1")
        rows["weight"] = rows["weight"].astype(float)
        tables.append(
            rows.groupby(
                ["epoch_id", "animal", "is_swr", "current_patch", *place],
                as_index=False,
            )["weight"].sum()
        )
    return pd.concat(tables, ignore_index=True), paths


def with_event_start_end(
    events: pd.DataFrame, folder: Path, n_epochs: int
) -> tuple[pd.DataFrame, list[Path]]:
    """``events`` with where each one's content starts and ends
    (``content_along_arms.event_start_end`` of the content-along-arms run, the
    decode's most probable place at each time bin); and the files read."""
    bins, paths = content_along_arms_bins(
        events,
        folder,
        n_epochs,
        ["time", "distance_to_animal_cm", "distance_to_well_cm"],
    )
    places = content_along_arms.event_start_end(bins)
    return events.join(places, on="event_id"), paths


def with_event_place_shares(
    events: pd.DataFrame, folder: Path, n_epochs: int
) -> tuple[pd.DataFrame, list[Path]]:
    """``events`` with each one's share of content time at a well, at a remote
    well and at the animal's own (``content_at_well``, ``content_remote_well``
    and ``content_own_well``: ``content_along_arms.event_place_shares`` of the
    content-mass run; the event table's own ``at_well`` is where the animal is),
    the number of wells it represents and of patches it spans
    (``content_n_wells``, ``content_n_patches``:
    ``content_along_arms.event_well_counts`` and ``event_patch_counts``) and
    its ``epoch_id``; and the files read.

    Raises
    ------
    ValueError
        If ``folder`` does not hold ``n_epochs`` epochs, or an event has no row.
    """
    paths = sorted(folder.glob("*.parquet"))
    if len(paths) != n_epochs:
        raise ValueError(f"{folder} holds {len(paths)} epochs, not {n_epochs}")
    tables = []
    for path in paths:
        rows = select_events_in(
            pd.read_parquet(
                path,
                columns=[
                    "event_id",
                    "epoch_id",
                    "arm",
                    "far",
                    "weight",
                    "distance_to_well_cm",
                ],
            ),
            events["event_id"],
        )
        shares = content_along_arms.event_place_shares(rows).add_prefix("content_")
        shares["content_n_wells"] = content_along_arms.event_well_counts(rows)
        shares["content_n_patches"] = content_along_arms.event_patch_counts(rows)
        shares["epoch_id"] = path.stem
        tables.append(shares.reset_index())
    joined = events.merge(
        pd.concat(tables, ignore_index=True), on="event_id", how="left", validate="1:1"
    )
    if joined["content_at_well"].isna().any():
        raise ValueError("an event has no row in the content-mass run")
    return joined, paths


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


def add_content(reads: SummaryReads) -> None:
    """Add where the events' content sits along the arms."""
    summary = reads.summary
    inputs = reads.inputs
    events = reads.events
    n_epochs = reads.n_epochs
    stale = reads.stale
    along_bins, along_paths = content_mass_along_arms(
        events, inputs.content_mass_along_arms, n_epochs
    )
    summary.add(
        *over(
            pend(
                content_occupancy_values(
                    along_bins,
                    f"{rel(inputs.content_mass_along_arms)} "
                    "(scripts/event_properties/measure_content_mass_along_arms.py)",
                ),
                stale(inputs.content_mass_along_arms),
            ),
            EVENTS_SUBSET,
        )
    )
    summary.inputs += [
        files_record("*.parquet", along_paths, ROOT_DIR),
        file_record(inputs.content_mass_along_arms / "manifest.json", ROOT_DIR),
    ]
