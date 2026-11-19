"""The other detectors' events against the non-local detector's.

Read from an event features run (``paths.event_features_root()``), for the
epochs its manifest marks ``ok`` and no others (``runs.io.ok_epochs``):
every inventory's compared events (``*_features.parquet``), each event's
membership against every other inventory (``*_pair_membership.parquet``) and
its state shares (``*_state_shares.parquet``, from
``scripts/detector_comparison/add_state_shares.py``).
The non-local detector's inventory is ``state_space``. An event's membership
against another inventory is known (``matched`` or ``unmatched``) only when the
event lies wholly inside the pair's shared exposure.

A row of an event is in a state when that state's probability is above 0.8, so
an event's state shares are the shares of its rows in each state, and in none
(:data:`~continuum_swr_replay.analysis.detector_comparison.features.STATE_SHARE_COLUMNS`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay import intervals
from continuum_swr_replay.analysis.detector_comparison.features import (
    STATE_SHARE_COLUMNS,
)
from continuum_swr_replay.analysis.detector_comparison.inventory_run import (
    SWR_DEFINITIONS,
    TABLE_DTYPES,
    TRIGGERED,
)
from continuum_swr_replay.analysis.detector_comparison.pooling import (
    cells_by_animal,
    counts_by_animal,
    session_tables,
)
from continuum_swr_replay.analysis.detector_comparison.schema import (
    MEMORYLESS,
    STATE_SPACE,
    typed_table,
)
from continuum_swr_replay.analysis.shared_statistics import animal_session_values
from continuum_swr_replay.paths import inventory_comparison_root
from continuum_swr_replay.runs.io import ok_table_paths, read_manifest
from continuum_swr_replay.tables.io import read_table

COVERINGS = {
    "kay": ("kay",),
    "swr": SWR_DEFINITIONS,
    "trigger": TRIGGERED,
    "widloski": (MEMORYLESS,),
}
"""The sets of inventories whose cover of the non-local event time is measured:
Kay's SWRs, the five SWR definitions together, those and MUA together, and the
Widloski detector's events."""

_KNOWN = ["matched", "unmatched"]

INVENTORY_TABLES = ("observations", "counts", "cells")
"""The inventory run's tables :func:`left_out_statistics` pools by animal."""


def _epoch_paths(run_dir: Path, name: str) -> list[Path]:
    """Table ``name`` of each epoch the run's manifest marks ``ok``.

    Raises
    ------
    ValueError
        If the run has no readable manifest, or it marks no epoch ``ok``.
    """
    paths = ok_table_paths(run_dir, name)
    if not paths:
        raise ValueError(f"the manifest of {run_dir} marks no epoch ok")
    return paths


def _read_epochs(run_dir: Path, name: str, **kwargs) -> pd.DataFrame:
    return pd.concat(
        (pd.read_parquet(path, **kwargs) for path in _epoch_paths(run_dir, name)),
        ignore_index=True,
    )


def other_detector_events(run_dir: Path) -> pd.DataFrame:
    """Each other detector's compared events: whether it overlaps a non-local
    event (``matched``), and its four state shares.

    Only events with a known membership are kept, as the comparison's
    ``directional`` table counts them.
    """
    keys = ["animal", "day", "epoch", "detector", "event_key"]
    shares = _read_epochs(run_dir, "state_shares")
    membership = _read_epochs(
        run_dir, "pair_membership", filters=[("other_detector", "==", STATE_SPACE)]
    )
    events = shares.merge(
        membership[[*keys, "membership"]], on=keys, how="inner", validate="1:1"
    )
    events = events[events["membership"].isin(_KNOWN)]
    return events.assign(matched=events["membership"] == "matched")


def nonlocal_event_memberships(
    run_dir: Path, directional: pd.DataFrame
) -> pd.DataFrame:
    """Each non-local event's membership against each other detector: whether
    that detector's events overlap it (``matched``).

    Only the memberships the comparison's ``directional`` table counts are kept:
    known ones (matched or unmatched) within the two detectors' shared exposure.

    Raises
    ------
    ValueError
        Unless the kept memberships give the directional table's counts.
    """
    memberships = _read_epochs(
        run_dir, "pair_membership", filters=[("detector", "==", STATE_SPACE)]
    )
    kept = memberships[
        memberships["membership"].isin(_KNOWN)
        & memberships["within_shared_exposure"].astype(bool)
    ]
    kept = kept.assign(matched=kept["membership"] == "matched")
    _check_counts(
        kept.groupby("other_detector")["matched"].agg(
            n_eligible="size", n_matched="sum"
        ),
        directional[directional["source_detector"] == STATE_SPACE]
        .groupby("target_detector")[["n_eligible", "n_matched"]]
        .sum(),
    )
    return kept


def check_against_directional(events: pd.DataFrame, directional: pd.DataFrame) -> None:
    """Raise unless the per-event memberships give the comparison's own counts."""
    _check_counts(
        events.groupby("detector")["matched"].agg(n_eligible="size", n_matched="sum"),
        directional[directional["target_detector"] == STATE_SPACE]
        .groupby("source_detector")[["n_eligible", "n_matched"]]
        .sum(),
    )


def _check_counts(mine: pd.DataFrame, theirs: pd.DataFrame) -> None:
    """Raise unless per-event counts (``mine``) equal the directional table's."""
    joined = theirs.join(mine, rsuffix="_events", how="outer")
    if not (
        (joined["n_eligible"] == joined["n_eligible_events"]).all()
        and (joined["n_matched"] == joined["n_matched_events"]).all()
    ):
        raise ValueError(f"memberships do not give the directional counts:\n{joined}")


@dataclass(frozen=True)
class Comparison:
    """An event features run's comparison, as the figure and the summary read it.

    Attributes
    ----------
    others : pd.DataFrame
        Each other detector's compared events (:func:`other_detector_events`).
    inventory_dir : Path
        The inventory comparison run the features run read.
    directional_path : Path
        Its ``report/directional_by_animal.csv``.
    directional : pd.DataFrame
        That table, checked against ``others`` (:func:`check_against_directional`).
    """

    others: pd.DataFrame
    inventory_dir: Path
    directional_path: Path
    directional: pd.DataFrame


def comparison_files(run_dir: Path) -> list[Path]:
    """Every file :func:`read_comparison` and :func:`nonlocal_event_memberships`
    read: the run's manifest and per-epoch tables, and the inventory run's
    directional table."""
    inventory_dir = (
        inventory_comparison_root()
        / read_manifest(run_dir)["settings"]["input_runs"]["inventory"]
    )
    return [
        run_dir / "manifest.json",
        *_epoch_paths(run_dir, "state_shares"),
        *_epoch_paths(run_dir, "pair_membership"),
        inventory_dir / "report" / "directional_by_animal.csv",
    ]


def read_comparison(run_dir: Path) -> Comparison:
    """The features run's compared events and its inventory run's directional
    table, checked against each other."""
    others = other_detector_events(run_dir)
    inventory_dir = (
        inventory_comparison_root()
        / read_manifest(run_dir)["settings"]["input_runs"]["inventory"]
    )
    directional_path = inventory_dir / "report" / "directional_by_animal.csv"
    directional = pd.read_csv(directional_path)
    check_against_directional(others, directional)
    return Comparison(others, inventory_dir, directional_path, directional)


def _matched_shares(
    events: pd.DataFrame, by: str, detectors: tuple[str, ...]
) -> pd.DataFrame:
    """Each detector's share of its events matched: pooled over events, and per
    animal as the mean over the animal's epochs of each epoch's share (event ->
    epoch -> animal), as :func:`content_class_animal_shares` takes it."""
    shares = {}
    for detector in detectors:
        mine = events[events[by] == detector]
        shares[detector] = {
            "pooled": mine["matched"].mean(),
            "animals": animal_session_values(
                mine["matched"].astype(float),
                mine["animal"],
                mine[["day", "epoch"]],
                "mean",
            ).to_numpy(),
        }
    return pd.DataFrame.from_dict(shares, orient="index").rename_axis("detector")


def nonlocal_events_overlapped(
    memberships: pd.DataFrame, detectors: tuple[str, ...]
) -> pd.DataFrame:
    """The share of the non-local events each detector's events overlap.

    Parameters
    ----------
    memberships : pd.DataFrame
        :func:`nonlocal_event_memberships`: one row per non-local event and
        other detector, with ``matched`` and the session columns.
    detectors : tuple of str
        The other detectors, in the order returned.

    Returns
    -------
    overlap : pd.DataFrame, shape (n_detectors, 2)
        Indexed by detector: ``pooled``, the matched non-local events over the
        eligible ones, and ``animals``, each animal's mean over its epochs of
        the epoch's share (an array, in the animals' sorted order).
    """
    return _matched_shares(memberships, "other_detector", detectors)


def events_overlapping_nonlocal(
    events: pd.DataFrame, detectors: tuple[str, ...]
) -> pd.DataFrame:
    """The share of each detector's events that overlap a non-local event.

    The other direction of :func:`nonlocal_events_overlapped`: 1 minus this is
    :func:`other_detector_statistics`'s ``no_nonlocal_fraction``, and for each
    animal it is :func:`content_class_animal_shares`' ``with_nonlocal``.

    Parameters
    ----------
    events : pd.DataFrame
        :func:`other_detector_events`.
    detectors : tuple of str
        The detectors, in the order returned.

    Returns
    -------
    overlap : pd.DataFrame, shape (n_detectors, 2)
        Indexed by detector: ``pooled`` and ``animals``, as for
        :func:`nonlocal_events_overlapped`.
    """
    return _matched_shares(events, "detector", detectors)


CONTENT_CLASSES = ("with_nonlocal", "mostly_local", "other")
"""What a detector's event holds: it overlaps a non-local event; it does not and
Local is above 0.8 in more than half its rows; or neither."""


def content_classes(events: pd.DataFrame) -> pd.Series:
    """Each compared event's :data:`CONTENT_CLASSES` class.

    An event that overlaps no non-local event and has no state shares is
    ``"other"``, as :func:`other_detector_statistics` counts it.
    """
    local = STATE_SHARE_COLUMNS[0]
    mostly_local = ~events["matched"] & (events[local] > 0.5)
    return pd.Series(
        np.select(
            [events["matched"].to_numpy(dtype=bool), mostly_local.to_numpy()],
            ["with_nonlocal", "mostly_local"],
            "other",
        ),
        index=events.index,
    )


def content_class_animal_shares(events: pd.DataFrame, detector: str) -> pd.DataFrame:
    """Each animal's share of one detector's events in each content class.

    An animal's share is the mean over its epochs of each epoch's share of the
    detector's events (event -> epoch -> animal), as
    :func:`other_detector_statistics` takes the animals' range.

    Returns
    -------
    pd.DataFrame, shape (n_animals, 3)
        Indexed by animal; one column per :data:`CONTENT_CLASSES`.
    """
    mine = events[events["detector"] == detector]
    classes = content_classes(mine)
    return pd.DataFrame(
        {
            name: animal_session_values(
                (classes == name).astype(float),
                mine["animal"],
                mine[["day", "epoch"]],
                "mean",
            )
            for name in CONTENT_CLASSES
        }
    )


def other_detector_statistics(events: pd.DataFrame) -> pd.DataFrame:
    """For each other detector, how its events sit against the non-local events.

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, n_columns)
        Each other detector's compared events (:func:`other_detector_events`),
        with ``detector``, ``animal``, ``day``, ``epoch``, ``matched`` and the
        four state shares. An event with no state shares is left out of the
        state statistics.

    Returns
    -------
    statistics : pd.DataFrame, shape (n_detectors, 15)
        Indexed by detector. "Mostly" a state is more than half the event's
        rows.

        - ``n_events``: its compared events.
        - ``no_nonlocal_fraction``: the share overlapping no non-local event,
          pooled, and ``no_nonlocal_fraction_animal_min`` and
          ``..._animal_max``, the lowest and highest of the animals' own
          shares, each event -> epoch -> animal: the mean over the animal's
          epochs of each epoch's share.
        - ``mostly_local_fraction``: the events overlapping no non-local event
          that are mostly Local, as a share of all its compared events (an
          event without state shares is not mostly Local).
        - Of the events overlapping no non-local event:
          ``no_nonlocal_mostly_local_fraction``,
          ``no_nonlocal_mostly_no_spike_fraction`` and
          ``no_nonlocal_mixed_fraction`` (neither).
        - Of the events overlapping a non-local event, the median share of
          their rows in each state (``overlap_local_share_median``,
          ``overlap_no_spike_share_median``, ``overlap_nonlocal_share_median``,
          ``overlap_uncertain_share_median``), and the shares of events with
          any Local row (``overlap_any_local_fraction``), any No-Spike row
          (``overlap_any_no_spike_fraction``) and every row non-local
          (``overlap_all_nonlocal_fraction``).
    """
    local, no_spike, non_local, uncertain = STATE_SHARE_COLUMNS
    rows = {}
    for detector, mine in events.groupby("detector"):
        alone = mine[~mine["matched"]].dropna(subset=list(STATE_SHARE_COLUMNS))
        overlapping = mine[mine["matched"]].dropna(subset=list(STATE_SHARE_COLUMNS))
        mostly_local = alone[local] > 0.5
        mostly_no_spike = alone[no_spike] > 0.5
        matched_by_animal = content_class_animal_shares(mine, detector)["with_nonlocal"]
        rows[detector] = {
            "n_events": len(mine),
            "no_nonlocal_fraction": (~mine["matched"]).sum() / len(mine),
            "no_nonlocal_fraction_animal_min": 1 - matched_by_animal.max(),
            "no_nonlocal_fraction_animal_max": 1 - matched_by_animal.min(),
            "mostly_local_fraction": mostly_local.sum() / len(mine),
            "no_nonlocal_mostly_local_fraction": mostly_local.mean(),
            "no_nonlocal_mostly_no_spike_fraction": mostly_no_spike.mean(),
            "no_nonlocal_mixed_fraction": (~mostly_local & ~mostly_no_spike).mean(),
            "overlap_local_share_median": overlapping[local].median(),
            "overlap_no_spike_share_median": overlapping[no_spike].median(),
            "overlap_nonlocal_share_median": overlapping[non_local].median(),
            "overlap_uncertain_share_median": overlapping[uncertain].median(),
            "overlap_any_local_fraction": (overlapping[local] > 0).mean(),
            "overlap_any_no_spike_fraction": (overlapping[no_spike] > 0).mean(),
            "overlap_all_nonlocal_fraction": (overlapping[non_local] == 1).mean(),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("detector")


def _intervals(table: pd.DataFrame, detectors: tuple[str, ...]) -> np.ndarray:
    rows = table[table["detector"].isin(detectors)]
    return rows[["event_start_s", "event_end_s"]].to_numpy(dtype=np.float64)


def left_out_seconds(run_dir: Path) -> dict[str, float]:
    """Seconds of non-local event time outside each other inventory, summed over
    the run's ``ok`` epochs.

    Each epoch's compared events (``*_features.parquet``: every inventory's
    whole-event-immobile events): the non-local detector's event time
    (``nonlocal``), the time Kay's SWRs span (``kay``), and, for each of
    :data:`COVERINGS`, the part of the non-local event time that no event of
    those inventories covers (``outside_<covering>``).

    Raises
    ------
    ValueError
        If the run has no readable manifest, or it marks no epoch ``ok``.
    """
    totals = dict.fromkeys(
        ("nonlocal", "kay", *(f"outside_{name}" for name in COVERINGS)), 0.0
    )
    for path in _epoch_paths(run_dir, "features"):
        table = pd.read_parquet(
            path, columns=["detector", "event_start_s", "event_end_s"]
        )
        nonlocal_ = _intervals(table, (STATE_SPACE,))
        totals["nonlocal"] += intervals.total_duration(nonlocal_)
        totals["kay"] += intervals.total_duration(_intervals(table, ("kay",)))
        for name, detectors in COVERINGS.items():
            outside = intervals.subtract(nonlocal_, _intervals(table, detectors))
            totals[f"outside_{name}"] += intervals.total_duration(outside)
    return totals


def inventory_animal_tables(inventory_dir: Path) -> dict[str, pd.DataFrame]:
    """The inventory run's counts and cells pooled by animal, from its tables.

    The run's ``observations``, ``counts`` and ``cells`` tables of the epochs its
    manifest marks ``ok`` (:data:`INVENTORY_TABLES`), pooled as the comparison's
    report pools them (``pooling.session_tables``, ``pooling.counts_by_animal``,
    ``pooling.cells_by_animal``).

    Returns
    -------
    dict of str to DataFrame
        ``counts_by_animal`` and ``cells_by_animal``.

    Raises
    ------
    ValueError
        If the run has no readable manifest, or it marks no epoch ``ok``.
    """
    manifest = read_manifest(inventory_dir)
    tables = {
        name: typed_table(
            map(read_table, _epoch_paths(inventory_dir, name)), TABLE_DTYPES[name]
        )
        for name in INVENTORY_TABLES
    }
    by_session = session_tables(manifest, tables)
    return {
        "counts_by_animal": counts_by_animal(by_session["counts"]),
        "cells_by_animal": cells_by_animal(by_session["cells"]),
    }


def left_out_statistics(run_dir: Path, inventory_dir: Path) -> dict[str, float]:
    """What an analysis that starts from another inventory leaves out of the
    non-local events, in events, in event time and per hour of rest.

    Parameters
    ----------
    run_dir : Path
        The event features run.
    inventory_dir : Path
        The inventory comparison run the event features run read; its
        exposures and cells are pooled from its tables
        (:func:`inventory_animal_tables`).

    Returns
    -------
    statistics : dict
        - ``rest_hours``: hours of analysed immobility the non-local detector
          observed (its exposure), summed over epochs.
        - ``nonlocal_time_s``, ``kay_time_s``: seconds the non-local events, and
          Kay's SWRs, span (:func:`left_out_seconds`).
        - For each of :data:`COVERINGS`, ``nonlocal_time_outside_<covering>_s``
          and ``..._fraction``: the non-local event time it does not cover, in
          seconds and as a share.
        - For ``kay`` and ``widloski``, and for ``trigger`` (no SWR definition
          and no MUA event; the comparison's cell
          ``sequence_matched_by_no_trigger``): ``nonlocal_events_outside_<name>``,
          the non-local events none of its events overlaps; ``..._fraction``,
          their share of the non-local events with a known membership; and
          ``..._per_hour``, per hour of ``rest_hours``.
    """
    pooled = inventory_animal_tables(inventory_dir)
    counts = pooled["counts_by_animal"]
    exposure_s = float(
        counts.loc[counts["detector"] == STATE_SPACE, "exposure_s"].sum()
    )
    hours = exposure_s / 3600.0
    seconds = left_out_seconds(run_dir)
    statistics = {
        "rest_hours": hours,
        "nonlocal_time_s": seconds["nonlocal"],
        "kay_time_s": seconds["kay"],
    }
    for name in COVERINGS:
        outside = seconds[f"outside_{name}"]
        statistics[f"nonlocal_time_outside_{name}_s"] = outside
        statistics[f"nonlocal_time_outside_{name}_fraction"] = (
            outside / seconds["nonlocal"]
        )

    membership = _read_epochs(
        run_dir, "pair_membership", filters=[("detector", "==", STATE_SPACE)]
    )
    known = membership[membership["membership"].isin(_KNOWN)]
    alone = (
        known.assign(alone=known["membership"] == "unmatched")
        .groupby("other_detector")["alone"]
        .agg(["sum", "size"])
    )
    cells = pooled["cells_by_animal"]
    no_trigger = cells[
        (cells["cell"] == "sequence_matched_by_no_trigger")
        & (cells["detector"] == STATE_SPACE)
    ][["n_events", "n_unmatched", "n_unknown"]].sum()
    outside_events = {
        "kay": (int(alone.loc["kay", "sum"]), int(alone.loc["kay", "size"])),
        "widloski": (
            int(alone.loc[MEMORYLESS, "sum"]),
            int(alone.loc[MEMORYLESS, "size"]),
        ),
        "trigger": (
            int(no_trigger["n_unmatched"]),
            int(no_trigger["n_events"] - no_trigger["n_unknown"]),
        ),
    }
    for name, (n, n_known) in outside_events.items():
        statistics[f"nonlocal_events_outside_{name}"] = n
        statistics[f"nonlocal_events_outside_{name}_fraction"] = n / n_known
        statistics[f"nonlocal_events_outside_{name}_per_hour"] = n / hours
    return statistics
