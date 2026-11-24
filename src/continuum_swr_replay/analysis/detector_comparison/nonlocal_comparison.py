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

The comparison is over the events that start inside a well visit (from the
first poke-in at a well to the final poke-out): a non-local event when the event
table's selection keeps it (``cohort_selection.select_visit_context``, passed as
its ``event_id``), another detector's event (an SWR, a MUA event, a Widloski
event) when its first sample is inside a visit (:func:`visit_starts`, from each
epoch's visits; ``event_definition.well_visits.in_well_visit``). The runs'
counts are checked against the inventory run's directional table over every
event first, and the selection is made after.

A row of an event is in a state when that state's probability is above 0.8, so
an event's state shares are the shares of its rows in each state, and in none
(:data:`~continuum_swr_replay.analysis.detector_comparison.features.STATE_SHARE_COLUMNS`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay import intervals
from continuum_swr_replay.analysis.detector_comparison.availability import (
    still_intervals,
    stored_coverage,
)
from continuum_swr_replay.analysis.detector_comparison.features import (
    STATE_SHARE_COLUMNS,
)
from continuum_swr_replay.analysis.detector_comparison.inventory_run import (
    MAX_GAP_S,
    SWR_DEFINITIONS,
    TABLE_DTYPES,
    TRIGGERED,
)
from continuum_swr_replay.analysis.detector_comparison.pooling import (
    detector_availability,
)
from continuum_swr_replay.analysis.detector_comparison.schema import (
    MEMORYLESS,
    STATE_SPACE,
    native_event_keys,
    typed_table,
)
from continuum_swr_replay.analysis.event_definition.well_visits import in_well_visit
from continuum_swr_replay.analysis.shared_statistics import animal_session_values
from continuum_swr_replay.parameters import DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC
from continuum_swr_replay.paths import inventory_comparison_root
from continuum_swr_replay.runs.io import ok_epochs, ok_table_paths, read_manifest
from continuum_swr_replay.tables.io import read_table
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS_DAY

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

INVENTORY_TABLES = ("observations", "cells")
"""The inventory run's tables :func:`left_out_statistics` reads."""


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


_EVENT_KEYS = ["animal", "day", "epoch", "detector", "event_key"]

#: An epoch's well visits: each one's first poke-in and final poke-out (s).
Visits = tuple[np.ndarray, np.ndarray]


def visit_starts(run_dir: Path, visits: Mapping[str, Visits]) -> pd.DataFrame:
    """Whether each compared event of the run starts inside a well visit.

    Parameters
    ----------
    run_dir : Path
        The event features run; its ``*_features.parquet`` of the epochs its
        manifest marks ``ok`` give each event's ``event_start_s``.
    visits : Mapping
        Each ok epoch's id to its visits' ``(arrival_times, departure_times)``.

    Returns
    -------
    pd.DataFrame
        One row per compared event: the event keys and
        ``starts_in_well_visit`` (its first sample is inside a visit).

    Raises
    ------
    ValueError
        If an ok epoch has no visits in ``visits``.
    """
    identifiers = ok_epochs(read_manifest(run_dir))
    missing = sorted(set(identifiers) - set(visits))
    if missing:
        raise ValueError(
            f"no well visits for {len(missing)} epochs, e.g. {missing[:3]}"
        )
    tables = []
    for identifier, path in zip(
        identifiers, _epoch_paths(run_dir, "features"), strict=True
    ):
        table = pd.read_parquet(path, columns=[*_EVENT_KEYS, "event_start_s"])
        starts = in_well_visit(
            table["event_start_s"].to_numpy(dtype=np.float64), *visits[identifier]
        )
        tables.append(table[_EVENT_KEYS].assign(starts_in_well_visit=starts))
    return pd.concat(tables, ignore_index=True)


def other_detector_events(run_dir: Path) -> pd.DataFrame:
    """Each other detector's compared events: whether it overlaps a non-local
    event (``matched``), and its four state shares.

    Only events with a known membership are kept, as the comparison's
    ``directional`` table counts them. Every such event, in a visit or not
    (:func:`read_comparison` selects).
    """
    keys = _EVENT_KEYS
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
    run_dir: Path, directional: pd.DataFrame, event_ids: Iterable[object]
) -> pd.DataFrame:
    """Each selected non-local event's membership against each other detector:
    whether that detector's events overlap it (``matched``).

    The memberships the comparison's ``directional`` table counts (known ones,
    matched or unmatched, within the two detectors' shared exposure) are
    checked against it, and then those of the non-local events in
    ``event_ids`` are kept.

    Parameters
    ----------
    run_dir : Path
        The event features run.
    directional : pd.DataFrame
        The inventory run's directional table.
    event_ids : iterable
        The event table's ``event_id`` of the selected non-local events (a
        non-local event's native key in the run is its ``event_id``).

    Raises
    ------
    ValueError
        Unless the memberships give the directional table's counts.
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
    selected = native_event_keys(kept["event_key"]).isin(
        {str(event_id) for event_id in event_ids}
    )
    return kept.loc[selected.to_numpy()].reset_index(drop=True)


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
        Each other detector's compared events (:func:`other_detector_events`)
        that start inside a well visit.
    visit_counts : pd.DataFrame
        Indexed by detector: ``n_in_visit``, its compared events that start
        inside a well visit (the rows of ``others``), and ``n_outside_visit``,
        those left out.
    inventory_dir : Path
        The inventory comparison run the features run read.
    directional_path : Path
        Its ``report/directional_by_animal.csv``.
    directional : pd.DataFrame
        That table, checked against ``others`` (:func:`check_against_directional`).
    """

    others: pd.DataFrame
    visit_counts: pd.DataFrame
    inventory_dir: Path
    directional_path: Path
    directional: pd.DataFrame


def comparison_files(run_dir: Path) -> list[Path]:
    """Every file of the run :func:`read_comparison` and
    :func:`nonlocal_event_memberships` read: the run's manifest and per-epoch
    tables, and the inventory run's directional table (the well visits are the
    caller's)."""
    inventory_dir = (
        inventory_comparison_root()
        / read_manifest(run_dir)["settings"]["input_runs"]["inventory"]
    )
    return [
        run_dir / "manifest.json",
        *_epoch_paths(run_dir, "features"),
        *_epoch_paths(run_dir, "state_shares"),
        *_epoch_paths(run_dir, "pair_membership"),
        inventory_dir / "report" / "directional_by_animal.csv",
    ]


def read_comparison(run_dir: Path, visits: Mapping[str, Visits]) -> Comparison:
    """The features run's compared events that start inside a well visit, and
    its inventory run's directional table.

    Every compared event is checked against the directional table first; then
    the events whose first sample is inside a visit are kept
    (:func:`visit_starts`).

    Parameters
    ----------
    run_dir : Path
        The event features run.
    visits : Mapping
        Each ok epoch's id to its visits' ``(arrival_times, departure_times)``.
    """
    others = other_detector_events(run_dir)
    inventory_dir = (
        inventory_comparison_root()
        / read_manifest(run_dir)["settings"]["input_runs"]["inventory"]
    )
    directional_path = inventory_dir / "report" / "directional_by_animal.csv"
    directional = pd.read_csv(directional_path)
    check_against_directional(others, directional)
    others = others.merge(
        visit_starts(run_dir, visits), on=_EVENT_KEYS, how="left", validate="1:1"
    )
    if others["starts_in_well_visit"].isna().any():
        raise ValueError("a compared event has no row in the run's features tables")
    inside = others["starts_in_well_visit"].astype(bool)
    visit_counts = pd.DataFrame(
        {
            "n_in_visit": inside.groupby(others["detector"]).sum(),
            "n_outside_visit": (~inside).groupby(others["detector"]).sum(),
        }
    ).astype(int)
    return Comparison(
        others.loc[inside.to_numpy()].reset_index(drop=True),
        visit_counts,
        inventory_dir,
        directional_path,
        directional,
    )


FIGURE_DETECTORS = ("kay", "karlsson", "yu", "zugaro", "carey", "mua", "memoryless")
"""The other detectors Figure 2e compares with the non-local detector, top to
bottom: the SWR definitions (Kay's, the SWR source, first), MUA and the
Widloski detector."""


def _matched_shares(
    events: pd.DataFrame, by: str, detectors: tuple[str, ...]
) -> pd.DataFrame:
    """Each detector's share of its events matched: per animal the mean over
    the animal's epochs of each epoch's share (event -> epoch -> animal), as
    :func:`content_class_animal_shares` takes it, and the mean over animals of
    the animals' shares."""
    shares = {}
    for detector in detectors:
        mine = events[events[by] == detector]
        animals = animal_session_values(
            mine["matched"].astype(float),
            mine["animal"],
            mine[["day", "epoch"]],
            "mean",
        )
        shares[detector] = {
            "animal_mean": animals.mean(),
            "animals": animals.to_numpy(),
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
        Indexed by detector: ``animals``, each animal's mean over its epochs
        of the epoch's share of the eligible non-local events it matches (an
        array, in the animals' sorted order); and ``animal_mean``, the mean
        over animals of those shares (Figure 2e's bars).
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
        Indexed by detector: ``animals`` and ``animal_mean``, as for
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


def content_class_means(statistics: pd.Series) -> pd.Series:
    """The mean over animals of the animals' shares of a detector's events in
    each content class, from its row of :func:`other_detector_statistics`:
    with non-local activity, one minus ``no_nonlocal_fraction_animal_mean``;
    mostly Local, ``mostly_local_fraction_animal_mean``; and the other events,
    the rest of those with no non-local activity.

    Parameters
    ----------
    statistics : pd.Series
        One detector's row of :func:`other_detector_statistics`.

    Returns
    -------
    pd.Series, shape (3,)
        Indexed by :data:`CONTENT_CLASSES`.
    """
    no_nonlocal = statistics["no_nonlocal_fraction_animal_mean"]
    mostly_local = statistics["mostly_local_fraction_animal_mean"]
    return pd.Series(
        {
            "with_nonlocal": 1 - no_nonlocal,
            "mostly_local": mostly_local,
            "other": no_nonlocal - mostly_local,
        }
    )


def other_detector_statistics(events: pd.DataFrame) -> pd.DataFrame:
    """For each other detector, how its events sit against the non-local events.

    Every statistic is event -> epoch -> animal -> mean over animals: each
    epoch's share (or median) over its events of the subset the statistic is
    about, each animal's mean (a share) or median (a per-event quantity) over
    its epochs with an event of that subset, then the mean over the animals
    with such an epoch. An epoch with no event of the subset is not one of its
    animal's epochs for that statistic, and an animal with none is left out of
    its mean over animals; a subset can hold few events, and a share is then
    a mean over few epochs' shares, each of few events.

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, n_columns)
        Each other detector's compared events (:func:`other_detector_events`),
        with ``detector``, ``animal``, ``day``, ``epoch``, ``matched`` and the
        four state shares. An event with no state shares is left out of the
        state statistics.

    Returns
    -------
    statistics : pd.DataFrame, shape (n_detectors, 19)
        Indexed by detector. "Mostly" a state is more than half the event's
        rows.

        - ``n_events``: its compared events.
        - ``no_nonlocal_fraction``: the share overlapping no non-local event;
          ``no_nonlocal_fraction_animal_mean``, the same number;
          ``no_nonlocal_fraction_animal_min`` and ``..._animal_max``, the
          lowest and highest of the animals' own shares (the mean over the
          animal's epochs of each epoch's share).
        - ``mostly_local_fraction``: the events overlapping no non-local event
          that are mostly Local, as a share of all its compared events (an
          event without state shares is not mostly Local), and
          ``mostly_local_fraction_animal_mean``, the same number
          (:func:`content_class_animal_shares`).
        - Of the events overlapping no non-local event (with state shares):
          ``no_nonlocal_mostly_local_fraction``,
          ``no_nonlocal_mostly_no_spike_fraction`` and
          ``no_nonlocal_mixed_fraction`` (neither); ``no_nonlocal_n_animals``,
          the animals with such an event.
        - Of the events overlapping a non-local event (with state shares), the
          median share of their rows in each state (``overlap_local_share_median``,
          ``overlap_no_spike_share_median``, ``overlap_nonlocal_share_median``,
          ``overlap_uncertain_share_median``: each epoch's median, each
          animal's median over its epochs, the mean over animals), and the
          shares of events with any Local row (``overlap_any_local_fraction``),
          any No-Spike row (``overlap_any_no_spike_fraction``) and every row
          non-local (``overlap_all_nonlocal_fraction``);
          ``overlap_n_animals``, the animals with such an event.
    """
    local, no_spike, non_local, uncertain = STATE_SHARE_COLUMNS
    session = ["day", "epoch"]

    def by_animal(values: pd.Series, rows: pd.DataFrame, statistic: str) -> float:
        return float(
            animal_session_values(
                values.astype(float), rows["animal"], rows[session], statistic
            ).mean()
        )

    rows = {}
    for detector, mine in events.groupby("detector"):
        alone = mine[~mine["matched"]].dropna(subset=list(STATE_SHARE_COLUMNS))
        overlapping = mine[mine["matched"]].dropna(subset=list(STATE_SHARE_COLUMNS))
        mostly_local = alone[local] > 0.5
        mostly_no_spike = alone[no_spike] > 0.5
        shares = content_class_animal_shares(mine, detector)
        matched_by_animal = shares["with_nonlocal"]
        rows[detector] = {
            "n_events": len(mine),
            "no_nonlocal_fraction": 1 - matched_by_animal.mean(),
            "no_nonlocal_fraction_animal_min": 1 - matched_by_animal.max(),
            "no_nonlocal_fraction_animal_max": 1 - matched_by_animal.min(),
            "no_nonlocal_fraction_animal_mean": 1 - matched_by_animal.mean(),
            "mostly_local_fraction": shares["mostly_local"].mean(),
            "mostly_local_fraction_animal_mean": shares["mostly_local"].mean(),
            "no_nonlocal_mostly_local_fraction": by_animal(mostly_local, alone, "mean"),
            "no_nonlocal_mostly_no_spike_fraction": by_animal(
                mostly_no_spike, alone, "mean"
            ),
            "no_nonlocal_mixed_fraction": by_animal(
                ~mostly_local & ~mostly_no_spike, alone, "mean"
            ),
            "no_nonlocal_n_animals": alone["animal"].nunique(),
            **{
                f"overlap_{name}_share_median": by_animal(
                    overlapping[column], overlapping, "median"
                )
                for name, column in (
                    ("local", local),
                    ("no_spike", no_spike),
                    ("nonlocal", non_local),
                    ("uncertain", uncertain),
                )
            },
            "overlap_any_local_fraction": by_animal(
                overlapping[local] > 0, overlapping, "mean"
            ),
            "overlap_any_no_spike_fraction": by_animal(
                overlapping[no_spike] > 0, overlapping, "mean"
            ),
            "overlap_all_nonlocal_fraction": by_animal(
                overlapping[non_local] == 1, overlapping, "mean"
            ),
            "overlap_n_animals": overlapping["animal"].nunique(),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("detector")


def _intervals(table: pd.DataFrame, detectors: tuple[str, ...]) -> np.ndarray:
    rows = table[table["detector"].isin(detectors)]
    return rows[["event_start_s", "event_end_s"]].to_numpy(dtype=np.float64)


def exposure_in_well_visits(
    observations: pd.DataFrame,
    set_aside: pd.DataFrame,
    time: NDArray,
    speed: NDArray,
    arrival_times: NDArray,
    departure_times: NDArray,
) -> tuple[float, float]:
    """One epoch's non-local detector exposure, and its part inside the well visits.

    The exposure is the inventory comparison's: the time the detector observed,
    less excluded time and the spans of the events set aside, inside the
    epoch's still intervals (``availability.stored_coverage``, rebuilt from the
    tables the comparison stored and the decode frame's speed trace). The
    part inside the visits is its intersection with ``[arrival, departure)``.

    Parameters
    ----------
    observations, set_aside : pd.DataFrame
        The inventory run's stored ``observations`` and ``set_aside`` tables of
        the epoch.
    time, speed : NDArray, shape (n_time,)
        The decode frame's grid (s) and head speed (cm/s)
        (``inventory_run.speed_trace``).
    arrival_times, departure_times : NDArray, shape (n_visits,)
        The epoch's well visits (``event_tables.prepare_data.well_visit_times``).

    Returns
    -------
    exposure_s, exposure_in_visit_s : float
        Seconds.
    """
    session = tuple(observations[["animal", "day", "epoch"]].iloc[0])
    session = (str(session[0]), int(session[1]), str(session[2]))
    still = {
        session: still_intervals(
            time, speed, DEFAULT_IMMOBILITY_THRESHOLD_CM_PER_SEC, MAX_GAP_S
        )
    }
    coverage = stored_coverage(observations, set_aside, still)
    exposure = coverage[(*session, STATE_SPACE)].exposure
    visits = np.column_stack(
        [
            np.asarray(arrival_times, dtype=np.float64),
            np.asarray(departure_times, dtype=np.float64),
        ]
    ).reshape(-1, 2)
    return (
        intervals.total_duration(exposure),
        intervals.total_duration(intervals.intersect(exposure, visits)),
    )


def _selected(table: pd.DataFrame, event_ids: Iterable[object]) -> np.ndarray:
    """Which rows of a table of the run's (or the inventory run's) non-local
    events are among ``event_ids``: a non-local event's native key is its
    ``event_id``."""
    return (
        native_event_keys(table["event_key"])
        .isin({str(event_id) for event_id in event_ids})
        .to_numpy()
    )


def left_out_epoch_seconds(
    run_dir: Path, event_ids: Iterable[object], visits: Mapping[str, Visits]
) -> pd.DataFrame:
    """Seconds of the selected non-local events' time outside each other
    inventory, in each of the run's ``ok`` epochs.

    Each epoch's compared events (``*_features.parquet``: every inventory's
    whole-event-immobile events, with ``animal``, ``day`` and ``epoch``): the
    time of the non-local events in ``event_ids`` (``nonlocal``), the time the
    Kay SWRs whose first sample is inside a well visit span (``kay``), and,
    for each of :data:`COVERINGS`, the part of that non-local time that no
    event of those inventories covers, wherever the covering event starts
    (``outside_<covering>``).

    Parameters
    ----------
    run_dir : Path
        The event features run.
    event_ids : iterable
        The selected non-local events' ``event_id``.
    visits : Mapping
        Each ok epoch's id to its visits' ``(arrival_times, departure_times)``.

    Returns
    -------
    pd.DataFrame
        One row per ok epoch: ``animal``, ``day``, ``epoch`` and the seconds.

    Raises
    ------
    ValueError
        If the run has no readable manifest, it marks no epoch ``ok``, or an
        ok epoch has no visits in ``visits``.
    """
    event_ids = list(event_ids)
    identifiers = ok_epochs(read_manifest(run_dir))
    if missing := sorted(set(identifiers) - set(visits)):
        raise ValueError(
            f"no well visits for {len(missing)} epochs, e.g. {missing[:3]}"
        )
    rows = []
    for identifier, path in zip(
        identifiers, _epoch_paths(run_dir, "features"), strict=True
    ):
        table = pd.read_parquet(
            path,
            columns=[
                *SESSION_COLUMNS_DAY,
                "detector",
                "event_key",
                "event_start_s",
                "event_end_s",
            ],
        )
        is_nonlocal = (table["detector"] == STATE_SPACE).to_numpy()
        selected = np.zeros(len(table), dtype=bool)
        selected[is_nonlocal] = _selected(table[is_nonlocal], event_ids)
        nonlocal_ = _intervals(table[selected], (STATE_SPACE,))
        kay = table[table["detector"] == "kay"]
        kay = kay[
            in_well_visit(
                kay["event_start_s"].to_numpy(dtype=np.float64), *visits[identifier]
            )
        ]
        row = dict(table[list(SESSION_COLUMNS_DAY)].iloc[0])
        row["nonlocal"] = intervals.total_duration(nonlocal_)
        row["kay"] = intervals.total_duration(_intervals(kay, ("kay",)))
        for name, detectors in COVERINGS.items():
            outside = intervals.subtract(nonlocal_, _intervals(table, detectors))
            row[f"outside_{name}"] = intervals.total_duration(outside)
        rows.append(row)
    return pd.DataFrame(rows)


def left_out_seconds(
    run_dir: Path, event_ids: Iterable[object], visits: Mapping[str, Visits]
) -> dict[str, float]:
    """:func:`left_out_epoch_seconds`, summed over the run's ``ok`` epochs:
    ``nonlocal``, ``kay`` and ``outside_<covering>`` for each of
    :data:`COVERINGS`, in seconds.

    Raises
    ------
    ValueError
        As :func:`left_out_epoch_seconds`.
    """
    seconds = left_out_epoch_seconds(run_dir, event_ids, visits)
    names = ("nonlocal", "kay", *(f"outside_{name}" for name in COVERINGS))
    return {name: float(seconds[name].sum()) for name in names}


def left_out_animal_shares(epochs: pd.DataFrame) -> dict[str, float]:
    """The shares and rates of :func:`left_out_statistics`, epoch -> animal ->
    mean over animals, from one row per epoch.

    A time share is each epoch's left-out seconds over its selected non-local
    seconds (epochs with such time); an event share each epoch's left-out
    events over its events with a known membership (epochs with such an
    event); a rate each epoch's left-out events per hour of its exposure
    inside the well visits (epochs the non-local detector observed, with an
    exposure; an observed epoch with no left-out event has a rate of 0). Each
    animal's mean over its epochs with a value, then the mean over the
    animals with one: an animal with no such epoch is left out.

    Parameters
    ----------
    epochs : pd.DataFrame, shape (n_epochs, n_columns)
        ``animal``, ``day``, ``epoch``; ``nonlocal`` and ``outside_<covering>``
        (s) for each of :data:`COVERINGS` (NaN for an epoch the run does not
        hold); ``hours`` (NaN for an epoch not observed); and for each of
        ``kay``, ``widloski`` and ``trigger``, ``<name>_outside`` (left-out
        events) and ``<name>_known`` (events with a known membership).

    Returns
    -------
    dict
        ``nonlocal_time_outside_<covering>_fraction``,
        ``nonlocal_events_outside_<name>_fraction`` and
        ``nonlocal_events_outside_<name>_per_hour``.
    """

    def mean_over_animals(values: pd.Series, rows: pd.DataFrame) -> float:
        return float(
            animal_session_values(
                values, rows["animal"], rows[["day", "epoch"]], "mean"
            ).mean()
        )

    shares = {}
    timed = epochs[epochs["nonlocal"] > 0]
    for name in COVERINGS:
        shares[f"nonlocal_time_outside_{name}_fraction"] = mean_over_animals(
            timed[f"outside_{name}"] / timed["nonlocal"], timed
        )
    for name in ("kay", "widloski", "trigger"):
        known = epochs[epochs[f"{name}_known"] > 0]
        shares[f"nonlocal_events_outside_{name}_fraction"] = mean_over_animals(
            known[f"{name}_outside"] / known[f"{name}_known"], known
        )
        observed = epochs[epochs["hours"] > 0]
        shares[f"nonlocal_events_outside_{name}_per_hour"] = mean_over_animals(
            observed[f"{name}_outside"].fillna(0) / observed["hours"], observed
        )
    return shares


def left_out_statistics(
    run_dir: Path,
    inventory_dir: Path,
    event_ids: Iterable[object],
    visits: Mapping[str, Visits],
    exposure: pd.DataFrame,
) -> dict[str, float]:
    """What an analysis that starts from another inventory leaves out of the
    selected non-local events, in events, in event time and per hour of rest
    inside the well visits.

    Parameters
    ----------
    run_dir : Path
        The event features run.
    inventory_dir : Path
        The inventory comparison run the event features run read: its
        observations (which sessions the non-local detector observed) and its
        cells (each event's membership against the triggered inventories) of
        the epochs its manifest marks ``ok``.
    event_ids : iterable
        The selected non-local events' ``event_id``
        (``cohort_selection.select_visit_context``).
    visits : Mapping
        Each ok epoch's id to its visits' ``(arrival_times, departure_times)``.
    exposure : pd.DataFrame
        One row per ok epoch (``animal``, ``day``, ``epoch``,
        ``exposure_in_visit_s``): the non-local detector's exposure inside the
        well visits (:func:`exposure_in_well_visits`).

    Returns
    -------
    statistics : dict
        - ``rest_hours``: hours of the non-local detector's exposure inside a
          well visit, summed over the sessions it observed.
        - ``nonlocal_time_s``, ``kay_time_s``: seconds the selected non-local
          events, and the Kay SWRs that start inside a visit, span, summed
          over the epochs (:func:`left_out_seconds`).
        - For each of :data:`COVERINGS`, ``nonlocal_time_outside_<covering>_s``,
          the selected non-local event time it does not cover (summed), and
          ``..._fraction``, that time as a share of the selected non-local
          time (:func:`left_out_animal_shares`: each epoch's, each animal's
          mean over its epochs, the mean over animals).
        - For ``kay`` and ``widloski``, and for ``trigger`` (no SWR definition
          and no MUA event; the comparison's cell
          ``sequence_matched_by_no_trigger``): ``nonlocal_events_outside_<name>``,
          the selected non-local events none of its events overlaps (wherever
          that event starts; summed); ``..._fraction``, their share of the
          selected events with a known membership; and ``..._per_hour``, per
          hour of exposure inside the visits (both
          :func:`left_out_animal_shares`).

    Raises
    ------
    ValueError
        If either run has no readable manifest or marks no epoch ``ok``, an ok
        epoch has no visits, or ``exposure`` lacks a session the non-local
        detector observed.
    """
    event_ids = list(event_ids)
    session = list(SESSION_COLUMNS_DAY)
    availability = detector_availability(
        typed_table(
            map(read_table, _epoch_paths(inventory_dir, "observations")),
            TABLE_DTYPES["observations"],
        )
    )
    observed = availability.loc[
        (availability["detector"] == STATE_SPACE) & availability["available"], session
    ]
    exposed = observed.merge(
        exposure[[*session, "exposure_in_visit_s"]],
        on=session,
        how="left",
        validate="1:1",
    )
    if exposed["exposure_in_visit_s"].isna().any():
        raise ValueError("the exposure lacks a session the non-local detector observed")
    exposed = exposed.assign(hours=exposed["exposure_in_visit_s"] / 3600.0)
    seconds = left_out_epoch_seconds(run_dir, event_ids, visits)
    statistics = {
        "rest_hours": float(exposed["hours"].sum()),
        "nonlocal_time_s": float(seconds["nonlocal"].sum()),
        "kay_time_s": float(seconds["kay"].sum()),
    }
    for name in COVERINGS:
        statistics[f"nonlocal_time_outside_{name}_s"] = float(
            seconds[f"outside_{name}"].sum()
        )

    membership = _read_epochs(
        run_dir, "pair_membership", filters=[("detector", "==", STATE_SPACE)]
    )
    known = membership[
        membership["membership"].isin(_KNOWN) & _selected(membership, event_ids)
    ].assign(alone=lambda t: t["membership"] == "unmatched")
    cells = typed_table(
        map(read_table, _epoch_paths(inventory_dir, "cells")), TABLE_DTYPES["cells"]
    )
    cells = cells[
        (cells["cell"] == "sequence_matched_by_no_trigger")
        & (cells["detector"] == STATE_SPACE)
    ].merge(observed, on=session, how="inner")
    cells = cells[_selected(cells, event_ids) & cells["membership"].isin(_KNOWN)]
    cells = cells.assign(alone=cells["membership"] == "unmatched")
    epochs = seconds.merge(exposed[[*session, "hours"]], on=session, how="outer")
    for name, table in (
        ("kay", known[known["other_detector"] == "kay"]),
        ("widloski", known[known["other_detector"] == MEMORYLESS]),
        ("trigger", cells),
    ):
        counts = (
            table.groupby(session)["alone"]
            .agg(["sum", "size"])
            .set_axis([f"{name}_outside", f"{name}_known"], axis=1)
            .reset_index()
        )
        epochs = epochs.merge(counts, on=session, how="outer")
        statistics[f"nonlocal_events_outside_{name}"] = int(table["alone"].sum())
    for key, value in left_out_animal_shares(epochs).items():
        statistics[key] = value
    return {
        key: statistics[key]
        for key in (
            "rest_hours",
            "nonlocal_time_s",
            "kay_time_s",
            *(
                f"nonlocal_time_outside_{name}_{unit}"
                for name in COVERINGS
                for unit in ("s", "fraction")
            ),
            *(
                f"nonlocal_events_outside_{name}{suffix}"
                for name in ("kay", "widloski", "trigger")
                for suffix in ("", "_fraction", "_per_hour")
            ),
        )
    }
