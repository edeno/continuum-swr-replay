"""The other detectors' events against the non-local detector's.

Read from an event features run (``paths.event_features_root()``), for the
epochs its manifest marks ``ok`` and no others (``feature_run.ok_epochs``):
every inventory's compared events (``*_features.parquet``), each event's
membership against every other inventory (``*_pair_membership.parquet``) and
its state shares (``*_state_shares.parquet``, from
``scripts/detection/add_state_shares.py``).
The non-local detector's inventory is ``state_space``. An event's membership
against another inventory is known (``matched`` or ``unmatched``) only when the
event lies wholly inside the pair's shared exposure.

A row of an event is in a state when that state's probability is above 0.8, so
an event's state shares are the shares of its rows in each state, and in none
(:data:`~continuum_swr_replay.analysis.inventory_comparison.features.STATE_SHARE_COLUMNS`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from continuum_swr_replay import intervals
from continuum_swr_replay.analysis.inventory_comparison.feature_run import (
    ok_epochs,
    output_path,
    read_manifest,
)
from continuum_swr_replay.analysis.inventory_comparison.features import (
    STATE_SHARE_COLUMNS,
)
from continuum_swr_replay.analysis.inventory_comparison.inventory_run import (
    SWR_DEFINITIONS,
    TRIGGERED,
)
from continuum_swr_replay.analysis.inventory_comparison.schema import (
    MEMORYLESS,
    STATE_SPACE,
)

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


def _epoch_paths(run_dir: Path, name: str) -> list[Path]:
    """Table ``name`` of each epoch the run's manifest marks ``ok``.

    Raises
    ------
    ValueError
        If the run has no readable manifest, or it marks no epoch ``ok``.
    """
    ok = ok_epochs(read_manifest(run_dir))
    if not ok:
        raise ValueError(f"the manifest of {run_dir} marks no epoch ok")
    return [output_path(identifier, run_dir, name) for identifier in ok]


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


def check_against_directional(events: pd.DataFrame, directional: pd.DataFrame) -> None:
    """Raise unless the per-event memberships give the comparison's own counts."""
    mine = events.groupby("detector")["matched"].agg(n_eligible="size", n_matched="sum")
    theirs = (
        directional[directional["target_detector"] == STATE_SPACE]
        .groupby("source_detector")[["n_eligible", "n_matched"]]
        .sum()
    )
    joined = theirs.join(mine, rsuffix="_events", how="outer")
    if not (
        (joined["n_eligible"] == joined["n_eligible_events"]).all()
        and (joined["n_matched"] == joined["n_matched_events"]).all()
    ):
        raise ValueError(f"memberships do not give the directional counts:\n{joined}")


def other_detector_statistics(events: pd.DataFrame) -> pd.DataFrame:
    """For each other detector, how its events sit against the non-local events.

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, n_columns)
        Each other detector's compared events (:func:`other_detector_events`),
        with ``detector``, ``animal``, ``matched`` and the four state shares.
        An event with no state shares is left out of the state statistics.

    Returns
    -------
    statistics : pd.DataFrame, shape (n_detectors, 14)
        Indexed by detector. "Mostly" a state is more than half the event's
        rows.

        - ``n_events``: its compared events.
        - ``no_nonlocal_fraction``: the share overlapping no non-local event,
          and ``no_nonlocal_fraction_animal_min`` and ``..._animal_max``, the
          lowest and highest of the animals' own shares.
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
        matched_by_animal = mine.groupby("animal")["matched"].mean()
        rows[detector] = {
            "n_events": len(mine),
            "no_nonlocal_fraction": (~mine["matched"]).sum() / len(mine),
            "no_nonlocal_fraction_animal_min": 1 - matched_by_animal.max(),
            "no_nonlocal_fraction_animal_max": 1 - matched_by_animal.min(),
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


def left_out_statistics(run_dir: Path, report_dir: Path) -> dict[str, float]:
    """What an analysis that starts from another inventory leaves out of the
    non-local events, in events, in event time and per hour of rest.

    Parameters
    ----------
    run_dir : Path
        The event features run.
    report_dir : Path
        The report of the inventory comparison the run read, with
        ``counts_by_animal.csv`` (each inventory's exposure) and
        ``cells_by_animal.csv`` (the comparison's cells).

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
    counts = pd.read_csv(report_dir / "counts_by_animal.csv")
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
    cells = pd.read_csv(report_dir / "cells_by_animal.csv")
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
