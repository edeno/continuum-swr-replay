"""The share of events with no SWR among the events whose content is resolved.

An event's width is the 95 % HPD length of the other tetrodes' position
estimate in the held-out tetrode run (``shared_content.WIDTH_COLUMN``). The
events with a width under :data:`RESOLVED_WIDTH_CM` are those whose estimate
points to a part of the track; in the widest group the agreement is no
better than an unrelated event of the same width (``shared_content``). This
module gives the share of events with no SWR among the resolved events and
among the wide ones, by the rule of ``event_statistics.overlap_class_statistics``
(each epoch's share, each animal's mean over its epochs, the mean over
animals), and the share of all events that are resolved
(:func:`resolved_share_statistics`).
"""

from __future__ import annotations

import pandas as pd

from continuum_swr_replay.analysis.event_evidence.shared_content import (
    RESOLVED_WIDTH_CM,
    WIDTH_COLUMN,
)
from continuum_swr_replay.analysis.event_properties.event_statistics import (
    overlap_class_statistics,
)
from continuum_swr_replay.analysis.shared_statistics import animal_session_values
from continuum_swr_replay.columns import require_columns
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS


def event_widths(events: pd.DataFrame, windows: pd.DataFrame) -> pd.Series:
    """Each event's width (cm, NaN when the held-out tetrode run has none).

    Parameters
    ----------
    events : pd.DataFrame
        The canonical event table's selected events, with ``event_id``.
    windows : pd.DataFrame
        ``tetrode_holdout.window_scores``; the event windows (``kind``
        ``"event"``) are read.

    Raises
    ------
    ValueError
        If an event has two event windows.
    """
    require_columns(windows, ["event_id", "kind", WIDTH_COLUMN], "the windows")
    mine = windows[windows["kind"] == "event"]
    if mine["event_id"].duplicated().any():
        raise ValueError("an event has two event windows")
    return events["event_id"].map(mine.set_index("event_id")[WIDTH_COLUMN])


def resolved_share_statistics(
    events: pd.DataFrame, windows: pd.DataFrame
) -> dict[str, dict[str, float] | float | int]:
    """The share of events with no SWR among the resolved and the wide events,
    and the share of events that are resolved.

    Parameters
    ----------
    events : pd.DataFrame
        The canonical event table's selected events: ``event_id``,
        ``overlap_class``, ``whole_event_speed_eligible`` and the session
        columns.
    windows : pd.DataFrame
        As :func:`event_widths` takes it.

    Returns
    -------
    dict
        ``resolved`` and ``wide``: ``overlap_class_statistics`` of the events
        under and at or above :data:`RESOLVED_WIDTH_CM`;
        ``resolved_share_animal_mean``, ``resolved_share_animal_min`` and
        ``resolved_share_animal_max``: the share of the events with a width
        that are resolved, each epoch's, each animal's mean over its epochs,
        their mean, the lowest and highest animal; ``n_without_width``: the
        events without a width, left out of every value.
    """
    width = event_widths(events, windows)
    known = events[width.notna()]
    resolved = (width[width.notna()] < RESOLVED_WIDTH_CM).astype(float)
    shares = animal_session_values(
        resolved,
        known["animal"],
        known[[c for c in SESSION_COLUMNS if c != "animal"]],
        "mean",
    )
    return {
        "resolved": overlap_class_statistics(known[resolved.to_numpy() == 1.0]),
        "wide": overlap_class_statistics(known[resolved.to_numpy() == 0.0]),
        "resolved_share_animal_mean": float(shares.mean()),
        "resolved_share_animal_min": float(shares.min()),
        "resolved_share_animal_max": float(shares.max()),
        "n_without_width": int(width.isna().sum()),
    }
