"""The held-out tetrode evidence of events decoded at low-rate places, and the
place information of the scored spikes against the events' controls.

An event is a low-rate event when its decoded place is among the lowest-rate
tenth of the track's places in the encoding model
(:mod:`.content_place_rate`: ``place_rate_rank`` below
``LOW_RATE_RANK``). Where the summed rate is low, a bin without spikes is
itself evidence for the place, so these events may be carried by the absence
of spikes more than by the waveform marks. This module splits the held-out
tetrode scores (``tetrode_holdout.window_scores``) of the events by that group
(:func:`low_rate_split_statistics`), gives the events outside it
(:func:`events_outside_low_rate`), and pairs each event with its Local-rest
control (:func:`event_minus_control`).
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from continuum_swr_replay.analysis.event_evidence.likelihood_advantage import (
    likelihood_advantage_statistics,
)
from continuum_swr_replay.analysis.event_properties.content_place_rate import (
    LOW_RATE_RANK,
    low_rate_share_statistics,
    rank_tenth_shares,
)
from continuum_swr_replay.columns import require_columns

PLACES = ("low_rate", "other")
"""The two groups of events by their decoded place's rate rank: below
``LOW_RATE_RANK`` and at or above it (an event without a rank is in neither)."""


def place_groups(event_ids: pd.Series, place_rate: pd.DataFrame) -> pd.Series:
    """Each event's group of :data:`PLACES` (NaN without a rank).

    Parameters
    ----------
    event_ids : pd.Series
        The events' ids.
    place_rate : pd.DataFrame
        The place-rate run's rows: ``event_id`` (unique) and ``place_rate_rank``.

    Raises
    ------
    ValueError
        If an event id is in ``place_rate`` twice.
    """
    require_columns(place_rate, ["event_id", "place_rate_rank"], "the place-rate run")
    if place_rate["event_id"].duplicated().any():
        raise ValueError("an event has two rows in the place-rate run")
    rank = event_ids.map(place_rate.set_index("event_id")["place_rate_rank"])
    group = pd.Series(pd.NA, index=event_ids.index, dtype="object")
    group[rank < LOW_RATE_RANK] = "low_rate"
    group[rank >= LOW_RATE_RANK] = "other"
    return group


def events_outside_low_rate(
    events: pd.DataFrame, place_rate: pd.DataFrame
) -> pd.DataFrame:
    """The events that are not low-rate events (those without a rank stay).

    Parameters
    ----------
    events : pd.DataFrame
        Rows with ``event_id`` (the canonical event table's selected events).
    place_rate : pd.DataFrame
        As :func:`place_groups` takes it.
    """
    return events[place_groups(events["event_id"], place_rate) != "low_rate"]


def low_rate_split_statistics(
    windows: pd.DataFrame,
    place_rate: pd.DataFrame,
    column: str,
    *,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """A held-out tetrode score of the events by place group and SWR group.

    Parameters
    ----------
    windows : pd.DataFrame
        ``tetrode_holdout.window_scores``' rows; the events (``kind``
        ``"event"``) are read.
    place_rate : pd.DataFrame
        As :func:`place_groups` takes it.
    column : str
        The score per spike (or per informative spike).
    n_boot, seed : int, optional
        The bootstrap's replicates and seed, the same for every row.

    Returns
    -------
    pd.DataFrame
        Indexed by ``place`` (:data:`PLACES`) and ``group``: the columns of
        ``likelihood_advantage_statistics`` (each epoch's median event, each
        animal's median epoch, the mean over animals, its 95 % interval, the
        animals above zero).
    """
    events = windows[windows["kind"] == "event"]
    places = place_groups(events["event_id"], place_rate)
    parts = []
    for place in PLACES:
        statistics = likelihood_advantage_statistics(
            events[places == place], column=column, n_boot=n_boot, seed=seed
        )
        parts.append(statistics.assign(place=place).reset_index())
    return pd.concat(parts, ignore_index=True).set_index(["place", "group"])


def event_minus_control(windows: pd.DataFrame, column: str) -> pd.DataFrame:
    """Each event's ``column`` minus its Local-rest control's.

    Parameters
    ----------
    windows : pd.DataFrame
        ``tetrode_holdout.window_scores``' rows: each event (``kind``
        ``"event"``) and its control (``"control"``), which carries its
        event's ``event_id``.
    column : str

    Returns
    -------
    pd.DataFrame
        The events' rows with ``{column}_minus_control`` (NaN when either is
        missing, or the event has no control).

    Raises
    ------
    ValueError
        If an event or a control appears twice for one ``event_id``.
    """
    events = windows[windows["kind"] == "event"]
    controls = windows[windows["kind"] == "control"]
    for name, rows in (("event", events), ("control", controls)):
        if rows["event_id"].duplicated().any():
            raise ValueError(f"an event_id has two {name} windows")
    control = events["event_id"].map(controls.set_index("event_id")[column])
    return events.assign(**{f"{column}_minus_control": events[column] - control})


LOW_RATE_MEASURES = (
    "gain_per_spike",
    "gain_mark_per_spike",
    "agreement_minus_null_per_spike",
    "mark_information_per_spike",
)
"""The held-out tetrode scores read by place group
(``tetrode_holdout.window_scores``' columns of a run with ``--extra-scores``):
the gain over local coding, its mark part, the agreement minus the
shuffled-event reference, and the place information of the scored spikes'
marks."""
LOW_RATE_FIGURE_MEASURES = LOW_RATE_MEASURES[1:]
"""The scores Figure 2's supplement draws by place group, one axis each."""


class LowRateStatistics(NamedTuple):
    """The low-rate group's statistics (:func:`low_rate_statistics`)."""

    shares: pd.DataFrame
    """``content_place_rate.low_rate_share_statistics``, indexed by group."""
    tenths: pd.DataFrame
    """``content_place_rate.rank_tenth_shares``, indexed by group and rank bin."""
    measures: dict[str, pd.DataFrame]
    """:func:`low_rate_split_statistics` of each of :data:`LOW_RATE_MEASURES`."""


def low_rate_statistics(
    windows: pd.DataFrame, place_rate: pd.DataFrame
) -> LowRateStatistics:
    """The share of events at low-rate places, by tenth of the rank, and the
    held-out tetrode scores by place group, as the events summary and Figure
    2's supplement read them.

    Parameters
    ----------
    windows : pd.DataFrame
        ``tetrode_holdout.window_scores`` of a run with ``--extra-scores`` (the
        selected events and their controls).
    place_rate : pd.DataFrame
        The place-rate run's rows of the selected events: ``event_id``,
        ``animal``, ``epoch_id``, ``group`` and ``place_rate_rank``.
    """
    return LowRateStatistics(
        low_rate_share_statistics(place_rate),
        rank_tenth_shares(place_rate),
        {
            column: low_rate_split_statistics(windows, place_rate, column)
            for column in LOW_RATE_MEASURES
        },
    )
