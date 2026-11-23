"""Summary statistics of the immobile non-local events.

Each function takes a table an analysis wrote (the canonical event table, a
reference table, a run's per-event metrics), except :func:`quadrant_fractions`,
which takes one session's per-time-bin content distances. Most return named
numbers: a dict,
or a table with one row per group (per animal, for
:func:`animal_occupancy_shares`). :func:`boundary_offsets` returns one row per
event (what a histogram draws), :func:`most_steps` a count with the properties
that attain it, :func:`animal_class_medians` each animal's median of each
class, and :func:`n_animals_in_direction` and
:func:`n_animals_non_swr_more_at_well` one count. They are the statistics the
events summary reports (``continuum_swr_replay.reported.events`` names and
describes them) and a figure that shows one of them calls the same function.

SWR status is the event table's ``is_swr``: strict positive-duration overlap
with an SWR. A statistic of the events (a median, a share) pools them over
epochs and animals. "Animals" statistics count the animals whose own value
meets the stated condition, and an animal's own value goes event -> epoch ->
animal
(:func:`~continuum_swr_replay.analysis.shared_statistics.animal_session_values`):
a median of its epochs' medians, or a mean of its epochs' means. Times are in
seconds.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_evidence.likelihood_advantage import (
    likelihood_advantage_statistics,
)
from continuum_swr_replay.analysis.shared_statistics import (
    animal_session_values,
    session_median_statistics,
)
from continuum_swr_replay.tables.session_keys import SESSION_COLUMNS, n_sessions

_WITHIN_ANIMAL = [column for column in SESSION_COLUMNS if column != "animal"]
"""The session columns that name a session within its animal."""

# --- how the events sit against SWRs ----------------------------------------------


def compute_swr_overlap_class_counts(
    classified_events: pd.DataFrame,
) -> dict[str, int]:
    """Count immobile non-local events by SWR-overlap class.

    Operates on the output of
    :func:`continuum_swr_replay.analysis.event_definition.classified_table.build_classified_nonlocal_event_table`,
    restricted to immobile events (``whole_event_speed_eligible``), and
    tallies the ``overlap_class`` column. The single overlap convention
    (strict positive-duration overlap, "contained" = some single SWR spans
    the event) lives in that builder, so this is a thin tally with no
    classification logic of its own.

    Parameters
    ----------
    classified_events : pd.DataFrame
        Per-non-local-event table with ``overlap_class`` and
        ``whole_event_speed_eligible`` columns. May be the builder's
        zero-row empty frame.

    Returns
    -------
    dict[str, int]
        Mapping ``{"swr_contained": n, "swr_partial": n,
        "non_swr": n, "total": n}``.

    """
    zero = {"swr_contained": 0, "swr_partial": 0, "non_swr": 0, "total": 0}
    if classified_events.empty:
        return zero
    # ``.loc`` + ``astype(bool)`` so the builder's zero-row empty frame
    # (where ``whole_event_speed_eligible`` is object-dtype) still indexes
    # cleanly rather than raising on a non-boolean mask.
    immobile = classified_events.loc[
        classified_events["whole_event_speed_eligible"].astype(bool)
    ]
    if immobile.empty:
        return zero
    counts = immobile["overlap_class"].value_counts().to_dict()
    return {
        "swr_contained": int(counts.get("swr_contained", 0)),
        "swr_partial": int(counts.get("swr_partial", 0)),
        "non_swr": int(counts.get("non_swr", 0)),
        "total": int(len(immobile)),
    }


OVERLAP_CLASSES = ("swr_contained", "swr_partial", "non_swr")
"""The SWR-overlap classes, inside an SWR first."""


def overlap_class_animal_shares(events: pd.DataFrame) -> pd.DataFrame:
    """Each animal's share of its events in each SWR-overlap class.

    An animal's share of a class is the mean over its epochs of each epoch's
    share of its events (event -> epoch -> animal); an epoch with no event of a
    class has a share of 0 in it.

    Parameters
    ----------
    events : pd.DataFrame
        The canonical event table: ``overlap_class``, ``animal`` and the session
        columns.

    Returns
    -------
    pd.DataFrame, shape (n_animals, 3)
        Indexed by animal; one column per :data:`OVERLAP_CLASSES`.
    """
    return pd.DataFrame(
        {
            overlap_class: animal_session_values(
                (events["overlap_class"] == overlap_class).astype(float),
                events["animal"],
                events[_WITHIN_ANIMAL],
                "mean",
            )
            for overlap_class in OVERLAP_CLASSES
        }
    )


def overlap_class_statistics(events: pd.DataFrame) -> dict[str, float]:
    """How the immobile non-local events divide among the SWR-overlap classes.

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, n_columns)
        The canonical event table, with ``overlap_class`` (``non_swr``,
        ``swr_contained`` or ``swr_partial``), ``whole_event_speed_eligible``
        and the session columns
        (:data:`~continuum_swr_replay.tables.session_keys.SESSION_COLUMNS`).

    Returns
    -------
    statistics : dict
        - ``n_events``: the events.
        - ``n_non_swr``, ``n_swr_contained``, ``n_swr_partial``: the events of
          each class.
        - ``non_swr_fraction``: the share overlapping no SWR, pooled over animals.
        - ``swr_contained_fraction``, ``swr_partial_fraction``: the shares
          lying inside an SWR and overlapping one in part, pooled the same way.
        - ``non_swr_fraction_animal_min``, ``non_swr_fraction_animal_max``: the
          lowest and highest of the animals' own shares. An animal's share of
          a class is the mean over its epochs of each epoch's share of its
          events (event -> epoch -> animal); an epoch with no event of a class
          has a share of 0 in it.
        - ``n_animals``, ``n_epochs``: animals and epochs with an event.
        - ``n_animals_non_swr_plurality``: animals whose non-SWR share is the
          largest of their three.
        - ``n_animals_non_swr_majority``: animals whose non-SWR share is above
          one half.

    Raises
    ------
    ValueError
        If an event is not immobile as a whole, or its class is none of the
        three.
    """
    if not events["whole_event_speed_eligible"].all():
        raise ValueError(
            "the event table holds events that are not whole-event immobile"
        )
    counts = compute_swr_overlap_class_counts(events)
    n_classed = counts["non_swr"] + counts["swr_contained"] + counts["swr_partial"]
    if n_classed != counts["total"]:
        raise ValueError(
            f"{counts['total'] - n_classed} events have an overlap_class that is "
            "none of non_swr, swr_contained and swr_partial"
        )
    shares = overlap_class_animal_shares(events)
    non_swr = shares["non_swr"]
    return {
        "n_events": counts["total"],
        "n_non_swr": counts["non_swr"],
        "n_swr_contained": counts["swr_contained"],
        "n_swr_partial": counts["swr_partial"],
        "non_swr_fraction": counts["non_swr"] / counts["total"],
        "swr_contained_fraction": counts["swr_contained"] / counts["total"],
        "swr_partial_fraction": counts["swr_partial"] / counts["total"],
        "non_swr_fraction_animal_min": non_swr.min(),
        "non_swr_fraction_animal_max": non_swr.max(),
        "n_animals": len(shares),
        "n_epochs": n_sessions(events),
        "n_animals_non_swr_plurality": (
            non_swr > shares.drop(columns="non_swr").max(axis=1)
        ).sum(),
        "n_animals_non_swr_majority": (non_swr > 0.5).sum(),
    }


#: The non-local states whose mean share of an event :func:`state_composition_statistics`
#: reports, by the event table's mean-probability column.
STATE_SHARE_COLUMNS = {
    "continuous": "continuous_prob_mean",
    "fragmented": "fragmented_prob_mean",
}


def state_composition_statistics(events: pd.DataFrame) -> pd.DataFrame:
    """The mean share of an SWR or no-SWR event in each non-local state, by animal.

    Each event counts alike, whatever its length: this is the mean over events of
    each event's share, not the share of all event time.

    An event's share in a state is its mean probability of that state over its
    bins (the event table's ``continuous_prob_mean`` and
    ``fragmented_prob_mean``). An animal's share is the mean over its epochs of
    each epoch's mean over events (``animal_session_values(..., "mean")``): the
    shares of one animal then add up as its events' do. The summary is the mean
    over animals.

    Parameters
    ----------
    events : pd.DataFrame
        The canonical event table: ``is_swr``, the session columns
        (:data:`~continuum_swr_replay.tables.session_keys.SESSION_COLUMNS`) and
        the columns of :data:`STATE_SHARE_COLUMNS`.

    Returns
    -------
    pd.DataFrame
        Indexed by group (``"no SWR"``, ``"SWR"``): for each state,
        ``{state}_animals`` (the animals' shares, in sorted animal order) and
        ``{state}_animal_mean``; ``animal_names``, ``n_animals`` and
        ``n_events``.
    """
    rows = {}
    is_swr = events["is_swr"].astype(bool)
    for group, mine in (("no SWR", events[~is_swr]), ("SWR", events[is_swr])):
        row: dict[str, object] = {"n_events": len(mine)}
        for state, column in STATE_SHARE_COLUMNS.items():
            per_animal = animal_session_values(
                mine[column], mine["animal"], mine[list(SESSION_COLUMNS)], "mean"
            )
            row[f"{state}_animals"] = per_animal.to_numpy()
            row[f"{state}_animal_mean"] = float(per_animal.mean())
        row["animal_names"] = per_animal.index.to_numpy()
        row["n_animals"] = len(per_animal)
        rows[group] = row
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("group")


def boundary_offsets(events: pd.DataFrame) -> pd.DataFrame:
    """Each SWR-overlapping event's bounds against its matched SWR's.

    The matched SWR is the one containing the event when there is one, else the
    earliest one overlapping it (the event table's ``matched_swr_start`` and
    ``matched_swr_end``).

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, n_columns)
        The canonical event table, with ``overlap_class``, ``event_start_s``,
        ``event_end_s``, ``matched_swr_start`` and ``matched_swr_end``.

    Returns
    -------
    offsets : pd.DataFrame, shape (n_overlapping_events, 2)
        ``start_offset_s`` (event start minus SWR start) and ``end_offset_s``
        (event end minus SWR end), indexed as ``events``.

    Raises
    ------
    ValueError
        If an SWR-overlapping event has no matched SWR bounds.
    """
    overlapping = events[events["overlap_class"] != "non_swr"]
    offsets = pd.DataFrame(
        {
            "start_offset_s": overlapping["event_start_s"]
            - overlapping["matched_swr_start"],
            "end_offset_s": overlapping["event_end_s"] - overlapping["matched_swr_end"],
        }
    )
    if offsets.isna().any(axis=None):
        raise ValueError("an SWR-overlapping event has no matched SWR bounds")
    return offsets


OFFSET_TAILS_S = {"twenty_five_ms": 0.025, "fifty_ms": 0.050}
"""How far (s) an event's bound may sit from its SWR's before it counts in the
tail (:func:`boundary_offset_statistics`), by the name its statistic carries."""


def boundary_offset_statistics(offsets: pd.DataFrame) -> dict[str, float]:
    """The typical offset of an SWR-overlapping event's bounds from its SWR's.

    Parameters
    ----------
    offsets : pd.DataFrame, shape (n_overlapping_events, 2)
        The SWR-overlapping events' :func:`boundary_offsets`.

    Returns
    -------
    statistics : dict
        ``n_overlapping``, the events overlapping an SWR;
        ``start_offset_median_s`` and ``end_offset_median_s``, the medians of
        their offsets; and, for each bound and each of :data:`OFFSET_TAILS_S`,
        ``{bound}_offset_beyond_{name}_fraction``, the share of the events whose
        offset is larger than that in magnitude.
    """
    statistics = {
        "n_overlapping": len(offsets),
        "start_offset_median_s": offsets["start_offset_s"].median(),
        "end_offset_median_s": offsets["end_offset_s"].median(),
    }
    for bound in ("start", "end"):
        magnitude_us = _offset_us(offsets[f"{bound}_offset_s"]).abs()
        for name, tail_s in OFFSET_TAILS_S.items():
            statistics[f"{bound}_offset_beyond_{name}_fraction"] = float(
                (magnitude_us > round(1e6 * tail_s)).mean()
            )
    return statistics


def _offset_us(offset_s: pd.Series) -> pd.Series:
    """An offset in whole microseconds, so that a bound exactly at a
    tolerance's edge is within it."""
    return np.round(1e6 * offset_s)


TIMING_TOLERANCE = "twenty_five_ms"
"""The :data:`OFFSET_TAILS_S` tolerance within which an event's bound is
together with its SWR's (:func:`boundary_timing_statistics`): the same 25 ms
as the ``*_beyond_twenty_five_ms_fraction`` statistics."""
TIMING_CLASSES = ("earlier", "together", "later")
"""How an event's bound sits against its SWR's, in order."""


def boundary_timing_classes(offsets: pd.DataFrame) -> pd.DataFrame:
    """Each SWR-overlapping event's start and end against its SWR's.

    A bound is ``"earlier"`` when its offset (event minus SWR,
    :func:`boundary_offsets`) is below minus the tolerance
    (:data:`TIMING_TOLERANCE`), ``"later"`` when above it, and ``"together"``
    within it (an offset exactly at the edge is together, as it is not in the
    ``*_beyond_*`` tail). So a positive start offset is an event that starts
    later than its SWR, a positive end offset one that ends later.

    Parameters
    ----------
    offsets : pd.DataFrame, shape (n_overlapping_events, 2)
        :func:`boundary_offsets`.

    Returns
    -------
    pd.DataFrame, shape (n_overlapping_events, 2)
        ``start`` and ``end``, each one of :data:`TIMING_CLASSES`, indexed as
        ``offsets``.
    """
    tolerance_us = round(1e6 * OFFSET_TAILS_S[TIMING_TOLERANCE])
    classes = {}
    for bound in ("start", "end"):
        offset_us = _offset_us(offsets[f"{bound}_offset_s"])
        classes[bound] = np.select(
            [offset_us < -tolerance_us, offset_us > tolerance_us],
            [TIMING_CLASSES[0], TIMING_CLASSES[2]],
            TIMING_CLASSES[1],
        )
    return pd.DataFrame(classes, index=offsets.index)


def boundary_timing_statistics(
    offsets: pd.DataFrame, animal: pd.Series
) -> pd.DataFrame:
    """The share of SWR-overlapping events in each of the nine combinations of
    their start and end against their SWR's (:func:`boundary_timing_classes`).

    Parameters
    ----------
    offsets : pd.DataFrame, shape (n_overlapping_events, 2)
        :func:`boundary_offsets`.
    animal : pd.Series
        Each event's animal, indexed as ``offsets`` (or a superset of it).

    Returns
    -------
    pd.DataFrame, shape (9, 4)
        Indexed by ``start`` and ``end`` (each of :data:`TIMING_CLASSES`, in
        order): ``share``, pooled over every event, as panels c and d pool;
        ``animal_min`` and ``animal_max``, the lowest and highest of each
        animal's share over its own events; ``n_events``, the events in the
        cell.
    """
    classes = boundary_timing_classes(offsets)
    cells = pd.MultiIndex.from_product(
        [TIMING_CLASSES, TIMING_CLASSES], names=["start", "end"]
    )
    counts = classes.value_counts(["start", "end"]).reindex(cells, fill_value=0)
    per_animal = (
        classes.assign(animal=animal.loc[classes.index])
        .groupby("animal")
        .value_counts(["start", "end"], normalize=True)
        .unstack("animal")
        .reindex(cells, fill_value=0.0)
        .fillna(0.0)
    )
    return pd.DataFrame(
        {
            "share": counts / counts.sum(),
            "animal_min": per_animal.min(axis=1),
            "animal_max": per_animal.max(axis=1),
            "n_events": counts,
        }
    )


# --- theta, ripple, MUA and gamma during events -----------------------------------


def reference_medians(reference: pd.DataFrame, signals: list[str]) -> pd.DataFrame:
    """The typical running and quiet-rest level of each signal.

    Parameters
    ----------
    reference : pd.DataFrame, shape (n_epochs, n_columns)
        The brain-state reference: for each signal, each epoch's median while
        running (``<signal>_running_median``) and in quiet rest
        (``<signal>_rest_median``), z-scored within the epoch.
    signals : list of str
        The signals (``"theta"``, ``"ripple"``, ...).

    Returns
    -------
    medians : pd.DataFrame, shape (n_signals, 2)
        ``running`` and ``rest``: the median over epochs of each, by signal.
    """
    return pd.DataFrame(
        {
            "running": {s: reference[f"{s}_running_median"].median() for s in signals},
            "rest": {s: reference[f"{s}_rest_median"].median() for s in signals},
        }
    )


_CLASS_WORDS = {False: "non_swr", True: "swr"}
"""Each class's word in a column name."""


def matched_reference_medians(
    reference: pd.DataFrame, signals: list[str]
) -> pd.DataFrame:
    """The typical level of each signal in running and quiet-rest windows as long
    as each class's events.

    Parameters
    ----------
    reference : pd.DataFrame, shape (n_epochs, n_columns)
        For each signal, class (``swr``, ``non_swr``) and regime, each epoch's
        ``<signal>_<regime>_matched_<class>`` (``gamma_power.matched_level``).
    signals : list of str

    Returns
    -------
    medians : pd.DataFrame, shape (2 * n_signals, 2)
        ``running`` and ``rest``, indexed by signal and ``is_swr``: the median
        over the epochs that have the level (an epoch with no event of the class
        has none).
    """
    return pd.DataFrame.from_dict(
        {
            (signal, is_swr): {
                regime: reference[f"{signal}_{regime}_matched_{word}"].median()
                for regime in ("running", "rest")
            }
            for signal in signals
            for is_swr, word in _CLASS_WORDS.items()
        },
        orient="index",
    ).rename_axis(["signal", "is_swr"])


def event_signal_statistics(
    events: pd.DataFrame,
    reference: pd.DataFrame,
    signals: Mapping[str, str],
    matched: bool = False,
) -> pd.DataFrame:
    """Each signal during events, against the same epochs' running and rest levels.

    An event's value of a signal is its mean over the event's bins of the
    epoch's z-scored signal. An animal's value for a class is event -> epoch
    -> animal: the median over its epochs with an event of the class of each
    epoch's median event value. It is compared with the median over the same
    epochs of each epoch's running (quiet-rest) level, so every epoch counts
    once, and an epoch with no event of the class does not count.

    An epoch's level is the median of its single running (quiet-rest) bins, or,
    with ``matched``, the level of running (quiet-rest) windows as long as the
    class's events (``<signal>_<regime>_matched_<class>``). A mean over an
    event's bins of a skewed signal sits above the median of single bins when
    nothing differs, so only the matched level is the same statistic as the
    events' value.

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, n_columns)
        One row per event, with ``is_swr``, the session columns
        (:data:`~continuum_swr_replay.tables.session_keys.SESSION_COLUMNS`) and each
        signal's column.
    reference : pd.DataFrame, shape (n_epochs, n_columns)
        The brain-state reference (:func:`reference_medians`), one row per epoch.
    signals : mapping of str to str
        Each signal's column in ``events``, by signal.
    matched : bool
        Compare with the levels of windows as long as the class's events.

    Returns
    -------
    statistics : pd.DataFrame, shape (2 * n_signals, 4)
        Indexed by signal and ``is_swr``:

        - ``median``: the median event value, z, over the class's events.
        - ``n_animals_below_running``: animals, of those with an event of the
          class, whose value is below the median running level of the same
          epochs.
        - ``n_animals_above_rest``: animals, of those with an event of the
          class, whose value is above the median quiet-rest level of the same
          epochs.
        - ``reaches_running_fraction``: the share of the class's events whose
          value reaches their own epoch's median running level.

    Raises
    ------
    ValueError
        If an event has no value of a signal, or its epoch no running or
        quiet-rest level of one.
    """
    rows = {}
    for signal, column in signals.items():
        joined = _with_signal_levels(events, reference, signal, column, matched)
        levels = _animal_levels(joined, column)
        for is_swr in (False, True):
            group = joined[joined["is_swr"] == is_swr]
            per_animal = levels.loc[is_swr]
            rows[(signal, is_swr)] = {
                "median": group[column].median(),
                "n_animals_below_running": (
                    per_animal["event"] < per_animal["running"]
                ).sum(),
                "n_animals_above_rest": (
                    per_animal["event"] > per_animal["rest"]
                ).sum(),
                "reaches_running_fraction": (
                    group[column] >= group["running_level"]
                ).mean(),
            }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(
        ["signal", "is_swr"]
    )


def animal_signal_levels(
    events: pd.DataFrame,
    reference: pd.DataFrame,
    signal: str,
    column: str,
    matched: bool = False,
) -> pd.DataFrame:
    """Each animal's event value of a signal and its running and quiet-rest levels.

    The values :func:`event_signal_statistics` compares, per class: the animal's
    event value (event -> epoch -> animal medians), and the median over the same
    epochs (its epochs with an event of the class) of each epoch's running and
    quiet-rest level.

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, n_columns)
        One row per event, with ``is_swr``, the session columns and ``column``.
    reference : pd.DataFrame, shape (n_epochs, n_columns)
        The brain-state reference, one row per epoch, with
        ``<signal>_running_median`` and ``<signal>_rest_median``, or, with
        ``matched``, ``<signal>_<regime>_matched_<class>``.
    signal : str
        The signal's name in the reference (``"theta"``).
    column : str
        The signal's column in ``events`` (``"theta_z"``).
    matched : bool
        Take each class's levels from windows as long as its events
        (:func:`event_signal_statistics`).

    Returns
    -------
    levels : pd.DataFrame, shape (n_classes * n_animals, 3)
        ``event``, ``running`` and ``rest``, z, indexed by ``is_swr`` and
        ``animal``. An animal with no event of a class has no row for it.

    Raises
    ------
    ValueError
        If an event has no value of the signal, or its epoch no running or
        quiet-rest level of it.
    """
    return _animal_levels(
        _with_signal_levels(events, reference, signal, column, matched), column
    )


def _with_signal_levels(
    events: pd.DataFrame,
    reference: pd.DataFrame,
    signal: str,
    column: str,
    matched: bool = False,
) -> pd.DataFrame:
    """``events`` joined to their epoch's levels, each event's as ``running_level``
    and ``rest_level``; raise if a value or level is missing.

    The levels are the epoch's medians of single bins, or, with ``matched``, the
    levels of windows as long as the event's class's events.
    """
    joined = events.merge(
        reference, on=list(SESSION_COLUMNS), how="left", validate="m:1"
    )
    is_swr = joined["is_swr"].to_numpy(dtype=bool)
    for regime in ("running", "rest"):
        if matched:
            joined[f"{regime}_level"] = np.where(
                is_swr,
                joined[f"{signal}_{regime}_matched_{_CLASS_WORDS[True]}"],
                joined[f"{signal}_{regime}_matched_{_CLASS_WORDS[False]}"],
            )
        else:
            joined[f"{regime}_level"] = joined[f"{signal}_{regime}_median"]
    # Counted under the reference's own column names, which the error names.
    source = "matched" if matched else "median"
    missing = (
        joined[[column, "running_level", "rest_level"]]
        .isna()
        .sum()
        .rename(
            {
                f"{regime}_level": f"{signal}_{regime}_{source}"
                for regime in ("running", "rest")
            }
        )
    )
    if missing.any():
        raise ValueError(
            f"events with no {signal} value, or whose epoch has no running or "
            f"quiet-rest level of it:\n{missing[missing > 0]}"
        )
    return joined


def _animal_levels(joined: pd.DataFrame, column: str) -> pd.DataFrame:
    """Per class and animal: the event value and the running and rest levels."""
    frames = {}
    for is_swr, group in joined.groupby("is_swr"):
        frames[is_swr] = pd.DataFrame(
            {
                name: animal_session_values(
                    group[source], group["animal"], group[_WITHIN_ANIMAL], "median"
                )
                for name, source in (
                    ("event", column),
                    ("running", "running_level"),
                    ("rest", "rest_level"),
                )
            }
        )
    return pd.concat(frames, names=["is_swr"])


def power_change_statistics(reference: pd.DataFrame, bands: list[str]) -> pd.DataFrame:
    """Each band's power during events against the same epoch's quiet rest, in dB.

    Parameters
    ----------
    reference : pd.DataFrame, shape (n_epochs, n_columns)
        One row per epoch, with ``animal`` and, for each band and class,
        ``<band>_db_non_swr`` and ``<band>_db_swr``: ``10 log10`` of the mean
        band power over the class's event samples over that of quiet rest.
    bands : list of str
        The bands (``"slow_gamma"``, ``"fast_gamma"``).

    Returns
    -------
    statistics : pd.DataFrame, shape (2 * n_bands, 2)
        Indexed by band and ``is_swr``: ``median`` over epochs, and
        ``n_animals_above_rest``, the animals whose median over their epochs is
        above 0 dB.

    Raises
    ------
    ValueError
        If an epoch has no value for a band and class.
    """
    rows = {}
    for band in bands:
        for is_swr, group in ((False, "non_swr"), (True, "swr")):
            column = f"{band}_db_{group}"
            if reference[column].isna().any():
                raise ValueError(
                    f"{int(reference[column].isna().sum())} epochs have no {column}"
                )
            summary = session_median_statistics(reference[column], reference["animal"])
            rows[(band, is_swr)] = {
                "median": summary["median"],
                "n_animals_above_rest": summary["n_animals_positive"],
            }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis(["band", "is_swr"])


def rest_matched_theta_statistics(events: pd.DataFrame) -> pd.DataFrame:
    """The theta-like share of duration-matched quiet rest, by SWR status.

    An event, or a quiet-rest window, is theta-like when its mean theta reaches
    its epoch's median running theta. Each event has windows of quiet rest with
    as many bins as it has; the reference share of a class is the mean over its
    events of their windows' theta-like share, so it carries the class's
    durations.

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, n_columns)
        One row per event, with ``is_swr``, the session columns
        (:data:`~continuum_swr_replay.tables.session_keys.SESSION_COLUMNS`),
        ``theta_z``, ``theta_running_median``, ``rest_windows`` (how many
        windows it has) and ``rest_theta_like_share`` (the share of them that
        are theta-like).

    Returns
    -------
    statistics : pd.DataFrame, shape (2, 2)
        Indexed by ``is_swr``: ``rest_theta_like_fraction``, the mean over the
        class's events of ``rest_theta_like_share``, and
        ``n_animals_above_rest``, the animals, of those with an event of the
        class, whose share of theta-like events is above their reference
        share. Both are event -> epoch -> animal: the mean over the animal's
        epochs with an event of the class of each epoch's share of theta-like
        events, against the same mean of each epoch's mean
        ``rest_theta_like_share``.

    Raises
    ------
    ValueError
        If an event has no window (no rest stretch of its epoch is as long), or
        lacks its theta, its epoch's running theta, its number of windows or
        their theta-like share.
    """
    missing = (
        events[
            ["theta_z", "theta_running_median", "rest_windows", "rest_theta_like_share"]
        ]
        .isna()
        .sum()
    )
    if missing.any():
        raise ValueError(f"events lack a value:\n{missing[missing > 0]}")
    if (events.rest_windows <= 0).any():
        raise ValueError(
            f"{int((events.rest_windows <= 0).sum())} events have no duration-matched "
            "quiet-rest window"
        )
    usable = events.assign(theta_like=events.theta_z >= events.theta_running_median)
    rows = {}
    for is_swr in (False, True):
        group = usable[usable.is_swr == is_swr]
        per_animal = pd.DataFrame(
            {
                name: animal_session_values(
                    group[source].astype(float),
                    group["animal"],
                    group[_WITHIN_ANIMAL],
                    "mean",
                )
                for name, source in (
                    ("events", "theta_like"),
                    ("rest", "rest_theta_like_share"),
                )
            }
        )
        rows[is_swr] = {
            "rest_theta_like_fraction": group.rest_theta_like_share.mean(),
            "n_animals_above_rest": (per_animal.events > per_animal.rest).sum(),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("is_swr")


# --- spike support ----------------------------------------------------------------


def spike_evidence_statistics(metrics: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """Each group's spike evidence, against matched quiet windows.

    Event-level medians, pooled over epochs; the animals' medians go through
    their epochs.

    Parameters
    ----------
    metrics : pd.DataFrame, shape (n_events_and_windows, n_columns)
        The spike-evidence run's metrics: one row per scored event or quiet
        window, with ``group``, ``animal``, ``epoch_id``, ``n_spikes``,
        ``n_tetrodes``, ``dll_per_spike`` (the log-likelihood ratio per spike of
        the decoded location against the animal's) and ``dist_to_gpi_peak``
        (track distance from the MAP to the ground-process intensity peak, cm).
    baseline : str
        The ``group`` of the matched quiet windows.

    Returns
    -------
    statistics : pd.DataFrame, shape (n_groups, 8)
        Indexed by group:

        - ``n_scored``: rows scored.
        - ``spikes_median`` (spikes), ``tetrodes_median`` (tetrodes),
          ``dll_per_spike_median`` (nats per spike), ``gpi_distance_median_cm``
          (cm): medians over the group's rows.
        - ``n_epochs_scored``: epochs with a row of the group.
        - ``n_epochs_above_baseline``, ``n_animals_above_baseline``: of the
          epochs (animals) with a row of the group, those whose median
          ``dll_per_spike`` is above that of their own quiet windows. NaN for
          the baseline itself. An animal's median is row -> epoch -> animal,
          the median over its epochs of each epoch's median, for the group
          over the epochs with a row of it and for the quiet windows over all
          the animal's epochs.

    Raises
    ------
    ValueError
        If an epoch with a scored row has no ``dll_per_spike`` of a quiet
        window to be compared with.
    """

    def per_animal(rows: pd.DataFrame) -> pd.Series:
        return animal_session_values(
            rows["dll_per_spike"], rows["animal"], rows["epoch_id"], "median"
        )

    base = metrics[metrics["group"] == baseline]
    epoch_base = base.groupby("epoch_id")["dll_per_spike"].median()
    animal_base = per_animal(base)
    without = epoch_base.reindex(metrics["epoch_id"].unique()).isna()
    if without.any():
        raise ValueError(
            f"{int(without.sum())} epochs have no {baseline!r} dll_per_spike to "
            f"compare with: {sorted(without.index[without])}"
        )
    rows = {}
    for group, group_rows in metrics.groupby("group"):
        epoch_median = group_rows.groupby("epoch_id")["dll_per_spike"].median()
        animal_median = per_animal(group_rows)
        is_baseline = group == baseline
        rows[group] = {
            "n_scored": len(group_rows),
            "spikes_median": group_rows["n_spikes"].median(),
            "tetrodes_median": group_rows["n_tetrodes"].median(),
            "dll_per_spike_median": group_rows["dll_per_spike"].median(),
            "gpi_distance_median_cm": group_rows["dist_to_gpi_peak"].median(),
            "n_epochs_scored": len(epoch_median),
            "n_epochs_above_baseline": np.nan
            if is_baseline
            else (epoch_median > epoch_base.reindex(epoch_median.index)).sum(),
            "n_animals_above_baseline": np.nan
            if is_baseline
            else (animal_median > animal_base.reindex(animal_median.index)).sum(),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("group")


# --- one class or two -------------------------------------------------------------


def peak_ripple_mode_statistics(
    distribution: pd.DataFrame, n_animals: int, alpha: float = 0.05
) -> dict[str, float]:
    """Whether the events' peak ripple power has one mode, and where it is.

    Parameters
    ----------
    distribution : pd.DataFrame, shape (n_rows, n_columns)
        The one-class-or-two analysis's per-animal distribution tests, with
        ``animal`` (an animal, or ``"pooled"``), ``power`` (the power measure),
        ``dip_p`` (Hartigan's dip test) and ``main_mode_z`` (the KDE's main
        mode, z).
    n_animals : int
        The animals the peak power must have a test for.
    alpha : float
        The level below which the dip test rejects one mode.

    Returns
    -------
    statistics : dict
        - ``n_animals_unimodal``: animals, of ``n_animals``, in which the dip
          test does not reject one mode of the peak power (``dip_p`` at or
          above ``alpha``).
        - ``mode_animal_min``, ``mode_animal_max``: the lowest and highest of
          the animals' main modes, z.

    Raises
    ------
    ValueError
        If the peak power does not have one test for each of ``n_animals``
        animals, or a test has no ``dip_p`` or no ``main_mode_z``.
    """
    peak = distribution[
        (distribution.animal != "pooled") & (distribution.power == "peak")
    ]
    if len(peak) != n_animals or peak.animal.nunique() != n_animals:
        raise ValueError(
            f"the peak-power tests do not hold one row for each of {n_animals} "
            f"animals: {sorted(peak.animal)}"
        )
    for column in ("dip_p", "main_mode_z"):
        if peak[column].isna().any():
            raise ValueError(
                f"{int(peak[column].isna().sum())} animals' peak-power tests have "
                f"no {column}"
            )
    return {
        "n_animals_unimodal": (peak.dip_p >= alpha).sum(),
        "mode_animal_min": peak.main_mode_z.min(),
        "mode_animal_max": peak.main_mode_z.max(),
    }


def animal_signal_counts(epoch_counts: pd.DataFrame) -> pd.DataFrame:
    """Each animal's time bins, and its non-local ones, by the level of a signal
    (ripple power or MUA).

    Parameters
    ----------
    epoch_counts : pd.DataFrame, shape (n_epochs * n_signal_bins, n_columns)
        One row per epoch and signal bin (``nonlocal_by_signal.nonlocal_counts``),
        with ``animal``, ``signal_low``, ``signal_high``, ``n_bins`` and
        ``n_nonlocal``.

    Returns
    -------
    counts : pd.DataFrame, shape (n_animals * n_signal_bins, 7)
        One row per animal and signal bin, in that order: ``animal``,
        ``signal_low``, ``signal_high``, ``n_bins`` and ``n_nonlocal`` summed
        over the animal's epochs, ``share`` (the non-local share of the bin's
        time bins, NaN where the animal has none) and ``nonlocal_fraction``
        (the bin's share of the animal's non-local bins).
    """
    counts = epoch_counts.groupby(
        ["animal", "signal_low", "signal_high"], as_index=False
    )[["n_bins", "n_nonlocal"]].sum()
    counts["share"] = counts["n_nonlocal"] / counts["n_bins"].where(
        counts["n_bins"] > 0
    )
    counts["nonlocal_fraction"] = counts["n_nonlocal"] / counts.groupby("animal")[
        "n_nonlocal"
    ].transform("sum")
    return counts


def _require_edge(counts: pd.DataFrame, threshold: float) -> None:
    if not (counts["signal_low"] == threshold).any():
        raise ValueError(f"{threshold} is not an edge of the signal's bins")


def nonlocal_below_threshold_statistics(
    counts: pd.DataFrame, threshold: float
) -> dict[str, float]:
    """How much of the non-local time has a signal below a threshold.

    Parameters
    ----------
    counts : pd.DataFrame, shape (n_animals * n_signal_bins, n_columns)
        :func:`animal_signal_counts`.
    threshold : float
        The signal's level, z; an edge of its bins.

    Returns
    -------
    statistics : dict
        - ``fraction_animal_mean``: the mean over animals of each animal's
          share of its non-local time bins whose signal is below ``threshold``.
        - ``fraction_animal_min``, ``fraction_animal_max``: the lowest and
          highest of the animals' shares.
        - ``n_animals``: the animals.

    Raises
    ------
    ValueError
        If ``threshold`` is not an edge of the signal's bins.
    """
    _require_edge(counts, threshold)
    below = (
        counts[counts["signal_high"] <= threshold]
        .groupby("animal")["nonlocal_fraction"]
        .sum()
    )
    return {
        "fraction_animal_mean": float(below.mean()),
        "fraction_animal_min": float(below.min()),
        "fraction_animal_max": float(below.max()),
        "n_animals": int(below.size),
    }


def nonlocal_share_above_statistics(
    counts: pd.DataFrame, threshold: float
) -> dict[str, float]:
    """The share of the time bins with a signal at or above a level that are
    decoded non-local.

    Parameters
    ----------
    counts : pd.DataFrame, shape (n_animals * n_signal_bins, n_columns)
        :func:`animal_signal_counts`.
    threshold : float
        The level, z; an edge of the signal's bins.

    Returns
    -------
    statistics : dict
        - ``share_animal_mean``: the mean over animals of each animal's
          non-local bins at or above ``threshold`` over its bins there.
        - ``share_animal_min``, ``share_animal_max``: the lowest and highest
          animal.
        - ``n_animals``: the animals.

    Raises
    ------
    ValueError
        If ``threshold`` is not an edge of the signal's bins.
    """
    _require_edge(counts, threshold)
    above = (
        counts[counts["signal_low"] >= threshold]
        .groupby("animal")[["n_nonlocal", "n_bins"]]
        .sum()
    )
    share = above["n_nonlocal"] / above["n_bins"]
    return {
        "share_animal_mean": float(share.mean()),
        "share_animal_min": float(share.min()),
        "share_animal_max": float(share.max()),
        "n_animals": int(share.size),
    }


def main_step_fits(
    steps: pd.DataFrame, n_knots: int, window_z: float, n_animals: int
) -> pd.DataFrame:
    """The step fits of the main setting: the Kay criterion, no covariate,
    ``n_knots`` knots and a window of ``window_z`` around the threshold.

    Parameters
    ----------
    steps : pd.DataFrame, shape (n_fits, n_columns)
        The one-class-or-two analysis's step fits (as
        :func:`threshold_step_statistics` takes them).
    n_knots : int
        The spline's knots in the main setting.
    window_z : float
        The half-width of the main setting's window around the threshold, z.
    n_animals : int
        The animals every property must have one fit for.

    Returns
    -------
    fits : pd.DataFrame, shape (n_properties * n_animals, n_columns)
        The rows of ``steps`` in the main setting, sorted by property and
        animal: one per property and animal.

    Raises
    ------
    ValueError
        If a property does not have one fit for each of ``n_animals`` animals
        in the main setting.
    """
    main = steps[
        (steps.n_knots == n_knots)
        & (steps.power == "criterion")
        & np.isclose(steps.window_z, window_z)
        & (steps.covariate == "none")
    ]
    fits = main.groupby("property").agg(
        n_fits=("animal", "size"), n_animals_fitted=("animal", "nunique")
    )
    if not (fits == n_animals).all(axis=None):
        raise ValueError(
            f"not every property has one fit for each of {n_animals} animals:\n{fits}"
        )
    return main.sort_values(["property", "animal"])


def threshold_step_statistics(
    steps: pd.DataFrame, n_knots: int, window_z: float, n_animals: int
) -> pd.DataFrame:
    """For each property, the animals with a step at the SWR threshold.

    Parameters
    ----------
    steps : pd.DataFrame, shape (n_fits, n_columns)
        The one-class-or-two analysis's step fits: one row per animal, property
        and setting, with ``n_knots``, ``power``, ``window_z``, ``covariate``,
        ``property``, ``has_step`` (the step term's interval excludes 0 and it
        lowers the held-out error) and ``step_share_overall`` (the term over
        the animal's SWR minus non-SWR difference).
    n_knots : int
        The spline's knots in the main setting.
    window_z : float
        The half-width of the main setting's window around the threshold, z.
    n_animals : int
        The animals every property must have one fit for.

    Returns
    -------
    statistics : pd.DataFrame, shape (n_properties, 3)
        Indexed by property, over :func:`main_step_fits` (the Kay criterion, no
        covariate), each of ``n_animals`` animals: ``n_animals_with_step``;
        ``n_animals_same_sign``, the animals whose step points the way of their
        SWR minus non-SWR difference (a positive share); and
        ``step_share_median``, the median over animals of the step as a share
        of that difference.

    Raises
    ------
    ValueError
        If a property does not have one fit for each of ``n_animals`` animals
        in the main setting.
    """
    main = main_step_fits(steps, n_knots, window_z, n_animals)
    return main.groupby("property").agg(
        n_animals_with_step=("has_step", "sum"),
        n_animals_same_sign=("step_share_overall", lambda share: (share > 0).sum()),
        step_share_median=("step_share_overall", "median"),
    )


def most_steps(
    statistics: pd.DataFrame, properties: list[str]
) -> tuple[int, list[str]]:
    """The most animals with a step in any of ``properties``
    (:func:`threshold_step_statistics`), and the properties that attain it."""
    with_step = statistics.loc[properties, "n_animals_with_step"]
    return int(with_step.max()), list(with_step.index[with_step == with_step.max()])


# --- properties -------------------------------------------------------------------


def class_medians(events: pd.DataFrame, column: str) -> dict[str, float]:
    """A per-event property's median in non-SWR and in SWR events.

    Plain medians of all events with a value, pooled over epochs and animals;
    not duration-controlled.

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, n_columns)
        One row per event, with ``is_swr`` and ``column``.
    column : str
        The property.

    Returns
    -------
    statistics : dict
        ``median_non_swr`` and ``median_swr``.
    """
    median = events.dropna(subset=[column]).groupby("is_swr")[column].median()
    return {"median_non_swr": median[False], "median_swr": median[True]}


def animal_class_medians(events: pd.DataFrame, column: str) -> pd.DataFrame:
    """Each animal's median of a per-event property in non-SWR and in SWR events.

    An animal's median of a class is event -> epoch -> animal: the median over
    its epochs with an event of the class (with a value) of each epoch's
    median.

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, n_columns)
        One row per event, with ``is_swr``, the session columns
        (:data:`~continuum_swr_replay.tables.session_keys.SESSION_COLUMNS`) and
        ``column``.
    column : str
        The property.

    Returns
    -------
    medians : pd.DataFrame, shape (n_animals, 2)
        Indexed by animal, with the columns ``non_swr`` and ``swr``. An animal
        without an event of a class (with a value) has no median there.
    """
    return pd.DataFrame(
        {
            _CLASS_WORDS[is_swr]: animal_session_values(
                group[column], group["animal"], group[_WITHIN_ANIMAL], "median"
            )
            for is_swr, group in events.groupby("is_swr")
        }
    )


HPD_TIGHT_CM = 50.0
"""The HPD coverage below which an event's posterior counts as tight (cm): it
lies between the two groups of events, a tight one near 20-25 cm and a diffuse
one spread over most of the track."""

HPD_BIN_EDGES_CM = np.geomspace(5.0, 500.0, 25)
"""The HPD coverage bins of the posterior-spread panel (cm), log-spaced: every
event's coverage lies inside them."""


def _hpd_table(events: pd.DataFrame) -> pd.DataFrame:
    """The events with a coverage: ``animal``, ``epoch_id`` (one label per
    session), ``group`` (``"no SWR"`` or ``"SWR"``) and ``hpd_coverage_cm``."""
    known = events.dropna(subset=["hpd_coverage_cm"])
    return pd.DataFrame(
        {
            "animal": known["animal"].to_numpy(),
            "epoch_id": known[list(SESSION_COLUMNS)]
            .astype(str)
            .agg("_".join, axis=1)
            .to_numpy(),
            "group": np.where(known["is_swr"].astype(bool), "SWR", "no SWR"),
            "hpd_coverage_cm": known["hpd_coverage_cm"].to_numpy(dtype=float),
        }
    )


def hpd_distribution(
    events: pd.DataFrame, edges: np.ndarray = HPD_BIN_EDGES_CM
) -> pd.DataFrame:
    """Each animal's distribution of the events' HPD coverage (the length of
    track that holds 95 % of the decoded probability, averaged over the
    event's bins), by class.

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, n_columns)
        One row per event, with ``animal``, the session columns, ``is_swr`` and
        ``hpd_coverage_cm`` (an event without one is left out).
    edges : np.ndarray, shape (n_edges,)
        Ascending bin edges (cm); a bin holds its low edge and not its high one.

    Returns
    -------
    pd.DataFrame
        One row per animal, class and bin: ``animal``, ``group``, ``bin_low``,
        ``bin_high`` and ``percent`` (the percent of the animal's events of the
        class in the bin).
    """
    table = _hpd_table(events)
    rows = []
    for (animal, group), mine in table.groupby(["animal", "group"]):
        counts, _ = np.histogram(mine["hpd_coverage_cm"], bins=edges)
        rows.append(
            pd.DataFrame(
                {
                    "animal": animal,
                    "group": group,
                    "bin_low": edges[:-1],
                    "bin_high": edges[1:],
                    "percent": 100 * counts / len(mine),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def hpd_tight_share_statistics(
    events: pd.DataFrame,
    *,
    cut_cm: float = HPD_TIGHT_CM,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """The share of each class's events whose HPD coverage is below a cut.

    Each epoch's share of its events of the class below ``cut_cm`` is reduced
    as the project reduces a per-epoch share
    (``likelihood_advantage.likelihood_advantage_statistics`` over one row per
    epoch, as ``restriction_totals.outside_swr_gain_statistics`` reduces its
    shares): each animal's median over its epochs, the mean over animals and
    its 95 % nested bootstrap interval.

    Parameters
    ----------
    events : pd.DataFrame
        As :func:`hpd_distribution` takes them.
    cut_cm : float, optional
    n_boot, seed : int, optional

    Returns
    -------
    dict
        ``rows``: indexed by group (``"no SWR"``, ``"SWR"``), with
        ``likelihood_advantage_statistics``' columns; and
        ``n_animals_non_swr_higher``: the animals whose no-SWR share is above
        their SWR share.
    """
    table = _hpd_table(events)
    shares = (
        table.assign(tight=table["hpd_coverage_cm"] < cut_cm)
        .groupby(["animal", "epoch_id", "group"], as_index=False)["tight"]
        .mean()
    )
    rows = likelihood_advantage_statistics(
        shares, column="tight", n_boot=n_boot, seed=seed
    ).loc[["no SWR", "SWR"]]
    by_group = {
        group: pd.Series(rows.loc[group, "animals"], rows.loc[group, "animal_names"])
        for group in rows.index
    }
    higher = by_group["no SWR"] > by_group["SWR"].reindex(by_group["no SWR"].index)
    return {"rows": rows, "n_animals_non_swr_higher": int(higher.sum())}


def n_animals_in_direction(events: pd.DataFrame, column: str, direction: str) -> int:
    """The animals whose own non-SWR median of a per-event property is
    ``direction`` than their SWR median.

    The medians are :func:`animal_class_medians`. An animal without one of the
    classes is in neither direction.

    Parameters
    ----------
    events : pd.DataFrame, shape (n_events, n_columns)
        One row per event, with ``is_swr``, the session columns
        (:data:`~continuum_swr_replay.tables.session_keys.SESSION_COLUMNS`) and
        ``column``.
    column : str
        The property.
    direction : {"lower", "higher"}
        The direction of the non-SWR median against the SWR median.

    Returns
    -------
    int

    Raises
    ------
    ValueError
        If ``direction`` is neither ``"lower"`` nor ``"higher"``.
    """
    if direction not in ("lower", "higher"):
        raise ValueError(f"direction {direction!r} is neither 'lower' nor 'higher'")
    medians = animal_class_medians(events, column)
    compare = pd.Series.lt if direction == "lower" else pd.Series.gt
    return int(compare(medians["non_swr"], medians["swr"]).sum())


# --- where content sits -----------------------------------------------------------

OCCUPANCY_QUADRANTS = ("remote_well", "own_well", "remote_path", "near_off")
"""The four places a content time bin is in, by its distance to the animal and
to the nearest well: at a remote well, at the animal's own well, on a remote
path, or near the animal off the wells."""

OCCUPANCY_PLACES = ("remote_well", "own_well", "at_well", "remote_path", "near_off")
"""Where content can sit: the four :data:`OCCUPANCY_QUADRANTS`, and at either
well (``at_well``)."""

FAR_CM = 20.0
"""Content farther than this from the animal (track distance, cm) is remote."""

AT_WELL_CM = 20.0
"""Content nearer than this to the nearest reward well (track distance, cm) is
at a well."""


def quadrant_fractions(d_animal: np.ndarray, d_well: np.ndarray) -> dict[str, float]:
    """Fraction of content *timebins* in each (far × at-well) quadrant.

    Timebin-weighted (where the decoded content spends its time), not
    event-weighted; longer events contribute more. Aggregation
    (:func:`content_occupancy_shares`) keeps this convention (weights
    per-session fractions by timebin count).

    Parameters
    ----------
    d_animal : np.ndarray, shape (n_bins,)
        Each non-local time bin's track distance (cm) from the decoded content
        to the animal. Far means above :data:`FAR_CM`.
    d_well : np.ndarray, shape (n_bins,)
        The same bins' track distance (cm) from the content to the nearest
        reward well. At a well means below :data:`AT_WELL_CM`.

    Returns
    -------
    dict of str to float
        The share of the bins in each of :data:`OCCUPANCY_QUADRANTS`.
    """
    far = d_animal > FAR_CM
    at_well = d_well < AT_WELL_CM
    return {
        "remote_well": float((far & at_well).mean()),
        "own_well": float((~far & at_well).mean()),
        "remote_path": float((far & ~at_well).mean()),
        "near_off": float((~far & ~at_well).mean()),
    }


def arm_occupancy_shares(event_arm: pd.DataFrame) -> pd.DataFrame:
    """Per-arm posterior-mass occupancy, normalized within SWR group.

    Parameters
    ----------
    event_arm : pd.DataFrame, shape (n_events * n_arms, n_columns)
        The event-arm table: ``is_swr``, ``candidate_arm`` and
        ``arm_posterior_mass`` per event and arm.

    Returns
    -------
    shares : pd.DataFrame, shape (n_arms, n_groups)
        Indexed by ``candidate_arm``, one column per group (``"nonSWR"``,
        ``"SWR"``): the group's posterior mass on the arm over its mass on every
        arm. NaN for an arm the group has no row for.
    """
    ea = event_arm.assign(
        group=np.where(event_arm["is_swr"].astype(bool), "SWR", "nonSWR")
    )
    occ = ea.groupby(["group", "candidate_arm"])["arm_posterior_mass"].sum().unstack(0)
    return occ / occ.sum()


def content_occupancy_shares(occupancy: pd.DataFrame) -> pd.DataFrame:
    """Where content sits against the animal and the reward wells.

    Parameters
    ----------
    occupancy : pd.DataFrame, shape (n_sessions * n_groups, n_columns)
        The content-occupancy table: per session and ``group`` (``"SWR"``,
        ``"nonSWR"``), ``n`` non-local time bins and the share of them at each
        place (``remote_well``, ``own_well``, ``remote_path``, ``near_off``).

    Returns
    -------
    shares : pd.DataFrame, shape (n_groups, 5)
        For each group, the share of its time bins at each of
        :data:`OCCUPANCY_PLACES`, weighted by time bins over sessions.

    Raises
    ------
    ValueError
        If a session with time bins has no share at a place.
    """
    return _weighted_shares(occupancy, ["group"])


def animal_occupancy_shares(occupancy: pd.DataFrame, group: str) -> pd.DataFrame:
    """Each animal's share of a group's content at each place.

    An animal's share is session -> animal: the mean over its sessions with
    time bins of the group of each session's share, each session weighing the
    same however many time bins it holds (``at_well`` is a session's
    ``remote_well`` plus ``own_well``).

    Parameters
    ----------
    occupancy : pd.DataFrame, shape (n_sessions * n_groups, n_columns)
        The content-occupancy table (:func:`content_occupancy_shares`), with
        ``animal`` and ``epoch_id``.
    group : str
        ``"SWR"`` or ``"nonSWR"``.

    Returns
    -------
    shares : pd.DataFrame, shape (n_animals, 5)
        Indexed by ``animal`` in sorted order, one column per place of
        :data:`OCCUPANCY_PLACES`. An animal with no session of the group with
        time bins is not in it.

    Raises
    ------
    ValueError
        If a session with time bins has no share at a place.
    """
    with_bins = _sessions_with_bins(occupancy)
    rows = with_bins[with_bins["group"] == group]
    rows = rows.assign(at_well=rows["remote_well"] + rows["own_well"])
    return pd.DataFrame(
        {
            place: animal_session_values(
                rows[place], rows["animal"], rows["epoch_id"], "mean"
            )
            for place in OCCUPANCY_PLACES
        },
        columns=list(OCCUPANCY_PLACES),
    ).rename_axis("animal")


def n_animals_non_swr_more_at_well(occupancy: pd.DataFrame) -> int:
    """The animals whose non-SWR content is at a well more often than their SWR
    content.

    An animal's share at a well of a group is its ``at_well`` of
    :func:`animal_occupancy_shares`: the mean over its sessions of each
    session's share.

    Parameters
    ----------
    occupancy : pd.DataFrame, shape (n_sessions * n_groups, n_columns)
        The content-occupancy table (:func:`content_occupancy_shares`), with
        ``animal`` and ``epoch_id``.

    Returns
    -------
    int
        An animal without one of the groups is not counted.

    Raises
    ------
    ValueError
        If a session with time bins has no share at a place.
    """
    with_bins = _sessions_with_bins(occupancy)
    at_well = pd.DataFrame(
        {
            group: animal_occupancy_shares(with_bins, group)["at_well"]
            for group in sorted(with_bins["group"].unique())
        }
    )
    return int((at_well["nonSWR"] > at_well["SWR"]).sum())


def _sessions_with_bins(occupancy: pd.DataFrame) -> pd.DataFrame:
    """The rows of ``occupancy`` with time bins; raise if one has no share at a
    place."""
    with_bins = occupancy[occupancy["n"] > 0]
    without = with_bins[list(OCCUPANCY_QUADRANTS)].isna().any(axis=1)
    if without.any():
        raise ValueError(
            f"{int(without.sum())} sessions with time bins have no share at a place"
        )
    return with_bins


def _weighted_shares(occupancy: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """The share of time bins at each place, weighted by time bins, per ``by``."""
    _sessions_with_bins(occupancy)
    occupancy = occupancy.assign(
        at_well=occupancy["remote_well"] + occupancy["own_well"]
    )

    def weighted(frame: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                place: float((frame[place] * frame["n"]).sum() / frame["n"].sum())
                for place in OCCUPANCY_PLACES
            }
        )

    return occupancy.groupby(by).apply(weighted, include_groups=False)
