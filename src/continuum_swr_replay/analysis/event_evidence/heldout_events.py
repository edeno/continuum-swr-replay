"""Non-local events chosen from held-out data: the canonical event rule applied to
the fold fit's held-out state probabilities.

The restriction run (:mod:`.nonlocal_restriction`, the fold fit) scores each fold
of whole trials under the whole decoder fit on the other folds and stores, per
bin, the ``full`` model's held-out smoothed state probabilities. The canonical
events were found from a decode fit on all the data, so a gain read over their
windows is read on time chosen by a decoder that saw those bins. Here the
events are found from the held-out probabilities instead, by the canonical rule,
on the canonical decode frame.

A fold is whole trial segments (the run to a well and the whole stay there),
but a segment longer than ``regression.folds.MAX_SEGMENT_S`` (30 s) is cut into
pieces of at most 30 s that can fall in different folds
(``regression.folds.trial_folds``). So a held-out event can span two folds (a
small share do); its probability trace and its per-bin scores then come from
two fits, each bin still scored under a fit that did not see it.

The two steps:

1. :func:`labeled_frame`: ``nonlocal_event_rule.detect_nonlocal_events`` on the
   summed non-local probability (strictly above 0.8 for at least 7 samples, the
   animal immobile throughout); the labels and the two non-local probabilities
   are put into the decode frame as the decoder saved it, and the frame is read
   through ``analyzed_time.with_swr_source`` (the probability contract is
   validated, events in excluded time are dropped, a kept event that overlaps
   an SWR or MUA event dropped for excluded time is dropped, the SWR labels are
   the SWR source's);
2. :func:`event_rows`: one row per event, its SWR group by strict
   positive-duration overlap with an SWR of the source
   (``nonlocal_event_rule.event_strict_overlap_keys``, the keys the canonical
   table's ``is_swr`` is made from), and whether it starts inside a well visit
   (its first bin is inside one).

Applied to the canonical decode's own probabilities, the two steps give the
canonical labels exactly (the runner checks this in every epoch).

:func:`shared_bin_events` says which events of one label vector share at least
one bin with an event of another on the same time axis ("found again"), and
:func:`found_again_statistics` summarizes it over the cohort.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from continuum_swr_replay.analysis.event_definition.analyzed_time import (
    with_swr_source,
)
from continuum_swr_replay.analysis.event_definition.nonlocal_event_rule import (
    detect_nonlocal_events,
    event_strict_overlap_keys,
)
from continuum_swr_replay.analysis.event_evidence.state_evidence.windows import (
    SWR_LABELS,
)
from continuum_swr_replay.analysis.shared_statistics import animal_session_values

LABEL_COLUMN = "non_local_event"
"""The decode frame's non-local event labels."""

PROBABILITY_COLUMNS = ("Non-Local_Continuous_prob", "Non-Local_Fragmented_prob")
"""The decode frame's two non-local state probabilities."""

GROUPS = ("all", SWR_LABELS[False], SWR_LABELS[True])
"""The groups :func:`found_again_statistics` reports: every event, then each SWR
group."""


def labeled_frame(
    raw: pd.DataFrame, continuous: np.ndarray, fragmented: np.ndarray
) -> pd.DataFrame:
    """The decode frame with the non-local events of a probability trace, read
    through the SWR source.

    Parameters
    ----------
    raw : pd.DataFrame
        One epoch's decode frame as the decoder saved it (not yet read through
        the SWR source), indexed by time (s), with ``head_speed`` (cm/s) and
        the columns ``analyzed_time.with_swr_source`` reads.
    continuous, fragmented : np.ndarray, shape (n_time,)
        P(Non-Local Continuous) and P(Non-Local Fragmented) at each row of
        ``raw``.

    Returns
    -------
    pd.DataFrame
        ``with_swr_source`` of ``raw`` with ``non_local_event`` the labels of
        ``detect_nonlocal_events`` on ``continuous + fragmented`` and the two
        probability columns (:data:`PROBABILITY_COLUMNS`) set to these;
        ``raw`` is not changed. Every other column (``Local_prob``,
        ``No-Spike_prob``, ``map_estimate``, the posterior statistics, ...) is
        still the saved decode's.

    Notes
    -----
    When the trace is a held-out decode's, the frame is not that decode: only
    ``non_local_event`` and the two non-local probabilities are from it.
    """
    _, labels = detect_nonlocal_events(
        xr.DataArray(continuous + fragmented, dims="time"),
        raw["head_speed"].to_numpy(dtype=float),
        raw.index,
    )
    frame = raw.copy(deep=False)
    frame[LABEL_COLUMN] = labels.to_numpy()
    frame[PROBABILITY_COLUMNS[0]] = continuous
    frame[PROBABILITY_COLUMNS[1]] = fragmented
    return with_swr_source(frame)


def label_bins(labels: np.ndarray) -> pd.DataFrame:
    """Each event's first and last bin.

    Parameters
    ----------
    labels : np.ndarray of int, shape (n_time,)
        An event label per bin, 0 outside events; each event's bins are one
        run.

    Returns
    -------
    pd.DataFrame
        One row per label, sorted: ``label``, ``s`` and ``e`` (the positions of
        its first and last bin, inclusive).

    Raises
    ------
    ValueError
        If an event's bins are not one run.
    """
    labels = np.asarray(labels)
    inside = np.flatnonzero(labels > 0)
    ids, first, counts = np.unique(
        labels[inside], return_index=True, return_counts=True
    )
    # the last bin of each label: its first bin in the reversed positions
    _, from_end = np.unique(labels[inside][::-1], return_index=True)
    s = inside[first]
    e = inside[inside.size - 1 - from_end]
    if (e - s + 1 != counts).any():
        raise ValueError("an event's bins are not one run")
    return pd.DataFrame({"label": ids, "s": s, "e": e})


def event_rows(frame: pd.DataFrame, in_well_visit: np.ndarray) -> pd.DataFrame:
    """One row per non-local event of a frame read through the SWR source.

    Parameters
    ----------
    frame : pd.DataFrame
        A decode frame read through the SWR source (:func:`labeled_frame`),
        indexed by time (s), with ``non_local_event``, ``SWR_event`` and the
        session columns ``animal``, ``day`` and ``epoch``.
    in_well_visit : np.ndarray of bool, shape (n_time,)
        Whether each row of ``frame`` is inside a well visit (the per-bin
        facts' ``in_well_visit``).

    Returns
    -------
    pd.DataFrame
        One row per event, by label: ``label``, ``s``, ``e`` (first and last
        bin, positions in ``frame``), ``event_start_s``, ``event_end_s`` (their
        times), ``group`` (``"SWR"`` when the event strictly overlaps an SWR of
        the source, else ``"no SWR"``, ``state_evidence.windows.SWR_LABELS``)
        and ``starts_in_well_visit`` (its first bin is inside a visit).

    Raises
    ------
    ValueError
        If ``in_well_visit`` is not one value per row of ``frame``.
    """
    in_well_visit = np.asarray(in_well_visit, dtype=bool)
    if in_well_visit.shape != (len(frame),):
        raise ValueError("in_well_visit is not one value per row of the frame")
    rows = label_bins(frame[LABEL_COLUMN].to_numpy())
    swr = {
        key[-1] for key in event_strict_overlap_keys(frame, LABEL_COLUMN, "SWR_event")
    }
    time = frame.index.to_numpy(dtype=float)
    is_swr = rows["label"].isin(swr).to_numpy()
    return rows.assign(
        event_start_s=time[rows["s"].to_numpy()],
        event_end_s=time[rows["e"].to_numpy()],
        group=np.where(is_swr, SWR_LABELS[True], SWR_LABELS[False]),
        starts_in_well_visit=in_well_visit[rows["s"].to_numpy()],
    )


def shared_bin_events(labels: np.ndarray, other: np.ndarray) -> pd.Series:
    """Whether each event of ``labels`` shares a bin with an event of ``other``.

    Parameters
    ----------
    labels, other : np.ndarray of int, shape (n_time,)
        Two event label vectors on the same time axis, 0 outside events.

    Returns
    -------
    pd.Series of bool
        Indexed by each label of ``labels`` (sorted): True when at least one of
        its bins is inside an event of ``other``.

    Raises
    ------
    ValueError
        If the two vectors are not on the same axis (not of one shape).
    """
    labels, other = np.asarray(labels), np.asarray(other)
    if labels.shape != other.shape:
        raise ValueError("the two label vectors are not on the same time axis")
    ids = np.unique(labels[labels > 0])
    shared = np.unique(labels[(labels > 0) & (other > 0)])
    return pd.Series(np.isin(ids, shared), index=pd.Index(ids, name="label"))


def found_again_statistics(events: pd.DataFrame, found: str) -> pd.DataFrame:
    """How many events of a set an event of the other set shares a bin with,
    over every event and by SWR group.

    Parameters
    ----------
    events : pd.DataFrame
        One row per event: ``epoch_id``, ``animal``, ``group`` and the bool
        column ``found``.
    found : str
        The column that says whether the event shares a bin with an event of
        the other set.

    Returns
    -------
    pd.DataFrame
        Indexed by ``group`` (:data:`GROUPS`): ``n_events``, ``n_found``,
        ``share`` (``n_found`` over ``n_events``, pooled over epochs and
        animals; NaN without events), ``share_animal_min`` and
        ``share_animal_max`` (the lowest and highest of the animals' own
        shares, each the mean over the animal's epochs of the epoch's share,
        event -> epoch -> animal) and ``n_animals``.
    """
    rows = {}
    for group in GROUPS:
        mine = events if group == "all" else events[events["group"] == group]
        hit = mine[found].astype(bool)
        by_animal = animal_session_values(
            hit.astype(float), mine["animal"], mine["epoch_id"], "mean"
        )
        rows[group] = {
            "n_events": len(mine),
            "n_found": int(hit.sum()),
            "share": hit.mean() if len(mine) else np.nan,
            "share_animal_min": by_animal.min(),
            "share_animal_max": by_animal.max(),
            "n_animals": len(by_animal),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("group")
