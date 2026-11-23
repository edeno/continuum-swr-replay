"""Shared content: during an event, do the tetrodes point to the same remote place?

The held-out tetrode run (:mod:`.tetrode_holdout`) scores each left-out tetrode
under the other tetrodes' position distribution against no place preference:
the window's ``agreement``, above zero only when the left-out tetrode's activity
is where the others point. Over all events it is near zero, because in many
events the others' distribution is spread over much of the track and there is
nothing specific to agree with. This module groups the events by that width.

An event's width is its ``hpd_cm`` from ``tetrode_holdout.window_scores`` (a
run made with ``--extra-scores``): the mean over its left-out tetrodes of the
mean over the window's scored bins of the 95 % HPD length of the other
tetrodes' position distribution (``tetrode_holdout.hpd_lengths``). The mean
over tetrodes is the natural summary: each left-out tetrode has its own
"others", and the agreement is summed over the same tetrodes. An event's
agreement per spike is ``agreement_per_spike``: summed over tetrodes and bins,
divided by the event's spikes, as the gain per spike is.

Against no place preference the agreement of a wide event is below zero even
when nothing is shared, so each bin also carries the shuffled-event reference
(``agreement_null_per_spike``: the window's spikes scored under another event
window's estimate of the same group and fold) and the agreement minus that
reference, event by event (``agreement_minus_null_per_spike``)
(:func:`shared_content_statistics`).

The events are grouped by width in fixed bins, :data:`WIDTH_BINS_CM`.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_evidence.likelihood_advantage import (
    likelihood_advantage_statistics,
)
from continuum_swr_replay.analysis.shared_statistics import animal_session_values

WIDTH_BINS_CM = (0.0, 25.0, 50.0, 100.0, 200.0, np.inf)
"""The width bins' edges (cm), each bin closed on the left: under 25, 25 to
50, 50 to 100, 100 to 200, and 200 or more."""

WIDTH_BIN_LABELS = ("<25", "25–50", "50–100", "100–200", "≥200")
"""Each width bin's label, in order."""

WIDTH_COLUMN = "hpd_cm"
"""The ``tetrode_holdout.window_scores`` column the events are binned on: the
95 % HPD length of the other tetrodes' position estimate (Figure 2h's x
axis)."""

SHARED_CONTENT_COLUMNS = {
    "agreement": "agreement_per_spike",
    "reference": "agreement_null_per_spike",
    "difference": "agreement_minus_null_per_spike",
}
"""The scores per spike of :func:`shared_content_statistics`, by name: the
agreement, the shuffled-event reference, and the agreement minus the
reference event by event."""


def width_bins(width_cm: pd.Series) -> pd.Series:
    """Each event's width bin, the label of :data:`WIDTH_BIN_LABELS` it falls
    in (a categorical in the bins' order)."""
    return pd.cut(
        width_cm, list(WIDTH_BINS_CM), right=False, labels=list(WIDTH_BIN_LABELS)
    )


def agreement_by_width_statistics(
    windows: pd.DataFrame,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    column: str = "agreement_per_spike",
) -> pd.DataFrame:
    """A score per spike of the events of each SWR group, by width bin.

    Parameters
    ----------
    windows : pd.DataFrame
        ``tetrode_holdout.window_scores`` of the events (``kind`` ``"event"``):
        ``animal``, ``epoch_id``, ``group``, :data:`WIDTH_COLUMN` and ``column``
        (NaN for an event without a value, which counts in no statistic).
    n_boot, seed : int, optional
        The bootstrap's replicates and seed, the same for every row.
    column : str, optional
        The score per spike, by default ``agreement_per_spike`` (one of
        :data:`SHARED_CONTENT_COLUMNS`' values).

    Returns
    -------
    pd.DataFrame
        Indexed by ``width_bin`` and ``group``:

        - ``estimate``, ``ci_low``, ``ci_high``, ``animals``, ``animal_names``,
          ``n_animals``, ``n_events`` and ``n_scored``: the score per spike as
          the held-out gain's statistic computes it
          (``likelihood_advantage_statistics``: each epoch's median event, each
          animal's median epoch, the mean over animals, its 95 % interval),
          and ``n_animals_positive``, the animals whose value is above zero;
        - ``n_epochs``: the epochs with an event of the group in the bin (an
          epoch without one has no row);
        - ``share_positive``, ``share_positive_animal_min``,
          ``share_positive_animal_max``: the share of events with a score
          above zero, per epoch, then each animal's median, then the mean over
          animals, and the lowest and highest animal;
        - ``share_of_events``: the share of the group's scored events that fall
          in the bin, pooled over every event;
        - ``share_of_events_animal_mean``, ``share_of_events_animal_min``,
          ``share_of_events_animal_max`` and ``share_of_events_animals``: that
          share per epoch (of the epoch's scored events of the group), each
          animal's mean over its epochs
          (``shared_statistics.animal_session_values(..., "mean")``), their
          mean, the lowest and highest animal, and the animals' values (in
          ``animal_names``' order).
    """
    scored = windows.dropna(subset=[column]).assign(
        width_bin=lambda t: width_bins(t[WIDTH_COLUMN])
    )
    totals = scored.groupby("group").size()
    rows = []
    for label in WIDTH_BIN_LABELS:
        mine = scored[scored["width_bin"] == label]
        statistics = likelihood_advantage_statistics(
            mine, column=column, n_boot=n_boot, seed=seed
        )
        for group, row in statistics.iterrows():
            events = mine[mine["group"] == group]
            positive = (events[column] > 0).astype(float)
            per_animal = animal_session_values(
                positive.groupby([events["animal"], events["epoch_id"]]).transform(
                    "mean"
                ),
                events["animal"],
                events["epoch_id"],
                "median",
            )
            everyone = scored[scored["group"] == group]
            shares = animal_session_values(
                (everyone["width_bin"] == label).astype(float),
                everyone["animal"],
                everyone["epoch_id"],
                "mean",
            ).reindex(row["animal_names"])
            rows.append(
                {
                    "width_bin": label,
                    "group": group,
                    **row.to_dict(),
                    "n_epochs": int(events["epoch_id"].nunique()),
                    "share_positive": float(per_animal.mean()),
                    "share_positive_animal_min": float(per_animal.min()),
                    "share_positive_animal_max": float(per_animal.max()),
                    "share_of_events": len(events) / totals[group],
                    "share_of_events_animal_mean": float(shares.mean()),
                    "share_of_events_animal_min": float(shares.min()),
                    "share_of_events_animal_max": float(shares.max()),
                    "share_of_events_animals": shares.to_numpy(),
                }
            )
    return pd.DataFrame(rows).set_index(["width_bin", "group"])


class SharedContentStatistics(NamedTuple):
    """Figure 2h's statistics (:func:`shared_content_statistics`), each
    :func:`agreement_by_width_statistics` of one of
    :data:`SHARED_CONTENT_COLUMNS`."""

    agreement: pd.DataFrame
    reference: pd.DataFrame
    difference: pd.DataFrame


def shared_content_statistics(
    windows: pd.DataFrame, *, n_boot: int = 1000, seed: int = 0
) -> SharedContentStatistics:
    """The agreement, its shuffled-event reference (on the events with a
    donor) and the agreement minus the reference, by width bin and group, as
    the events summary and Figure 2h read them.

    Parameters
    ----------
    windows : pd.DataFrame
        ``tetrode_holdout.window_scores`` of a run with ``--extra-scores``, the
        events only.
    n_boot, seed : int, optional
        The bootstrap's replicates and seed, the same for every row.
    """
    return SharedContentStatistics(
        *(
            agreement_by_width_statistics(windows, n_boot=n_boot, seed=seed, column=c)
            for c in SHARED_CONTENT_COLUMNS.values()
        )
    )
