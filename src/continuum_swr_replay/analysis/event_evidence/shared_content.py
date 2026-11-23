"""Shared content: during an event, do the tetrodes point to the same remote place?

The held-out tetrode run (:mod:`.tetrode_holdout`) scores each left-out tetrode
under the other tetrodes' position distribution against no place preference:
the window's ``agreement``, above zero only when the left-out tetrode's activity
is where the others point. Over all events it is near zero, because in many
events the others' distribution is spread over much of the track and there is
nothing specific to agree with. This module groups the events by that spread.

An event's spread is its ``spread_cm`` from ``tetrode_holdout.window_scores``:
the mean over its left-out tetrodes of the mean over the window's scored bins of
the effective width of the other tetrodes' position distribution
(``exp(entropy)`` place bins times the place-bin size, cm). The mean over
tetrodes is the natural summary: each left-out tetrode has its own "others", and
the agreement is summed over the same tetrodes. An event's agreement per spike
is ``agreement_per_spike``: summed over tetrodes and bins, divided by the
event's spikes, as the gain per spike is.

The events are grouped by spread in fixed bins, :data:`SPREAD_BINS_CM`, declared
before the result was seen: an earlier look at all windows in thirds of spread
(narrow events agreeing, wide ones not) motivated the analysis, and the edges
were not chosen to improve it. Every animal has events of both kinds in every
bin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_evidence.likelihood_advantage import (
    likelihood_advantage_statistics,
)
from continuum_swr_replay.analysis.shared_statistics import animal_session_values

SPREAD_BINS_CM = (0.0, 25.0, 50.0, 100.0, np.inf)
"""The spread bins' edges (cm), each bin closed on the left: under 25, 25 to
50, 50 to 100, and 100 or more. Fixed in advance (module docstring)."""

SPREAD_BIN_LABELS = ("<25", "25–50", "50–100", "≥100")
"""Each spread bin's label, in order."""

SPREAD_COLUMN = "spread_cm"
"""The ``tetrode_holdout.window_scores`` column the events are binned on: the
one place the width measure is chosen (Figure 2h's x axis follows it)."""


def spread_bins(spread_cm: pd.Series) -> pd.Series:
    """Each event's spread bin, the label of :data:`SPREAD_BIN_LABELS` it
    falls in (a categorical in the bins' order)."""
    return pd.cut(
        spread_cm, list(SPREAD_BINS_CM), right=False, labels=list(SPREAD_BIN_LABELS)
    )


def agreement_by_spread_statistics(
    windows: pd.DataFrame, *, n_boot: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """The agreement per spike of the events of each SWR group, by spread bin.

    Parameters
    ----------
    windows : pd.DataFrame
        ``tetrode_holdout.window_scores`` of the events (``kind`` ``"event"``):
        ``animal``, ``epoch_id``, ``group``, :data:`SPREAD_COLUMN` and
        ``agreement_per_spike`` (NaN for an event without spikes, which counts
        in no statistic).
    n_boot, seed : int, optional
        The bootstrap's replicates and seed, the same for every row.

    Returns
    -------
    pd.DataFrame
        Indexed by ``spread_bin`` and ``group``:

        - ``estimate``, ``ci_low``, ``ci_high``, ``animals``, ``animal_names``,
          ``n_animals``, ``n_events`` and ``n_scored``: the agreement per spike
          as the held-out gain's statistic computes it
          (``likelihood_advantage_statistics``: each epoch's median event, each
          animal's median epoch, the mean over animals, its 95 % interval);
        - ``n_epochs``: the epochs with an event of the group in the bin (an
          epoch without one has no row);
        - ``share_positive``, ``share_positive_animal_min``,
          ``share_positive_animal_max``: the share of events with an agreement
          above zero, per epoch, then each animal's median, then the mean over
          animals, and the lowest and highest animal;
        - ``share_of_events``: the share of the group's scored events that fall
          in the bin, pooled over every event.
    """
    scored = windows.dropna(subset=["agreement_per_spike"]).assign(
        spread_bin=lambda t: spread_bins(t[SPREAD_COLUMN])
    )
    totals = scored.groupby("group").size()
    rows = []
    for label in SPREAD_BIN_LABELS:
        mine = scored[scored["spread_bin"] == label]
        statistics = likelihood_advantage_statistics(
            mine, column="agreement_per_spike", n_boot=n_boot, seed=seed
        )
        for group, row in statistics.iterrows():
            events = mine[mine["group"] == group]
            positive = (events["agreement_per_spike"] > 0).astype(float)
            per_animal = animal_session_values(
                positive.groupby([events["animal"], events["epoch_id"]]).transform(
                    "mean"
                ),
                events["animal"],
                events["epoch_id"],
                "median",
            )
            rows.append(
                {
                    "spread_bin": label,
                    "group": group,
                    **row.to_dict(),
                    "n_epochs": int(events["epoch_id"].nunique()),
                    "share_positive": float(per_animal.mean()),
                    "share_positive_animal_min": float(per_animal.min()),
                    "share_positive_animal_max": float(per_animal.max()),
                    "share_of_events": len(events) / totals[group],
                }
            )
    return pd.DataFrame(rows).set_index(["spread_bin", "group"])
