"""The non-local restriction contrast over all immobile time outside SWRs.

The windows of :mod:`.restriction_windows` are the decoder's own events, so a
gain read on them is read on time the decoder chose. This module reads the same
contrast, ``full - swr`` (:mod:`.nonlocal_restriction`), over time chosen
without the decoder: every bin of the category ``immobile_outside_swr``, set by
speed and ripple detection only (:func:`.nonlocal_restriction.bin_categories`).
There ``swr`` may not use the non-local states and ``full`` may, so the
contrast is what the non-local states buy outside SWRs.

Per epoch (:func:`epoch_outside_swr_gains`), for each score family
(:data:`.nonlocal_restriction.FAMILY_SUFFIX`):

- the gain per second: the category's summed ``full - swr`` (nats) divided by
  its duration, its bins times the decode's time step
  (:func:`.nonlocal_restriction.restriction_contrasts`);
- the share in events with no SWR: the summed ``full - swr`` of the epoch's
  "no SWR" event windows divided by the category's, NaN when the category's
  is not positive. The windows are not wholly inside the category: 12 of the
  42,974 "no SWR" events of the 242-epoch run hold 32 bins inside an SWR the
  ``swr`` model used (:mod:`.restriction_windows`), which count in the
  numerator and are ``immobile_in_swr`` bins, not the category's. A window with
  a bin in excluded time is left out of the numerator, as
  :func:`.restriction_windows.window_contrasts` leaves it out of the window
  statistics (none in the canonical windows).

Over the cohort (:func:`outside_swr_gain_statistics`) each measure is reduced
as the window statistics are
(``likelihood_advantage.likelihood_advantage_statistics``, with one row per
epoch): an animal's value is the median over its epochs, the estimate is the
mean over animals, and its 95 % interval is from the nested bootstrap of the
animals and then their epochs (``shared_statistics.nested_bootstrap_draws``,
``shared_statistics.bootstrap_interval``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_evidence.likelihood_advantage import (
    likelihood_advantage_statistics,
)
from continuum_swr_replay.analysis.event_evidence.nonlocal_restriction import (
    FAMILY_SUFFIX,
    IMMOBILE_OUTSIDE_SWR,
    SCORE_COLUMNS,
    restriction_contrasts,
)
from continuum_swr_replay.analysis.event_evidence.state_evidence.windows import (
    SWR_LABELS,
)

MEASURES = ("gain_per_s", "share_in_no_swr_events")
"""The two per-epoch measures of each family, by their column stem."""


def epoch_outside_swr_gains(
    sums: pd.DataFrame, bin_s: pd.Series, windows: pd.DataFrame
) -> pd.DataFrame:
    """Each epoch's ``full - swr`` over its immobile bins outside SWRs, and the
    share of it in its "no SWR" event windows, both score families.

    Parameters
    ----------
    sums : pd.DataFrame
        The cohort's per-epoch category sums of the restriction run, one row
        per epoch and category: ``epoch_id``, ``animal``, ``category``,
        ``n_bins`` and the six summed score columns
        (:data:`.nonlocal_restriction.SCORE_COLUMNS`).
    bin_s : pd.Series
        Each epoch's decode time step (s), indexed by ``epoch_id``.
    windows : pd.DataFrame
        The cohort's windows (:func:`.restriction_windows.restriction_window_sums`'
        rows with ``epoch_id``): ``kind``, ``group``, ``n_bins_excluded`` and the
        summed score columns. An epoch with no window adds 0 to its share's
        numerator.

    Returns
    -------
    pd.DataFrame
        One row per epoch of ``sums``, sorted by ``epoch_id``: ``epoch_id``,
        ``animal``, ``n_bins`` and ``seconds`` of the category, and for each
        family (suffix ``""`` for the filter score, ``"_two_sided"``):
        ``full_minus_swr`` (nats over the category), ``gain_per_s``,
        ``no_swr_events_full_minus_swr`` (nats over the "no SWR" event windows)
        and ``share_in_no_swr_events``.

    Raises
    ------
    ValueError
        If an epoch has no ``immobile_outside_swr`` row or more than one, or no
        time step.
    """
    epochs = sorted(sums["epoch_id"].unique())
    outside = sums[sums["category"] == IMMOBILE_OUTSIDE_SWR]
    if not outside["epoch_id"].is_unique or set(outside["epoch_id"]) != set(epochs):
        raise ValueError(
            "every epoch must have exactly one immobile_outside_swr row of sums"
        )
    if missing := sorted(set(epochs) - set(bin_s.index)):
        raise ValueError(f"{len(missing)} epochs have no time step: {missing[:3]}")
    no_swr_events = windows[
        (windows["kind"] == "event")
        & (windows["group"] == SWR_LABELS[False])
        & (windows["n_bins_excluded"] == 0)
    ]
    event_sums = no_swr_events.groupby("epoch_id").agg(
        {
            f"{model}{suffix}": "sum"
            for suffix in FAMILY_SUFFIX.values()
            for model in ("full", "swr")
        }
    )
    rows = []
    columns = ["n_bins", *SCORE_COLUMNS]
    for epoch_id, mine in outside.sort_values("epoch_id").groupby("epoch_id"):
        contrasts = restriction_contrasts(
            mine.set_index("category")[columns],
            [IMMOBILE_OUTSIDE_SWR],
            float(bin_s[epoch_id]),
        )
        out = {
            "epoch_id": epoch_id,
            "animal": mine["animal"].iloc[0],
            "n_bins": contrasts["n_bins"],
            "seconds": contrasts["seconds"],
        }
        for suffix in FAMILY_SUFFIX.values():
            total = contrasts[f"full_minus_swr{suffix}"]
            in_events = (
                float(
                    event_sums.loc[epoch_id, f"full{suffix}"]
                    - event_sums.loc[epoch_id, f"swr{suffix}"]
                )
                if epoch_id in event_sums.index
                else 0.0
            )
            out[f"full_minus_swr{suffix}"] = total
            out[f"gain_per_s{suffix}"] = contrasts[f"full_minus_swr{suffix}_per_s"]
            out[f"no_swr_events_full_minus_swr{suffix}"] = in_events
            out[f"share_in_no_swr_events{suffix}"] = (
                in_events / total if total > 0 else np.nan
            )
        rows.append(out)
    return pd.DataFrame(rows)


def outside_swr_gain_statistics(
    epochs: pd.DataFrame, *, n_boot: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Each measure's per-animal values, their mean and its interval, by score
    family, with the epochs that have a value and those above zero.

    Parameters
    ----------
    epochs : pd.DataFrame
        :func:`epoch_outside_swr_gains`' rows.
    n_boot, seed : int, optional
        The bootstrap's replicates and seed, the same for every row.

    Returns
    -------
    pd.DataFrame
        One row per family and measure (:data:`MEASURES`): those keys,
        ``estimate`` (the mean over animals of each animal's median over its
        epochs), ``ci_low``, ``ci_high`` (the 95 % interval), ``animals`` (the
        animals' values, sorted animal order), ``animal_names``, ``n_animals``,
        ``n_animals_positive`` (animals whose value is above zero), ``n_epochs``
        (epochs with a value) and ``n_epochs_positive`` (epochs whose value is
        above zero).
    """
    parts = []
    for family, suffix in FAMILY_SUFFIX.items():
        for measure in MEASURES:
            column = f"{measure}{suffix}"
            table = epochs[["epoch_id", "animal", column]].assign(group=measure)
            row = likelihood_advantage_statistics(
                table, column=column, n_boot=n_boot, seed=seed
            ).loc[measure]
            known = epochs[column].dropna()
            parts.append(
                {
                    "family": family,
                    "measure": measure,
                    "estimate": row["estimate"],
                    "ci_low": row["ci_low"],
                    "ci_high": row["ci_high"],
                    "animals": row["animals"],
                    "animal_names": row["animal_names"],
                    "n_animals": row["n_animals"],
                    "n_animals_positive": int((row["animals"] > 0).sum()),
                    "n_epochs": len(known),
                    "n_epochs_positive": int((known > 0).sum()),
                }
            )
    return pd.DataFrame(parts)
