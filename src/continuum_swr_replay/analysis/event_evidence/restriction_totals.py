"""The non-local restriction contrasts over all immobile time, and outside SWRs.

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

- the share in the SWR-overlapping events' time outside the SWR: the summed
  ``full - swr`` of the epoch's "SWR" event windows over their
  ``immobile_outside_swr`` bins only (the window table's ``*_outside_swr``
  columns, :func:`.restriction_windows.restriction_window_sums`) divided by the
  category's, NaN when the category's is not positive;
- the gain per second in the rest of that time, outside every event: the
  category's summed ``full - swr`` less the event windows' over their
  ``immobile_outside_swr`` bins, divided by the category's duration less
  theirs. The events do not overlap one another (checked), so this is the
  category's time that no event holds. Every event window enters, whether or
  not it has an excluded bin: its ``immobile_outside_swr`` bins are the
  category's either way.

Figure 2g's first axis reads the decoders over all immobile time
(:func:`epoch_immobile_gains`): per epoch, over the bins of both immobile
categories (``immobile_in_swr`` and ``immobile_outside_swr``), each decoder's
gain over the local-only decoder per second, ``(swr - none)`` and
``(full - none)`` over their duration; their difference ``full - swr`` per
second (above zero where the full decoder predicts the spikes better than the
SWR-only one); and the share of the full decoder's gain the SWR-only decoder
captures, ``(swr - none) / (full - none)``, NaN when the full decoder's gain is
not positive.

Over the cohort (:func:`outside_swr_gain_statistics`,
:func:`immobile_gain_statistics`) each measure is reduced
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
    IMMOBILE_IN_SWR,
    IMMOBILE_OUTSIDE_SWR,
    SCORE_COLUMNS,
    restriction_contrasts,
)
from continuum_swr_replay.analysis.event_evidence.restriction_windows import (
    OUTSIDE_SWR_SUFFIX,
    require_outside_swr_sums,
)
from continuum_swr_replay.analysis.event_evidence.state_evidence.windows import (
    SWR_LABELS,
)

MEASURES = (
    "gain_per_s",
    "share_in_no_swr_events",
    "share_in_swr_events_outside_swr",
    "outside_events_gain_per_s",
)
"""The per-epoch measures of each family over the immobile time outside SWRs,
by their column stem (:func:`epoch_outside_swr_gains`)."""

IMMOBILE_MEASURES = (
    "swr_minus_none_per_s",
    "full_minus_none_per_s",
    "full_minus_swr_per_s",
    "share_captured_by_swr_only",
)
"""The per-epoch measures of each family over all immobile time, by their
column stem (:func:`epoch_immobile_gains`)."""


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
        rows with ``epoch_id``): ``kind``, ``group``, ``s``, ``e``,
        ``n_bins_excluded``, the summed score columns and the columns over the
        bins outside SWRs (:data:`.restriction_windows.OUTSIDE_SWR_COLUMNS`). An
        epoch with no window adds 0 to its shares' numerators.

    Returns
    -------
    pd.DataFrame
        One row per epoch of ``sums``, sorted by ``epoch_id``: ``epoch_id``,
        ``animal``, ``n_bins`` and ``seconds`` of the category,
        ``outside_events_seconds`` (the category's seconds no event holds), and
        for each family (suffix ``""`` for the filter score, ``"_two_sided"``):
        ``full_minus_swr`` (nats over the category), ``gain_per_s``,
        ``no_swr_events_full_minus_swr`` (nats over the "no SWR" event windows),
        ``share_in_no_swr_events``, ``swr_events_outside_swr_full_minus_swr``
        (nats over the "SWR" event windows' bins of the category),
        ``share_in_swr_events_outside_swr``, ``outside_events_full_minus_swr``
        (nats over the category's bins no event holds) and
        ``outside_events_gain_per_s`` (NaN without such bins).

    Raises
    ------
    ValueError
        If an epoch has no ``immobile_outside_swr`` row or more than one, or no
        time step; if ``windows`` lacks the columns over the bins outside SWRs
        (a window run older than them,
        :func:`.restriction_windows.require_outside_swr_sums`); or if two event
        windows of an epoch overlap.
    """
    require_outside_swr_sums(windows)
    epochs = sorted(sums["epoch_id"].unique())
    outside = sums[sums["category"] == IMMOBILE_OUTSIDE_SWR]
    if not outside["epoch_id"].is_unique or set(outside["epoch_id"]) != set(epochs):
        raise ValueError(
            "every epoch must have exactly one immobile_outside_swr row of sums"
        )
    if missing := sorted(set(epochs) - set(bin_s.index)):
        raise ValueError(f"{len(missing)} epochs have no time step: {missing[:3]}")
    events = windows[windows["kind"] == "event"].sort_values(["epoch_id", "s"])
    same_epoch = events["epoch_id"].to_numpy()[1:] == events["epoch_id"].to_numpy()[:-1]
    if (same_epoch & (events["s"].to_numpy()[1:] <= events["e"].to_numpy()[:-1])).any():
        raise ValueError("two event windows of an epoch overlap")
    models = [
        f"{model}{suffix}"
        for suffix in FAMILY_SUFFIX.values()
        for model in ("full", "swr")
    ]
    kept = events["n_bins_excluded"] == 0
    event_sums = (
        events[kept & (events["group"] == SWR_LABELS[False])]
        .groupby("epoch_id")[models]
        .sum()
    )
    outside_models = [f"{model}{OUTSIDE_SWR_SUFFIX}" for model in models]
    flank_sums = (
        events[kept & (events["group"] == SWR_LABELS[True])]
        .groupby("epoch_id")[outside_models]
        .sum()
    )
    in_events = events.groupby("epoch_id")[
        ["n_bins_outside_swr", *outside_models]
    ].sum()

    def summed(table: pd.DataFrame, epoch_id: str, column: str) -> float:
        """An epoch's sum of ``column`` in ``table``, 0 when it has no row."""
        return float(table.loc[epoch_id, column]) if epoch_id in table.index else 0.0

    rows = []
    columns = ["n_bins", *SCORE_COLUMNS]
    for epoch_id, mine in outside.sort_values("epoch_id").groupby("epoch_id"):
        contrasts = restriction_contrasts(
            mine.set_index("category")[columns],
            [IMMOBILE_OUTSIDE_SWR],
            float(bin_s[epoch_id]),
        )
        rest_seconds = contrasts["seconds"] - (
            summed(in_events, epoch_id, "n_bins_outside_swr") * float(bin_s[epoch_id])
        )
        out = {
            "epoch_id": epoch_id,
            "animal": mine["animal"].iloc[0],
            "n_bins": contrasts["n_bins"],
            "seconds": contrasts["seconds"],
            "outside_events_seconds": rest_seconds,
        }
        for suffix in FAMILY_SUFFIX.values():
            total = contrasts[f"full_minus_swr{suffix}"]
            no_swr = summed(event_sums, epoch_id, f"full{suffix}") - summed(
                event_sums, epoch_id, f"swr{suffix}"
            )
            full_out, swr_out = (
                f"{model}{suffix}{OUTSIDE_SWR_SUFFIX}" for model in ("full", "swr")
            )
            flanks = summed(flank_sums, epoch_id, full_out) - summed(
                flank_sums, epoch_id, swr_out
            )
            rest = total - (
                summed(in_events, epoch_id, full_out)
                - summed(in_events, epoch_id, swr_out)
            )
            out[f"full_minus_swr{suffix}"] = total
            out[f"gain_per_s{suffix}"] = contrasts[f"full_minus_swr{suffix}_per_s"]
            out[f"no_swr_events_full_minus_swr{suffix}"] = no_swr
            out[f"share_in_no_swr_events{suffix}"] = (
                no_swr / total if total > 0 else np.nan
            )
            out[f"swr_events_outside_swr_full_minus_swr{suffix}"] = flanks
            out[f"share_in_swr_events_outside_swr{suffix}"] = (
                flanks / total if total > 0 else np.nan
            )
            out[f"outside_events_full_minus_swr{suffix}"] = rest
            out[f"outside_events_gain_per_s{suffix}"] = (
                rest / rest_seconds if rest_seconds > 0 else np.nan
            )
        rows.append(out)
    return pd.DataFrame(rows)


def epoch_immobile_gains(sums: pd.DataFrame, bin_s: pd.Series) -> pd.DataFrame:
    """Each epoch's decoders over all its immobile time: the SWR-only and full
    decoders' gains over the local-only decoder per second, their difference,
    and the share of the full decoder's gain the SWR-only decoder captures.

    Parameters
    ----------
    sums : pd.DataFrame
        The cohort's per-epoch category sums of the restriction run, as
        :func:`epoch_outside_swr_gains` takes them.
    bin_s : pd.Series
        Each epoch's decode time step (s), indexed by ``epoch_id``.

    Returns
    -------
    pd.DataFrame
        One row per epoch of ``sums``, sorted by ``epoch_id``: ``epoch_id``,
        ``animal``, ``n_bins`` and ``seconds`` of the immobile bins (both
        immobile categories), and for each family (suffix ``""`` for the filter
        score, ``"_two_sided"``): ``swr_minus_none_per_s``,
        ``full_minus_none_per_s`` and ``full_minus_swr_per_s`` (nats per second
        of immobile time, NaN without immobile bins),
        ``share_captured_by_swr_only`` (``swr - none`` over ``full - none``,
        NaN when ``full - none`` is not positive) and ``full_above_swr_only``
        (``full - swr`` above zero).

    Raises
    ------
    ValueError
        If an epoch has no time step.
    """
    epochs = sorted(sums["epoch_id"].unique())
    if missing := sorted(set(epochs) - set(bin_s.index)):
        raise ValueError(f"{len(missing)} epochs have no time step: {missing[:3]}")
    columns = ["n_bins", *SCORE_COLUMNS]
    rows = []
    for epoch_id, mine in sums.sort_values("epoch_id").groupby("epoch_id"):
        contrasts = restriction_contrasts(
            mine.set_index("category")[columns],
            [IMMOBILE_IN_SWR, IMMOBILE_OUTSIDE_SWR],
            float(bin_s[epoch_id]),
        )
        out = {
            "epoch_id": epoch_id,
            "animal": mine["animal"].iloc[0],
            "n_bins": contrasts["n_bins"],
            "seconds": contrasts["seconds"],
        }
        for suffix in FAMILY_SUFFIX.values():
            swr_gain = contrasts[f"swr_minus_none{suffix}"]
            full_gain = contrasts[f"full_minus_none{suffix}"]
            for name in ("swr_minus_none", "full_minus_none", "full_minus_swr"):
                out[f"{name}_per_s{suffix}"] = contrasts[f"{name}{suffix}_per_s"]
            out[f"share_captured_by_swr_only{suffix}"] = (
                swr_gain / full_gain if full_gain > 0 else np.nan
            )
            out[f"full_above_swr_only{suffix}"] = (
                contrasts[f"full_minus_swr{suffix}"] > 0
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
    return _epoch_measure_statistics(epochs, MEASURES, n_boot=n_boot, seed=seed)


def immobile_gain_statistics(
    epochs: pd.DataFrame, *, n_boot: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Each measure over all immobile time: per-animal values, their mean and
    its interval, by score family, with the epochs that have a value, those
    above zero and those where the full decoder is above the SWR-only one.

    Parameters
    ----------
    epochs : pd.DataFrame
        :func:`epoch_immobile_gains`' rows.
    n_boot, seed : int, optional
        The bootstrap's replicates and seed, the same for every row.

    Returns
    -------
    pd.DataFrame
        One row per family and measure (:data:`IMMOBILE_MEASURES`), with the
        columns of :func:`outside_swr_gain_statistics` and
        ``n_epochs_full_above_swr_only`` (the family's epochs whose
        ``full - swr`` is above zero, the same on each of its rows).
    """
    table = _epoch_measure_statistics(
        epochs, IMMOBILE_MEASURES, n_boot=n_boot, seed=seed
    )
    table["n_epochs_full_above_swr_only"] = [
        int(epochs[f"full_above_swr_only{FAMILY_SUFFIX[family]}"].sum())
        for family in table["family"]
    ]
    return table


def _epoch_measure_statistics(
    epochs: pd.DataFrame, measures: tuple[str, ...], *, n_boot: int, seed: int
) -> pd.DataFrame:
    """Per family and measure: each animal's median over its epochs, their
    mean, its interval, and the epochs and animals with a value and above zero
    (the columns :func:`outside_swr_gain_statistics` documents)."""
    parts = []
    for family, suffix in FAMILY_SUFFIX.items():
        for measure in measures:
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
