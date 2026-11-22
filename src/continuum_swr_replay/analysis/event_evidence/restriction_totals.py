"""The non-local restriction contrasts over all immobile time, and outside SWRs.

The windows of :mod:`.restriction_windows` are the decoder's own events, so a
gain read on them is read on time the decoder chose. This module reads the
contrasts (:mod:`.nonlocal_restriction`) over time chosen without the decoder:
the bins of the immobile categories, set by speed and ripple detection only
(:func:`.nonlocal_restriction.bin_categories`), and only those inside a well
visit (``event_definition.well_visits.in_well_visit``), the time subset of
Figures 2 to 5.

**The time sums.** Per epoch, :func:`restriction_time_sums` sums the per-bin
scores, bins and spikes (the per-bin facts, :mod:`.bin_facts`) of each
combination of ``category``, ``in_well_visit`` and ``in_event_window`` (the bin
lies in the window of a given event of either group: the events detected from
the held-out decode, ``scripts/event_evidence/detect_heldout_events.py``): every
bin is in exactly one row, so any union of rows is summed by adding them.

**All immobile time** (:func:`epoch_immobile_gains`, Figure 2g's first axis):
per epoch, over the bins of both immobile categories (``immobile_in_swr`` and
``immobile_outside_swr``) inside a well visit, each decoder's gain over the
local-only decoder, ``(swr - none)`` and ``(full - none)``, and their
difference ``full - swr``, each per spike in those bins and per second of them;
and the share of the full decoder's gain the SWR-only decoder captures,
``(swr - none) / (full - none)``, NaN when the full decoder's gain is not
positive.

**Immobile time outside SWRs** (:func:`epoch_outside_swr_gains`): per epoch,
over the ``immobile_outside_swr`` bins inside a well visit, where ``swr`` may
not use the non-local states and ``full`` may, so ``full - swr`` is what the
non-local states buy outside SWRs:

- the gain per spike and per second;
- the share in the given events with no SWR, and the share in the given events'
  time outside the SWR of those that overlap one: the events' summed
  ``full - swr`` over their ``immobile_outside_swr`` bins inside a well visit
  (the window table's ``*_outside_swr_in_visit`` columns,
  :func:`.restriction_windows.window_bin_fact_sums`) divided by the epoch's
  over the same kind of bins, so numerator and denominator are summed on the
  same bins. An event that starts inside a visit and ends after its final
  poke-out adds its bins inside the visit only. NaN when the epoch's total is
  not positive. A window with a bin in excluded time is left out of the
  numerators, as :func:`.restriction_windows.window_contrasts` leaves it out of
  the window statistics;
- the gain per spike and per second in the rest of that time, the bins inside
  no event window of any group (``in_event_window`` false): every event the
  time sums were given counts, whichever events the shares are given.

Over the cohort (:func:`outside_swr_gain_statistics`,
:func:`immobile_gain_statistics`) each measure is reduced
as the window statistics are
(``likelihood_advantage.likelihood_advantage_statistics``, with one row per
epoch): an animal's value is the median over its epochs, the estimate is the
mean over animals, and its 95 % interval is from the nested bootstrap of the
animals and then their epochs (``shared_statistics.nested_bootstrap_copies``,
``shared_statistics.bootstrap_interval``).
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from continuum_swr_replay.analysis.event_evidence.likelihood_advantage import (
    likelihood_advantage_statistics,
)
from continuum_swr_replay.analysis.event_evidence.nonlocal_restriction import (
    CATEGORIES,
    CONTRASTS,
    FAMILY_SUFFIX,
    IMMOBILE_IN_SWR,
    IMMOBILE_OUTSIDE_SWR,
    SCORE_COLUMNS,
    ratio_or_nan,
)
from continuum_swr_replay.analysis.event_evidence.restriction_windows import (
    IN_VISIT_SUFFIX,
    _check_facts_time,
    _window_bounds,
    restriction_window_statistics,
)
from continuum_swr_replay.analysis.event_evidence.state_evidence.windows import (
    SWR_LABELS,
)
from continuum_swr_replay.analysis.event_evidence.tetrode_holdout import window_rows

TIME_SUM_KEYS = ("category", "in_well_visit", "in_event_window")
"""The columns that name a row of an epoch's time sums
(:func:`restriction_time_sums`)."""

MEASURES = (
    "gain_per_spike",
    "gain_per_s",
    "share_in_no_swr_events",
    "share_in_swr_events_outside_swr",
    "outside_events_gain_per_spike",
    "outside_events_gain_per_s",
)
"""The per-epoch measures of each family over the immobile time outside SWRs,
by their column stem (:func:`epoch_outside_swr_gains`)."""

IMMOBILE_MEASURES = (
    *(
        f"{contrast}_{unit}"
        for unit in ("per_spike", "per_s")
        for contrast in CONTRASTS
    ),
    "share_captured_by_swr_only",
)
"""The per-epoch measures of each family over all immobile time, by their
column stem (:func:`epoch_immobile_gains`): ``full_minus_swr``,
``swr_minus_none`` and ``full_minus_none`` per spike and per second, and the
share the SWR-only decoder captures."""


def restriction_time_sums(
    bins: pd.DataFrame, facts: pd.DataFrame, windows: pd.DataFrame
) -> pd.DataFrame:
    """One epoch's bins, spikes and summed scores by category, well visit and
    event window.

    Parameters
    ----------
    bins : pd.DataFrame
        The epoch's per-bin table of the restriction run, one row per decode
        bin in time order: ``time`` (s), ``category`` and the six score columns
        (:data:`.nonlocal_restriction.SCORE_COLUMNS`).
    facts : pd.DataFrame
        The epoch's per-bin facts (``bin_facts.bin_facts``) on the same grid:
        ``time``, ``n_spikes`` and ``in_well_visit``.
    windows : pd.DataFrame
        The epoch's windows: ``kind`` and the first and last bin ``s`` and
        ``e`` (inclusive). Only the ``"event"`` windows mark
        ``in_event_window``.

    Returns
    -------
    pd.DataFrame
        One row for each combination of :data:`TIME_SUM_KEYS`
        (:data:`.nonlocal_restriction.CATEGORIES` by ``in_well_visit`` by
        ``in_event_window``, 16 rows, those without bins included): ``n_bins``,
        ``n_spikes`` (int64) and each score's float64 sum (0 without bins).

    Raises
    ------
    ValueError
        If the facts' time is not the per-bin table's, a bin's category is not
        one of the categories, or a window reaches outside ``bins``.
    """
    _check_facts_time(bins, facts)
    category = bins["category"].astype(str).to_numpy()
    if unknown := sorted(set(category) - set(CATEGORIES)):
        raise ValueError(f"bins in unknown categories: {unknown}")
    s, e = _window_bounds(windows[windows["kind"] == "event"], len(bins))
    in_event = np.zeros(len(bins), dtype=bool)
    in_event[window_rows(s, e)] = True
    frame = bins[list(SCORE_COLUMNS)].astype(np.float64)
    frame.insert(0, "n_spikes", facts["n_spikes"].to_numpy(dtype=np.int64))
    frame.insert(0, "n_bins", np.int64(1))
    keys = [
        pd.Series(category, name="category"),
        pd.Series(facts["in_well_visit"].to_numpy(dtype=bool), name="in_well_visit"),
        pd.Series(in_event, name="in_event_window"),
    ]
    summed = frame.reset_index(drop=True).groupby(keys).sum()
    every = pd.MultiIndex.from_tuples(
        list(itertools.product(CATEGORIES, (False, True), (False, True))),
        names=list(TIME_SUM_KEYS),
    )
    out = summed.reindex(every, fill_value=0).reset_index()
    out[["n_bins", "n_spikes"]] = out[["n_bins", "n_spikes"]].astype(np.int64)
    return out


def _chosen_time(
    time_sums: pd.DataFrame,
    categories: tuple[str, ...],
    *,
    outside_events: bool = False,
) -> pd.DataFrame:
    """Each epoch's sums over the rows of ``categories`` inside a well visit
    (and outside every event window only, when asked): ``n_bins``,
    ``n_spikes`` and the scores, indexed by ``epoch_id``, every epoch of
    ``time_sums`` included."""
    rows = time_sums["category"].isin(categories)
    rows &= time_sums["in_well_visit"].astype(bool)
    if outside_events:
        rows &= ~time_sums["in_event_window"].astype(bool)
    columns = ["n_bins", "n_spikes", *SCORE_COLUMNS]
    epochs = sorted(time_sums["epoch_id"].unique())
    return (
        time_sums[rows].groupby("epoch_id")[columns].sum().reindex(epochs, fill_value=0)
    )


def _check_time_sums(time_sums: pd.DataFrame, bin_s: pd.Series) -> None:
    """Refuse time sums with an epoch's row of :data:`TIME_SUM_KEYS` twice, or
    an epoch without a time step."""
    if time_sums.duplicated(["epoch_id", *TIME_SUM_KEYS]).any():
        raise ValueError("an epoch has a row of the time sums twice")
    epochs = set(time_sums["epoch_id"])
    if missing := sorted(epochs - set(bin_s.index)):
        raise ValueError(f"{len(missing)} epochs have no time step: {missing[:3]}")


def epoch_outside_swr_gains(
    time_sums: pd.DataFrame,
    bin_s: pd.Series,
    windows: pd.DataFrame,
) -> pd.DataFrame:
    """Each epoch's ``full - swr`` over its immobile bins outside SWRs inside a
    well visit, per spike and per second, the share of it in the
    given events, and the rest outside every event, both score families.

    Parameters
    ----------
    time_sums : pd.DataFrame
        The cohort's time sums (:func:`restriction_time_sums`' rows with
        ``epoch_id`` and ``animal``).
    bin_s : pd.Series
        Each epoch's decode time step (s), indexed by ``epoch_id``.
    windows : pd.DataFrame
        The windows of the events the shares are of
        (:func:`.restriction_windows.restriction_window_sums`' rows with
        ``epoch_id`` and :data:`.restriction_windows.BIN_FACT_COLUMNS`): ``kind``,
        ``group``, ``n_bins_excluded`` and the sums over the bins outside SWRs.
        Only the ``"event"`` windows count. An epoch with no window adds 0 to
        its shares' numerators.

    Returns
    -------
    pd.DataFrame
        One row per epoch of ``time_sums``, sorted by ``epoch_id``:
        ``epoch_id``, ``animal``, ``n_bins``, ``n_spikes`` and ``seconds`` of
        the chosen bins, ``outside_events_n_spikes`` and
        ``outside_events_seconds`` (of those inside no event window), and for
        each family (suffix ``""`` for the filter score, ``"_two_sided"``):
        ``full_minus_swr`` (nats), ``gain_per_spike``, ``gain_per_s``,
        ``no_swr_events_full_minus_swr``, ``share_in_no_swr_events``,
        ``swr_events_outside_swr_full_minus_swr``,
        ``share_in_swr_events_outside_swr``, ``outside_events_full_minus_swr``,
        ``outside_events_gain_per_spike`` and ``outside_events_gain_per_s``
        (NaN without spikes, bins or a positive total, as each divides).

    Raises
    ------
    ValueError
        If an epoch has a row of the time sums twice or no time step.
    """
    _check_time_sums(time_sums, bin_s)
    total = _chosen_time(time_sums, (IMMOBILE_OUTSIDE_SWR,))
    rest = _chosen_time(time_sums, (IMMOBILE_OUTSIDE_SWR,), outside_events=True)
    models = [
        f"{model}{suffix}{IN_VISIT_SUFFIX}"
        for suffix in FAMILY_SUFFIX.values()
        for model in ("full", "swr")
    ]
    events = windows[(windows["kind"] == "event") & (windows["n_bins_excluded"] == 0)]
    by_group = {
        group: events[events["group"] == SWR_LABELS[swr]]
        .groupby("epoch_id")[models]
        .sum()
        .reindex(total.index, fill_value=0.0)
        for group, swr in (("no_swr", False), ("swr", True))
    }
    animals = time_sums.groupby("epoch_id")["animal"].first()
    step = bin_s.reindex(total.index).to_numpy(dtype=float)
    out = pd.DataFrame(
        {
            "epoch_id": total.index,
            "animal": animals.reindex(total.index).to_numpy(),
            "n_bins": total["n_bins"].to_numpy(),
            "n_spikes": total["n_spikes"].to_numpy(),
            "seconds": total["n_bins"].to_numpy() * step,
            "outside_events_n_spikes": rest["n_spikes"].to_numpy(),
            "outside_events_seconds": rest["n_bins"].to_numpy() * step,
        }
    )
    for suffix in FAMILY_SUFFIX.values():
        gain = (total[f"full{suffix}"] - total[f"swr{suffix}"]).to_numpy()
        rest_gain = (rest[f"full{suffix}"] - rest[f"swr{suffix}"]).to_numpy()
        no_swr, flanks = (
            (
                by_group[group][f"full{suffix}{IN_VISIT_SUFFIX}"]
                - by_group[group][f"swr{suffix}{IN_VISIT_SUFFIX}"]
            ).to_numpy()
            for group in ("no_swr", "swr")
        )
        out[f"full_minus_swr{suffix}"] = gain
        out[f"gain_per_spike{suffix}"] = list(map(ratio_or_nan, gain, out["n_spikes"]))
        out[f"gain_per_s{suffix}"] = list(map(ratio_or_nan, gain, out["seconds"]))
        out[f"no_swr_events_full_minus_swr{suffix}"] = no_swr
        out[f"share_in_no_swr_events{suffix}"] = list(map(ratio_or_nan, no_swr, gain))
        out[f"swr_events_outside_swr_full_minus_swr{suffix}"] = flanks
        out[f"share_in_swr_events_outside_swr{suffix}"] = list(
            map(ratio_or_nan, flanks, gain)
        )
        out[f"outside_events_full_minus_swr{suffix}"] = rest_gain
        out[f"outside_events_gain_per_spike{suffix}"] = list(
            map(ratio_or_nan, rest_gain, out["outside_events_n_spikes"])
        )
        out[f"outside_events_gain_per_s{suffix}"] = list(
            map(ratio_or_nan, rest_gain, out["outside_events_seconds"])
        )
    return out


def epoch_immobile_gains(time_sums: pd.DataFrame, bin_s: pd.Series) -> pd.DataFrame:
    """Each epoch's decoders over all its immobile time inside a well visit: the SWR-only and full decoders' gains over the local-only decoder
    per spike and per second, their difference, and the share of the full
    decoder's gain the SWR-only decoder captures.

    Parameters
    ----------
    time_sums : pd.DataFrame
        The cohort's time sums, as :func:`epoch_outside_swr_gains` takes them.
    bin_s : pd.Series
        Each epoch's decode time step (s), indexed by ``epoch_id``.

    Returns
    -------
    pd.DataFrame
        One row per epoch of ``time_sums``, sorted by ``epoch_id``:
        ``epoch_id``, ``animal``, ``n_bins``, ``n_spikes`` and ``seconds`` of
        the chosen immobile bins (both immobile categories), and for each
        family (suffix ``""`` for the filter score, ``"_two_sided"``): each
        contrast of :data:`.nonlocal_restriction.CONTRASTS` per spike and per
        second (``<contrast>_per_spike``, ``<contrast>_per_s``; NaN without
        spikes or bins), ``share_captured_by_swr_only`` (``swr - none`` over
        ``full - none``, NaN when ``full - none`` is not positive) and
        ``full_above_swr_only`` (``full - swr`` above zero).

    Raises
    ------
    ValueError
        If an epoch has a row of the time sums twice or no time step.
    """
    _check_time_sums(time_sums, bin_s)
    total = _chosen_time(time_sums, (IMMOBILE_IN_SWR, IMMOBILE_OUTSIDE_SWR))
    animals = time_sums.groupby("epoch_id")["animal"].first()
    seconds = total["n_bins"].to_numpy() * bin_s.reindex(total.index).to_numpy(float)
    spikes = total["n_spikes"].to_numpy()
    out = pd.DataFrame(
        {
            "epoch_id": total.index,
            "animal": animals.reindex(total.index).to_numpy(),
            "n_bins": total["n_bins"].to_numpy(),
            "n_spikes": spikes,
            "seconds": seconds,
        }
    )
    for suffix in FAMILY_SUFFIX.values():
        gains = {
            contrast: (total[f"{a}{suffix}"] - total[f"{b}{suffix}"]).to_numpy()
            for contrast, (a, b) in CONTRASTS.items()
        }
        for unit, divisor in (("per_spike", spikes), ("per_s", seconds)):
            for contrast, gain in gains.items():
                out[f"{contrast}_{unit}{suffix}"] = list(
                    map(ratio_or_nan, gain, divisor)
                )
        out[f"share_captured_by_swr_only{suffix}"] = list(
            map(ratio_or_nan, gains["swr_minus_none"], gains["full_minus_none"])
        )
        out[f"full_above_swr_only{suffix}"] = gains["full_minus_swr"] > 0
    return out


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


# --- Figure 2g ---------------------------------------------------------------

RESTRICTION_FIGURE_FAMILY = "filter"
"""Figure 2g, the decoders compared, in one score family (the events summary's
values of it carry the panel's tag; the other family's do not)."""
RESTRICTION_FIGURE_DECODERS = ("swr_minus_none_per_spike", "full_minus_none_per_spike")
"""Figure 2g's first axis, "All immobile time": each decoder's gain over the
local-only decoder per spike of the immobile time inside a well visit, by its
measure (:data:`IMMOBILE_MEASURES`), in drawing order (SWR-only, Full)."""
RESTRICTION_FIGURE_DIFFERENCE = "full_minus_swr_per_spike"
"""The measure of the two decoders' difference, which the line joining an
animal's two dots on Figure 2g's first axis shows."""
RESTRICTION_FIGURE = {
    "parts": ["outside_swr"],
    "families": [RESTRICTION_FIGURE_FAMILY],
    "contrasts": ["full_minus_swr"],
    "units": ["per_spike"],
    "kinds": ["event"],
}
"""Figure 2g's second axis, "Immobile time outside SWR bounds": the choice of
``restriction_windows.restriction_window_statistics`` (its keywords), the full
decoder minus the SWR-only decoder per spike over the bins outside SWRs of the
events detected from the held-out decode."""
RESTRICTION_FIGURE_GROUPS = (SWR_LABELS[False], SWR_LABELS[True])
"""The second axis' groups in drawing order: events with no SWR, then the time
outside the SWR of the events that overlap one."""


def restriction_figure_statistics(
    windows: pd.DataFrame, time_sums: pd.DataFrame, bin_s: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Figure 2g's two axes, as the events summary computes them.

    Parameters
    ----------
    windows : pd.DataFrame
        The held-out event run's events of every epoch
        (``scripts/event_evidence/detect_heldout_events.py``), those that start
        inside a well visit (``starts_in_well_visit``).
    time_sums : pd.DataFrame
        The held-out event run's time sums of every epoch
        (:func:`restriction_time_sums`).
    bin_s : pd.Series
        Each epoch's decode time step (s), indexed by ``epoch_id``.

    Returns
    -------
    decoders : pd.DataFrame
        :func:`immobile_gain_statistics` over the immobile time inside a well
        visit, indexed by ``family`` and ``measure``.
    parts : pd.DataFrame
        ``restriction_windows.restriction_window_statistics`` with
        :data:`RESTRICTION_FIGURE` over ``windows``, indexed by ``group``.
    """
    decoders = immobile_gain_statistics(
        epoch_immobile_gains(time_sums, bin_s)
    ).set_index(["family", "measure"])
    parts = restriction_window_statistics(windows, **RESTRICTION_FIGURE).set_index(
        "group"
    )
    return decoders, parts
