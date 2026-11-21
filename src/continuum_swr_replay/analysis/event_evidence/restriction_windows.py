"""The non-local restriction models over the windows of Figure 2's panel g.

The per-bin scores of the three models (``none``, ``swr``, ``full``;
:mod:`.nonlocal_restriction`) summed over each canonical non-local event and
over its matched Local-rest control window: the windows of the held-out tetrode
score (``state_evidence.windows.epoch_windows``), so the two methods are read on
the same bins and summarized by the same statistic
(``likelihood_advantage.likelihood_advantage_statistics``: one value per animal,
their mean and its nested-bootstrap interval). The held-out encodings are the
same only when the two runs used the same fold design (``fold_design`` in each
run's manifest settings; both runners default to whole trials,
``tetrode_holdout.trial_folds``). A held-out tetrode run on contiguous blocks
of time (``tetrode_holdout.time_blocks``), as the run made before that option
was, differs from the restriction run in more than what is scored.

A window's contrast ``a - b`` (:data:`CONTRASTS`) is its sum of ``a``'s per-bin
score less its sum of ``b``'s, in nats, for each score family
(:data:`.nonlocal_restriction.FAMILY_SUFFIX`). It is read two ways:

- per spike: divided by the window's spike count over every tetrode (the
  held-out tetrode run's ``n_spikes``, summed over tetrodes), NaN without
  spikes, as ``tetrode_holdout.window_scores`` does;
- per second: divided by the window's duration, its bins times the decode's
  time step (``intervals.sample_step`` of the per-bin table's time).

A contrast is also read over a part of the window (:data:`PART_UNITS`): its bins
in the category ``immobile_outside_swr`` only, per spike in them (the spikes of
every tetrode the decoder put in those bins, from the per-bin facts,
:mod:`.bin_facts`; NaN without spikes) and per second of them (the window
table's ``*_outside_swr`` columns). In an event that overlaps an SWR these are
its bins outside the SWR, where ``swr`` may not use the non-local states; in an
event without an SWR, all its bins but the few inside an SWR the ``swr`` model
used. Figure 2g's second axis reads ``full - swr`` there, per spike, over the
windows of the events detected from the held-out decode
(``scripts/event_evidence/detect_heldout_events.py`` sums them with the same
functions).

The per-bin facts also give each window's sums over its ``immobile_outside_swr``
bins inside a well visit (:func:`window_bin_fact_sums`,
:data:`BIN_FACT_COLUMNS`): the numerators of the shares of
``restriction_totals.epoch_outside_swr_gains``, on the same bins as their
denominator.

``full - swr`` in events without an SWR is what the non-local states buy outside
SWRs, where ``swr`` may not use them; in the Local-rest controls it is the same
contrast where the decoder saw no non-local activity; ``swr - none`` in events
that overlap an SWR is what calling the SWR's bins non-local buys. "Without an
SWR" is the event table's class (strict overlap with a kept SWR), not the
``swr`` model's bins: an event that only shares an edge bin with an SWR, or
that lies in an SWR the cohort drops for speed, is without an SWR and still
holds bins where ``swr`` is forced non-local. The table counts them
(``n_bins_in_swr``) and the runner's records total them
(``n_no_swr_events_with_in_swr_bins``, ``n_in_swr_bins_of_no_swr_events``: 12
of the 42,974 such events and 32 bins in the 242-epoch run).

The filter family's window sum is the window's share of the model's log
likelihood; the two-sided family's is a sum of scores that do not add up to a
likelihood (:mod:`.nonlocal_restriction`), read the same way.

**Windows with an excluded bin or a score that is not finite.** The sums are
taken over every bin of the window whatever its category, and the table counts
its bins in excluded time (``n_bins_excluded``), its bins outside the immobile
categories (``n_bins`` less ``n_bins_immobile``) and its bins with any score that
is not finite (``n_bins_not_finite``). A score that is not finite makes that
column's sum NaN; it is not replaced. :func:`window_contrasts` gives a window
with an excluded bin NaN in every contrast, so the statistics leave it out and
count it (``n_events`` less ``n_scored``); neither happens in the canonical
windows, which lie outside excluded time by the analyzed-time rule, and whose
bins the restriction run scored finite (its sums raise otherwise).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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
)
from continuum_swr_replay.analysis.event_evidence.tetrode_holdout import (
    window_rows,
    window_sums,
)
from continuum_swr_replay.intervals import sample_step

CONTRASTS = {
    "full_minus_swr": ("full", "swr"),
    "swr_minus_none": ("swr", "none"),
    "full_minus_none": ("full", "none"),
}
"""Each contrast's name and its two models, ``a - b``."""

UNITS = {"per_spike": "n_spikes", "per_s": "duration_s"}
"""Each unit of a contrast and the window's column it is divided by."""

OUTSIDE_SWR_SUFFIX = "_outside_swr"
"""The suffix of a window's columns over its bins in the category
``immobile_outside_swr`` only (:func:`restriction_window_sums`)."""

OUTSIDE_SWR_COLUMNS = (
    "n_bins_outside_swr",
    "duration_outside_swr_s",
    *(f"{column}{OUTSIDE_SWR_SUFFIX}" for column in SCORE_COLUMNS),
)
"""The window table's columns over the bins outside SWRs."""

PART_UNITS = {"whole": tuple(UNITS), "outside_swr": tuple(UNITS)}
"""Each part of a window a contrast is read over and its units: the whole
window, or only its bins in the category ``immobile_outside_swr``, per spike in
them (``n_spikes_outside_swr``) and per second of them."""

IN_VISIT_SUFFIX = "_outside_swr_in_visit"
"""The suffix of a window's columns over its bins in the category
``immobile_outside_swr`` that are inside a well visit
(:func:`window_bin_fact_sums`)."""

BIN_FACT_COLUMNS = (
    "n_spikes_outside_swr",
    "n_bins_outside_swr_in_visit",
    "n_spikes_outside_swr_in_visit",
    *(f"{column}{IN_VISIT_SUFFIX}" for column in SCORE_COLUMNS),
)
"""The window table's columns from the per-bin facts
(:func:`window_bin_fact_sums`)."""

WINDOW_KEYS = ("window_id", "event_id", "kind", "group")
"""The columns that name a window (``epoch_windows``)."""

EVENT_TABLE_NOTE = "event_table_note"
"""The window run's settings key of the note it records when the event table it
read is not the one the held-out tetrode run read: ``event_table_sha256`` (the
window run's) and ``tetrode_holdout_event_table_sha256`` (the held-out tetrode
run's). The windows are still matched to the held-out tetrode run's
(:func:`window_spike_counts`), and the events summary marks every value read
from a run with the note pending."""


def restriction_window_sums(bins: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    """Each window's bins and summed scores of the three models.

    Parameters
    ----------
    bins : pd.DataFrame
        The epoch's per-bin table of the restriction run, one row per decode
        bin in time order: ``time`` (s), the six score columns
        (:data:`.nonlocal_restriction.SCORE_COLUMNS`), ``in_swr`` and
        ``category``.
    windows : pd.DataFrame
        The epoch's windows (``epoch_windows``): :data:`WINDOW_KEYS` and the
        first and last bin ``s`` and ``e`` (inclusive), positions in ``bins``.

    Returns
    -------
    pd.DataFrame
        One row per window, in ``windows``' order: :data:`WINDOW_KEYS`, ``s``,
        ``e``, ``n_bins``, ``n_bins_in_swr`` (bins with ``in_swr``),
        ``n_bins_immobile`` (bins in an immobile category), ``n_bins_excluded``
        (bins in excluded time), ``n_bins_not_finite`` (bins with any score
        that is not finite), ``duration_s`` (``n_bins`` times the time grid's
        step) and the float64 sum of each score column (NaN where a bin of the
        window has that score not finite); then the same over the window's
        bins in the category ``immobile_outside_swr`` only
        (:data:`OUTSIDE_SWR_COLUMNS`): ``n_bins_outside_swr``,
        ``duration_outside_swr_s`` and each score's float64 sum over them,
        ``<score>_outside_swr`` (0 without such bins, NaN where one of them has
        that score not finite).

    Raises
    ------
    ValueError
        If a window is empty or reaches outside ``bins``.
    """
    s = windows["s"].to_numpy(dtype=np.int64)
    e = windows["e"].to_numpy(dtype=np.int64)
    if (e < s).any() or (s < 0).any() or (e >= len(bins)).any():
        raise ValueError("a window is empty or reaches outside the per-bin table")
    rows = window_rows(s, e)
    scores = bins[list(SCORE_COLUMNS)].to_numpy(dtype=np.float64)[rows]
    finite = np.isfinite(scores)
    category = bins["category"].astype(str).to_numpy()[rows]
    outside = (category == IMMOBILE_OUTSIDE_SWR)[:, np.newaxis]
    count_names = [
        "n_bins_in_swr",
        "n_bins_immobile",
        "n_bins_excluded",
        "n_bins_not_finite",
        "n_bins_outside_swr",
    ]
    counts = np.column_stack(
        [
            bins["in_swr"].to_numpy(dtype=bool)[rows],
            np.isin(category, [IMMOBILE_IN_SWR, IMMOBILE_OUTSIDE_SWR]),
            category == "excluded",
            ~finite.all(axis=1),
            outside[:, 0],
        ]
    ).astype(np.float64)
    # every bin, then the bins outside SWRs only: each block is the bins with
    # a score not finite, then the scores summed over the finite ones
    blocks = [
        (~finite, np.where(finite, scores, 0.0)),
        (~finite & outside, np.where(finite & outside, scores, 0.0)),
    ]
    summed = window_sums(
        np.column_stack([counts, *(part for block in blocks for part in block)]),
        rows,
        s,
        e,
    )
    n_bins = e - s + 1
    step = sample_step(bins["time"].to_numpy(dtype=float))
    counted = {
        name: np.rint(summed[:, j]).astype(np.int64)
        for j, name in enumerate(count_names)
    }
    n_scores = len(SCORE_COLUMNS)
    block_sums = []
    for b in range(len(blocks)):
        start = len(count_names) + 2 * b * n_scores
        sums = summed[:, start + n_scores : start + 2 * n_scores]
        sums[summed[:, start : start + n_scores] > 0] = np.nan
        block_sums.append(sums)
    # the columns of the whole window first, as they were before the bins
    # outside SWRs were summed, then those
    out = windows[[*WINDOW_KEYS, "s", "e"]].reset_index(drop=True)
    out["n_bins"] = n_bins
    for name in count_names[:-1]:
        out[name] = counted[name]
    out["duration_s"] = n_bins * step
    for j, column in enumerate(SCORE_COLUMNS):
        out[column] = block_sums[0][:, j]
    out["n_bins_outside_swr"] = counted["n_bins_outside_swr"]
    out["duration_outside_swr_s"] = counted["n_bins_outside_swr"] * step
    for j, column in enumerate(SCORE_COLUMNS):
        out[f"{column}{OUTSIDE_SWR_SUFFIX}"] = block_sums[1][:, j]
    return out


def window_bin_fact_sums(
    bins: pd.DataFrame, facts: pd.DataFrame, windows: pd.DataFrame
) -> pd.DataFrame:
    """Each window's spikes, and its sums over its bins outside SWRs inside a
    well visit, from the epoch's per-bin facts.

    Parameters
    ----------
    bins : pd.DataFrame
        The epoch's per-bin table of the restriction run, as
        :func:`restriction_window_sums` takes it.
    facts : pd.DataFrame
        The epoch's per-bin facts (``bin_facts.bin_facts``), one row per bin of
        the same time grid: ``time``, ``n_spikes`` and ``in_well_visit``.
    windows : pd.DataFrame
        The epoch's windows, as :func:`restriction_window_sums` takes them.

    Returns
    -------
    pd.DataFrame
        One row per window, in ``windows``' order: ``n_spikes_bins`` (the
        window's spikes over all its bins, which must be the held-out tetrode
        run's ``n_spikes``) and :data:`BIN_FACT_COLUMNS`:
        ``n_spikes_outside_swr`` (spikes in its ``immobile_outside_swr``
        bins), ``n_bins_outside_swr_in_visit`` and
        ``n_spikes_outside_swr_in_visit`` (bins and spikes of those inside a
        well visit) and each score's float64 sum over the latter,
        ``<score>_outside_swr_in_visit`` (0 without such bins; the restriction
        run's scores are finite, its sums raise otherwise).

    Raises
    ------
    ValueError
        If the facts' time is not the per-bin table's, or a window is empty or
        reaches outside ``bins``.
    """
    if not np.array_equal(
        facts["time"].to_numpy(dtype=float), bins["time"].to_numpy(dtype=float)
    ):
        raise ValueError("the per-bin facts' time is not the per-bin table's")
    s = windows["s"].to_numpy(dtype=np.int64)
    e = windows["e"].to_numpy(dtype=np.int64)
    if (e < s).any() or (s < 0).any() or (e >= len(bins)).any():
        raise ValueError("a window is empty or reaches outside the per-bin table")
    rows = window_rows(s, e)
    spikes = facts["n_spikes"].to_numpy(dtype=np.float64)[rows]
    outside = bins["category"].astype(str).to_numpy()[rows] == IMMOBILE_OUTSIDE_SWR
    in_visit = outside & facts["in_well_visit"].to_numpy(dtype=bool)[rows]
    scores = bins[list(SCORE_COLUMNS)].to_numpy(dtype=np.float64)[rows]
    summed = window_sums(
        np.column_stack(
            [
                spikes,
                spikes * outside,
                in_visit,
                spikes * in_visit,
                np.where(in_visit[:, np.newaxis], scores, 0.0),
            ]
        ),
        rows,
        s,
        e,
    )
    out = pd.DataFrame(
        {
            name: np.rint(summed[:, j]).astype(np.int64)
            for j, name in enumerate(["n_spikes_bins", *BIN_FACT_COLUMNS[:3]])
        }
    )
    for j, column in enumerate(SCORE_COLUMNS):
        out[f"{column}{IN_VISIT_SUFFIX}"] = summed[:, 4 + j]
    return out


def outside_swr_window_counts(windows: pd.DataFrame) -> pd.DataFrame:
    """Per kind and group, the windows, those with bins outside SWRs, those with
    such bins but no spike in them (no value per spike there), and those with
    such bins outside a well visit (a window that runs past the final poke-out
    of its visit, or lies outside one).

    Parameters
    ----------
    windows : pd.DataFrame
        :func:`restriction_window_sums`' rows with :data:`BIN_FACT_COLUMNS`,
        ``kind`` and ``group``.

    Returns
    -------
    pd.DataFrame
        Indexed by ``kind`` and ``group``: ``n_windows``,
        ``n_with_bins_outside_swr``, ``n_without_spikes_outside_swr`` and
        ``n_with_bins_outside_swr_outside_visit``.
    """
    with_bins = windows["n_bins_outside_swr"] > 0
    return (
        windows.assign(
            n_windows=1,
            n_with_bins_outside_swr=with_bins,
            n_without_spikes_outside_swr=with_bins
            & (windows["n_spikes_outside_swr"] == 0),
            n_with_bins_outside_swr_outside_visit=windows["n_bins_outside_swr"]
            > windows["n_bins_outside_swr_in_visit"],
        )
        .groupby(["kind", "group"], sort=True)[
            [
                "n_windows",
                "n_with_bins_outside_swr",
                "n_without_spikes_outside_swr",
                "n_with_bins_outside_swr_outside_visit",
            ]
        ]
        .sum()
        .astype(np.int64)
    )


def window_spike_counts(holdout: pd.DataFrame, windows: pd.DataFrame) -> pd.Series:
    """Each window's spike count over every tetrode, from the held-out tetrode
    run's rows of the same epoch.

    Parameters
    ----------
    holdout : pd.DataFrame
        The epoch's ``tetrode_holdout`` table, one row per window and tetrode:
        :data:`WINDOW_KEYS`, ``tetrode``, ``n_bins`` and ``n_spikes``.
    windows : pd.DataFrame
        The epoch's windows (``epoch_windows``).

    Returns
    -------
    pd.Series
        ``n_spikes`` (int), indexed by ``window_id`` in ``windows``' order.

    Raises
    ------
    ValueError
        If the two runs' windows differ by id, event, kind, group or length
        (the held-out tetrode table does not hold a window's first bin, so a
        window of the same length at another place is not seen here; the
        runner stops unless both runs read the same participation tables, and
        records a note, :data:`EVENT_TABLE_NOTE`, when they read different
        event tables), or if a window does not have one row for each of the table's
        tetrodes (its spike count would miss a tetrode).
    """
    keys = list(WINDOW_KEYS)
    by_window = holdout.groupby(keys, sort=False)
    tetrodes = by_window["tetrode"].agg(["size", "nunique"])
    n_tetrodes = holdout["tetrode"].nunique()
    if not (tetrodes.to_numpy() == n_tetrodes).all():
        raise ValueError(
            "a window of the held-out tetrode run does not have one row per tetrode"
        )
    held = by_window["n_spikes"].sum().reset_index()
    held["n_bins"] = by_window["n_bins"].max().to_numpy()
    if not (by_window["n_bins"].nunique() == 1).all():
        raise ValueError("a window of the held-out tetrode run has two lengths")
    ours = windows[keys].assign(n_bins=windows["e"] - windows["s"] + 1)
    merged = ours.merge(
        held, on=[*keys, "n_bins"], how="outer", indicator=True, validate="one_to_one"
    )
    unmatched = merged[merged["_merge"] != "both"]
    if len(unmatched):
        raise ValueError(
            f"{len(unmatched)} windows are not the held-out tetrode run's: "
            f"{unmatched[keys + ['n_bins', '_merge']].head().to_dict('records')}"
        )
    spikes = merged.set_index("window_id")["n_spikes"].astype(np.int64)
    return spikes.reindex(windows["window_id"].to_numpy())


def contrast_column(contrast: str, family: str, unit: str, part: str = "whole") -> str:
    """The name of a window's contrast column, e.g. ``full_minus_swr_per_spike``,
    ``full_minus_swr_two_sided_per_s`` or, over the bins outside SWRs,
    ``full_minus_swr_outside_swr_per_s``."""
    where = OUTSIDE_SWR_SUFFIX if part == "outside_swr" else ""
    return f"{contrast}{FAMILY_SUFFIX[family]}{where}_{unit}"


def window_contrasts(
    windows: pd.DataFrame, parts: Sequence[str] = tuple(PART_UNITS)
) -> pd.DataFrame:
    """Each window's contrasts, per spike and per second, in both families,
    over the whole window and over its bins outside SWRs.

    Parameters
    ----------
    windows : pd.DataFrame
        :func:`restriction_window_sums`' rows with ``n_spikes``.
    parts : sequence of str, optional
        The parts of the window (:data:`PART_UNITS`); by default both. The part
        ``"outside_swr"`` reads the columns :data:`OUTSIDE_SWR_COLUMNS` and
        :data:`BIN_FACT_COLUMNS`.

    Returns
    -------
    pd.DataFrame
        ``windows`` with one column per part, contrast, family and unit
        (:func:`contrast_column`). Over the whole window: the contrast's window
        sum (nats) divided by ``n_spikes`` (NaN without spikes) or by
        ``duration_s``. So the two units do not cover the same windows: a
        window without spikes has a value per second and none per spike (153
        Local-rest controls in the cohort; the statistics' ``n_scored`` counts
        the windows with a value). Outside SWRs: the contrast summed over the
        window's ``immobile_outside_swr`` bins divided by their spikes,
        ``n_spikes_outside_swr`` (NaN without a spike there), or by their
        duration, ``duration_outside_swr_s``; NaN for a window with no such bin
        (an event wholly inside its SWR). Every contrast of a window with a bin
        in excluded time is NaN.
    """
    out = windows.copy()
    kept = out["n_bins_excluded"].to_numpy() == 0
    for part in _chosen(parts, PART_UNITS, "part"):
        if part == "outside_swr":
            seconds = out["duration_outside_swr_s"]
            spikes = out["n_spikes_outside_swr"]
            divisors = {
                "per_spike": spikes.where(spikes > 0).to_numpy(float),
                "per_s": seconds.where(seconds > 0).to_numpy(float),
            }
            where = OUTSIDE_SWR_SUFFIX
        else:
            divisors = {
                "per_spike": out["n_spikes"].where(out["n_spikes"] > 0).to_numpy(float),
                "per_s": out["duration_s"].to_numpy(float),
            }
            where = ""
        for contrast, (a, b) in CONTRASTS.items():
            for family, suffix in FAMILY_SUFFIX.items():
                difference = (
                    out[f"{a}{suffix}{where}"].to_numpy()
                    - out[f"{b}{suffix}{where}"].to_numpy()
                )
                for unit in PART_UNITS[part]:
                    out[contrast_column(contrast, family, unit, part)] = np.where(
                        kept, difference / divisors[unit], np.nan
                    )
    return out


def restriction_window_statistics(
    windows: pd.DataFrame,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    families: Sequence[str] | None = None,
    contrasts: Sequence[str] | None = None,
    units: Sequence[str] | None = None,
    kinds: Sequence[str] | None = None,
    parts: Sequence[str] = ("whole",),
) -> pd.DataFrame:
    """Each contrast's per-animal values, their mean and its interval, by
    score family, unit, kind of window, group and part of the window.

    For each part of the window, family, contrast and unit
    (:func:`window_contrasts`' columns) and each kind of window (``"event"``,
    ``"control"``):
    ``likelihood_advantage_statistics`` over that kind's windows, per group
    (an animal's value is the median over its epochs of each epoch's median
    window; the estimate is the mean over animals; the 95 % interval is from
    the nested bootstrap of animals, epochs and windows), as the held-out
    tetrode figure's ``tetrode_holdout.holdout_statistics`` does.

    Parameters
    ----------
    windows : pd.DataFrame
        The cohort's windows: :func:`restriction_window_sums`' rows with
        ``n_spikes``, ``epoch_id`` and ``animal``.
    n_boot, seed : int, optional
        The bootstrap's replicates and seed, the same for every row.
    families, contrasts, units, kinds : sequence of str, optional
        Only these score families (:data:`.nonlocal_restriction.FAMILY_SUFFIX`),
        contrasts (:data:`CONTRASTS`), units (:data:`UNITS`) and kinds of window;
        each defaults to all of them. Each row's bootstrap starts from ``seed``
        on its own, so a row is the same whichever others are computed.
    parts : sequence of str, optional
        The parts of the window (:data:`PART_UNITS`): ``"whole"`` (the
        default) and ``"outside_swr"``, the window's bins in the category
        ``immobile_outside_swr``, read per spike in them and per second. A
        window with no such bin, or (per spike) no spike in them, has no value
        there and is counted in ``n_events`` but not in ``n_scored``
        (:func:`outside_swr_window_counts` counts the two apart).

    Returns
    -------
    pd.DataFrame
        One row per chosen family, contrast, unit, kind, group and part: those
        keys, ``estimate``, ``ci_low``, ``ci_high``, ``animals`` (the animals'
        values), ``animal_names``, ``n_events`` (windows), ``n_scored``
        (windows with a value), ``n_animals`` and ``n_animals_positive``
        (animals whose value is above zero).

    Raises
    ------
    ValueError
        If a choice is unknown, or no window is of a chosen kind.
    """
    families = _chosen(families, FAMILY_SUFFIX, "family")
    contrasts = _chosen(contrasts, CONTRASTS, "contrast")
    units = _chosen(units, UNITS, "unit")
    parts = _chosen(parts, PART_UNITS, "part")
    scored = window_contrasts(windows, parts)
    if kinds is not None:
        missing = set(kinds) - set(scored["kind"])
        if missing:
            raise ValueError(f"no windows of kind {sorted(missing)}")
        scored = scored[scored["kind"].isin(kinds)]
    rows = []
    for part in parts:
        for family in families:
            for contrast in contrasts:
                for unit in (u for u in units if u in PART_UNITS[part]):
                    column = contrast_column(contrast, family, unit, part)
                    for kind, mine in scored.groupby("kind", sort=True):
                        statistics = likelihood_advantage_statistics(
                            mine, column=column, n_boot=n_boot, seed=seed
                        ).reset_index()
                        rows.append(
                            statistics.assign(
                                family=family,
                                contrast=contrast,
                                unit=unit,
                                kind=kind,
                                part=part,
                            )
                        )
    table = pd.concat(rows, ignore_index=True)
    table["n_animals_positive"] = [int((a > 0).sum()) for a in table["animals"]]
    keys = ["family", "contrast", "unit", "kind", "group", "part"]
    return table[keys + [c for c in table.columns if c not in keys]]


def _chosen(
    chosen: Sequence[str] | None, known: Mapping[str, object], name: str
) -> list[str]:
    """``chosen`` in ``known``'s order, or all of ``known`` when None.

    Raises
    ------
    ValueError
        If ``chosen`` names one that ``known`` does not hold.
    """
    if chosen is None:
        return list(known)
    unknown = set(chosen) - set(known)
    if unknown:
        raise ValueError(f"unknown {name}: {sorted(unknown)}")
    return [key for key in known if key in chosen]
