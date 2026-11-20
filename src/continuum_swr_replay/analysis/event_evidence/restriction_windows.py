"""The non-local restriction models over the windows of Figure 2's panel h.

The per-bin scores of the three models (``none``, ``swr``, ``full``;
:mod:`.nonlocal_restriction`) summed over each canonical non-local event and
over its matched Local-rest control window: the windows of the held-out tetrode
score (``state_evidence.windows.epoch_windows``), so the two methods are read on
the same bins and summarized by the same statistic
(``likelihood_advantage.likelihood_advantage_statistics``: one value per animal,
their mean and its nested-bootstrap interval). The held-out encodings are not
the same: the restriction run's folds are whole trials
(``tetrode_holdout.trial_folds``), the held-out tetrode run's are contiguous
blocks of time (``tetrode_holdout.time_blocks``), so the two differ in more
than what is scored.

A window's contrast ``a - b`` (:data:`CONTRASTS`) is its sum of ``a``'s per-bin
score less its sum of ``b``'s, in nats, for each score family
(:data:`.nonlocal_restriction.FAMILY_SUFFIX`). It is read two ways:

- per spike: divided by the window's spike count over every tetrode (the
  held-out tetrode run's ``n_spikes``, summed over tetrodes), NaN without
  spikes, as ``tetrode_holdout.window_scores`` does;
- per second: divided by the window's duration, its bins times the decode's
  time step (``intervals.sample_step`` of the per-bin table's time).

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

WINDOW_KEYS = ("window_id", "event_id", "kind", "group")
"""The columns that name a window (``epoch_windows``)."""


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
        window has that score not finite).

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
    counts = np.column_stack(
        [
            bins["in_swr"].to_numpy(dtype=bool)[rows],
            np.isin(category, [IMMOBILE_IN_SWR, IMMOBILE_OUTSIDE_SWR]),
            category == "excluded",
            ~finite.all(axis=1),
            ~finite,
        ]
    ).astype(np.float64)
    summed = window_sums(
        np.column_stack([counts, np.where(finite, scores, 0.0)]), rows, s, e
    )
    n_counts = 4
    n_not_finite = summed[:, n_counts : n_counts + len(SCORE_COLUMNS)]
    sums = summed[:, n_counts + len(SCORE_COLUMNS) :]
    sums[n_not_finite > 0] = np.nan
    n_bins = e - s + 1
    out = windows[[*WINDOW_KEYS, "s", "e"]].reset_index(drop=True)
    out["n_bins"] = n_bins
    for j, name in enumerate(
        ["n_bins_in_swr", "n_bins_immobile", "n_bins_excluded", "n_bins_not_finite"]
    ):
        out[name] = np.rint(summed[:, j]).astype(np.int64)
    out["duration_s"] = n_bins * sample_step(bins["time"].to_numpy(dtype=float))
    for j, column in enumerate(SCORE_COLUMNS):
        out[column] = sums[:, j]
    return out


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
        runner checks that both runs read the same event and participation
        tables), or if a window does not have one row for each of the table's
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


def contrast_column(contrast: str, family: str, unit: str) -> str:
    """The name of a window's contrast column, e.g. ``full_minus_swr_per_spike``
    or ``full_minus_swr_two_sided_per_s``."""
    return f"{contrast}{FAMILY_SUFFIX[family]}_{unit}"


def window_contrasts(windows: pd.DataFrame) -> pd.DataFrame:
    """Each window's contrasts, per spike and per second, in both families.

    Parameters
    ----------
    windows : pd.DataFrame
        :func:`restriction_window_sums`' rows with ``n_spikes``.

    Returns
    -------
    pd.DataFrame
        ``windows`` with one column per contrast, family and unit
        (:func:`contrast_column`): the contrast's window sum (nats) divided by
        ``n_spikes`` (NaN without spikes) or by ``duration_s``. Every contrast
        of a window with a bin in excluded time is NaN. So the two units do not
        cover the same windows: a window without spikes has a value per second
        and none per spike (153 Local-rest controls in the cohort; the
        statistics' ``n_scored`` counts the windows with a value).
    """
    out = windows.copy()
    kept = out["n_bins_excluded"].to_numpy() == 0
    divisors = {
        "per_spike": out["n_spikes"].where(out["n_spikes"] > 0).to_numpy(float),
        "per_s": out["duration_s"].to_numpy(float),
    }
    for contrast, (a, b) in CONTRASTS.items():
        for family, suffix in FAMILY_SUFFIX.items():
            difference = out[f"{a}{suffix}"].to_numpy() - out[f"{b}{suffix}"].to_numpy()
            for unit in UNITS:
                out[contrast_column(contrast, family, unit)] = np.where(
                    kept, difference / divisors[unit], np.nan
                )
    return out


def restriction_window_statistics(
    windows: pd.DataFrame, *, n_boot: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Each contrast's per-animal values, their mean and its interval, by
    score family, unit, kind of window and group.

    For each family, contrast and unit (:func:`window_contrasts`' columns) and
    each kind of window (``"event"``, ``"control"``):
    ``likelihood_advantage_statistics`` over that kind's windows, per group
    (an animal's value is the median over its epochs of each epoch's median
    window; the estimate is the mean over animals; the 95 % interval is from
    the nested bootstrap of animals, epochs and windows), as panel h's
    ``tetrode_holdout.holdout_statistics`` does.

    Parameters
    ----------
    windows : pd.DataFrame
        The cohort's windows: :func:`restriction_window_sums`' rows with
        ``n_spikes``, ``epoch_id`` and ``animal``.
    n_boot, seed : int, optional
        The bootstrap's replicates and seed, the same for every row.

    Returns
    -------
    pd.DataFrame
        One row per family, contrast, unit, kind and group: those keys,
        ``estimate``, ``ci_low``, ``ci_high``, ``animals`` (the animals'
        values), ``animal_names``, ``n_events`` (windows), ``n_scored``
        (windows with a value) and ``n_animals``.
    """
    scored = window_contrasts(windows)
    parts = []
    for family in FAMILY_SUFFIX:
        for contrast in CONTRASTS:
            for unit in UNITS:
                column = contrast_column(contrast, family, unit)
                for kind, mine in scored.groupby("kind", sort=True):
                    part = likelihood_advantage_statistics(
                        mine, column=column, n_boot=n_boot, seed=seed
                    ).reset_index()
                    parts.append(
                        part.assign(
                            family=family, contrast=contrast, unit=unit, kind=kind
                        )
                    )
    table = pd.concat(parts, ignore_index=True)
    keys = ["family", "contrast", "unit", "kind", "group"]
    return table[keys + [c for c in table.columns if c not in keys]]
