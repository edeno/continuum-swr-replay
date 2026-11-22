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
combination of :data:`TIME_SUM_KEYS`: ``category``, ``in_well_visit``,
``in_event_window`` (the bin lies in the window of a given event of either
group: the events detected from the held-out decode,
``scripts/event_evidence/detect_heldout_events.py``), ``swr_observed`` (SWR
detection was observed at the bin, the per-bin facts'), ``beyond_swr_margin_s``
(the largest margin of :data:`OUTSIDE_SWR_MARGINS_S` that the bin is farther
than from every bin of the scoring mask ``in_swr``, 0 when it is within the
smallest; :func:`swr_margin_bands`) and ``place_sampling_at_least_s`` (the
largest of :data:`PLACE_SAMPLING_THRESHOLDS_S` that the bin's
``place_sampling_s`` reaches, 0 below the smallest or where the place is
unknown; :func:`place_sampling_bands`). Every bin is in exactly one row, so
any union of rows is summed by adding them, and the sums over the first three
keys alone are the sums over every value of the other three.

**Outside SWRs** is :func:`.restriction_windows.outside_swr_bins`: the category
``immobile_outside_swr`` where SWR detection was observed. An immobile bin
outside every SWR of the source where detection was not observed (a gap in the
stored ripple band) is not known to be outside an SWR, so no sum outside SWRs
holds it; :func:`outside_swr_not_observed_counts` counts those left out. The
sums over all immobile time keep them: they are immobile time, inside an SWR
or not.

**All immobile time** (:func:`epoch_immobile_gains`; Figure 2g's first axis
draws the paired difference ``full - swr`` per spike):
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

- the gain per spike and per second (the second column of Figure 2g's first
  axis);
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
- the share outside those events, the rest: 1 minus the two shares, so the three
  add to 1 (Figure 2g's second axis draws the three). It holds the bins inside no
  event window and the bins of the windows the two shares leave out;
- the gain per spike and per second in the rest of that time, the bins inside
  no event window of any group (``in_event_window`` false): every event the
  time sums were given counts, whichever events the shares are given.

**Parts of the time outside SWRs** (:func:`epoch_outside_swr_subset_gains`,
:func:`outside_swr_subset_statistics`): the same gain, per spike and per
second, over only the bins outside SWRs inside a well visit that are farther
than a margin of :data:`OUTSIDE_SWR_MARGINS_S` from every bin of the scoring
mask (on each side), or whose ``place_sampling_s`` is at least a threshold of
:data:`PLACE_SAMPLING_THRESHOLDS_S`: the seconds of training data at the
animal's place under the bin's own fold (``tetrode_holdout.place_sampling``:
the decode's training bins outside the fold and its margin whose place bin is
the bin's own, times the bin width). With each, the share of the epoch's bins
outside SWRs inside a well visit that the part keeps.

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

from continuum_swr_replay.analysis.event_evidence.bin_facts import (
    require_swr_observed,
)
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
    WINDOW_BIN_COLUMNS,
    _check_facts_time,
    _window_bounds,
    not_observed_outside_swr_bins,
    outside_swr_bins,
)
from continuum_swr_replay.analysis.event_evidence.state_evidence.windows import (
    SWR_LABELS,
)
from continuum_swr_replay.analysis.event_evidence.tetrode_holdout import window_rows
from continuum_swr_replay.analysis.shared_statistics import animal_session_values
from continuum_swr_replay.intervals import sample_step

OUTSIDE_SWR_MARGINS_S = (0.1, 0.25, 0.5)
"""The margins (s, on each side) of the parts of the time outside SWRs that lie
farther than the margin from every bin of the scoring mask
(:func:`epoch_outside_swr_subset_gains`)."""

PLACE_SAMPLING_THRESHOLDS_S = (10.0, 60.0)
"""The thresholds (s) of the parts of the time outside SWRs whose
``place_sampling_s`` is at least the threshold
(:func:`epoch_outside_swr_subset_gains`)."""

TIME_SUM_KEYS = (
    "category",
    "in_well_visit",
    "in_event_window",
    "swr_observed",
    "beyond_swr_margin_s",
    "place_sampling_at_least_s",
)
"""The columns that name a row of an epoch's time sums
(:func:`restriction_time_sums`)."""

TIME_SUM_BIN_COLUMNS = (*WINDOW_BIN_COLUMNS, "place_sampling_s")
"""The per-bin table's columns :func:`restriction_time_sums` reads."""

MEASURES = (
    "gain_per_spike",
    "gain_per_s",
    "share_in_no_swr_events",
    "share_in_swr_events_outside_swr",
    "share_outside_events",
    "outside_events_gain_per_spike",
    "outside_events_gain_per_s",
)
"""The per-epoch measures of each family over the immobile time outside SWRs,
by their column stem (:func:`epoch_outside_swr_gains`)."""

SUBSET_MEASURES = ("gain_per_spike", "gain_per_s")
"""The per-epoch measures of each family over a part of the time outside SWRs,
by their column stem (:func:`epoch_outside_swr_subset_gains`)."""

SUBSETS = {
    "beyond_swr_margin": ("beyond_swr_margin_s", OUTSIDE_SWR_MARGINS_S),
    "place_sampling": ("place_sampling_at_least_s", PLACE_SAMPLING_THRESHOLDS_S),
}
"""Each kind of part of the time outside SWRs: the time sums' key it is chosen
by and its thresholds (s)."""

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


def swr_margin_bands(in_swr: np.ndarray, bin_s: float) -> np.ndarray:
    """Each bin's largest margin of :data:`OUTSIDE_SWR_MARGINS_S` that it is
    farther than from every bin of the scoring mask.

    The distance is counted in bins: a margin of ``m`` s is
    ``round(m / bin_s)`` bins, and a bin is farther than it when the nearest
    bin of the mask, on either side, is more than that many bins away.

    Parameters
    ----------
    in_swr : np.ndarray of bool, shape (n_time,)
        The scoring mask (the per-bin table's ``in_swr``).
    bin_s : float
        The time step (s).

    Returns
    -------
    np.ndarray of float64, shape (n_time,)
        The largest margin (s) the bin is farther than, or 0.0 when it is
        within the smallest (every bin of the mask is). With no bin in the
        mask, every bin is farther than every margin.
    """
    in_swr = np.asarray(in_swr, dtype=bool)
    position = np.arange(in_swr.size)
    mask = np.flatnonzero(in_swr)
    if mask.size == 0:
        distance = np.full(in_swr.size, np.inf)
    else:
        after = np.searchsorted(mask, position)
        to_next = np.where(
            after < mask.size, mask[np.minimum(after, mask.size - 1)] - position, np.inf
        )
        to_previous = np.where(
            after > 0, position - mask[np.maximum(after - 1, 0)], np.inf
        )
        distance = np.minimum(to_next, to_previous)
    out = np.zeros(in_swr.size)
    for margin in OUTSIDE_SWR_MARGINS_S:
        out[distance > round(margin / bin_s)] = margin
    return out


def place_sampling_bands(place_sampling_s: np.ndarray) -> np.ndarray:
    """Each bin's largest threshold of :data:`PLACE_SAMPLING_THRESHOLDS_S` that
    its ``place_sampling_s`` reaches.

    Parameters
    ----------
    place_sampling_s : np.ndarray, shape (n_time,)
        The seconds of training data at the animal's place under the bin's
        fold (the per-bin table's ``place_sampling_s``), NaN where the place is
        unknown.

    Returns
    -------
    np.ndarray of float64, shape (n_time,)
        The largest threshold (s) the value is at least, or 0.0 below the
        smallest or where it is NaN.
    """
    sampling = np.asarray(place_sampling_s, dtype=np.float64)
    out = np.zeros(sampling.size)
    for threshold in PLACE_SAMPLING_THRESHOLDS_S:
        out[sampling >= threshold] = threshold
    return out


def restriction_time_sums(
    bins: pd.DataFrame, facts: pd.DataFrame, windows: pd.DataFrame
) -> pd.DataFrame:
    """One epoch's bins, spikes and summed scores by category, well visit,
    event window, SWR detection observed, margin from the scoring mask and
    place sampling.

    Parameters
    ----------
    bins : pd.DataFrame
        The epoch's per-bin table of the restriction run, one row per decode
        bin in time order: :data:`TIME_SUM_BIN_COLUMNS`, of which ``time``
        (s), ``category``, the six score columns
        (:data:`.nonlocal_restriction.SCORE_COLUMNS`), ``in_swr`` and
        ``place_sampling_s`` are read.
    facts : pd.DataFrame
        The epoch's per-bin facts (``bin_facts.bin_facts``) on the same grid:
        ``time``, ``n_spikes``, ``in_well_visit`` and ``swr_observed``.
    windows : pd.DataFrame
        The epoch's windows: ``kind`` and the first and last bin ``s`` and
        ``e`` (inclusive). Only the ``"event"`` windows mark
        ``in_event_window``.

    Returns
    -------
    pd.DataFrame
        One row for each combination of :data:`TIME_SUM_KEYS`
        (:data:`.nonlocal_restriction.CATEGORIES` by ``in_well_visit`` by
        ``in_event_window`` by ``swr_observed`` by ``beyond_swr_margin_s``
        (0 and :data:`OUTSIDE_SWR_MARGINS_S`) by ``place_sampling_at_least_s``
        (0 and :data:`PLACE_SAMPLING_THRESHOLDS_S`), 384 rows, those without
        bins included): ``n_bins``, ``n_spikes`` (int64) and each score's
        float64 sum (0 without bins).

    Raises
    ------
    ValueError
        If the facts' time is not the per-bin table's, the facts have no
        ``swr_observed``, a bin's category is not one of the categories, or a
        window reaches outside ``bins``.
    """
    _check_facts_time(bins, facts)
    require_swr_observed(facts)
    category = bins["category"].astype(str).to_numpy()
    if unknown := sorted(set(category) - set(CATEGORIES)):
        raise ValueError(f"bins in unknown categories: {unknown}")
    s, e = _window_bounds(windows[windows["kind"] == "event"], len(bins))
    in_event = np.zeros(len(bins), dtype=bool)
    in_event[window_rows(s, e)] = True
    frame = bins[list(SCORE_COLUMNS)].astype(np.float64)
    frame.insert(0, "n_spikes", facts["n_spikes"].to_numpy(dtype=np.int64))
    frame.insert(0, "n_bins", np.int64(1))
    bin_s = sample_step(bins["time"].to_numpy(dtype=float))
    keys = [
        pd.Series(category, name="category"),
        pd.Series(facts["in_well_visit"].to_numpy(dtype=bool), name="in_well_visit"),
        pd.Series(in_event, name="in_event_window"),
        pd.Series(facts["swr_observed"].to_numpy(dtype=bool), name="swr_observed"),
        pd.Series(
            swr_margin_bands(bins["in_swr"].to_numpy(dtype=bool), bin_s),
            name="beyond_swr_margin_s",
        ),
        pd.Series(
            place_sampling_bands(bins["place_sampling_s"].to_numpy()),
            name="place_sampling_at_least_s",
        ),
    ]
    summed = frame.reset_index(drop=True).groupby(keys).sum()
    every = pd.MultiIndex.from_tuples(
        list(
            itertools.product(
                CATEGORIES,
                (False, True),
                (False, True),
                (False, True),
                (0.0, *OUTSIDE_SWR_MARGINS_S),
                (0.0, *PLACE_SAMPLING_THRESHOLDS_S),
            )
        ),
        names=list(TIME_SUM_KEYS),
    )
    out = summed.reindex(every, fill_value=0).reset_index()
    out[["n_bins", "n_spikes"]] = out[["n_bins", "n_spikes"]].astype(np.int64)
    return out


def _chosen_time(time_sums: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    """Each epoch's sums over ``rows`` of the time sums that are inside a well
    visit: ``n_bins``, ``n_spikes`` and the scores, indexed by ``epoch_id``,
    every epoch of ``time_sums`` included."""
    rows = np.asarray(rows, dtype=bool) & time_sums["in_well_visit"].to_numpy(bool)
    columns = ["n_bins", "n_spikes", *SCORE_COLUMNS]
    epochs = sorted(time_sums["epoch_id"].unique())
    return (
        time_sums[rows].groupby("epoch_id")[columns].sum().reindex(epochs, fill_value=0)
    )


def _outside_swr_rows(
    time_sums: pd.DataFrame,
    *,
    outside_events: bool = False,
    beyond_swr_margin_s: float = 0.0,
    place_sampling_at_least_s: float = 0.0,
) -> np.ndarray:
    """The rows of the time sums outside SWRs
    (:func:`.restriction_windows.outside_swr_bins`), outside every event window
    only when asked, and only those farther than a margin from the scoring
    mask or at least a place sampling, when given (0 keeps every row)."""
    rows = outside_swr_bins(time_sums["category"], time_sums["swr_observed"])
    if outside_events:
        rows &= ~time_sums["in_event_window"].to_numpy(bool)
    for key, at_least, known in (
        ("beyond_swr_margin_s", beyond_swr_margin_s, OUTSIDE_SWR_MARGINS_S),
        (
            "place_sampling_at_least_s",
            place_sampling_at_least_s,
            PLACE_SAMPLING_THRESHOLDS_S,
        ),
    ):
        if at_least not in (0.0, *known):
            raise ValueError(f"{key} {at_least} is not 0 or one of {known}")
        rows &= time_sums[key].to_numpy(float) >= at_least
    return rows


def _check_time_sums(time_sums: pd.DataFrame, bin_s: pd.Series | None) -> None:
    """Refuse time sums of an earlier design (without every key of
    :data:`TIME_SUM_KEYS`), with an epoch's row twice, or (unless ``bin_s`` is
    None) an epoch without a time step."""
    if missing := [key for key in TIME_SUM_KEYS if key not in time_sums.columns]:
        raise ValueError(
            f"the time sums have no {missing}: they were made before those keys. "
            "Rerun scripts/event_evidence/measure_bin_facts.py and "
            "scripts/event_evidence/detect_heldout_events.py"
        )
    if time_sums.duplicated(["epoch_id", *TIME_SUM_KEYS]).any():
        raise ValueError("an epoch has a row of the time sums twice")
    if bin_s is None:
        return
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
        ``share_in_swr_events_outside_swr``, ``share_outside_events`` (1 minus
        the two shares), ``outside_events_full_minus_swr``,
        ``outside_events_gain_per_spike`` and ``outside_events_gain_per_s``
        (NaN without spikes, bins or a positive total, as each divides).

    Raises
    ------
    ValueError
        If an epoch has a row of the time sums twice or no time step.
    """
    _check_time_sums(time_sums, bin_s)
    total = _chosen_time(time_sums, _outside_swr_rows(time_sums))
    rest = _chosen_time(time_sums, _outside_swr_rows(time_sums, outside_events=True))
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
        out[f"share_outside_events{suffix}"] = (
            1.0
            - out[f"share_in_no_swr_events{suffix}"]
            - out[f"share_in_swr_events_outside_swr{suffix}"]
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
    total = _chosen_time(
        time_sums, time_sums["category"].isin([IMMOBILE_IN_SWR, IMMOBILE_OUTSIDE_SWR])
    )
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


def outside_swr_not_observed_counts(time_sums: pd.DataFrame) -> dict[str, int]:
    """The immobile bins outside every SWR of the source, inside a well visit,
    where SWR detection was not observed: left out of every sum outside SWRs
    (:func:`.restriction_windows.not_observed_outside_swr_bins`).

    Parameters
    ----------
    time_sums : pd.DataFrame
        The cohort's time sums (:func:`restriction_time_sums`' rows with
        ``epoch_id``).

    Returns
    -------
    dict
        ``n_bins`` (those bins over the cohort) and ``n_epochs`` (epochs with
        at least one).

    Raises
    ------
    ValueError
        If the time sums are of an earlier design (:data:`TIME_SUM_KEYS`).
    """
    _check_time_sums(time_sums, None)
    left_out = not_observed_outside_swr_bins(
        time_sums["category"], time_sums["swr_observed"]
    )
    by_epoch = _chosen_time(time_sums, left_out)["n_bins"]
    return {"n_bins": int(by_epoch.sum()), "n_epochs": int((by_epoch > 0).sum())}


def epoch_outside_swr_subset_gains(
    time_sums: pd.DataFrame,
    bin_s: pd.Series,
    *,
    beyond_swr_margin_s: float = 0.0,
    place_sampling_at_least_s: float = 0.0,
) -> pd.DataFrame:
    """Each epoch's ``full - swr`` over a part of its immobile bins outside
    SWRs inside a well visit, per spike and per second, and the share of those
    bins the part keeps, both score families.

    Parameters
    ----------
    time_sums : pd.DataFrame
        The cohort's time sums, as :func:`epoch_outside_swr_gains` takes them.
    bin_s : pd.Series
        Each epoch's decode time step (s), indexed by ``epoch_id``.
    beyond_swr_margin_s : float, optional
        Only the bins farther than this margin (s, one of
        :data:`OUTSIDE_SWR_MARGINS_S`) from every bin of the scoring mask; 0
        (the default) keeps every bin.
    place_sampling_at_least_s : float, optional
        Only the bins whose ``place_sampling_s`` is at least this (s, one of
        :data:`PLACE_SAMPLING_THRESHOLDS_S`); 0 (the default) keeps every bin.

    Returns
    -------
    pd.DataFrame
        One row per epoch of ``time_sums``, sorted by ``epoch_id``:
        ``epoch_id``, ``animal``, ``n_bins``, ``n_spikes`` and ``seconds`` of
        the kept bins, and for each family (suffix ``""`` for the filter score,
        ``"_two_sided"``): ``full_minus_swr`` (nats), ``gain_per_spike``,
        ``gain_per_s`` (NaN without spikes or bins); and ``share_of_bins``
        (kept bins over every bin outside SWRs inside a well visit, NaN
        without such bins; one for both families).

    Raises
    ------
    ValueError
        If an epoch has a row of the time sums twice or no time step, or a
        threshold is not 0 or one of the named ones.
    """
    _check_time_sums(time_sums, bin_s)
    every = _chosen_time(time_sums, _outside_swr_rows(time_sums))
    kept = _chosen_time(
        time_sums,
        _outside_swr_rows(
            time_sums,
            beyond_swr_margin_s=beyond_swr_margin_s,
            place_sampling_at_least_s=place_sampling_at_least_s,
        ),
    )
    animals = time_sums.groupby("epoch_id")["animal"].first()
    step = bin_s.reindex(kept.index).to_numpy(dtype=float)
    out = pd.DataFrame(
        {
            "epoch_id": kept.index,
            "animal": animals.reindex(kept.index).to_numpy(),
            "n_bins": kept["n_bins"].to_numpy(),
            "n_spikes": kept["n_spikes"].to_numpy(),
            "seconds": kept["n_bins"].to_numpy() * step,
        }
    )
    out["share_of_bins"] = list(map(ratio_or_nan, kept["n_bins"], every["n_bins"]))
    for suffix in FAMILY_SUFFIX.values():
        gain = (kept[f"full{suffix}"] - kept[f"swr{suffix}"]).to_numpy()
        out[f"full_minus_swr{suffix}"] = gain
        out[f"gain_per_spike{suffix}"] = list(map(ratio_or_nan, gain, out["n_spikes"]))
        out[f"gain_per_s{suffix}"] = list(map(ratio_or_nan, gain, out["seconds"]))
    return out


def outside_swr_subset_statistics(
    time_sums: pd.DataFrame, bin_s: pd.Series, *, n_boot: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Each part of the time outside SWRs (:data:`SUBSETS`): each measure's
    per-animal values, their mean and its interval, by score family.

    Parameters
    ----------
    time_sums : pd.DataFrame
        The cohort's time sums, as :func:`epoch_outside_swr_gains` takes them.
    bin_s : pd.Series
        Each epoch's decode time step (s), indexed by ``epoch_id``.
    n_boot, seed : int, optional
        The bootstrap's replicates and seed, the same for every row.

    Returns
    -------
    pd.DataFrame
        One row per part (``subset``, a key of :data:`SUBSETS`, and its
        ``threshold_s``), family and measure (:data:`SUBSET_MEASURES`), with
        the columns of :func:`outside_swr_gain_statistics`, over
        :func:`epoch_outside_swr_subset_gains`' rows.
    """
    parts = []
    for subset, (key, thresholds) in SUBSETS.items():
        for threshold in thresholds:
            epochs = epoch_outside_swr_subset_gains(
                time_sums, bin_s, **{key: threshold}
            )
            table = _epoch_measure_statistics(
                epochs, SUBSET_MEASURES, n_boot=n_boot, seed=seed
            )
            table.insert(0, "threshold_s", threshold)
            table.insert(0, "subset", subset)
            parts.append(table)
    return pd.concat(parts, ignore_index=True)


def outside_swr_subset_shares(
    time_sums: pd.DataFrame, bin_s: pd.Series
) -> pd.DataFrame:
    """Each part of the time outside SWRs (:data:`SUBSETS`): the share of the
    bins outside SWRs inside a well visit it keeps, over the cohort, with no
    interval.

    The share is the same in both score families, and it is a description of
    the part, not an effect: an animal's value is the median over its epochs
    (``shared_statistics.animal_session_values``), the estimate the mean over
    animals, and the lowest animal's value is given beside it. (A t interval
    of a share near 1 can pass 1.)

    Parameters
    ----------
    time_sums : pd.DataFrame
        The cohort's time sums, as :func:`epoch_outside_swr_gains` takes them.
    bin_s : pd.Series
        Each epoch's decode time step (s), indexed by ``epoch_id``.

    Returns
    -------
    pd.DataFrame
        Indexed by ``subset`` and ``threshold_s``: ``estimate``,
        ``animal_min``, ``n_animals`` and ``n_epochs`` (epochs with a share).
    """
    rows = []
    for subset, (key, thresholds) in SUBSETS.items():
        for threshold in thresholds:
            epochs = epoch_outside_swr_subset_gains(
                time_sums, bin_s, **{key: threshold}
            )
            per_animal = animal_session_values(
                epochs["share_of_bins"], epochs["animal"], epochs["epoch_id"], "median"
            )
            rows.append(
                {
                    "subset": subset,
                    "threshold_s": threshold,
                    "estimate": float(per_animal.mean()),
                    "animal_min": float(per_animal.min()),
                    "n_animals": len(per_animal),
                    "n_epochs": int(epochs["share_of_bins"].notna().sum()),
                }
            )
    return pd.DataFrame(rows).set_index(["subset", "threshold_s"])


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
RESTRICTION_FIGURE_DIFFERENCE = "full_minus_swr_per_spike"
"""Figure 2g's first axis, first column: the full decoder minus the SWR-only
decoder per spike of all the immobile time inside a well visit, each epoch's
paired difference (:data:`IMMOBILE_MEASURES`)."""
RESTRICTION_FIGURE_OUTSIDE_SWR = "gain_per_spike"
"""Figure 2g's first axis, second column: the full decoder minus the SWR-only
decoder per spike of the immobile time outside SWRs inside a well visit
(:data:`MEASURES`)."""
RESTRICTION_FIGURE_SHARES = (
    "share_in_no_swr_events",
    "share_in_swr_events_outside_swr",
    "share_outside_events",
)
"""Figure 2g's second axis: where that gain outside SWRs sits, as shares of it
(:data:`MEASURES`), in drawing order: in the no-SWR events, in the time outside
the SWR of the events partly in one (both detected from the held-out decode),
and outside those events."""
RESTRICTION_FIGURE = {
    "parts": ["outside_swr"],
    "families": [RESTRICTION_FIGURE_FAMILY],
    "contrasts": ["full_minus_swr"],
    "units": ["per_spike"],
    "kinds": ["event"],
}
"""The choice of ``restriction_windows.restriction_window_statistics`` (its
keywords) for the full decoder minus the SWR-only decoder per spike over the
bins outside SWRs of each event, which the events summary reports (Figure 2g
drew it before it drew the total outside SWRs)."""
RESTRICTION_FIGURE_GROUPS = (SWR_LABELS[False], SWR_LABELS[True])
"""The groups of :data:`RESTRICTION_FIGURE`, in order: events with no SWR, then
the time outside the SWR of the events that overlap one."""


def restriction_figure_statistics(
    windows: pd.DataFrame, time_sums: pd.DataFrame, bin_s: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Figure 2g's statistics, as the events summary computes them.

    Parameters
    ----------
    windows : pd.DataFrame
        The held-out event run's events of every epoch
        (``scripts/event_evidence/detect_heldout_events.py``), those that start
        inside a well visit (``starts_in_well_visit``), as
        :func:`epoch_outside_swr_gains` takes them.
    time_sums : pd.DataFrame
        The held-out event run's time sums of every epoch
        (:func:`restriction_time_sums`).
    bin_s : pd.Series
        Each epoch's decode time step (s), indexed by ``epoch_id``.

    Returns
    -------
    decoders : pd.DataFrame
        :func:`immobile_gain_statistics` over the immobile time inside a well
        visit, indexed by ``family`` and ``measure`` (the first axis' first
        column).
    outside : pd.DataFrame
        :func:`outside_swr_gain_statistics` over the immobile time outside SWRs
        inside a well visit, with the shares of it in the events of
        ``windows``, indexed by ``family`` and ``measure`` (the first axis'
        second column and the second axis).
    """
    decoders = immobile_gain_statistics(
        epoch_immobile_gains(time_sums, bin_s)
    ).set_index(["family", "measure"])
    outside = outside_swr_gain_statistics(
        epoch_outside_swr_gains(time_sums, bin_s, windows)
    ).set_index(["family", "measure"])
    return decoders, outside
