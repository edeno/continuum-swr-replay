"""Summary statistics of the per-window table, over epochs nested in animals.

About 218 epochs are nested in 5 animals, and each epoch holds hundreds to
thousands of windows that are not independent. Each epoch is reduced to one
value, and the inference takes the animal as the unit of replication.

- **Each epoch's value.**
  - For an evidence metric, and for its paired bout - matched difference, its
    median or its mean over windows by the kind of metric. The rule is set in
    one place, :data:`EPOCH_SUMMARY` (read by :func:`epoch_summary_kind`):
    - the **median** for the log-likelihood ratios: local, remote against the
      animal or a mismatched location, local against S's best remote bin and
      the No-Spike log-ratio, with their per-spike versions. They are
      heavy-tailed: a burst of spikes on most tetrodes scores far below zero.
      In one pilot chimi epoch three of 81 immobile No-Spike bouts (86 to 137
      spikes on 19 to 22 tetrodes) score -1641, -1407 and -1016 nats under the
      running-only encoding, so that epoch's mean is -45 while its median is
      +5.4.
    - the **mean** for expected counts, counts, shares and distances: E_C,
      C's spike count, silent shares, MAP_S's distances and the
      coordinated-silence counts. They are bounded and deterministic or nearly
      so, and a median can sit on ties: a bout and its matched window at the
      same position share E_C exactly.
    - The other summary is reported as secondary. Both were chosen on the five
      pilot (development) epochs, before the 218-epoch run that produces the
      reported numbers.
  - For a calibration curve, the epoch's own OLS slope (:func:`epoch_slopes`).
  - For coordinated silence, the epoch's own ratio of sums of silent-run
    counts.
- **Primary inference, at the animal level.** An animal's summary is the mean
  of its epochs' values. With A animals:
  - ``{p}animal_estimate`` is the mean of the animal summaries, each animal
    weighted equally;
  - ``{p}animal_ci_lo`` / ``{p}animal_ci_hi`` is its 95 % interval from the
    nested bootstrap (:func:`nested_mean_intervals`: animals, then epochs
    within each): the estimate +/- t(0.975, A - 1) sqrt(A / (A - 1)) times the
    replicates' SD
    (:func:`~continuum_swr_replay.analysis.shared_statistics.bootstrap_interval`).
    The percentiles of the replicates would under-cover with five animals
    (Saravanan, Berman & Sober 2020, section 3.1.3). With one epoch per
    animal this interval is the t-interval over the animal summaries;
    resampling the epochs within each animal widens it;
  - ``{p}p_animal_t`` is the two-sided one-sample t-test of the animal
    summaries, A - 1 degrees of freedom (:func:`animal_t_test`; with balanced
    data, the intercept test of a random-intercept model);
  - ``{p}animals_pos`` and ``{p}epochs_pos`` are k of A animals and k of N
    epochs with a value > 0.
  - The interval and the test are NaN with fewer than two animals.
- **Secondary.**
  - ``{p}epoch_estimate`` with ``{p}epoch_ci_lo`` / ``{p}epoch_ci_hi``: the
    mean of the epoch values, each epoch weighted equally, with its 95 %
    interval from the same replicates by the same rule.
  - ``{p}p_epoch_signflip``: the sign-flip p over epochs (:func:`sign_flip_p`),
    which treats the epochs as independent and so ignores the animals.
- **Column scheme.** The prefix ``{p}`` says what the columns describe: "" a
  bout metric, ``diff_`` its paired difference, ``other_`` (``other_diff_``)
  the same over the secondary epoch summary, ``slope_`` a calibration curve,
  and ``excess_{e}_`` coordinated silence's excess over local coding under
  encoding ``e`` (:data:`ENCODING_LABELS`). ``epoch_summary`` names each row's
  primary epoch summary.
  Calibration bins carry only the estimates and their intervals.
- **Descriptive.** Window-level counts, medians, quartiles, shares > 0 and
  means, pooled over epochs.
- **Calibration verdicts read the animal-level interval.** ``rises`` and ``top_positive``
  say "yes" when it lies above 0, else "no", but only when the top and the
  bottom probability bins each hold at least :data:`MIN_BIN_WINDOWS` windows,
  pooled over epochs (otherwise :data:`TOO_FEW_WINDOWS`), and at least two
  animals give the interval (otherwise :data:`TOO_FEW_ANIMALS`). An epoch's
  slope needs two bins of that many windows.
- **Window level: opt-in, not for inference** (``window_level=True``). Columns
  ending ``_window_level`` resample or sign-flip windows as if they were
  independent. They ignore the nesting, are kept only so the prototype's
  statistics can be compared, and take most of a summary's run time.

Reference: Saravanan V, Berman GJ, Sober SJ (2020). Application of the
hierarchical bootstrap to multi-level data in neuroscience. Neurons,
Behavior, Data Analysis and Theory 3(5). PMID 33644783, PMC7906290.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import t as student_t

from continuum_swr_replay.analysis.event_definition.well_visits import in_well_visit
from continuum_swr_replay.analysis.event_evidence.state_evidence.windows import (
    REGIMES,
    STATES,
    SWR_LABELS,
    Seed,
    _with_keys,
    keyed_rng,
)
from continuum_swr_replay.analysis.shared_statistics import (
    bootstrap_interval,
    nested_bootstrap_draws,
    nested_codes,
)

ENCODING_LABELS = {
    "runcv": "running-only, time-held-out",
    "run": "running-only",
    "can": "canonical",
}
"""The evidence columns' prefixes, one per encoding (:data:`ENCODING_HELD_OUT`)."""

PRIMARY_ENCODING = "runcv"

ENCODING_HELD_OUT = {
    "runcv": "primary: fit on the running bins of the other fold's alternating "
    "30 s blocks, so a window's own spikes are not in it, but for the bins of a "
    "window that straddles a block edge (cv_insample_share), and an edge bin "
    "next to the other fold's running bins, which the library's padding of each "
    "training run by about one bin can reach",
    "run": "secondary: fit on every running bin, so it includes running "
    "windows' own spikes",
    "can": "secondary: the decode's own, fit on every non-ripple bin, so it "
    "includes non-ripple windows' own spikes",
}
"""What each encoding holds out."""

PRIMARY_METRICS: Mapping[str, tuple[str, ...]] = {
    "Local": ("runcv_loc_mark",),
    "No-Spike": ("runcv_ns_logratio",),
    "Non-Local Continuous": ("runcv_nl_null_mark", "runcv_nl_anim_mark"),
    "Non-Local Fragmented": ("runcv_nl_null_mark", "runcv_nl_anim_mark"),
}
"""The evidence each state's claim is judged by, under the primary encoding."""

SECONDARY_METRICS = (
    "runcv_loc_vs_Sremote_mark",
    "runcv_nlr_anim_mark",
    "runcv_loc_full",
    "runcv_nl_anim_full",
    "runcv_ns_count",
    "runcv_ns_silent_frac",
    "runcv_ns_expected",
    "runcv_mapS_dist_cm",
    "runcv_mapSr_dist_cm",
    "run_loc_mark",
    "run_nl_null_mark",
    "run_nl_anim_mark",
    "run_ns_logratio",
    "run_ns_expected",
    "can_loc_mark",
    "can_nl_null_mark",
    "can_nl_anim_mark",
    "can_ns_logratio",
    "can_ns_expected",
)
"""Reported for the bouts of every state: more evidence under the primary
encoding, and the primary evidence under the two secondary encodings."""

CALIBRATION_METRICS: Mapping[str, tuple[str, ...]] = {
    "Local": (
        "runcv_loc_mark",
        "runcv_loc_vs_Sremote_mark",
        "run_loc_mark",
        "can_loc_mark",
    ),
    "No-Spike": (
        "runcv_ns_logratio",
        "runcv_ns_count",
        "runcv_ns_silent_frac",
        "run_ns_logratio",
        "can_ns_logratio",
    ),
    "Non-Local Continuous": (
        "runcv_nl_null_mark",
        "runcv_nl_anim_mark",
        "run_nl_null_mark",
        "run_nl_anim_mark",
        "can_nl_null_mark",
        "can_nl_anim_mark",
    ),
    "Non-Local Fragmented": (
        "runcv_nl_null_mark",
        "runcv_nl_anim_mark",
        "run_nl_null_mark",
        "run_nl_anim_mark",
        "can_nl_null_mark",
        "can_nl_anim_mark",
    ),
}
"""The calibration curves: each metric against its state's probability."""

PROBABILITY_BINS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0000001)
PROBABILITY_BIN_LABELS = ("0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0")
MIN_BIN_WINDOWS = 10
"""Windows a calibration bin needs for a verdict (pooled over epochs) or to
count toward an epoch's slope (within the epoch)."""

TOO_FEW_WINDOWS = "too few windows"
TOO_FEW_ANIMALS = "too few animals"
"""A calibration verdict that is neither "yes" nor "no", and why."""

EPOCH_SUMMARY: Mapping[str, str] = {
    **dict.fromkeys(
        (
            "loc_mark",
            "loc_full",
            "loc_vs_Sremote_mark",
            "nl_anim_mark",
            "nl_anim_full",
            "nl_null_mark",
            "nl_null_full",
            "nlr_anim_mark",
            "nlr_anim_full",
            "ns_logratio",
        ),
        "median",
    ),
    **dict.fromkeys(
        (
            "ns_expected",
            "ns_count",
            "sel_count",
            "ns_silent_frac",
            "ns_pred_silent",
            "cond_S_silent",
            "cond_both_silent",
            "cond_pred_C_silent",
            "mapS_dist_cm",
            "mapSr_dist_cm",
        ),
        "mean",
    ),
}
"""Each evidence stem's epoch summary over windows: the median for the
log-likelihood ratios, which are heavy-tailed, and the mean for the expected
counts, counts, shares and distances, which are bounded (see the module notes).
MAP_S's distances from the animal (``mapS_dist_cm``, ``mapSr_dist_cm``) are
bounded by the track's length, so they take the mean.
A stem is a column name without its encoding prefix (``runcv_``, ``run_``,
``can_``) or
per-spike suffix (``_ps``); a paired difference follows its metric."""

SUMMARY_SEED = 20261004

SPLIT_COLUMN = "split"
"""Names the half decode a window's labels come from, when an epoch's table
holds several (Task 13's half decodes): a bout is paired with its matched window
within its split, and its epoch's value pools the splits' windows. The counts
then count window x split: ``n`` sums each split's bouts, so labels shared by
the splits (the canonical decode's, scored by each split's own half) count each
bout once per split, and the calibration verdicts' window gate
(:data:`MIN_BIN_WINDOWS`) counts calibration windows once per split. The
inference does not change: each epoch still enters as one value."""


def _split_keys(windows: pd.DataFrame) -> list[str]:
    """``[SPLIT_COLUMN]`` when the table has one, else nothing."""
    return [SPLIT_COLUMN] if SPLIT_COLUMN in windows else []


_RESAMPLE_CELLS = 1 << 24
"""Random draws per chunk of a resampling test, so memory stays bounded."""


def _chunk(n_values: int, n_draws: int) -> int:
    return int(max(1, min(n_draws, _RESAMPLE_CELLS // max(n_values, 1))))


def sign_flip_p(x: NDArray, *, n_flips: int = 10_000, seed: Seed) -> float:
    """Two-sided sign-flip p of mean(x) = 0.

    Exact over all 2**n sign patterns when there are no more of them than
    ``n_flips`` (n <= 13 at the default); otherwise ``n_flips`` seeded random
    flips, ``(hits + 1) / (n_flips + 1)``. Five epochs allow p = 2 / 32 at
    best. Non-finite values are dropped; NaN when none is left.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return float("nan")
    observed = abs(x.mean())
    if 2**n <= n_flips:
        signs = 1 - 2 * ((np.arange(2**n)[:, None] >> np.arange(n)) & 1)
        return float(np.mean(np.abs(signs @ x) / n >= observed * (1 - 1e-12)))
    stream = keyed_rng(seed, "flip")
    hits = 0
    chunk = _chunk(n, n_flips)
    for k in range(0, n_flips, chunk):
        b = min(chunk, n_flips - k)
        signs = stream.integers(0, 2, size=(b, n), dtype=np.int8) * 2 - 1
        hits += int((np.abs(signs @ x) / n >= observed * (1 - 1e-12)).sum())
    return (hits + 1) / (n_flips + 1)


def stratified_bootstrap_ci(
    x: NDArray, strata: NDArray, *, n_boot: int = 10_000, seed: Seed
) -> tuple[float, float]:
    """95 % percentile bootstrap CI of the pooled mean, resampling within strata.

    Window level: the strata keep their sizes but windows are treated as
    independent, so with nested data it ignores the nesting and is not for
    inference (:func:`nested_mean_intervals` is). Non-finite values are
    dropped; NaNs when fewer than two are left.
    """
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    x, strata = x[finite], np.asarray(strata)[finite]
    if x.size < 2:
        return (float("nan"), float("nan"))
    stream = keyed_rng(seed, "boot")
    parts = [x[strata == g] for g in pd.unique(strata)]
    means = np.zeros(n_boot)
    chunk = _chunk(x.size, n_boot)
    for k in range(0, n_boot, chunk):
        b = min(chunk, n_boot - k)
        total = np.zeros(b)
        for part in parts:
            total += part[stream.integers(0, part.size, size=(b, part.size))].sum(1)
        means[k : k + b] = total / x.size
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _slope(p: NDArray, x: NDArray) -> NDArray:
    centered = p - p.mean(-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        result: NDArray = (centered * (x - x.mean(-1, keepdims=True))).sum(-1) / (
            centered**2
        ).sum(-1)
    return result


def ols_slope_ci(
    p: NDArray, x: NDArray, strata: NDArray, *, n_boot: int = 2000, seed: Seed
) -> tuple[float, float, float]:
    """Window-level OLS slope of ``x`` on ``p``, with a 95 % bootstrap CI.

    Windows are resampled within strata, as if independent: it ignores the
    nesting and is not for inference (:func:`epoch_slopes` is). Pairs with a
    non-finite value are dropped.

    Returns
    -------
    slope, ci_lo, ci_hi : float
    """
    p, x = np.asarray(p, dtype=float), np.asarray(x, dtype=float)
    finite = np.isfinite(p) & np.isfinite(x)
    p, x, strata = p[finite], x[finite], np.asarray(strata)[finite]
    if p.size < 3:
        return (float("nan"),) * 3
    estimate = float(_slope(p, x))
    stream = keyed_rng(seed, "slope")
    by_stratum = [np.flatnonzero(strata == g) for g in pd.unique(strata)]
    slopes = np.zeros(n_boot)
    chunk = _chunk(p.size, n_boot)
    for k in range(0, n_boot, chunk):
        b = min(chunk, n_boot - k)
        idx = np.concatenate(
            [g[stream.integers(0, g.size, size=(b, g.size))] for g in by_stratum],
            axis=1,
        )
        slopes[k : k + b] = _slope(p[idx], x[idx])
    lo, hi = np.nanpercentile(slopes, [2.5, 97.5])
    return estimate, float(lo), float(hi)


def _ratio_ci(
    num: NDArray, den: NDArray, strata: NDArray, *, n_boot: int = 2000, seed: Seed
) -> tuple[float, float]:
    """95 % bootstrap CI of sum(num) / sum(den), resampling windows within strata.

    Window level: ignores the nesting, not for inference.
    """
    stream = keyed_rng(seed, "ratio")
    parts = [(num[strata == g], den[strata == g]) for g in pd.unique(strata)]
    out = np.zeros(n_boot)
    chunk = _chunk(num.size, n_boot)
    for k in range(0, n_boot, chunk):
        b = min(chunk, n_boot - k)
        top, bottom = np.zeros(b), np.zeros(b)
        for nu, de in parts:
            idx = stream.integers(0, nu.size, size=(b, nu.size))
            top += nu[idx].sum(1)
            bottom += de[idx].sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[k : k + b] = np.where(bottom > 0, top / bottom, np.nan)
    lo, hi = np.nanpercentile(out, [2.5, 97.5])
    return float(lo), float(hi)


def _describe(values: NDArray, prefix: str = "") -> dict[str, float]:
    """Median, quartiles, share > 0 and mean of the finite values (NaN if none)."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        names = ("median", "q25", "q75", "frac_pos", "mean")
        return {f"{prefix}{name}": float("nan") for name in names}
    q25, median, q75 = np.percentile(finite, [25, 50, 75])
    return {
        f"{prefix}median": float(median),
        f"{prefix}q25": float(q25),
        f"{prefix}q75": float(q75),
        f"{prefix}frac_pos": float(np.mean(finite > 0)),
        f"{prefix}mean": float(finite.mean()),
    }


def epoch_summaries(values: NDArray, epochs: NDArray, animals: NDArray) -> pd.DataFrame:
    """Each epoch's median and mean over its finite values: one row per epoch.

    The median is the primary summary: a few artifact windows can drag the
    mean (see the module notes).

    Parameters
    ----------
    values : ndarray, shape (n_windows,)
    epochs, animals : ndarray, shape (n_windows,)
        Each window's epoch, and that epoch's animal.

    Returns
    -------
    pd.DataFrame
        Indexed by epoch, sorted: ``animal``, ``median``, ``mean`` and ``n``
        (its finite values). An epoch with no finite value is left out.
    """
    frame = pd.DataFrame(
        {
            "epoch": np.asarray(epochs),
            "animal": np.asarray(animals),
            "value": np.asarray(values, dtype=float),
        }
    )
    frame = frame[np.isfinite(frame["value"].to_numpy())]
    grouped = frame.groupby("epoch", sort=True)
    return pd.DataFrame(
        {
            "animal": grouped["animal"].first(),
            "median": grouped["value"].median(),
            "mean": grouped["value"].mean(),
            "n": grouped["value"].size(),
        }
    )


def epoch_summary_kind(metric: str) -> str:
    """The epoch summary a metric is inferred from: "median" or "mean".

    Read from :data:`EPOCH_SUMMARY` by the metric's stem: its name without the
    encoding prefix (``runcv_``, ``run_``, ``can_``) and the per-spike suffix
    (``_ps``).

    Raises
    ------
    ValueError
        No summary is set for the metric, so it is not to be summarized.
    """
    stem = metric.removesuffix("_ps")
    for prefix in ENCODING_LABELS:
        stem = stem.removeprefix(f"{prefix}_")
    if stem not in EPOCH_SUMMARY:
        raise ValueError(f"no epoch summary is set for {metric!r}")
    return EPOCH_SUMMARY[stem]


def _roles(metric: str) -> tuple[tuple[str, str], tuple[str, str]]:
    """The metric's (column prefix, epoch summary) pairs: primary, then other."""
    kind = epoch_summary_kind(metric)
    return ("", kind), ("other_", "mean" if kind == "median" else "median")


def epoch_slopes(
    p: NDArray, x: NDArray, epochs: NDArray, animals: NDArray
) -> pd.DataFrame:
    """Each epoch's own OLS slope of ``x`` on ``p``: one row per epoch.

    A slope within each epoch cannot come from differences between epochs: a
    pooled slope can rise because the epochs with more windows at a high
    probability also carry more evidence everywhere.

    Parameters
    ----------
    p, x : ndarray, shape (n_windows,)
    epochs, animals : ndarray, shape (n_windows,)

    Returns
    -------
    pd.DataFrame
        Indexed by epoch, sorted: ``animal``, ``slope`` and ``n`` (finite
        pairs). An epoch enters only when at least two probability bins
        (:data:`PROBABILITY_BINS`) each hold :data:`MIN_BIN_WINDOWS` or more of
        its finite pairs, so no slope rests on a narrow range of ``p``.
    """
    p, x = np.asarray(p, dtype=float), np.asarray(x, dtype=float)
    finite = np.isfinite(p) & np.isfinite(x)
    frame = pd.DataFrame(
        {
            "epoch": np.asarray(epochs)[finite],
            "animal": np.asarray(animals)[finite],
            "p": p[finite],
            "x": x[finite],
        }
    )
    grouped = frame.groupby("epoch", sort=True)
    dp = frame["p"] - grouped["p"].transform("mean")
    dx = frame["x"] - grouped["x"].transform("mean")
    sums = pd.DataFrame({"sxy": dp * dx, "sxx": dp * dp}).groupby(frame["epoch"]).sum()
    out = pd.DataFrame({"animal": grouped["animal"].first(), "n": grouped["p"].size()})
    label = pd.cut(
        frame["p"], PROBABILITY_BINS, right=False, labels=PROBABILITY_BIN_LABELS
    )
    per_bin = frame.groupby(["epoch", label], observed=True).size()
    populated = (per_bin >= MIN_BIN_WINDOWS).groupby(level="epoch").sum()
    keep = populated.reindex(out.index, fill_value=0) >= 2
    out["slope"] = sums["sxy"] / sums["sxx"]
    return out[keep]


def nested_mean_intervals(
    summaries: NDArray, animals: NDArray, *, n_boot: int = 10_000, seed: Seed
) -> dict[str, float]:
    """95 % intervals of the animal-level and epoch-weighted means of per-epoch
    summaries, from one nested bootstrap.

    The replicates are
    :func:`~continuum_swr_replay.analysis.shared_statistics.nested_bootstrap_draws`
    with the epochs as the sessions (each its own row): the animals with
    replacement, then each drawn animal's epochs with replacement. On it, the
    animal-level mean is the mean over the drawn animals (each as often as it
    was drawn) of each one's mean over its drawn epochs, and the epoch-weighted
    mean the mean over every drawn epoch. Each interval is
    :func:`~continuum_swr_replay.analysis.shared_statistics.bootstrap_interval`
    of the statistic on the data and its replicates over the animals: the
    statistic plus or minus t(0.975, A - 1) sqrt(A / (A - 1)) times the
    replicates' SD.

    Windows are not resampled. Each epoch enters through its summary over its
    windows, whose spread across epochs already contains the windows' sampling
    noise; resampling them as well would add that noise a second time. The epoch
    is the unit within an animal.

    Parameters
    ----------
    summaries : ndarray, shape (n_epochs,)
        One value per epoch; non-finite ones are dropped.
    animals : ndarray, shape (n_epochs,)
        Each epoch's animal.
    n_boot : int, optional
    seed : int or sequence
        The replicates come from ``keyed_rng(seed, "hierarchical")``.

    Returns
    -------
    dict
        ``animal_estimate``, ``animal_ci_lo``, ``animal_ci_hi``,
        ``epoch_estimate``, ``epoch_ci_lo`` and ``epoch_ci_hi``. The estimates
        are NaN without any epoch, the intervals with fewer than two animals.
    """
    values = np.asarray(summaries, dtype=float)
    finite = np.isfinite(values)
    values, animals = values[finite], np.asarray(animals)[finite]
    nan = float("nan")
    if values.size == 0:
        return dict.fromkeys(
            [
                "animal_estimate",
                "animal_ci_lo",
                "animal_ci_hi",
                "epoch_estimate",
                "epoch_ci_lo",
                "epoch_ci_hi",
            ],
            nan,
        )
    animal_codes, epoch_codes = nested_codes(animals, np.arange(values.size))
    n_animals = int(animal_codes.max()) + 1
    animal_estimate = float(_animal_summaries(values, animals).mean())
    epoch_estimate = float(values.mean())
    stream = keyed_rng(seed, "hierarchical")
    draws = nested_bootstrap_draws(animal_codes, epoch_codes, stream, n_boot)
    epochs_drawn = draws.rows.astype(float)
    animals_drawn = draws.animals.astype(float)
    member = np.eye(n_animals)[animal_codes]  # (n_epochs, n_animals)
    sums = epochs_drawn @ (member * values[:, None])
    counts = epochs_drawn @ member
    own_means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    animal_draws = (animals_drawn * own_means).sum(1) / animals_drawn.sum(1)
    epoch_draws = (epochs_drawn @ values) / epochs_drawn.sum(1)
    animal = bootstrap_interval(animal_estimate, animal_draws, n_animals)
    epoch = bootstrap_interval(epoch_estimate, epoch_draws, n_animals)
    return {
        "animal_estimate": animal_estimate,
        "animal_ci_lo": animal.low,
        "animal_ci_hi": animal.high,
        "epoch_estimate": epoch_estimate,
        "epoch_ci_lo": epoch.low,
        "epoch_ci_hi": epoch.high,
    }


def _animal_summaries(values: NDArray, animals: NDArray) -> NDArray:
    """Each animal's mean epoch value, by animal: shape (n_animals,)."""
    by_animal: NDArray = (
        pd.Series(values, dtype=float).groupby(animals, sort=True).mean().to_numpy()
    )
    return by_animal


class AnimalTTest(NamedTuple):
    """The animal-level test of :func:`animal_t_test`.

    ``estimate`` is the mean of the A animal summaries; ``p`` the two-sided
    one-sample t-test p against 0, with A - 1 degrees of freedom.
    ``zero_variance``: every animal summary is the same, so there is no test
    (``p`` is NaN). Its interval is :func:`nested_mean_intervals`'.
    """

    estimate: float
    p: float
    zero_variance: bool


def _t_inference(by_animal: NDArray) -> AnimalTTest:
    """:class:`AnimalTTest` of the animal summaries, shape (n_animals,)."""
    nan = float("nan")
    n = by_animal.size
    if n == 0:
        return AnimalTTest(nan, nan, False)
    estimate = float(by_animal.mean())
    if n < 2:
        return AnimalTTest(estimate, nan, False)
    if np.ptp(by_animal) == 0:  # identical: their float SD can be ~1e-17, not 0
        return AnimalTTest(estimate, nan, True)
    sem = float(by_animal.std(ddof=1) / np.sqrt(n))
    p = float(2 * student_t.sf(abs(estimate / sem), df=n - 1))
    return AnimalTTest(estimate, p, False)


def animal_t_test(summaries: NDArray, animals: NDArray) -> AnimalTTest:
    """The mean of the animal summaries and its t-test against 0.

    Each animal enters once, as the mean of its epochs' summaries, however
    many epochs it has (one is enough). With A animals the two-sided test has
    A - 1 degrees of freedom; with balanced data it is the intercept test of a
    random-intercept model of epochs within animals.

    Parameters
    ----------
    summaries : ndarray, shape (n_epochs,)
        One value per epoch; non-finite ones are dropped.
    animals : ndarray, shape (n_epochs,)
        Each epoch's animal.

    Returns
    -------
    AnimalTTest
        ``estimate`` (NaN without any animal), ``p`` (NaN with fewer than two
        animals), and ``zero_variance``.
    """
    values = np.asarray(summaries, dtype=float)
    finite = np.isfinite(values)
    return _t_inference(_animal_summaries(values[finite], np.asarray(animals)[finite]))


def _epoch_statistics(
    summaries: pd.Series | NDArray,
    animals: pd.Series | NDArray,
    *,
    n_boot: int,
    n_flips: int,
    seed: Seed,
    signflip: bool = True,
) -> dict[str, Any]:
    """The inference on per-epoch values, keyed by column name (module notes).

    Primary: ``animal_estimate`` with its interval ``animal_ci_lo`` /
    ``animal_ci_hi`` (:func:`nested_mean_intervals`), ``p_animal_t`` and
    ``zero_variance`` (:func:`animal_t_test`), and ``animals_pos`` /
    ``epochs_pos`` ("k/A" and "k/N" > 0). Secondary: ``epoch_estimate`` with
    its interval ``epoch_ci_lo`` / ``epoch_ci_hi`` from the same replicates, and
    ``p_epoch_signflip`` unless ``signflip`` is False. Non-finite values are
    dropped.
    """
    values = np.asarray(summaries, dtype=float)
    animal = np.asarray(animals)
    finite = np.isfinite(values)
    values, animal = values[finite], animal[finite]
    by_animal = _animal_summaries(values, animal)
    test = _t_inference(by_animal)
    intervals = nested_mean_intervals(values, animal, n_boot=n_boot, seed=seed)
    level: dict[str, Any] = {
        "animal_estimate": test.estimate,
        "animal_ci_lo": intervals["animal_ci_lo"],
        "animal_ci_hi": intervals["animal_ci_hi"],
        "p_animal_t": test.p,
        "zero_variance": test.zero_variance,
        "animals_pos": f"{int((by_animal > 0).sum())}/{by_animal.size}",
        "epochs_pos": f"{int((values > 0).sum())}/{values.size}",
        "epoch_estimate": intervals["epoch_estimate"],
        "epoch_ci_lo": intervals["epoch_ci_lo"],
        "epoch_ci_hi": intervals["epoch_ci_hi"],
    }
    if signflip:
        level["p_epoch_signflip"] = sign_flip_p(values, n_flips=n_flips, seed=seed)
    return level


_INTERVALS = (
    "animal_estimate",
    "animal_ci_lo",
    "animal_ci_hi",
    "zero_variance",
    "epoch_estimate",
    "epoch_ci_lo",
    "epoch_ci_hi",
)
"""The estimates and intervals that a calibration bin and coordinated
silence's observed ratio carry, without the tests."""


def _prefixed(level: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    """:func:`_epoch_statistics` columns under ``prefix`` (the module notes)."""
    return {f"{prefix}{name}": value for name, value in level.items()}


def _summary_seed(seed: Seed, kind: str) -> Seed:
    """The seed of one epoch summary's statistics.

    The mean's keep ``seed``, the one they had while the mean was primary, so
    they can be checked against those earlier outputs; the median's add
    ``kind``.
    """
    return seed if kind == "mean" else _with_keys(seed, kind)


def _verdict(enough: bool, holds: bool) -> str:
    """The verdict on ``holds``: "yes" or "no", or :data:`TOO_FEW_WINDOWS`."""
    if not enough:
        return TOO_FEW_WINDOWS
    return "yes" if holds else "no"


def _ci_verdict(enough: bool, ci_lo: float) -> str:
    """The verdict that the animal-level interval lies above 0, or why none.

    :data:`TOO_FEW_WINDOWS` without ``enough`` windows; :data:`TOO_FEW_ANIMALS`
    when fewer than two animals give the interval (it is NaN).
    """
    if enough and not np.isfinite(ci_lo):
        return TOO_FEW_ANIMALS
    return _verdict(enough, ci_lo > 0)


def paired_bouts(windows: pd.DataFrame) -> pd.DataFrame:
    """Every bout with its matched window's columns, suffixed ``_base``.

    Within its epoch and, when the table has one, its :data:`SPLIT_COLUMN`.

    Raises
    ------
    ValueError
        A bout has no matched window.
    """
    keys = ["epoch_id", *_split_keys(windows), "state", "regime"]
    paired = windows[windows.kind == "bout"].merge(
        windows[windows.kind == "baseline"],
        left_on=[*keys, "bout_s"],
        right_on=[*keys, "matched_bout_s"],
        suffixes=("", "_base"),
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if (paired["_merge"] != "both").any():
        raise ValueError("a bout lacks its matched window")
    return paired


def bout_animal_summaries(
    windows: pd.DataFrame, *, metrics: Mapping[str, Sequence[str]]
) -> pd.DataFrame:
    """Each animal's summary of bout metrics and of their paired differences.

    An animal's summary is the mean of its epochs' summaries over windows, the
    median or the mean by metric (:func:`epoch_summary_kind`): the values whose
    mean and interval :func:`bout_statistics` reports as ``animal_*`` and
    ``diff_animal_*``.

    Parameters
    ----------
    windows : pd.DataFrame
        The per-window table, as for :func:`bout_statistics`.
    metrics : mapping
        State -> metric columns.

    Returns
    -------
    pd.DataFrame
        ``state``, ``regime``, ``metric``, ``animal``, ``value`` (the bouts'
        summary) and ``diff`` (the paired bout - matched summary).
    """
    paired = paired_bouts(windows)
    rows = []
    for state, state_metrics in metrics.items():
        for regime in REGIMES:
            cell = paired[(paired.state == state) & (paired.regime == regime)]
            if cell.empty:
                continue
            epochs, animals = cell.epoch_id.to_numpy(), cell.animal.to_numpy()
            for metric in state_metrics:
                kind = epoch_summary_kind(metric)
                x = cell[metric].to_numpy(dtype=float)
                diff = x - cell[f"{metric}_base"].to_numpy(dtype=float)
                by_animal = {
                    name: epoch_summaries(values, epochs, animals)
                    .groupby("animal")[kind]
                    .mean()
                    for name, values in (("value", x), ("diff", diff))
                }
                frame = pd.DataFrame(by_animal).rename_axis("animal").reset_index()
                rows.append(frame.assign(state=state, regime=regime, metric=metric))
    columns = ["state", "regime", "metric", "animal", "value", "diff"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.concat(rows, ignore_index=True)[columns]


def bout_statistics(
    windows: pd.DataFrame,
    *,
    metrics: Mapping[str, Sequence[str]] | None = None,
    n_flips: int = 10_000,
    n_boot: int = 10_000,
    seed: Seed = SUMMARY_SEED,
    window_level: bool = False,
) -> pd.DataFrame:
    """Bouts against their matched windows, per state x regime x metric.

    - **Epoch level** (the module's column scheme): ``epoch_summary`` is the
      metric's epoch summary (:func:`epoch_summary_kind`), and each epoch
      enters as that summary of its bouts' values (no prefix) and of its
      bouts' paired differences (``diff_``). Primary: the animal-level
      ``animal_*`` columns with ``animals_pos`` and ``epochs_pos``. Secondary:
      the epoch-weighted ``epoch_*`` columns and ``p_epoch_signflip``, and all
      of it again over the other epoch summary (``other_``, ``other_diff_``).
    - **Descriptive**, windows pooled over epochs:
      - ``n``, ``n_found``, ``n_epochs`` and ``n_animals``;
      - ``median``, ``q25``, ``q75``, ``frac_pos`` and ``mean``, and the same
        for the matched windows (``base_*``);
      - ``diff_mean``;
      - per held-out spike: ``ps_median``, ``ps_mean`` and ``base_ps_median``;
      - the median spikes, bins and bout bins.
    - **Window level, only with** ``window_level`` (windows as if
      independent, not for inference): ``p_signflip_window_level``,
      ``ci_lo_window_level`` / ``ci_hi_window_level``,
      ``p_diff_window_level``, ``diff_ci_lo_window_level`` /
      ``diff_ci_hi_window_level``.

    Parameters
    ----------
    windows : pd.DataFrame
        The per-window table: ``epoch_id``, ``animal``, ``kind``, ``state``,
        ``regime``, ``bout_s``, ``matched_bout_s``, ``n_found`` and the metrics.
    metrics : mapping, optional
        State -> metric columns; default each state's :data:`PRIMARY_METRICS`
        and every :data:`SECONDARY_METRICS`.
    window_level : bool, optional
        Add the window-level columns (default False).

    Raises
    ------
    ValueError
        A bout has no matched window.
    """
    if metrics is None:
        metrics = {
            state: (*PRIMARY_METRICS[state], *SECONDARY_METRICS) for state in STATES
        }
    bouts = windows[windows.kind == "bout"]
    paired = paired_bouts(windows)
    rows = []
    for state, state_metrics in metrics.items():
        for regime in REGIMES:
            cell = paired[(paired.state == state) & (paired.regime == regime)]
            found = bouts[(bouts.state == state) & (bouts.regime == regime)]
            n_found = int(
                found.groupby(["epoch_id", *_split_keys(found)]).n_found.first().sum()
            )
            epochs, animals = cell.epoch_id.to_numpy(), cell.animal.to_numpy()
            for metric in state_metrics:
                key = (state, regime, metric)
                row: dict[str, Any] = {
                    "state": state,
                    "regime": regime,
                    "metric": metric,
                    "primary": metric in PRIMARY_METRICS.get(state, ()),
                    "epoch_summary": epoch_summary_kind(metric),
                    "n": len(cell),
                    "n_found": n_found,
                    "n_epochs": int(cell.epoch_id.nunique()),
                    "n_animals": int(cell.animal.nunique()),
                }
                if len(cell) == 0:
                    rows.append(row)
                    continue
                x = cell[metric].to_numpy(dtype=float)
                xb = cell[f"{metric}_base"].to_numpy(dtype=float)
                diff = x - xb
                row.update(_describe(x))
                row.update(_describe(xb, prefix="base_"))
                row["diff_mean"] = _describe(diff)["mean"]
                quantities = {
                    "": (epoch_summaries(x, epochs, animals), ("epoch",)),
                    "diff_": (
                        epoch_summaries(diff, epochs, animals),
                        ("diff", "epoch"),
                    ),
                }
                for role, which in _roles(metric):
                    for quantity, (summaries, seed_keys) in quantities.items():
                        level = _epoch_statistics(
                            summaries[which],
                            summaries["animal"],
                            n_boot=n_boot,
                            n_flips=n_flips,
                            seed=_summary_seed(
                                _with_keys(seed, *key, *seed_keys), which
                            ),
                        )
                        row.update(_prefixed(level, f"{role}{quantity}"))
                if window_level:
                    row["p_signflip_window_level"] = sign_flip_p(
                        x, n_flips=n_flips, seed=_with_keys(seed, *key)
                    )
                    (
                        row["ci_lo_window_level"],
                        row["ci_hi_window_level"],
                    ) = stratified_bootstrap_ci(
                        x, epochs, n_boot=n_boot, seed=_with_keys(seed, *key)
                    )
                    row["p_diff_window_level"] = sign_flip_p(
                        diff, n_flips=n_flips, seed=_with_keys(seed, *key, "diff")
                    )
                    (
                        row["diff_ci_lo_window_level"],
                        row["diff_ci_hi_window_level"],
                    ) = stratified_bootstrap_ci(
                        diff, epochs, n_boot=n_boot, seed=_with_keys(seed, *key, "diff")
                    )
                if f"{metric}_ps" in cell:
                    ps = _describe(cell[f"{metric}_ps"].to_numpy(dtype=float))
                    base_ps = _describe(cell[f"{metric}_ps_base"].to_numpy(dtype=float))
                    row.update(
                        ps_median=ps["median"],
                        ps_mean=ps["mean"],
                        base_ps_median=base_ps["median"],
                    )
                for column, name in [
                    ("n_spikes", "median_n_spikes"),
                    ("n_bins", "median_bins"),
                    ("bout_len", "median_bout_bins"),
                ]:
                    if column in cell:
                        row[name] = float(cell[column].median())
                rows.append(row)
    return pd.DataFrame(rows)


def calibration_statistics(
    windows: pd.DataFrame,
    *,
    metrics: Mapping[str, Sequence[str]] | None = None,
    n_boot: int = 10_000,
    n_boot_window_level: int = 2000,
    n_flips: int = 10_000,
    seed: Seed = SUMMARY_SEED,
    window_level: bool = False,
    regimes: Sequence[str] = REGIMES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibration curves: evidence against each state's probability.

    Calibration windows are binned by their mean posterior of the state (0-0.2,
    ..., 0.8-1.0).

    - **Primary, animal level:**
      - A bin's ``animal_estimate`` is the mean over animals of each animal's
        mean epoch summary in the bin (``epoch_summary``,
        :func:`epoch_summary_kind`), with its interval ``animal_ci_lo`` /
        ``animal_ci_hi`` (:func:`nested_mean_intervals`).
      - A curve's slope is each epoch's own OLS slope (:func:`epoch_slopes`,
        which takes only epochs with two bins of :data:`MIN_BIN_WINDOWS`
        windows), inferred under the prefix ``slope_`` (the module's column
        scheme).
      - The verdicts read the animal-level interval. They need the top and the bottom
        bins to hold at least :data:`MIN_BIN_WINDOWS` windows each, pooled
        (``n_top``, ``n_bottom``); otherwise they read
        :data:`TOO_FEW_WINDOWS`, and :data:`TOO_FEW_ANIMALS` when fewer than
        two animals give the interval.
      - ``rises`` is "yes" when the slope's interval lies above 0, else "no".
      - ``top_positive`` is "yes" when the top bin's interval
        (``top_animal_ci_lo``) lies above 0, else "no".
    - **Secondary:** a bin's epoch-weighted ``epoch_estimate`` with its
      interval ``epoch_ci_lo`` / ``epoch_ci_hi``; all of a
      bin's estimates again over the other epoch summary (``other_``); a
      curve's ``top_other_animal_estimate``, ``slope_epoch_*`` and
      ``slope_p_epoch_signflip``.
    - **Descriptive:** a bin's ``n``, ``n_epochs``, pooled ``mean``,
      ``median`` and ``frac_pos``; a curve's ``n``, ``top_mean`` and
      ``top_minus_bottom``.
    - **Window level, only with** ``window_level`` (not for inference):
      - per bin, ``ci_lo_window_level`` / ``ci_hi_window_level``;
      - per curve, the pooled ``slope_window_level`` with its window-resampling
        CI, and the prototype's calls ``ci_separated_window_level``,
        ``rises_window_level`` and ``top_positive_window_level``, as "yes",
        "no" or :data:`TOO_FEW_WINDOWS` under the same pooled counts.

    Parameters
    ----------
    windows : pd.DataFrame
        Needs ``kind``, ``regime``, ``epoch_id``, ``animal``, ``p_{state}``
        and the metrics.
    metrics : mapping, optional
        State -> metric columns; default :data:`CALIBRATION_METRICS`.
    n_boot, n_boot_window_level : int, optional
        Resamples of the nested and of the window-level bootstraps.
    window_level : bool, optional
        Add the window-level columns (default False).
    regimes : sequence of str, optional
        The regimes summarized (default :data:`REGIMES`).

    Returns
    -------
    bins, trends : pd.DataFrame
    """
    if metrics is None:
        metrics = CALIBRATION_METRICS
    calibration = windows[windows.kind == "calib"]
    bins, trends = [], []
    for regime in regimes:
        cell = calibration[calibration.regime == regime]
        epochs, animals = cell.epoch_id.to_numpy(), cell.animal.to_numpy()
        for state, state_metrics in metrics.items():
            p = cell[f"p_{state}"].to_numpy(dtype=float)
            label = pd.cut(
                cell[f"p_{state}"],
                PROBABILITY_BINS,
                right=False,
                labels=PROBABILITY_BIN_LABELS,
            ).to_numpy()
            for metric in state_metrics:
                roles = _roles(metric)
                x = cell[metric].to_numpy(dtype=float)
                curve = []
                for name in PROBABILITY_BIN_LABELS:
                    m = (label == name) & np.isfinite(x)
                    row: dict[str, Any] = {
                        "regime": regime,
                        "state": state,
                        "metric": metric,
                        "pbin": name,
                        "epoch_summary": roles[0][1],
                        "n": int(m.sum()),
                        "n_epochs": int(np.unique(epochs[m]).size),
                    }
                    if m.sum() >= 2:
                        row.update(
                            mean=float(x[m].mean()),
                            median=float(np.median(x[m])),
                            frac_pos=float((x[m] > 0).mean()),
                        )
                        if window_level:
                            (
                                row["ci_lo_window_level"],
                                row["ci_hi_window_level"],
                            ) = stratified_bootstrap_ci(
                                x[m],
                                epochs[m],
                                n_boot=n_boot_window_level,
                                seed=_with_keys(seed, regime, state, metric, name),
                            )
                    summaries = epoch_summaries(x[m], epochs[m], animals[m])
                    bin_seed = _with_keys(seed, regime, state, metric, name, "epoch")
                    for role, which in roles:
                        level = _epoch_statistics(
                            summaries[which],
                            summaries["animal"],
                            n_boot=n_boot,
                            n_flips=n_flips,
                            seed=_summary_seed(bin_seed, which),
                            signflip=False,
                        )
                        row.update({f"{role}{c}": level[c] for c in _INTERVALS})
                    curve.append(row)
                bins += curve
                slopes = epoch_slopes(p, x, epochs, animals)
                level = _epoch_statistics(
                    slopes["slope"],
                    slopes["animal"],
                    n_boot=n_boot,
                    n_flips=n_flips,
                    seed=_with_keys(seed, regime, state, metric, "epoch"),
                )
                bottom, top = curve[0], curve[-1]
                enough = bottom["n"] >= MIN_BIN_WINDOWS and top["n"] >= MIN_BIN_WINDOWS
                trend: dict[str, Any] = {
                    "regime": regime,
                    "state": state,
                    "metric": metric,
                    "epoch_summary": roles[0][1],
                    "n": int(np.isfinite(x).sum()),
                    "n_bottom": bottom["n"],
                    "n_top": top["n"],
                    "n_epochs": len(slopes),
                    **_prefixed(level, "slope_"),
                    "rises": _ci_verdict(enough, level["animal_ci_lo"]),
                    "top_animal_estimate": top["animal_estimate"],
                    "top_animal_ci_lo": top["animal_ci_lo"],
                    "top_positive": _ci_verdict(enough, top["animal_ci_lo"]),
                    "top_other_animal_estimate": top["other_animal_estimate"],
                    "top_mean": top.get("mean", np.nan),
                    "top_minus_bottom": (
                        top["mean"] - bottom["mean"] if enough else np.nan
                    ),
                }
                if window_level:
                    slope, lo, hi = ols_slope_ci(
                        p,
                        x,
                        epochs,
                        n_boot=n_boot_window_level,
                        seed=_with_keys(seed, regime, state, metric),
                    )
                    separated = enough and (
                        top["ci_lo_window_level"] > bottom["ci_hi_window_level"]
                    )
                    trend.update(
                        slope_window_level=slope,
                        slope_ci_lo_window_level=lo,
                        slope_ci_hi_window_level=hi,
                        ci_separated_window_level=_verdict(enough, separated),
                        rises_window_level=_verdict(enough, separated and lo > 0),
                        top_positive_window_level=_verdict(
                            enough, enough and top["ci_lo_window_level"] > 0
                        ),
                    )
                trends.append(trend)
    return pd.DataFrame(bins), pd.DataFrame(trends)


def rate_calibration(windows: pd.DataFrame) -> pd.DataFrame:
    """Observed / expected held-out spikes in the calibration windows.

    Per regime and encoding, pooled (``animal == "all"``) and per animal:
    C's spike count summed over windows against E_C summed over windows,
    with the observed and predicted shares of silent C. The observed counts
    are the same under every encoding; they are read from the primary one.
    Descriptive.
    """
    observed_count = f"{PRIMARY_ENCODING}_ns_count"
    observed_silent = f"{PRIMARY_ENCODING}_ns_silent_frac"
    calibration = windows[windows.kind == "calib"]
    rows = []
    for regime in REGIMES:
        cell = calibration[calibration.regime == regime]
        for prefix, label in ENCODING_LABELS.items():
            groups = [("all", cell)] + [
                (str(animal), g) for animal, g in cell.groupby("animal", sort=True)
            ]
            for animal, g in groups:
                observed = float(g[observed_count].sum())
                expected = float(g[f"{prefix}_ns_expected"].sum())
                rows.append(
                    {
                        "regime": regime,
                        "encoding": label,
                        "animal": animal,
                        "n_windows": len(g),
                        "observed_spikes": observed,
                        "expected_spikes": expected,
                        "observed_over_expected": (
                            observed / expected if expected > 0 else np.nan
                        ),
                        "obs_silent_share": float(g[observed_silent].mean()),
                        "pred_silent_share": float(
                            g[f"{prefix}_ns_pred_silent"].mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def coordinated_silence(
    windows: pd.DataFrame,
    *,
    n_boot: int = 10_000,
    n_boot_window_level: int = 2000,
    n_flips: int = 10_000,
    seed: Seed = SUMMARY_SEED,
    window_level: bool = False,
) -> pd.DataFrame:
    """Held-out coordinated silence: P(C silent | S silent) against local coding.

    Computed for each state's bouts and matched windows, and for the
    calibration windows, by P(No-Spike) bin and all together. Local coding makes
    the halves independent given position, so its prediction for
    P(C silent | S silent) is exp(-E_C) averaged over the S-silent runs, under
    either encoding.

    - **Primary, animal level.** Each epoch's observed ratio and local
      coding's predicted ratio are ratios of the epoch's own sums of
      silent-run counts: counts, so a mean-type summary (:data:`EPOCH_SUMMARY`).
      - ``obs_animal_estimate`` is the observed ratios' mean over animals, with
        its interval ``obs_animal_ci_lo`` / ``obs_animal_ci_hi``; the
        secondary ``obs_epoch_*`` are their epoch-weighted mean and its
        interval (:func:`nested_mean_intervals`).
      - ``pred_{e}_animal_estimate`` is encoding ``e``'s prediction, its mean
        over animals (``pred_{e}_epoch_estimate``, over epochs).
      - Each epoch's observed - predicted is inferred under the prefix
        ``excess_{e}_`` (the module's column scheme).
      - The observed silences are the same under every encoding; they are read
        from the primary one's columns.
    - **Descriptive:** pooled ratios of sums over windows and splits:
      ``obs_C_silent_given_S_silent``, ``pred_given_S_{e}``, ``obs_C_silent``
      and ``pred_C_silent_{e}``.
    - **Window level, only with** ``window_level`` (not for inference):
      ``obs_given_ci_lo_window_level`` / ``_hi`` (windows resampled within
      epochs).
    """
    primary = PRIMARY_ENCODING  # the observed silences: the same under every encoding
    groups: list[tuple[tuple[str, str, str, str], pd.DataFrame]] = []
    for regime in REGIMES:
        for kind in ("bout", "baseline"):
            for state in STATES:
                d = windows[
                    (windows.kind == kind)
                    & (windows.state == state)
                    & (windows.regime == regime)
                ]
                groups.append(((regime, kind, state, "all"), d))
        cell = windows[(windows.kind == "calib") & (windows.regime == regime)]
        label = pd.cut(
            cell["p_No-Spike"],
            PROBABILITY_BINS,
            right=False,
            labels=PROBABILITY_BIN_LABELS,
        ).to_numpy()
        for name in PROBABILITY_BIN_LABELS:
            groups.append(
                ((regime, "calib", "P(No-Spike) bin", name), cell[label == name])
            )
        groups.append(((regime, "calib", "all windows", "all"), cell))
    rows = []
    for (regime, kind, state, name), d in groups:
        if len(d) == 0:
            continue
        silent_s = d[f"{primary}_cond_S_silent"].to_numpy(dtype=float)
        both = d[f"{primary}_cond_both_silent"].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "regime": regime,
            "kind": kind,
            "state": state,
            "pbin": name,
            "n_windows": len(d),
            "runs_S_silent": int(silent_s.sum()),
            "obs_C_silent": float(d[f"{primary}_ns_silent_frac"].mean()),
            **{
                f"pred_C_silent_{prefix}": float(d[f"{prefix}_ns_pred_silent"].mean())
                for prefix in ENCODING_LABELS
            },
        }
        if silent_s.sum() > 0:
            row["obs_C_silent_given_S_silent"] = float(both.sum() / silent_s.sum())
            for prefix in ENCODING_LABELS:
                row[f"pred_given_S_{prefix}"] = float(
                    d[f"{prefix}_cond_pred_C_silent"].sum() / silent_s.sum()
                )
            if window_level and (silent_s > 0).sum() >= 5:
                (
                    row["obs_given_ci_lo_window_level"],
                    row["obs_given_ci_hi_window_level"],
                ) = _ratio_ci(
                    both,
                    silent_s,
                    d.epoch_id.to_numpy(),
                    n_boot=n_boot_window_level,
                    seed=_with_keys(seed, regime, kind, state, name),
                )
            sums = d.groupby("epoch_id", sort=True).agg(
                animal=("animal", "first"),
                silent=(f"{primary}_cond_S_silent", "sum"),
                both=(f"{primary}_cond_both_silent", "sum"),
                **{
                    prefix: (f"{prefix}_cond_pred_C_silent", "sum")
                    for prefix in ENCODING_LABELS
                },
            )
            sums = sums[sums["silent"] > 0]
            observed = sums["both"] / sums["silent"]
            row["n_epochs"] = len(sums)
            level = _epoch_statistics(
                observed,
                sums["animal"],
                n_boot=n_boot,
                n_flips=n_flips,
                seed=_with_keys(seed, regime, kind, state, name, "epoch"),
                signflip=False,
            )
            row.update({f"obs_{c}": level[c] for c in _INTERVALS})
            for prefix in ENCODING_LABELS:
                predicted = sums[prefix] / sums["silent"]
                row[f"pred_{prefix}_animal_estimate"] = float(
                    _animal_summaries(
                        predicted.to_numpy(), sums["animal"].to_numpy()
                    ).mean()
                )
                row[f"pred_{prefix}_epoch_estimate"] = float(predicted.mean())
                level = _epoch_statistics(
                    observed - predicted,
                    sums["animal"],
                    n_boot=n_boot,
                    n_flips=n_flips,
                    seed=_with_keys(seed, regime, kind, state, name, prefix, "excess"),
                )
                row.update(_prefixed(level, f"excess_{prefix}_"))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_state_evidence(
    windows: pd.DataFrame, *, seed: Seed = SUMMARY_SEED, window_level: bool = False
) -> dict[str, pd.DataFrame]:
    """Every summary of the per-window table, pooled over its epochs.

    ``{p}_loc_vs_Sremote_mark`` (and ``_ps``) are added as the negatives of
    ``{p}_nlr_anim_mark``, for each encoding prefix ``p`` the table has, so
    that positive supports the Local claim.
    ``window_level`` adds the window-level columns, which take most of the run
    time; the other columns do not depend on it.

    Returns
    -------
    dict
        ``bouts`` (:func:`bout_statistics`), ``calibration`` and ``trends``
        (:func:`calibration_statistics`), ``rate_calibration`` and
        ``silence`` (:func:`coordinated_silence`).
    """
    windows = windows.assign(
        **{
            f"{prefix}_loc_vs_Sremote_mark{tail}": -windows[
                f"{prefix}_nlr_anim_mark{tail}"
            ]
            for prefix in ENCODING_LABELS
            if f"{prefix}_nlr_anim_mark" in windows
            for tail in ("", "_ps")
        }
    )
    bins, trends = calibration_statistics(windows, seed=seed, window_level=window_level)
    return {
        "bouts": bout_statistics(windows, seed=seed, window_level=window_level),
        "calibration": bins,
        "trends": trends,
        "rate_calibration": rate_calibration(windows),
        "silence": coordinated_silence(windows, seed=seed, window_level=window_level),
    }


#: The decoder states split by SWR coupling (:func:`split_non_local_by_swr`).
NON_LOCAL_STATES = ("Non-Local Continuous", "Non-Local Fragmented")


def split_non_local_by_swr(windows: pd.DataFrame, swr: np.ndarray) -> pd.DataFrame:
    """One epoch's Non-Local bouts and their matched windows, labeled by SWR.

    A bout is "SWR" when some bin of its full span ``[bout_s, bout_e]`` is in
    an SWR, else "no SWR"; its matched window takes the bout's label. The
    label is appended to ``state`` (``"Non-Local Continuous | SWR"``), so
    :func:`bout_statistics` runs on each subset as on any state.

    Parameters
    ----------
    windows : pd.DataFrame
        One epoch's per-window table (``kind``, ``state``, ``regime``,
        ``bout_s``, ``bout_e``, ``matched_bout_s``).
    swr : np.ndarray, shape (n_time,)
        Whether each decode bin is in an SWR, on the decode's time bins.

    Returns
    -------
    pd.DataFrame
        The Non-Local bout and baseline rows with an ``swr`` column and the
        label appended to ``state``.

    Raises
    ------
    ValueError
        When two bouts share a (state, regime, start) key, or a matched window
        has no bout.
    """
    w = windows[
        windows["kind"].isin(["bout", "baseline"])
        & windows["state"].isin(NON_LOCAL_STATES)
    ].copy()
    in_swr = np.r_[0, np.cumsum(np.asarray(swr, dtype=bool))]
    bouts = w[w["kind"] == "bout"]
    start = bouts["bout_s"].to_numpy(dtype=int)
    end = bouts["bout_e"].to_numpy(dtype=int)
    label = pd.Series(
        np.where(
            in_swr[end + 1] - in_swr[start] > 0, SWR_LABELS[True], SWR_LABELS[False]
        ),
        index=pd.MultiIndex.from_arrays(
            [bouts["state"], bouts["regime"], bouts["bout_s"].astype(int)]
        ),
    )
    if label.index.duplicated().any():
        raise ValueError("duplicate bout keys")
    key = pd.MultiIndex.from_arrays(
        [
            w["state"],
            w["regime"],
            np.where(w["kind"] == "bout", w["bout_s"], w["matched_bout_s"]).astype(int),
        ]
    )
    w["swr"] = label.reindex(key).to_numpy()
    if w["swr"].isna().any():
        raise ValueError("a matched window has no bout")
    w["state"] = w["state"] + " | " + w["swr"]
    return w


IN_VISIT_CALIBRATION_METRICS: Mapping[str, tuple[str, ...]] = {
    "Local": ("runcv_loc_mark_ps",),
    "Non-Local Continuous": ("runcv_nl_null_mark_ps",),
    "Non-Local Fragmented": ("runcv_nl_null_mark_ps",),
}
"""The calibration Figure 2's supplement draws, over the immobile calibration
windows that start inside a well visit, under the primary encoding, per spike
of the scored tetrodes (``_ps``: each run's score over the scored half's
spikes, averaged over the runs in which that half spiked; a window in which it
never spiked has no value): for Local, the animal's place against a
mismatched place; for the non-local states, the place one half of the
tetrodes decodes against a mismatched place, scored on the other half.
No-Spike is not drawn: a confident No-Spike label forces both halves silent
(the module's caveats)."""


def calibration_windows_in_well_visits(
    windows: pd.DataFrame,
    arrival_times: np.ndarray,
    departure_times: np.ndarray,
) -> pd.DataFrame:
    """One epoch's immobile calibration windows that start inside a well visit.

    A window counts when its first sample (``start_time``) is inside a visit
    (``event_definition.well_visits.in_well_visit``), as an immobile bout does
    (:func:`mark_bouts_in_well_visits`).

    Parameters
    ----------
    windows : pd.DataFrame
        One epoch's windows: ``kind``, ``regime`` and ``start_time`` (s).
    arrival_times, departure_times : np.ndarray, shape (n_visits,)
        The epoch's well visits (``event_tables.prepare_data.well_visit_times``).

    Returns
    -------
    pd.DataFrame
        The rows of ``windows`` kept.
    """
    calibration = (windows["kind"] == "calib") & (windows["regime"] == "immobile")
    inside = in_well_visit(
        windows["start_time"].to_numpy(dtype=float), arrival_times, departure_times
    )
    return windows[calibration.to_numpy() & inside]


def mark_bouts_in_well_visits(
    windows: pd.DataFrame,
    time: np.ndarray,
    arrival_times: np.ndarray,
    departure_times: np.ndarray,
) -> pd.DataFrame:
    """One epoch's Non-Local bouts and matched windows, marked by the well visits.

    An immobile bout counts when its first sample (``time[bout_s]``) is inside
    a well visit (``event_definition.well_visits.in_well_visit``); a running
    bout counts wherever it happens, as running time does (the animal does not
    run inside a visit). The matched window goes with its bout: both rows of
    the pair carry the bout's ``starts_in_well_visit`` and
    ``in_analyzed_subset`` (whether the pair is read). The matched windows were
    drawn from all of the epoch's time in the bout's regime, so each pair also
    carries ``matched_starts_in_well_visit``, whether the matched window's
    first sample (``time[s]``) is inside a visit, to be counted
    (:func:`bout_visit_counts`), not to select by.

    Parameters
    ----------
    windows : pd.DataFrame
        From :func:`split_non_local_by_swr`: ``kind`` (``"bout"`` or
        ``"baseline"``), ``state``, ``regime``, ``s``, ``bout_s``,
        ``matched_bout_s`` and ``start_time`` (``time[s]``, the scored window's
        first sample).
    time : np.ndarray, shape (n_time,)
        The decode's time bins (s), which ``s`` and ``bout_s`` index.
    arrival_times, departure_times : np.ndarray, shape (n_visits,)
        The epoch's well visits (``event_tables.prepare_data.well_visit_times``).

    Returns
    -------
    pd.DataFrame
        A copy of ``windows`` with ``starts_in_well_visit``,
        ``matched_starts_in_well_visit`` and ``in_analyzed_subset``.

    Raises
    ------
    ValueError
        If a window's ``start_time`` is not ``time[s]`` (the windows were made
        on another time grid), or a matched window has no bout.
    """
    time = np.asarray(time, dtype=float)
    s = windows["s"].to_numpy(dtype=int)
    if not np.array_equal(time[s], windows["start_time"].to_numpy(dtype=float)):
        raise ValueError("the windows' start_time is not the decode time at s")
    is_bout = (windows["kind"] == "bout").to_numpy()
    first = np.where(is_bout, windows["bout_s"], windows["matched_bout_s"]).astype(int)
    key = pd.MultiIndex.from_arrays([windows["state"], windows["regime"], first])
    bout_in = pd.Series(
        in_well_visit(time[first[is_bout]], arrival_times, departure_times),
        index=key[is_bout],
    )
    matched_in = pd.Series(
        in_well_visit(time[s[~is_bout]], arrival_times, departure_times),
        index=key[~is_bout],
    )
    out = windows.copy()
    out["starts_in_well_visit"] = bout_in.reindex(key).to_numpy()
    out["matched_starts_in_well_visit"] = matched_in.reindex(key).to_numpy()
    if out["starts_in_well_visit"].isna().any():
        raise ValueError("a matched window has no bout")
    if out["matched_starts_in_well_visit"].isna().any():
        raise ValueError("a bout has no matched window")
    out = out.astype(
        {"starts_in_well_visit": bool, "matched_starts_in_well_visit": bool}
    )
    out["in_analyzed_subset"] = out["starts_in_well_visit"] | (
        out["regime"] == "running"
    )
    return out


def bout_visit_counts(windows: pd.DataFrame) -> pd.DataFrame:
    """The bouts kept and left out by the well visits, per state and regime.

    Parameters
    ----------
    windows : pd.DataFrame
        From :func:`mark_bouts_in_well_visits`, over any epochs.

    Returns
    -------
    pd.DataFrame
        One row per ``state`` (with its SWR label) and ``regime``:
        ``n_bouts``; ``n_kept``, the bouts read (``in_analyzed_subset``: an
        immobile bout whose first sample is inside a well visit, every running
        bout); ``n_left_out``, the others; and ``n_kept_matched_outside_visit``,
        the kept bouts whose matched window starts outside every visit.
    """
    bouts = windows[windows["kind"] == "bout"]
    kept = bouts["in_analyzed_subset"]
    return (
        bouts.assign(
            kept=kept,
            left_out=~kept,
            matched_outside=kept & ~bouts["matched_starts_in_well_visit"],
        )
        .groupby(["state", "regime"])
        .agg(
            n_bouts=("kept", "size"),
            n_kept=("kept", "sum"),
            n_left_out=("left_out", "sum"),
            n_kept_matched_outside_visit=("matched_outside", "sum"),
        )
        .reset_index()
    )
