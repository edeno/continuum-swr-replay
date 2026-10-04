"""Summary statistics of the per-window table, over epochs nested in animals.

The epoch is the unit of inference: about 218 epochs nested in 5 animals,
each holding hundreds to thousands of windows that are not independent.

- **Primary (inference).**
  - Each epoch enters as one value: its median over windows
    (:func:`epoch_summaries`) or, for a calibration curve, its own slope
    (:func:`epoch_slopes`).
  - CIs come from a hierarchical bootstrap (:func:`hierarchical_bootstrap_ci`)
    of the mean of the epoch values: animals are resampled, then epochs within
    each resampled animal (Saravanan, Berman & Sober 2020).
  - Tests are two-sided sign-flip tests on the epoch values (:func:`sign_flip_p`).
  - Consistency is reported as k of N epochs and k of A animals (an animal's
    value is the mean of its epochs') with a value > 0.
  - These columns carry ``epoch`` in their name, as do the calibration
    verdicts ``rises`` and ``top_positive``.
- **Why each epoch's median.** The per-window evidence is heavy-tailed. A burst
  of spikes on most tetrodes scores far below zero: in one pilot chimi epoch,
  three of 81 immobile No-Spike bouts (86 to 137 spikes on 19 to 22 tetrodes)
  score -1641, -1407 and -1016 nats under the running-only encoding, so that
  epoch's mean is -45 while its median is +5.4. A few windows drag an epoch's
  mean; its median is the robust summary. The choice was made on the five
  pilot (development) epochs, before the 218-epoch run that produces the
  reported numbers.
- **Secondary.** The same statistics of each epoch's *mean*: the columns named
  ``mean`` where the primary ones say ``median``, and the counts ending
  ``_mean``.
- **Descriptive.** Window-level counts, medians, quartiles, shares > 0 and
  means, pooled over epochs.
- **Calibration verdicts need windows.** ``rises`` and ``top_positive`` say
  "yes" or "no" only when the top and the bottom probability bins each hold at
  least :data:`MIN_BIN_WINDOWS` windows, pooled over epochs (otherwise
  :data:`TOO_FEW_WINDOWS`), and when at least two epochs give the statistic
  (otherwise :data:`TOO_FEW_EPOCHS`). An epoch's slope needs two bins of that
  many windows.
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
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from continuum_swr_replay.analysis.hippocampus.state_evidence.windows import (
    REGIMES,
    STATES,
    Seed,
    _with_keys,
    keyed_rng,
)

PRIMARY_METRICS: Mapping[str, tuple[str, ...]] = {
    "Local": ("run_loc_mark",),
    "No-Spike": ("run_ns_logratio", "can_ns_logratio"),
    "Non-Local Continuous": ("run_nl_null_mark", "run_nl_anim_mark"),
    "Non-Local Fragmented": ("run_nl_null_mark", "run_nl_anim_mark"),
}
"""The evidence each state's claim is judged by (``run_`` = running-only
encoding, ``can_`` = the decode's own)."""

SECONDARY_METRICS = (
    "run_loc_vs_Sremote_mark",
    "run_nlr_anim_mark",
    "can_loc_mark",
    "run_loc_full",
    "can_nl_null_mark",
    "can_nl_anim_mark",
    "run_nl_anim_full",
    "run_ns_count",
    "run_ns_silent_frac",
    "run_ns_expected",
    "can_ns_expected",
    "run_mapS_dist_cm",
    "run_mapSr_dist_cm",
)
"""Reported for the bouts of every state."""

CALIBRATION_METRICS: Mapping[str, tuple[str, ...]] = {
    "Local": ("run_loc_mark", "run_loc_vs_Sremote_mark", "can_loc_mark"),
    "No-Spike": (
        "run_ns_logratio",
        "can_ns_logratio",
        "run_ns_count",
        "run_ns_silent_frac",
    ),
    "Non-Local Continuous": (
        "run_nl_null_mark",
        "run_nl_anim_mark",
        "can_nl_null_mark",
        "can_nl_anim_mark",
    ),
    "Non-Local Fragmented": (
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
TOO_FEW_EPOCHS = "too few epochs"
"""A calibration verdict that is neither "yes" nor "no", and why."""

SUMMARIES = ("median", "mean")
"""Each epoch's summary over its windows: primary, then secondary."""

SUMMARY_SEED = 20261004

ENCODING_LABELS = {"run": "running-only", "can": "canonical"}
"""The evidence columns' prefixes: the running-only refit and the decode's own."""

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
    inference (:func:`hierarchical_bootstrap_ci` is). Non-finite values are
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
    centred = p - p.mean(-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        result: NDArray = (centred * (x - x.mean(-1, keepdims=True))).sum(-1) / (
            centred**2
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


def hierarchical_bootstrap_ci(
    summaries: NDArray, animals: NDArray, *, n_boot: int = 10_000, seed: Seed
) -> tuple[float, float]:
    """95 % hierarchical bootstrap CI of the mean of per-epoch summaries.

    Saravanan, Berman & Sober (2020). Each replicate resamples the animals with
    replacement and then, within each resampled animal, its epochs with
    replacement; its statistic is the mean over every resampled epoch, the same
    epoch-weighted mean as the estimate. Vectorized over replicates.

    Windows are not resampled. Each epoch enters through its summary over its
    windows, whose spread across epochs already contains the windows' sampling
    noise; resampling them as well would add that noise a second time. The epoch
    is the unit of inference.

    Parameters
    ----------
    summaries : ndarray, shape (n_epochs,)
        One value per epoch; non-finite ones are dropped.
    animals : ndarray, shape (n_epochs,)
        Each epoch's animal.
    n_boot : int, optional
    seed : int or sequence

    Returns
    -------
    ci_lo, ci_hi : float
        NaN when fewer than two epochs are left.
    """
    summaries = np.asarray(summaries, dtype=float)
    finite = np.isfinite(summaries)
    values, animals = summaries[finite], np.asarray(animals)[finite]
    if values.size < 2:
        return (float("nan"), float("nan"))
    codes, names = pd.factorize(animals)
    stream = keyed_rng(seed, "hierarchical")
    picks = stream.integers(0, len(names), size=(n_boot, len(names)))
    total = np.zeros(n_boot)
    count = np.zeros(n_boot)
    for k in range(len(names)):
        own = values[codes == k]
        replicate = np.nonzero(picks == k)[0]  # one entry for each draw of animal k
        draws = stream.integers(0, own.size, size=(replicate.size, own.size))
        np.add.at(total, replicate, own[draws].sum(1))
        np.add.at(count, replicate, own.size)
    lo, hi = np.percentile(total / count, [2.5, 97.5])
    return float(lo), float(hi)


def _epoch_statistics(
    summaries: pd.Series | NDArray,
    animals: pd.Series | NDArray,
    *,
    n_boot: int,
    n_flips: int,
    seed: Seed,
) -> dict[str, Any]:
    """The primary statistics of per-epoch values.

    Returns their mean (``mean``), its hierarchical CI (``ci_lo``, ``ci_hi``),
    the sign-flip p over the epochs (``p``), and how many epochs and animals
    (an animal's value is the mean of its epochs') are > 0 (``epochs``,
    ``animals``, as "k/N").
    """
    values = np.asarray(summaries, dtype=float)
    animal = np.asarray(animals)
    finite = np.isfinite(values)
    values, animal = values[finite], animal[finite]
    by_animal = pd.Series(values).groupby(animal).mean()
    lo, hi = hierarchical_bootstrap_ci(values, animal, n_boot=n_boot, seed=seed)
    return {
        "mean": float(values.mean()) if values.size else float("nan"),
        "ci_lo": lo,
        "ci_hi": hi,
        "p": sign_flip_p(values, n_flips=n_flips, seed=seed),
        "epochs": f"{int((values > 0).sum())}/{values.size}",
        "animals": f"{int((by_animal > 0).sum())}/{by_animal.size}",
    }


def _summary_seed(seed: Seed, kind: str) -> Seed:
    """The seed of one epoch summary's statistics.

    The mean's keep ``seed``, the one they had while the mean was primary, so
    they can be checked against those earlier outputs; the median's add
    ``kind``.
    """
    return seed if kind == "mean" else _with_keys(seed, kind)


def _level_columns(
    values: NDArray,
    epochs: NDArray,
    animals: NDArray,
    *,
    prefix: str,
    counts: str,
    n_boot: int,
    n_flips: int,
    seed: Seed,
) -> dict[str, Any]:
    """The epoch-level columns of ``values``, over each epoch's median and mean.

    For the median (primary): ``{prefix}epoch_median``, its CI
    ``{prefix}epoch_median_ci_lo`` / ``_ci_hi``, ``p_{prefix}epoch_median``,
    and ``epochs_{counts}`` / ``animals_{counts}``. For the mean (secondary):
    the same with ``mean`` for ``median``, the counts ending ``_mean``.
    """
    summaries = epoch_summaries(values, epochs, animals)
    columns: dict[str, Any] = {}
    for kind in SUMMARIES:
        level = _epoch_statistics(
            summaries[kind],
            summaries["animal"],
            n_boot=n_boot,
            n_flips=n_flips,
            seed=_summary_seed(seed, kind),
        )
        tail = "" if kind == "median" else f"_{kind}"
        columns[f"{prefix}epoch_{kind}"] = level["mean"]
        columns[f"{prefix}epoch_{kind}_ci_lo"] = level["ci_lo"]
        columns[f"{prefix}epoch_{kind}_ci_hi"] = level["ci_hi"]
        columns[f"p_{prefix}epoch_{kind}"] = level["p"]
        columns[f"epochs_{counts}{tail}"] = level["epochs"]
        columns[f"animals_{counts}{tail}"] = level["animals"]
    return columns


def _verdict(enough: bool, holds: bool) -> str:
    """The verdict on ``holds``: "yes" or "no", or :data:`TOO_FEW_WINDOWS`."""
    if not enough:
        return TOO_FEW_WINDOWS
    return "yes" if holds else "no"


def _epoch_verdict(enough: bool, ci_lo: float) -> str:
    """The verdict that an epoch-level CI lies above 0, or why there is none.

    :data:`TOO_FEW_WINDOWS` without ``enough`` windows; :data:`TOO_FEW_EPOCHS`
    when fewer than two epochs give the statistic (its CI is NaN).
    """
    if enough and not np.isfinite(ci_lo):
        return TOO_FEW_EPOCHS
    return _verdict(enough, ci_lo > 0)


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

    - **Primary, epoch level, over each epoch's median** (see the module notes):
      - ``epoch_median``: the mean over epochs of each epoch's median.
      - ``epoch_median_ci_lo`` / ``epoch_median_ci_hi``: its hierarchical
        bootstrap CI.
      - ``p_epoch_median``: the sign-flip p over the epoch medians.
      - ``epochs_pos`` / ``animals_pos``: k of N epochs and k of A animals
        with a median > 0.
      - The paired bout - baseline difference, each epoch entering as the
        median of its bouts' paired differences: ``diff_epoch_median``,
        ``diff_epoch_median_ci_lo`` / ``_ci_hi``, ``p_diff_epoch_median``,
        ``epochs_above_base`` and ``animals_above_base``.
    - **Secondary, epoch level, over each epoch's mean:** ``epoch_mean``,
      ``epoch_mean_ci_lo`` / ``_ci_hi``, ``p_epoch_mean``, ``epochs_pos_mean``
      and ``animals_pos_mean``, and for the difference ``diff_epoch_mean``,
      ``diff_epoch_mean_ci_lo`` / ``_ci_hi``, ``p_diff_epoch_mean``,
      ``epochs_above_base_mean`` and ``animals_above_base_mean``.
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
    keys = ["epoch_id", "state", "regime"]
    bouts = windows[windows.kind == "bout"]
    paired = bouts.merge(
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
    rows = []
    for state, state_metrics in metrics.items():
        for regime in REGIMES:
            cell = paired[(paired.state == state) & (paired.regime == regime)]
            found = bouts[(bouts.state == state) & (bouts.regime == regime)]
            n_found = int(found.groupby("epoch_id").n_found.first().sum())
            epochs, animals = cell.epoch_id.to_numpy(), cell.animal.to_numpy()
            for metric in state_metrics:
                key = (state, regime, metric)
                row: dict[str, Any] = {
                    "state": state,
                    "regime": regime,
                    "metric": metric,
                    "primary": metric in PRIMARY_METRICS.get(state, ()),
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
                row.update(
                    _level_columns(
                        x, epochs, animals, prefix="", counts="pos",
                        n_boot=n_boot, n_flips=n_flips,
                        seed=_with_keys(seed, *key, "epoch"),
                    )
                )  # fmt: skip
                row.update(_describe(xb, prefix="base_"))
                row["diff_mean"] = _describe(diff)["mean"]
                row.update(
                    _level_columns(
                        diff, epochs, animals, prefix="diff_", counts="above_base",
                        n_boot=n_boot, n_flips=n_flips,
                        seed=_with_keys(seed, *key, "diff", "epoch"),
                    )
                )  # fmt: skip
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibration curves: evidence against each state's probability.

    Calibration windows are binned by their mean posterior of the state (0-0.2,
    ..., 0.8-1.0).

    - **Primary, epoch level:**
      - A bin's ``epoch_median`` is the mean over epochs of each epoch's median
        in the bin, with its hierarchical CI ``epoch_median_ci_lo`` /
        ``epoch_median_ci_hi``.
      - A curve's ``epoch_slope`` is the mean of the epochs' own OLS slopes
        (:func:`epoch_slopes`, which takes only epochs with two bins of
        :data:`MIN_BIN_WINDOWS` windows), with its hierarchical CI, the
        sign-flip ``p_epoch_slope`` and ``epochs_rising`` / ``animals_rising``.
      - The verdicts need the top and the bottom bins to hold at least
        :data:`MIN_BIN_WINDOWS` windows each, pooled (``n_top``,
        ``n_bottom``); otherwise they read :data:`TOO_FEW_WINDOWS`, and
        :data:`TOO_FEW_EPOCHS` when fewer than two epochs give the CI.
      - ``rises`` is "yes" when the epoch slope's CI lies above 0, else "no".
      - ``top_positive`` is "yes" when the top bin's ``epoch_median`` CI
        (``top_epoch_median_ci_lo``) lies above 0, else "no".
    - **Secondary, over each epoch's mean:** a bin's ``epoch_mean`` with
      ``epoch_mean_ci_lo`` / ``epoch_mean_ci_hi``; a curve's ``top_epoch_mean``.
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
        Resamples of the hierarchical and of the window-level bootstraps.
    window_level : bool, optional
        Add the window-level columns (default False).

    Returns
    -------
    bins, trends : pd.DataFrame
    """
    if metrics is None:
        metrics = CALIBRATION_METRICS
    calibration = windows[windows.kind == "calib"]
    bins, trends = [], []
    for regime in REGIMES:
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
                x = cell[metric].to_numpy(dtype=float)
                curve = []
                for name in PROBABILITY_BIN_LABELS:
                    m = (label == name) & np.isfinite(x)
                    row: dict[str, Any] = {
                        "regime": regime,
                        "state": state,
                        "metric": metric,
                        "pbin": name,
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
                    for kind in SUMMARIES:
                        values = summaries[kind]
                        row[f"epoch_{kind}"] = (
                            float(values.mean()) if len(values) else float("nan")
                        )
                        (
                            row[f"epoch_{kind}_ci_lo"],
                            row[f"epoch_{kind}_ci_hi"],
                        ) = hierarchical_bootstrap_ci(
                            values,
                            summaries["animal"],
                            n_boot=n_boot,
                            seed=_summary_seed(
                                _with_keys(seed, regime, state, metric, name, "epoch"),
                                kind,
                            ),
                        )
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
                    "n": int(np.isfinite(x).sum()),
                    "n_bottom": bottom["n"],
                    "n_top": top["n"],
                    "n_epochs": len(slopes),
                    "epoch_slope": level["mean"],
                    "epoch_slope_ci_lo": level["ci_lo"],
                    "epoch_slope_ci_hi": level["ci_hi"],
                    "p_epoch_slope": level["p"],
                    "epochs_rising": level["epochs"],
                    "animals_rising": level["animals"],
                    "rises": _epoch_verdict(enough, level["ci_lo"]),
                    "top_epoch_median": top["epoch_median"],
                    "top_epoch_median_ci_lo": top["epoch_median_ci_lo"],
                    "top_positive": _epoch_verdict(enough, top["epoch_median_ci_lo"]),
                    "top_epoch_mean": top["epoch_mean"],
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
    with the observed and predicted shares of silent C. Descriptive.
    """
    calibration = windows[windows.kind == "calib"]
    rows = []
    for regime in REGIMES:
        cell = calibration[calibration.regime == regime]
        for prefix, label in ENCODING_LABELS.items():
            groups = [("all", cell)] + [
                (str(animal), g) for animal, g in cell.groupby("animal", sort=True)
            ]
            for animal, g in groups:
                observed = float(g.run_ns_count.sum())
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
                        "obs_silent_share": float(g.run_ns_silent_frac.mean()),
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

    - **Primary, epoch level.** Each epoch's observed ratio and local coding's
      predicted ratio are ratios of the epoch's own sums of silent-run counts
      (no per-window evidence, so no median).
      - ``epoch_obs_given_S`` is the observed ratios' mean over epochs, with
        its hierarchical CI ``epoch_obs_given_S_ci_lo`` / ``_ci_hi``.
      - ``epoch_pred_given_S_run`` / ``_can`` are the predictions' means.
      - ``epoch_excess_run`` / ``_can`` is the mean of each epoch's observed -
        predicted. It comes with its hierarchical CI, the sign-flip
        ``p_epoch_excess_*``, and ``epochs_above_pred_*`` /
        ``animals_above_pred_*``.
    - **Descriptive:** pooled ratios of sums over windows and splits:
      ``obs_C_silent_given_S_silent``, ``pred_given_S_run`` / ``_can``,
      ``obs_C_silent``, ``pred_C_silent_run`` / ``_can``.
    - **Window level, only with** ``window_level`` (not for inference):
      ``obs_given_ci_lo_window_level`` / ``_hi`` (windows resampled within
      epochs).
    """
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
        silent_s = d.run_cond_S_silent.to_numpy(dtype=float)
        both = d.run_cond_both_silent.to_numpy(dtype=float)
        row: dict[str, Any] = {
            "regime": regime,
            "kind": kind,
            "state": state,
            "pbin": name,
            "n_windows": len(d),
            "runs_S_silent": int(silent_s.sum()),
            "obs_C_silent": float(d.run_ns_silent_frac.mean()),
            "pred_C_silent_run": float(d.run_ns_pred_silent.mean()),
            "pred_C_silent_can": float(d.can_ns_pred_silent.mean()),
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
                silent=("run_cond_S_silent", "sum"),
                both=("run_cond_both_silent", "sum"),
                run=("run_cond_pred_C_silent", "sum"),
                can=("can_cond_pred_C_silent", "sum"),
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
            )
            row["epoch_obs_given_S"] = level["mean"]
            row["epoch_obs_given_S_ci_lo"] = level["ci_lo"]
            row["epoch_obs_given_S_ci_hi"] = level["ci_hi"]
            for prefix in ENCODING_LABELS:
                predicted = sums[prefix] / sums["silent"]
                row[f"epoch_pred_given_S_{prefix}"] = float(predicted.mean())
                level = _epoch_statistics(
                    observed - predicted,
                    sums["animal"],
                    n_boot=n_boot,
                    n_flips=n_flips,
                    seed=_with_keys(seed, regime, kind, state, name, prefix, "excess"),
                )
                row[f"epoch_excess_{prefix}"] = level["mean"]
                row[f"epoch_excess_{prefix}_ci_lo"] = level["ci_lo"]
                row[f"epoch_excess_{prefix}_ci_hi"] = level["ci_hi"]
                row[f"p_epoch_excess_{prefix}"] = level["p"]
                row[f"epochs_above_pred_{prefix}"] = level["epochs"]
                row[f"animals_above_pred_{prefix}"] = level["animals"]
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_state_evidence(
    windows: pd.DataFrame, *, seed: Seed = SUMMARY_SEED, window_level: bool = False
) -> dict[str, pd.DataFrame]:
    """Every summary of the per-window table, pooled over its epochs.

    ``run_loc_vs_Sremote_mark`` (and ``_ps``) are added as the negatives of
    ``run_nlr_anim_mark``, so that positive supports the Local claim.
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
        run_loc_vs_Sremote_mark=-windows["run_nlr_anim_mark"],
        run_loc_vs_Sremote_mark_ps=-windows["run_nlr_anim_mark_ps"],
    )
    bins, trends = calibration_statistics(windows, seed=seed, window_level=window_level)
    return {
        "bouts": bout_statistics(windows, seed=seed, window_level=window_level),
        "calibration": bins,
        "trends": trends,
        "rate_calibration": rate_calibration(windows),
        "silence": coordinated_silence(windows, seed=seed, window_level=window_level),
    }
