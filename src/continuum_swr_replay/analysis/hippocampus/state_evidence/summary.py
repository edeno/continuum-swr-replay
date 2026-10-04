"""Summary statistics of the per-window table, pooled over epochs."""

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
"""A calibration bin with fewer windows is not compared."""

SUMMARY_SEED = 20261004

ENCODING_LABELS = {"run": "running-only", "can": "canonical"}
"""The evidence columns' prefixes: the running-only refit and the decode's own."""

_RESAMPLE_CELLS = 1 << 24
"""Random draws per chunk of a resampling test, so memory stays bounded."""


def _chunk(n_values: int, n_draws: int) -> int:
    return int(max(1, min(n_draws, _RESAMPLE_CELLS // max(n_values, 1))))


def sign_flip_p(x: NDArray, *, n_flips: int = 10_000, seed: Seed) -> float:
    """Two-sided sign-flip p of mean(x) = 0, ``(hits + 1) / (n_flips + 1)``.

    Non-finite values are dropped; NaN when none is left.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return float("nan")
    stream = keyed_rng(seed, "flip")
    observed = abs(x.mean())
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

    Non-finite values are dropped; NaNs when fewer than two are left.
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

    Windows are resampled within strata. Pairs with a non-finite value are
    dropped.

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
    """95 % bootstrap CI of sum(num) / sum(den), resampling within strata."""
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


def _consistency(
    values: pd.Series, baseline: pd.Series, groups: pd.Series
) -> tuple[str, str]:
    """How many groups have a positive mean, and a mean above their baseline's."""
    mean = values.groupby(groups).mean()
    base = baseline.groupby(groups).mean()
    return (
        f"{int((mean > 0).sum())}/{len(mean)}",
        f"{int((mean > base).sum())}/{len(mean)}",
    )


def bout_statistics(
    windows: pd.DataFrame,
    *,
    metrics: Mapping[str, Sequence[str]] | None = None,
    n_flips: int = 10_000,
    n_boot: int = 10_000,
    seed: Seed = SUMMARY_SEED,
) -> pd.DataFrame:
    """Bouts against their matched windows, per state x regime x metric.

    Bouts are pooled over epochs. Each metric gets its median [IQR], share
    > 0, mean with a two-sided sign-flip p and a 95 % bootstrap CI (resampled
    within epochs), evidence per held-out spike (``{metric}_ps`` when present),
    the paired bout - baseline difference with the same tests, and the count of
    epochs and of animals whose mean is > 0 and above their baseline's.

    Parameters
    ----------
    windows : pd.DataFrame
        The per-window table: ``epoch_id``, ``animal``, ``kind``, ``state``,
        ``regime``, ``bout_s``, ``matched_bout_s``, ``n_found`` and the metrics.
    metrics : mapping, optional
        State -> metric columns; default each state's :data:`PRIMARY_METRICS`
        and every :data:`SECONDARY_METRICS`.

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
                }
                if len(cell) == 0:
                    rows.append(row)
                    continue
                x = cell[metric].to_numpy(dtype=float)
                xb = cell[f"{metric}_base"].to_numpy(dtype=float)
                diff = x - xb
                epochs = cell.epoch_id.to_numpy()
                row.update(_describe(x))
                row["p_signflip"] = sign_flip_p(
                    x, n_flips=n_flips, seed=_with_keys(seed, *key)
                )
                row["ci_lo"], row["ci_hi"] = stratified_bootstrap_ci(
                    x, epochs, n_boot=n_boot, seed=_with_keys(seed, *key)
                )
                row.update(_describe(xb, prefix="base_"))
                row["diff_mean"] = _describe(diff)["mean"]
                row["p_diff"] = sign_flip_p(
                    diff, n_flips=n_flips, seed=_with_keys(seed, *key, "diff")
                )
                row["diff_ci_lo"], row["diff_ci_hi"] = stratified_bootstrap_ci(
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
                row["epochs_pos"], row["epochs_above_base"] = _consistency(
                    cell[metric], cell[f"{metric}_base"], cell.epoch_id
                )
                row["animals_pos"], row["animals_above_base"] = _consistency(
                    cell[metric], cell[f"{metric}_base"], cell.animal
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
    n_boot: int = 2000,
    seed: Seed = SUMMARY_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibration curves: evidence against each state's probability.

    Calibration windows are binned by their mean posterior of the state (0-0.2,
    ..., 0.8-1.0); each bin has n, mean with a 95 % bootstrap CI (resampled
    within epochs), median and share > 0. A curve's trend is the window-level
    OLS slope with a bootstrap CI. It **rises** when the slope's CI is above 0
    and the top bin's CI is above the bottom bin's; its **top bin is > 0** when
    that CI is above 0. Bins with fewer than :data:`MIN_BIN_WINDOWS` windows
    are not compared.

    Parameters
    ----------
    windows : pd.DataFrame
        Needs ``kind``, ``regime``, ``epoch_id``, ``p_{state}`` and the metrics.
    metrics : mapping, optional
        State -> metric columns; default :data:`CALIBRATION_METRICS`.

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
        epochs = cell.epoch_id.to_numpy()
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
                    }
                    if m.sum() >= 2:
                        row.update(
                            mean=float(x[m].mean()),
                            median=float(np.median(x[m])),
                            frac_pos=float((x[m] > 0).mean()),
                        )
                        row["ci_lo"], row["ci_hi"] = stratified_bootstrap_ci(
                            x[m],
                            epochs[m],
                            n_boot=n_boot,
                            seed=_with_keys(seed, regime, state, metric, name),
                        )
                    curve.append(row)
                bins += curve
                slope, lo, hi = ols_slope_ci(
                    p,
                    x,
                    epochs,
                    n_boot=n_boot,
                    seed=_with_keys(seed, regime, state, metric),
                )
                trend: dict[str, Any] = {
                    "regime": regime,
                    "state": state,
                    "metric": metric,
                    "n": int(np.isfinite(x).sum()),
                    "slope": slope,
                    "slope_ci_lo": lo,
                    "slope_ci_hi": hi,
                }
                bottom, top = curve[0], curve[-1]
                compared = (
                    bottom["n"] >= MIN_BIN_WINDOWS and top["n"] >= MIN_BIN_WINDOWS
                )
                separated = compared and top["ci_lo"] > bottom["ci_hi"]
                trend.update(
                    n_top=top["n"],
                    top_mean=top.get("mean", np.nan),
                    top_minus_bottom=(
                        top["mean"] - bottom["mean"] if compared else np.nan
                    ),
                    ci_separated=bool(separated),
                    rises=bool(separated and lo > 0),
                    top_positive=bool(top["n"] >= MIN_BIN_WINDOWS and top["ci_lo"] > 0),
                )
                trends.append(trend)
    return pd.DataFrame(bins), pd.DataFrame(trends)


def rate_calibration(windows: pd.DataFrame) -> pd.DataFrame:
    """Observed / expected held-out spikes in the calibration windows.

    Per regime and encoding, pooled (``animal == "all"``) and per animal:
    C's spike count summed over windows against E_C summed over windows,
    with the observed and predicted shares of silent C.
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
    windows: pd.DataFrame, *, n_boot: int = 2000, seed: Seed = SUMMARY_SEED
) -> pd.DataFrame:
    """Held-out coordinated silence: P(C silent | S silent) against local coding.

    Pooled over windows and splits (a ratio of sums), for each state's bouts
    and matched windows, and for the calibration windows by P(No-Spike) bin and
    all together. Local coding makes the halves independent given position, so
    its prediction for P(C silent | S silent) is exp(-E_C) averaged over the
    S-silent runs, under either encoding.
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
            if (silent_s > 0).sum() >= 5:
                row["obs_given_ci_lo"], row["obs_given_ci_hi"] = _ratio_ci(
                    both,
                    silent_s,
                    d.epoch_id.to_numpy(),
                    n_boot=n_boot,
                    seed=_with_keys(seed, regime, kind, state, name),
                )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_state_evidence(
    windows: pd.DataFrame, *, seed: Seed = SUMMARY_SEED
) -> dict[str, pd.DataFrame]:
    """Every summary of the per-window table, pooled over its epochs.

    ``run_loc_vs_Sremote_mark`` (and ``_ps``) are added as the negatives of
    ``run_nlr_anim_mark``, so that positive supports the Local claim.

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
    bins, trends = calibration_statistics(windows, seed=seed)
    return {
        "bouts": bout_statistics(windows, seed=seed),
        "calibration": bins,
        "trends": trends,
        "rate_calibration": rate_calibration(windows),
        "silence": coordinated_silence(windows, seed=seed),
    }
