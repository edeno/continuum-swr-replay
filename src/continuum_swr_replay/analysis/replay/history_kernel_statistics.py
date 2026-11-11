"""Summary statistics of the event-history kernels of the duration-aware rate fits.

The duration-aware refits (:mod:`.duration_aware_history`, one fit per animal
and population on the animal's stacked sessions; :mod:`.ridge_history_cv`, the
same with a ridge prior and a held-out comparison) write, per fit, each history
block's kernel: the log rate ratio of one past event at each lag, with its
standard error, and the expected onsets behind each lag. The functions here
read those kernels at their measurable lags and summarize them over animals.

A lag is measurable when it has at least :data:`HISTORY_MIN_SUPPORT` expected
onsets and its standard error is at most :data:`HISTORY_MAX_SE`; a lag that is
not is left out, never read as an effect. Lag index ``j`` is ``j + 1`` bins of
``BIN_S`` after the past event's anchor (its onset or its end). Lags are
reported in ms, kernels as rate ratios.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from continuum_swr_replay.analysis.replay.model_specs import BIN_S

HISTORY_MIN_SUPPORT = 3.0
"""A kernel's lag is read when it has at least this many expected onsets ..."""

HISTORY_MAX_SE = 0.5
"""... and its Wald standard error on the log scale is at most this."""

_WALD_Z = 1.96
"""The normal quantile of a two-sided 95 % interval."""


def lag_ms(index: int) -> int:
    """Lag index ``index`` of a history kernel in ms: ``index + 1`` bins of ``BIN_S``."""
    return round((index + 1) * BIN_S * 1e3)


def lag_index(ms: float) -> int:
    """The history-kernel index of a lag of ``ms`` milliseconds."""
    return round(ms / (BIN_S * 1e3)) - 1


def measurable_kernel(record: dict, rung: str, block: str) -> dict | None:
    """One fit's kernel of one history block, with its measurable lags.

    Parameters
    ----------
    record : dict
        One animal and population's result of the pooled duration-aware refit:
        ``duration_aware`` (rung -> fit, with ``ok``, ``converged`` and
        ``kernels``: block -> ``log_rr`` and ``se`` per lag) and ``support``
        (block -> expected onsets per lag).
    rung : str
        The rung whose kernel is read.
    block : str
        The history block (``"self_onset"``, ``"self_offset"``).

    Returns
    -------
    kernel : dict or None
        ``log_rr`` and ``se``, shape (n_lags,), and ``ok``, whether each lag is
        measurable. None when the rung did not fit, did not converge, or lacks
        the block.
    """
    fit = record["duration_aware"].get(rung)
    if (
        fit is None
        or not fit["ok"]
        or not fit["converged"]
        or block not in fit["kernels"]
    ):
        return None
    k = fit["kernels"][block]
    log_rr, se = np.asarray(k["log_rr"]), np.asarray(k["se"])
    ok = (np.asarray(record["support"][block]) >= HISTORY_MIN_SUPPORT) & (
        se <= HISTORY_MAX_SE
    )
    return {"log_rr": log_rr, "se": se, "ok": ok}


def kernel_peak(kernel: dict) -> dict | None:
    """The measurable lag at which a kernel's rate ratio is largest.

    The lag is chosen for being the largest, so its own 95 % Wald interval would
    clear 1 more often than 5 % of the time with no effect at any lag. The
    interval is therefore widened for the number of measurable lags searched
    (Bonferroni: two-sided level 0.05 / m).

    Parameters
    ----------
    kernel : dict
        A kernel with its measurable lags (:func:`measurable_kernel`).

    Returns
    -------
    peak : dict or None
        ``rate_ratio``, the largest rate ratio over the measurable lags;
        ``lag_ms``, its lag; and ``above_one``, whether its widened interval
        lies above a rate ratio of 1. None when no lag is measurable.
    """
    if not kernel["ok"].any():
        return None
    j = int(np.nanargmax(np.where(kernel["ok"], kernel["log_rr"], np.nan)))
    lr, se = kernel["log_rr"][j], kernel["se"][j]
    z = norm.isf(0.025 / int(kernel["ok"].sum()))
    return {
        "rate_ratio": float(np.exp(lr)),
        "lag_ms": lag_ms(j),
        "above_one": bool(lr - z * se > 0),
    }


def self_history_statistics(records: Sequence[dict], rung: str) -> pd.DataFrame:
    """Each population's self-history kernels, over its animals' pooled fits.

    The end-locked kernel (``self_offset``) is read at its first lag and at its
    peak; the onset-locked kernel (``self_onset``), where the rung carries one,
    at its peak. An animal whose lag is not measurable is left out of that
    lag's statistics.

    Parameters
    ----------
    records : sequence of dict
        The pooled refit's results, one per animal and population
        (:func:`measurable_kernel`), each with ``animal`` and ``outcome``.
    rung : str
        The rung whose kernels are read.

    Returns
    -------
    statistics : pd.DataFrame
        One row per population (``outcome``):

        - ``n_fits``: the animals' fits.
        - ``n_first_lag_measurable``: fits whose first end-locked lag is
          measurable.
        - ``first_lag_rate_ratio_median``: the median over those of the rate
          ratio at the first lag after the previous event's end.
        - ``n_first_lag_suppressed``, ``n_first_lag_raised``: of those, fits
          whose 95 % Wald interval at that lag lies below, and above, a rate
          ratio of 1.
        - ``end_peak_rate_ratio_median``, ``end_peak_lag_ms_median``: the
          median over animals of the end-locked kernel's peak rate ratio and
          of its lag (:func:`kernel_peak`).
        - ``n_end_peaks_above_one``: animals whose end-locked peak has its
          widened interval above 1.
        - ``onset_peak_rate_ratio_median``, ``onset_peak_lag_ms_median``,
          ``n_onset_peaks_above_one``: the same of the onset-locked kernel.

    Raises
    ------
    ValueError
        If an animal's fit of ``rung`` failed, did not converge or has no
        end-locked kernel, or no animal of a population has a measurable first
        lag.
    """
    by_population: dict[str, dict[str, dict]] = {}
    for record in records:
        by_population.setdefault(record["outcome"], {})[record["animal"]] = record
    rows = {}
    for outcome, by_animal in by_population.items():
        ends, onsets, first = [], [], []
        n_suppressed = n_raised = 0
        for record in by_animal.values():
            end = measurable_kernel(record, rung, "self_offset")
            onset = measurable_kernel(record, rung, "self_onset")
            if end is None:
                raise ValueError(
                    f"{outcome}/{record['animal']}: {rung} failed or did not converge"
                )
            if end["ok"][0]:
                lr, se = end["log_rr"][0], end["se"][0]
                first.append(float(np.exp(lr)))
                n_suppressed += bool(lr + _WALD_Z * se < 0)
                n_raised += bool(lr - _WALD_Z * se > 0)
            if (peak := kernel_peak(end)) is not None:
                ends.append(peak)
            if onset is not None and (peak := kernel_peak(onset)) is not None:
                onsets.append(peak)
        if not first:
            raise ValueError(f"{outcome}: no animal's {lag_ms(0)} ms lag is measurable")
        rows[outcome] = {
            "n_fits": len(by_animal),
            "n_first_lag_measurable": len(first),
            "first_lag_rate_ratio_median": float(np.median(first)),
            "n_first_lag_suppressed": n_suppressed,
            "n_first_lag_raised": n_raised,
            "end_peak_rate_ratio_median": np.median([p["rate_ratio"] for p in ends]),
            "end_peak_lag_ms_median": np.median([p["lag_ms"] for p in ends]),
            "n_end_peaks_above_one": sum(p["above_one"] for p in ends),
            "onset_peak_rate_ratio_median": np.median(
                [p["rate_ratio"] for p in onsets]
            ),
            "onset_peak_lag_ms_median": np.median([p["lag_ms"] for p in onsets]),
            "n_onset_peaks_above_one": sum(p["above_one"] for p in onsets),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("outcome")


def self_history_range_statistics(by_population: pd.DataFrame) -> dict[str, float]:
    """The populations' end-locked self-history statistics, together.

    Parameters
    ----------
    by_population : pd.DataFrame
        One row per population (:func:`self_history_statistics`).

    Returns
    -------
    statistics : dict
        - ``n_fits``, ``n_first_lag_measurable``, ``n_first_lag_suppressed``,
          ``n_first_lag_raised``: the fits of every population, summed.
        - ``first_lag_rate_ratio_low``, ``first_lag_rate_ratio_high``: the
          lowest and highest over the populations of the median first-lag rate
          ratio.
        - ``end_peak_rate_ratio_low``, ``end_peak_rate_ratio_high``: the same
          of the median end-locked peak.
        - ``end_peak_lag_ms_low``, ``end_peak_lag_ms_high``: the shortest and
          longest of the median end-locked peak lag.
        - ``n_end_peaks_above_one_min``: the fewest, over the populations,
          animals whose end-locked peak has its widened interval above 1.
    """
    statistics = {
        name: int(by_population[name].sum())
        for name in (
            "n_fits",
            "n_first_lag_measurable",
            "n_first_lag_suppressed",
            "n_first_lag_raised",
        )
    }
    for name, column in (
        ("first_lag_rate_ratio", "first_lag_rate_ratio_median"),
        ("end_peak_rate_ratio", "end_peak_rate_ratio_median"),
        ("end_peak_lag_ms", "end_peak_lag_ms_median"),
    ):
        statistics[f"{name}_low"] = by_population[column].min()
        statistics[f"{name}_high"] = by_population[column].max()
    statistics["n_end_peaks_above_one_min"] = int(
        by_population["n_end_peaks_above_one"].min()
    )
    return statistics


def cross_history_statistics(
    records: Sequence[dict], outcome: str, cross_rung: str, at_ms: float
) -> dict[str, float]:
    """One stream's rate after the other stream's events end, over animals.

    Parameters
    ----------
    records : sequence of dict
        The ridge cross-validated comparison's results, one per animal and
        response population: ``animal``, ``outcome``, ``gain_per_event`` (the
        held-out log-likelihood gain per held-out event of the self+cross rung
        over the self-only rung), ``support`` (block -> expected onsets per
        lag) and ``models`` (rung -> fit, with ``full_converged``, ``cv``
        (``n_nonconverged``) and ``kernels``: block -> ``log_rr`` and its
        posterior ``sd`` per lag).
    outcome : str
        The response population.
    cross_rung : str
        The self+cross rung, whose end-locked cross kernel (``cross_offset``)
        is read.
    at_ms : float
        The lag after the other stream's event ended, in ms.

    Returns
    -------
    statistics : dict
        - ``n_animals``: the population's animals.
        - ``n_measurable``: animals whose lag has at least
          :data:`HISTORY_MIN_SUPPORT` expected onsets.
        - ``rate_ratio_median``: the median over those of the rate ratio at
          the lag.
        - ``n_animals_above_one``: of those, animals whose 95 % interval
          (posterior mode +- 1.96 posterior SD) lies above 1.
        - ``heldout_gain_median``: the median over all the animals of the
          held-out gain, in nats per event.
        - ``n_animals_gain_positive``: animals whose held-out gain is positive.

    Raises
    ------
    ValueError
        If the population has no record, a ridge fit of it did not converge,
        or no animal's lag is measurable.
    """
    lag = lag_index(at_ms)
    records = [record for record in records if record["outcome"] == outcome]
    if not records:
        raise ValueError(f"{outcome}: no ridge cross-history fit")
    rate_ratios, gains, n_above = [], [], 0
    for record in records:
        for rung, fit in record["models"].items():
            if not (fit["full_converged"] and fit["cv"]["n_nonconverged"] == 0):
                raise ValueError(
                    f"{outcome}/{record['animal']}: ridge {rung} did not converge"
                )
        gains.append(record["gain_per_event"])
        kernel = record["models"][cross_rung]["kernels"]["cross_offset"]
        lr, sd = kernel["log_rr"][lag], kernel["sd"][lag]
        if record["support"]["cross_offset"][lag] < HISTORY_MIN_SUPPORT:
            continue
        rate_ratios.append(float(np.exp(lr)))
        n_above += bool(lr - _WALD_Z * sd > 0)
    if not rate_ratios:
        raise ValueError(f"{outcome}: no animal's {at_ms} ms lag is measurable")
    return {
        "n_animals": len(records),
        "n_measurable": len(rate_ratios),
        "rate_ratio_median": np.median(rate_ratios),
        "n_animals_above_one": n_above,
        "heldout_gain_median": np.median(gains),
        "n_animals_gain_positive": int(sum(gain > 0 for gain in gains)),
    }
