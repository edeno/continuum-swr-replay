"""Summary statistics of the mPFC unit population.

Each statistic takes a table an analysis wrote (the unit table of
:func:`~continuum_swr_replay.analysis.pfc.load_population.load_population_data`,
or a table of what the bundles' provenance records per session) and returns
named numbers: a dict, or a table with one row per group. They are the
statistics the mpfc summary reports (``continuum_swr_replay.reported.mpfc``
names and describes them). Two functions select units instead:
:func:`called_units` returns a mask and :func:`units_at_or_above_rate` the
rows of the units at or above a firing rate. The timing statistics of the
peri-event curves are in :mod:`~continuum_swr_replay.analysis.pfc.timing`;
:func:`swr_without_replay_timing_statistics` applies one of them to the units
called for SWRs without replay.

A unit is *tested* for an event class when its plus-one empirical p-value
(:data:`~continuum_swr_replay.analysis.pfc.load_population.HELDOUT_P_COLUMNS`)
is finite, and *called* when that p-value is below ``alpha``. The fractions of
units are descriptive, pooled over sessions and animals: they are not
population inference. A unit's *peak* for a curve is its
``<curve>: peak_log_modulation``, the fit's largest log-rate change over the
reported window, signed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from scipy.stats import false_discovery_control

from continuum_swr_replay.analysis.pfc.load_population import (
    HELDOUT_P_COLUMNS,
    MATCHED_NULL_OBSERVED_COLUMNS,
    significant_fraction,
)
from continuum_swr_replay.analysis.pfc.timing import (
    DIRECTIONS,
    peak_and_onset,
    population_rise_timing,
)
from continuum_swr_replay.decoding.identifiers import parse_nwb_filename


def lowest_and_highest(values: pd.Series) -> dict[str, object]:
    """The lowest and the highest of a group's values, and which group has each.

    Parameters
    ----------
    values : pd.Series, shape (n_groups,)
        One value per group, indexed by the group's name.

    Returns
    -------
    extremes : dict
        ``min`` and ``max``, the lowest and highest value (a missing value is
        left out), and ``min_label`` and ``max_label``, the first group that
        has each.
    """
    return {
        "min": values.min(),
        "min_label": values.idxmin(),
        "max": values.max(),
        "max_label": values.idxmax(),
    }


# --- the cohort and its fits ---------------------------------------------------------


def cohort_count_statistics(counts: pd.DataFrame) -> dict[str, float]:
    """The bins and events the unit models were fit on, over sessions.

    Parameters
    ----------
    counts : pd.DataFrame, shape (n_sessions, n_columns)
        One row per session: the immobile 2 ms bins before
        (``n_bins_immobile``) and after (``n_bins_fit``) the restriction to
        kept stops, the non-local events (``n_nonlocal_before``,
        ``n_nonlocal_after``) and SWRs (``n_ripple_before``,
        ``n_ripple_after``) before and after it, and the kept non-local events
        that are SWR-coupled (``n_nonlocal_swr``).

    Returns
    -------
    statistics : dict
        - ``n_bins_fit``, ``n_bins_immobile``, ``n_nonlocal_kept``,
          ``n_nonlocal_before``: each count summed over sessions. A session's
          missing count is left out of that sum.
        - ``bins_fit_fraction``, ``nonlocal_kept_fraction``,
          ``swrs_kept_fraction``: the share of the immobile bins, non-local
          events and SWRs the restriction keeps, pooled (summed kept over
          summed before).
        - ``swr_coupled_share_session_median``: the median over sessions of
          the share of a session's kept non-local events that are SWR-coupled.
          A session with no kept non-local event has no share and is left out
          of the median.
    """
    # Each column is summed by itself, so that a column made fractional by a
    # missing count leaves the other totals whole.
    totals = {
        name: counts[name].sum()
        for name in (
            "n_bins_fit",
            "n_bins_immobile",
            "n_nonlocal_after",
            "n_nonlocal_before",
            "n_ripple_after",
            "n_ripple_before",
        )
    }
    swr_share = counts["n_nonlocal_swr"] / counts["n_nonlocal_after"]
    return {
        "n_bins_fit": totals["n_bins_fit"],
        "n_bins_immobile": totals["n_bins_immobile"],
        "bins_fit_fraction": totals["n_bins_fit"] / totals["n_bins_immobile"],
        "n_nonlocal_kept": totals["n_nonlocal_after"],
        "n_nonlocal_before": totals["n_nonlocal_before"],
        "nonlocal_kept_fraction": (
            totals["n_nonlocal_after"] / totals["n_nonlocal_before"]
        ),
        "swrs_kept_fraction": totals["n_ripple_after"] / totals["n_ripple_before"],
        "swr_coupled_share_session_median": swr_share.median(),
    }


def fit_count_statistics(fits: pd.DataFrame) -> dict[str, float]:
    """How many of the unit models' fits did not converge, over sessions.

    Parameters
    ----------
    fits : pd.DataFrame, shape (n_sessions, 3)
        One row per session: its neuron x model fits (``n_model_fits``), those
        that ran to the iteration limit (``n_nonconverged_model_fits``) and the
        neurons with at least one such fit
        (``n_neurons_with_nonconverged_model``).

    Returns
    -------
    statistics : dict
        The three counts summed over sessions, under their column names (a
        session's missing count is left out of that sum), and
        ``nonconverged_percent``: the non-converged fits as a percent of the
        fits.
    """
    totals = {
        name: fits[name].sum()
        for name in (
            "n_model_fits",
            "n_nonconverged_model_fits",
            "n_neurons_with_nonconverged_model",
        )
    }
    return {
        "n_model_fits": totals["n_model_fits"],
        "n_nonconverged_model_fits": totals["n_nonconverged_model_fits"],
        "n_neurons_with_nonconverged_model": totals[
            "n_neurons_with_nonconverged_model"
        ],
        "nonconverged_percent": (
            100 * totals["n_nonconverged_model_fits"] / totals["n_model_fits"]
        ),
    }


def prior_binding_percent(population: pd.DataFrame, curves: Sequence[str]) -> pd.Series:
    """How often the prior moves a curve's peak, per curve.

    A unit is well posed for a curve when its unpenalised fit converged
    (``<curve>: mle_converged``) and has no separated column
    (``<curve>: mle_separated``). The prior binds on it when its penalised peak
    differs from the unpenalised one by more than that peak's standard error
    (``<curve>: prior_binds``). A missing flag counts as False.

    Parameters
    ----------
    population : pd.DataFrame, shape (n_units, n_columns)
        The unit table, with the three flags of every curve.
    curves : sequence of str
        The curves (``"SWR"``, ``"non_local_SWR"``, ...).

    Returns
    -------
    percent : pd.Series, shape (n_curves,)
        By curve: the well-posed units the prior binds on, as a percent of the
        well-posed units. NaN, with a NumPy warning, for a curve with no
        well-posed unit.
    """
    shares = {}
    for curve in curves:
        # ``== True`` reads a missing flag as False.
        converged = population[f"{curve}: mle_converged"].eq(True)
        separated = population[f"{curve}: mle_separated"].eq(True)
        well_posed = converged & ~separated
        binds = population[f"{curve}: prior_binds"].eq(True)
        shares[curve] = 100 * (binds & well_posed).sum() / well_posed.sum()
    return pd.Series(shares)


# --- the units called for each test ---------------------------------------------------

#: The p-value column of SWRs without replay, which only the extended event set
#: has (it is not one of ``HELDOUT_P_COLUMNS``).
SWR_WITHOUT_REPLAY_P_COLUMN = "SWR_no_non_local: p_heldout"


def _tested_and_called(pvalue: pd.Series, alpha: float) -> tuple[NDArray, NDArray]:
    """Which units are tested for one test (a finite p-value) and which are
    called (a p-value below ``alpha``). Two boolean arrays, shape (n_units,)."""
    values = pvalue.to_numpy(dtype=float)
    return np.isfinite(values), values < alpha


def called_units(population: pd.DataFrame, name: str, alpha: float) -> NDArray:
    """Which units are called for one test.

    Parameters
    ----------
    population : pd.DataFrame, shape (n_units, n_columns)
        The unit table.
    name : str
        A key of
        :data:`~continuum_swr_replay.analysis.pfc.load_population.HELDOUT_P_COLUMNS`.
    alpha : float
        The raw empirical-p threshold.

    Returns
    -------
    called : NDArray of bool, shape (n_units,)
        True where the unit's p-value is below ``alpha``; False for an
        untested unit.
    """
    return _tested_and_called(population[HELDOUT_P_COLUMNS[name]], alpha)[1]


def called_fractions(population: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """The share of tested units called, for every test.

    Parameters
    ----------
    population : pd.DataFrame, shape (n_units, n_columns)
        The unit table.
    alpha : float
        The raw empirical-p threshold.

    Returns
    -------
    fractions : pd.DataFrame, shape (n_tests, 3)
        Indexed by the keys of
        :data:`~continuum_swr_replay.analysis.pfc.load_population.HELDOUT_P_COLUMNS`:
        ``n_called``, ``n_tested`` and ``fraction``, the called units as a
        share of the tested ones
        (:func:`~continuum_swr_replay.analysis.pfc.load_population.significant_fraction`).

    Raises
    ------
    ZeroDivisionError
        If no unit is tested for one of the tests.
    """
    rows = {}
    for name, (n_called, n_tested) in significant_fraction(
        population, alpha=alpha
    ).items():
        rows[name] = {
            "n_called": n_called,
            "n_tested": n_tested,
            "fraction": n_called / n_tested,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def called_fraction_by_animal(
    population: pd.DataFrame, name: str, alpha: float
) -> pd.DataFrame:
    """The share of each animal's tested units called for one test.

    Parameters
    ----------
    population : pd.DataFrame, shape (n_units, n_columns)
        The unit table, with ``animal``.
    name : str
        A key of
        :data:`~continuum_swr_replay.analysis.pfc.load_population.HELDOUT_P_COLUMNS`.
    alpha : float
        The raw empirical-p threshold.

    Returns
    -------
    fractions : pd.DataFrame, shape (n_animals, 3)
        Indexed by animal, in sorted order: ``n_called``, ``n_tested`` and
        ``fraction`` as in :func:`called_fractions`, over the animal's units.

    Raises
    ------
    ZeroDivisionError
        If an animal has no unit tested for ``name``.
    """
    rows = {}
    for animal, frame in population.groupby("animal"):
        n_called, n_tested = significant_fraction(frame, alpha=alpha)[name]
        rows[animal] = {
            "n_called": n_called,
            "n_tested": n_tested,
            "fraction": n_called / n_tested,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def bh_called_fraction(population: pd.DataFrame, name: str, alpha: float) -> float:
    """The share of tested units with a Benjamini-Hochberg q below ``alpha``.

    BH is applied once, over the pooled p-values of every unit tested for
    ``name`` (:func:`scipy.stats.false_discovery_control`).

    Parameters
    ----------
    population : pd.DataFrame, shape (n_units, n_columns)
        The unit table.
    name : str
        A key of
        :data:`~continuum_swr_replay.analysis.pfc.load_population.HELDOUT_P_COLUMNS`.
    alpha : float
        The threshold on q.

    Returns
    -------
    fraction : float
        Units with q below ``alpha``, as a share of the tested units. NaN, with
        a NumPy warning, when no unit is tested.

    Raises
    ------
    ValueError
        If the finite p-values are not the units
        :func:`~continuum_swr_replay.analysis.pfc.load_population.significant_fraction`
        counts as tested.
    """
    pvalues = population[HELDOUT_P_COLUMNS[name]].to_numpy(dtype=float)
    pvalues = pvalues[np.isfinite(pvalues)]
    if len(pvalues) != significant_fraction(population, alpha=alpha)[name][1]:
        raise ValueError(f"the {name} p-values are not the units it was tested on")
    return (false_discovery_control(pvalues, method="bh") < alpha).mean()


def best_model_fractions(
    population: pd.DataFrame,
    models: Sequence[str],
    column: str = "min_heldout_model",
) -> dict[str, float]:
    """The share of units whose best model is each of ``models``.

    Parameters
    ----------
    population : pd.DataFrame, shape (n_units, n_columns)
        The unit table, with each unit's best model by held-out deviance in
        ``column``.
    models : sequence of str
        The models to count.
    column : str
        The column naming each unit's best model.

    Returns
    -------
    fractions : dict
        By model: the units whose best model it is, as a share of the units
        with a best model (a unit with none is left out). NaN when no unit has
        a best model.
    """
    best = population[column]
    n_known = int(best.notna().sum())
    return {
        model: float((best == model).sum() / n_known) if n_known else np.nan
        for model in models
    }


def units_at_or_above_rate(
    population: pd.DataFrame,
    min_firing_rate: float,
    curves: Mapping[str, NDArray] | None = None,
) -> tuple[pd.DataFrame, dict[str, NDArray]]:
    """The units whose firing rate on the fitted bins reaches a cutoff.

    Parameters
    ----------
    population : pd.DataFrame, shape (n_units, n_columns)
        The unit table, with ``immobility_avg_firing_rate`` in Hz.
    min_firing_rate : float
        The cutoff, in Hz. A unit exactly at it is kept.
    curves : mapping of str to NDArray, each shape (n_lags, n_units), optional
        The units' curves, one column per row of ``population``.

    Returns
    -------
    kept : pd.DataFrame, shape (n_kept, n_columns)
        The rows of the kept units, with a fresh integer index.
    kept_curves : dict of str to NDArray, each shape (n_lags, n_kept)
        The kept units' columns of each curve; empty when no curves are given.
    """
    rate = population["immobility_avg_firing_rate"].to_numpy(dtype=float)
    keep = rate >= min_firing_rate
    kept_curves = {name: values[:, keep] for name, values in (curves or {}).items()}
    return population[keep].reset_index(drop=True), kept_curves


# --- the null's geometry ---------------------------------------------------------------


def called_without_improvement(population: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Units called for a test whose own observed improvement is not positive.

    A unit's p-value compares its observed held-out improvement on the
    matchable events
    (:data:`~continuum_swr_replay.analysis.pfc.load_population.MATCHED_NULL_OBSERVED_COLUMNS`)
    with its null draws, so a unit can be called while its observed
    improvement is zero or negative.

    Parameters
    ----------
    population : pd.DataFrame, shape (n_units, n_columns)
        The unit table.
    alpha : float
        The raw empirical-p threshold.

    Returns
    -------
    statistics : pd.DataFrame, shape (n_tests, 3)
        Indexed by the keys of
        :data:`~continuum_swr_replay.analysis.pfc.load_population.HELDOUT_P_COLUMNS`:
        ``n_called``, the units called; ``n_flat``, those of them whose
        observed improvement is at most zero (a called unit with no observed
        improvement recorded is not counted); and ``fraction``, ``n_flat`` as
        a share of ``n_called`` (NaN, with a NumPy warning, when no unit is
        called).
    """
    rows = {}
    for name, column in HELDOUT_P_COLUMNS.items():
        called = population[column] < alpha
        observed = population[MATCHED_NULL_OBSERVED_COLUMNS[name]]
        n_flat, n_called = (called & (observed <= 0)).sum(), called.sum()
        rows[name] = {
            "n_called": n_called,
            "n_flat": n_flat,
            "fraction": n_flat / n_called,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def session_value_statistics(values: Sequence[float]) -> dict[str, float]:
    """The median and range over sessions of one value per session.

    Parameters
    ----------
    values : sequence of float, length n_sessions
        One value per session (for example each session's median share of an
        engagement null draw's windows that land on the shipped control).

    Returns
    -------
    statistics : dict
        ``median``, ``min`` and ``max`` over sessions.
    """
    by_session = pd.Series(values)
    return {
        "median": by_session.median(),
        "min": by_session.min(),
        "max": by_session.max(),
    }


def unmatchable_event_statistics(windows: pd.DataFrame) -> dict[str, object]:
    """Events with no legal within-bout control, which enter neither side of a test.

    Parameters
    ----------
    windows : pd.DataFrame, shape (n_sessions * n_classes, 4)
        One row per session and event class: ``session``, ``event_class``, the
        class's events in the session (``total``) and those with no legal
        control (``excluded``).

    Returns
    -------
    statistics : dict
        - ``percent_class_min``, ``percent_class_max``: the lowest and highest,
          over event classes, of the percent of a class's events that are
          excluded, events summed over sessions; ``class_min`` and
          ``class_max`` name the classes.
        - ``worst_session``, ``worst_event_class``: the session and class with
          the largest excluded share, among those with an event.
        - ``worst_excluded``, ``worst_total``: its excluded events and its
          events.

    Raises
    ------
    ValueError
        If no session holds an event of any class.
    """
    by_class = windows.groupby("event_class")[["total", "excluded"]].sum()
    excluded = 100 * by_class["excluded"] / by_class["total"]
    with_events = windows[windows["total"] > 0]
    if with_events.empty:
        raise ValueError("no session holds an event of any class")
    worst = with_events.loc[(with_events["excluded"] / with_events["total"]).idxmax()]
    return {
        "percent_class_min": excluded.min(),
        "class_min": excluded.idxmin(),
        "percent_class_max": excluded.max(),
        "class_max": excluded.idxmax(),
        "worst_session": worst["session"],
        "worst_event_class": worst["event_class"],
        "worst_excluded": worst["excluded"],
        "worst_total": worst["total"],
    }


# --- SWR-coupled against non-SWR replay ---------------------------------------------


def call_overlap_statistics(
    first_called: NDArray,
    second_called: NDArray,
    first_peak: NDArray,
    second_peak: NDArray,
) -> dict[str, float]:
    """Whether the same units are called for two event classes.

    Over the units given, which the caller restricts to those tested for both
    classes.

    Parameters
    ----------
    first_called, second_called : NDArray of bool, shape (n_units,)
        Whether each unit is called for the first and for the second class.
    first_peak, second_peak : NDArray, shape (n_units,)
        Each unit's peak for the two classes; NaN where a unit has none.

    Returns
    -------
    statistics : dict
        - ``n_tested``: the units given.
        - ``n_both``, ``n_first_only``, ``n_second_only``, ``n_neither``: those
          called for both classes, for one only and for neither.
        - ``n_both_expected``: the units called for both if the two calls were
          independent: the product of the two called shares, times
          ``n_tested``. NaN when no unit is given.
        - ``odds_ratio``: ``n_both * n_neither / (n_first_only *
          n_second_only)``. NaN when either one-only count is zero: the ratio
          is then infinite or undefined.
        - ``rank_correlation``: the Spearman correlation of the two peaks over
          the units with both peaks. NaN with fewer than three such units.
        - ``same_sign_fraction_both``: of the units called for both, the share
          whose two peaks have the same sign (a unit missing a peak counts as
          not the same). NaN when no unit is called for both.
    """
    n = len(first_called)
    both = first_called & second_called
    n_both = int(both.sum())
    n_first_only = int((first_called & ~second_called).sum())
    n_second_only = int((~first_called & second_called).sum())
    n_neither = int((~first_called & ~second_called).sum())
    statistics: dict[str, float] = {
        "n_tested": n,
        "n_both": n_both,
        "n_first_only": n_first_only,
        "n_second_only": n_second_only,
        "n_neither": n_neither,
        "n_both_expected": np.nan,
        "odds_ratio": np.nan,
        "rank_correlation": np.nan,
        "same_sign_fraction_both": np.nan,
    }
    if n:
        statistics["n_both_expected"] = first_called.mean() * second_called.mean() * n
    if n_first_only and n_second_only:
        statistics["odds_ratio"] = (n_both * n_neither) / (n_first_only * n_second_only)
    with_peaks = np.isfinite(first_peak) & np.isfinite(second_peak)
    if with_peaks.sum() > 2:
        statistics["rank_correlation"] = stats.spearmanr(
            first_peak[with_peaks], second_peak[with_peaks]
        ).statistic
    if n_both:
        statistics["same_sign_fraction_both"] = float(
            (np.sign(first_peak[both]) == np.sign(second_peak[both])).mean()
        )
    return statistics


def replay_class_overlap_statistics(
    population: pd.DataFrame, alpha: float
) -> dict[str, float]:
    """Whether the same units are called for SWR-coupled and non-SWR replay.

    Over the units tested for both ``non_local_SWR`` (SWR-coupled replay) and
    ``non_local_no_SWR`` (non-SWR replay).

    Parameters
    ----------
    population : pd.DataFrame, shape (n_units, n_columns)
        The unit table, with ``nwb_file_name`` (the animal is read from it) and
        the two classes' p-values and peaks.
    alpha : float
        The raw empirical-p threshold.

    Returns
    -------
    statistics : dict
        - ``n_tested``: the units tested for both.
        - ``n_both``, ``n_swr_only``, ``n_non_swr_only``, ``n_neither``: those
          called for both classes, for one only and for neither.
        - ``n_both_expected``, ``odds_ratio``, ``rank_correlation``,
          ``same_sign_fraction_both``: as in :func:`call_overlap_statistics`,
          SWR-coupled replay first, pooled over the units tested for both.
        - ``odds_ratio_animal_min``, ``odds_ratio_animal_max``: the lowest and
          highest of the animals' own odds ratios, over the animals with a
          unit tested for both.
        - ``n_animals_odds_ratio_above_one``: animals whose own ratio is above 1.
        - ``rank_correlation_called``: the rank correlation over the units
          called for either class.
        - ``swr_larger_fraction_called``: of the units called for either, the
          share whose absolute peak is larger for SWR-coupled replay (a unit
          missing a peak counts as not larger).
        - ``abs_peak_median_called_swr``, ``abs_peak_median_called_non_swr``:
          the median absolute peak for a class over the units tested for both
          that are called for that class and have the peak.

    Raises
    ------
    ValueError
        If the odds ratio of the pooled units or of an animal is undefined:
        no unit is called for SWR-coupled replay only, or none for non-SWR
        replay only.
    """
    swr_tested, swr_called = _tested_and_called(
        population[HELDOUT_P_COLUMNS["non_local_SWR"]], alpha
    )
    non_tested, non_called = _tested_and_called(
        population[HELDOUT_P_COLUMNS["non_local_no_SWR"]], alpha
    )
    tested = swr_tested & non_tested
    frame = population[tested]
    swr_called, non_called = swr_called[tested], non_called[tested]
    swr_peak = frame["non_local_SWR: peak_log_modulation"].to_numpy(dtype=float)
    non_peak = frame["non_local_no_SWR: peak_log_modulation"].to_numpy(dtype=float)
    animals = (
        frame["nwb_file_name"].map(lambda name: parse_nwb_filename(name)[0]).to_numpy()
    )

    def overlap_of(units: NDArray) -> dict[str, float]:
        """The overlap statistics of some of the units tested for both."""
        return call_overlap_statistics(
            swr_called[units], non_called[units], swr_peak[units], non_peak[units]
        )

    pooled = overlap_of(np.ones(len(frame), dtype=bool))
    by_animal = {animal: overlap_of(animals == animal) for animal in np.unique(animals)}
    for what, overlap in [
        ("pooled units", pooled),
        *((f"animal {animal}", overlap) for animal, overlap in by_animal.items()),
    ]:
        if np.isnan(overlap["odds_ratio"]):
            raise ValueError(
                f"odds ratio undefined for {what}: a one-class-only cell is empty "
                f"(both {overlap['n_both']}, one-only {overlap['n_first_only']} "
                f"and {overlap['n_second_only']}, neither {overlap['n_neither']})"
            )
    animal_ratios = [overlap["odds_ratio"] for overlap in by_animal.values()]
    either = swr_called | non_called
    return {
        "n_tested": pooled["n_tested"],
        "n_both": pooled["n_both"],
        "n_swr_only": pooled["n_first_only"],
        "n_non_swr_only": pooled["n_second_only"],
        "n_neither": pooled["n_neither"],
        "n_both_expected": pooled["n_both_expected"],
        "odds_ratio": pooled["odds_ratio"],
        "odds_ratio_animal_min": min(animal_ratios),
        "odds_ratio_animal_max": max(animal_ratios),
        "n_animals_odds_ratio_above_one": sum(r > 1 for r in animal_ratios),
        "rank_correlation": pooled["rank_correlation"],
        "rank_correlation_called": overlap_of(either)["rank_correlation"],
        "same_sign_fraction_both": pooled["same_sign_fraction_both"],
        "swr_larger_fraction_called": float(
            (np.abs(swr_peak[either]) > np.abs(non_peak[either])).mean()
        ),
        "abs_peak_median_called_swr": np.nanmedian(np.abs(swr_peak[swr_called])),
        "abs_peak_median_called_non_swr": np.nanmedian(np.abs(non_peak[non_called])),
    }


def replay_split_statistics(population: pd.DataFrame) -> dict[str, float]:
    """How often splitting the non-local response by SWR status helps a unit.

    Parameters
    ----------
    population : pd.DataFrame, shape (n_units, n_columns)
        The unit table, with the held-out deviance improvement of the split
        model over the unsplit one (``non_local_SWR + non_local_no_SWR vs.
        non_local: heldout_delta_dev``) and each unit's best model among
        constant, non-local and the split (``min_heldout_model_replay_split``).

    Returns
    -------
    statistics : dict
        - ``improves_fraction``: of the units with an improvement, the share
          whose improvement is positive.
        - ``best_model_fraction``: of the units with a best model, the share
          whose best model is the split (:func:`best_model_fractions`).
    """
    split_model = "non_local_SWR + non_local_no_SWR"
    split = population[f"{split_model} vs. non_local: heldout_delta_dev"]
    return {
        "improves_fraction": float((split.dropna() > 0).mean()),
        "best_model_fraction": best_model_fractions(
            population, [split_model], column="min_heldout_model_replay_split"
        )[split_model],
    }


def peak_modulation_statistics(
    population: pd.DataFrame, curves: Sequence[str]
) -> pd.DataFrame:
    """The typical size and direction of the units' peaks, per curve.

    Parameters
    ----------
    population : pd.DataFrame, shape (n_units, n_columns)
        The unit table, with ``<curve>: peak_log_modulation`` for every curve.
        A unit with no peak for a curve is left out of that curve's statistics.
    curves : sequence of str
        The curves (``"SWR"``, ``"matched_control"``, ...).

    Returns
    -------
    statistics : pd.DataFrame, shape (n_curves, 3)
        Indexed by curve:

        - ``abs_peak_median``: the median over units of the absolute peak, on
          the log-rate scale.
        - ``rate_multiplier_median``: ``exp`` of that median, the median unit's
          largest rate change as a multiplier.
        - ``suppressed_fraction``: the share of units whose peak is a decrease.
    """
    rows = {}
    for curve in curves:
        peak = population[f"{curve}: peak_log_modulation"].dropna()
        rows[curve] = {
            "abs_peak_median": peak.abs().median(),
            "rate_multiplier_median": float(np.exp(peak.abs().median())),
            "suppressed_fraction": float((peak < 0).mean()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


# --- SWRs without replay ---------------------------------------------------------------


def _session_event_counts(sessions: pd.DataFrame) -> pd.Series:
    """Each session's SWRs without replay, indexed by ``(nwb_file_name,
    epoch_name)``.

    Raises
    ------
    ValueError
        If a session is listed more than once, or has no count.
    """
    n_events = sessions.set_index(["nwb_file_name", "epoch_name"])["n_swr_no_non_local"]
    repeated = n_events.index[n_events.index.duplicated()]
    if len(repeated):
        raise ValueError(f"sessions listed more than once: {sorted(set(repeated))}")
    if n_events.isna().any():
        raise ValueError(
            "sessions with no count of SWRs without replay: "
            f"{sorted(n_events.index[n_events.isna()])}"
        )
    return n_events


def swr_without_replay_event_statistics(
    sessions: pd.DataFrame, min_events: int
) -> dict[str, float]:
    """How many kept SWRs overlap no non-local event, over sessions.

    Parameters
    ----------
    sessions : pd.DataFrame, shape (n_sessions, 5)
        One row per session: ``nwb_file_name``, ``epoch_name``, its kept SWRs
        in the stop cohort that overlap no non-local event
        (``n_swr_no_non_local``), those that overlap one
        (``n_swr_with_non_local``), and those left out of both because they
        overlap a window the analysed-time rule dropped
        (``n_swr_no_non_local_unknown``).
    min_events : int
        The floor a session's SWRs without replay are compared with.

    Returns
    -------
    statistics : dict
        - ``n_events``: SWRs without replay, summed over sessions.
        - ``events_fraction``: ``n_events`` as a share of the three counts
          together, summed over sessions.
        - ``n_unknown``: the left-out SWRs, summed over sessions.
        - ``events_session_median``: the median per session of the SWRs
          without replay.
        - ``n_sessions_min_events``: sessions with at least ``min_events`` of
          them.

    Raises
    ------
    ValueError
        If a session is listed more than once, or lacks one of the three
        counts.
    """
    n_events = _session_event_counts(sessions)
    others = sessions[["n_swr_with_non_local", "n_swr_no_non_local_unknown"]]
    if others.isna().any(axis=None):
        raise ValueError(
            "a session has no count of its SWRs with replay or of those left out"
        )
    total = int(n_events.sum())
    with_replay = int(others["n_swr_with_non_local"].sum())
    unknown = int(others["n_swr_no_non_local_unknown"].sum())
    return {
        "n_events": total,
        "events_fraction": total / (total + with_replay + unknown),
        "n_unknown": unknown,
        "events_session_median": float(n_events.median()),
        "n_sessions_min_events": int((n_events >= min_events).sum()),
    }


def _swr_without_replay_subsets(
    population: pd.DataFrame, sessions: pd.DataFrame, min_events: int
) -> dict[str, NDArray]:
    """The units of each group of sessions: every session (``"all"``), and the
    sessions with at least ``min_events`` SWRs without replay
    (``"min_events"``). A boolean mask over units for each."""
    n_events = _session_event_counts(sessions).to_dict()
    session_events = np.array(
        [
            n_events[(name, epoch)]
            for name, epoch in zip(
                population["nwb_file_name"], population["epoch_name"], strict=True
            )
        ]
    )
    return {
        "all": np.ones(len(population), dtype=bool),
        "min_events": session_events >= min_events,
    }


def swr_without_replay_unit_statistics(
    population: pd.DataFrame,
    sessions: pd.DataFrame,
    *,
    alpha: float,
    min_events: int,
    three_class_model: str,
) -> pd.DataFrame:
    """How units respond to SWRs without replay, against SWR-coupled replay.

    Read from the extended event set, in which SWRs that overlap no non-local
    event (``SWR_no_non_local``) have their own matched null and are fit beside
    the two replay classes in one three-class model. Both compared peaks come
    from that model: ``SWR_no_non_local`` and ``non_local_SWR_3class``.

    Parameters
    ----------
    population : pd.DataFrame, shape (n_units, n_columns)
        The extended unit table: ``nwb_file_name``, ``epoch_name``, the
        p-values of ``SWR_no_non_local`` and ``non_local_SWR``, the two peaks,
        the held-out deviance improvement of the three-class model over the
        replay split (``<three_class_model> vs. non_local_SWR +
        non_local_no_SWR: heldout_delta_dev``) and each unit's best model
        (``min_heldout_model_three_class``).
    sessions : pd.DataFrame, shape (n_sessions, n_columns)
        One row per session, with ``nwb_file_name``, ``epoch_name`` and
        ``n_swr_no_non_local`` (:func:`swr_without_replay_event_statistics`).
    alpha : float
        The raw empirical-p threshold.
    min_events : int
        The floor on a session's SWRs without replay for the second group.
    three_class_model : str
        The three-class model's name.

    Returns
    -------
    statistics : pd.DataFrame, shape (2, n_statistics)
        Indexed by group of sessions (``"all"``: every session;
        ``"min_events"``: sessions with at least ``min_events`` SWRs without
        replay). A count over an empty set of units is 0; any other statistic
        over an empty set is NaN.

        - ``n_units``: the group's units; ``n_tested``: those tested for SWRs
          without replay; ``n_called``: those called for them.
        - ``called_fraction``: ``n_called`` as a share of ``n_tested``.
        - ``n_overlap_tested``: units tested for both SWRs without replay and
          SWR-coupled replay. Over those units, with SWRs without replay the
          first class (:func:`call_overlap_statistics`): ``n_overlap_both``,
          ``n_overlap_swr_no_replay_only``, ``n_overlap_swr_replay_only`` and
          ``n_overlap_neither`` (its four counts), ``n_overlap_expected`` (its
          ``n_both_expected``), ``overlap_odds_ratio`` (its ``odds_ratio``),
          ``rank_correlation`` and ``same_sign_fraction`` (its
          ``same_sign_fraction_both``).
        - ``abs_peak_median_swr_no_replay``, ``abs_peak_median_swr_replay``:
          the median of each absolute peak over the group's units with that
          peak; ``rate_multiplier_median_*``: ``exp`` of it;
          ``abs_peak_median_called_*``: the same median over the units called
          for SWRs without replay (for both peaks).
        - ``adds_to_split_fraction``: of the units with an improvement of the
          three-class model over the replay split, the share whose
          improvement is positive.
        - ``best_model_fraction``: of the units with a best model, the share
          whose best model is the three-class model
          (:func:`best_model_fractions`).

    Raises
    ------
    KeyError
        If a unit's session is not in ``sessions``.
    ValueError
        If a session is listed more than once in ``sessions``, or has no count
        of SWRs without replay.
    """
    subsets = _swr_without_replay_subsets(population, sessions, min_events)
    new_tested, new_called = _tested_and_called(
        population[SWR_WITHOUT_REPLAY_P_COLUMN], alpha
    )
    swr_tested, swr_called = _tested_and_called(
        population[HELDOUT_P_COLUMNS["non_local_SWR"]], alpha
    )
    new_peak = population["SWR_no_non_local: peak_log_modulation"].to_numpy(dtype=float)
    swr_peak = population["non_local_SWR_3class: peak_log_modulation"].to_numpy(
        dtype=float
    )
    improves = population[
        f"{three_class_model} vs. non_local_SWR + non_local_no_SWR: heldout_delta_dev"
    ].to_numpy(dtype=float)

    columns = [
        "n_units",
        "n_tested",
        "called_fraction",
        "n_called",
        "n_overlap_tested",
        "n_overlap_both",
        "n_overlap_swr_no_replay_only",
        "n_overlap_swr_replay_only",
        "n_overlap_neither",
        "n_overlap_expected",
        "overlap_odds_ratio",
        "rank_correlation",
        "same_sign_fraction",
        "abs_peak_median_swr_no_replay",
        "rate_multiplier_median_swr_no_replay",
        "abs_peak_median_called_swr_no_replay",
        "abs_peak_median_swr_replay",
        "rate_multiplier_median_swr_replay",
        "abs_peak_median_called_swr_replay",
        "adds_to_split_fraction",
        "best_model_fraction",
    ]
    rows = {}
    for subset, keep in subsets.items():
        row: dict[str, float] = dict.fromkeys(columns, np.nan)
        tested, called = keep & new_tested, keep & new_called
        row["n_units"] = int(keep.sum())
        row["n_tested"] = int(tested.sum())
        row["n_called"] = int(called.sum())
        if tested.any():
            row["called_fraction"] = called.sum() / tested.sum()
        both_tested = tested & swr_tested
        overlap = call_overlap_statistics(
            called[both_tested],
            swr_called[both_tested],
            new_peak[both_tested],
            swr_peak[both_tested],
        )
        row["n_overlap_tested"] = overlap["n_tested"]
        row["n_overlap_both"] = overlap["n_both"]
        row["n_overlap_swr_no_replay_only"] = overlap["n_first_only"]
        row["n_overlap_swr_replay_only"] = overlap["n_second_only"]
        row["n_overlap_neither"] = overlap["n_neither"]
        row["n_overlap_expected"] = overlap["n_both_expected"]
        row["overlap_odds_ratio"] = overlap["odds_ratio"]
        row["rank_correlation"] = overlap["rank_correlation"]
        row["same_sign_fraction"] = overlap["same_sign_fraction_both"]
        for word, peak in (("swr_no_replay", new_peak), ("swr_replay", swr_peak)):
            finite = keep & np.isfinite(peak)
            if not finite.any():
                continue
            median_abs = float(np.median(np.abs(peak[finite])))
            row[f"abs_peak_median_{word}"] = median_abs
            row[f"rate_multiplier_median_{word}"] = float(np.exp(median_abs))
            called_peak = peak[called & np.isfinite(peak)]
            if called_peak.size:
                row[f"abs_peak_median_called_{word}"] = float(
                    np.median(np.abs(called_peak))
                )
        improves_finite = keep & np.isfinite(improves)
        if improves_finite.any():
            row["adds_to_split_fraction"] = float(
                (improves[improves_finite] > 0).mean()
            )
        row["best_model_fraction"] = best_model_fractions(
            population[keep],
            [three_class_model],
            column="min_heldout_model_three_class",
        )[three_class_model]
        rows[subset] = row
    return pd.DataFrame.from_dict(rows, orient="index", columns=columns)


def swr_without_replay_called_by_animal(
    population: pd.DataFrame,
    sessions: pd.DataFrame,
    *,
    alpha: float,
    min_events: int,
) -> pd.DataFrame:
    """The share of each animal's tested units called for SWRs without replay.

    Parameters
    ----------
    population : pd.DataFrame, shape (n_units, n_columns)
        The extended unit table, with ``animal``, ``nwb_file_name``,
        ``epoch_name`` and ``SWR_no_non_local: p_heldout``.
    sessions : pd.DataFrame, shape (n_sessions, n_columns)
        One row per session (:func:`swr_without_replay_unit_statistics`).
    alpha : float
        The raw empirical-p threshold.
    min_events : int
        The floor on a session's SWRs without replay for the second group.

    Returns
    -------
    fractions : pd.DataFrame, shape (n_groups_with_animals, 1)
        Indexed by group of sessions (``subset``: ``"all"`` or
        ``"min_events"``) and ``animal``, for the animals with a tested unit in
        the group: ``called_fraction``, the animal's called units as a share of
        its tested units.
    """
    subsets = _swr_without_replay_subsets(population, sessions, min_events)
    new_tested, new_called = _tested_and_called(
        population[SWR_WITHOUT_REPLAY_P_COLUMN], alpha
    )
    animals = population["animal"].to_numpy()
    index, fractions = [], []
    for subset, keep in subsets.items():
        tested = keep & new_tested
        for animal in np.unique(animals[tested]):
            own = tested & (animals == animal)
            index.append((subset, animal))
            fractions.append(new_called[own].sum() / own.sum())
    return pd.DataFrame(
        {"called_fraction": fractions},
        index=pd.MultiIndex.from_arrays(
            [[s for s, _ in index], [a for _, a in index]], names=["subset", "animal"]
        ),
    )


def swr_without_replay_timing_statistics(
    population: pd.DataFrame,
    curves: NDArray,
    lag_time: NDArray,
    sessions: pd.DataFrame,
    *,
    alpha: float,
    min_events: int,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """When the response to SWRs without replay peaks and starts to rise.

    Over the units called for SWRs without replay, split by the sign of each
    unit's peak, the rise timing of their sign-aligned mean curve
    (:func:`~continuum_swr_replay.analysis.pfc.timing.population_rise_timing`),
    resampling recording days (``nwb_file_name``).

    Parameters
    ----------
    population : pd.DataFrame, shape (n_units, n_columns)
        The extended unit table, with ``nwb_file_name``, ``epoch_name`` and
        ``SWR_no_non_local: p_heldout``.
    curves : NDArray, shape (n_lags, n_units)
        Each unit's curve for SWRs without replay.
    lag_time : NDArray, shape (n_lags,)
        Lag of each row, in seconds.
    sessions : pd.DataFrame, shape (n_sessions, n_columns)
        One row per session (:func:`swr_without_replay_unit_statistics`).
    alpha : float
        The raw empirical-p threshold.
    min_events : int
        The floor on a session's SWRs without replay for the second group.
    n_boot, seed : int
        Recording-day resamples, and their seed.

    Returns
    -------
    statistics : pd.DataFrame, shape (4, 8)
        Indexed by group of sessions (``subset``) and ``direction``
        (``"enhanced"``: the peak is an increase; ``"suppressed"``: a
        decrease):

        - ``n_units``: the called units of that direction. A called unit whose
          curve is not finite or is zero everywhere has no peak, and is in
          neither direction.
        - ``peak_ms``, ``rise_onset_ms``, ``rise_onset_ms_ci_low``,
          ``rise_onset_ms_ci_high``, ``rise_size``, ``rise_size_ci_low``,
          ``rise_size_ci_high``: the statistics of
          :func:`~continuum_swr_replay.analysis.pfc.timing.population_rise_timing`
          for those units (lags in ms, sizes on the log-rate scale), NaN when
          the direction has no unit.
    """
    subsets = _swr_without_replay_subsets(population, sessions, min_events)
    _, new_called = _tested_and_called(population[SWR_WITHOUT_REPLAY_P_COLUMN], alpha)
    days = population["nwb_file_name"].astype(str).to_numpy()
    peak = peak_and_onset(curves, lag_time).peak
    rows = {}
    for subset, keep in subsets.items():
        called = keep & new_called
        for direction, sign in DIRECTIONS.items():
            chosen = called & (np.sign(peak) == sign)
            rows[(subset, direction)] = {
                "n_units": int(chosen.sum()),
                **population_rise_timing(
                    curves[:, chosen],
                    peak[chosen],
                    days[chosen],
                    lag_time,
                    n_boot=n_boot,
                    seed=seed,
                ),
            }
    statistics = pd.DataFrame.from_dict(rows, orient="index")
    statistics.index.names = ["subset", "direction"]
    return statistics
