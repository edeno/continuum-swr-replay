"""The mpfc summary's unit-population values (question 4.1, Figure 4c and
4d): the cohort, the estimator and its fits, the prior binding, the called
fractions, the null's geometry, and the overlap and size of the responses.
The statistics are computed in ``analysis.pfc.population_statistics``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pandas as pd

from continuum_swr_replay.analysis.behavior.coefficient_statistics import group_range
from continuum_swr_replay.analysis.pfc import population_statistics
from continuum_swr_replay.analysis.pfc.heldout import empirical_pvalue_floor
from continuum_swr_replay.analysis.pfc.load_population import (
    CALLED_ALPHA as ALPHA,
)
from continuum_swr_replay.analysis.pfc.population_statistics import (
    BEST_MODELS,
    EFFECT_CURVES,
)
from continuum_swr_replay.reported.mpfc.common import (
    ANIMAL_WORDS,
    CLASS_LABELS,
    TAGS,
)
from continuum_swr_replay.reported.mpfc.inputs import (
    STEM,
)
from continuum_swr_replay.reported.values import (
    Value,
)

#: The unit tests Figure 4 reports, by the word their keys use.
CLASSES = {
    "swr": "SWR",
    "nonlocal": "non_local",
    "non_swr_replay": "non_local_no_SWR",
    "swr_replay": "non_local_SWR",
    "engagement": "engagement",
}


#: The curves whose prior binding the audit counts (compare_pfc_cohorts.CURVE_NAMES).
CURVES = ("SWR", "no_SWR", "non_local_SWR", "non_local_no_SWR", "matched_control")


def _uniform(sidecars: list[dict[str, Any]], name: str) -> Any:
    """The one value every sidecar records for ``name``."""
    values = {json.dumps(s.get(name), sort_keys=True) for s in sidecars}
    if len(values) != 1:
        raise ValueError(f"the bundles disagree on {name}: {sorted(values)}")
    return sidecars[0].get(name)


# --- the cohort -----------------------------------------------------------------


def cohort_values(
    sidecars: list[dict[str, Any]],
    population: pd.DataFrame,
    cohort: dict[str, Any],
    make: Callable[..., Value],
) -> list[Value]:
    """Which sessions, units, bins and events the unit GLM was fit on.

    Bins and events are summed over sessions, before and after the at-well
    stop restriction; ``compare_pfc_cohorts.session_cohort_table`` shows the
    same totals.
    """
    if len(sidecars) != cohort["n_pfc_cohort"]:
        raise ValueError(
            f"{len(sidecars)} bundles, but the PFC cohort holds "
            f"{cohort['n_pfc_cohort']} epochs"
        )
    if _uniform(sidecars, "stop_cohort") != "at_well":
        raise ValueError("the canonical bundles are not the at-well cohort")
    stats = population_statistics.cohort_count_statistics(
        pd.DataFrame([s["cohort_counts"] for s in sidecars])
    )
    return [
        make(
            "sessions",
            len(sidecars),
            "count",
            "count",
            "epochs with an mPFC unit bundle (the PFC cohort: analysed epochs with "
            "curated cortical spike sorting)",
        ),
        make(
            "analysed_epochs",
            cohort["n_analysed"],
            "count",
            "count",
            "analysed epochs the PFC cohort is drawn from",
        ),
        make(
            "animals",
            population["animal"].nunique(),
            "count",
            "count",
            "animals in the mPFC population",
        ),
        make(
            "units",
            len(population),
            "count",
            "count",
            "mPFC units in the population: every unit with a usable fit (no "
            "firing-rate cutoff; a unit with no spike in the fitted bins has none) "
            "and finite modulation curves",
            tags=TAGS["units"],
        ),
        make(
            "bins_fit",
            stats["n_bins_fit"],
            "count",
            "count",
            "immobile 2 ms bins inside a kept stop, the bins the GLM is fit on, "
            "summed over sessions",
        ),
        make(
            "bins_immobile",
            stats["n_bins_immobile"],
            "count",
            "count",
            "immobile 2 ms bins before the stop restriction, summed over sessions",
        ),
        make(
            "bins_fit_fraction",
            stats["bins_fit_fraction"],
            "fraction",
            "pct",
            "share of the immobile bins the stop restriction keeps, pooled",
        ),
        make(
            "nonlocal_events_kept",
            stats["n_nonlocal_kept"],
            "count",
            "count",
            "non-local events whose onset is in a kept stop, summed over sessions",
        ),
        make(
            "nonlocal_events_before",
            stats["n_nonlocal_before"],
            "count",
            "count",
            "the session's non-local events before the stop restriction, summed",
        ),
        make(
            "nonlocal_events_kept_fraction",
            stats["nonlocal_kept_fraction"],
            "fraction",
            "pct",
            "share of the non-local events the stop restriction keeps, pooled",
        ),
        make(
            "swrs_kept_fraction",
            stats["swrs_kept_fraction"],
            "fraction",
            "pct",
            "share of the SWRs the stop restriction keeps, pooled",
        ),
        make(
            "swr_coupled_share_session_median",
            stats["swr_coupled_share_session_median"],
            "fraction",
            "dec2",
            "median over sessions of the share of kept non-local events that are "
            "SWR-coupled (the canonical event table's is_swr)",
        ),
    ]


# --- the estimator and its fits ------------------------------------------------


def estimator_values(
    sidecars: list[dict[str, Any]], make: Callable[..., Value]
) -> list[Value]:
    """The prior, folds and null the bundles state, and what the fits did.

    Raises
    ------
    ValueError
        The bundles disagree on a setting, or were not fit with the ridge
        estimator and the held-out inference.
    """
    if (_uniform(sidecars, "estimator"), _uniform(sidecars, "inference")) != (
        "ridge_poisson_glm",
        "blocked_heldout_deviance",
    ):
        raise ValueError("the bundles are not the ridge, held-out bundles")
    if _uniform(sidecars, "empirical_pvalue_method") != "plus_one":
        raise ValueError("the bundles' p-values are not plus-one corrected")
    prior = _uniform(sidecars, "implied_prior_sd")
    draws = _uniform(sidecars, "n_control_draws")
    floor = empirical_pvalue_floor(draws)
    fits = population_statistics.fit_count_statistics(
        pd.DataFrame(
            [
                {
                    name: s[name]
                    for name in (
                        "n_model_fits",
                        "n_nonconverged_model_fits",
                        "n_neurons_with_nonconverged_model",
                    )
                }
                for s in sidecars
            ]
        )
    )
    return [
        make(
            "penalty_sigma",
            _uniform(sidecars, "penalty_sigma_log_rate"),
            "quantity",
            "int",
            "prior standard deviation of an event-basis coefficient",
            "log rate",
        ),
        make(
            "prior_sd_per_lag",
            prior["per_lag"],
            "quantity",
            "dec2",
            "the prior's implied standard deviation of a curve at one lag",
            "log rate",
        ),
        make(
            "prior_multiplier_per_lag_low",
            prior["per_lag_multiplier_low"],
            "quantity",
            "sig2",
            "lower end of the rate multiplier that per-lag sd allows at 95 %",
            "rate multiplier",
        ),
        make(
            "prior_multiplier_per_lag_high",
            prior["per_lag_multiplier_high"],
            "quantity",
            "sig2",
            "upper end of that multiplier",
            "rate multiplier",
        ),
        make(
            "prior_sd_sustained",
            prior["sustained"],
            "quantity",
            "dec2",
            "the prior's implied standard deviation of a curve sustained over the "
            "reported window",
            "log rate",
        ),
        make(
            "prior_multiplier_sustained_low",
            prior["sustained_multiplier_low"],
            "quantity",
            "sig2",
            "lower end of the rate multiplier that sustained sd allows at 95 %",
            "rate multiplier",
        ),
        make(
            "prior_multiplier_sustained_high",
            prior["sustained_multiplier_high"],
            "quantity",
            "sig2",
            "upper end of that multiplier",
            "rate multiplier",
        ),
        make(
            "folds",
            _uniform(sidecars, "n_folds"),
            "count",
            "count",
            "blocked folds of whole immobility runs the held-out deviance is scored on",
        ),
        make(
            "null_draws",
            draws,
            "count",
            "count",
            "matched-control draws in each unit's null",
        ),
        make(
            "min_pvalue_denominator",
            floor["denominator"],
            "count",
            "count",
            "R + 1: the smallest plus-one empirical p-value is one over it",
        ),
        make(
            "min_pvalue",
            floor["min_pvalue"],
            "fraction",
            "sig2",
            "the smallest plus-one empirical p-value, 1/(R + 1)",
        ),
        make(
            "model_fits",
            fits["n_model_fits"],
            "count",
            "count",
            "neuron x model fits in the bundles",
        ),
        make(
            "nonconverged_model_fits",
            fits["n_nonconverged_model_fits"],
            "count",
            "count",
            "of those, fits that ran to maxiter without converging",
        ),
        make(
            "nonconverged_neurons",
            fits["n_neurons_with_nonconverged_model"],
            "count",
            "count",
            "neurons carrying at least one non-converged fit",
        ),
        make(
            "nonconverged_percent",
            fits["nonconverged_percent"],
            "quantity",
            "sig2",
            "non-converged fits as a percent of model fits",
            "%",
        ),
    ]


def prior_binding_values(
    population: pd.DataFrame, make: Callable[..., Value]
) -> list[Value]:
    """How often the prior moves a curve's peak, per curve.

    Among the well-posed
    units of a curve (unpenalised fit converged and no separated column), the
    share whose penalised peak differs from the unpenalised one by more than
    that peak's standard error (the bundle's ``prior_binds``). The values are
    the lowest and highest share over the five curves;
    ``compare_pfc_cohorts.shrinkage_table`` shows each curve's share.
    """
    # A curve with no well-posed unit has no share and is left out.
    shares = group_range(
        population_statistics.prior_binding_percent(population, CURVES)
        .dropna()
        .to_dict()
    )
    return [
        make(
            "prior_binds_percent_min",
            shares["low"],
            "quantity",
            "sig2",
            f"lowest per-curve percent of well-posed units the prior binds on "
            f"({shares['low_group']})",
            "%",
        ),
        make(
            "prior_binds_percent_max",
            shares["high"],
            "quantity",
            "sig2",
            f"highest per-curve percent of well-posed units the prior binds on "
            f"({shares['high_group']})",
            "%",
        ),
    ]


# --- the unit fractions (Figure 4d) -----------------------------------------------


def called_fraction_values(
    population: pd.DataFrame, make: Callable[..., Value]
) -> list[Value]:
    """Figure 4d: the share of units at raw empirical p below ``ALPHA``, per test.

    The denominator is the units tested for that class
    (:func:`~continuum_swr_replay.analysis.pfc.load_population.significant_fraction`).
    Engagement is also given per animal.
    """
    fractions = population_statistics.called_fractions(population, ALPHA)
    values = []
    for word, name in CLASSES.items():
        stats = fractions.loc[name]
        label = CLASS_LABELS[name]
        values += [
            make(
                f"{word}_called_fraction",
                stats["fraction"],
                "fraction",
                "pct",
                f"share of tested units at raw empirical p < {ALPHA:g} for {label}, "
                "pooled (descriptive)",
            ),
            make(
                f"{word}_units_called",
                stats["n_called"],
                "count",
                "count",
                f"units at raw empirical p < {ALPHA:g} for {label}",
            ),
            make(
                f"{word}_units_tested",
                stats["n_tested"],
                "count",
                "count",
                f"units with a finite p-value for {label}",
            ),
        ]
    by_animal = population_statistics.called_fraction_by_animal(
        population, "engagement", ALPHA
    )
    for animal, stats in by_animal.iterrows():
        values.append(
            make(
                f"engagement_called_fraction_{ANIMAL_WORDS[animal]}",
                stats["fraction"],
                "fraction",
                "pct",
                f"share of {animal}'s tested units at raw empirical p < {ALPHA:g} for "
                "the engagement test: the mean over its sessions of each session's "
                "share",
            )
        )
    values.append(
        make(
            "engagement_bh_fraction",
            population_statistics.bh_called_fraction(population, "engagement", ALPHA),
            "fraction",
            "pct",
            "share of the tested units with a Benjamini-Hochberg q below "
            f"{ALPHA:g} for the engagement test, BH applied once over the pooled "
            "plus-one empirical p-values of every tested unit "
            "(analysis.shared_statistics.benjamini_hochberg). For reference only: the units "
            "are not independent, and the p-values sit on a 1/(R + 1) grid",
            tags=TAGS["modulation"],
        )
    )
    best = population_statistics.best_model_fractions(
        population, list(BEST_MODELS.values())
    )
    for word, model in BEST_MODELS.items():
        values.append(
            make(
                f"best_model_{word}_fraction",
                best[model],
                "fraction",
                "pct",
                f"share of units whose best model by held-out deviance is {model}",
                tags=TAGS["figure_4d"],
            )
        )
    return values


# --- the null's geometry -----------------------------------------------------------


def null_geometry_values(
    population: pd.DataFrame,
    sidecars: list[dict[str, Any]],
    make: Callable[..., Value],
) -> list[Value]:
    """Three caveats on the empirical p-values, which
    ``scripts/pfc/measure_pfc_null_geometry.py`` reports per session.

    - Units called at raw p below ``ALPHA`` whose own observed held-out improvement (on
      the matchable subset) is not positive.
    - How much of the engagement null's windows land on the shipped control:
      each session's median over draws, then the median and range over
      sessions.
    - Events with no legal within-bout control, which enter neither side:
      per class, summed over sessions.
    """
    flat = population_statistics.called_without_improvement(population, ALPHA)
    engagement = flat.loc["engagement"]
    # A class with no called unit has no share and is left out.
    class_shares = group_range(flat["fraction"].drop("engagement").dropna().to_dict())
    overlap = population_statistics.session_value_statistics(
        [s["engagement_null_overlap"]["median"] for s in sidecars]
    )
    unmatchable = population_statistics.unmatchable_event_statistics(
        pd.DataFrame(
            [
                {
                    "session": s["artifact"].removesuffix(f"{STEM}.pkl"),
                    "event_class": event_class,
                    "total": w["n_events_total"],
                    "excluded": w["n_events_excluded_unmatchable"],
                }
                for s in sidecars
                for event_class, w in s["null_windows"].items()
            ]
        )
    )
    worst_label = f"{unmatchable['worst_session']} / {unmatchable['worst_event_class']}"
    return [
        make(
            "engagement_called_flat_fraction",
            engagement["fraction"],
            "fraction",
            "pct",
            "of the units called by the engagement test, the share whose own "
            "observed held-out improvement is not positive",
        ),
        make(
            "engagement_called_flat",
            engagement["n_flat"],
            "count",
            "count",
            "units called by the engagement test with no positive observed improvement",
        ),
        make(
            "class_called_flat_fraction_min",
            class_shares["low"],
            "fraction",
            "pctnum",
            f"lowest such share over the four event classes ({class_shares['low_group']})",
        ),
        make(
            "class_called_flat_fraction_max",
            class_shares["high"],
            "fraction",
            "pct",
            f"highest such share over the four event classes ({class_shares['high_group']})",
        ),
        make(
            "engagement_null_overlap_median",
            overlap["median"],
            "fraction",
            "pct",
            "median over sessions of each session's median share of an engagement "
            "null draw's windows that land on the shipped matched control",
        ),
        make(
            "engagement_null_overlap_session_min",
            overlap["min"],
            "fraction",
            "pctnum",
            "lowest session median of that share",
        ),
        make(
            "engagement_null_overlap_session_max",
            overlap["max"],
            "fraction",
            "pct",
            "highest session median of that share",
        ),
        make(
            "unmatchable_percent_class_min",
            unmatchable["percent_class_min"],
            "quantity",
            "sig2",
            "lowest percent, over event classes, of a class's events with no legal "
            f"within-bout control, summed over sessions ({unmatchable['class_min']})",
            "%",
        ),
        make(
            "unmatchable_percent_class_max",
            unmatchable["percent_class_max"],
            "quantity",
            "sig2",
            f"highest such percent ({unmatchable['class_max']})",
            "%",
        ),
        make(
            "unmatchable_worst_excluded",
            unmatchable["worst_excluded"],
            "count",
            "count",
            f"events with no legal control in the worst session and class ({worst_label})",
        ),
        make(
            "unmatchable_worst_total",
            unmatchable["worst_total"],
            "count",
            "count",
            f"events of that class in that session ({worst_label})",
        ),
    ]


# --- the summary -------------------------------------------------------------------


def overlap_and_effect_values(population: pd.DataFrame, make) -> list[Value]:
    """Do the same mPFC units respond to SWR-coupled and non-SWR replay, how much,
    and do they respond alike?

    - **Overlap:** units called (raw empirical p below ``ALPHA``) for SWR-coupled replay
      (``non_local_SWR``), for non-SWR replay (``non_local_no_SWR``), for both or
      neither, among units tested for both; the count of both expected if the two
      calls were independent; their odds ratio, pooled and per animal.
    - **Effect size:** each unit's ``peak_log_modulation`` (the ridge fit's
      largest log-rate change over the reported window) for each class, as a
      median of its absolute value; the matched quiet-rest control's is the
      noise floor. ``exp`` of a log modulation is a rate multiplier.
    - **Alike:** across units, the rank correlation of the two classes' peak
      modulations, and among units called for both the share with the same sign;
      the share of units whose held-out deviance improves when the non-local
      response is split by SWR status (``non_local_SWR + non_local_no_SWR vs.
      non_local``), and whose best model by held-out deviance is that split.
    """
    overlap = population_statistics.replay_class_overlap_statistics(population, ALPHA)
    split = population_statistics.replay_split_statistics(population)
    values = [
        make(
            "overlap_units_tested",
            overlap["n_tested"],
            "count",
            "count",
            "units with a finite p-value for both SWR-coupled and non-SWR replay",
        ),
        make(
            "overlap_units_both",
            overlap["n_both"],
            "count",
            "count",
            f"of those, units called (raw p < {ALPHA:g}) for both",
        ),
        make(
            "overlap_units_swr_replay_only",
            overlap["n_swr_only"],
            "count",
            "count",
            "units called for SWR-coupled replay and not for non-SWR replay",
        ),
        make(
            "overlap_units_non_swr_replay_only",
            overlap["n_non_swr_only"],
            "count",
            "count",
            "units called for non-SWR replay and not for SWR-coupled replay",
        ),
        make(
            "overlap_units_neither",
            overlap["n_neither"],
            "count",
            "count",
            "units called for neither",
        ),
        make(
            "overlap_units_both_expected",
            overlap["n_both_expected"],
            "quantity",
            "int",
            "units called for both expected if the two calls were independent "
            "(product of the two called shares times the units tested)",
            "units",
        ),
        make(
            "overlap_odds_ratio",
            overlap["odds_ratio"],
            "quantity",
            "sig2",
            "odds ratio of being called for non-SWR replay given called for SWR-coupled "
            "replay, pooled over units (1 = independent)",
        ),
        make(
            "overlap_odds_ratio_animal_min",
            overlap["odds_ratio_animal_min"],
            "quantity",
            "sig2",
            "lowest per-animal odds ratio",
        ),
        make(
            "overlap_odds_ratio_animal_max",
            overlap["odds_ratio_animal_max"],
            "quantity",
            "sig2",
            "highest per-animal odds ratio",
        ),
        make(
            "overlap_animals_odds_ratio_above_one",
            overlap["n_animals_odds_ratio_above_one"],
            "count",
            "count",
            "animals whose odds ratio is above 1",
        ),
        make(
            "overlap_rank_correlation",
            overlap["rank_correlation"],
            "quantity",
            "dec2",
            "Spearman correlation over units tested for both of the peak log modulation "
            "for SWR-coupled replay and for non-SWR replay",
        ),
        make(
            "overlap_rank_correlation_called",
            overlap["rank_correlation_called"],
            "quantity",
            "dec2",
            "the same over units called for either",
        ),
        make(
            "overlap_same_sign_fraction_both",
            overlap["same_sign_fraction_both"],
            "fraction",
            "pct",
            "of the units called for both, the share whose two peak modulations have the "
            "same sign (both up or both down)",
        ),
        make(
            "split_improves_fraction",
            split["improves_fraction"],
            "fraction",
            "pct",
            "share of units whose held-out deviance improves when the non-local response "
            "is split by SWR status (non_local_SWR + non_local_no_SWR vs. non_local)",
        ),
        make(
            "split_best_model_fraction",
            split["best_model_fraction"],
            "fraction",
            "pct",
            "share of units whose best model by held-out deviance (among constant, "
            "non_local, and the split) is the split by SWR status",
        ),
        make(
            "split_swr_larger_fraction_called",
            overlap["swr_larger_fraction_called"],
            "fraction",
            "pct",
            "of units called for either, the share whose absolute peak modulation is "
            "larger for SWR-coupled replay than for non-SWR replay",
        ),
    ]
    effects = population_statistics.peak_modulation_statistics(
        population, list(EFFECT_CURVES.values())
    )
    for word, column in EFFECT_CURVES.items():
        stats = effects.loc[column]
        label = {
            "swr_replay": "SWR-coupled replay",
            "non_swr_replay": "non-SWR replay",
            "swr": "SWRs",
            "matched_control": "matched quiet-rest control windows (the noise floor)",
        }[word]
        values += [
            make(
                f"effect_abs_peak_median_{word}",
                stats["abs_peak_median"],
                "quantity",
                "dec2",
                f"median over units of the absolute peak log-rate modulation for {label}",
                "log rate",
            ),
            make(
                f"effect_rate_multiplier_median_{word}",
                stats["rate_multiplier_median"],
                "quantity",
                "sig2",
                f"exp of that median: the median unit's largest rate change for {label}, as "
                "a multiplier (up or down)",
                "x",
            ),
            make(
                f"effect_suppressed_fraction_{word}",
                stats["suppressed_fraction"],
                "fraction",
                "pct",
                f"share of units whose peak modulation for {label} is a decrease",
            ),
        ]
    for word, called_median in (
        ("swr_replay", overlap["abs_peak_median_called_swr"]),
        ("non_swr_replay", overlap["abs_peak_median_called_non_swr"]),
    ):
        values.append(
            make(
                f"effect_abs_peak_median_called_{word}",
                called_median,
                "quantity",
                "dec2",
                f"median absolute peak log-rate modulation of the units called for "
                f"{'SWR-coupled' if word == 'swr_replay' else 'non-SWR'} replay, "
                "among the units tested for both replay classes",
                "log rate",
            )
        )
    return values
