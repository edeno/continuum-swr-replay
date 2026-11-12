"""The mPFC summary's cohort, estimator, called-fraction, null-geometry and overlap
values: which statistic reaches which key, on twelve hand-made units and two
sidecars."""

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.pfc.load_population import (
    HELDOUT_P_COLUMNS,
    MATCHED_NULL_OBSERVED_COLUMNS,
)
from continuum_swr_replay.reported.mpfc.common import TAGS
from continuum_swr_replay.reported.mpfc.inputs import STEM
from continuum_swr_replay.reported.mpfc.population import (
    CURVES,
    called_fraction_values,
    cohort_values,
    estimator_values,
    null_geometry_values,
    overlap_and_effect_values,
    prior_binding_values,
)
from continuum_swr_replay.reported.values import value_factory

NAN = np.nan
SPLIT = "non_local_SWR + non_local_no_SWR"

# Units 0-3 and 10 are chimi's, 4-9 and 11 wilbur's. For the two replay classes:
#
#   unit             0  1  2  3 | 4  5  6  7  8  9 | 10  11
#   SWR-coupled      x  x  .  . | x  x  x  .  .  x |  -   x     (- not tested)
#   non-SWR          x  .  x  . | x  x  .  x  .  x |  x   .
#
# Tested for both: every unit but 10. Called for both: 0, 4, 5, 9; for
# SWR-coupled replay only: 1, 6, 11; for non-SWR replay only: 2, 7; for neither:
# 3, 8. chimi's four tested units are one of each.
_P = {
    "SWR": [0.01] * 3 + [0.5] * 9,
    "non_local": [0.01] * 2 + [0.5] * 10,
    "non_local_SWR": [
        0.01,
        0.01,
        0.5,
        0.5,
        0.01,
        0.01,
        0.01,
        0.5,
        0.5,
        0.01,
        NAN,
        0.01,
    ],
    "non_local_no_SWR": [
        0.01,
        0.5,
        0.01,
        0.5,
        0.01,
        0.01,
        0.5,
        0.01,
        0.5,
        0.01,
        0.01,
        0.5,
    ],
    # called: chimi's 0, 1 and 2 and wilbur's 4
    "engagement": [0.01, 0.01, 0.01, 0.5, 0.01] + [0.5] * 7,
}
# The observed improvement of a called unit is not positive for: engagement,
# units 1 and 2 of its four; SWR, unit 2 of its three; all non-local events,
# none of its two; SWR-coupled replay, units 0, 1 and 4 of its seven; non-SWR
# replay, unit 10 of its seven.
_OBSERVED = {
    "engagement": [1, 0, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "SWR": [1, 1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    "non_local": [1] * 12,
    "non_local_SWR": [-1, -1, 1, 1, -1, 1, 1, 1, 1, 1, 1, 1],
    "non_local_no_SWR": [1] * 10 + [0, 1],
}
_PEAKS = {
    "non_local_SWR": [1.0, -0.8, 0.3, 0.1, 2.0, -1.5, 0.6, 0.2, -0.1, 0.9, 5.0, 0.7],
    "non_local_no_SWR": [
        0.5,
        0.2,
        -0.7,
        0.05,
        1.0,
        1.2,
        0.1,
        0.8,
        0.15,
        -0.4,
        -3.0,
        0.3,
    ],
    "SWR": [0.4, -0.4] * 6,
    "matched_control": [0.05] * 12,
}


@pytest.fixture
def population() -> pd.DataFrame:
    chimi = [True] * 4 + [False] * 6 + [True, False]
    frame = pd.DataFrame(
        {
            "animal": np.where(chimi, "chimi", "wilbur"),
            "nwb_file_name": np.where(
                chimi, "chimi20200212_.nwb", "wilbur20210329_.nwb"
            ),
            "min_heldout_model": ["constant"] * 5
            + ["SWR"] * 4
            + ["non_local"] * 2
            + ["SWR + non_local_no_SWR"],
            f"{SPLIT} vs. non_local: heldout_delta_dev": [1.0] * 3 + [-1.0] * 8 + [NAN],
            "min_heldout_model_replay_split": [SPLIT] * 2 + ["constant"] * 10,
        }
    )
    for name, column in HELDOUT_P_COLUMNS.items():
        frame[column] = _P[name]
        frame[MATCHED_NULL_OBSERVED_COLUMNS[name]] = _OBSERVED[name]
    for curve, peaks in _PEAKS.items():
        frame[f"{curve}: peak_log_modulation"] = peaks
    # every unit is well posed; the prior binds on the first one, two, ... units
    for n_binds, curve in enumerate(CURVES, start=1):
        frame[f"{curve}: mle_converged"] = True
        frame[f"{curve}: mle_separated"] = False
        frame[f"{curve}: prior_binds"] = np.arange(12) < n_binds
    return frame


def _sidecar(name: str, counts: tuple, fits: tuple, overlap: float, windows: dict):
    return {
        "artifact": f"{name}{STEM}.pkl",
        "stop_cohort": "at_well",
        "cohort_counts": dict(
            zip(
                (
                    "n_bins_fit",
                    "n_bins_immobile",
                    "n_nonlocal_after",
                    "n_nonlocal_before",
                    "n_ripple_after",
                    "n_ripple_before",
                    "n_nonlocal_swr",
                ),
                counts,
                strict=True,
            )
        ),
        "estimator": "ridge_poisson_glm",
        "inference": "blocked_heldout_deviance",
        "empirical_pvalue_method": "plus_one",
        "implied_prior_sd": {
            "per_lag": 1.5,
            "per_lag_multiplier_low": 0.05,
            "per_lag_multiplier_high": 19.0,
            "sustained": 3.0,
            "sustained_multiplier_low": 0.003,
            "sustained_multiplier_high": 350.0,
        },
        "penalty_sigma_log_rate": 3,
        "n_folds": 5,
        "n_control_draws": 999,
        "n_model_fits": fits[0],
        "n_nonconverged_model_fits": fits[1],
        "n_neurons_with_nonconverged_model": fits[2],
        "engagement_null_overlap": {"median": overlap},
        "null_windows": {
            event_class: {
                "n_events_total": total,
                "n_events_excluded_unmatchable": excluded,
            }
            for event_class, (total, excluded) in windows.items()
        },
    }


@pytest.fixture
def sidecars() -> list[dict]:
    return [
        _sidecar(
            "a",
            (100, 200, 10, 20, 5, 10, 5),
            (10, 1, 1),
            0.1,
            {"SWR": (10, 1), "non_local": (20, 10)},
        ),
        _sidecar(
            "b",
            (300, 600, 30, 80, 15, 70, 6),
            (30, 3, 2),
            0.3,
            {"SWR": (30, 3), "non_local": (4, 3)},
        ),
    ]


def _values(function, *inputs) -> dict:
    make = value_factory(function, "synthetic", TAGS["modulation"])
    return {v.key: v for v in function(*inputs, make)}


def test_cohort_values_pool_the_counts(population, sidecars):
    values = _values(
        cohort_values, sidecars, population, {"n_pfc_cohort": 2, "n_analysed": 7}
    )
    assert values["sessions"].value == 2
    assert values["analysed_epochs"].value == 7
    assert values["animals"].value == 2
    assert values["units"].value == 12
    assert (values["bins_fit"].value, values["bins_immobile"].value) == (400, 800)
    assert values["bins_fit_fraction"].value == pytest.approx(0.5)
    assert values["nonlocal_events_kept"].value == 40
    assert values["nonlocal_events_before"].value == 100
    assert values["nonlocal_events_kept_fraction"].value == pytest.approx(0.4)
    # 20 of 80 SWRs
    assert values["swrs_kept_fraction"].value == pytest.approx(0.25)
    # the sessions' SWR-coupled shares are 5 of 10 and 6 of 30
    assert values["swr_coupled_share_session_median"].value == pytest.approx(0.35)


def test_cohort_values_refuse_another_number_of_bundles(population, sidecars):
    with pytest.raises(ValueError, match="2 bundles, but the PFC cohort holds 3"):
        _values(
            cohort_values, sidecars, population, {"n_pfc_cohort": 3, "n_analysed": 7}
        )


def test_estimator_values_copy_the_settings_and_sum_the_fits(sidecars):
    values = _values(estimator_values, sidecars)
    assert values["penalty_sigma"].value == 3
    assert values["prior_sd_per_lag"].value == 1.5
    assert values["prior_sd_sustained"].value == 3.0
    assert values["prior_multiplier_per_lag_low"].value == 0.05
    assert values["prior_multiplier_per_lag_high"].value == 19.0
    assert values["prior_multiplier_sustained_low"].value == 0.003
    assert values["prior_multiplier_sustained_high"].value == 350.0
    assert values["folds"].value == 5
    assert values["null_draws"].value == 999
    assert values["min_pvalue_denominator"].value == 1000
    assert values["min_pvalue"].value == 0.001
    assert values["model_fits"].value == 40
    assert values["nonconverged_model_fits"].value == 4
    assert values["nonconverged_neurons"].value == 3
    assert values["nonconverged_percent"].value == pytest.approx(10.0)


def test_prior_binding_values_are_the_lowest_and_highest_curve(population):
    values = _values(prior_binding_values, population)
    # the prior binds on 1 of 12 units for the first curve and 5 of 12 for the last
    assert values["prior_binds_percent_min"].value == pytest.approx(100 / 12)
    assert f"({CURVES[0]})" in values["prior_binds_percent_min"].description
    assert values["prior_binds_percent_max"].value == pytest.approx(500 / 12)
    assert f"({CURVES[-1]})" in values["prior_binds_percent_max"].description


def test_prior_binding_values_leave_out_a_curve_with_no_well_posed_unit(population):
    # the last curve's units all have a separated column: it has no share
    population[f"{CURVES[-1]}: mle_separated"] = True
    with np.errstate(invalid="ignore"):
        values = _values(prior_binding_values, population)
    assert values["prior_binds_percent_max"].value == pytest.approx(400 / 12)
    assert f"({CURVES[-2]})" in values["prior_binds_percent_max"].description


def test_called_fraction_values_per_class_and_per_animal(population):
    values = _values(called_fraction_values, population)
    expected = {
        "swr": (3, 12),
        "nonlocal": (2, 12),
        "swr_replay": (7, 11),
        "non_swr_replay": (7, 12),
        "engagement": (4, 12),
    }
    for word, (called, tested) in expected.items():
        assert values[f"{word}_units_called"].value == called, word
        assert values[f"{word}_units_tested"].value == tested, word
        assert values[f"{word}_called_fraction"].value == pytest.approx(called / tested)
    # chimi: three of its five units; wilbur: one of its seven
    assert values["engagement_called_fraction_chimi"].value == pytest.approx(0.6)
    assert values["engagement_called_fraction_wilbur"].value == pytest.approx(1 / 7)
    # Benjamini-Hochberg: the four p-values of 0.01 become 0.01 * 12 / 4 = 0.03
    assert values["engagement_bh_fraction"].value == pytest.approx(4 / 12)
    for word, n_units in (("constant", 5), ("swr", 4), ("nonlocal", 2), ("joint", 1)):
        assert values[f"best_model_{word}_fraction"].value == pytest.approx(
            n_units / 12
        ), word


def test_null_geometry_values_name_the_lowest_and_highest(population, sidecars):
    values = _values(null_geometry_values, population, sidecars)
    assert values["engagement_called_flat"].value == 2
    assert values["engagement_called_flat_fraction"].value == pytest.approx(0.5)
    # over the four event classes: none of two for all non-local events, three
    # of seven for SWR-coupled replay
    assert values["class_called_flat_fraction_min"].value == 0.0
    assert "(non_local)" in values["class_called_flat_fraction_min"].description
    assert values["class_called_flat_fraction_max"].value == pytest.approx(3 / 7)
    assert "(non_local_SWR)" in values["class_called_flat_fraction_max"].description
    assert values["engagement_null_overlap_median"].value == pytest.approx(0.2)
    assert values["engagement_null_overlap_session_min"].value == 0.1
    assert values["engagement_null_overlap_session_max"].value == 0.3
    # SWRs: 4 of 40 events have no control; all non-local events: 13 of 24
    assert values["unmatchable_percent_class_min"].value == pytest.approx(10.0)
    assert "(SWR)" in values["unmatchable_percent_class_min"].description
    assert values["unmatchable_percent_class_max"].value == pytest.approx(1300 / 24)
    assert "(non_local)" in values["unmatchable_percent_class_max"].description
    # the worst session and class is b's non-local events, 3 of 4 (not a's 10 of 20)
    assert values["unmatchable_worst_excluded"].value == 3
    assert values["unmatchable_worst_total"].value == 4
    assert "(b / non_local)" in values["unmatchable_worst_total"].description


def test_overlap_and_effect_values_keep_the_two_replay_classes_apart(population):
    values = _values(overlap_and_effect_values, population)
    assert values["overlap_units_tested"].value == 11
    assert values["overlap_units_both"].value == 4
    assert values["overlap_units_swr_replay_only"].value == 3
    assert values["overlap_units_non_swr_replay_only"].value == 2
    assert values["overlap_units_neither"].value == 2
    # 7 of 11 called for SWR-coupled replay, 6 of 11 for non-SWR replay
    assert values["overlap_units_both_expected"].value == pytest.approx(42 / 11)
    assert values["overlap_odds_ratio"].value == pytest.approx(4 * 2 / (3 * 2))
    # chimi: one unit of each kind; wilbur: 3 both, 2 and 1 one-only, 1 neither
    assert values["overlap_odds_ratio_animal_min"].value == pytest.approx(1.0)
    assert values["overlap_odds_ratio_animal_max"].value == pytest.approx(1.5)
    assert values["overlap_animals_odds_ratio_above_one"].value == 1
    # of units 0, 4, 5 and 9, the peaks of 5 and 9 have opposite signs
    assert values["overlap_same_sign_fraction_both"].value == 0.5
    # three of the eleven units with an improvement; two of twelve best models
    assert values["split_improves_fraction"].value == pytest.approx(3 / 11)
    assert values["split_best_model_fraction"].value == pytest.approx(2 / 12)
    # of the nine units called for either, the SWR-coupled peak is the larger
    # for all but units 2 and 7
    assert values["split_swr_larger_fraction_called"].value == pytest.approx(7 / 9)
    # |peak| over all twelve units: the middle two are 0.7 and 0.8, and for
    # non-SWR replay 0.4 and 0.5
    assert values["effect_abs_peak_median_swr_replay"].value == pytest.approx(0.75)
    assert values["effect_abs_peak_median_non_swr_replay"].value == pytest.approx(0.45)
    assert values["effect_abs_peak_median_swr"].value == pytest.approx(0.4)
    assert values["effect_abs_peak_median_matched_control"].value == pytest.approx(0.05)
    assert values["effect_rate_multiplier_median_swr"].value == pytest.approx(
        np.exp(0.4)
    )
    assert values["effect_suppressed_fraction_swr"].value == 0.5
    assert values["effect_suppressed_fraction_matched_control"].value == 0.0
    assert values["effect_suppressed_fraction_swr_replay"].value == pytest.approx(0.25)
    # Over the units tested for both that are called for the class: unit 10,
    # with the largest non-SWR peak (3.0), is not tested for SWR-coupled replay.
    #   SWR-coupled (0, 1, 4, 5, 6, 9, 11): 0.6, 0.7, 0.8, 0.9, 1.0, 1.5, 2.0
    #   non-SWR (0, 2, 4, 5, 7, 9): 0.4, 0.5, 0.7, 0.8, 1.0, 1.2
    assert values["effect_abs_peak_median_called_swr_replay"].value == pytest.approx(
        0.9
    )
    assert values["effect_abs_peak_median_called_non_swr_replay"].value == (
        pytest.approx(0.75)
    )


def test_the_overlap_descriptions_name_the_called_threshold(population, monkeypatch):
    # The calls do not move between 0.01 and 0.5, so only the text says which
    # threshold the counts are at.
    monkeypatch.setattr("continuum_swr_replay.reported.mpfc.population.ALPHA", 0.02)
    values = _values(overlap_and_effect_values, population)
    assert values["overlap_units_both"].value == 4
    assert values["overlap_units_both"].description == (
        "of those, units called (raw p < 0.02) for both"
    )
