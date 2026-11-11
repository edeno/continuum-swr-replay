"""The statistics of the mPFC unit population, on tables small enough to check by hand."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.pfc import population_statistics as stats
from continuum_swr_replay.analysis.pfc.load_population import (
    HELDOUT_P_COLUMNS,
    MATCHED_NULL_OBSERVED_COLUMNS,
)

ALPHA = 0.05
NAN = np.nan


def test_lowest_and_highest_name_the_first_group_with_each():
    values = pd.Series({"a": 3.0, "b": 1.0, "c": 3.0, "d": NAN})
    assert stats.lowest_and_highest(values) == {
        "min": 1.0,
        "min_label": "b",
        "max": 3.0,
        "max_label": "a",
    }


# --- the cohort and its fits ---------------------------------------------------------


def test_cohort_count_statistics_pool_the_counts_and_take_the_session_median():
    counts = pd.DataFrame(
        {
            "n_bins_fit": [100, 300, 0],
            "n_bins_immobile": [200, 600, 200],
            "n_nonlocal_after": [10, 30, 20],
            "n_nonlocal_before": [20, 80, 100],
            "n_ripple_after": [5, 15, 0],
            "n_ripple_before": [10, 70, 20],
            "n_nonlocal_swr": [5, 6, 20],
        }
    )
    result = stats.cohort_count_statistics(counts)
    assert (result["n_bins_fit"], result["n_bins_immobile"]) == (400, 1000)
    # pooled: 400 of 1000 (the mean of the sessions' shares would be 1/3)
    assert result["bins_fit_fraction"] == pytest.approx(0.4)
    assert (result["n_nonlocal_kept"], result["n_nonlocal_before"]) == (60, 200)
    assert result["nonlocal_kept_fraction"] == pytest.approx(0.3)
    assert result["swrs_kept_fraction"] == pytest.approx(20 / 100)
    # the sessions' SWR-coupled shares are 0.5, 0.2 and 1.0 (pooled: 31/60)
    assert result["swr_coupled_share_session_median"] == pytest.approx(0.5)


def test_fit_count_statistics_sum_sessions():
    fits = pd.DataFrame(
        {
            "n_model_fits": [10, 30],
            "n_nonconverged_model_fits": [1, 3],
            "n_neurons_with_nonconverged_model": [1, 2],
        }
    )
    assert stats.fit_count_statistics(fits) == {
        "n_model_fits": 40,
        "n_nonconverged_model_fits": 4,
        "n_neurons_with_nonconverged_model": 3,
        "nonconverged_percent": pytest.approx(10.0),
    }


def test_prior_binding_percent_is_over_the_well_posed_units():
    population = pd.DataFrame(
        {
            # units 0, 1 and 3 are well posed: 2 is separated, 4 did not
            # converge and 5 has no convergence flag; a missing flag is False
            "SWR: mle_converged": [True, True, True, True, False, NAN],
            "SWR: mle_separated": [False, False, True, NAN, False, False],
            "SWR: prior_binds": [True, False, True, True, True, True],
            # every unit is well posed and the prior binds on one, by a missing flag
            "no_SWR: mle_converged": [True] * 6,
            "no_SWR: mle_separated": [False] * 6,
            "no_SWR: prior_binds": [NAN, False, False, True, False, False],
        }
    )
    with warnings.catch_warnings():
        # pandas warns that filling a missing flag in an object column will change
        warnings.simplefilter("ignore", FutureWarning)
        percent = stats.prior_binding_percent(population, ["SWR", "no_SWR"])
    # SWR: the prior binds on units 0 and 3 of the three well-posed ones
    assert percent["SWR"] == pytest.approx(100 * 2 / 3)
    assert percent["no_SWR"] == pytest.approx(100 * 1 / 6)
    assert list(percent.index) == ["SWR", "no_SWR"]


# --- the units called for each test ---------------------------------------------------


def _units(n: int, **columns) -> pd.DataFrame:
    """``n`` units, tested and not called for every test unless ``columns`` says."""
    frame = pd.DataFrame({column: [0.5] * n for column in HELDOUT_P_COLUMNS.values()})
    for name, values in columns.items():
        frame[HELDOUT_P_COLUMNS.get(name, name)] = values
    return frame


def test_called_units_are_below_alpha_and_tested():
    population = _units(4, SWR=[0.01, 0.05, NAN, 0.049])
    # exactly alpha is not called; an untested unit is not called
    np.testing.assert_array_equal(
        stats.called_units(population, "SWR", ALPHA), [True, False, False, True]
    )


def test_called_fractions_are_over_the_tested_units():
    population = _units(
        4, SWR=[0.01, 0.2, NAN, 0.04], engagement=[0.01, 0.01, 0.01, 0.2]
    )
    fractions = stats.called_fractions(population, ALPHA)
    assert list(fractions.index) == list(HELDOUT_P_COLUMNS)
    swr = fractions.loc["SWR"]
    assert (swr["n_called"], swr["n_tested"]) == (2, 3)
    assert swr["fraction"] == pytest.approx(2 / 3)
    assert fractions.loc["engagement", "fraction"] == pytest.approx(0.75)
    assert fractions.loc["non_local", "n_called"] == 0


def test_called_fraction_by_animal_uses_each_animals_own_tested_units():
    population = _units(5, engagement=[0.01, NAN, 0.5, 0.01, 0.01]).assign(
        animal=["a", "a", "b", "b", "b"]
    )
    fractions = stats.called_fraction_by_animal(population, "engagement", ALPHA)
    assert list(fractions.index) == ["a", "b"]
    # a: one tested unit, called; b: two of three
    assert fractions.loc["a", "fraction"] == 1.0
    assert fractions.loc["a", "n_tested"] == 1
    assert fractions.loc["b", "fraction"] == pytest.approx(2 / 3)
    # the other tests are read from their own columns: nothing is called for SWR
    assert (
        stats.called_fraction_by_animal(population, "SWR", ALPHA)["fraction"]
        .eq(0)
        .all()
    )


def test_bh_called_fraction_adjusts_the_tested_units_p_values():
    population = _units(5, engagement=[0.001, 0.03, 0.04, 0.9, NAN])
    # Three of the four tested units have raw p < 0.05. Benjamini-Hochberg over
    # the four: 0.9 stays 0.9; 0.04 * 4/3 = 0.053; 0.03 * 4/2 = 0.06, lowered
    # to 0.053; 0.001 * 4 = 0.004. Only the last is below 0.05.
    assert stats.bh_called_fraction(population, "engagement", ALPHA) == 0.25


def test_best_model_fractions_are_shares_of_every_unit():
    population = pd.DataFrame({"min_heldout_model": ["constant", "SWR", "SWR", None]})
    fractions = stats.best_model_fractions(population, ["constant", "SWR", "non_local"])
    # the unit with no best model stays in the denominator
    assert fractions == {"constant": 0.25, "SWR": 0.5, "non_local": 0.0}
    other = pd.DataFrame({"best": ["a", "b"]})
    assert stats.best_model_fractions(other, ["a"], column="best") == {"a": 0.5}


def test_units_at_or_above_rate_keep_the_cutoff_itself():
    population = pd.DataFrame(
        {"immobility_avg_firing_rate": [0.5, 1.0, NAN, 2.0], "unit": [0, 1, 2, 3]}
    )
    curves = {"swr": np.arange(8).reshape(2, 4)}
    kept, kept_curves = stats.units_at_or_above_rate(population, 1.0, curves)
    assert list(kept["unit"]) == [1, 3]
    assert list(kept.index) == [0, 1]
    np.testing.assert_array_equal(kept_curves["swr"], [[1, 3], [5, 7]])
    kept, kept_curves = stats.units_at_or_above_rate(population, 1.0)
    assert list(kept["unit"]) == [1, 3] and kept_curves == {}


# --- the null's geometry ---------------------------------------------------------------


def test_called_without_improvement_counts_called_units_at_or_below_zero():
    population = _units(4, SWR=[0.01, 0.01, 0.2, 0.01], engagement=[0.01] * 4)
    for name, column in MATCHED_NULL_OBSERVED_COLUMNS.items():
        population[column] = 1.0 if name != "SWR" else [1.0, 0.0, -1.0, -0.5]
    population[MATCHED_NULL_OBSERVED_COLUMNS["engagement"]] = [2.0, 1.0, 0.5, -3.0]
    with np.errstate(invalid="ignore"):
        result = stats.called_without_improvement(population, ALPHA)
    # SWR: units 0, 1 and 3 are called; 1 (exactly zero) and 3 did not improve.
    # Unit 2 did not improve either, but is not called.
    swr = result.loc["SWR"]
    assert (swr["n_called"], swr["n_flat"]) == (3, 2)
    assert swr["fraction"] == pytest.approx(2 / 3)
    assert result.loc["engagement", "fraction"] == 0.25
    # no unit is called for the other classes: the share is undefined
    assert result.loc["non_local", "n_called"] == 0
    assert np.isnan(result.loc["non_local", "fraction"])


def test_session_value_statistics_are_the_median_and_range():
    assert stats.session_value_statistics([0.2, 0.9, 0.4, 0.5]) == {
        "median": pytest.approx(0.45),
        "min": 0.2,
        "max": 0.9,
    }


def _windows(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["session", "event_class", "total", "excluded"])


def test_unmatchable_event_statistics_pool_classes_and_find_the_worst_session():
    windows = _windows(
        [
            ("s1", "SWR", 10, 1),
            ("s1", "replay", 0, 0),
            ("s2", "SWR", 30, 3),
            ("s2", "replay", 20, 10),
            ("s3", "replay", 4, 3),
        ]
    )
    result = stats.unmatchable_event_statistics(windows)
    # SWR: 4 of 40 events; replay: 13 of 24
    assert result["percent_class_min"] == pytest.approx(10.0)
    assert result["class_min"] == "SWR"
    assert result["percent_class_max"] == pytest.approx(100 * 13 / 24)
    assert result["class_max"] == "replay"
    # the largest share is s3's 3 of 4, not s2's larger count of 10 of 20
    assert (result["worst_session"], result["worst_event_class"]) == ("s3", "replay")
    assert (result["worst_excluded"], result["worst_total"]) == (3, 4)


def test_unmatchable_event_statistics_refuse_a_table_without_events():
    with pytest.raises(ValueError, match="no session holds an event of any class"):
        stats.unmatchable_event_statistics(_windows([("s1", "SWR", 0, 0)]))


# --- SWR-coupled against non-SWR replay ---------------------------------------------


def test_odds_ratio_of_a_two_by_two_table():
    assert stats.odds_ratio(3, 2, 2, 5) == 3.75
    with pytest.raises(ValueError, match="odds ratio undefined for this table"):
        stats.odds_ratio(3, 2, 0, 5)


#: (animal's file, p SWR-coupled, p non-SWR, peak SWR-coupled, peak non-SWR)
_CHIMI, _WILBUR = "chimi20200212_.nwb", "wilbur20210329_.nwb"
_OVERLAP_UNITS = [
    (_CHIMI, 0.01, 0.01, 1.0, 0.5),  # 0: both
    (_CHIMI, 0.01, 0.01, -1.2, 0.2),  # 1: both, opposite signs
    (_CHIMI, 0.01, 0.5, 0.3, 0.6),  # 2: SWR-coupled only
    (_CHIMI, 0.5, 0.01, 0.2, -0.55),  # 3: non-SWR only
    (_CHIMI, 0.5, 0.5, -0.1, -0.15),  # 4: neither
    (_CHIMI, 0.01, 0.5, 0.4, 0.1),  # 5: SWR-coupled only
    (_WILBUR, 0.01, 0.01, 2.0, 1.5),  # 6: both
    (_WILBUR, 0.01, 0.5, -1.1, -0.25),  # 7: SWR-coupled only
    (_WILBUR, 0.5, 0.01, 0.15, 0.9),  # 8: non-SWR only
    (_WILBUR, 0.5, 0.5, 0.05, -0.3),  # 9: neither
    (_WILBUR, 0.5, 0.5, -0.6, 0.3),  # 10: neither
    (_WILBUR, 0.5, 0.5, 0.8, 0.7),  # 11: neither
    # not tested for SWR-coupled replay: in none of the statistics
    (_CHIMI, NAN, 0.01, 9.0, -9.0),
    (_WILBUR, 0.01, NAN, -9.0, 9.0),
]


def _overlap_population(units: list[tuple]) -> pd.DataFrame:
    frame = pd.DataFrame(
        units,
        columns=[
            "nwb_file_name",
            HELDOUT_P_COLUMNS["non_local_SWR"],
            HELDOUT_P_COLUMNS["non_local_no_SWR"],
            "non_local_SWR: peak_log_modulation",
            "non_local_no_SWR: peak_log_modulation",
        ],
    )
    # the animal is read from the file name, not from this column
    return frame.assign(animal="x")


def test_replay_class_overlap_statistics_on_twelve_tested_units():
    result = stats.replay_class_overlap_statistics(
        _overlap_population(_OVERLAP_UNITS), ALPHA
    )
    assert result["n_tested"] == 12
    assert (
        result["n_both"],
        result["n_swr_only"],
        result["n_non_swr_only"],
        result["n_neither"],
    ) == (3, 3, 2, 4)
    # 6 of 12 are called for SWR-coupled replay and 5 for non-SWR replay:
    # 12 * (6/12) * (5/12)
    assert result["n_both_expected"] == pytest.approx(2.5)
    assert result["odds_ratio"] == pytest.approx(3 * 4 / (3 * 2))
    # chimi: 2 both, 2 and 1 one-only, 1 neither: (2 * 1) / (2 * 1), which is
    # not above one. wilbur: 1 both, 1 and 1 one-only, 3 neither.
    assert result["odds_ratio_animal_min"] == pytest.approx(1.0)
    assert result["odds_ratio_animal_max"] == pytest.approx(3.0)
    assert result["n_animals_odds_ratio_above_one"] == 1
    # Ranks of the two peaks over units 0-11 (no ties):
    #   SWR-coupled: 11, 1, 8, 7, 4, 9, 12, 2, 6, 5, 3, 10
    #   non-SWR:      8, 6, 9, 1, 4, 5, 12, 3, 11, 2, 7, 10
    # squared rank differences sum to 9+25+1+36+0+16+0+1+25+9+16+0 = 138
    assert result["rank_correlation"] == pytest.approx(1 - 6 * 138 / (12 * 143))
    # Over the eight units called for either (0, 1, 2, 3, 5, 6, 7, 8):
    #   SWR-coupled: 7, 1, 5, 4, 6, 8, 2, 3; non-SWR: 5, 4, 6, 1, 3, 8, 2, 7
    # squared rank differences sum to 4+9+1+9+9+0+0+16 = 48
    assert result["rank_correlation_called"] == pytest.approx(1 - 6 * 48 / (8 * 63))
    # of units 0, 1 and 6, unit 1's peaks have opposite signs
    assert result["same_sign_fraction_both"] == pytest.approx(2 / 3)
    # |SWR-coupled| > |non-SWR| for units 0, 1, 5, 6 and 7 of the eight
    assert result["swr_larger_fraction_called"] == pytest.approx(5 / 8)
    # called for SWR-coupled (0, 1, 2, 5, 6, 7): |peak| 0.3, 0.4, 1.0, 1.1, 1.2, 2.0
    assert result["abs_peak_median_called_swr"] == pytest.approx(1.05)
    # called for non-SWR (0, 1, 3, 6, 8): |peak| 0.2, 0.5, 0.55, 0.9, 1.5
    assert result["abs_peak_median_called_non_swr"] == pytest.approx(0.55)


def test_replay_class_overlap_statistics_refuse_an_animal_without_a_one_only_unit():
    # without unit 7, wilbur has no unit called for SWR-coupled replay only
    units = [unit for i, unit in enumerate(_OVERLAP_UNITS) if i != 7]
    with pytest.raises(ValueError, match="odds ratio undefined for animal wilbur"):
        stats.replay_class_overlap_statistics(_overlap_population(units), ALPHA)


def test_replay_split_statistics_leave_out_units_without_a_value():
    split = "non_local_SWR + non_local_no_SWR"
    population = pd.DataFrame(
        {
            f"{split} vs. non_local: heldout_delta_dev": [1.0, -1.0, NAN, 0.5, 0.0],
            "min_heldout_model_replay_split": [
                split,
                "constant",
                None,
                "non_local",
                "constant",
            ],
        }
    )
    result = stats.replay_split_statistics(population)
    # four units have an improvement; two are positive (zero is not)
    assert result["improves_fraction"] == 0.5
    # four units have a best model; for one it is the split
    assert result["best_model_fraction"] == 0.25


def test_peak_modulation_statistics_per_curve():
    population = pd.DataFrame(
        {
            "SWR: peak_log_modulation": [0.5, -1.0, NAN, -2.0],
            "matched_control: peak_log_modulation": [0.1, 0.3, -0.2, 0.2],
        }
    )
    result = stats.peak_modulation_statistics(population, ["SWR", "matched_control"])
    # SWR: |peak| 0.5, 1.0, 2.0 over the three units with a peak (mean 1.17)
    assert result.loc["SWR", "abs_peak_median"] == pytest.approx(1.0)
    assert result.loc["SWR", "rate_multiplier_median"] == pytest.approx(np.e)
    assert result.loc["SWR", "suppressed_fraction"] == pytest.approx(2 / 3)
    # matched control: |peak| 0.1, 0.2, 0.2, 0.3
    control = result.loc["matched_control"]
    assert control["abs_peak_median"] == pytest.approx(0.2)
    assert control["rate_multiplier_median"] == pytest.approx(np.exp(0.2))
    assert control["suppressed_fraction"] == 0.25


# --- SWRs without replay ---------------------------------------------------------------

THREE_CLASS = "three"
MIN_EVENTS = 10
_S1, _S2, _S3 = (_CHIMI, "02_r1"), (_WILBUR, "02_r1"), (_WILBUR, "04_r2")

#: SWRs without replay, with replay, and left out, per session. Session 3 has
#: exactly the floor of SWRs without replay.
SESSIONS = pd.DataFrame(
    [(*_S1, 4, 40, 1), (*_S2, 30, 60, 0), (*_S3, 10, 20, 2)],
    columns=[
        "nwb_file_name",
        "epoch_name",
        "n_swr_no_non_local",
        "n_swr_with_non_local",
        "n_swr_no_non_local_unknown",
    ],
)

#: (session, animal, p SWRs without replay, p SWR-coupled replay, peak SWRs
#: without replay, peak SWR-coupled replay, improvement, best model)
_EXTENDED_UNITS = [
    (_S1, "chimi", 0.01, 0.01, 1.0, 0.5, 1.0, THREE_CLASS),  # 0: both
    (_S1, "chimi", 0.01, 0.5, -0.4, 0.2, -1.0, "constant"),  # 1: no-replay only
    (_S1, "chimi", 0.5, 0.5, 0.2, -0.6, 0.5, None),  # 2: neither
    (_S1, "chimi", NAN, 0.01, NAN, 0.3, NAN, "constant"),  # 3: untested
    (_S2, "wilbur", 0.01, 0.01, -2.0, -1.0, 2.0, THREE_CLASS),  # 4: both
    (_S2, "wilbur", 0.01, NAN, 0.6, 0.1, -0.5, THREE_CLASS),  # 5: replay untested
    # 6: neither (a p-value of exactly alpha is not called)
    (_S2, "wilbur", 0.05, 0.05, 0.1, 0.4, 0.0, "constant"),
    (_S3, "wilbur", 0.5, 0.01, -0.3, 0.8, 0.1, "non_local"),  # 7: replay only
    # 8: no-replay only (exactly alpha for SWR-coupled replay)
    (_S3, "wilbur", 0.01, 0.05, 0.8, -0.2, -0.2, "constant"),
    (_S3, "wilbur", 0.04, 0.01, 1.4, -0.9, 0.3, THREE_CLASS),  # 9: both, opposite
]


def _extended_population() -> pd.DataFrame:
    return pd.DataFrame(
        [(*session, *rest) for session, *rest in _EXTENDED_UNITS],
        columns=[
            "nwb_file_name",
            "epoch_name",
            "animal",
            "SWR_no_non_local: p_heldout",
            HELDOUT_P_COLUMNS["non_local_SWR"],
            "SWR_no_non_local: peak_log_modulation",
            "non_local_SWR_3class: peak_log_modulation",
            f"{THREE_CLASS} vs. non_local_SWR + non_local_no_SWR: heldout_delta_dev",
            "min_heldout_model_three_class",
        ],
    )


def test_swr_without_replay_event_statistics_sum_sessions():
    result = stats.swr_without_replay_event_statistics(SESSIONS, MIN_EVENTS)
    assert result == {
        "n_events": 44,
        # 44 without replay, 120 with, 3 left out
        "events_fraction": pytest.approx(44 / 167),
        "n_unknown": 3,
        "events_session_median": 10.0,
        # the session with exactly ten counts
        "n_sessions_min_events": 2,
    }


def test_swr_without_replay_unit_statistics_over_every_session():
    result = stats.swr_without_replay_unit_statistics(
        _extended_population(),
        SESSIONS,
        alpha=ALPHA,
        min_events=MIN_EVENTS,
        three_class_model=THREE_CLASS,
    ).loc["all"]
    assert (result["n_units"], result["n_tested"]) == (10, 9)
    # called: units 0, 1, 4, 5, 8 and 9
    assert result["n_called"] == 6
    assert result["called_fraction"] == pytest.approx(6 / 9)
    # Tested for both: every tested unit but 5. Called for both: 0, 4, 9;
    # no-replay only: 1, 8; replay only: 7; neither: 2, 6.
    assert result["n_overlap_tested"] == 8
    assert (
        result["n_overlap_both"],
        result["n_overlap_swr_no_replay_only"],
        result["n_overlap_swr_replay_only"],
        result["n_overlap_neither"],
    ) == (3, 2, 1, 2)
    # 5 of 8 called for SWRs without replay, 4 for SWR-coupled replay:
    # 8 * (5/8) * (4/8)
    assert result["n_overlap_expected"] == pytest.approx(2.5)
    assert result["overlap_odds_ratio"] == pytest.approx(3 * 2 / (2 * 1))
    # Ranks over units 0, 1, 2, 4, 6, 7, 8, 9:
    #   without replay: 7, 2, 5, 1, 4, 3, 6, 8; SWR-coupled: 7, 5, 3, 1, 6, 8, 4, 2
    # squared rank differences sum to 0+9+4+0+4+25+4+36 = 82
    assert result["rank_correlation"] == pytest.approx(1 - 6 * 82 / (8 * 63))
    # of units 0, 4 and 9, unit 9's peaks have opposite signs
    assert result["same_sign_fraction"] == pytest.approx(2 / 3)
    # |peak| without replay over the nine units with one:
    # 0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.4, 2.0
    assert result["abs_peak_median_swr_no_replay"] == pytest.approx(0.6)
    assert result["rate_multiplier_median_swr_no_replay"] == pytest.approx(np.exp(0.6))
    # over the six called units: 0.4, 0.6, 0.8, 1.0, 1.4, 2.0
    assert result["abs_peak_median_called_swr_no_replay"] == pytest.approx(0.9)
    # |peak| for SWR-coupled replay over all ten:
    # 0.1, 0.2, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9, 1.0
    assert result["abs_peak_median_swr_replay"] == pytest.approx(0.45)
    assert result["rate_multiplier_median_swr_replay"] == pytest.approx(np.exp(0.45))
    # over the six units called for SWRs without replay: 0.1, 0.2, 0.2, 0.5, 0.9, 1.0
    assert result["abs_peak_median_called_swr_replay"] == pytest.approx(0.35)
    # nine improvements, five positive (zero is not)
    assert result["adds_to_split_fraction"] == pytest.approx(5 / 9)
    # nine units have a best model; for four it is the three-class model
    assert result["best_model_fraction"] == pytest.approx(4 / 9)


def test_swr_without_replay_unit_statistics_over_sessions_with_enough_events():
    result = stats.swr_without_replay_unit_statistics(
        _extended_population(),
        SESSIONS,
        alpha=ALPHA,
        min_events=MIN_EVENTS,
        three_class_model=THREE_CLASS,
    ).loc["min_events"]
    # sessions 2 and 3 (30 and exactly 10 events): units 4-9
    assert (result["n_units"], result["n_tested"], result["n_called"]) == (6, 6, 4)
    assert result["called_fraction"] == pytest.approx(4 / 6)
    # tested for both: 4, 6, 7, 8, 9; both: 4, 9; one only: 8 and 7; neither: 6
    assert result["n_overlap_tested"] == 5
    assert (
        result["n_overlap_both"],
        result["n_overlap_swr_no_replay_only"],
        result["n_overlap_swr_replay_only"],
        result["n_overlap_neither"],
    ) == (2, 1, 1, 1)
    assert result["n_overlap_expected"] == pytest.approx(5 * (3 / 5) * (3 / 5))
    assert result["overlap_odds_ratio"] == pytest.approx(2.0)
    # ranks over units 4, 6, 7, 8, 9: 1, 3, 2, 4, 5 and 1, 4, 5, 3, 2:
    # squared rank differences sum to 0+1+9+1+9 = 20, so rho = 1 - 120/120
    assert result["rank_correlation"] == pytest.approx(0.0, abs=1e-12)
    assert result["same_sign_fraction"] == 0.5
    # |peak| without replay: 0.1, 0.3, 0.6, 0.8, 1.4, 2.0; called: 0.6, 0.8, 1.4, 2.0
    assert result["abs_peak_median_swr_no_replay"] == pytest.approx(0.7)
    assert result["abs_peak_median_called_swr_no_replay"] == pytest.approx(1.1)
    # |peak| SWR-coupled: 0.1, 0.2, 0.4, 0.8, 0.9, 1.0; called: 0.1, 0.2, 0.9, 1.0
    assert result["abs_peak_median_swr_replay"] == pytest.approx(0.6)
    assert result["abs_peak_median_called_swr_replay"] == pytest.approx(0.55)
    assert result["adds_to_split_fraction"] == 0.5
    assert result["best_model_fraction"] == 0.5


def test_swr_without_replay_unit_statistics_are_missing_over_no_units():
    result = stats.swr_without_replay_unit_statistics(
        _extended_population(),
        SESSIONS,
        alpha=ALPHA,
        min_events=31,
        three_class_model=THREE_CLASS,
    ).loc["min_events"]
    assert (result["n_units"], result["n_tested"]) == (0, 0)
    assert result.drop(["n_units", "n_tested"]).isna().all()


def test_swr_without_replay_statistics_refuse_a_unit_of_an_unknown_session():
    with pytest.raises(KeyError, match="wilbur20210329_.nwb"):
        stats.swr_without_replay_unit_statistics(
            _extended_population(),
            SESSIONS.iloc[:1],
            alpha=ALPHA,
            min_events=MIN_EVENTS,
            three_class_model=THREE_CLASS,
        )


def test_swr_without_replay_called_by_animal():
    result = stats.swr_without_replay_called_by_animal(
        _extended_population(), SESSIONS, alpha=ALPHA, min_events=MIN_EVENTS
    )
    assert list(result.index) == [
        ("all", "chimi"),
        ("all", "wilbur"),
        ("min_events", "wilbur"),
    ]
    # chimi: units 0 and 1 of its three tested units; wilbur: 4, 5, 8, 9 of six
    np.testing.assert_allclose(result["called_fraction"], [2 / 3, 4 / 6, 4 / 6])


# The report window: +/-200 ms at 500 Hz.
LAG_MS = 2 * np.arange(-100, 101)
LAG_TIME = LAG_MS / 1000.0


def _ramp(peak_ms: int, offset: float) -> np.ndarray:
    """``offset`` until event onset, then a line reaching ``offset + 1`` at
    ``peak_ms`` and staying there."""
    return offset + np.clip(LAG_MS / peak_ms, 0.0, 1.0)


def test_swr_without_replay_timing_statistics_of_the_called_units_mean_curve():
    # Called units with an increase (0, 5, 8) share one curve and those with a
    # decrease (1, 4, 9) another, so their mean curves are those curves. The
    # other units peak at -100 ms and would move the mean if they were included.
    early = 5.0 * (LAG_MS == -100)
    up, down = _ramp(98, 0.2), -_ramp(58, 0.2)
    curves = np.column_stack(
        [up, down, early, early, down, up, -early, early, up, down]
    )
    result = stats.swr_without_replay_timing_statistics(
        _extended_population(),
        curves,
        LAG_TIME,
        SESSIONS,
        alpha=ALPHA,
        min_events=MIN_EVENTS,
        n_boot=50,
        seed=0,
    )
    assert list(result.index) == [
        ("all", "enhanced"),
        ("all", "suppressed"),
        ("min_events", "enhanced"),
        ("min_events", "suppressed"),
    ]
    np.testing.assert_array_equal(result["n_units"], [3, 3, 2, 2])
    for subset in ("all", "min_events"):
        enhanced = result.loc[(subset, "enhanced")]
        # The line reaches its top at 98 ms. It is half way up from its value
        # at -200 ms between 48 and 50 ms. (Half way up from zero, 0.6 of 1.2,
        # would be 40 ms.)
        assert enhanced["peak_ms"] == pytest.approx(98)
        assert enhanced["rise_onset_ms"] == pytest.approx(50)
        # every resample averages the same curve
        assert enhanced["rise_onset_ms_ci_low"] == pytest.approx(50)
        assert enhanced["rise_onset_ms_ci_high"] == pytest.approx(50)
        # the top less the value at -200 ms, not the top itself (1.2)
        assert enhanced["rise_size"] == pytest.approx(1.0)
        suppressed = result.loc[(subset, "suppressed")]
        assert suppressed["peak_ms"] == pytest.approx(58)
        assert suppressed["rise_onset_ms"] == pytest.approx(30)
        assert suppressed["rise_size"] == pytest.approx(1.0)

    # with only session 2 kept, each direction has one called unit: no timing
    alone = stats.swr_without_replay_timing_statistics(
        _extended_population(),
        curves,
        LAG_TIME,
        SESSIONS,
        alpha=ALPHA,
        min_events=20,
        n_boot=50,
        seed=0,
    ).loc["min_events"]
    np.testing.assert_array_equal(alone["n_units"], [1, 1])
    assert alone.drop(columns="n_units").isna().all(axis=None)
