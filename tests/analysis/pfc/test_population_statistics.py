"""The statistics of the mPFC unit population, on tables small enough to check by hand."""

from __future__ import annotations

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


def test_cohort_count_statistics_keep_whole_totals_beside_a_missing_count():
    counts = pd.DataFrame(
        {
            "n_bins_fit": [100, 300],
            "n_bins_immobile": [200, 600],
            "n_nonlocal_after": [10, 30],
            "n_nonlocal_before": [20, 80],
            # one session has no count of its kept SWRs
            "n_ripple_after": [5, NAN],
            "n_ripple_before": [10, 70],
            "n_nonlocal_swr": [5, 6],
        }
    )
    result = stats.cohort_count_statistics(counts)
    # the other totals stay whole numbers: 400, not 400.0
    for name, total in (("n_bins_fit", 400), ("n_bins_immobile", 800)):
        assert isinstance(result[name], np.integer), name
        assert result[name] == total
    assert isinstance(result["n_nonlocal_kept"], np.integer)
    # the missing count is left out of its sum: 5 of 80
    assert result["swrs_kept_fraction"] == pytest.approx(5 / 80)


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


def test_fit_status_reads_a_missing_flag_as_false():
    population = pd.DataFrame(
        {
            # 0 well posed; 1 separated; 2 did not converge and is separated
            # (not converged comes first); 3 has no convergence flag; 4 has no
            # separation flag
            "SWR: mle_converged": [True, True, False, NAN, True],
            "SWR: mle_separated": [False, True, True, False, NAN],
        },
        index=[10, 11, 12, 13, 14],
    )
    status = stats.fit_status(population, "SWR")
    assert list(status.index) == [10, 11, 12, 13, 14]
    assert list(status) == [
        "well posed",
        "separated",
        "nonconverged",
        "nonconverged",
        "well posed",
    ]


def test_fit_status_refuses_a_missing_column():
    population = pd.DataFrame(
        {"SWR: mle_converged": [True, False], "SWR: mle_separated": [False, False]}
    )
    with pytest.raises(KeyError, match="no_SWR: mle_converged"):
        stats.fit_status(population, "no_SWR")
    with pytest.raises(KeyError, match="SWR: mle_separated"):
        stats.fit_status(population.drop(columns="SWR: mle_separated"), "SWR")
    with pytest.raises(KeyError, match="SWR: mle_converged"):
        stats.prior_binding_percent(
            population.drop(columns="SWR: mle_converged").assign(
                **{"SWR: prior_binds": False}
            ),
            ["SWR"],
        )


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


def _sessions(population: pd.DataFrame, animals: list, epochs: list) -> pd.DataFrame:
    """``population`` with each unit's animal and session (a day per animal)."""
    return population.assign(
        animal=animals, nwb_file_name=[f"{a}_day" for a in animals], epoch_name=epochs
    )


def test_called_fraction_by_animal_averages_each_animals_sessions():
    population = _sessions(
        _units(5, engagement=[0.01, NAN, 0.5, 0.01, 0.01]),
        ["a", "a", "b", "b", "b"],
        ["02_r1", "02_r1", "02_r1", "02_r1", "04_r2"],
    )
    fractions = stats.called_fraction_by_animal(population, "engagement", ALPHA)
    assert list(fractions.index) == ["a", "b"]
    # a: one tested unit, called; b: one of two in its first session, one of
    # one in its second: the mean of 1/2 and 1 is 3/4 (pooled, two of three)
    assert fractions.loc["a", "fraction"] == 1.0
    assert fractions.loc["a", "n_tested"] == 1
    assert fractions.loc["b", "fraction"] == pytest.approx(3 / 4)
    assert (fractions.loc["b", "n_called"], fractions.loc["b", "n_tested"]) == (2, 3)
    assert fractions.loc["b", "n_sessions"] == 2
    # the other tests are read from their own columns: nothing is called for SWR
    assert (
        stats.called_fraction_by_animal(population, "SWR", ALPHA)["fraction"]
        .eq(0)
        .all()
    )


def test_called_fraction_by_animal_refuses_an_animal_without_a_tested_unit():
    population = _sessions(
        _units(3, engagement=[0.01, NAN, NAN]), ["a", "b", "b"], ["02_r1"] * 3
    )
    with pytest.raises(ValueError, match=r"no unit tested for engagement: \['b'\]"):
        stats.called_fraction_by_animal(population, "engagement", ALPHA)


def test_bh_called_fraction_adjusts_the_tested_units_p_values():
    population = _units(5, engagement=[0.001, 0.02, 0.04, 0.9, NAN])
    # Three of the four tested units have raw p < 0.05. Benjamini-Hochberg over
    # the four: 0.9 stays 0.9; 0.04 * 4/3 = 0.053; 0.02 * 4/2 = 0.04;
    # 0.001 * 4 = 0.004. Two are below 0.05. (Bonferroni, times 4, leaves one:
    # 0.004, 0.08, 0.16. So does Benjamini-Yekutieli, which multiplies the
    # Benjamini-Hochberg values by 1 + 1/2 + 1/3 + 1/4.)
    assert stats.bh_called_fraction(population, "engagement", ALPHA) == 0.5


def test_best_model_fractions_are_shares_of_the_units_with_a_best_model():
    population = pd.DataFrame({"min_heldout_model": ["constant", "SWR", "SWR", None]})
    fractions = stats.best_model_fractions(population, ["constant", "SWR", "non_local"])
    # the unit with no best model is left out: three units, not four
    assert fractions == {
        "constant": pytest.approx(1 / 3),
        "SWR": pytest.approx(2 / 3),
        "non_local": 0.0,
    }
    other = pd.DataFrame({"best": ["a", "b"]})
    assert stats.best_model_fractions(other, ["a"], column="best") == {"a": 0.5}
    nobody = pd.DataFrame({"best": [None, None]})
    assert np.isnan(stats.best_model_fractions(nobody, ["a"], column="best")["a"])


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


def test_unmatchable_events_by_class_sum_sessions_and_skip_empty_ones():
    windows = _windows(
        [
            ("s1", "SWR", 10, 1),
            ("s1", "replay", 0, 0),
            ("s2", "SWR", 30, 3),
            ("s2", "replay", 20, 10),
            ("s3", "replay", 4, 3),
            ("s3", "control", 0, 0),
        ]
    )
    result = stats.unmatchable_events_by_class(windows).set_index("event_class")
    assert list(result.index) == ["SWR", "control", "replay"]
    # replay: three sessions (s1 has no event of it), 13 of 24 events; its
    # largest session share is s3's 3 of 4, not s2's 10 of 20
    replay = result.loc["replay"]
    assert replay["n_sessions"] == 3
    assert (replay["n_events_total"], replay["n_events_excluded"]) == (24, 13)
    assert replay["max_session_frac_excluded"] == pytest.approx(0.75)
    assert replay["frac_excluded"] == pytest.approx(13 / 24)
    assert result.loc["SWR", "frac_excluded"] == pytest.approx(0.1)
    # a class with no event in any session has no share
    assert np.isnan(result.loc["control", "max_session_frac_excluded"])
    assert np.isnan(result.loc["control", "frac_excluded"])


def test_engagement_null_overlap_statistics_name_the_worst_median_session():
    overlap = pd.DataFrame(
        {
            "session": ["s1", "s2", "s3", "s4"],
            "median_overlap_fraction": [0.1, 0.4, 0.2, np.nan],
            "max_overlap_fraction": [0.5, 0.45, 0.9, np.nan],
        }
    )
    assert stats.engagement_null_overlap_statistics(overlap) == {
        "n_sessions": 3,
        "median_over_sessions": pytest.approx(0.2),
        "min_over_sessions": 0.1,
        "max_over_sessions": 0.4,
        # the largest median, not the largest single draw (s3)
        "worst_session": "s2",
        "max_single_draw_overlap": 0.9,
    }


def test_unmatchable_event_statistics_refuse_a_table_without_events():
    with pytest.raises(ValueError, match="no session holds an event of any class"):
        stats.unmatchable_event_statistics(_windows([("s1", "SWR", 0, 0)]))


# --- SWR-coupled against non-SWR replay ---------------------------------------------


def _calls(first: str, second: str) -> tuple[np.ndarray, np.ndarray]:
    """Two boolean arrays from two strings of ``1`` (called) and ``0``."""
    return (
        np.array([c == "1" for c in first]),
        np.array([c == "1" for c in second]),
    )


def test_call_overlap_statistics_of_two_calls():
    # seven units: 2 called for both, 3 for the first only, 1 for the second
    # only, 1 for neither
    first, second = _calls("1111100", "1100010")
    first_peak = np.array([1.0, -2.0, 0.5, NAN, 3.0, 0.2, -0.4])
    second_peak = np.array([0.5, 1.0, 0.7, 0.1, 0.9, NAN, -0.3])
    result = stats.call_overlap_statistics(first, second, first_peak, second_peak)
    assert result["n_tested"] == 7
    assert (
        result["n_both"],
        result["n_first_only"],
        result["n_second_only"],
        result["n_neither"],
    ) == (2, 3, 1, 1)
    # 5 of 7 are called for the first class and 3 of 7 for the second
    assert result["n_both_expected"] == pytest.approx(7 * (5 / 7) * (3 / 7))
    # both times neither over the two one-only counts: (2 * 1) / (3 * 1)
    assert result["odds_ratio"] == pytest.approx(2 / 3)
    # Units 3 and 5 miss a peak. Ranks of the other five (0, 1, 2, 4, 6):
    #   first: 4, 1, 3, 5, 2; second: 2, 5, 3, 4, 1
    # squared rank differences sum to 4 + 16 + 0 + 1 + 1 = 22
    assert result["rank_correlation"] == pytest.approx(1 - 6 * 22 / (5 * 24))
    # of units 0 and 1, called for both, unit 1's peaks have opposite signs
    assert result["same_sign_fraction_both"] == 0.5


def test_call_overlap_statistics_without_a_one_only_unit_has_no_odds_ratio():
    peak = np.arange(4.0)
    # no unit is called for the second class only
    first, second = _calls("1110", "1000")
    assert np.isnan(
        stats.call_overlap_statistics(first, second, peak, peak)["odds_ratio"]
    )
    # nor, the other way round, for the first only
    assert np.isnan(
        stats.call_overlap_statistics(second, first, peak, peak)["odds_ratio"]
    )
    # with one of each the ratio is defined, and zero when none is called for both
    first, second = _calls("1100", "0110")
    assert stats.call_overlap_statistics(first, second, peak, peak)["odds_ratio"] == 1.0
    first, second = _calls("1000", "0110")
    assert stats.call_overlap_statistics(first, second, peak, peak)["odds_ratio"] == 0.0


def test_call_overlap_statistics_over_too_few_units_are_missing():
    # two units with both peaks: no rank correlation; none called for both
    first, second = _calls("100", "010")
    result = stats.call_overlap_statistics(
        first, second, np.array([1.0, 2.0, NAN]), np.array([2.0, 1.0, 3.0])
    )
    assert np.isnan(result["rank_correlation"])
    assert np.isnan(result["same_sign_fraction_both"])
    nobody = np.zeros(0, dtype=bool)
    nothing = stats.call_overlap_statistics(nobody, nobody, np.zeros(0), np.zeros(0))
    assert (nothing["n_tested"], nothing["n_both"], nothing["n_neither"]) == (0, 0, 0)
    assert np.isnan(nothing["n_both_expected"])
    assert np.isnan(nothing["odds_ratio"])


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
    animals = {_CHIMI: "chimi", _WILBUR: "wilbur"}
    return frame.assign(animal=frame["nwb_file_name"].map(animals))


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
    with pytest.raises(
        ValueError,
        match=r"odds ratio undefined for animal wilbur: a one-class-only cell is "
        r"empty \(both 1, one-only 0 and 1, neither 3\)",
    ):
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
    counts = ["n_units", "n_tested", "n_called", "n_overlap_tested"] + [
        f"n_overlap_{cell}"
        for cell in ("both", "swr_no_replay_only", "swr_replay_only", "neither")
    ]
    assert (result[counts] == 0).all()
    assert result.drop(counts).isna().all()


def test_swr_without_replay_statistics_refuse_a_unit_of_an_unknown_session():
    with pytest.raises(KeyError, match="wilbur20210329_.nwb"):
        stats.swr_without_replay_unit_statistics(
            _extended_population(),
            SESSIONS.iloc[:1],
            alpha=ALPHA,
            min_events=MIN_EVENTS,
            three_class_model=THREE_CLASS,
        )


def test_swr_without_replay_statistics_refuse_a_session_listed_twice():
    # session 1 again, with another count: neither row can be the session's
    twice = pd.concat(
        [SESSIONS, SESSIONS.iloc[:1].assign(n_swr_no_non_local=50)], ignore_index=True
    )
    message = r"sessions listed more than once: \[\('chimi20200212_.nwb', '02_r1'\)\]"
    with pytest.raises(ValueError, match=message):
        stats.swr_without_replay_event_statistics(twice, MIN_EVENTS)
    with pytest.raises(ValueError, match=message):
        stats.swr_without_replay_unit_statistics(
            _extended_population(),
            twice,
            alpha=ALPHA,
            min_events=MIN_EVENTS,
            three_class_model=THREE_CLASS,
        )


@pytest.mark.parametrize(
    ("column", "message"),
    [
        ("n_swr_no_non_local", "sessions with no count of SWRs without replay"),
        ("n_swr_with_non_local", "a session has no count of its SWRs with replay"),
        ("n_swr_no_non_local_unknown", "a session has no count of its SWRs with"),
    ],
)
def test_swr_without_replay_event_statistics_refuse_a_missing_count(column, message):
    # a missing count is not a count of zero
    sessions = SESSIONS.astype({column: float})
    sessions.loc[1, column] = NAN
    with pytest.raises(ValueError, match=message):
        stats.swr_without_replay_event_statistics(sessions, MIN_EVENTS)


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
    # The called units with an increase are 0 (chimi's day), 5 and 8 (two epochs
    # of wilbur's day); their lines reach the top at 98, 58 and 18 ms. Those
    # with a decrease (1, 4, 9) share one curve. The other units peak at
    # -100 ms and would move the mean if they were included.
    early = 5.0 * (LAG_MS == -100)
    down = -_ramp(58, 0.2)
    curves = np.column_stack(
        [
            _ramp(98, 0.2),
            down,
            early,
            early,
            down,
            _ramp(58, 0.2),
            -early,
            early,
            _ramp(18, 0.2),
            down,
        ]
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
    # Every session, increases. Each line is half way up from its value at
    # -200 ms (0.2) at half its ramp: 49, 29 and 9 ms. The mean of the three
    # reaches the top (1.2) at 98 ms and is half way up, 0.7, from 20 ms:
    # (20/98 + 20/58 + 1) / 3 = 0.516, while at 18 ms it is 0.498.
    enhanced = result.loc[("all", "enhanced")]
    assert enhanced["peak_ms"] == pytest.approx(98)
    assert enhanced["rise_onset_ms"] == pytest.approx(20)
    # the top less the value at -200 ms, not the top itself (1.2)
    assert enhanced["rise_size"] == pytest.approx(1.0)
    # The replicates resample the animals, their days and the days' units: a
    # mean of these lines is half way up between the earliest line's crossing
    # (unit 8 alone, 10 ms) and the latest's (unit 0 alone, 50 ms).
    assert 10 <= enhanced["rise_onset_ms_ci_low"] <= 20
    assert 20 <= enhanced["rise_onset_ms_ci_high"] <= 50
    assert enhanced["rise_size_ci_low"] == pytest.approx(1.0)
    assert enhanced["rise_size_ci_high"] == pytest.approx(1.0)
    # Sessions with enough events, increases: units 5 and 8, of one day of one
    # animal. Their mean is half way up from 14 ms ((14/58 + 14/18) / 2 = 0.510,
    # at 12 ms 0.437); the units are resampled within the day, and a replicate
    # of unit 8 alone crosses at 10 ms, of unit 5 alone at 30 ms (resampling
    # the days alone would give 14 ms every time).
    enough = result.loc[("min_events", "enhanced")]
    assert enough["peak_ms"] == pytest.approx(58)
    assert enough["rise_onset_ms"] == pytest.approx(14)
    assert enough["rise_onset_ms_ci_low"] == pytest.approx(10)
    assert enough["rise_onset_ms_ci_high"] == pytest.approx(30)
    for subset in ("all", "min_events"):
        suppressed = result.loc[(subset, "suppressed")]
        assert suppressed["peak_ms"] == pytest.approx(58)
        assert suppressed["rise_onset_ms"] == pytest.approx(30)
        assert suppressed["rise_size"] == pytest.approx(1.0)


def test_swr_without_replay_timing_statistics_of_one_unit_and_of_none():
    down = -_ramp(58, 0.2)
    curves = np.column_stack([down] * 5 + [_ramp(98, 0.2)] + [down] * 4)

    def timing(min_events: int) -> pd.DataFrame:
        return stats.swr_without_replay_timing_statistics(
            _extended_population(),
            curves,
            LAG_TIME,
            SESSIONS,
            alpha=ALPHA,
            min_events=min_events,
            n_boot=10,
            seed=0,
        ).loc["min_events"]

    # with only session 2 kept, each direction has one called unit (5 and 4),
    # and its mean curve is that unit's own
    alone = timing(20)
    np.testing.assert_array_equal(alone["n_units"], [1, 1])
    assert alone.loc["enhanced", "peak_ms"] == pytest.approx(98)
    assert alone.loc["enhanced", "rise_onset_ms"] == pytest.approx(50)
    assert alone.loc["suppressed", "peak_ms"] == pytest.approx(58)
    assert alone.loc["suppressed", "rise_onset_ms"] == pytest.approx(30)
    # no session has 31 events: no unit, and no timing
    nobody = timing(31)
    np.testing.assert_array_equal(nobody["n_units"], [0, 0])
    assert nobody.drop(columns="n_units").isna().all(axis=None)


# --- population curves -----------------------------------------------------------------


def test_session_mean_curve_leaves_out_a_unit_with_a_non_finite_lag():
    # lags x units; the middle unit is NaN at lag 0 and is left out whole.
    unit_curves = np.array([[1.0, NAN, 3.0], [2.0, 5.0, 4.0]])

    np.testing.assert_allclose(stats.session_mean_curve(unit_curves), [2.0, 3.0])
    with pytest.raises(ValueError, match="finite"):
        stats.session_mean_curve(np.array([[NAN], [1.0]]))


def test_animal_curve_sem_is_over_animals_at_each_lag():
    # Animal means at lag 0: 1, 2, 4 (mean 7/3, sample SD sqrt(7/3)); at lag 1
    # all equal.
    per_animal = {
        "a": np.array([1.0, 5.0]),
        "b": np.array([2.0, 5.0]),
        "c": np.array([4.0, 5.0]),
    }

    sem = stats.animal_curve_sem(per_animal)

    np.testing.assert_allclose(sem, [np.sqrt(7 / 3) / np.sqrt(3), 0.0])
    assert np.isnan(stats.animal_curve_sem({"a": np.array([1.0, 5.0])})).all()


def test_animal_balanced_unit_curve_weights_animals_equally():
    # Columns are units, in no session order. Animal a: session a1 holds three
    # units (mean [3, 0]), a2 one ([1, 2]), so a is [2, 1]. Animal b: b1 holds
    # [4, 4] and a unit with a NaN, left out, so b is [4, 4]. Animal c's only
    # unit has a NaN, so its session and c drop out.
    unit_curves = np.array(
        [
            # a1   b1    a2   a1   c1   b1   a1
            [1.0, 4.0, 1.0, 2.0, NAN, NAN, 6.0],
            [0.0, 4.0, 2.0, 0.0, 1.0, 100.0, 0.0],
        ]
    )
    sessions = np.array(["a1", "b1", "a2", "a1", "c1", "b1", "a1"])
    animals = np.array(["a", "b", "a", "a", "c", "b", "a"])

    population, sem = stats.animal_balanced_unit_curve(unit_curves, sessions, animals)

    # The mean over animals a and b; pooling the five finite units would give
    # [2.8, 1.2], and weighting the three sessions equally [8/3, 2].
    np.testing.assert_allclose(population, [3.0, 2.5])
    # Over two animals: |a - b| / 2 at each lag.
    np.testing.assert_allclose(sem, [1.0, 1.5])


def test_animal_balanced_unit_curve_has_no_band_with_one_animal():
    unit_curves = np.array([[1.0, 3.0], [0.0, 2.0]])

    population, sem = stats.animal_balanced_unit_curve(
        unit_curves, np.array(["s1", "s2"]), np.array(["a", "a"])
    )

    np.testing.assert_allclose(population, [2.0, 1.0])
    assert np.isnan(sem).all()
