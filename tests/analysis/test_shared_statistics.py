"""The domain-free statistics, on inputs small enough to check by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis import shared_statistics as stats

NAN = np.nan

# --- the signed-rank test -------------------------------------------------------------


def test_six_differences_of_one_sign_give_the_smallest_exact_p():
    # Only the all-positive and the all-negative of the 2 ** 6 sign patterns are
    # as extreme: p = 2 / 64.
    assert stats.signed_rank_p([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) == pytest.approx(2 / 64)
    assert stats.signed_rank_p([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]) == pytest.approx(
        2 / 64
    )


def test_the_exact_p_counts_the_sign_patterns_as_extreme():
    # The negative ranks sum to 2. Three of the 64 patterns have a rank sum of
    # at most 2 ({}, {1}, {2}): two-sided p = 2 * 3 / 64.
    assert stats.signed_rank_p([1.0, -2.0, 3.0, 4.0, 5.0, 6.0]) == pytest.approx(6 / 64)


def test_five_differences_are_too_few_to_test():
    # Their smallest two-sided p is 2 / 2 ** 5 = 0.0625: no test at 0.05.
    assert np.isnan(stats.signed_rank_p([1.0, 2.0, 3.0, 4.0, 5.0]))


def test_zeros_are_not_differences():
    # Seven values, but two are zero: five differences remain.
    assert np.isnan(stats.signed_rank_p([0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 5.0]))
    # Eight values with two zeros: six differences, tested as six.
    assert stats.signed_rank_p(
        [0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 5.0, 6.0]
    ) == pytest.approx(2 / 64)
    assert np.isnan(stats.signed_rank_p([0.0] * 8))


def test_missing_and_infinite_values_are_left_out():
    six = [1.0, -2.0, 3.0, 4.0, 5.0, 6.0]
    assert stats.signed_rank_p([NAN, *six, np.inf]) == pytest.approx(6 / 64)
    # Five finite differences and two that are not: too few.
    assert np.isnan(stats.signed_rank_p([1.0, 2.0, NAN, 3.0, 4.0, -np.inf, 5.0]))
    assert np.isnan(stats.signed_rank_p([]))


def test_a_series_is_read_by_its_values():
    values = pd.Series([1.0, -2.0, 3.0, 4.0, 5.0, 6.0], index=[9, 3, 7, 1, 0, 2])
    assert stats.signed_rank_p(values) == pytest.approx(6 / 64)


# --- Benjamini-Hochberg ---------------------------------------------------------------


def test_the_step_up_carries_a_smaller_adjusted_p_down_the_ranks():
    # Sorted: 0.01, 0.03, 0.04, 0.20; times m / rank: 0.04, 0.06, 0.0533, 0.20.
    # The second-ranked p's own value, 0.06, is larger than the third's, so it
    # takes the third's: 0.16 / 3.
    adjusted = stats.benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
    np.testing.assert_allclose(adjusted, [0.04, 0.16 / 3, 0.16 / 3, 0.20])


def test_a_missing_p_is_not_a_test():
    # The family is the four finite p-values: m = 4, not 6.
    adjusted = stats.benjamini_hochberg([0.01, NAN, 0.04, 0.03, np.inf, 0.20])
    np.testing.assert_allclose(adjusted, [0.04, NAN, 0.16 / 3, 0.16 / 3, NAN, 0.20])


def test_tied_p_values_get_one_adjusted_p():
    # times m / rank: 0.06, 0.03, 0.5 -> the tie takes the smaller, 0.03.
    adjusted = stats.benjamini_hochberg([0.02, 0.5, 0.02])
    np.testing.assert_allclose(adjusted, [0.03, 0.5, 0.03])


def test_a_family_of_one_is_its_own_p():
    np.testing.assert_allclose(stats.benjamini_hochberg([0.3]), [0.3])
    np.testing.assert_allclose(stats.benjamini_hochberg([NAN, 0.3]), [NAN, 0.3])


def test_a_family_without_a_test_is_all_missing():
    assert stats.benjamini_hochberg([]).shape == (0,)
    assert np.isnan(stats.benjamini_hochberg([NAN, NAN])).all()


def test_the_adjusted_p_is_never_above_one():
    adjusted = stats.benjamini_hochberg([0.9, 0.95, 0.6])
    # times m / rank: 1.8, 1.425, 0.95 -> every one takes 0.95
    np.testing.assert_allclose(adjusted, [0.95, 0.95, 0.95])


def test_the_adjusted_p_is_statsmodels_to_the_last_digit():
    # The adjusted values are reported, so their arithmetic is pinned: the
    # statsmodels form (p over rank / m), which can differ from
    # scipy.stats.false_discovery_control (p times m / rank) in the last digit.
    from statsmodels.stats.multitest import multipletests

    p = np.random.default_rng(0).uniform(size=200) ** 3
    np.testing.assert_array_equal(
        stats.benjamini_hochberg(p), multipletests(p, method="fdr_bh")[1]
    )


# --- the animal-balanced summary ------------------------------------------------------


def test_session_median_statistics_count_sessions_and_animals_by_sign():
    # sorted: -3, -1, 0, 1, 2, 4 -> median 0.5
    values = pd.Series([1.0, 2.0, -3.0, 4.0, 0.0, -1.0])
    # animal medians: a 1.5, b 0.5, c -0.5 -> median 0.5
    animal = pd.Series(["a", "a", "b", "b", "c", "c"])
    assert stats.session_median_statistics(values, animal) == {
        "n_sessions": 6,
        "median": 0.5,
        "n_positive": 3,
        "n_negative": 2,
        "positive_fraction": 0.5,
        "negative_fraction": pytest.approx(1 / 3),
        "n_animals": 3,
        "animal_median": 0.5,
        "n_animals_positive": 2,
        "n_animals_negative": 1,
        "n_animals_same_sign": 2,
    }


def test_the_animal_median_is_balanced_over_animals_not_sessions():
    # Animal a has five sessions at 1, b and c one session each at -2 and -4:
    # the session median is 1, the median over animals is -2.
    values = pd.Series([1.0, 1.0, 1.0, 1.0, 1.0, -2.0, -4.0])
    animal = pd.Series(["a"] * 5 + ["b", "c"])
    result = stats.session_median_statistics(values, animal)
    assert result["median"] == 1.0
    assert result["animal_median"] == -2.0
    assert result["n_animals_positive"] == 1
    assert result["n_animals_negative"] == 2
    # one animal (a) has the sign of the session median
    assert result["n_animals_same_sign"] == 1


def test_an_animal_s_number_is_its_median_and_zero_has_no_sign():
    # animal medians: a 2 (its mean is 4), b 3, c 0, d -6 -> median 1
    values = pd.Series([1.0, 2.0, 9.0, 3.0, 0.0, -6.0])
    animal = pd.Series(["a", "a", "a", "b", "c", "d"])
    result = stats.session_median_statistics(values, animal)
    assert result["animal_median"] == 1.0
    assert result["n_animals_positive"] == 2
    assert result["n_animals_negative"] == 1
    assert result["n_animals_same_sign"] == 2  # the session median is 1.5


def test_same_sign_follows_a_negative_median():
    # sorted: -4, -2, -1, 3 -> median -1.5; animal medians a -1.5, b 3, c -4
    values = pd.Series([-1.0, -2.0, 3.0, -4.0])
    animal = pd.Series(["a", "a", "b", "c"])
    result = stats.session_median_statistics(values, animal)
    assert result["median"] == -1.5
    assert result["animal_median"] == -1.5
    assert result["n_animals_same_sign"] == 2


def test_the_animals_must_be_indexed_as_the_values():
    values = pd.Series([1.0, 2.0, -3.0], index=[0, 1, 2])
    # Animals of other rows: the last session would have no animal, and the
    # animals' medians would be over two sessions of three.
    with pytest.raises(ValueError, match="not indexed alike"):
        stats.session_median_statistics(
            values, pd.Series(["a", "a", "b"], index=[0, 1, 7])
        )


def test_a_missing_value_is_refused():
    # Counted, it would be a session that is neither positive nor negative.
    values = pd.Series([1.0, np.nan, -3.0])
    with pytest.raises(ValueError, match="missing"):
        stats.session_median_statistics(values, pd.Series(["a", "a", "b"]))


def test_an_animal_without_a_session_is_not_an_animal_of_the_summary():
    # The category c has no session; a 1.5 and b -3 are the animals.
    values = pd.Series([1.0, 2.0, -3.0])
    animal = pd.Series(pd.Categorical(["a", "a", "b"], categories=["a", "b", "c"]))
    result = stats.session_median_statistics(values, animal)
    assert result["n_animals"] == 2
    assert result["animal_median"] == -0.75
    assert result["n_animals_positive"] == 1
    assert result["n_animals_negative"] == 1


def test_a_zero_session_median_has_the_sign_of_the_zero_animals_only():
    # sorted: -2, 0, 0, 3 -> median 0; animal medians a 0, b -2, c 1.5
    values = pd.Series([0.0, -2.0, 0.0, 3.0])
    animal = pd.Series(["a", "b", "c", "c"])
    result = stats.session_median_statistics(values, animal)
    assert result["median"] == 0.0
    assert result["n_animals_same_sign"] == 1  # a, whose own median is 0
