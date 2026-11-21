"""The domain-free statistics, on inputs small enough to check by hand."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as student_t

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


# --- information criteria and pseudo-R² ---------------------------------------------


def test_aic_is_twice_the_parameters_less_twice_the_log_likelihood():
    assert stats.akaike_information_criterion(-10.5, 3) == 27.0
    assert stats.akaike_information_criterion(-10.5, 0) == 21.0


def test_bic_penalizes_each_parameter_by_the_log_of_the_observations():
    assert stats.bayesian_information_criterion(-10.5, 2, np.e**3) == pytest.approx(
        27.0
    )
    # No parameters: no penalty, whatever n.
    assert stats.bayesian_information_criterion(-10.5, 0, 50) == 21.0


def test_information_criteria_take_one_fit_per_element():
    llf = np.array([-1.0, -2.0, np.nan])
    np.testing.assert_array_equal(
        stats.akaike_information_criterion(llf, 2), [6.0, 8.0, np.nan]
    )
    np.testing.assert_array_equal(
        stats.bayesian_information_criterion(llf, np.array([0, 1, 1]), 1),
        [2.0, 4.0, np.nan],
    )


def test_pseudo_r2_is_one_less_the_ratio_to_the_null():
    assert stats.mcfadden_pseudo_r2(-30.0, -40.0) == 0.25
    assert stats.mcfadden_pseudo_r2(-40.0, -40.0) == 0.0


@pytest.mark.parametrize("llf_null", [0.0, np.nan, -np.inf])
def test_pseudo_r2_is_missing_without_a_finite_nonzero_null(llf_null):
    assert np.isnan(stats.mcfadden_pseudo_r2(-30.0, llf_null))


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


def test_a_missing_value_is_left_out_and_counts_in_nothing():
    # Animal c's only session is missing, so c is not an animal; a's missing
    # session is not a session. Counted, a missing value would be a session that
    # is neither positive nor negative, and in a's median it would be skipped
    # while c would count with a NaN median.
    values = pd.Series([1.0, np.nan, 2.0, -3.0, np.nan], index=[10, 11, 12, 13, 14])
    animal = pd.Series(["a", "a", "a", "b", "c"], index=values.index)
    assert stats.session_median_statistics(values, animal) == {
        "n_sessions": 3,
        "median": 1.0,
        "n_positive": 2,
        "n_negative": 1,
        "positive_fraction": pytest.approx(2 / 3),
        "negative_fraction": pytest.approx(1 / 3),
        "n_animals": 2,
        "animal_median": -0.75,
        "n_animals_positive": 1,
        "n_animals_negative": 1,
        "n_animals_same_sign": 1,
    }


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


def test_animal_medians_are_each_animals_median_over_its_rows_in_sorted_order():
    values = pd.Series([1.0, 9.0, 3.0, 4.0, np.nan, 5.0, 7.0])
    animal = pd.Series(["b", "b", "a", "a", "c", "b", "b"])
    result = stats.animal_medians(values, animal)
    # b has 1, 9, 5, 7 (even count, median 6; its mean is 5.5), a has 3, 4
    assert list(result.index) == ["a", "b"]
    assert list(result) == [3.5, 6.0]


def test_animal_medians_need_the_animals_indexed_as_the_values():
    with pytest.raises(ValueError, match="indexed alike"):
        stats.animal_medians(pd.Series([1.0, 2.0]), pd.Series(["a", "b"], index=[5, 6]))


# --- the nested bootstrap -------------------------------------------------------------


def _two_animals() -> pd.DataFrame:
    """Animal a: session s1 holds 1, 2, 9 (median 2, mean 4), s2 holds 10 and
    a missing value; animal b: one session, 5 and 6. Session s1 of b shares
    a's label but is b's own."""
    return pd.DataFrame(
        {
            "value": [1.0, 2.0, 9.0, 10.0, NAN, 5.0, 6.0],
            "animal": ["a", "a", "a", "a", "a", "b", "b"],
            "day": "d1",
            "epoch": ["s1", "s1", "s1", "s2", "s2", "s1", "s1"],
        }
    )


def test_animal_session_values_go_row_then_session_then_animal():
    rows = _two_animals()
    medians = stats.animal_session_values(
        rows["value"], rows["animal"], rows["epoch"], "median"
    )
    # a: the median of 2 and 10 is 6 (its rows' median is 5.5); b: 5.5
    assert medians.to_dict() == {"a": 6.0, "b": 5.5}
    means = stats.animal_session_values(
        rows["value"], rows["animal"], rows[["day", "epoch"]], "mean"
    )
    # a: the mean of 4 and 10 is 7 (its rows' mean is 5.5); b: 5.5
    assert means.to_dict() == {"a": pytest.approx(7.0), "b": pytest.approx(5.5)}


def test_animal_session_values_leave_out_a_session_without_a_value():
    rows = _two_animals()
    rows.loc[3, "value"] = NAN  # s2 of a has no value left
    medians = stats.animal_session_values(
        rows["value"], rows["animal"], rows["epoch"], "median"
    )
    assert medians["a"] == 2.0


def test_animal_session_values_refuse_another_statistic_or_index():
    rows = _two_animals()
    with pytest.raises(ValueError, match="statistic 'max'"):
        stats.animal_session_values(rows["value"], rows["animal"], rows["epoch"], "max")
    with pytest.raises(ValueError, match="not indexed alike"):
        stats.animal_session_values(
            rows["value"], rows["animal"].iloc[:3], rows["epoch"], "mean"
        )
    with pytest.raises(ValueError, match="not indexed alike"):
        stats.animal_session_values(
            rows["value"], rows["animal"], rows["epoch"].iloc[::-1], "mean"
        )


def test_weighted_median_equals_median_of_repeated_sample():
    rng = np.random.default_rng(0)
    for _ in range(50):
        values = np.sort(rng.normal(size=rng.integers(1, 12)))
        weights = rng.integers(0, 4, values.size)
        if weights.sum() == 0:
            assert np.isnan(stats.weighted_median(values, weights))
            continue
        expected = np.median(np.repeat(values, weights))
        assert stats.weighted_median(values, weights) == pytest.approx(expected)


def _design(n_animals=5, n_sessions=8, n_events=40):
    animal = np.repeat(np.arange(n_animals), n_sessions * n_events)
    session = np.repeat(np.arange(n_animals * n_sessions), n_events)
    return animal, session


def test_nested_counts_keep_the_totals_and_the_nesting():
    animal, session = _design()
    rng = np.random.default_rng(1)
    for _ in range(20):
        counts = stats.nested_bootstrap_draw(animal, session, rng).rows
        # every level draws as many as it has: the total is unchanged
        assert counts.sum() == animal.size
        # a drawn session's events total a multiple of its size
        per_session = pd.Series(counts).groupby(session).sum()
        assert (per_session % 40 == 0).all()
        per_animal = pd.Series(counts).groupby(animal).sum()
        assert (per_animal % (8 * 40) == 0).all()


def test_nested_bootstrap_spread_tracks_between_animal_variation():
    # an animal-level effect dominates the median's uncertainty: the
    # nested bootstrap must see it (a flat event bootstrap would not)
    animal, session = _design()
    rng = np.random.default_rng(2)
    offsets = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    values = offsets[animal] + rng.normal(0, 0.1, animal.size)
    medians = []
    for _ in range(200):
        counts = stats.nested_bootstrap_draw(animal, session, rng).rows
        order = np.argsort(values)
        medians.append(stats.weighted_median(values[order], counts[order]))
    flat = [np.median(rng.choice(values, values.size)) for _ in range(200)]
    assert np.std(medians) > 10 * np.std(flat)


def test_the_draw_counts_each_level_consistently():
    # Unequal sizes: animal 0 has sessions of 1 and 3 rows, animal 1 one of 2.
    animal = np.array([0, 0, 0, 0, 1, 1])
    session = np.array([0, 1, 1, 1, 2, 2])
    sizes = np.array([1, 3, 2])
    rng = np.random.default_rng(4)
    for _ in range(50):
        draw = stats.nested_bootstrap_draw(animal, session, rng)
        assert draw.animals.sum() == 2
        # each draw of an animal draws as many sessions as it has
        assert draw.sessions[:2].sum() == 2 * draw.animals[0]
        assert draw.sessions[2] == draw.animals[1]
        # each draw of a session draws as many rows as it has
        per_session = np.bincount(session, weights=draw.rows, minlength=3)
        np.testing.assert_array_equal(per_session, draw.sessions * sizes)


def test_many_draws_at_once_keep_the_nesting_and_one_is_the_single_draw():
    # Unequal sizes: animal 0 has sessions of 1 and 3 rows, animal 1 one of 2.
    animal = np.array([0, 0, 0, 0, 1, 1])
    session = np.array([0, 1, 1, 1, 2, 2])
    sizes = np.array([1, 3, 2])
    draws = stats.nested_bootstrap_draws(animal, session, np.random.default_rng(4), 200)
    assert draws.animals.shape == (200, 2) and draws.rows.shape == (200, 6)
    np.testing.assert_array_equal(draws.animals.sum(axis=1), 2)
    np.testing.assert_array_equal(draws.sessions[:, :2].sum(1), 2 * draws.animals[:, 0])
    np.testing.assert_array_equal(draws.sessions[:, 2], draws.animals[:, 1])
    per_session = np.stack(
        [np.bincount(session, weights=r, minlength=3) for r in draws.rows]
    )
    np.testing.assert_array_equal(per_session, draws.sessions * sizes)
    # both animals and every row of the three-row session vary over replicates
    assert (draws.rows.std(axis=0) > 0).all()
    # one draw at once is the single draw, and leaves the generator alike
    rng_one, rng_many = np.random.default_rng(9), np.random.default_rng(9)
    single = stats.nested_bootstrap_draw(animal, session, rng_one)
    batch = stats.nested_bootstrap_draws(animal, session, rng_many, 1)
    for a, b in zip(single, batch, strict=True):
        np.testing.assert_array_equal(a, b[0])
    assert rng_one.integers(10**9) == rng_many.integers(10**9)


def test_bootstrap_interval_is_the_estimate_plus_minus_the_corrected_t_half_width():
    # Draws 1, 2, 3, 4: SD (ddof=1) sqrt(5/3). Five animals: t(0.975, 4) is
    # 2.7764451, the correction sqrt(5/4). The interval is centered on the
    # estimate, not on the draws (mean 2.5).
    interval = stats.bootstrap_interval(10.0, [1.0, 2.0, 3.0, 4.0], 5)
    half = 2.7764451052 * np.sqrt(5 / 4) * np.sqrt(5 / 3)
    assert interval.sd == pytest.approx(np.sqrt(5 / 3))
    assert interval.low == pytest.approx(10.0 - half)
    assert interval.high == pytest.approx(10.0 + half)
    # Two animals: t(0.975, 1) is 12.7062047, the correction sqrt(2).
    two = stats.bootstrap_interval(0.0, [1.0, 2.0, 3.0, 4.0], 2)
    assert two.high == pytest.approx(12.7062047362 * np.sqrt(2) * np.sqrt(5 / 3))


def test_bootstrap_interval_clips_each_bound_to_the_statistics_range():
    # Same draws as above: half-width 2.7764451 * sqrt(5/4) * sqrt(5/3) = 4.008.
    half = 2.7764451052 * np.sqrt(5 / 4) * np.sqrt(5 / 3)
    clipped = stats.bootstrap_interval(1.0, [1.0, 2.0, 3.0, 4.0], 5, (-2.0, 3.0))
    assert clipped.low == -2.0 and clipped.high == 3.0
    assert clipped.sd == pytest.approx(np.sqrt(5 / 3))
    inside = stats.bootstrap_interval(1.0, [1.0, 2.0, 3.0, 4.0], 5, (-10.0, 10.0))
    assert inside.low == pytest.approx(1.0 - half)
    assert inside.high == pytest.approx(1.0 + half)
    missing = stats.bootstrap_interval(1.0, [1.0], 5, (-2.0, 3.0))
    assert np.isnan(missing.low) and np.isnan(missing.high)


def test_bootstrap_interval_of_a_mean_over_five_animals_is_their_t_interval():
    # Every one of the 5**5 resamples of five animal values, once each: the
    # exact bootstrap distribution of their mean. Its SD (ddof=0) is
    # sqrt(4/5) s/sqrt(5), so the interval is scipy's t-interval up to the
    # ddof=1 factor sqrt(3125/3124) on the half-width.
    values = np.array([0.3, -1.2, 2.5, 0.9, 4.0])
    draws = np.array([values[list(i)].mean() for i in product(range(5), repeat=5)])
    interval = stats.bootstrap_interval(values.mean(), draws, 5)
    low, high = student_t.interval(
        0.95, 4, loc=values.mean(), scale=values.std(ddof=1) / np.sqrt(5)
    )
    factor = np.sqrt(3125 / 3124)
    assert interval.high - values.mean() == pytest.approx(
        (high - values.mean()) * factor, rel=1e-12
    )
    assert values.mean() - interval.low == pytest.approx(
        (values.mean() - low) * factor, rel=1e-12
    )


def test_bootstrap_interval_leaves_out_non_finite_draws_and_needs_two_animals():
    with_nan = stats.bootstrap_interval(1.0, [1.0, NAN, 2.0, np.inf, 3.0, 4.0], 5)
    assert with_nan == stats.bootstrap_interval(1.0, [1.0, 2.0, 3.0, 4.0], 5)
    one_animal = stats.bootstrap_interval(1.0, [1.0, 2.0], 1)
    assert np.isnan(one_animal.low) and np.isnan(one_animal.high)
    assert one_animal.sd == pytest.approx(np.sqrt(0.5))
    one_draw = stats.bootstrap_interval(1.0, [1.0, NAN], 5)
    assert np.isnan(one_draw.low) and np.isnan(one_draw.sd)
    no_estimate = stats.bootstrap_interval(NAN, [1.0, 2.0], 5)
    assert np.isnan(no_estimate.low) and np.isnan(no_estimate.high)
    # All draws equal: no width.
    flat = stats.bootstrap_interval(2.0, [3.0, 3.0, 3.0], 5)
    assert (flat.low, flat.high, flat.sd) == (2.0, 2.0, 0.0)


def test_nested_codes_follow_the_sorted_labels_and_refuse_a_shared_session():
    animal, session = stats.nested_codes(
        np.array(["b", "a", "b"]), np.array(["s2", "s1", "s3"])
    )
    np.testing.assert_array_equal(animal, [1, 0, 1])
    np.testing.assert_array_equal(session, [1, 0, 2])
    with pytest.raises(ValueError, match="more than one animal"):
        stats.nested_codes(np.array([0, 0, 1, 1]), np.array([0, 0, 0, 1]))


# --- curves over sessions and animals ----------------------------------------------


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


# --- session -> recording day -> animal --------------------------------------------

# Two animals. Animal "a": day 1 holds sessions 1.0 and 3.0, day 2 holds 8.0;
# animal "b": one day of sessions -1.0, 0.0 and 4.0.
THREE_LEVEL = pd.DataFrame(
    {
        "value": [1.0, 3.0, 8.0, -1.0, 0.0, 4.0],
        "animal": ["a", "a", "a", "b", "b", "b"],
        "day": ["d1", "d1", "d2", "d1", "d1", "d1"],
        "session": ["s1", "s2", "s3", "s1", "s2", "s3"],
    }
)


def test_the_three_level_bootstrap_estimate_is_the_mean_of_animal_means_of_day_means():
    # a: days (2 + 8) / 2 = 5; b: 1. Mean over animals 3. The day label d1 of
    # animal b is not animal a's day d1.
    result = stats.animal_day_session_bootstrap(
        THREE_LEVEL.value, THREE_LEVEL.animal, THREE_LEVEL.day, n_draws=50, seed=0
    )
    assert result.estimate == pytest.approx(3.0)
    assert result.n_animals == 2
    assert result.draws.shape == (50,)


def test_each_three_level_draw_is_the_mean_of_means_of_its_drawn_counts():
    rng = np.random.default_rng(0)
    animal = np.repeat([0, 1, 2], 6)
    day = np.repeat(np.arange(6), 3)  # two days of three sessions per animal
    values = rng.normal(size=(18, 4))  # a curve of four points per session
    result = stats.animal_day_session_bootstrap(values, animal, day, n_draws=30, seed=7)
    # The same counts, drawn again with the same seed: animals, days, sessions.
    codes = stats.nested_codes(animal, day)
    draws = stats.nested_bootstrap_draws(*codes, np.random.default_rng(7), 30)
    for k in range(30):
        expected = np.zeros(4)
        for a in range(3):
            if draws.animals[k, a] == 0:
                continue
            day_means, day_counts = [], []
            for d in np.unique(day[animal == a]):
                rows = draws.rows[k, day == d]
                if draws.sessions[k, d] == 0:
                    continue
                day_means.append(rows @ values[day == d] / rows.sum())
                day_counts.append(draws.sessions[k, d])
            animal_mean = np.average(day_means, axis=0, weights=day_counts)
            expected += draws.animals[k, a] * animal_mean / 3
        np.testing.assert_allclose(result.draws[k], expected, rtol=1e-12)


def test_the_three_level_bootstrap_leaves_out_a_session_without_a_finite_value():
    values = THREE_LEVEL.value.to_numpy().copy()
    values[2] = np.nan  # animal a keeps one day, mean 2
    result = stats.animal_day_session_bootstrap(
        values, THREE_LEVEL.animal, THREE_LEVEL.day, n_draws=10, seed=0
    )
    assert result.estimate == pytest.approx((2.0 + 1.0) / 2)
