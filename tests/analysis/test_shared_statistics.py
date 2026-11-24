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


def test_weighted_medians_are_the_weighted_median_row_by_row():
    rng = np.random.default_rng(0)
    values = np.sort(rng.normal(size=(200, 6)), axis=1)
    weights = rng.integers(0, 4, size=(200, 6))
    weights[:3] = 0  # rows without a draw
    expected = [
        stats.weighted_median(v, w) for v, w in zip(values, weights, strict=True)
    ]
    np.testing.assert_allclose(stats.weighted_medians(values, weights), expected)
    assert np.isnan(expected[:3]).all()


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


def test_copies_give_every_slot_one_animal_and_that_animals_sessions():
    session_animal = np.array([0, 0, 0, 1, 1, 2])
    copies = stats.nested_bootstrap_copies(
        session_animal, np.random.default_rng(3), 500
    )
    assert copies.animal.shape == (500, 3)
    assert copies.sessions.shape == (500, 3, 6)
    assert set(np.unique(copies.animal)) == {0, 1, 2}
    # a slot draws the sessions of its own animal only, as many as it has
    own = session_animal == copies.animal[:, :, None]
    assert (copies.sessions[~own] == 0).all()
    np.testing.assert_array_equal(
        copies.sessions.sum(axis=2), np.bincount(session_animal)[copies.animal]
    )
    # two slots of one animal draw its sessions apart
    twice = (copies.animal[:, 0] == 0) & (copies.animal[:, 1] == 0)
    assert twice.sum() > 20
    assert (copies.sessions[twice, 0] != copies.sessions[twice, 1]).any()
    # every session is drawn once a replicate on average, as on the data
    np.testing.assert_allclose(copies.sessions.sum(axis=1).mean(axis=0), 1.0, atol=0.15)
    again = stats.nested_bootstrap_copies(session_animal, np.random.default_rng(3), 500)
    np.testing.assert_array_equal(copies.animal, again.animal)
    np.testing.assert_array_equal(copies.sessions, again.sessions)


def test_every_drawn_copy_of_a_session_has_its_own_resample():
    session_animal = np.array([0, 0, 1, 1, 1])
    copies = stats.nested_bootstrap_copies(
        session_animal, np.random.default_rng(0), 200
    )
    asked = []

    def resample(session: int, n: int) -> np.ndarray:
        asked.append((session, n))
        # resample k of session s has the value s + k / 1000: all distinct
        return session + np.arange(n) / 1000

    values = stats.session_copy_values(copies, resample)
    assert values.shape == (200, 2, 3)  # the largest animal has three sessions
    # each session is asked once, for as many resamples as it has copies
    assert asked == [(s, int(copies.sessions[:, :, s].sum())) for s in range(5)]
    known = values[np.isfinite(values)]
    # every resample is used, and none twice
    assert known.size == np.unique(known).size == copies.sessions.sum()
    for draw, slot in np.ndindex(200, 2):
        mine = values[draw, slot]
        drawn = mine[np.isfinite(mine)]
        # as many copies of each session as the slot drew, NaN after them
        np.testing.assert_array_equal(
            np.bincount(np.floor(drawn).astype(int), minlength=5),
            copies.sessions[draw, slot],
        )
        assert np.isnan(mine[drawn.size :]).all()


def test_a_resample_without_a_value_is_refused_not_read_as_no_copy():
    copies = stats.nested_bootstrap_copies(
        np.array([0, 0, 1]), np.random.default_rng(0), 20
    )

    def resample(session: int, n: int) -> np.ndarray:
        values = np.full(n, float(session))
        if session == 1:
            values[:2] = np.nan
        return values

    with pytest.raises(ValueError, match=r"session 1: 2 of \d+ resamples have no"):
        stats.session_copy_values(copies, resample)


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


def test_the_animal_balanced_estimate_is_the_mean_of_animal_means_of_day_means():
    # a: days (2 + 8) / 2 = 5; b: 1. Mean over animals 3. The day label d1 of
    # animal b is not animal a's day d1.
    result = stats.animal_balanced_bootstrap(
        THREE_LEVEL.value, THREE_LEVEL.animal, THREE_LEVEL.day, n_draws=50, seed=0
    )
    assert result.estimate == pytest.approx(3.0)
    assert result.n_animals == 2
    assert result.draws.shape == (50,)


def test_the_draws_resample_an_animals_sessions_not_its_days():
    """One animal: day d1 holds sessions 0 and 6, day d2 holds session 3. A
    draw takes three of the animal's sessions with replacement; each drawn day's
    mean is over its drawn sessions, and a day with none is absent. Drawing
    0, 0, 6 gives day d1 alone, mean 2: a value that resampling days first
    cannot give (it always draws whole days, so d1 alone would be four draws
    from {0, 6})."""
    values = np.array([0.0, 6.0, 3.0])
    day = np.array(["d1", "d1", "d2"])
    possible = set()
    for picks in product(range(3), repeat=3):
        drawn = np.bincount(picks, minlength=3)
        means = [
            values[(day == d) & (drawn > 0)]
            @ drawn[(day == d) & (drawn > 0)]
            / drawn[day == d].sum()
            for d in ("d1", "d2")
            if drawn[day == d].sum() > 0
        ]
        possible.add(round(float(np.mean(means)), 12))
    result = stats.animal_balanced_bootstrap(
        values, np.zeros(3), day, n_draws=2000, seed=1
    )
    assert result.estimate == pytest.approx(3.0)
    assert {round(float(v), 12) for v in result.draws} == possible
    assert 2.0 in possible


def test_each_draw_is_the_day_then_animal_mean_of_its_drawn_sessions():
    rng = np.random.default_rng(0)
    animal = np.repeat([0, 1, 2], 6)
    day = np.repeat(np.arange(6), 3)  # two days of three sessions per animal
    values = rng.normal(size=(18, 4))  # a curve of four points per session
    result = stats.animal_balanced_bootstrap(values, animal, day, n_draws=30, seed=7)
    # The same copies, drawn again with the same seed: animals, then sessions.
    copies = stats.nested_bootstrap_copies(animal, np.random.default_rng(7), 30)
    for k in range(30):
        own_means = []
        for a, drawn in zip(copies.animal[k], copies.sessions[k], strict=True):
            day_means = [
                drawn[day == d] @ values[day == d] / drawn[day == d].sum()
                for d in np.unique(day[animal == a])
                if drawn[day == d].sum() > 0
            ]
            own_means.append(np.mean(day_means, axis=0))
        np.testing.assert_allclose(
            result.draws[k], np.mean(own_means, axis=0), rtol=1e-12
        )


def test_an_animal_drawn_twice_takes_its_day_means_once_for_each_copy(monkeypatch):
    """Animal a: day d1 holds sessions 0 and 6, day d2 holds session 3; animal
    b: one session. A replicate draws a twice. One copy draws session 0 three
    times: day d1 alone, mean 0. The other draws each session once: days 3 and
    3, mean 3. The replicate is (0 + 3) / 2 = 1.5. With the two copies'
    sessions pooled, day d1 would be (4 * 0 + 6) / 5 = 1.2 and the replicate
    (1.2 + 3) / 2 = 2.1."""
    copies = stats.NestedCopies(
        np.array([[0, 0]]), np.array([[[3, 0, 0, 0], [1, 1, 1, 0]]])
    )
    monkeypatch.setattr(stats, "nested_bootstrap_copies", lambda *_: copies)
    result = stats.animal_balanced_bootstrap(
        [0.0, 6.0, 3.0, 10.0],
        ["a", "a", "a", "b"],
        ["d1", "d1", "d2", "d1"],
        n_draws=1,
        seed=0,
    )
    assert result.estimate == pytest.approx((3.0 + 10.0) / 2)
    np.testing.assert_allclose(result.draws, [1.5])


def test_the_animal_balanced_bootstrap_leaves_out_a_session_without_a_finite_value():
    values = THREE_LEVEL.value.to_numpy().copy()
    values[2] = np.nan  # animal a keeps one day, mean 2
    result = stats.animal_balanced_bootstrap(
        values, THREE_LEVEL.animal, THREE_LEVEL.day, n_draws=10, seed=0
    )
    assert result.estimate == pytest.approx((2.0 + 1.0) / 2)


def test_animal_day_session_values():
    t = THREE_LEVEL
    means = stats.animal_day_session_values(t.value, t.animal, t.day, t.session)
    # a: day means 2 and 8 -> 5; b: one day, mean 1.
    pd.testing.assert_series_equal(
        means, pd.Series([5.0, 1.0], index=pd.Index(["a", "b"])), check_names=False
    )
    # Rows within a session are reduced first: two rows of session s3 of a.
    rows = pd.concat([t, t.iloc[[2]].assign(value=10.0)], ignore_index=True)
    means = stats.animal_day_session_values(
        rows.value, rows.animal, rows.day, rows.session
    )
    assert means["a"] == pytest.approx((2.0 + 9.0) / 2)
    with pytest.raises(ValueError, match="indexed alike"):
        stats.animal_day_session_values(
            t.value, t.animal.set_axis(range(10, 16)), t.day, t.session
        )


def _days_table() -> pd.DataFrame:
    """Two animals of four days of two sessions each."""
    rng = np.random.default_rng(3)
    animal = np.repeat(["a", "b"], 8)
    day = np.tile(np.repeat(["d1", "d2", "d3", "d4"], 2), 2)
    session = np.tile(["s1", "s2"], 8)
    return pd.DataFrame(
        {
            "value": rng.normal(0.5, 1.0, 16),
            "animal": animal,
            "day": day,
            "session": session,
        }
    )


def test_day_level_test():
    t = _days_table()
    day_values = t.groupby(["animal", "day"]).value.mean()
    p = stats.day_level_test(t.value, t.animal, t.day)
    assert np.isfinite(p)
    assert p == stats.signed_rank_p(day_values.to_numpy())
    # Five days are too few.
    few = t[~((t.animal == "b") & t.day.isin(["d2", "d3", "d4"]))]
    assert np.isnan(stats.day_level_test(few.value, few.animal, few.day))


def test_estimand_summary():
    t = THREE_LEVEL
    summary = stats.estimand_summary(
        t.value, t.animal, t.day, t.session, n_draws=500, seed=11
    )
    boot = stats.animal_balanced_bootstrap(
        t.value, t.animal, t.day, n_draws=500, seed=11
    )
    interval = stats.bootstrap_interval(3.0, boot.draws, 2)
    assert summary["estimate"] == pytest.approx(3.0)
    assert summary["low"] == interval.low and summary["high"] == interval.high
    assert summary["sd"] == interval.sd
    assert summary["n_animals"] == 2 and summary["n_days"] == 3
    assert summary["n_sessions"] == 6
    assert summary["n_animals_same_sign"] == 2  # a: 5, b: 1
    assert np.isnan(summary["day_p"])  # three days
    assert summary["n_draws"] == 500 and summary["seed"] == 11
    flipped = stats.estimand_summary(
        -t.value + 2.0, t.animal, t.day, t.session, n_draws=500, seed=11
    )
    # a: -3, b: 1; the estimate -1 is negative, one animal shares its sign.
    assert flipped["estimate"] == pytest.approx(-1.0)
    assert flipped["n_animals_same_sign"] == 1
    d = _days_table()
    days = stats.estimand_summary(
        d.value, d.animal, d.day, d.session, n_draws=50, seed=0
    )
    assert days["day_p"] == stats.day_level_test(d.value, d.animal, d.day)
    own = stats.animal_day_session_values(d.value, d.animal, d.day, d.session)
    assert days["n_animals_same_sign"] == int(
        (np.sign(own) == np.sign(days["estimate"])).sum()
    )
    shifted = t.day.set_axis(range(10, 16))
    with pytest.raises(ValueError, match="values, animal, day and session"):
        stats.estimand_summary(
            t.value, t.animal, shifted, t.session, n_draws=10, seed=0
        )
    with pytest.raises(ValueError, match="values, animal and day are not"):
        stats.day_level_test(t.value, t.animal, shifted)
    with pytest.raises(ValueError, match="more than one value"):
        twice = pd.concat([t, t.iloc[[0]]], ignore_index=True)
        stats.estimand_summary(
            twice.value, twice.animal, twice.day, twice.session, n_draws=10, seed=0
        )


def test_a_session_without_a_finite_value_is_left_out_of_everything():
    """An infinite session value is left out as a missing one is: of the
    estimate, the counts, the animals' signs and its day's value."""
    d = _days_table()
    infinite, missing = d.copy(), d.copy()
    infinite.loc[0, "value"] = np.inf
    missing.loc[0, "value"] = np.nan
    a = stats.estimand_summary(
        infinite.value,
        infinite.animal,
        infinite.day,
        infinite.session,
        n_draws=50,
        seed=0,
    )
    b = stats.estimand_summary(
        missing.value, missing.animal, missing.day, missing.session, n_draws=50, seed=0
    )
    assert a == b
    assert a["n_sessions"] == len(d) - 1 and a["n_days"] == 8
    assert stats.day_level_test(
        infinite.value, infinite.animal, infinite.day
    ) == stats.day_level_test(missing.value, missing.animal, missing.day)
    values = stats.animal_day_session_values(
        infinite.value, infinite.animal, infinite.day, infinite.session
    )
    assert np.isfinite(values).all()


def test_no_finite_value_is_refused_by_name():
    t = THREE_LEVEL.assign(value=[np.nan, np.inf, np.nan, np.nan, -np.inf, np.nan])
    with pytest.raises(ValueError, match="no session has a finite value"):
        stats.animal_balanced_bootstrap(t.value, t.animal, t.day, n_draws=10, seed=0)
    with pytest.raises(ValueError, match="no session has a finite value"):
        stats.estimand_summary(t.value, t.animal, t.day, t.session, n_draws=10, seed=0)


def test_the_draws_follow_the_sorted_order_of_the_animal_labels():
    t = _days_table()
    one = stats.animal_balanced_bootstrap(t.value, t.animal, t.day, n_draws=20, seed=3)
    relabeled = t.animal.map({"a": "z", "b": "y"})
    two = stats.animal_balanced_bootstrap(t.value, relabeled, t.day, n_draws=20, seed=3)
    assert one.estimate == pytest.approx(two.estimate)
    assert not np.allclose(one.draws, two.draws)


# --- ranks and signs blind to the order of a sum -----------------------------------

# Eight days of two animals whose day means are, in exact arithmetic,
# 0.1, 0.1, -0.1, 0.2, 0.3, -0.2, 0.1, 0.4: ties among |d|. Each day's sessions
# are written two ways, so that the floating means differ in the last bit.
EXACT_DAY_MEANS = [0.1, 0.1, -0.1, 0.2, 0.3, -0.2, 0.1, 0.4]
SESSIONS_ONE = [
    [0.1, 0.1, 0.1],
    [0.1],
    [-0.1, -0.1, -0.1],
    [0.2],
    [0.3],
    [-0.2],
    [0.1],
    [0.4],
]
SESSIONS_TWO = [
    [0.1],
    [0.3, -0.1],
    [-0.1],
    [0.1, 0.3],
    [0.3, 0.3, 0.3],
    [-0.1, -0.3],
    [0.1, 0.1, 0.1],
    [0.7, 0.1],
]


def _day_table(sessions: list[list[float]]) -> pd.DataFrame:
    rows = []
    for d, values in enumerate(sessions):
        for s, v in enumerate(values):
            rows.append(
                {
                    "value": v,
                    "animal": "a" if d < 4 else "b",
                    "day": f"d{d}",
                    "session": f"s{s}",
                }
            )
    return pd.DataFrame(rows)


def test_the_day_p_does_not_depend_on_the_last_bit_of_a_tie():
    one, two = _day_table(SESSIONS_ONE), _day_table(SESSIONS_TWO)
    expected = stats.signed_rank_p(np.array(EXACT_DAY_MEANS))
    for t in (one, two):
        assert stats.day_level_test(t.value, t.animal, t.day) == expected
        summary = stats.estimand_summary(
            t.value, t.animal, t.day, t.session, n_draws=20, seed=0
        )
        assert summary["day_p"] == expected


def test_an_animal_at_zero_does_not_have_the_estimates_sign():
    """Animal b's sessions average to zero in exact arithmetic; in floating
    point the sum can leave a residue of either sign, in any order."""
    rng = np.random.default_rng(0)
    b_values = np.array([0.1, 0.2, -0.3, 0.7, -0.7, 0.3, -0.1, -0.2])
    for _ in range(20):
        shuffled = rng.permutation(b_values)
        t = pd.DataFrame(
            {
                "value": np.r_[[1.0, 2.0], shuffled],
                "animal": ["a", "a"] + ["b"] * 8,
                "day": ["d1", "d2"] + ["d1"] * 8,
                "session": ["s1", "s1"] + [f"s{i}" for i in range(8)],
            }
        )
        summary = stats.estimand_summary(
            t.value, t.animal, t.day, t.session, n_draws=10, seed=0
        )
        assert summary["estimate"] > 0
        assert summary["n_animals_same_sign"] == 1


def test_on_continuous_values_the_snap_changes_nothing():
    for seed in range(20):
        t = _days_table().assign(value=np.random.default_rng(seed).normal(0.2, 1.0, 16))
        summary = stats.estimand_summary(
            t.value, t.animal, t.day, t.session, n_draws=10, seed=0
        )
        day_means = t.groupby(["animal", "day"]).value.mean().to_numpy()
        assert summary["day_p"] == stats.signed_rank_p(day_means)
        own = stats.animal_day_session_values(t.value, t.animal, t.day, t.session)
        assert summary["n_animals_same_sign"] == int(
            (np.sign(own) == np.sign(summary["estimate"])).sum()
        )


@pytest.mark.parametrize("column", ["animal", "day", "session"])
def test_a_missing_label_is_refused_by_name(column):
    t = THREE_LEVEL.copy()
    t[column] = t[column].astype(object)
    t.loc[[1, 4], column] = None
    message = f"the {column} labels have 2 missing values"
    with pytest.raises(ValueError, match=message):
        stats.estimand_summary(t.value, t.animal, t.day, t.session, n_draws=10, seed=0)
    with pytest.raises(ValueError, match=message):
        stats.animal_day_session_values(t.value, t.animal, t.day, t.session)
    if column != "session":
        with pytest.raises(ValueError, match=message):
            stats.day_level_test(t.value, t.animal, t.day)
        with pytest.raises(ValueError, match=message):
            stats.animal_balanced_bootstrap(
                t.value, t.animal, t.day, n_draws=10, seed=0
            )


def test_animal_mean_is_the_mean_over_the_animals_with_a_value_at_each_level():
    from continuum_swr_replay.analysis.shared_statistics import animal_mean

    table = pd.DataFrame(
        {
            "animal": ["a", "a", "b", "b", "c"],
            "bin_low": [0.0, 1.0, 0.0, 1.0, 0.0],
            "percent": [10.0, 90.0, 30.0, 70.0, np.nan],
        }
    )
    means = animal_mean(table, "bin_low", "percent")
    # bin 0: a 10, b 30 (c has no value) -> 20; bin 1: a 90, b 70 -> 80
    assert means.index.tolist() == [0.0, 1.0]
    assert means.tolist() == pytest.approx([20.0, 80.0])
    with pytest.raises(ValueError):
        animal_mean(pd.concat([table, table.iloc[[0]]]), "bin_low", "percent")


def test_session_value_statistics_take_each_animals_mean_or_median_of_its_sessions():
    """a: sessions 1, 1 and 0 (mean 2/3, median 1); b: one session, 0.5. The
    mean over animals of the means is 7/12, of the medians 3/4."""
    from continuum_swr_replay.analysis.shared_statistics import session_value_statistics

    values, animal = [1.0, 1.0, 0.0, 0.5], ["a", "a", "a", "b"]
    mean = session_value_statistics(
        values, animal, statistic="mean", n_boot=200, rng=np.random.default_rng(0)
    )
    np.testing.assert_allclose(mean.per_animal.to_numpy(), [2 / 3, 0.5])
    assert mean.estimate == pytest.approx(7 / 12)
    assert mean.interval.low <= mean.estimate <= mean.interval.high
    median = session_value_statistics(
        values, animal, statistic="median", n_boot=200, rng=np.random.default_rng(0)
    )
    assert median.estimate == pytest.approx(0.75)
    # the same replicates (one seed): each slot's copies of the sessions
    np.testing.assert_array_equal(
        np.isnan(mean.copy_values), np.isnan(median.copy_values)
    )


def test_session_value_statistics_of_equal_values_have_no_spread():
    from continuum_swr_replay.analysis.shared_statistics import session_value_statistics

    result = session_value_statistics(
        [0.3] * 5, ["a", "a", "b", "c", "c"], statistic="median", n_boot=100,
        rng=np.random.default_rng(1),
    )  # fmt: skip
    assert result.estimate == pytest.approx(0.3)
    assert (result.interval.low, result.interval.high) == pytest.approx((0.3, 0.3))
    with pytest.raises(ValueError, match="'mean' or 'median'"):
        session_value_statistics(
            [1.0], ["a"], statistic="sum", n_boot=1, rng=np.random.default_rng(0)
        )
    with pytest.raises(ValueError, match="finite"):
        session_value_statistics(
            [np.nan], ["a"], statistic="mean", n_boot=1, rng=np.random.default_rng(0)
        )


# --- the mean over animals with the t interval -------------------------------------


def test_animal_mean_interval_by_hand():
    # Values 1, 2, 3, -4, 18 over five animals: mean 4, SD (ddof=1) and its
    # standard error over sqrt(5), t(0.975, 4) = 2.7764451052. A missing
    # animal value is left out; four of the five share the mean's sign.
    values = pd.Series([1.0, 2.0, 3.0, -4.0, 18.0, NAN], index=list("abcdef"))
    summary = stats.animal_mean_interval(values)
    se = np.sqrt(((values.dropna() - 4.0) ** 2).sum() / 4 / 5)
    assert summary["estimate"] == pytest.approx(4.0)
    assert summary["se"] == pytest.approx(se)
    assert summary["low"] == pytest.approx(4.0 - 2.7764451052 * se)
    assert summary["high"] == pytest.approx(4.0 + 2.7764451052 * se)
    assert summary["n_animals"] == 5
    assert summary["n_animals_same_sign"] == 4


def test_animal_mean_interval_needs_two_animals_for_an_interval():
    one = stats.animal_mean_interval(pd.Series([0.7], index=["a"]))
    assert one["estimate"] == pytest.approx(0.7)
    assert np.isnan(one["low"]) and np.isnan(one["high"]) and np.isnan(one["se"])
    assert one["n_animals"] == 1
    none = stats.animal_mean_interval(pd.Series([NAN], index=["a"]))
    assert np.isnan(none["estimate"]) and none["n_animals"] == 0
    with pytest.raises(ValueError, match="one value per animal"):
        stats.animal_mean_interval(pd.Series([1.0, 2.0], index=["a", "a"]))


# --- the reduction of one value per unit of the fit ----------------------------------

#: Two animals: "a" with days d1 (sessions 1 and 2) and d2 (session 3), "b" with
#: day d1 (session 1) and day d2 (sessions 1 and 2), and "c" with one session.
_SESSION_VALUES = pd.Series(
    [1.0, 3.0, 5.0, -2.0, 4.0, 6.0, 0.5],
    index=pd.MultiIndex.from_tuples(
        [
            ("a", "d1", "e1"),
            ("a", "d1", "e2"),
            ("a", "d2", "e3"),
            ("b", "d1", "e1"),
            ("b", "d2", "e1"),
            ("b", "d2", "e2"),
            ("c", "d1", "e1"),
        ],
        names=["animal", "session_date", "epoch"],
    ),
)


def test_unit_summary_session_is_estimand_summary():
    index = _SESSION_VALUES.index
    labels = [pd.Series(index.get_level_values(k), index=index) for k in range(3)]
    expected = stats.estimand_summary(_SESSION_VALUES, *labels, n_draws=300, seed=4)
    got = stats.unit_summary(_SESSION_VALUES, "session", n_draws=300, seed=4)
    np.testing.assert_equal({k: got[k] for k in expected}, expected)
    assert got["fit_by"] == "session" and got["n_units"] == 7


def test_unit_summary_day_one_session_per_day():
    # One value per day: each day is its own session. Animal a: days 2 and 5,
    # mean 3.5; b: 1 and 5, mean 3; c: 0.5. The estimate is their mean.
    days = pd.Series(
        [2.0, 5.0, 1.0, 5.0, 0.5],
        index=pd.MultiIndex.from_tuples(
            [("a", "d1"), ("a", "d2"), ("b", "d1"), ("b", "d2"), ("c", "d1")],
            names=["animal", "session_date"],
        ),
    )
    got = stats.unit_summary(days, "day", n_draws=300, seed=4)
    animal = pd.Series(days.index.get_level_values(0), index=days.index)
    day = pd.Series(days.index.get_level_values(1), index=days.index)
    expected = stats.estimand_summary(days, animal, day, day, n_draws=300, seed=4)
    assert got["estimate"] == pytest.approx((3.5 + 3.0 + 0.5) / 3)
    for key in ("estimate", "low", "high", "sd", "day_p", "n_animals_same_sign"):
        np.testing.assert_equal(got[key], expected[key])
    assert (got["n_animals"], got["n_days"], got["n_units"]) == (3, 5, 5)
    assert np.isnan(got["n_sessions"])


def test_unit_summary_animal_is_t_interval():
    # Five animal values 1, 2, 3, -4, 18 (one NaN left out): mean 4, the
    # interval 4 +/- t(0.975, 4) SD / sqrt(5); no day-level test.
    values = pd.Series(
        [1.0, 2.0, 3.0, -4.0, 18.0, NAN],
        index=pd.Index(list("abcdef"), name="animal"),
    )
    got = stats.unit_summary(values, "animal", n_draws=300, seed=4)
    sd = np.sqrt(((values.dropna() - 4.0) ** 2).sum() / 4)
    half = 2.7764451052 * sd / np.sqrt(5)
    assert got["estimate"] == pytest.approx(4.0)
    assert (got["low"], got["high"]) == pytest.approx((4.0 - half, 4.0 + half))
    # sd is the SD whose bootstrap_interval scale gives the same half width
    assert got["sd"] * stats.draws_sd_scale(5) == pytest.approx(sd / np.sqrt(5))
    assert np.isnan(got["day_p"])
    assert (got["n_animals"], got["n_units"], got["n_animals_same_sign"]) == (5, 5, 4)
    assert np.isnan(got["n_days"]) and np.isnan(got["n_sessions"])


def test_unit_summary_same_keys_at_every_unit():
    session = _SESSION_VALUES
    day = session.groupby(level=[0, 1]).mean()
    animal = day.groupby(level=0).mean()
    keys = [
        set(stats.unit_summary(values, fit_by, n_draws=50, seed=1))
        for values, fit_by in ((session, "session"), (day, "day"), (animal, "animal"))
    ]
    assert keys[0] == keys[1] == keys[2]


def test_unit_summary_refuses_values_not_keyed_by_the_unit():
    with pytest.raises(ValueError, match="day"):
        stats.unit_summary(_SESSION_VALUES, "day", n_draws=50, seed=1)
    with pytest.raises(ValueError, match="fit_by"):
        stats.unit_summary(_SESSION_VALUES, "epoch", n_draws=50, seed=1)
    with pytest.raises(ValueError, match="more than one value"):
        stats.unit_summary(
            pd.Series([1.0, 2.0], index=pd.Index(["a", "a"], name="animal")),
            "animal",
            n_draws=50,
            seed=1,
        )
