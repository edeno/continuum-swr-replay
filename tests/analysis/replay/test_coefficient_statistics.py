"""The across-session summaries of per-session fits, on tables small enough to check by hand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from continuum_swr_replay.analysis.replay import coefficient_statistics as stats

NAN = np.nan


def _fits(coefs, animals, *, term="x", sd=0.1, separated_terms=None, **columns):
    """One fit per coefficient, each its own session of its animal."""
    n = len(coefs)
    frame = pd.DataFrame(
        {
            "animal": list(animals),
            "session_date": list(range(n)),
            "epoch": ["02_r1"] * n,
            f"coef_{term}": np.asarray(coefs, dtype=float),
            f"sd_{term}": sd,
            "separated_terms": separated_terms
            if separated_terms is not None
            else [""] * n,
        }
    )
    return frame.assign(**columns)


# --- the shared summary ---------------------------------------------------------------


def test_fit_rows_keep_the_rows_with_every_label():
    fits = pd.DataFrame(
        {
            "model": ["M0", "M0", "M1", "M1"],
            "outcome": ["swr", "non_local", "swr", "non_local"],
            "aic": [1.0, 2.0, 3.0, 4.0],
        }
    )
    assert list(stats.fit_rows(fits, model="M1")["aic"]) == [3.0, 4.0]
    assert list(stats.fit_rows(fits, model="M1", outcome="swr")["aic"]) == [3.0]
    assert list(stats.fit_rows(fits)["aic"]) == [1.0, 2.0, 3.0, 4.0]


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


# --- one coefficient ------------------------------------------------------------------


def test_coefficient_statistics_leave_out_sessions_without_the_coefficient():
    fits = pd.DataFrame(
        {
            "animal": ["a", "a", "b", "b", "c"],
            "beta": [0.5, 1.0, 1.5, 2.0, NAN],
        }
    )
    result = stats.coefficient_statistics(fits, "beta")
    assert result["n_sessions"] == 4
    assert result["median"] == 1.25
    # the animal with no coefficient is not an animal of the summary
    assert result["n_animals"] == 2
    assert result["animal_median"] == 1.25  # a 0.75, b 1.75
    assert result["n_animals_positive"] == 2
    assert result["exp_median"] == pytest.approx(3.4903429574618414)  # e ** 1.25
    assert result["exp_animal_median"] == pytest.approx(3.4903429574618414)
    # four sessions, all positive: the exact two-sided p is 2 / 2 ** 4
    assert result["signed_rank_p"] == pytest.approx(0.125)


def test_the_signed_rank_p_is_of_the_sessions_against_zero():
    fits = pd.DataFrame(
        {"animal": ["a", "b", "c", "d", "e"], "beta": [1.0, -2.0, 3.0, 4.0, 5.0]}
    )
    # The negative ranks sum to 2. Of the 32 sign patterns, three have a rank
    # sum of at most 2 ({}, {1}, {2}): two-sided p = 2 * 3 / 32.
    result = stats.coefficient_statistics(fits, "beta")
    assert result["signed_rank_p"] == pytest.approx(0.1875)
    assert result["exp_median"] == pytest.approx(20.085536923187668)  # e ** 3
    assert result["exp_animal_median"] == pytest.approx(20.085536923187668)


def test_exp_of_the_two_medians_are_separate_numbers():
    fits = pd.DataFrame({"animal": ["a"] * 3 + ["b"], "beta": [0.0, 0.0, 0.0, 2.0]})
    result = stats.coefficient_statistics(fits, "beta")
    assert result["exp_median"] == 1.0  # session median 0
    # animal medians 0 and 2 -> 1 -> e
    assert result["exp_animal_median"] == pytest.approx(2.718281828459045)


def test_the_gate_leaves_out_separated_runaway_and_missing_sessions():
    fits = _fits(
        # kept: 0.2, 0.4, 0.6, 1.0; then separated, runaway (200 * 0.1 >= 10), missing
        [0.2, 0.4, 0.6, 1.0, 3.0, 200.0, NAN],
        ["a", "a", "b", "b", "c", "c", "c"],
        separated_terms=["", "y", "", "", "y|x", "", ""],
        bse_x=0.2,
    )
    result = stats.gated_coefficient_statistics(fits, "x")
    assert result["n_sessions"] == 4
    assert result["median"] == 0.5
    assert result["n_animals"] == 2
    assert result["animal_median"] == pytest.approx(0.55)  # a 0.3, b 0.8
    assert result["session_se_median"] == pytest.approx(0.2)
    # z = 1, 2, 3, 5: three above 1.96
    assert result["significant_fraction"] == 0.75
    assert result["exp_median"] == pytest.approx(1.6487212707001282)  # e ** 0.5
    assert result["signed_rank_p"] == pytest.approx(0.125)


def test_significance_is_read_on_the_clustered_standard_error():
    fits = _fits(
        [0.2, 0.4, 0.6, 1.0], ["a", "a", "b", "b"], bse_x=0.2, bse_cluster_x=0.25
    )
    result = stats.gated_coefficient_statistics(fits, "x")
    assert result["session_se_median"] == pytest.approx(0.25)
    # z = 0.8, 1.6, 2.4, 4: two above 1.96
    assert result["significant_fraction"] == 0.5


def test_fits_without_a_standard_error_use_their_p_values():
    fits = _fits(
        [0.2, 0.4, 0.6, 1.0], ["a", "a", "b", "b"], pval_x=[0.5, 0.04, 0.2, 0.01]
    )
    result = stats.gated_coefficient_statistics(fits, "x")
    assert np.isnan(result["session_se_median"])
    assert result["significant_fraction"] == 0.5


def test_the_exp_median_ratio_divides_each_fit_s_own_gated_session_median():
    log = np.log
    # a: session median log 4 (its median over animals is log 8); the runaway
    # fit is left out. b: log 2.
    a = _fits([log(4), log(4), log(4), log(16), 500.0], ["a", "a", "a", "b", "b"])
    b = _fits([log(1), log(2), log(4)], ["a", "b", "c"])
    assert stats.gated_exp_median_ratio(a, b, "x") == pytest.approx(2.0)
    assert stats.gated_exp_median_ratio(b, a, "x") == pytest.approx(0.5)


# --- one coefficient in two fits of the same sessions ---------------------------------


def _split_fits(
    differences: dict[str, list[float]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per animal, one session per difference: fit a has 1 + d, fit b has 1."""
    rows = [
        {
            "animal": animal,
            "session_date": session,
            "epoch": "02_r1",
            "coef_x": 1.0,
            "sd_x": 0.3,
            "separated_terms": "",
            "d": d,
        }
        for animal, diffs in differences.items()
        for session, d in enumerate(diffs)
    ]
    b = pd.DataFrame(rows)
    a = b.assign(coef_x=b["coef_x"] + b["d"])
    return a.drop(columns="d"), b.drop(columns="d")


def test_the_difference_is_within_session_then_over_animals():
    a, b = _split_fits(
        {
            "a": [0.5] * 6,
            "b": [-0.2] * 6,
            "c": [0.1] * 6,
            "d": [0.3] * 6,
            "e": [-0.4] * 6,
        }
    )
    result = stats.gated_coefficient_difference_statistics(a, b, "x")
    assert result["animal_median"] == pytest.approx(0.1)
    assert result["n_animals_positive"] == 3
    assert result["n_animals"] == 5
    assert result["n_sessions"] == 30
    # the difference is a - b, not b - a
    assert stats.gated_coefficient_difference_statistics(b, a, "x")[
        "animal_median"
    ] == pytest.approx(-0.1)


def test_the_difference_drops_a_session_missing_or_gated_in_either_fit():
    a, b = _split_fits({animal: [0.2] * 6 for animal in "abcde"})
    a = a.drop(a.index[a["session_date"] == 0])  # five sessions have no fit a
    result = stats.gated_coefficient_difference_statistics(a, b, "x")
    assert result["n_sessions"] == 25
    assert result["animal_median"] == pytest.approx(0.2)

    b.loc[(b["animal"] == "a") & (b["session_date"] == 1), "separated_terms"] = "x"
    assert stats.gated_coefficient_difference_statistics(a, b, "x")["n_sessions"] == 24


def test_the_difference_refuses_two_fits_of_one_session():
    a, b = _split_fits({"a": [0.2, 0.3]})
    with pytest.raises(pd.errors.MergeError):
        stats.gated_coefficient_difference_statistics(pd.concat([a, a]), b, "x")


# --- a rung against its parent --------------------------------------------------------


def _rung(aic, n_stops=None) -> pd.DataFrame:
    n = len(aic)
    return pd.DataFrame(
        {
            "animal": ["a", "a", "b", "b"][:n],
            "session_date": list(range(n)),
            "epoch": ["02_r1"] * n,
            "aic": aic,
            "n_stops": n_stops if n_stops is not None else [5] * n,
        }
    )


def test_the_aic_difference_is_child_minus_parent_within_session():
    child = _rung([10.0, 20.0, 30.0, 40.0])
    parent = _rung([12.0, 19.0, 35.0, NAN])
    # differences -2, 1, -5; the session without a parent AIC is left out
    result = stats.aic_difference_statistics(child, parent)
    assert result["n_sessions"] == 3
    assert result["median"] == -2.0
    assert result["negative_fraction"] == pytest.approx(2 / 3)
    assert result["animal_median"] == -2.75  # a -0.5, b -5
    assert result["n_animals"] == 2
    assert result["n_animals_negative"] == 2


def test_the_aic_difference_needs_both_fits_of_a_session():
    child = _rung([10.0, 20.0, 30.0, 40.0])
    parent = _rung([12.0, 19.0, 35.0])  # no fourth session
    assert stats.aic_difference_statistics(child, parent)["n_sessions"] == 3


def test_the_aic_difference_can_require_the_same_stops():
    child = _rung([10.0, 20.0, 30.0, 40.0])
    parent = _rung([12.0, 19.0, 35.0, 44.0], n_stops=[5, 6, 5, 5])
    # the second session's rungs saw different stops: -2, -5, -4 remain
    result = stats.aic_difference_statistics(child, parent, same=("n_stops",))
    assert result["n_sessions"] == 3
    assert result["median"] == -4.0
    assert result["negative_fraction"] == 1.0
    assert stats.aic_difference_statistics(child, parent)["n_sessions"] == 4


# --- separation -----------------------------------------------------------------------


def test_separated_fraction_counts_only_a_true_flag():
    fits = pd.DataFrame({"separated": [True, False, None, True]})
    assert stats.separated_fraction(fits) == 0.5


def test_fits_carrying_need_every_term():
    fits = pd.DataFrame({"coef_x": [1.0, NAN, 2.0], "coef_y": [1.0, 1.0, NAN]})
    assert list(stats.fits_carrying(fits, ["x"]).index) == [0, 2]
    assert list(stats.fits_carrying(fits, ["x", "y"]).index) == [0]


def test_term_separation_counts_fits_flagging_any_of_the_terms():
    fits = pd.DataFrame({"separated_terms": ["x|y", "", None, "z", "y", "xx|y"]})
    # x or z is named by the first and fourth fits; "xx" is another term
    assert stats.term_separation_statistics(fits, ["x", "z"]) == {
        "n_fits": 6,
        "n_flagged": 2,
        "flagged_fraction": pytest.approx(1 / 3),
    }
    assert stats.term_separation_statistics(fits, ["y"])["n_flagged"] == 3


def test_rung_separation_range_is_over_the_matching_rungs():
    fits = pd.DataFrame(
        {
            "model": ["S1_a"] * 4 + ["S2_a"] * 3 + ["S10_b"] * 2 + ["S11_c"],
            "separated": [True] * 4 + [True, False, False] + [True, True] + [False],
            "separated_terms": ["t"] * 4 + ["t", "t|u", "u"] + ["u", ""] + ["t"],
        }
    )
    # S2_a: 3 fits, 1 separated, 2 flagging t; S10_b: 2 fits, 2 separated, 0 flagging
    assert stats.rung_separation_range(fits, "t", r"S([2-9]|10)_") == {
        "n_fits_min": 2,
        "n_fits_max": 3,
        "n_separated_min": 1,
        "n_separated_max": 2,
        "n_flagged_min": 0,
        "n_flagged_max": 2,
    }


def test_audited_drop_statistics_name_the_first_row_with_the_most_drops():
    audited = pd.DataFrame(
        {
            "model": ["m1", "m2", "m3", "m4"],
            "term": ["t1", "t2", "t3", "t4"],
            "n_sessions": [100, 90, 80, 95],
            "n_dropped": [0, 12, 12, 3],
        }
    )
    assert stats.audited_drop_statistics(audited) == {
        "n_terms_losing_sessions": 3,
        "most_dropped": 12,
        "most_dropped_of": 102,
        "most_dropped_model": "m2",
        "most_dropped_term": "t2",
    }
